//===- AirChannelIndexFlattener.cpp - Flatten multi-dim air.channel ops --===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --air-channel-flatten-indices: flatten multi-dimensional air.channel
// declarations and their put/get index operands to scalar channels.
//
// Background
// ----------
// air.channel @foo [M, N] declares M×N independent channels, one per (i,j)
// index pair.  SPMD specialization (air-specialize-channel-broadcast) is
// supposed to reduce all channel declarations to [1,1] before Pass B
// (--air-channel-to-conduit) runs.  However, programs that still carry
// rank-2 (or higher) channel declarations cannot be processed by Pass B,
// which only handles [1,1] scalar channels.
//
// This pass performs the flattening statically:
//
//   air.channel @foo [M, N]
//     → M×N scalar channel declarations @foo_0_0 ... @foo_{M-1}_{N-1}
//
//   air.channel.put @foo[%ci, %cj]  (arith.constant indices i, j)
//     → air.channel.put @foo_{i}_{j}  (chan_name attribute rewritten)
//
//   air.channel.get @foo[%ci, %cj]  (arith.constant indices i, j)
//     → air.channel.get @foo_{i}_{j}
//
// Dynamic indices (loop IVs, block args) cannot be resolved statically.
// The pass emits a hard error (emitError + signalPassFailure) on any
// dynamic-index put/get on a multi-dimensional channel.
//
// Algorithm
// ---------
// Phase 1: Walk air.channel declaration ops.  For each with rank > 1 and all
//   static (integer constant) bounds, record the shape and generate the flat
//   name list.  Rank-1 [1] or [N] channels are left alone (they are already
//   scalar or 1-D, and Pass B handles them).  Rank-0 channels are trivially
//   scalar and are also skipped.
//
// Phase 2: For each multi-dim channel declaration, emit M×N new
//   air.channel declarations with rank [1, 1] (same as SPMD output) named
//   @foo_0_0, @foo_0_1, ..., @foo_{M-1}_{N-1}.
//
// Phase 3: Walk all air.channel.put and air.channel.get ops.  For those
//   that reference a multi-dim channel (by chan_name), extract the index
//   operands.  If all indices are arith.constant, rewrite chan_name to the
//   flat name @foo_{i}_{j}.  If any index is dynamic, emit a hard error.
//
// Phase 4: Erase the original multi-dim air.channel declaration ops.
//
// Notes
// -----
// - Only rank-2 channels with static bounds are supported.  Higher-rank
//   channels would require generalization but are not present in the
//   mlir-air corpus.
// - The channel `size` attribute (not operand bounds) records the shape.
//   We read it from the "size" DenseI64ArrayAttr on the declaration op.
// - This pass does NOT modify the put/get operand_segment_sizes — the index
//   operands are simply removed from the segment after rewriting chan_name.
//   The Pass B operand-segment decoder treats the indices as a separate
//   segment (nidx), so dropping index operands would break parsing.
//   Instead, we rewrite only chan_name and leave the index operands in place;
//   Pass B will warn about multi-dim indices on the resulting [1,1] channel.
//   To fully remove index operands, a separate cleanup pass would be needed.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx::conduit {

using ::mlir::ModuleOp;

#define GEN_PASS_DECL_AIRCHANNELINDEXFLATTENER
#define GEN_PASS_DEF_AIRCHANNELINDEXFLATTENER
#include "air/Conversion/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers (mirrors the subset used by Pass B — no AIR dialect dependency)
// ---------------------------------------------------------------------------

static bool isAirChannelDecl(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.channel";
}

static bool isAirChannelPut(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.channel.put";
}

static bool isAirChannelGet(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.channel.get";
}

static std::string getSymName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name"))
    return attr.getValue().str();
  return "";
}

static std::string getChanName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("chan_name"))
    return attr.getValue().str();
  if (auto attr = op->getAttr("chan_name")) {
    if (auto symRef = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr))
      return symRef.getValue().str();
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
      return strAttr.getValue().str();
  }
  return "";
}

/// Try to extract a compile-time integer constant from an SSA value.
static std::optional<int64_t> tryExtractConstInt(mlir::Value v) {
  mlir::Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  if (auto cOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp))
    return cOp.value();
  if (auto cOp = mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp))
    return cOp.value();
  if (auto cOp = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp)) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cOp.getValue()))
      return intAttr.getValue().getSExtValue();
  }
  return std::nullopt;
}

/// Return the flat channel name for a multi-dim channel with indices (i, j).
static std::string flatName(llvm::StringRef base, int64_t i, int64_t j) {
  return (base + "_" + std::to_string(i) + "_" + std::to_string(j)).str();
}

/// Retrieve operand segment sizes from "operand_segment_sizes".
static llvm::SmallVector<int32_t> getOperandSegments(mlir::Operation *op) {
  llvm::SmallVector<int32_t> segs;
  auto attr =
      op->getAttrOfType<mlir::DenseI32ArrayAttr>("operand_segment_sizes");
  if (attr)
    for (int32_t v : attr.asArrayRef())
      segs.push_back(v);
  return segs;
}

// ---------------------------------------------------------------------------
// Pass struct
// ---------------------------------------------------------------------------

struct AirChannelIndexFlattenerPass
    : impl::AirChannelIndexFlattenerBase<AirChannelIndexFlattenerPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    mlir::MLIRContext *ctx = module.getContext();

    // -----------------------------------------------------------------------
    // Phase 1: collect multi-dimensional channel declarations.
    //
    // A channel declaration looks like:
    //   "air.channel"() {sym_name = "foo", size = array<i64: 2, 3>} : () -> ()
    //
    // We look for declarations where the "size" attribute is a DenseI64Array
    // with more than one element AND at least one element > 1.  Rank-1 [N]
    // channels are treated as scalar (N instances of a 1-D channel) and are
    // not flattened by this pass — Pass B handles them directly.
    // -----------------------------------------------------------------------

    // Map: original channel name → shape [M, N].
    llvm::StringMap<llvm::SmallVector<int64_t>> multiDimChannels;

    // Collect all channel decl ops for processing.
    llvm::SmallVector<mlir::Operation *> allChannelDecls;
    llvm::SmallVector<mlir::Operation *> multiDimDeclsToErase;

    module.walk([&](mlir::Operation *op) {
      if (!isAirChannelDecl(op))
        return;
      allChannelDecls.push_back(op);

      std::string name = getSymName(op);
      if (name.empty())
        return;

      // Read the "size" attribute: array<i64: M, N>.
      auto sizeAttr = op->getAttr("size");
      if (!sizeAttr)
        return;

      // Try DenseI64ArrayAttr (array<i64: ...>) or DenseIntElementsAttr.
      llvm::SmallVector<int64_t> shape;
      if (auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(sizeAttr)) {
        for (int64_t d : dense.asArrayRef())
          shape.push_back(d);
      } else if (auto intArr =
                     mlir::dyn_cast<mlir::DenseIntElementsAttr>(sizeAttr)) {
        for (auto v : intArr.getValues<int64_t>())
          shape.push_back(v);
      } else {
        return; // Unrecognized format — leave alone.
      }

      // Only handle rank-2 channels with both dims > 1 that need flattening.
      // Scalar [1, 1] or rank-1 channels are left alone.
      if (shape.size() < 2)
        return;
      bool needsFlattening = false;
      for (int64_t d : shape) {
        if (d > 1)
          needsFlattening = true;
      }
      if (!needsFlattening)
        return;

      multiDimChannels[name] = shape;
      multiDimDeclsToErase.push_back(op);
    });

    if (multiDimChannels.empty())
      return; // Nothing to do.

    // -----------------------------------------------------------------------
    // Phase 2: emit M×N scalar channel declarations for each multi-dim channel.
    //
    // New declarations are emitted with size = array<i64: 1, 1> (scalar) and
    // all other attributes inherited from the original declaration.
    // -----------------------------------------------------------------------

    for (mlir::Operation *op : multiDimDeclsToErase) {
      std::string name = getSymName(op);
      auto &shape = multiDimChannels[name];
      int64_t M = shape[0];
      int64_t N = (shape.size() >= 2) ? shape[1] : 1;

      builder.setInsertionPoint(op);
      mlir::Location loc = op->getLoc();

      // Build the attribute set for each new channel: copy all attrs from
      // the original, but replace sym_name and size.
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          std::string newName = flatName(name, i, j);

          // Collect attrs: all original minus sym_name and size; then add new.
          llvm::SmallVector<mlir::NamedAttribute> attrs;
          for (auto &attr : op->getAttrs()) {
            llvm::StringRef attrName = attr.getName().strref();
            if (attrName == "sym_name" || attrName == "size")
              continue;
            attrs.push_back(attr);
          }
          attrs.push_back(
              mlir::NamedAttribute(mlir::StringAttr::get(ctx, "sym_name"),
                                   mlir::StringAttr::get(ctx, newName)));
          attrs.push_back(
              mlir::NamedAttribute(mlir::StringAttr::get(ctx, "size"),
                                   mlir::DenseI64ArrayAttr::get(ctx, {1, 1})));

          mlir::OperationState state(loc, "air.channel");
          state.addAttributes(attrs);
          builder.create(state);
        }
      }
    }

    // -----------------------------------------------------------------------
    // Phase 3: rewrite chan_name on put/get ops that reference multi-dim
    // channels.  Index operands are decoded from operand_segment_sizes.
    //
    // For each put/get:
    //   - If the channel is not multi-dim: skip.
    //   - Extract the index operands (segment 1 in operand_segment_sizes).
    //   - If all indices are arith.constant: rewrite chan_name.
    //   - If any index is dynamic: hard error + signalPassFailure().
    // -----------------------------------------------------------------------

    bool passFailed = false;
    module.walk([&](mlir::Operation *op) {
      if (!isAirChannelPut(op) && !isAirChannelGet(op))
        return;

      std::string chanName = getChanName(op);
      if (chanName.empty())
        return;

      auto multiIt = multiDimChannels.find(chanName);
      if (multiIt == multiDimChannels.end())
        return; // Not a multi-dim channel — leave alone.

      auto &shape = multiIt->second;
      int64_t M = shape[0];
      int64_t N = (shape.size() >= 2) ? shape[1] : 1;

      // Decode index operands.
      auto segs = getOperandSegments(op);
      // operand_segment_sizes = [ndeps, nidx, 1(memref), noffsets, nsizes,
      // nstrides]
      if (segs.size() < 2) {
        op->emitError()
            << "air-channel-flatten-indices: channel @" << chanName
            << " put/get has no operand_segment_sizes; cannot decode indices";
        passFailed = true;
        signalPassFailure();
        return;
      }
      int32_t ndeps = segs[0];
      int32_t nidx = segs[1];

      if (nidx == 0) {
        // No index operands — this is a scalar access on a multi-dim channel.
        // Treat as [0, 0].
        std::string newName = flatName(chanName, 0, 0);
        op->setAttr("chan_name", mlir::FlatSymbolRefAttr::get(ctx, newName));
        return;
      }

      mlir::OperandRange allOps = op->getOperands();
      if ((int32_t)allOps.size() < ndeps + nidx) {
        op->emitError() << "air-channel-flatten-indices: channel @" << chanName
                        << " put/get: operand count " << allOps.size()
                        << " < ndeps+nidx = " << ndeps + nidx;
        passFailed = true;
        signalPassFailure();
        return;
      }

      // Extract index values.
      llvm::SmallVector<int64_t> indices;
      bool hasDynamic = false;
      for (int32_t k = 0; k < nidx; ++k) {
        mlir::Value idxVal = allOps[ndeps + k];
        auto maybeConst = tryExtractConstInt(idxVal);
        if (!maybeConst) {
          hasDynamic = true;
          break;
        }
        indices.push_back(*maybeConst);
      }

      if (hasDynamic) {
        op->emitError()
            << "air-channel-flatten-indices: channel @" << chanName
            << " has dynamic index operand(s); cannot statically determine "
               "the target channel @"
            << chanName
            << "[i][j]. "
               "Use air-specialize-channel-broadcast before this pass to "
               "specialize dynamic indices to constants.";
        passFailed = true;
        signalPassFailure();
        return;
      }

      // Validate bounds.
      int64_t i = (indices.size() >= 1) ? indices[0] : 0;
      int64_t j = (indices.size() >= 2) ? indices[1] : 0;
      if (i < 0 || i >= M || j < 0 || j >= N) {
        op->emitError() << "air-channel-flatten-indices: channel @" << chanName
                        << " index [" << i << ", " << j << "] out of bounds "
                        << "[" << M << ", " << N << "]";
        passFailed = true;
        signalPassFailure();
        return;
      }

      // Rewrite chan_name to the flat scalar channel.
      std::string newName = flatName(chanName, i, j);
      op->setAttr("chan_name", mlir::FlatSymbolRefAttr::get(ctx, newName));
    });

    // -----------------------------------------------------------------------
    // Phase 4: erase original multi-dim channel declaration ops.
    // Only erase if no failure was signaled — on failure, put/get ops that
    // triggered the error may still reference the original decl (they were
    // not rewritten), so erasing would leave dangling symbol references.
    // -----------------------------------------------------------------------
    if (!passFailed) {
      for (mlir::Operation *op : multiDimDeclsToErase)
        op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAirChannelIndexFlattenerPass() {
  return std::make_unique<AirChannelIndexFlattenerPass>();
}

} // namespace xilinx::conduit
