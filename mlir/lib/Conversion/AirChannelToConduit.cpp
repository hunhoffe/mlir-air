//===- AirChannelToConduit.cpp - AIR Channel → Conduit IR (Pass B)
//-*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pass B of the Conduit lowering pipeline: lift air.channel.* ops into
// Conduit Tier 3 memref-DMA IR.
//
// Architecture:
//
//   aie.objectfifo.*  ──┐
//                       ├──► Conduit IR ──► aie.dma_bd / aie.lock / aie.buffer
//   air.channel.*     ──┘
//
//   (ObjectFifoToConduit.cpp)   (this file)   (ConduitToDMA.cpp)
//
// Design note: why generic op matching
// -------------------------------------
// The AIR dialect (mlir-air) is a separate repository with its own build.
// The mlir-aie aie-opt tool does NOT link against the AIR dialect library.
// To parse air.channel.* programs, the caller must use
//   aie-opt --allow-unregistered-dialect --air-channel-to-conduit ...
// This pass therefore matches ops by their string name
// ("air.channel.put", "air.channel.get", etc.) rather than by C++ type.
// All operand / attribute access goes through the generic MLIR API.
//
// Supported mappings
// ------------------
//
// 1. air.channel declaration (Symbol op, no operands):
//      air.channel @name [1, 1]
//    → conduit.create {name="name", depth=1}
//      The element_type is left unset (unknown until a put/get is seen).
//      A second pass fills element_type from the memref operand of the
//      first put/get that references this channel.
//
// 2. air.channel.put (blocking or async):
//      %tok = air.channel.put async [%deps] @chan[%i,%j]
//                 (%buf[%o0,%o1][%s0,%s1][%st0,%st1]) : (memref<...>)
//    → %tok = conduit.put_memref_async
//                 {name="chan", num_elems=<product(sizes)>,
//                  offsets=<static offsets or []>,
//                  sizes=<static sizes or []>,
//                  strides=<static strides or []>}
//                 : !conduit.dma.token
//      The %tok SSA value is replaced with the new !conduit.dma.token.
//
// 3. air.channel.get (blocking or async):
//    → conduit.get_memref_async {same attrs} : !conduit.dma.token
//
// 4. air.wait_all:
//      %t = air.wait_all async [%dep0, %dep1]
//    → %t = conduit.wait_all_async %dep0, %dep1
//              : (!conduit.dma.token, ...) -> !conduit.dma.token
//      Blocking (no result):
//      air.wait_all [%dep0, %dep1]
//    → conduit.wait_all %dep0, %dep1
//
// 5. air.async.token type → !conduit.dma.token
//    (via SSA replacement; no explicit type conversion needed because
//    conduit ops produce !conduit.dma.token results directly)
//
// Coverage
// --------
// - Static-shape SPSC programs: handled (offsets/sizes/strides from
// arith.constant extracted)
// - Static strides from arith.constant (index or integer type): extracted
// correctly
// - channel_type propagation: "dma_packet" → routing_mode = "packet" on
// conduit.create;
//   "cascade" → routing_mode = "cascade" (put/get rewritten to
//   conduit.put_cascade / conduit.get_cascade); "dma_stream" / absent → circuit
//   default (no routing_mode attr)
// - broadcast_shape: emits a diagnostic warning (not silently dropped); full
// broadcast
//   topology lowering is a future TODO
// - Multi-dimensional channel indices [M,N]: warned and dropped; only [1,1]
// scalar channels
//   supported (multi-dim channels require a pre-pass to specialize indices)
// - Async token threading: structural only (air.async.token →
// !conduit.dma.token)
// - Dynamic offsets/strides (SSA non-constant, e.g. loop IVs): hard error
//   (placeholder substitution produces wrong DMA descriptors;
//   emitError+signalPassFailure)
//
// Known limitations (documented honestly)
// ----------------------------------------
// - Only [1,1] scalar channels supported; multi-dimensional indices ignored.
// - num_elems is computed from static sizes only; dynamic sizes emit a hard
//   error (run --air-hierarchy-to-aie --air-split-devices first to specialize).
// - Offset/size/stride Index SSA values are extracted when they come from
//   arith.constant (ConstantIndexOp, ConstantIntOp, or generic ConstantOp with
//   integer attribute).  Truly dynamic values (loop induction variables, block
//   arguments, etc.) cause a hard error (emitError + signalPassFailure) because
//   placeholder substitution produces incorrect DMA descriptors.
//   Full dynamic operand threading is a future TODO.
// - The blocking (non-async) put/get forms with no result SSA value are
//   lowered to the async form with the result token unused.  This is safe
//   because the token is not consumed by any downstream op in the original.
// - air.execute regions (async wrappers): correctly passed through as
// unregistered
//   ops (with --allow-unregistered-dialect); the memref SSA values they yield
//   are consumed by put/get and correctly decoded by Phase 2b element_type
//   patching.
// - broadcast_shape: conduit.create has no broadcast_shape field; full
// broadcast
//   topology (capacity, routing hints) is deferred. A warning is emitted.
// - cascade channels: the air.channel.put/get operand is a memref; Pass B emits
//   a memref.load before conduit.put_cascade (put path) or memref.store after
//   conduit.get_cascade (get path).  Multi-element cascade memrefs (>1 element)
//   are not yet supported — only element [0] is transferred.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <map>
#include <set>
#include <string>

namespace xilinx::conduit {

using ::mlir::ModuleOp;

#define GEN_PASS_DECL_AIRCHANNELTOCONDUIT
#define GEN_PASS_DEF_AIRCHANNELTOCONDUIT
#include "air/Conversion/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return true if this is an air.channel declaration op.
static bool isAirChannelDecl(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.channel";
}

/// Return true if this is air.channel.put (async or blocking).
static bool isAirChannelPut(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.channel.put";
}

/// Return true if this is air.channel.get (async or blocking).
static bool isAirChannelGet(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.channel.get";
}

/// Return true if this is air.wait_all.
static bool isAirWaitAll(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.wait_all";
}

/// Extract the sym_name attribute from a channel declaration op.
/// Returns empty string if not found.
static std::string getSymName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name"))
    return attr.getValue().str();
  return "";
}

/// Extract the channel symbol name from a channel.put / channel.get op.
/// These ops carry a FlatSymbolRefAttr named "chan_name".
static std::string getChanName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("chan_name"))
    return attr.getValue().str();
  // Fallback: look for a symbol ref in any attribute named "chan_name"
  if (auto attr = op->getAttr("chan_name")) {
    if (auto symRef = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr))
      return symRef.getValue().str();
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
      return strAttr.getValue().str();
  }
  return "";
}

/// Collapse a rank≥3 MemRefType to rank-2 by folding all leading dimensions
/// into the first.  For example, memref<2x128x64xbf16> → memref<256x64xbf16>.
/// Returns the original type unchanged if rank ≤ 2 or any dim is dynamic.
static mlir::MemRefType collapseToRank2(mlir::MemRefType mt) {
  if (mt.getRank() <= 2)
    return mt;
  // Fold all dims except the last into a single leading dim.
  int64_t leading = 1;
  for (int64_t i = 0, e = mt.getRank() - 1; i < e; ++i) {
    int64_t d = mt.getDimSize(i);
    if (mlir::ShapedType::isDynamic(d))
      return mt; // cannot collapse dynamic dims; return as-is
    leading *= d;
  }
  int64_t last = mt.getDimSize(mt.getRank() - 1);
  // Drop the original layout (its affine map rank won't match the new shape).
  // Use identity layout (empty MemRefLayoutAttrInterface) for the collapsed
  // type.
  return mlir::MemRefType::get({leading, last}, mt.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               mt.getMemorySpace());
}

/// Try to extract a compile-time integer value from an SSA value defined by
/// an arith constant op (ConstantIndexOp, ConstantIntOp, or the generic
/// arith::ConstantOp with an IntegerAttr).  Returns the value on success,
/// or std::nullopt for truly dynamic (non-constant) values.
///
/// Note: ConstantIndexOp, ConstantIntOp, and ConstantFloatOp are C++ wrapper
/// classes that all share the same MLIR op class (arith::ConstantOp) and the
/// same TypeID.  The dyn_cast<ConstantIndexOp> / dyn_cast<ConstantIntOp>
/// dispatches succeed based on each wrapper's classof() predicate, which
/// inspects the result type of the underlying ConstantOp.
static std::optional<int64_t> tryExtractConstInt(mlir::Value v) {
  mlir::Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  if (auto cOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp))
    return cOp.value();
  if (auto cOp = mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp))
    return cOp.value();
  // Generic arith::ConstantOp fallback: handles integer constants whose result
  // type (e.g., i32, i64) doesn't satisfy ConstantIndexOp or ConstantIntOp's
  // classof().  Use getSExtValue() to correctly handle negative stride/offset
  // constants (e.g., negative strides for reverse iteration).
  if (auto cOp = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp)) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cOp.getValue())) {
      return intAttr.getValue().getSExtValue();
    }
  }
  return std::nullopt;
}

/// Compute num_elems as the product of static integer constant sizes.
/// The sizes are Index-typed SSA values; we look for arith.constant defs.
/// If any size is dynamic (not a constant), returns 0.
static int64_t computeNumElems(mlir::ValueRange sizes) {
  if (sizes.empty())
    return 1; // scalar: 1 element
  int64_t prod = 1;
  for (mlir::Value v : sizes) {
    auto maybeVal = tryExtractConstInt(v);
    if (!maybeVal)
      return 0; // Not a static constant — signal dynamic
    prod *= *maybeVal;
  }
  return prod;
}

/// Extract static integer values from an Index SSA value range.
/// Non-static values are represented as -1 (ShapedType::kDynamic).
static llvm::SmallVector<int64_t> extractStaticInts(mlir::ValueRange vals) {
  llvm::SmallVector<int64_t> result;
  for (mlir::Value v : vals) {
    auto maybeVal = tryExtractConstInt(v);
    result.push_back(maybeVal.value_or(-1));
  }
  return result;
}

// ---------------------------------------------------------------------------
// AIR channel.put / channel.get operand layout
// ---------------------------------------------------------------------------
//
// The AIR assembly format for channel.put:
//   custom<AsyncDependencies>(type($async_token), $async_dependencies)
//   $chan_name `[` ($indices^)? `]`
//   `(` $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)?
//   `]` `)` ...
//
// AttrSizedOperandSegments is set, so the op has:
//   attribute "operandSegmentSizes" : array<i32: ndeps, nidx, 1, noffsets,
//   nsizes, nstrides>
//
// MLIR renamed this from "operand_segment_sizes" (old snake_case) to
// "operandSegmentSizes" (camelCase) as a properties-based inherent attribute.
// We try both names for forward/backward compatibility.
//
// We use this to slice the operand list.

/// Retrieve operand segment sizes from the "operandSegmentSizes" attribute.
static llvm::SmallVector<int32_t> getOperandSegments(mlir::Operation *op) {
  llvm::SmallVector<int32_t> segs;
  // Try current MLIR name first (camelCase, stored as property).
  auto attr = op->getAttrOfType<mlir::DenseI32ArrayAttr>("operandSegmentSizes");
  if (!attr)
    // Fallback: old MLIR name (snake_case, stored in attribute dict).
    attr = op->getAttrOfType<mlir::DenseI32ArrayAttr>("operand_segment_sizes");
  if (attr) {
    for (int32_t v : attr.asArrayRef())
      segs.push_back(v);
  }
  return segs;
}

// ---------------------------------------------------------------------------
// Broadcast Step 2 helpers
// ---------------------------------------------------------------------------

/// Try to find the enclosing aie.core op for an op, and return its tile
/// [col, row] coordinates.  Returns std::nullopt if the op is not inside
/// an aie.core region (e.g., it lives in a plain function or air.herd).
static std::optional<std::pair<int64_t, int64_t>>
tryGetEnclosingCoreTile(mlir::Operation *op) {
  mlir::Operation *parent = op->getParentOp();
  while (parent) {
    if (auto coreOp = mlir::dyn_cast<AIE::CoreOp>(parent)) {
      AIE::TileOp tile = coreOp.getTileOp();
      return std::make_pair((int64_t)tile.getCol(), (int64_t)tile.getRow());
    }
    parent = parent->getParentOp();
  }
  return std::nullopt;
}

/// Find the MemTile in the given column by walking aie.tile ops inside the
/// enclosing aie.device.  Returns a "tile(col,row)" string for the relay tile,
/// or "" if no MemTile is found.
static std::string findMemTileInColumn(mlir::Operation *contextOp,
                                       int64_t col) {
  // Walk up to find the enclosing aie.device.
  auto deviceOp = contextOp->getParentOfType<AIE::DeviceOp>();
  if (!deviceOp) {
    // Try walking down from the module.
    mlir::Operation *parent = contextOp;
    while (parent) {
      parent->walk([&](AIE::DeviceOp d) { deviceOp = d; });
      if (deviceOp)
        break;
      parent = parent->getParentOp();
    }
  }
  if (!deviceOp)
    return "";

  const AIE::AIETargetModel &tm = AIE::getTargetModel(deviceOp);
  std::string result;
  // First, check instantiated TileOps in the column.
  deviceOp.walk([&](AIE::TileOp tileOp) {
    if (result.empty() && (int64_t)tileOp.getCol() == col &&
        tm.isMemTile(tileOp.getCol(), tileOp.getRow())) {
      llvm::raw_string_ostream os(result);
      os << "tile(" << tileOp.getCol() << "," << tileOp.getRow() << ")";
    }
  });
  // Fallback: consult the target model if no TileOp was instantiated.
  if (result.empty()) {
    for (int row = 0; row < tm.rows(); ++row) {
      if (tm.isMemTile(col, row)) {
        llvm::raw_string_ostream os(result);
        os << "tile(" << col << "," << row << ")";
        break;
      }
    }
  }
  return result;
}

/// For an air.channel.put/get op NOT inside aie.core, infer the tile by
/// examining the memref operand's memory space:
///   memory space 1 (L2) → MemTile
///   no memory space / 0 (L3) → shim tile (same col as MemTile, row 0)
///
/// Returns {tileCoord, isShim}.  If no MemTile exists, returns {{}, false}.
static std::pair<std::optional<std::pair<int64_t, int64_t>>, bool>
inferNonCoreTile(mlir::Operation *op) {
  auto deviceOp = op->getParentOfType<AIE::DeviceOp>();
  if (!deviceOp)
    return {{}, false};

  const AIE::AIETargetModel &tm = AIE::getTargetModel(deviceOp);

  // Check memref operand's memory space.
  bool hasL2Memref = false;
  auto segs = getOperandSegments(op);
  if (segs.size() >= 3) {
    int32_t ndeps = segs[0];
    int32_t nidx = segs[1];
    int32_t memrefPos = ndeps + nidx;
    if (memrefPos < static_cast<int32_t>(op->getNumOperands())) {
      mlir::Value memrefVal = op->getOperand(memrefPos);
      if (auto mt = mlir::dyn_cast<mlir::MemRefType>(memrefVal.getType())) {
        if (auto memSpace = mt.getMemorySpace()) {
          if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(memSpace)) {
            if (intAttr.getInt() == 1)
              hasL2Memref = true;
          }
        }
      }
    }
  }

  // Find first MemTile in the device.
  std::optional<std::pair<int64_t, int64_t>> memTile;
  deviceOp.walk([&](AIE::TileOp tileOp) {
    if (!memTile && tm.isMemTile(tileOp.getCol(), tileOp.getRow()))
      memTile = {(int64_t)tileOp.getCol(), (int64_t)tileOp.getRow()};
  });

  if (!memTile)
    return {{}, false};

  if (hasL2Memref)
    return {memTile, false}; // MemTile endpoint
  else
    return std::make_pair(
        std::make_optional(std::make_pair(memTile->first, (int64_t)0)),
        true); // Shim tile endpoint
}

// ---------------------------------------------------------------------------
// Main pass struct
// ---------------------------------------------------------------------------

struct AirChannelToConduitPass
    : impl::AirChannelToConduitBase<AirChannelToConduitPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<ConduitDialect>();
    // AIE dialect needed: Pass B emits aie.put_cascade / aie.get_cascade
    // directly for cascade channels (routing_mode="cascade").
    registry.insert<xilinx::AIE::AIEDialect>();
    // memref and arith needed for cascade load/store in put/get lowering.
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    mlir::MLIRContext *ctx = module.getContext();

    // DMA token type for conduit (put/get_memref_async and wait_all_async
    // all return !conduit.dma.token).
    auto conduitTokenTy = DMATokenType::get(ctx);

    // Collect aie.device scopes: process each independently to avoid
    // channel-name collisions when --air-hierarchy-to-aie emits multiple
    // device blocks with identically-named channel declarations.
    // Post-hierarchy IR (with aie.device blocks) is the only production input.
    llvm::SmallVector<mlir::Operation *> scopes;
    module.walk([&](AIE::DeviceOp d) { scopes.push_back(d.getOperation()); });

    for (mlir::Operation *scopeOp : scopes) {

      // Phase 1: collect air.channel declarations → build name→create map.
      // We'll emit conduit.create for each; element_type filled in Phase 2.

      // Map: channel name → conduit.create op (for later patching of
      // element_type)
      llvm::StringMap<mlir::Operation *> channelCreateOps;

      // Map: broadcast alias name → source channel name.
      // Used to propagate element_type from source to per-consumer aliases
      // after Phase 2b, since put/get ops reference only the source name.
      llvm::StringMap<std::string> aliasToSourceChannel;

      // Collect channel decl ops for deferred erasure.
      llvm::SmallVector<mlir::Operation *> channelDeclsToErase;

      // Collect put/get/wait_all ops for rewriting.
      llvm::SmallVector<mlir::Operation *> putGetToRewrite;
      llvm::SmallVector<mlir::Operation *> waitAllToRewrite;

      // Tile coordinate collection: when air.channel.put/get ops are enclosed
      // in aie.core regions (i.e., after --air-hierarchy-to-aie), collect tile
      // coordinates so conduit.create can carry producer_tile / consumer_tiles.
      // Without these, Pass C cannot allocate buffers/locks/flows.
      //
      // broadcastConsumerTiles: channel name → list of consumer [col, row]
      // pairs
      //   (also used for broadcast Step 2 per-consumer conduit.create aliases).
      // channelProducerTile: channel name → producer [col, row]
      //   (first put op's enclosing core tile wins).
      // channelConsumerTiles: channel name → list of consumer [col, row] pairs
      //   (all get ops, regardless of broadcast).
      llvm::StringMap<llvm::SmallVector<std::pair<int64_t, int64_t>>>
          broadcastConsumerTiles;
      llvm::StringMap<std::pair<int64_t, int64_t>> channelProducerTile;
      llvm::StringMap<llvm::SmallVector<std::pair<int64_t, int64_t>>>
          channelConsumerTiles;

      // Shim endpoint tracking: for channels where put/get ops at device body
      // level reference L3 (external) buffers, record the external buffer SSA
      // values and tile coordinates.
      llvm::StringMap<llvm::SmallVector<mlir::Value>> shimExtBufs;
      llvm::StringMap<std::pair<int64_t, int64_t>> shimTileCoords;

      // Broadcast guard for Phase 6 infer-rates: track which channel names were
      // detected as broadcast channels (broadcast_shape attribute present and
      // product > 1).  Phase 6 skips rate annotation for these channels because
      // their broadcast capacity = product(broadcast_shape) represents fan-out
      // count, not buffer slots.  M7 would misinterpret the inflated capacity
      // as buffer capacity and produce wrong CSDF occupancy checks.
      llvm::StringSet<> broadcastChannelNames;

      // Walk and collect all ops of interest.
      scopeOp->walk([&](mlir::Operation *op) {
        if (isAirChannelDecl(op)) {
          // Only collect declarations that live inside a device body.
          // After --air-hierarchy-to-aie, the renamed @channel_XX declarations
          // are direct children of the aie.device body and pass this check.
          // Module-level original declarations (e.g., @L3ToL2Chan1) are NOT
          // inside any DeviceOp and are excluded here; they are handled by
          // the module-level cleanup after the per-device loop (lines below).
          if (op->getParentOfType<AIE::DeviceOp>() != nullptr)
            channelDeclsToErase.push_back(op);
        } else if (isAirChannelPut(op) || isAirChannelGet(op)) {
          putGetToRewrite.push_back(op);
          std::string chanName = getChanName(op);
          if (!chanName.empty()) {
            if (auto tileCoord = tryGetEnclosingCoreTile(op)) {
              if (isAirChannelPut(op)) {
                // Record producer tile (first put wins).
                if (!channelProducerTile.count(chanName))
                  channelProducerTile[chanName] = *tileCoord;
              } else {
                // Record consumer tile.
                channelConsumerTiles[chanName].push_back(*tileCoord);
                broadcastConsumerTiles[chanName].push_back(*tileCoord);
              }
            } else if (!op->getParentOfType<mlir::func::FuncOp>()) {
              // Op not inside aie.core AND not inside func.func — infer tile
              // from memref memory space.  This handles post-hierarchy IR where
              // device-body-level ops represent MemTile/shim DMA transfers.
              // Skip when inside func.func (pre-hierarchy IR lacks the
              // memory-space annotations needed for reliable inference).
              // L2 (memory space 1) → MemTile; L3 (no memory space) → shim.
              auto [inferred, isShim] = inferNonCoreTile(op);
              if (inferred) {
                if (isAirChannelPut(op)) {
                  if (!channelProducerTile.count(chanName))
                    channelProducerTile[chanName] = *inferred;
                } else {
                  channelConsumerTiles[chanName].push_back(*inferred);
                }
                // Track external buffer for shim-level channels.
                // Only record when the memref operand is an aie.external_buffer
                // (not a function argument or regular memref.alloc).
                if (isShim) {
                  auto opSegs = getOperandSegments(op);
                  if (opSegs.size() >= 3) {
                    int32_t memrefPos = opSegs[0] + opSegs[1];
                    if (memrefPos < (int32_t)op->getNumOperands()) {
                      mlir::Value memrefVal = op->getOperand(memrefPos);
                      if (auto *defOp = memrefVal.getDefiningOp()) {
                        if (mlir::isa<AIE::ExternalBufferOp>(defOp)) {
                          shimExtBufs[chanName].push_back(memrefVal);
                        }
                      }
                    }
                  }
                  shimTileCoords[chanName] = *inferred;
                }
              }
            }
          }
        } else if (isAirWaitAll(op))
          waitAllToRewrite.push_back(op);
      });

      // Phase 1b: pre-scan for existing conduit.create ops so Phase 2 does not
      // emit duplicates when a conduit.create with tile info already exists.
      scopeOp->walk([&](Create existingCreate) {
        auto nameAttr = existingCreate.getName();
        if (!nameAttr.empty()) {
          if (channelCreateOps.count(nameAttr.str())) {
            channelCreateOps[nameAttr.str()]->emitWarning(
                "air-channel-to-conduit: duplicate conduit.create for '")
                << nameAttr.str() << "' — earlier entry replaced";
          }
          channelCreateOps[nameAttr.str()] = existingCreate.getOperation();
        }
      });

      // Phase 2: emit conduit.create for each air.channel declaration.
      for (mlir::Operation *op : channelDeclsToErase) {
        std::string name = getSymName(op);
        if (name.empty()) {
          // No sym_name — skip.
          op->emitWarning("air-channel-to-conduit: could not resolve channel "
                          "name — op dropped");
          continue;
        }

        // Skip if a conduit.create with this name already exists (e.g., one
        // with tile info inserted by a prior placement step).
        if (channelCreateOps.count(name))
          continue;

        builder.setInsertionPoint(op);
        mlir::Location loc = op->getLoc();

        // 5a: Propagate channel_type → routing_mode on conduit.create.
        //   "dma_packet" → routing_mode = Packet
        //   "cascade"    → routing_mode = Cascade
        //   "dma_stream" / absent → leave routing_mode absent (circuit default)
        RoutingModeAttr routingMode{};
        if (auto ctAttr = op->getAttrOfType<mlir::StringAttr>("channel_type")) {
          llvm::StringRef ct = ctAttr.getValue();
          if (ct == "dma_packet") {
            routingMode = RoutingModeAttr::get(ctx, RoutingMode::Packet);
          } else if (ct == "cascade") {
            routingMode = RoutingModeAttr::get(ctx, RoutingMode::Cascade);
          }
          // "dma_stream" → leave routingMode absent (circuit default)
        }

        // 5b: Propagate broadcast_shape → conduit capacity.
        //
        // broadcast_shape = [d0, d1, ...] describes the fan-out topology:
        //   broadcast capacity = product(broadcast_shape) (total number of consumers)
        //
        // Step 2 (broadcast topology):
        //   If consumer tile coordinates are available (i.e., air.channel.get
        //   ops were found inside aie.core regions), emit:
        //     - Per-consumer conduit.create aliases: @name_c0, @name_c1, ...
        //     - conduit.scatter{src=@name, dsts=[@name_c0,...]}
        //   If consumer tile coords are not available (usual case — tile
        //   placement runs before or separately), emit a remark and leave
        //   consumer_tiles empty.
        int64_t broadcastCapacity = 1;
        bool isBroadcast = false;
        if (auto bsAttr = op->getAttr("broadcast_shape")) {
          if (auto denseAttr =
                  mlir::dyn_cast<mlir::DenseI64ArrayAttr>(bsAttr)) {
            for (int64_t dim : denseAttr.asArrayRef())
              broadcastCapacity *= dim;
            isBroadcast = true;
            // Record broadcast channel name so Phase 6 can skip rate
            // annotation.
            broadcastChannelNames.insert(name);
            // Check if we have consumer tile coordinates from aie.core
            // enclosure.
            auto tileIt = broadcastConsumerTiles.find(name);
            bool hasTileCoords = (tileIt != broadcastConsumerTiles.end() &&
                                  !tileIt->second.empty());
            if (hasTileCoords) {
              op->emitRemark()
                  << "air-channel-to-conduit: channel @" << name
                  << " broadcast_shape=" << bsAttr
                  << " → broadcast capacity = " << broadcastCapacity
                  << "; found " << tileIt->second.size()
                  << " consumer tiles from aie.core enclosure; "
                     "emitting conduit.scatter for distribute.";
            } else {
              // No tile coords available — usual case (AIR before AIE
              // lowering).
              op->emitRemark()
                  << "air-channel-to-conduit: channel @" << name
                  << " broadcast_shape=" << bsAttr
                  << " → broadcast capacity = " << broadcastCapacity
                  << "; consumer tile coordinates not available (requires "
                     "tile-placement pre-pass). conduit.create emitted with "
                     "correct capacity; consumer_tiles left empty.";
            }
          } else if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(bsAttr)) {
            // Handle ArrayAttr of IntegerAttr (e.g., [4 : index, 4 : index]).
            bool allInts = true;
            for (auto elem : arrayAttr) {
              if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(elem))
                broadcastCapacity *= intAttr.getInt();
              else {
                allInts = false;
                break;
              }
            }
            if (allInts && !arrayAttr.empty()) {
              isBroadcast = true;
              broadcastChannelNames.insert(name);
              auto tileIt = broadcastConsumerTiles.find(name);
              bool hasTileCoords = (tileIt != broadcastConsumerTiles.end() &&
                                    !tileIt->second.empty());
              if (hasTileCoords) {
                op->emitRemark()
                    << "air-channel-to-conduit: channel @" << name
                    << " broadcast_shape=" << bsAttr
                    << " → broadcast capacity = " << broadcastCapacity
                    << "; found " << tileIt->second.size()
                    << " consumer tiles from aie.core enclosure; "
                       "emitting conduit.scatter for distribute.";
              } else {
                op->emitRemark()
                    << "air-channel-to-conduit: channel @" << name
                    << " broadcast_shape=" << bsAttr
                    << " → broadcast capacity = " << broadcastCapacity
                    << "; consumer tile coordinates not available (requires "
                       "tile-placement pre-pass). conduit.create emitted with "
                       "correct capacity; consumer_tiles left empty.";
              }
            } else {
              op->emitWarning()
                  << "air-channel-to-conduit: channel @" << name
                  << " has broadcast_shape=" << bsAttr
                  << " with non-integer elements; capacity defaulting to 1";
            }
          } else {
            // Non-dense, non-array broadcast_shape: fall back to warning.
            op->emitWarning()
                << "air-channel-to-conduit: channel @" << name
                << " has broadcast_shape=" << bsAttr
                << " in unrecognized format; capacity defaulting to 1";
          }
        }

        // Propagate fusion_group hint from air.channel declaration if present.
        mlir::StringAttr fusionGroupAttr{};
        if (auto fg = op->getAttrOfType<mlir::StringAttr>("fusion_group"))
          fusionGroupAttr = fg;

        // Emit conduit.create for the source (producer) side.
        // element_type will be patched after put/get scan below.
        mlir::Operation *createOp = builder.create<Create>(
            loc, mlir::StringAttr::get(ctx, name),
            /*element_type=*/mlir::TypeAttr{},
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 0),
            /*routing_mode=*/routingMode,
            /*sync_mode=*/SyncModeAttr{},
            /*producer_rates=*/mlir::DenseI64ArrayAttr{},
            /*consumer_rates=*/mlir::DenseI64ArrayAttr{},
            /*fusion_group=*/fusionGroupAttr,
            /*bd_repeat=*/mlir::IntegerAttr{},
            /*dma_repeat=*/mlir::IntegerAttr{},
            /*producer_dimensions=*/nullptr,
            /*consumer_dimensions=*/nullptr);

        channelCreateOps[name] = createOp;

        // Note: producer_tile/consumer_tiles attrs are no longer emitted —
        // tile coordinates are inferred from IR structure via inferAllTiles().
        // Pass C reads tiles from aie.core Acquire/GetMemrefAsync ops and
        // aie.shim_dma_allocation ops.

        // Broadcast Step 2: if consumer tile coordinates are known, emit
        // per-consumer conduit.create aliases and a conduit.scatter.
        //
        // This is only possible when air.channel.get ops appear inside aie.core
        // regions (i.e., after the air-to-aie lowering pass).  In the common
        // case (air dialect before placement), broadcastConsumerTiles is empty
        // and this block is skipped.
        if (isBroadcast) {
          auto tileIt = broadcastConsumerTiles.find(name);
          if (tileIt != broadcastConsumerTiles.end() &&
              !tileIt->second.empty()) {
            // Deduplicate consumer tile coordinates before broadcast Step 2.
            // Post-hierarchy channels (e.g., @channel_XX from
            // --air-hierarchy-to-aie) often have broadcast_shape > 1 but
            // all consumers on the SAME physical tile (temporal multiplexing,
            // not spatial fan-out).  Skip distribute when only 1 unique
            // consumer tile remains — it's effectively point-to-point.
            std::set<std::pair<int64_t, int64_t>> uniqueConsumers;
            for (auto &coord : tileIt->second)
              uniqueConsumers.insert(coord);
            llvm::SmallVector<std::pair<int64_t, int64_t>> consumerCoords(
                uniqueConsumers.begin(), uniqueConsumers.end());

            if (consumerCoords.size() <= 1) {
              // Single consumer tile — no spatial fan-out needed.
              // conduit.create already has capacity and consumer_tiles set;
              // Pass C handles this as a standard DMA channel.
              continue;
            }

            // Build per-consumer conduit names and conduit.create aliases.
            llvm::SmallVector<std::string> dstNames;
            // Emit consumer creates immediately after the source create.
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(createOp);

            // Pre-extract element_type from put/get ops for this channel so
            // per-consumer conduit.create aliases are created with a valid
            // element_type (required attribute since Sprint 6).
            mlir::TypeAttr bcastElemType{};
            for (mlir::Operation *pgOp : putGetToRewrite) {
              if (getChanName(pgOp) != name)
                continue;
              auto pgSegs = getOperandSegments(pgOp);
              if (pgSegs.size() >= 3) {
                int32_t pgMemrefPos = pgSegs[0] + pgSegs[1];
                if (pgMemrefPos <
                    static_cast<int32_t>(pgOp->getNumOperands())) {
                  mlir::Value memrefVal = pgOp->getOperand(pgMemrefPos);
                  if (auto mt = mlir::dyn_cast<mlir::MemRefType>(
                          memrefVal.getType())) {
                    bcastElemType =
                        mlir::TypeAttr::get(collapseToRank2(mt));
                    break;
                  }
                }
              }
            }

            for (size_t i = 0; i < consumerCoords.size(); ++i) {
              std::string dstName = name + "_c" + std::to_string(i);
              dstNames.push_back(dstName);

              // Per-consumer conduit.create (each consumer gets its own
              // independent BD chain).
              auto consCreate = builder.create<Create>(
                  loc, mlir::StringAttr::get(ctx, dstName),
                  /*element_type=*/bcastElemType,
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 0),
                  /*routing_mode=*/routingMode,
                  /*sync_mode=*/SyncModeAttr{},
                  /*producer_rates=*/mlir::DenseI64ArrayAttr{},
                  /*consumer_rates=*/mlir::DenseI64ArrayAttr{},
                  /*fusion_group=*/fusionGroupAttr,
                  /*bd_repeat=*/mlir::IntegerAttr{},
                  /*dma_repeat=*/mlir::IntegerAttr{},
                  /*producer_dimensions=*/nullptr,
                  /*consumer_dimensions=*/nullptr);
              channelCreateOps[dstName] = consCreate.getOperation();
              aliasToSourceChannel[dstName] = name;
            }

            // Group consumers by column so each scatter uses its column-local
            // MemTile instead of routing everything through column 0.
            std::map<int64_t, llvm::SmallVector<size_t>> consumersByCol;
            for (size_t ci = 0; ci < consumerCoords.size(); ++ci)
              consumersByCol[consumerCoords[ci].first].push_back(ci);

            // Emit one conduit.scatter per column group.
            for (auto &[col, indices] : consumersByCol) {
              llvm::SmallVector<mlir::Attribute> dstsAttrs;
              for (size_t idx : indices)
                dstsAttrs.push_back(
                    mlir::FlatSymbolRefAttr::get(ctx, dstNames[idx]));

              // Determine relay MemTile from this column.
              std::string memtileStr =
                  findMemTileInColumn(createOp, col);
              if (memtileStr.empty()) {
                // Fallback: try producer tile column.
                auto prodIt2 = channelProducerTile.find(name);
                if (prodIt2 != channelProducerTile.end())
                  memtileStr =
                      findMemTileInColumn(createOp, prodIt2->second.first);
              }
              builder.create<ScatterOp>(
                  loc, mlir::FlatSymbolRefAttr::get(ctx, name),
                  mlir::ArrayAttr::get(ctx, dstsAttrs),
                  mlir::StringAttr::get(ctx, memtileStr),
                  /*offsets=*/mlir::DenseI64ArrayAttr{});
            }
          }
        }
      }

      // Phase 2b: scan put/get ops to extract element_type for conduit.create.
      //
      // Two-pass strategy: prefer L2 consumer (get) buffer types over L3
      // producer (put) source types.  For L3→L2 relay channels, the put's
      // memref is the full DRAM tensor (e.g. memref<768x64xbf16> = 96KB)
      // while the get's L2 buffer is the per-tile relay size (e.g.
      // memref<96x64xbf16> = 12KB).  Using the L2 size matches the original
      // air design's explicit buffer allocation and avoids MemTile overflow.
      //
      // Helper lambda: extract the memref type from a put/get op.
      auto extractMemRefType =
          [](mlir::Operation *op) -> std::optional<mlir::MemRefType> {
        auto segs = getOperandSegments(op);
        if (segs.size() >= 3) {
          int32_t ndeps = segs[0];
          int32_t nidx = segs[1];
          int32_t memrefPos = ndeps + nidx;
          if (static_cast<int32_t>(op->getNumOperands()) > memrefPos) {
            mlir::Value memrefVal = op->getOperand(memrefPos);
            if (auto mt =
                    mlir::dyn_cast<mlir::MemRefType>(memrefVal.getType()))
              return mt;
          }
        } else if (op->getNumOperands() >= 1) {
          if (auto mt =
                  mlir::dyn_cast<mlir::MemRefType>(op->getOperand(0).getType()))
            return mt;
        }
        return std::nullopt;
      };

      // Pass 1: scan get ops with L2 memory space (1) — preferred source for
      // element_type because these are the actual MemTile relay buffers.
      for (mlir::Operation *op : putGetToRewrite) {
        if (!isAirChannelGet(op))
          continue;
        std::string chanName = getChanName(op);
        if (chanName.empty())
          continue;
        auto it = channelCreateOps.find(chanName);
        if (it == channelCreateOps.end())
          continue;
        auto createTypedOp =
            mlir::dyn_cast<Create>(it->second);
        if (!createTypedOp || createTypedOp.getElementTypeAttr())
          continue;
        auto optMt = extractMemRefType(op);
        if (!optMt)
          continue;
        mlir::MemRefType mt = *optMt;
        // Only prefer get ops whose buffer lives in L2 (memory space 1).
        if (auto msAttr =
                mlir::dyn_cast_or_null<mlir::IntegerAttr>(mt.getMemorySpace()))
          if (msAttr.getInt() == 1) {
            mlir::MemRefType collapsed = collapseToRank2(mt);
            createTypedOp.setElementTypeAttr(
                mlir::TypeAttr::get(collapsed));
          }
      }

      // Pass 1.5: scan put ops with L2 memory space (1) for output channels.
      //
      // Output channels (L2→L3, e.g. MemTile→DRAM) have get ops in L3 space
      // and put ops in L2 space.  Pass 1 only checks get ops, so it misses
      // these.  The put op's L2 memref carries the correct relay tile size
      // (e.g., memref<96x64xbf16>) instead of the full L3 output tensor size
      // (e.g., memref<4096x64xbf16>) which would cause MemTile overflow.
      for (mlir::Operation *op : putGetToRewrite) {
        if (!isAirChannelPut(op))
          continue;
        std::string chanName = getChanName(op);
        if (chanName.empty())
          continue;
        auto it = channelCreateOps.find(chanName);
        if (it == channelCreateOps.end())
          continue;
        auto createTypedOp =
            mlir::dyn_cast<Create>(it->second);
        if (!createTypedOp || createTypedOp.getElementTypeAttr())
          continue;
        auto optMt = extractMemRefType(op);
        if (!optMt)
          continue;
        mlir::MemRefType mt = *optMt;
        // Only use put ops whose buffer lives in L2 (memory space 1).
        if (auto msAttr =
                mlir::dyn_cast_or_null<mlir::IntegerAttr>(mt.getMemorySpace()))
          if (msAttr.getInt() == 1) {
            mlir::MemRefType collapsed = collapseToRank2(mt);
            createTypedOp.setElementTypeAttr(
                mlir::TypeAttr::get(collapsed));
          }
      }

      // Pass 2: for channels still unpatched, fall back to any put/get op
      // (preserves the original first-encountered behavior).
      for (mlir::Operation *op : putGetToRewrite) {
        std::string chanName = getChanName(op);
        if (chanName.empty())
          continue;
        auto it = channelCreateOps.find(chanName);
        if (it == channelCreateOps.end())
          continue;
        auto createTypedOp =
            mlir::dyn_cast<Create>(it->second);
        if (!createTypedOp || createTypedOp.getElementTypeAttr())
          continue;
        auto optMt = extractMemRefType(op);
        if (!optMt)
          continue;
        mlir::MemRefType collapsed = collapseToRank2(*optMt);
        createTypedOp.setElementTypeAttr(mlir::TypeAttr::get(collapsed));
      }

      // Phase 2b.5: propagate element_type from source channels to broadcast
      // consumer aliases.  Per-consumer conduit.create ops are emitted with a
      // null TypeAttr because the element_type is not yet known at emission
      // time.  Phase 2b patches the source channel (referenced by put/get ops)
      // but never patches the aliases (not referenced by any put/get).
      for (auto &kv : aliasToSourceChannel) {
        llvm::StringRef aliasName = kv.first();
        const std::string &srcName = kv.second;

        auto aliasIt = channelCreateOps.find(aliasName);
        auto srcIt = channelCreateOps.find(srcName);
        if (aliasIt == channelCreateOps.end() ||
            srcIt == channelCreateOps.end())
          continue;

        auto aliasCreate = mlir::dyn_cast<Create>(aliasIt->second);
        auto srcCreate = mlir::dyn_cast<Create>(srcIt->second);
        if (!aliasCreate || !srcCreate)
          continue;

        // Only propagate if source has been patched and alias hasn't.
        if (aliasCreate.getElementTypeAttr() || !srcCreate.getElementTypeAttr())
          continue;

        aliasCreate.setElementTypeAttr(srcCreate.getElementTypeAttr());
      }

      // Phase 2b.7: Persist inferred tile coordinates as discardable
      // attributes on conduit.create ops for Source 8 in inferAllTiles().
      //
      // Pass B has tile info from enclosing aie.core (tryGetEnclosingCoreTile)
      // and memory-space-based inference (inferNonCoreTile) that is lost when
      // the original air.channel put/get ops are erased in Phase 3/5.
      // Storing these as attrs on conduit.create allows inferAllTiles() to
      // recover producer/consumer tile coordinates for channels that have
      // no Acquire/Release, no conduit_channel, and no dma_channel_group.
      for (auto &[chName, createOp] : channelCreateOps) {
        auto prodIt = channelProducerTile.find(chName);
        if (prodIt != channelProducerTile.end()) {
          auto [col, row] = prodIt->second;
          createOp->setAttr("air_producer_tile",
                            mlir::DenseI64ArrayAttr::get(ctx, {col, row}));
        }
        auto consIt = channelConsumerTiles.find(chName);
        if (consIt != channelConsumerTiles.end() && !consIt->second.empty()) {
          llvm::SmallVector<int64_t> coords;
          for (auto &[col, row] : consIt->second) {
            coords.push_back(col);
            coords.push_back(row);
          }
          createOp->setAttr("air_consumer_tiles",
                            mlir::DenseI64ArrayAttr::get(ctx, coords));
        }
      }

      // Phase 2c: create shim aie.tile ops for channels with shim endpoints
      // (producer or consumer at row 0).
      //
      // After --air-hierarchy-to-aie, shim tiles (row 0) are NOT created by
      // that pass — only compute tiles (row >= 2) and MemTiles (row 1) exist.
      // Pass C needs shim tiles for shim DMA allocation, so we create them here.
      {
        AIE::DeviceOp deviceOp;
        if (auto d = mlir::dyn_cast<AIE::DeviceOp>(scopeOp))
          deviceOp = d;
        else
          scopeOp->walk([&](AIE::DeviceOp d) {
            if (!deviceOp)
              deviceOp = d;
          });

        if (deviceOp && !shimExtBufs.empty()) {
          // Build existing tile cache.
          llvm::DenseMap<std::pair<int64_t, int64_t>, AIE::TileOp> tileCache;
          deviceOp.walk(
              [&](AIE::TileOp t) { tileCache[{t.getCol(), t.getRow()}] = t; });

          // Collect unique shim tile coords needed.
          std::set<std::pair<int64_t, int64_t>> shimTileCoordsNeeded;
          for (auto &[name, coord] : shimTileCoords)
            shimTileCoordsNeeded.insert(coord);

          // Create missing shim tiles.
          for (auto &[col, row] : shimTileCoordsNeeded) {
            if (!tileCache.count({col, row})) {
              // Insert shim tile after existing tiles in the device body.
              mlir::Operation *insertAfter = nullptr;
              deviceOp.walk([&](AIE::TileOp t) { insertAfter = t; });
              if (insertAfter)
                builder.setInsertionPointAfter(insertAfter);
              else
                builder.setInsertionPointToStart(
                    &deviceOp.getBodyRegion().front());
              auto shimTile =
                  builder.create<AIE::TileOp>(deviceOp.getLoc(), col, row);
              tileCache[{col, row}] = shimTile;
            }
          }

        }
      }

      // Phase 2d: merge shim relay channels.
      //
      // After --air-hierarchy-to-aie, a single L3→L2 channel array (e.g.,
      // L3ToL2Chan1 [1,4]) gets expanded into multiple individual channels
      // (channel_34, channel_36, channel_38, channel_40).  Each maps to the
      // same shim→MemTile endpoint pair.  With 4 K channels + 4 V channels,
      // the MemTile needs 8 S2MM channels, exceeding the AIE2 limit of 6.
      //
      // Fix: merge channels that share the same (shim producer tile,
      // external buffer) into one canonical channel.  The surviving channel
      // accumulates all put/get ops, which Pass C lowers into a BD chain
      // on a single physical DMA channel.
      llvm::StringMap<std::string> channelMergeMap;
      {
        // Group merge candidates by external buffer SSA value.
        // Only consider channels where the producer is a shim tile (row 0).
        llvm::DenseMap<mlir::Value, llvm::SmallVector<std::string>>
            extBufGroups;
        for (auto &[name, bufs] : shimExtBufs) {
          auto coordIt = shimTileCoords.find(name);
          if (coordIt == shimTileCoords.end() || coordIt->second.second != 0)
            continue; // Not a shim producer.
          // Also verify this channel has a MemTile consumer (row 1).
          auto consIt = channelConsumerTiles.find(name);
          if (consIt == channelConsumerTiles.end() || consIt->second.empty() ||
              consIt->second[0].second != 1)
            continue; // Consumer is not a MemTile.
          if (bufs.empty())
            continue;
          extBufGroups[bufs[0]].push_back(name.str());
        }

        for (auto &[extBuf, names] : extBufGroups) {
          if (names.size() <= 1)
            continue;
          std::sort(names.begin(), names.end());
          std::string canonical = names[0];

          for (size_t i = 1; i < names.size(); ++i) {
            channelMergeMap[names[i]] = canonical;

            // Erase duplicate conduit.create.
            auto createIt = channelCreateOps.find(names[i]);
            if (createIt != channelCreateOps.end()) {
              createIt->second->erase();
              channelCreateOps.erase(createIt);
            }
          }

          scopeOp->emitRemark()
              << "air-channel-to-conduit: merged shim relay channels ["
              << llvm::join(names, ", ") << "] into canonical @" << canonical
              << " (reduces MemTile S2MM from " << names.size() << " to 1)";
        }

      }

      // Phase 3: rewrite air.channel.put / air.channel.get → conduit
      // put/get_memref_async.
      //
      // SSA threading:
      //   The original air.channel.put result is !air.async.token (opaque in
      //   aie-opt).  We replace all uses with the new !conduit.dma.token.
      //
      // PASSB-DEP-001 fix: air.wait_all fan-in tokens as deps.
      //   When a dep token is still !air.async.token but its defining op is
      //   air.wait_all, Phase 4 has not yet run and the wait_all result is not
      //   yet a DMATokenType.  However, the wait_all's OWN operands ARE already
      //   conduit tokens (the put/get ops that fed them have already been
      //   processed and replaceAllUsesWith was called).
      //
      //   Fix: when we encounter an !air.async.token dep whose defining op is
      //   air.wait_all, we pre-emit a conduit.wait_all_async over the resolved
      //   sub-tokens right here in Phase 3.  The result is used as the dep.
      //   We record the pre-emitted op in preEmittedWaitAll so Phase 4 skips
      //   re-emitting it (Phase 4 uses the pre-emitted result for
      //   replaceAllUsesWith).
      //
      //   This is recursive up to one level: if the wait_all's operands are
      //   themselves !air.async.token (e.g., from air.execute), those are still
      //   dropped (same as before).  Only DMATokenType operands are threaded.

      // Map: air.wait_all op → the conduit.wait_all_async value pre-emitted for
      // it. Phase 4 uses this to skip re-emitting and to replace uses
      // correctly.
      llvm::DenseMap<mlir::Operation *, mlir::Value> preEmittedWaitAll;

      llvm::SmallVector<mlir::Operation *> putGetToErase;

      for (mlir::Operation *op : putGetToRewrite) {
        std::string chanName = getChanName(op);
        // Apply merge map: redirect merged channel names to canonical.
        auto mergeIt = channelMergeMap.find(chanName);
        if (mergeIt != channelMergeMap.end())
          chanName = mergeIt->second;
        if (chanName.empty()) {
          // Cannot identify channel — skip.
          op->emitWarning("air-channel-to-conduit: could not resolve channel "
                          "name — op dropped");
          continue;
        }

        builder.setInsertionPoint(op);
        mlir::Location loc = op->getLoc();

        // Decode operand segments.
        auto segs = getOperandSegments(op);
        // Expected: [ndeps, nidx, 1 (memref), noffsets, nsizes, nstrides]
        int32_t ndeps = 0, nidx = 0, noffsets = 0, nsizes = 0, nstrides = 0;
        if (segs.size() >= 6) {
          ndeps = segs[0];
          nidx = segs[1];
          // segs[2] == 1 (memref)
          noffsets = segs[3];
          nsizes = segs[4];
          nstrides = segs[5];
        }

        // Slice operands.
        mlir::OperandRange allOps = op->getOperands();
        int32_t base = 0;
        // Collect async dependency tokens that have already been converted to
        // conduit DMA token types.  Only DMATokenType (and the AIR async token,
        // if present) are valid DMA dependency operands — WindowTokenType is
        // a window-slot token and must not appear in DMA dep chains.
        //
        // PASSB-DEP-001: also handle air.wait_all results (still
        // !air.async.token at this point).  Recursively resolve their operands
        // into DMATokenType and pre-emit a conduit.wait_all_async in-place.
        llvm::SmallVector<mlir::Value> depTokens;
        for (int32_t i = 0; i < ndeps; ++i) {
          mlir::Value dep = allOps[base + i];
          if (mlir::isa<DMATokenType>(dep.getType())) {
            // Already a conduit token (prior put/get was rewritten in program
            // order).
            depTokens.push_back(dep);
          } else if (mlir::Operation *defOp = dep.getDefiningOp();
                     defOp && isAirWaitAll(defOp)) {
            // PASSB-DEP-001: dep comes from an air.wait_all that Phase 4 hasn't
            // rewritten yet.  Resolve the wait_all's operands (which are
            // already conduit tokens via earlier replaceAllUsesWith calls) and
            // pre-emit a conduit.wait_all_async to represent the fan-in fence.
            //
            // Check if we already pre-emitted a wait_all_async for this op.
            auto preIt = preEmittedWaitAll.find(defOp);
            if (preIt != preEmittedWaitAll.end()) {
              // Reuse previously pre-emitted result.
              depTokens.push_back(preIt->second);
            } else {
              // Collect sub-tokens: only DMATokenType operands of the wait_all
              // are forwarded; non-DMA tokens (e.g., air.execute results) are
              // still dropped (same policy as the primary filter above).
              llvm::SmallVector<mlir::Value> subTokens;
              for (mlir::Value subDep : defOp->getOperands()) {
                if (mlir::isa<DMATokenType>(subDep.getType()))
                  subTokens.push_back(subDep);
              }
              if (!subTokens.empty()) {
                // Pre-emit conduit.wait_all_async immediately before the
                // current put/get op so the SSA value dominates it.
                auto preWait = builder.create<WaitAllAsync>(
                    defOp->getLoc(), conduitTokenTy, subTokens);
                mlir::Value preWaitVal = preWait.getResult();
                preEmittedWaitAll[defOp] = preWaitVal;
                depTokens.push_back(preWaitVal);
              }
              // If subTokens is empty (all operands were non-DMA), nothing to
              // thread — the dep is dropped (same as non-wait_all
              // !air.async.token).
            }
          }
          // else: non-DMA token (air.execute, etc.) — silently dropped per
          // documented limitation.
        }
        base += ndeps;
        // indices (ignored — only [1,1] channels supported)
        if (nidx > 0)
          op->emitWarning()
              << "AirChannelToConduit: " << nidx
              << " multi-dimensional channel index operand(s) dropped for @"
              << chanName << "; only [1,1] scalar channels are supported";
        base += nidx;
        // memref
        mlir::ValueRange offsetsRange, sizesRange, stridesRange;
        mlir::MemRefType memrefType = nullptr;
        if (static_cast<int32_t>(allOps.size()) >=
            base + 1 + noffsets + nsizes + nstrides) {
          // B-7: Handle rank≥3 memref operands by collapsing leading
          // dimensions. air.channel uses a flat 1-D view (offsets/sizes/strides
          // are scalar); the memref shape only matters for element_type
          // patching in Phase 2b. Collapse memref<A×B×C×T> → memref<(A*B)×C×T>
          // → ... → memref<N×M×T> so that the element_type stored on
          // conduit.create is rank-2.
          mlir::Value memrefOperand = allOps[base];
          memrefType =
              mlir::dyn_cast<mlir::MemRefType>(memrefOperand.getType());
          if (memrefType) {
            if (memrefType.getRank() >= 3) {
              op->emitWarning("air-channel-to-conduit: rank-")
                  << memrefType.getRank() << " memref operand for @" << chanName
                  << " collapsed to rank-2 for element_type patching "
                     "(leading dims folded into first dim)";
            }
          }
          base += 1; // skip memref operand itself
          offsetsRange = allOps.slice(base, noffsets);
          base += noffsets;
          sizesRange = allOps.slice(base, nsizes);
          base += nsizes;
          stridesRange = allOps.slice(base, nstrides);
        }

        // Compute num_elems from static sizes.
        // PassB-empty-sizes fix: when sizesRange is empty (nsizes==0), the
        // full buffer should be transferred.  Use the memref type's total
        // element count instead of defaulting to 1.
        int64_t numElems = computeNumElems(sizesRange);
        if (sizesRange.empty() && memrefType && memrefType.hasStaticShape()) {
          numElems = memrefType.getNumElements();
        } else if (numElems == 0 ||
                   (sizesRange.empty() && memrefType &&
                    !memrefType.hasStaticShape())) {
          op->emitError(
              "dynamic channel size not supported in "
              "--air-channel-to-conduit; run --air-hierarchy-to-aie "
              "--air-split-devices before this pass to specialize channel "
              "sizes");
          signalPassFailure();
          continue;
        }

        // Determine routing mode first — cascade channels skip the
        // dynamic-strides check because offsets/sizes/strides are irrelevant
        // (we only load/store element[0] of the memref, regardless of the DMA
        // descriptor).
        bool isCascade = false;
        {
          auto it = channelCreateOps.find(chanName);
          if (it != channelCreateOps.end()) {
            if (auto createTypedOp = mlir::dyn_cast<Create>(it->second)) {
              auto rmOpt = createTypedOp.getRoutingMode();
              if (rmOpt && *rmOpt == RoutingMode::Cascade)
                isCascade = true;
            }
          }
        }

        // Extract static values for the structured attrs (DMA path only).
        auto offsetVals = extractStaticInts(offsetsRange);
        auto sizeVals = extractStaticInts(sizesRange);
        auto strideVals = extractStaticInts(stridesRange);

        // Check for dynamic (non-constant) offset/size/stride values.
        // Dynamic values cannot be lowered correctly for DMA channels — the
        // emitted BD descriptors would use placeholder values producing silent
        // data corruption.  Cascade channels are exempt: they ignore
        // offsets/sizes/strides entirely (only element[0] is transferred).
        if (!isCascade) {
          bool hasDynamic = false;
          for (auto &v : offsetVals) {
            if (v < 0) {
              hasDynamic = true;
            }
          }
          for (auto &v : sizeVals) {
            if (v < 0) {
              hasDynamic = true;
            }
          }
          for (auto &v : strideVals) {
            if (v < 0) {
              hasDynamic = true;
            }
          }
          if (hasDynamic) {
            op->emitError()
                << "air-channel-to-conduit: channel @" << chanName
                << " has dynamic offset/size/stride operands (e.g., loop IVs "
                   "or "
                   "block arguments) that cannot be extracted statically; "
                   "placeholder substitution would produce incorrect DMA "
                   "descriptors and silent data corruption on hardware";
            putGetToErase.push_back(op);
            signalPassFailure();
            continue;
          }
        }

        // For cascade channels, warn on non-trivial (non-zero) constant offsets
        // and non-trivial sizes/strides, and error on fully dynamic operands.
        // Only element[0] is ever transferred; strided slices are silently
        // wrong without these diagnostics.
        //
        // NOTE: the error flag cascadeHadError is checked AFTER all three loops
        // so that we can report all bad operands in one pass before skipping
        // the op. Using `continue` inside the inner loops would only skip to
        // the next element in that loop — it would NOT skip the outer
        // putGetToRewrite loop.
        bool cascadeHadError = false;
        if (isCascade) {
          // Check offsets: warn if constant non-zero, error if dynamic.
          for (mlir::Value v : offsetsRange) {
            auto maybeVal = tryExtractConstInt(v);
            if (!maybeVal) {
              op->emitError()
                  << "air-channel-to-conduit: cascade channel @" << chanName
                  << " has a fully dynamic offset operand; intent cannot be "
                     "inferred — only element[0] will be transferred "
                     "regardless";
              signalPassFailure();
              cascadeHadError = true;
            } else if (*maybeVal != 0) {
              op->emitWarning()
                  << "air-channel-to-conduit: cascade channel @" << chanName
                  << " has a non-zero offset (" << *maybeVal
                  << "); only element[0] will be transferred — "
                     "strided-slice semantics are not supported for cascade";
            }
          }

          // Check sizes: warn if non-trivial constant (not matching full
          // element), error if dynamic.
          for (mlir::Value v : sizesRange) {
            auto maybeVal = tryExtractConstInt(v);
            if (!maybeVal) {
              op->emitError()
                  << "air-channel-to-conduit: cascade channel @" << chanName
                  << " has a fully dynamic size operand; intent cannot be "
                     "inferred — only element[0] will be transferred "
                     "regardless";
              signalPassFailure();
              cascadeHadError = true;
            } else if (*maybeVal != 1) {
              // Size of 1 is trivial (single-element cascade); warn on larger.
              op->emitWarning()
                  << "air-channel-to-conduit: cascade channel @" << chanName
                  << " has a non-unit size (" << *maybeVal
                  << "); only element[0] will be transferred — "
                     "multi-element cascade slices are not supported";
            }
          }

          // Check strides: warn if non-trivial constant, error if dynamic.
          for (mlir::Value v : stridesRange) {
            auto maybeVal = tryExtractConstInt(v);
            if (!maybeVal) {
              op->emitError()
                  << "air-channel-to-conduit: cascade channel @" << chanName
                  << " has a fully dynamic stride operand; intent cannot be "
                     "inferred — only element[0] will be transferred "
                     "regardless";
              signalPassFailure();
              cascadeHadError = true;
            } else if (*maybeVal != 1) {
              // Stride of 1 is the trivial (unit) stride; warn on others.
              op->emitWarning()
                  << "air-channel-to-conduit: cascade channel @" << chanName
                  << " has a non-unit stride (" << *maybeVal
                  << "); only element[0] will be transferred — "
                     "non-unit strides are not supported for cascade";
            }
          }
        }
        // Skip this op entirely if a cascade error was signaled.
        // Still add to putGetToErase so Phase 5 channel-decl cleanup doesn't
        // find stale AIR uses and emit a spurious "remaining uses" error.
        if (cascadeHadError) {
          putGetToErase.push_back(op);
          continue;
        }

        bool isPut = isAirChannelPut(op);
        mlir::Operation *newOp = nullptr;

        if (isCascade) {
          // Cascade channels: the kernel (e.g. attn.cc) manages cascade data
          // movement via C++ intrinsics (get_scd/put_scd loops) — these are not
          // expressed in MLIR IR at all. The routing connection is established
          // by routing_mode="cascade" on conduit.create, which Pass C converts
          // to aie.cascade_flow (zero locks, zero DMA budget). No
          // aie.put_cascade or aie.get_cascade ops are emitted here; just erase
          // the air.channel op.
          putGetToErase.push_back(op);
          continue;
        } else {
          // Normal DMA path: emit conduit put_memref_async or get_memref_async.
          if (isPut) {
            newOp = builder.create<PutMemrefAsync>(
                loc, conduitTokenTy,
                mlir::FlatSymbolRefAttr::get(ctx, chanName),
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       numElems),
                mlir::DenseI64ArrayAttr::get(ctx, offsetVals),
                mlir::DenseI64ArrayAttr::get(ctx, sizeVals),
                mlir::DenseI64ArrayAttr::get(ctx, strideVals), depTokens,
                /*producer_dimensions=*/mlir::Attribute{});
          } else {
            newOp = builder.create<GetMemrefAsync>(
                loc, conduitTokenTy,
                mlir::FlatSymbolRefAttr::get(ctx, chanName),
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       numElems),
                mlir::DenseI64ArrayAttr::get(ctx, offsetVals),
                mlir::DenseI64ArrayAttr::get(ctx, sizeVals),
                mlir::DenseI64ArrayAttr::get(ctx, strideVals), depTokens,
                /*consumer_dimensions=*/mlir::Attribute{});
          }

          // Replace all uses of the old async token result with the new token.
          if (op->getNumResults() >= 1 && newOp->getNumResults() >= 1)
            op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
        }

        putGetToErase.push_back(op);
      }

      // NOTE: putGetToErase erasure is deferred to after Phase 4 (wait_all
      // processing).  Erasing put/get ops here would crash when an errored op's
      // async token result is still used by an air.wait_all operand (the
      // error path skips replaceAllUsesWith, leaving dangling uses).  Phase 4
      // rewrites and erases the wait_all ops first, which removes those uses.

      // Phase 4: rewrite air.wait_all → conduit.wait_all /
      // conduit.wait_all_async.
      //
      // air.wait_all has:
      //   args: variadic !air.async.token (async_dependencies)
      //   result (optional): !air.async.token (async_token, present when async)
      //
      // Mapping:
      //   result present → conduit.wait_all_async %deps : (...) ->
      //   !conduit.dma.token no result      → conduit.wait_all %deps
      //
      // PASSB-DEP-001: if an air.wait_all was already pre-emitted in Phase 3
      // (because it appeared as a dep of an air.channel.put/get), skip
      // re-emitting it.  Instead, call replaceAllUsesWith to forward any
      // remaining uses of the original !air.async.token result to the
      // pre-emitted conduit token.

      llvm::SmallVector<mlir::Operation *> waitAllToErase;

      for (mlir::Operation *op : waitAllToRewrite) {
        // PASSB-DEP-001: skip air.wait_all ops that were already pre-emitted in
        // Phase 3 as part of dep resolution for a put/get op.  Their result was
        // already replaced via the pre-emitted WaitAllAsync; we only need to
        // update any remaining uses of the original air.async.token result.
        auto preIt = preEmittedWaitAll.find(op);
        if (preIt != preEmittedWaitAll.end()) {
          // Forward any remaining !air.async.token uses to the pre-emitted
          // token.
          if (op->getNumResults() >= 1)
            op->getResult(0).replaceAllUsesWith(preIt->second);
          waitAllToErase.push_back(op);
          continue;
        }

        builder.setInsertionPoint(op);
        mlir::Location loc = op->getLoc();

        // Filter operands: only pass conduit token types to conduit.wait_all.
        // Non-conduit types (e.g., residual !air.async.token or i1) are
        // dropped.
        llvm::SmallVector<mlir::Value> conduitDeps;
        for (mlir::Value dep : op->getOperands())
          if (mlir::isa<DMATokenType, WindowTokenType>(dep.getType()))
            conduitDeps.push_back(dep);

        bool hasResult = (op->getNumResults() >= 1);

        if (hasResult) {
          auto newOp =
              builder.create<WaitAllAsync>(loc, conduitTokenTy, conduitDeps);
          op->getResult(0).replaceAllUsesWith(newOp.getResult());
        } else {
          builder.create<WaitAll>(loc, conduitDeps);
        }

        waitAllToErase.push_back(op);
      }

      for (mlir::Operation *op : waitAllToErase)
        op->erase();

      // Erase original put/get ops (deferred from after Phase 3 — see note
      // above). Reverse order: later ops may reference earlier ops' results as
      // deps. Erasing later ops first removes those uses before we erase the
      // defining op.
      for (mlir::Operation *op : llvm::reverse(putGetToErase))
        op->erase();

      // Phase 5: erase air.channel declaration ops (after all put/get refs are
      // gone).
      //
      // After Phase 3+4, the only remaining references to a channel symbol are
      // in conduit ops (conduit.put_memref_async {name = @chan}, etc.) —
      // symbolKnownUseEmpty counts these, but they are EXPECTED: conduit.create
      // @chan already exists and takes over as the canonical symbol definition
      // once air.channel @chan is erased.  Erasing air.channel while conduit
      // ops still reference @chan is safe because conduit.create @chan is the
      // new authoritative definition.
      //
      // We only error if old AIR put/get ops still reference the channel, which
      // indicates a rewrite failure (not the normal conduit-reference case).
      for (mlir::Operation *op : channelDeclsToErase) {
        if (auto symOp = mlir::dyn_cast<mlir::SymbolOpInterface>(op)) {
          llvm::StringRef name = symOp.getNameAttr().getValue();
          // Check for surviving AIR op uses only — conduit uses are expected.
          bool hasAirUses = false;
          scopeOp->walk([&](mlir::Operation *user) {
            if ((isAirChannelPut(user) || isAirChannelGet(user)) &&
                getChanName(user) == name)
              hasAirUses = true;
          });
          if (hasAirUses) {
            op->emitError("air-channel-to-conduit: channel decl '")
                << name << "' has remaining uses after rewrite — cannot erase";
            signalPassFailure();
            continue;
          }
        }
        op->erase();
      }

      // Phase 6: optionally run conduit-infer-rates inline.
      //
      // When inferRates is true (the default), walk the module and attach
      // producer_rates / consumer_rates to conduit.create ops that have no
      // explicit rates, by collecting num_elems from put_memref_async /
      // get_memref_async ops in program order.  This mirrors what the
      // standalone --conduit-infer-rates pass does, but avoids requiring the
      // user to add an extra pass flag.
      //
      // Set --air-channel-to-conduit-infer-rates=false to opt out.
      if (inferRates) {
        // Collect num_elems sequences for each channel name.
        llvm::StringMap<llvm::SmallVector<int64_t>> putElemsMap;
        llvm::StringMap<llvm::SmallVector<int64_t>> getElemsMap;
        llvm::StringMap<bool> hasDynElems;

        scopeOp->walk([&](PutMemrefAsync op) {
          auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
          if (!nameAttr)
            return;
          llvm::StringRef name = nameAttr.getValue();
          if (hasDynElems.count(name))
            return;
          auto ne = op->getAttrOfType<mlir::IntegerAttr>("num_elems");
          if (!ne) {
            hasDynElems[name] = true;
            return;
          }
          putElemsMap[name].push_back(ne.getInt());
        });

        scopeOp->walk([&](GetMemrefAsync op) {
          auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
          if (!nameAttr)
            return;
          llvm::StringRef name = nameAttr.getValue();
          if (hasDynElems.count(name))
            return;
          auto ne = op->getAttrOfType<mlir::IntegerAttr>("num_elems");
          if (!ne) {
            hasDynElems[name] = true;
            return;
          }
          getElemsMap[name].push_back(ne.getInt());
        });

        scopeOp->walk([&](Create op) {
          if (op.getProducerRates().has_value() ||
              op.getConsumerRates().has_value())
            return;
          llvm::StringRef name = op.getSymName();
          if (name.empty())
            return;
          // Broadcast guard: skip rate annotation for broadcast channels.
          // Their capacity = product(broadcast_shape) is a fan-out count, not
          // buffer slots; M7 would misinterpret it.
          if (broadcastChannelNames.count(name))
            return;
          if (hasDynElems.count(name))
            return;
          auto pIt = putElemsMap.find(name);
          auto gIt = getElemsMap.find(name);
          bool hasPuts = (pIt != putElemsMap.end() && !pIt->second.empty());
          bool hasGets = (gIt != getElemsMap.end() && !gIt->second.empty());
          if (!hasPuts || !hasGets)
            return;
          // MVE-2: sliding-window guard.
          // If max(get num_elems) > min(put num_elems), the consumer fetches
          // more elements per step than the producer sends — sliding-window
          // pattern. Attaching rates would give M6 an unbalanced consumer rate,
          // causing a false rejection.  Skip rate annotation and emit a remark.
          int64_t maxGet = *llvm::max_element(gIt->second);
          int64_t minPut = *llvm::min_element(pIt->second);
          if (maxGet > minPut) {
            op->emitRemark(
                "conduit-air-channel: skipping CSDF rate annotation for "
                "sliding-window channel '")
                << name << "' (max get_elems=" << maxGet
                << " > min put_elems=" << minPut
                << "); use explicit producer_rates/consumer_rates with "
                   "window_size";
            return;
          }
          op->setAttr("producer_rates",
                      mlir::DenseI64ArrayAttr::get(ctx, pIt->second));
          op->setAttr("consumer_rates",
                      mlir::DenseI64ArrayAttr::get(ctx, gIt->second));
        });
      }

      // Phase 6.5: Set dma_channel_group on packet-mode conduit.create ops
      // that target the same consumer tile.
      //
      // When multiple packet-mode channels share the same consumer tile, they
      // must share one physical S2MM DMA port (differentiated by packet_id in
      // BD headers).  Pass C uses the dma_channel_group attribute to group
      // these channels onto one S2MM port instead of allocating separate ports
      // per channel (which would exhaust the 2-port S2MM budget on compute
      // tiles).
      //
      // Group key: "pkt_{col}_{row}" for each unique consumer tile coordinate.
      // Non-packet and cascade channels are excluded.
      {
        // Build: consumer tile coord → list of packet-mode conduit.create ops.
        llvm::DenseMap<std::pair<int64_t, int64_t>,
                       llvm::SmallVector<mlir::Operation *>>
            pktGroupByTile;

        scopeOp->walk([&](Create createOp) {
          auto rm = createOp.getRoutingMode();
          if (!rm || *rm != RoutingMode::Packet)
            return;
          // Read consumer tile directly from discardable attr on conduit.create.
          auto consTilesAttr =
              createOp->getAttrOfType<mlir::DenseI64ArrayAttr>("consumer_tiles");
          if (!consTilesAttr || consTilesAttr.size() < 2)
            return;
          // consumer_tiles is a flat [col, row] array; use first pair.
          std::pair<int64_t, int64_t> coord = {consTilesAttr[0],
                                                consTilesAttr[1]};
          pktGroupByTile[coord].push_back(createOp.getOperation());
        });

        for (auto &[coord, ops] : pktGroupByTile) {
          if (ops.size() <= 1)
            continue; // No sharing needed for single-channel tiles.
          std::string groupName = "pkt_" + std::to_string(coord.first) + "_" +
                                  std::to_string(coord.second);
          for (mlir::Operation *op : ops) {
            op->setAttr("dma_channel_group",
                        mlir::StringAttr::get(ctx, groupName));
          }
        }
      }

    } // end for (scopeOp : scopes)

    // After all device scopes: erase module-level air.channel decls.
    // These are the original high-level names (e.g., @L3ToL2Chan1) that
    // --air-hierarchy-to-aie leaves at module body level with no put/get users.
    {
      llvm::SmallVector<mlir::Operation *> moduleLevelDecls;
      for (auto &op : module.getBody()->getOperations()) {
        if (isAirChannelDecl(&op))
          moduleLevelDecls.push_back(&op);
      }
      for (auto *op : moduleLevelDecls) {
        if (auto symOp = mlir::dyn_cast<mlir::SymbolOpInterface>(op)) {
          if (!mlir::SymbolTable::symbolKnownUseEmpty(symOp.getNameAttr(),
                                                      module.getOperation()))
            continue;
        }
        op->erase();
      }
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory + registration
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAirChannelToConduitPass() {
  return std::make_unique<AirChannelToConduitPass>();
}

} // namespace xilinx::conduit
