//===- AIRHierarchyToAIEPass.cpp ---------------------------------*- C++
//-*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// --air-hierarchy-to-aie: Lowers AIR hierarchy (launch/segment/herd) to
// aie.device / aie.tile / aie.core while preserving air.channel declarations
// and put/get/wait_all ops.  This is the intended entry point for the Conduit
// pipeline on AIR programs:
//
//   air-opt --air-hierarchy-to-aie \
//           --air-channel-to-conduit \
//           --conduit-to-dma <file.mlir>
//
// Implementation: runs steps 1-5 of the AIRToAIE pipeline and stops before
// buffer allocation and channel/DMA lowering.
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToAIEPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "air-hierarchy-to-aie"

using namespace mlir;

namespace xilinx {
namespace air {

class AIRHierarchyToAIEPass
    : public air::impl::AIRHierarchyToAIEBase<AIRHierarchyToAIEPass> {

public:
  AIRHierarchyToAIEPass() = default;
  AIRHierarchyToAIEPass(const AIRHierarchyToAIEPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<xilinx::AIEX::AIEXDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<DLTIDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module);
    builder.setInsertionPointToStart(module.getBody());

    // Parse the device string.
    auto device = AIE::symbolizeAIEDevice(clDevice);
    if (!device) {
      module.emitOpError("Invalid aie.device option '") << clDevice << "'";
      signalPassFailure();
      return;
    }

    // Build conversion options from pass flags.
    AIRToAIEConversionOptions options = {
        /* .col_offset = */ static_cast<int64_t>(clColOffset),
        /* .row_offset = */ static_cast<int64_t>(clRowOffset),
        /* .emit_while = */ clEmitWhileLoop,
        /* .emit_herd_lock = */ clEmitHerdLock,
        /* .generate_shim_dma = */ false,
        /* .insert_trace_packet_flow = */ false,
        /* .use_packet_flow_at_shim_dmas = */ false,
        /* .use_lock_race_condition_fix = */ clUseLockRaceConditionFix,
        /* .device = */ *device};

    // Step 1: renumber memcpy ops at module level.
    air::renumberMemcpyIfOps(&module.getRegion());

    // Step 2: create aie.device ops with aie.tile/aie.core for each
    // segment/herd.
    std::vector<
        std::tuple<AIE::DeviceOp, air::HerdOp, AIRToAIEConversionOptions>>
        aie_devices;
    std::map<AIE::TileOp, air::HerdOp> tileToHerdMap;
    createAIEModulesAndOutlineCores(module, aie_devices, tileToHerdMap,
                                    options);

    // Steps 3-5: for each unique device, run hierarchy lowering.
    std::set<AIE::DeviceOp> seen;
    for (auto &p : aie_devices) {
      auto deviceOp = std::get<0>(p);
      air::HerdOp h = std::get<1>(p);
      auto device_options = std::get<2>(p);

      if (seen.find(deviceOp) != seen.end())
        continue;
      seen.insert(deviceOp);

      // Step 3: clone L2/L3 memcpy ops into the device.
      // For hierarchy-only lowering we clone both L2 and L3 (matching the
      // non-objectfifo path of --air-to-aie).
      air::LaunchOp targetLaunch = h->getParentOfType<air::LaunchOp>();
      cloneL2AndL3MemcpysToDeviceOp(
          builder, deviceOp, module, /*clone_l2*/ true, /*clone_l3*/ true,
          device_options.use_lock_race_condition_fix, targetLaunch);

      // Step 3b: rewire out-of-device async-token operands.
      //
      // cloneL2AndL3MemcpysToDeviceOp clones launch-level scf.for /
      // air.channel.put / air.channel.get into the device, but air.wait_all
      // is intentionally NOT cloned (see AIRToAIEPass.cpp's
      // cloneL2AndL3MemcpysToDeviceOp walk: WaitAllOp is in the "advance"
      // list, not the "clone" list).  As a result, a cloned scf.for's
      // !air.async.token init_arg still references the original launch's
      // air.wait_all.  When Step 4's lowerScfAirTokens then propagates that
      // init_arg into channel.put async dependencies via replaceAllUsesWith,
      // and Step 6 erases the launch (via dropAllDefinedValueUses), the
      // cloned channel.put ends up with a NULL operand for its async dep.
      //
      // Fix: walk the device and replace any !air.async.token operand whose
      // defining op (or block-arg parent op) lives outside this device with
      // a fresh in-device air.wait_all (no deps, satisfied token).
      {
        auto *ctx = deviceOp->getContext();
        deviceOp.walk([&](Operation *op) {
          if (op == deviceOp.getOperation())
            return;
          for (OpOperand &operand : op->getOpOperands()) {
            Value v = operand.get();
            if (!v || !llvm::isa<air::AsyncTokenType>(v.getType()))
              continue;
            Operation *defOp = v.getDefiningOp();
            bool inDevice = false;
            if (defOp) {
              inDevice = deviceOp->isAncestor(defOp);
            } else if (auto blockArg = llvm::dyn_cast<BlockArgument>(v)) {
              if (Operation *parentOp = blockArg.getOwner()->getParentOp())
                inDevice = deviceOp->isAncestor(parentOp);
            }
            if (inDevice)
              continue;
            OpBuilder b(op);
            auto fresh = air::WaitAllOp::create(
                b, op->getLoc(), air::AsyncTokenType::get(ctx),
                SmallVector<Value>{});
            operand.set(fresh.getAsyncToken());
          }
        });
      }

      // Step 4: lower AIR control-flow constructs.
      specializeHerdAffineIf(deviceOp);
      lowerAirExecute(deviceOp);
      lowerScfAirTokens(deviceOp);

      // Step 5: specialize channel bundles and clean up orphans.
      std::map<std::string, std::string> chan_to_chan_map;
      specializeChannelBundle(deviceOp, chan_to_chan_map);
      if (deviceOp->hasAttr("segment_unroll_x") ||
          deviceOp->hasAttr("segment_unroll_y"))
        removeOrphanedChannels(deviceOp);
    }

    // Step 6: erase the original air.launch ops.
    //
    // createAIEModulesAndOutlineCores clones the herd/segment bodies into
    // aie.device regions but leaves the originals in the func.func body.
    // The Conduit pipeline needs them gone: --air-channel-to-conduit (Pass B)
    // would otherwise convert air.channel.put/get in both the aie.device
    // (correct) and the residual func body (wrong — causes type mismatches
    // because air.herd/affine.yield still expect !air.async.token, not
    // !conduit.dma.token).
    //
    // Erase air.launch ops (which contain air.segment and air.herd).
    // Module-level air.channel declarations are kept — Pass B needs them.
    //
    // Before erasing, drop all SSA uses within each op's body to prevent
    // 'operation destroyed but still has uses' assertions.  The original
    // launch/segment/herd bodies contain async token dependency chains
    // (air.wait_all → scf.for iter_args → air.channel.put/get) that form
    // cross-region use-def edges.  Region destruction erases ops one by one,
    // hitting the assertion when an inner op still has uses from another
    // inner op that hasn't been erased yet.  dropAllDefinedValueUses()
    // drops uses of ALL values defined by the op or its nested regions
    // (results + block arguments), then dropAllReferences() clears all
    // operand references.  Together they make the recursive erase safe.
    // The real work has already been cloned into aie.device and lowered.
    auto safeEraseWithRegions = [](Operation *op) {
      // Drop all uses of values defined by this op or its nested regions
      // (results, block arguments, nested op results).
      op->dropAllDefinedValueUses();
      // Drop all operand references within the op and its nested regions.
      op->dropAllReferences();
      op->erase();
    };

    SmallVector<air::LaunchOp> launches;
    module.walk([&](air::LaunchOp op) { launches.push_back(op); });
    for (auto op : launches)
      safeEraseWithRegions(op);

    // Also erase any top-level air.herd ops not inside a launch (implicit
    // segment case).
    SmallVector<air::HerdOp> topHerds;
    module.walk([&](air::HerdOp op) {
      if (!op->getParentOfType<AIE::DeviceOp>())
        topHerds.push_back(op);
    });
    for (auto op : topHerds)
      safeEraseWithRegions(op);

    // Erase any top-level air.segment ops not inside a launch.
    SmallVector<air::SegmentOp> topSegs;
    module.walk([&](air::SegmentOp op) {
      if (!op->getParentOfType<AIE::DeviceOp>())
        topSegs.push_back(op);
    });
    for (auto op : topSegs)
      safeEraseWithRegions(op);

    // The IR now has:
    //   - aie.device / aie.tile / aie.core  (hierarchy lowered)
    //   - air.channel declarations inside aie.device
    //   - air.channel.put/get/wait_all inside aie.core bodies
    //   - No aie.objectfifo / aie.lock / aie.flow / aie.dma_bd
    //   - No residual air.launch / air.segment / air.herd
    //
    // Ready for --air-channel-to-conduit (Pass B) + --conduit-to-dma (Pass C).
  }
};

std::unique_ptr<mlir::Pass> createAIRHierarchyToAIEPass() {
  return std::make_unique<AIRHierarchyToAIEPass>();
}

} // namespace air
} // namespace xilinx
