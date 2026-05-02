//===- air_depth_promote.mlir - Looped air.channel via Conduit ---*- MLIR -*-===//
//
// Replacement test for async_gemm_w_pingpong_to_locks.mlir / construct_ping_pong.mlir.
// Tests that air.channel.put/get inside scf.for loops are correctly lowered
// to Conduit Tier 3 ops by Pass B.
//
// Pipeline: --air-channel-to-conduit
//
// Scenario: Producer and consumer exchange data in a loop (scf.for).
// Pass B converts air.channel.put/get inside loop bodies to conduit Tier 3 ops.
//
// Note: --conduit-depth-promote only applies to Tier 2 (ObjectFIFO-originated)
// conduit ops, not Tier 3 (air.channel-originated). Depth promotion for
// air.channel programs would require extending depth-promote to handle Tier 3
// or doing promotion at the air.channel level.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --allow-unregistered-dialect \
// RUN:   --air-channel-to-conduit \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: module

// conduit.create emitted with depth=0 sentinel (resolved by --conduit-depth-promote).
// CHECK: conduit.create @chan
// CHECK-SAME: depth = 0

// Tier 3 ops inside loop body.
// CHECK: scf.for
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan

// CHECK: scf.for
// CHECK: conduit.get_memref_async
// CHECK-SAME: name = @chan

// No residual air ops.
// CHECK-NOT: air.channel

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    // Simple SPSC channel.
    "air.channel"() {sym_name = "chan", size = [1, 1]} : () -> ()

    // Producer tile sends data in a loop.
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %alloc = memref.alloc() : memref<32xi32, 2>
      scf.for %i = %c0 to %c8 step %c1 {
        "air.channel.put"(%alloc)
            {chan_name = @chan,
             operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
            : (memref<32xi32, 2>) -> ()
      }
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }

    // Consumer tile receives data in a loop.
    aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %alloc = memref.alloc() : memref<32xi32, 2>
      scf.for %i = %c0 to %c8 step %c1 {
        "air.channel.get"(%alloc)
            {chan_name = @chan,
             operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
            : (memref<32xi32, 2>) -> ()
      }
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }
  }
}
