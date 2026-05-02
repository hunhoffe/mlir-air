//===- air_compute_to_compute_full.mlir - Full B+C pipeline -----*- MLIR -*-===//
//
// Replacement test for async_one_core_gemm_to_npu.mlir and general air-to-aie
// conversion tests. Exercises the full Pass B + Pass C pipeline from
// air.channel to aie.buffer/lock.
//
// Pipeline: --air-channel-to-conduit --conduit-to-dma
//
// Scenario: Two compute tiles on xcve2802. Tile (0,3) produces data and
// tile (0,4) consumes it. Both put/get ops are inside aie.core regions,
// so Pass B tile_coords fix propagates tile coordinates into conduit.create.
// Pass C then lowers to hardware ops (buffer + locks + use_lock in cores).
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s

// CHECK: aie.device(xcve2802)

// Pass C allocates buffer and locks.
// CHECK-DAG: aie.buffer
// CHECK-DAG: aie.lock

// Pass C emits use_lock in core bodies.
// CHECK: aie.use_lock

// No residual conduit or air ops.
// CHECK-NOT: conduit.create
// CHECK-NOT: air.channel

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    // Simple SPSC channel.
    "air.channel"() {sym_name = "compute_chan", size = [1, 1]} : () -> ()

    // Producer core at tile (0,3).
    aie.core(%tile_0_3) {
      %alloc = memref.alloc() : memref<16xi32, 2>
      "air.channel.put"(%alloc)
          {chan_name = @compute_chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<16xi32, 2>) -> ()
      memref.dealloc %alloc : memref<16xi32, 2>
      aie.end
    }

    // Consumer core at tile (0,4).
    aie.core(%tile_0_4) {
      %alloc = memref.alloc() : memref<16xi32, 2>
      "air.channel.get"(%alloc)
          {chan_name = @compute_chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<16xi32, 2>) -> ()
      memref.dealloc %alloc : memref<16xi32, 2>
      aie.end
    }
  }
}
