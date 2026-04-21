//===- air_shim_dma.mlir - Shim DMA transfer via Conduit --------*- MLIR -*-===//
//
// Replacement test for air_shimcpy_to_aie.mlir / air_shimcpy_to_npu.mlir.
// Tests shim-to-compute data transfer through the Conduit pipeline.
//
// Pipeline: --air-channel-to-conduit (Pass B only)
//
// Scenario: Shim tile sends data through a channel to a compute tile.
// The async put/get pattern models the DMA shimcpy operation.
// Tests that Pass B correctly converts async air.channel.put/get with
// DMA descriptors to conduit Tier 3 ops.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --allow-unregistered-dialect \
// RUN:   --air-channel-to-conduit \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: module

// Channel becomes conduit.create.
// CHECK: conduit.create @dma_chan

// Async put becomes conduit.put_memref_async with descriptor.
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @dma_chan
// CHECK-SAME: num_elems = 1024
// CHECK-SAME: offsets = array<i64: 0>
// CHECK-SAME: sizes = array<i64: 1024>
// CHECK-SAME: strides = array<i64: 1>

// Async get becomes conduit.get_memref_async.
// CHECK: conduit.get_memref_async
// CHECK-SAME: name = @dma_chan
// CHECK-SAME: num_elems = 1024

// wait_all converted to conduit.wait_all_async.
// CHECK: conduit.wait_all_async

// No residual air ops.
// CHECK-NOT: air.channel
// CHECK-NOT: air.wait_all

module {
  aie.device(xcve2802) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Shim-to-compute channel.
    "air.channel"() {sym_name = "dma_chan", size = [1, 1]} : () -> ()

    func.func @shimcpy_test(%src: memref<1024xi32>, %dst: memref<1024xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index

      // Shim sends 1024 elements (async put with 1-D descriptor).
      %tok0 = "air.channel.put"(%src, %c0, %c1024, %c1)
          {chan_name = @dma_chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<1024xi32>, index, index, index) -> !air.async.token

      // Compute tile receives 1024 elements (async get with 1-D descriptor).
      %tok1 = "air.channel.get"(%dst, %c0, %c1024, %c1)
          {chan_name = @dma_chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<1024xi32>, index, index, index) -> !air.async.token

      // Barrier on both transfers.
      %merged = "air.wait_all"(%tok0, %tok1)
          : (!air.async.token, !air.async.token) -> !air.async.token

      return
    }
  }
}
