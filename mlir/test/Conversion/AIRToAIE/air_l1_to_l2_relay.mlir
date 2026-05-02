//===- air_l1_to_l2_relay.mlir - L1->L2 DMA relay via Conduit ---*- MLIR -*-===//
//
// Replacement test for air_channel_to_objectfifo_L1toL2.mlir.
// Tests L1 (compute tile) to L2 (MemTile) DMA relay through the
// Conduit pipeline. Uses aie.device directly (no hierarchy pass needed).
//
// Scenario: MemTile <-> Compute tile bidirectional data path.
// MemTile ops are outside aie.core (no tile coord propagation); compute
// ops inside aie.core get tile coords from Pass B tile_coords fix.
//
// Tests Pass B only (no Pass C): verifies conduit.create emitted for each
// channel, air.channel declarations removed.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s

// CHECK-LABEL: module

// Both channels become conduit.create ops.
// CHECK: conduit.create @ch_in

// CHECK: conduit.create @ch_out

// Put/get converted to conduit Tier 3 ops.
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @ch_in

// CHECK: conduit.get_memref_async
// CHECK-SAME: name = @ch_in

// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @ch_out

// CHECK: conduit.get_memref_async
// CHECK-SAME: name = @ch_out

// No residual air ops.
// CHECK-NOT: air.channel

module {
  aie.device(xcve2802) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_3 = aie.tile(0, 3)

    // L2->L1 channel (MemTile to compute tile)
    "air.channel"() {sym_name = "ch_in", size = [1, 1]} : () -> ()
    // L1->L2 channel (compute tile to MemTile)
    "air.channel"() {sym_name = "ch_out", size = [1, 1]} : () -> ()

    // MemTile buffer
    %l2_buf = aie.buffer(%tile_0_1) {sym_name = "l2_buf"} : memref<32xi32, 1>

    // MemTile sends data to compute tile
    "air.channel.put"(%l2_buf)
        {chan_name = @ch_in,
         operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
        : (memref<32xi32, 1>) -> ()

    // Compute tile receives data, does work, sends back
    aie.core(%tile_0_3) {
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.get"(%alloc)
          {chan_name = @ch_in,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<32xi32, 2>) -> ()
      "air.channel.put"(%alloc)
          {chan_name = @ch_out,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }

    // MemTile receives result
    "air.channel.get"(%l2_buf)
        {chan_name = @ch_out,
         operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
        : (memref<32xi32, 1>) -> ()
  }
}
