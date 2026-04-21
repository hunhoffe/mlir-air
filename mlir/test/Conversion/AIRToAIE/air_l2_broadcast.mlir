//===- air_l2_broadcast.mlir - L2 broadcast via Conduit --------*- MLIR -*-===//
//
// Replacement test for air_channel_to_objectfifo_L2_broadcast.mlir.
// Tests 1→2 broadcast from MemTile to two compute tiles through the
// Conduit pipeline. Uses aie.device directly.
//
// Scenario: MemTile distributes data to 2 compute tiles.
// broadcast_shape = [1, 2] means 1 producer → 2 consumers.
// Consumer get ops inside aie.core → Pass B broadcast Step 2 extracts
// tile coordinates and emits per-consumer conduit.create aliases.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s

// CHECK: remark{{.*}}found 2 consumer tiles from aie.core enclosure

// CHECK-LABEL: module

// Source conduit.create with element_type (broadcast fan-out).
// CHECK: conduit.create @bcast
// CHECK-SAME: element_type = memref<32xi32, 1>

// Per-consumer aliases from broadcast Step 2.
// CHECK: conduit.create @bcast_c0
// CHECK-SAME: element_type = memref<32xi32, 1>
// CHECK: conduit.create @bcast_c1
// CHECK-SAME: element_type = memref<32xi32, 1>
// Per-column scatter ops (one per consumer column, each with column-local MemTile).
// CHECK: conduit.scatter{src = @bcast, dsts = [@bcast_c0]
// CHECK-SAME: memtile = "tile(0,1)"
// CHECK: conduit.scatter{src = @bcast, dsts = [@bcast_c1]
// CHECK-SAME: memtile = "tile(1,1)"

// No residual air ops.
// CHECK-NOT: air.channel

module {
  aie.device(xcve2802) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)

    // Broadcast channel: MemTile → 2 compute tiles
    "air.channel"() {sym_name = "bcast", size = [1, 1],
                     broadcast_shape = array<i64: 1, 2>} : () -> ()

    // MemTile: source of broadcast
    %l2_buf = aie.buffer(%tile_0_1) {sym_name = "l2_buf"} : memref<32xi32, 1>
    "air.channel.put"(%l2_buf)
        {chan_name = @bcast,
         operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
        : (memref<32xi32, 1>) -> ()

    // Compute tile 0: consumer 0
    aie.core(%tile_0_3) {
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.get"(%alloc)
          {chan_name = @bcast,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }

    // Compute tile 1: consumer 1
    aie.core(%tile_1_3) {
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.get"(%alloc)
          {chan_name = @bcast,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }
  }
}
