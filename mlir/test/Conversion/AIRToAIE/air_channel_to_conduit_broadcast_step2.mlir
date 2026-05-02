// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s
//
// Pass B broadcast_shape Step 2: consumer tile coordinate extraction.
//
// When air.channel.get ops appear inside aie.core regions (i.e., after the
// air-to-aie lowering pass has run), Pass B can extract the enclosing tile
// coordinates and emit:
//   - Per-consumer conduit.create aliases (@bcast_c0, @bcast_c1)
//   - conduit.distribute {srcs = [@bcast], dsts = [@bcast_c0, @bcast_c1]}
//
// This test exercises the aie.core enclosure path.
//
// Topology: channel @bcast broadcasts to 2 consumers at tiles (2,2) and (3,2).
// (Row 2 = core tiles on xcve2302; row 0 = shim, row 1 = memtile.)
//
// Expected IR after --air-channel-to-conduit:
//   conduit.create with name="bcast", slot_elems =2  (source)
//   conduit.create with name="bcast_c0", //   conduit.create with name="bcast_c1", //   conduit.distribute with srcs=["bcast"], dsts=["bcast_c0","bcast_c1"]
//
// Note: the remark "found 2 consumer tiles from aie.core enclosure" is emitted.

// CHECK: remark{{.*}}found 2 consumer tiles from aie.core enclosure

// CHECK-LABEL: module

// Source conduit.create with element_type.
// CHECK: conduit.create @bcast
// CHECK-SAME: element_type = memref<16xi32>

// Consumer alias for tile (2,2).
// CHECK: conduit.create @bcast_c0
// CHECK-SAME: element_type = memref<16xi32>
// Consumer alias for tile (3,2).
// CHECK: conduit.create @bcast_c1
// CHECK-SAME: element_type = memref<16xi32>
// Per-column scatter ops (one per consumer column, each with column-local MemTile).
// CHECK: conduit.scatter{src = @bcast, dsts = [@bcast_c0]
// CHECK-SAME: memtile = %mem_tile_2_1
// CHECK: conduit.scatter{src = @bcast, dsts = [@bcast_c1]
// CHECK-SAME: memtile = %mem_tile_3_1

// No residual air.channel declarations.
// CHECK-NOT: air.channel {

module {
  aie.device(xcve2302) {
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %buf_2 = aie.buffer(%tile_2_2) {sym_name = "buf_2"} : memref<16xi32>
    %buf_3 = aie.buffer(%tile_3_2) {sym_name = "buf_3"} : memref<16xi32>

    // Broadcast channel: 1 producer → 2 consumers.
    "air.channel"() {sym_name = "bcast", size = [1, 1],
                     broadcast_shape = array<i64: 1, 2>} : () -> ()

    // Consumer 0 in tile (2,2).
    %core_2_2 = aie.core(%tile_2_2) {
      "air.channel.get"(%buf_2)
          {chan_name = @bcast,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<16xi32>) -> ()
      aie.end
    }

    // Consumer 1 in tile (3,2).
    %core_3_2 = aie.core(%tile_3_2) {
      "air.channel.get"(%buf_3)
          {chan_name = @bcast,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<16xi32>) -> ()
      aie.end
    }
  }
}
