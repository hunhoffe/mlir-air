// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s
//
// Pass B (--air-channel-to-conduit) broadcast_shape topology test.
//
// Verifies that air.channel declarations with a broadcast_shape attribute:
//   1. Emit a remark (not a warning) naming the computed capacity.
//   2. Emit conduit.create with slot_elems = product(broadcast_shape) = 4.
//   3. Retain routing_mode = #conduit.routing_mode<packet> from channel_type = "dma_packet".
//
// broadcast_shape = [1, 4] → slot_elems = 1 × 4 = 4.
//
// Consumer tile coordinates are NOT available at this stage (no tile-placement
// pre-pass). conduit.create is emitted with correct capacity but empty
// consumer_tiles; full wiring requires a subsequent placement pass.
//
// Note: FileCheck is run on combined stdout+stderr (2>&1) so it can check
// the remark emitted by mlir's diagnostic system.

// Output order (stderr+stdout combined):
//   line 1: remark: ... broadcast_shape ... slot_elems = 4 ...
//   line 2-3: source echo + note
//   line 4: module {
//   line 5: conduit.create {...slot_elems = 4...name = @bcast_chan...routing_mode = #conduit.routing_mode<packet>}
//
// Checks must follow the output order.

// 1. Remark contains "broadcast_shape" and "capacity = 4".
// CHECK: remark{{.*}}broadcast_shape
// CHECK-SAME: capacity = 4

// 2. Module opens (comes before conduit.create in output).
// CHECK: module {

// 3. conduit.create with correct element_type, name, and routing_mode.
// CHECK: conduit.create @bcast_chan
// CHECK-SAME: element_type = memref<8x8xi32>
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>

// 4. No further air.channel ops in module body (source echoes already passed).
// CHECK-NOT: air.channel

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "bcast_chan", size = [1, 1],
                     broadcast_shape = array<i64: 1, 4>,
                     channel_type = "dma_packet"} : () -> ()
    aie.core(%tile_0_3) {
      %src = memref.alloca() : memref<8x8xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %tok0 = "air.channel.put"(%src, %c0, %c0, %c8, %c8, %c8, %c1)
          {chan_name = @bcast_chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
          : (memref<8x8xi32>, index, index, index, index, index, index)
          -> !air.async.token
      aie.end
    }
  }
}
