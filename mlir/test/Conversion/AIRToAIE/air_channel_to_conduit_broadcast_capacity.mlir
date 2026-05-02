// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s
//
// Pass B broadcast_shape capacity propagation test.
//
// Verifies that broadcast_shape dimensions are multiplied to produce the
// correct capacity on conduit.create for several common fan-out shapes:
//
//   channel_1x2: broadcast_shape=[1,2] → slot_elems = 2
//   channel_2x1: broadcast_shape=[2,1] → slot_elems = 2
//   channel_2x2: broadcast_shape=[2,2] → slot_elems = 4
//   channel_1x4: broadcast_shape=[1,4] → slot_elems = 4
//   channel_scalar: no broadcast_shape → slot_elems = 1 (default)
//
// Consumer tile coordinates are not available at Pass B time; conduit.create
// is emitted with correct capacity and empty consumer_tiles.
// Full wiring requires a tile-placement pre-pass to populate consumer_tiles.

// CHECK: remark{{.*}}channel_1x2{{.*}}capacity = 2
// CHECK: remark{{.*}}channel_2x1{{.*}}capacity = 2
// CHECK: remark{{.*}}channel_2x2{{.*}}capacity = 4
// CHECK: remark{{.*}}channel_1x4{{.*}}capacity = 4

// CHECK-LABEL: module

// channel_1x2: element_type from put memref
// CHECK: conduit.create @channel_1x2
// CHECK-SAME: element_type = memref<16xi32>

// channel_2x1: element_type from put memref
// CHECK: conduit.create @channel_2x1
// CHECK-SAME: element_type = memref<16xi32>

// channel_2x2: element_type from put memref
// CHECK: conduit.create @channel_2x2
// CHECK-SAME: element_type = memref<16xi32>

// channel_1x4: element_type from put memref
// CHECK: conduit.create @channel_1x4
// CHECK-SAME: element_type = memref<16xi32>

// channel_scalar: no broadcast_shape, element_type from put memref
// CHECK: conduit.create @channel_scalar
// CHECK-SAME: element_type = memref<16xi32>

// No residual air.channel ops.
// CHECK-NOT: air.channel

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "channel_1x2", size = [1, 1],
                     broadcast_shape = array<i64: 1, 2>} : () -> ()
    "air.channel"() {sym_name = "channel_2x1", size = [1, 1],
                     broadcast_shape = array<i64: 2, 1>} : () -> ()
    "air.channel"() {sym_name = "channel_2x2", size = [1, 1],
                     broadcast_shape = array<i64: 2, 2>} : () -> ()
    "air.channel"() {sym_name = "channel_1x4", size = [1, 1],
                     broadcast_shape = array<i64: 1, 4>} : () -> ()
    "air.channel"() {sym_name = "channel_scalar", size = [1, 1]} : () -> ()
    aie.core(%tile_0_3) {
      %buf = memref.alloca() : memref<16xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      "air.channel.put"(%buf, %c0, %c1, %c1)
          {chan_name = @channel_1x2, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<16xi32>, index, index, index) -> ()
      "air.channel.put"(%buf, %c0, %c1, %c1)
          {chan_name = @channel_2x1, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<16xi32>, index, index, index) -> ()
      "air.channel.put"(%buf, %c0, %c1, %c1)
          {chan_name = @channel_2x2, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<16xi32>, index, index, index) -> ()
      "air.channel.put"(%buf, %c0, %c1, %c1)
          {chan_name = @channel_1x4, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<16xi32>, index, index, index) -> ()
      "air.channel.put"(%buf, %c0, %c1, %c1)
          {chan_name = @channel_scalar, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<16xi32>, index, index, index) -> ()
      aie.end
    }
  }
}
