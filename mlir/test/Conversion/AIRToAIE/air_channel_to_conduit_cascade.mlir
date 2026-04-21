// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B (--air-channel-to-conduit) cascade channel_type test.
//
// Verifies that air.channel declarations with channel_type = "cascade" emit
// a conduit.create with routing_mode = #conduit.routing_mode<cascade>, and
// that the put/get ops are ERASED (the kernel C++ manages cascade data
// movement via get_scd/put_scd intrinsics — Pass B does not emit
// aie.put_cascade / aie.get_cascade).
//
// Uses memref<1xvector<16xi32>>: element type vector<16xi32> = 512 bits (AIE2).

// CHECK-LABEL: module

// --- Cascade channel: conduit.create with routing_mode = cascade ---
// CHECK: conduit.create @cas_chan
// CHECK-SAME: depth = 0 : i64
// CHECK-SAME: element_type = memref<1xvector<16xi32>>
// CHECK-SAME: routing_mode = #conduit.routing_mode<cascade>

// --- air.channel declaration is erased ---
// CHECK-NOT: air.channel

// --- put/get ops are erased (kernel manages cascade intrinsics) ---
// CHECK-NOT: air.channel.put
// CHECK-NOT: air.channel.get
// CHECK-NOT: aie.put_cascade
// CHECK-NOT: aie.get_cascade

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "cas_chan", size = [1, 1],
                     channel_type = "cascade"} : () -> ()
    aie.core(%tile_0_3) {
      %src = memref.alloca() : memref<1xvector<16xi32>>
      %dst = memref.alloca() : memref<1xvector<16xi32>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      "air.channel.put"(%src, %c0, %c1, %c1)
          {chan_name = @cas_chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<1xvector<16xi32>>, index, index, index) -> ()
      "air.channel.get"(%dst, %c0, %c1, %c1)
          {chan_name = @cas_chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<1xvector<16xi32>>, index, index, index) -> ()
      aie.end
    }
  }
}
