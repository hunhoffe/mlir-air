// RUN: air-opt --allow-unregistered-dialect "--air-channel-to-conduit=infer-rates=true" %s 2>&1 | FileCheck %s
//
// Pass B broadcast guard for Phase 6 infer-rates.
//
// Verifies two things:
//
// 1. SPSC channel (@spsc_chan): no broadcast_shape → rates ARE annotated.
// 2. Broadcast channel (@bcast_chan, broadcast_shape=[1,4]): rates are NOT
//    annotated even with infer-rates=true, because slot_elems =4 is a fan-out
//    count (not buffer slots) and M7 would misinterpret it.
//
// MLIR prints attributes alphabetically; conduit.create for spsc_chan appears
// first (declaration order).
//
// The broadcast remark appears on stderr (merged via 2>&1) before the module
// output. Use CHECK-LABEL on "module" to skip past the remark lines.

// CHECK-LABEL: module

// SPSC: consumer_rates and producer_rates attached.
// CHECK: conduit.create @spsc_chan
// CHECK-SAME: consumer_rates = array<i64: 64>
// CHECK-SAME: producer_rates = array<i64: 64>

// Broadcast channel: rates NOT annotated (broadcast is multi-consumer,
// slot_elems was a fan-out count; in Sprint 6 that attr is gone).
// After matching bcast_chan create, verify NO producer_rates or consumer_rates.
// CHECK: conduit.create @bcast_chan
// CHECK-NOT: producer_rates
// CHECK-NOT: consumer_rates

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "spsc_chan", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "bcast_chan", size = [1, 1],
                     broadcast_shape = array<i64: 1, 4>} : () -> ()
    aie.core(%tile_0_3) {
      %buf = memref.alloca() : memref<64xi32>
      "air.channel.put"(%buf)
          {chan_name = @spsc_chan, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<64xi32>) -> ()
      "air.channel.get"(%buf)
          {chan_name = @spsc_chan, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<64xi32>) -> ()
      "air.channel.put"(%buf)
          {chan_name = @bcast_chan, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<64xi32>) -> ()
      "air.channel.get"(%buf)
          {chan_name = @bcast_chan, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<64xi32>) -> ()
      aie.end
    }
  }
}