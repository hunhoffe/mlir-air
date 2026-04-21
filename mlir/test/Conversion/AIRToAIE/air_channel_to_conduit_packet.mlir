// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B (--air-channel-to-conduit) channel_type propagation test.
//
// Verifies that air.channel declarations with channel_type = "dma_packet" produce
// a conduit.create with routing_mode = #conduit.routing_mode<packet>, while channels without channel_type
// (dma_stream default) produce a conduit.create with no routing_mode attribute.
//
// Sprint item 5a: channel_type → routing_mode propagation.

// CHECK-LABEL: module

// --- Packet channel: routing_mode = #conduit.routing_mode<packet> ---
// CHECK:   conduit.create @pkt_chan
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>

// --- Stream channel: no routing_mode attribute ---
// CHECK:   conduit.create @stream_chan
// CHECK-NOT: routing_mode

// CHECK-NOT: air.channel

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "pkt_chan", size = [1, 1],
                     channel_type = "dma_packet"} : () -> ()
    "air.channel"() {sym_name = "stream_chan", size = [1, 1]} : () -> ()
    aie.core(%tile_0_3) {
      %src = memref.alloca() : memref<4x4xi32>
      %dst = memref.alloca() : memref<4x4xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %tok0 = "air.channel.put"(%src, %c0, %c0, %c4, %c4, %c4, %c1)
          {chan_name = @pkt_chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
          : (memref<4x4xi32>, index, index, index, index, index, index)
          -> !air.async.token
      %tok1 = "air.channel.get"(%dst, %c0, %c0, %c4, %c4, %c4, %c1)
          {chan_name = @stream_chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
          : (memref<4x4xi32>, index, index, index, index, index, index)
          -> !air.async.token
      aie.end
    }
  }
}
