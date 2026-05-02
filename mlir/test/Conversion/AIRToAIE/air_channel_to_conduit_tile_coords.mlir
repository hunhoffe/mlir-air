// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B tile coordinate propagation test.
//
// When air.channel.put/get ops appear inside aie.core regions (i.e., after
// --air-hierarchy-to-aie), Pass B propagates the enclosing core's tile
// coordinates into the conduit.create op's producer_tile and consumer_tiles
// attributes.  Pass C requires these to allocate locks and flows on the
// correct tiles.
//
// Input: hierarchy-produced IR with:
//   - aie.device(xcve2802) containing two tiles
//   - aie.core(%tile_2_3) with air.channel.put (producer)
//   - aie.core(%tile_2_4) with air.channel.get (consumer)
//   - air.channel @channel_0 declaration inside the device
//
// Expected output:
//   conduit.create with // CHECK: conduit.create @channel_0
module @test_tile_coords {
  aie.device(xcve2802) @segment_0 {
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %core_2_3 = aie.core(%tile_2_3) {
      %alloc = memref.alloc() : memref<32x32xbf16, 2>
      "air.channel.put"(%alloc) {chan_name = @channel_0, id = 1 : i32, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>} : (memref<32x32xbf16, 2>) -> ()
      memref.dealloc %alloc : memref<32x32xbf16, 2>
      aie.end
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %alloc = memref.alloc() : memref<32x32xbf16, 2>
      "air.channel.get"(%alloc) {chan_name = @channel_0, id = 2 : i32, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>} : (memref<32x32xbf16, 2>) -> ()
      memref.dealloc %alloc : memref<32x32xbf16, 2>
      aie.end
    }
    "air.channel"() {broadcast_shape = array<i64>, sym_name = "channel_0"} : () -> ()
  }
}
