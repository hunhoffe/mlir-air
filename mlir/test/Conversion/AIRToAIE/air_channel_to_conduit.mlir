// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B (--air-channel-to-conduit) basic test: lowers air.channel.put/get to Conduit Tier 3 ops.
//
// Input: post-hierarchy IR (air.channel inside aie.device, put/get inside aie.core).
// This matches the format produced by --air-hierarchy-to-aie, which is the
// only production input Pass B ever receives.
//
// The program contains:
//   - one air.channel declaration @chan [1, 1] inside aie.device
//   - one air.channel.put (async) with 8x8 descriptor inside aie.core
//   - one air.channel.get (async) with matching descriptor inside aie.core
//   - one air.wait_all (async)
//
// Expected output after --air-channel-to-conduit:
//   - conduit.create @chan {depth=0, element_type=...}
//     with element_type inferred from the put memref operand type
//   - conduit.put_memref_async with name="chan", num_elems=64, offsets/sizes/strides extracted
//   - conduit.get_memref_async with name="chan", num_elems=64, matching descriptor
//   - conduit.wait_all_async joining the two tokens
//   - NO air.channel / air.wait_all ops remaining

// CHECK-LABEL: module
//
// --- Channel declaration becomes conduit.create ---
// Attributes are printed in alphabetical order:
//   depth, element_type
// CHECK:   conduit.create @chan
// CHECK-SAME: depth = 0
// CHECK-SAME: element_type = memref<8x8xi32>
//
// --- air.channel.put becomes conduit.put_memref_async ---
// Static descriptor: offsets=[0,0], sizes=[8,8], strides=[8,1], num_elems=8*8=64
// CHECK:   %[[TOK0:.*]] = conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: num_elems = 64
// CHECK-SAME: offsets = array<i64: 0, 0>
// CHECK-SAME: sizes = array<i64: 8, 8>
// CHECK-SAME: strides = array<i64: 8, 1>
// CHECK-SAME: : !conduit.dma.token
//
// --- air.channel.get becomes conduit.get_memref_async ---
// CHECK:   %[[TOK1:.*]] = conduit.get_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: num_elems = 64
// CHECK-SAME: offsets = array<i64: 0, 0>
// CHECK-SAME: sizes = array<i64: 8, 8>
// CHECK-SAME: strides = array<i64: 8, 1>
// CHECK-SAME: : !conduit.dma.token
//
// --- air.wait_all async becomes conduit.wait_all_async ---
// CHECK:   conduit.wait_all_async %[[TOK0]], %[[TOK1]]
// CHECK-SAME: (!conduit.dma.token, !conduit.dma.token) -> !conduit.dma.token
//
// --- No residual AIR ops ---
// CHECK-NOT: air.channel
// CHECK-NOT: air.wait_all

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)

    // Air channel declaration inside aie.device (post-hierarchy format).
    "air.channel"() {sym_name = "chan", size = [1, 1]} : () -> ()

    aie.core(%tile_0_3) {
      %src = memref.alloca() : memref<8x8xi32>
      %dst = memref.alloca() : memref<8x8xi32>

      // Static constants for the memref descriptor.
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index

      // air.channel.put async: offsets=[0,0], sizes=[8,8], strides=[8,1], num_elems=64.
      %tok0 = "air.channel.put"(%src, %c0, %c0, %c8, %c8, %c8, %c1)
          {chan_name = @chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
          : (memref<8x8xi32>, index, index, index, index, index, index)
          -> !air.async.token

      // air.channel.get async: same descriptor shape.
      %tok1 = "air.channel.get"(%dst, %c0, %c0, %c8, %c8, %c8, %c1)
          {chan_name = @chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
          : (memref<8x8xi32>, index, index, index, index, index, index)
          -> !air.async.token

      // air.wait_all async: fan-in over both tokens.
      %merged = "air.wait_all"(%tok0, %tok1)
          : (!air.async.token, !air.async.token) -> !air.async.token

      aie.end
    }
  }
}
