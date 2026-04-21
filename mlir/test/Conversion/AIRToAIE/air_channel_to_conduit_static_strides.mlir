// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit --verify-diagnostics %s
//
// Pass B strides extraction test: static constants + dynamic error.
//
// Verifies three scenarios:
//
//   1. Static 1-D descriptor: offsets/sizes/strides from arith.constant (index type).
//      Pass B must extract correct values.
//
//   2. Duplicate index constant stride: arith.constant 4 : index (second index
//      constant in the same block).  Ensures Pass B handles multiple constants
//      correctly.
//
//   3. Dynamic (non-constant) offset: a function block argument.
//      Pass B must emit a hard error and fail — placeholder substitution
//      produces wrong DMA descriptors (stride=0 reads same address repeatedly).
//
// This test uses --verify-diagnostics to check the expected-error on scenario 3.
// Static extraction correctness (scenarios 1 & 2) is covered by
// air_channel_to_conduit.mlir which tests 2-D static descriptors end-to-end.

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "chan_index", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "chan_i32",   size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "chan_dyn",   size = [1, 1]} : () -> ()

    // Scenarios 1 & 2: static constants — no error.
    aie.core(%tile_0_3) {
      %src = memref.alloca() : memref<128xi32>
      %dst = memref.alloca() : memref<128xi32>
      %c16  = arith.constant 16 : index
      %c32  = arith.constant 32 : index
      %c1   = arith.constant 1  : index
      %c0   = arith.constant 0  : index
      %c64  = arith.constant 64 : index
      %s4   = arith.constant 4  : index

      %tok0 = "air.channel.put"(%src, %c16, %c32, %c1)
          {chan_name = @chan_index, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<128xi32>, index, index, index) -> !air.async.token
      %tok1 = "air.channel.get"(%dst, %c16, %c32, %c1)
          {chan_name = @chan_index, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<128xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%tok0, %tok1) : (!air.async.token, !air.async.token) -> ()

      %src2 = memref.alloca() : memref<128xi32>
      %tok2 = "air.channel.put"(%src2, %c0, %c64, %s4)
          {chan_name = @chan_i32, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<128xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%tok2) : (!air.async.token) -> ()

      // Scenario 3: dynamic offset via memref.load (non-constant).
      %dyn_buf = memref.alloca() : memref<1xindex>
      %dynoff = memref.load %dyn_buf[%c0] : memref<1xindex>
      %src3 = memref.alloca() : memref<128xi32>
      // expected-error @+1 {{air-channel-to-conduit: channel @chan_dyn has dynamic offset/size/stride operands}}
      %tok3 = "air.channel.put"(%src3, %dynoff, %c64, %c1)
          {chan_name = @chan_dyn, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<128xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%tok3) : (!air.async.token) -> ()
      aie.end
    }
  }
}