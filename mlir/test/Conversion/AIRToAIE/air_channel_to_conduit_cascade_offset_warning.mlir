// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit --verify-diagnostics %s
//
// Pass B cascade offset/size/stride diagnostic test.
//
// Verifies that non-trivial offset/size/stride operands on cascade channels
// produce the correct diagnostics:
//
//   1. Constant non-zero offset → warning mentioning "element[0]"
//   2. Constant non-unit size   → warning mentioning "element[0]"
//   3. Constant non-unit stride → warning mentioning "element[0]"
//   4. Fully dynamic (non-constant) offset → hard error (intent cannot be inferred)
//
// Trivial (zero-offset, unit-size, unit-stride) operands on cascade channels
// produce no diagnostic (tested in air_channel_to_conduit_cascade.mlir).

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "cas_nonzero_off", size = [1, 1], channel_type = "cascade"} : () -> ()
    "air.channel"() {sym_name = "cas_nonunit_sz",  size = [1, 1], channel_type = "cascade"} : () -> ()
    "air.channel"() {sym_name = "cas_nonunit_st",  size = [1, 1], channel_type = "cascade"} : () -> ()
    "air.channel"() {sym_name = "cas_dyn_off",     size = [1, 1], channel_type = "cascade"} : () -> ()
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index

      %src1 = memref.alloca() : memref<1xvector<16xi32>>
      // expected-warning @+1 {{non-zero offset}}
      "air.channel.put"(%src1, %c4, %c1, %c1)
          {chan_name = @cas_nonzero_off, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<1xvector<16xi32>>, index, index, index) -> ()

      %src2 = memref.alloca() : memref<4xvector<16xi32>>
      // expected-warning @+1 {{non-unit size}}
      "air.channel.put"(%src2, %c0, %c4, %c1)
          {chan_name = @cas_nonunit_sz, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xvector<16xi32>>, index, index, index) -> ()

      %src3 = memref.alloca() : memref<1xvector<16xi32>>
      // expected-warning @+1 {{non-unit stride}}
      "air.channel.put"(%src3, %c0, %c1, %c2)
          {chan_name = @cas_nonunit_st, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<1xvector<16xi32>>, index, index, index) -> ()

      // Scenario 4: dynamic offset via memref.load
      %dyn_buf = memref.alloca() : memref<1xindex>
      %dyn = memref.load %dyn_buf[%c0] : memref<1xindex>
      %src4 = memref.alloca() : memref<1xvector<16xi32>>
      // expected-error @+1 {{fully dynamic offset operand}}
      "air.channel.put"(%src4, %dyn, %c1, %c1)
          {chan_name = @cas_dyn_off, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<1xvector<16xi32>>, index, index, index) -> ()

      aie.end
    }
  }
}