// RUN: not air-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s
//
// Regression test (A-4 + A-5): dynamic-stride put must emit a hard error and
// not crash due to the original op being left in the IR with uses.
//
// A-4: After signalPassFailure, the op must be added to putGetToErase so it is
//      cleaned up and does not leave dangling channel-use references.
// A-5: Phase 5 must not erase the channel decl if it still has remaining uses;
//      it must emit an error and skip the erase.
//
// The dynamic stride is a function argument (%dyn_stride), which cannot be
// extracted as a static constant. This triggers the dynamic-operand error path.
//
// CHECK: error:{{.*}}dynamic offset/size/stride operands

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "dyn_chan", size = [1, 1]} : () -> ()
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %dyn_buf = memref.alloca() : memref<1xindex>
      %dyn_stride = memref.load %dyn_buf[%c0] : memref<1xindex>
      %src = memref.alloca() : memref<8x8xi32>
      %tok0 = "air.channel.put"(%src, %c0, %c0, %c8, %c8, %c8, %dyn_stride)
          {chan_name = @dyn_chan, operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
          : (memref<8x8xi32>, index, index, index, index, index, index)
          -> !air.async.token
      aie.end
    }
  }
}