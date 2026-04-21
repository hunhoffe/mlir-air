// RUN: not air-opt --allow-unregistered-dialect --air-channel-flatten-indices %s 2>&1 | FileCheck %s
//
// --air-channel-flatten-indices error test.
//
// A put op with a dynamic index (function argument, not arith.constant) on
// a multi-dim channel must produce a hard error.
//
// CHECK: error{{.*}}dynamic index operand
// CHECK-SAME: matrix

module {
  "air.channel"() {sym_name = "matrix", size = array<i64: 2, 2>} : () -> ()

  func.func @test(%buf : memref<4xi32>, %i : index, %j : index) {
    // Dynamic indices (%i, %j are function arguments, not constants).
    "air.channel.put"(%i, %j, %buf)
        {chan_name = @matrix,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<4xi32>) -> ()
    return
  }
}
