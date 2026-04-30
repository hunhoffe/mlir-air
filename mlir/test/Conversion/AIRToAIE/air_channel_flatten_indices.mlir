// RUN: air-opt --allow-unregistered-dialect --air-channel-flatten-indices %s | FileCheck %s
//
// --air-channel-flatten-indices test.
//
// Verifies that a 2×2 multi-dimensional air.channel declaration and its
// put/get ops with static constant indices are flattened to 4 scalar channels.
//
//   @matrix [2, 2]  →  @matrix_0_0, @matrix_0_1, @matrix_1_0, @matrix_1_1
//
// put @matrix[%c0, %c0]  →  chan_name rewritten to @matrix_0_0
// put @matrix[%c0, %c1]  →  chan_name rewritten to @matrix_0_1
// get @matrix[%c1, %c0]  →  chan_name rewritten to @matrix_1_0
// get @matrix[%c1, %c1]  →  chan_name rewritten to @matrix_1_1

// CHECK-LABEL: module

// Flat channel declarations emitted (four scalar channels).
// MLIR prints attributes in alphabetical order: size before sym_name.
// CHECK-DAG: "air.channel"() {size = [1, 1], sym_name = "matrix_0_0"
// CHECK-DAG: "air.channel"() {size = [1, 1], sym_name = "matrix_0_1"
// CHECK-DAG: "air.channel"() {size = [1, 1], sym_name = "matrix_1_0"
// CHECK-DAG: "air.channel"() {size = [1, 1], sym_name = "matrix_1_1"

// Original multi-dim channel is erased (no "size = [2, 2]").
// CHECK-NOT: size = [2, 2]

// Rewritten channel names for put ops.
// CHECK-DAG: chan_name = @matrix_0_0
// CHECK-DAG: chan_name = @matrix_0_1

// Rewritten channel names for get ops.
// CHECK-DAG: chan_name = @matrix_1_0
// CHECK-DAG: chan_name = @matrix_1_1

module {
  // Multi-dimensional 2×2 channel declaration.
  "air.channel"() {sym_name = "matrix", size = [2, 2]} : () -> ()

  func.func @test(%buf : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // put @matrix[0, 0] — operand_segment_sizes: [0 deps, 2 idx, 1 memref, 0 offsets, 0 sizes, 0 strides]
    "air.channel.put"(%c0, %c0, %buf)
        {chan_name = @matrix,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<4xi32>) -> ()

    // put @matrix[0, 1]
    "air.channel.put"(%c0, %c1, %buf)
        {chan_name = @matrix,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<4xi32>) -> ()

    // get @matrix[1, 0]
    "air.channel.get"(%c1, %c0, %buf)
        {chan_name = @matrix,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<4xi32>) -> ()

    // get @matrix[1, 1]
    "air.channel.get"(%c1, %c1, %buf)
        {chan_name = @matrix,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<4xi32>) -> ()

    return
  }
}
