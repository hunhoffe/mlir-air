// RUN: not air-opt --allow-unregistered-dialect --air-channel-flatten-indices %s 2>&1 | FileCheck %s
//
// Regression test (A-6): AirChannelIndexFlattener must NOT erase the channel
// decl when a dynamic-index error occurs.
//
// Before the fix, Phase 4 erased multiDimDeclsToErase unconditionally, even
// when signalPassFailure() had been called. This left the put/get ops (which
// could not be rewritten due to the dynamic index) with dangling symbol
// references — the @matrix decl was erased but the put still referenced it.
//
// After the fix, Phase 4 is guarded by `!passFailed`, so the decl is preserved
// when an error occurs. The test verifies:
//   1. The dynamic-index error is emitted.
//   2. The pass exits with failure (via `not aie-opt`).
//
// CHECK: error{{.*}}dynamic index operand
// CHECK-SAME: matrix_rank2

module {
  "air.channel"() {sym_name = "matrix_rank2", size = array<i64: 2, 3>} : () -> ()

  func.func @test(%buf : memref<4xi32>, %dyn_i : index) {
    %c0 = arith.constant 0 : index
    // First index is dynamic (%dyn_i cannot be extracted as a constant).
    // This triggers the error path and sets passFailed=true.
    "air.channel.put"(%dyn_i, %c0, %buf)
        {chan_name = @matrix_rank2,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<4xi32>) -> ()
    return
  }
}
