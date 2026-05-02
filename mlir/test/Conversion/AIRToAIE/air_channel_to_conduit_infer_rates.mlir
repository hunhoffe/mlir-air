// RUN: air-opt --allow-unregistered-dialect "--air-channel-to-conduit=infer-rates=true" %s 2>&1 | FileCheck %s
// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s --check-prefix=NOINFER
//
// --air-channel-to-conduit infer-rates pipeline integration test.
//
// When inferRates=true, the pass attaches producer_rates and
// consumer_rates to conduit.create based on the num_elems of put/get ops.
//
// When inferRates=false (the default), rates are NOT attached.
//
// This test uses full-buffer transfers (no sizes → num_elems=64 from
// memref<64xi32>).  Phase 2b patches slot_elems from the sentinel (1) to
// 64, and Phase 6 infers rates [64] from num_elems.
//
// Topology: channel @chan with one put and one get over memref<64xi32>.

// Explicit inferRates=true: rates should be attached.
// MLIR prints attributes alphabetically: consumer_rates, name, producer_rates.
// CHECK-LABEL: module
// CHECK: conduit.create @chan
// CHECK-SAME: consumer_rates = array<i64: 64>
// CHECK-SAME: producer_rates = array<i64: 64>

// Default (inferRates=false): rates must NOT be attached.
// NOINFER-LABEL: module
// NOINFER: conduit.create @chan
// NOINFER-NOT: producer_rates
// NOINFER-NOT: consumer_rates

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "chan", size = [1, 1]} : () -> ()
    aie.core(%tile_0_3) {
      %buf = memref.alloca() : memref<64xi32>
      "air.channel.put"(%buf)
          {chan_name = @chan, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<64xi32>) -> ()
      "air.channel.get"(%buf)
          {chan_name = @chan, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<64xi32>) -> ()
      aie.end
    }
  }
}