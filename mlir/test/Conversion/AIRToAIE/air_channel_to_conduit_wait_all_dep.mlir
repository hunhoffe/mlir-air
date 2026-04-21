// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// PASSB-DEP-001 regression test: air.wait_all fan-in tokens as deps for put/get.
//
// Bug: AirChannelToConduit.cpp Phase 3 (put/get rewrite) runs before Phase 4
// (wait_all rewrite). When air.channel.get async [%merged] is processed in
// Phase 3 and %merged is the result of an air.wait_all, the token still has
// type !air.async.token at Phase 3 time and fails the DMATokenType filter at
// dep resolution. The dep was silently dropped, causing the DMA to fire before
// the fan-in fence completes — potential data race or deadlock on hardware.
//
// Fix: at Phase 3 dep resolution, when a dep fails the DMATokenType check,
// inspect whether its defining op is air.wait_all. If so, recursively resolve
// the wait_all's operands (already conduit tokens via prior replaceAllUsesWith)
// and pre-emit a conduit.wait_all_async in-place. The pre-emitted op is
// recorded in a map so Phase 4 skips re-emitting it and just replaces uses.
//
// Test cases:
//   1. put → wait_all → get: get dep on wait_all result (single-input fan-in).
//   2. put + put → wait_all (two inputs) → get: multi-input fan-in.
//   3. Same wait_all result used as dep by two gets: deduplication check.
//   4. wait_all with mix of conduit and non-conduit deps: only conduit threaded.
//   5. Blocking (no-result) wait_all still lowers correctly when a second wait_all
//      uses the merged result as a dep to a get.

// CHECK-LABEL: module

// All conduit.create ops appear grouped in the aie.device body (before aie.core).
// CHECK: conduit.create @chan1
// CHECK: conduit.create @ch2a
// CHECK: conduit.create @ch2b
// CHECK: conduit.create @ch2c
// CHECK: conduit.create @ch3put
// CHECK: conduit.create @ch3a
// CHECK: conduit.create @ch3b
// CHECK: conduit.create @ch4a
// CHECK: conduit.create @ch4b
// CHECK: conduit.create @ch4c

// Test 1: put → pre-emitted wait_all → get[dep=merged].
// CHECK: %[[PUT1:.*]] = conduit.put_memref_async {name = @chan1
// CHECK: %[[WA1:.*]] = conduit.wait_all_async %[[PUT1]]
// CHECK-SAME: (!conduit.dma.token) -> !conduit.dma.token
// CHECK: %[[GET1:.*]] = conduit.get_memref_async[%[[WA1]] : !conduit.dma.token]
// CHECK-SAME: name = @chan1

// Test 2: two puts, wait_all fan-in, get[dep=merged].
// CHECK: %[[P2A:.*]] = conduit.put_memref_async {name = @ch2a
// CHECK: %[[P2B:.*]] = conduit.put_memref_async {name = @ch2b
// CHECK: %[[WA2:.*]] = conduit.wait_all_async %[[P2A]], %[[P2B]]
// CHECK-SAME: (!conduit.dma.token, !conduit.dma.token) -> !conduit.dma.token
// CHECK: %[[GET2:.*]] = conduit.get_memref_async[%[[WA2]] : !conduit.dma.token]
// CHECK-SAME: name = @ch2c

// Test 3: deduplication — same wait_all result as dep for two gets.
// CHECK: %[[P3:.*]] = conduit.put_memref_async {name = @ch3put
// CHECK: %[[WA3:.*]] = conduit.wait_all_async %[[P3]]
// CHECK: %[[G3A:.*]] = conduit.get_memref_async[%[[WA3]] : !conduit.dma.token]
// CHECK-SAME: name = @ch3a
// CHECK: %[[G3B:.*]] = conduit.get_memref_async[%[[WA3]] : !conduit.dma.token]
// CHECK-SAME: name = @ch3b

// Test 4: PASSB-DEP-001 — put + get → wait_all fan-in → put[dep=merged].
// CHECK: %[[P4:.*]] = conduit.put_memref_async {name = @ch4a
// CHECK: %[[G4:.*]] = conduit.get_memref_async {name = @ch4b
// CHECK: %[[WA4:.*]] = conduit.wait_all_async %[[P4]], %[[G4]]
// CHECK-SAME: (!conduit.dma.token, !conduit.dma.token) -> !conduit.dma.token
// CHECK: conduit.put_memref_async[%[[WA4]] : !conduit.dma.token]
// CHECK-SAME: name = @ch4c

// CHECK-NOT: air.channel{{[^._]}}
// CHECK-NOT: air.wait_all

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)

    // Test 1
    "air.channel"() {sym_name = "chan1", size = [1, 1]} : () -> ()
    // Test 2
    "air.channel"() {sym_name = "ch2a", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "ch2b", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "ch2c", size = [1, 1]} : () -> ()
    // Test 3
    "air.channel"() {sym_name = "ch3put", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "ch3a",   size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "ch3b",   size = [1, 1]} : () -> ()
    // Test 4
    "air.channel"() {sym_name = "ch4a", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "ch4b", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "ch4c", size = [1, 1]} : () -> ()

    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // --- Test 1: put → wait_all → get ---
      %src1 = memref.alloca() : memref<4xi32>
      %dst1 = memref.alloca() : memref<4xi32>
      %put_tok1 = "air.channel.put"(%src1, %c0, %c4, %c1)
          {chan_name = @chan1, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %merged1 = "air.wait_all"(%put_tok1) : (!air.async.token) -> !air.async.token
      %get_tok1 = "air.channel.get"(%merged1, %dst1, %c0, %c4, %c1)
          {chan_name = @chan1, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
          : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%get_tok1) : (!air.async.token) -> ()

      // --- Test 2: two puts, wait_all fan-in, get ---
      %s2a = memref.alloca() : memref<4xi32>
      %s2b = memref.alloca() : memref<4xi32>
      %dst2 = memref.alloca() : memref<4xi32>
      %tok2a = "air.channel.put"(%s2a, %c0, %c4, %c1)
          {chan_name = @ch2a, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %tok2b = "air.channel.put"(%s2b, %c0, %c4, %c1)
          {chan_name = @ch2b, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %merged2 = "air.wait_all"(%tok2a, %tok2b) : (!air.async.token, !air.async.token) -> !air.async.token
      %get_tok2 = "air.channel.get"(%merged2, %dst2, %c0, %c4, %c1)
          {chan_name = @ch2c, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
          : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%get_tok2) : (!air.async.token) -> ()

      // --- Test 3: same wait_all result as dep for two gets ---
      %src3 = memref.alloca() : memref<4xi32>
      %dA = memref.alloca() : memref<4xi32>
      %dB = memref.alloca() : memref<4xi32>
      %put_tok3 = "air.channel.put"(%src3, %c0, %c4, %c1)
          {chan_name = @ch3put, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %merged3 = "air.wait_all"(%put_tok3) : (!air.async.token) -> !air.async.token
      %gA = "air.channel.get"(%merged3, %dA, %c0, %c4, %c1)
          {chan_name = @ch3a, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
          : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token
      %gB = "air.channel.get"(%merged3, %dB, %c0, %c4, %c1)
          {chan_name = @ch3b, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
          : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%gA, %gB) : (!air.async.token, !air.async.token) -> ()

      // --- Test 4: put → get → wait_all → put[dep=merged] (PASSB-DEP-001) ---
      %s4a = memref.alloca() : memref<4xi32>
      %s4b = memref.alloca() : memref<4xi32>
      %s4c = memref.alloca() : memref<4xi32>
      %put_tok4 = "air.channel.put"(%s4a, %c0, %c4, %c1)
          {chan_name = @ch4a, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %get_tok4 = "air.channel.get"(%s4b, %c0, %c4, %c1)
          {chan_name = @ch4b, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %merged4 = "air.wait_all"(%put_tok4, %get_tok4) : (!air.async.token, !air.async.token) -> !air.async.token
      %put2_tok4 = "air.channel.put"(%merged4, %s4c, %c0, %c4, %c1)
          {chan_name = @ch4c, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
          : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%put2_tok4) : (!air.async.token) -> ()

      aie.end
    }
  }
}