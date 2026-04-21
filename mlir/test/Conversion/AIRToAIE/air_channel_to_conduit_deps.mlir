// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B dep-list threading test.
//
// Verifies that async dependency tokens from prior air.channel.put/get results
// are correctly threaded through to the emitted conduit put/get_memref_async ops.
//
// The key mechanism: Pass B calls replaceAllUsesWith() on the old air async token
// result after emitting the conduit op. This redirects all uses (including dep-list
// operands of subsequent ops) to the new !conduit.dma.token. When a later put/get
// is processed, its dep operand already has DMATokenType and passes the isa<> filter.
//
// Test cases:
//   1. put-then-get: get dep on put token (put→get chain, common in producer-consumer)
//   2. get-then-put: put dep on get token (get→put chain, common in read-modify-write)
//   3. wait_all fan-in: wait_all with two conduit deps (both from put/get)
//   4. wait_all dep propagation: put → wait_all(put_tok) → get(dep=merged_wait_all_result)
//      Tests that the merged token from wait_all propagates as a dep to a subsequent get.
//      This is the exact PASSB-DEP-001 regression scenario (fixed in Task #24).
//
// NOT tested here (known limitation, documented in AirChannelToConduit.cpp):
//   - air.execute tokens as deps: these remain !air.async.token and are dropped
//     because there is no conduit equivalent for non-DMA ordering tokens.

// CHECK-LABEL: module

// All conduit.create ops appear grouped in the aie.device body (before aie.core).
// CHECK: conduit.create @putGet
// CHECK: conduit.create @inChan
// CHECK: conduit.create @outChan
// CHECK: conduit.create @waChan
// CHECK: conduit.create @waDep

// Test 1: put-then-get dep chain (inside aie.core body).
// CHECK: %[[PUT:.*]] = conduit.put_memref_async {name = @putGet
// CHECK-SAME: : !conduit.dma.token
// CHECK: %[[GET:.*]] = conduit.get_memref_async
// CHECK-SAME: [%[[PUT]] : !conduit.dma.token]
// CHECK-SAME: name = @putGet

// Test 2: get-then-put dep chain.
// CHECK: %[[GTOK:.*]] = conduit.get_memref_async
// CHECK-SAME: name = @inChan
// CHECK: %[[PTOK:.*]] = conduit.put_memref_async
// CHECK-SAME: [%[[GTOK]] : !conduit.dma.token]
// CHECK-SAME: name = @outChan

// Test 3: wait_all fan-in.
// CHECK: %[[W0:.*]] = conduit.put_memref_async
// CHECK-SAME: name = @waChan
// CHECK: %[[W1:.*]] = conduit.get_memref_async
// CHECK-SAME: name = @waChan
// CHECK: conduit.wait_all_async %[[W0]], %[[W1]]
// CHECK-SAME: (!conduit.dma.token, !conduit.dma.token) -> !conduit.dma.token

// Test 4: PASSB-DEP-001 — put → pre-emitted wait_all → get[dep=merged].
// CHECK: %[[WD_PUT:.*]] = conduit.put_memref_async {name = @waDep
// CHECK: %[[WD_MERGED:.*]] = conduit.wait_all_async %[[WD_PUT]]
// CHECK-SAME: (!conduit.dma.token) -> !conduit.dma.token
// CHECK: conduit.get_memref_async[%[[WD_MERGED]] : !conduit.dma.token]
// CHECK-SAME: name = @waDep

// No residual air ops.
// CHECK-NOT: air.channel{{[^._]}}
// CHECK-NOT: air.wait_all

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)
    "air.channel"() {sym_name = "putGet", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "inChan", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "outChan", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "waChan", size = [1, 1]} : () -> ()
    "air.channel"() {sym_name = "waDep", size = [1, 1]} : () -> ()
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // Test 1: put-then-get dep chain.
      %src1 = memref.alloca() : memref<4xi32>
      %dst1 = memref.alloca() : memref<4xi32>
      %put_tok = "air.channel.put"(%src1, %c0, %c4, %c1)
          {chan_name = @putGet, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %get_tok = "air.channel.get"(%put_tok, %dst1, %c0, %c4, %c1)
          {chan_name = @putGet, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
          : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%get_tok) : (!air.async.token) -> ()

      // Test 2: get-then-put dep chain.
      %src2 = memref.alloca() : memref<4xi32>
      %dst2 = memref.alloca() : memref<4xi32>
      %get_tok2 = "air.channel.get"(%dst2, %c0, %c4, %c1)
          {chan_name = @inChan, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %put_tok2 = "air.channel.put"(%get_tok2, %src2, %c0, %c4, %c1)
          {chan_name = @outChan, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
          : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%put_tok2) : (!air.async.token) -> ()

      // Test 3: wait_all fan-in with two conduit deps.
      %src3 = memref.alloca() : memref<4xi32>
      %dst3 = memref.alloca() : memref<4xi32>
      %wput = "air.channel.put"(%src3, %c0, %c4, %c1)
          {chan_name = @waChan, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %wget = "air.channel.get"(%dst3, %c0, %c4, %c1)
          {chan_name = @waChan, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %wmerged = "air.wait_all"(%wput, %wget) : (!air.async.token, !air.async.token) -> !air.async.token
      "air.wait_all"(%wmerged) : (!air.async.token) -> ()

      // Test 4: put → wait_all(put_tok) → get(dep=merged_wait_all_result). PASSB-DEP-001.
      %src4 = memref.alloca() : memref<4xi32>
      %dst4 = memref.alloca() : memref<4xi32>
      %dput_tok = "air.channel.put"(%src4, %c0, %c4, %c1)
          {chan_name = @waDep, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<4xi32>, index, index, index) -> !air.async.token
      %dmerged = "air.wait_all"(%dput_tok) : (!air.async.token) -> !air.async.token
      %dget_tok = "air.channel.get"(%dmerged, %dst4, %c0, %c4, %c1)
          {chan_name = @waDep, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
          : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token
      "air.wait_all"(%dget_tok) : (!air.async.token) -> ()

      aie.end
    }
  }
}