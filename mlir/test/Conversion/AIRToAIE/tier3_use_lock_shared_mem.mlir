// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
//
// Regression test: Tier 3 use_lock on adjacent tiles via air.channel pipeline.
//
// Adjacent tiles (2,3) and (2,4): Pass B generates conduit.create, Pass C
// lowers to shared-memory routing with locks.  Core-side use_lock ops
// synchronize producer and consumer via shared buffer (no DMA).
//
// Verifies:
//   1. Both cores reference the same lock pair (shared-memory routing)
//   2. Producer: acquire prodLock, release consLock
//   3. Consumer: acquire consLock, release prodLock
//   4. No residual conduit ops

// CHECK:       %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}, 0) {init = 1
// CHECK:       %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}, 1) {init = 0

// --- Producer core ---
// CHECK:       aie.core
// CHECK:         aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK-NEXT:    aie.use_lock(%[[CONS_LOCK]], Release, 1)

// --- Consumer core: same locks, opposite polarity ---
// CHECK:       aie.core
// CHECK:         aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK-NEXT:    aie.use_lock(%[[PROD_LOCK]], Release, 1)

// --- No residual conduit ops ---
// CHECK-NOT: conduit.put_memref_async
// CHECK-NOT: conduit.get_memref_async
// CHECK-NOT: conduit.create

module @test_tier3_shared_mem {
  aie.device(xcve2802) @segment_0 {
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %core_2_3 = aie.core(%tile_2_3) {
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.put"(%alloc) {chan_name = @channel_0, id = 1 : i32, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>} : (memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.get"(%alloc) {chan_name = @channel_0, id = 2 : i32, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>} : (memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }
    "air.channel"() {sym_name = "channel_0"} : () -> ()
  }
}
