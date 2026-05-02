// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit %s | aie-opt --conduit-to-dma - | FileCheck %s
//
// Pass B + Pass C end-to-end test: air.channel.put/get → conduit Tier 3 → aie hardware ops.
//
// Pipeline: --allow-unregistered-dialect --air-channel-to-conduit --conduit-to-dma
//
// Design note on tile placement:
//   Pass B (--air-channel-to-conduit) converts air.channel.put/get to
//   conduit.put_memref_async/get_memref_async, and emits conduit.create for each
//   air.channel declaration.  The emitted conduit.create has empty producer_tile /
//   consumer_tiles because AIR channel ops do not carry tile coordinates — those
//   come from a separate tile-placement step between Pass B and Pass C.
//
//   This test provides structural tile info via aie.shim_dma_allocation (for
//   shim producer) and aie.core (for compute consumer) so that Pass C's
//   inferAllTiles() can determine tile assignments.
//
// Input:
//   - aie.device(npu1_1col) with two tiles: shim(0,0) and compute(0,2)
//   - conduit.create @mychan with element_type=memref<64xi32>
//   - aie.shim_dma_allocation for shim producer (MM2S)
//   - aie.core for compute consumer with conduit.acquire/release
//   - air.channel.put and air.channel.get with 1-D descriptor:
//       offsets=[0], sizes=[64], strides=[1] → num_elems=64
//
// Expected after --air-channel-to-conduit:
//   - conduit.put_memref_async {name="mychan", num_elems=64, offsets=[0], sizes=[64], strides=[1]}
//   - conduit.get_memref_async {same attrs}
//   - No residual air.channel / air.wait_all ops
//
// Expected after --conduit-to-dma:
//   - aie.buffer on tile(0,2) : memref<64xi32>
//   - aie.lock (prod_lock, init=1) and (cons_lock, init=0) on tile(0,2)
//   - aie.shim_dma_allocation on tile(0,0)
//   - aie.flow: tile(0,0) DMA:0 → tile(0,2) DMA:0
//   - aie.mem on tile(0,2) with aie.dma_start(S2MM, 0, ...) and aie.dma_bd

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {

// --- Consumer-tile buffer: conduit.create → aie.buffer ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "mychan_cons_buff_0"

// --- Consumer-tile locks ---
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "mychan_cons_prod_lock_0"
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "mychan_cons_cons_lock_0"

// --- func body: put_memref_async and get_memref_async are erased by Phase 7 ---
// CHECK:     func.func @test(

// --- Shim DMA allocation (from input, precedes generated locks) ---
// CHECK:     aie.shim_dma_allocation @mychan_shim_alloc

// --- Shim-side locks and flow ---
// CHECK:     aie.lock(%{{.*}}tile_0_0
// CHECK-SAME:   sym_name = "mychan_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_0
// CHECK-SAME:   sym_name = "mychan_cons_lock_0"
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)

// --- Tile DMA region: conduit.create → aie.mem with S2MM BD ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%{{.*}}mychan_cons_buff_0
// CHECK:       aie.use_lock(%[[CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.end
// CHECK:     }

// --- No residual air.channel or conduit.* ops ---
// CHECK-NOT: air.channel
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.put_memref_async
// CHECK-NOT: conduit.get_memref_async
// CHECK-NOT: conduit.put_memref
// CHECK-NOT: conduit.get_memref

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // conduit.create without tile attrs (structural tile info below).
    conduit.create @mychan {depth = 1 : i64,
                    element_type = memref<64xi32>}

    // Structural tile info: compute tile(0,2) consumes @mychan.
    %core = aie.core(%tile_0_2) {
      %w = conduit.acquire {name = @mychan, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<64xi32>>
      %e = conduit.subview_access %w {index = 0 : i64} : !conduit.window<memref<64xi32>> -> memref<64xi32>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<64xi32>>
      aie.end
    }

    func.func @test(%src: memref<64xi32>, %dst: memref<64xi32>) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index

      // air.channel.put async with 1-D descriptor:
      //   offsets=[0], sizes=[64], strides=[1], num_elems=64
      // operand_segment_sizes = [ndeps=0, nidx=0, nmemref=1, noffsets=1, nsizes=1, nstrides=1]
      %tok0 = "air.channel.put"(%src, %c0, %c64, %c1)
          {chan_name = @mychan,
           operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<64xi32>, index, index, index) -> !air.async.token

      // air.channel.get async with same descriptor
      %tok1 = "air.channel.get"(%dst, %c0, %c64, %c1)
          {chan_name = @mychan,
           operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<64xi32>, index, index, index) -> !air.async.token

      return
    }

    // Structural tile info: shim tile(0,0) is the MM2S producer for @mychan.
    aie.shim_dma_allocation @mychan_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @mychan}
  }
}
