//===- air_channel_to_pools.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-channels-to-pools' -split-input-file | FileCheck %s
// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-channels-to-pools' -split-input-file | aie-opt --split-input-file --aie-objectFifo-stateful-transform | FileCheck %s --check-prefix=LOWERED

// An L1-to-L1 channel becomes a pool on each core's tile, a core endpoint and
// a DMA endpoint per pool, and a route between the DMA endpoints. mlir-aie's
// own passes then give the pools buffers and locks, the endpoints channels and
// BD chains, and the cores their lock accesses: nothing on the AIR side
// emitted a lock or a descriptor.

// CHECK-LABEL: aie.device(xcvc1902)
// CHECK-DAG:   %[[T11:.*]] = aie.tile(1, 1)
// CHECK-DAG:   %[[T12:.*]] = aie.tile(1, 2)
// CHECK:       aie.objectfifo.pool @air_channel_0_prod_pool(%[[T11]]) {depth = 1 : i32} : memref<32xi32> {
// CHECK:         aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32}
// CHECK:       aie.objectfifo.core_endpoint @air_channel_0_prod(%[[T11]]) fills @air_channel_0_prod_pool
// CHECK:       aie.objectfifo.dma_endpoint @air_channel_0_prod_dma(%[[T11]]) drains @air_channel_0_prod_pool {fifoName = "air_channel_0"}
// CHECK:       aie.objectfifo.pool @air_channel_0_cons_pool(%[[T12]]) {depth = 1 : i32} : memref<32xi32> {
// CHECK:       aie.objectfifo.dma_endpoint @air_channel_0_cons_dma(%[[T12]]) fills @air_channel_0_cons_pool {fifoName = "air_channel_0"}
// CHECK:       aie.objectfifo.core_endpoint @air_channel_0_cons(%[[T12]]) drains @air_channel_0_cons_pool
// CHECK:       aie.route from @air_channel_0_prod_dma to [@air_channel_0_cons_dma]
// CHECK:       aie.core(%[[T12]]) {
// CHECK:         %[[OBJ:.*]] = aie.objectfifo.acquire @air_channel_0_cons(1) : memref<32xi32>
// CHECK:         aie.objectfifo.release @air_channel_0_cons(1)
// CHECK:       aie.core(%[[T11]]) {
// CHECK:         aie.objectfifo.acquire @air_channel_0_prod(1) : memref<32xi32>
// CHECK:         aie.objectfifo.release @air_channel_0_prod(1)

// LOWERED-LABEL: aie.device(xcvc1902)
// LOWERED-DAG:   aie.buffer({{.*}}) {sym_name = "air_channel_0_prod_buff_0"} : memref<32xi32>
// LOWERED-DAG:   aie.buffer({{.*}}) {sym_name = "air_channel_0_cons_buff_0"} : memref<32xi32>
// LOWERED:       aie.flow({{.*}}, DMA : 0, {{.*}}, DMA : 0)
// LOWERED:       aie.core
// LOWERED:         aie.use_lock
// LOWERED:       aie.core
// LOWERED:         aie.use_lock
// LOWERED:       aie.mem
// LOWERED:         aie.dma_start(MM2S, 0
// LOWERED:         aie.use_lock
// LOWERED:         aie.dma_bd({{.*}} : memref<32xi32> offset = 0 len = 32)
// LOWERED:       aie.mem
// LOWERED:         aie.dma_start(S2MM, 0
// LOWERED:         aie.use_lock
// LOWERED:         aie.dma_bd({{.*}} : memref<32xi32> offset = 0 len = 32)

aie.device(xcvc1902) {
  %0 = aie.tile(1, 1)
  %1 = aie.tile(1, 2)
  air.channel @channel_0 [1, 1]
  %2 = aie.core(%1) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    air.channel.get  @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    aie.end
  }
  %3 = aie.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    air.channel.put  @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    aie.end
  }
}

// -----

// An end with no put is L3: the route starts at a shim endpoint the runtime
// drives, and the get's strided access becomes the DMA endpoint's dimensions
// rather than a BD the AIR side has to lay out. A packet channel puts its
// header on the route, pinned id and all.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK-DAG:   %[[T02:.*]] = aie.tile(0, 2)
// CHECK:       aie.route_endpoint @air_channel_1_shim_in(%{{.*}}) DMA {fifoName = "air_channel_1"}
// CHECK:       aie.objectfifo.pool @air_channel_1_cons_pool(%[[T02]]) {depth = 2 : i32} : memref<8x4xi32>
// CHECK:       aie.objectfifo.dma_endpoint @air_channel_1_cons_dma(%[[T02]]) fills @air_channel_1_cons_pool {dimensions = #aie<bd_dim_layout_array_array{{\[}}[<size = 4, stride = 8>, <size = 8, stride = 1>]]>, fifoName = "air_channel_1"}
// CHECK:       aie.route from @air_channel_1_shim_in to [@air_channel_1_cons_dma] {packet = #aie.packet_info<pkt_id = 5>}

// LOWERED-LABEL: aie.device(npu1_1col)
// LOWERED:       aie.packet_flow(5) {
// LOWERED:       aie.shim_dma_allocation @air_channel_1_shim_alloc({{.*}}, MM2S, 0, <pkt_id = 5>)
// LOWERED:       aie.mem
// LOWERED:         aie.dma_bd({{.*}} : memref<8x4xi32> offset = 0 len = 32 sizes = [4, 8] strides = [8, 1])

aie.device(npu1_1col) {
  %t = aie.tile(0, 2)
  air.channel @channel_1 [1, 1] {channel_type = "npu_dma_packet", packet_ids = [5], buffer_resources = 2 : i64}
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %alloc = memref.alloc() : memref<8x4xi32, 2>
    air.channel.get  @channel_1[] (%alloc[%c0, %c0] [%c4, %c8] [%c8, %c1]) : (memref<8x4xi32, 2>)
    memref.dealloc %alloc : memref<8x4xi32, 2>
    aie.end
  }
}
