//===- air_gemm_multidim.mlir - Multi-dim GEMM via Conduit -----*- MLIR -*-===//
//
// Replacement test for async_gemm_to_locks.mlir / async_gemm_to_objectfifo.mlir.
// Tests 2x2 multi-dimensional air.channel indices with --air-channel-flatten-indices
// pre-pass. Uses aie.device directly (pre-hierarchy IR).
//
// Scenario: 2x2 grid of compute tiles, each gets its own channel via
// flatten-indices. Demonstrates that --air-channel-flatten-indices +
// --air-channel-to-conduit correctly handles the [M,N] -> scalar mapping.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --allow-unregistered-dialect \
// RUN:   --air-channel-flatten-indices \
// RUN:   --air-channel-to-conduit \
// RUN:   %s | FileCheck %s

// After flatten-indices, @data_in[2,2] becomes 4 scalar channels.
// Each gets its own conduit.create.

// CHECK-LABEL: module

// Flattened conduit.create ops (one per scalar channel).
// CHECK-DAG: conduit.create @data_in_0_0
// CHECK-DAG: conduit.create @data_in_0_1
// CHECK-DAG: conduit.create @data_in_1_0
// CHECK-DAG: conduit.create @data_in_1_1
// Conduit put/get ops reference scalar channel names.
// CHECK-DAG: conduit.put_memref_async {{.*}}name = @data_in_0_0
// CHECK-DAG: conduit.put_memref_async {{.*}}name = @data_in_0_1
// CHECK-DAG: conduit.put_memref_async {{.*}}name = @data_in_1_0
// CHECK-DAG: conduit.put_memref_async {{.*}}name = @data_in_1_1

// No residual multi-dim channel or air ops.
// CHECK-NOT: size = array<i64: 2, 2>
// CHECK-NOT: air.channel

module {
  aie.device(xcve2802) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)

    // 2x2 channel declaration (multi-dimensional)
    "air.channel"() {sym_name = "data_in", size = array<i64: 2, 2>} : () -> ()

    // MemTile: distributes data to 4 tiles via @data_in[i,j]
    %l2_buf = aie.buffer(%tile_0_1) {sym_name = "l2_buf"} : memref<128xi32, 1>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // put @data_in[0, 0]
    "air.channel.put"(%c0, %c0, %l2_buf)
        {chan_name = @data_in,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<128xi32, 1>) -> ()

    // put @data_in[0, 1]
    "air.channel.put"(%c0, %c1, %l2_buf)
        {chan_name = @data_in,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<128xi32, 1>) -> ()

    // put @data_in[1, 0]
    "air.channel.put"(%c1, %c0, %l2_buf)
        {chan_name = @data_in,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<128xi32, 1>) -> ()

    // put @data_in[1, 1]
    "air.channel.put"(%c1, %c1, %l2_buf)
        {chan_name = @data_in,
         operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
        : (index, index, memref<128xi32, 1>) -> ()

    // Compute tile (0,0) gets @data_in[0, 0]
    aie.core(%tile_0_3) {
      %c0_0 = arith.constant 0 : index
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.get"(%c0_0, %c0_0, %alloc)
          {chan_name = @data_in,
           operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
          : (index, index, memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }

    // Compute tile (0,1) gets @data_in[0, 1]
    aie.core(%tile_1_3) {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.get"(%c0_1, %c1_1, %alloc)
          {chan_name = @data_in,
           operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
          : (index, index, memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }

    // Compute tile (1,0) gets @data_in[1, 0]
    aie.core(%tile_0_4) {
      %c1_2 = arith.constant 1 : index
      %c0_2 = arith.constant 0 : index
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.get"(%c1_2, %c0_2, %alloc)
          {chan_name = @data_in,
           operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
          : (index, index, memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }

    // Compute tile (1,1) gets @data_in[1, 1]
    aie.core(%tile_1_4) {
      %c1_3 = arith.constant 1 : index
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.get"(%c1_3, %c1_3, %alloc)
          {chan_name = @data_in,
           operand_segment_sizes = array<i32: 0, 2, 1, 0, 0, 0>}
          : (index, index, memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }
  }
}
