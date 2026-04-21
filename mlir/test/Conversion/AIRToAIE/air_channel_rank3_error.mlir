// RUN: air-opt --allow-unregistered-dialect --air-channel-to-conduit --verify-diagnostics %s
//
// Regression test: B-7 — rank-3 memref operand in air.channel.put is handled
// by collapsing leading dimensions to rank-2 (emitting a warning, not an error).
//
// memref<4x4x4xi32> → memref<16x4xi32> (4*4=16 leading dims folded).
// The air.channel.put offsets/sizes/strides are 1-D descriptors independent of
// memref shape, so the DMA descriptor itself is unaffected; only element_type
// on conduit.create is collapsed to rank-2.
//
// Uses air-opt (which has the AIR dialect registered) so that actual
// air.channel syntax can be used.
//
// The channel decl is inside aie.device so Pass B collects it for erasure.
// The put op is inside aie.core so Pass B processes it (rank-3 warning).
// After Phase 5 fix: air.channel @chan3d is erased even though the replacement
// conduit.put_memref_async still references @chan3d — that reference is valid
// because conduit.create @chan3d takes over as the canonical symbol definition.

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)

    "air.channel"() {sym_name = "chan3d", size = [1, 1]} : () -> ()

    aie.core(%tile_0_3) {
      %src = memref.alloca() : memref<4x4x4xi32, 2>
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index

      // air.channel.put with a rank-3 memref (4x4x4xi32) — B-7 collapses
      // to rank-2 and emits a warning instead of a hard error.
      // expected-warning @below {{air-channel-to-conduit: rank-3 memref operand for @chan3d collapsed to rank-2}}
      "air.channel.put"(%src, %c0, %c0, %c0, %c4, %c4, %c4, %c16, %c4, %c1)
          {chan_name = @chan3d,
           operand_segment_sizes = array<i32: 0, 0, 1, 3, 3, 3>}
          : (memref<4x4x4xi32, 2>, index, index, index, index, index, index,
             index, index, index) -> ()
      aie.end
    }
  }
}
