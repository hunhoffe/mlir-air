# Lowering `air.channel` through pools and endpoints

A spike on whether `air.channel` can lower onto mlir-aie's pool and endpoint
form, the level `aie.objectfifo` itself splits into, so that the two projects
share one allocator, one DMA lowering and one core lowering. The answer is
that the one-to-one DMA case lowers today with no AIR-side lock or descriptor
code at all, and the table below says how far the rest of the direct path
reaches before the level runs out of vocabulary.

## What was built

`--air-to-aie='test-patterns=lower-air-channels-to-pools'` lowers a channel
with one put and one get to:

- a pool per L1 end on its core's tile, depth `buffer_resources`, one segment
  covering the object;
- a core endpoint and a DMA endpoint per pool, the transfer's sizes and
  strides on the DMA endpoint as its `dimensions`;
- an `aie.route_endpoint` on a shim for an end with no put or get, the
  runtime's to drive, with `fifoName` so allocation emits the shim
  allocation the AIR runtime looks up;
- an `aie.route` between the two DMA ends, carrying `#aie.packet_info` when
  the channel is `npu_dma_packet`, pinned id included;
- an acquire of one object and a release in place of the core's alloc and
  dealloc.

Then `aie-opt --aie-objectFifo-stateful-transform` gives the pools buffers
and locks, the endpoints channels and BD chains (a two-descriptor rotation
for a depth of two, the strided access as `sizes` and `strides` on each
descriptor, the packet header on each descriptor of a packet route), and the
cores their lock accesses. `air_channel_to_pools.mlir` runs both halves. The
pattern is 250 lines; the direct path in `AIRToAIEPass.cpp` and
`AIRToAIESchedulingUtils.cpp` has 46 sites that create a lock, a lock use, a
descriptor, a DMA start or a flow, spread over 12,700 lines.

## Against the existing objectfifo path

`test-patterns=lower-air-channels` already lowers a channel to
`aie.objectfifo`, and after mlir-aie's split pass lands in the same place. The
difference is what the AIR side can say:

| Channel property | On `aie.objectfifo` | On pools and endpoints |
| --- | --- | --- |
| put and get sizes and strides | `dimensionsToStream` for the producer, `dimensionsFromStreamPerConsumer` for each consumer; one transform for the whole fifo | one `dimensions` attribute per DMA endpoint, per segment |
| padding | `padDimensions` on the fifo | `padDimensions` on the draining endpoint |
| a buffer the design already owns | not expressible; the fifo allocates | `buffers = [@mine]` on the pool, kept by allocation |
| pinned DMA channel | `prod_dma_channel`, `cons_dma_channels` | `channelIndex` on the endpoint |
| packet switching | `transport = #aie.transport<dma, packet = <...>>` | `packet = #aie.packet_info<...>` on the route |
| L2 relay, split, join | `aie.objectfifo.link` | a pool on the mem tile with segments, one DMA endpoint per side |

So the pool level is the objectfifo level with the per-end and per-segment
knobs exposed, which is what a channel put or get carries. Nothing the
objectfifo path emits is lost, and the two share every pass below the split.

## Against the direct path

The direct path's test names are the inventory of what it does that
`aie.objectfifo` never did. Each row says whether the pool level carries it.

| Direct-path feature (test) | Pool level | How |
| --- | --- | --- |
| ping-pong (`to_locks_ping_pong`, `buffer_resources`) | yes | pool depth |
| strided transfers, padding (`pad`) | yes | endpoint `dimensions`, `padDimensions` |
| pinned DMA channel (`dma_channel_pin`) | yes | endpoint `channelIndex` |
| L2 fan-in, fan-out (`memtile_chain_lock_v2_fanin`, `_fanout`) | yes | mem-tile pool with segments |
| broadcast (`specialize_channel_broadcast`) | yes | one route, several destinations |
| packet with one id, pinned or open (`packet_single_producer_one_flow`) | yes | route header, `--aie-assign-packet-ids` |
| column pins and affinity (`l2_memtile_column_pin*`) | yes | logical tiles, `aie-place-tiles` |
| conditional put or get (`to_locks_scf_if`) | yes | lower-cores keeps the runtime buffer index |
| shim BD programs (`air_shimcpy_*_with_shim_dma_bds`) | yes | `aie.route_endpoint` plus AIR's own runtime sequence |
| unequal trip counts, n-buffer rotation (`unequal_trip_no_rotation`, `n_buffer_rotation`) | partly | the chain rotates over the pool's depth; a rotation that differs from the depth is not expressible |
| shared L1 buffer between channels (`to_locks_shared_buffer`, `shared_l1_*`) | partly | a pool can name the buffer, but two pools cannot share one, and the verifier wants one filler and one drainer per segment |
| peeled rotation, index-switch relays (`peeled_rotation`, `index_switch_*`) | no | core-side scheduling the level does not describe; the core body would have to carry it |
| prefix and suffix descriptors (`prefix_suffix_bd`) | no | one descriptor shape per segment; a chain whose first or last descriptor differs has no spelling |
| producer refeed, chain-lock refeed (`producer_refeed`, `memtile_chain_lock_v2_refeed`) | no | a pool drains to one place; feeding a drained object back is a second route into the same pool the level rejects |
| many puts into one get (`packet_two_producers_one_channel`, `mm2s_flows_*`) | no | a route has one source, and a route endpoint may be named by one route |
| multi-id packet demux, per-descriptor packet tags (`packet_multi_id_*`, `bd_tag`) | no | one header per route; the packet type and id are the same on every descriptor of an endpoint |
| cascade (`npu_cascade`) | no at this level | cascade lowers from `aie.objectfifo` (transport `cascade`) and never reaches pools; a channel would go through the objectfifo op instead |
| MMIO (`npu_mmio`) | not applicable | no DMA, no flow |

Nine of the seventeen rows lower with what the level has; the rest split
between things the level could grow (a rotation length, a second route into a
pool, a per-descriptor header) and things that are core-side scheduling and
belong in the core body whichever way the data moves.

## What sharing would take

1. **A second source per endpoint.** Many-to-one is the largest gap and the
   one AIR uses most (every `packet_*_producers_*` test). A route already has
   several destinations; letting an endpoint be named by several routes, each
   with its own header, is a verifier change and an allocate change.
2. **Per-segment descriptor shape, not per-endpoint.** Prefix and suffix
   descriptors and per-descriptor packet tags are both "this segment's
   descriptor differs". `dimensions` is already per segment; `packet` is not.
3. **Rotation length as a pool property.** `iterCount` bounds a chain and
   `repeatCount` replays an object; a chain that rotates over fewer objects
   than the pool holds has no attribute. This covers the `unequal_trip` and
   `n_buffer_rotation` rows.
4. **Buffer sharing between pools.** Two pools naming one buffer with
   disjoint lifetimes is what `shared_l1_*` wants. The verifier's one-filler,
   one-drainer rule is per segment and would stay; the sharing is at the
   buffer.

None of the four is AIR-specific, so each would land in mlir-aie first and
AIR would consume it. Until then the pool path covers the channel shapes the
objectfifo path covers, with the per-end knobs the objectfifo op hides, and
the direct path stays for the rest.

## Two things learned on the way

- mlir-air's tests check the printed form of `#aie.packet_info`. Once
  mlir-aie stops printing `pkt_type = 0` (its #34), seven AIRToAIE tests
  need the same one-token edit on their CHECK lines. Nothing else in the
  suite moves.
- The `use-objectfifo` flag on `--air-to-aie` is the objectfifo path's
  production entry point, and it is covered: three tests turn it on
  (`async_gemm_to_objectfifo`, `air_channel_to_objectfifo_L1toL2`,
  `air_channel_to_objectfifo_L2_broadcast`) and twelve pass it off
  explicitly. The `lower-air-channels` test pattern exercises the same
  pattern in isolation. A pools path would sit beside it as a third
  backend, or replace it once it covers what those three tests cover.
