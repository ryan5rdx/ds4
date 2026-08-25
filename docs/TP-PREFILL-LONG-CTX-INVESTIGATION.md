# TP Prefill Performance Drop at Long Context — Investigation

Target: **DeepSeek V4 Flash**, 2× M2 Ultra, Tensor Parallelism over RDMA.
Symptom: TP **prefill** throughput collapses as context grows
(381 → 183 t/s from 2k → 128k), and node power **drops** during prefill
(~120 W at <50k → ~100 W at ~180k+). Lower power at larger context = the GPU is
**under-utilized / waiting**, not compute-bound.

## What the code says (TP = 2 nodes)

### 1. What TP actually splits
- **Routed experts**: ownership-split 50/50 — each rank holds half the experts.
  This is the *compute* that scales with TP and keeps both GPUs busy.
- **Attention**: split differently in the two phases, and the distinction is
  load-bearing for everything below.
  - *Decode* splits attention **heads** (`tp_split_attn = g->tp_world==2`, in the
    decode encoder).
  - *Prefill* splits attention **rows** (`tp_row_split_attn`, in
    `metal_graph_encode_layer_attention_batch`), and **only for `pos0 == 0`
    chunks** — every split shape is gated on `const bool zero_prefix = pos0 == 0`.
    With the default 4096-token chunking, a 131072-token prefill row-splits
    chunk 0 and runs the **entire** attention path (q_b, attention core, output
    projection, indexer, compressor) replicated on both ranks for the other 31
    chunks.
- **Attention output**: reconstructed via one "big gate" (full-batch RDMA
  hidden-state exchange) per layer.
- **Shared experts**: replicated, only row-split above
  `DS4_TP_PREFILL_SPLIT_MIN` (default 32 tokens).

### 2. What grows with context — and is NOT split
- **Compressed KV cache** grows linearly: `comp_cap = ctx_size / ratio + 2`
  (ratio-4 layers → `ctx/4`). The CPU path computes it in `kv_cache_init`; the
  one that matters on the Metal pair is `g->layer_comp_cap[il]` in
  `metal_graph_alloc_raw_cap()`. The raw SWA cache is bounded by the sliding
  window (`DS4_N_SWA`), so it does *not* grow.
- **The indexer top-k scoring** over that growing compressed cache is
  **replicated on both ranks**. The TP row-split comment at the top of
  `metal_graph_encode_layer_attention_batch()` in `ds4.c` is explicit:

  > "ratio-4 layer with indexer top-k, whose per-token score/top-k selection
  > **stays replicated** while the attention consumption splits by rows"

  and earlier: "q_a and the KV path stay full (**both ranks need every row's
  KV, and the compressor/indexer keep updating their state from full rows**)".

So: as context grows, each rank does an ever-larger *identical* indexer score +
top-k pass over the whole compressed KV, plus full-row compressor/indexer state
updates. This work (a) scales with context, (b) is duplicated on both GPUs (no
TP speedup), and (c) is memory/latency-bound (score all compressed keys, select
top-k), not heavy-math bound.

### 3. Why power drops
At small context, prefill is dominated by the **split expert FFN** (dense
compute → both GPUs busy → high power, ~120 W). As context grows, the
**replicated indexer/compressor/compressed-attention** pass becomes a larger
fraction of each chunk. That pass is memory-bound and duplicated across ranks,
so the GPU spends proportionally more time reading the growing compressed KV /
waiting on the per-layer big-gate RDMA exchange rather than doing parallel
compute → lower utilization → lower power (~100 W). This matches the observed
prefill t/s collapse almost exactly tracking `comp_cap ∝ ctx`.

## Improvement options (ordered by expected value / risk)

### A0. Lift the `pos0 == 0` restriction on prefill row splitting (do this first)
All the `tp_row_split_attn` shapes are gated on `zero_prefix`, so on a 128k
prefill only the first 4096-token chunk splits at all and the other ~31 chunks
duplicate the *whole* attention path on both ranks — not just the indexer top-k.
That makes this a strictly larger win than A and probably a cheaper one: the
split machinery already exists and is proven on chunk 0, so the work is
establishing that the compressor/indexer state stays consistent when a chunk
starts at `pos0 > 0`. Measure first: the theory above predicts most of the
long-context prefill loss lives here, and if it does, A and B are refinements
on top rather than the main event.

### A. Split the indexer score/top-k across ranks
Currently replicated (identical work on both GPUs). Make rank r score only its
half of the compressed rows, then merge top-k via one small RDMA reduction per
layer. This directly parallelizes the context-proportional work. Risk: top-k
selection must stay identical across ranks for the attention rows to be
consistent — needs a deterministic merge. Moderate complexity.

### B. Split / shard the compressed KV cache across ranks (peer-read)
Port the existing CUDA-TP cache-duplication + peer-read machinery to the Metal
TP path: `cuda_tp_attn_cache_dup`, `cuda_tp_attn_peer_read`,
`layer_attn_comp_cache_tp[]`, `ds4_gpu_tensor_copy_xdev`
(`ds4.c:20207-20275`, `23240-23300`). Each rank stores half the compressed rows
and reads the peer's half over RDMA. Cuts per-rank compressed-KV footprint and
indexer cost in half. The CUDA path is a working template. Moderate-high
complexity (RDMA read path for Metal), but the infra exists.

### C. Reduce replicated full-row work during chunked prefill
The compressor/indexer "keep updating their state from full rows" even when
attention rows are split. If the compressor state update can be made
rank-local / incremental per chunk (only the new chunk's rows), the
context-proportional replicated cost shrinks. Investigate
`metal_graph_attn_comp_prefill_target` and the indexer state update path.

### D. Tune chunk size (`DS4_METAL_PREFILL_CHUNK`)
Larger chunks amortize the per-layer big-gate RDMA exchange over more rows,
raising the compute:communication ratio. Cheap to test; may help the "waiting
on the wire" component. Note the existing `DS4_TP_SUBGATE_PIPELINE` was
measured net-negative on M5 Max — retune on M2 Ultra.

### E. Increase raw-window coverage vs compressed (model/arch level)
`DS4_N_SWA` bounds raw attention, so beyond the window everything goes through
the replicated indexer path. Not a code knob (model-fixed), listed for
completeness.

## How to verify on the pair
- Re-run the TP sweep with `DS4_METAL_PREFILL_CHUNK` varied (e.g. 2048/4096/8192)
  to isolate the chunk-size / RDMA-exchange contribution.
- Use `DS4_METAL_ABLATE_INDEXER_SCORE` / `_TOPK` (diagnostic, changes output) to
  confirm indexer cost share at 128k vs 8k.
- Profile the per-layer stage timers. Two gotchas that cost time if you miss
  them: the `indexer=%.3f attn_rows=%.3f` line is the *second* fprintf in that
  block and requires **`DS4_PREFILL_PROFILE_TOKEN=1`** on top of the profile
  flag (the first line only reports `hc_norm/q/kv/token_loop/out`); and both
  live in the **CPU** batch path (`layer_attention_raw_swa_batch`), so for the
  Metal TP pair use the Metal stage profilers instead
  (`DS4_METAL_INDEXER_STAGE_PROFILE`, `DS4_METAL_Q_STAGE_PROFILE`,
  `metal_graph_layer_stage_profile_boundary`).
- Log `pos0` per chunk alongside the stage timers to confirm the A0 hypothesis
  directly: if only the `pos0 == 0` chunk shows a split attention cost and every
  later chunk shows the full replicated cost, A0 is the dominant term.

## Status
- [x] Root-cause analysis (above). Note: the first revision of this document
      claimed prefill splits attention *heads*; it splits *rows*, and only on
      `pos0 == 0` chunks. Corrected 2026-08-24, and it added option A0.
- [ ] Option A/B/C prototype on a branch off `tp-multi-slot-batching`.
- [ ] Re-measure sweep + power after a fix.
