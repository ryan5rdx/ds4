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
- **Attention heads**: split across ranks (`tp_split_attn = g->tp_world==2`).
- **Attention output**: reconstructed via one "big gate" (full-batch RDMA
  hidden-state exchange) per layer.
- **Shared experts**: replicated, only row-split above
  `DS4_TP_PREFILL_SPLIT_MIN` (default 32 tokens).

### 2. What grows with context — and is NOT split
- **Compressed KV cache** grows linearly: `comp_cap = ctx_size / ratio + 2`
  (ratio-4 layers → `ctx/4`; see `kv_cache_init`, `ds4.c:12383`). The raw SWA
  cache is bounded by the sliding window (`DS4_N_SWA`), so it does *not* grow.
- **The indexer top-k scoring** over that growing compressed cache is
  **replicated on both ranks**. The comment at `ds4.c:28893` is explicit:

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

### A. Split the indexer score/top-k across ranks (highest value)
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
- Instrument or profile the per-layer stage timers
  (`ds4: prefill detail layer %u ... indexer=%.3f attn_rows=%.3f` at
  `ds4.c:13532`) to confirm the indexer/attn stage grows with context while the
  FFN stage is flat.

## Status
- [x] Root-cause analysis (above).
- [ ] Option A/B/C prototype on a branch off `tp-multi-slot-batching`.
- [ ] Re-measure sweep + power after a fix.
