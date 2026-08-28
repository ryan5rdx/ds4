# Optimizations

Every speedup this fork carries over upstream, with the measurement that
justifies it. Split by prefill and decode, because the two are bound by
different things on this hardware and the same idea rarely pays in both.

For the raw measurement record — platform roofs, closed avenues, and the
hazards that produced wrong numbers — see [`benchmarks.md`](benchmarks.md).
Bugfixes and operability work with no throughput claim are in
[`fixes.md`](fixes.md).

**Rig.** Two Apple M2 Ultra, 60 GPU cores, 128 GB each, tensor-parallel over
Thunderbolt RDMA, Metal backend. DeepSeek V4 Flash MXFP4, 145.26 GiB total,
76.73 GiB resident per rank.

---

## Where it stands

Full sweep against pipeline-parallel at the same commit, `--step-mul 2`, 128
generated tokens per point. Tokens per second.

| ctx | TP prefill | PP prefill | prefill winner | TP decode | PP decode | decode winner |
|---:|---:|---:|---|---:|---:|---|
| 2048 | **423.03** | 316.98 | TP +34% | **40.91** | 27.67 | TP +48% |
| 4096 | **417.54** | 310.46 | TP +35% | **36.47** | 26.29 | TP +39% |
| 8192 | **500.16** | 348.21 | TP +44% | **35.98** | 26.08 | TP +38% |
| 16384 | **480.79** | 459.38 | TP +5% | **35.43** | 25.54 | TP +39% |
| 32768 | 461.20 | **530.00** | PP +15% | **33.73** | 24.11 | TP +40% |
| 65536 | 427.08 | **520.42** | PP +22% | **31.52** | 22.52 | TP +40% |
| 131072 | 367.29 | **444.38** | PP +21% | **28.13** | 20.57 | TP +37% |

Cold 131k prefill under tensor parallelism: **402.64 t/s**.

Tensor parallelism wins prefill to 16k and decode at every context; pipeline
parallelism wins prefill from 32k up. Both are worth keeping, and neither is
strictly better.

In a `--step-mul 2` sweep each row is the *increment* from the previous point,
so the 131072 row is the 65536→131072 chunk rather than a cold 131k prefill.

---

## Prefill

Prefill is the part of this fork that paid. The arc at 131k, each step measured
against the one above it:

| step | t/s | gain | output |
|---|---:|---:|---|
| pre-fork base | 183.4 | — | |
| upstream indexer stack | 221.50 | +20.8% | |
| non-zero-position row split | 237.44 | +7.2% | logits perturbed 0.049 |
| indexer row split | 283.64 | +19.5% | bit-identical |
| masked / low-top-k split | 342.25 | +20.5% | bit-identical |
| flash-attention simdgroup count | **367.29** | +7.3% | bit-identical |

**+100% over the pre-fork base, +65.8% over the same branch with the splits
disabled.** Cold single point 259.90 → 402.64 (+54.9%).

### Tensor-parallel row splitting

Three of the four gains above are row splits: a prefill chunk's rows are
divided across the pair instead of replicated. Each is gated by its own
environment variable and is default-on.

**Non-zero positions — +7.2% at 131k.** The attention row split previously
applied only to the first chunk of a prompt, so a long prefill split one chunk
and replicated the rest. All three kernels on that path now take an explicit
`(origin, rows)` pair, so a row sub-range is expressible rather than only a
prefix.

**Indexer — +19.5%, bit-identical.** The indexer score and top-k are row-split
alongside the attention rows each rank already owns. Because the split follows
rows a rank already holds, it adds no cross-rank merge and no gate traffic.

**Masked and low-top-k chunks — +20.5%, bit-identical.** Extends the split to
chunks carrying a compressed-key mask, which were previously refused outright.

Row splitting needs enough rows to divide, so it engages only above a minimum
chunk size and does not transfer to small batches. Both ranks must compute the
same split or the per-layer gates stop matching, which is why the gates are
read identically on each side rather than derived independently.

*Correctness.* The non-zero-position split perturbs logits by 0.049 against a
0.0055 control-vs-control baseline — about two f16 ULP — with 128/128 tokens and
305/305 argmax positions identical. The other two are bit-identical: 0 of
129,280 logits differ. Decode is unaffected by all three.

### Kernels

**Flash attention, four simdgroups rather than eight — +7.3% at 131k,
bit-identical.** For the non-vec dk512 kernel, four doubles the tiles produced
per barrier region and avoids an unroll that doubles live simdgroup matrices.
Shared memory pins one threadgroup per core either way, so the wider count buys
no occupancy — it only costs. `DS4_METAL_FA_NSG` restores the previous value.

**Register-resident indexer scorer — bit-identical.** The staged K tile stays in
simdgroup registers across the head loop instead of being reloaded from
threadgroup memory per head, halving threadgroup loads per matrix multiply.
Staging, barriers and the per-pair reduction order are unchanged. *From upstream
PR #831 (Adrian Galilea).* Gate: `make test-indexer-scorer`.

**Exact streaming top-512 — replaces a block argsort plus merge cascade.** A
running 512th-best threshold lets a single scan discard a candidate only when it
cannot belong to the final set; survivors compact into a 2048-slot threadgroup
buffer. Keys pack score and index into one 64-bit word under a (score
descending, index ascending) total order, so ties resolve deterministically and
the emitted list does not depend on compaction order — which matters, because
compaction uses a threadgroup atomic and is therefore unordered. Selected for
`top_k == 512` with at least 32 rows. *Ported from the CUDA stream selector in
upstream PR #832 (Adrian Galilea).* Rollback:
`DS4_METAL_DISABLE_TOPK_STREAM512`. Gate: `make test-topk-stream512`.

Between them, #831 and #832 are the largest prefill kernel wins on this branch.

---

## Decode

Decode is latency-bound here, and the honest summary is that the decode work
largely did not pay. The individually measured changes are at or below the
measurement floor at short context, and what gains exist are long-context.

Why: decode runs at the top P-state with full residency at roughly **8.19%** of
peak arithmetic throughput (2.26 / 27.6 FLOP/byte), drawing half of prefill's
power. Encoder spans already overlap ~2× with no idle pool inside a command
buffer — span union is 100% of the buffer, gap ≈ 0.000 ms at both 2k and 131k.
So stalls are *inside* kernels, not between them, and the entire
dispatch-reduction class of ideas cannot pay: there are no gaps to close.

The byte floor is 5.93 GB/rank/token at 449.2 GB/s = 13.20 ms; decode is 1.84×
off it.

**Track record: eleven ex-ante decode throughput estimates were measured on this
rig; the best realised was +0.12%.** Estimates here should be discounted
heavily, particularly anything in the grid-widening or occupancy class, which is
0-for-4. The closed-avenue table in [`benchmarks.md`](benchmarks.md) lists what
was tried.

### What shipped

**Shared-memory alias in the decode indexer scorer — +17.4% on the kernel,
byte-identical.** The score scratch is aliased over the staged-key buffer; their
lifetimes are disjoint, so allocating the maximum rather than the sum halves
threadgroup memory and lifts residency from one threadgroup per core to two on a
kernel bounded by occupancy rather than bandwidth. The grid is unchanged.

Worth little end-to-end at short context: indexed attention requires more than
1024 compressed rows, so at 2k with a ratio-4 layer (512 rows) the indexed
branch never executes at all. The crossover is context > 4096, and at 2k the
change is worth ~0.02 ms/token.

**Four rows per thread in the routed MXFP4 down projection — bit-identical.** An
r4 twin of the routed sum6 kernel accumulates four output rows per thread rather
than two, quartering the grid for the same work. Accumulation order within a row
is unchanged; only the number of rows a thread walks. Gate and up stay at two
rows, where widening measured as a regression.

The row count is shared between host and kernel — the dispatch computes the grid
from it and the kernel strides by it — so they are one quantity and must move
together.

**Split-K reduce sized to the keys actually present.** The reduce walked a fixed
number of work groups regardless of how many held keys, so a decode step
attending a short context reduced over empty partials. It now takes the live
count. `speed-bench/metal_flash_attn_decode_bench.c` is the bit-exactness
harness.

**Five decode fast paths admitted on earlier Apple silicon.** These kernels were
gated on device family rather than on capability, so hardware that supports the
same operations fell back unnecessarily. Extends the fusions ported in *upstream
PR #778 (david)*. No effect on the reference rig, which already took the fast
paths; the benefit is to earlier parts.

### What decode gained overall

The sweep's decode column predates some of this work. Later re-baselines on the
same configuration read **41.19 t/s at 2k** (three interleaved repeats, sd
0.015) and **29.4–29.8 at 131k**. Cross-session anchors span 41.1–42.3 at 2k — a
±1.4% spread that bounds the resolution of any single-run decode comparison, and
is wider than most of the individual changes above.

---

## Speculation: measured and declined

Both drafters were built, measured, and are not fundable on this hardware. The
n-gram proposer was removed; the DSpark path is carried but its production
policy correctly declines to speculate.

- **n-gram**: 1.097× on prose, 1.034× on code, with 91–97% of steps committing
  nothing.
- **DSpark**: verify cost fits `V(k) = 34.83 + 20.64k` ms, and `verify_layer` is
  **99.97%** of it — fixed GPU layer-encode overhead, not data movement, so it
  is immune to quantisation or bandwidth. Break-even needs mean commit > 4.41 at
  5 rows against 1.658 observed. **A completely free drafter would still reach
  only 24.89 t/s against a 41.96 t/s baseline** — for k ≤ 4 the break-even
  exceeds the maximum possible commit, so a perfect drafter loses.

Full numbers in [`benchmarks.md`](benchmarks.md) §5.

---

## Rollback and verification

Every default-on kernel change has a rollback environment variable and a
model-free gate that runs on any Metal device:

| change | rollback | gate |
|---|---|---|
| streaming top-512 | `DS4_METAL_DISABLE_TOPK_STREAM512` | `make test-topk-stream512` |
| register-resident scorer | `DS4_METAL_DISABLE_INDEXER_SCORES_TILED5` | `make test-indexer-scorer` |
| flash-attention simdgroups | `DS4_METAL_FA_NSG` | prefill A/B, bit-identical |
| routed MXFP4 r4 down | pipeline selection | `make test-mxfp4-metal` |
| prefill row splits | one variable per split | logit-delta gate, above |

`ds4_gpu_last_indexer_scorer()` and `ds4_gpu_last_indexer_topk()` report which
kernel a call actually selected, so an A/B test can prove it compared two paths
rather than one against itself. Both gates assert the arms differ before
comparing output — without that, a test whose arms silently resolve to the same
path passes green while comparing a path against itself.
