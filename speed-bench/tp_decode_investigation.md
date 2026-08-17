# DS4 Flash tensor-parallel decode: investigation record

Rig: **2 x Mac Studio M2 Ultra, 60-core GPU, 128 GB each**, DeepSeek V4 Flash MXFP4,
tensor parallel over Thunderbolt RDMA. Branch `tp-frontends-phase1`.

> **2026-08-14 audit note:** the original conclusion that routed MoE was the
> only remaining target is superseded by the addendum below.  In particular,
> the reported decode-split sweep did not execute the attention kernel used at
> ctx 512.  The historical measurements are retained after the addendum rather
> than silently rewritten.

## 2026-08-14 Apple/Metal audit addendum

### What the fresh audit invalidated

- The flat `DS4_METAL_DECODE_SPLITS=12..30` sweep in §8 did **not** test the
  ctx-512 attention core.  That setting is read only by indexed sparse
  attention (`ds4_metal.m:28983+`).  The indexed path requires more than the
  default 1024 compressed rows and more rows than top-k 512
  (`ds4.c:19724-19755,22989-22993`), roughly beyond 4K context for ratio 4.
  Ctx 512 -> 1024 remains in raw/gathered FlashAttention, where the split count
  was independently hard-coded to 32 (`ds4_metal.m:26311,27851`).  Therefore
  the claim that attention was not split-K/occupancy bound was unsupported.
- The `attncore ~= 45 MB/token ~= 17 GB/s` model omitted its largest temporary.
  With 32 local TP heads, the vec kernel writes, and the reduce kernel reads,
  `32 * NWG * (512 + 2) * 4` bytes per layer.  At NWG 32 that is 2,105,344
  bytes/layer, or **172.672 MiB/token** of partial-buffer payload over 43 layers
  (about 183 MiB when the reducer's repeated stats loads are counted), before
  KV staging or KV reads.  The 60.17 MB/layer table in §3 models projection
  weights; it is not an attention-core traffic invariant.
- At ctx 512 the old vec grid dispatches 44,032 workgroups/token, but only about
  10,496 carry a 32-key chunk over the 512 generated tokens.  Roughly 76% of
  the grid is empty, yet those groups still initialize state, read Q, and write
  an identity partial.  Attention is a measured optimization target.
- `GPU busy` is not necessarily useful compute: fast TP synchronization spins
  on the GPU (`ds4_metal.m:10084-10149`, `metal/dsv4_misc.metal:6636-6664`).
  Also, ablation deltas already include the dispatches removed with a stage;
  adding a separate `1021 * 1.9 us` row double-counts launch time.  The table in
  §4 is useful attribution evidence, but is not an additive decomposition and
  does not establish a 1.76 ms residual.
- The isolated MoE harness is world 1 with `add_in == NULL`
  (`tests/bench_moe_mxfp4_decode.c:153-195`).  Live TP is world 2 and folds a
  shared addend (`ds4.c:25087-25109`), selecting different Metal
  specializations (`ds4_metal.m:38355-38422`).  Its ~400 GB/s result does not
  prove the live TP kernel ceiling.  The uniform-router `E[max]` assumption is
  also unverified against real selected-expert traces.

### Exact compact decode FlashAttention

The raw/gathered vec kernel processes 32 keys per simdgroup.  The implementation
now keeps the legacy NWG-32 `NSG`, computes

```
chunks = ceil(n_keys / 32)
needed = ceil(chunks / baseline_NSG)
```

and, for TP world 2 only, chooses the smallest compiled bucket in
`{4, 5, 12, 16, 24, 32}` that is at least `needed`.  The 8 bucket remains
available through `DS4_METAL_DECODE_NWG=8` for experiments; the default goes
straight to 12 after 5 so a ctx-512 run does not compile an NWG-8 PSO for only
its first three generated tokens and then compile NWG 12 mid-measurement.
World 1 remains at 32, preserving the 64-head packed specialization.

| layer class | layers | effective keys over gen 512 | 32-key chunks | default NWG |
|---|---:|---:|---:|---:|
| uncompressed | 2 | 128 | 4 | 4 |
| ratio 128 | 20 | 132..136 | 5 | 5 |
| ratio 4 | 21 | 256..384 | 8..12 | 12 |

This is not an approximate reassociation.  Since `bucket >= needed`, no compact
workgroup wraps around its split stride.  Every real chunk retains the same
initial `iwg * NSG + sgitg` assignment as NWG 32.  The reduce kernel maps lanes
outside the compact bucket to the old empty identity
`S=0, M=-FLT_MAX/2, partial=0`, preserving the same 32-lane max/sum tree.  The
inverse-RoPE fused reducer shares this corrected body.  NWG 1 is intentionally
rejected because the vec kernel's NWG-1 direct-output mode does not emit the
stats consumed by these reducers.

At the benchmark shape the default reduces vec workgroups from 44,032 to
11,520/token and partial-buffer read+write payload from 172.672 to about
45.2 MiB/token.  It also reduces redundant Q loads and shared-state setup.
Metal function-constant specialization can still change compiler codegen, so
algorithmic exactness is not accepted without the bitwise harness below.

### Validation and target-rig A/B

The model-free harness uses the production TP shape (32 heads x 512), gathered
F16 compressed KV, and the fused inverse-RoPE reducer.  It checks raw/gathered
outputs against NWG 32 at key-count and bucket boundaries.

```sh
make metal-flash-attn-decode-bench
./speed-bench/metal_flash_attn_decode_bench --correctness-only
./speed-bench/metal_flash_attn_decode_bench --warmup 2 --samples 8 --tokens 32
```

The harness is a geometry microbenchmark: its 43 calls reuse one synthetic KV
allocation and omit intervening model work and TP gates.  Use it to reject bad
split choices, not to forecast the full 2.68 ms stage.

For authoritative two-node throughput, run the §10 worker and coordinator
commands in interleaved A/B/B/A order on both machines:

```sh
# A: legacy control
DS4_METAL_FAST_SYNC=1 DS4_METAL_DECODE_NWG=32 ...

# B: exact compact default
env -u DS4_METAL_DECODE_NWG DS4_METAL_FAST_SYNC=1 ...
```

Compare median `gen_steady_tps` after PSOs are warm and verify greedy output
identity.  Repeat diagnostic arms with `DS4_TP_ABLATE=attncore` on **both**
ranks to isolate the attention delta.  Keep dispatch/GPU profiles off for the
headline run, then enable them separately for attribution.

### Reprioritized work

1. **Attention:** after the compact A/B, build a dense short-context sibling of
   the existing indexed heads-8 split kernel.  Direct raw-ring/F16-compressed
   reads and cross-head KV reuse can remove gathered staging and avoid the
   generic kernel's per-head KV loads.  Use enough splits (roughly 12-16) to
   keep 60 cores occupied; the old split sweep provides no evidence here.
2. **MoE:** collect representative and held-out route data with the existing
   expert profiler, then calculate the empirical per-layer rank critical path.
   Extend the live-shape harness to world 2 with its non-null shared addend.
   Compare co-occurrence-aware placement, graduated replication, and routed
   MXFP4 intra-expert slicing only after that evidence exists.  Shared Q8
   slicing does not by itself derisk the routed MXFP4 geometry.
3. **TP gates:** 16 KB in 38 us is latency/software dominated, not a measured
   fabric floor.  `ds4_tp.c:908-970` signals every send and polls the shared CQ.
   Measure zero/1KB/16KB messages, then test periodic signaled completions,
   receive-window/mutex cost, and service-thread affinity.  Five microseconds
   saved across 86 gates is 0.43 ms/token.
4. **Prefill:** establish a two-node baseline before optimizing.  Sweep prompt
   length and prefill chunks 128..4096, record both-rank big-gate time and GPU
   stages, and interleave rollback arms for the six imported prefill fusions.
   Decode matvec conclusions do not transfer to prefill GEMMs.

**Result: 37.00 -> 41.21 t/s at 0 context (+11.4%).** The token is now fully
decomposed and every stage except one runs at its hardware rate. This document
is written so a new session can pick up without re-deriving anything.

---

## 1. Current state

| baseline | t/s | ms/token | how measured |
|---|---|---|---|
| 0 context | **41.21** | 24.266 | `ds4` REPL, `--temp 0`, empty prompt |
| ctx 512 | **41.56** | 24.062 | `ds4-bench`, the stage breakdown below |

Both are with `DS4_METAL_FAST_SYNC=1`. The ctx-512 figure is the one to use for
stage work: it is reproducible to ~1% and every ablation below is against it.

**The only stage carrying recoverable waste is the routed MoE, and the waste is
load imbalance, not kernel inefficiency.** Worth ~1.47 ms = +2.7 t/s. Everything
else is at or near its measured kernel ceiling.

---

## 2. Hardware and what to expect from it

- **M2 Ultra, 60-core GPU variant, Apple8 family.** 800 GB/s is a fixed chip
  spec and is **identical on the 60- and 76-core parts** — memory bandwidth does
  not scale with GPU core count, and per-core bandwidth is *higher* on the
  60-core. Core count matters here through occupancy, not bandwidth.
- Apple8 lacks: native bf16, Apple9 dynamic caching, any GPU matrix unit
  (`simdgroup_matrix` lowers to FP32 ALUs), Metal 4 TensorOps. `grep -rl
  "bfloat\|bf16" metal/` returns nothing.
- **Achieved rates measured on this rig** (see §5): Q8_0 matvec 517-581 GB/s
  (65-73% of spec), MXFP4 MoE ~400 GB/s (50%). Those are the real ceilings, not
  800.
- GPU sits pinned at its top P-state (P8, 1398 MHz) for the whole token; DVFS is
  not a factor (`8183c7e`).
- Decode draws ~98 W system power vs ~108 W prefill. Not diagnostic — that gap
  is normal for matvec vs GEMM.
- **`powermetrics` has no DRAM bandwidth sampler on Apple silicon.** Derive
  bandwidth as bytes/token divided by GPU-busy/token instead.

**Dev machine caveat.** The development box in this project is an **M1 Max,
64 GB**. It is fine for building and for `ds4_test --metal-kernels`, and useless
for timing — repeated runs of the same bench there returned 0.2%-of-peak
garbage rows because the machine is busy. Never quote a performance number from
it.

---

## 3. Model architecture (verified against the layout validator)

`DS4_SHAPE_FLASH`, `ds4.c:582`:

```
n_layer 43          n_embd 4096        n_vocab 129280
n_head 64           n_head_kv 1        n_head_dim 512     n_value_dim 512
n_rot 64            n_out_group 8      n_lora_q 1024      n_lora_o 1024
n_expert 256        n_expert_used 6    n_expert_shared 1  n_ff_exp 2048
n_kv_lora 512       n_key_mla 256      n_value_mla 256
n_indexer_head 64   n_indexer_head_dim 128   n_indexer_top_k 512
n_hc 4              n_hc_sinkhorn_iter 20    n_hash_layer 3   n_swa 128
```

**`n_leading_dense` is 0 for Flash.** The `.n_leading_dense = 3` at `ds4.c:686`
belongs to `DS4_SHAPE_GLM52`. All 43 layers are routed. This has been
mis-assumed at least twice.

Derived dims that matter and are easy to get wrong:

- `q_dim = n_head * n_head_dim = 32768` (`ds4.c:5070`)
- `out_low_dim = n_out_group * n_lora_o = 8192` (`ds4.c:5071`)
- Compressor ratio per layer (`ds4.c:1120-1124`): `il < 2 -> 0`, even -> 4,
  odd -> 128. So **2 never-compressed + 21 ratio-4 + 20 ratio-128**. A layer
  becomes "gathered" once its compressor has emitted; ratio-128 layers stay raw
  until pos 128.

### Verified byte model

Q8_0 = 34 B / 32 values = 1.0625 B/param. MXFP4 = 17 B / 32 = 0.53125 B/param.

| tensor | shape | MB/layer | per rank under TP |
|---|---|---|---|
| `attn_q_a` | [4096, 1024] | 4.46 | 4.46 (replicated) |
| `attn_q_b` | [1024, 32768] | 35.65 | **17.83** (head half) |
| `attn_kv` | [4096, 512] | 2.23 | 2.23 (replicated) |
| `attn_output_a` | [4096, 8192] | 35.65 | **17.83** (4 of 8 groups) |
| `attn_output_b` | [8192, 4096] | 35.65 | **17.83** (k-slice 4096 of 8192) |
| **attention total** | | | **60.17 MB/layer = 2.587 GB/token** |

That 60.17 reproduces `tests/bench_q8_attn_shapes.c`'s independently derived
60.2 MB, so the model is trustworthy. **Any byte figure not reconciling against
this number is wrong** — that mistake was made three times in this
investigation and each time it produced a phantom optimisation target.

Routed experts: **13.369 MB per expert per layer** (gate+up+down, MXFP4).
256 experts x 43 layers = **137 GiB**. Per-rank shard 68.5 GiB routed +
~8.2 GiB replicated = **76.73 GiB** (matches the startup log). Model file is
145.26 GiB, so it does **not** fit one node — TP is mandatory and no
single-node baseline is measurable.

### The attention output decomposition

`attn_output_a` is **block-diagonal**: 8 groups, group *g* maps input slice *g*
(8 heads x 512 = 4096 dims) to low slice *g* (1024 dims). Each rank holds 32
heads = 4 input groups and produces exactly those 4 groups' low dims.
`attn_output_b` then k-slices the 8192-wide low by the same 4/8 split, so **each
rank reduces over k=4096** and the two partials sum at the ATTN gate.

Both stages therefore land on k=4096, the *fast* bench shape. There is no bad
geometry hiding here.

---

## 4. Measured per-stage budget

At ctx 512, gen 512, control **41.56 t/s = 24.062 ms/token**. Every row marked
*ablated* is a real measurement: `DS4_TP_ABLATE=<chain>` on both ranks, delta
against control. Every delta was ~100% GPU-busy time, never bubble.

| stage | ms | % | source |
|---|---|---|---|
| routed MoE | **6.17** | 25.7% | ablated (`moe`, 55.98 t/s) |
| TP gate sync | **3.30** | 13.7% | 86 gates x 38 us measured (`5adc371`) |
| attn output (`out_a`+`out_b`) | **2.88** | 12.0% | ablated (`attnout`, 47.21) |
| attn core (vec+reduce) | **2.68** | 11.1% | ablated (`attncore`, 46.77) |
| `q_b` projection | **1.56** | 6.5% | ablated (`qb`, 44.44) |
| shared expert | ~1.15 | 4.8% | ablated (`shared`, 42.10 — confounded, see §9) |
| HC mix | **0.85** | 3.5% | ablated (`hcpre`, 43.14) |
| bubble (non-GPU) | **0.66** | 2.7% | wall minus GPU busy, flat in every arm |
| `q_a` + `kv` | ~0.59 | 2.5% | derived: 0.287 GB @ ~490 GB/s |
| output head | ~0.52 | 2.2% | derived: 0.281 GB @ ~540 GB/s |
| dispatch overhead | ~1.94 | 8.1% | derived: 1021 x 1.9 us |
| residual | ~1.76 | 7.3% | compressor, router, norms, misc |

GPU busy is **97.3%** of wall. The bubble is a flat 0.66 ms in every arm and
does not move.

### Achieved vs isolated kernel rate — this is the key table

| stage | GB/token | ms | achieved | isolated bench | ratio |
|---|---|---|---|---|---|
| attn output | 1.533 | 2.880 | **532 GB/s** | 517-581 | **~100%** |
| `q_b` | 0.767 | 1.559 | **492 GB/s** | ~413-541 | **~100%** |
| routed MoE | 2.264* | 6.170 | **367 GB/s** | ~400 | **92%** |

\* MoE critical path is **E[max(k, 6-k)] = 3.9375 experts**, not 3.0, because
each of the 43 FFN gates waits for the slower rank. 3.9375/3 x 1.725 = 2.264 GB.

**Conclusion: the kernels are fine.** The single remaining inefficiency is the
expert straggler.

---

## 5. Isolated kernel rates (measured on the rig)

`tests/bench_q8_attn_shapes` and `tests/bench_moe_mxfp4_decode`. Build (needs
only `ds4_metal.o`, no GGUF — the Q8 harness creates its own scratch file):

```sh
cc -O3 -ffast-math -mcpu=native -I. -c tests/bench_q8_attn_shapes.c -o tests/bench_q8_attn_shapes.o
cc -O3 -o tests/bench_q8_attn_shapes tests/bench_q8_attn_shapes.o ds4_metal.o \
   -lm -pthread -framework Foundation -framework Metal
```

| shape | GB/s | % of 800 |
|---|---|---|
| k=2048 -> 8192 | 581 | 73% |
| k=4096 -> 4096 | 541 | 68% |
| k=8192 -> 2048 | 517 | 65% |
| k=1024 -> 16384 | 413 | 52% |
| k=512 -> 32768 | 283 | 35% |
| MXFP4 routed MoE | ~400 | 50% |

**Use a large N.** The stock harness ran N=28 dispatches per command buffer,
which inflated per-iteration time ~18% with submit overhead and understated the
rate as ~440. `DS4_BENCH_MAP_GB=16` sets N to the region count (963) and gives
541-581. The MoE bench is the control proving this is an N effect and not a
working-set effect: it swept 1.6 -> 13.7 GB with N fixed at 128 and stayed flat
at 397-402 GB/s.

Run each config **twice and use the second** — the first faults in the mmap.
Watch the engine's own `warmup` line to confirm.

`DS4_BENCH_CHAIN=1` runs the square shape as a dependency chain instead of N
independent matvecs. Result: **452.2 vs 450.3 GB/s, identical.** Expected, since
`ds4_gpu_compute_encoder` builds the batch encoder with
`[cb computeCommandEncoder]` — `MTLDispatchTypeSerial` — so those dispatches
were already serialised. Dependency stalls are ruled out.

---

## 6. Dispatch accounting

- **1021 dispatches per token** at ctx 512 (340.25/cb x 3 cbs/token).
- **3 command buffers per token** at pos >= 128, **2** below — the second split
  only engages at `pos >= 128` (`ds4.c:27033`). This is why short 0-context runs
  barely exercise it, and why `dispatches/cb` is not comparable across runs of
  different length.
- **Marginal cost of one dispatch: ~1.9 us**, from the cleanest measurement we
  have (the `kv` arm adds exactly 43 dispatches for 0.081 ms). Corroborating:
  ballast gives 3.74 us at the first increment (independent dispatches, a
  floor), `ba132ba`'s mask arm gave 4.4 us. **Use 1.9-4.4 us, not 8.6.**
- `make check-dispatch-count` asserts every dispatch site is wrapped in
  `DS4_DISP`. The invariant has rotted twice; 16 sites were unwrapped as of
  `90605b1`, under-reporting by ~186/token.
- `DS4_METAL_DISPATCH_BALLAST=N` emits N no-op one-thread dispatches per layer
  for in-situ calibration. Sweep N in {0,1,2,4} and fit d(ms)/d(43N).

**Dispatch removal is not a productive strategy here.** 1021 dispatches x 1.9 us
= 1.94 ms total, and a realistic fusion campaign was scoped at 185 dispatches =
0.35 ms = +0.6 t/s. See §8.

---

## 7. The one remaining target: MoE expert straggler

Routed experts are sharded **contiguously 128/128**. With 6 experts selected
uniformly, the per-layer critical path is `E[max(k, 6-k)] = 252/64 = 3.9375`
experts against an ideal 3.0 — and because each layer gates independently, the
imbalance does **not** average out.

**Worth 6.170 x (1 - 3/3.9375) = 1.47 ms -> ~44.3 t/s (+2.7).**

Three designs were scoped:

| | balance | extra memory | mapping | verdict |
|---|---|---|---|---|
| **A. Partial expert replication** | E[max] ~3.16 | **+27 GiB** | two contiguous ranges | least code |
| **B. Intra-expert split, in place** | 3.0 | none | **~22k spans — blocked** | infeasible |
| **C. Intra-expert split, repacked GGUF** | **3.0** | none | one contiguous span | cleanest end state |

**A.** Overlap the shards (rank 0 owns 0-177, rank 1 owns 78-255, 100 shared).
Both ranks compute the identical router output, so they can deterministically
assign each selected expert to whichever owner is lighter. Costs ~27 GiB, taking
each node to ~103 GiB against a ~114 GiB wired limit, which drops usable context
from ~1M to roughly 650k.

**B.** Split the 2048 intermediate. `ffn_gate_exps` is (4096, 2048, 256) with
dim[0] contiguous, so each expert's half is contiguous *within that expert* —
but the halves interleave across 256 experts, giving 256 spans per tensor per
layer, ~22k total, against the 219 buffers used today. `ffn_down_exps` is
(2048, 4096, 256) so its split axis is the inner dim, i.e. strided 544-byte
reads, not expressible as spans at all. Blocked.

**C.** Repack the GGUF once so each rank's slice is contiguous. Gives perfect
3.0 balance, one span per tensor, no extra memory, and incidentally makes the
`down` read contiguous. Costs a conversion tool and a TP-layout-specific model
file.

**Efficiency of intra-expert splitting is already derisked**: the shared expert
*already* uses exactly this decomposition (`shared_tp_local = shared_dim / 2`,
`ds4.c:23922`) and measures at its predicted rate. The obstacle is purely
mapping, not throughput.

### Scaling to 4 nodes — do not expect much

| N | E[max] | ideal | imbalance |
|---|---|---|---|
| 2 | 3.938 | 3.0 | 1.31x |
| 4 | 2.818 | 1.5 | **1.88x** |

MoE critical path improves only 1.40x, not 2x, and relative imbalance gets
*worse*. Meanwhile half the attention is replicated (`q_a`, `kv` don't split)
and gate cost roughly doubles for a 2-hop all-reduce. Estimated 4-node total
~22.9 ms = ~43.7 t/s: **+5% for double the hardware.** Intra-expert splitting
would improve that to ~48 t/s, which is why C matters more at 4 nodes than 2.

On all-reduce libraries: the gate payload is 16 KB exchanged 86 times per token
at 38 us — **latency-bound, not bandwidth-bound**. QuickReduce's central trick
is inline-quantized all-reduce, a bandwidth optimisation, and buys nothing.
MLX's `jaccl` is more relevant as it is Apple/Thunderbolt-native. But no library
beats the fabric: **the lever is fewer gates, not faster ones.**

---

## 8. Negative results — do not retry these

| tried | result |
|---|---|
| `DS4_METAL_MODEL_UNTRACKED=1` | 41.71 vs 41.61, noise. Hazard tracking serialises *across queues*; one serial queue has nothing to serialise |
| `DS4_METAL_DECODE_SPLITS` 12/16/20/24/30 | 41.56 / 41.62 / 41.58 / 41.71 / 41.68 — **flat**. 48 -> 120 threadgroups changed nothing, so `attncore` is not split-K occupancy bound |
| Dependency-chain bench mode | 452.2 vs 450.3 GB/s, identical |
| `32ef898` three TP gate-exclusion removals | +0.019 ms, nothing |
| Host-side caching generally | `0c0ee5e` removed 0.87 ms of host time; none of it converted to throughput |
| Concurrent decode encoder | Only one `memoryBarrierWithResources` exists in 43k lines; dependent dispatches cannot overlap. Argued down, not tested |
| Fusing router with shared gate/up | Fused kernel hardcodes `NSG = 4` (`metal/dense.metal:555`) while TP forces q8 `nsg = 2` (`ds4_metal.m:5098`). Not exact |
| packed32 flash reduce at 32 heads | Correct but **1.35 t/s slower** — tuned for the 64-head grid, halving it underfills 60 cores. Reverted (`fe4674f`) |
| Fused router projection+select under TP | **Broke output** (repetition loops). Reverted (`2b04539`). See §9 |
| Logits over TCP | recv ~= 145 us fixed + bytes/1.5 GB/s. 318 us total, 1.3% of token. Both RDMA and top-k routes rejected |
| 20 Sinkhorn iterations | Register-only loop inside the producer (`metal/dsv4_hc.metal:550`), zero dispatches. Dead hypothesis |
| `attnout` / `out_b` restructuring | **Phantom target.** Believed to be 2x its rate; it runs at ~100%. The error was a 3x-low byte estimate and a wrong k (512 vs the actual 4096) |

---

## 9. Instruments and their caveats

### Ablation chains

`DS4_TP_ABLATE=chain[,chain]`, **same value on both ranks**. Output is
semantically wrong; only the timing is meaningful.

- In `ds4.c` via `metal_graph_tp_ablate()`: `hcpre`, `router`, `kv`, `qb`,
  `attnout`, `shared`
- In `ds4_metal.m` via `ds4_gpu_ablate_chain()` (for stages with too many or too
  conditional call sites): `moe`, `attncore`
- **`compidx` has no call site** and never had one. It reports 0 ms because it
  is unimplemented, not free.

Three caveats, all learned the hard way:

1. **Anything upstream of expert selection inflates its own cost.** A stale
   top-6 unbalances the contiguous shard. The `router` arm removed 92 dispatches
   and still ran **0.574 ms slower** — about 0.77 ms of added straggler. That
   arm is unusable for attribution. Arms whose ablation leaves the input still
   *varying* (`qb`, `attnout`, `attncore`) keep selection effectively random and
   are much cleaner.
2. **Some "ablations" are fusion rollbacks.** `kv` and `shared` do not skip
   work — they disable a fusion so an unfused path runs, **adding** dispatches
   (+43 each). Check the dispatch delta before interpreting.
3. **`hcpre` removes work, not dispatches.** Its dispatch count is bit-identical
   to control; only GPU busy differs.

### Counters

```sh
DS4_METAL_DISPATCH_PROFILE=1   # dispatches N over M cbs
DS4_METAL_GPU_BUSY_PROFILE=1   # gpu busy accum, every 64 cbs
DS4_TP_GATE_PROFILE=1          # per-gate wait/exchange; cumulative, never resets
DS4_TP_LOGITS_PROFILE=1        # per-token logits exchange
DS4_TP_LOGITS_PROBE_DIV=N      # shrink the logits payload to separate wire from skew
DS4_METAL_DISPATCH_BALLAST=N   # N no-op dispatches/layer for marginal-cost calibration
```

Read steady state from **cbs 512 -> 1536**, past prefill. Compute
`ms/cb x 3` for per-token GPU busy.

### Local tests (the success predictor)

Changes with a byte-exact test that runs **locally without the model** held 3
for 3. Changes validated only by reading held 0 for 2. Since the model does not
fit one node, `speed-bench/metal_decode_schedule_bench` (single-node) **cannot
run** — the only on-rig check is a greedy-output diff at `--temp 0`.

```sh
make check-dispatch-count
./ds4_test --metal-kernels     # no model needed, ~10 min
make test-mxfp4-metal
```

---

## 10. Benchmark procedure

```sh
# worker (machine B) — needs no ctx flags or prompt
DS4_METAL_FAST_SYNC=1 \
./ds4-bench -m "$MODEL" --tensor-parallel --role worker \
  --coordinator <ip> 1234 --transport rdma --rdma-device rdma_en6

# coordinator (machine A)
DS4_METAL_FAST_SYNC=1 DS4_METAL_DISPATCH_PROFILE=1 DS4_METAL_GPU_BUSY_PROFILE=1 \
./ds4-bench -m "$MODEL" \
  --tensor-parallel --role coordinator --listen 0.0.0.0 1234 \
  --transport rdma --rdma-device rdma_en7 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 512 --ctx-max 512 --gen-tokens 512
```

Read **`gen_steady_tps`** (8th CSV column) — it drops the first token.

**Traps, all hit at least once:**

- **`--warm-weights` hangs the machine.** It touches every page of the whole
  145.26 GiB GGUF (`ds4.c:3141`) with no awareness of the TP shard, thrashing a
  128 GB node. The TP startup already pre-faults the rank's own ~76 GiB.
- **`--ctx-max` is the sweep endpoint and must be <= the prompt length.**
  `promessi_sposi.txt` is 419,509 tokens. If it appears twice on the command
  line the last wins.
- **`--ctx-alloc` decouples allocation from the sweep**, so you can keep a large
  KV allocation while measuring at 512.
- The `ds4` REPL **cannot cap generation** (`-n` does not bound it), so ablation
  arms terminate at whatever length garbage output reaches. Use `ds4-bench`.
- ctx 512 keeps every token above pos 128 (fixed 3 cbs/token) and below ~2048
  where the indexer engages (`ds4.c:22966`).
- Noise floor is **~1%**. Interleave arms; ~10% machine drift has been seen
  between distant windows.

### RDMA nuances

- `--listen 0.0.0.0` works for the TCP handshake, but the RDMA GID must resolve
  to the address on the cabled Thunderbolt interface. Pass `--rdma-device`
  explicitly and confirm `transport=rdma` in the log — a silent TCP fallback
  would change gate cost and invalidate any comparison.
- Confirm `ds4: TP fast release fence enabled` appears. Without
  `DS4_METAL_FAST_SYNC` the gates cost ~180 us each instead of 38, worth ~15 ms.
- **AppleThunderboltRDMA accepts RDMA WRITE work requests and never executes
  them** (`ds4_tp.c:118-127`); only UC QPs exist. One-sided transfer is
  permanently off the table.
- Gate payload is 16 KB in exactly one RDMA message. Logits ride the **plain TCP
  control socket**, not RDMA (`ds4_tp_send_logits_half`).
- `tp_rdma_big_gate_exchange` (`ds4_tp.c:1078`) refuses when
  `recv_window_active`, which the decode gate lookahead sets on the first gate
  and clears only in the drain — so it is unavailable for the whole of decode.

---

## 11. Landed changes

| commit | change | measured |
|---|---|---|
| `01a56db` | admit pre-M5 to the decode pair/affine RoPE path | **-1.16 ms** |
| `65b4cf2` | MXFP4 routed down matvec, 4 rows per thread | **-0.53 ms** |
| `32ef898` | three TP gate-exclusion removals | none |
| `ba132ba` | pre-M5 gathered KV stage + persistent zero mask | **-1.13 ms** |
| `bb576ed` | pre-M5 compressor pair state store (fixed 23 test failures) | test-only |
| `36d2758` | pre-M5 output HC weights4 | ~0.02 ms |
| `a7ad597` | six prefill fusion gates cherry-picked from upstream #770 | prefill only |
| `8b0e380` | wrap all 229 dispatch sites + `check-dispatch-count` | instrumentation |
| `15492bf` | dispatch ballast calibration | instrumentation |
| `62aa3ff`, `51921a2` | ablation chains for attention stages, shared, MoE | instrumentation |
| `648f300` | tunable `DS4_METAL_DECODE_SPLITS` | inert by default |
| `6808243` | let a TP worker start `ds4-bench` without coordinator args | fix |

Reverted: `fe4674f` (packed32 at 32 heads, -1.35 t/s), `2b04539` (fused router
project+select, broke output).

### The recurring defect

**A device-*name* string test where the surrounding code used a device-*family*
predicate.** `ds4_gpu_device_name_contains("M3")` versus
`ds4_gpu_device_is_pre_m5_apple_silicon()`. An M3 Ultra satisfies both, so the
original port validated green on the author's hardware and was silently dead on
every other pre-M5 part. Eight decode sites fixed here; upstream PR #770 found
the same defect independently and reports **the same 29 failing assertions**.
Its extra seven sites are all prefill, six of which are now cherry-picked.

Worth re-running `grep -n 'device_name_contains' ds4_metal.m` periodically.

---

## 12. Upstream context

- **#770** (kk1987) — the same family-gate fix, 15 sites. Six cherry-picked in
  `a7ad597`; the eight overlapping ours left in our form; `27563` (shared kvpad)
  deliberately not taken — it removes no dispatch and carries a layout hazard.
- **#743** (ryan5rdx) — our own polled release fence, still open.
- **#799** (sethconvex) — per-stream command-queue overlap, ~1.7x for
  **multi-session batching**. Explicitly excludes TP.
- **#590** (gilbert-barajas) — replay-free DSpark partial accepts, +40% greedy.
  Relevant if DSpark is revisited.
- **#628** (devteapot) — model-provider refactor moving sources under
  `models/<model>/`. Would be a major merge event for this fork.

**Merge-shape lesson:** git conflicts are not the safety signal. Replaying the
last merge without the pre-emptive revert gives *zero* conflicts in
`metal/moe.metal` — the `_r4` kernel twins merged textually clean and would have
left half of every down projection silently unwritten. Host selection blocks
conflict loudly; kernel insertions merge silently. Put fork semantics where git
will fight you.

---

## 13. Deferred: DSpark

Enabling DSpark drops 37 -> ~24 t/s. Four verified TP-specific causes:

1. Verify-batch gates have **no fence path** — `ds4_metal.m:10116` waits
   unconditionally, and the service thread's `req.rows > 0` branch shadows the
   fence branch. 4.1-7.7 ms per verify block.
2. The verify batch **forfeits the TP attention split** (`ds4.c:28181` requires
   `tp_batch_rows != n_tokens`; verify sets them equal), re-reading 4.886 GB
   instead of 2.587.
3. The verify output head is **not vocab-split** — full 562.6 MB.
4. The routed MoE runs **row by row** with four malloc/free per row per layer.

Fixing 1-3 removes ~11-15 ms per verify block, but post-fix lands at ~30-36 t/s
— probably still worse than DSpark off. It is opt-in and off in the baseline, so
it contributes nothing to the decode goal.

---

## 14. Where to start a new investigation

1. **Re-establish the baseline** with §10. Expect 41.5-41.7 at ctx 512.
2. **The only measured target left is the MoE straggler** (§7), 1.47 ms /
   +2.7 t/s. Pick Option A or C.
3. If looking for something new, the unmeasured remainder is **~1.76 ms of
   residual** (compressor, router, norms) plus **~1.94 ms of dispatch
   overhead** — both small and both hard.
4. **`attncore` at 2.68 ms is the one stage not explained by bandwidth.** It
   moves ~45 MB at ctx 512, i.e. ~17 GB/s, and is not split-K occupancy bound
   (§8). Nobody has worked out what it *is* bound by. That is the most
   interesting open question in the file.
5. Before quoting any byte figure, reconcile it against the 60.17 MB/layer in
   §3. Three separate wrong conclusions in this investigation came from skipping
   that check.
