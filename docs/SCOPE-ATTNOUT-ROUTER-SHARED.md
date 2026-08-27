# Scoping: attn_output / router / shared-expert decode stages

Branch `upstream-metal-wins`. **Analysis only — no source changed.**

Roofs used throughout:
- streaming memory **760 GB/s measured** (`BENCHMARKS-TP-PP.md:195-207`, both hosts) — not 800, not 400
- FP32 ALU **~21.5 TFLOP/s** (60 cores x 128 lanes x 2 flop/FMA x 1.398 GHz).
  Equivalent **issue** roof: 60 x 128 x 1.398e9 = **10.74 T lane-instructions/s** — the more useful
  number for a dequant matvec, where most instructions are not FMAs.
- TP2: each rank holds 32 of 64 heads, 4 of 8 output groups, 128 of 256 routed experts,
  half the shared-expert intermediate.

Targets as briefed (U12 stage profile, `BENCHMARKS-TP-PP.md:246-266`):

| stage | ms @131k | % of 2k token |
|---|---|---|
| `attn_output` | 3.804 | 15.4% |
| `router` | 1.108 | 4.6% |
| `shared_gate_up` + `shared_down` | 1.622 | 6.7% |

---

## 0. Method and provenance

**Where the target numbers come from.** `BENCHMARKS-TP-PP.md:246-266` (U12, 2026-08-27,
build `b99dfa3`, mat worker / lanfear coordinator, RDMA, gen 128, 32k and 131k).
Instrument is `DS4_METAL_GPU_STAGE_TIMESTAMPS=1`: every `DS4_METAL_PROFILE_DECODE_STAGE`
marker commits the command batch and tags it (`ds4.c:28953-28955` ->
`ds4_metal.m:10822-10837`); the per-stage number is `GPUEndTime - GPUStartTime`
summed over that stage's command buffers (`ds4_metal.m:10847-10861`).

**Three properties of that instrument that bear on every number below:**

1. **A stage's number is command-buffer GPU-busy, not kernel time.** Anything encoded into
   the same buffer before the marker is inside the stage — including TP fence spins
   (see 1.1) — and per-command-buffer start/end overhead is inside it too.
2. **The instrumented run is ~11% slower than the real token.** U12 reports
   `total gpu_busy` 37.99 ms at 131k against a 34.31 ms uninstrumented token. Stage
   numbers are therefore *upper* bounds with a systematic inflation of roughly that size,
   unevenly distributed (small stages pay proportionally more cb overhead).
3. **The U12 table is a top-12, not a decomposition.** It sums to 29.88 of 32.70 ms at 32k;
   ~2.8 ms sits in unreported markers (`docs/TP-A0-ROWSPLIT-TEST-PLAN.md:1697-1703`).

**Epoch discipline.** Three distinct epochs appear in this document and are never subtracted
from one another:

| epoch | control | source |
|---|---|---|
| **E1** ctx-512 ablation | 41.56 t/s / 24.062 ms | `tp_decode_investigation.md` §4 |
| **E2** M2 ablation battery, 2026-08-26 | 28.29 t/s @131k / 35.35 ms | `BENCHMARKS-TP-PP.md:585-594` |
| **E3** U12 stage profile, 2026-08-27 | 29.15 t/s @131k / 34.31 ms; 37.99 ms instrumented busy | `BENCHMARKS-TP-PP.md:246-266` |

Where two epochs agree on a derived rate that is treated as corroboration, never as a
subtractable difference.

**Byte-model reconciliation.** Every byte figure below is re-derived from the encoder call
site and then checked against the verified 60.17 MB/layer attention model in
`tp_decode_investigation.md` §3 before any rate is quoted, per §14.6.

---

## 1. What is actually in each span

Stage spans run marker to marker. All three markers live in
`metal_graph_encode_decode_layer_phase` in `ds4.c`.

### 1.1 `attn_output` — span `ds4.c:23582` (`attn_inv_rope`) -> `ds4.c:23870`

Under TP2 decode (`tp_attn_decode_split` true, `fuse_attn_out_hc` false because it requires
`g->tp_world < 2`, `ds4.c:23596-23602`) the span contains **four dispatches per layer**, not two:

| # | kernel | grid (threadgroups) | threads/TG | tg memory | site |
|---|---|---|---|---|---|
| 1 | `kernel_dsv4_attn_out_low_q8_0_f32` (`out_a`) | **(512, 1, 4) = 2048** | (32, 4) = **128** | 256 B | `ds4_metal.m:26398-26411`, grid at `31331` |
| 2 | `kernel_mul_mv_q8_0_f32` (`out_b`, k-sliced) | **(2048, 1, 1) = 2048** | (32, 2) = **64** | 256 B | `ds4_metal.m:26091-26094` |
| 3 | `kernel_dsv4_tp_flag_set_coherent` | 1 | 1 | — | `ds4_metal.m:10484-10496` |
| 4 | `kernel_dsv4_tp_fence_wait` | 1 | 1 | — | `ds4_metal.m:10420-10432` |

Dispatches 3 and 4 are the **ATTN TP gate**, encoded at `ds4.c:23856`
(`ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_ATTN)`) — *before* the marker, into the same
command batch (`ds4_gpu_tp_gate_encode` requires and reuses `g_batch_cb`,
`ds4_metal.m:10450`, `10480`, `10506`). `kernel_dsv4_tp_fence_wait` **spins on the GPU**
until the host service thread publishes the peer's release word. That spin is
`GPUEndTime - GPUStartTime` time and is therefore **inside the reported 3.804 ms.**

This is the single most important structural fact about this stage, and it was not
visible from the stage name. Corroboration from the sibling stage: the FFN gate
(`ds4.c:25335`) lands the same way inside `ffn_hc_post`, and `ffn_hc_post` measures
1.812 ms against `attn_hc_post`'s 0.461 ms for comparable HC work — a ~1.35 ms
difference in the same run and the same direction. (This is a consistency check across
two stages of *one* run, not a cross-epoch subtraction.)

Notes:
- `ds4.c:23802`'s comment "**2560 threadgroups**" is **stale**. The pair dispatches
  2048 + 2048 = **4096** threadgroups per layer. `N_R0_Q8_0` is 2 (`ds4_metal.m:4595`),
  so `out_b`'s grid is `out_dim / 2 = 2048`, not `out_dim / 8 = 512`.
- `DS4_TP_ABLATE=attnout` (`ds4.c:23803`) removes **only dispatches 1 and 2**. It does not
  remove the gate. So an `attnout` ablation delta prices the projection pair, while the
  stage marker prices projection pair + gate. They are measuring different things, on top
  of being different epochs.
- The island-1 decode-graph capture at `ds4.c:23613-23648` is disabled under TP
  (`island_b_ok` requires `fuse_attn_out_hc`, which requires `tp_world < 2`). Not in play.
- `ds4_gpu_tp_gate_encode` calls `ds4_gpu_close_batch_encoder()` twice per gate
  (`ds4_metal.m:10498`, `10500`), so the span also pays two extra compute-encoder
  open/close cycles per layer.

### 1.2 `router` — span `ds4.c:24023` (`ffn_norm`) -> `ds4.c:24137`

`metal_graph_decode_cpu_router_applicable` is false for this model — it needs PRO+Q4_K or
SSD-streaming IQ2_XXS (`ds4.c:20585-20611`) — so the GPU path runs. The fused
router+shared-gate/up path and the fused router-project+select path are both
**disabled under TP** by `g->tp_world < 2` at `ds4.c:24042`. What remains is
**two dispatches per layer**:

| # | kernel | grid | threads/TG | tg memory | site |
|---|---|---|---|---|---|
| 1 | `kernel_mul_mv_f16_f32_4` (router logits, F16 4096x256) | **128** | (32, 8) = **256** | 256 B | `ds4_metal.m:20818-20843`; nsg/nr0 from `ds4_metal.m:5299-5321` |
| 2 | `kernel_dsv4_router_transform_finalize_weights_one_simd` | **1** | 256 | 4096 B | `ds4_metal.m:33508-33513`; kernel `metal/dsv4_misc.metal:4923-5024` |

`metal_graph_decode_set_hash_selected_override` (`ds4.c:24129`) returns immediately unless
the layer has `ffn_gate_tid2eid` (`ds4.c:21115`); at most the 3 hash layers can add work.

Dispatch 2 is a **full 256-element bitonic sort in one threadgroup on one of 60 cores** to
extract the top 6 — 36 compare-exchange stages, 6 of them across simdgroups with a
threadgroup barrier, followed by a `tid == 0` serial loop of six dependent
`volatile device` loads (`metal/dsv4_misc.metal:5009-5018`).

### 1.3 `shared_gate_up` — span `ds4.c:25030` (`routed_moe`) -> `ds4.c:25143`

Under TP2, `tp_split_shared` is taken (`ds4.c:25060-25092`). `routed_moe` at `ds4.c:25030`
is an **empty marker** on this path — the routed MoE actually runs at `ds4.c:25296` under
`tp_fold_ffn` and reports as `routed_moe_folded`, which is why U12's table has no
`routed_moe` row. So the span holds **one dispatch**:

| kernel | grid | threads/TG | tg memory | site |
|---|---|---|---|---|
| `kernel_dsv4_shared_gate_up_swiglu_q8_0` | **512** | (32, 2) = **64** | 512 B | `ds4_metal.m:20202-20234` |

`out_dim = shared_tp_local = shared_dim / 2 = 1024`, `in_dim = 4096`, `nr0 = 2` -> 512
threadgroups. `nsg = 2` because `ds4_gpu_make_q8_0_mv_dispatch` returns 2 under TP2
(`ds4_metal.m:5274-5284`).

### 1.4 `shared_down` — span `ds4.c:25143` -> `ds4.c:25291`

One dispatch:

| kernel | grid | threads/TG | tg memory | site |
|---|---|---|---|---|
| `kernel_mul_mv_q8_0_f32` (k-sliced) | **2048** | (32, 2) = **64** | 256 B | `ds4_metal.m:26072-26094` |

`full_in_dim = 2048`, `k_off = tp_rank * 1024`, `k_cnt = 1024`, `out_dim = 4096`
(`ds4.c:25273-25281`). With `k_cnt = 1024` the block count is `nb = 32` and the loop
stride is `NSG * NQ = 16`, so **each thread runs exactly 2 loop iterations**.

---

## 2. `attn_output`

### 2.1 Bytes and FLOPs per token per rank

Shape at the decode call site (`ds4.c:23804-23815`, values from `DS4_SHAPE_FLASH`):
`group_dim = n_head_dim * (n_head / n_out_group) = 512 * 8 = 4096`; `rank = n_lora_o = 1024`;
`n_groups_total = 8`; `group_cnt = 4` (this rank's half); `out_dim = n_embd = 4096`.

Q8_0 row bytes = `(k / 32) * 34`.

| tensor | rows | k read | bytes/layer |
|---|---:|---:|---:|
| `attn_output_a`, 4 owned groups | 4 x 1024 | 4096 | 4 x 1024 x 4352 = **17,825,792** |
| `attn_output_b`, k-window 4096 of 8192 | 4096 | 4096 | 4096 x 4352 = **17,825,792** |
| activations (heads in 64 KiB, low w+r 32 KiB, out w 16 KiB) | | | 114,688 |
| **total** | | | **35,766,272 B = 35.77 MB** |

**Reconciliation (mandatory check):** §3's verified model gives `attn_output_a` 17.83 MB
and `attn_output_b` 17.83 MB per rank per layer = 35.65 MB of weights. This derivation
gives 35.65 MB of weights + 0.11 MB of activations. **Agrees.** x43 layers =
**1.538 GB/token** (§4 quotes 1.533 GB, weights only — the 0.3% gap is the activations).

FLOPs: `2 x (4 x 1024 x 4096)` MAC/layer = 33.55 M MAC = **67.1 MFLOP/layer**,
**2.886 GFLOP/token**.

Instruction count (the roof that actually matters for a Q8_0 matvec): the inner loop
(`metal/dense.metal:161-178`) costs, per MAC, one int8->float convert plus one FMA, with
1/8 of a `char8` weight load, 1/16 of a `half` scale load and 1/16 of an activation
`float` load amortised over `NR0 = 2` rows — call it **~2.3 lane-instructions per MAC**.
33.55 M x 2.3 x 43 = **3.3 G lane-instructions/token**.

### 2.2 Achieved rate against both roofs

| basis | ms | GB/s | % of 760 | GFLOP/s | % of 21.5 T | issue % of 10.74 T |
|---|---:|---:|---:|---:|---:|---:|
| **E3** stage span (proj + gate) | 3.804 | **404** | **53%** | 759 | 3.5% | 8.1% |
| **E2** `attnout` ablation delta @131k (proj only) | 2.93* | **525** | **69%** | 985 | 4.6% | 10.6% |
| **E1** `attnout` ablation delta @512 (proj only) | 2.880 | 534 | 70% | 1002 | 4.7% | 10.8% |

\* E2: control 28.29 t/s (35.35 ms), `attnout` arm +9.0% -> 30.84 t/s (32.42 ms);
delta 2.93 ms (`BENCHMARKS-TP-PP.md:587,590`). Independent epoch, independent instrument.

**Classification: memory-bound, at ~69-70% of the streaming roof, with ~0.9 ms of the
3.804 ms stage span being something other than the projection pair.**

Two conclusions, and they are different questions:

**(a) The projection pair is at 69-70% of the roof, robustly.** E1 and E2 are separate
epochs, separate instruments (ctx-512 vs 131k, both ablation deltas) and land at 534 and
525 GB/s. That is not noise and it is not the ~100% that §4 claimed — §4 compared against
the *isolated bench* (517-581 GB/s for the same kernel family), which only proves the
engine costs nothing over standalone. Against the hardware it is 70%. The remaining 30%
is ~0.9 ms/token, **3.7% of the 2k token** if fully recovered.

**(b) ~0.9 ms of the stage span is not the projection pair at all.** 3.804 (E3 span) vs
2.93 (E2 projection-only) is a cross-epoch comparison and must not be quoted as a
difference — but the span provably contains the ATTN gate fence spin, which is the only
other GPU work in it, and the `ffn_hc_post` / `attn_hc_post` contrast inside E3 puts a
same-run gate-sized quantity in the same range. **`attn_output` is being over-credited as
a kernel target by roughly a quarter of its headline number.**

### 2.3 Why the kernel is at 70% — and what the shape data already rules out

Not underfill. 4096 threadgroups per layer on 60 cores (2048 x 128 threads + 2048 x 64
threads). This is the opposite end of the constraint from `packed32` (32 heads), the fused
head-norm/RoPE grid (32 TGs) and the LLT scorer (1 resident/core). **The four-sighting
underfill lead does not apply to `attn_output`; do not open it here.**

Not ALU. 8-11% of the span at the issue roof, 3.5-4.7% at the FLOP roof.

Not the engine. §4's isolated-vs-achieved comparison, whatever else it got wrong, does
establish that the in-engine kernel matches the same kernel standalone.

The informative evidence is the isolated shape sweep (`tp_decode_investigation.md` §5),
all at a constant ~17.8 MB of weights:

| shape (k -> N) | threadgroups | bytes/TG | GB/s | % of 760 |
|---|---:|---:|---:|---:|
| 512 -> 32768 | 16384 | 1,088 | 283 | 37% |
| 1024 -> 16384 | 8192 | 2,176 | 413 | 54% |
| 2048 -> 8192 | 4096 | 4,352 | **581** | **76%** |
| **4096 -> 4096** (this stage) | **2048** | **8,704** | **541** | **71%** |
| 8192 -> 2048 | 1024 | 17,408 | 517 | 68% |

The rate is a **hump in k, peaking at k=2048**, and the same total bytes move in every
row. That is the signature of a **latency / memory-level-parallelism limit, not a
bandwidth limit** — consistent with the standing "DS4F decode is latency-bound" finding.
Below k=2048 the per-thread loop is too short to amortise the prologue and the
`helper_mv_reduce_and_write` tail (at k=1024, `nb=32` and `NSG*NQ=32`, so each thread runs
**one** iteration); above it, the threadgroup count falls and with it the number of
independent outstanding loads the memory system sees. At k=4096 the kernel sits just past
the hump on the low-parallelism side.

Per-thread memory-level parallelism at this stage's shape (`metal/dense.metal:161-178`):
each iteration issues 8 activation floats (reused across both rows), then per row one
`char8` of weights and one `half` scale — about **4-6 independent loads in flight per
thread**, with `nb = 128` blocks and stride `NSG*NQ`, giving **4 iterations** for `out_a`
(NSG=4) and **8** for `out_b` (NSG=2). The loop is not unrolled and carries an
accumulator, so cross-iteration prefetch depends entirely on the compiler.

One structural cost worth naming precisely: **Q8_0's block is 34 bytes.** A thread's
weight load starts at byte `ib*34 + 2 + il*8`, whose alignment mod 8 cycles
`2,4,6,0` — only one block in four is 8-byte aligned, so three quarters of the `char8`
weight loads straddle. The simdgroup as a whole does cover its 8 blocks (272 B)
essentially contiguously, so this is a per-transaction cost rather than a wasted-bytes
cost, but it is the mechanical reason a Q8_0 matvec tops out around 70% where a plain
streaming kernel reaches 95%.

### 2.4 What has already been tried, and must not be re-proposed

- **`DS4_METAL_Q8_MV_NSG`** was swept (T4, `BENCHMARKS-TP-PP.md:707-711`): the TP default
  of 2 is optimal and higher is monotonically worse (nsg=4 is **-3.0%** at 131k). The
  global nsg lever is closed.
- **`attnout` / `out_b` restructuring** on a 3x-low byte estimate with k=512 was the §8
  phantom. The byte model above is reconciled against §3 and does not repeat it.
- `DS4_METAL_MODEL_UNTRACKED`, dependency-chain execution, concurrent encoders: all
  §8 negatives, and U6 additionally shows allocation path costs nothing (all seven
  arms within 1% at ~760 GB/s).

**One thing T4 did *not* cover, contrary to the plan text.** `docs/TP-A0-ROWSPLIT-TEST-PLAN.md:648`
asserts `DS4_METAL_Q8_MV_NSG` "carries ... attention-output low". It does not.
`out_a` calls `ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4)` with a
**literal 4** (`ds4_metal.m:26399`, `26410`), and `ds4_gpu_get_mul_mv_pipeline` takes nsg
as an argument with no env override (`ds4_metal.m:2960-2993`). Only `out_b`, `q_a`, `kv`,
`q_b` and the shared expert follow the env. **`out_a`'s NSG has never been swept**, and it
sits at the value T4 found to be 3% worse everywhere else.

---

## 3. `router`

### 3.1 Bytes and FLOPs per token per rank

`ffn_gate_inp` is **F16 [n_embd 4096, n_expert 256]** — asserted at the call site
(`ds4.c:24046-24048`) — and is **replicated on both ranks** (the two ranks must select the
same experts, so there is no TP split here).

| item | bytes/layer |
|---|---:|
| `ffn_gate_inp` F16 4096 x 256 | 2,097,152 |
| `ffn_exp_probs_b` bias, 256 f32 | 1,024 |
| activations: `ffn_norm` in 16 KiB, logits/probs 8 KiB, selected+weights 48 B | ~17,900 |
| **total** | **~2,116,000 = 2.12 MB** |

x43 layers = **91.0 MB/token**. FLOPs: 4096 x 256 MAC = **2.10 MFLOP/layer**,
**90.2 MFLOP/token**.

**Reconciliation:** this is not attention traffic, so §3's 60.17 MB/layer does not apply.
The independent check is the shard accounting: 43 x 2.097 MB = 90.2 MB of router weights
sits inside §3's "~8.2 GiB replicated" per-rank figure, and the F16 4096x256 shape is
asserted in the code, not assumed.

### 3.2 Achieved rate against both roofs

| | value | % of roof |
|---|---:|---:|
| 91.0 MB / 1.108 ms | **82 GB/s** | **10.8% of 760 GB/s** |
| 90.2 MFLOP / 1.108 ms | **81 GFLOP/s** | **0.38% of 21.5 TFLOP/s** |
| per layer | **25.8 us** for 2 dispatches and 2.12 MB | |

**Classification: neither roof. The router is latency-, dispatch- and instrument-bound.**
At the 760 GB/s roof its weights would take **0.12 ms/token**. It costs 1.108 ms. There is
no bandwidth headroom to reclaim here because bandwidth is not what it is spending.

### 3.3 Where the 25.8 us/layer plausibly goes — and why this cannot be ablated

§9 is explicit that `DS4_TP_ABLATE=router` is **unusable**: the arm ran 0.574 ms *slower*
while removing 92 dispatches, because a stale top-6 unbalances the contiguous expert
shard. So the following is an **inference from code structure, clearly marked as such**,
not a measurement:

| component | estimate/layer | basis |
|---|---:|---|
| router F16 matvec | ~5-8 us | 2.1 MB; 128 TGs on 60 cores is 2.1 waves; each thread covers 4096/(32x8) = 16 elements per row — a very short kernel dominated by launch and memory latency |
| bitonic top-6 select | ~2-3 us | 1 threadgroup on 1 of 60 cores; 36 stages, 6 threadgroup barriers, then 6 dependent `volatile device` loads on `tid == 0` (`metal/dsv4_misc.metal:5009-5018`) |
| 2 dispatches | ~4-9 us | §6's marginal 1.9-4.4 us/dispatch |
| stage-profile command buffer | ~4-5 us | see below |
| **sum** | **~15-25 us** | against 25.8 us measured |

**The instrument tax is material for a stage this small.** E3 reports 37.99 ms of stage
GPU-busy against a 34.31 ms uninstrumented token (and, at §4's 97.3% busy fraction, against
~33.4 ms of real GPU busy) — a **~4.6 ms overshoot** spread across the run's command
buffers. The router stage owns one command buffer per layer, so on the order of
**0.18-0.22 ms of its 1.108 ms is the profiler**, i.e. ~16-20%. The same absolute tax
applies to `shared_gate_up` (~19-23%) and `shared_down` (~27-33%), and is negligible for
`attn_output` (~5%).

**This is recoverable exactly, at zero cost.** `ds4_gpu_stage_report` already prints
`buffers=%zu` per stage (`ds4_metal.m:10870-10873`). U12's table dropped that column.
Recording it, plus one `DS4_METAL_DISPATCH_BALLAST`-style calibration of per-cb overhead,
turns the tax from an estimate into a subtraction *within one run*.

### 3.4 Two fusions already exist for this stage, and both are TP-blocked

- **`kernel_dsv4_router_shared_gate_up_q8_0`** (`ds4_metal.m:20254-20351`) folds the
  router matvec into the shared gate/up SwiGLU — one dispatch on the same `ffn_norm`
  input. Blocked by `g->tp_world < 2` (`ds4.c:24042`). The surrounding control flow
  already supports the resulting order (`router_shared_done` makes `ds4.c:25113` skip the
  separate shared gate/up), so what is missing is the TP lane offset
  (`gate_offset + tp_lane_off`, `out_dim = tp_half`) and an exactness decision.
  §8's objection stands and must be answered: the fused kernel dispatches `(32, 8, 1)`
  (`ds4_metal.m:20342`) while TP runs the standalone shared gate/up at
  `nsg = 2`, so the reduction order differs and the result is **not bit-identical to
  today's TP output** — though it is identical *between ranks*, which is what TP
  determinism requires.
- **`kernel_dsv4_router_project_select_fused`** (`ds4_metal.m:20353`,
  `metal/dsv4_misc.metal:5026`) folds matvec+select into one dispatch using a grid-wide
  `atomic_uint completion` software barrier. **Already tried under TP and it broke output
  with repetition loops — reverted in `2b04539` (§8).** It is additionally gated on
  `ds4_gpu_device_is_m5_apple_silicon()` (`ds4.c:24058`), so it is dead code on M2 Ultra.
  Do not retry without first explaining the failure; a grid-wide spin barrier is not safe
  when the grid can exceed resident capacity.

---

## 4. Shared expert (`shared_gate_up` + `shared_down`)

### 4.1 Bytes and FLOPs per token per rank

`ffn_gate_shexp` / `ffn_up_shexp` are **Q8_0 [4096, 2048]** and `ffn_down_shexp` is
**Q8_0 [2048, 4096]** — asserted at `ds4.c:5227-5229`; `shared_dim = ffn_gate_shexp->dim[1]
= DS4_N_FF_EXP = 2048` (`ds4.c:22117`).

| stage | what this rank reads | bytes/layer |
|---|---|---:|
| `shared_gate_up` | 1024 gate rows + 1024 up rows, each k=4096: `2 x 1024 x 4352` | 8,912,896 |
| | + x in 16 KiB, gate/up/mid out 12 KiB | 28,672 |
| `shared_down` | 4096 rows, k-window 1024 of 2048: `4096 x 1088` | 4,456,448 |
| | + mid in 4 KiB, out 16 KiB | 20,480 |
| **shared expert weights total** | | **13,369,344 = 13.37 MB** |

x43 layers: `shared_gate_up` **384.5 MB/token**, `shared_down` **192.5 MB/token**,
combined **577.0 MB/token**.

**Reconciliation — and it is exact.** §3 states routed experts cost **13.369 MB per expert
per layer** (MXFP4, full 2048 intermediate). This rank's *half* of the Q8_0 shared expert
is **13,369,344 B — the same number to the byte**, because
`3 x 4096 x 2048 x 0.53125 = 3 x 4096 x 1024 x 1.0625`. Two independently derived figures
landing on the same integer is a strong check on both the shape and the quant sizes.

FLOPs: `shared_gate_up` 8.39 M MAC = 16.8 MFLOP/layer (**721 MFLOP/token**);
`shared_down` 4.19 M MAC = 8.39 MFLOP/layer (**361 MFLOP/token**).

### 4.2 Achieved rate against both roofs

| stage | ms | GB/s | % of 760 | GFLOP/s | % of 21.5 T |
|---|---:|---:|---:|---:|---:|
| `shared_gate_up` | 0.965 | **398** | **52%** | 748 | 3.5% |
| `shared_down` | 0.657 | **293** | **39%** | 549 | 2.6% |
| **combined** | 1.622 | **356** | **47%** | 667 | 3.1% |

**Classification: memory-shaped work running at roughly half the streaming roof — the same
band as the MoE matvec's 54% (U6), and not "at its predicted rate".**

§7's claim that the shared expert "measures at its predicted rate" was made against the
pre-U6 assumption that ~400 GB/s was the ceiling. `shared_gate_up`'s 398 GB/s is exactly
that number, which is why it looked closed. **Against the measured 760 GB/s roof it is
52%, and `shared_down` is 39%.** §7 used this to argue that intra-expert splitting is
"already derisked ... the obstacle is purely mapping, not throughput". That conclusion
should be re-read: the split is not free, it is merely no worse than the number §7
compared it against. See 4.3.

### 4.3 `shared_down`: the TP k-split moved the kernel to the worst point on its own curve

This is the clearest mechanism found in this scoping exercise.

§5's isolated Q8_0 matvec sweep, at a constant ~17.8 MB, is a **hump in k**:

| k | GB/s | % of 760 |
|---:|---:|---:|
| 512 | 283 | 37% |
| 1024 | 413 | 54% |
| **2048** | **581** | **76%** |
| 4096 | 541 | 71% |
| 8192 | 517 | 68% |

`ffn_down_shexp` is naturally a **k=2048** matvec — the peak of that curve.
`shared_tp_local = shared_dim / 2` (`ds4.c:24150` in this tree — the brief's `23922` is a
stale line number; applied at `ds4.c:25277-25278`)
k-slices it to **k=1024**, one step down the falling side. The mechanism is visible in the
kernel: with `k_cnt = 1024`, `nb = 32` and the loop stride is `NSG * NQ = 16`
(`metal/dense.metal:161`), so **every thread executes exactly 2 loop iterations** — barely
enough to amortise the prologue's 64-bit address arithmetic and the
`helper_mv_reduce_and_write` tail (2 `simd_sum`, a threadgroup barrier, 2 more
`simd_sum`). In situ it measures **293 GB/s**, below even the sweep's k=1024 point.

**The fix is a different split axis, not a different kernel.** Split `ffn_down_shexp` by
**output rows** instead of by k: rank *r* computes rows `[r*2048, (r+1)*2048)` over the
full k=2048.

- **Bytes are identical**: `2048 rows x 2176 B = 4,456,448 B` — the same figure as today.
- **k per row goes 1024 -> 2048** (the peak of the curve) and per-thread iterations 2 -> 4.
- **Grid goes 2048 -> 1024 threadgroups.** This is the one risk: it is a halving, and the
  underfill constraint has four sightings. 1024 TGs x 64 threads is still 17 TGs/core, so
  it is not the packed32 regime, but it moves in that direction and the sweep's k=8192 row
  (1024 TGs) sits at 517 GB/s, not 581 — so expect somewhere in 450-520, not 581.
- **Correctness composes with the existing gate.** The FFN gate already *sums* the two rank
  partials (`ds4.c:25332`, `25337-25338`), and disjoint row ownership sums correctly
  provided the unowned half of `shared_out` is zero. Since rank *r* always owns the same
  rows, that is a **one-time zero at buffer allocation**, not a per-layer memset — so the
  saving is not eaten by a new dispatch.
- **Not bit-identical to today** (single full-k dot vs. two half-k partials summed), though
  identical between ranks.

**Sizing:** 0.657 ms at 293 GB/s -> ~0.40-0.46 ms at 420-480 GB/s. **Saves 0.20-0.26 ms =
0.8-1.1% of the 2k token (+0.35-0.45 t/s).** Real, cheap, and small.

### 4.4 `shared_gate_up`: a 512-threadgroup grid, halved by the same split

`out_dim = tp_half = 1024` and `nr0 = 2` gives **512 threadgroups of 64 threads** — 32,768
threads, ~546 per core. World-1 would dispatch 1024. This is the packed32 pattern
("halving a 64-head grid underfills 60 cores") applied to the shared expert, and it is the
**fifth sighting** of the underfill constraint.

But the same evidence says smaller threadgroups are not automatically the answer, and the
obvious knobs are closed: `nsg` is swept and pinned (T4: 2 optimal, 4 is -3.0%), and
`N_R0_Q8_0` is a compile-time 2 shared by the fused HC kernels
(`ds4_metal.m:4595`, and `ds4_metal.m:5286-5297` documents what breaks when it is widened
without widening the kernel).

The available lever is therefore **not to shrink the work per threadgroup but to give the
dispatch more to do**: fold the router matvec into it (3.4), which raises the grid from
512 to 640 threadgroups, removes one dispatch and one full-GPU barrier per layer, and lets
the router's 2.1 MB ride behind the shared expert's 8.9 MB stream instead of paying its own
launch and latency.

### 4.5 The FFN front-end as a whole

`router` + `shared_gate_up` + `shared_down` = **2.73 ms/token for 668 MB = 245 GB/s, 32% of
the roof**, spent as **four serialized dispatches per layer** (128, 1, 512 and 2048
threadgroups) separated by full-GPU barriers on a single serial encoder
(`ds4_gpu_compute_encoder` builds `MTLDispatchTypeSerial`, §5). Two of those four grids are
too small to fill the machine. **The structural lever for this 11.2% of the token is
reducing the number of serialized FFN front-end dispatches, not speeding any one kernel.**

## 5. Ranked candidates

TBD

## 6. Instrumentation that would settle the open questions

TBD
