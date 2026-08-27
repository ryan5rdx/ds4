# Inventory: DECODE per-token cost budget

Extracted 2026-08-27 for the consolidated reference. Assignment: **decode-budget**.
Sources read in full or near-full: `docs/TP-A0-ROWSPLIT-TEST-PLAN.md` (3453 l),
`BENCHMARKS-TP-PP.md` (1799 l), `speed-bench/tp_decode_investigation.md` (652 l),
`speed-bench/tp_mtp_hunt.md` (365 l), `docs/SCOPE-HC-STAGES.md`,
`docs/SCOPE-ATTNOUT-ROUTER-SHARED.md`, `docs/SCOPE-TP-GATE-OVERLAP.md`,
`docs/SCOPE-MOE-ATTN.md`, `docs/TP-PREFILL-LONG-CTX-INVESTIGATION.md`.

Citations are `file:line` or `file §n` as the sources give them.

---

## 1. Provenance key — rigs, builds, instruments, epochs

### 1.1 Machines

| id | what | when it may be quoted |
|---|---|---|
| **rig** | 2× Mac Studio M2 Ultra, **60-core** GPU, 128 GB each, TP2 over Thunderbolt RDMA. Hosts named **mat** and **lanfear** (coordinator varies by campaign; M0/U11/U12 = lanfear coord + mat worker). 800 GB/s spec/chip; **measured streaming roof ~760 GB/s** (U6). FP32 ALU peak ~21–21.5 TFLOP/s @1.398 GHz (P8, pinned all token, `8183c7e`). | all throughput/t/s and all in-engine stage numbers |
| **M1 Max dev box** | 32-core, 64 GB, single 400 GB/s die, **no model weights, no TP** | model-free harnesses and source analysis **only**. `tp_decode_investigation.md §2`: "repeated runs of the same bench there returned 0.2%-of-peak garbage rows… **Never quote a performance number from it**." Standalone→rig transfer factor **~0.63** (only calibrated once, on R8). |

Model: DeepSeek-V4-Flash MXFP4, 145.26 GiB file, 76.73 GiB shard/rank
(68.5 GiB routed + ~8.2 GiB replicated). Does **not** fit one node → **no
single-node decode baseline is measurable** (`tp_decode_investigation.md §3`).

### 1.2 Builds / commits that carry decode numbers

| build | date | what it is | health |
|---|---|---|---|
| `tp-frontends-phase1` era, `5adc371`, `32ef898`, `ba132ba`, `01a56db`, `65b4cf2` | pre-2026-08-19 | the ctx-512 decode investigation epoch (E1) | clean, but **ctx 512 only**, indexer inactive |
| `3746eae1` | 2026-08-25 | A0-off inertness baseline | clean |
| `f3668a1`/`627ccc1` | 2026-08-25/26 | R10 gate+decode profiling, "Current state" sweep | **decode numbers taken with `iogpu.wired_limit_mb = 0`** — see §2.3 |
| `a0cf853` | 2026-08-26 | R11 replicate-attn | wired=0 era, but R11's A/B was in-session |
| `5866100` tree build | 2026-08-26 | **M0 re-baseline, pinned shard (120000)** | clean; the baseline all later A/Bs compare to |
| `f45b535` | — | consolidated prefill-split defaults on; `DS4_TP_DECODE_REPLICATE_ATTN` deleted | — |
| `b99dfa3` | 2026-08-27 | U11 / U12 stage profile @32k+131k | clean; unprofiled control 34.59 t/s @32k = 28.91 ms |
| `26ef09f` | 2026-08-27 | adds `q_a_kv_proj`/`q_lora_norm`/`q_path` markers (from `863e8fa`) — U12b | clean |
| `c13e3bb` | 2026-08-27 | **U15, first-ever 2k stage profile** + HC arms | clean |
| `345de30` | 2026-08-27 | 5-arm battery (gate profiler, FA split, stage profile, q8 shapes, attn_out nsg) | clean; **predates C2**; the `attn_tp_gate` marker here is MISPLACED (§2.2) |
| **`6b962db`** | 2026-08-27 | iteration 2 (attn_out_proj boundary, sampled FA, corrected gate formula, n-gram) | **CONTAMINATED — C2 defect, §2.1** |
| `da63283` | 2026-08-27 | C2 revert | — |
| `a861150`, `df0037e`, `867cae3`, `8943015`, `28ecec4`, `177b50a`, `69a3b86` | 2026-08-27 | instruments (per-slot gate profiler + `attn_tp_gate` marker; `DS4_METAL_GPU_ENCODER_TIMESTAMPS`; `bench_qkv_norm`; decode-FA boundaries; decode stage report wired into main TP loop; `DS4_TP_GATE_FASTPATH`; TP worker model-view crash fix) | — |

### 1.3 Measurement epochs (never subtract across them)

| epoch | control token | source |
|---|---|---|
| **E1** ctx-512 ablation | 41.56 t/s = **24.062 ms**; 0-ctx 41.21 t/s = 24.266 ms | `tp_decode_investigation.md §1/§4` |
| **E2** M2 ablation battery, 2026-08-26 | 33.92 t/s @32k = 29.48 ms; 31.69 @65k; **28.29 t/s @131k = 35.35 ms** | `BENCHMARKS-TP-PP.md` M2 |
| **E3** U12 stage profile, 2026-08-27 (`b99dfa3`) | 29.15 t/s @131k = 34.31 ms real; **37.99 ms instrumented busy**; 32k 34.59 t/s = 28.91 ms real, **32.70 ms instrumented** | `BENCHMARKS-TP-PP.md` U12/U11 |
| **E4** U15 2k stage profile (`c13e3bb`) | **28.613 ms instrumented busy** vs **~24.34 ms real (41.1 t/s)** | `BENCHMARKS-TP-PP.md` U15 |
| **E5** 5-arm battery (`345de30`) | 28.657 ms busy @2k / 38.402 @131k | `BENCHMARKS-TP-PP.md` |
| **E6** iteration 2 (`6b962db`, contaminated) | 28.174 ms busy @2k / 37.712 @131k | `BENCHMARKS-TP-PP.md` |
| **E0** 2026-08-26 stage profile (pre-U12) | 30.43 ms busy @32k / 36.36 @131k; span 30.73/36.67; **gap 0.307/0.308 ms** | `BENCHMARKS-TP-PP.md` "Stage profile" |

### 1.4 Instruments and their known distortions

| instrument | env | what it measures | distortion |
|---|---|---|---|
| stage timestamps | `DS4_METAL_GPU_STAGE_TIMESTAMPS=1` (+`_LAYER`, `_DETAIL`) | per-stage GPU time; **each stage becomes its own command buffer** (~900/token vs the normal 3) | **+13% @32k, +18% @2k; ≈4.2 µs/boundary ≈ 0.18 ms per marker per token.** Does NOT change kernel selection under TP2 (both fusions it disables are already TP-disabled) |
| encoder timestamps | `DS4_METAL_GPU_ENCODER_TIMESTAMPS=1` (`df0037e`) | GPU counter at **encoder** boundaries inside one CB | **~1–3%** overhead. `atStageBoundary` supported, `atDispatchBoundary` **not**. Validated: 4× work → 3.82×, 2× → 1.92×, encoder sum within 4% of CB span. **Not yet used on the rig.** |
| ablation chains | `DS4_TP_ABLATE=chain[,chain]`, **same value both ranks** | in-situ cost of a chain (t/s delta) | output is semantically wrong; three caveats in §6.2 |
| gate profiler | `DS4_TP_GATE_PROFILE=1` (+ per-slot ATTN/FFN split, `a861150`) | per-gate wait/exchange, cumulative, never resets | `gpu-wait` is **encode-to-satisfied**, i.e. contains all inter-gate compute — not exclusive idle |
| dispatch/busy profiles | `DS4_METAL_DISPATCH_PROFILE=1`, `DS4_METAL_GPU_BUSY_PROFILE=1` | dispatches N over M cbs; busy accum every 64 cbs | read steady state cbs 512→1536; `ms/cb × 3` = per-token busy |
| ballast | `DS4_METAL_DISPATCH_BALLAST=N` | N no-op 1-thread dispatches/layer; fit d(ms)/d(43N) | measures **overlapping** dispatches only (3.74 µs), not barrier dispatches (22 µs) |
| FA stage boundaries | `DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1` (`8943015`) | `gather`/`packed`/`fa_core`/`reduce` | **BROKEN for decode** — uses a host wall clock around `end_commands()`/`begin_commands()`, so it times a **command-buffer round trip** (~0.25–0.4 ms), not kernel time. §2.4 |
| standalone harnesses | `tests/bench_q8_attn_shapes`, `bench_moe_mxfp4_decode`, `bench_indexer_score`, `bench_membw`, `bench_qkv_norm`, `speed-bench/metal_flash_attn_decode_bench` | isolated kernel rates | see §7; indexer harness **over-predicts engine cost by ~1.9×** |

Read `gen_steady_tps` (8th CSV column) — it drops the first token.
Noise floor **~1%**; ~10% machine drift between distant windows; interleave arms.

---

## 2. Contamination and correction rules — apply these to every figure

### 2.1 Build `6b962db` — the C2 defect (rule 3, case 1)
`out_low_nsg = 8` was live, so `kernel_dsv4_attn_out_low_q8_0_f32` was **compiled
for 8 simdgroups and dispatched with 4**: each simdgroup covered 1/8 of k instead
of 1/4, so the threadgroup read **half the k range** — faster and wrong.
A throughput sweep cannot see garbage output; it measured as a **1.7% "win"**.

**The contamination is narrow and quantified** (clean `345de30` vs broken `6b962db`, @2k):

| stage @2k | clean 345de30 | broken 6b962db | Δ |
|---|---|---|---|
| routed_moe | 4.911 | 4.927 | +0.016 |
| q_a_kv_proj | 2.152 | 2.141 | −0.011 |
| q_path | 1.874 | 1.876 | +0.002 |
| q_lora_norm | 1.673 | 1.675 | +0.002 |
| attn_inv_rope | 3.813 | 3.784 | −0.029 |
| **attention region** | 3.719 (mislabelled combined) | 2.383 + 0.994 = 3.377 | **+0.342 ms of C2 artifact** |

Consequences: (a) every unrelated stage from `6b962db` is usable (≤0.03 ms drift);
(b) `attn_out_proj`'s broken 2.383 must be read as **~2.73 ms clean**;
(c) **arm 3's straggler must use the clean number** — garbage attention feeds the
router, so expert selection under C2 is not the model's distribution: the ATTN/FFN
exchange delta grew **+11.6 µs (clean) → +15.4 µs (broken)**, i.e. degenerate
routing concentrating experts. Straggler = **0.50 ms**, not 0.66.

### 2.2 The `attn_tp_gate` marker mislabel (rule from the brief)
In `345de30` the new `attn_tp_gate` boundary was placed **after**
`ds4_gpu_tp_gate_encode` (`ds4.c:23856`), so the span still began at
`attn_inv_rope` and **swallowed the out_a/out_b projections**. It read
**3.719 ms @2k / 3.750 @131k** while `attn_output` fell to ~0.000.

> **The 3.72 ms is the projections, not the gate.** Fixed by adding an
> `attn_out_proj` boundary immediately *before* the encode. Measured split
> (`6b962db`, then corrected for C2): `attn_out_proj` **~2.73 ms clean**
> (2.383 broken), `attn_tp_gate` **0.994 ms @2k / 0.912 ms @131k**.
> The real ATTN gate is **~0.91–0.99 ms**.

Cross-check: `ffn_tp_gate` 1.643 + `attn_tp_gate` 0.994 = **2.64 ms** total gate
spin, against the independent differencing estimate ~2.7 ms
(`ffn_hc_post − attn_hc_post` = 1.812 − 0.461 = 1.351 ms = 43 × 31.4 µs). **Two
independent methods agree.** FFN gate costs **1.65×** the ATTN gate — the excess
is the routed-shard straggler.

### 2.3 `iogpu.wired_limit_mb = 0` era
The sysctl was `0` on both hosts for the whole R10/R11 era (runtime-only; lost on
reboot; lanfear was rebooted mid-campaign 2026-08-26). All 27 surviving R10/R11
coordinator logs carry the `wired_limit_mb is 0` warning → the 76.7 GiB shard
paged lazily.

- **Decode t/s survives**: moved ≤1.1% (2048: 40.91→41.15; 131k: 28.09→28.24–28.40).
- **Gate exchange does not**: row exchange @131k **49.7 → 24.8 µs (−50%)**;
  @2k **34.8 → 23.6 µs (−32%)**; row wait @131k 401.4 → 375.0; @2k 292.6 → 247.1;
  BIG exchange @131k 22,531 → 17,091 µs (−24%).
- Rule: `wired_limit_mb is 0` anywhere in a log → **discard the arm**.

### 2.4 Batch-context / round-trip contamination of the flash-attn split
Two attempts to decompose decode attention are unusable:
- **`345de30` arm 2**: batch-context profile, runs at **13 t/s @2k vs 41 unprofiled**;
  per-call `fa_core` 0.31–0.33 ms, `reduce` 0.40–0.49 ms; per-call sums ≈30 ms
  against a 3.8 ms stage marker.
- **`6b962db` arm 2** (sampled, `..._EVERY`=43): `fa_core` 0.251 ms, `reduce`
  0.274 ms per token over 123 calls/128 tokens; still **6× out**.
- **Root cause**: the boundary takes `ds4_gpu_now_ms()` (host wall clock) either
  side of `end_commands()`/`begin_commands()` → it measures a **command-buffer
  round trip** (independently ~423 µs wall / 110 µs GPU-busy per 1-dispatch cb,
  `bench_qkv_norm`). Fix requires `MTLCounterSampleBuffer` **inside one command
  buffer**. **A2 stays unsized; stop reporting A0 as built.**

### 2.5 Standing calibration factors
| factor | value | scope |
|---|---|---|
| stage-marker tax | **~0.18 ms/marker** (4.2 µs/boundary × ~43) → 13% @32k, 18% @2k | subtract from every stage-timestamp figure |
| standalone (M1 Max) → rig | **×0.63** | only calibrated on R8; treat M1 Max speedups as upper bounds |
| indexer harness → engine | **÷1.88–1.9** | `bench_indexer_score` over-predicts (7.20 ms projected vs 3.84 ms in-situ) |
| ablation vs profile (hcpre) | ablation covers only ~43% of the span | §6.2 |

---

## 3. Reconciled decode budget @ 2k (best current figures)

Token: **24.34 ms real (41.1 t/s)**, from `c13e3bb`/M0-class controls.
Instrumented busy 28.613 ms (E4). Net column = profiled − 0.18 ms marker tax.

| stage | profiled ms (E4, `c13e3bb`) | **net ms (best)** | % of 24.34 ms | rate / notes |
|---|---|---|---|---|
| `routed_moe_folded` | 4.901 | **4.72** | 19.4% | 1.7246 GB → **365 GB/s = 48% of 760** (51% of the stride-17-adjusted 713); 6.49 GFLOP → 1.38 TFLOP/s = 6.4% ALU |
| `attn_inv_rope` (**mostly FlashAttention**, see §4) | 3.806 | **3.63** | 14.9% | 20–100 MB DRAM → **6–28 GB/s (0.8–3.7%)**; 1.107 GFLOP → **305 GFLOP/s (1.4%)**; ~35× its own instruction-issue floor |
| `attn_output` (= `attn_out_proj` + ATTN gate) | 3.731 | **3.55** | 14.6% | span 404 GB/s = 53%; **net of gate 524–534 GB/s = 69–70% of 760** |
| ├ `attn_out_proj` (out_a + out_b) | ~2.73 clean (2.383 on C2 build) | ~2.55 | ~10.5% | context-invariant |
| └ `attn_tp_gate` | 0.994 | ~0.81 | ~3.3% | the real ATTN gate |
| `q_a_kv_proj` | 2.150 | **1.97** | 8.1% | 0.279 GB → **131 GB/s = 17% of roof** |
| `ffn_hc_post` (75% FFN TP gate) | 1.921 | **1.74** | 7.2% | whole span 4.3 GB/s = 0.6%; **1.351 ms of it is the FFN gate** (= 43 × 31.4 µs) → ~0.39 ms real HC-post work |
| ├ `ffn_tp_gate` (separate marker) | 1.643 | ~1.46 | ~6% | measured directly in E5/E6 |
| `q_path` (q_b + per-head RMS norm + RoPE tail) | 1.875 | **1.70** | 7.0% | ~0.744 GB → **~400 GB/s = 53%**; the norm/RoPE tail is only **~0.36 ms** |
| `q_lora_norm` | 1.684 | **1.50** | 6.2% | **~0.55 MB → 0.33 GB/s = 0.04% of roof**; 39 µs/layer; **2 threadgroups on 60 cores**, 43×/token — worst underfill in the engine |
| `attn_hc_pre` | 1.135 | **0.96** | 3.9% | 0.0388 GB unique → 33.6 GB/s = **4.4%**; 0.0445 GFLOP → 38.5 GFLOP/s = **0.18%**; grid **6 TG × 512 thr** |
| `ffn_hc_pre` | 1.109 | **0.93** | 3.8% | 34.6 GB/s = **4.6%**; same 6×512 grid |
| `router` | 1.101 | **0.92** | 3.8% | 91.0 MB → **82 GB/s = 10.8%**; 81 GFLOP/s = **0.38%**; 25.8 µs/layer for **2 dispatches** |
| `shared_gate_up` | 0.974 | **0.79** | 3.3% | 0.383 GB → **398 GB/s = 52%**; 512 TG |
| `shared_down` | 0.665 | **0.49** | 2.0% | 0.192 GB → **293 GB/s = 39%** — TP k-split moved it from k=2048 (peak) to k=1024 |
| `attn_hc_post` | 0.471 | **0.29** | 1.2% | 0.00705 GB → 15.3 GB/s = **2.0%**; grid **16 TG × 256 thr**; identical kernel to `ffn_hc_post`'s expand **with no gate** — this is the gate-pricing control |
| `compressor_indexer` | 0.201 | **0.02** | 0.1% | collapses at short context, as predicted |
| **reported sum** | **25.72** | — | — | |
| **unreported markers** (`attn_norm`, `kv_path`, `compressor_proj/update/quantize/commit`, `indexer_compressor_*`) | **~2.89** | — | ~9% at 32k | exists as markers, never reported. **Report every stage next time.** |
| **total gpu_busy (instrumented)** | **28.613** | 24.34 real | 100% | |

**Fixed vs growing**: 88.5% of the 2k token is context-invariant — only
`compressor_indexer` and `attn_inv_rope` move with context (plan §"88% of the
short-context token is context-invariant", subtotal 21.53 ms / 88.5%).

**Gate spin total @2k**: `ffn_tp_gate` 1.643 + `attn_tp_gate` 0.994 = **2.64 ms**
(range quoted elsewhere 1.75–2.70 ms = 7.2–11.1%).

---

## 4. Reconciled decode budget @ 32k / 65k / 131k

### 4.1 Stage profile, all epochs side by side (ms/token, instrumented)

| stage | E0 32k | E0 131k | E3 32k (`b99dfa3`) | E3 131k | E5 2k/131k (`345de30`) | E6 2k/131k (`6b962db`, C2) |
|---|---|---|---|---|---|---|
| `compressor_indexer` | 5.45 | **10.51** | 4.957 | 9.668 | 0.199 / 9.712 | 0.199 / 9.643 |
| `q_path` (whole, pre-split) | 5.48 | 5.47 | **5.474** | **5.472** | — (split) | — (split) |
| ├ `q_a_kv_proj` | — | — | 2.137 | 2.136 | 2.152 / 2.140 | 2.141 / 2.144 |
| ├ `q_lora_norm` | — | — | 1.680 | 1.680 | 1.673 / 1.678 | 1.675 / 1.676 |
| └ `q_path` (q_b + norm + RoPE tail) | — | — | 1.862 | 1.863 | 1.874 / 1.869 | 1.876 / 1.859 |
| `routed_moe_folded` | 4.67 | 5.40 | 4.990 | 4.950 | 4.911 / 4.963 | 4.927 / 4.784 |
| `attn_inv_rope` | 3.42 | 4.27 | 3.389 | 4.258 | 3.813 / 4.220 | 3.784 / 4.260 |
| `attn_output` | 3.73 | 3.80 | 3.746 | 3.804 | ~0.000 (absorbed) | ~0.000 (absorbed) |
| ├ `attn_out_proj` | — | — | — | — | — | 2.383 / 2.384 (**clean ≈2.73**) |
| ├ `attn_tp_gate` | — | — | — | — | 3.719 / 3.750 **(MISLABEL)** | 0.994 / 0.912 |
| `ffn_hc_post` | 2.19 | 1.48 | 1.834 | 1.812 | — | — |
| └ `ffn_tp_gate` | — | — | — | — | 1.581 / 1.494 | 1.643 / 1.651 |
| `attn_hc_pre` | 1.16 | 1.14 | 1.142 | 1.155 | — | — |
| `ffn_hc_pre` | 1.13 | 1.11 | 1.119 | 1.121 | — | — |
| `router` | 1.11 | 1.09 | 1.112 | 1.108 | — | — |
| `shared_gate_up` | 0.99 | 0.98 | 0.985 | 0.965 | — | — |
| `shared_down` | 0.66 | 0.66 | 0.660 | 0.657 | — | — |
| `attn_hc_post` | 0.46 | 0.46 | 0.467 | 0.461 | — | — |
| **total gpu_busy** | **30.43** | **36.36** | **32.70** | **37.99** | 28.657 / 38.402 | 28.174 / 37.712 |
| span | 30.73 | 36.67 | — | — | — | — |
| **gap (stall)** | **0.307** | **0.308** | — | — | — | — |

**Attribution is closed by construction**: the stage sum equals `total gpu_busy`
exactly (30.427 / 36.360), and the **gap is ~0.31 ms (~1%) at both contexts.
Decode is ~99% GPU-busy — there is no stall to find.** (But see §5: busy ≠
occupied.)

### 4.2 Best current 131k figures (E3, `b99dfa3`, minus 0.18 ms/marker)

Token: 34.31 ms real (29.15 t/s); 37.99 ms instrumented.

| stage | profiled | net | % of 34.31 |
|---|---|---|---|
| `compressor_indexer` | 9.668 | 9.49 | 27.7% |
| `q_a_kv_proj`+`q_lora_norm`+`q_path` | 5.472 (2.136/1.680/1.863) | 4.93 | 14.4% |
| `routed_moe_folded` | 4.950 | 4.77 | 13.9% |
| `attn_inv_rope` | 4.258 | 4.08 | 11.9% |
| `attn_output` (proj + ATTN gate ~0.91) | 3.804 | 3.62 | 10.6% |
| `ffn_hc_post` (incl. FFN gate ~1.35) | 1.812 | 1.63 | 4.8% |
| `attn_hc_pre` | 1.155 | 0.98 | 2.8% |
| `ffn_hc_pre` | 1.121 | 0.94 | 2.7% |
| `router` | 1.108 | 0.93 | 2.7% |
| `shared_gate_up` | 0.965 | 0.79 | 2.3% |
| `shared_down` | 0.657 | 0.48 | 1.4% |
| `attn_hc_post` | 0.461 | 0.28 | 0.8% |

### 4.3 65k — what exists
No stage profile has ever been run at 65k. **Only ablation percentages (M2) and
t/s sweeps exist** (see §6.1, §11). This is a gap.

### 4.4 Decode t/s vs context (headline arcs)

| ctx | `tp-multi-slot-batching` (old base) | `f3668a1` "current state" (wired=0) | **M0 pinned `5866100`** | `b99dfa3` (U11, TIGHT on) |
|---|---|---|---|---|
| 2048 | 40.87 | 40.91 | **41.09** (gate arm 41.15) | — |
| 4096 | 36.76 | 36.47 | 36.58 | — |
| 8192 | 36.07 | 35.98 | 36.04 | — |
| 16384 | 35.04 | 35.43 | 35.39 | — |
| 32768 | 32.98 | 33.73 | 33.71 | 34.59 |
| 65536 | 29.80 | 31.52 | 31.56 | 32.19 |
| 131072 | 25.33 | 28.13 | **28.34** (cold 28.40) | 29.15 |

E1 ctx-512 epoch: 41.56 t/s (ctx 512) / 41.21 (0 context). Later target-only
observation 41.98 t/s; n-gram-inert arm 42.00 @2k, 29.55 @131k.

---

## 5. The token against its own floor (the "why 41 t/s" analysis)

**Bytes per token per rank**, from stages whose byte models reconcile against
`tp_decode_investigation.md §3`: routed MoE **1.725 GB** + q_a+kv **0.279** +
q_b **0.744** + attn out_a/out_b **1.538** = **4.29 GB** (excludes
shared/router/HC/indexer).

| at | ms | t/s |
|---|---|---|
| 760 GB/s (measured streaming roof) | 5.6 | 177 |
| **450 GB/s (best real Q8_0 kernel on this rig)** | **9.5** | **105** |
| **actual** | **24.34** | **41.1** |

**Decode is 2.6× off a rate a production kernel has actually achieved.** The gap
is ~15 ms — not the 4.34 ms that separates 41 from 50 t/s.

**Mechanism: exposed memory latency on a serial dependency chain.**
1021 dispatches/token, `MTLDispatchTypeSerial` by default; the concurrent path
(`g_batch_encoder_concurrent`, `ds4_metal.m:1028`) is armed only in
`ds4_gpu_parallel_ffn_start` and `parallel_ffn_route_eligible` carries
`g->tp_world < 2` itself (`ds4.c:~22180`). **Under TP2 nothing in decode runs
concurrently.**

**Six independent sightings of underfill** (none of them looked for):

| site | grid |
|---|---|
| `q_lora_norm` | **2 threadgroups** on 60 cores, 43×/token |
| head-norm / RoPE tail | 32 threadgroups (`ds4_metal.m:22589`, `MTLSizeMake(n_head, n_tok, 1)`) |
| flash reduce | **32 threadgroups every layer** (`ds4_metal.m:29334`) |
| 22 of 43 layers | dispatch only 128–160 threadgroups (vec kernel) |
| indexer LLT | **1 threadgroup resident/core**, smem-capped (20,512 B) |
| `packed32` flash reduce at 32 heads | correct but **−1.35 t/s** — underfills |
| (7th) HC producers | 6 TG × 512; HC expand 16 TG × 256 |

**Dispatch cost is situational** — `bench_qkv_norm` (`867cae3`), batched with
rotated outputs: **~22 µs/dispatch, flat across a 64× work range**:

| q_n | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|
| µs/dispatch | 22.6 | 21.4 | 23.5 | 23.0 |

Reconciles with the ballast's 1.9–3.7 µs: **a dispatch costs ~2–4 µs if it
overlaps and ~22 µs if everything waits on it.** The engine's 34.9 µs/layer for
`q_lora_norm` = ~22 µs + ~13 µs surrounding serialisation.

**Why TP2 is only ~1.6×** (decomposition of the 24.34 ms token):

| | ms |
|---|---|
| work (scales with bytes) | 17.0 |
| latency (does not) | 4.9 |
| gate (TP-only, does not) | 2.7 |

Single node = 2×17.0 + 4.9 = **39.0 ms (25.7 t/s)** vs TP2's 24.34 → **1.60×**.
The 31% that does not scale is exactly the latency-bound part.

**Power**: decode ~30 W GPU / 90–98 W system; prefill ~55–60 W GPU / 108–120 W
system; idle ~0.1 W. Decode is 54–58% of prefill power at both 2k and 131k and
**flat through the whole window**. `powermetrics` has **no DRAM bandwidth
sampler** on Apple silicon — derive bandwidth as bytes/token ÷ GPU-busy/token.
**Correction:** the gate spin does *not* explain the 30 W (8–11% of the token at
zero power predicts ~53 W). The power gap is latency-bound matvec vs prefill GEMM.

**Fixed-work savings propagate additively:**

| fixed work cut | 2k t/s | 131k t/s |
|---|---|---|
| — | 41.1 | 29.1 |
| 2.0 ms | 44.8 | 31.0 |
| 3.0 ms | 46.9 | 31.9 |
| **4.34 ms** | **50.0** | **33.4** |
| 6.0 ms | 54.5 | 35.3 |

<!-- SECTION-BREAK-1 -->
