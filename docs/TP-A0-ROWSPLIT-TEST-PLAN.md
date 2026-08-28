# TP2 rig test plan — prefill and decode

Started as the A0 row-split plan; A0 shipped and is default-on, and this is
now the standing request channel for TP performance work on the pair.

Branch: `upstream-metal-wins`
Rig: 2× M2 Ultra 60-core 128GB, tensor-parallel over Thunderbolt RDMA
Model: DeepSeek V4 Flash MXFP4, 145.26 GiB (76.73 GiB shard per rank)

## Status (2026-08-25)

This doc is the **request** side of a loop: results go into
`BENCHMARKS-TP-PP.md`, asks go here.

| run | state | outcome |
|---|---|---|
| 1 — inertness / baseline | done | 221.50 t/s @131k prefill, 28.06 decode. Indexer stack +20.8%, LLT decode +10.8% vs old base. Startup hang did not reproduce at `3746eae`. |
| 2 — correctness | done | **Pass.** 128/128 tokens identical, 305/305 argmax identical, max Δlogit 0.049 vs a 0.0055 control-vs-control baseline (~2 f16 ULP). Criterion rewritten below. |
| 3 — A0 throughput | done | +7.2% @131k (221.50 → 237.44), peaking +11.1% at 8k. Real, correct, decode untouched — but **not** the predicted ~1.5×. |
| 3b — cold single point | done | 259.90 → 281.20 t/s, +8.2%. |
| R1 — indexer stage profile | done | score 317.7 : topk 10.0 : attention 41.3 ms @comp 32768. **score ≈ 8× attention**, scales linearly with comp. Model confirmed. |
| R2 — gate profile | done | BIG 1419 → 2132 gates; per-gate wait *fell* 318 → 190 ms, exchange 30.5 → 23.7 ms. Effective 2.2–2.8 GB/s. |
| R3 — utilisation | done | **~90% in both arms at both 8k and 131k.** Flat, not A0-induced, not context-dependent. |
| R4 — argsort canon | done + **fixed** | +5.6% @2048, +2.5% @4096 when disabled. Now gated on `n_tokens >= 32` (`6fa977c`). |
| R5 — indexer split | done | **Bit-identical** (0/129,280 logits differ). score 317.7 → 138.2 ms, topk 10.0 → 5.2, attention unchanged. Sweep 131k 237.44 → **283.64** (+19.5%), cold 281.20 → **322.64**. |
| R6 — size ratio-128 layers | done | Prize **311.2 ms/layer-chunk × 20 layers**, ~71% of the odd layer. Odd-layer attention is *larger* than a ratio-4 layer's (255.3 vs 182.8 ms) and grows with context. Two of this doc's premises were wrong — see below. |
| R7 — static-mixed split | done | **Bit-identical** (0/129,280). Sweep 131k 284.03 → **342.25** (+20.5%), cold 322.64 → **380.27** (+18.2%). Both inside the pre-recorded projection (~340 / ~380). |
| R8 — FlashAttention `nsg` | done, **default flipped** | **Bit-identical** (0/129,280). Sweep 131k 342.18 → **367.29** (+7.3%), cold 380.27 → **402.64** (+5.9%). Correct direction and shape, but **under** the projected +10–15% — see the transfer factor below. |
| R9 — `nqptg` ceiling | scoped, no runs | A 2× standalone kernel discounts to ~+6–8% end-to-end. Prototype only if the restructure shows ≥1.5–2×. |
| R10 — gate path + decode | done, no code change | Sub-gate re-test: **wash at every context, avenue closed**. Link ceiling: **4.4/4.1 GB/s per direction** (4 KiB WR cap — 64 MiB messages structurally impossible); gate path at ~65–75% of it. First decode profiling: decode is **stall-bound** (~30 W vs ~55 W prefill), NWG default already optimal (plateau 8–32), the 2048 decode floor is **per-gate fixed cost**, and the only lever left is the 86 gates/token — a design change. Full numbers in `BENCHMARKS-TP-PP.md`. |
| R11 — decode gate count | done — **clean negative, avenue closed** | Decode **−8.4 to −11.8% at all 7 contexts**, prefill unmoved. Cost is flat ~3.25 ms/token at both 2k and 131k. The gate wait was never idle: it is the window the peer's half of the attention runs in. Falsifies R10e's headline — see below. |
| R12 — decode data-gathering | **requested below** | Two zero-code sweeps: the TP-excluded command-buffer split schedule, and the never-run dispatch-ballast slope. Both target the same hypothesis. |
| R13 — audit findings | **partly verified below** | `DS4_METAL_FAST_SYNC` is default-off and gates the TP decode split (ops check, possibly live); MXFP4 fixed-route MoE decode is TP-disabled by a host clause; one more device-name gate missed by the earlier sweep. |

**Arc at ctx 131072 (sweep):** 183.4 (old base) → 221.50 (upstream indexer
stack) → 237.44 (A0) → 283.64 (indexer split) → 342.25 (static-mixed split) →
**367.29** (FA nsg). **+100%** over the old base, **+65.8%** over the flag-off
baseline on this branch. Cold single point 259.90 → **402.64** (+54.9%).
Decode untouched throughout (28.18 steady).

**Correction — TP does *not* beat PP at long context.** That claim was made
against the stale `pp-rdma-new` figure (334.53) flagged below. Re-measured on
the same commit, PP at 131k is **444.38**, and the picture is:

| | ≤ 16k | ≥ 32k | decode |
|---|---|---|---|
| winner | **TP** (+34% @2k, +44% @8k, +5% @16k) | **PP** (+15% @32k, +22% @65k, +21% @131k) | **TP everywhere** (+37–48%) |

Most of PP's jump is **our own work, not the transport**: the old PP table was
built off `tp-multi-slot-batching`, which predates the upstream indexer stack
(+20.8% on TP) and R8's `nsg=4`. Neither is TP-specific —
`ds4_gpu_flash_attn_nonvec_nsg()` sits on the shared Metal path — so PP
inherited both. 1.208 × 1.07 = 1.29 against an observed 1.33, leaving little
for the TCP-over-TB change.

The lesson to carry: several wins in this series lifted *both* architectures,
so TP-only deltas do not move the TP-vs-PP comparison by the same amount. Score
cross-architecture claims only against a same-commit PP run.

**Standalone-to-rig transfer factor: ~0.63.** R8 is the first result measured
both ways, and the two disagree. Backing the kernel speedup out of the
end-to-end numbers — the FA kernel is ~24% of the sweep-131k row and ~20% of
the cold run post-R7 — gives **~1.39× on M2 Ultra from both**, against
**2.19× measured standalone on M1 Max**. The two independent back-outs agreeing
to two decimals is what makes this a usable calibration rather than noise.

So the mechanism transferred and the magnitude did not. Discount future
standalone M1 Max numbers by roughly this factor before projecting, and treat
them as upper bounds. The projection for R8 quoted the full 2.19× while only
*verbally* flagging the microarchitecture risk — the flag was right, using the
undiscounted number anyway was not.

Worth noting what the changes before R8 had in common: none of them made the
GPU faster. A0, the indexer split and the static-mixed split each *removed
replicated work*. That well is now dry — the stages that must stay full-width
(`kv_path` + `compressor`) are only ~3.5–5.5 ms/layer — so everything after R8
is kernel efficiency or fixed overhead.

**PP column — resolved.** The R7c PP figures were indeed unsourced; the 131072
cell was carried over from the stale `pp-rdma-new` table. Re-measured on the
same commit and recorded as its own table. Flagging it was right; the guess
attached to the flag was not — I reasoned from the fresh column trending
*below* the old one at 65536 that a re-measured 131k would also be lower and
widen TP's margin. It came in far higher (444.38 vs 334.53) and reversed the
result. A number worth flagging as unsourced is a number not worth
extrapolating from either.

**Do not quote `speed-bench/m2_ultra.csv` or the "1.85× PP scaling" figure as
scaling references.** The former uses a fixed 2048-token increment per frontier
(vs this sweep's doubling increment) and cannot be running this model on one
128 GiB node; the latter compares a two-node **Q4** run against a single-process
**Q2** reference, which `speed-bench/README.md:264` itself flags as
"directional rather than an apples-to-apples speedup".

### Two readings to correct before they propagate

**A0 did not make gate stalls worse.** Per-gate wait *fell* (318 → 190 ms)
because each swap moves a smaller row range. In aggregate 1419×318 = 451 s
became 2132×190 = 405 s, a ~10% improvement, against +17% total wire time. The
+50% gate count was not the drag it looked like, and R3 confirms it: idle is
identical in both arms.

**`avg gpu-wait` is not additive stall.** 1419 × 318 ms = 451 s against a ~504 s
prefill at 90% residency cannot both be true unless the waits overlap — and they
do: the service thread keeps many gates in flight, and the metric measures
encode-to-satisfied per gate, not exclusive idle. Use it as a *ratio* against
exchange (the cost is drain/wait, not the link), never as a time budget.

The honest stall number is R3's flat ~10%, which bounds *idle-recovery* at
roughly +11%.

**Superseded, and it matters: the "exchange is only ~9% of the run, so the RDMA
staging/window workstream is closed" conclusion was wrong.** It was drawn from
R2, before the splits raised the BIG gate count to 2752. The R9 profile makes
the total explicit: 2752 × 21.8 ms = **60 s of wire in a 325 s cold prefill,
18.5%**, and it is on the critical path rather than overlapped. Wire is
therefore twice the share that argument assumed and larger than the entire idle
budget it was weighed against. R10a/R10b reopen it.

The underlying mistake was reasoning about a *ratio* (wait:wire per gate) as
though it bounded a *total*. Per-gate wire being small next to per-gate wait
says nothing about what wire sums to across thousands of gates — and the gate
count is exactly what this workstream kept increasing.

**Why Run 3 came in low, and it is not noise.** A0 splits the attention
*consumption*, which `top_k` caps at a constant `(512 + 128) × 64 × 512 × 2`
= 4.19e7 MACs/token/layer. It does **not** split the indexer *scoring*, which is
`n_comp × 64 × 128` and grows linearly with context — 2.68e8 MACs/token/layer at
131k, i.e. **6.4× larger**. The measured gain falls exactly as that ratio rises
(8k: 0.4× → +11.1%; 131k: 6.4× → +7.2%), which is the mechanism, not scatter.

Consequence: **A0 structurally cannot fix long-context degradation** — the term
it splits does not grow with context. The term that does is still replicated on
both ranks. Splitting the indexer is the actual fix, and unlike A0 it adds **no
gate traffic**, because score/top-k are per-query-row: each rank scores its own
rows, feeds its own top-k into its own attention rows, and the existing
`attn_out` swap already recombines everything downstream.

| | compute removed /token/layer | gates added | GPU idle |
|---|---|---|---|
| A0 (landed) | 4.19e7 MACs | +23/chunk (43 → 66) | worse (~90% util observed) |
| indexer split (next) | 2.68e8 MACs | **0** | unchanged |

## Consolidated defaults (`f45b535`)

The three measured prefill wins are **default ON** and no longer need flags.
Each keeps an escape hatch — set it to `0` on BOTH ranks to disable — so the
A/B protocol still works and a regression can be bisected without a rebuild.
Asymmetry still deadlocks the big gates; only the direction of the default
changed.

| flag | default | evidence |
|---|---|---|
| `DS4_TP_PREFILL_SPLIT_NONZERO` (A0) | **on** | +7.2% @131k, Run 2 correctness pass |
| `DS4_TP_PREFILL_SPLIT_INDEXER` | **on** | +19.5%, bit-identical (R5a) |
| `DS4_TP_PREFILL_SPLIT_STATIC_MIXED` | **on** | +20.5%, bit-identical (R7b) |
| `DS4_METAL_FA_NSG` | **4** | +7.3% @131k, bit-identical (R8) |

**Removed:** `DS4_TP_DECODE_REPLICATE_ATTN` and its code path. R11 measured it
at −8.4 to −11.8% decode at all seven contexts; a clean negative on a decode
hot path is worth deleting rather than carrying. `tp_attn_decode_split`
collapses back to `g->tp_world == 2`, with the predicate comment recording why
so the idea is not re-derived from the gate count.

**Closed, no code to remove** (upstream knobs, left at their defaults):
`DS4_TP_SUBGATE_PIPELINE` (R10a wash), `DS4_METAL_DECODE_NWG` (R10d, default
already optimal).

**Benchmark arms should now run with no TP flags at all.** A "flag-off"
baseline means explicitly setting the three to `0`.

## Open requests

### HC scoping result — 2026-08-27 — three corrections, and one finding bigger than the stages themselves

Full report: `docs/SCOPE-HC-STAGES.md`. **The 4.56 ms is not 4.56 ms of HC
work**, and the credible headroom is **~0.6–1.0 ms (2.5–4.1% of the 2k token)**,
not 18.7%. Three things it establishes matter beyond the HC stages.

**1. The TP gate is a 1-thread GPU spin loop, and it is inside these stages.**
`ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_FFN)` sits inside the `ffn_hc_post`
span (`ds4.c:25335`, marker at `:25377`), and its release fence is
`kernel_dsv4_tp_fence_wait` (`metal/dsv4_misc.metal:7333`) — a bounded spin that
blocks the command buffer for the whole RDMA round trip. `attn_hc_post` runs the
**identical kernel at the identical grid with no gate, in the same run**, so the
difference prices the gate directly:

> **1.812 − 0.461 = 1.351 ms = 43 × 31.4 µs/gate**, against an independently
> measured ~38 µs/gate.

**This is what the 30 W is.** A 1-thread spin keeps the GPU "busy" at
essentially zero power, which is why the stage profile reports 99% busy with a
0.31 ms gap while the part draws a quarter of its envelope. The "there is no
stall" conclusion was correct about *idle* and wrong about *useful work* — the
stall is inside the busy time, wearing a kernel's clothes.

With the ATTN gate inside `attn_output` too, **~2.7 ms/token — 11% at 2k — is
gate spin.**

**2. `attn_output` is not a target.** Net of the ATTN gate it prices at ~640
GB/s = **84% of roof**. §8's "phantom target" warning holds; the sibling agent's
largest item is pre-empted.

**3. Every stage number in this document is inflated ~13%.**
`DS4_METAL_GPU_STAGE_TIMESTAMPS` makes each stage its own command buffer (~900
per token against 3), and U12's 32k sum is **32.70 ms profiled vs 28.91
unprofiled on the same build `b99dfa3`** — 3.79 ms over ~903 boundaries =
**4.2 µs each, ≈0.18 ms per marker**. Net of their own boundary:

| stage | profiled | net |
|---|---|---|
| `q_a_kv_proj` | 2.137 | 1.957 |
| `q_lora_norm` | 1.680 | **1.500** |
| `ffn_hc_post` | 1.812 | 1.632 |
| `attn_hc_post` | 0.461 | 0.281 |

U16 survives this comfortably — 1.50 ms on 2 threadgroups is still the worst
ratio in the engine — but **quote net numbers from here on.**

### What this does to the queue

**U14's shared-shift comes back up, with a measured mechanism.** I deprioritised
it as an abstract load-balance argument worth 0.82 ms. It is now visible as
**1.351 ms of measured spin in `ffn_hc_post`**, and the reason the peer is late
is precisely the `E[max(k, 6−k)] = 3.9375` imbalance §7 describes. Shifting the
shared expert to the routed-light rank attacks exactly this, costs no memory and
no repack, and is **fixed work — it pays at every context**. That puts it level
with U16 and above the entire HC candidate pool.

**Grid widening on the HC producers is already a recorded negative.** `d81a28f`
on branch `pr-778` (not in this branch) implements the obvious 6→10 threadgroup
spread, bit-identical, for **+0.12% on an M5 Max** — because TG0 alone carries
four matvec rows *plus* the whole 4096-wide collapse+RMS epilogue, and spreading
does not touch it. Do not re-propose it; H2 (unloading TG0) is the version with
a mechanism, and it is not bit-exact.

**Three zero-code arms to fold into the next rig run**, all from the report:
`DS4_TP_ABLATE=hcpre` (resolves an unexplained 2.7× disagreement between the
ablation's 0.85 ms and the profile's 2.276 ms);
`DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE=1` (tests whether the
fusion's three device-scope `seq_cst` fences cost more than the dispatch they
removed — a possible free default flip); and a three-line marker after
`ds4.c:25335` that turns finding 1 from inference into a measured row.

### attn_output / router / shared scoping result — 2026-08-27

Full report: `docs/SCOPE-ATTNOUT-ROUTER-SHARED.md`.

**It independently reached the gate finding**, from the other side: the ATTN
gate (`kernel_dsv4_tp_flag_set_coherent` + `kernel_dsv4_tp_fence_wait`) is
encoded at `ds4.c:23856` into the same batch, before the marker, so
**0.4–1.35 ms of `attn_output`'s 3.804 ms is gate, not kernel.** Two agents
working different stages converged on the same 1-thread spin. That is now the
best-supported finding in the document.

**Net of the gate, `attn_output` is 524–534 GB/s = 69–70% of roof** — not the
84% I estimated from the first report, and not §4's "~100%". The byte model was
re-derived from the encoder and reconciles to §3 within **0.3%**, so the
`attnout` phantom is not repeated. It is not underfill either (4096
threadgroups/layer).

**The 30% gap is the Q8_0 matvec's k-curve, and it is program-wide.**
Confirmed model-free on an M1 Max (`b95e7ff`) — identical 17.83 MB at every
shape, rate varying **2.9×**:

| k | GB/s | % of peak |
|---|---|---|
| 512 | 89.8 | 22.4% |
| 1024 | 113.7 | 28.4% |
| 2048 | 185.4 | 46.4% |
| 4096 | 253.1 | 63.3% |
| 8192 | 261.5 | 65.4% |

Stable under 1% across four passes. **The rig's curve is a different shape** —
§5 records 283/413/**581**/541/517, a hump peaking at k=2048 — so the corrective
must be validated there, but both machines agree the rate is strongly
k-dependent at constant bytes.

**`shared_down` is the clean consequence.** `shared_tp_local = shared_dim/2`
(`ds4.c:24150`) k-slices a natural **k=2048** matvec down to **k=1024** — from
the peak of the rig's curve to well below it — and measures **293 GB/s, 39% of
roof**. The M1 Max ratio for that same step is 185.4/113.7 = **1.63×**, against
the rig's 581/293 = 1.98×. **Row-splitting instead reads identical bytes at the
peak k**, and the summing FFN gate composes if the unowned half is zeroed once
at allocation. This is a TP-split-axis choice, not a kernel rewrite.

**`router` has no kernel headroom** — 82 GB/s and 0.4% of ALU peak, but only two
dispatches (a 128-threadgroup matvec and a *single-threadgroup* 256-element
bitonic sort). 25.8 µs/layer is latency, dispatch and ~16–20% profiler tax. Only
dispatch removal moves it, and the router+shared fusion is TP-blocked at
`ds4.c:24042`.

**Two corrections to standing docs.** `ds4.c:23802`'s "2560 threadgroups" is
stale — the actual grid is 4096. And **this plan at :648 is wrong** that T4
swept attention-output-low: `out_a` passes a **literal `nsg=4`**
(`ds4_metal.m:26399`) that `DS4_METAL_Q8_MV_NSG` cannot reach, so it sits
unswept at the value T4 measured as 3% worse everywhere else.

### Consolidated ranking — 2026-08-27

Net of the ~0.18 ms/marker profiler tax, and with both scoping reports in:

| item | ms | % of 2k token | confidence |
|---|---|---|---|
| **gate spin** (ATTN + FFN) | ~2.7 | 11.1% | measured twice, independently |
| ~~U16~~ | — | **Diagnosed, not implemented.** The kernel is ~22 µs of pure dispatch latency, flat across a 64× work range, so a fusion buys nothing. Same fix as the gate spin: concurrent scheduling. **Held behind the gate agent.** |
| **C1** Q8_0 k-curve, program-wide | 1.13–1.40 | 4.6–5.8% | 2.9× spread confirmed model-free |
| **U14** shared-shift | ≤0.74 | **re-sized down** from 0.82, which exceeded the whole imbalance. Design changes from "shift all" to "shift to equalise", which reaches zero FFN-gate imbalance in the δ<S regime. Gated on the per-slot gate measurement. |
| **HC pool** (H4/H6/H2) | 0.6–1.0 | 2.5–4.1% | scoped; grid-widening already negative |
| **C3** row-split `shared_down` | ~0.25 | 0.8–1.1% | k-curve consequence |
| **C2** `out_a` nsg (unswept) | ~0.25 | 0.8–1.2% | one literal to change |

**Sum of the non-gate items: 3.75–4.92 ms against the 4.34 ms needed for 50 t/s
at 2k.** For the first time the target is covered by *identified, measured-gap*
items rather than by extrapolation. Temper that with this project's hit rate —
roughly one in five proposals has survived contact — but the estimates are now
grounded in roofline gaps rather than guesses.

**The gate spin is the largest single item and the least understood.** It is not
obviously recoverable — an RDMA round trip has to happen — but 11% of the token
in a 1-thread spin deserves its own investigation before more kernel work. The
question is whether it can be overlapped rather than removed.

### U15 result — the model holds, and the 2.7× disagreement resolves — 2026-08-27

**Context-invariance is confirmed.** Every stage at 2k matches its 32k value
within noise except `compressor_indexer`, which collapses **4.957 → 0.201 ms**
exactly as predicted. The short-context program rests on a measurement now, not
an inference.

**One reclassification: `attn_inv_rope` is not context-growing.** It reads
**3.806 @2k / 3.393 @32k / 4.252 @131k** — it *dips* then rises, so its 32k→131k
slope was misleading. At 2k it is the **second-largest stage**. Consistent with
it being mostly FlashAttention taking a different branch at short context, and
with U13 being worthless to the short-context program.

**The profiler tax at 2k is 18%, not 13%** — 28.613 ms profiled against ~24.34
unprofiled. Same fixed per-boundary cost over a smaller token, so the correction
is larger here. Net of ~0.18 ms/marker:

| stage | net ms | % of the 24.34 ms token |
|---|---|---|
| `routed_moe_folded` | 4.72 | **19.4%** |
| `attn_inv_rope` | 3.63 | **14.9%** |
| `attn_output` | 3.55 | **14.6%** (0.4–1.35 of it ATTN gate) |
| `q_a_kv_proj` | 1.97 | 8.1% |
| `ffn_hc_post` | 1.74 | 7.2% (**1.35 of it FFN gate**) |
| `q_path` | 1.70 | 7.0% |
| `q_lora_norm` | **1.50** | **6.2%** |
| `attn_hc_pre` | 0.96 | 3.9% |
| `ffn_hc_pre` | 0.93 | 3.8% |
| `router` | 0.92 | 3.8% |
| `shared_gate_up` | 0.79 | 3.3% |
| `shared_down` | 0.49 | 2.0% |
| `attn_hc_post` | 0.29 | 1.2% |
| `compressor_indexer` | 0.02 | 0.1% |

**The gate decomposition holds at 2k.** `ffn_hc_post` net 1.74 − 1.351 gate =
**0.39 ms** of real HC-post work, against `attn_hc_post`'s **0.29 ms** for the
identical kernel with no gate. Two independent stages, same answer.

**The 2.7× profile-vs-ablation disagreement is resolved, and it is not a
measurement error.** `DS4_TP_ABLATE=hcpre` takes `attn_hc_pre` 1.135→0.645 and
`ffn_hc_pre` 1.109→0.627 — it removes only ~43% of each span and **leaves
1.27 ms still in them**. So the ablation chain simply does not cover the whole
stage; §9's caveat applies exactly. The profile is right, the ablation is right,
and they measure different things. **Close this open item.**

Consequence: the real ablatable HC-pre pool is **~0.76 ms end-to-end**, and the
HC candidate pool shrinks toward the bottom of its 0.6–1.0 ms range.

**The pre-norm fusion is a net win — do not flip it.** Disabling it costs
**+0.58 ms** (attn_hc_pre 1.135→1.416, ffn_hc_pre 1.109→1.398). The three
device-scope `seq_cst` fences cost less than the dispatch they remove. That was
one of the three speculative zero-code arms; it closes as *current default is
correct*.

### Revised short-context ranking after U15

| item | ms | % of the 2k token | status |
|---|---|---|---|
| **gate spin** (ATTN + FFN) | 1.75–2.7 | **7–11%** | confirmed at 2k; largest, least understood |
| **C1** Q8_0 k-curve, program-wide | 1.13–1.40 | 4.6–5.8% | 2.9× spread confirmed model-free |
| ~~U16~~ | — | **Diagnosed, not implemented.** The kernel is ~22 µs of pure dispatch latency, flat across a 64× work range, so a fusion buys nothing. Same fix as the gate spin: concurrent scheduling. **Held behind the gate agent.** |
| **U14** shared-shift | ≤0.74 | **re-sized down** from 0.82, which exceeded the whole imbalance. Design changes from "shift all" to "shift to equalise", which reaches zero FFN-gate imbalance in the δ<S regime. Gated on the per-slot gate measurement. |
| **C3** row-split `shared_down` | ~0.25 | ~1.0% | k-curve consequence; `shared_down` is only 0.49 ms net |
| **C2** `out_a` nsg (unswept) | ~0.25 | ~1.0% | one literal |
| **HC pool** | **~0.5** | ~2% | **downgraded** — ablation says 0.76 ms exists at all |

Non-gate sum **≈4.0–5.2 ms** against the **4.34 ms** needed for 50 t/s. Still
covered, but the HC contribution shrank and the two largest items (`routed_moe`
4.72 and `attn_inv_rope` 3.63, together 34% of the token) remain without a
proposal.

### Five-arm battery — one win banked, C1 falsified, and two of my instruments were wrong — 2026-08-27

**C2 banked and shipped.** `DS4_METAL_ATTN_OUT_LOW_NSG` swept 1/2/4/8 →
38.86/40.66/41.15/**41.85** t/s at 2k and 28.02/28.79/29.20/**29.61** at 131k.
**Monotonically better with width; nsg=8 is +1.7% at 2k, +1.4% at 131k.** T4's
"nsg=4 is ~3% worse everywhere else" does **not** reproduce on the rig. Default
moved 4 → 8. This is the second banked win after U10, and it came from a literal
that was simply unreachable by the sweep that was supposed to cover it.

**Arm 1 — the straggler is real: U14 and §7 survive.** ATTN and FFN exchange are
not equal — the FFN gate, which alone sits behind the routed shard, trails by
**+11.6 µs at 2k / +8.1 µs at 131k**.

**But my instrument's derived figure double-counted, and I am correcting it
down.** It printed `43 × 2 × delta`. Since `delta = E|s|/2` is *already* the
per-layer excess on the critical path — the token advances at the slower rank's
pace, so the cost is `E[max] − E[mean] = E|s|/2` — the per-token figure is
`43 × delta`:

| ctx | delta | printed | **correct** |
|---|---|---|---|
| 2k | 11.6 µs | 1.00 ms | **0.50 ms** |
| 131k | 8.1 µs | 0.70 ms | **0.35 ms** |

So the straggler is **0.50 ms at 2k (2.1% of the token) and 0.35 ms at 131k**,
and it is *smaller* at long context. **U14 survives but is now a ~2% item**, down
from the 0.82 ms I first claimed and the ≤0.74 ms first correction. Formula
fixed in `ds4_metal.m`.

**Arm 3 — my `attn_tp_gate` marker renamed a stage instead of splitting it.**
It reads 3.719 ms at 2k against the old `attn_output`'s 3.731, and `attn_output`
fell to ~0.000. The cause: I placed the boundary *after* `ds4_gpu_tp_gate_encode`
(`ds4.c:23856`), so the span still began at `attn_inv_rope` and swallowed the
out_a/out_b projections. **The 3.72 ms is the projections, not the gate.** Fixed
by adding an `attn_out_proj` boundary immediately *before* the encode, so the
three spans separate: projections → gate → post-gate add. **Re-run arm 3.**

**Arm 4 — C1 is falsified as specified, and C3 is supported.** The rig's Q8_0
curve peaks at **k=4096 (445–458 GB/s)** and dips at 8192 — a different shape
from the M1 Max's monotonic climb *and* from §5's historical peak at k=2048
(283/413/581/541/517). The current absolute numbers are also well below §5's, so
**§5's curve should not be quoted further**. C1's premise (a fixable shape) does
not survive contact.

C3 does: **k=1024 → k=2048 is 279 → 410 GB/s = 1.47×** on the current rig. On
`shared_down`'s 0.49 ms net that is **0.16 ms, 0.64% of the 2k token** —
real, small, and now measured rather than inferred.

**Arm 2 — my flash-attn split does not reconcile, and the reason is my
implementation.** Per-call `fa_core` ~0.31–0.33 ms and `reduce` ~0.40–0.49 ms
at 2k, but the per-call sums come to ~30 ms against the 3.8 ms stage marker, and
the profiled run does 13 t/s against 41 unprofiled. **I mirrored the prefill
pattern, which requires batch context — and batch context distorts decode
beyond usefulness.** A0 therefore does not yet decompose the 3.63 ms, and A2
stays unsized. Needs a non-batch decode path before it is worth re-running.

### Repack-on-load — makes M3 and §7C the same mechanism, and kills the objection that buried §7C — 2026-08-27

**Both M3 and §7C are pure byte reorganisations of data we already have.**
Neither changes a value: MXFP4 dequant is `d = e8m0_to_f32(e)` times a 16-entry
table of exact E2M1 values, so a planar split of `{uchar e; uchar qs[16]}` into
an aligned payload plane and a scale plane is bit-identical by construction, and
§7C's intra-expert split is just a different ordering of the same expert bytes.

**So do it in the engine at load, not offline.** The rank already knows at load
time whether it is in TP mode and which shard it owns. Making the layout a
runtime property removes, at a stroke, the two things that got §7C struck:

- ~~"costs a conversion tool"~~ — no tool.
- ~~"a TP-layout-specific model file"~~ — no second artefact, no format
  versioning, no "which GGUF is this" problem.

**And it makes §7B's blocker disappear too.** §7B was rejected as *"~22k spans —
blocked"*, because a rank's half of each expert is not contiguous in the mapped
file and `ffn_down_exps` splits on the inner dim. That is an *addressing*
problem, and it only exists while you are reading in place. **If the rank builds
its own contiguous copy — read scattered, write contiguous, once at load — there
are no spans.** Each rank writes only its own slice, so its destination is
*half* the routed weights.

**Sizing, from the verified byte model** (`13.37 MB`/expert, matching the T8
bench exactly): 128 owned experts × 43 layers = **73.6 GB/rank** of routed MoE;
147.2 GB across both.

| cost | assessment |
|---|---|
| **Load time** | 73.6 GB read+write per rank. At a realistic 5–10 GB/s effective that is **~7–15 s** added to startup. Today the loader is mmap-lazy and copies nothing. |
| **Peak memory** | The destination is ~73.6 GB (half that for the §7C variant). Source pages are clean and file-backed, so `MADV_DONTNEED` as the stream advances keeps steady-state RSS roughly unchanged — **but only if it is written as a stream, not "allocate both, then copy."** |
| **Wired limit** | 120 GB configured; destination + non-MoE + KV needs checking before committing. |
| **SSD streaming** | **Incompatible as stated.** `ssd_streaming` pages experts on demand; you cannot pre-repack what is loaded lazily. Either exclude that mode or repack per streamed chunk. |
| **Existing hook** | None. Weights are strictly mmap-in-place (`newBufferWithBytesNoCopy` over the GGUF); `ds4_layer_pack.c` is layer-to-device placement, unrelated. This is a new loader path. |

**It is not a cache-locality change, and getting that right shrinks M3.** Both
layouts are read strictly sequentially — a matvec walks blocks in order — so
cache lines are equally well used and the same bytes come out of DRAM either
way. What planar actually changes is **alignment**: at a 17-byte stride the
16-byte payload is never naturally aligned, so it cannot be fetched as one
aligned vector load and ~6% of accesses straddle a cache line.

**Both halves of that mechanism are already measured, and they are small.**

| measurement | result |
|---|---|
| **U8** — stride 17 vs stride 16, same 16-byte payload, on the rig | 713.2 vs 758.0 GB/s = **5.9%** |
| **M1** — 8 scalar byte loads → 2 alignment-1 `packed_uchar4` loads | **null** (734.2 vs 729.2 ms, byte-identical) |

5.9% of `routed_moe`'s 4.72 ms is **0.28 ms — 1.1% of the 2k token**, against
M3's estimated 2.30–2.45 ms. **An 8× gap between the estimate and the mechanism
we have already measured.** M1 says the instruction-count half is worth nothing;
U8 says the straddle half is worth ~6%. Neither supports 9.7%.

**So M3's estimate is not credible on the evidence, and the residual 48%→100% is
probably not layout at all** — more likely nibble unpacking and LUT traffic (M2),
or simply that a 4-bit format cannot reach the rate of one that does no
unpacking. Worth remembering MXFP4 is already the faster format *per weight*:
365 GB/s × 1.88 weights/byte ≈ 687 Gweights/s against Q8_0's ~547.

**Consequence: justify repack-on-load on §7C alone.** §7C is a load-balance
change whose value does not depend on any of this, and it is measured directly
by the per-slot gate profiler. If M3 rides along for free once the loader
exists, fine — but it should not be the reason to build it.

**Both ranks must repack, and it is not symmetric work.** The worker has to run
the same conversion when it sees TP, and for §7C the two ranks write *different*
slices, so this is not a case where "both run the same binary" is sufficient
reassurance. Three specific hazards, all with precedent in this tree:
the decision must derive only from inputs both ranks share (TP active, world
size, rank index) and never from a one-sided env — `DS4_TP_ABLATE` already
carries a "same value on both ranks" warning, and `DS4_METAL_FAST_SYNC` silently
collapses the decode split when set on only one; the ~7–15 s cost is paid by
both, and the coordinator will block on the worker; and a rank that skips the
repack while its peer performs it produces silently wrong output rather than an
error, so the layout choice needs to be part of the handshake, not an
assumption.

*(SSD streaming is not a constraint — confirmed out of scope.)*

**Two independent gates, one shared mechanism.** This makes both items *cheaper
to try* — it does not make either more likely to pay:

- **M3 is gated on M2.** M1 already falsified the load-shape half of the dequant
  premise (byte-identical, 734.2 vs 729.2 ms, nothing). **M2 — the byte-indexed
  `float2` LUT, ~30 lines and no loader work — is the remaining cheap test.** If
  it is also null there is nothing for a repack to fix, and the loader work
  would be wasted.
- **§7C is gated on the per-slot gate profiler** (next run, arm 1), which prices
  the straggler directly. Its value is load balance and is **independent of M3's
  premise** — so §7C can survive M2 killing M3, and vice versa.

**Order: run M2 and arm 1 first. Build the loader only for whichever survives.**
Both surviving is what makes the shared mechanism worth its ~7–15 s of startup.

### MoE / attention scoping — two different problems, and a correction to T8 — 2026-08-27

Full report: `docs/SCOPE-MOE-ATTN.md`.

**Correction — `routed_moe_folded` is 365 GB/s, not the ~410 this plan has been
quoting, and T8 priced kernels production cannot run.** The 410 is U1's
*isolated world-1 harness*. **All five MXFP4 specialisations are gated on
`add_in == NULL && tp_world == 1`, so none of them can ever fire on the folded
TP path.** T8's "the ladder is worth nothing" conclusion survives — but for a
stronger reason than it recorded: those kernels are unreachable under TP, not
merely unprofitable. Live rate is **365 GB/s = 48% of roof**; bytes reconcile
exactly against §3 (3.0 owned experts × 13.369 MB × 43 = 1.7246 GB).

**The k-curve does not explain the MoE — clean negative.** gate/up run at
k=4096 and down at k=2048, the rig's *two best* Q8_0 points (541 and 581 GB/s).
Unlike `shared_down`, there is no bad geometry here.

**What is left is per-byte work.** `block_mxfp4` is `{uchar e; uchar qs[16]}` at
**alignment 1**, so the payload reads as 8 scalar byte loads
(`metal/moe.metal:4501`) and dequant is 16 threadgroup LUT lookups per 8 bytes —
~2.9 memory instructions per weight byte against ~1.5 for the Q8_0 matvec.

**M1 tested and reverted — the instruction-count model did not predict.**
Rewriting all nine sites to alignment-1 `packed_uchar4` loads was
**byte-identical** (FNV-1a fingerprint unchanged) and **measured nothing**:
interleaved GPU-time A/B gave 734.2/734.2 ms patched against 729.2/734.5
baseline, a gap inside the baseline's own spread. Not shipped. **M2 (byte-indexed
`float2` LUT) is untested and now doubtful** — and it would add ~1.9 KB of
threadgroup memory, which U7/U10 showed is the binding constraint on residency.

**`attn_inv_rope` is a parallelism problem, near neither roof by 1.5 orders of
magnitude** — 6–28 GB/s (0.8–3.7%), 305 GFLOP/s (1.4%), ~35× its own
instruction-issue floor. Four causes, all structural: the flash reduce is
hard-wired to exactly **32 threadgroups on 60 cores** every layer
(`ds4_metal.m:29334`); the vec kernel is one simdgroup per threadgroup, capped
at ~9 resident/core by 3,328 B of shared memory; **22 of 43 layers dispatch only
128–160 threadgroups**; and KV is re-read **64×** (32 heads × K-then-V on the
same buffer). This is the sixth sighting of the underfill constraint and by far
the worst.

**A0 built (`8943015`).** The decode attention could not be decomposed at all —
`ds4_gpu_flash_attn_stage_profile_boundary` existed but was wired only into the
four *prefill* encoders, which is why the restructuring estimate honestly spans
0–8%. `ds4_gpu_encode_flash_attention_gathered_heads` now takes the command
buffer by pointer and carries four boundaries — `gather`, `packed`, `fa_core`,
`reduce` — so `DS4_METAL_FLASH_ATTN_STAGE_PROFILE` splits the 3.63 ms in one
run. Inert unless the env is set.

**The one large item, and it now has a partner.** M3 — repack MXFP4 planar (an
aligned `qs` plane plus a separate scale plane) — is **2.30–2.45 ms, 9.5–10.1%
of the token**, high cost, and **shares its conversion tool with §7 option C**
(the intra-expert repack for the straggler). Together those are **~15% of the
token for one offline tool and one TP-specific model file.** That is a
materially different proposition from either alone, and it is the first thing in
this document that could move decode by more than a few percent.

### Gate-overlap scoping — mostly irreducible, and four corrections — 2026-08-27

Full report: `docs/SCOPE-TP-GATE-OVERLAP.md`.

**Verdict: ~1.3 ms of the 1.75–2.70 ms is hardware one-way latency** for a 16 KB
exchange that must happen 86 times. That line has four levers only: fewer gates
(structurally impossible), a smaller payload (not bit-exact), a faster fabric,
or more tokens per gate. **Overlap does not exist to be found — it has to be
manufactured.**

**Correction 1 — I was wrong that the gate spin explains the 30 W.** I asserted
that twice. 8–11% of the token at zero power predicts **~53 W, not 30 W**. The
"GPU busy ⇒ useful work" correction stands, but the power gap is mostly
latency-bound matvec versus prefill GEMM. Retracted.

**Correction 2 — M0's "exchange is 6–7% of gate time" divides by the wrong
denominator.** `DS4_TP_GATE_PROFILE`'s `gpu-wait` leg is encode-to-execute lead
*plus all inter-gate compute*, because `ds4_gpu_tp_gate_encode` enqueues at
**encode** time (`ds4_metal.m:10513-10525`), not execute time. The GPU's actual
stall is the **exchange** leg (24.8 µs), not `gpu-wait` (375 µs). 24.8 µs + two
dispatches + two forced encoder breaks ≈ 30 µs, which reconciles with the
31.4 µs differential. **Exchange is ~65–80% of the gate; wire is ~60% of that.**
T1's post-mortem reason ("dominated by local GPU completion") is wrong — its
null result stands, but not for the reason recorded.

**Correction 3 — §8's concurrent-encoder dismissal is factually wrong.** A
`MTLDispatchTypeConcurrent` path **exists and ships** (`ds4_metal.m:1028-1030`,
`:9526-9733`, live at `ds4.c:24985`). The single `memoryBarrierWithResources` is
the one level break inside the one concurrent region, not evidence that overlap
is impossible. It is disabled under TP2 only because `fuse_shared_down_hc`
requires `tp_world < 2` (`ds4.c:24182-24187`). Also relevant: `ds4_gpu_tp_big_gate_kick`/`_wait`
already implements a kick/wait split for prefill, explicitly "to interleave more
GPU work with the wire exchange" — and its comment records a **measured
stale-payload failure** when arrival used a flag word with no fence behind it.
**That is the go/no-go risk for any overlap work.** `MTLSharedEvent` release is
closed by measurement (~186 µs/gate); a wider fence is closed (21% vs 4%).

**Correction 4 — U14 is over-sized and, as specified, unachievable.** Since the
ATTN gate is symmetric and carries no straggler,
`exchange_FFN − exchange_ATTN = E|s|/2` exactly. The measured 23.6 µs pooled
exchange bounds **E|s| ≤ 34.4 µs, so the straggler is ≤ 0.74 ms/token
(est. ~0.3 ms)**. U14's uniform `E[max] = 3.9375` model implies 1.29 ms of
excess — **incompatible at any non-negative software cost**, so its 0.82 ms win
exceeds the entire imbalance. But we are in the `δ < S` regime, where the
*correct* design is "shift the shared expert to **equalise**", not "shift all" —
and that drives FFN-gate imbalance to **zero** rather than partway. **Re-size
U14 to ≤0.74 ms and change its design.**

*(One thing in the report I do not follow: it lists C3 as dying with the
straggler. C3 is the `shared_down` row-split, which is a k-curve consequence —
k=1024 sits below the kernel's peak — and is independent of load balance. Left
in the queue.)*

### Instrument built for the next run — `a861150`

The report's recommended measurement, implemented rather than queued:

- **Per-slot gate profiler.** `DS4_TP_GATE_PROFILE` now splits row gates by
  ATTN/FFN and prints both averages, their delta, and the implied per-token
  straggler bound. `req.gate` was already in the queue struct, so this perturbs
  nothing on the GPU.
- **`attn_tp_gate` stage marker**, mirroring `ffn_tp_gate`.

**This is a decisive arm: if the two gates measure equal, the straggler is zero
and U14 plus all three §7 designs are dead in one run.** Also queued from the
report, zero engine change: re-run M3's `uc_pingpong` with 4×4 KB chained
against 1×16 KB — minutes, worth ≤0.34 ms.

### U16 — measured before implementing, and the fix changed — 2026-08-27

I set out to fuse the q-LoRA norm away. Measuring the kernel first
(`tests/bench_qkv_norm`, `867cae3`) showed the fusion would have bought nothing.

**The kernel does no work.** Batched with rotated outputs to avoid
write-after-write serialisation, per-dispatch cost converges to **~22 µs** — and
sweeping `q_n` from 256 to 16384, **a 64× range of work, leaves it flat at
21–23 µs**:

| q_n | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|
| µs/dispatch | 22.6 | 21.4 | 23.5 | 23.0 |

It is **entirely per-dispatch latency**. The engine's 34.9 µs/layer is that
~22 µs plus ~13 µs of surrounding serialisation.

**This reconciles with §6 rather than contradicting it.** The ballast measured
~1.9–3.7 µs marginal, but ballast dispatches are independent no-ops inserted
*among* real work and pipeline with it. `q_lora_norm` is the opposite: a
dependency barrier with two threadgroups and nothing to overlap, so it pays full
end-to-end latency. **Both numbers are right; they measure different
situations.** Record the distinction — a dispatch's cost in this engine is
~2–4 µs if it overlaps and **~22 µs if everything waits on it.**

**So U16's fix is not a kernel change and not a partial fusion.** Folding only
the q half into `q_b` would leave the kv norm/RoPE/store dispatch in place, and
the dispatch *is* the cost — the saving would be ~0. Removing the whole dispatch
needs the kv half folded somewhere too, and its consumer is the KV cache, which
persists across tokens and cannot absorb it.

**U16 therefore converges on the gate problem.** Both are dependency barriers on
a serial encoder with nothing scheduled alongside; both are fixed by giving the
GPU concurrent work, not by making a kernel faster. The relevant prior is §8's
*"Concurrent decode encoder — only one `memoryBarrierWithResources` exists in
43k lines; dependent dispatches cannot overlap. **Argued down, not tested.**"*
The batch encoder already supports `MTLDispatchTypeConcurrent`
(`g_batch_encoder_concurrent`, `ds4_metal.m:1026`), and at that point in the
layer the **compressor projection is independent of the q path** — it reads
`attn_norm`, not `qr` — so there is real work available to overlap.

**Held behind the gate-overlap agent**, which is investigating exactly this
mechanism at larger scale (~2.7 ms vs U16's ~0.95 ms). Implementing concurrent
scheduling twice, differently, would be worse than waiting for its answer.

**What this reprices.** If a dependency-barrier dispatch costs ~22 µs, the
engine's small dispatches are worth far more than §6's campaign estimate
suggested — but only the ones that block. Identifying *which* dispatches are
barriers with nothing alongside is now a first-class question, and
`bench_qkv_norm` is the instrument for pricing any of them in isolation.

### Iteration 2 ran on the C2-broken build — what survives — 2026-08-27

Build `6b962db` had `out_low_nsg = 8` live, so `kernel_dsv4_attn_out_low_q8_0_f32`
was compiled for 8 simdgroups and dispatched with 4. Each simdgroup covered
1/8 of k instead of 1/4, so the threadgroup read **half the k range** — faster
and wrong. The previous 5-arm battery (build `345de30`) predates C2 and is
**clean**; that gives a clean control for the same procedure on the same rig.

**The contamination is narrow and quantifiable.** Every unrelated stage matches
between the two runs within 0.03 ms:

| stage @2k | clean | broken | Δ |
|---|---|---|---|
| routed_moe | 4.911 | 4.927 | +0.016 |
| q_a_kv_proj | 2.152 | 2.141 | −0.011 |
| q_path | 1.874 | 1.876 | +0.002 |
| q_lora_norm | 1.673 | 1.675 | +0.002 |
| attn_inv_rope | 3.813 | 3.784 | −0.029 |

So the runs are comparable and the artifact is localised. In the attention
region: clean 3.719 (the mislabelled combined figure) against broken
2.383 + 0.994 = 3.377 — **+0.342 ms of C2 artifact**.

**Arm 1 survives with one correction.** `attn_out_proj` is real and
context-invariant, but its clean value is **~2.73 ms, not 2.38**. The headline
finding stands and is important: **the 3.72 ms was a mislabel, and the actual
ATTN gate is only ~0.99 ms.**

**The gate is now directly measured rather than differenced:** `ffn_tp_gate`
1.643 + `attn_tp_gate` 0.994 = **2.64 ms**, against my earlier ~2.7 ms estimate
from differencing `ffn_hc_post` against `attn_hc_post`. Two independent methods
agreeing. And the split is informative — **the FFN gate costs 1.65× the ATTN
gate, which is the straggler, since only the FFN gate sits behind the routed
shard.**

**Arm 3 must use the CLEAN number.** Garbage attention output feeds the router,
so expert selection under C2 is not the model's real distribution — and it
shows: the delta grew from **+11.6 µs (clean) to +15.4 µs (broken)**, consistent
with degenerate routing concentrating experts. **The straggler is ~0.50 ms, not
0.66.** U14 and §7 survive at **0.50 ms, 2.1% of the token** — the third
successive downward revision of that item (0.82 → 0.74 → 0.66 → 0.50).

### Arm 2 cannot work as built, and the sampling fix did not address the real defect

Per-call `fa_core` 0.25–0.41 ms and `reduce` 0.27–0.35 ms, against a 3.78 ms
stage for all 43 layers. Sampling cut the calls from 10,496 to 123 as intended,
and the reconciliation is **still 6× out**.

**The reason is the boundary mechanism itself, not the call count.**
`ds4_gpu_flash_attn_stage_profile_boundary` takes `ds4_gpu_now_ms()` — a **host
wall clock** — either side of `ds4_gpu_end_commands()` / `begin_commands()`. It
therefore measures a **command-buffer round trip**, not kernel time. The
0.25–0.41 ms figures are exactly the round-trip cost `bench_qkv_norm` measured
independently (423 µs wall, 110 µs GPU-busy per cb at one dispatch per buffer).

This works for prefill, where a stage is milliseconds and the round trip is
noise. **At decode, where stages are microseconds, the round trip is the entire
measurement.** A0 as designed can never decompose the decode attention.

**What it needs instead:** GPU timestamps sampled *within one command buffer*
(`MTLCounterSampleBuffer` at dispatch boundaries), not encoder splits. That is a
different instrument, not a parameter change. **A2 stays unsized, and I should
stop reporting A0 as built.**

### RETRACTION: the "22 µs dependency-barrier dispatch" was my measurement error — 2026-08-27

The 50 t/s fan-out caught this and it is the most consequential error I have
made on this project, because several later conclusions were built on it.

**What I claimed.** `tests/bench_qkv_norm` showed ~22 µs per dispatch, *flat
across a 64× range of work*, so the kernel did no work and the cost was pure
exposed dispatch latency. From that: a dependency-barrier dispatch costs ~22 µs
against the ballast's 3.74 µs; U16's fix must therefore be dispatch removal
rather than kernel optimisation; and `1021 × 22 µs ≈ the 24.34 ms token`.

**What was wrong.** I swept `argv[1]` = `q_n` from 256 to 16384 while holding
`argv[2]` = `kv_n` at 512. **`q_n` is not the cost driver.** Sweeping the other
parameter:

| kv_n | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|
| µs/dispatch | 10.7 | 21.3 | 76.9 | 293.7 |

**It scales almost linearly.** A fit over the work-dominated end gives
**0.0353 µs/element + 4.6 µs fixed**, which predicts 9.2 / 22.7 at kv_n 128 and
512 against 10.7 / 21.3 measured. At the production shape the split is **~18 µs
of work and ~5 µs fixed — 80/20, not 0/100.**

**So the fixed cost is ~5 µs, consistent with the ballast's 3.74 µs.** The two
numbers never needed reconciling; I invented a "barrier versus overlapped"
distinction to explain a gap that did not exist.

**What this invalidates.**

1. **U16's kill reason is void.** I concluded that fusing the q-LoRA norm "buys
   nothing because the dispatch is the cost". The kernel is 80% work at the
   production shape, so **optimising it is exactly the right move** — the
   opposite of what I recorded. U16 is reopened.
2. **Grid widening is back on the table for this kernel.** 2 threadgroups doing
   512 elements in ~18 µs is ~36 ns/element — a serialised chain
   (norm → RoPE → quantize → store) with almost no parallelism. The "widening
   has failed twice" caution came from *other* kernels and was over-generalised
   here.
3. **The `1021 × 22 µs ≈ token` arithmetic is dead.** It was numerology resting
   on a wrong per-dispatch price.
4. **The 30 W explanation is weakened but not void.** The underfill sightings
   and the 99%-busy / ~20%-utilised gap stand on their own evidence; what falls
   is the specific claim that per-dispatch latency accounts for most of the
   token.

**The methodology lesson, which is the durable part.** I swept a parameter,
observed flatness, and concluded "no work happens here" — without checking that
the parameter I varied was the one the kernel spends time on. **A flat response
to a swept parameter means either the work is fixed, or you swept the wrong
knob.** Distinguishing them costs one more sweep. Add to the rules in §
Methodology: *when a sweep comes back flat, sweep a second parameter before
concluding anything.*

### Queued work, built or scoped 2026-08-27

#### Built: `DS4_METAL_GPU_ENCODER_TIMESTAMPS` — a profiler that does not distort

`df0037e`. Samples the GPU timestamp counter at **encoder** boundaries inside one
command buffer, instead of ending the command buffer and reading
`cb.GPUStartTime`. Overhead **~1–3%** against the existing profiler's **13–18%**.

Probed the hardware before designing: `atStageBoundary` is supported,
`atDispatchBoundary` is **not**, so per-encoder is the finest granularity
available. Mechanism validated against known workloads — 4× work measured 3.82×,
2× measured 1.92×, encoder sum within 4% of the command-buffer span.

**Why this matters beyond one stage.** Every stage figure in this document
carries a 13–18% correction, and two conclusions were wrong because of
measurement distortion rather than the thing being measured. This removes the
correction.

**Limitation, stated plainly:** a stage that is one encoder yields one number.
Decomposing `attn_inv_rope` further needs encoder splits at the sub-stage points
— much cheaper than the command-buffer splits they replace, but not free and
**not yet done**.

**Rig use:** add `DS4_METAL_GPU_ENCODER_TIMESTAMPS=1` to the iteration-3
re-baseline. It cross-checks the old profiler on the same run, and if the two
disagree the new one is the one to believe.

#### Scoped: speculation under TP2 — and a terminology correction that matters

**Three different things have been conflated, including by me:**

| | what it is |
|---|---|
| **MTP** | a draft *model* file (`--mtp FILE`), i.e. the support model |
| **DSpark** | the speculative *driver* that uses it — `--dspark` **requires** `--mtp` |
| **n-gram** (PR #846) | a *drafter* needing no model at all, just history matching |

So we are **not** "using MTP instead of DSpark". n-gram **replaces the MTP
support model as the drafter, while sharing DSpark's batch verify** — the code
says so at `ds4.c:69992`: *"N-gram speculation replaces the support-model drafter
rather than stacking with it: both end in the same batch verify."*

**Consequence: switching drafter does not avoid the TP problem.** The exclusion
is in the shared verify/session path (`s->distributed`, `ds4.c:69931`), which
n-gram inherits.

**What it does avoid is the reason DSpark lost.** From
`speed-bench/tp_mtp_hunt.md` (2026-08-19, single-node, ctx-512 Promessi):

| arm | steady decode | acceptance |
|---|---|---|
| target only | **41.98 t/s** | — |
| published 5.99 GB support, forced d5 | 21.29–21.35 | 189/1596 = **11.84%** |
| full-fat MXFP4 support after the F32-HC fix | 19.91 | 157/1767 = **8.89%** |
| low-yield production policy | 41.57 | verifier never launched |

**DSpark's pass cost ~3× a decode step, and that is dominated by a 5.99 GB
support-model forward every cycle.** n-gram's draft is a hash lookup — free. The
last row is the tell: with no verifier launched, drafting alone costs ~1%.

**So the open question is verify cost alone, which nobody has isolated.** If a
verify pass with N rows costs ~1× a decode step (weights are read once per pass
regardless of rows), 50 t/s needs acceptance **L = 1.22**. If it costs 2×, it
needs **L = 2.43**.

**And acceptance is fixture-dependent in exactly the way that favours us.** The
same doc records llama.cpp at ~50% on predictable code generation and ~25% on
literary prose, and warns their "mean 3.5" includes the unconditional target
token. **Promessi prose is the worst case; this rig's production workload is
tool-calling and structured output**, which is where n-gram drafting is
strongest. The 8.89–11.84% figures are prose numbers with a weak drafter.

**Do not start this without reading `tp_mtp_hunt.md` in full.** It documents a
completed attempt at the structural TP work and the reasons it was set aside.
The remaining pieces are the batched verify across the pair — the machinery
exists, since multi-row batches already cross the gates in prefill — and
checkpoint/rewind on rejection under the mirrored-session protocol, which does
not. **Days, not hours.**

### Speculation is structurally excluded under TP2 — the wiring task is not a wiring task — 2026-08-27

Before writing an n-gram call site I checked whether one would help. It would
not. The exclusion is architectural, not a missing caller.

1. **TP requires a distributed role.** `--tensor-parallel requires --role
   coordinator or --role worker` (`ds4_tp.c:531-534`). TP is expressed *through*
   the distributed machinery; there is no TP without it.
2. **A coordinator session therefore gets a distributed handle.**
   `if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR)` →
   `ds4_dist_session_create(&s->distributed, ...)` (`ds4.c:60993`, `:61153`).
3. **And the speculative entry bails on exactly that**, first thing
   (`ds4.c:69931`):

```c
    if (s->distributed) {
        if (!accepted) return 0;
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
```

**So under TP2, speculation of any kind — n-gram or DSpark — falls straight
through to a plain single-token eval and returns 1.** Adding a bench call site
would have measured baseline throughput a second time, which is precisely the
failure the last arm already produced.

**Why this matters more than the earlier finding.** "Unreachable from the bench"
implied a call site would fix it. It would not: on this rig, in the
configuration we actually run, the feature cannot engage at all.

**It also resolves the DSpark number.** "Forced Promessi runs remain 19–21 t/s"
cannot have come from a TP coordinator, or it would have read 41 (the
fall-through). It was single-node — so that figure says nothing about
speculation under TP, in either direction. Note `speed-bench/tp_mtp_hunt.md` and
the `tp-mtp-hunt` branch exist precisely because someone already attempted the
structural TP work; **read that before re-attempting it.**

**Re-sizing the prize, because it is still the largest one available.** Weights
are read once per *pass* regardless of row count, so a verify pass with N draft
rows costs about what a 1-row decode costs. Throughput is then `L / 24.34 ms`
for acceptance length L:

| L | t/s |
|---|---|
| 1.00 (today) | 41.1 |
| **1.22** | **50.1** |
| 1.50 | 61.6 |

**50 t/s needs only L = 1.22** — one extra token accepted on 22% of steps — *if*
verify is near-free. The production workload is tool-calling and structured
output, which is where n-gram drafting is strongest; Promessi prose is its worst
case. **But if verify costs 2× a decode step, break-even moves to L = 2.00 and
50 t/s needs L = 2.43.** Both unknowns fall out of one run, once it can run.

**Cost: this is days, not hours.** It needs the batched verify to work across
the pair — draft is deterministic from token history so both ranks can derive it
independently, and multi-row batches already cross the gates during prefill, so
the machinery exists. What does not exist is checkpoint/rewind on rejection under
the mirrored-session protocol. **Do not start it without reading
`tp_mtp_hunt.md` first.**

### U5 / n-gram speculation is UNREACHABLE, not merely unrun — 2026-08-27

Found while running the arm: `DS4_NGRAM_SPEC=1` on the plain bench decode loop
returned **42.00 t/s with zero draft, verify or spec activity**. That is the
baseline. **The arm never engaged the feature.** Traced and confirmed:

- `ds4_session_eval` routes straight to `ds4_session_eval_probe_tp`
  (`ds4.c:64765`). No speculation, no n-gram.
- The n-gram dispatch lives at `ds4.c:69996`, inside
  `ds4_session_eval_speculative_argmax_impl` (`:69923`).
- That impl has exactly two public entries, `ds4_session_eval_speculative_argmax`
  and `..._excluding` (`:70745`, `:70755`).
- **The only caller of either, anywhere, is `ds4_bench.c:866`** — gated on
  `cfg.dspark && !cfg.dspark_strict`, and `--dspark` **requires `--mtp FILE`**
  (`ds4_bench.c:398`).
- **`ds4_server.c` never calls either entry.** It parses `--dspark` and sets
  `c.engine.dspark`, but in the engine that flag only selects DSpark weight
  loading and prints a warning at open (`ds4.c:59879`); the server's generation
  loop has no speculative path at all.

**So there is no speculation in the production serving path, and n-gram in
particular can only be reached from the bench with an MTP file loaded** — a
draft-model mechanism entirely separate from n-gram drafting.

**Three consequences.**

1. **Every previous framing of U5 as "implemented, never run" was wrong.** It is
   implemented and *not wired up*. Running it needs a call site, not a rig slot.
2. **The recorded DSpark negative does not transfer.** "Forced Promessi runs
   remain 19–21 t/s" was measured through `--dspark --mtp`, i.e. an MTP draft
   model. It says nothing about n-gram drafting, whose draft is nearly free.
   Those two were being treated as one datum.
3. **This raises speculation's cost and lowers its risk simultaneously** —
   more work than an env flag, but the reason it has never shown a benefit is
   now known to be "it never ran", not "it does not work".

**Correction owed to the running 50 t/s workflow:** its speculation lens was
briefed with "implemented and never run on the rig", which is materially
incomplete. Whatever it concludes about acceptance rates must be re-read against
the fact that wiring is a prerequisite. Amend the synthesis when it lands.

## Why decode is at 41 t/s, why the GPU draws 30 W, and why TP2 is only ~1.6× — 2026-08-27

These are one question. Answering it properly says the remaining 1% items
cannot reach 50 t/s and points at what could.

### The token is 2.6× off its own realistic floor

Bytes per token per rank, from the stages whose byte models reconcile against
§3 (routed MoE 1.725 GB, q_a+kv 0.279, q_b 0.744, attn out_a/out_b 1.538):
**4.29 GB**, excluding shared/router/HC/indexer.

| at | ms | t/s |
|---|---|---|
| 760 GB/s (streaming roof) | 5.6 | 177 |
| **450 GB/s (best real Q8_0 kernel on this rig)** | **9.5** | **105** |
| **actual** | **24.34** | **41.1** |

**Even against a rate we have actually observed from a production kernel, decode
is 2.6× slow.** The gap is ~15 ms — not the 4.34 ms that separates us from
50 t/s. No combination of the queued 1% items closes it, and that is the honest
reason 50 t/s has stayed out of reach.

### The mechanism: exposed memory latency on a serial dependency chain

**The decode graph is 1021 dispatches and they are fully serialized.** Metal's
default encoder is `MTLDispatchTypeSerial`. A concurrent path exists and ships
(`g_batch_encoder_concurrent`, `ds4_metal.m:1028`) but is armed in exactly one
place — `ds4_gpu_parallel_ffn_start` — and the fusion it belongs to,
`fuse_shared_down_hc`, requires **`g->tp_world < 2`** (`ds4.c:24191`). **Under
TP2 nothing in decode runs concurrently.**

So every dispatch waits for its predecessor whether or not it depends on it, and
most are far too small to hide a DRAM round trip. Direct measurement
(`tests/bench_qkv_norm`): a dependency-barrier dispatch costs **~22 µs and is
flat across a 64× range of work** — it is pure exposed latency.

**Six independent sightings of the same underfill, none of them looked for:**

| | |
|---|---|
| `q_lora_norm` | **2 threadgroups** on 60 cores, 43×/token |
| head-norm/RoPE | 32 threadgroups |
| flash reduce | 32 threadgroups, every layer |
| 22 of 43 layers | dispatch only 128–160 threadgroups |
| indexer LLT | **1 threadgroup resident/core**, smem-capped |
| `packed32` at 32 heads | correct but −1.35 t/s — underfills |

**That is the 30 W.** The GPU is 99% *busy* and ~20% *utilised*: few threads
resident, ALUs waiting on memory, memory at a fifth of its rate. Prefill has the
same kernels with thousands of rows in flight, so it fills the machine and draws
60 W. **Busy ≠ occupied**, and the stage profiler only ever measured busy.

*Bound on this theory, stated honestly:* the dispatch ballast measures **3.74 µs**
per added no-op, not 22 µs. Those no-ops touch no memory, so they have no DRAM
latency to expose. **The floor is not "22 µs per dispatch" — it is "~20 µs for a
dispatch that must reach DRAM before it can start and has nothing to overlap
with."** The engine has many of those; it does not have 1021 of them.

### Which is also why TP2 is only ~1.6×

Splitting across two nodes halves the *bytes* per dispatch. It does not change
the *number* of dispatches, and it adds 86 gate exchanges. Only the work-bound
part scales:

| | ms |
|---|---|
| work (scales with bytes) | 17.0 |
| latency (does not) | 4.9 |
| gate (TP-only, does not) | 2.7 |

Single node = 2 × 17.0 + 4.9 = **39.0 ms (25.7 t/s)** against TP2's 24.34 —
**1.60×, not 2×.** The shortfall is exactly the **31%** of the token that does
not scale, and that 31% is precisely the latency-bound part. **TP2's scaling
deficit and the single-node inefficiency are the same defect**, which is why
fixing it pays twice.

### Restructuring options, ranked by leverage

1. **Concurrent encoder — I mis-sized this, twice over. Corrected 2026-08-27.**

   I called it "blocked only by an unrelated fusion's `tp_world < 2` guard" and
   "the cheapest structural change". Both wrong.

   **It is deliberate, and the code says so.** `parallel_ffn_route_eligible`
   carries `g->tp_world < 2` *itself* (`ds4.c:~22180`), not via some other
   fusion, and the comment directly above it reads: *"A concurrent compute
   encoder removes all of the generic routed-MoE function's implicit dispatch
   ordering. **Keep admission deliberately narrower than the normal decode
   path**: fixed resident IQ2 pair-SwiGLU, direct Q2 top-6 sum, no
   readback/profiling/TP detours."* It also excludes SSD streaming, every
   `cuda_tp_*` mode, profiling, directional steering and debug prefixes.

   **The reason is sound.** Under `MTLDispatchTypeConcurrent` dispatches may
   overlap unless separated by an explicit barrier, and the decode graph relies
   on the serial encoder for *implicit* ordering throughout. Enabling it broadly
   means converting ~24 dispatches per layer from implicit to explicit
   dependency ordering. Miss one and the result is silent corruption — the same
   failure mode as the C2 regression, which shipped a wrong-output kernel that
   looked like a 1.7% win because a throughput sweep cannot see garbage.

   **And the prize is smaller than I implied, and lands at the wrong end.**
   Overlap is bounded by the *smaller* of the two independent branches. The q
   path (`q_a_kv_proj` + `q_lora_norm` + `q_path` = 5.17 ms) and the
   compressor/kv path both start from `attn_norm`, so they are genuinely
   independent — but the compressor branch is ~0.5 ms at 2k and ~9.7 ms at 131k:

   | | bound on the saving |
   |---|---|
   | 2k | **~0.5 ms (2.1%)** — there is almost nothing to overlap with |
   | 131k | ~5.2 ms (15.1%) |

   **So it does not serve the 50 t/s short-context goal at all** — at 2k the
   compressor branch is nearly empty. It is a long-context lever, and an
   expensive, corruption-prone one. **Demoted from first to last.**
2. **Collapse a layer's ~24 dispatches toward 1–3.** The precedent is in-tree:
   `kernel_dsv4_comp_row_finalize_f32` already folds seven tiny per-layer
   dispatches into one, bit-exactly. Every additional fusion removes an exposed
   round trip, and U16 showed the dispatch *is* the cost.
3. **Speculation, reframed.** Not "raises arithmetic intensity" — **it amortises
   the fixed latency over N tokens.** The same 1021 serialized dispatches
   produce N tokens instead of 1, so the non-scaling 31% is divided by the
   acceptance length. This is the only lever that attacks the floor without
   touching a single kernel, and `DS4_NGRAM_SPEC` is implemented and still
   unrun.
4. **Raise occupancy where a grid is provably too small** — the six sightings
   above. But note two attempts already failed (`packed32` −1.35 t/s, the
   `pr-778` HC spread +0.12%), so treat per-kernel widening as the *low*-leverage
   option, not the obvious one.

**What this reprices.** The queued items sum to 1–3 ms against a ~15 ms gap.
They are not wrong, but they are not the path to 50 t/s. Item 1 is a day's work
against a defect worth several milliseconds, and item 3 costs nothing but a run.

### Falsifier resolved: slot reuse it was — and the confirmed overlap is the bigger finding — 2026-08-27

On the fixed build the inferred tick **pins at exactly 1.000 ns at 2k**, against
0.632–0.642 before. The pre-registered falsifier said a tick that stopped
varying with context meant slot reuse, and it did. **The counter is 1.0 ns on
both parts; the sub-nanosecond readings were corruption.**

**Coverage stays at ~197%, and for the first time that number means something.**
With the corruption gone it is a genuine measurement: **the encoder spans really
do overlap about 2×.**

**That is worth more than the calibration fix, because it explains three
standing nulls at once.** 175 encoders in a 23.8 ms token would average 136 µs
if they did not overlap; measured they average 268 µs, so each span extends
~132 µs past its own work — about one neighbour. A fixed per-encoder offset
would have to be 132 µs to explain that, which is not plausible; a pipelining
depth of ~2 is. **The GPU already overlaps adjacent encoders.** Therefore:

| null | why it was a null |
|---|---|
| R12a split schedule, flat within 0.9% | boundaries already overlap, so moving them changes nothing |
| ballast marginal dispatch, 3.74 µs | an added dispatch hides inside the pipeline |
| dispatch-count reduction, dead | the count was never the constraint |

Three independent results, one mechanism. **Stop treating the decode graph as
"1021 strictly serialised dispatches" — it is not, and I asserted that for most
of today.** What survives is that the *work* is latency-bound within each
kernel; what does not is that the *scheduling* is the problem.

**Instrument change: a normalised column.** Raw spans sum to ~2× the wall clock
and are useless as a budget, so the report now prints raw, normalised
(`× cb/sum`) and share-of-cb side by side. The normalised column sums to the
command buffer, which is what a per-stage budget needs. It assumes overlap is
uniform across encoders — where it is not, the error is redistributed rather
than removed — and both columns are printed so that assumption stays visible.

**Arm B is now usable.** Re-run on a build ≥ this commit and take the
**normalised** column as the budget; treat raw as an upper bound.

### Arm B re-run — one bug, not two, and the data says which — 2026-08-27

The calibrated re-run reported **tick 0.632–0.642 ns and coverage ~198%
OVERLAPPING at 2k**, and **tick ~1.000 ns with ~198% coverage at 131k**. The
bench read that as two independent causes: an M2 Ultra tick of ~0.63 ns *plus*
genuine 2× encoder overlap. **I think it is one cause, and the data rules the
other reading out.**

**A hardware tick period cannot vary with context length.** 0.632 ns at 2k
against 1.000 ns at 131k is not a property of the counter. And overlap cannot
move the inferred tick: the scale is `cb_seconds / (hi − lo)`, and overlap
changes the *sum* of spans, not their extent. A tick of 0.63 requires `hi − lo`
in ticks to exceed the command buffer's duration in nanoseconds by **1.57×** —
impossible for encoders inside one buffer.

**One bug explains both numbers: the sample slots were global, not per command
buffer.** Encoders from a second in-flight buffer appended to the first one's
range, so `hi − lo` spanned ~2 buffers (inferred tick ~0.5–0.65) while the
report divided the span sum by *one* buffer's wall clock (coverage ~200%). That
1.000 ns at 131k is the M1 Max value exactly, which is the tell: **the true tick
is 1.0 ns on both parts**, and the 2k reading was corruption.

**Fixed:** slots are now owned by a command buffer, the range resets when a
different buffer starts filling it, and a report against a foreign buffer is
refused rather than mis-scaled. While debugging I observed **command-buffer
addresses being recycled**, so pointer identity alone would let a new buffer
reuse a freed address and append to a stale range — the report therefore
disowns the range when it completes.

**The falsifier, stated in advance.** If the next run still shows the inferred
tick varying with context, my explanation is wrong and the bench's two-cause
reading stands. If the tick pins at ~1.000 ns at both contexts, it was slot
reuse and **coverage becomes a real overlap measurement for the first time** —
at which point a coverage figure meaningfully above 100% would mean the
per-encoder spans genuinely overlap and are upper bounds.

**Until then, treat both arm-B runs' absolute numbers as void** — including the
calibrated `reduce` figures of 199 µs/call at 2k and 311 at 131k. The
instrument's *distortion* claim is unaffected and still holds: throughput came
back at baseline (42.04 / 29.70 t/s) on both runs.

## Arm R1 and Arm S — the two measurements that matter next, both built — 2026-08-27

Everything below is implemented and building. Neither needs code from the rig.

### Arm R1 — name the 2.42 ms inside `attn_inv_rope`

**Why this and not a candidate.** B6 split the bracket for the first time and
found only a third of it: `fa_core` 0.560 + `reduce` 0.66 = 1.22 of 3.64 ms.
The other **2.42 ms is 10% of the decode token and larger than every surviving
optimization candidate combined** (best case 1.22 ms). Launch explains 0.461 ms
(133 dispatches × 3.464 µs) and KV staging traffic 0.038 ms. **~1.9 ms is
unnamed.** No candidate is worth a rig slot before this is.

**Built (`ds4_metal.m`).** The bracket had **4** label sites, all of them
flash-attn phases, so the two calls that dominate the rest were invisible. Added
**5** more via `DS4_METAL_PROFILE_BRACKET_STAGE`, taking coverage to **9**:

| new label | call | site |
|---|---|---|
| `idx_sort` | `kernel_dsv4_sort_i32_rows_asc` | `ds4_metal.m:30745` |
| `idx_attn_split` | indexed-mixed attention, split path | `:30781` |
| `idx_split_red` | split reduce | `:30792` |
| `idx_attn` | indexed-mixed attention, unsplit | `:30815` |
| `rope_tail` | `kernel_dsv4_head_rms_norm_rope_tail_f32` | `:23050` |

These cover `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` (4 clusters) and
`ds4_gpu_rope_tail_tensor` (1) — previously **zero** labels between them.

**Run — two passes, same as B6:**

```
DS4_METAL_GPU_ENCODER_TIMESTAMPS=1                                            # 2k + 131k, gen 128
DS4_METAL_GPU_ENCODER_TIMESTAMPS_SPLIT=1 DS4_METAL_GPU_ENCODER_TIMESTAMPS=1   # 2k + 131k, gen 128
```

**Read in order:**

1. **Pass 1's composite line.** With 9 labels in one batch segment the collapse
   is *worse*, not better — expect composites like `idx_sort..reduce+9`. That is
   the instrument working; it is why pass 2 exists.
2. **Pass 2, the five new spans × their per-token counts.** Sum with `fa_core`
   (0.560) and `reduce` (0.66). **The residual against 3.64 ms is the answer.**
3. **If the nine labels now sum to ≈3.64 ms**, the 2.42 ms is named and we have a
   new target list ranked by span.
4. **If a large residual persists**, the remaining work is not in these calls at
   all, and the next step is the call site in `ds4.c` rather than the encoder.

**Pre-registered falsifier.** If pass 2's nine spans sum to *more* than 3.64 ms,
the SPLIT perturbation is inflating spans faster than it resolves them — already
suspected, since a dev-box `DS4_METAL_DECODE_NWG` sweep of 2→32 (16× the reduce
traffic) did not move the reduce span. In that case treat every pass-2 number as
a ceiling and read only the **ratios** between the nine.

**RESULT 2026-08-27** (build `179c105`): **the falsifier fired; the residual is
still not named.** Pass 1 collapses the whole 9-label bracket into one composite
(`rope_tail..reduce` @2k, three `rope_tail..*` forms @131k); honest reads hold
(gap ≈ 0, conc ~1.97, throughput baseline 42.03 / 29.37). Pass 2 @131k inflates
pathologically (`idx_attn` 38511 µs, total 212 ms ≫ 4.24) — the exact
pre-registered failure. Pass 2 @2k splits only three labels cleanly:
rope_tail 0.222 + fa_core 0.537 + reduce 0.662 = **1.42 ms**; `idx_*` and
gather/packed stay fused (2k) or inflate (131k). **The ~1.9 ms is still
unexplained — next step is the call site in `ds4.c`, not the encoder.** Full
data in `BENCHMARKS-TP-PP.md` §Arm R1.

### Arm S — the n-gram commit rate, never once measured on this rig

**Why it is the highest-value single session available.** Speculation is the only
route to 50 t/s left. A drafted step emits `1 + commit` tokens for `T + V`, so it
wins iff `commit > V/T`. Everything turns on one number nobody has measured.

**Built, and deliberately offline.** `ds4_ngram_propose` (`ds4.c:69749`) is a pure
function of token history, so the commit rate is computable without a GPU, without
a model, and — crucially — **without the four known defects in the live
speculative path** (`s->logits` never refreshed on exit, `drafts[0]` pushed without
an eval, the bound returning 0 on the ideal periodic case). Those break the
*implementation*; this measures the *opportunity*, which is what gates the week.

- **`DS4_NGRAM_TRACE=<path>`** (`ds4.c:53375`) — passive. Writes every committed
  decode token id, one per line. Does **not** enable speculation and does not
  touch the decode path beyond an `fputs`.
- **`tests/bench_ngram_accept`** — replays the trace through the **shipped**
  proposer (linked, not copied) simulating the real cycle: propose at *i*, commit
  the longest matching prefix *c*, restart at *i+1+c*. Sweeps k ∈ {2,3,4,5,6,8} ×
  depth ∈ {2,3,4,5,8} in one pass and reports offered%, mean commit, tokens/step,
  speedup and the commit-length distribution.

**Run:**

```
DS4_NGRAM_TRACE=/tmp/toks.txt ./ds4-bench ...        # any real workload; coding is the one that matters
make tests/bench_ngram_accept
./tests/bench_ngram_accept /tmp/toks.txt --vt 4.459  # today's 5-row verify
./tests/bench_ngram_accept /tmp/toks.txt --vt 2.06   # corrected ~50 ms verify floor
```

Use a **real coding session**, not a synthetic prompt — n-gram drafting lives or
dies on repetition, and the PI coding harness is the workload that matters.
A few thousand tokens minimum; the harness refuses below 64.

**Validated here** on two controls with known answers:
- period-8 synthetic: commit saturates at depth (1.985 / 2.968 / 3.950 / 4.926)
  and caps at period−k = 5, faithfully reproducing the shipped proposer's own
  look-ahead bound;
- uniform random: 0.0% offered, 0.000 commit, every configuration a loss.

**What the controls already tell us, before any rig data.** At **today's** verify
cost the *perfectly periodic* trace — 98.5% offer rate, 4.926 mean commit, the
best case that can exist — reaches only **1.086×**, and only at depth 5. Every
shallower depth loses. **So the binding constraint is the verify cost, not the
drafter.** A real coding trace will land far below the periodic ceiling.

**Decision rule.** If the real trace clears ~2.0 mean commit at `--vt 2.06`, MTP
is fundable *conditional on* the verify floor being fixed first — chiefly
`ds4.c:37202` setting `tp_batch_rows = n_tokens`, which disables the row split at
`ds4.c:29237` and costs 10.88 ms of the 108 ms verify. If it does not clear 2.0
even at the corrected floor, **speculation is dead on this workload** and 42–43.5
t/s is the ceiling — which is worth knowing for one session's cost rather than a
week's.

## The decode program after B6 — what survives, and where the next rig slot goes — 2026-08-27

Output of a 41-agent enumeration against the corrected budget, every candidate
attacked by an independent refuter. **The headline is negative and should be
taken at face value: the new findings do not open a path to 50 t/s.** 50 needs
4.26 ms (24.260 → 20.000); the entire surviving inventory is **1.22 ms best
case = 29% of the gap.**

### Ranked program

Anchor 24.260 ms = 41.220 t/s.

| candidate | stage | ms/token | +t/s | cum t/s | effort | first measurement |
|---|---|---|---|---|---|---|
| **item C** — remove 86 `flag_set_coherent` dispatches | TP gate | **0.510** (0.30–0.51) | +0.88 | 42.10 | days | rig ballast **slope** {0,2,8,16} interleaved — the absolute still rests on a 2-point delta |
| **U16 / item A** — parallelise 7 serial FP8 amax blocks | q_lora_norm (1.50) | **0.35** (0.25–0.50) | +0.60 | 42.70 | days | `tests/bench_qkv_norm` block-count bisection on a rig node; model-free, minutes, no TP |
| **U14′** — variable asymmetric shared-expert split | ffn_tp_gate straggler (0.59) | **0.30** (0.20–0.48) | +0.53 | 43.23 | days | **already satisfied**: gate profile gives +12.2–12.7 µs/layer |
| **G-b** — H heads per `fa_core` threadgroup | attn_inv_rope / fa_core (0.560) | **0.15** (0.13–0.22) | +0.26 | 43.49 | week+ | rig head sweep H=4/8/16/32, 10 min, one node |
| *fence-scope reduction* (not queued) | TP gate | 0.201 | +0.34 | — | days | needs a memory-model argument first |

**Discounted for the measured reversal rate.** Eleven ex-ante estimates on this
rig; **best realised +0.12%**: C2 +1.7%→reverted, packed32 −1.35 t/s, pr-778
+0.12%, U9 0.77×, M1 0.0%, T8 0 to −5%, T1 ±0.6%, R11 −8.4 to −11.8%, R12a within
0.9%, 32ef898 +0.019 ms. Four of those (packed32, pr-778, U9, M1) are the
grid-widening/occupancy class — **0-for-4** — which is exactly G-b's class. So
U16 and U14′ (removal of a proven serial chain; correction of a *directly
measured* imbalance) get 60% realisation and G-b gets 25%:
`0.35×0.6 + 0.30×0.6 + 0.15×0.25 = 0.43 ms` → **41.97 t/s**.

**Honest range 42.0–43.5 t/s, central ~42.0–42.4. The 43–45 belief should move
down.** It rested on three supports that are all gone: a 1.10 ms `compressor_update`
plug, a 3.63 ms bracket assumed to be mostly `fa_core`, and three banked wins
worth ~0 at 2k.

### The one thing bigger than the whole program

B6 puts `fa_core` + `reduce` at `0.560 + 0.66 = 1.22 ms` of the 3.64 ms
`attn_inv_rope` bracket — **33.5%. The other 2.42 ms (10.0% of the token) is
unlabelled**, because only four FA label sites exist, so KV staging and
`ds4_gpu_rope_tail_tensor` (`ds4.c:23570`) fall outside them. Launch accounts for
`133 × 3.464 µs = 0.461 ms` and KV staging traffic for
`16.97 MB / 449.2 GB/s = 0.038 ms`, leaving **~1.9 ms unexplained — more than the
entire surviving program's best case, and nobody knows what it is.**

**This is where the next rig slot goes.** Not on any candidate in the table.
Add label sites across the whole bracket and re-run enc-ts; it eliminates the
±0.18 ms/marker tax instead of re-weighting it. (Note `ds4_metal.m:1427` skips the
timestamped descriptor when the encoder is concurrent, so this and any C1-class
work can never validate each other.)

### B6's own numbers are ceilings

An independent dev-box replication under `SPLIT=1` swept `DS4_METAL_DECODE_NWG`
2→32 — a **16× change in reduce traffic** (131 kB → 2.10 MB) — and **the reduce
span did not move**. At forced NWG=32, where the reduce dispatch is byte-identical
in every layer, spans alternated 40.5 / 82.3 / 41.3 / 78.1 / 40.3 / 88.8 µs,
tracking the neighbouring `fa_core`. Treat **30.5 µs/call as an upper bound** on
reduce and 25.8 as approximately right or high on `fa_core`. The 2.42 ms remainder
is unaffected — it is a subtraction, so inflation inside the labels only makes it
larger.

### Newly dead (do not re-propose)

- **`compressor_update` 1.10 ms — a PHANTOM.** True cost 0.021–0.046 ms. The
  1.10 ms was bookkeeping. Kill: `DS4_METAL_GPU_STAGE_TIMESTAMPS=1 …_DETAIL=1`,
  2 min, predict `buffers=41` never 43 and `gpu_ms ≈ 0` on 3 of 4 tokens.
- **A1 reduce DV split** — reduce is 0.66 ms total; wave arithmetic caps the win
  at 1.33× = 0.025–0.073 ms, 5–14× inside the ±0.34 ms anchor spread.
- **G-a V-reuse** — a C=32 key block is `32×512×2 = 32,768 B`, exactly 100% of
  `maxThreadgroupMemoryLength`, on top of an existing 3,328 B = **36,096 B against
  a 32,768 B limit**. Dropping to C=16 changes the online-softmax blocking and
  loses the bit-identity that was G-a's whole justification.
- **CU-EMIT-FUSE** — the merge as specified is a **data race**: both pools share
  `g_compressor_pool_softmax` at offset 0.
- **W1/W2/pr-778 HC widening** — `d81a28f` already did it bit-identically, +0.12%.
  W1's 60-TG grid-wide rendezvous is a hang class.
- **R1 router top-6 fusion** — `ds4_metal.m:34020` already collapses the tail
  3 dispatches → 1 on M2 Ultra; removable is 49, not 86.
- **F1 store_one fold** — `state_already_stored = qkv_pair_quad_fused`
  (`ds4.c:22881`) is already true; store_one fires zero times per token.
- **R13c shared-kvpad**, **pool lane widening**, **M2 byte-indexed LUT**,
  **attn_out_proj "anomaly"** (adopting the 2.867 ms re-baseline *adds* 0.32 ms to
  the book), **marker-tax reweights** (re-rank nothing; tightest gap 0.010 ms).

### MTP is the only route to 50, and it is not fundable yet

A drafted step emits (1+commit) tokens for T+V, so it wins iff `commit > V/T`.
Today `V = 12332.9/114 = 108.18 ms` per 5-row verify against `T = 24.260`, so
break-even is **commit > 4.459 of a maximum 5** — near-perfect acceptance of four
drafts every step. Dead as it stands.

Corrected verify floor: 21.2 (MoE at a 4.77× expert union) + 10.88 (attention
reads 4.886 GB because `ds4.c:37202` sets `tp_batch_rows = n_tokens`, which
disables the row split at `ds4.c:29237`) + 15.9 (dense) + 2.7 (gates) + 1.25
(unsplit output head) = **~50 ms**. At V=50, break-even is commit > 2.06; at a
sustained commit of 3, `(24.26+50)/4 = 18.57 ms = ` **53.9 t/s**; at 2.5, 47.1 t/s.

Cost: fix the three §13 causes, then four confirmed n-gram defects — `s->logits` is
never refreshed on any exit path of `ds4.c:69770-69921` so the caller re-samples
stale logits and emits a duplicate token every cycle; `drafts[0]` is pushed at
`ds4.c:69812` with no eval so its KV row is never written; `ds4.c:69759`'s bound
returns 0 on the ideal periodic case. The corpus's own post-fix estimate is
**30–36 t/s — still below the 41.2 baseline.** Weeks, currently negative EV.

**The single cheapest thing that would change this: no n-gram acceptance rate has
ever been measured on this rig, on any fixture.** One session. It is the gate on
whether MTP is worth weeks, and it is the only measurement in this document that
could move the ceiling rather than the estimate.

### Model-free probes run here — item C roughly doubles, and the tax constant holds — 2026-08-27

Two harnesses (`tests/bench_stage_marker_tax.m`, `tests/bench_flagset_tax.m`),
both model-free, run on the M1 Max dev box. Absolute µs are M1 Max; the
**ratios** are what transfer to the rig.

**1. `DS4_METAL_DISPATCH_BALLAST` is not representative of item C — it under-prices
it by 71%.** Ballast dispatches `kernel_ds4_dispatch_ballast`
(`metal/unary.metal:320`), a pure no-op. The real dispatch under
`DS4_METAL_FAST_SYNC` is `kernel_dsv4_tp_flag_set_coherent`
(`metal/dsv4_misc.metal:7403`), which brackets a single 4-byte store with **two
system-scope seq_cst device fences**. Slope fit over N = 43…430:

| shape | no-op | relaxed store | **coherent** | fences alone |
|---|---|---|---|---|
| same encoder | 2.698 | 2.726 (+1.0%) | **3.384 (+25.4%)** | 0.658 µs (19%) |
| **encoder per dispatch — the live gate** | 2.281 | 2.364 (+3.6%) | **3.904 (+71.2%)** | **1.540 µs (39%)** |

**Two things fall out.**

- **Item C is worth about twice what arm D priced.** Scaling the rig's ballast
  figure by the live-gate ratio `3.904/2.281 = 1.712` gives
  `3.4638 × 1.712 =` **`5.93 µs/dispatch`**, so `86 × 5.93 =` **`0.510 ms/token`**
  — 2.1% of the token, **+0.88 t/s** — against 0.298 ms priced off ballast. Margin
  over the 2.907 µs threshold goes from 1.19× to **2.04×**. Item C is comfortably
  alive, and this settles the "is 3.46 µs an upper or lower bound" question with a
  measurement rather than an argument: **lower, by 71%.**
- **NEW candidate, independent of item C: the cost is the fences, not the store.**
  A relaxed atomic store is within 3.6% of a no-op; the two system-scope seq_cst
  fences are **39% of the coherent dispatch**. Weakening them — one fence instead
  of two, or device scope instead of system scope — is worth
  `2.34 µs × 86 = 0.201 ms` = **+0.34 t/s** *without removing a single dispatch*,
  and is far cheaper to implement than item C. **Do not ship this on the number
  alone.** Those fences exist to make the TP gate flag visible across the pair;
  weakening them risks exactly the race the gate prevents. It needs a
  memory-model argument first, then the top-1 + bounded-Δlogit gate, then a
  soak. Flagged, not queued.

**2. The 0.18 ms/marker tax constant is independently corroborated — and one
1.0–1.3 ms phantom is killed.**

The harness was built to test whether `ds4_gpu_flush_commands`
(`ds4_metal.m:9813`) committing **unconditionally** — even when
`g_batch_has_work == NO` — charges a stage that encoded nothing with one command
buffer per layer. **It does not, in the busy accounting: an empty command buffer
reports 0.026 µs busy.** So there is no 1.0–1.3 ms instrument floor hiding in the
stage profile. (It does cost ~2.85 µs of *wall* per empty buffer, so anything
reading wall rather than busy would still see it.)

What the arm pairs do show is the per-command-buffer tax, which is what a stage
marker actually costs:

| work | separate cb each | batched in one cb | **per-cb busy tax** |
|---|---|---|---|
| no-op | 5.200 µs | 2.448 µs | **2.75 µs** |
| pool-shaped | 11.586 µs | 6.262 µs | **5.32 µs** |

At 43 layers that is **0.118–0.229 ms per marker**, and the standing 0.18 ms/marker
constant sits inside that range. The audit flagged the constant as soft — derived
from one 32k pair comparing a profiled stage *sum* against an unprofiled *wall*
clock. It now has independent support from a different method on different
hardware. **Keep 0.18, and keep the ±0.18 error bar.**

**Caveat on both:** M1 Max is 32 cores on one die, the rig is 60 cores over two.
Command-buffer submission overhead is driver/queue-dominated so it should transfer,
but the absolute microseconds should be re-measured on the rig before anything is
banked on them. The *ratios* — 1.712× for coherent-vs-noop, fences at 39% — are the
durable results.

### Arm B6 — the label was the bug: `reduce` was never a reduce — 2026-08-27

B5's falsifier fired exactly as pre-registered, and the answer it points to is
**not** the cross-instrument operating point I nominated. It is that the label
has never meant what every arm since B assumed.

**What B5 did settle, and it stands.** `conc mean ≈ 2.0` with `conc max = 2`, and
`union = 100%` of cb with `gap = 0.000 ms` at both contexts. Pipelining depth
really is ~2 (not a span-endpoint artefact), and **there is no idle pool between
encoders** — consistent with arm E's 100% residency at max clock. The stalls are
*inside* the encoders. That is the most useful thing this instrument has produced
and it does not depend on any label being correct.

**Why `fair` barely moved.** With `conc` identically 2, fair-share and the uniform
divide are arithmetically the *same operation* — `fair = raw/2`, `norm = raw/1.97`.
157.4 vs 159.1 was a null test by construction. B5 could not have separated cause
3; that is a flaw in B5's design, not evidence about cause 3.

**The actual defect, proven in code.** In the batch,
`ds4_gpu_end_compute_encoder` is a **no-op** — it returns when
`enc == g_batch_enc` (`ds4_metal.m:1389`, pre-fix) — and `ds4_gpu_compute_encoder`
hands back the same encoder **without reserving a new timestamp slot**. The chain:

1. Decode runs inside `begin_commands`, so `ds4_gpu_command_buffer(&owned)`
   (`ds4_metal.m:963`) returns `g_batch_cb` with `owned = 0`.
2. The four flash-attn label sites — `gather` (`:29617`), `packed` (`:29678`),
   `fa_core` (`:29698`), `reduce` (`:29726`) — sit in
   `ds4_gpu_encode_flash_attention_gathered_heads` with **no `close_batch_encoder`
   anywhere between them**.
3. So all four write `g_ts_labels[g_ts_n - 1]` — the same slot — and
   `ds4_gpu_ts_name_last` overwrote unconditionally. **`reduce`, being last, won.**

**The span reported as `reduce` is the whole batch segment**, not the reduce
dispatch. Which also explains the count that never made sense: **27/token is the
number of batch segments whose last label happened to be `reduce`**, not a number
of reduce dispatches — which is why it matched no partition of the 43 layers
(21 ratio-4 + 20 ratio-128 + 2 raw). And a segment-sized span outweighing
`attn_inv_rope` is not a paradox at all; it is the expected result.

**Everything per-encoder from arm B through B5 is retired.** The tick (1.000 ns),
`conc ≈ 2`, `union = 100%`, `gap ≈ 0` and the throughput baselines all survive —
none of them depend on a label. Every µs/call figure does not.

**Landed (`9a9f6ce`).** Labels are counted per span rather than overwritten, and
composites print as `first..last+n` with a summary line:

```
ds4: enc-ts <tag>: N of M spans carry MULTIPLE stage labels -- per-stage attribution is not available
```

Plus `DS4_METAL_GPU_ENCODER_TIMESTAMPS_SPLIT=1`, which forces a real encoder
boundary at every `end_compute_encoder` so each stage gets its own span. It
perturbs — on `bench_decode_rows` here it turns 1 span of 1.334 ms into 24 spans
over a 1.988 ms buffer (+49%) — so it is opt-in and its **absolute** numbers are
not a budget. Ending an encoder is a stronger barrier than reusing one, so split
mode can only over-serialise, never mis-order.

**What to run — two passes.**

```
# pass 1 -- honest default, tells us how bad the collapse is
DS4_METAL_GPU_ENCODER_TIMESTAMPS=1                                            # 2k + 131k, gen 128

# pass 2 -- true per-stage spans, perturbed
DS4_METAL_GPU_ENCODER_TIMESTAMPS=1 DS4_METAL_GPU_ENCODER_TIMESTAMPS_SPLIT=1   # 2k + 131k, gen 128
```

**Read in this order:**

1. **Pass 1, the composite line.** This is the falsifier for the whole diagnosis.
   If it reports **0 composite spans**, the collapse does not fire on the rig, my
   diagnosis is wrong, and B5's numbers stand as they were. If it reports a large
   fraction, every per-call figure from B onward is confirmed dead.
2. **Pass 1, the composite labels themselves.** Expect
   `gather..reduce+4` (or `fa_core..reduce+2`) on the FA spans. The `+n` says how
   many stages are fused into one measurement.
3. **Pass 2, `reduce` alone.** Now a real span. Multiply by its own count from the
   same run — *not* by 27, which was never a dispatch count. It must come in under
   `attn_inv_rope`; note pass 2 is perturbed, so treat it as a **ratio** against
   the other FA stages in the same run, not as an absolute.
4. **Pass 2, the FA split.** `gather : packed : fa_core : reduce` as shares. This
   is what the arm was originally for: which part of flash-attn to attack.
5. **Both passes, `conc`/`gap`/throughput.** Pass 1 must still read ~42 / ~29.6 t/s
   and `gap ≈ 0`. Pass 2 will read worse; record how much so we know the
   perturbation size.

**Pre-registered falsifier.** 0 composite spans in pass 1 refutes the diagnosis
outright. If instead pass 2's `reduce` still exceeds `attn_inv_rope` after being
given its own boundary, then the containment assumption itself is wrong — i.e.
this FA path is reached from somewhere outside the
`compressor_indexer`→`attn_inv_rope` bracket — and the next step is to instrument
the call site rather than the encoder.

**Zero-cost falsifier, already resolved — cause 4 confirmed.** Across all four
enc-ts arms, `reduce` is the **only** FA label ever reported: 10 mentions in the
arm-B write-ups, **zero** for `gather`, `packed` or `fa_core`. If the four phases
had their own encoders, all four labels would appear. They do not.

**Independent kill by arithmetic.** The reduce kernel's own traffic is
`tmp = nrows·head_dim·nwg·4 + nrows·2·nwg·4` = 1,579,008 B at ratio-4 (nwg 24) and
328,960 B at ratio-128 (nwg 5): `21×1,579,008 + 20×328,960 = 39,738,368 B / 41 =
969 kB/call`, which at 760 GB/s is **1.28 µs/call** against a reported 157–160.
**123× off.** No plausible inefficiency spans two orders of magnitude; the span is
not the kernel.

**Two refinements to the run matrix.**

- **Do not co-run `DS4_METAL_DECODE_STAGE_PROFILE=1`.** It ends a command buffer
  per marker, which shreds the enc-ts spans and moves the operating point to
  28.83 ms. B5's "same run" requirement is better met by deriving `attn_inv_rope`
  from **pass 2's own spans** — the sum of the FA phase spans between
  `compressor_indexer` and `attn_inv_rope` — than by stacking two instruments.
- **Bounded distortion.** Pass 2's perturbation is `172 × 3.464 µs = 0.596 ms`
  = 2.5% of the token, so it is usable as a ratio between FA phases.

**If pass 2's `reduce` reads near 1.28 µs/call**, item G's ceiling is the `fa_core`
half, and the 46–48 t/s branch becomes fundable. That is the real prize here.

### Corrections owed to `BENCHMARKS-TP-PP.md` (rig-owned — listed, not applied)

From the arms C–F audit (50 agents, 27 findings survived adversarial refutation,
31 refuted). Ordered by consequence.

| # | claim as written | correction |
|---|---|---|
| 1 | Arm C's table is a **decode** shape | **It is prefill-only.** All four `DS4_METAL_PROFILE_FLASH_ATTN_STAGE` sites are prefill encoders (`ds4_metal.m:28387/28669/29018/29272`); the print string is literally "Metal FlashAttention prefill stage" (`:11345`); and `heads=64` is structurally impossible for TP2 decode, which passes 32. `n_keys = 2048/2064/2560` is a prefill tile extent and **must not size `attn_inv_rope`.** |
| 2 | — | **Decode `n_keys` derived: 128 / 144 / 640.** `n_comp = 0/16/512` does transfer; with `n_raw = 128` fixed by `metal_graph_raw_span_for_batch` at `n_tokens=1`, window `DS4_N_SWA=128` (`ds4.c:19764-19780`). **640 lands inside the 512–700 PASS band, so item G is not killed.** |
| 3 | ratio-128 = 21 layers | **20.** `2 + 21 + 21 = 44 > 43`. Line counts confirm exactly: `2×4 + 20×6 + 21×5 = 233`, and it is **one** chunk, not two. ratio-128 gets a 6th line because pad fires (`2064 % 64 = 16`); ratio-4 does not (`2560 % 64 = 0`), `ds4_metal.m:28305-28307`. |
| 4 | `≈ 0.28 ms ≈ 3.3 µs/dispatch` | `= 0.2979 ms = **3.464 µs/dispatch**`. Threshold is `250/86 = 2.907`, so the margin is **1.19×**, not 1.14×. |
| 5 | "treat 3.3 µs as an **upper** bound" | **Backwards on its own premise.** If added dispatches hide in the ~2× overlap, the marginal price is *depressed* — 3.46 µs is a **lower** bound. |
| 6 | "Confirms the 608 MB compressor accounting" | **Confirms 440.40 of 608.17 MB (72.4%).** `DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_QUAD_FUSE` gates only the `ratio==4` branch (`ds4.c:22478`); the 20 ratio-128 layers fuse behind a *different*, unset var `..._PAIR_COMPRESSOR_FUSE` (`ds4.c:22536`) and still fired. **Do not quote 608/0.903 = 673 GB/s.** Corrected, the books are *better*: 632 GB/s out vs 609 GB/s in, 3.8% agreement. |
| 7 | `compressor_update` fused = 1.278 | **Unmeasured.** The value appears nowhere else in the corpus and Iteration 3 does not list the stage. The 0.009 ms residual is conditional on it. |
| 8 | Arm E's reason: "busy at top P-state, modest power" | **Right verdict, weak reason.** The decisive fact is that **prefill runs at 1394–1398 MHz — at or 0.3% *below* decode's pinned 1398 — while drawing 1.78× the power.** A lower-clocked phase drawing more power excludes DVFS outright. 1398 MHz is also the clock the 21 TFLOP/s roof was measured at, so "2% of peak FLOPs" is not a DVFS artefact. |
| 9 | Arm E's residency column | **Conflicts with `BENCHMARKS-TP-PP.md:1505`** ("the sampler on these boxes reports GPU *power*, not residency %"). One of the two is wrong. Attach the raw powermetrics stanza or strike one. Also record n ≈ 15 samples, one session, and that at `-i 200` one sample averages 8.24 tokens. |
| 10 | Arm F "stage deltas are the read" | Safe only on rows with the **same marker count in both arms**. `q_a_kv_proj` is 43 in both (−0.697 quotable); `compressor_proj` exists in one arm only, so its +0.903 and the +0.215 total each carry ~0.18 ms of pure instrument. |

**Latent defect, inert today.** `ds4_gpu_decode_dispatch_ballast` takes `owned`
(`ds4_metal.m:1768`) and never tests it before
`ds4_gpu_finish_command_buffer(cb, owned, …)` (`:1779`), unlike the real gate's
`if (!cb || owned) return 0;`. Harmless while decode always has `g_batch_cb` open
(`owned = 0`), but any future call outside an open batch would report a full
command-buffer round trip as a "dispatch price".

**The budget book is overdrawn.** Measured markers sum to 24.12 ms = 99.4% of the
24.26 ms token, leaving 0.14 ms — but ~19 marker sites (`ds4.c:22417-25394`) plus
the output head are unreported, and `output_norm` alone was priced at 0.52 ms.
**Deficit ≥ 0.38 ms.** Either the flat 0.18 ms/marker over-corrects, or one of
`attn_out_proj` (601 GB/s — above every Q8_0 rate on this rig) and `q_a_kv_proj`
(455 GB/s, a byte-weighted average of two different streams) is wrong by ~0.4 ms.

**New candidate.** `compressor_update` at **1.10 ms net (4.5%)** reads only 2.79 MB
(`21×1024×4×2 + 20×512×128×2`), so it is **not** bandwidth-bound and has **no
roofline defence** — the only stage in the table in that position besides
`attn_inv_rope`.

**RESULT 2026-08-27** (build `eb90b5f`, two passes): **the collapse is real and
total.** Pass 1 reports **3456 `fa_core..reduce` composite spans @2k (27/token)
and 2432 @131k (19/token)** — exactly the mysterious counts from every run since
B. `reduce` was the whole batch segment (all four FA labels wrote one slot;
`reduce`, last, overwrote). **Everything per-encoder from arm B through B5 is
retired.** Pass 2 (SPLIT) gives true spans: **reduce = 30.5 µs/call @2k →
0.66 ms/token, 69.2 µs/call @131k → 0.60 ms/token** — tiny against
`attn_inv_rope` 3.64/4.24 ms. The 2k overshoot was a composite-span artefact, not
a paradox. FA split @2k ≈ fa_core 46% : reduce 54%. Pass 2 perturbs (−23% t/s,
gap opens to 2.7-6.3 ms) as designed; conc 1.78. Full data in
`BENCHMARKS-TP-PP.md` §Arm B6.

### Arm B5 — stop dividing, decompose: the `fair` column — 2026-08-27

B4 did its job: the LOSS line went in, and the answer was **not** loss. The
`reduce` count is identical to run 3 (3456 @2k, 2432 @131k — exactly 27×128 and
19×128, so the per-token counts are real), banking did not move it, and the 2k
overshoot survived at 1.18×. Cause 3 is what is left. Two corrections to B4's
write-up before the fix:

**1. `raw` is not a usable upper bound either.** B4 concludes "treat raw as the
upper bound and stop quoting norm". Raw is 2.33× over at 2k:

| ctx | raw product | norm product | `attn_inv_rope` | container bound on `reduce` |
|---|---|---|---|---|
| 2k | 27 × 314.0 = **8.48 ms** (2.33×) | 27 × 159.5 = 4.31 ms (1.18×) | 3.64 ms | **≤ 134.8 µs/call** |
| 131k | 19 × 339.3 = 6.45 ms (1.52×) | 19 × 170.1 = 3.23 ms (0.76×) | 4.24 ms | ≤ 223.2 µs/call |

Raw is an upper bound only in the vacuous sense. The one sound statement is the
container bound: **`reduce` ≤ 134.8 µs/call at 2k**, so `norm` over-states by
≥18% and `raw` by ≥133%. Neither column is a budget.

**2. The LOSS line is not a red herring — it is the mechanism, read sideways.**
B4 dismissed loss because the `reduce` count did not move. Correct, but the loss
rate is a direct measure of *command-buffer structure*, and the structure differs
enormously between the two contexts:

| ctx | cbs/token | dropped | reported cbs/token | `reduce` per **reported** cb |
|---|---|---|---|---|
| 2k | 3.03 | 66% | 1.02 | **26.4** |
| 131k | 13.27 | 8% | 12.26 | **1.55** |

At 2k all 27 `reduce` spans land in **one** command buffer; at 131k they are
spread 1.55 per buffer across twelve. `norm` divides by a **cb-wide** factor, so
that factor lands on `reduce` in completely different ways at the two contexts —
and 2k is precisely the one that fails to reconcile. That is cause 3 with a
mechanism, and it says the fix is not a better divisor.

**The fix, landed: a `fair` column.** Sweep the interval set; wherever *k*
encoders are in flight, credit each *dt/k*. Overlap is then priced per encoder
from its own timestamps instead of smeared uniformly, and the column sums to the
**union** of the spans by construction. Two new outputs:

```
ds4: enc-ts <tag> ... raw <a> us   norm <b> us   fair <c> us   <share>%
ds4: enc-ts <tag>: union <U> ms = <P>% of cb (gap <G> ms), conc mean <M> max <K>
```

`ds4_gpu_ts_fair_share` is exposed in `ds4_gpu.h` and unit-tested against
hand-computed interval sets — `make test-ts-fairshare`, 8 cases, CPU-only, no GPU
and no model. The nested case is the one that matters: for a 100-tick encoder with
two 10-tick encoders inside it, fair bills the big one **93.3** while a uniform
divide bills it **83.3** and over-bills each small one by 2.5×. That is the exact
error shape `norm` has been making.

**`union` and `conc` are the new instruments, and they answer a bigger question
than `reduce`.** Because `spt` is calibrated so `hi−lo` maps onto the cb wall
clock, `union/cb` can never exceed 100%; what it measures is the fraction of the
buffer in which *any* encoder is running. So `gap = cb − union` is **the idle pool
inside the command buffer** — and arm E says the GPU is at 100% residency and max
clock, which predicts `gap ≈ 0`. If `gap` comes back large, arm E's residency and
this instrument disagree and one of them is wrong. If `gap ≈ 0`, the stalls are
*inside* the encoders, which points the remaining work at kernel occupancy rather
than at scheduling — and that is the single most decision-relevant number left in
the decode program.

`conc mean` is the honest version of the "~197% coverage" figure: coverage
double-counts a span whenever two overlap, `conc` says how deep the pipelining
actually runs.

**What to run.** Unchanged flags, same two contexts:

```
DS4_METAL_GPU_ENCODER_TIMESTAMPS=1     # 2k and 131k, gen 128
```

**Read in this order:**

1. **`conc mean`.** If it is ~2.0 at both contexts, pipelining depth really is 2
   and the banked "2× overlap" conclusion stands. If it is ~1.0 while coverage
   still reads ~197%, the coverage figure is an artefact of the span endpoints and
   **three banked nulls lose their explanation** (R12a's flat split schedule, the
   cheap marginal ballast dispatch, dead dispatch-count reduction) — note that arm
   D's 3.46 µs/dispatch is already in mild tension with "dispatches are free".
2. **`gap`** (per the paragraph above). Report it as a fraction of cb.
3. **`fair` for `reduce`**, × 27 @2k / × 19 @131k. It must come in under
   `attn_inv_rope` (3.64 / 4.24 ms). At 2k it must be **≤ 134.8 µs/call**.
4. **Throughput** — ~42 t/s @2k, ~29.7 @131k, else the instrument is distorting.

**Pre-registered falsifier.** If `fair × count` still exceeds `attn_inv_rope` at
2k, then non-uniform overlap was *not* the cause either, all three candidates are
dead, and the remaining suspect is the cross-instrument comparison itself — B's
token is 23.65 ms (42.29 t/s) while the stage profile that produced
`attn_inv_rope = 3.64 ms` ran at 28.83 ms gpu_busy, a 22% different operating
point. In that case the next arm is to re-measure `attn_inv_rope` and the enc-ts
`reduce` **in the same run**, and until then no per-encoder figure enters the
budget at all.

**RESULT 2026-08-27** (build `a59d3af`): **the falsifier fired.** conc mean
1.97-1.99 / max 2 (pipelining depth is really ~2 — overlap conclusion stands),
union = 100% of cb, **gap ≈ 0.000 ms** (no idle pool in the buffer; consistent
with arm E's 100% residency — stalls are inside encoders). But `fair` for
`reduce` barely moved from norm — **157.4 vs 159.1 µs/call @2k** — and the
product is still **4.25 ms > `attn_inv_rope` 3.64 ms (1.17×)**. All three
candidates (loss, slot aliasing, non-uniform overlap) are dead. Per the
falsifier, the remaining suspect is the cross-instrument comparison: B runs at
~23.7 ms/token vs the stage profile's 28.83 ms gpu_busy (~22% different
operating point). **Next: re-measure `attn_inv_rope` and enc-ts `reduce` in the
same run; no per-encoder figure enters the budget until then.** Throughput
baseline (42.15 / 29.62 t/s). Full data in `BENCHMARKS-TP-PP.md` §Arm B5.

### Arm B4 — the tick is settled, the *call count* is not; one more run — 2026-08-27

Run 3 closed the falsifier: **tick = 1.000 ns at both contexts**, so the counter
unit question is finished and the ~197% coverage is a real 2× overlap. Two things
in run 3 are still not usable, and one of them is arithmetically impossible.

**The impossibility.** `reduce` is a sub-phase *of* `attn_inv_rope` — the decode
flash-attn call sits between the `compressor_indexer` marker (`ds4.c:23330`) and
the `attn_inv_rope` marker (`ds4.c:23582`) — so it cannot cost more than the
stage that contains it. Run 3 says it does:

| ctx | calls/token | norm µs/call | product | `attn_inv_rope` | ratio |
|---|---|---|---|---|---|
| 2k | 27 | 159.7 | **4.31 ms** | 3.64 ms | **1.18× — impossible** |
| 131k | 19 | 171.3 | 3.25 ms | 4.24 ms | 0.77× — plausible |

**`calls/token` was measuring the instrument, not the graph.** It read 41 at 2k
before slot-scoping and 27 after, and slot-scoping changed only *which ranges get
reported* — never how many encoders the engine builds. Two silent-loss paths in
the old code:

1. `ds4_gpu_ts_report` returned without printing for a foreign cb, and the next
   cb to encode zeroed the range. Any buffer committed to `g_pending_cbs`
   (`ds4_metal.m:9595`, `:9618`) is never reported at all, so its whole range
   vanished with no trace.
2. Slots were reserved at **encode** time but written at **GPU execution** time,
   and committed buffers keep executing while later ones encode. A single
   512-slot arena therefore let an in-flight buffer scribble into the range being
   measured — the same class as the original global-slot bug, only narrower, and
   **context-dependent**: 2k runs many short buffers, 131k runs few long ones,
   which is the right shape to explain why only 2k fails to reconcile.

**Fixed in this build.** Ranges are banked (4 × 512 slots, `g_ts_base` advances
past a dropped range so in-flight writes cannot alias), and every report now
prints a cumulative loss line:

```
ds4: enc-ts LOSS (cumulative): N of M ranges never reported (K encoders), J encoders unsampled at the 512-slot range cap
```

**Third candidate, not yet excluded.** Normalisation divides every span by the
same coverage factor, i.e. it assumes overlap is uniform. If the big FA `reduce`
encoder pipelines with its neighbour more than the small encoders do, a uniform
divide over-states it — which would also produce a 2k overshoot. Banking and the
loss counter do not test this; the residual after them does.

**What to run.** Same two contexts, same build flags as run 3, plus nothing new:

```
DS4_METAL_GPU_ENCODER_TIMESTAMPS=1   # 2k and 131k, as run 3
```

**Read in this order:**

1. **The LOSS line.** If `N/M` is ~0, loss is not the cause and cause 3 is what
   remains. If `N/M` is ~1/3 at 2k, the 41→27 drift is explained and the real
   `calls/token` is the reported count scaled by `M/(M−N)`.
2. **Recompute the product** with the corrected count. It must come in **under**
   `attn_inv_rope` (3.64 ms @2k, 4.24 @131k). If it still exceeds it, the
   uniform-overlap assumption is broken and the normalised column is not a budget
   — in which case treat *raw* as the upper bound and stop quoting `norm`.
3. **Throughput** — must still read ~42 t/s @2k / ~29.7 @131k, else the
   instrument has started distorting.

**Pre-registered falsifier.** If the LOSS line reads ~0 dropped ranges at *both*
contexts, then loss was never the mechanism, and the 2k overshoot is either
slot aliasing (now fixed, so the product should have moved) or non-uniform
overlap (so it will not have). Whichever it is, run 4 separates them, because
banking is already in and only cause 3 can survive it.

**Until this reconciles, `norm µs/call` stays out of the token budget.** The
per-call figure may well be right; the count it gets multiplied by is not.

**RESULT 2026-08-27** (build `c9e0f72`, run 4): the 2k overshoot **survives
banking**. LOSS reads 257/388 = 66% @2k but only 8% @131k — yet `reduce`'s
count is identical to run 3 (3456 / 2432) and norm unchanged (159.5 / 170.1),
so loss drops other ranges, not `reduce`; the 41→27 drift is not a loss
artifact. Product still **4.31 ms > attn_inv_rope 3.64 ms @2k (1.18×)**;
131k reconciles (3.23 < 4.24, 0.76×). **Cause 3 (non-uniform overlap) is what
remains — norm column is not a budget; treat raw as upper bound, stop quoting
norm.** Throughput at baseline (42.29 / 29.71 t/s). Full data in
`BENCHMARKS-TP-PP.md` §Arm B4.

### Arm B — the instrument runs clean but reports wrong; fixed, needs a re-run — 2026-08-27

**What worked.** 42.08 t/s at 2k and 29.71 at 131k — **baseline throughput**, so
the 1–3% distortion claim holds and the run is undistorted. 175 encoders per
token, against 14 stage markers. The mechanism is sound.

**What did not.** The reported spans are too large. The per-report sum is
~44 ms at 2k against a 23.8 ms token, and the labelled `reduce` span reads
313 µs/call × 41 calls = 12.8 ms against `attn_inv_rope`'s 3.64 ms. **Do not use
arm B's absolute numbers.**

**Diagnosis.** `MTLCounterResultTimestamp.timestamp` is a GPU *tick*, not
guaranteed nanoseconds, and I divided by 1000 assuming it was. That assumption
validated on the M1 Max — re-checked just now, it infers **exactly 1.000 ns/tick
with 93–96% coverage and no overlap** — so it is plausibly a different timebase
on M2 Ultra. The other candidate is genuine span overlap, where the GPU
pipelines one encoder's setup against the previous encoder's drain.

**Fix: stop assuming, calibrate.** Each report now derives seconds-per-tick from
the command buffer's own `GPUStartTime`/`GPUEndTime` — documented seconds — and
prints, on the same line: total span, the cb span, **coverage %**, an
`OVERLAPPING` marker past 105%, and the **inferred ns/tick**. So the next run
distinguishes the two causes by itself:

- `tick ≈ 0.54 ns` → it was a unit difference and the calibrated figures are now
  correct outright.
- `tick ≈ 1.0 ns` with coverage > 100% → real overlap, and per-encoder figures
  are **upper bounds**.

Putting the cb span on the same line also fixes an interpretation trap: a report
fires **per command buffer**, and decode issues 3 per token, so comparing a
per-report sum against a per-*token* wall clock is not a like-for-like
comparison. The line is now self-contained.

**Two limits worth stating.** With a single encoder the calibration is
degenerate — coverage is forced to 100% and tells you nothing. And because the
tick scale is derived from `hi − lo` across the encoders, any command-buffer
overhead outside them inflates the inferred tick slightly, so coverage is a mild
over-estimate.

**Re-run arm B** on a build carrying this fix before any of its numbers enter the
budget. Everything else in Phase 0 is unaffected.

### Iteration 3 results — crash fix holds, baseline restored, R12a dead — 2026-08-27

Build `06c2c6b`, i.e. **before** the encoder-timestamp work, so three of the
seven Phase 0 arms ran and the rest still need a build ≥ `225c884`.

**Arm A — the TP crash fix holds. PASS.** Prefilled 4095 fresh (493.64 t/s),
resumed 4095→4099 so the first chunk was exactly **one token** — the precise
trigger. **Zero** "not covered by mapped model views", no resumed-prefill
failure, no GPU timeout, no transport failure. `69a3b86` is validated against
the shape that produced the original crash. **Close this.**

**Arm B — clean re-baseline, and both hand-corrections were close.**

| | predicted | measured | error |
|---|---|---|---|
| `attn_out_proj` | 2.73 | **2.867** | 5% |
| straggler @2k | 0.50 | **0.545** | 8% |

So the C2-contamination analysis was sound: iteration 2's 2.38 was the broken
build's artefact, and differencing against the clean control recovered the right
answer to within 5%. **U14 / §7C stand at ~0.52–0.55 ms.**

Net of the ~18% profiler tax, the 2k budget now reads:

| stage | net ms | share of the 24.44 ms token |
|---|---|---|
| `routed_moe_folded` | 4.72 | 19.3% |
| `attn_inv_rope` | 3.64 | **14.9%** |
| `attn_out_proj` | 2.69 | 11.0% |
| `q_a_kv_proj` | 1.96 | 8.0% |
| `q_path` | 1.69 | 6.9% |
| `q_lora_norm` | 1.49 | 6.1% |
| `ffn_tp_gate` | 1.38 | 5.6% |
| `attn_tp_gate` | 0.76 | 3.1% |
| `compressor_indexer` | 0.02 | 0.1% |
| **sum** | **18.33** | **75.0%** |

**Treat these as the last numbers from the old instrument.** Arm B of the new
Phase 0 list re-takes all of it at 1–3% distortion instead of 18%, and the
standing gate applies: where the two disagree by more than 0.3 ms, believe the
new one.

**Arm G — R12a is a flat null and the avenue is dead.** Six split schedules at
131k: 29.31 / 29.21 / 29.24 / 29.38 / 29.19 / 29.12 t/s — **all within 0.9%**.

**And that is worth more than the null.** If command-buffer boundaries carried
real cost, moving from 4/0 to 2/32 would have shown it. They do not. **This is a
second, independent refutation of the round-trip framing I retracted an hour
ago** — the ballast said the marginal dispatch is cheap, the `kv_n` sweep said
the kernel is 80% work, and now the split schedule says command-buffer count
does not matter either. Three instruments, one answer. **Remove R12a from the
queue and stop proposing command-buffer or dispatch-count reductions.**

**Still to run, on a build ≥ `225c884`:** the encoder-timestamp re-baseline, the
flash-attn shape fields, the ballast arm, `powermetrics`, and the
compressor-accounting arm.

### Next run — iteration 4 — 2026-08-27

**Read `docs/PATH-TO-50TPS.md` first.** Its verdict is that 50 t/s is **not
reachable** with anything identified — believed 43–45, with 46–48 contingent on
one unsized item. Discounted budget across seven surviving candidates is
**0.90 ms against 4.34 needed.** It also corrected two of my figures: the byte
model is **5.93 GB/rank/token, not 4.29**, so the floor is **13.2 ms and decode
is 1.85× off it, not 2.6×**.

**Rebuild first — the build must carry `d50bbaa` or later.** Arm B's tick
calibration landed there, and without it that arm reports unusable numbers.

**What iteration 3 settled, so it is not re-run:** the TP crash fix holds
(arm A, PASS against the exact one-token-chunk trigger); the baseline is
restored and both hand-corrections landed within 5–8% (`attn_out_proj` 2.867,
straggler 0.545); and R12a is a flat null across six split schedules inside
0.9%, which — with the ballast and the `kv_n` sweep — is the third independent
refutation of the round-trip framing. **Command-buffer and dispatch-count
reduction should not be proposed again.**

#### Rig — **five arms left**, all zero-code, one session

Of the original seven: **A passed and is closed**, **G is a flat null and the
avenue is dead**, and **B must be re-run** — it ran, but on a build predating
the tick calibration, so its numbers cannot be used. Build must carry
`d50bbaa` or later.

| # | arm | decides | outcome |
|---|---|---|---|
| **A** | **Validate the TP crash fix.** Prefill **4095 tokens**, checkpoint, continue. Watch the worker for "not covered by mapped model views". | Gates a shipped fix (`69a3b86`). Single-box variant in `docs/BUG-TP-WORKER-MODEL-VIEW-2026-08-27.md`. | **PASS 2026-08-27** (build `06c2c6b`). Prefill 4095 fresh, resume 4095→4099 (`--step-incr 4`, first chunk = 1 token, `to_boundary=1`): **zero** "not covered" / "resumed prefill failed" / GPU-timeout lines. Fix holds. |
| **B** | **`DS4_METAL_GPU_ENCODER_TIMESTAMPS=1`** at 2k and 131k. **Build ≥ `05f402d`.** | **Still the one that matters.** Re-baselines the budget at 1–3% distortion instead of 18%, in ~175 spans instead of 14. **Gate: where it disagrees with the hand-corrected table by >0.3 ms on any stage, believe it and re-rank before writing code.** | **RE-RUN DONE 2026-08-27** (build `05f402d`, slot-scoped + normalised). **Falsifier resolved: tick pins at 1.000 ns both contexts** (counter is 1.0 ns on both parts; the 0.63 ns was slot-reuse corruption). Coverage ~197% OVERLAPPING is now a genuine measurement — encoder spans really overlap ~2×, which independently explains the R12a/ballast/dispatch-count nulls. **Normalised `reduce` (the budget): 159.7 µs/call @2k, 171.3 @131k** (raw 314/339). Throughput baseline (42.25 / 29.75 t/s). See `BENCHMARKS-TP-PP.md` §Arm B (re-run 3). |
| **C** | **`DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1`** at 2k — **read only the shape fields** | `n_keys` and `n_comp` per layer, the missing input to sizing the `attn_inv_rope` item. Its *timings* are still host round trips; ignore them. | **DONE 2026-08-27** (build `05f402d`). Shapes at 2k: ratio-0 → comp=0/keys=2048 (~2 layers); ratio-128 → comp=16/keys=2064 (21); ratio-4 → comp=512/keys=2560 (21). heads=64 dim=512 window=128. |
| **D** | **`DS4_METAL_DISPATCH_BALLAST` ∈ {0,2}** at 2k | The dispatch price **in the live graph**. Settles an 11.6× spread that four candidates live or die on — and my retracted 22 µs claim came from getting this wrong in isolation. | **DONE 2026-08-27** (build `05f402d`). bal0 41.22 / bal2 40.72 t/s — Δ0.50 t/s for +86 no-ops ≈ **3.3 µs/dispatch** (above the 2.9 µs kill threshold, so item C not dropped by this arm alone; treat as upper bound given the ~2× overlap). |
| **E** | **`powermetrics --samplers gpu_power`** during decode vs `tests/bench_membw` | Retires the "30 W / 20% utilised" premise either way. The fan-out believes it dissolves. | **DONE 2026-08-27**. Decode **~33.4 W @ 1398 MHz max clock, 100% residency** — busy at top P-state, not under-clocked/idle. Prefill ~59-60 W, idle ~0.08 W. bench_membw 38 W active-burst / 66 W peak, DVFS ~950-1398. **Premise retired; DVFS is not the decode gap.** |
| **F** | **`DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_QUAD_FUSE=1`** + stage profiler | Confirms the 608 MB compressor accounting the corrected byte model rests on. | **DONE 2026-08-27** (build `05f402d`). **Confirms it.** Unfusing drops q_a_kv_proj 2.138→**1.441 ms** (Δ0.697) and a **compressor_proj row appears at 0.903 ms**. Compressor reads land where the byte model put them. |
| ~~G~~ | ~~R12a command-buffer split schedule~~ — **done 2026-08-27: flat null, all six arms within 0.9%. Avenue dead.** — arms 4/0, 2/8, 2/16, 2/32, 3/12, 4/12. **`DS4_METAL_FAST_SYNC=1` on both ranks** or every arm collapses to one buffer and reads as a false null. | Bit-identical, free, and the only queued item that attacks command-buffer round-trip cost. | **FLAT NULL 2026-08-27** (build `06c2c6b`, 131k, `DS4_METAL_FAST_SYNC=1`). All arms within 0.9%: 4/0 29.31, 2/8 29.21, 2/16 29.24, 2/32 29.38, 3/12 29.19, 4/12 29.12 t/s. Split schedule makes no measurable difference at 131k. |

**Also recorded 2026-08-27** (clean re-baseline on `06c2c6b`, pre-encoder-timestamp build): `attn_out_proj` = **2.867 ms** ctx-invariant (confirming ~2.73, not the C2-broken 2.38); `attn_tp_gate` 0.935 ms; total gpu_busy 28.83 / 38.37 ms; straggler **~0.52-0.55 ms/token** (gate delta +12.2-12.7 µs). Full data in `BENCHMARKS-TP-PP.md` §Iteration 3. Note: arms B/C/D/E/F supersede/re-baseline these figures with the low-distortion encoder-timestamp instrument, so treat the above as the last pre-instrument numbers. |

#### Built for this run

- **`DS4_METAL_GPU_ENCODER_TIMESTAMPS`** (`df0037e`) — GPU timestamp counters at
  encoder boundaries inside one command buffer. Probed first: `atStageBoundary`
  supported, `atDispatchBoundary` not. Validated against known workloads (4×
  work → 3.82×, 2× → 1.92×, encoder sum within 4% of the cb span).
- **A0 rewired onto it** (this commit) — the four decode flash-attn phases
  (`gather`, `packed`, `fa_core`, `reduce`) now just *name* the encoder they
  already close, instead of ending the command buffer and bracketing it with a
  host clock. **This is what finally splits `attn_inv_rope`**, the only stage the
  fan-out found off its roof and the only place 4.34 ms could come from. The
  sampling knob added earlier is deleted — with no cb splits there is nothing to
  sample around.
  *Caveat:* the timestamp path is gated on `!g_batch_encoder_concurrent`, so this
  instrument and any concurrent-encoder work are mutually blind.
- **`make check-dispatch-count` is green again** — two sites I left unwrapped in
  the U9 top-k are fixed.

#### Not runnable — needs code, not a rig slot

- **n-gram / speculation.** Structurally excluded under TP2: the speculative
  entry bails on `s->distributed` (`ds4.c:69931`), which every TP coordinator
  session has. Days of work, and `speed-bench/tp_mtp_hunt.md` must be read first.

#### Live implementation queue

**U16 is reopened** — its kill reason was my measurement error; the kernel is
~80% work at the production shape, so optimising it is the right move and grid
widening is back on the table *for that kernel*. Then C3 row-split `shared_down`
(0.16 ms), U14/§7C shared-shift (**0.50 ms, measured**), A1 reduce split
(0.1–0.3 ms). **Hold all of it until arm B lands** — it re-baselines everything
these are ranked against.

### Sequencing — run in this order

Ordered by (information gained) / (rig time), not by expected gain. The first
two are minutes and one of them can invalidate three completed runs.

| # | Do | Time | Why first |
|---|---|---|---|
| 0 | **`sysctl iogpu.wired_limit_mb` on both hosts** — **done 2026-08-26**: it **was** `0` on both hosts; all 27 surviving R10/R11 coordinator logs carry the `wired_limit_mb is 0` warning, so **R10d/R10e/R11 decode numbers are suspect** (flat, stall-shaped, neither-bound profile = lazy paging). Set to `120000` on both; prerequisite added to `BENCHMARKS-TP-PP.md`. | — | If it was 0 after the 2026-08-26 reboot, the shard was paging lazily and **R10d, R10e and R11 are all suspect**. Set it to `120000` on both and note it in the prerequisites (done — `BENCHMARKS-TP-PP.md`). |
| 0b | Confirm `DS4_METAL_FAST_SYNC=1` is in the **`ds4-server`** launch path — **resolved 2026-08-26: nothing to confirm**. There is no `ds4-server` launch path in the repo and no production `ds4-server` deployment on the bench hosts; the knob is bench-only until a server path exists | — | Bench always sets it; production may not. Worth ~186 µs of a 508 µs gate, and without it the decode command-buffer split is a no-op under TP. |
| **0c** | ~~M0 — re-baseline on a pinned shard~~ — **done 2026-08-26. Decode survives within 1%; the gate exchange halved at 131k (49.7 → 24.8 µs); prefill moved +6–25% but the cause is confounded. T1 re-sizes down to ~+2%.** | — | **Step 0 came back positive: the limit was `0` on both hosts.** Everything measured on 2026-08-26 was on a lazily-paged shard. This re-run decides how much of R10c/R10e/R11 survives and re-sizes T1. Nothing else should run first. |
| 1 | **M3** — **done 2026-08-26** (`uc_lat2`, byte-verified, n=2000/arm): half-RTT **8.0 µs (4 KB)** / **14.5–15.5 µs p50 (16 KB, single WR)** — both ≪20 µs → **T1 open** (~30 µs/gate ≈ 2.5 ms/token at 131k). Single 16 KB UC WR confirmed working on this stack. One transient first-ping UC drop seen; T1 needs a re-arm/retry path. Recorded in `BENCHMARKS-TP-PP.md`. | — | Decides T1, the largest sized item, before any code is written. ≲20 µs half-RTT → ~2.5 ms/token available; ~45 µs → T1 closes. |
| 2 | ~~Env battery T2 + T3 + T4~~ — **done 2026-08-26. T2 +0.9% @131k and monotonic to the top of the tested range (not peaked); T3 mixed; T4 default already optimal.** Follow-up: sweep 28/31 — **done 2026-08-26, peaked at 24 (28/31 regress); default set to 24, T2 closed.** | — | One small real win, two nulls. |
| 3 | ~~T8 pricing~~ — **done 2026-08-26. The specialisation ladder is worth nothing to −5%; the generic path is faster. T8 is dead.** The routed MoE matvec is bandwidth-bound at ~400 GB/s. | — | 30 minutes of pricing saved five kernel patches for a negative. |
| 4 | ~~M2~~ — **done 2026-08-26.** Ablation at 32k/65k/131k. **Routed MoE is the dominant decode stage (~22–25%)**; **the indexer is the long-context story** (score +5.9→12.2%, topk +7.9→17.7% as ctx 32k→131k — the largest attributed slice of the 11.1 ms); attnout ~9–10%, qb/attncore ~5%, hcpre ~2%. **Refined by the stage profile (done below): the indexer cost is `compressor_indexer` (+5.06 of +5.93 ms growth), and the ~13 ms residual is real compute, not stall.** | ~2 h | The 11.1 ms/token of unattributed long-context decode growth — the largest unknown in the document. |
| 4b | ~~Stage profile at 32k/131k~~ — **done 2026-08-26.** Decode gpu_busy 30.43/36.36 ms, **gap ~0.31 ms (1%) — decode is ~99% GPU-busy, no stall.** Stage sum = busy exactly (100% attribution). compressor_indexer = the long-context term (+5.06 ms). **The ~13 ms M2 residual is real compute (unablated stages), not idle.** | ~20 min | Bounds router/shared/kv/compressor and decomposes the ~13 ms floor without ablation. |
| 5 | ~~R12a~~ split-schedule sweep + **R12b** reduced ballast arm + **encoder-boundary instrument** — **R12b and encoder-boundary MOOT** (probe stall; stage profile shows no stall). R12a split-schedule still valid if wanted. | ~1.5 h | R12b is now a confirmation, not a discovery; the encoder-boundary half is the genuinely unmeasured one. |
| 6 | ~~R13~~ — **merged into U5 (row 13)**, same arms. | — | Deduplicated. |
| 7 | ~~T1~~ — **done 2026-08-26: dead.** `DS4_TP_GATE_FASTPATH` is a wash (±0.6% decode, no gate-exchange change) and is **not bit-identical** (logits shift up to 2.3, top-1 preserved). Stay default-off. ~~T8 port~~ — dead, see row 3. | — | Both are code, both are gated on a measurement above. |
| 8 | Cleanup batch: T13, and the unreachable `llt16`/`llt32` instantiations (**T9 → U3/U10; T6 → U9; T10 → U13** — all sized before the stage profile) | — | Small, low-risk, individually sub-1%. |
| ~~9~~ | ~~U1~~ — **done 2026-08-26; verdict superseded by U6.** The ~408–410 GB/s plateau is the MoE matvec's access pattern, not the platform. | — | Reversed by U6. |
| ~~10~~ | ~~U2~~ — **done 2026-08-26; measured the *direct* fallback, not production's LLT kernel** (see U7b), so its roofline does not describe the shipped path. | — | Right method, wrong kernel. |
| **11** | ~~U3~~ — **folded into U10.** U2 gated it off as "latency-bound, halving bytes will not pay" — correct on bandwidth, but U2 measured the *direct* fallback, not production's LLT kernel, and the real argument for F16 was never bandwidth. It is that `sk` is 80% of the threadgroup-memory budget. See U10. | — | Killed for the right reason on the wrong kernel; returns with a different mechanism. |
| **12** | **U4 — TP row-split the decode indexer** — **UN-DEFERRED 2026-08-27: U9 failed, so its prize is intact** | ~3 h | `compressor_indexer` is **9.67 ms at 131k**, the largest stage, and *both halves* (score and top-k) are computed identically on both ranks — neither `ds4_gpu_indexer_score_one_tensor` nor `ds4_gpu_indexer_topk_tensor` takes a rank or world argument. Row-split with local top-k and a merge is exact. **~4 ms net at 131k (11.7%); ~0 at 2k.** Largest single long-context item left. |
| **13** | **U5 — R13 n-gram arms, re-prioritised** | ~1 h | Raises arithmetic intensity without requiring any kernel to get faster — the structural answer to a latency-bound decode. |
| ~~14~~ | ~~U6~~ — **done 2026-08-27. Roof is ~760 GB/s (94–95% of spec), all seven arms within 1%.** Allocation path and concurrency cost nothing; the loader is exonerated. **Reverses U1, voids T8's ceiling claim.** | — | Every stage had been scored against an assumed roof. |
| ~~15~~ | ~~U7 — indexer LLT scoring on the rig~~ — **done 2026-08-27. 1565 GFLOP/s, 72% core scaling, 7.3% of ALU peak.** U2's 1015 was the *direct* fallback, so the non-scaling worry was an artefact. Deficit is latency hiding: one threadgroup resident per core. | — | Sized the largest stage against the right kernel for the first time. |
| ~~16~~ | ~~U8 — MoE block granularity~~ — **done 2026-08-27. 17-byte stride costs 5.9%, not 46%.** `blk-16` hits 758 GB/s ≈ roof, so it is not load granularity in any form. **Dequant/accumulate are what remain.** | — | Eliminated one of three suspects for the matvec's 54%. |
| ~~17~~ | ~~U10a~~ — **done 2026-08-27. NSG=4 beats the default +5–13% on the Ultra; NSG=2 −12%.** Residency beats NK there — U10 confirmed ON. | — | The M1 Max ranking did not transfer, which was the question. |
| ~~18~~ | ~~U9~~ — **built and measured 2026-08-27: negative, default off.** 0.77× on the M1 Max. The premise was wrong — the argsort fallback is *already* a 32-block hierarchy, so the parallelism U9 added already existed. Exact vs CPU ground truth; the argsort path is the one that deviates under ties. | — | Cost a day; caught by interleaved A/B before it shipped. |
| ~~17~~ | ~~U10a~~ — **done 2026-08-27. NSG=4 beats the default by +5–13% on the Ultra; NSG=2 −12%. Residency beats NK there — U10 is ON.** | — | The M1 Max ranking did not transfer, which was the question. |
| ~~19~~ | ~~U10~~ — **done and productionised 2026-08-27. +17.4% on the rig, bit-identical, default-on**, prefill neutral (U11). Delivered ~+1.6% end-to-end. | — | Banked. |
| ~~20~~ | ~~U11 — confirm U10 does not regress prefill~~ — **done 2026-08-27: neutral.** Prefill 393.03 vs 393.37 t/s @131k; first-token 36.5 vs 35.5 ms. No nsg4-style spike. **U10 stays default-on.** | — | One arm, closed. |
| ~~21~~ | ~~U12 — price `q_path` (5.47 ms) and `attn_inv_rope` (4.27 ms)~~ — **done 2026-08-27.** q_path 5.474/5.472 ms/token (ctx-invariant, 15%); attn_inv_rope 3.389→4.258 ms (grows +26%, 11.7%). 26.8% of token priced. | — | 26.8% of the token, now attributed. |
| ~~22~~ | ~~U13 — arm the inverse-RoPE fuse on the indexed branch (was T10)~~ — **sizing invalidated by U12 correction 1; U12b arm 1 DONE 2026-08-27: standalone RoPE is +0.133 ms @32k / +0.512 ms @131k — U13's prize is ~0.5 ms at 131k, marginal.** | ~1 d | The 4.27 ms stage is mostly attention; the RoPE tail is ~0.5 ms at 131k. |
| ~~23~~ | ~~U14 — MoE expert straggler~~ — **deprioritised 2026-08-27.** Design C is a repacked TP-specific model file for 3.5%; B blocked; A costs 27 GiB. Recorded that whole-expert reassignment provably cannot help (variance, not mean), and that §7 missed a cheaper option — shifting the shared expert to the routed-light rank, 2.3% for no memory and no repack. | — | Poor return against U12/U13; kept so it is not re-derived. |

Steps 0–3 are about four hours of rig time and settle whether the last three
campaigns are valid, whether the largest sized item is real, and whether the
MoE port is worth writing.

**M0 does not gate everything.** Two queued items are **model-free harnesses**
that never load the shard, so the wired limit is irrelevant to them and they
can run immediately, on the dev box, in parallel with the rig work:
`tests/bench_indexer_score` (T3 pre-screen, CPU-reference checked) and
`tests/bench_moe_mxfp4_decode 256` (T8 pricing — though *that* one the audit
says to run on the rig, since the M1 Max mispredicts MoE shapes). Everything
that loads the model waits for M0, because M0 is not merely a re-validation:
it is the **new baseline every subsequent A/B compares against**.

### R9 — the `nqptg = 8` ceiling (scoping, no code yet)

R8 took the attention kernel from ~2.4 to ~3.3 TFLOP/s on the rig. `q_path` on
the same GPU does **14.7**, so the kernel is still ~4.4× off a plain GEMM and
remains the largest stage in prefill. What is left is structural, and R8's own
shared-memory arithmetic identifies it exactly.

The kernel processes **8 query rows per KV pass**, which holds arithmetic
intensity near **8 FLOP/byte** — far below the ~134 FLOP/byte ridge where this
GPU becomes compute-bound. It cannot go higher because shared memory is already
28,672 B of a 32,768 B budget, and `nqptg=16` would need 57,344 B and fails to
launch (verified). The budget breaks down as:

| region | contents | bytes at Q=8 |
|---|---|---|
| `so` | **fp32 output accumulator**, `Q*PV*4` | **16,384** |
| `sq` | Q tile, `Q*DK*2` | 8,192 |
| `ss` | scores, `Q*SH*4` | 4,096 |

`so` alone is more than half. Moving it into registers — each of the 4
simdgroups owning `DV/4 = 128` accumulator lanes — would leave ~12 KB and admit
`nqptg=16` or `24`, doubling or tripling KV reuse per pass.

There is a second, related prize the same change unlocks: this model is
**MQA — `n_head_kv = 1`**, so all 64 query heads share one K/V. The dispatch is
currently `(query_block, head)`, so the same K/V is re-read **64 times**, which
is where the ~82 GB of KV traffic per attention call comes from. Handling
several heads per threadgroup would cut that proportionally, and it needs the
same shared-memory headroom.

Both are real kernel work, not a knob. Before committing to it, worth deciding
whether a prototype is warranted given the ~0.63 transfer factor: a standalone
M1 Max prototype can establish whether the restructure works and roughly how
much intensity it buys, but its speedup number should be discounted before it
justifies anything. **No code yet — flagging the target and the cost.**

Cheaper alternatives already ruled out: `C` tuning (C=128 slower on both NSG
values *and* skips less work — 1280 vs 1216 executed keys/row; C=32 will not
build), and the CPU/ANE offload evaluated below.

### R13 — decode audit findings (verified against source)

From a code audit of decode fast paths that are off for our configuration.
Every claim below I re-checked at the anchor; three further audit items did
**not** survive verification and are recorded at the end so they are not
re-raised. A fuller ranked list is still outstanding.

#### R13a — `DS4_METAL_FAST_SYNC` is default-off and gates the TP decode split — **ops check first**

`g_tp_fast_sync = getenv("DS4_METAL_FAST_SYNC") != NULL` (`ds4_metal.m:10263`)
— **default off, and nothing in the tree sets it**. A repo-wide search finds it
only in docs (`BENCHMARKS-TP-PP.md:83`, `README.md:622`,
`QA_BEFORE_RELEASES.md:275`) and one read in `ds4_distributed.c:793`. It is not
in any `ds4-server` launch path.

Two things depend on it under TP:

1. **The fast release fence.** Its own comment: *"Resuming the command
   processor from `g_tp_cpu_event` costs ~186 µs of a 508 µs gate"*
   (`ds4_metal.m:9948`). At 86 gates/token that is up to **~16 ms/token**.
   Against the measured 24.5 ms token at 2048 that is a ~40% decode loss —
   an upper bound, since not every gate need serialise, but R11 established
   these gates are close to serial.
2. **The decode command-buffer split.** `ds4_gpu_tp_split_safe()` returns 0
   without it (`ds4_metal.m:10369`), and the split site *"falls closed to one
   command buffer per TP token"* (`ds4.c:28425`).

Corroboration that the bench arms *do* have it: our measured row-gate wait is
292.6 µs, consistent with 508 − 186 ≈ 322 and not with 508. So **no published
number is affected** — the bench protocol always sets it. The question is only
whether production does.

**Action: check the `ds4-server` launch command on both ranks.** If it is
missing, that is the largest single decode number in this document and it costs
nothing to fix. Consider whether it should be a default rather than an env var
the operator has to remember — the reason it is opt-in ("relies on an
undocumented Metal qualifier, so fall back to the shared event if anything is
unavailable") is a *capability* argument, and the capability is already probed
separately via `g_tp_flag_gates` and `g_tp_release_words`.

**This also constrains R12a.** The split schedule sweep is meaningless without
`DS4_METAL_FAST_SYNC=1` on both ranks: `tp_split_safe()` would return 0 and
every arm would collapse to one command buffer, producing a flat null result
for the wrong reason. Set it explicitly in every R12a arm.

#### R13b — MXFP4 fixed-route MoE decode is disabled under TP — **corrected**

**What I got wrong.** I wrote that the `tp_world == 1 && tp_expert_base == 0`
clauses at `ds4_metal.m:39387`/`:39399` are what disable the fixed-route path.
They are not the binding constraint. Under TP decode `tp_fold_ffn` is true
(`ds4.c:24961`) so `add_in = metal_graph_shared_out(g)` (`ds4.c:25302`), and a
non-NULL `add_in` disqualifies `use_mxfp4_moe_decode_tg_multiple`
(`ds4_metal.m:39360`) — on which every other specialisation is layered. All
five fall out *before* the `tp_world` clauses are evaluated.

**And we are not falling off a cliff.** What runs is the generic nsg=1
pair-SwiGLU (`:39443`) and the nsg=1 sum6 swapped to the down-r4 twin
(`:39497`), so we keep the two structurally biggest wins — the single-simdgroup
decode mapping and down-r4 (worth +0.74 t/s per `32ef898`) — and the 2-dispatch
fused shape. A hard guard at `ds4_metal.m:39544` makes TP *fail loudly* rather
than degrade to a per-expert scalar loop.

What we lose is four layers of compile-time specialisation on top:
`tg_multiple`, `fixed_route_pair`/`_sum6` (bake `nei0=6` and the 0..255 expert
range as constants), `sum6_full_rows`, and `static_trip_*` (compile-time K-walk
trip counts).

**The sizing reference in the tree is invalid, in our favour.**
`tp_decode_investigation.md:289-296` reports routed MoE at 367 GB/s vs ~400
isolated — "92%", *"the kernels are fine"*. The same document invalidates that
at `:49-54`: the isolated harness is **world 1 with `add_in == NULL`**, i.e. it
measures the *fully specialised* kernel while production TP runs the *generic*
one. The 8% gap is against the wrong reference and the real TP ceiling has
never been measured. Likewise `:428` records `32ef898`'s tg_multiple
TP-exclusion removal as *"+0.019 ms, nothing"* — necessarily inert, since
`add_in` blocks `tg_multiple` regardless of `tp_world`. Do not read either as
evidence the specialisations are worthless.

**Measure before writing anything.** `tests/bench_moe_mxfp4_decode` with
`n_total_expert = 256` (the default 128 fails the shape check), toggling each
`DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_{TG_MULTIPLE,FIXED_ROUTE_PAIR,
FIXED_ROUTE_SUM6,SUM6_FULL_ROWS,STATIC_TRIP}`. **On the rig, not the M1 Max dev
box.** That prices each layer at the live shape before any kernel work.

**If it prices well**, the port is the generic sibling's two lines
(`ds4_tp_owns_expert` test + `- tp_expert_base` rebase, `metal/moe.metal:4578-
4584`) across five kernels (`:4593, 4734, 6478, 6523, 6616`), plus handling
`add_in` so `tg_multiple` is reachable under the fold, plus dropping the host
clauses. Hazard: the `nr0`/grid-width mismatch behind the `d9eba30` revert
(`ds4_metal.m:39135-39148`) silently halves the grid and leaves the upper half
of the down projection unwritten. Also fix the stale comment at `:39485-39487`,
which claims `tg_multiple` needs `g_tp_split_world == 1`; its predicate has no
such check.

#### R13c — one more device-name gate the earlier sweep missed

`ds4_metal.m:28922`:
```c
const bool use_shared_kvpad = has_kvpad &&
    ds4_gpu_device_name_contains("M3") &&
    getenv("DS4_METAL_DISABLE_SHARED_KV_PAD") == NULL;
```
A bare device-*name* test with no `pre_m5 || m5` family arm, on the decode
gathered-FlashAttention path — the 20 ratio-128 layers. This is exactly the
defect class `speed-bench/tp_decode_investigation.md:574-584` documents
(*"a device-name string test where the surrounding code used a device-family
predicate"*), of which eight were already fixed; this one was missed. It only
fires when `n_keys % 32 != 0`, so the hit rate is context-dependent and the
yield is probably small. Widen following `ds4_metal.m:20915-20922`.

#### Did not survive verification — do not re-raise

- **`hc_rms_scale_project`'s `n_rows > 8u` gate** (`ds4_metal.m:21770`) — not
  on the decode path; all three callers are prefill/batch.
- **cluster2 HC norm-mix M5 gate** (`ds4_metal.m:43592`) — guards a fallback we
  do not take; the fusion we do take already selects the cluster2 pre-norm
  kernel unconditionally.
- **`vec_hc` M5 gate** (`ds4_metal.m:44656`) — only reachable through the
  `fuse_attn_out_hc` branch, which is TP-disabled.

Also recorded: there is no `#if 0` anywhere in the decode sources, and
`DS4_METAL_NORM_RSQRT_DISABLE` being default-on is a deliberate determinism
requirement (`metal/norm.metal:231`), **not** a missed optimisation — do not
flip it for speed.

### M0 — re-baseline on a pinned shard — **DONE 2026-08-26**

`iogpu.wired_limit_mb` was `0` on both hosts for the whole R10/R11 era. M0
re-ran the decode baseline with it at `120000`. Full tables in
`BENCHMARKS-TP-PP.md`.

**What survived, and what did not.**

| | verdict |
|---|---|
| R10a sub-gate wash, R10b link ceiling, R10c stall-bound, R10d NWG plateau, R11 negative | **all survive** — decode moved ≤1.1%, and R11's A/B was in-session anyway |
| R10e gate *exchange* numbers | **wrong** — 49.7 → **24.8 µs** at 131k (−50%), 34.8 → 23.6 at 2k |
| The R9-vs-R10e "discordant" wire spread | **explained**: it was the wired limit. R9's 29.9 µs was near-pinned; R10e's 49.7 was paged |
| T1's sizing | **halved and then some** — see below |

So the suspicion was right about the gates and wrong about decode throughput.
Lazy paging cost the *exchange* half its speed and cost decode t/s almost
nothing — worth remembering as a shape: a misconfiguration can be invisible in
the headline metric and dominant in a component of it.

**T1 re-sized down to ~+2%.** At 24.8 µs the exchange sits within ~9–10 µs of
M3's probe one-way (14.5–15.5 µs, same 16 KB single WR) rather than 3–4× above
it. Upper bound is now ~9 µs/gate × 86 = **0.8 ms/token ≈ +2% at 131k**, and
that assumes the whole delta to a ping-pong RTT is recoverable, which it is
not. `DS4_TP_GATE_FASTPATH` is built and default-off, so it is still a cheap
A/B — but it is no longer the largest item and should not be scheduled as if it
were.

**Correcting my own consequences table.** I wrote that prefill was "unaffected —
prefill's 86–91% residency is not a paging signature." Prefill moved **+6% to
+25%** (131k sweep 367.29 → 393.53, cold 402.64 → 435.67). The residency
argument was wrong: prefill touches nearly every expert per 4096-token chunk
where decode touches six per token, so per-command-buffer residency validation
scales with prefill's working set and barely with decode's. That mechanism fits
the split perfectly — prefill ~+10%, decode ~+1%.

**But the prefill delta is confounded and must not be quoted.** M0 also switched
the prompt from the `/tmp` word-soup to `speed-bench/promessi_sposi.txt`, which
the R12 note warned has a different token/byte ratio and different routing
entropy. Pinning is the more likely driver by the mechanism above, but nothing
separates them. Consequences:

- **The prefill arc ends at 367.29.** Do not extend it with 393.53; the two
  numbers are not the same experiment. M0 opens a new baseline series.
- A same-prompt control (word-soup on the pinned shard) would separate them in
  one arm, if anyone wants the attribution. It is not needed for anything
  currently queued.
- All prefill A/Bs in R5–R8 remain valid — both arms shared prompt and shard
  state within each comparison.

**Where the decode critical path actually is.** At 131k a row gate waits
**375 µs** to exchange **24.8 µs** — wire is 6–7% of gate time. The write-up
reads that as "the rest is waiting on local GPU completion", which is true but
close to the tautology this doc keeps tripping on: the wait *contains* the
compute, so 86 × 375 µs ≈ the token by construction.

The informative content is what is now excluded. Link: 6–7%, closed. Gate
count: closed by R11. Dispatch: bounded at ~6% by
`tp_decode_investigation.md`'s own 1.9 µs. What remains inside the 375 µs is
**compute**, which points the queue at the things that make compute cheaper —
the T2/T3/T4 env battery, T8's MoE specialisations, T9 — and at **M2**, which
is the only item that can say where the 11.1 ms of long-context growth goes.

### Env battery + T8 pricing — **DONE 2026-08-26**

Full tables in `BENCHMARKS-TP-PP.md`. Four results, one of which closes a
workstream and one of which narrows M2 substantially.

**T8 — dead, and the pricing step is why we know cheaply.** The five MXFP4
routed-MoE decode specialisations are worth **nothing to −5%**: the
fully-generic arm runs 0.1988 ms/iter against the full ladder's 0.2092
(~403 vs ~383 GB/s), non-overlapping across three runs each. The audit ranked
T8 as the largest byte-mover with unknown upside and promoted it to step 3 on
the strength of a two-line-per-kernel fix. Thirty minutes of pricing saved
writing five kernel patches for a regression. **This is the model for every
future "cheap kernel patch, unknown upside" item.**

**And the reason it is dead narrows M2.** The routed MoE matvec is
**bandwidth-bound at ~400 GB/s** — roughly half of the M2 Ultra's 800 and good
for a GEMV. Specialisation cannot buy what is not there. Since routed MoE is
~1.7 GB/token/node of the ~3.3 GB total, that is ~4.3 ms of a 35.5 ms token,
**near its ceiling and therefore not where the deficit lives.** ~31 ms of the
token is something else, and M2's unattributed 11.1 ms is inside it.

One caveat on how far to carry that: the harness is world-1, no gates, tight
loop, so ~400 GB/s is an *isolated-kernel* ceiling. It shows the kernel is
near-optimal; whether the *stage* achieves it in production is what
`DS4_TP_ABLATE=moe` in M2 would show. (Note the R13b caveat inverts here —
the harness's world-1 arm measures the generic path, which is what production
actually runs, so for once the isolated number is the relevant one.)

**T2 — the one real win, and it is not finished.** Decode split-K is
**monotonic from 12 → 24**: +0.9% @131k, +1.3% @32k, across six arms. The
default (12) is not optimal. Monotonicity across the whole range is the
evidence here, not any single delta.

Note the occupancy story does not explain it. Under TP each rank has 32 heads
in groups of 8 = 4 groups, so `splits=15` already gives 60 threadgroups on 60
cores; gains continuing to `splits=24` (96 threadgroups) means we want
**oversubscription for latency hiding**, not exact fill. **Follow-up: sweep 28
and 31** (the code allows 2..31). If it is still climbing, the default should
move and the comment at `ds4_metal.m:29988` — which reasons from exact fill —
needs rewriting.

**T3 — mixed, close it.** LLT `nsg4` helps 32k steady (+1.4%) and hurts
65k/131k (−1.7%/−0.6%), with first-token latency spiking to 189 ms @32k versus
31 ms for the control. Not a win at the contexts we care about.

**T4 — already optimal, monotonically.** `DS4_METAL_Q8_MV_NSG`'s TP-specific
default of 2 is the best value: n=1 −0.9%, n=3 −1.6%, n=4 −3.0%, n=6 −4.1%.
The `parallel_full_ffn` confound was isolated properly (t4_2 with the env set
vs control: −0.1%).

**Scoreboard.** Every *knob* has now returned null or ≤1% (NWG, Q8_MV_NSG, MoE
specialisations, sub-gate, LLT nsg4), and every *structural* change has
returned negative (gate count via R11, replication). The two survivors are T2
at ~+1% and T1 at ~+2%. That is a strong signal that the remaining decode
headroom is not in the knobs, and it makes **M2 — attributing the 11.1 ms —
the only item left with real upside.**

### R14 — ranked decode targets from the audit

Full list with anchors is in the audit; this records the ordering, the two
items that outrank everything I had queued, and one correction to my own
reasoning. **Sequencing: M1a/M1b first (minutes, and M1a can invalidate two
campaigns), then the env battery, then the T8 pricing, then the instruments.**

#### M1a — check `iogpu.wired_limit_mb` on both hosts — **do this before anything else**

`ds4.c:59886` — if `glm_graph_wired_limit_bytes()` returns 0, TP **skips the
residency set entirely** and the 76.7 GiB shard pages lazily. The sysctl is
runtime-only and does not survive a reboot. `README.md:704` documents it;
**`BENCHMARKS-TP-PP.md`'s prerequisites do not list it**, though they do list
the RDMA setup as reboot-sensitive.

The R10 ops notes record lanfear being **rebooted mid-campaign on 2026-08-26**,
`/tmp` wiped, TB IPs re-applied — and nothing told anyone to re-apply the wired
limit. A lazily-paging shard produces exactly the flat, stall-shaped,
neither-bandwidth-nor-compute-bound decode profile we have been chasing.

**If it was 0, R10d, R10e and R11 are all suspect.** Check
`sysctl iogpu.wired_limit_mb` on both hosts and grep every saved bench log for
`wired_limit_mb is 0`. Then add it to the prerequisites section.

#### T1 — row-gate exchange **latency** — the largest sized item, and I dismissed it

**My error.** I concluded "the link is not decode's problem" from wire being
~12% of gate time. That is the ratio-vs-total trap this doc has now flagged
five times. In absolute terms: **86 × 49.7 µs = 4.27 ms of a 35.55 ms token —
12% of the token**, strictly serial, with the local GPU spinning in
`kernel_dsv4_tp_fence_wait` throughout.

And it is not the fabric. The payload is a fixed 16,384 B (one WR, no
chunking), which at R10b's measured 4.4 GB/s is **3.7 µs of wire**. The
exchange takes 49.7. So **~46 µs/gate is software**, and the same measurement
moved 29.9 → 49.7 µs between R9 and R10e for identical bytes — scheduling, not
transport. R10b measured *streamed bandwidth* and correctly closed that
question; it never addressed one 16 KB request/response.

Three ordered changes in `tp_rdma_gate_exchange()` (`ds4_tp.c:1011-1073`):

1. **Hoist the receive re-arm.** `tp_rdma_post_gate_recv()` sits at `:1070`,
   *after* the wait loop — one more verb call between "peer data arrived" and
   "GPU unblocked". Verified. ~0.1–0.25 ms/token.
2. **Stop signalling every send.** `IBV_SEND_SIGNALED` on every send (`:1045`)
   makes `tp_rdma_drain_cq` chew a send CQE before reaching each recv CQE. The
   send completion is not needed for correctness; signal every Nth. This is
   `tp_decode_investigation.md:139-143`'s own open item #3, never executed.
   Needs care with `send_outstanding` or the send queue overflows.
3. **Service-thread affinity** — it tight-spins (`1<<20` spins before
   `sched_yield`) on the same P-cluster as the Metal encode thread.

**Gate it on M3 first:** `uc_pingpong` at 4 KB and 16 KB. Half-RTT ≲20 µs means
~30 µs/gate is recoverable — **2.5 ms/token, +8% at 131k**. ~45 µs means T1 is
closed and we have documented a hard floor. Minutes to find out.

**M3 + M0 result: T1 is open but small — ~+2%, not the 9% first sized.**
M3 measured hardware half-RTT at 8.0 µs (4 KB) / 14.5–15.5 µs p50 (16 KB, the
gate shape), byte-verified ×2000 per arm. M0 then showed the 49.7 µs exchange
that motivated T1 was a lazy-paging artifact: on a pinned shard it is **24.8
µs**, within ~9–10 µs of the probe. Upper bound **0.8 ms/token ≈ +2% at 131k**,
and that assumes the full delta to a ping-pong RTT is recoverable, which it is
not.

`DS4_TP_GATE_FASTPATH` (`177b50a`) implements both changes and is default-off,
so it costs one A/B arm to settle. No retry path: UC on this stack is treated
as lossless. A single 16,384 B UC SEND WR needs no chunking.

#### T2 follow-up — proceed, but size it honestly — **DONE 2026-08-26: peaked at 24, T2 closed**

**Verdict: run it. It is one rig session for two arms, and it is the last
unfinished item from the env battery.** But M2 caps what it can be worth, and
this document should say so rather than leaving T2 described as "the one real
win."

`DS4_METAL_DECODE_SPLITS` repartitions the online-softmax reduce inside the
attention decode kernels — i.e. it lives in `attncore` + `attnout`, which M2
measures at **1.91 + 2.92 = 4.83 ms of the 35.35 ms token at 131k**. A ~1% total
gain is consistent with a ~7% improvement *within* those stages, a reasonable
return for a constant; but T2's ceiling is 4.83 ms however the sweep comes out,
and realistically a fraction of it. **It is a ~1% item.**

**State the acceptance bar explicitly, because T1 just exposed an
inconsistency.** T2 is *not* bit-exact — it repartitions a reduction, exactly
the class of change that killed T1. But T1 was rejected for being a wash **and**
perturbing logits (up to 2.3, top-1 preserved). The bar that actually applies
is **a measurable win, top-1 preserved, and bounded Δlogit**; T1 failed on the
win, not on the bit-exactness, so T2 at +1% can pass the bar T1 failed. Judge it
that way, and record Δlogit alongside t/s so the two are comparable.

**Two outcomes.**
- Still climbing at 28/31 → move the default and rewrite the comment at
  `ds4_metal.m:29988`, which reasons from exact fill; gains past 60
  threadgroups on 60 cores mean oversubscription for latency hiding, not
  occupancy.
- **Flat or regressing → set the default to the measured peak and close
  T2 — the branch taken (below).**

**Outcome — done 2026-08-26: peaked at 24; set the default to 24; close
T2.** The follow-up sweep at 28/31 regressed at 131k (28.48 and 28.45 vs
24's 28.68); the trend is no longer monotonic — it turns over at 28:

| splits | 32k | 65k | 131k |
|---|---|---|---|
| 12 (control) | 33.79 | 31.71 | 28.43 |
| **24 (peak)** | 34.24 | 31.99 | **28.68** |
| 28 | 34.19 | 31.99 | 28.48 |
| 31 | 34.24 | 31.94 | 28.45 |

Per the decision rule above: set the default to the measured peak (24) and
close T2; do not move to 28/31. Recorded in `BENCHMARKS-TP-PP.md` (T2
follow-up section).

**Interaction to keep in view:** with `DS4_NGRAM_SPEC` on, verify steps set
`decode_splits = 1` (`ds4_metal.m:30001`) and T2 is inert on those steps. If
n-gram ever defaults on, T2's value shrinks by the acceptance rate. Measure with
n-gram off as before, but do not productionise the two independently without
re-checking the combination.

#### The env battery — one rig session, `DS4_NGRAM_SPEC` off

**T2 — `DS4_METAL_DECODE_SPLITS`.** Its own comment names our rig: *"12 was
chosen on unsplit hardware... Under tensor parallelism each rank holds 32
heads, so the grid halves to 4 × 12 = 48 — under one per core on a 60-core M2
Ultra... occupancy-bound rather than bandwidth-bound"* (`ds4_metal.m:29988`).
The only sweep of it ran at **ctx 512, where the flag is inert** — the doc
retracts its own negative at `:24-31`. Sweep {8, 12, 15, 16, 20, 24} at 32768
and 131072. Not bit-exact (repartitions the online-softmax reduce); judge on
top-1 + bounded Δlogit.

**T3 — `DS4_METAL_INDEXER_LLT_NSG4=1`.** Off because upstream measured it
*"prefill-negative/decode-slightly-positive"* — but `4333606` wires LLT into
**decode only** on this branch, so the prefill-negative half cannot fire.
Structurally the same shape as R8's `nsg=8`. Pre-screen free with
`tests/bench_indexer_score` (model-free, CPU-reference checked). While there:
the LLT scorer's own **+6–7% at 117k is unverified on this rig**; control is
`DS4_METAL_DISABLE_INDEXER_LLT=1`.

**T4 — `DS4_METAL_Q8_MV_NSG`.** `ds4_metal.m:5272` gives TP **nsg=2** and
world-1 **nsg=4**, with no comment explaining why. TP-*aware* but never swept.
Carries `q_a`, `kv`, `q_b`, shared gate/up/down, attention-output low. Sweep
{1,2,3,4,6}; set `DS4_BENCH_MAP_GB=16` or the harness under-reports ~18%.

**Interaction to respect:** with `DS4_NGRAM_SPEC` on, verify steps run
`n_tokens > 1`, which sets `decode_splits = 1` (`ds4_metal.m:30001`) and makes
**T2 inert on those steps**. Measure the battery with n-gram off.

#### M2 — attribute the 11.1 ms — the largest unknown — **DONE 2026-08-26**

Every decode stage number in this tree is from **ctx 512, where the indexer
path is entirely inactive**. That token is 24.06 ms; ours at 131k is 35.55.
**11.1 ms/token — 31% of the long-context token — has never been attributed**,
and code reading accounts for maybe 3–4 ms of it.

**Outcome.** Full tables in `BENCHMARKS-TP-PP.md`. The routed MoE is the
dominant decode stage (~22–25%, larger than T8's isolated ~12% because
production carries the shard exchange). The indexer is the long-context
story — score +5.9→12.2%, topk +7.9→17.7% as ctx goes 32k→131k, making the
indexer the largest attributed slice of the 11.1 ms (caveat: ablating it also
removes its contribution to MoE top-6 selection, so `indexer` and `moe` are
not disjoint). attnout ~9–10%, qb/attncore ~5%, hcpre ~2%. No chain is free.

**Converted to milliseconds, M2 says something the percentages hide.** A t/s
gain of X from removing a chain is `t_ctrl · X/(1+X)` of in-situ time, so:

| chain | 32k (29.48 ms token) | 131k (35.35 ms token) | growth |
|---|---|---|---|
| routed MoE | 5.90 | 6.38 | +0.48 |
| indexer topk | 2.16 | 5.32 | **+3.16** |
| indexer score | 1.64 | 3.84 | **+2.20** |
| attnout | 2.80 | 2.92 | +0.12 |
| attncore | 1.27 | 1.91 | +0.64 |
| qb | 1.54 | 1.78 | +0.24 |
| hcpre | 0.75 | 0.76 | +0.01 |
| **attributed** | **16.05** | **22.90** | +6.85 |
| **residual** | **13.43 (46%)** | **12.45 (35%)** | **−0.98** |

Two findings, and the second is the one that matters.

**1. The context term is the indexer, essentially in full.** Measured growth
32k→131k is +5.87 ms; the two indexer chains alone supply +5.36 ms of it. The
other five chains together contribute +1.49 ms, and the sum over-counts by
0.98 ms — which is exactly the indexer/MoE overlap the caveat predicts. So the
11.1 ms long-context penalty is an indexer story with a small attention tail,
and nothing else in the token grows meaningfully with context.

**2. There is a ~13 ms floor that no ablated chain touches, and it does not
move with context.** The residual is 13.43 ms at 32k and 12.45 ms at 131k —
flat, and the apparent shrink is just the double-count. Against the 24.34 ms
token at 2k it is ~55% of the token; at 131k it is still 35%. **It is the
largest single item in the decode budget, larger than the routed MoE, and
stage ablation cannot see it by construction** — it is not in any stage.

**What the residual is not.** It is unattributed, not known-to-be-overhead —
an earlier draft of this section overstated it as "not compute at all," which
is wrong. M2 ablated seven chains; the decode token contains several more that
the battery deliberately skipped because they do not ablate cleanly: **`router`**
(unusable — ran 0.574 ms *slower* while removing 92 dispatches), **`shared`**
and **`kv`** (fusion rollbacks that *add* dispatches), plus the q_a/kv
projections, the compressor update, HC post-combine, and per-token embedding /
final norm / logits / sampling. The shared expert and router in particular are
real, irreducible work. So the honest statement is: **~13 ms of the token is
flat in context and unmeasured**, and its split between genuine unablated
compute and true stall is the open question — not a foregone conclusion.

What does survive is the negative pattern: T1 (wire latency, wash), T8 (MoE
kernel specialisation, negative) and R11 (gate count, negative) all attacked
chains *inside* the attributed 16–23 ms, and none of them moved the needle.

**Per-dispatch cost cannot be the floor — our own ballast numbers rule it
out.** The marginal dispatch cost measured in-situ is **~1.9 µs**, and decode
issues **1021 dispatches/token**, so the entire dispatch budget is **1.94 ms**
(`:781`). To reach 13 ms the marginal cost would have to be 12.7 µs — **6.7×**
what we measured. Whatever the recoverable part of the residual is, dispatch
count accounts for at most 1.94 ms of it, which is consistent with this
document already concluding that "dispatch removal is not a productive
strategy here."

That left two candidates — unablated compute, or encoder boundaries and gate
waits. **The stage profile has since resolved it: unablated compute, with a
0.31 ms gap and no stall.** The reasoning below is kept because the conclusion
it reached — that dispatch removal cannot be the lever — was right, and it is
now measured rather than inferred.

**Decomposing the compute half does not need ablation, and that is the
unlock.** The
three chains M2 skipped are precisely the ones that cannot be ablated cleanly —
but they are all already instrumented in the decode stage profiler
(`DS4_METAL_PROFILE_DECODE_STAGE("router")` at `ds4.c:24123`, and the same for
`shared`, `compressor_proj`, `compressor_update`, `compressor_commit`). A
single `DS4_METAL_DECODE_STAGE_PROFILE` run at 32k and 131k reports their cost
directly, with no semantically-wrong output and no dispatch-count confound.
**That is a ~20-minute measurement and it converts the largest unknown in this
document into a table.** It should run before anything else.

**Consequence for sequencing.** The remaining stage-level items are worth ~1%
between them, and the floor is ~13 ms. Order:

1. **Stage-profile run at 32k/131k** (above) — **DONE 2026-08-26. The floor is
   real compute, not stall.** `DS4_METAL_GPU_STAGE_TIMESTAMPS` reports decode
   gpu_busy 30.43 ms @32k / 36.36 @131k with a **gap of only ~0.31 ms at both**
   — the decode GPU is ~99% busy, and the stage sum equals busy exactly, so
   there is no hidden idle time. The 13 ms M2 residual is the sum of stages
   the ablation set did not cover (q_path remainder, compressor_indexer,
   attn_inv_rope, router, shared, ffn/attn hc), all real compute.
   **compressor_indexer is the long-context term**: +5.06 of the +5.93 ms
   growth 32k→131k. (Note: a one-line instrument fix was needed — the decode
   stage report was only wired on the speculative `_top` decode variant, not
   the main TP decode loop; added in `28ecec4`.)
2. **R12b reduced-ballast arm + the encoder-boundary instrument** — **MOOT.
   They probe stall, and the stage profile shows there is no stall (gap
   ~0.31 ms, 1%).** Do not run. The floor is compute; per-layer dispatch
   cannot be the 13 ms.
3. **T2 follow-up** (28/31) — see below. Cheap, but a loose end, not a lever.
   Still the only remaining queued measurement.

### U-series — requests — 2026-08-27

Five items from the throughput reopening. **U1 first**: it decides how to read
the other four, and it is fifteen minutes.

#### U1 — is ~400 GB/s the platform or our kernels? — **DONE 2026-08-26; verdict SUPERSEDED by U6 2026-08-27**

**Outcome.** Pinned at ~408–410 GB/s across maps 3.19–25.5 GiB. Initially read
as one M2 Max die / platform; **U6 reversed this** — the part streams at ~760
GB/s on the same `mmap` path, so U1's plateau is the MoE matvec kernel's
access pattern (~54% of achievable), not the platform. Kernel headroom
re-opens; the loader is exonerated.

**Where:** the rig (M2 Ultra, 800 GB/s spec). **Not** the M1 Max — its peak
*is* 400 GB/s, so it cannot discriminate. One rank, no TP, no model, matching
T8's conditions exactly.

**Method.** Push the working set well past any cache with the existing
model-free harness and watch whether achieved bandwidth ever exceeds 400 GB/s:

```
tests/bench_moe_mxfp4_decode 256 256 6     # T8's original point, the ~400 GB/s reading
tests/bench_moe_mxfp4_decode 512 512 6
tests/bench_moe_mxfp4_decode 512 512 12    # more distinct experts streamed per iter
tests/bench_moe_mxfp4_decode 1024 1024 12
```
Report MB/iter and GB/s for each. Three runs per arm; these are small and
noisy.

**Decision.**
- **Any arm > ~450 GB/s** → 400 was a working-set or kernel artifact, the
  "near the M2 Ultra ceiling" claim at `BENCHMARKS-TP-PP.md:226` is retracted,
  and matvec tuning re-opens across the board (including, separately from its
  dead specialisation ladder, the MoE stage at 5.4 ms).
- **Nothing exceeds ~410 GB/s at any size** → we are pinned at exactly one
  M2 Max die's worth of bandwidth. That points at placement/interleave across
  UltraFusion rather than at any kernel, which would be a **platform-wide ~2×**
  and by far the largest thing in this document. Escalate; do not tune kernels
  until it is understood.

**Note for the benchmark doc (resolved by U6 2026-08-27):** the T8 sentence
at `BENCHMARKS-TP-PP.md` ("bandwidth-bound at ~400 GB/s — near the M2 Ultra
ceiling") has been corrected — the Ultra ceiling is 800 GB/s and the matvec
is not bandwidth-bound at all (~54% of the achievable streaming rate).

#### U2 — indexer-score roofline and working-set sweep — **gates U3 — DONE 2026-08-26**

**Outcome.** Latency-bound: 39.7 GB/s at 65536 = ~10% of the 400 GB/s
platform (one threadgroup per row, per-thread row count 1), GPU-busy
~linear in `n_comp`. **U3 will not pay** — halving bytes of a kernel at
~10% of bandwidth buys little time; go to the restructure instead.
Correctness gate: worst rel 7.6e-3 at row 17391 (same row at 32768 and
65536) — expected FP32 tree-vs-sequential reduction tolerance, benign for
ranking.

**Where:** model-free, so the dev box works, but **also run it on the rig** —
the ratio between the two is itself informative and the 0.63 transfer factor
applies to any projection from the M1 Max.

**Baseline already taken (M1 Max, 2026-08-27):** `bench_indexer_score 32768`
gives 460 GFLOP/s (**4.4%** of ~10.4 TFLOP/s) and 14.4 GB/s of K-cache traffic
(**3.6%** of 400 GB/s), GPU-busy 1.01 ms/dispatch → 21.2 ms across 21 layers,
about 2× the rig's measured 10.5 ms. Three kernel arms
(`DS4_METAL_DISABLE_INDEXER_LLT=1`, `DS4_METAL_INDEXER_LLT_NSG4=1`, default)
landed within 0.7% of each other.

**Method.** Sweep the working set and watch how time scales:

```
for n in 4096 8192 16384 32768 65536 ; do
    DS4_METAL_GPU_BUSY_PROFILE=1 tests/bench_indexer_score $n 300
done
```

**Decision — this is the fork that matters.**
- **GPU-busy roughly linear in `n_comp`, GB/s flat and low** → genuinely
  latency-bound per byte, and **U3 will not pay**: halving the bytes of a
  kernel that is at 4% of bandwidth buys little time. Go instead to a
  restructure (below).
- **GPU-busy sublinear / GB/s rising with `n_comp`** → the small-working-set
  arms are overhead-dominated and the kernel does stream at larger sizes.
  **U3 then pays roughly its byte reduction**, and is the cheap win.
- **Flat GPU-busy across sizes** → fixed per-dispatch cost dominates and the
  answer is neither U3 nor a restructure but batching layers per dispatch.

**The restructure to price if the first branch holds.** 64 heads score against
one K-cache: **16.78 MB/dispatch with perfect reuse, 1.07 GB without.** The
distance between those is the whole optimisation. Report, for the current
kernel, threadgroup count, threads/threadgroup and per-thread row count so we
can see how far it is from the ~2,060-threadgroup configuration ds4 uses for a
comparable Q4_K shape (`ds4_metal.m:19628`).

**Also resolve, while here:** the harness fails its own correctness gate —
`8459/32768 bit-exact, 24127 within 1e-5, worst rel 7.567e-03`, over its 1e-3
threshold, printing "kernel looks wrong". Probably benign for a ranking use,
but it must not stay unexplained on the stage we are about to spend three days
optimising. Say whether it is an expected FP32-vs-reference tolerance or a
real defect.

#### U3 — T9 re-sized: indexer compressed cache F32 → F16 — **gated on U2 — GATED OFF by U2 2026-08-26**

**Outcome.** U2 reported latency-bound, so per its own gate U3's honest
prize is near zero — do not start the F32→F16 code work; go to the
restructure instead.

**Why it moved tier.** The cache is allocated F32 at `ds4.c:17373`
(`layer_comp_cap[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float)`): at 131k that
is 32768 × 128 × 4 B = **16.78 MB/layer**, **352 MB/token** across the 21
ratio-4 layers. T9 halves it. The plan filed this at "~0.2–0.4 ms" in the
sub-1% bucket — a figure written before the stage profile priced the stage at
**10.5 ms**.

**Do not start until U2 reports.** If the kernel is latency-bound rather than
bandwidth-bound, halving the bytes does not halve the time, and the honest
prize could be near zero. U2 is thirty minutes and decides whether this is a
~2–4 ms item or a null.

**Correctness.** The e2m1 losslessness argument in the original T9 note still
needs stating explicitly against the *decode* path, not just prefill. Gate on
top-1 preserved plus bounded Δlogit, per the T2 bar. Memory: ~176 MB freed.

#### U4 — TP row-split the decode indexer — the largest single item here

**The finding.** `metal_graph_tp_split_indexer()` is gated on
`DS4_TP_PREFILL_SPLIT_INDEXER` and is **prefill-only** (`ds4.c:29056-29062`).
The decode call site passes the full `g->layer_n_index_comp[il]` with **no rank
and no world argument** (`ds4.c:23241`). Both ranks therefore compute the
identical full indexer score every decode token, on the single largest stage in
the profile (10.5 ms at 131k).

**Design.** Split the compressed-row range by rank; each rank scores its half
and takes a **local top-k**; exchange the two candidate lists; merge to the
global top-k.

- **Exactness:** top-k of the union of two local top-ks *is* the global top-k,
  provided each side contributes k candidates. This is exact, not approximate.
- **Tie-breaking is the one correctness risk.** Equal scores must break
  deterministically and identically on both ranks — break by global row index,
  not by local position, or the two ranks can select different key sets and
  diverge. Call this out in the implementation and test it directly with a
  synthetic tied-score case.
- **Wire cost is negligible:** k = 512 candidates × (score + index) = ~4 KB per
  layer, ~86 KB/token across 21 layers, ~20 µs at the measured 4.4 GB/s link
  ceiling — against a ~5 ms saving.
- **Latency is the real cost, not bytes.** 21 additional exchanges per token at
  ~25 µs each is ~0.5 ms if they serialise. **Ride the existing per-layer gate
  exchange rather than adding a new one**; if that is not possible, report the
  serialised cost so the net is honest.

**Prize:** ~5 ms of a 36 ms token, ~14%. Report t/s at 2k/32k/131k against the
M0 baseline, top-1 preservation over ≥7 steps, and the measured exchange cost
separately from the stage saving.

**Prerequisite:** U1, so we know whether the halved per-rank work actually
converts to time or just moves us along a flat part of the curve.

#### U6 — why ~400 GB/s on an 800 GB/s part — **DONE 2026-08-27: the roof is ~760 GB/s, not 400**

**Outcome.** `bench_membw` on mat (M2 Ultra): all seven arms land at
**752–762 GB/s — 94–95% of the 800 GB/s spec**, within 1% of each other, at
both 16 and 32 GiB per arm. `mmap-file` (ds4's own path, pointed at the real
GGUF) is 756.7–759.9 — identical to `metal-private` (757.4–761.7) and
`two-queue` (752–755 aggregate). **Allocation path costs nothing; concurrency
costs nothing.** A plain grid-stride kernel saturates *both* dies on ds4's own
`mmap` path. **U1's ~408–410 GB/s is therefore the MoE matvec kernel — ~54%
of achievable bandwidth — not the platform.** The placement hypothesis and
both rivals are falsified together; the loader is exonerated and the headroom
is in the matvec's access pattern (MXFP4 17-byte blocks, gather/scatter,
dequant interleave). See `BENCHMARKS-TP-PP.md` §U6.

**The die-placement hypothesis was mine, and it is dead.** I argued that
file-backed `mmap` pages wrapped with `newBufferWithBytesNoCopy`
(`ds4.c:2544`, `ds4_metal.m:2023`) were landing on one die, that all 60 cores
were therefore contending for one controller's 400 GB/s, and that this
"predicts exactly the number we measure." It predicted a number that does not
exist. `mmap-file` is 756.7–759.9 GB/s — indistinguishable from
`metal-private`. Both rival explanations died with it: concurrency costs
nothing, and there is no architectural ceiling at 400.

What survives is the narrower half of the argument, which was the useful half:
**U1 established "not working-set size," not "not our kernels."** All four of
its arms shared one kernel and one allocation path. That gap was real, and
closing it is what produced the roof.

The M1 Max pre-screen is what makes this decisive rather than arguable. It
established the probe saturates a single die, so 760 on the Ultra cannot be
explained away as a weak kernel, and the hypothesis dies outright.


**The instrument now exists: `tests/bench_membw`.** `bench_moe_mxfp4_decode` is
a matvec, not a bandwidth probe, and only exercises the mmap path. The new
harness is self-contained (Metal + Foundation only, no `ds4_metal.o`) and runs
one grid-stride `uint4` streaming kernel — no dependent chains, 2M threads,
with a host-supplied `store` flag the compiler cannot fold so the loads cannot
be sunk — across seven allocation arms of identical size:

| arm | allocation | tests |
|---|---|---|
| `metal-shared` | `newBufferWithLength:` + `StorageModeShared` | Metal-placed, CPU-visible |
| `mmap-anon` | `NoCopy` over `MAP_PRIVATE\|MAP_ANON` | file-backed vs anonymous |
| `mmap-file` | `NoCopy` over `MAP_SHARED` file mmap | **the current model path** |
| `mmap-untracked` | as above + `HazardTrackingModeUntracked` | the `DS4_METAL_MODEL_UNTRACKED` knob |
| `mmap-parallel` | as `mmap-file`, faulted in by 16 threads | **first-touch placement — the cheap fix** |
| `metal-private` | `newBufferWithLength:` + `StorageModePrivate` | Metal-placed, GPU-local |
| `two-queue` | 2× Private on two queues, overlapped | per-dispatch concurrency limit |

```
make tests/bench_membw
BENCH_MEMBW_FILE=/path/to/model.gguf ./tests/bench_membw 16 20
```

`BENCH_MEMBW_FILE` points the file arms at a real file — **use the GGUF on the
rig**, which makes `mmap-file` literally the engine's own load path. Arms are
freed as they finish so peak footprint is ~2× the arm size; 16 GiB/arm wants
~32 GiB free.

**Pre-screened on the M1 Max (single 400 GB/s die), 2026-08-27:** every arm
lands at **357–365 GB/s — 89–91% of that part's spec**, and all seven are
within 2% of each other. Two things follow, and both are what make the rig run
decisive. The kernel is a *valid* probe: it saturates a single-die part, so a
low number on the Ultra cannot be blamed on a weak kernel. And allocation path
costs nothing on one die — no paging or hazard-tracking penalty — so if the
arms *do* separate on the Ultra, die locality is the only thing left that
distinguishes them.

**Expect on the rig:** the same kernel at 89–91% of an 800 GB/s part would be
~710–730 GB/s. If `mmap-file` instead lands near the 408–410 GB/s U1 measured,
that is one die exactly, and the spread across the other six arms says which
fix applies.

**Decision.**
- **D (or F) ≫ A** → placement or concurrency, and it is *ours to fix*. At
  ~1.7–2× on every weight read this would be worth more than every other item
  in this document combined, and it would re-open T8 and the whole matvec
  queue. The fix would be a loader change — stage weights into Metal-allocated
  memory, or fault the mapping in from several threads so first-touch spreads
  the pages — not a kernel change.
- **All arms ~410** → genuine architectural ceiling for GPU reads on this part.
  Then 400 GB/s is the real roof, U2's 39.7 GB/s is 10% of a *correct* roof,
  and the kernel-level work stands as planned.

**Actual outcome (2026-08-27): neither branch fired — the third case.** All
arms landed at **752–762 GB/s (94–95% of 800), within 1%**, so neither
placement nor concurrency nor an architectural ceiling explains U1's ~410.
The part saturates both dies on ds4's own `mmap` path; the ~410 plateau is
the **MoE matvec kernel** at ~54% of achievable bandwidth. Matvec access-
pattern tuning re-opens (loader exonerated); U1's "escalate, do not tune
kernels" is retracted.

**Run on the rig only.** The M1 Max is a single 400 GB/s die and cannot
discriminate any of this. One useful pre-screen *is* available on the dev box
though: if A and D differ materially even on a single-die part, the penalty is
paging or hazard tracking rather than die locality, which is worth knowing
before the rig time is spent.

**Sequencing: this goes ahead of U3, U4 and U5.** Every one of them is sized in
milliseconds saved against an assumed bandwidth roof, and U6 decides whether
that roof is 400 or 800.

#### Re-scoring the decode queue against a measured 760 GB/s roof

Every stage in this document was being scored against an *assumed* roof. It is
now measured, and it is 760 — not the 400 U1 inferred and T8 asserted.

| stage (131k) | time | achieved | of 760 roof |
|---|---|---|---|
| `compressor_indexer` | 10.51 ms | ~40 GB/s (U2) | **~5%** |
| `routed_moe_folded` | 5.40 ms | ~410 GB/s (U1) | **~54%** |
| `q_path` | 5.47 ms | not measured | — |
| `attn_inv_rope` | 4.27 ms | not measured | — |
| streaming probe | — | 760 GB/s | 95% |

**The indexer is ~16× off the roof and it is the largest stage.** At
`n_comp=32768` the score kernel moves 16.78 MB in 0.352 ms; at the measured
streaming rate that transfer is ~0.022 ms. U2 already named the cause — **one
threadgroup per compressed row, per-thread row count 1** — so each threadgroup
loads 512 B, runs a full `simd_sum` tree, and exits, reloading the query for
every row. This is not a byte-count problem, which is why killing U3 was
right; it is a work-per-threadgroup problem.

**Two stages remain unmeasured** (`q_path` 5.47 ms, `attn_inv_rope` 4.27 ms).
Together they are 9.7 ms, and nobody has priced their achieved bandwidth. Fold
that into U7's sweep if it is cheap.

#### U7 — the indexer is compute-bound, not bandwidth-bound — **rewritten 2026-08-27, my first version was wrong**

**Retracting the premise.** I wrote U7 as "one threadgroup per compressed row —
batch rows to fix it," taking U2's kernel-config note at face value. Both are
wrong, and I should have read the dispatch before writing a request on it.

- The decode path is **not** the per-row kernel. `n_head == 64 && head_dim ==
  128` (`ds4_metal.m:18214`) selects `kernel_dsv4_indexer_scores_llt`
  (`<NBPTG=8, T_NSG=8>`), dispatching `(n_comp + 63)/64` threadgroups of 256
  threads — **64 keys per threadgroup**, not one. `DS4_N_INDEXER_HEAD` is a
  model constant and is **not** TP-split, so both ranks take this path.
  `kernel_dsv4_indexer_score_one_direct` is the fallback, which U2 appears to
  have described instead; its "(32,4)" is a threadgroup shape, not a head count.
- **The row batching I was going to propose is already there.** The kernel
  stages Q in 8-head tiles and keeps each K row resident across all 64 heads.

**And comparing it to a 760 GB/s streaming roof was the wrong roof.** With K
reused across 64 heads the kernel does 64 × 128 × 2 = 16,384 FLOP per 512-byte
key — **32 FLOP/byte by construction.** The harness's own two columns confirm
the reuse is real, at every size:

| n_comp | GFLOP/s | GB/s | ratio |
|---|---|---|---|
| 16384 | 731 | 22.9 | **31.9** |
| 32768 | 1015 | 31.7 | **32.0** |
| 65536 | 1271 | 39.7 | **32.0** |

A kernel at 32 FLOP/byte should never approach streaming bandwidth; 40 GB/s is
what 1271 GFLOP/s *looks* like here, not an independent symptom. **The
indexer's "~5% of roof" was an artefact of measuring an arithmetic-dense kernel
against a bandwidth roof.**

**The real question, and it is still a big one.** Against FLOPs the kernel runs
at **1271 GFLOP/s of roughly 21 TFLOP/s FP32 (60-core M2 Ultra) — about 6%.**
The stage is still 10.51 ms and still the largest; only the diagnosis changes,
from "make it stream" to "find out why the ALUs are idle." Candidates worth
pricing, in order:

1. **The f32→half staging.** The kernel comment says "f32 rows staged to half"
   before the 8×8 simdgroup steps. The compressed cache is F32 on disk *and* in
   memory (`ds4.c:17373`), so every key is converted on the fly, every layer,
   every token. **Storing it as F16 would remove the conversion entirely** —
   note this is T9/U3, which was killed on bandwidth grounds that no longer
   apply. **U3 should be reconsidered for a completely different reason than it
   was proposed.**
2. **Occupancy.** 256 threads and `sharedf` threadgroup memory per threadgroup;
   report threadgroups-in-flight per core and whether shared memory or
   registers are the limiter.
3. **The simdgroup step over depth 128** in 8×8 tiles — 16 sequential matrix
   steps per key tile, each dependent on the last.

**What to actually run.** There is no rows-per-threadgroup knob to sweep, and
adding one would be pointless. The only existing variants are
`kernel_dsv4_indexer_scores_llt16` / `llt32` (`NBPTG` = 16/32), which vary the
**token** axis and are therefore **inert at decode** (`n_tokens = 1`), and
`llt_nsg4` (`T_NSG=4`), already measured as a wash. A real sweep needs **new
template instantiations varying `T_NSG`** — `<8,2>` and `<8,16>` — which is a
two-line change in `metal/dsv4_misc.metal` plus pipeline names. That is the
first cheap experiment; the F16 staging is the first substantive one.

**Sequencing unchanged in one respect:** this still supersedes U4. A 10.51 ms
stage split across ranks saves ~5 ms, but fixing 6% ALU utilisation is worth
more and makes the split worth proportionally less. Fix the kernel first.

#### U7 — instrumented, and the occupancy hypothesis is already dead — 2026-08-27

**Built.** `DS4_METAL_INDEXER_LLT_NSG` ∈ {2, 4, 8} selects simdgroups per
threadgroup; the dispatch is now generalised from `NK = 8*NSG` rather than two
hardcoded branches, and reproduces the old constants exactly at 8 and 4.
`DS4_METAL_INDEXER_LLT_NSG4` still works as an alias so the T3 arm reproduces.
New instantiation `kernel_dsv4_indexer_scores_llt_nsg2`.

**NSG=16 is not buildable** and that is the finding. Staging is
`sk[NK*128]half + sq[8*128]half + sw[8]f32 + sqk[NK*8]f32`, so NSG=16 needs
**38,944 B against a 32,768 B limit.** The residency ladder:

| NSG | NK | threads | smem | threadgroups resident / core |
|---|---|---|---|---|
| 2 | 16 | 64 | 6,688 | 4 |
| 4 | 32 | 128 | 11,296 | 2 |
| **8** (default) | 64 | 256 | 20,512 | **1** |
| 16 | 128 | 512 | 38,944 | **does not fit** |

**Swept on the M1 Max — more residency is *worse*, so the occupancy hypothesis
is falsified:**

| NSG | GFLOP/s | vs default |
|---|---|---|
| 8 (default) | 1050.6 | — |
| 4 | 1036.4 | −1.4% |
| 2 | 784.9 | **−25%** |

Q staging is amortised over NK keys, so shrinking NK to buy residency loses
more than it gains. **The trend says bigger NK is better and we are pinned at
NSG=8 by threadgroup memory** — one threadgroup per core is not the problem to
fix, it is the price of the largest NK that fits.

**All three arms are bit-identical** (`BENCH_DUMP`, 32768 rows, byte-for-byte).
NSG changes how many keys a threadgroup owns, not the per-key reduction order,
so unlike T1 and T2 this knob needs no correctness gate.

**Correction to my own M1 Max number.** I reported 460 GFLOP/s / "4.4% of peak"
earlier from a **stale binary** (`tests/bench_indexer_score`, built 2026-08-24,
predating the LLT wiring). A fresh build gives **1050.6** — the earlier figure
was the pre-LLT direct path. Anything computed from 460 is void.

**The question that now matters most, and it is for the rig.** My fresh M1 Max
number is **1050 GFLOP/s**; U2 measured **1015 on the M2 Ultra** at the same
`n_comp`. A 60-core part matching a 32-core part means **the kernel does not
scale with cores at all.** If that reproduces it is worth far more than any
NSG arm — it says the grid or the staging serialises. **First rig action: re-run
`bench_indexer_score 32768` on a freshly built binary and confirm or kill the
non-scaling.** Check U2's binary date before trusting its 1015.

**The remaining lever, and U3 returns for a third reason.** To grow NK past 64
the `sk` staging must shrink. It exists to convert F32 keys to half; if the
index cache were stored F16 (`ds4.c:17373`) the kernel could read K directly,
freeing ~16 KB and admitting NSG=16. So T9/U3 — proposed for bandwidth, killed
for bandwidth — is back on **two** independent grounds: it removes a per-key
format conversion, and it unlocks the NK the sweep says we want. Re-open it
after the scaling question is settled.

#### U7b — the scaling puzzle is probably an artefact, and U2 measured the wrong kernel

Fresh build on the M1 Max, same `n_comp=32768`:

| path | GFLOP/s |
|---|---|
| `kernel_dsv4_indexer_score_one_direct` (fallback) | 456.5 |
| `kernel_dsv4_indexer_scores_llt` (**production default**) | 1075.9 |

**U2's 1015 on the rig is almost certainly the *direct* path**, not LLT — its
own note describes `kernel_dsv4_indexer_score_one_direct` and its `(32,4)`
threadgroup shape. Against direct-on-M1-Max (456.5) that is **2.22×**, right in
line with core count × clock (60/32 × 1398/1296 ≈ 2.02). **So the kernel scales
fine and there is no non-scaling mystery** — I was comparing my LLT number
against U2's direct number.

Two consequences. **U2's roofline is about a kernel production does not run**,
so its "~5% of roof" should not be carried forward. And the rig's real LLT rate
is likely ~2100 GFLOP/s, which puts the *scoring* half of the stage at roughly
0.26 ms/layer × 21 ≈ **5.4 ms** — leaving the other ~5 ms of the 10.51 ms stage
somewhere else. M2 already said where: **`indexer topk` was 5.32 ms against
`indexer score` 3.84 ms.** Selection costs more than scoring.

**Still confirm on the rig with a freshly built binary** — this is inference
from a dev-box A/B, not a measurement.

#### U7c — the rig number lands: 1565 GFLOP/s, and it scales at 72%, not 100%

Rig, fresh build, `n_comp=32768`, LLT path: **1565 GFLOP/s** against the M1
Max's 1075.9.

**Scaling is real but sub-linear, and my ~2100 estimate was 39% high.**

| | M1 Max (32c @1.296) | M2 Ultra (60c @1.398) |
|---|---|---|
| measured | 1075.9 | **1565** |
| ratio | — | **1.45×** |
| ideal (cores × clock) | — | 2.02× |
| **scaling efficiency** | — | **72%** |
| FP32 peak | 10.6 TFLOP/s | 21.5 TFLOP/s |
| **achieved** | **10.1%** | **7.3%** |

So the non-scaling worry is dead (U7b was right that far) but **the bigger part
is the *less* efficient one** — 7.3% of ALU peak against 10.1%. Something costs
more on the Ultra, and the obvious candidate is memory latency: with
threadgroup memory allowing **exactly one threadgroup resident per core**
(U7's ladder), there is almost nothing to hide latency with, and the Ultra has
more latency to hide. That is consistent with the NSG sweep, with T2's
turnover, and with the constraint recalled from the earlier investigation.

**A calibration worth keeping: the harness over-predicts this kernel by ~1.9×.**
At 1565 GFLOP/s the scoring half works out to 0.343 ms/layer × 21 = **7.20 ms**,
but M2's in-situ ablation attributed **3.84 ms** to `indexer score`. Ratio
**1.88×**. That is the harness's documented failure mode — *"run the kernel
alone and every stall becomes wall time; run it inside the graph and other work
hides them"* — now quantified for this kernel. **Divide standalone indexer
numbers by ~1.9 before projecting engine impact**, alongside the ~0.63
standalone-to-rig factor already in this document. Note the in-situ numbers are
the trustworthy ones, and they still put top-k (5.32 ms) above scoring (3.84).

#### U10a — run the NSG sweep on the rig — **zero code, and it decides U10 — DONE 2026-08-27: U10 is ON**

**Outcome.** NSG sweep on mat (M2 Ultra), n_comp=32768, 300 dispatches,
`DS4_METAL_GPU_BUSY_PROFILE=1`, repeated 2–3× per arm for stability:

| NSG | GFLOP/s (runs) | vs default |
|---|---|---|
| **8** (default) | 1525.2 / 1607.4 / 1683.0 | — |
| **4** | **1720.7 / 1813.8 / 1704.4** | **+12.8% / +12.8% / +1.3%** |
| 2 | 1342.2 | −12% |

NSG=4 beats NSG=8 in every run (~+5–13%), and NSG=2 is clearly worse — the
**first outcome in the decision table**. The M1 Max ranking (−1.4% at NSG=4)
does **not** transfer to the Ultra: residency is worth more than NK there.
**U10 is on** — dropping `sk` buys 7× residency at unchanged NK, strictly
better than what NSG=4 trades for. (NSG=4 confounds halved NK with doubled
residency, so it is suggestive, but the win is strong evidence because U10
gets the residency without paying the NK.) See `BENCHMARKS-TP-PP.md` §U10a.

**This is the gap in the U7 rig run.** The rig measured only the default
(NSG=8, 1565 GFLOP/s). The `DS4_METAL_INDEXER_LLT_NSG` knob is built, committed
and bit-identical across arms — **the sweep just has not been run there**, and
it is the free test of U10's entire premise.

```
for n in 8 4 2 ; do
  DS4_METAL_INDEXER_LLT_NSG=$n DS4_METAL_GPU_BUSY_PROFILE=1 \
    tests/bench_indexer_score 32768 300
done
```

**Why it decides U10.** On the M1 Max more residency was *worse* — NSG=4
−1.4%, NSG=2 −25% — because shrinking NK loses more on Q-staging amortisation
than residency gains. But U7c's whole argument is that **the Ultra is short of
latency hiding specifically**, at 72% scaling efficiency and 7.3% of ALU peak
against the M1 Max's 10.1%. If that is right, the residency/NK trade should sit
at a **different point on the Ultra than on the M1 Max**.

| rig outcome | reading | consequence |
|---|---|---|
| NSG=4 ≥ NSG=8 | residency is worth more than NK on the Ultra — the M1 Max ranking does **not** transfer | **U10 is on**: dropping `sk` buys 7× residency at *unchanged* NK, strictly better than what NSG=4 trades for |
| NSG=4 ≈ −1.4% as on M1 Max | the trade behaves the same on both parts | U10 weakens sharply — residency is not the deficit, and the 72% is something else |
| NSG=4 clearly worse | NK dominates even harder at scale | **U10 is off.** Look at the 8×8 simdgroup step chain instead |

Note NSG=4 confounds two changes (NK halved *and* residency doubled), so it can
only ever be suggestive — but a **win** there is strong evidence for U10 because
U10 gets the residency without paying the NK. Run this before writing any U10
code.

#### U10b — sizing U9 against U10 end-to-end, and resolving the T3 contradiction

**The apparent contradiction first, because it looks worse than it is.** U10a's
harness says NSG=4 is **+12.8%** at `n_comp=32768`. T3's *production* A/B of the
same flag, at the ctx that corresponds to exactly that `n_comp` (131072 / 4),
measured **−1.7%**. Opposite signs on the same knob at the same operating point.

They reconcile on sizing. The score kernel is **3.84 ms of a 36.36 ms token —
10.9%.** A +12.8% kernel win is therefore **+1.2% end-to-end**, and T3's decode
deltas at 131k ran −1.7% / −0.6% against a noise floor of about ±1%. **The
predicted effect sits inside T3's noise.** No contradiction, but no production
confirmation either — and note T3 also recorded first-token latency spiking to
189 ms vs 31 ms control, so the flag is clearly *prefill*-negative and must not
be flipped globally on decode evidence.

**Now size the two live items against the token, which I failed to do when I
called U10 "the strongest structural lever."**

| | share of token | 2× | 4× | 7× |
|---|---|---|---|---|
| **U9** — top-k, 5.32 ms | **15.0%** | **+7.5%** | **+11.3%** | +13.0% |
| **U10** — score, 3.84 ms | 10.9% | +5.4% | +8.2% | +9.3% |

**U9 is the larger item, and the gap is wider than the table shows.** Top-k is
**algorithmic** — a full bitonic sort replaced by an O(n) select, where 4× is a
conservative ask. U10 is a **constant-factor** occupancy tune on a kernel
already running a reasonable algorithm, where 7× residency will not become 7×
throughput; the honest expectation is the 1.3–2× band, i.e. **+2.5–5.4%**.

**Revised recommendation: U9 first, U10 second.** I had them roughly co-equal
and led with U10 because U10a was cheap to decide. That was sequencing
convenience, not sizing.

**Two things that follow.**

1. **Do not flip the NSG=4 default on harness evidence.** The end-to-end effect
   (~1.2%) is below what our decode A/B can resolve, and T3 says it is
   prefill-negative. If it is wanted, it needs a decode-only gate and a
   production A/B with enough repeats to resolve ~1%, which M0-class runs
   currently cannot.
2. **U10's ceiling should be re-stated honestly wherever it appears.** "7×
   residency" is the mechanism, not the outcome. Against the token it is a
   **~2.5–5.4%** item — worth doing, not worth doing first.

**And the general lesson, which has now cost us twice.** T9 and T6 were both
mis-sized because they were priced before the stage profile existed. U10 was
mis-*ranked* because I priced the kernel and not the token. **Every future item
in this document should carry its end-to-end share, not just its kernel-level
multiple.**

#### U10 — drop the `sk` staging buffer: 1 → 7 threadgroups resident per core — **cheap half DONE on the rig 2026-08-27**

**Rig A/B (mat, M2 Ultra, 2026-08-27): `DS4_METAL_INDEXER_LLT_TIGHT` wins at
+17.4%** (mean 2015.6 vs 1717.2 GFLOP/s at n_comp=32768, 3 runs), and beats
NSG=4 (+2.4% there) too. **Bit-identical** to default (32750/32768 exact, same
worst-rel row 22878). The implemented cheap half (alias `sq`/`sw`/`sqk` over
`sk`, residency 1 → 2 at unchanged NK) is confirmed at Ultra scale; end-to-end
it is a ~2–4% token item (see U10b). **U10c** (F16 cache, 2 → 7 resident) is
gated on.

**Productionised: `TIGHT` is now the default**, with
`DS4_METAL_INDEXER_LLT_TIGHT=0` as the rollback. Unlike T3's nsg4 it **does not
change the grid** — only the threadgroup-memory allocation — so it carries none
of the prefill risk that made nsg4 spike first-token latency to 189 ms. Worth
confirming prefill on the next rig run regardless: the same single-token scorer
runs per token in the prefill tail loop (`ds4.c:30833`), so this is not a
decode-only path.

**End-to-end it delivers ~+1.6%** — score is 3.84 ms of 36.36, and +17.4% saves
0.57 ms. **U10b predicted +1.7%**, so the end-to-end sizing discipline is now
validated against a measurement and should stay.

**Dead code found while wiring this:** `kernel_dsv4_indexer_scores_llt16` and
`_llt32` (`NBPTG` 16/32) are referenced by **no host call site**. The LLT
pipelines have exactly one dispatch site, `ds4_gpu_indexer_score_one_tensor`,
which always passes `n_tokens = 1`, so the `NBPTG` template axis is inert
everywhere and both instantiations are unreachable. Add to the cleanup batch.

### Where the decode token stands after U9 and U10 — 2026-08-27

| stage | ms | share of 36.36 | status |
|---|---|---|---|
| `q_path` | 5.47 | **15.0%** | **never priced** |
| `routed_moe_folded` | 5.40 | 14.9% | 54% of roof; dequant/accumulate (U8) |
| indexer top-k | 5.32 | 14.6% | **U9 failed**; residual idea below |
| `attn_inv_rope` | 4.27 | **11.7%** | **never priced** |
| indexer score | 3.84 | 10.6% | **U10 done: +17.4% kernel → +1.6% token** |

**`q_path` + `attn_inv_rope` are 9.74 ms — 26.8% of the token — and neither has
ever been looked at.** With U9 dead that is the largest unexplored area in this
document, larger than anything queued. It is also cheap to start: the stage
profile already names them, so the work is identifying the kernels and pricing
their achieved rate against the 760 GB/s roof and the ~21 TFLOP/s ALU peak,
exactly as U7 and U8 did.

**Residual U9-shaped idea, far cheaper than U9 was.** The argsort path's own
split has never been swept: `nth` is the largest power of two ≤
`maxTotalThreadsPerThreadgroup`, and `npr = ceil(n_comp/nth)` falls out — 32
blocks of 1024 at `n_comp` 32768. Nothing has tested whether that is the right
point, and U10 just demonstrated this kernel family is residency-sensitive, so
smaller blocks are worth one arm. Zero code if `nth` is made env-overridable.

### Recovered prior work — `speed-bench/tp_decode_investigation.md` — 2026-08-27

The dispatch-reduction dead end is documented, and reading it back turns up two
corrections to this plan and one queued target we had lost.

**The dispatch campaign was abandoned on arithmetic, not on threads.** §6:
*"Dispatch removal is not a productive strategy here. 1021 dispatches × 1.9 µs
= 1.94 ms, and a realistic fusion campaign was scoped at 185 dispatches =
0.35 ms = +0.6 t/s."* That is the same conclusion U6/U7 reached independently,
so nothing to re-tread — **and it retroactively explains why U9 was a bad bet**:
it was a dispatch-shaped idea in a tree that had already priced dispatch-shaped
ideas at +0.6 t/s.

**The "not enough threads" memory is a different entry**, §8: *"packed32 flash
reduce at 32 heads — correct but **1.35 t/s slower**; tuned for the 64-head
grid, halving it **underfills 60 cores**. Reverted (`fe4674f`)."* Same family as
T2's turnover at 112 threadgroups and U7's smem-capped residency. Three
independent observations of one constraint.

**Correction 1 — §4's "the kernels are fine" was measured against the wrong
roof.** Its key table:

| stage | GB/token | ms | achieved | isolated bench | ratio |
|---|---|---|---|---|---|
| attn output | 1.533 | 2.880 | 532 GB/s | 517–581 | ~100% |
| `q_b` | 0.767 | 1.559 | 492 GB/s | ~413–541 | ~100% |
| routed MoE | 2.264 | 6.170 | 367 GB/s | ~400 | 92% |

Those ratios compare **in-engine achieved against the same kernel measured in
isolation** — they show the engine loses nothing versus standalone. They do
**not** show the kernels are near the hardware. U6 put the streaming roof at
**760 GB/s**, so 532 GB/s is **70%** and the isolated bench itself is 68–76%.
*"Conclusion: the kernels are fine"* should read *"the engine is not the
problem; the kernels are at ~70% of what the part can stream."* That is exactly
the error U6 corrected for the MoE, and it was sitting in this file for the
attention path too.

**Correction 2 — U12 must not repeat the `attnout` phantom.** §8: *"`attnout` /
`out_b` restructuring — **phantom target**. Believed to be 2× its rate; it runs
at ~100%. The error was a 3×-low byte estimate and a wrong k (512 vs the actual
4096)."* And §14.6, in the file's own words: **"Before quoting any byte figure,
reconcile it against the 60.17 MB/layer in §3. Three wrong conclusions in this
investigation came from skipping that check."** Fold that into U12 as a
prerequisite, not advice.

Also §14.1, which we re-learned the hard way in U9: *"treat changes below about
1% as noise until interleaved."* It was already written down.

#### U14 — the MoE expert straggler (recovered from §7) — fully scoped, never queued here

**Worth ~1.29 ms at 131k (3.5% of the token); §7 measured it at 1.47 ms /
+2.7 t/s at its own baseline.** Routed experts are sharded contiguously
128/128, so with 6 experts selected uniformly the per-layer critical path is
`E[max(k, 6−k)] = 3.9375` against an ideal 3.0 — **and because each layer gates
independently the imbalance does not average out.**

Three designs were already scoped (§7): **A** partial replication (E[max]≈3.16,
**+27 GiB**, drops usable context ~1M → ~650k); **B** in-place intra-expert
split (**blocked** — ~22k spans, and `ffn_down_exps` splits on the inner dim so
it is not expressible as spans at all); **C** repacked GGUF (perfect 3.0, one
span per tensor, no extra memory, and incidentally makes the `down` read
contiguous — cleanest end state, costs a conversion tool and a TP-specific
model file).

**Throughput is already derisked:** the shared expert *already* uses exactly
this decomposition (`ds4.c:23922`) and hits its predicted rate. **The obstacle
is mapping, not throughput.**

**Deprioritised 2026-08-27 — recorded so it is not re-derived, not queued.**
Design C is a repacked TP-layout-specific model file plus a conversion tool for
3.5%, and B is blocked, and A costs 27 GiB and a third of the usable context.
That is poor return against U12/U13. Two things are worth writing down first.

**Whole-expert reassignment cannot fix this, so C really is the only route to
perfect balance.** The imbalance is *per-token variance*, not mean load: even
with every expert equally popular, `E[max(k, 6−k)] = 3.9375`. No permutation of
which experts live on which rank changes that, and popularity-balancing the
partition fixes the mean while leaving the variance untouched. Splitting every
expert's intermediate dimension is the only static scheme that makes both ranks
do exactly half of *whatever* is selected.

**But §7 missed a fourth option, and it is much cheaper.** `router`,
`shared_gate_up`, `routed_moe` and `shared_down` all run **before the FFN
gate**, so the gate waits on `max_over_ranks(shared_share + routed_share)` —
and the shared expert is split by a fixed 50/50 `shared_tp_local = shared_dim/2`
(`ds4.c:23922`). Give the routed-*light* rank more of the shared expert and the
max drops:

| per layer @131k | heavy rank | light rank | max |
|---|---|---|---|
| today (shared 50/50) | 0.1256 + 0.0191 | 0.0658 + 0.0191 | **0.1447** |
| all shared → light rank | 0.1256 + 0 | 0.0658 + 0.0381 | **0.1256** |

**0.82 ms/token — 2.3% of the token, 64% of design C's prize** — with no extra
memory, no repacked GGUF, and reusing a decomposition that already exists and
already hits its predicted rate. Note the ceiling: even handing the light rank
*all* of the shared expert does not fully balance (it would need 128% of it),
so the win is exactly the heavy rank's shared share and no more.

**The real obstacle is dispatch shape, not mapping.** `k` is only known after
the router kernel runs *on the GPU*, and the host encodes dispatches ahead of
time; reading `k` back would add a sync that costs more than the win. It needs
either an indirect command buffer or — simpler — both ranks dispatching the
full shared grid with each threadgroup predicated on `k` from a device buffer,
so the unassigned side early-outs. Not bit-exact (it repartitions the shared
expert's sums), so T2 bar.

**If this is ever revisited, do the shared-shift, not the repack.**

#### U11 — confirm U10 does not regress prefill — **run first — DONE 2026-08-27: neutral, U10 stays default-on**

**Outcome.** Prefill A/B on the rig (mat worker / lanfear coord, build
`b99dfa3`), default (TIGHT on) vs `DS4_METAL_INDEXER_LLT_TIGHT=0`, 32k/65k/131k
with `promessi_sposi.txt`:

| ctx | arm | prefill t/s | first-token ms | steady t/s |
|---|---|---|---|---|
| 32k | TIGHT on | 501.92 | 272.9* | 34.59 |
| 32k | TIGHT=0 | 517.19 | 31.1 | 33.92 |
| 65k | TIGHT on | 460.94 | 32.8 | 32.19 |
| 65k | TIGHT=0 | 461.72 | 32.5 | 31.68 |
| 131k | TIGHT on | 393.03 | 36.5 | 29.15 |
| 131k | TIGHT=0 | 393.37 | 35.5 | 28.42 |

*\*The 272.9 ms at 32k TIGHT-on is a cold-start/warm-up artifact — the first
frontier of the first arm includes model load and shader compile, and the same
arm's 65k/131k first-token is 32.8/36.5 ms. **Treat the whole 32k row as void
rather than as −3%**: the warm-up contaminates that arm's prefill t/s as well,
so it is not evidence either way. The 65k and 131k rows are clean and are what
answers the question.*

**Prefill is neutral:** 501.92 vs 517.19 (−3% at 32k, warm-up), 460.94 vs
461.72 (−0.2%), 393.03 vs 393.37 (−0.1% at 131k). First-token at 131k 36.5 vs
35.5 ms (within noise). **No prefill-catastrophic spike** — nothing like T3's
nsg4's 189 ms. Decode steady-state is slightly *better* with TIGHT on (29.15
vs 28.42 t/s at 131k). **U10 stays default-on.** See `BENCHMARKS-TP-PP.md`
§U11.

U10 is default-on. The one dimension nobody has measured is prefill, and T3 is
the reason to bother: `nsg4` was decode-positive and **prefill-catastrophic**,
spiking first-token latency to 189 ms against a 31 ms control. The mechanism
there was a doubled threadgroup count; U10 does not touch the grid, only the
allocation, so the prior is that it is neutral-to-positive. That is a prior, not
a measurement.

**It matters because this scorer is not decode-only.** `ds4_gpu_indexer_score_one_tensor`
has exactly one dispatch site, but two callers: decode (`ds4.c:23241`) and the
**prefill tail loop** (`ds4.c:30833`), which calls it once per token for every
token past the raw prefix — thousands of calls in a 4096-token chunk.

**Arms:** default (TIGHT on) vs `DS4_METAL_INDEXER_LLT_TIGHT=0`, prefill sweep
at 32k/131k with `speed-bench/promessi_sposi.txt`. **Report first-token latency
explicitly**, not just steady-state t/s — that is the number T3 caught it on.

**Decision:** prefill neutral or better → U10 stays default-on, done. Prefill
regression → gate TIGHT to the decode call site only, which is a one-line
change since the two callers are distinct.

#### U12 — price `q_path` and `attn_inv_rope` — **26.8% of the token — DONE 2026-08-27**

**Outcome.** Stage-profile at 32k/131k (build `b99dfa3`): **q_path 5.474 →
5.472 ms/token** (context-invariant, 15.0% of token) and **attn_inv_rope
3.389 → 4.258 ms** (+26% with context, 11.7% of token). Together 9.7 ms =
26.8% of the 36.36 ms token. q_path is the largest non-indexer decode stage;
attn_inv_rope grows with context. See `BENCHMARKS-TP-PP.md` §U12.

Together they are **9.74 ms of the 36.36 ms token**, larger than any item ever
queued in this document, and neither has been priced against a roof. Both are
already named by the stage profile, so this is measurement, not archaeology.

**What they are** (read from the stage boundaries, so the request is concrete):

- **`q_path`, 5.47 ms, 15.0%** (`ds4.c:22776`) — the whole query path: the
  `q_a`/`q_b` projections, head RMS norm, and the RoPE tail. It has a fusion,
  `ds4_gpu_head_rms_norm_rope_tail_tensor` (`ds4.c:22749`), with a standalone
  fallback of `head_rms_norm` + `rope_tail` when it does not fire. **First
  question: does the fused path actually fire in production at 131k, or are we
  paying two extra dispatches per layer?** M2's `qb` ablation was only 1.78 ms,
  so ~3.7 ms of this stage is *not* the q_b projection and has never been
  attributed.
- **`attn_inv_rope`, 4.27 ms, 11.7%** (`ds4.c:23568`) — a standalone
  64-threadgroup inverse-RoPE dispatch applied to the attention output heads.

**Prerequisite, per §14.6 of `tp_decode_investigation.md`:** reconcile every
byte figure against the verified 60.17 MB/layer model in its §3 **before**
quoting a rate. Three wrong conclusions in that investigation came from skipping
this, including the `attnout` phantom target — believed 2× off, actually at
~100%, from a 3×-low byte estimate and a wrong k. `q_path` is the same family of
kernel, so the same trap is live.

**Note what §4 already priced:** `q_b` at 492 GB/s and attn output at 532 GB/s,
both ~100% of their *isolated bench* — but only ~70% of U6's 760 GB/s roof. So
the open question for `q_path` is not "does the engine lose against standalone"
(answered: no) but "why is the standalone kernel at 70%".

**Method, mirroring U7/U8:** stage-profile at 32k and 131k with per-kernel
attribution; compute achieved GB/s and GFLOP/s for each and place them against
the **760 GB/s** roof and the **~21 TFLOP/s** ALU peak. Report which of the two
roofs each is near, exactly as U7 did — that is what turned the indexer from
"slow" into "occupancy-bound", and it is the step that has produced every real
win here.

**Report the end-to-end share alongside any kernel multiple** (U10b).

#### U12 — **partially done 2026-08-27.** Times landed; the roofline did not — and `attn_inv_rope` is mis-named

**Delivered:** per-stage times at 32k/131k. `q_path` **5.474 → 5.472 ms —
context-invariant**; `attn_inv_rope` **3.389 → 4.258 ms, +26% with context**.
Together 26.8% of the token, as expected.

**Not delivered:** the roofline. No GB/s, no GFLOP/s, and no reconciliation
against the 60.17 MB/layer model — which was the *prerequisite*, and the whole
point. We now know what these stages cost; we still do not know why.

**Correction 1 — `attn_inv_rope` is mostly FlashAttention, not RoPE.** Stage
spans run marker-to-marker, and this one begins at `compressor_indexer`
(`ds4.c:23316`) and ends at `ds4.c:23568`. `ds4_gpu_attention_decode_heads_tensor`
— the decode FlashAttention call — sits at `ds4.c:23539`, **inside that span.**
Confirming: there is **no separate attention stage anywhere in the profile's
twelve entries**, so the attention core has to be here. The +26% growth with
context is attention behaving normally, not a RoPE tail getting slower.

**This invalidates U13's sizing.** U13 was written against "a 4.27 ms standalone
inverse-RoPE dispatch". Most of that 4.258 ms is attention. The RoPE tail's real
share is unmeasured and could be a small fraction of it. **Do not write U13
until it is isolated** — and do not reuse M2's `attncore` (1.91 ms) to subtract,
because that is a different measurement epoch and mixing epochs is exactly the
error §14.6 warns about.

**Isolating it is free.** `DS4_METAL_DISABLE_PRE_M5_ATTN_INV_ROPE_FUSE=1`
(`ds4.c:22094`) forces the *gathered* branch onto the standalone RoPE as well.
Default vs that arm prices the standalone dispatch across the 20 gathered
layers, which scales directly to the 21 indexed layers that pay it today. **Zero
code, one arm, and it is the number U13 actually needs.**

**Correction 2 — `q_path` roofline, computed here since U12 did not.** From the
shapes: `q_a` is 4096 × 1024 and `q_b` is 1024 × 16384 per rank, Q8_0 at ~1.031
B/weight → **21.6 MB/layer/rank, 0.930 GB/token**. Reconciliation check per
§14.6: `q_b` alone is 0.744 GB against §4's measured **0.767 GB** — agrees, so
the byte model is sound.

**0.930 GB over 5.472 ms = 170 GB/s = 22% of the 760 GB/s roof.** That is the
number U12 was asked for, and it says `q_path` is *not* bandwidth-limited —
consistent with it being context-invariant.

**Where to look next in `q_path`.** The fused head-norm + RoPE-tail kernel
dispatches `MTLSizeMake(n_head, n_tok, 1)` (`ds4_metal.m:22589`) — at decode
that is **32 threadgroups on 60 cores**, one per head, for 512 floats each.
Same underfill family as packed32-at-32-heads (−1.35 t/s), T2's turnover at
112, and U7's smem-capped residency. But **do not size this by subtracting §4's
`q_b` from U12's `q_path`** — different epochs. Get it by adding stage markers
between `q_a`, `q_b` and the norm/RoPE tail: three lines, and it splits 5.47 ms
into three numbers that are all from one run.

### U12b outcome — my hypothesis is dead, and a better target fell out — 2026-08-27

**Arm 2 — `q_path` split three ways** (context-invariant at both 32k and 131k):

| sub-stage | ms | bytes/token | achieved | of 760 roof |
|---|---|---|---|---|
| `q_a_kv_proj` | 2.137 | 0.279 GB | 131 GB/s | **17%** |
| `q_lora_norm` | 1.680 | **~0.55 MB** | **0.33 GB/s** | **0.04%** |
| `q_path` (q_b + per-head norm/RoPE) | 1.862 | ~0.744 GB | ~400 GB/s | 53% |

**The per-head norm/RoPE underfill hypothesis is dead.** Phase B is 1.862 ms
against a ~1.5 ms `q_b`-alone estimate, so the tail is **~0.36 ms** — not the
1.5–2.5 ms I put in the sizing table. The U12b decision table called this
correctly ("remaining `q_path` ≈ 1.5 ms → the cost is in `q_a_kv_proj`, look
there"), which is the first time the pre-registered prediction has paid off.
The 32-threadgroup grid at `ds4_metal.m:22589` is real but it is not costing us
anything worth chasing.

**`q_lora_norm` is the find: 1.68 ms — 6.9% of the 2k token — moving ~0.55 MB.**
That is **0.33 GB/s, 0.04% of roof**, and **39 µs per layer to normalise a
1024-element vector**. Twenty times the 1.9 µs marginal dispatch cost, so it is
not dispatch count either. Nothing else measured in this project is this far
from its roof — the indexer at 5% of ALU peak was the previous record and this
is two orders of magnitude below that.

**Which variant fires — resolved from source 2026-08-27, and it explains the
number.** The span (`ds4.c:22612`–`:22732`) selects among four variants. The
first, `dsv4_qkv_rms_norm_rows_kv_rope`, is gated on `g->cuda_qkv_kv_rope_fuse`
and is **CUDA-only**, so Metal takes either
`dsv4_qkv_rms_norm_kv_rope_fp8_store` or `dsv4_qkv_rms_norm_rows`. **Both
dispatch two threadgroups:**

- `..._kv_rope_fp8_store` → `MTLSizeMake(1, 2, 1)` (`ds4_metal.m:22402`)
- `..._rows` → `MTLSizeMake(rows, 2, 1)`, and the decode call site passes
  **`rows = 1`** (`ds4.c:22701`)

**So `q_lora_norm` runs 2 threadgroups on a 60-core GPU, 43 times per token —
1/30th of the machine.** That is the most extreme underfill measured anywhere in
this engine; the previous worst was the 32-threadgroup head-norm/RoPE, which
turned out to cost only 0.36 ms.

**39 µs is far too long even for two threadgroups**, so the cost is not the
norm's arithmetic — a 1024-element reduction is microseconds. It is a
**serialisation point**: the norm depends on the q_a/kv projection and q_b
depends on the norm, so with two threadgroups resident the GPU drains and
refills around it, 43 times a token. This is also why it does not contradict
§6's "dispatch removal is not productive" — that concerns the *marginal* 1.9 µs
cost of an additional dispatch, whereas this is one dispatch costing 20× that
because it has no parallelism to hide behind.

**The fix direction is fusion, and the tree has precedent for exactly this
shape.** `kernel_dsv4_comp_row_finalize_f32` already collapses seven tiny
single-row dispatches per layer into one two-threadgroup dispatch while
preserving each kernel's reduction tree bit-exactly (`ds4.c:22966-22971`), and
`ds4_gpu_head_rms_norm_rope_tail_tensor` already fuses norm+RoPE on the head
path. Folding the q-LoRA norm into the preceding projection or the following
q_b removes 43 serialisation points per token.

**Prize: up to 1.68 ms — 6.9% at 2k, 4.9% at 131k** — though a fusion will not
recover all of it, since the arithmetic still has to happen somewhere.

`q_a_kv_proj` at 17% of roof is the second target — 2.14 ms, 8.8% of the 2k
token, and a plain Q8_0 pair projection that ought to run near where `q_b`
does (53%). If it reached `q_b`'s rate it would be ~0.7 ms, saving **~1.4 ms**.

**Together these two are 3.82 ms — 15.7% of the 2k token — and both are far
enough from their roofs that this is now the best-supported headroom in the
document.**

### U13 — **effectively dead, and dead outright for the short-context program**

**Arm 1** forced the gathered branch onto the standalone RoPE:

| ctx | default | norope | delta |
|---|---|---|---|
| 32k | 3.393 | 3.526 | **+0.133 ms** |
| 131k | 4.252 | 4.764 | **+0.512 ms** |

Scaling by 21/20 for the indexed layers puts U13's prize at **~0.14 ms at 32k
and ~0.54 ms at 131k**. Against the plan's gate (under ~0.3 ms → dead) it is
marginal at 131k and **clearly under at 32k**, so for the short-context program
it is worth essentially nothing.

**And treat the 0.512 ms as an upper bound.** A fixed-size rotate cannot grow
with context, so the delta is not purely the RoPE dispatch — disabling the fuse
also switches the attention reduce between its fused and unfused variants, and
that difference scales with the reduce's size. Some unknown fraction of the
0.512 ms is the reduce, not the RoPE.

**Verdict: deprioritise U13** below `q_lora_norm` and `q_a_kv_proj`. It is a day
of not-bit-exact work for ~1.6% at long context only.

## The short-context program — 2026-08-27

**Target: 50 t/s at low context, from ~41 today.** The reason to pursue this
ahead of more long-context kernel work is structural, and it is the most useful
thing the stage profile has told us.

### 88% of the short-context token is context-invariant

Separating U12's 32k and 131k columns by whether a stage moves:

| stage | ms | % of the 2k token | priced? |
|---|---|---|---|
| `q_path` | 5.47 | **22.5%** | 170 GB/s = **22% of roof** |
| `routed_moe_folded` | 4.99 | **20.5%** | ~410 GB/s = **54% of roof** |
| `attn_output` | 3.75 | **15.4%** | only in an older epoch, and once a phantom target |
| `ffn_hc_post` | 1.83 | 7.5% | **never** |
| `attn_hc_pre` | 1.14 | 4.7% | **never** |
| `ffn_hc_pre` | 1.12 | 4.6% | **never** |
| `router` | 1.11 | 4.6% | **never** |
| `shared_gate_up` | 0.98 | 4.0% | **never** |
| `shared_down` | 0.66 | 2.7% | **never** |
| `attn_hc_post` | 0.47 | 1.9% | **never** |
| **fixed subtotal** | **21.53** | **88.5%** | |

Only `compressor_indexer` and `attn_inv_rope` grow with context. **Everything
else is paid identically at 2k and at 131k.**

**34% of the 2k token has never been priced against any roof** — the four HC
stages (4.56 ms) and `attn_output` (3.75 ms). That is a larger unexamined
fraction than long context ever had, and it is why this is the better place to
spend effort now: the long-context queue is picked-over ground, this is not.

### Fixed-work savings propagate, additively

A saving in fixed work is the *same number of milliseconds* at every context —
so it is a smaller *percentage* at long context, but free there:

| fixed work cut | 2k | 131k |
|---|---|---|
| — | 41.1 t/s | 29.1 t/s |
| 2.0 ms | 44.8 | 31.0 |
| 3.0 ms | 46.9 | 31.9 |
| **4.34 ms** | **50.0** | **33.4** |
| 6.0 ms | 54.5 | 35.3 |

**So hitting 50 t/s at 2k lands ~33.4 t/s at 131k** — a +15% long-context gain
for work aimed at short context. Reaching 35 at 131k needs 5.73 ms, more than
the fixed pool realistically holds, so that additionally requires attacking
`compressor_indexer`. Note the effect is **additive, not ratio-preserving**:
removing shared fixed cost slightly *worsens* the 2k→131k ratio even as both
improve.

### U15 — stage-profile at 2k — **DONE 2026-08-27: context-invariance confirmed, HC arms resolved**

**Outcome.** First 2k profile (build `c13e3bb`): every stage matches its 32k
value within noise except `compressor_indexer` (0.201 ms @2k vs 4.957 @32k)
— the long-context term collapses exactly as predicted. Short-context token
(28.61 ms gpu_busy) is dominated by context-invariant stages.
`DS4_TP_ABLATE=hcpre` removes only 0.76 ms despite the profile attributing
2.24 ms to hc_pre — the ~2.7× disagreement reproduces at 2k.
`DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE=1` costs **+0.58 ms** —
the fusion is a net win, not a free flip. See `BENCHMARKS-TP-PP.md` §U15.

Everything above is inferred from 32k/131k invariance. Sound, but indirect, and
the two largest unexamined blocks deserve a direct measurement at the context
they are being justified by.

```
DS4_METAL_GPU_STAGE_TIMESTAMPS=1   # ctx 2048, gen 128, same procedure as U12
```

**Report every stage, not a top-12.** U12's table listed 12 rows summing to
29.88 ms against a reported 32.70 ms total at 32k — so **~2.8 ms (9%) sat in
stages that exist as markers but were not reported**: `attn_norm`, `kv_path`,
`compressor_proj`, `compressor_update`, `compressor_quantize`,
`compressor_commit`, `indexer_compressor_*`. At 2k that residue is a larger
share than several of the stages we are chasing. It also now includes the two
new `q_path` sub-stages from `863e8fa`.

### Scoping agents dispatched 2026-08-27

Two agents are scoping the unpriced blocks against the 760 GB/s / ~21 TFLOP/s
roofs, with the byte-reconciliation and epoch rules as hard constraints:

- `docs/SCOPE-HC-STAGES.md` — the four HC stages, 4.56 ms / 18.7%.
- `docs/SCOPE-ATTNOUT-ROUTER-SHARED.md` — `attn_output`, `router`, shared
  expert; 6.50 ms / 26.7%. Briefed explicitly on the `attnout` phantom-target
  history and on why §4's "~100% of isolated bench" does not answer the
  roofline question.

### U12b — the two arms that turn 26.8% from "priced" into "explained" — **DONE 2026-08-27**

**Outcome.** Arm 1 (inverse-RoPE isolation): standalone RoPE dispatch is
**+0.133 ms @32k / +0.512 ms @131k** (norope vs default, across the 20 gathered
layers; ×21/20 for the indexed). Most of `attn_inv_rope` is attention, not
RoPE — U13's prize is ~0.5 ms at 131k, marginal per the under-0.3 ms gate.
Arm 2 (q_path split): q_a_kv_proj 2.14 / q_lora_norm 1.68 / q_path (q_b +
norm + RoPE tail) 1.86 ms — all context-invariant; q_path is mostly q_b, and
q_a_kv_proj + q_lora_norm are the two larger pieces to look at next. See
`BENCHMARKS-TP-PP.md` §U12b.

**Both are built and committed. Neither needs code on the rig side.**

**Arm 1 — isolate the standalone inverse RoPE from FlashAttention. Zero code.**

```
DS4_METAL_DISABLE_PRE_M5_ATTN_INV_ROPE_FUSE=1   # vs default, 32k and 131k
```

The flag (`ds4.c:22094`) forces the **gathered** branch onto the standalone
RoPE as well. Default vs this arm prices that dispatch across the 20 gathered
layers; scale by 21/20 for the indexed layers that pay it unconditionally
today. The delta **is** U13's prize — the 4.258 ms `attn_inv_rope` figure is
not, because most of it is the attention core.

Expect a *small* number. If the delta is under ~0.3 ms, **U13 is dead** and the
4.27 ms is simply attention doing its job at long context.

**Arm 2 — `q_path` split three ways.** `863e8fa` adds two boundaries, so the
existing stage-profile run now reports:

| new stage | contents |
|---|---|
| `q_a_kv_proj` | the fused q_a/kv Q8_0 pair projection |
| `q_lora_norm` | the q-LoRA RMS norm (+ fused KV RoPE where it fires) |
| `q_path` | Phase B: q_b projection + per-head RMS norm + RoPE tail |

Same command as U12 (`DS4_METAL_GPU_STAGE_TIMESTAMPS=1`, 32k and 131k) — the
markers are inert unless profiling is on, and stage names pass through
dynamically, so nothing else changes.

**What each outcome means.** `q_b` should be ≈0.744 GB/token; at the ~492 GB/s
§4 measured for that kernel it is ~1.5 ms. So:

- **remaining `q_path` ≈ 1.5 ms** → q_b is the whole of it, the per-head
  norm/RoPE is free, and the cost is in `q_a_kv_proj` — look there.
- **remaining `q_path` ≫ 1.5 ms** → the per-head norm/RoPE tail is the target,
  and the suspect is its `n_head × n_tok` grid (`ds4_metal.m:22589`) — **32
  threadgroups on 60 cores** at decode, one per head for 512 floats. That is
  the same underfill family as packed32-at-32-heads (−1.35 t/s), T2's turnover
  at 112, and U7's smem cap, and it would be the fourth sighting of one
  constraint.

**Do not size this by subtracting §4's `q_b` from U12's `q_path`** — different
epochs, and §14.6 records three wrong conclusions from exactly that shortcut.
That is why the markers exist.

#### U13 — arm the inverse-RoPE fuse on the indexed branch — **T10 — SIZING INVALIDATED, see U12 correction 1**

**T10 is filed in the row-8 cleanup batch at "~0.04–0.09 ms" and the stage it
targets is 4.27 ms.** That is the fourth item mis-sized the same way — T9, T6,
U10 and now T10 — all priced before the stage profile existed.

**The finding.** `fuse_attn_inv_rope` (`ds4.c:22093`) defers the inverse RoPE
into the FlashAttention reduce, so "the separate 64-threadgroup RoPE dispatch
disappears". It is armed **only in the `else` branch** — gathered/non-indexed
attention (`ds4.c:23531`). **The indexed branch never arms it**, so all **21
ratio-4 layers** pay the standalone dispatch every decode token, and those are
exactly the layers that dominate at long context.

**Why it was left this way is not recorded**, and that is the first thing to
establish — the comment explains the mechanism, not the restriction. The
indexed path's attention reduce may not own a whole head row the way the
gathered path does, in which case the fusion is not applicable and the honest
answer is a cheaper standalone kernel instead. **Determine that before writing
anything.**

**Prize if it is applicable:** up to ~4.27 ms minus whatever the gathered
branch's share already is — call it **2–4 ms, or 6–11% of the token**, which
would make it the largest banked win in this document. **Gated on U12**, which
prices the stage properly and says how much of it is the indexed branch.

**Correctness:** moving the rotation into the reduce changes FP32 accumulation
order, so this is not bit-exact. T2 bar — measurable win, top-1 preserved,
bounded Δlogit — and note `attn_inv_rope_fuse_armed` exists precisely because
the backend's consumed flag is process-global and a stale value would make an
indexed layer skip its required standalone RoPE (`ds4.c:22098-22101`). Any
change here must keep that guard intact.

This is where U7c points, and it is the strongest structural lever found so far.

At NSG=8 the 20,512 B of threadgroup memory breaks down as:

| buffer | bytes | purpose |
|---|---|---|
| `sk[64*128]half` | **16,384** | **F32 keys converted to half** |
| `sq[8*128]half` | 2,048 | query tile |
| `sqk[512]f32` | 2,048 | partial scores |
| `sw[8]f32` | 32 | head weights |

**`sk` is 80% of the budget and exists only because the cache is F32.** The
index compressed cache is allocated `n * DS4_N_INDEXER_HEAD_DIM * sizeof(float)`
(`ds4.c:17373`), so every key is converted F32→half on the way in. **Store it
F16 and the kernel can `simdgroup_load` keys straight from device memory:
threadgroup memory drops to 4,128 B and residency goes from 1 to 7 threadgroups
per core — at unchanged NK.**

That is the occupancy the kernel has never had, and U7c says occupancy is
exactly what the Ultra is short of.

*(Aside from U6's second-host run: on lanfear the Metal-allocated arms reach
791 GB/s against the mmap arms' 758–764 — a repeatable **~4%** placement
effect that mat does not show. So "allocation path costs nothing" is very
slightly too strong: it costs 0–4% depending on host, and the true roof is
~790. Real, but two orders of magnitude below what the placement hypothesis
predicted, and not worth a loader change.)* It also removes a per-key format conversion
and halves the cache's 352 MB/token of reads, though on the evidence neither of
those is the main prize.

**This is T9/U3 for the fourth time, and the first time with a mechanism worth
the work.** Proposed for bandwidth, killed for bandwidth (correctly — the
kernel is at 32 FLOP/byte), then revived for conversion cost, then for larger
NK. The real argument is none of those: **it buys 7× residency.**

**Risks to price, honestly.**
- `simdgroup_load` from device memory may be slower per access than from
  threadgroup memory, and staging exists partly to make the 8×8 matrix steps
  cheap. **The win is occupancy; the cost is per-access. Measure, do not
  assume.** A cheap intermediate exists: keep `sk` but halve it by staging 32
  keys instead of 64 — 8,192 B, 3 resident — which separates "residency helps"
  from "device-memory loads hurt" without an F16 cache conversion.
- Apple8 has **no matrix unit** — `simdgroup_matrix` lowers to FP32 ALUs
  (`speed-bench/tp_decode_investigation.md:178`) — so there is no hardware
  matmul path being given up here, which makes the device-memory variant less
  risky than it would be on a part with real tensor hardware.
- F16 keys change score values. The scores feed a **ranking**, and U2
  established the existing kernel already disagrees with a CPU reference at
  7.6e-3 from reduction association. Gate on selected-index-set equality
  against the F32 path, not on score equality.

**Order the experiment cheapest-first:** halve `sk` to 32 keys (no format
change, isolates the occupancy question) → if residency helps, do the F16 cache
and drop `sk` entirely.

#### U9 — decode sorts 32,768 scores to take 512 — **the largest untouched item**

**What decode actually does.** `ds4_gpu_indexer_topk_tensor` has a purpose-built
streaming selector, `kernel_dsv4_indexer_topk_stream512`, gated on:

```c
if (top_k == 512u && n_tokens >= 32u && ...)     // ds4_metal.m:18658
```

**Decode is `n_tokens == 1`, so it never qualifies.** It falls through to a full
`kernel_argsort_f32_i32_desc` block sort plus merge — **a complete bitonic sort
of 32,768 scores per layer to extract the top 512**, 21 layers per token. That
is O(n log²n) to select 1.5% of the input, and it explains why M2 measured
top-k (5.32 ms) as *more expensive than the scoring it selects from* (3.84 ms).

**Why the gate exists — and why removing it is not the fix.** `stream512`
dispatches `MTLSizeMake(n_tokens, 1, 1)` threadgroups of 512 threads: **one
threadgroup per token.** At decode that is a single threadgroup on a 60-core
GPU, i.e. 1/60th of the machine. Naively enabling it for `n_tokens == 1` would
be algorithmically better and wall-clock worse. **This is the same "cannot run
enough threads at once" constraint seen before** — the T2 sweep put the useful
oversubscription ceiling at ~96 threadgroups on 60 cores (1.6/core, turnover at
112), and the U7 NSG ladder is capped at one threadgroup per core by
threadgroup memory.

**So the fix is a genuine restructure, and it is the right kind.** Single-token
selection has to be spread across threadgroups rather than compressed into one:

1. **Two-pass radix / histogram select.** Histogram the score exponents across
   many threadgroups, prefix-sum to find the 512th-largest threshold, then one
   filtering pass. **O(n) with full grid occupancy**, versus O(n log²n) on one
   threadgroup.
2. **Per-block partial top-512 then merge.** Simpler, reuses `stream512`'s body
   per block, needs a small merge over `ceil(n_comp/block)` candidate lists.

Option 1 is the better ceiling; option 2 is the cheaper first experiment and
can reuse an existing kernel body.

**Prize.** 5.32 ms of a 36.36 ms token at 131k — **the largest single
sub-component of the largest stage**, and it is algorithmic rather than a
constant-factor tune, which is why it is worth more than any knob swept so far.

**T6 already spotted the gate and mis-sized it.** It is filed in "Lower tier …
individually sub-1%" as *"`stream512` top-k at `n_tokens == 1` — decode never
takes it; 0.16–0.46 ms"*. That figure predates M2 and the stage profile. **T6 is
withdrawn from the cleanup batch and superseded by U9** — the same mistake T9
made, and for the same reason: sized before the stage was priced.

**Correctness.** Top-k selection is exact if ties break deterministically. A
threshold/histogram select must handle the case where more than 512 scores tie
at the threshold value — break by ascending row index, matching whatever the
argsort currently produces, and test it directly with a synthetic all-equal
score vector. Compare selected-index sets against the argsort path with
`BENCH_DUMP`-style GPU-vs-GPU diffing, not against a CPU reference.

#### U8 — diagnose the MoE matvec's 54% — re-opened by U6

**Prize.** `routed_moe_folded` is 5.40 ms at 131k streaming at ~410 of 760
GB/s. Closing that gap is ~2.5 ms.

**This is not a T8 re-run.** T8 priced five specialisations *within* the
existing access pattern and found them worth nothing to −5%; that result stands
and those patches stay dead. What T8's wrong ceiling foreclosed is the
different question: **why does this kernel stream at 410 when the part does
760?** Deliverable is a diagnosis, not a patch — apportion the shortfall
between the 17-byte MXFP4 block granularity (unaligned against any natural
vector width), the expert gather/scatter, and the dequant interleave.

**Bracket built, and it already eliminates one of the three candidates.**
Rather than touch the MoE harness, the arm went into `bench_membw`, which is
validated against the streaming roof: `stream_blocks` walks fixed-size blocks
and consumes a 16-byte payload from each, so running it at stride 16 and
stride 17 isolates the MXFP4 layout cost from dequant arithmetic and from the
expert gather. At 17 bytes a simdgroup's 32 threads span 544 B at 32 odd
offsets, every load straddling a 16-byte boundary.

**M1 Max result: stride 17 costs 4.7%** — 348.2 GB/s against 365.2 aligned,
both ~91% of that part's roof. **So the 17-byte MXFP4 block granularity is not
what puts the matvec at 54%.** It is worth ~5%, not ~46%.

**Rig confirmation (mat, M2 Ultra, 2026-08-27): stride 17 costs 5.9%** — 713.2
GB/s against 758.0 aligned. The M1 Max finding holds at Ultra scale. So the
block granularity is worth ~6%, not the ~46% gap.

That leaves the expert gather and the dequant/accumulate path. Re-run the two
block arms on the rig to confirm the 4.7% holds at Ultra scale (done — 5.9%),
then attribute the rest between gather and dequant — the gather is 6 experts ×
3 tensors = 18 large contiguous regions per token, which *should* stream, so
**dequant and accumulation are now the leading suspects.** See
`BENCHMARKS-TP-PP.md` §U8.

#### U5 — n-gram speculation on the rig (R13 arms, re-prioritised)

Unchanged in content from R13, promoted in priority. `DS4_NGRAM_SPEC` is
implemented and has **never run on the rig**. It is the one queued item that
raises arithmetic intensity without requiring any kernel to get faster: verify
steps run `n_tokens > 1`, turning decode matvecs into small matmuls, which is
the textbook response to a latency-bound decode.

Run the existing R13 arms (inertness, correctness, decode A/B on repetitive vs
novel text, using `speed-bench/promessi_sposi.txt`). **Report acceptance rate
alongside t/s** — the t/s delta is meaningless without it. Note the known
interaction: with speculation on, `decode_splits = 1` (`ds4_metal.m:30001`), so
**T2 is inert on verify steps** and the two must not be productionised
independently without re-checking the combination.

### Correction — "99% busy" is a time measurement, not a throughput one — 2026-08-27

The stage profile shows the decode GPU busy 99% of the token. That does **not**
mean it is saturated, and the rig's power draw says it is not: **decode pulls
~30 W GPU / 90 W system against ~60 W GPU / 120 W system in prefill.** Same
GPU, half the power. A GPU stalled on memory latency is "busy" and cool.

**The T8 "near the ceiling" claim was wrong and it closed a workstream on
that error — now corrected and resolved by U6 (2026-08-27).** It read: "the
routed MoE matvec is bandwidth-bound at ~400 GB/s — **near the M2 Ultra
ceiling**." The M2 Ultra ceiling is **800 GB/s**; 400 GB/s was read as exactly
one M2 *Max* die, raising the single-die locality hypothesis. **U6 falsified
the placement hypothesis**: the part streams at ~760 GB/s on ds4's own `mmap`
path, so the 400.0 is the MoE matvec kernel at ~54% of achievable bandwidth,
not the ceiling and not one die. The workstream re-opens as matvec access-
pattern tuning, not as a loader/platform fix.

**The largest stage is nowhere near any roof.** `tests/bench_indexer_score
32768` on an M1 Max (400 GB/s, ~10.4 TFLOP/s FP32):

| | measured | of peak |
|---|---|---|
| FLOPs | 460 GFLOP/s | **4.4%** |
| K-cache DRAM | 14.4 GB/s (16.78 MB/dispatch) | **3.6%** |
| GPU busy | 1.01 ms/dispatch → 21.2 ms for 21 layers | ≈2× the rig's 10.5 ms ✓ |

The bench reproduces the production stage cost, and the kernel sits at ~4% of
*both* roofs — the signature of a latency-bound kernel, not a bandwidth-bound
one. Three arms (`DS4_METAL_DISABLE_INDEXER_LLT`, `DS4_METAL_INDEXER_LLT_NSG4`,
default) came back within 0.7% of each other: **the knobs we have do not touch
it.** (The harness also flags its own correctness check — worst relative error
7.6e-3 against the CPU reference, over its 1e-3 threshold. Probably benign for
a ranking use, but it should not be left unexplained.)

### The queue this reopens, ranked

**1. The decode indexer is fully replicated across TP ranks — the biggest
stage, computed twice.** `metal_graph_tp_split_indexer()` is gated on
`DS4_TP_PREFILL_SPLIT_INDEXER` and is **prefill-only** (`ds4.c:29056-29062`);
the decode call site passes the full `layer_n_index_comp[il]` with no rank or
world argument (`ds4.c:23241`). Split the row range per rank, take a local
top-k on each, exchange the two candidate lists and merge — **exact**, because
top-k of the union of two local top-ks is the global top-k. The exchange is
~4 KB/layer against a 10.5 ms stage. **Upside ~5 ms of 36 ms (14%).** This is
the TP-topology lever.

**2. T9 is mis-sized by an order of magnitude.** The indexer compressed cache
is allocated **F32** (`ds4.c:17373`): 32768 × 128 × 4 B = 16.78 MB/layer, **352
MB/token** across 21 ratio-4 layers at 131k. T9 (F32→F16) halves that. It is
currently in the "lower tier … individually sub-1%" bucket at "~0.2–0.4 ms" —
a figure written before the stage profile priced the stage at 10.5 ms.

**3. Restructure the kernel for K-cache reuse.** 64 heads score against the
same cache: 16.78 MB/dispatch with perfect reuse, **1.07 GB without**. The
gap between those two numbers is the whole optimisation.

**4. Speculative decode (`DS4_NGRAM_SPEC`, R13).** Implemented, never run on
the rig. Turns matvec into matmul and is the textbook fix for a latency-bound
decode.

**5. One test for the UltraFusion hypothesis** — a pure streaming-read kernel
on the rig. If it also caps at ~400 GB/s, placement is the problem, not the
kernels.

**The real lever is now the stage costs.** compressor_indexer (10.5 ms @131k,
the entire long-context growth), q_path (5.5), routed_moe (5.4), attn_inv_rope
(4.3). Any future decode work must target these stages, not stall.

**Post-mortem on how the floor was mis-framed, kept because it cost two wrong
turns.** The residual was first described here as "not compute at all," then as
splitting between unablated compute and stall, with "per-layer versus
per-token" offered as the discriminator. All three were wrong, and the stage
profile settled it in twenty minutes: **the gap is 0.31 ms. There was never any
stall to find.**

Two observations from that detour still hold and are worth keeping.

- **Nearly everything in a decode token is per-layer**, so that axis was never
  going to discriminate anything. Encoder boundaries are per-layer — gates per
  token are `DS4_N_LAYER * DS4_TP_GATES_PER_LAYER` (`ds4.c:60435`), so 86 gates
  = 43 × 2 and the 172 close/reopen events = 86 × 2 — but so are `router`,
  `shared`, `kv` and the compressor.
- **That makes the Qwen extrapolation more robust, not less.** Because the
  token is per-layer nearly end to end, the 48/43 scaling applies to
  essentially all of it, and it no longer depends on a compute/stall split that
  turns out not to exist. ~14.5 ms stands (see
  `docs/QWEN38-FLASH-NEXT-PORT-PLAN.md` §10) — and it is now a *compute* floor,
  which means the GDN design's dispatch-count objective is aimed at the wrong
  target.

**Method note.** Ablation and stage profiling answer different questions, and
reaching for ablation first cost the detour above. Ablation prices *what a
chain contributes in situ* but can only cover chains that ablate cleanly — M2
had to skip `router`, `shared` and `kv` for exactly that reason, and the
skipped set was the residual. Stage timestamps price *everything*, need no
semantically-wrong output, and sum to the measured busy time so the attribution
is closed by construction. **Profile first, ablate second.**

The ablation method (re-run `DS4_TP_ABLATE` chains at 32k and 131k, plus
`DS4_METAL_ABLATE_INDEXER_SCORE`/`_TOPK`, which are already wired into decode
at `ds4.c:23240`/`:23257` and had never been run at long context). Caveats
from `tp_decode_investigation.md:454-466`: `router` is unusable (ran 0.574 ms
*slower* while removing 92 dispatches), `kv`/`shared` are fusion rollbacks that
*add* dispatches, `compidx` has no call site and reports 0.

#### Lower tier

T6 (`stream512` top-k at `n_tokens == 1` — decode never takes it; 0.16–0.46
ms), T7 (fuse gate flag-set with fence-wait, 86 dispatches + 86 encoder
boundaries), T9 (indexer compressed cache F32 → F16; lossless by the e2m1
argument, ~0.2–0.4 ms + 176 MB), T10 (inverse-RoPE fuse, armed only on the
non-indexed branch so all 21 ratio-4 layers pay it at long context, ~0.04–0.09
ms), T13 (the `use_shared_kvpad` device-name gate, R13c above). T5 speculative.
T11 (ICB decode replay) only if the encoder-boundary slope surprises.

#### Structurally blocked — do not re-propose

`fuse_attn_out_hc` and `fuse_shared_down_hc` cannot fuse across a gate
boundary, and with `DS4_TP_DECODE_REPLICATE_ATTN` deleted that is now
permanent. Router + shared gate/up fusion is independently non-exact under TP
(hardcodes NSG=4 against TP's nsg=2). `packed32` flash reduce needs
`n_head == 64`; TP has 32, already reverted at −1.35 t/s. `parallel_full_ffn`
is IQ2_XXS/Q2_K only.

### R12 — two decode sweeps — **BOTH RESCOPED, read this before running**

A source audit found that both R12 arms were requested on stale premises. The
sweeps are still worth running; the reasons and the expected sizes are not what
was written.

#### R12a — command-buffer split schedule — reframed

**What I got wrong.** I wrote that "the adaptive tuning is disabled under TP,
so TP runs the flat 4/none while single-node reaches 2/32." Only half true, and
the half that matters is the other one:

- The **second** split has **no TP exclusion at all**. `ds4.c:27266-27271` says
  so explicitly: *"No TP exclusion here... excluding `tp_world == 2` a second
  time only cost the second split its whole effect under tensor parallelism."*
- It is instead capped at **`pos < 3328u`** — for everyone, TP or not.
- The **first** split's `tp_world != 2u` guards (`ds4.c:27144`, `:27156`) sit
  inside the `pos >= 128 && pos < 2048` and `pos >= 2048 && pos < 2816` windows,
  so they only bite at short context.

**Net: at any context ≥ 4k, TP and single-node run the same schedule — one
split after layer 4, two command buffers per token.** The TP exclusion I
highlighted is a short-context effect.

That makes the sweep *more* interesting, not less, but for a different reason:
at 131k **nobody** gets a second split, so setting `_SECOND_SPLIT_LAYERS`
explicitly engages a configuration the adaptive path never grants at long
context — untested for every configuration, not just ours. Arms unchanged
(4/0, 2/8, 2/16, 2/32, 3/12, 4/12), still bit-identical by construction.

**Hard prerequisite:** `DS4_METAL_FAST_SYNC=1` on both ranks. Without it
`ds4_gpu_tp_split_safe()` returns 0 and every arm collapses to one command
buffer — a flat null for the wrong reason. See R13a.

#### R12b — dispatch ballast — **demoted; it has already been run**

**What I got wrong.** I wrote that the instrument "has never been run" and
quoted a 4.4–10.6 µs band. Both come from the stale comment at
`ds4_metal.m:1317-1332`. `speed-bench/tp_decode_investigation.md:341-360` has
the actual in-situ measurements:

- **1021 dispatches/token** at ctx 512.
- Marginal cost **~1.9 µs**, from the cleanest arm (`kv` adds exactly 43
  dispatches for 0.081 ms). **Ballast itself gives 3.74 µs**; the mask arm 4.4.
  Its verdict: *"Use 1.9–4.4 µs, not 8.6."*
- And the conclusion I should have found before requesting this: **"Dispatch
  removal is not a productive strategy here."** 1021 × 1.9 µs = **1.94 ms**, and
  a realistic fusion campaign was scoped at 185 dispatches = 0.35 ms = +0.6 t/s.

So the per-dispatch hypothesis is bounded at **~8% at ctx 512 and ~6% at 131k
even if you removed every dispatch** — not the 10–25% I projected. The R11
"492 GB/s marginal vs 135 average" observation is still real, but it cannot be
worth what I claimed.

**Keep a reduced arm**: N ∈ {0, 2, 4} at **131072 only**, to confirm the
ctx-512 slope holds at long context where the indexer stage is live. One
sweep, not a campaign.

**Pair it with the genuinely unmeasured half — encoder boundaries.** Each of
the 86 gates/token does flag-set dispatch → `close_batch_encoder()` → *fresh*
encoder for the fence-wait spin → close again (`ds4_metal.m:10412-10434`,
`:10477-10506`): **172 encoder close/reopen events per token**, which ballast
cannot see because it emits its no-ops *inside* the open encoder. Build the
sibling instrument (N extra close/reopen pairs per decode layer, fit
`d(ms/token)/d(43N)`, ~20 lines) and run both in the same session. External
datapoint: upstream #590 measured **53.4 → 55.3 t/s** for removing *one*
encoder close/reopen from a checkpoint copy.

#### Prompt file — unchanged

Switch to `speed-bench/promessi_sposi.txt` as previously described.

## Closed work — results in `BENCHMARKS-TP-PP.md`, lessons kept here

Full tables live in the benchmark doc. What is kept below is only the reasoning
that would otherwise be re-derived — the corrections, the mechanisms, and the
things that turned out not to be true.

| run | outcome | the part worth remembering |
|---|---|---|
| Runs 1–3b, R1–R4 | A0 landed, +7.2% @131k | R4 found the argsort canon comparator was not token-count gated like its siblings; fixed in `6fa977c`. |
| R5 indexer split | bit-identical, +19.5% | Bit-identical *because* the split moves no FlashAttention block geometry — it only reassigns which rank computes an order-invariant per-row quantity. A0 perturbed (0.049) for the opposite reason. |
| R6 sizing | prize 311.2 ms × 20 layers | Two premises of this doc were wrong. See below. |
| R7 static-mixed split | bit-identical, +20.5% | The "mask needs a token-axis slice" blocker did not exist on the `pos0 > 0` path: that mask is rebuilt per call from the origin, so it slices itself. |
| R8 FA `nsg` | bit-identical, +7.3% | The standalone→rig transfer factor. See below. |
| R9 `nqptg` | scoped, no runs | A 2× standalone kernel discounts to ~+6–8% end-to-end. Prototype only if a restructure shows ≥1.5–2×. |
| R10a sub-gate | wash | The decisive argument was that a *prefill* flag moved *decode* by 10% in one arm — which proved the control run was bad, not that the flag worked. |
| R10b link ceiling | 4.4/4.1 GB/s | Its stated reason (UC SEND EPERMs >4096 B) was later disproved by M3. |
| R10c/d/e, R11 | see M0 | **All taken on a shard with `iogpu.wired_limit_mb = 0`.** Suspect. |
| R11 replicated attention | −8.4 to −11.8% | The gate wait is not idle — it is the window the peer's half of the attention runs in. Flag and code deleted in `f45b535`. |
| ANE / CPU offload | closed | See the section below; do not re-open. |

### R6's two corrected premises

**"`comp = 1024` on ratio-128 means their attention is small."** Backwards. It
is 255.3 ms against the ratio-4 layer's 182.8. Having *no indexer* is the
point: `top_k` caps a ratio-4 row at 512 + window keys, while a ratio-128 row
attends its entire compressed cache — and that count grows with context. Fewer
compressed keys, but all of them, beats more keys with only 512 read. The error
was reasoning from cache size instead of keys-read-per-row.

**"The static-mixed path needs a token-axis mask slice."** True of the
*zero-prefix* call site, not the `pos0 > 0` one, where `use_comp_mask` is
always 0 and the mask comes from `ds4_gpu_fill_mixed_decode_batch_mask()`.

### The standalone-to-rig transfer factor: ~0.63

R8 is the only result measured both ways. Backing the kernel speedup out of the
end-to-end numbers gives **~1.39× on M2 Ultra** from both the sweep and the
cold run independently, against **2.19× measured standalone on M1 Max**.
Discount future standalone M1 Max numbers accordingly and treat them as upper
bounds. R8's projection quoted the full 2.19× while only *verbally* flagging
the microarchitecture risk — flagging a risk is not pricing it in.

### The metric trap, five times over

`Σ(per-gate wait) ≈ token time` is **tautological** wherever gates bracket all
the work, because `wait` contains the compute. It is not evidence of overhead.
This doc has now drawn a wrong conclusion from it three separate times (R2's
"451 s of stall", R10e's "the floor is per-gate fixed cost", and the
"93% of 2048 prefill is gate time" comparison, which measured wait+wire against
a wire-only baseline). Related: a per-gate *ratio* does not bound a *total* —
"exchange is only ~9% of the run" was drawn from R2 and superseded once the
splits tripled the gate count.

When reading gate data, use: wire as a share of the token (absolute), and how
wait scales with context. Not the sum, and not the ratio alone.

### Heterogeneous compute (ANE / CPU) — closed, do not re-open

**Premise correction first:** the GPU is not bad at matmul. Scoring R6's stages
against their FLOP counts gives `q_path` **14.7 TFLOP/s**, `output_proj` 8.9,
`attention` ~2.4. The problem was one kernel at ~6× below what the same GPU
does on a plain GEMM in the same layer — hence R8/R9, not more silicon.

**ANE — four independent blockers.** Bandwidth roof 24–51 GB/s (M1) / ~145
(M5), not unified-memory bandwidth. Ultra does not aggregate: the driver
*"steers whole independent submissions to the least-busy engine die and never
exchanges tensor data between them"*. fp16 fails on our worst case — the guide
names output projection and down-projection as breaking in fp16 e2e, and `o_a`
is K=32768, with the M2-generation MAC saturating at 2¹⁵. One compiled program
per concrete shape, 0.23 ms dispatch floor. Its own head-to-head has ANE at 3.6
vs GPU 7.3 TFLOP/s on a K=4096 GEMM and classifies large-batch matmul as "GPU
regime throughout". Probed locally: ANE is a *candidate* for every op tried but
CoreML never schedules it, even for a canonical conv stack.

**CPU/AMX** measured at **1.32–1.79 TFLOP/s** on our shapes — ~20% more compute
against a GPU doing 14.7, and not free, since those cores drive the Metal
encoder and the RDMA service thread.

One cross-check worth keeping: AMX reproduces the GPU's `q_b`-vs-`o_a`
asymmetry (1.78 vs 1.32, same direction as 14.7 vs 8.9), so the K=32768
reduction being slower is a property of the shape on two independent engines.

## Protocol

### Prerequisites

Operational prerequisites (same commit, env symmetry, RDMA setup, **the
`iogpu.wired_limit_mb=120000` sysctl**, `DS4_METAL_FAST_SYNC`) live in
`BENCHMARKS-TP-PP.md` and are the authoritative copy. Two that bite here
specifically:

- **Env symmetry.** Anything that changes the per-layer gate count
  **deadlocks** the exchange if asymmetric; it does not degrade. The exception
  is `DS4_TP_DECODE_REPLICATE_ATTN`-class changes to `gates_per_token`, which
  the TP hello rejects at handshake (`ds4_tp.c:1386`).
- **The sweep is incremental.** `ds4_bench.c` advances the frontier from the
  previous point, so the `131072` row measures the 65536→131072 increment and
  its mean attended position is ~3N/4. Use it for A/B against itself; use a
  cold single point for anything cross-machine.

### Rig runbook — per-campaign operations

Operational lessons from R10–R12, M3, and the M0 re-baseline. The prerequisites
above say *what* must be true; this says *how the campaign actually runs* and
where it quietly breaks.

**After every reboot of either host, in this order** (all runtime-only, all
lost on reboot):

1. `./setup-rdma-net.sh` on **both** hosts (`~/Downloads/rdma-tb4/tests`), then
   `./check-roce-v2-gid.sh` — expect the IPv4-mapped RoCEv2 GID.
2. `sudo sysctl iogpu.wired_limit_mb=120000` on **both** hosts.
3. Re-stage campaign artifacts: macOS **`/tmp` is wiped on reboot** — driver
   scripts, the results tree, prompt copies, everything goes. Keep master
   copies in the Linux repo or `~/Downloads`; the standard prompt now lives
   persistently at `~/Downloads/promessi_sposi.txt` on both hosts.
4. Re-establish ssh: key auth between the two Macs (both directions) and from
   the Linux box. If a host was rebuilt, re-copy `id_ed25519.pub` into
   `~/.ssh/authorized_keys` and `ssh-keyscan` the peer. No `sshpass`/`expect`
   anywhere; `pexpect` works if a one-shot password prompt is unavoidable. The
   password must never land in a doc, commit, or durable file — `/tmp` askpass
   helpers are per-session and die with `/tmp`.
5. Rebuild and **verify the build is current**: `make -j ds4-bench` on both,
   then confirm with `strings ds4-bench | grep <a new env var from the commit>`
   plus the binary timestamp. A stale same-name binary is the classic silent
   wrong-run. Note: rsyncing the tree *without* `.git` leaves the Mac git tags
   stale — verify the tree (binary strings, `git diff --stat` on the Linux
   side), not the tag.

**Per-arm driver requirements** (the campaign drivers in `/tmp/*_driver.sh`
follow this; a new driver must too):

- `cd $REPO` before launching `ds4-bench` — it finds `metal/*.metal` relative
  to cwd, not the binary.
- Coordinator first, then worker via ssh; **verify the worker actually
  started** — count `ps aux | grep -F 'ds4-bench -m' | grep -v grep | grep -v
  'zsh -c'` on the worker ~5 s after launch — and kill the coordinator if it
  did not. The coordinator's `waiting for worker` has **no timeout**: a failed
  worker launch hangs the arm forever. (The `zsh -c` exclusion matters: the
  ssh wrapper shell self-matches the pattern.)
- The worker's log redirect targets a directory on **the worker host** —
  `mkdir -p` it there first *and* verify it (`test -d` over ssh). A missing
  dir makes the backgrounded worker shell die before exec'ing `ds4-bench`,
  silently.
- `--dump-frontier-logits-dir DIR` does **not** mkdir DIR — pre-create it, or
  the arm aborts into a header-only CSV.
- Stale-process cleanup before each arm (single-instance guard per host):
  `pkill -9 -f 'ds4-bench -m'` on both hosts, then verify zero matches.
- Progress monitoring: watch `--csv` row growth and
  `sudo powermetrics --samplers gpu_power -i 5000 -n 1` (prefill ~55–60 W,
  decode ~30 W, idle ~0.1 W). Under `nohup` the log is fully buffered and
  flushes only on **normal exit** — `kill -9` loses the whole buffer (all
  M0 worker logs came back empty for exactly this reason), and an empty log
  mid-arm is normal, not a hang. The coordinator exits normally so its log
  flushes and is the one to read.
- End-of-arm: wait for the coordinator process to **exit** (a fixed `sleep` +
  `pkill` will murder a live sweep — the 131k chunk takes minutes), `pkill`
  stragglers, copy the CSV into the results tree, and check row count ≥
  expected (header-only = early abort; log the tail for triage).
- Driver hygiene: **define bash functions before first use** — a call to an
  undefined function under `2>/dev/null 2>&1` dies silently as "command not
  found" (M0 first launch, 2026-08-26: the remote mkdir/launch helpers were
  called before their definitions, so the worker never started and the
  coordinator sat in `waiting for worker`). Remote operations must fail loudly
  (`|| { log; return 1; }`) and be verified (`test -d`, process count); never
  trust the exit code of a `… && nohup … & echo started` chain, which always
  exits 0.
- Timestamps in driver/watch logs are **Mac time**; the Linux box runs ~3 h
  ahead.

**Known infrastructure failure modes:**

| symptom | cause | action |
|---|---|---|
| IPv4-mapped GID at index 2 with a hole at index 1 after a link flap or `--reset`; `check-roce-v2-gid.sh` / `uc_pingpong` / `jaccl` fail RTR errno=1 | the GID table is **not stable across flaps** — a stale invalidated slot is kept and the new entry lands after it | ds4 is immune (it scans the table, `ds4_tp.c:780-790`) — run ds4 as the link check. For probes, find the real index with `ibv_devinfo -d rdma_enX -v` and pass it in; do not flap blindly hoping it compacts |
| UC ping-pong probe hangs after connect, no WC error on either side | **UC first-packet race**: closing the OOB TCP socket before the first data-plane SEND lets the initiator's first SEND arrive before the responder's recv is armed; UC has no retransmit, so one drop is permanent | keep the OOB socket open and `barrier()` before the first ping (documented in `uc_bench.c`) |
| UC probe hang ≈ n × per-iteration poll deadline (e.g. 2000 × 500 ms) | a single silently dropped UC WR — UC mode has no retry of any kind | the barrier fix above; for long runs add a spin watchdog and treat ≫p50 as a drop, not a stall |
| `bad bytes` ≈ 100 % of payload in one direction only | the responder replied with its own never-written send buffer instead of echoing what it received | the responder must `memcpy(send, recv, size)` before replying |
| `kill -9` does not clear a stuck rank; process sits in `U`/`U+` state | uninterruptible kernel wait (RDMA/IOKit teardown) | reboot the host, then run the post-reboot checklist above; a wedged pair never recovers in-process (`tp->failed` is sticky) |
| worker GPU `kIOGPUCommandBufferCallbackErrorTimeout` on a one-shot 200k-token cold prefill | 200k in one command buffer is beyond what the worker tolerates (observed once during prompt verification) | keep bench ctx ≤ 131072; a 200k-context run needs chunked prefill first |
| remote `sed -i` inside an ssh single-quoted string dies with "may not be used with stdin" | the quoting breaks sed's redirection | patch remote C files with a `python3` heredoc, or pull the file local and edit it |
| a note claiming >4096 B UC SEND fails with EPERM | stale comment in `uc_bench.c`; **false on this stack** — ds4 posts one 16,384 B UC SEND per gate all day, and the `uc_lat2` 16 KB single-WR runs are byte-verified both directions | single 16 KB WR is fine; do not chunk what ds4 sends as one WR |

### The correctness gate

Byte-identical is **not** a usable criterion on this rig: two identical control
runs already differ. The gate is **top-1 identity plus max |Δlogit| bounded
against a same-session control-vs-control baseline**.

Baselines observed so far: **0.0055** (R2-era, 16k prompt) and **0.688** (R11's
prompt, ~125× worse, with 5 of 7 frontiers in a degenerate repetition regime
where near-tied logits amplify anything). A candidate "inside the in-session
baseline" means little when the baseline is 0.688 — **fix the prompt before
trusting this harness on anything non-bit-identical.** Changes that come back
bit-identical (R5, R7, R8) need no baseline at all.

### Failure signatures

| symptom | meaning |
|---|---|
| gate wait hangs, or `tp: worker sync send failed` | asymmetric env on a gate-count flag |
| TP hello rejects at startup | asymmetric `gates_per_token` — the safe failure mode |
| `kIOGPUCommandBufferCallbackErrorTimeout` on the worker | watchdog kill; check whether it reproduces with the flag off |
| a knob shows ~0 effect at every context | confirm it reaches its predicate before believing the null — several knobs here are gated on context, quant, or `tp_world` |
| `wired_limit_mb is 0` anywhere in the log | discard the arm and re-run |

Any hang → **restart both ranks**; `tp->failed` is sticky and a wedged pair
does not recover in-process.

### Rollback

The three prefill splits are default-on; set each to `0` on both ranks to
disable without a rebuild. Everything currently under test
(`DS4_NGRAM_SPEC`, `DS4_TP_GATE_FASTPATH`, `DS4_METAL_FA_NSG=8`) is opt-in or
has an escape hatch, so no rollback needs a rebuild.
