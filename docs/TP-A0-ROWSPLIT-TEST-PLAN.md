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

### Sequencing — run in this order

Ordered by (information gained) / (rig time), not by expected gain. The first
two are minutes and one of them can invalidate three completed runs.

| # | Do | Time | Why first |
|---|---|---|---|
| 0 | **`sysctl iogpu.wired_limit_mb` on both hosts** — **done 2026-08-26**: it **was** `0` on both hosts; all 27 surviving R10/R11 coordinator logs carry the `wired_limit_mb is 0` warning, so **R10d/R10e/R11 decode numbers are suspect** (flat, stall-shaped, neither-bound profile = lazy paging). Set to `120000` on both; prerequisite added to `BENCHMARKS-TP-PP.md`. | — | If it was 0 after the 2026-08-26 reboot, the shard was paging lazily and **R10d, R10e and R11 are all suspect**. Set it to `120000` on both and note it in the prerequisites (done — `BENCHMARKS-TP-PP.md`). |
| 0b | Confirm `DS4_METAL_FAST_SYNC=1` is in the **`ds4-server`** launch path — **resolved 2026-08-26: nothing to confirm**. There is no `ds4-server` launch path in the repo and no production `ds4-server` deployment on the bench hosts; the knob is bench-only until a server path exists | — | Bench always sets it; production may not. Worth ~186 µs of a 508 µs gate, and without it the decode command-buffer split is a no-op under TP. |
| **0c** | ~~M0 — re-baseline on a pinned shard~~ — **done 2026-08-26. Decode survives within 1%; the gate exchange halved at 131k (49.7 → 24.8 µs); prefill moved +6–25% but the cause is confounded. T1 re-sizes down to ~+2%.** | — | **Step 0 came back positive: the limit was `0` on both hosts.** Everything measured on 2026-08-26 was on a lazily-paged shard. This re-run decides how much of R10c/R10e/R11 survives and re-sizes T1. Nothing else should run first. |
| 1 | **M3** — **done 2026-08-26** (`uc_lat2`, byte-verified, n=2000/arm): half-RTT **8.0 µs (4 KB)** / **14.5–15.5 µs p50 (16 KB, single WR)** — both ≪20 µs → **T1 open** (~30 µs/gate ≈ 2.5 ms/token at 131k). Single 16 KB UC WR confirmed working on this stack. One transient first-ping UC drop seen; T1 needs a re-arm/retry path. Recorded in `BENCHMARKS-TP-PP.md`. | — | Decides T1, the largest sized item, before any code is written. ≲20 µs half-RTT → ~2.5 ms/token available; ~45 µs → T1 closes. |
| 2 | ~~Env battery T2 + T3 + T4~~ — **done 2026-08-26. T2 +0.9% @131k and monotonic to the top of the tested range (not peaked); T3 mixed; T4 default already optimal.** Follow-up: sweep T2 at 28/31. | — | One small real win, two nulls. |
| 3 | ~~T8 pricing~~ — **done 2026-08-26. The specialisation ladder is worth nothing to −5%; the generic path is faster. T8 is dead.** The routed MoE matvec is bandwidth-bound at ~400 GB/s. | — | 30 minutes of pricing saved five kernel patches for a negative. |
| 4 | ~~M2~~ — **done 2026-08-26.** Ablation at 32k/65k/131k. **Routed MoE is the dominant decode stage (~22–25%)**; **the indexer is the long-context story** (score +5.9→12.2%, topk +7.9→17.7% as ctx 32k→131k — the largest attributed slice of the 11.1 ms); attnout ~9–10%, qb/attncore ~5%, hcpre ~2%. **Refined by the stage profile (done below): the indexer cost is `compressor_indexer` (+5.06 of +5.93 ms growth), and the ~13 ms residual is real compute, not stall.** | ~2 h | The 11.1 ms/token of unattributed long-context decode growth — the largest unknown in the document. |
| 4b | ~~Stage profile at 32k/131k~~ — **done 2026-08-26.** Decode gpu_busy 30.43/36.36 ms, **gap ~0.31 ms (1%) — decode is ~99% GPU-busy, no stall.** Stage sum = busy exactly (100% attribution). compressor_indexer = the long-context term (+5.06 ms). **The ~13 ms M2 residual is real compute (unablated stages), not idle.** | ~20 min | Bounds router/shared/kv/compressor and decomposes the ~13 ms floor without ablation. |
| 5 | ~~R12a~~ split-schedule sweep + **R12b** reduced ballast arm + **encoder-boundary instrument** — **R12b and encoder-boundary MOOT** (probe stall; stage profile shows no stall). R12a split-schedule still valid if wanted. | ~1.5 h | R12b is now a confirmation, not a discovery; the encoder-boundary half is the genuinely unmeasured one. |
| 6 | **R13 n-gram rig arms** (inertness / correctness / decode A/B on repetitive vs novel text) | ~1 h | Independent of the above; run whenever convenient. |
| 7 | ~~T1~~ — **done 2026-08-26: dead.** `DS4_TP_GATE_FASTPATH` is a wash (±0.6% decode, no gate-exchange change) and is **not bit-identical** (logits shift up to 2.3, top-1 preserved). Stay default-off. ~~T8 port~~ — dead, see row 3. | — | Both are code, both are gated on a measurement above. |
| 8 | Cleanup batch: T6, T9, T10, T13 | — | Small, low-risk, individually sub-1%. |

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

#### T2 follow-up — proceed, but size it honestly — **2026-08-27**

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
- Flat or regressing → set the default to the measured peak and close T2.

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
