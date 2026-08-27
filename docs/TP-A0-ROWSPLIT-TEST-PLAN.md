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
| 2 | ~~Env battery T2 + T3 + T4~~ — **done 2026-08-26. T2 +0.9% @131k and monotonic to the top of the tested range (not peaked); T3 mixed; T4 default already optimal.** Follow-up: sweep 28/31 — **done 2026-08-26, peaked at 24 (28/31 regress); default set to 24, T2 closed.** | — | One small real win, two nulls. |
| 3 | ~~T8 pricing~~ — **done 2026-08-26. The specialisation ladder is worth nothing to −5%; the generic path is faster. T8 is dead.** The routed MoE matvec is bandwidth-bound at ~400 GB/s. | — | 30 minutes of pricing saved five kernel patches for a negative. |
| 4 | ~~M2~~ — **done 2026-08-26.** Ablation at 32k/65k/131k. **Routed MoE is the dominant decode stage (~22–25%)**; **the indexer is the long-context story** (score +5.9→12.2%, topk +7.9→17.7% as ctx 32k→131k — the largest attributed slice of the 11.1 ms); attnout ~9–10%, qb/attncore ~5%, hcpre ~2%. **Refined by the stage profile (done below): the indexer cost is `compressor_indexer` (+5.06 of +5.93 ms growth), and the ~13 ms residual is real compute, not stall.** | ~2 h | The 11.1 ms/token of unattributed long-context decode growth — the largest unknown in the document. |
| 4b | ~~Stage profile at 32k/131k~~ — **done 2026-08-26.** Decode gpu_busy 30.43/36.36 ms, **gap ~0.31 ms (1%) — decode is ~99% GPU-busy, no stall.** Stage sum = busy exactly (100% attribution). compressor_indexer = the long-context term (+5.06 ms). **The ~13 ms M2 residual is real compute (unablated stages), not idle.** | ~20 min | Bounds router/shared/kv/compressor and decomposes the ~13 ms floor without ablation. |
| 5 | ~~R12a~~ split-schedule sweep + **R12b** reduced ballast arm + **encoder-boundary instrument** — **R12b and encoder-boundary MOOT** (probe stall; stage profile shows no stall). R12a split-schedule still valid if wanted. | ~1.5 h | R12b is now a confirmation, not a discovery; the encoder-boundary half is the genuinely unmeasured one. |
| 6 | **R13 n-gram rig arms** (inertness / correctness / decode A/B on repetitive vs novel text) | ~1 h | Independent of the above; run whenever convenient. |
| 7 | ~~T1~~ — **done 2026-08-26: dead.** `DS4_TP_GATE_FASTPATH` is a wash (±0.6% decode, no gate-exchange change) and is **not bit-identical** (logits shift up to 2.3, top-1 preserved). Stay default-off. ~~T8 port~~ — dead, see row 3. | — | Both are code, both are gated on a measurement above. |
| 8 | Cleanup batch: T6, T10, T13 (**T9 removed — promoted to U3**) | — | Small, low-risk, individually sub-1%. |
| **9** | **U1 — streaming-read ceiling on the rig**, one rank, no TP | ~15 min | **done 2026-08-26** — pinned at ~408–410 GB/s across maps 3.19–25.5 GiB. Initially read as one M2 Max die / platform; **superseded by U6 2026-08-27** — the part streams at ~760 GB/s on the same `mmap` path, so U1's plateau is the MoE matvec kernel's access pattern, not the platform. Kernel headroom re-opens. **Run first.** Decides whether ~400 GB/s is the platform or our kernels, and therefore how to read every number below. Also re-scores T8. |
| **10** | **U2 — indexer-score roofline + working-set sweep** | ~30 min | **done 2026-08-26.** Latency-bound per byte — 39.7 GB/s at 65536 = ~10% of the 400 GB/s platform, one threadgroup per row; GPU-busy ~linear in `n_comp`. **U3 will not pay** (honest prize near zero); go to the restructure. Correctness flag (worst rel 7.6e-3, row 17391) = expected FP32 tree-vs-sequential tolerance, benign. The largest stage sits at ~4% of *both* roofs. |
| **11** | **U3 — T9 re-sized: indexer cache F32→F16** | ~1 h | **gated off by U2 2026-08-26.** U2 reported latency-bound, so per its own gate the honest prize is near zero — do not start the F32→F16 code work; go to the restructure. 352 MB/token at 131k, halved. Was filed sub-1% at "0.2–0.4 ms"; the stage is 10.5 ms. **Gated on U2.** |
| **12** | **U4 — TP row-split the decode indexer** | ~3 h | The biggest stage is computed *twice* today, once per rank. ~5 ms of 36 ms. Largest single item in this document. |
| **13** | **U5 — R13 n-gram arms, re-prioritised** | ~1 h | Raises arithmetic intensity without requiring any kernel to get faster — the structural answer to a latency-bound decode. |
| **14** | **U6 — why ~400 GB/s on an 800 GB/s part** — allocation-path + concurrency arms | ~1 h + harness | **done 2026-08-27 — roof is ~760 GB/s, not 400.** `bench_membw` on mat: all seven arms 752–762 GB/s (94–95% of 800), within 1%. Allocation path costs nothing; concurrency costs nothing. The part saturates both dies on ds4's own `mmap` path. **U1's ~408–410 is the MoE matvec kernel (~54% of achievable), not the platform** — kernel headroom re-opens, loader exonerated. |

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

#### U10 — drop the `sk` staging buffer: 1 → 7 threadgroups resident per core

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
