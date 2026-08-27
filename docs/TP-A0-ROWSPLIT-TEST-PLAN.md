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
| 0b | Confirm `DS4_METAL_FAST_SYNC=1` is in the **`ds4-server`** launch path | 2 min | Bench always sets it; production may not. Worth ~186 µs of a 508 µs gate, and without it the decode command-buffer split is a no-op under TP. |
| **0c** | **M0 — re-baseline decode with the wired limit set** (sweep + cold 131k + `DS4_TP_GATE_PROFILE`) | ~1 h | **Step 0 came back positive: the limit was `0` on both hosts.** Everything measured on 2026-08-26 was on a lazily-paged shard. This re-run decides how much of R10c/R10e/R11 survives and re-sizes T1. Nothing else should run first. |
| 1 | **M3** — **done 2026-08-26** (`uc_lat2`, byte-verified, n=2000/arm): half-RTT **8.0 µs (4 KB)** / **14.5–15.5 µs p50 (16 KB, single WR)** — both ≪20 µs → **T1 open** (~30 µs/gate ≈ 2.5 ms/token at 131k). Single 16 KB UC WR confirmed working on this stack. One transient first-ping UC drop seen; T1 needs a re-arm/retry path. Recorded in `BENCHMARKS-TP-PP.md`. | — | Decides T1, the largest sized item, before any code is written. ≲20 µs half-RTT → ~2.5 ms/token available; ~45 µs → T1 closes. |
| 2 | **Env battery: T2 + T3 + T4** at 32k and 131k, `DS4_NGRAM_SPEC` **off** | ~1.5 h | Three knobs, no code, no correctness arm needed beyond top-1 + bounded Δlogit. T3 pre-screens free on the dev box with `tests/bench_indexer_score`. |
| 3 | **T8 pricing** — `tests/bench_moe_mxfp4_decode 256` with the five disable envs, **on the rig** | 30 min | Prices four layers of MoE specialisation before writing the port. Model-free harness. |
| 4 | **M2** — ablation battery at 32k and 131k, incl. the never-run indexer ablations | ~2 h | The 11.1 ms/token of unattributed long-context decode growth — the largest unknown in the document. |
| 5 | **R12a** split-schedule sweep + **R12b** reduced ballast arm + the encoder-boundary instrument | ~1.5 h | R12b is now a confirmation, not a discovery; the encoder-boundary half is the genuinely unmeasured one. |
| 6 | **R13 n-gram rig arms** (inertness / correctness / decode A/B on repetitive vs novel text) | ~1 h | Independent of the above; run whenever convenient. |
| 7 | T1 implementation if step 1 says yes; T8 port if step 3 says yes | — | Both are code, both are gated on a measurement above. |
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

### M0 — re-baseline decode; the 2026-08-26 numbers were taken on a paged shard

**Step 0 came back positive.** `iogpu.wired_limit_mb` was **0 on both hosts**,
and all 27 surviving R10/R11 coordinator logs carry the `wired_limit_mb is 0`
warning. With it at 0, `ds4.c:59886` takes `ds4_gpu_model_residency_skip(1)`
and TP never declares the 76.7 GiB shard resident, so Metal validates/pages it
per use instead of pinning it.

**Why this is not a small correction.** A shard that pages produces a decode
profile that is stall-shaped, flat with context, and neither bandwidth- nor
compute-bound — which is *the exact profile* R10c/R10e/R11 were built to
explain. The conclusions drawn from it may be describing a misconfiguration
rather than the engine.

| result | status now |
|---|---|
| R10c decode is stall-bound (~30 W vs 55.5 W prefill) | **suspect** — page-fault stalls look identical to gate stalls in a power trace |
| R10d `DECODE_NWG` plateau 8–32 | **shape probably survives** (all arms shared the defect), absolutes do not |
| R10e per-gate waits (292.6 / 401.4 µs) | **suspect** — residency validation inflates exactly this measure |
| R11 replicated attention, −8.4 to −11.8% | **sign probably survives, magnitude suspect** — replication adds ~1.6 GB/token of weight reads, which a paged shard punishes harder than a pinned one |
| R1–R9, all prefill | **unaffected** — pre-date the reboot; and prefill's 86–91% residency is not a paging signature |
| The +65.8% sweep / +54.9% cold prefill arc | **unaffected** |

**A concrete hypothesis this re-run tests.** R9 measured row-gate wire at
**29.9 µs**; R10e measured **49.7 µs** for identical 16,384-byte payloads,
+66%, which the bench flagged as discordant and widened the error bar for. R9
is 2026-08-25, before the reboot; R10e is 2026-08-26, after. **If the gate wire
returns to ~30 µs with the limit set, the discrepancy was the wired limit** —
and that halves T1's prize (see below). If it stays at ~50 µs, the spread has
another cause and T1 is worth what M3 says.

**Do this before anything else in the table.** Re-run: the 7-point sweep, cold
131k, and `DS4_TP_GATE_PROFILE=1` at 2048 and 131072. That is one session and
it re-validates or discards three campaigns.

**Also worth deciding after it lands:** whether R11 deserves a re-test.
`DS4_TP_DECODE_REPLICATE_ATTN` was deleted in `f45b535` on the strength of a
measurement now known to be taken on a paged shard. The mechanism argument
("the gate wait is the window the peer's half runs in") is not obviously
residency-dependent, so I would not restore it speculatively — but if M0 moves
decode materially, reverting that one hunk and re-running is cheap.

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

**M3 result (2026-08-26): T1 is open, but its size now depends on M0.**
Half-RTT **8.0 µs (4 KB)** and **14.5–15.5 µs p50 (16 KB, single WR — the gate
shape)**, both well under the 20 µs threshold, byte-verified ×2000 per arm.

Sizing, and the caveat that matters: against R10e's 49.7 µs the software
overhead is ~35 µs/gate = **3.0 ms/token, +9% at 131k**. Against R9's 29.9 µs
for the *same payload* it is ~15 µs/gate = **1.3 ms/token, +3.7%**. M0 decides
which wire number is real — see above. Treat T1 as "open, worth 1.3–3.0
ms/token" until then, and do not quote the upper end.

One implementation note from the probe: a single 16,384 B UC SEND WR is
confirmed working on this stack, so the gate needs no chunking.

**No retry path needed** (owner's call, 2026-08-26): UC on this Apple stack is
treated as lossless. The single first-ping drop M3 saw is the documented UC
first-packet race in the probe's own OOB setup, not steady-state loss —
`uc_bench`'s skeleton exists precisely because closing the OOB socket early
triggers it. ds4's independent evidence is stronger than the probe's: R10e ran
2,752 big gates and 11,008 row gates per cold run, all day, with no lost gate.
So T1 does not carry a re-arm/retry design; `timeout_sec` remains the
correctness backstop. Full table in `BENCHMARKS-TP-PP.md` (M3 section).

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

#### M2 — attribute the 11.1 ms — the largest unknown

Every decode stage number in this tree is from **ctx 512, where the indexer
path is entirely inactive**. That token is 24.06 ms; ours at 131k is 35.55.
**11.1 ms/token — 31% of the long-context token — has never been attributed**,
and code reading accounts for maybe 3–4 ms of it.

Re-run the ablation battery at 32k and 131k (`DS4_TP_ABLATE` chains, plus
`DS4_METAL_ABLATE_INDEXER_SCORE`/`_TOPK`, which are already wired into decode
at `ds4.c:23240`/`:23257` and have never been run at long context). Caveats
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
