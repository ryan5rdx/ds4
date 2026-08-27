# TP2 prefill: row-split-at-pos0>0 (A0) rig test plan

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
MoE port is worth writing. Nothing below step 3 should start before they land.

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

**M3 result (2026-08-26):** half-RTT **8.0 µs (4 KB)** and **14.5–15.5 µs p50
(16 KB, single WR — the gate shape)**, both under the threshold. **T1 is open.**
A single 16,384 B UC SEND WR is confirmed working on this stack (byte-verified
×2000 per arm), and one transient first-ping UC drop across the whole battery
means the implementation needs a re-arm/retry path. Full table in
`BENCHMARKS-TP-PP.md` (M3 section).

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

### R11 — halve the decode gate count (`DS4_TP_DECODE_REPLICATE_ATTN=1`) — **DONE, 2026-08-26, NEGATIVE: decode −8.5 to −11.8%, recorded in `BENCHMARKS-TP-PP.md`**

**What it cost, and why that number is the finding.** Decode fell uniformly:
B/A = 0.882 / 0.912 / 0.912 / 0.913 / 0.916 / 0.913 / 0.916 across the sweep.
In per-token terms that is **+3.28 ms at 2048 and +3.24 ms at 131072** — flat
to within 1% across a 64× context range. Flat means the cost is the duplicated
*weight* traffic (+1.6 GB/token/node of `q_b`/`o_a`, which does not scale with
context) and **not** the full-head KV re-read (which would). At ~500 GB/s that
+1.6 GB is ~3.2 ms: the measured cost is the duplicated weights, essentially
in full.

**Now the part that matters. Removing 43 gates saved approximately nothing.**
Those gates carried 43 × 293 µs ≈ **12.6 ms/token** of wait at 2048, against a
24.5 ms token. Had that been transport/sync overhead, deleting it would have
paid for the 3.2 ms of extra traffic several times over. Net was +3.25 ms,
i.e. the ~12.6 ms of removed wait bought ~0 — because a TP gate wait is not
idle, it is **the window in which the peer rank's half of the attention runs**.
Replication does not remove that work, it moves it onto both ranks and
serialises what used to be two half-attentions in parallel.

**This falsifies R10e's headline, and the error was mine.** R10e concluded
"short-context decode is entirely per-gate fixed cost" from 86 × ~284 µs ≈
24.4 ms ≈ the 41 t/s floor. That sum is the same tautology this doc has now
flagged four times: Σwait ≈ token time wherever gates bracket the work, because
`wait` *contains* the compute. I wrote that warning into the R10e request and
then used the quantity it warns about to argue the floor was overhead — and
built R11 on it.

The corrected reading, and it closes the subject: subtracting the duplicated
traffic from the measured delta leaves **~0 recoverable per-gate sync
overhead**. The decode gate mechanism is not costing us anything worth
reclaiming; the 86 gates/token are measuring compute. **Decode is closed as a
TP-structural problem.** What remains is making the compute itself cheaper —
kernels, quantisation, or fewer active parameters — not restructuring the
exchange.

**Two harness findings worth acting on before the next campaign.**

*The correctness gate has degraded ~125×.* In-session control-vs-control max
|Δlogit| is **0.688**, against the R2-era 0.0055 — at logit magnitude ~27 that
is 2.5% relative, and 5 of 7 frontiers fall into a degenerate "of/and"
repetition regime where near-tied logits amplify any perturbation. The
candidate (0.665) passing "inside the in-session baseline" is therefore a very
weak pass. The degenerate prompt is the likely cause, but `dedc830`
(compressor-frontier rewind) is also new in this build and the frontier chain
exercises restore paths, so **confirm the cause rather than assume it** before
this harness gates anything non-bit-identical again. R5/R7/R8's bit-identical
results are unaffected — 0 differing logits needs no baseline.

*Warm-sweep prefill is ±5% with occasional single-point ~13% dips*, evidenced
by two opposite-direction outliers (32768 and 65536) in runs where the flag
cannot reach prefill. Same-day A/B is the only clean comparison. This does not
disturb the large deltas in the arc (R7 +20.5%, R8 +7.3% at 131k), but it does
mean the small ones — R8's +1.5% at 4096, +2.8% at 16384, and R10a's ±1%
agreements — sit at or inside the noise floor and should not be quoted as
individually resolved.


Built on `upstream-metal-wins`. Opt-in, default off, must be set on BOTH ranks.
Builds warning-clean; `ds4_test --server`, `ds4_agent_test`,
`ds4-eval --self-test-extractors` and `test_engine_mgpu_placement` (123/123)
all pass.

R10e closed out every knob and left exactly one lever: **the 86 gates/token**.
Short-context decode is entirely per-gate fixed cost (86 × ~284 µs ≈ 24.4 ms
≈ the measured 41 t/s floor), wire is only ~12% of gate time at both contexts,
and NWG is already optimal. This is the design change that moves the count.

**What it does.** Replicates the whole decode attention block instead of
splitting it, so the per-layer ATTN gate disappears and the schedule goes from
2 gates/layer to 1 — `start = DS4_TP_GATE_FFN, step = 2, per_token = 43`.

**The precedent is already in the same function.**
`ds4_engine_tp_gate_schedule()` gives GLM exactly this shape today ("One FFN
gate per sparse layer"), so neither the schedule form nor the transport's
arithmetic-progression slot mapping needs changing. Prefill is untouched: it
uses `ds4_gpu_tp_big_gate_encode()` with its own sequencing, not the row-gate
schedule.

**This is a trade, and it can lose.** Correcting my own framing when I proposed
it — this is not "one branch". Three decode splits have to come off together
(the head split, the independently group-sliced output projection at the
`tp_attn_decode_split` branch, and the gate itself) or the gate slot carries a
partial nobody sums. All three now key off one predicate. The cost is that each
rank runs the full attention block: roughly **+1.6 GB/token/node** of weight
traffic on Flash, because `q_b` and `o_a` are 32768-wide.

The bet is R10c: decode runs at ~54% of prefill power, so it is stall-bound,
and added reads should land in existing bubbles while removed gates take real
wall time. If decode were bandwidth-bound this would be strictly worse. **I
would not be surprised by a negative result** — record it either way, because a
clean negative closes the last lever and says the 86-gate structure is load
bearing.

**R11a — inertness.** Flag unset must reproduce the current arm exactly; the
default path is untouched.

**R11b — correctness** (16384, greedy). Expect **not** bit-identical, unlike
R5/R7/R8: replication changes the attention output from `rank0_partial +
rank1_partial` to a single full-width computation, so the summation order
differs. Judge it on the R2-era criterion — top-1 identity plus max |Δlogit|
bounded against the ~0.0055 control-vs-control baseline. A large divergence
means a genuine bug, not accumulation order.

**R11c — throughput**: decode at 2048 and 131072, plus a full sweep to confirm
prefill is unmoved. Decode is the metric; prefill should be flat to within
noise, and if it is not, something leaked out of the decode path.

No projection recorded, deliberately. The two terms — per-gate fixed cost saved
and duplicated attention traffic added — are each uncertain by more than their
difference, and R8 is the standing reminder of what happens when a shaky
estimate gets quoted as a number. Direction is genuinely unknown; that is why
it is worth one run.

**Asymmetry fails safe here**, unlike the prefill split flags:
`gates_per_token` is exchanged in the TP hello and a mismatch is rejected at
handshake (`ds4_tp.c:1386`) rather than deadlocking mid-run.

### R10 — the gate path, and the first hard look at decode — **DONE, 2026-08-26, recorded in `BENCHMARKS-TP-PP.md`**

**Two readings to fix before they propagate.**

**The "93% of 2048 prefill wall time is BIG-gate wait+wire, versus 18.5% at
131k" comparison is apples-to-oranges.** The 18.5% figure is **wire only**; the
93% is **wait + wire**, and `wait` contains the layer's compute. Σ(wait+wire) ≈
wall time is near-tautological wherever gates bracket all the work — at 131k
the same sum is 319 s of 325 s, i.e. **98%**, *higher* than 2048's 93%. The
comparable number is wire alone: 86 × 13.07 ms = 1.12 s of 4.96 s = **~23% at
2048** against 19.1% at 131k. Similar, slightly worse at short context. No R10
conclusion changes, but "the gate *is* the prefill at 2048" reads as an
overhead finding when it is a bracketing artifact — the third instance of this
exact trap in this doc, after the two corrections above.

**R10c confirms its prior via a proxy, not directly.** These boxes report GPU
power, not residency, so the 86–91% prefill residency figure has no
counterpart. Decode at 30 W against prefill's 55.5 W is 54% of prefill power;
multiplied through prefill's ~88% residency that is a **~48% residency
equivalent** — the top edge of the predicted 30–50% band, so confirmed but only
just, and power is not linear in residency under DVFS. The qualitative finding
(decode is not compute-saturated) is solid; the number is directional.

**And one prediction that was simply wrong.** R10d was requested on the
suspicion that `DECODE_NWG`'s bucket heuristic was "exactly the class of
tuned-once constant `nsg=8` turned out to be". It is not: the default 32 sits at
the top of the plateau on **both** contexts (40.91 @2k, 0.03% off best @131k).
The knob is a real lever — NWG=2 costs 14% — with nothing left to tune. One for
two on suspecting heuristics; worth remembering before the next such request.


**No code.** Everything here is an existing knob or a measurement. R9's own
arithmetic put a kernel rewrite at +6–8%; the R9 gate profile put **BIG-gate
wire at 60 s of a 325 s cold prefill — 18.5%, on the critical path and not
overlapped**. That is the larger target, and it is also the structural reason
PP wins at long context: PP moves activations twice per pipeline pass, TP moves
them once per layer per chunk, so TP's transfer cost scales with layers ×
chunks and PP's does not.

**R10a — re-test `DS4_TP_SUBGATE_PIPELINE=1`** (both ranks; full sweep + cold
131k against the current arm). Documented as "measured net-negative on the M5
Max pair", but that predates this entire workstream, when BIG wait:wire was
**8.0 : 1**. It is now **4.3 : 1** with per-gate wait halved (190 → 93 ms).
Overlapping sub-chunks is worth much more when wire is a larger share of the
gate, so the regime it was rejected in no longer exists. Cheapest possible
probe of an 18.5% target: one flag, no code. If it is still net-negative,
record it and the sub-gate avenue closes for good.

**R10b — link ceiling at the real message size.** Effective BIG-gate wire is
2.8–3.1 GB/s (67,108,864 B / 21.8 ms) against a link we treat as ~5 GB/s.
Before anyone tries to overlap the transfer, establish whether it can simply go
faster: run `uc_pingpong` (or the existing RDMA test harness) at **64 MiB**
message size, not the default, and record achieved GB/s each way. If the link
delivers ~5 GB/s at that size, the gate path is leaving ~40% on the floor and
staging/window tuning is back on the table — the workstream this doc closed
earlier was closed on the argument that exchange was only ~9% of the run, which
the 18.5% figure supersedes.

**Decode.** Prefill has had all the attention; decode has been "untouched" in
every arm since R3 and has never been profiled on its own. It is now the
weaker half of the story in absolute terms — 28.13 t/s at 131k against 367
prefill — and it is where TP's remaining advantage lives (+37–48% over PP at
every context). Three cheap measurements, in priority order:

**R10c — decode residency and power** (`powermetrics --samplers gpu_power`
during the *generation* phase only, at 2048 and 131072). **This has never been
measured** — every residency figure in this doc is from prefill. It decides the
other two: prefill sits at 86–91%, and if decode sits far below that, decode is
stall-bound and the gate chain is the target; if it is also ~90%, decode is
compute-bound and the kernels are.

Prior expectation, so it can be scored: **low**, 30–50%. Decode moves ~3.3 GB
per token per node (experts + attention weights + KV) which is ~4 ms at 800
GB/s, against a measured 35.5 ms/token at 131k — about 12% of bandwidth and,
per the standing note, ~2% of peak FLOPs. If residency comes back at ~90% that
prediction is wrong in an interesting way and worth stopping on.

**R10d — `DS4_METAL_DECODE_NWG` sweep** (2, 4, 8, 12, 16, 24, 32; both ranks;
at 2048 and 131072). An existing env knob (`ds4_gpu_flash_attn_decode_nwg`,
range 2–32) whose default is a **coarse bucket heuristic** — "bound runtime PSO
compilation while still removing most empty work". That is exactly the class of
tuned-once constant `nsg=8` turned out to be, and it costs nothing to sweep.
Note it only branches on `ds4_gpu_tp_world_is_two()`, so world-1 keeps NWG=32
regardless — sweep under TP.

**R10e — decode gate profile at two contexts** (`DS4_TP_GATE_PROFILE=1` at
2048 and 131072). We have ROW-gate data only at 131k: 11,008 gates
(= 128 tokens × 86, the fixed per-token schedule) at 418 µs wait / 29.9 µs
wire. One context cannot separate fixed overhead from context-dependent
compute; two can.

What to read from it, and what **not** to. Decode's 86 gates run in fixed
sequential order, so unlike the bulk prefill gates they cannot overlap — which
means Σwait ≈ token time is close to **tautological** (86 × 418 µs = 35.9 ms
against a measured 35.5 ms/token) and is *not* evidence of anything. Two prior
readings in this doc went wrong exactly here. The informative quantities are:

- **wire as a fraction of wait** — currently 29.9/418 = **7.2%**, i.e. the link
  is not decode's problem and R10b will not help decode;
- **how wait scales with context** — 2048 decode is 40.91 t/s → 284 µs/gate
  implied, 131k is 418 µs. If wire is flat ~29 µs at both, the growth is
  compute and the floor is ~284 µs/gate × 86 = 24.4 ms/token ≈ 41 t/s, which
  is precisely the measured 2048 number. That would mean **short-context decode
  is entirely per-gate fixed cost**, and the only lever with real headroom is
  reducing the 86 gates/token (2 per layer) — a design change, not a knob, but
  worth knowing before anyone tunes kernels for it.

### R8 — FlashAttention simdgroup count — **DONE, default flipped**

`DS4_METAL_FA_NSG` now defaults to 4; set `=8` to restore the old value for
A/B. Bit-identical on the rig as predicted, inertness exact, positive at every
context, and the gain grows with context (1.5% @4k → 7.3% @131k) exactly as the
mechanism implies. The one off-trend point is 2048 at +8.5%, which is above 4k
and 8k; 2048 is a single `pos0 == 0` chunk served by the zero-prefix path,
which uses the same kernel, so it is not inert here — but the size looks like
scatter rather than signal.

What follows is the original request, kept for the projection record.

**Why.** `nsg=8`, which `head_dim >= 512` has always selected, is the worst
legal value for this kernel at `C = 64`, on two compounding counts:

- `NC = (C/8)/NSG` is the number of 8×8 QK tiles a simdgroup produces per key
  block. `NSG=8` pins `NC` at its floor of **1** — one tile, then
  `simdgroup_store` and the barriers — where `NSG=4` does two, halving sync
  overhead per unit work. A `static_assert((C/8) % NSG == 0)` makes 8 the
  largest value the kernel admits at all.
- the QK loop unrolls by `MIN(DK8/2, 4*NSG)`, jumping from 16× to a full 32× at
  `NSG=8` and doubling live simdgroup matrices. No occupancy is won back:
  shared memory is `Q*(DK + 2*PV + 2*SH)*2` = 28,672 B against a 32,768 B
  limit, so **exactly one threadgroup is resident per core** either way. (The
  same budget is why `nqptg` cannot go to 16 — it would need 57,344 B and
  simply fails to launch. `head_dim=512` is what caps the query tile at 8.)

**Measured standalone** (M1 Max, ds4's own shader extracted and run against the
real kernel; executed keys counted from the mask, not assumed):

| shape | nsg=8 | nsg=4 | speedup | differing floats |
|---|---|---|---|---|
| odd ratio-128, late 131k chunk | 852.4 ms | 387.2 ms | **2.20×** | 0 / 134,217,728 |
| odd ratio-128, mid 64k chunk | 509.4 ms | 236.1 ms | **2.16×** | 0 / 134,217,728 |
| odd ratio-128, early 8k chunk | 210.0 ms | 97.7 ms | **2.15×** | 0 / 134,217,728 |
| short chunk (1024 tok) | 203.7 ms | 105.0 ms | **1.94×** | 0 / 33,554,432 |

Context for the size of that: at `nsg=8` the kernel runs at **0.76 TFLOP/s**
while an equal-FLOP GEMM on the same GPU reaches **5.2–6.0** — a ~7× gap, which
independently reproduces the ~6× implied by R6 on the M2 Ultra (attention 2.4
vs `q_path` 14.7 TFLOP/s). `nsg=4` recovers about half of it.

**R8a — inertness.** Flag unset must reproduce the R7 arm. Should be exact:
the default path is untouched.

**R8b — correctness** (16384, `DS4_TP_FORCE_DENSE_ATTN_OUT=1` both arms,
greedy). Expect **bit-identical**, as it was standalone on 134M output floats
across four shapes. `nsg` changes only how work is partitioned across
simdgroups, not accumulation order within a QK tile. If logits differ at all,
stop and report — that would mean the partitioning is not accumulation-neutral
on this hardware, which the standalone run says it is.

**R8c — throughput**: full sweep + cold 131k, all four flags on, vs the R7 arm.

Projection, recorded before the run: if the standalone ratio transfers, the
FlashAttention consumption roughly halves. Post-R7 that stage is ~128 ms on an
odd layer and ~39 ms on an even one, so the saving is ≈ 1.4 s + 0.4 s per late
chunk out of ~12.5 s → **sweep 131k 342.25 → ~375–395, cold 380.27 → ~415–440**
(+10–15%).

Weaker than R7's projection, for a reason worth stating: R7 extrapolated from
measurements on *this rig*, whereas this extrapolates from a **24-core M1 Max
to a 60-core M2 Ultra**. The 32 KB threadgroup-memory limit is architectural so
the occupancy argument carries, but the barrier-vs-register tradeoff that makes
`nsg=4` win could scale differently with core count. Treat the direction as
solid and the magnitude as unverified — that is exactly what R8c is for.

If it lands, flip the default in `ds4_gpu_flash_attn_nonvec_nsg()` and drop the
env knob.

**Outcome:** direction, shape and bit-identity all landed; the magnitude did
not — +7.3% against a projected +10–15%. Default flipped, but the knob is
**kept** (`=8` restores the old value) rather than dropped: it costs nothing
and it is the only way to re-A/B this on new silicon or after a kernel change.
The scoring caveat above was the right one to raise and the wrong one to then
ignore in the number.

### R7 — static-mixed row split — **DONE, passed**

Bit-identical, and both pre-recorded projections landed (sweep ~340 → 342.25,
cold ~380 → 380.27). Full results in `BENCHMARKS-TP-PP.md`. Notable: 2048
came in at ~0%, which is the correct inertness signal — a 2048-token prompt is
a single `pos0 == 0` chunk, so the flag has nothing to act on.

R6's single-threaded 43 MB mask-fill concern **did not bite**, as the follow-up
measurement predicted: ported and timed exactly, that loop is 5.25 ms serial,
2.1% of the `attention` stage and ~0.8% of a full 131k prefill. Parallelising
it with `dispatch_apply` gives 4.1×, and it is still not worth doing.

### Heterogeneous compute (ANE / CPU) — **evaluated and closed, 2026-08-26**

Asked whether the ANE or the CPU could take some of the prefill matmul load.
Answer: no for ANE, marginal for CPU. Recorded so it is not re-opened.

**The premise needed adjusting first.** Prefill *is* matmul-bound, but the GPU
is not bad at matmul. Scoring R6's stages against their FLOP counts: `q_path`
**14.7 TFLOP/s** (68% of the M2 Ultra's 21.5 TF fp32 peak), `output_proj`
**8.9**, `attention` **~2.4**. The problem was one kernel at ~6× below what the
same GPU does on a plain GEMM 30 ms earlier in the same layer — hence R8, not
more silicon.

**ANE — four independent blockers**, from the ANE guide plus local probing:

- **Bandwidth.** ANE's roof is 24–51 GB/s on M1, ~145 GB/s on M5. It does not
  see unified-memory bandwidth; the rig's GPU has ~800 GB/s per node.
- **Ultra does not aggregate.** The driver "steers whole independent
  submissions to the least-busy engine die and never exchanges tensor data
  between them," and the collective-enable capability byte reads zero on all 28
  targets. An M2 Ultra is two independent 16-core ANEs; a single graph cannot
  span dies.
- **fp16 numerics fail on our worst case.** The guide names value, **output
  projection** and down-projection as the projections that break in fp16 e2e.
  `o_a` is K=32768 — and on M2 (A14-generation ANE) the MAC output stage
  saturates at 2¹⁵ = 32768; the non-saturating path arrives at A15/M3.
- **Shape tax.** One compiled program per concrete shape, no dynamic dims
  without entitlement, 0.23 ms dispatch floor.

Throughput loses anyway: the guide's M1 head-to-head is ANE 3.6 vs GPU 7.3
TFLOP/s on a K=4096 GEMM, and it classifies large-batch matmul as "GPU regime
throughout" — which is exactly what 4096-token prefill is.

Probed locally on an M1 Max via CoreML at ds4's real shapes: ANE appears as a
*candidate* for every op tried (`candidates=CPU|GPU|NeuralEngine`) but CoreML
never once scheduled it, including for a canonical 3×3 conv stack. That matches
the guide — under CoreML the device is "a scheduling outcome, not a choice."
Forcing it means the private `e5rt_*` Espresso API.

On zero-copy: every boundary "charges a tensor repack between the engine
channel-interleaved fp16 layout and the host layout," IOSurface sharing is
listed as reachable-but-unexercised, and overlapping streams is called out as
unsound. Unified memory does not buy a free hand-off here.

**CPU/AMX** measured on the same M1 Max via Accelerate at ds4's shapes:
**1.32–1.79 TFLOP/s** fp32 (`q_a` 1.75, `q_b` 1.78, `o_a` 1.32, `ffn` 1.79).
Scaling by P-cluster count an M2 Ultra should reach ~3–3.5 — ~20% more compute
against a GPU doing 14.7, and not free, since those cores already drive the
Metal encoder and the RDMA gate service thread. Poor trade against R8.

One incidental cross-check worth keeping: AMX reproduces the GPU's
`q_b`-vs-`o_a` asymmetry (1.78 vs 1.32, same direction as 14.7 vs 8.9), so the
K=32768 reduction being slower is a property of the shape on two independent
engines, not a ds4 defect.

### R6 — size the ratio-128 layers — **DONE**, and it corrected this doc

Results in `BENCHMARKS-TP-PP.md`. The prize is large: `q_path` + `attention` +
`output_proj` on an odd layer = **311.2 ms**, ~71% of that layer's chunk time,
×20 layers ≈ **6.2 s of every late chunk**. The hard floor (`kv_path` +
`compressor`, which must stay full-width) is only ~3.5–5.5 ms/layer, so
row-splitting's ceiling on these layers really is the whole attention block.

**The data validates itself.** Odd `output_proj` / even `output_proj` =
34.857 / 17.190 = **2.03×** — almost exactly the replicated-vs-split ratio, on a
stage whose cost has nothing to do with `ratio`. That is an internal control
saying the profiler is faithful and that these layers are indeed running
full-width.

Two things this doc asserted, both wrong:

**1. "`comp = 1024` on ratio-128 means their attention is small."** Backwards.
It is 255.3 ms against the ratio-4 layer's 182.8. Having *no indexer* is the
whole point: `top_k` caps a ratio-4 row at 512 + window keys, while a ratio-128
row attends its entire compressed cache — 1024 + window at 131k, and **growing
with context**. Fewer compressed keys, but all of them, beats more compressed
keys with only 512 read. I inverted the comparison by reasoning from cache size
instead of from keys-read-per-row.

The growth confirms it: allocated `n_keys` rises only 7.2% from pos 81920 to
126976 (4896 → 5248), but the stage rises 33% (192.0 → 255.3). Cost tracks
*visible* keys per row (784 → 1136, +45%), not allocated ones — so the kernel's
block skipping is working, and what grows is the compressed span.

**2. "The static-mixed path carries a mask that needs a token-axis slice."**
Not on this path. `use_comp_mask` is set only inside the `ratio == 4 &&
n_comp > top_k` branch, which then takes `use_indexed_comp` and never reaches
this kernel — so the `pos0 > 0` mixed call always passes `comp_mask = NULL`.
Its mask is the one `ds4_gpu_fill_mixed_decode_batch_mask()` rebuilds per call
from `(pos0, n_tokens, n_raw, n_comp, window, ratio)`. Hand it this rank's
origin and row count and it regenerates exactly this rank's half: **the mask
slices itself.** The blocker I described is real, but it belongs to the
*zero-prefix* static-mixed path, which is a different call site.

So R7 was a predicate relaxation plus one call site passing
`q_work_rows`/`attn_pos0` instead of `n_tokens`/`pos0` — not the "real work" this
doc budgeted for. Corroboration that the surrounding machinery was always ready:
ratio-128 layers **already row-split at chunk 0** through the zero-prefix
static-mixed kernel, so nothing about those layers was unsafe to split.

Method note for future sizing runs: `DS4_METAL_LAYER_STAGE_PROFILE` **blocks the
CPU** at every boundary (`ds4_gpu_end_commands()`), unlike
`DS4_METAL_GPU_STAGE_TIMESTAMPS`. Its numbers are wall time per stage —
CPU-side work included, which is what made the mask fill visible — but they
serialise a layer that would otherwise overlap, so read one profiled layer
against another profiled layer, never against a clean run.

### R5 — indexer split (the main event; `589e93e`) — **DONE, passed**

Splits the score + top-k by query row, which R1 measured at ~61% of the
layer-chunk against the ~8% A0 splits. Needs **no extra gates** — each rank
scores its own rows and feeds its own top-k into attention rows it already
owns — so unlike A0 it should convert compute saving straight to throughput.

**Requires A0 on.** It reuses `tp_row0`/`tp_rows`/`attn_pos0`.

```
DS4_TP_PREFILL_SPLIT_NONZERO=1 DS4_TP_PREFILL_SPLIT_INDEXER=1    # BOTH ranks
```

**R5a — correctness first**, same protocol as Run 2: `DS4_TP_FORCE_DENSE_ATTN_OUT=1`
on both arms, temp 0, compare against the A0-only arm (not against flags-off).
Gate: tokens byte-identical, argmax identical at every frontier, `max |Δlogit|`
in the same ULP class as the 0.0055 control baseline. Expect a *smaller*
deviation than A0's 0.049 — this split does not move the FlashAttention block
geometry, it only changes which rows a rank scores.

**R5b — throughput**: cold single point at 131072, plus the full sweep. Compare
against the A0-on arm (237.44 sweep / 281.20 cold), not the baseline.

| | A0 only | A0 + indexer | Δ |
|---|---|---|---|
| cold 131072 | 281.20 | | |
| sweep 131072 | 237.44 | | |
| sweep 8192 | 422.05 | | |

**R5c — re-run R1 with the split on.** `score` per layer-chunk should roughly
halve (317.7 → ~160 ms) if the split is doing what it claims. This is the
cleanest confirmation that the mechanism works, independent of end-to-end noise.

No throughput prediction from me this time. The last two were wrong — 7× high on
A0, and then wrong about which term the gate cost lived in. R1's score share
bounds it; the rig decides the rest.

### Answered — kept for protocol reference

R1–R4 are complete; see the status table. Commands retained below.

### R1 — indexer stage breakdown at 131k (highest value)

Sizes the indexer split before it is built. Everything above is arithmetic; this
is the measurement that replaces it.

```
DS4_METAL_FAST_SYNC=1 DS4_TP_PREFILL_SPLIT_NONZERO=1 \
DS4_METAL_INDEXER_STAGE_PROFILE=1 \
  ./ds4-bench ... --ctx-start 131072 --ctx-max 131072
```

Emits per layer per chunk:
```
ds4: metal indexer stage layer=<il> pos=<pos0> tokens=<n> comp=<n_comp> score=<ms> ms
                                                                        topk=<ms> ms
                                                                   attention=<ms> ms
```

Record the three stage times for **a few even layers ≥ 2** (they carry the
indexer) at a late chunk, plus `comp=`. Paste a representative handful of lines
rather than the whole log.

**Caveat:** the boundary helper calls `ds4_gpu_end_commands()` /
`ds4_gpu_begin_commands()` around each stage, so it forces a command-buffer
split per stage. Throughput under this flag is **not** comparable to a clean
run — only the relative score : topk : attention split is meaningful.

What it decides: if `score` dominates `attention` by the predicted ~6×, the
indexer split is the next change and its ceiling is roughly the `score` share.
If it does not, the model above is wrong and I want to know before building.

### R2 — gate profile, both arms

Turns "~90% GPU util" into a number, and gives the bandwidth figure that decides
whether the RDMA workstream is worth anything.

```
DS4_TP_GATE_PROFILE=1 ...  --ctx-start 131072 --ctx-max 131072
```
once with `DS4_TP_PREFILL_SPLIT_NONZERO=1`, once without.

From the **BIG** line record, for each arm:

| | A0 off | A0 on |
|---|---|---|
| `gates` (expect 43 → 66 per chunk) | | |
| avg **gpu wait** µs — the pipeline bubble | | |
| avg **exchange** µs | | |

Effective per-direction bandwidth = `67,108,864 / avg_exchange_us`.

The gpu-wait delta between arms is the true cost of A0's added gates, and is the
number I got wrong by estimating wire time only.

### R3 — GPU utilisation shape

Was the ~90% seen with A0 **on**, **off**, or both? And is the idle flat across
the sweep or growing with context? Flat implicates gate count; growing
implicates something else. A rough `powermetrics --samplers gpu_power` reading
at 8k and at 131k in each arm is enough.

### R4 — short-context regression (cheap, low priority)

Run 1 showed −2.2% prefill and −2.0% decode at 4096 vs the old base, above the
~1% noise floor. Suspect is #832's canonical argsort comparator: unlike the rest
of the indexer stack it is **not** token-count gated, so it applies where the
sort is a large fraction.

```
DS4_METAL_DISABLE_ARGSORT_CANON=1     # 2048 and 4096 points only
```

If that recovers the 2%, it should be gated on `n_tokens >= 32` like its
siblings and I will patch it.

## What is being measured

Two independent changes landed on this branch. **They must be separated**, or the
A0 result will be contaminated by the upstream prefill stack.

| group | commits | expected effect |
|---|---|---|
| upstream indexer stack | `4760333`, `557ebf4`, `4624e5f`, `1718151`, `ead8786`, `5321d86`, `b7dc56c` | prefill +~24% at 131k, bit-exact |
| LLT decode scorer | `4333606` | decode +6–7% at long ctx only |
| **A0 row split** | `309c0e2`, `534648e`, `82d9e9a` | **prefill, opt-in, the thing under test** |
| server/correctness | `958b248`, `92c4a5a`, `4cc4497`, `471d11b`, `5994f7d`, `d2367a3`, `318e6eb` | no prefill throughput effect |

**The old 183.4 t/s at 131072 is from `tp-multi-slot-batching` and is NOT the A0
baseline.** A0's baseline is this branch with the flag off, which should already
be faster because of the indexer stack.

## Hard prerequisites

1. **Same commit on both hosts.** `git rev-parse HEAD` must match on lanfear and
   mat. Rebuild both; do not reuse a stale binary.
2. **Env symmetry.** `DS4_TP_PREFILL_SPLIT_NONZERO` changes the per-layer gate
   count. An asymmetric setting **deadlocks the gate exchange**, it does not
   degrade. Set it on both ranks or neither. Same for
   `DS4_TP_FORCE_DENSE_ATTN_OUT`.
3. **RDMA device names are host-local and the docs disagree.** `BENCHMARKS-TP-PP.md`
   has lanfear=`rdma_en6`, mat=`rdma_en7`; `speed-bench/README.md` has them
   reversed. Confirm against the actual hardware before the first run — a wrong
   name is a failed QP bring-up, not a silent slowdown.
4. **The sweep is incremental.** `ds4_bench.c:780-795` advances the frontier from
   the previous ctx point, so the `131072` row measures the 65536→131072
   increment (65536 tokens), not a 131k prefill from scratch. Every chunk in that
   row has `pos0 > 0`, which is why it is the cleanest A0 signal in the sweep —
   but it also means the row's mean attended position is ~3N/4, not N/2, so it is
   **not comparable** to any single-node number whose protocol we do not know.
   Use the sweep for A/B against itself, and Run 3b below for anything
   cross-machine.

## Run 0 — build and sanity

On **both** hosts:

```
git fetch && git checkout upstream-metal-wins && git rev-parse HEAD
make -j ds4-bench ds4-server ds4
```

Sanity, short prompt, all new flags unset. Expect normal output and no new
warnings in stderr.

## Run 1 — inertness (do this first, it is the cheapest failure)

Flag unset. Confirms A0 is a true no-op when off, and establishes the A0 baseline.

**lanfear (coordinator):**
```
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role coordinator \
  --tensor-parallel --transport rdma --listen 0.0.0.0 1234 \
  --rdma-device rdma_en6 --prompt-file /tmp/bench_long.txt \
  --ctx-start 2048 --ctx-max 131072 --step-mul 2 \
  --gen-tokens 128 --csv /tmp/a0_baseline.csv
```

**mat (worker):**
```
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role worker \
  --coordinator 192.168.0.6 1234 --transport rdma --tensor-parallel \
  --rdma-device rdma_en7
```

Pass criteria: completes with zero faults; prefill at 131072 is **≥ the old
183.4 t/s**. If the indexer stack is working it should be nearer ~228.

## Run 2 — correctness (must pass before any throughput claim)

Bit-equality of split vs unsplit. **`DS4_TP_FORCE_DENSE_ATTN_OUT=1` on both arms
and both ranks** — without it the arms differ by the output-projection kernel as
well as by the split, and the comparison is meaningless.

Use a fixed prompt long enough to cross a chunk boundary (>8k tokens), temp 0.

**Arm A (control):**
```
DS4_METAL_FAST_SYNC=1 DS4_TP_FORCE_DENSE_ATTN_OUT=1 \
  ./ds4-bench ... --gen-tokens 128 > /tmp/a0_ctrl.txt
```

**Arm B (candidate):** identical plus `DS4_TP_PREFILL_SPLIT_NONZERO=1` on both
ranks.

```
cmp -s /tmp/a0_ctrl.txt /tmp/a0_cand.txt && echo IDENTICAL || echo MISMATCH
```

**Superseded criterion (2026-08-25).** The original gate here was
"byte-identical logits". That is unsatisfiable on this rig and always was:
**two identical control runs** (flag off) already differ in all 129,280 dumped
logits, max abs 0.0055, argmax identical. The unsplit TP path is not
bit-deterministic run to run, so cross-run bit equality cannot be a gate for
anything.

The earlier reasoning — "each output row is an independent accumulation over the
same key sequence, so the split cannot change how a row is computed" — is right
about the per-row math and wrong about the surrounding blocking. The split
changes `n_raw` and `raw_start`, so the SWA ring is linearised from a different
offset and the FlashAttention block geometry moves with it (`has_kvpad` from
`n_keys`, `bc_mask` from `n_tokens % 8`). Same keys, same values, different
block boundaries, different rounding.

**Actual pass criteria — all three:**

1. Generated tokens **byte-identical** to the control at temp 0. This is the
   strong one: it means every sampling decision matched.
2. Frontier **argmax identical at every dumped frontier**.
3. `max |Δlogit|` within a few ULP of f16 at the observed logit magnitude, and
   within ~10× the control-vs-control baseline measured in the same session.
   Record both numbers; the baseline is the scale, not zero.

Measured 2026-08-25: control-vs-control 0.0055, split-vs-control 0.049 at logit
magnitude ~27 (f16 ULP there is ~0.026, so ~2 ULP). 128/128 tokens identical,
305/305 argmax identical. **Pass.**

**If criteria 1 or 2 fail, stop.** Criterion 3 alone drifting means investigate
before trusting the throughput numbers, not necessarily abort.

## Run 3 — A0 throughput sweep

`DS4_TP_PREFILL_SPLIT_NONZERO=1` on both ranks, `DS4_TP_FORCE_DENSE_ATTN_OUT`
**unset** (it costs throughput).

```
DS4_METAL_FAST_SYNC=1 DS4_TP_PREFILL_SPLIT_NONZERO=1 \
  ./ds4-bench ... --csv /tmp/a0_split.csv
```

### Results table

| ctx | old branch | Run 1 (flag off) | Run 3 (flag on) | Δ vs Run 1 |
|---|---|---|---|---|
| 2048 | 381.3 | | | |
| 4096 | 353.8 | | | |
| 8192 | 377.1 | | | |
| 16384 | 346.3 | | | |
| 32768 | 310.4 | | | |
| 65536 | 252.6 | | | |
| **131072** | **183.4** | | | |

Expected shape: **~0% at 2048** (chunk 0 only, already split), rising with
context, largest at 131072. A0 does nothing at short context by construction —
if you see a gain at 2048 something else changed.

Target at 131072: **~1.5× over Run 1.** Derivation: the pair executes ~8.9e10
FLOPs/token of which ~45% duplicates the peer; halving the splittable part
predicts 1.51×, and the measured TP2-vs-single-node ratio after discounting the
M2/M3 gap independently gives 1.52.

## Run 3b — cold single-point prefill (do this, it is the honest number)

The sweep rows are incremental suffixes. For a number that means something on
its own — and for any comparison against a single-node figure — run one **cold**
point, flag off then flag on:

```
--ctx-start 131072 --ctx-max 131072      # no --step-mul; one cold prefill
```

This is the number to quote. The sweep tells you the *shape* of the gain across
context; this tells you the *size* of it at the context you care about, without
the 3N/4 mean-position artefact.

| | flag off | flag on | Δ |
|---|---|---|---|
| cold 131072 prefill t/s | | | |

## Run 4 — free diagnostics while the rig is up

Costs one extra run and decides whether the RDMA workstream is worth anything.

```
DS4_TP_GATE_PROFILE=1 DS4_METAL_FAST_SYNC=1 ... --ctx-start 131072 --ctx-max 131072
```

The profile prints per gate kind (ROW / VERIFY / BIG): gate count, average GPU
wait µs, average exchange µs. From the **BIG** line record:

- `gates` — sanity check: should be ~16 chunks × 43 layers = 688 with the flag
  off, roughly double with it on (A0 adds the attn_out row swap per layer)
- `avg exchange us` → **effective per-direction bandwidth =
  67,108,864 / avg_exchange_us** (bytes/µs = MB/s)
- `avg gpu wait us` → the pipeline bubble per gate. This is the number I could
  only estimate statically; if it is large relative to the exchange, the cost is
  the encoder drain and not the wire.

The bandwidth number decides WS3+4 on its own: two of its three proposed gain
arms required bandwidth above TB4 line rate, i.e. were impossible.

Also worth capturing, same run:
```
DS4_METAL_GPU_STAGE_TIMESTAMPS=1 DS4_METAL_GPU_STAGE_TIMESTAMPS_LAYER=<il> ...
```
(`_LAYER` restricts to one layer, `_DETAIL` adds sub-stages). Pick a ratio-4
layer — even `il` ≥ 2 — since those carry the indexer. This gives the per-stage
split of the ~519 ms/layer-chunk: indexer scoring vs attention core vs
projections. That sizes the *next* change — splitting indexer score/top-k, which
needs no cross-rank merge because it is per-query-row, and is the largest
remaining win once A0 lands.

## Failure signatures

| symptom | meaning |
|---|---|
| Gate wait hangs, or `tp: worker sync send failed` | **asymmetric env.** The flag is set on one rank only. |
| `kIOGPUCommandBufferCallbackErrorTimeout` on the worker | watchdog kill; check whether it also happens with the flag off (then it is pre-existing, see #852) |
| Row-range view rejected / prefill aborts | a bounds case the even-chunk guard did not cover — capture `n_tokens`, `pos0`, `il` |
| Run 2 mismatch | real correctness defect in the split |
| Run 3 gain ≈ 0 at all ctx | flag not reaching the predicate — confirm with `DS4_LOG` that split chunks are firing |
| Run 3 gain at 2048 | contamination; something other than A0 changed |

## Abort criteria

- Run 2 mismatch → revert to Run 1 config, report the prompt and ctx.
- Any hang → both ranks must be restarted (`tp->failed` is sticky and never
  cleared; a wedged pair does not recover in-process).
- Worker exits with a GPU command-buffer error → expected behaviour as of
  `03cbf99`; it now exits loudly rather than lingering as a zombie.

## Rollback

Everything under test is opt-in and off by default. Unsetting
`DS4_TP_PREFILL_SPLIT_NONZERO` restores current behaviour with no rebuild. To
drop A0 entirely: `git revert 82d9e9a 534648e 309c0e2`.
