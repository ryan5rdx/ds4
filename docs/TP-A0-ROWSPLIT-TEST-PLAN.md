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
| R11 — decode gate count | **built, requested below** | `DS4_TP_DECODE_REPLICATE_ATTN=1`. Replicates attention on decode, dropping 86 gates/token to 43. GLM already runs this way. A real trade, not a free win. |

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

## Open requests

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

### R11 — halve the decode gate count (`DS4_TP_DECODE_REPLICATE_ATTN=1`)

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
