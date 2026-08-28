# Benchmarks

Measurement record for the Metal / tensor-parallel fork. Everything here was
taken on the reference rig unless stated otherwise; entries are kept because
they are still load-bearing for a decision, not for completeness. Superseded
runs have been dropped.

**Rig.** Two Apple M2 Ultra, 60 GPU cores, 128 GB each, tensor-parallel over
Thunderbolt RDMA, Metal backend. Model: DeepSeek V4 Flash MXFP4, 145.26 GiB
total, 76.73 GiB resident per rank.

**Reading the numbers.** Quote `gen_steady_tps`, not `gen_tps` — the latter is
skewed by the first token. In a `--step-mul 2` sweep each context row is the
*increment* from the previous point, so the 131072 row is the 65536→131072
chunk, not a cold 131k prefill. Cold single points are called out explicitly.

---

## 1. Current performance

Full sweep, TP versus pipeline-parallel at the same commit, `--step-mul 2`,
128 generated tokens per point.

| ctx | TP prefill | PP prefill | prefill winner | TP decode | PP decode | decode winner |
|---|---|---|---|---|---|---|
| 2048 | 423.03 | 316.98 | TP +34% | 40.91 | 27.67 | TP +48% |
| 4096 | 417.54 | 310.46 | TP +35% | 36.47 | 26.29 | TP +39% |
| 8192 | 500.16 | 348.21 | TP +44% | 35.98 | 26.08 | TP +38% |
| 16384 | 480.79 | 459.38 | TP +5% | 35.43 | 25.54 | TP +39% |
| 32768 | 461.20 | 530.00 | PP +15% | 33.73 | 24.11 | TP +40% |
| 65536 | 427.08 | 520.42 | PP +22% | 31.52 | 22.52 | TP +40% |
| 131072 | 367.29 | 444.38 | PP +21% | 28.13 | 20.57 | TP +37% |

Cold 131k prefill under TP: **402.64 t/s**.

**TP wins prefill up to 16k and decode at every context; PP wins prefill from
32k up.** Both configurations are worth keeping.

Later decode re-baselines on the same configuration read slightly higher than
the sweep's decode column, which predates some of the decode work: **41.19 t/s
at 2k** (three interleaved repeats, sd 0.015) and **29.4–29.8 at 131k**.
Cross-session anchors span 41.1–42.3 at 2k, a ±1.4% spread that bounds the
resolution of any single-run decode comparison.

---

## 2. Platform roofs

These bound every optimization estimate and are worth re-reading before
proposing one.

| quantity | measured | notes |
|---|---|---|
| streaming read, one rank | **760 GB/s** | 94–95% of the 800 GB/s spec |
| FP32 compute | ~21 TFLOP/s | 60 cores × 128 × 2 × 1.398 GHz |
| Q8_0 matvec, best case | 445–458 GB/s at k=4096 | peaks at 4096, dips at 8192 |
| decode effective bandwidth | ~192–244 GB/s | ~25–32% of roof |
| decode ALU utilisation ceiling | **8.19%** | roofline: 2.26 / 27.6 FLOP/byte |
| GPU clock during decode | 1398 MHz (maximum) | 100% residency, 33.4 W |
| GPU power during prefill | 59–60 W at 1394–1398 MHz | same clock, 1.78× the power |

**The allocation path costs nothing.** Seven allocation arms — Metal shared
buffers, anonymous mmap, file-backed mmap (the engine's own load path),
untracked, and parallel-faulted — all measure 756–761 GB/s at both 16 GiB and
32 GiB. There is no gain available from changing how weights are mapped.

**Decode is latency-bound, not bandwidth-bound or power-bound.** It runs at the
top P-state with full residency while drawing half of prefill's power, and a
genuinely bandwidth-saturated streaming kernel at 760 GB/s draws only 38 W. So
power tracks neither clock nor bytes, and the low wattage is not evidence of
headroom in either direction. The arithmetic intensity is what shows the
headroom: ~8% of peak ALU.

### RDMA transport

Verified ping-pong, 2000 iterations per arm, every reply byte-checked in both
directions (µs, min / p50 / mean / p99 / max):

| arm | RTT | half-RTT p50 | drops |
|---|---|---|---|
| 4 KB | 11.0 / 16.0 / 16.4 / 25.0 / 789 | **8.0** | 0 |
| 16 KB | 19.0 / 29.0 / 40.6 / 113 / 1608 | **14.5** | 0 |

A single 16,384 B UC SEND work request — the gate's actual shape — works on this
stack with no chunking. Hardware one-way at the gate payload is ~15 µs against
~49.7 µs observed in-engine, so roughly 35 µs per gate is software rather than
wire. Link ceiling is 4.4 / 4.1 GB/s per direction with a 4 KiB write cap.

UC has no retransmit: one transient first-ping drop occurred across the whole
battery, so any latency-sensitive gate rework needs a re-arm path.

---

## 3. Prefill

The optimization arc at 131k, each step measured against the previous:

| step | t/s | gain | output |
|---|---|---|---|
| pre-workstream base | 183.4 | — | |
| upstream indexer stack | 221.50 | +20.8% | |
| nonzero-prefix row split | 237.44 | +7.2% | perturbed logits 0.049 |
| indexer split | 283.64 | +19.5% | bit-identical |
| static-mixed split | 342.25 | +20.5% | bit-identical |
| flash-attention simdgroup count | **367.29** | +7.3% | bit-identical |

**+100% over the pre-workstream base, +65.8% over the flag-off baseline on this
branch.** Cold single point 259.90 → 402.64 (+54.9%).

Three of the four are **row splits** — they divide a prefill chunk's rows across
ranks. That is why they need a minimum chunk size and why they do not transfer
to small batches.

Correctness gate: the nonzero-prefix split perturbs logits by 0.049 against a
0.0055 control-vs-control baseline (~2 f16 ULP) with 128/128 tokens and 305/305
argmax identical. The other three are bit-identical, 0 of 129,280 logits
differing.

---

## 4. Decode

### Stage budget at 2k

Anchor 24.26 ms/token. Net of a ~0.18 ms/marker profiler tax, which is itself
soft — every net figure carries ±0.18 ms, i.e. 100% of its own correction.

| stage | net ms | % | note |
|---|---|---|---|
| routed MoE | 4.72 | 19.5 | 367 vs ~400 GB/s isolated — at its roof |
| attention + inverse RoPE bracket | 3.63 | 15.0 | the only stage materially off its roof |
| attention output projection | 2.55 | 10.5 | implied 601 GB/s, above every measured Q8_0 rate — suspect |
| q/kv projection | 1.97 | 8.1 | byte-weighted average of a Q8_0 and an F16 stream |
| q path | 1.70 | 7.0 | |
| q-LoRA norm | 1.50 | 6.2 | one dispatch of two threadgroups, ~35 µs/layer |
| FFN gate exchange | 1.40 | 5.8 | |
| compressor update | 1.10 | 4.5 | **bookkeeping artefact — true cost 0.02–0.05 ms** |
| attention HC pre | 0.96 | 4.0 | |
| FFN HC pre | 0.93 | 3.8 | |
| router | 0.92 | 3.8 | |
| attention gate exchange | 0.81 | 3.3 | |
| shared gate/up | 0.79 | 3.3 | |
| shared down | 0.49 | 2.0 | |
| remainder | 0.65 | 2.7 | |

Byte floor: 5.93 GB/rank/token at 449.2 GB/s = **13.20 ms**; decode is 1.84× off
it. The book is overdrawn by at least 0.38 ms — roughly nineteen marker sites
plus the output head are unreported, and the output norm alone was priced at
0.52 ms.

### Scheduling

- Encoder spans **overlap ~2×**: mean concurrency 1.95–1.99, maximum exactly 2.
- **No idle pool inside a command buffer**: span union is 100% of the buffer,
  gap ≈ 0.000 ms, at both 2k and 131k.
- Therefore stalls are *inside* kernels, not between them, and reordering or
  merging dispatches to close gaps cannot pay — there are no gaps.
- Marginal dispatch cost is **nonlinear**: 2.49 µs at the 86-dispatch scale,
  0.98 µs at 688, because later dispatches hide in the 2× overlap. Fitted
  t/s = 41.106 − 0.0694·N over ballast N ∈ {0,2,8,16}, three interleaved repeats.

### Kernel-level

- Indexer LLT scoring: **1565 GFLOP/s**, 72% core scaling; occupancy is the
  lever, not bandwidth. The tight-shared-memory alias is **+17.4%** on the
  kernel and bit-identical, and does not regress prefill — but it is worth only
  ~0.02 ms/token at 2k because the indexed path is unreachable there.
- MXFP4 17-byte block granularity costs ~6%, not the 46% once suspected.
- Indexed attention requires more than 1024 compressed rows, so at 2k with a
  ratio-4 layer (512 rows) the indexed branch never executes. The crossover is
  context > 4096.

---

## 5. Speculation

Both drafters were measured and neither is fundable on this hardware.

### n-gram

Offline replay of captured token traces through the shipped proposer, with the
cost model `V(k) = V_fixed + k·V_marginal` charged only on cycles that offer a
draft.

| trace | tokens | best | mean commit | offered |
|---|---|---|---|---|
| prose | 6143 | 1.097× | 0.30 | 9.9% |
| code | 3938 | 1.034× | 0.15 | 4.0% |

91.4% and 96.5% of steps commit nothing. Both clear their own break-even —
which is `(V/T) × offer_rate`, not a flat mean commit — but only by enough to
reach 1.03–1.10×, a rounding error. Calibration controls: a uniform-random
trace reads exactly 1.000×, a period-8 synthetic 1.569×.

### DSpark

Forced 5-row drafts, 512 tokens, against a 41.96 t/s no-speculation baseline:

| arm | t/s | acceptance |
|---|---|---|
| published support model | 21.32 | 189/1596 = 11.8% |
| with verifier head split | 20.18 | 151/1785 = 8.5% |
| full-precision support model | 19.91 | 157/1767 = 8.9% |
| low-yield production policy | 41.57 | verifier never launched |

Where the time goes: proposal 3412 ms (14.2%), verify 12,333 ms (51.4%), target
8237 ms (34.3%). Those tile the wall clock to 0.004%, so **nothing overlaps**.

Fitted verify cost, from a dedicated profile run (n=50 at 2 rows, n=7 at 3):

```
V(2) = 76.12 ms    V(3) = 96.76 ms    ->  V(k) = 34.83 + 20.64k
```

The intercept is real but the fit does not extrapolate: V(5) predicts 138 ms
against 108.18 measured, because a 5-row draft takes a native 5-row tile that
d2/d3 do not. **Longer drafts are cheaper per row than the linear fit implies**,
which is why the optimum here runs toward longer drafts rather than shorter.

**The fixed term is GPU layer-encode overhead, not data movement.**
`verify_layer` is **99.97%** of the verify in every run; upload and readback are
0.26–0.5 ms each. So the cost the byte model cannot see lives inside the
43-layer batch encode, as fixed launch overhead — the same latency-bound
character as decode, and immune to quantisation or bandwidth.

**Which term dominates depends on the workload.** Under a forced policy on prose
the split inverts to propose 3859 ms against verify 1794 ms, with the proposal
chain 86% of propose. Both terms are latency-bound; neither is a bandwidth wall.

**The production policy correctly declines to speculate on prose.** Left at its
defaults it backs off after four attempts, launches no verifier at all, and
returns exactly the no-speculation baseline (40.33 t/s). Measuring the verify at
all requires `DS4_DSPARK_TP_LOW_YIELD_POLICY=0`, and volume requires the
scheduler off as well.

Break-even is mean commit > V/T = 4.41 at 5 rows, against 1.658 observed. A
*completely free* drafter would still reach only 24.89 t/s. Shorter drafts do
not help here: the large fixed term means the optimum runs toward longer drafts,
the opposite of the guidance published for compute-rich single-GPU parts.

The propose is also latency-bound — ~1.86 GB of traffic implies 2.4 ms at roof
against 10.7 ms measured, i.e. ~174 GB/s, essentially the decode rate.

---

## 6. Closed avenues

Kept so they are not re-investigated. Each was measured, not argued.

| avenue | result |
|---|---|
| replicated decode attention to halve gate count | **−8.4 to −11.8% decode** at all seven contexts |
| widening attention-output simdgroups | measured faster only because it skipped half the weight stream; produced incorrect output |
| packed-32 attention at 32 heads | −1.35 t/s |
| hierarchical top-k | 0.77× |
| byte-indexed MXFP4 lookup table | predicted +15.4%, measured −0.32% |
| MoE decode specialization | 0 to −5% |
| row-gate fastpath | ±0.6%, a wash |
| split command-buffer schedule | flat, within 0.9% |
| sub-gate pipelining | wash at every context |
| decode workgroup count retune | default already optimal |
| host allocation strategy | 756–761 GB/s across all seven arms |
| CPU or Neural Engine drafting | 17.9 t/s and no execution path respectively |
| third-node drafting | needs 1.62 GB of target weights plus the support model; does not fit |

**Track record: eleven ex-ante decode throughput estimates were measured; the
best realised was +0.12%.** Estimates on this path should be discounted heavily,
particularly anything in the grid-widening or occupancy class, which is 0-for-4.

---

## 7. Measurement hazards

Every one of these produced a wrong published number at least once.

- **Sequential A/B blocks are worthless.** The same arm has varied 56–124 ms
  across repeats. Interleave arms and report a fitted slope, not a two-point
  difference.
- **A throughput delta is not a result without a correctness gate.** The bar is
  top-1 preserved plus a bounded logit delta. A change that skips work measures
  faster.
- **Paired quantities must move together.** A baked pipeline constant and the
  dispatch argument that must match it are one quantity, not two.
- **The stage profiler ends a command buffer per marker**, costing ~0.18 ms per
  marker and moving the operating point by up to 22%. Do not compare a
  stage-profiled figure against an unprofiled one.
- **Encoder timestamps cannot sub-divide a batch segment.** The batch reuses one
  encoder across many dispatches, so labels collapse onto a single span. Forcing
  a boundary per stage resolves them but perturbs throughput by 20–30% and
  inflates long-context spans by 50×.
- **Scaling a ratio measured on one machine onto another is not a measurement.**
  A dev-box-derived per-dispatch cost came out 3.3× too high on the rig.
- **Derived byte models drift from reality.** A propose call was documented as
  streaming 5.99 GB when it streams ~1.86 GB, because only the experts the draft
  rows route to are read.
