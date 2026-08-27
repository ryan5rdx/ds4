# Path to 50 t/s at low-context decode — decision document

**Date:** 2026-08-27 · **Baseline:** 41.1 t/s = 24.34 ms/token at ctx 2048, TP2,
2× M2 Ultra 60-core over Thunderbolt RDMA, Metal.
**Target:** 50 t/s = 20.00 ms/token. Saving required: **4.34 ms (17.8%)**.

Inputs: eight scouted lenses, each adversarially refuted; a completeness critic;
a premise challenger; a risk reviewer. Everything load-bearing below was
re-verified against the tree at `HEAD = 6f37a55`. Where a reviewer's number
disagreed with the code or with an on-disk measurement, the code and the
measurement win and the disagreement is named.

---

## 1. VERDICT

**No. 50 t/s is not reachable at low context on TP2 with anything currently
identified. The number I believe is 43–45 t/s from the sequenced plan below,
with 46–48 t/s as an upper branch contingent on one item nobody has been able to
size because the instrument for it has never been run.**

Three claims, in order of how much they change the picture.

### 1.1 The premise the campaign has been run on is wrong in the campaign's favour, twice, and against it once

**The floor is not 9.5 ms.** The 4.29 GB/token/rank figure is §3's *attention
total* (2.587 GB) plus the balanced routed-MoE share (1.725 GB). It omits every
replicated stream. I re-derived the byte model from the layout assertions at
`ds4.c:5124-5180` and it comes to **5.93 GB of weights** — the omissions are the
attention + indexer compressors (**608 MB**, which I recomputed independently
and which matches to the digit), the shared expert (575 MB), the output head
(281 MB), the router (90 MB) and the HC mix (68 MB). At a flat 450 GB/s the
floor is **13.2 ms (76 t/s)**, not 9.5 ms (105 t/s). Decode is **~1.8× off its
floor, not 2.6×**. The premise challenger found this independently and validated
it against the artifact: non-routed weights sum to 8.19 GiB against the 8.26 GiB
implied by (file 145.26 GiB − routed 137 GiB), and §3's own startup-log
reconciliation of 76.73 GiB. I believe it. **This is the single most important
correction in the document** — it re-bases every "% of roof" claim anyone makes
from here on.

**760 GB/s is not the roof for this work.** `tp_decode_investigation.md` §2
already says so: *"Q8_0 matvec 517–581 GB/s, MXFP4 MoE ~400 GB/s. Those are the
real ceilings, not 800."* Priced against each stream's own measured shape rate,
the bandwidth-bound 58% of the token runs at **~90% of its own roofline**. That
is why weeks of nsg / nr0 / format / fusion work returned nothing: it was the
correct outcome, not a failure of imagination.

**But the "central anomaly" is real, and it is one stage, not the whole graph.**
Every stage in the token is at 90–100% of its own kernel's isolated rate — with
exactly one exception. `attn_inv_rope` (3.63 ms net, 14.9%) has a DRAM floor of
~0.2 ms, a FLOP floor of ~0.1 ms and a cache-traffic figure of ~496 GB/s from a
655 KB/layer working set. It is 1–2 orders of magnitude from every roof it has.
**It is also nearly context-invariant** — raw 3.813 ms at 2k against 4.220 at
131k (`BENCHMARKS-TP-PP.md:336-344`), i.e. +11% over a 64× context increase — so
it is a short-context item, not a long-context one.

### 1.2 The 30 W / "20% utilised" anomaly dissolves, and I would stop citing it

At 2.26 FLOP/byte against a 27.6 FLOP/byte machine balance, decode sits 12×
inside the memory-bound region; the **maximum** ALU utilisation this workload can
ever reach at the memory roof is 8.2%. A ~30 W draw at 248 GB/s sustained
reconciles to 28–35 W with no residual (DRAM+PHY ~10 W, on-die ~3 W, ALU+dequant
5–8 W, clock+leakage 10–14 W). Prefill's 60 W is a GEMM number; you cannot draw
GEMM power on a matvec at any efficiency. The alternative reading — 5× headroom
— requires 1242 GB/s, which is 1.6× the chip's spec. **The power observation and
the bandwidth roof are the same constraint counted twice.** I believe the premise
challenger here, and there is a 5-minute `powermetrics` falsifier in §5 that
retires it either way.

### 1.3 The arithmetic

Nothing survives the refutation stage at a size that matters. Summing **every
item in the scouted pool — survivors and refuted, at full value, at 100%
confidence, with every double-count left in — gives 3.54 ms against 4.34
needed.** My own discounted total, with two items the pool never scouted added
back, is **0.90 ms**, or **0.60 ms** if the one unsized item is excluded.

| scenario | saving | ms/token | t/s |
|---|---|---|---|
| baseline | — | 24.34 | 41.1 |
| discounted plan, A–F only (§2) | 0.60 ms | 23.74 | 42.1 |
| discounted plan, A–G | 0.90 ms | 23.44 | **42.7** |
| all of A–F land at their top end | 1.57 ms | 22.77 | 43.9 |
| + FlashAttention restructuring at 1.0 ms | 2.57 ms | 21.77 | 45.9 |
| + FlashAttention restructuring at 2.0 ms | 3.57 ms | 20.77 | **48.1** |
| absolute physical floor (§3.4) | 6.3 ms | ~18.0 | ~55 |

**50 t/s is not physically excluded — the floor is ~18 ms — but it is excluded by
the identified work.** Reaching it requires taking roughly two thirds of the
~10.8 ms latency block, and the only structure in that block big enough to fund
it is FlashAttention. Nobody can size that today, because **the instrument built
to decompose it measures a command-buffer round trip with a host wall clock and
reports figures 6× the stage they belong to** (`ds4_metal.m:11040-11050`;
retracted in-tree at `TP-A0-ROWSPLIT-TEST-PLAN.md:765-780`), and the instrument
that would fix it — `DS4_METAL_GPU_ENCODER_TIMESTAMPS`, `df0037e`, 1–3% overhead,
already built and validated — **has never been run on the rig.**

### 1.4 What I would tell the programme

1. **Set the target at 43–45 t/s for TP2 low context** and say so publicly. That
   is +5–10% on a graph that is already 90% efficient on its bandwidth half, and
   it is honest. 46–48 requires the FlashAttention item to land near its top end.
2. **Stop funding lens rounds against the current stage table.** Six lenses ran
   and the sentence at `TP-A0-ROWSPLIT-TEST-PLAN.md:392` — *"the two largest
   items remain without a proposal"* — is still true, because the table
   pre-labelled them as such and every lens obeyed the labelling. **13.5 ms of
   the token, 55%, has no surviving candidate**; the two largest stages, 8.35 ms
   together, have none at all.
3. **Spend the next rig session on measurement, not on arms.** Five zero-code
   runs and one ~10-line instrument rewire (§5) decide more than every candidate
   in the pool combined, and two of them are already-built instruments sitting
   unused.
4. **If 50 t/s is a hard requirement, it is a TP4 decision, not a kernel
   decision.** Per-rank bytes fall to ~3.5 GB; the ~10.8 ms latency block does
   not scale, so the projection is ~19–20 ms = **50–53 t/s**, which independently
   reproduces §7's planning ranges (47–51 disjoint quarters, 52–56 cyclic,
   53–58 repacked, `tp_decode_investigation.md:412-417`). Those depend on a
   three-peer 16 KiB collective reaching 40–60 µs p50 and on a true six-link
   mesh. It is a topology programme, not a decode optimisation.

---

## 2. THE BUDGET

### 2.1 Surviving candidates

Confidence factors are mine and are applied to the 2k column. "Bit-id?" is the
strongest correctness gate available for the item; see §4 for the full gates.

| # | candidate | 2k ms | 131k ms | conf | factor | disc. | cost | bit-id? |
|---|---|---|---|---|---|---|---|---|
| A | FP8 amax block loop, parallelised (`q_lora_norm`) | 0.30 | 0.30 | med | 0.60 | **0.18** | days | **yes, provable** |
| B | MoE straggler → variable shared-expert split | 0.30 | 0.30 | med | 0.40 | **0.12** | weeks | no (T2 bar) |
| C | TP-gate `flag_set` dispatch fold (86/token) | 0.22 | 0.22 | med | 0.40 | **0.09** | days + heavy verify | intended, not provable |
| D | `uc_pingpong` 4×4 KB chained vs 1×16 KB | 0.15 | 0.15 | med-low | 0.40 | **0.06** | **minutes, zero code** | n/a (wire only) |
| E | Concurrent encoder, 1–3 hand-proved sibling pairs | 0.25 | 0.25 | low | 0.25 | **0.06** | weeks | no (race, not error) |
| F | Q4_K on `attn_q_b` / `out_a` / `out_b` only | 0.35 | 0.35 | low | 0.25 | **0.09** | days + quality gate | no |
| G | **FlashAttention restructuring** (§3.3.1) | 0.0–2.0, mid 1.0 | 0.0–2.4 | mech. med, size **unknown** | 0.30 | **0.30** | weeks–months | G-a variant only |
| | **raw total** | **1.57 + G** | | | | | | |
| | **discounted, A–F only** | | | | | **0.60** | | |
| | **discounted, A–G** | | | | | **0.90** | | |

**0.90 ms discounted against 4.34 ms needed — 21% of the target. Without the
unsized FlashAttention item it is 0.60 ms, 14%. It falls short, by 3.4 ms.**
Even the raw total of every item at full value, with G at its optimistic 2.0 ms,
is 3.57 ms and does not reach 50 t/s.

**On the 131k column.** Every item here targets a context-invariant stage, so the
absolute saving carries over almost unchanged — but the token is longer
(`compressor_indexer` alone goes 0.199 → 9.712 ms), so the *relative* value
falls: 0.90 ms is 3.7% at 2k and 2.3% at 131k against the 38.4 ms raw token.
**G is the only item whose 131k value exceeds its 2k value**, because
`attn_inv_rope` grows 3.813 → 4.220 raw.

### 2.2 Repricings I applied, and why

- **C was scouted at 0.49 ms and refuted to 0.30.** I priced it lower still.
  The dispatch price is contested across an 11.6× range (§8); 86 one-thread
  no-memory dispatches at the rig's own in-situ marginal cost of **1.9 µs**
  (`tp_decode_investigation.md` §6, which says in terms *"use 1.9–4.4 µs, not
  8.6"*) is **0.16 ms**; at the ballast's 3.74 µs it is 0.32. I took 0.22 and
  note that `DS4_METAL_DISPATCH_BALLAST` in the live graph settles it for free.
- **E is four pool entries, one mechanism.** "Concurrency islands" (0.30),
  "C1 concurrent decode encoder" (0.60), "globally-concurrent encoder" (0.25),
  "un-disqualify the concurrent FFN encoder" (0.10) are the same
  `MTLDispatchTypeConcurrent` change at `ds4_metal.m:1129-1130`. **Counted once.**
  The three estimates that actually measured something converge on 0.2–0.4 ms;
  the one that survived review is the one whose "28% recoverable" is explicitly
  *"inference, from reading ds4.c."* `TP-A0-ROWSPLIT-TEST-PLAN.md:968-1002` —
  titled *"Concurrent encoder — I mis-sized this, twice over. Corrected
  2026-08-27"* — sizes it at ~0.5 ms and demotes it to last. No lens cited it.
- **B was not in the pool at all.** Nobody was assigned load balance. It is a
  *measured* item: `TP-A0-ROWSPLIT-TEST-PLAN.md:743-762` prices the straggler at
  **0.50 ms** from a direct clean-vs-broken comparison (+11.6 µs/layer), the
  fourth successive downward revision (0.82 → 0.74 → 0.66 → 0.50).
  `SCOPE-TP-GATE-OVERLAP.md:498` item 2 estimates ~0.3 ms for the cheap variable
  shared-split variant, which drives FFN-gate imbalance to zero without a
  repacked GGUF. I took 0.30.
- **D was not in the pool either.** `SCOPE-TP-GATE-OVERLAP.md:496` item 1: *"zero
  engine change, minutes, ≤ 0.34 ms (1.4%), no gate."* It is the cheapest live
  item in the entire campaign and no lens mentioned it.
- **F was scouted at ~1.0 ms and refuted to 0.5.** I took 0.35. Its base was
  overstated 27%: `attn_out_proj` is **2.73 ms, not 3.72** — the 3.72 is a
  pre-split mislabel that contained the 0.99 ms ATTN TP gate, corrected in-tree
  on 2026-08-27 (`ds4.c:23860` and `:23865` are two markers today). **Only the
  reduce-the-bytes reviewer caught this; the brief handed the stale number to all
  six lenses.** The 1.26× speed factor is also not a Q4_K measurement — it is
  MXFP4-MoE-gather vs Q8_0-dense, cross-kernel.
- **A was scouted at 0.5 ms and refuted to 0.4 on M1 Max.** I took 0.30 for the
  rig. The M1 Max → M2 Ultra transfer is entirely unmeasured, and the one
  same-kernel same-shape cross-machine comparison on record is **4.13×** in the
  rig's favour (`attn_q_b`: 119 GB/s dev box vs 492 GB/s rig). Four of six lenses
  made their decisive measurement on the dev box.

### 2.3 The corrected stage table, and what has no candidate at all

This is the reference table for the rest of the document. It is the brief's net
table with the `attn_out_proj` / `attn_tp_gate` marker split applied
(`TP-A0-ROWSPLIT-TEST-PLAN.md:743-752`; the brief carried the pre-split label
3.72, which is also a *raw* figure sitting in a net table) and the FFN gate
broken out of `ffn_hc_post`. Net = raw − 0.18 ms/marker, the campaign's own
correction.

| stage | net ms | % | surviving candidate |
|---|---:|---:|---|
| `routed_moe_folded` | 4.72 | 19.4% | **none** — 367 vs ~400 GB/s isolated = 92% |
| `attn_inv_rope` | 3.63 | 14.9% | G only, unsized |
| `attn_out_proj` | 2.55 | 10.5% | F (partial) |
| `q_a_kv_proj` | 1.97 | 8.1% | **none, and correctly so** — 895.6 MB / 1.97 ms = **455 GB/s** |
| `q_path` | 1.70 | 7.0% | F (`q_b` half) |
| `q_lora_norm` | 1.50 | 6.2% | A |
| `ffn_tp_gate` | 1.40 | 5.8% | B, C, D |
| `attn_hc_pre` | 0.96 | 3.9% | **none** |
| `ffn_hc_pre` | 0.93 | 3.8% | **none** |
| `router` | 0.92 | 3.8% | E only |
| `attn_tp_gate` | 0.81 | 3.3% | C, D |
| `shared_gate_up` | 0.79 | 3.2% | E only |
| `shared_down` | 0.49 | 2.0% | **none** (the Q4_K residue, ~0.1–0.16 ms, was refuted out) |
| `ffn_hc_post` residue | 0.34 | 1.4% | **none** |
| `attn_hc_post` | 0.29 | 1.2% | **none** |
| `compressor_indexer` | 0.02 | 0.1% | n/a — this is the 131k term (9.71 ms there) |
| **sum of markers** | **23.02** | **94.6%** | |
| **residual, incl. the output head** | **1.32** | **5.4%** | **none, contents unknown** |
| **token** | **24.34** | | |

**≈13.5 ms — 55% of the token — has no surviving candidate**, and the two
largest stages, 8.35 ms together, have none at all.
`TP-A0-ROWSPLIT-TEST-PLAN.md:391-393` already said so in writing before this
round started: *"the two largest items (`routed_moe` 4.72 and `attn_inv_rope`
3.63, together 34% of the token) remain without a proposal."* Six lenses ran and
that sentence is still true.

**Two corrections to the brief's framing of this table:**

1. *"~9% of the token sits in stages that have markers but were never
   reported"* is wrong. The markers sum to 94.6%; the residual is **1.32 ms /
   5.4%**. The 9% is the *profiled-epoch* gap (28.657 busy − ~25.7 stage sum),
   which is not net of the 0.18 ms/marker tax and should not have been carried
   into a net table. Scouts were told there was 2.2 ms of unexamined slack.
2. **The `attn_out_proj` row is internally inconsistent and R1 must settle it.**
   1533.4 MB in 2.55 ms is **601 GB/s**, above every Q8_0 rate ever measured on
   this rig (best 581 at k=2048→8192). Using the raw 2.73 gives 562, still above
   the k=4096 isolated 541. The clean ablated figure at ctx 512 is 2.88 ms =
   532 GB/s = 98% of isolated, which is the only self-consistent one. Either the
   flat 0.18 ms/marker correction is wrong for this marker, or the profile is,
   or the byte model is. **This is precisely what a 1–3%-distortion instrument
   removes**, and it is a live example of why every number above carries a
   hand-correction I would rather not be carrying.

## 3. WHERE THE 15 MS GOES

Short answer: **3.7 ms of it was never a gap** (the byte model omitted a third of
the weights), **~1.3 ms is irreducible wire**, and the remaining **~9.9 ms is
work that never had a byte floor and to which a bandwidth roofline was never
applicable.** The engine is ~98% efficient on the half of the token that has a
roofline and roughly 40-50% efficient on the half that is a latency graph.

### 3.1 The corrected byte model — derived from the layout assertions, not from another estimate

Per rank, per token, at ctx 2048. Q8_0 = 34 B/32 = 1.0625 B/param; MXFP4 =
17 B/32 = 0.53125; F16 = 2. Shapes from `weights_validate_layout`
(`ds4.c:5124-5180`) and `DS4_SHAPE_FLASH` (`ds4.c:581`).

| stream | shape / count | MB/token | split? |
|---|---|---:|---|
| `attn_q_a` | [4096,1024] Q8_0 × 43 | 191.6 | replicated |
| `attn_kv` | [4096,512] Q8_0 × 43 | 95.8 | replicated |
| `attn_q_b` | [1024,32768] Q8_0 × 43 | 766.7 | head half |
| `attn_output_a` | [4096,8192] Q8_0 × 43 | 766.7 | 4 of 8 groups |
| `attn_output_b` | [8192,4096] Q8_0 × 43 | 766.7 | k-slice |
| *attention subtotal* | | *2587.5* | **= §3's verified 60.17 MB/layer** ✓ |
| `attn_compressor_kv` + `_gate` | F16, 21 × 2 × [4096,1024] + 20 × 2 × [4096,512] | 520.1 | **replicated, omitted** |
| `indexer_compressor_kv` + `_gate` | F16, 21 × 2 × [4096,256] | 88.1 | **replicated, omitted** |
| `hc_attn_fn` + `hc_ffn_fn` | F16, 43 × 2 × [16384,24] | 67.6 | **replicated, omitted** |
| `ffn_gate_inp` (router) | F16, 43 × [4096,256] | 90.2 | **replicated, omitted** |
| shared gate + up + down | Q8_0, 43 × 3 × [4096,2048] ÷ 2 (`ds4.c:24159`) | 574.9 | **half, omitted** |
| routed MoE, 3 local experts | MXFP4, 43 × 3 × 13.369 MB | 1724.6 | shard |
| output head | Q8_0 [4096,129280] ÷ 2 | 281.3 | **vocab half, omitted** |
| **weights total** | | **5934** | |
| KV / compressed-KV cache DRAM footprint @2k | 43 × ~2 MB | ~90 | |
| flash partials, activations, HC re-reads | | ~100 | mostly SLC |
| **total DRAM traffic** | | **~6.1 GB** | |

`ratio` per layer is `il<2 → 0`, even → 4, odd → 128 (`ds4.c:1120-1124`), so 21
layers carry the ratio-4 compressors (`comp_width = 2 × 512`) plus the indexer
compressors, 20 carry the ratio-128 ones (`comp_width = 512`). The compressors
are read **unconditionally every token** — the quad-compressor fusion at
`ds4.c:22478-22515` fires inside the `q_a_kv_proj` marker span
(`ds4.c:22431 → 22612`), and only the *state append* is position-gated.

Two independent cross-checks, neither of which is another estimate:

- Non-routed weights over the whole layout sum to **8.19 GiB** against the
  **8.26 GiB** implied by (model file 145.26 GiB − routed experts 137 GiB).
  Match to 0.8%. §3's startup-log figure of 76.73 GiB resident per rank
  reconciles as 68.5 routed + ~8.2 non-routed.
- **`q_a_kv_proj` closes on its own.** Stage bytes = q_a 191.6 + kv 95.8 +
  compressors 608.2 = **895.6 MB in 1.97 ms = 455 GB/s**, i.e. exactly the rig's
  measured k=4096 Q8 ceiling of 445–458 (`BENCHMARKS-TP-PP.md:352-357`). Under
  the old byte model the same stage read as 146 GB/s and one lens picked it as
  its headline underperformer. It has no headroom at all.

### 3.2 The corrected floor

| floor method | ms | t/s |
|---|---|---|
| 4.29 GB @ 450 GB/s (**the brief's**) | 9.53 | 105 |
| 5.93 GB @ 450 GB/s flat | **13.19** | **76** |
| 5.93 GB, each stream at its own measured shape rate (§5 of the investigation) | **13.18** | **76** |
| actual | 24.34 | 41.1 |

The two corrected methods agreeing to 0.01 ms is a coincidence of rounding, but
they are genuinely independent constructions and they land in the same place.
**Gap = 11.15 ms, ratio 1.85×.** The brief's 15 ms and 2.6× are both artifacts of
methodology rule 1 failing at the top of the budget instead of inside a stage —
which is the direction the rule was written to prevent.

### 3.3 Attribution of the 11.15 ms

Against the §2.3 net table. Each stage's floor is its own bytes at its own
measured shape rate from `tp_decode_investigation.md` §5.

| term | ms | how it is derived |
|---|---:|---|
| **FlashAttention above every roof it has** | **3.43** | 3.63 measured − ~0.20 DRAM floor. §3.3.1 |
| **HC mix, four markers, above its 0.33 ms byte floor** | **2.19** | (0.96 + 0.93 + 0.34 + 0.29) − 0.33. **But ablation caps the removable part at 0.76 ms** |
| **`q_lora_norm` — reads no weights at all** | **1.50** | 43 dispatches; the serial FP8 amax tree is 81% of the kernel |
| **TP wire, irreducible** | **1.30** | 86 gates × ~15 µs one-way at 16 KB |
| **TP gates above the wire floor** | **0.91** | (0.81 + 1.40) − 1.30. **0.50 of it is the measured MoE straggler** |
| **residual, incl. the output head** | **0.80** | 1.32 − an *assumed* 0.52 for the head. **Unfalsifiable today — no marker exists** |
| **`router` above its 0.30 ms byte floor** | **0.62** | ~0.22 is profiler tax, ~0.3 is the 1-threadgroup bitonic select + 2 dispatches (`SCOPE-ATTNOUT-ROUTER-SHARED.md` §3.2-3.3) |
| **routed MoE above its isolated MXFP4 rate** | **0.41** | 367 vs ~400 GB/s = 92% |
| bandwidth slack elsewhere (`q_a_kv` 0.12, shared 0.11) | 0.23 | 90–97% of shape rate |
| `attn_out_proj`, `q_path` | 0 | both at or above their isolated rates — see the §2.3 caveat |
| **total attributed** | **11.4** | against 11.15 + 0.20 (FA DRAM) + 1.30 (wire) = 12.65 of the 24.34 that is not weight bytes; the ~1 ms slack is the accumulated 0.18 ms/marker hand-correction |

**Attributed is not recoverable, and the distinction is the whole point.**

- **~1.3 ms is physically irreducible under TP2** (wire).
- **~2.2 ms is measured as un-removable.** `DS4_TP_ABLATE=hcpre` deletes the
  entire HC mix and recovers **0.76 ms end-to-end** against 0.97 ms of per-stage
  delta and 2.24 ms of profile attribution — the *2.7× profile-vs-ablation
  disagreement*, reproduced at 2k (`BENCHMARKS-TP-PP.md:409-413`). Whatever the
  profile says is in those stages, only a third of it exists. **And if that 2.7×
  is a general property of small stages rather than an HC-specific one, every
  item in §2 is over-priced by up to the same factor** — see Q3.
- **~0.4 ms in the MoE is closed** — 92% of isolated, and both format levers
  (M1 packed loads, M3 planar repack) are dead.
- **~3.4 ms in FlashAttention is the only large term whose mechanism is unknown**
  and whose instrument has never worked.
- **~0.8 ms is genuinely unattributed** and one of its known contents — the
  output head — has no decode-stage marker anywhere in `ds4.c`, so its 0.52 ms
  is an assumption used to declare itself at the roof.

#### 3.3.1 Why FlashAttention is the only stage off its roof, and by how much

`attn_inv_rope` spans `ds4.c:23330 → 23582` and is overwhelmingly the decode
FlashAttention call. Reconciled against the model rather than asserted:

> **Everything in this subsection rests on one unmeasured input: `n_keys` at
> ctx 2048.** I take ~640 as an inference from the indexer's 512-way top-k plus
> the raw window, supported by the stage's near-context-invariance (3.813 → 4.220
> raw over a 64× context increase, which is only possible if attended keys are
> bounded). **R2 prints the exact value with no code change.** If `n_keys` is in
> fact ~2048, the cache-traffic figure rises to ~1.6 TB/s and this whole
> subsection collapses.

- **DRAM.** The KV working set is `n_keys × 512 × 2 B` per layer. At ctx 2048
  with the indexer bounding attended keys, ~640 keys gives ~655 KB/layer,
  ~28 MB/token, plus the flash partial buffers (`nrows × 512 × nwg × 4` written
  then read) at ~60 MB. **Floor ~0.2 ms. Measured 3.63.**
- **FLOPs.** 32 local heads × ~640 keys × (512 QK + 512 PV) × 2 = ~1.8 GFLOP/token
  = **~500 GFLOP/s = 2.4% of the 21 TFLOP/s FP32 peak.** Floor ~0.09 ms.
- **Cache traffic.** `k` and `v` are the *same buffer* in this MLA layout
  (`nb21 == nb11`, both `g_flash_attn_kv_buffer`), and the kernel loads the row
  twice — once at `metal/flash_attn.metal:1130` for the QK dot and again at
  `:1218` for the PV accumulation. With 32 local heads that is a **64× re-read of
  a 655 KB working set**: ~1.8 GB/token of L1/L2 load traffic in 3.63 ms =
  **~496 GB/s.** That is DRAM-class bandwidth being extracted from a cache-
  resident working set that should deliver multiples of it.
- **Grid.** `nwg` is *not* hard-wired to 32 as the brief states — under TP2
  `ds4_gpu_flash_attn_decode_nwg` (`ds4_metal.m:3819-3842`) buckets it to
  4/5/12/16/24/32 by key count, so at 2k it is ~24. The `fa_core` grid is
  `(1, n_head, nwg)` = ~768 threadgroups of **one simdgroup each**, residency-
  capped near 9/core by 3,328 B of threadgroup memory. The `reduce` grid is
  `nrows = n_head` = **32 threadgroups on 60 cores** (`ds4_metal.m:29526`; the
  vec dispatch is at `:29498`), which is a real underfill but a small kernel.
- **Instruction issue.** ~1,600 lane-instructions per threadgroup × 768 / 60
  cores / 4 issue pipelines ≈ **4–15 µs/layer** against ~84 µs/layer measured.

**Every floor available says the same thing: 3–10× at least, and the largest
single pool of time in the token.** But note carefully what this does *not* say.
It does not say the fix is "more threadgroups" — methodology rule 7, and the
`packed32` experiment (−1.35 t/s) and pr-778 (+0.12%) both failed exactly that
argument. The mechanism that the byte model points at is **arithmetic intensity,
not occupancy**: batch H heads per threadgroup so one KV row load serves H
heads, and/or reuse the K load for the PV pass. Nothing in the tree does this —
`kernel_dsv4_flash_attn_vec_packed32_reduce_rope_f16_dk512_dv512`
(`ds4_metal.m:29309`) fuses vec+reduce into one threadgroup per head, which is a
*dispatch* fusion, not a head batching, and under TP2 it never fires anyway
because its admission requires `n_head == 64` and the rank has 32.

**I am deliberately not putting a number on this.** The size depends on `n_keys`
at ctx 2048, which I could not determine by reading, and on where inside the
stage the time actually is, which **no working instrument has ever reported**.
Both are settled by two runs in §5, one of them zero-code.

### 3.4 The absolute physical floor

| term | ms | why it is a floor |
|---|---|---|
| weight bytes at each stream's shape rate | 13.18 | §3.1–3.2 |
| TP wire, 86 gates | 1.30 | measured half-RTT, cannot overlap a barrier |
| FlashAttention at 8× intensity improvement | ~1.0 | still 5× its issue floor |
| everything else, halved | ~2.5 | dispatch floor at ~600 narrow dispatches × 1.9–4.4 µs = 1.1–2.6 ms alone |
| **floor** | **~18.0** | **~55 t/s** |

50 t/s = 20.0 ms sits **1.11× above this floor**. It is not physically excluded.
It requires the FlashAttention rewrite *and* roughly half of the remaining
latency block, which is a multi-month kernel programme, not a campaign of 1%
items.

## 4. SEQUENCED PLAN

**Programme rule, before any item.** No candidate ships on a throughput A/B
alone. Four of the seven degrade output *smoothly* and get *faster* when broken,
and that is not a coincidence: in every case the corruption comes from skipping
work, and skipped work is faster. This is exactly how `3f17e83` shipped as
default and had to be reverted by `da63283`. Every arm produces a paired output
artefact — bit-identical logits from **both ranks**, or top-1 + max |Δlogit| +
perplexity where bit-identity is impossible.

**Prerequisite, half a day, blocks two items.** `make check-dispatch-count` is
**red on this tree** — I ran it: *"dispatch counter incomplete: 239 dispatch
sites, 238 DS4_DISP (want 240)"*, with the two unwrapped sites at
`ds4_metal.m:18739` and `:18974`, both in the compressor/indexer region. Any
item whose claim is "N dispatches removed" (C, E) is being audited by a counter
that is already miscounting. It is the only automated guard the tree has against
exactly the class of "host and shader disagree about the graph" error that
produced `da63283`. Fix it first.

### Phase 0 — measurement. One rig session, all six arms are zero-code.

This phase decides more than every candidate in the pool combined, and its
marginal cost over the existing baseline run is minutes.

| # | arm | decides | code |
|---|---|---|---|
| 0.1 | `DS4_METAL_GPU_ENCODER_TIMESTAMPS=1` at 2k | **re-baselines the whole budget at 1–3% distortion instead of 18%**, decomposes into ~172 labelled spans instead of 14, locates the ~1.3 ms residual and the output head | **none** |
| 0.2 | `DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1` at 2k, **read only the shape fields** | `n_keys` and `n_comp` per layer at ctx 2048 — the single missing input to sizing item G | **none** |
| 0.3 | `DS4_METAL_DISPATCH_BALLAST` ∈ {0,2} at 2k | the dispatch price in the **live graph**, settling an 11.6× spread that four candidates live or die on; prices item C exactly | **none** |
| 0.4 | `powermetrics --samplers gpu_power` during decode vs `tests/bench_membw` | retires the "30 W / 20% utilised" premise either way | **none** |
| 0.5 | `DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_QUAD_FUSE=1` + stage profiler | confirms the 608 MB compressor accounting, i.e. §3.1 | **none** |
| 0.6 | startup shard log, itemised by tensor class | confirms replicated-vs-split before §3.1 is quoted anywhere else | none |

**Gate on Phase 0:** if 0.1 disagrees with the hand-corrected net table by more
than ~0.3 ms on any stage, **believe 0.1** (that is what the tree says at
`ds4_metal.m:1033-1041` and what `TP-A0-ROWSPLIT-TEST-PLAN.md:807-809` already
queued) and re-rank §2 before writing any code.

### Phase 1 — the two items worth doing whatever Phase 0 says

**1.1 — D: `uc_pingpong`, 4 × 4 KB chained vs 1 × 16 KB.** Minutes, zero engine
change, ≤ 0.34 ms. `SCOPE-TP-GATE-OVERLAP.md:319-325` shows the observed
14.5–15.5 µs is ~4 µs above the wire model and proposes chained WRs as the
explanation. *Correctness gate: none — it is a transport-level A/B on a
model-free harness. If it wins, the engine change is a WR-posting change with
byte-identical payloads, gated on bit-identical logits on both ranks.*

**1.2 — A: parallelise the FP8 amax block loop.** Days. The strongest candidate
in the pool and the only one where bit-identity is a *provable* property: `max`
is exact, associative and commutative over the finite non-negative `abs()` set,
and `fp8_scale = exp2(ceil(log2(amax/448)))` is a pure function of `amax`. The
loop is at `metal/norm.metal:340-372` — 7 serial blocks, 8 `threadgroup_barrier`s
each, 64 of 256 threads live — and it is **81% of the kernel** (in-kernel
bisection: 4.03/10.25/12.96/21.36/39.74 µs at 0/1/3/7/15 blocks).

*Correctness gate, all five required:*
1. Runtime guard `n_simdgroups >= (n_nope + 63)/64` **and** `n_nope % 64 == 0`,
   with fallback to the serial loop. At `n_lora_q = 1024` the dispatch has 256
   threads = 8 simdgroups ≥ 7 blocks, but the thread count derives from `q_n`,
   not `kv_n`; a 512-rank q-LoRA would give 4 simdgroups and blocks 4–6 would
   **never be computed**, silently, and faster. The `% 64` clause is required
   because the existing serial loop has a latent stale-scratch read that never
   fires at `n_nope = 448`, and a `simd_max` over live lanes only would not
   reproduce it.
2. Host `setThreadgroupMemoryLength` computed from the **same `#define`** the
   shader uses. Today it is `(32u + 64u) * sizeof(float)` = **384 B with
   zero slack** (`ds4_metal.m:22551`) against a shader that aliases `scratch =
   shmem_f32 + 32` (`metal/norm.metal:345`). The parallel form needs
   32 + 7×64 = 1920 B. Metal does not bounds-check dynamic threadgroup memory.
   **Two hand-written constants that must move together is literally the C2
   shape.**
3. Byte-exact KV-cache dump, old vs new, ≥64 decode steps — plus a
   prefill-then-decode-the-same-text equality check, because the same amax loop
   exists at `metal/dsv4_kv.metal:132` and `:230` and `metal/dsv4_rope.metal:543`
   and `:711`, and changing only the decode copy is correct but must be *shown*
   to be (the failure mode is prefix-cache-reuse divergence, invisible otherwise).
4. Bit-identical logits on both ranks, ≥1024 greedy tokens, at pos < 128 and
   pos > 128.
5. U11-style prefill non-regression at 2k/32k/131k. Expected to be a clean null;
   if it is not, something else moved.

*Ledger caveat to carry:* at pos ≥ 128 the token splits across 3 command
buffers, so on 2–3 layers `phase != METAL_DECODE_LAYER_FULL` and the triple
fusion does not fire at all. Those layers get nothing. This caps the win below
43/43 of the stage.

### Phase 2 — the ten lines that decide whether 46 t/s exists

**2.1 — Rewire A0's four boundaries onto the encoder-timestamp path.** The four
sites (`gather`, `packed`, `fa_core`, `reduce`) currently call
`ds4_gpu_flash_attn_stage_profile_boundary`, which ends the command buffer and
takes a **host wall clock** either side (`ds4_metal.m:11040-11050`). At decode
the round trip *is* the measurement, which is why it reports per-call figures 6×
the stage they belong to — stated in the tree at `ds4_metal.m:1030-1034` and
retracted at `TP-A0-ROWSPLIT-TEST-PLAN.md:765-780`. Replace each with
`ds4_gpu_ts_label()` + an encoder close. ~10 lines. **This splits the largest
unexplained block in the token in one run.**

*Caveat found while reading:* the timestamp path is gated on
`!g_batch_encoder_concurrent` (`ds4_metal.m:1117`), so item E and this instrument
are mutually blind. Sequence accordingly.

**2.2 — Decide item G on the 2.1 + 0.2 data, not before.** If `fa_core` carries
most of the 3.63 ms and `n_keys` is small, the mechanism in §3.3.1 is the target
and the item is worth funding. If the time is in `gather`/`reduce`/dispatch
latency, it is a different and much smaller item, and the 46-48 t/s branch does
not exist.

### Phase 3 — medium builds, re-ranked on Phase 0 output

**3.1 — B: variable shared-expert split to equalise rank arrival.** The
prerequisite named at `SCOPE-TP-GATE-OVERLAP.md:498` (*"I1 must show
`exchange_FFN > exchange_ATTN`"*) is **already satisfied**: the direct marker
split gives `ffn_tp_gate` 1.643 against `attn_tp_gate` 0.994, a 1.65× ratio, and
only the FFN gate sits behind the routed shard. The straggler is measured at
**0.50 ms** (`TP-A0-ROWSPLIT-TEST-PLAN.md:760`, +11.6 µs/layer on clean
attention — note the broken-C2 run gave +15.4 µs, because degenerate routing
concentrates experts; use the clean number). The real obstacle is dispatch
shape: `k` is only known after the router runs on the GPU, so this needs either
an ICB or both ranks dispatching the full shared grid with each threadgroup
predicated on `k` from a device buffer.
*Correctness gate: not bit-exact (it repartitions the shared expert's sums), so
T2 bar — top-1 preserved over ≥2000 greedy tokens with reported max |Δlogit|,
on both ranks, plus U11 prefill.*
*Do the shared-shift, not the repack.* Design C (repacked TP-layout GGUF) buys
0.50 ms for a conversion tool and a TP-specific model file; the shared-shift
buys ~0.3 ms for neither. Design A costs 27 GiB and a third of the usable
context. Design B is blocked (`ffn_down_exps` splits on the inner dim).

**3.2 — C: TP-gate `flag_set` fold. Only if 0.3 returns ≥ 0.25 ms.** If ballast
prices 86 one-thread dispatches below that, **do not write the code**. This is
the highest-risk item on the list by a distance: the encoder-level buffer hazard
being deleted is the *only* ordering guarantee that the payload is visible before
the arrival flag (`ds4_metal.m:10704-10718`: a `MTLSizeMake(1,1,1)` x `MTLSizeMake(1,1,1)` dispatch
followed immediately by `ds4_gpu_close_batch_encoder()`).
Replacing it with a completion counter requires the last threadgroup's
system-scope coherent store to be ordered after every producer's device-release
**as observed by the CPU across a DMA read**, which Metal does not specify. A
torn payload makes the peer reduce a stale partial: logits degrade smoothly and
**t/s goes up, because the gate wait shortens — which is the claimed win.** The
TP2 wire cannot catch it: `tp_hello_exchange` (`ds4_tp.c:1396-1444`) compares
model identity only, and the per-gate frame (`ds4_tp.c:1621`) is
`{magic, layer, gate, seq}` with **no payload checksum**.
*Correctness gate: bit-identical **pre-reduce partials** dumped from both ranks
(not just final logits) over ≥1024 tokens at pos < 128 and pos > 128; plus a
**widened-window stress arm** with an artificial delay injected into the payload
kernel's slowest threadgroup, because a clean run at the native ~3.4 µs window is
a probabilistic sample of a race, not a proof; plus prefill, since a stale
attention partial during prefill corrupts the KV cache for the whole generation
and is invisible to first-token latency.*

**3.3 — F: Q4_K on `attn_q_b` / `attn_output_a` / `attn_output_b`.** Model-file
change only; the paths are live and I confirmed them (`ds4.c:4403-4443`,
`kernel_mul_mv_q4_K_dense_f32` at `ds4_metal.m:19869`, TP fallback at
`ds4.c:26253-26273`, group0 handling at
`ds4_metal.m:26730-26753`). **Use `--tensor-type` overrides, never the
`--attention-proj q4_K` family flag** — `is_attention_projection`
(`gguf-tools/deepseek4-quantize.c:1198-1202`) also matches `attn_q_a` and
`attn_kv`, and `attn_kv` is the KV-LoRA down-projection whose 4-bit error is
written into the FP8 KV cache and **persists and compounds for the whole
generation**. That surface is unpriced and invisible in a short greedy diff.
*Correctness gate: (i) a numerical gate on the kernels first, against an F32
reference on the **block-diagonal** `kernel_dsv4_attn_out_low_q4_K_f32` shape
(`nei0/ne02 = group_cnt`), not a dense [4096,4096] matvec — the existing harness
benches the wrong kernel and has **zero output verification**; (ii) ≥2000 greedy
tokens, top-1 agreement + max |Δlogit| + perplexity on a held-out fixture;
(iii) both TP2 ranks and single node, since only the TP path takes the
two-dispatch fallback; (iv) U11 prefill — the Q4_K attn-out-low kernel runs
nsg=2 / 32 B threadgroup memory against Q8_0's nsg=4 / 256 B, so prefill can
regress while decode wins, and `ds4_gpu_attention_output_q4_K_batch_tensor` hard-
returns at `n_tokens < 32u` (`ds4_metal.m:25985`) so the two paths are different
kernels.*
*Structural safety note, unique to this item:* `gguf_bytes` differs after
requantisation, so a half-requantised rank pair aborts in the hello. It is the
only item on the list with cross-rank safety by construction.
*Also note:* `ds4_gpu_attention_output_low_q4_K_slice_tensor` is
**production-dead code today** with no test coverage beyond `tests/test_q4k_dot.c`.
First execution in a TP2 decode is not where you want to discover a wrong slice.

### Phase 4 — expensive, and only with Phase 2 data in hand

**4.1 — G: FlashAttention restructuring.** Two variants, do them in this order.
- **G-a, V-reuse (possibly bit-identical):** `k` and `v` are the same buffer;
  hoist the row load so one fetch serves both the QK dot and the PV
  accumulation, preserving arithmetic order exactly. Halves the cache traffic. If
  the order is genuinely preserved this is gateable by bit-identity, which makes
  it the only safe way into this kernel. Blocker to check first: staging a C=32
  key block is 32 KB of threadgroup memory, at the limit.
- **G-b, head batching (not bit-identical):** H heads per threadgroup, one KV
  row load serving H heads. Divides cache traffic by H. Changes the reduction
  tree, so top-1 + bounded Δlogit only.
*Do not justify either as "more threadgroups".* Two grid-widening attempts have
already failed outright here (`packed32` at −1.35 t/s, pr-778 at +0.12%), and the
argument that survives is the traffic argument, not the occupancy argument.

**4.2 — E: concurrent encoder, at most 3 hand-proved sibling pairs.** Never a
graph-wide sweep of 238 sites. Phase 1 (barrier at every edge) is provably
equivalent, free, **and yields exactly zero**; all the value and all the risk are
in barrier removal. The only well-shaped pair in the graph is `router` (0.92) ∥
`shared_gate_up` (0.79); `SCOPE-TP-GATE-OVERLAP.md:124` shows the rest of the
FFN region is chained off `ffn_norm`, and `tp_fold_ffn` (`ds4.c:24984`) makes
shared and routed a hard dependency by construction.
*Correctness gate: for each pair, (i) a written read/write-set proof for both
`n_tok == 1` and `n_tok >= 32` — the batch encoder is shared with prefill, where
the same site has a different grid and different write extents; (ii) a debug
build that reinserts a barrier at every removed edge and asserts byte-equality
of the full logits buffer, ≥4096 tokens, both ranks; (iii) a widened-window
stress arm; (iv) U11 prefill.*
*Two structural obstacles to price before starting:* the concurrent path is
force-disabled whenever the stage profiler runs (`ds4.c:22180`, `:22230`), so the
campaign's only decomposition instrument cannot see the change it is meant to
validate — and neither can the encoder-timestamp profiler (`ds4_metal.m:1117`).
And the TP fence-wait spin (`kernel_dsv4_tp_fence_wait`, one thread) becomes a
**livelock** surface if a spinning threadgroup is co-resident with the one that
would release it.

## 5. RIG REQUESTS

Priority order. **R1–R5 and R7 are zero-code** — they set an environment variable
on an existing binary. **R6 needs ~10 lines written first.** All TP arms must run
with **identical env on both ranks**: `tp_hello_exchange` (`ds4_tp.c:1396-1444`)
compares `gguf_bytes / model_id / n_layer / n_embd / n_vocab / quant_bits /
gate_slot_start / gate_slot_step / gates_per_token` and **nothing about env vars,
build defines or kernel selection**, so a one-rank arm produces silently wrong
logits with no error anywhere.

Common prelude for every rig arm (both hosts): `sudo sysctl
iogpu.wired_limit_mb=120000`; coordinator = lanfear (`rdma_en6`), worker = mat
(`rdma_en7`); launch coordinator first.

Short-context arm skeleton, referred to below as **`<2K-ARM>`**:

```sh
# lanfear (coordinator)
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role coordinator \
  --tensor-parallel --transport rdma --listen 0.0.0.0 1234 \
  --rdma-device rdma_en6 --prompt-file ~/Downloads/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --gen-tokens 128 --csv /tmp/<name>.csv
# mat (worker)
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role worker \
  --coordinator 192.168.0.6 1234 --transport rdma --tensor-parallel \
  --rdma-device rdma_en7
```

---

### R1 — Re-baseline on the non-distorting profiler. **Zero code. Highest value on the list.**

```sh
DS4_METAL_GPU_ENCODER_TIMESTAMPS=1 <2K-ARM>          # both ranks
```

**Decides:** the entire budget. Every number in this document, in the brief, and
in every scout report is downstream of a table taken at **18% distortion** with a
hand-applied 0.18 ms/marker correction. This instrument (`df0037e`,
`ds4_metal.m:1026-1135`) samples the GPU timestamp counter at encoder boundaries
inside one command buffer at **1–3%** overhead, and the engine already creates
~172 labelled encoders per token. It was built on 2026-08-27, queued at
`TP-A0-ROWSPLIT-TEST-PLAN.md:807-809`, and **has never been run.**

Specifically it settles, in one run and with no cross-epoch subtraction: where
the ~1.3 ms residual is; where the **output head** is (it has no decode-stage
marker anywhere in `ds4.c`, and its 0.52 ms is an *assumption* — 281 MB divided
by an assumed 540 GB/s, then used to declare the stage "at the roof", which is
circular); whether `attn_out_proj` net is 2.55 or 2.73; and whether the 2.7×
profile-vs-ablation disagreement on the HC stages is an instrument artifact or
real. Buffer holds 512 encoders ≈ 3 tokens; report fires per batch, so read the
`enc-ts` lines, not just the summary.

**Watch for:** `g_ts_n < DS4_GPU_TS_MAX_ENCODERS` silently drops encoders past
512 within one command buffer. If the reported encoder count pins at 512, reduce
the sampled window before believing the sums.

### R2 — The one missing input to sizing FlashAttention. **Zero code.**

```sh
DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1 \
DS4_METAL_FLASH_ATTN_STAGE_PROFILE_EVERY=43 <2K-ARM>
```

**Decides:** `n_keys` and `n_comp` per layer at ctx 2048, which is the input I
could not obtain by reading and on which the entire sizing of item G depends.

**Read the shape fields, not the timings.** The line is
`mode=%s tokens=%u comp=%u keys=%u heads=%u dim=%u window=%u ratio=%u
<stage>=%.3f ms` and the shape fields are exact. The `%.3f ms` values are
**6× inflated** — the boundary calls `ds4_gpu_end_commands()` and takes a host
wall clock (`ds4_metal.m:11040-11050`), so at decode it measures a command-buffer
round trip, not kernel time. This is already known and written down
(`TP-A0-ROWSPLIT-TEST-PLAN.md:765-780`); the arm is being run *only* for the
shapes. Note the print says *"FlashAttention prefill stage"* even for decode
calls — the discriminator is `mode=decode_gathered` (`ds4_metal.m:11046`), not
the word "prefill". Do not discard the output on that basis.

Prediction to falsify: if `n_keys` at 2k is ~512–700 (indexer-bounded), the 64×
KV re-read model in §3.3.1 holds and item G's ceiling is 1.5–2.5 ms. If `n_keys`
is ~2048, the cache-traffic figure rises to ~1.6 TB/s and the kernel is much
closer to a real roof than §3.3.1 claims — in which case **item G collapses and
the 46-48 t/s branch does not exist.**

### R3 — Settle the dispatch price in the live graph. **Zero code.**

```sh
for N in 0 1 2 4; do DS4_METAL_DISPATCH_BALLAST=$N <2K-ARM>; done   # both ranks
```

**Decides:** four candidates at once. `ds4_gpu_decode_dispatch_ballast`
(`ds4_metal.m:1433-1475`, called at the head of every decode layer,
`ds4.c:22151`) emits N extra one-thread no-op dispatches **per layer**, so
`d(ms/token)/d(43N)` is the in-situ marginal launch cost. **N=2 is exactly the 86
dispatches item C proposes to remove.**

Four incompatible dispatch prices are in simultaneous use across the reviews,
spanning **11.6×**: 1.9 µs (rig in-situ, `tp_decode_investigation.md` §6, which
says in terms *"use 1.9–4.4, not 8.6"*), 3.6–3.74 µs (ballast + a synthetic twin
on M1 Max), 5.74 µs (imported from the HC pre-norm fuse, an atypical
one-threadgroup 80 KB cold-read site), and 22 µs (`tests/bench_qkv_norm`, **now
refuted** — that sweep varied `DS4_N_HEAD_DIM`, which *is* the amax block count,
so it measured the serial loop, not launch latency). **Decision rule: if
d/d(dispatch) < 2.9 µs, drop item C without writing any code.**

### R4 — Retire the power premise. **Zero code, 5 minutes, one node.**

```sh
sudo powermetrics --samplers gpu_power -i 200        # during <2K-ARM>
sudo powermetrics --samplers gpu_power -i 200        # during ./tests/bench_membw 4
```

**Decides:** whether the "30 W against a 120 W envelope, 20% utilised" reading
carries any information. If `bench_membw` at its 760 GB/s arm draws ~50 W and
decode draws ~30 W, power tracks bytes and the headroom reading is dead. If both
read ~30 W, power is static-dominated and the observation was never evidence
either way. **Both outcomes retire it.** Also capture the sustained GPU clock:
nobody has checked whether decode runs at a lower residency-driven P-state than
prefill, and if it does, some fraction of the gap is DVFS and no kernel change
touches it.

### R5 — Confirm the compressor accounting. **Zero code, rule-2 clean.**

```sh
DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_QUAD_FUSE=1 \
DS4_METAL_GPU_STAGE_TIMESTAMPS=1 <2K-ARM>
```

**Decides:** §3.1, i.e. whether the corrected byte model can be quoted. Unfusing
the quad compressor (`ds4.c:22478`) moves the 608 MB of compressor reads out of
the `q_a_kv_proj` span and into `compressor_proj` (`ds4.c:22977`). **Prediction:
`q_a_kv_proj` collapses to ~0.6–0.7 ms and a `compressor_proj` row appears at
~1.3–1.5 ms.** If `q_a_kv_proj` stays near 1.97, the compressor accounting is
wrong and the corrected floor moves back down. One binary, one run, no
cross-epoch subtraction.

*(The pre-M5 name is the right one for M2 Ultra; `DS4_METAL_DISABLE_M5_...` is
the M5 twin and will not fire on this rig.)*

### R6 — Split the 3.63 ms. **Needs ~10 lines written first.**

Rewire the four A0 boundaries — `DS4_METAL_PROFILE_DECODE_FA_STAGE("gather")` at
`ds4_metal.m:29420`, `("packed")` at `:29481`, `("fa_core")` at `:29501`,
`("reduce")` at `:29529`, macro defined at `:29191` — to call
`ds4_gpu_ts_label()` plus an encoder close instead of
`ds4_gpu_flash_attn_stage_profile_boundary()`. Then:

```sh
DS4_METAL_GPU_ENCODER_TIMESTAMPS=1 DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1 <2K-ARM>
```

**Decides:** whether item G — and therefore the 46-48 t/s branch — exists. This is the only
measurement that can tell `fa_core` from `reduce` from dispatch latency inside
the largest unexplained block in the token. Do it after R1 so the two instruments
can be cross-checked on the same run.

*Blocker to respect:* the timestamp path is gated on
`!g_batch_encoder_concurrent` (`ds4_metal.m:1117`), so this and item E cannot be
measured together, ever.

### R7 — The cheapest live item in the campaign. **Zero code, minutes.**

`M3 uc_pingpong`, 4 × 4 KB chained WRs against 1 × 16 KB
(`SCOPE-TP-GATE-OVERLAP.md:319-325, :496`). ≤ 0.34 ms (1.4%), no engine change,
no correctness gate. It has been sitting at the top of a scope document's
ranked list since 2026-08-27 and no lens mentioned it.

### Not requested, and why

- **Any `DS4_METAL_Q8_MV_NSG` sweep.** T4 already swept it on the rig:
  n=3 −1.6%, n=4 −3.0%, n=6 −4.1%, monotonically worse
  (`BENCHMARKS-TP-PP.md:920-924`). Three separate lenses proposed a point on that
  curve. Setting the env var at all also disables `parallel_full_ffn`
  (`ds4.c:22197`), which is a confound.
- **Any `DS4_TP_ABLATE=hcpre` arm.** Already run in the same 2026-08-27 battery:
  −0.76 ms, `BENCHMARKS-TP-PP.md:409-413`.
- **Any `DS4_TP_LOGITS_PROBE_DIV` sweep.** Already run;
  `tp_decode_investigation.md:434` records the two-term fit
  (145 µs fixed + bytes/1.5 GB/s = 318 µs total, 1.3% of token) in the
  *"Negative results — do not retry these"* table, with both the RDMA and the
  top-k routes rejected.
- **Any speculation / multi-row arm.** `speed-bench/tp_mtp_hunt.md` has the
  end-to-end answer on this rig: 19.9–21.4 t/s against a 41.1 baseline, 8.9–11.8%
  acceptance.
- **Anything measured on the M1 Max as a decision input.** Four of six lenses
  made their decisive measurement there. The one same-kernel same-shape
  cross-machine comparison on record is **4.13×** (`attn_q_b`, 119 GB/s dev box
  vs 492 GB/s rig, `tp_decode_investigation.md` §4-5). Dev-box arms are for
  *mechanism*, never for *size*.

## 6. WHAT WAS RULED OUT

### 6.1 Killed in this round

| item | verdict | reason |
|---|---|---|
| **Row manufacturing / multi-row decode** | **dead, −10 ms** | Already run end-to-end on this rig. `speed-bench/tp_mtp_hunt.md`: forced d5 = **21.29–21.35 t/s** (189/1596 = 11.84%); support + verifier head split = 20.18; full-fat MXFP4 = 19.91 (8.89%). Line 151: *"the observed average acceptance was far below break-even."* The cost curve in `tests/bench_decode_rows.c:66-71` sweeps five **dense** shapes and contains **no routed MoE** — 4 rows × 6-of-256 can touch up to 24 distinct experts per layer, so MoE bytes go up ~4×, not 1.2×. §13 records that fixing the forfeited TP attention split, the unsplit verify head and the per-row malloc still lands at 30–36 t/s, i.e. still below baseline. |
| **Floor-cost verify block + n-gram at depth 2** | **dead** | Its cost model `V(N) = 42.5 + 13.13N` is fitted through two numbers from **different arms with different policies** (`tp_mtp_hunt.md:149` is a mean over 114 verifiers; `:153` is one d2 verifier, n=1) — methodology rule 2, verbatim. Redone at its own best case (h=0.5, p=0.6): 20.97 ms/token = **47.7 t/s**, i.e. it misses the target arithmetically even before the assumptions are challenged. At the only measured proposal frequency on the benchmark prompt (490/502 cycles with no draft) it saves 1.5 ms; at the currently measured verifier it **loses 4 ms**. |
| **O1 — `nsg` 2→8 on the q_a/kv pair matvec** | **dead, three ways** | (a) Wrong kernel: 41 of 43 layers take `ds4_gpu_qkv_pair_quad_compressor_store_tensor`, whose pipeline is built with a **literal 8** (`ds4_metal.m:21742`) and whose Q8 half is `constexpr short NSG = 4` — it is already at the requested width and `DS4_METAL_Q8_MV_NSG` cannot reach it. (b) Wrong byte model: the stage runs at **455 GB/s**, at the ceiling (§3.1). (c) Already swept on the rig: T4, monotonically worse to −4.1%. |
| **O2 — router `nr0` 2→1** | **dead, and dangerous** | `kernel_mul_mv_t_t_4_disp` (`metal/dense.metal:873-887`) dispatches on `args.nr0` with `case 2:` (`:883`) and `case 4:` and **no default**. `nr0=1` falls through, the router logits buffer is never written, and the stage collapses — presenting as a **large speedup** while a stale router unbalances the expert shard. This is `da63283` move for move. The sizing was also already refuted: `SCOPE-ATTNOUT-ROUTER-SHARED.md` §3.2-3.3 puts the router F16 matvec at 0.22–0.34 ms of the 1.108 ms stage. |
| **O3 — widen the HC producer's phase A/B** | **dead** | The falsifier has already been run, on this build, at 2k: `DS4_TP_ABLATE=hcpre` removes **0.76 ms** (`BENCHMARKS-TP-PP.md:409-413`). Applying the scout's own decision rule retires it. The one prior widening of this kernel (pr-778 / `d81a28f`, 6→10 threadgroups) returned **+0.12%, bit-identical**, because TG0 alone carries the entire 4096-wide collapse + RMS epilogue and spreading does not touch it (`SCOPE-HC-STAGES.md:39-43, :538-546`). |
| **HC mix split-K** | **dead** | Same epilogue argument. Split-K shrinks TG0's matvec from 131 KB to 16 KB and leaves the single-threadgroup epilogue on the critical path untouched — pr-778's failure restated. Ceiling from the only end-to-end number (0.65 ms net for both mixes) at a 2.2× kernel speedup is 0.35 ms, before the reduce fold. The harness that produced the 2.5× is also **not the shipped kernel** (contiguous 4-rows-per-threadgroup vs an NB=32/NF=16 blocked stride with two 8-simdgroup clusters) and its byte model omits the `x` traffic entirely. |
| **`mul_mv_ext` at `r1ptg=1` for Q8_0** | **dead** | The kernel does not exist. `metal/dense.metal:1878-1881` instantiates only `_r1_2/_3/_4/_5`; `ds4_gpu_mv_ext_name` has no `r1ptg==1` case for q8 or f16; `ds4_gpu_mv_ext_r1ptg(1)` returns 0. The proposed one-line gate change yields `fn_name == NULL` and a hard failure. The evidence was also a dev-box pathology: the rig achieves **492 GB/s** on the target shape against the dev box's 119. |
| **`nr0=4` for the Q8_0 matvec** | **dead, −0.15 ms** | Killed by its own falsifier on its own harness: at the shipped TP2 `nsg=2`, `nr0=4` is **5–6% slower**, reproducibly, across four runs. The scout compared `ds4_2_2` against `ds4_4_4` and attributed a combined `nsg`+`nr0` change to `nr0` alone. `kernel_mul_mv_q8_0_f32_r4` also already shipped once and was deleted in `0ad494e`, and `a12e73d` already measured `nr0=4` on the rig as a regression for MXFP4 gate/up. |
| **Per-shape `nsg` table** | **dead, 0.07 ms** | 42% of the claimed addressable bytes (`attn_output_a`, 17.83 MB/layer) run on `kernel_dsv4_attn_out_low_q8_0_f32` with a **hardcoded literal 4** and no env override, i.e. already at the target width. The realisable remainder is ~0.066 ms = **0.27% of the token**, below the campaign's own 1% triage floor. |
| **Q4_K shared expert and output head** | **dead** | The shared-expert half prices the wrong gate: the Q8_0 fusions at `ds4.c:22162` and `:24052` are **already dead under TP2** (`g->tp_world < 2`); the gate that fires is `ds4.c:25089`, where Q4_K falls into the three-dispatch `else if (ok)` branch at `:25101-25120` — **+86 dispatches/token**, unbudgeted. The output-head half prices a stage that **has no decode marker** and whose 0.52 ms was back-derived from an assumed rate and then used to declare the stage "at the roof" — circular. 4-bit `lm_head` is also the single most quality-damaging quantisation available. *Residue worth keeping separately:* `shared_down` already routes through a type-generic single-dispatch k-slice (`ds4.c:25279-25288`) and is the slowest "roofline" stage at 391 GB/s — a genuine drop-in worth maybe 0.10–0.16 ms. |
| **Fused Q4_K TP attention-output kernel** | **dead, premise false** | `ds4_gpu_attention_output_q8_tp_tensor` is **not one fused dispatch** — it encodes `kernel_dsv4_attn_out_low_q8_0_f32` then tail-calls `ds4_gpu_matmul_q8_0_kslice_tensor`. Two dispatches. The Q4_K fallback is also two. The format switch adds **zero** dispatches and the entire 0.4 ms was fictional. (This is a net positive for item F: the dispatch objection against it evaporates.) |
| **Distributed greedy argmax (8-byte max/id exchange)** | **dead, already measured** | `tp_decode_investigation.md:434`, in the section headed *"Negative results — do not retry these"*: *"Logits over TCP — recv ≈ 145 µs fixed + bytes/1.5 GB/s. 318 µs total, 1.3% of token. Both RDMA and top-k routes rejected."* An 8-byte (max,id) exchange is the k=1 case of the rejected top-k route. Only the 172 µs bytes term is recoverable; the 145 µs fixed term is rank skew already booked as the straggler, so crediting it here **double-counts**. Net after the new GPU top-1 reduce dispatch: ~0.13 ms. Note also `metal_graph_encode_output_head_split_top1` (`ds4.c:25543`) already implements this design for the CUDA path — the parallelism exists. |
| **Un-disqualify the concurrent FFN encoder under TP2** | **dead** | The gate is not `tp_world < 2`: `ds4.c:22171-22193` requires `IQ2_XXS / IQ2_XXS / Q2_K`, so for MXFP4 Flash the region is **dead code at any `tp_world`** — an unvalidated template, not a shipped proof. It also duplicates `SCOPE-TP-GATE-OVERLAP.md:498` item 3 in a strictly worse form (hiding the shared expert under the routed MoE, which runs at 92% of roof, instead of under the fence spin, which is genuinely idle). And undoing `tp_fold_ffn` restores 43 add dispatches that the scout named and never priced. |
| **Zero-perturbation per-encoder `MTLCounterSampleBuffer` profiler** | **dead — it already exists and is cheaper than the proposal** | Cited the wrong function: the decode budget comes from `DS4_METAL_PROFILE_DECODE_STAGE` → `metal_graph_layer_stage_profile_boundary` (`ds4.c:20396`, `:22208`), not from the FlashAttention boundary it describes. That decode profiler already has a GPU-timestamp mode that does **not** CPU-wait (`ds4_gpu_stage_flush`, `ds4_metal.m:10967` — commits without waiting, batches the `GPUStartTime` reads afterwards) *and* the encoder-timestamp instrument of R1 already ships. The proposal's own cost, 11–20 µs per boundary, is **2.6–4.8× more** than the 4.2 µs/boundary it would replace, and 43 encoders × 11–20 µs = 1.9–3.5% of the token fails its own ≤1% falsifier. |

### 6.2 Correctness findings, not speed items

**`DS4_NGRAM_SPEC` is unreachable and, if reached, corrupts output.** Four
defects, all confirmed by reading:
1. Unreachable at all three call sites (`ds4_cli.c:599`, `ds4_server.c:12054`,
   `ds4_bench.c:857`) without an MTP/DSpark support model. **A planned arm with
   `DS4_NGRAM_SPEC=1` on a plain TP run would silently measure baseline.**
2. `s->logits` is never refreshed on **any** exit path of the cycle body
   (`ds4.c:69775-69921`) — the only occurrences are a comment at `:69793` and a
   read at `:69796`; `row_logits` is passed `NULL` at `:69851` and
   `metal_graph_read_spec_logits_row` is never called, unlike the DSpark
   equivalent at `ds4.c:65995-66018`. Every ngram cycle re-emits its last
   committed token.
3. The no-match branch (`:69812-69818`) pushes `drafts[0]` with no eval, leaving
   an unwritten KV row.
4. The match bound at `ds4.c:69759` (`start + k + n < hist_len - k`) truncates:
   a pure period-3 history with the default `k=3` returns **0** proposals. The
   guarding assertion at `tests/test_engine_mgpu_placement.c:717`
   (`n < 2 || out[1] == 5`) passes either way and is not a test.

**Recommendation: delete the path, or add a hard refusal when `DS4_NGRAM_SPEC` is
set on an unreachable configuration.** Fixing it makes reachable a path whose
best measured economics on this rig are a 30–40% regression, behind an env var,
carrying a two-rank wire protocol that must stay in lockstep. The proposed
correctness gate is also circular — the byte-identical greedy diff cannot run
until defect 1 is fixed. Unit tests close it with no rig time: assert KV frontier
`== checkpoint.len` after a no-match step, assert exact proposal length for
period-3/4/6 histories, assert `s->logits` is refreshed on every exit path.

### 6.3 Standing kill list — do not re-propose

Carried forward unchanged: T1 row-gate fastpath; T8 MXFP4 MoE decode
specialisations (also gated on `add_in == NULL && tp_world == 1`, so they can
never fire under TP); R11 gate-count reduction; R12b dispatch ballast as a
*change* (as an *instrument* it is R3 and is wanted); U1 "400 GB/s is the
platform" (reversed by U6); U3 indexer cache F32→F16; U9 hierarchical top-k;
M1 `packed_uchar4` MXFP4 loads; M3 planar MXFP4 repack; C1 Q8_0 k-curve
reshaping; U13 inverse-RoPE fuse on the indexed branch; pr-778 HC producer grid
widening; Sinkhorn parallelisation; `MTLSharedEvent` gate release (~186 µs/gate);
router+shared fusion (hardcodes NSG=4 against TP's nsg=2); wider fence kernel
(21% vs 4%); replicating attention to delete the ATTN gate (+7.1 ms for −0.9);
PP2 (27.67 vs 40.91 t/s at 2k).

**One correction to that list.** **U16's kill reason is void.** "Fusing the
q-LoRA norm — the kernel is pure dispatch latency, so fusing only the q half
leaves the dispatch and saves nothing" rests entirely on the 22 µs-flat reading
from `tests/bench_qkv_norm`, and that reading is wrong: `tests/bench_qkv_norm.c:62`
takes `argv[2]` as `DS4_N_HEAD_DIM` with `n_rot` pinned at 64, so the sweep
varied the **amax block count**, not the work per block. In-kernel bisection
gives 4.03 / 10.25 / 12.96 / 21.36 / 39.74 µs at 0/1/3/7/15 blocks — **81% of
that kernel is real serial work.** U16 stays dead only because item A supersedes
it by removing the work rather than the dispatch. **Do not use the 22 µs figure
to reject anything else** — two live refutations in this round still lean on it
(a "+22 µs added top-1 reduce" and a "43 × 22 µs = +0.95 ms of restored add
dispatches"; at the rig's in-situ 1.9 µs the latter is 0.08 ms and the objection
inverts).

## 7. OPEN QUESTIONS

Ordered by how much each would change the plan.

**Q1 — Where inside `attn_inv_rope` do the 3.63 ms go?** Unresolved by everyone,
and it is 14.9% of the token and the only stage far from every roof it has.
Nobody proposed it, for a reason that is structural rather than intellectual:
A0, the instrument built to answer it, uses a host wall clock across a
command-buffer round trip and reports figures 6× the stage they belong to. *What
it would take:* R6, ~10 lines. Nothing else in this document has that
cost-to-value ratio.

**Q2 — What is `n_keys` at ctx 2048?** The whole §3.3.1 mechanism, and therefore
the size of item G, turns on it. I could not derive it by reading: it is
`n_raw + n_comp` where the compressor ratio varies per layer and the indexer's
512-way top-k may or may not be engaged. *What it would take:* R2, zero code —
the shape fields are already printed.

**Q3 — Is the profiler over-attributing small stages by 2.7×, systematically?**
`DS4_TP_ABLATE=hcpre` deletes the HC mix and recovers **0.76 ms** against
**2.24 ms** of profile attribution. Nobody has explained the discrepancy. If it
is a general property of small stages rather than something specific to the HC
kernels, then **every item in §2 is over-priced by up to 2.7×** — including item
A, whose entire base is a 1.50 ms profiled stage. This is the largest
uncertainty in the budget and it is not in anyone's error bars. *What it would
take:* R1 cross-checked against R3; if the encoder-timestamp sums track the
command-buffer span to 4% (as the instrument's own validation claims) and still
show 2.24 ms of HC, the discrepancy is real and lives in the ablation, not the
profile.

**Q4 — Where is the output head?** There is no `DS4_METAL_PROFILE_DECODE_STAGE`
for it anywhere in `ds4.c` — only `DS4_METAL_PROFILE_OUTPUT_STAGE("output_norm")`
at `ds4.c:25496`. Its 0.52 ms is 281 MB divided by an assumed 540 GB/s, and that
assumption has been used to declare the stage "at the roof". Circular, and it is
~2% of the token. *R1 settles it.*

**Q5 — What is in the ~1.3 ms residual?** About 20 marker sites exist at
`ds4.c:22417-25394` that never appear in any reported table (`attn_norm`,
`kv_path`, `compressor_proj`, `compressor_update`, `compressor_quantize`,
`compressor_commit`, `indexer_compressor_*`, `ffn_norm`, `attn_output`, and the
non-TP `routed_moe` / `shared_*` variants). *R1 settles it.*

**Q6 — What is the dev-box → rig transfer function?** There is exactly **one**
same-kernel same-shape cross-machine data point and it is **4.13×**. Four of six
lenses made their decisive measurement on the M1 Max. Item A's entire size
estimate is a dev-box number. *What it would take:* run `tests/bench_qkv_norm`'s
block-count bisection on one rig node — model-free, minutes — before item A is
sized in any ledger.

**Q7 — Are the compressors, shared gate/up, router and output head genuinely
replicated?** §3.1 was derived from layout assertions and call sites, not from a
measured resident set. If any is in fact split, the corrected floor over-counts
and the campaign acquires a *second* wrong headline. *R5 and Phase-0 arm 0.6
settle it, both free.* **Do not quote §3.1 externally until they have run.**

**Q8 — Can a Metal completion-counter release be made correct across a CPU DMA
read?** Item C requires the last producer threadgroup's system-scope coherent
store to be ordered after every other threadgroup's device-scope release *as
observed by the host over Thunderbolt DMA*. Metal does not specify this. Nobody
could resolve it from documentation, and a throughput measurement cannot detect
its failure. *What it would take:* a standalone Metal + host-DMA torture harness
with widened race windows — which is most of the engineering cost of the item
itself, which is a reason to run R3 first and probably not to do it at all.

**Q9 — Does decode run at a lower GPU P-state than prefill?** Never checked. If
it does, some fraction of the 30 W vs 60 W observation is DVFS and no kernel
change touches it. *R4, 5 minutes.*

**Q10 — Why does the E[max] straggler model over-predict by 2–3×?** The uniform
model gives `E[max(k, 6−k)] = 3.9375` → 1.1–1.47 ms; the direct clean measurement
gives **0.50 ms**. Candidate explanations (skewed expert popularity correlating
with the contiguous 128/128 shard; ranks not being synchronised at layer start
in the way the model assumes; the FFN gate overlapping part of the skew with the
wire) were not distinguished by anyone. I use the measurement, but the model is
what people keep re-deriving, so the discrepancy is worth closing.

**Q11 — Do the 172 encoder closes per token put a hard floor under item E?**
`ds4_gpu_close_batch_encoder` is called per gate (`ds4_metal.m:10643`), and the
token additionally splits across 3 command buffers at pos ≥ 128. That partitions
the graph into ~86 inter-gate regions of ~12 dispatches each, capping sibling
width before any barrier analysis begins. Nobody priced this.

**Q12 — Is TP4 actually available?** The §7 projection (~50–53 t/s) is the only
route to the target that anyone has identified, but Apple RDMA is point-to-point
per cable, so TP4 needs a full mesh — six cables and a per-peer device
configuration. Whether the hardware exists and whether the engine's transport
layer admits it is a topology question this document did not investigate.

---

## 8. Where reviewers disagreed, and which I believe

**8.1 — The dispatch price. Four numbers, spanning 11.6×, all in simultaneous
use.** 1.9 µs (rig in-situ, §6 of the investigation, which says in terms *"use
1.9–4.4, not 8.6"*), 3.6–3.74 µs (ballast in-engine, plus an M1 Max synthetic
twin at 3.597), 5.74 µs (HC pre-norm fuse), 22 µs (`bench_qkv_norm`).
**I believe: 22 µs is dead** — the sweep varied the amax block count, not the
work per block, and the same reviewer who established that also reproduced the
in-kernel slope. **5.74 is an atypical site** — it is a one-threadgroup kernel
reading ~80 KB cold, so its restore cost is launch + cold-start + lost cache
residency. **The live-graph answer is 1.9–3.74 and R3 settles it in one arm.**
The disagreement matters because item C, item E and the entire (refuted) fusion
campaign are priced off it, and two surviving refutations still lean on the dead
22 µs figure in a way that inverts their conclusions.

**8.2 — The concurrent encoder. Four estimates: 0.10, 0.25, 0.30, 0.60.** All
four are the same `MTLDispatchTypeConcurrent` mechanism.
**I believe 0.2–0.4, and I note that the pool's survivor selection was
anti-correlated with evidence quality here:** the three estimates that actually
measured something converge on 0.2–0.4; the one that survived review is the one
whose "28% recoverable" is explicitly *"inference, from reading ds4.c"*, and four
of that same reviewer's own dead-ends argue the opposite. The best-derived
number in the tree is `TP-A0-ROWSPLIT-TEST-PLAN.md:968-1002` at ~0.5 ms —
written the same day, titled *"I mis-sized this, twice over"*, and demoted to
last. No lens cited it.

**8.3 — Is the shared-expert overlap bit-exact?** One reviewer promises
"bit-identical"; `SCOPE-TP-GATE-OVERLAP.md:498` says **"not bit-exact"** and
notes it carries a §3.4 stale-payload risk described as *a measured failure
mode*. **I believe the standing document.** Repartitioning a sum across a
different set of contributors is not bit-exact except by accident, and the
reviewer who claimed otherwise did not engage with the disagreement.

**8.4 — The MoE straggler: 0.50, 1.1, 1.29 or 1.47 ms?** **I believe 0.50** —
`TP-A0-ROWSPLIT-TEST-PLAN.md:760`, a direct clean-vs-broken comparison at
+11.6 µs/layer, and the fourth successive downward revision of the same item
(0.82 → 0.74 → 0.66 → 0.50). The larger figures are all the uniform `E[max]`
model, which has now over-predicted four times running. See Q10. Note also that
the straggler is **not additive with the TP-gate items** — it lives inside the
same measured gate budget (2.64 ms raw / 2.21 net), and the FFN gate's 1.65×
excess over the ATTN gate is exactly where it sits.

**8.5 — The floor: 9.5 ms or 13.2 ms?** **I believe 13.2**, and I re-derived it
independently from `ds4.c:5124-5180` rather than adopting either reviewer's
figure. Two reviewers reached it separately (the premise challenger from the
layout tables, the O1 refuter from the compressor call sites) and their 608 MB
compressor terms agree to the digit. It is corroborated by the artifact
(8.19 GiB derived vs 8.26 GiB implied by file-minus-routed) and by
`q_a_kv_proj` closing at 455 GB/s. **This retires the "2.6× off its floor"
headline.** Caveat: Q7.

**8.6 — How far is FlashAttention from its roof?** `TP-A0-ROWSPLIT-TEST-PLAN.md:577-582`
says 6–28 GB/s and 305 GFLOP/s; my reconciliation gives ~500 GFLOP/s and
~496 GB/s of *cache* traffic against a ~90 MB DRAM footprint. **Same conclusion,
different frame, and the frame matters.** The doc applies a DRAM roofline to a
kernel whose working set is cache-resident, which makes it look 1.5 orders of
magnitude off and points at "fill the grid". The intensity frame makes it look
3–10× off and points at "read the KV once and serve H heads with it". The second
is the one with a mechanism, and the first is the one that has already produced
two failed experiments (`packed32` at −1.35 t/s, pr-778 at +0.12%).

**8.7 — "The flash reduce is hard-wired to 32 every layer".** The brief says
this; the code disagrees for the decode path. Under TP2,
`ds4_gpu_flash_attn_decode_nwg` (`ds4_metal.m:3819-3842`) buckets `nwg` to
4/5/12/16/24/32 by key count. What *is* true, and is a different statement, is
that the **reduce grid is `nrows = n_head` = 32 threadgroups** on 60 cores
(`ds4_metal.m:29526`). Two claims were collapsed into one; only the second
survives, and it is a small kernel.

**8.8 — The residual: 0.96 ms or ~9%?** **I believe 0.96 from the completeness
critic's arithmetic** — the 14 net stages sum to 23.38 of 24.34 — **and I revise
it up to ~1.3 ms** because the brief's net table carries `attn_out_proj` at 3.72,
which is both the pre-split mislabel and a *raw* number in a net table. Scouts
were told there was 2.2 ms of unexamined slack; there is ~1.3.

**8.9 — Fix `DS4_NGRAM_SPEC` or delete it?** One reviewer says fix; the risk
reviewer says fixing it is riskier than leaving it. **I believe delete, or add a
hard refusal.** Fixing it makes reachable a path measured at a 30–40% regression
on this rig, behind an env var, with a two-rank protocol to keep in lockstep —
and defect 2 (stale `s->logits` on every exit path) is a live output-corruption
bug the moment anyone loads a DSpark support GGUF, so it needs *some* action
today either way.

**8.10 — Something everyone agreed on that I want to record.** Every reviewer
who checked concluded that a throughput measurement cannot gate any of the
top-ranked items, because in each case the corruption comes from skipping work
and skipped work is faster. That is not a coincidence and it is not paranoia —
it is the mechanism of `3f17e83` → `da63283`, and it is why §4 opens with a
programme rule rather than an item.

---

## 9. Provenance

Verified against the tree at `HEAD = 6f37a55` on 2026-08-27. Every file:line in
this document was opened. Figures quoted from `BENCHMARKS-TP-PP.md`,
`speed-bench/tp_decode_investigation.md`, `docs/TP-A0-ROWSPLIT-TEST-PLAN.md`,
`docs/SCOPE-TP-GATE-OVERLAP.md`, `docs/SCOPE-HC-STAGES.md`,
`docs/SCOPE-ATTNOUT-ROUTER-SHARED.md` and `speed-bench/tp_mtp_hunt.md` are
attributed inline. The byte model in §3.1 was re-derived from
`weights_validate_layout` rather than adopted from any reviewer.

**Numbers in this document that are *not* measurements and should be treated as
such:** the output head's 0.52 ms (assumption, Q4); `n_keys ≈ 640` at ctx 2048
(inference, Q2); every FlashAttention floor in §3.3.1 (derived from that
inference); item G's 0.0–2.0 ms range (unsized by construction); item A's rig
value (transferred from an M1 Max measurement across a gap whose only calibration
point is 4.13×, Q6); the ~18 ms physical floor in §3.4 (a construction, not a
measurement).

**This document does not claim a win.** It claims that the target is out of
reach by roughly 3.4 ms, that ~3.7 ms of the supposed 15 ms gap never existed,
and that the next thing to spend money on is four environment variables and ten
lines of instrument rewiring — not another arm.
