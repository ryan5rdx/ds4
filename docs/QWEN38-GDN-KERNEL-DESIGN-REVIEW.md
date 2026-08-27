# Adversarial review — `QWEN38-GDN-KERNEL-DESIGN.md`

Reviewed: `docs/QWEN38-GDN-KERNEL-DESIGN.md` (1,324 lines), branch `tp-multi-slot-batching`.
Stance: adversarial. Purpose: input to a go/no-go on a ~62k-line port.
Method: every `file:line` citation in the design was opened; every arithmetic claim in §2, §3,
§4 and §7 was recomputed; §2.2's separability result was re-derived from §1.2/§1.3.

**Ranking key** — `[GNG]` bears on go/no-go · `[DES]` forces a design change ·
`[LOC]` moves the line estimate · `[ACC]` accuracy defect that does not move the conclusion.

---

## Findings, ranked

### 1. `[GNG]` R1's falsifier cannot discriminate its own two hypotheses, and both branches cost the same ~1.5 ms

**Claim.** §3.1 and R1: the ~13 ms flat decode residual is *either* per-encoder-boundary
(172/token) *or* per-layer. "If the boundary slope explains the residual, §3's design is
right and GDN adds zero boundaries. If the *layer* slope explains it, stop and re-plan."
The proposed ~20-line instrument is billed as "half a day of work that determines whether
this port's decode target is reachable."

**Why it's wrong.** Encoder boundaries *are* per-layer. `ds4_engine_tp_gate_schedule()`
(`ds4.c:60432-60436`, verified) sets `per_token = DS4_N_LAYER * DS4_TP_GATES_PER_LAYER`, and
the 172 boundaries are 43 layers × 2 gates × 2 events. Under the boundary hypothesis, Qwen's
48 layers fire 96 gates → **192 boundaries, +20 per token**. At the design's own 75.6 µs/boundary
that is **+1.51 ms/token** — numerically identical to the +1.5 ms R1 assigns to the *per-layer*
branch (13 ms / 43 × 5 = 1.51 ms), because they are the same quantity computed two ways.
The dispatch hypothesis also scales with layers (1021/43 = 23.7 per layer). All three candidate
explanations are per-layer; the only hypothesis under which 48 layers is free is a fixed
per-token cost, and 13 ms of fixed per-token cost is not on the table.

So the instrument, whatever it returns, does not gate the port — it only tells you *which*
name to give a penalty you pay regardless.

**Second error, in the other direction.** `TP-A0-ROWSPLIT-TEST-PLAN.md:668-679` states
explicitly that the residual is "unattributed, not known-to-be-overhead", that an earlier
draft calling it "not compute at all" was **wrong**, and that it contains the router, shared
expert, q_a/kv projections, compressor update, HC post-combine, and per-token
embedding/norm/logits/sampling — "The shared expert and router in particular are real,
irreducible work." The design converts a conditional upper bound ("*if* boundaries carried
the whole non-compute remainder they would be ~75 µs each", `:697-700`) into its working
hypothesis and drops the hedge.

That cuts the other way on the extrapolation: most of DS4's per-layer residual is DS4's own
MoE router + shared expert + compressor. Qwen's 36 GDN layers have none of those. So
`13 ms / 43 × 48` is a category error in *both* directions — the residual is neither pure
overhead nor transferable per-layer.

**What is actually certain.** The +10 gates / +20 encoder boundaries per token is structural
and not conditional on any hypothesis. §4.3 presents this as a pure win — *"The gate schedule
needs no change at all… **Zero lines change in the transport**"* — and §3.1 point 1 states
*"The GDN design in §4 adds **zero** gates per layer."* Both are true per layer and both hide
that the model has 11.6% more layers, so the token pays 11.6% more gate exchanges. §4.2's
"6 KB of slack per work request" is a *payload* win; over Thunderbolt RDMA the gate cost is
latency-dominated, so a smaller payload does not offset more gates. Nothing in the document
nets these against each other.

**Consequence.** Go/no-go framing changes. The correct statement is: *the port starts
~1.0–1.5 ms/token behind DS4-Flash on layer-count alone, under every hypothesis the tree
currently entertains, and no GDN kernel design recovers it.* Whether that is fatal depends
on the whole-model byte budget, not on the instrument.

**What to do instead.** Run the instrument the test plan actually ranks first
(`TP-A0-ROWSPLIT-TEST-PLAN.md:718-721`): the `DS4_METAL_DECODE_STAGE_PROFILE` run at 32k/131k,
~20 minutes, which bounds how much of the 13 ms is real compute. That *is* decision-relevant,
because compute in DS4's router/shared/compressor does not transfer to Qwen and would shrink
the hole. The design cites `:736-740` and skips `:718-721`.

**Confidence: high** on the arithmetic and on the misread of the source; **medium** on the
size of the residual hole, which is exactly what the stage profile would settle.

---

### 2. `[GNG]/[DES]` D1 — the biggest decode kernel is the one kernel that gets no occupancy analysis, and its grid is two orders of magnitude narrower than ds4's existing path for the same shape

**Claim.** §3.2/§3.3: ten stages fuse into **D1**, *"Grid: one threadgroup per 128-channel
slab of the local `conv_dim = 5120`, so **40 threadgroups**… Weight-bandwidth bound: reads
… ≈ 13.0 MB/rank/layer."* This fusion is one of the three dispatches the headline result
rests on.

**Problem A — the grid is understated and inconsistent with the fusion table.** The §3.2
table puts `in_proj_z` [6144←2560] in D1. `z` is 3072 rows local; it is not in `conv_dim`.
D1 must produce 5120 (qkv) + 3072 (z) + 24 + 24 = **8,240 rows**, i.e. ≥ 64 threadgroups,
not 40. The 13.0 MB figure in the same paragraph is computed from 8,240 rows, so the byte
count and the grid count in adjacent sentences are derived from different row counts.

**Problem B — and this is the real one.** ds4's existing decode matvec for the exact shape
D1 replaces, on the quant the model actually ships (`UD-Q4_K_XL` →
`ds4_gpu_matmul_q4_tensor`, `ds4_metal.m:19622`): `nsg = 2`,
`nxpsg = ds4_gpu_mv_ext_nxpsg(2560, 1) = 16` → `nypsg = 2`, `r0ptg = 4`, grid
`out_dim / r0ptg` × 64 threads (`ds4_metal.m:19628-19643`, verified). For `out_dim = 8240`
that is **2,060 threadgroups / 131,840 threads**. (The f16 path is wider still:
`ds4_gpu_make_plain_mv_dispatch(2560, 0)` → `nsg = 8`, `nr0 = 2`, `ds4_metal.m:5297-5319`,
dispatched `out_dim / nr0` × 256 threads at `ds4_metal.m:20698-20701` = **4,120 threadgroups
/ 1,054,720 threads**.)

D1 proposes **64 threadgroups / 8,192 threads** for the same 13 MB read — 32× fewer
threadgroups and 16× fewer threads than the path it replaces.

At 8,192 threads over 60 cores you have ~136 threads/core, roughly 4 simdgroups, against a
core that can host on the order of 24. Decode on this rig is *latency-bound* (the memory
already says so: ~192 GB/s of ~800). The one lever the design itself identifies for a
bandwidth-bound kernel — "the only lever is bandwidth utilisation, and that is set by how
many cores are busy" (§3.5) — is applied to D2 (3.15 MB) and to K3, and **not applied to D1
(13.0 MB), which moves 4× the bytes.**

**Problem C — the cap is structural, not a tuning miss.** The row granularity of D1 is
pinned at 128 by the l2norm fusion ("reduction over the 128 contiguous channels of one head
— inside a 128-aligned channel slab"). You cannot go below 128 rows/threadgroup without a
cross-threadgroup reduction, and you cannot split the `k` axis (2560) across threadgroups
without either a second dispatch or a reduction-order change that §6 forbids. So D1's grid
is capped at 8240/128 = 64 by construction.

A partial rescue exists — make the grid heterogeneous, keeping 128-row slabs only for the
2,048 q/k rows that need the l2norm and using fine row slabs for v/z — but even then the
16 q/k threadgroups form a 3.24 MB tail running on ~27% of the machine, and the dispatch
cannot retire until they do. The document does not propose this and does not price it.

**Consequence.** The fusion D1 buys is worth, by the design's own accounting, 68–158 µs/token
per removed dispatch. A 1.5–2.5× bandwidth shortfall on 13 MB × 36 layers is
**~0.4–0.9 ms/token**. On the design's own numbers the fusion is a **net loss of 3–10×** —
the same argument the design correctly makes against fusing the recurrence into D1 in §3.4,
not applied to D1 itself. The "3 dispatches per GDN layer" headline is achieved by making
the dominant kernel narrow.

**What to do.** Either (a) split D1 into `in_proj` (existing wide mv path, pinned per §6) +
a small `gdn_decode_conv_norm` post-kernel — 4 dispatches/layer, ~+68–158 µs, and D1's
bandwidth problem disappears; or (b) keep the fusion but measure achieved GB/s at 64
threadgroups before committing. The document should contain the same table it wrote for D2
in §3.5, for D1.

**Confidence: high** that the analysis is missing and the grid figure is wrong;
**medium-high** that the fusion is net-negative (the achieved-bandwidth-vs-threadgroups
curve on this rig is not in the tree).

---

### 3. `[GNG]/[LOC]` §5.2's "seven sites" undercounts the polarity trap by ~3–4×, and S1's oracle is much weaker than claimed

**Claim.** §5.2 enumerates **seven** `ratio == 0` sites, says *"convert all seven call sites
in a single mechanical commit"*, budgets **250 lines** (§7.1), and gives S1 the oracle
"existing DS4-Flash decode + prefill regression … **bit-identical logits**".

**All seven cited sites are real and correctly quoted** (`ds4.c:17322`, `:35718`, `:35764`,
`:51992`, `:53237`, `:53277`, `:1135` — each verified). But `grep -c "ratio == 0\|ratio != 0"
ds4.c` returns **54**, and the ones §5.2 missed are in exactly the classes it says matter most:

| missed site | what it guards | Qwen consequence |
|---|---|---|
| `ds4.c:52388` | **CPU** session payload size | GDN state omitted from the CPU-path payload |
| `ds4.c:52468` | per-layer GPU payload size (`ds4_session_save_layer_payload` sibling) | streamed save omits GDN |
| `ds4.c:52649` | GPU payload **write** | silent omission |
| `ds4.c:53064` | GPU payload **read** | silent omission |
| `ds4.c:53632` | CPU payload **write** | silent omission |
| `ds4.c:53717` | CPU payload **read** | silent omission |
| `ds4.c:54094`, `:54254` | CPU restore / payload read | silent omission |
| `ds4.c:53314` | `spec_prefix` **restore** (the 4-slot ring) | spec prefix restore skips GDN |
| `ds4.c:12566` | `kv_cache_finish_prefill_states` (CPU) | prefill finalisation skips GDN |
| `ds4.c:16334`, `:17191`, `:38234`, `:38260`, `:50655` | memory **planning/accounting** | planner under-reports by 112 MiB/slot → admission over-commits (this is R11, arriving through a door the design did not check) |

§5.2 lists the GPU payload *sizer* (`:51992`) but not the GPU payload *writer* or *reader*;
that pairing is the single most likely way to ship a save that sizes correctly and writes
nothing. R3 anticipates "sites that express the same idea differently" — the actual failure
is that a dozen sites use the **identical idiom the design grepped for** and were not
enumerated.

**Second problem: S1's oracle is close to vacuous.** `ds4_expected_layer_compress_ratio()`
(verified) gives DS4-Flash `ratio == 0` on layers **0 and 1 only**, and those layers have no
compressor state at all — so a DS4-Flash regression exercises the `ratio == 0` branches only
in the "nothing to do" direction. A conversion that gets `ds4_layer_has_recurrent_state()`
subtly wrong for a *hypothetical* recurrent layer produces bit-identical DS4-Flash logits.
S1 as specified cannot fail for the reason S1 exists.

**Third problem, upstream of all of it.** `ds4_layer_compress_ratio()` (`ds4.c:1113-1117`)
opens with `if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK4) return 0;`, and
`ds4.c:5624-5632` hard-`exit(1)`s if the GGUF's ratio array does not match
`ds4_expected_layer_compress_ratio()`'s DeepSeek pattern. So "Qwen's GGUF reuses
`compress_ratios` with 0 = GDN and 4 = QSA" cannot be read at all until the family plumbing
and the validator are extended — and until then Qwen's ratio set is `{0}` everywhere, not
`{0, 4}`. This makes R12 *worse*, not better: `ds4_compressor_rewind_align()` would return
**1**, not the 4 the design states.

**Consequence.** The polarity commit is ~600–900 lines, not 250; it spans the CPU cache path
and the memory planner, not just the GPU state path; and it needs a *new* oracle (a
synthetic model with a recurrent-flagged layer, or a GLM-family round-trip) because the
DS4-Flash regression cannot see it.

**Confidence: high.**

---

### 4. `[GNG]/[DES]` §6.3's pinned projection is not realizable as one kernel family across `ntile ∈ {1, 8, 32}`, and the `mul_mm` row of the §6.2 table is an assumption, not a proof

**The narrower claim in §6.2 is correct, and I verified it line by line.** In
`kernel_mul_mv_ext_q4_f32_impl` (`metal/dense.metal:1596-1692`): the k-loop is
`for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg)` with `tx = tiisg % nxpsg`, the
per-row accumulator is `sumf[ir1]` with `dot(lx[ch], y4[ir1][ch*nxpsg])`, and the shuffle
tree is keyed on `nxpsg` alone. `r1ptg` appears only as an unrolled loop bound over
independent rows. **For this kernel family, the token tile provably cannot perturb bits and
`nxpsg` provably can.** §6.2's reduction of §1.5 2(c) is sound *as stated*, and it is the
best piece of engineering reasoning in the document.

**But the design it licenses does not follow.**

**(a) `ntile = 32` is not reachable in this family.** `mul_mv_ext` holds `device const float4
* y4[r1ptg]` and `float sumf[r1ptg]` in registers and is host-gated at `n_tok <= 8`
(`ds4_metal.m:20708`, `:19622`, `:21576` — all verified). At `r1ptg = 32` the arithmetic
intensity is 32 MACs per ~0.62 B of weight = 103 FLOP/B, which at 700 GB/s demands
72 TFLOP/s against a ~21.5 TFLOP/s machine — the kernel is compute-bound and every one of
its FLOPs is a scalar FMA. A kernel that reaches `ntile = 32` efficiently is a tiled GEMM,
which is a different k-blocking and therefore a different reduction tree. §6.3 rule 2 waves
`ntile ∈ {1, 8, 32}` through as "a token-axis parameter" without noticing that crossing
1 → 32 crosses the kernel-family line §6.2 itself says must be pinned.

**(b) The `mul_mm` row of the §6.2 table is unsupported.** The table marks
`mul_mm`'s `NR1` as "token / no". `NK = 32` at `dense.metal:1933` and the loop at `:2002`
are correctly quoted, but the actual reduction happens inside `mm.run(mB, mA, cT)` — a Metal
4 TensorOps cooperative matmul whose internal accumulation order is not specified by the
language and is a function of the operand tile shapes, one of which is `NR1`. "Provably
cannot perturb bits" is not available here; it is an empirical question. (The design's own
§6.4 cases B/C/D would in fact test it — but §6.2 states the conclusion as a theorem and
§6.3 builds on it.) Also missing from the table: `g_quality_mode`, which changes `nr0`
and `smem` for some shapes at `ds4_metal.m:20680`, and the `n_tok >= 192 && n_tok % 32 != 0`
switch at `ds4_metal.m:19180`.

**(c) The prefill cost of the pin is the largest unpriced item in the document.** If the
GDN projections cannot use `mul_mm`, then per 2048-token prefill chunk per rank the GDN
`in_proj` + `out_proj` are 29 M weights × 2048 tokens × 2 = **~119 GFLOP per layer**,
**~4.3 TFLOP over 36 layers** — an order of magnitude more than the 203 GFLOP the design
carefully budgets for K3 in §2.5, and it appears nowhere. Whatever the pinned kernel's
efficiency ratio to `mul_mm` is, it multiplies 4.3 TFLOP, not 0.2 TFLOP. Additionally, a
matvec-shaped kernel at `ntile = 32` re-reads the weights 2048/32 = 64× → 41 GB/chunk/rank
→ ~59 ms at 700 GB/s, also unbudgeted.

**Consequence.** §6.3's four function constants and its 350-line budget describe a kernel
that either (i) is a hand-written GEMM with an explicit fixed k-loop, usable at all `ntile`,
in which case it is a `mul_mm`-class kernel (`kernel_mul_mm` alone is ~200 lines in
`dense.metal` *before* per-quant dequant paths and host plumbing) and the prefill efficiency
question must be measured; or (ii) is an mv-family kernel, in which case prefill regresses
badly. The document does not choose. This is the single item most likely to move the
line estimate and the prefill projection.

**Confidence: high** on (a) and (b); **medium** on the size of (c), which needs a microbench.

---

### 5. `[DES]` K1's stated tiling cannot do both the conv halo and the l2norm, and the fix does not fit in threadgroup memory

**Claim.** §2.3: *"Grid: `(ceil(T/64), conv_dim_local/64)` … a threadgroup owning a 64-token
× 64-channel tile loads a 67-token halo. Halo smem: `67 × 64 × 4 B = 17,152 B`. Under
budget."* Two bullets later: *"`l2norm` is over the 128 contiguous channels of one `q`/`k`
head, so a **128-aligned channel tile** does it with one 128-lane reduction."*

A 64-channel tile holds **half** a head. The l2norm cannot be computed in it. Widening the
tile to 128 channels gives a halo of `67 × 128 × 4 = **34,304 B**`, which **exceeds the
32,768 B `maxThreadgroupMemoryLength`** the design itself establishes in §2.0.

Escapes, all of which cost something the document has not paid:
- 32-token × 128-channel tiles: `35 × 128 × 4 = 17,920 B` ✓ — but doubles the halo overhead
  ratio (3 of every 35 rows are halo instead of 3 of 67) and doubles the threadgroup count.
- Keep 64 channels and split the l2norm sum across two threadgroups — needs a device-scope
  reduction, i.e. **a seventh prefill dispatch**, contradicting §2.8's ledger.
- Store the halo in `half` — changes the conv reduction inputs and breaks the fp32 story.

**Consequence.** §2.3 re-tiles and §2.8's "6 dispatches per GDN layer per prefill chunk"
becomes 6 or 7 depending on the fix. Not fatal — prefill is not dispatch-sensitive, as the
design correctly says — but it is a concrete instance of the design's tile shapes not having
been checked against each other.

**Confidence: high.**

---

### 6. `[DES]` The threadgroup-memory occupancy argument is arithmetically self-contradictory

Two statements, both load-bearing:

- §2.0: *"I hold them under ~25 KiB **so the pipeline is not occupancy-capped by smem**."*
- §2.5: *"16,896 < 32,768 ✓ — comfortably, and **small enough that two K3 threadgroups
  co-reside per core**, which matters because K3's grid (96) does not evenly cover 60 cores."*

**`2 × 16,896 = 33,792 > 32,768`.** If the per-core threadgroup-memory pool equals the
per-threadgroup cap the design queried (32,768 B on Apple8), K3 co-resides **one**
threadgroup per core, by 1,024 bytes. The stated justification for K3's smem budget fails on
its own numbers.

If instead the per-core pool is larger (AGX reverse-engineering suggests 64 KiB of tile
memory per core; Apple does not publish it, and `maxThreadgroupMemoryLength` is a
per-threadgroup limit, not a pool size), then `2 × 25,344 = 50,688 ≤ 65,536` and K2 also
co-resides two — which makes §2.0's "under ~25 KiB so we are not occupancy-capped" a
statement about nothing. **The two claims cannot both be doing work.** Under the
conservative reading, 25.3 KiB is precisely the one-threadgroup-per-core case §2.0 says it
is avoiding.

The fix is cheap (drop K3's double-buffer slab → 8,704 B, or use `[64,16]` slabs), but the
document should say which pool size it assumes and how it will verify it — occupancy is not
readable from `maxThreadgroupMemoryLength`; it needs a pipeline-state
`maxTotalThreadsPerThreadgroup` probe or an occupancy microbench.

Separately, §2.4's table is a **static sum, not a peak**. Step 6 (`A_intra = (Q Kᵀ) ⊙ D`)
needs two `[64,32]` operand tiles simultaneously; with `sA` (16,384 B) still declared that
is 33,536 B unless the kernel aliases the dead `A_ut` storage or `simdgroup_load`s one
operand straight from device. Feasible, but not stated, and the budget table implies
otherwise. (Step 2 *is* fine with one slab, because `K_β Kᵀ = diag(β) K Kᵀ` — the design does
not say so, but the algebra rescues it.)

**Confidence: high** on the arithmetic; **medium** on which pool size applies.

---

### 7. `[ACC]/[DES]` §2.5's re-read ledger is 8× low and contradicts §2.4 on the same page

§2.4: *"`U [64,128] fp32` 32 KiB, `W [64,128] fp32` 32 KiB, `A_intra [64,64] fp32` 16 KiB =
80 KiB. At T=2048, TP2: `768 × 80 KiB = **60 MiB** of scratch per layer."* Correct.

§2.5, one page later: *"Per layer at T=2048/TP2: `4 × (W 3 MiB + Qe 3 MiB + K 3 MiB +
A 1.5 MiB) = 42 MiB` re-read vs 10.5 MiB at `G=1`. 31.5 MiB extra per layer × 36 = 1.1 GiB
… **~1.6 ms at 700 GB/s**."*

`W` is `[24 heads][2048 tokens][128] fp32` = **24 MiB per layer**, not 3 MiB — as §2.4's own
`768 × 32 KiB` implies. Same for `Qe` and `K`; `A_intra` is 12 MiB. So:

| | design | recomputed |
|---|---|---|
| re-read at `G=1` | 10.5 MiB/layer | **84 MiB/layer** |
| re-read at `G=4` | 42 MiB/layer | **336 MiB/layer** |
| extra over 36 layers | 1.1 GiB | **9.07 GiB** |
| time at 700 GB/s | ~1.6 ms | **~13 ms** per 2048-token chunk per rank |

The **conclusion survives** (13 ms against a ~1–2 s prefill chunk is still ~1%, so `G=4`
still buys its occupancy), but the ledger the conclusion is drawn from is wrong by 8×, and
the same 8× would flip the sign if anyone reuses these numbers for `G=8` at a larger `T`.
Flagging because §2.5 is presented as a quantified trade and the reader is invited to trust
the arithmetic.

**Confidence: high.**

---

### 8. `[DES]` The `G=4` occupancy argument uses core-count as a proxy where it does not apply (K3), and the wave arithmetic favours a smaller `G`

**K3 (prefill, §2.5).** 96 threadgroups on 60 cores is 1.6 waves; the machine executes
`ceil` → 2 waves → 80% utilisation. `G=2` gives 48 threadgroups, each with 2× the work, so
1 wave × 2 units = **the same wall time** with **half** the redundant `W`/`Qe`/`K`/`A`
traffic. Under a whole-wave model `G=2` weakly dominates `G=4` for K3, and the document
never considers it. (`G=2` is in fact ruled out — 128 dk × 64 dv per threadgroup is 64
registers/lane for `S` alone, on top of an already-tight ~90 — but that is a *register*
argument, not the bandwidth argument §2.5 makes. The right answer is reached for the wrong
reason, which means the reasoning will not generalise when `T`, `G` or the head count moves.)

**D2 (decode, §3.5).** The table's "cores busy" column is a proxy for thread-level
parallelism, which is the thing that actually sets achieved bandwidth on a latency-bound
kernel. That is defensible for D2, and the ±40% caveat is honest. But the *same* proxy is
never applied to D1 (finding 2), which is the kernel it would have condemned.

**Confidence: medium-high.**

---

### 9. `[DES]` D2's state layout is described in a type that cannot express D2's operations

§3.3: *"simdgroup `sg` within it holds `S[:, 8sg:8sg+8]` = 16 `float8x8` tiles = 32
regs/lane"*, then a body consisting of `S *= exp(g)`, `m = kᵀS`, `S += k ⊗ d`, `o = qᵀS`.

None of the last three is a `simdgroup_matrix` operation. MSL gives
`simdgroup_matrix` only `simdgroup_load/store`, `simdgroup_multiply[_accumulate]`,
`make_filled_simdgroup_matrix`, and `thread_elements()` — and `thread_elements()` hands back
a lane's two elements **without a documented (row, col) mapping**, which is exactly what a
rank-1 update `S[dk,dv] += k[dk]·d[dv]` needs. Expressing `k ⊗ d` as an MMA wastes 7/8 of
each tile.

The correct formulation is plain per-thread registers with an explicit `(dk,dv) → (lane,reg)`
map, `m[dv]` closed by an intra-simdgroup `simd_sum` over the 32 dk-lanes. That is cheap and
fine — 16 simd reductions per token — but it does mean §3.3's *"no cross-simdgroup reduction
occurs anywhere in the chunk body"* should read "no cross-**simdgroup**", not "no cross-lane":
D2 has 16 cross-lane reductions per token whose shuffle order must be pinned for
run-to-run determinism (S3's own oracle).

K3's use of `simdgroup_float8x8` is by contrast legitimate — `V' += W·S`, `O += A·Ṽ`,
`S += Kdᵀ·Ṽ` are all genuine MMAs, and `S *= exp(gc[63])` is an index-free elementwise scale
that `thread_elements()` handles. Note however that Apple8 has no matrix hardware;
`simdgroup_matrix` lowers to shuffles + FMAs on the same ALUs, so R6's fallback ("half
operands + fp32 accumulate") buys less than R6 assumes — the win is fp16 ALU rate, not a
tensor-unit path.

**Confidence: high** on the MSL surface; **medium** on how much it costs to fix (probably
little).

---

### 10. `[DES]` §5.3's checkpoint spacing collides head-on with the branch this work lands on

§5.3 recommends `CHKPT_SPACING = 4096` with `K = 2`, and makes
`ds4_session_rewind_align()` return **4096** for a GDN model.

`ds4_session_rewind_align()` is not "how far back you may go"; it is the granularity every
rewind **snaps down to** — `ds4.h:489-492`: *"A rewind can therefore land up to this many
tokens below the requested position"* (`ds4.c:70899`, `ds4.c:1136-1145` — all verified).
Today `ds4_compressor_rewind_align()` returns `lcm(4, 128) = **128**` for DS4-Flash
(`ds4_expected_layer_compress_ratio()` gives ratios `{0, 4, 128}`). Going to 4096 is a
**32× coarsening**: a 10-token rewind would re-prefill up to 4,096 tokens, ~4 s at 1000 t/s.

The three most recent commits on this branch (`ed63073`, `b5c5a24`, `f8b73c6`) are all about
making chat/tool-binding rewinds *land precisely* and *reuse the live prefix*. §5.3's own
option table correctly rejects "reset `S = 0` and force a full re-prefill" because it
"destroys the live-prefix reuse the branch exists for" — and then adopts a spacing that
destroys it for any rewind shorter than 4,096 tokens, which is most of them. The document
flags this as a *"Guess, flagged"* and says measure the distribution first; that is the right
instinct, but the finding should be ranked as a design blocker for the branch, not a tuning
note.

Cheaper alternatives the document does not consider: spacing tied to
`DS4_PREFILL_CHUNK_MIN = 512` with `K = 8` (same 448 MiB, 8× finer landing), or a two-level
ring (coarse 4096 + fine 512 near the frontier).

Related, and separately wrong: **§5.6 and R11 omit the checkpoint ring from the multi-slot
total.** §5.6 says "~110 MiB/rank/slot including the shadow… Four slots = **440 MiB/rank**."
Adding §5.3's `2 × 56.1 MiB` ring per slot gives **~222 MiB/slot → ~890 MiB/rank at 4 slots**.
R11's falsifier ("arithmetic against the branch's actual slot budget") would be run against
the wrong number.

**Confidence: high** on the arithmetic and on the align semantics; **medium** on how bad the
rewind regression is in practice (needs the distribution measurement the design asks for).

---

### 11. `[DES]` Shadow-and-swap needs a first-step/subsequent-step asymmetry the design elides

§5.5's proposal is **correct and clearly better than a 56 MiB memcpy per verify block** —
this is the right call and it composes with `spec_frontier_snapshot`/`restore`
(`ds4.c:53223`/`:53260`, verified) as described. Two gaps:

1. *"The verify pass runs D2 with `S_out = shadow`, `S_in = live`"* holds only for **step 1**
   of a verify block. Step 2 must read the shadow, not the live state. So D2 needs
   `(S_in, S_out)` as separate arguments with `S_in = live, S_out = shadow` on the first step
   and `S_in = S_out = shadow` thereafter. Solvable with a runtime argument (not a function
   constant, so no determinism exposure), but it means the verify encode loop is step-aware,
   which matters if blocks are encoded once and replayed.
2. §5.2 does not list `ds4.c:53314` (`spec_prefix` **restore**, gated by
   `spec_prefix_valid_mask`) among the polarity sites, and §5.5 addresses only the
   `spec_prefix1_*` **allocation** at `:17344-17356`. Gating prefix1 off for GDN (the design's
   recommendation, which I agree with) has to be done at the restore site too or the mask
   will point at buffers that were never allocated.

**Confidence: high.**

---

### 12. `[LOC]` §7.1's "honest all-in number" has no weight-loading, GGUF, or TP-split rows at all

The table covers `metal/gdn.metal`, `ds4_metal.m`, `ds4_gpu_args.h`, `ds4.c` state/encode/
lifecycle, `ds4.h`, and tests. It contains **nothing** for:

- GGUF tensor-name mapping and shape/quant validation for 9 new per-layer tensors × 36 layers
  (`in_proj_qkv/z/b/a`, `conv1d`, `dt_bias`, `A_log`, `norm`, `out_proj`);
- the config/metadata validators — and note `ds4.c:5624-5631` currently `exit(1)`s on any
  compress-ratio array that does not match the DeepSeek pattern (finding 3);
- the TP weight split, including the 3-segment `in_proj_qkv` descriptor or the ~40-line
  load-time relayout §4.1 itself recommends (R9);
- the family/variant plumbing that `ds4_layer_compress_ratio()` requires before §5.2's
  premise is even readable.

Combined with finding 3 (polarity commit 250 → 600–900) and finding 4 (the pinned projection
is a `mul_mm`-class kernel, not 350 lines), **~6,150 is optimistic by roughly 30–60%**;
8,000–10,000 is the defensible band for the scope §7.1 claims to cover. Minor: the
`ds4_gpu_args.h` row is misplaced — that file is 84 lines and the Metal arg structs actually
live in the `.metal` sources (e.g. `ds4_metal_args_mul_mv_ext` at `metal/dense.metal:52`)
mirrored into `ds4_gpu.h`.

**Confidence: medium-high** on the omissions; **medium** on the multiplier.

---

### 12b. `[DES]` The register budget is given for three kernels and omitted for the four that need it most, and its failure model is binary

The design estimates registers/lane for **K2 (~45)**, **K3 (~90)** and **D2 (~42)**. It gives
**no estimate at all** for **D1**, **D3**, **K1** or **K4** — and D1 is the ten-stage fusion
the whole §3 result rests on. A D1 threadgroup holding a 128-row output accumulator plus a
dequant staging tile plus 4 conv taps per channel plus the l2norm partials is not obviously
under any ceiling; §3.3 simply does not ask.

Second, the failure model is wrong in kind. §2.5 and R5 treat 128 regs/lane as a cliff — "if
the compiler spills, drop to `G = 8`". On Apple GPUs occupancy against register usage is a
**staircase, not a cliff**: a kernel using ~90 registers/lane loses concurrent simdgroups per
core long before it spills, which on a latency-bound machine is the entire problem. So K3 at
~90 regs/lane is already the interesting case even in the *good* outcome, and R5's falsifier
("read the register report; if it spills…") will report "no spill" and conclude wrongly.
The right falsifier is an occupancy probe (`maxTotalThreadsPerThreadgroup` on the compiled
pipeline state, plus a wave-count microbench), not a spill check.

Third, the "128 regs/lane at full occupancy on Apple8" figure is not in Apple's published
feature tables; it comes from AGX reverse-engineering. The document states it as verified
hardware fact alongside `maxThreadgroupMemoryLength = 32768`, which *is* published. Those two
should not carry the same confidence.

Minor, related: shadow-and-swap (§5.5) is "a pointer swap (free)" only while the decode graph
is re-encoded each token. `TP-A0-ROWSPLIT-TEST-PLAN.md` lists ICB decode replay (T11) as a
live option; under a replayed indirect command buffer a buffer-pointer swap is not free and
the shadow scheme would need double-encoding or an offset argument. Worth one line in §5.5.

---

### 13. `[ACC]` §2.6's "57% more arithmetic" is the reciprocal of its own numbers

*"`128×128 × 128×128` matmul per combine, i.e. **2.1 MFLOP** per combine against 7.3 MFLOP
for the whole serial chunk body. A Blelloch scan does `2n` combines … so a tree scan costs
~**57% more** arithmetic."* `2n × 2.1 / (n × 7.34) = 0.57` — that is 57% **of**, i.e. the
tree scan would be *cheaper*, the opposite of the stated conclusion. The slip is that
`128³ = 2.1 M` is **MACs**, so a combine is **4.19 MFLOP**, giving `2 × 4.19 / 7.34 = 1.14`
→ **14% more**, not 57%.

The conclusion (serial wins) still holds, and the *real* reason the design gives — that the
`dv` axis already fills the machine, so there is no parallelism to buy — is the sound one.
But a 14% margin is not the comfortable margin the section presents.

---

### 14. `[ACC]` "3 dispatches per GDN layer" excludes the gate machinery

§3.3/§3.6 count D1+D2+D3 and compare 108/token against DS4's 1021/token and 23.7/layer.
That is not like-for-like: DS4's 23.7/layer *includes* the ATTN and FFN gate flag-set,
fence-wait and combine dispatches (`TP-A0-ROWSPLIT-TEST-PLAN.md` T7: "fuse gate flag-set with
fence-wait, **86 dispatches + 86 encoder boundaries**"), plus the FFN/MoE chain. The GDN
block's true per-layer cost is 3 (kernels) + ~3 (ATTN gate: encode, fence, `ds4_gpu_add_tensor`
at `ds4.c:23842-23847`) + the FFN chain. The headline should be labelled "3 dispatches for
the GDN block", and the 108-vs-1021 comparison dropped.

Positively: I audited for smuggled dispatches and **found none** — no blits (shadow-swap is a
pointer swap), no zero-fills in steady state (`metal_tensor_fill_f32` is session-reset only),
no format conversion (fp32 state throughout), no argument buffers (ds4 uses `setBytes`, e.g.
`ds4_metal.m:20686`), and `repeat_interleave` is correctly not materialised.

---

### 15. `[ACC]` Smaller items

- **§4.5 prefill wire total.** "37% smaller per gate" is right (2560 vs 4096), but the
  *total* is 96 × 20.97 MB = 2.01 GB against DS4's 86 × 33.55 MB = 2.89 GB — **30%** smaller,
  not 37%, because there are 11.6% more gates. Same netting error as finding 1.
- **§3.3 D3** says "the threadgroup reads the 6144-element input anyway" — under TP2 it is
  3072 (the section is otherwise TP2-local). The "~50 ns" for the redundant RMS pass is
  per-threadgroup hand-waving: with ~320 `out_proj` threadgroups it is 320 × 12 KiB = 3.8 MB
  of *requests* per layer. Almost certainly SLC-resident so the DRAM cost is ~0, but the
  latency of a second pass over the vector in every threadgroup is not 50 ns and is not
  bounded in the document.
- **§2.4 step 1** "inclusive scan, 64 elements, **1 simd**" — a simdgroup is 32 lanes on
  Apple8; a 64-element scan is two simd scans plus a carry. Trivial, but the scan algorithm
  must be fixed (Hillis-Steele ≠ sequential in fp32) for split-invariance, and the document
  pins the UT block size for exactly this reason without pinning the scan.
- **§5.3(a)** states `ds4_compressor_rewind_align()` "returns **4**" for Qwen. Given
  `ds4_layer_compress_ratio()`'s family guard it returns **1**. Strengthens R12.
- **§6.4** has no case for (i) a rewind before the split (R12 notices this), (ii) **TP2**
  split-invariance (S9 tests per-head bit-identity, not split-invariance under TP), or
  (iii) **multi-slot** — which port plan §7a calls out as *"a new axis, and it lands on the
  branch we are on."* Omitting the multi-slot case from the oracle that exists to guard
  exactly that hazard is a gap.

---

## Checked and found sound

Spend no more review effort here.

- **§2.2 / §3 column separability — the cleverest claim, and it is correct.** Re-derived from
  §1.3: `V' = W·S[:,c]`, `U[:,c] = A_ut(β⊙V)[:,c]`, `Ṽ[:,c] = U[:,c] − V'[:,c]`,
  `O_inter[:,c] = (Q⊙e^g)S[:,c]`, `A_intra Ṽ[:,c]`, and
  `S[:,c] ← S[:,c]e^{g_{C-1}} + (K⊙e^{g_{C-1}-g})ᵀ Ṽ[:,c]`. Every state-touching term is
  indexed by `dv` on both sides; `A_intra = (QKᵀ)⊙D` carries no `dv` index and no state.
  The decode recurrence in §1.2 is separable for the same reason (`m[dv]` reads only column
  `dv`, `d[dv]` feeds only column `dv`). **Bit-identity to `G=1` holds** — each column's
  reduction sequence is untouched by the partition. `W`, `Q`, `K` are also `dv`-free and
  therefore shared, which the design correctly accounts for as the re-read cost (its
  magnitude is finding 7, not its existence).
- **§1.2/§1.3 algorithm transcription**, including the post-update read of `o_t` (the
  diagonal of `D` is 1, so the chunked form agrees), the `Q`-carries-`1/√128` / `K`-does-not
  split, the absence of a separate `masked_fill`, and the observation that every exponential
  is ≤ 1 so there is no overflow path and no max-subtraction trick.
- **§0 shape audit.** 48 v-heads / 16 k-heads / 128 / 128 / conv 4 → 3 MiB/layer, 108 MiB,
  and the conv state at 120 KiB/layer that the brief omitted. The three independent
  confirmations of the 48-head reading are sound. The two RMSNorm conventions and
  `conv1d bias=False` are exactly the kind of trap worth surfacing.
- **§6.2's narrower determinism condition**, verified line by line against
  `metal/dense.metal:1596-1692`. The theorem is right; only its application (finding 4) is not.
- **§4.1 head split.** 48/2 = 24 v-heads, 16/2 = 8 k-heads, 3:1 intact, no k-head shared.
  The TP4/TP8/TP16 extension is right. The 3-segment `in_proj_qkv` hazard (R9) is real and
  the load-time relayout is the right mitigation.
- **§4.3 gate-schedule expressibility.** Verified against `ds4.c:60433-60436` and
  `tp_gate_slot()` (`ds4_tp.c:941-946`): the affine `(0, 1, 96)` triple works, and the
  argument that a non-arithmetic firing set becomes a silent OOB write at
  `tp_rdma_post_gate_recv()` (`ds4_tp.c:975`) is correct. (What the section gets wrong is
  the *cost*, not the mechanism — finding 1.)
- **§4.2 payload.** `vec_bytes = n_embd * 4` confirmed at `ds4_tp.h:126`; 10,240 B is under
  the 16 KB single-WR cap that `tp_rdma_post_gate_recv()`'s comment describes DS4-Flash as
  sitting exactly on.
- **§4.4 / §4.6.** The `out_proj` k-split divergence is a pre-existing accepted property, and
  the observation that the two ranks hold **disjoint, non-redundant** state — so
  `ds4_tp_mark_failed()` must imply session invalidation and both ranks must save — is a real
  finding the port plan does not make.
- **§3.4's rejection of folding the recurrence into D1** (3×/12× redundant q/k projection,
  ~0.97 ms vs 68–158 µs) — correct, quantified, and the right call. Likewise the rejection of
  a persistent-threadgroup D2+D3 on co-residency grounds: Metal genuinely gives no
  device-scope barrier and no co-residency guarantee.
- **No world-1 assumptions found.** §2.3/§2.4/§2.5/§3.3/§5.1 all use `_local` head and channel
  counts consistently, and the grid arithmetic (`5120/64 = 80`, `24 × 32 = 768`,
  `2048 × 24/8 = 6144`) checks out. The recurrent and conv states correctly never cross the
  wire, and **the GDN path introduces no new synchronisation point** — that claim survives.
- **§5.5's shadow-and-swap** beats snapshot/restore by the margin claimed (modulo finding 11).
- **Arithmetic that I recomputed and found correct** (so nobody re-does it): all three
  threadgroup-memory *sums* — K2 `16,384 + 8,192 + 768 = 25,344`, K3
  `8,192 + 8,192 + 512 = 16,896`, D2 `512 + 512 + 128 = 1,152` (it is the *interpretation* of
  these numbers that is wrong, finding 6, not the addition); §0's memory ladder
  (`128·128·4 = 64 KiB`, `×48 = 3 MiB`, `×36 = 108 MiB`; conv `10240·3·4 = 120 KiB`,
  `×36 = 4.22 MiB`); §5.1's TP2 state (`24·4·128·32·4 = 1.5 MiB`, `×36 = 54 MiB`; conv
  `5120·3·4 = 60 KiB`, `×36 = 2.11 MiB`; total 56.1 MiB/rank); §2.5's FLOP count
  (`3·64·128·128 + 64·64·128 = 3.67 M` MACs → 406 GFLOP/2048-token chunk → 198 MFLOP/token);
  §3.3/§3.6's byte ledger (`8240·2560·0.6175 = 13.03 MB`, `out_proj 4.86 MB`, state
  `3.15 MB`, layer total `21.2 MB`, `×36 = 765 MB`, `/532 GB/s = 1.44 ms`, state = 14.8% of
  bytes); §3.6's dispatch savings (`540 × 1.9 µs = 1.03 ms`); K2's register count
  (`64 output tiles / 4 simdgroups × 2 floats/lane = 32 regs`); K3's tile decomposition
  (`S` 16 tiles, `Ṽ`/`O`/`V'` 8 tiles each); §4.5's per-gate prefill payload
  (`2048·2560·4 = 20.97 MB`). Finding 7 is the one place the arithmetic is wrong; finding 13
  is the one place a ratio is inverted.
- **Citation quality is high.** I opened every `file:line` in the document — `ds4.c:1135`,
  `:2577`, `:12266`, `:12293`, `:17171`, `:17322`, `:17344`, `:17358`, `:22966`, `:23838`,
  `:35718`, `:35764`, `:51992`, `:53223`, `:53237`, `:53260`, `:53277`, `:70812`, `:70899`,
  `:60420`; `ds4_metal.m:3447`, `:5372`, `:5378`, `:10412`, `:10477`, `:18278`, `:19134`,
  `:19622`, `:20467`, `:20631`, `:20676`, `:20708`, `:21576`, `:28943`, `:31044`, `:36183`,
  `:36363`, `:43504`, `:43771`; `metal/dense.metal:1607`, `:1613`, `:1638`, `:1668`, `:1933`,
  `:2002`; `ds4_tp.h:32`, `:126`, `:145`; `ds4_tp.c:941`, `:975`;
  `TP-A0-ROWSPLIT-TEST-PLAN.md:713-723`, `:731-740`. **Every one resolves to what the
  document says it does.** The only drift found: `dense.metal:1913-1914` is quoted for the
  `NR1` "widest token tile that evenly divides the batch" rule, but those lines are a comment
  about `M % NR0` and `K % NK`; and the `ds4_gpu_args.h` line-count row (finding 12).

---

## Verdict

**The document is a credible basis for a go/no-go decision on the kernels, and not a credible
basis for one on the port.**

What it does well is unusual and worth saying plainly: the separability result is correct and
is genuinely the thing that makes a serial recurrent scan fit a 60-core GPU; the determinism
reasoning in §6.2 is a real theorem, correctly derived, that I verified against the shader
source; the shape audit caught two silent-correctness traps (the two RMSNorm conventions, the
missing conv state) that the port plan had wrong; §4's TP2 analysis is right on every
mechanism I could check; the citations are accurate to a degree I do not usually see; and the
document is honest about which numbers are guesses. §5.5 and §3.4 are both correct calls made
for correct, quantified reasons.

What it does not survive is contact with the two questions the go/no-go actually turns on.

**First, the decode case is not established.** The design's own headline — 3 dispatches per
GDN layer — is achieved by fusing the layer's largest bandwidth consumer into a kernel with
64 threadgroups where ds4 currently uses 4,120, and D1 is the one kernel in the document that
receives no occupancy analysis (finding 2). The dispatch-count objective it is optimising is
worth 68–158 µs per removed dispatch by its own arithmetic, and the fusion plausibly costs
several times that. Meanwhile the falsifier it proposes for the residual cannot distinguish
its own two hypotheses, because both are per-layer (finding 1), and the +20 encoder boundaries
Qwen pays for having 48 layers instead of 43 are certain under every hypothesis and are
presented in §4.3 as "zero lines change."

**Second, the port-integration case is materially understated.** The polarity trap is 3–4×
larger than enumerated, spans the CPU cache path and the memory planner as well as the GPU
state path, and its assigned oracle cannot fail for the reason it exists (finding 3). The
pinned projection — the item the port plan itself calls "the only item with no prior art and
no free lunch" — is specified as a kernel that cannot exist in the family §6.2 proves things
about, and its prefill cost (~4.3 TFLOP/chunk/rank) is 20× the FLOP budget the document
carefully computes for K3 and appears nowhere (finding 4). ~6,150 lines is optimistic by
roughly a third to a half.

**Recommendation.** Do not treat this as go/no-go-ready. Three cheap things would make it so,
in this order:

1. **Run the stage profiler, not the boundary instrument** (`TP-A0-ROWSPLIT-TEST-PLAN.md:718-721`,
   ~20 min). It bounds how much of the 13 ms is DS4-specific compute that does *not* transfer
   to Qwen. That number, not the boundary slope, determines the size of the hole.
2. **Microbench achieved GB/s versus threadgroup count** for a 13 MB weight read at 128
   threads/threadgroup on this rig, at 40, 64, 256 and 4,120 threadgroups. Half a day, and it
   settles finding 2 — which is the difference between "3 dispatches/layer" being the design's
   best result and being its worst decision.
3. **Prototype `kernel_gdn_in_proj` at `ntile = 1` and `ntile = 32` and diff the outputs**
   before anything else in §7.2's build order. If a single k-reduction tree cannot span both,
   §6.3 collapses and with it the decode/prefill cost model.

None of the findings above is unfixable and none of them touches the separability result the
design is built on. But the specific numbers that would go into a go/no-go — decode
milliseconds per token and total lines — are both wrong in the optimistic direction, and the
one instrument the document nominates to resolve the largest uncertainty is the wrong
instrument.
