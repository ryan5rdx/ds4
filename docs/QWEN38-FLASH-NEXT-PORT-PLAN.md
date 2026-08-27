# Porting Qwen3.8-Flash-Next to ds4 — plan

Date 2026-08-26. Rig: 2× M2 Ultra 60-core 128GB, TP2 over Thunderbolt RDMA, Metal.
Sources: the tech report (`/tmp/tech_report.pdf`), the HF reference
`modeling_qwen4_exp.py`, SGLang PR #36497 and vLLM PR #53896 (both open,
unmerged, shipping day-0 Docker images), llama.cpp #27742, and the ds4 tree at
`upstream-metal-wins`. Supersedes the Qwen half of
`docs/NEW-MODEL-FEASIBILITY-2026-08-26.md`; that doc's GLM half still stands.

## 1. Verdict

**Feasible, ~62,000 lines Metal-only, and a correct model exists at ~22,000 of
them.** The architecture is a better *shape* for TP2 than DeepSeek-V4-Flash —
narrower gate payload, a residual that never crosses the wire, 36 of 48 layers
that do not degrade with context. But the specific optimisations that produced
our +65.8% prefill arc are worth **maybe +3% here**, because the replicated
compute they removed does not exist on this model, and the replicated compute
this model adds cannot be removed.

Two facts should drive the decision more than the line count:

- **65% of the cost is volume caused by the absence of a per-family seam**, not
  by the model. Only the GDN kernels (~3.5k lines) are genuinely unknown work.
- **Three independent correctness oracles exist**, one provable from the
  report's own equations. That makes the go/no-go gate unusually clean.

## 2. What it is

48 layers, hidden 2560, head_dim 256, ~180B total / ~6.2B active.

| | |
|---|---|
| layer pattern | **36 Gated DeltaNet : 12 QSA**, full attention at `idx % 4 == 3` |
| QSA | GQA 24:2, 4-head MQA indexer, 4:1 key pooling, top-512 of a 2048 budget, partial RoPE 64/256, interleaved mRoPE, sigmoid output gate |
| GDN | 48 v-heads / 16 k-heads, `[48,128,128]` fp32 state, fused depthwise causal ShortConv (kernel 4, no dilation), per-head scalar decay |
| MoE | every layer: 512 experts, top-10 + 1 shared, ffn 640 |
| residual | 4-branch Gated Residual, rank 320, **96 modules, no layernorms anywhere** |
| extras | 51B n-gram hash table at one layer; 1 MTP layer; zero-centred RMSNorm throughout |

### Implementation traps, from the reference rather than the report

These are the things a port gets silently wrong.

- **GDN's output gate is `sigmoid` for this model** — but the
  `Qwen4ExpTextRMSNormGated` class *defaults to SiLU* and takes the activation
  from `config.output_gate_type`. Follow the class default and the model is
  quietly wrong. (QSA's attention gate is separately sigmoid.)
- **L2 norm is `x * rsqrt(sum(x²) + 1e-6)`, not `F.normalize`.** The docstring
  says it mirrors FLA's convention.
- **Decay is `g = -exp(A_log) * softplus(a + dt_bias)`, per-head scalar, and
  unbounded** — unlike GLM-5.3-Flash's KDA, which takes a `-5.0` lower-bound
  branch. The two linear-attention variants differ here; do not share the code
  without a flag.
- **Chunked-scan chunk size is 64**, and **the reference asserts no numerical
  equivalence** between the chunked and sequential forms. See §7 — this is the
  sharpest correctness constraint in the port.
- Zero-centred RMSNorm is `output * (1.0 + weight)`. ds4 has no such form, and
  norms are fused into ~15 kernels, so it propagates.

## 3. The decisive TP fact

**GDN's output RMSNorm is per-head.** Verified in the reference:

```python
self.norm = Qwen4ExpTextRMSNormGated(self.head_v_dim, ...)   # weight (head_v_dim,)
core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
variance = hidden_states.pow(2).mean(-1, keepdim=True)       # over head_v_dim only
```

So a GDN layer is **head-parallel with exactly one cross-rank exchange, at the
end, after `W_o`** — the position DS4's ATTN gate already occupies.

Consequences:

| | DS4-Flash | Qwen3.8-Flash-Next |
|---|---|---|
| gates/token (decode) | 86 (43×2) | **96** (48×2) |
| payload/gate | 4096×4 = 16,384 B | **2560×4 = 10,240 B** |
| bytes/token/direction | 1.41 MB | **0.98 MB (−30%)** |
| schedule | affine `(0, 1, 86)` | **affine `(0, 1, 96)`** — `tp_gate_slot()` unchanged |
| WRs/gate | 1, *exactly* at the 16 KB cap | 1, with 6 KB of slack |

**The 4-wide residual does not cross the wire.** `ds4_tp.h:126`:
*"vec_bytes = n_embd * 4 (f32 partials, never quantized on the wire)"* — the 4
is `sizeof(float)`, not `n_hc`. Each rank keeps its own 4-wide residual locally
and the HC expand consumes n_embd partials after the gate.

**If GDN were replicated instead of head-split**, the firing pattern becomes
{FFN on all 48, ATTN on 12} — not an arithmetic progression, and no
`(start, step, per_token)` triple expresses it. Since `tp_rdma_post_gate_recv()`
computes an unchecked slab address from the schedule, a wrong table is a silent
out-of-bounds write. **Head-split GDN even if the compute case were marginal.**

## 4. Weights, metadata, and memory

**We are using `unsloth/Qwen3.8-Flash-Next-GGUF`.** That closes the conversion
question entirely — no converter to write, no metadata namespace to invent.

### The GGUF metadata namespace (read from the real file, 2026-08-27)

`general.architecture = "qwen4exp"` — one arch string to add at
`config_validate_model` (`ds4.c`), which today accepts exactly `"glm-dsa"` and
silently falls through to the DeepSeek validator for anything else.

The `qwen4exp.*` namespace maps cleanly onto `ds4_shape`:

| key | value | key | value |
|---|---|---|---|
| `block_count` | 48 | `expert_count` | 512 |
| `embedding_length` | 2560 | `expert_used_count` | 10 |
| `attention.head_count` | 24 | `expert_feed_forward_length` | 640 |
| `attention.head_count_kv` | **2** | `expert_shared_feed_forward_length` | 640 |
| `attention.key_length` / `value_length` | 256 / 256 | `hyper_connection.count` | 4 |
| `attention.indexer.head_count` | 4 | `hyper_connection.low_rank` | 320 |
| `attention.indexer.key_length` | 128 | `full_attention_interval` | 4 |
| `attention.indexer.top_k` | 2048 | `context_length` | 262144 |
| `rope.dimension_count` | 64 | `rope.freq_base` | 1e7 |
| `rope.dimension_sections` | [11,11,10,0] | `embedding_length_per_layer_input` | 160 |
| `ssm.state_size` | 128 | `ssm.inner_size` | 6144 |
| `ssm.group_count` | 16 | `ssm.time_step_rank` | 48 |
| `ssm.conv_kernel` | 4 | | |

**Two findings that materially reduce the port:**

**`qwen4exp.attention.compress_ratios = [0,0,0,4, 0,0,0,4, …]`** — the GGUF
uses **ds4's own per-layer convention**, with `4` on the QSA layers
(`idx % 4 == 3`) and `0` on the GDN layers. `g_ds4_compress_ratios[]` can read
it directly; what changes is only the *interpretation* of `ratio == 0`, which
means "SWA-only" for DeepSeek and "GDN layer" for Qwen. That is a
family-dependent reinterpretation of an existing array rather than the parallel
layer-kind enum the audit assumed we would need.

**The whole n-gram hash specification is in the metadata as first-class keys** —
`ple.layer_multipliers` `[23703573157769, 20109073645365, 8052911324071]`,
`ple.head_vocab_sizes` (16 primes just above 2e7), `ple.head_offsets`,
`ple.heads_per_ngram` 8, `ple.ngram_size` 3, `ple.conv_kernel` 4,
`ple.layers [1]`. The multipliers match what was independently read out of the
safetensors, and `ple.layers = [1]` settles the report-says-2 /
weights-say-1 discrepancy. Only the *composition rule* still has to be read out
of llama.cpp's reader — but since the GGUF carries exactly these keys, that
reader is the specification.

**And it confirms a latent defect with a real file:** `tokenizer.ggml.pre =
"qwen35"`. ds4 never reads that key (§10), so this GGUF would be tokenised with
the DeepSeek JoyAI splitter, silently. `tokenizer.ggml.model = "gpt2"`, i.e.
byte-level BPE, which is the algorithm ds4 already implements.

### Available quantisations

`UD-IQ1_S`, `UD-IQ1_M`, `UD-Q2_K_XL`, `UD-Q3_K_XL`, `UD-IQ3_XXS`, `UD-IQ4_XS`,
`UD-Q4_K_XL`. **There is no Q8, Q6_K or Q5_K.** Highest is **UD-Q4_K_XL at
103.6 GiB** (4 shards; `general.file_type` 15 = Q4_K_M base with Unsloth
dynamic per-tensor overrides, imatrix-calibrated over 45 chunks).

| | model | + KV/state @131k | of 117 GiB usable |
|---|---|---|---|
| single node | 103.6 GiB | ~3.2 | **~107 — fits, no room to spare** |
| **TP2** | ~52/rank + replication | ~1.6 | **~55–60/rank — comfortable** |

Both work; TP2 has the headroom for multi-slot batching and matches the rig.

Two caveats. **UD-Q4_K_XL is a *mixed* quant**, so the loader must tolerate a
heterogeneous per-tensor type mix, and ds4's read side has gaps there — Q5_K/Q6_K
are MoE-`mm`-only with no `mv`, and a Q8_K routed prefill requests kernels that
exist in no `.metal` file (`ds4_metal.m:31044`). **And the repo is one day old
with zero downloads** — nobody has validated these quants. Budget for the
possibility that they are wrong, and lean on the S2 logits oracle.

One upside for §8: modelling GDN weights at Q8 was conservative. At ~4.94
bits/param average they are ~0.64 GB/rank rather than 1.04, taking decode from
~3.5 to ~3.1 GB/token/rank — roughly 11% better than estimated.

## 5. Do our optimisations transfer?

**About a third.**

| optimisation | DS4 result | Qwen |
|---|---|---|
| A0 row-split at `pos0 > 0` | +7.2% @131k | **12 of 48 layers only — the row axis is closed for recurrent layers.** ≤ +2%, but flat in context rather than declining |
| Indexer split | +19.5%, bit-identical | **Transfers, still bit-identical — prize ~28× smaller** (4 heads × 12 layers vs 64 × 21). ≲1% |
| Static-mixed split | +20.5%, bit-identical | **No analogue.** Qwen has no indexer-less layer class |
| FA `nsg` 8→4 | +7.3%, bit-identical | Method transfers, value does not — **re-sweep**. And `nqptg` *unblocks*: head_dim 256 halves shmem to 16,384 B at Q=8, so Q=12 fits with no restructure where DS4 is stuck at 8 |
| MoE expert split | structural | Transfers, better balance, and **partial replication becomes affordable** (+11.7 GiB/rank vs +27 on DS4) |
| Decode head split | structural | **Becomes the only TP axis**, prefill included |
| Gate schedule | 86, affine | 96, affine — see §3 |
| *neg:* replicate decode attn | −8.4 to −11.8% | Still harmful, and worse: replicating GDN adds a second read of the state half |
| *neg:* sub-gate pipelining | wash | Still a wash — smaller payload, less to overlap |
| *neg:* `DECODE_NWG` optimal | plateau 8–32 | **Must be re-swept.** The null is shape-evidence, not knob-evidence |

**Why A0 cannot apply to a recurrent layer.** A0's correctness rests on
attention row *i* being a pure function of its *position* against a
position-indexed cache, so `(pos0 + tp_row0, tp_rows)` has a closed form. A GDN
row's output depends on `S_{t-1}`, a function of every prior row's values and
gates. There is no `attn_pos0` analogue. Row-splitting a GDN chunk is a
category error, not merely hard — rank 1 starting at T/2 needs rank 0's output.

**Why the indexer split still works.** QSA pools on the *key* axis: the pooled
block key is identical for every query, so scoring stays per-query-row. R5's
argument holds verbatim and it should come back bit-identical.

## 6. New opportunities this architecture creates

1. **Long-context decode should barely degrade.** DS4 falls −31% from 2k to
   131k. Qwen's only context-growing terms are the indexer key read and top-k
   on 12 layers; the 36 GDN layers are O(1). Modelled at **−5 to −8%**. This is
   the architectural answer to the problem R10/R11 could not optimise away.
2. **Partial expert replication becomes affordable.** One expert index across
   all 48 layers is 125 MB against DS4's 1.08 GB, so overlapping the shards
   costs +11.7 GiB/rank rather than +27. Worth ~3.4% end-to-end.
3. **Block-split the decode indexer** — see §9, and it applies to DS4 *today*.
4. **`nqptg` unblocks** — the R9 ceiling we could not reach on DS4.
5. **MTP with reported 4.06 mean accepted length**, versus DSpark's measured
   8.9–11.8% acceptance here. Treat the report's figure as an upper bound on a
   favourable fixture: `tp_mtp_hunt.md` already dismantled a comparable public
   claim (the mean includes the unconditional target token, and the fixture was
   predictable code rather than prose).
6. **A free correctness oracle.** QSA is bit-identical to dense attention below
   2051 cached tokens — provable from the report's own equations, since
   `⌊n/4⌋ ≤ 512` makes top-k the identity. Runs on the dev box with no model.

## 7. What gets worse

1. **The 96 GR modules.** 634M active params — 10.6% of the active budget —
   dense, read on **both ranks**, ~19× DS4's HC mixer traffic. Splitting was
   priced at **1.8 ms/token of new gates to save 0.63 ms** of duplicated read:
   net negative, and vLLM independently marks GR `disable_tp=True`. This is
   replicated work the campaign's most productive vein cannot mine.
2. **State determinism — reframed; my earlier statement of this was wrong.**
   I wrote that "the chunked and sequential kernels must agree bit-for-bit by
   construction" and called it the sharpest constraint in the port. That is not
   achievable, not required, and not what anyone does. Corrected in §7a below.

3. **Every MXFP4 MoE decode specialisation dies** — all are keyed on
   DS4-Flash geometry constants (`n_total_expert==256`, `nei0==6`,
   `gate_row_bytes==2176`, …). A fused `sum11` is a **prerequisite**, not an
   optimisation: `ds4_metal.m:39541` fails loudly without the fused path.
4. **512-expert routing costs on three axes**: 11 matmuls/token/layer vs 7,
   each expert 5× smaller (worse per-dispatch efficiency), 2.3× the selection
   work in prefill. Watch R4's argsort-canon lesson.
5. **The recurrent state cannot be rewound.** 108 MiB/sequence, halving to
   54 MiB/rank under head split. `ds4_session_rewind()` becomes
   restore-from-snapshot, not integer truncation.
6. **`dedc830` half-generalises, and the useful half is the half that does.**
   The *rebuild* does not — it needs a bounded window. But the **alignment**
   does, and `ds4_session_rewind_align()` already snaps to `lcm(non-zero
   ratios)` = **128, a multiple of 64**. It is already the right shape. What
   that code is really doing is canonicalising the carried state — making it a
   function of a fixed grid rather than of the chunk schedule — and GDN's state
   at a 64-aligned position is a *complete* object, so it gets that
   canonicality without needing the rebuild at all.

   Independent convergence worth knowing: vLLM and SGLang both arrived at
   `lcm(kernel chunk, cache page)` as the only depth at which a recurrent state
   may be named. That is `ds4_session_rewind_align()` generalised.

## 7a. State determinism — what actually has to hold

Three comparisons matter, and they are not equally hard.

**Chunked vs chunked at different boundaries — holds, and this is the one that
matters.** The chunked gated delta rule is a pure reassociation (a closed-form
unit-lower-triangular inverse, terminating in exactly `chunk_size` steps — no
truncated series). Splitting a sequence is bitwise-identical **provided three
riders hold**, confirmed empirically by vLLM PR #49827: 64-aligned splits came
back bitwise equal, while splitting at [73, 128, 192] changed **131,061 of
131,072 final-state elements**. Non-aligned splitting does not degrade the
state, it replaces it.

| rider | ds4 status |
|---|---|
| every carried-state boundary is a multiple of **64** | **violated** — the final chunk of a range is `remaining` (arbitrary, `ds4.c:36960`), and resumed prefill realigns to `prefill_cap`, not 64, from an arbitrary `pos0` |
| the state handoff is **fp32** | free to satisfy; costs ~1.6% of decode versus bf16 |
| **nothing selects a kernel by sequence length or batch row count** | **violated — see below** |

**Chunked prefill → sequential decode — a non-issue**, as suspected. Every
production stack does exactly this. The state handed over is a valid GDN state
and decode continues correctly from it; nobody treats the one-time offset as a
defect. One trap: the HF reference hardcodes `initial_state=None` on the chunk
path, so *chunked prefill with a carried state is off the reference's tested
path entirely*, and FLA's own test suite has no coverage for it either.

**Speculative verify — I over-weighted this, and ds4 already shipped the
concession.** `README.md:212-218` documents it verbatim: a long greedy DSpark
run "may therefore diverge from a run without DSpark after an otherwise valid
accepted block… use ordinary decoding, `--quality`, or `--dspark-strict` when
byte-for-byte reproducibility with one-token decode is required." GDN widens
the blast radius (divergence enters persistent state rather than appended rows)
but creates no new problem class. Both vLLM and SGLang sidestep it anyway by
verifying GDN with the **recurrent** kernel, so verify-vs-decode is not a
chunked-vs-recurrent mismatch at all.

### The hazard I missed, and it is live on the current branch

**ds4's matmul dispatch is keyed on batch row count.** Verified:

```c
static int16_t ds4_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok) {
    if ((in_dim % 256u) == 0 && n_tok < 3) return 16;   // ds4_metal.m:5372
    ...
}
```
plus `ds4_gpu_mv_ext_r1ptg()` switching on `n_tok`, and `ds4_metal.m:20676`
taking a different kernel entirely at `n_tok == 1`, `<= 8`, and above.

`nxpsg` is the **reduction partition width along `in_dim`**. So the same row, in
a differently-sized batch, gets a different dot-product reduction tree — and
this sits *upstream* of any GDN scan, on the q/k/v/g projections.

For attention today this is bounded: a perturbed K row shifts one attention
weight. It is also exactly why `metal_graph_refresh_ratio4_compressor_state`
exists (`ds4.c:28504`: the full-chunk path "uses the matrix-matrix path; mixing
those two accumulation orders changes a few FP8 rounding decisions in later
chunks"). **For GDN the same perturbation enters the state and persists.**

Three consequences, most likely first:

1. **The ragged final prefill chunk** permanently offsets the state, and no
   choice of GDN kernel fixes it.
2. **Multi-slot batching makes slot count part of session provenance.** A
   session that ran alone and one that ran alongside three others would
   diverge. This is a *new axis*, and it lands on the branch we are on.
3. The DSpark batch verifier's projections differ from decode's regardless of
   the GDN scan.

**Checked on the current model: DS4 is clean today, and structurally so.**
The decode compressor projection passes a literal `1` for the row count
(`ds4.c:22951` and `:22957`), so it is a single-row matvec no matter how many
sessions are co-batched. And both native cross-session row-fusing paths bail
out under TP — `metal_graph_native_session_batch_qkv_supported` (`ds4.c:64831`)
and `..._shared_supported` (`ds4.c:64784`) each test `e->tp.active` and return
false — so on this rig sessions batch at the outer level only and every
projection still runs at `n_tok == 1`. Neither path touches the compressor in
any case.

The exposure is therefore *prospective*, and it is worth naming because it
points straight at this branch's own goal: the outer TP session batching is
already on by default (`DS4_METAL_TP_SESSION_BATCH` only disables it when
explicitly `0`, `ds4.c:64747`), so the natural next step is to let the native
row-fusing paths run under TP. The moment they do, projection reduction trees
become slot-count-dependent. For DS4 that is bounded — a shifted attention
weight. For GDN it would be permanent.

### The mitigation package

1. **Align, serialise, and run the residue recurrently.** Mandate every
   internal chunk boundary at a multiple of 64; write the inter-chunk state
   loop strictly serially (free — we author the kernel, so we inherit nobody's
   parallel scan); run the final `N mod 64` tokens through the *recurrent*
   kernel from the last aligned state. `metal_graph_prefill_decode_streaming_range`
   (`ds4.c:32222`) already does prefill-by-decode-loop, and
   `ds4_session_rewind_align()` already provides the alignment. Add a validator
   asserting `chunk % 64 == 0` on every non-terminal chunk.
2. **fp32 state.** ~1.6% of decode. Cheap insurance; vLLM calls it "necessary,
   not sufficient."
3. **Pin the projection dispatch for GDN inputs** — the mitigation for the
   hazard above, and **the only item with no prior art and no free lunch.**
   Force the q/k/v/g/beta projections onto one dispatch variant regardless of
   `n_tok`, analogous to `g_quality_mode`. Without it, slot count is part of
   session identity.
4. **Gate CI on teacher-forced KL, never on greedy top-1.** vLLM's report is
   unambiguous: "a single qualitatively-neutral logit difference flips an
   argmax and the greedy path diverges forever afterward." Field calibration:
   FLA gates at 0.5% relative RMS; vLLM's Mamba kernels at `atol=1e-2`; SGLang
   at KL ≤ 0.003–0.005. Measured chunk-vs-recurrent divergence is **~1.2e-4**,
   agreed by three independent sources.

## 8. Estimated throughput

**This is my own first-order model, not a validated one.** Two attempts at a
dedicated modelling pass died on server errors, so I built it directly. The
arithmetic is reproducible from the numbers below; the *effective-rate*
assumption is doing all the work, and our own ~0.63 transfer-factor lesson says
to treat the optimistic end as an upper bound.

### The model validates against a known quantity first

Summing active parameters by component gives **5.995 B against the card's
~6 B** — 0.1%. That is the check that the component breakdown is right before
any throughput claim rests on it.

| | routed | shared | GDN | QSA | GR | router | (head) |
|---|---|---|---|---|---|---|---|
| active params | 2.359 B | 0.236 | 2.087 | 0.617 | **0.633** | 0.063 | 0.636 |

### Decode — bytes per token per rank

MXFP4 experts, Q8 dense, fp32 state, TP2 with GDN head-split, GR replicated,
experts split by id with a top-10 straggler factor of 6.23/10.

| component | MB | note |
|---|---|---|
| routed experts | 780.9 | straggler-adjusted |
| GDN weights | 1043.3 | head-split |
| GR | 633.1 | **replicated — cannot be split (§7.1)** |
| output head | 317.8 | vocab-split |
| QSA weights | 308.7 | head-split |
| shared expert | 118.0 | row-split |
| router + n-gram proj | 95.9 | |
| GDN state r/w | 113.0 | fp32; **56.6 at bf16**, which production stacks use |
| KV read | 50.0 | capped at 2048 selected + tail — **context-independent** |
| indexer keys | 1.3 → 50.0 | the only term that grows: 2k → 131k |
| **total** | **3.46 → 3.51 GB** | **+1.4% across a 64× context range** |

Against DS4-Flash's ~3.3 GB. **Roughly the same bytes, but almost none of the
context slope** — that is the architectural claim, quantified.

### Decode — throughput

DS4's measured effective bandwidth falls 123 → 93.5 GB/s from 2k to 131k, and
that decline *is* its context-dependent work (indexer scoring on 21 layers × 64
heads). Qwen's equivalent term is 28× smaller, so its effective rate should
hold near its short-context value rather than decaying.

| ctx | bytes/rank | at 123 GB/s | with a 2× dispatch penalty | DS4 actual |
|---|---|---|---|---|
| 2048 | 3.46 GB | 35.5 t/s | ~31 | 41.09 |
| 131072 | 3.51 GB | 35.0 t/s | ~30 | 28.34 |

**Estimate: 31–38 t/s at 2k, 29–36 t/s at 131k — degradation of −3 to −6%
against DS4's −31%.** Likely *slower* than DS4 at short context and *faster*
at long, with a crossover somewhere around 16–32k.

### Prefill — FLOPs per 4096-token chunk, at 131k

| | Qwen | DS4-Flash |
|---|---|---|
| MoE | 19.33 (36.7%) | 53.2 |
| GDN projections | 17.01 (32.3%) | — |
| QSA / attention projections | 5.06 (9.6%) | 26.6 |
| **GR** | **5.15 (9.8%)** | — |
| attention core | 2.48 (4.7%) | 19.4 |
| **indexer scoring** | **1.65 (3.1%)** | **46.0 (32%)** |
| **total** | **52.6 TFLOP** | **145.4 TFLOP** |

**Qwen is 0.36× the prefill FLOPs.** The dominant term is the indexer: 4 heads
on 12 layers versus 64 heads on 21 gives `(64/4) × (21/12) = 28×` less work,
which alone accounts for 44 of the 93 TFLOP difference. Active params (6.0 vs
~9.7 B) account for most of the rest.

GDN's *recurrence* is negligible and I checked it rather than assuming: at
chunk 64 the per-head state update is ~4e6 MACs per (chunk, head), giving
~0.9 TFLOP/chunk against 17 for its projections.

DS4 measures 10.41 s/chunk at 131k → **13.97 TFLOP/s effective**. At the same
rate Qwen would be 3.77 s/chunk = **~1070 t/s**. Both mixes are ~80%
matmul-shaped, so the rate should be comparable.

**Estimate: 600–900 t/s at 131k** — discounting the FLOP-parity figure hard for
an immature kernel set, more dispatches, and the GR modules' small rank-320
matmuls. Against DS4's measured 393.53.

### Sensitivity — what would change the answer

| assumption | if wrong | effect |
|---|---|---|
| **dispatch count** (assumed ~2× DS4's 1021: 48 layers, GDN has more ops each) | 3× instead | decode −15 to −20%; **this is the dominant decode risk** |
| GDN weight quant (assumed Q8) | BF16 | +1.04 GB/token/rank → decode −25%. **The single most consequential quantisation decision in the model**, and the feasibility doc got it wrong |
| effective TFLOP/s in prefill | 10 instead of 14 | prefill ~750 t/s |
| GR read fusion | not fused | +168 MB/module materialised in prefill — untenable, must be fused end-to-end |
| state dtype | bf16 not fp32 | decode +1.6%, and halves per-slot memory |

### What dominates, and therefore what to optimise

**Prefill**: MoE (37%) and GDN projections (32%) — both plain matmul, both
already well served. GR at 9.8% is the only novel cost. Nothing here resembles
DS4's indexer problem, which is why the A0/indexer/static-mixed arc has no
analogue.

**Decode**: GDN weights (30%), routed experts (22%), GR (18%). The context-
dependent part is 1.4% of bytes. So decode optimisation should target the GDN
weight path and quantisation — **not** the context-scaling work that consumed
R1–R14 on DS4.

### The caveat that matters most

DS4 decode is **latency-bound, not bandwidth-bound** — it achieves 93–135 GB/s
of 800, and T8 has since shown the routed MoE matvec alone runs at ~400 GB/s in
isolation. So a bytes model is a *floor*, not a forecast, and the gap between
the two on DS4 is exactly the 11.1 ms M2 is chasing. If that gap is a fixed
per-layer cost, Qwen inherits it 48/43 = 1.12× over. If it scales with the
context-dependent work, Qwen largely escapes it.

**Which of those is true is the single measurement that would most reduce the
uncertainty here — and it is M2, on the model we already have.** That is a
strong argument for finishing M2 before committing to this port, independent of
anything about Qwen.

## 9. Do this first — it applies to DS4 today

**DS4's decode indexer is fully replicated.** `ds4_gpu_indexer_score_one_tensor()`
(`ds4_gpu.h:534`) takes no rank, world or base parameter, and `ds4.c:23241`
calls it with the full `layer_n_index_comp[il]` and all 64 heads. Both ranks
score every compressed key on all 21 ratio-4 layers, every decode token —
~353 MB/token of duplicated F32 reads at 131k, and one of the few decode terms
that *grows with context*, which is what M2's unattributed 11.1 ms is chasing.

We split this for prefill in R5. The decode analogue was never done.

Head-splitting will not work (the score sums over heads). **Block-splitting is
exact**: rank 0 scores keys `[0, n/2)`, rank 1 `[n/2, n)`, each takes a local
top-512, and the merge is lossless because any member of the global top-512 has
at most 511 elements above it globally, hence at most 511 in its own half.
Exchange is 512 (score, index) pairs = 4 KB/layer — M3's cheap 8.0 µs arm.

**Size it before building it**: `DS4_METAL_ABLATE_INDEXER_SCORE=1` at 131k
gives the whole stage cost and half of it is the ceiling. That ablation is
already in M2's scope and has never been run at long context. My bytes estimate
says ~0.9 ms/token total, so ~0.44 ms saved ≈ **1.2%** — more if the kernel is
occupancy- rather than bandwidth-bound.

## 10. Four latent defects, all worth fixing regardless

| | anchor |
|---|---|
| `ds4_tp_hash_check()` — the only cross-rank divergence checksum — **has no caller** | `ds4_tp.c:2277`, decl `ds4_tp.h:239` |
| `tokenizer.ggml.pre` is **never read**; the pre-tokenizer is chosen by family alone | comments at `ds4.c:39254`, `:39374` |
| Unknown architecture **silently falls through** to the DeepSeek validator | `config_validate_model`, no `else`; dies on `deepseek4.block_count` |
| `DS4_MAX_HEAD_KV = 1` is **inert** — one occurrence, its own definition | `ds4.c:506`; the real MQA assumption is elsewhere |

The first matters most here: we are considering a model whose per-rank state can
silently desync, and the mechanism built to detect exactly that is dead code.

## 11. Staging plan

Each stage independently verifiable. **S0–S3 ≈ 22,000 lines and produces a
correct model** — that is the go/no-go gate.

| stage | content | verified by | lines |
|---|---|---|---|
| **S0** | No model. Wire up `ds4_tp_hash_check()`. Snapshot-based rewind against the *existing* bounded state. Fix the arch fallthrough and read `tokenizer.ggml.pre`. Land the still-owed on-rig rewind check from `dedc830`. | existing suites; rewind-vs-cold-prefill logits | ~800 |
| **S1** | GGUF conversion, `DS4_SHAPE_QWEN4`, validators, binders, Qwen2 BPE, chat template. No inference. | tokenizer round-trip vs HF; every tensor binds | ~3,500 |
| **S2** | CPU reference forward: dense QSA, naive sequential GDN, GR, n-gram on CPU. | **logits vs HF, first 64 tokens** — the golden oracle | ~4,000 |
| **S3** | Metal trunk: MoE + dense QSA + GR + zero-centred norms + mRoPE; GDN still sequential. | bit-compare vs S2. **A correct, slow model exists here.** | ~14,000 |
| **S4** | GDN chunked-scan prefill kernel. | vs S3; pin the tolerance, do not hand-wave §7.2 | ~3,000 |
| **S5** | QSA indexer + top-k. | **below 2051 tokens assert bit-identity with S3's dense path**; above, Jaccard vs reference — llama.cpp's ReLU-per-head-before-sum trap moved it 0.794 → 0.975, make that a regression assertion | ~2,500 |
| **S6** | Session lifecycle: checkpoint ring, rewind, spec shadows, disk KV v3, per-slot budgeting. | rewind-then-resume vs cold prefill; 8-slot soak | ~2,500 |
| **S7** | Performance: fused decode kernels, GR read fusion, dispatch reduction, n-gram prefetch overlap. | speed harness; no logit drift vs S5 | ~8,000+ |
| **S8** | TP — head-split GDN 24/24, 96-gate affine schedule. | two-rank bit-identity via the now-live `hash_check` | ~5,000 |

## 12. Write this down before the first knob sweep

**A first knob sweep on Qwen will return flat nulls across the board, and that
is the predicted result of the predicates, not evidence about the knobs.**

| knob | predicate | Qwen | verdict |
|---|---|---|---|
| `use_mxfp4_moe_decode_*` (5 levels) | `n_total_expert==256, nei0==6, gate_row_bytes==2176, …` | 512, 10, all differ | dead |
| `hc_rms_scale_project` | `in_dim==16384\|\|28672 && out_dim==24` | 10240, 320 | dead |
| `fuse_norm_mix` | `hc_dim==16384 && mix_hc==24` | 10240, 320 | dead |
| `packed32` flash reduce | `n_head == 64` | 24 / 48 | dead |
| `parallel_full_ffn` | IQ2_XXS / Q2_K only | MXFP4 | dead |
| `DS4_METAL_DECODE_NWG` | plateau measured at 32 heads × 512 | QSA 12×256 | re-sweep |

Recording that prediction in advance is how we avoid re-learning
`DECODE_SPLITS`' history, where a null had to be retracted because the sweep
ran where the flag was inert.

Two further lessons carry unchanged: the **~0.63 standalone-to-rig transfer
factor** (harder here — with zero GDN prior art in the tree, *every* recurrent
kernel number will be born on the M1 Max), and **build any GDN harness at world
2 with the head split and a non-null addend from day one**, because R13b showed
a world-1 harness measured a more specialised kernel than production runs and
made a "92%, the kernels are fine" conclusion worthless.

## 10. M2 lands — the decode case, re-grounded on measurement (2026-08-27)

M2 was the measurement gating this decision. It is in, and it changes both
halves of the decode argument — one for the better, one for the worse.

**The context term is the indexer, and Qwen deletes most of it.** M2 attributes
essentially the whole of DS4's 11.1 ms long-context penalty to the sparse
indexer: converted to time, the two indexer chains supply **+5.36 ms of the
+5.87 ms measured growth** from 32k to 131k (`docs/TP-A0-ROWSPLIT-TEST-PLAN.md`,
M2 section). DS4 pays that on its **21 ratio-4 layers of 43 (49%)**. Qwen has
**12 QSA layers of 48 (25%)**, and the other 36 are GDN — O(1) state, no cache
to score, no growth with context at all. The "36 of 48 layers that do not
degrade with context" claim in §1 now has a measured mechanism and a measured
magnitude behind it, rather than being an architectural assertion.

**But the floor does not improve, and probably worsens.** M2's other finding is
a **~13 ms/token residual that no stage ablation touches and that does not move
with context** — 46% of the token at 32k, 35% at 131k, larger than the routed
MoE. If that floor is per-layer dispatch and synchronisation cost, Qwen
inherits it scaled by layer count: **13 × 48/43 ≈ 14.5 ms before GDN adds a
single dispatch of its own.**

Re-estimating decode on that basis, with the §8 figure of ~3.1 GB/token/rank at
the ~400 GB/s effective bandwidth T8 measured:

| | DS4 measured | Qwen estimated |
|---|---|---|
| 2k | 41.1 t/s | ~45 t/s (+10%) |
| 131k | 28.3 t/s | ~45 t/s (**+59%**) |

≈ 7.8 ms of memory time on a ≈ 14.5 ms floor, roughly flat in context.

Three things follow.

1. **Qwen's decode win is a long-context win, not a general one.** At short
   context it is within noise of what we already have. Anyone expecting a
   uniform speedup from a 6 B-active model should be corrected by this table:
   the floor, not the parameter count, sets short-context decode.
2. **"One dispatch per layer" in the GDN decode kernel is load-bearing, not a
   nice-to-have.** It is the difference between inheriting the floor and
   inflating it across 36 additional layers. This is now the single most
   important constraint on the kernel design, and `docs/QWEN38-GDN-KERNEL-DESIGN.md`
   §3 should be judged primarily on it.
3. **The floor may be worth more than the port — but that is not yet
   established.** ~13 ms/token sits on the model we already run, against
   ~7 ms/token of long-context gain from a ~62,000-line port. The caution is
   that the 13 ms is *unattributed*, not *recoverable*: M2 skipped `router`,
   `shared` and `kv` because they do not ablate cleanly, so the residual
   contains the shared expert, the router and the KV store — real work that no
   amount of stall-hunting removes. The recoverable fraction is unknown. A
   single `DS4_METAL_DECODE_STAGE_PROFILE` run at 32k/131k bounds those three
   directly and is ~20 minutes; **it should precede any Qwen implementation
   work**, because it prices both sides of this trade at once.

**The open question that decides how the table above reads:** whether the floor
is per-layer or per-token. Per-layer cost scales with `n_layer` and Qwen is
worse on it; per-token fixed cost does not scale and Qwen keeps the full ~13 ms
floor, landing nearer ~48 t/s. That is a measurement, not an argument, and it
should be made on DS4 before any Qwen code is written.
