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

## 4. Memory and the deployment fork

At ds4's quant mix (MXFP4 routed experts, BF16 elsewhere):

| config | GPU-resident | n-gram table | total |
|---|---|---|---|
| single node, table in RAM at FP8 | 73.5 GiB | 47.7 | **121 GiB ✗** (>119 available) |
| single node, table requantised ~4-bit | 73.5 | 25.3 | 99 GiB ✓ |
| single node, table SSD-streamed | 73.5 | page cache | ✓ |
| **TP2, table split 8/8 by hash head** | 29.9 + 13.6 | 23.9 | **67 GiB/rank ✓** |

The official FP8 release quantises **the 512 routed experts (block-wise
`[128,128]`) and the n-gram table (per-tensor)**; everything else stays BF16.

**TP2 is the comfortable configuration**, and it is the one matching both the
rig and our existing investment. Single-node is possible but requires either
SSD-streaming the table or requantising it below FP8 — and
`ds4_tp_validate_engine_options` refuses SSD streaming under TP, so the two
paths are mutually exclusive as the code stands.

TP2 is also fine for the *released* FP8 weights via a refined block scale
(`moe_intermediate_size = 640 = 5×128`; TP2 → 320 → gcd 64). It is TP8 that is
rejected, not TP2.

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
2. **The chunked/sequential accumulation-order problem.** DS4 buys its way out
   by rebuilding the compressor frontier from the last 4 tokens
   (`ds4.c:28503-28509`, explicitly to avoid mixing accumulation orders). **For
   GDN there is no bounded rebuild** — the state is unbounded. The chunked and
   sequential kernels must agree bit-for-bit by construction. In my judgement
   this is the sharpest correctness constraint in the port.
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
6. **`dedc830` does not generalise** — and its own comment says why. The
   alignment is legal *because* at an lcm boundary the correct frontier is a
   known constant. A recurrent state has no such position other than 0. The
   pattern that does generalise is `spec_frontier_snapshot/restore`, already an
   O(state) copy at 12 MiB today versus 108 MiB then — a 9× scale-up of working
   code, not a new mechanism.

## 8. Estimated throughput

*Pending — a dedicated modelling pass is in flight. Will give prefill and
decode across the seven sweep contexts with sensitivity ranges, anchored to our
measured achieved rates rather than peaks.*

First-order expectation, to be replaced: decode roughly comparable to or better
than DS4-Flash at short context and **substantially flatter with context**, on
the strength of §6.1. Prefill is the less certain half — the 96 GR modules add
~3.5% of a cold 131k prefill in pure FLOPs, but the 512-expert routing and the
loss of every MoE decode specialisation cut the other way.

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
