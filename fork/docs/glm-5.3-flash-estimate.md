# GLM-5.3-Flash on the TP2 rig — projected performance

Estimate only. Nothing here is measured on GLM; every number is scaled from
DeepSeek-V4-Flash measurements on the same hardware. Confidence is stated per
section. See `benchmarks.md` for the DS4F measurements this rests on.

## Architecture

From the published config (`model_type: glm5_next`), reconciled against the
weight index to 0.2%.

| | GLM-5.3-Flash | DS4F |
|---|---|---|
| layers | 45 (3 dense + **42 MoE**) | 43 |
| attention split | **34 linear (KDA) : 11 sparse MLA** | 43 MLA |
| hidden | 4096 | 4096 |
| vocab | 154,880 | 129,280 |
| routed experts / top-k | 288 / **8** | 256 / 6 |
| shared experts | 1 | 1 |
| MoE intermediate | 2048 | 2048 |
| MLA | kv_lora 512, q_lora 1536, qk/v head 256, **NoPE** | kv 512, q_lora 1024, head 512 |
| indexer | top-2048, 32 heads, 4:1 k-pool | top-512, 64 heads, 4:1 on even layers |
| context | 1,048,576 | 131,072 |
| MTP | 1 draft layer | 3-stage DSpark |
| extra | 24-block ViT, mHC, clamped SwiGLU | HC |

Active parameters: routed 8.456 B, shared 1.057 B, dense FFN 0.453 B, attention
6.051 B (34 KDA × 137.73 M + 11 MLA × 124.40 M), head/embed 0.634 B each, ViT
~0.547 B. Total **17.3 B text / 17.8 B with vision**, against a published 18 B.

**`DS4_SHAPE_GLM52` already carries most of this** — 154,880 vocab, top-8,
32 indexer heads at top-2048, kv-LoRA 512, MLA key/value 256, 3 leading dense
layers, 1 next-token-prediction layer. The shape table is closer to ready than
the kernel set is.

## Projected throughput

Per-rank decode bytes, accounting for what TP actually splits (routed experts by
ownership; attention heads, attention-output groups, shared-expert lanes and the
LM head by compute):

| quant | GiB/rank | GB/rank/token | decode @2k | notes |
|---|---|---|---|---|
| UD-Q2_K_XL | 50.8 | 4.94 | **37–45** | only quant loadable under TP |
| UD-IQ3_XXS | 55.9 | 5.10 | 36–44 | routed type rejected at load |
| UD-Q3_K_XL | 68.9 | 5.48 | 34–41 | routed type rejected at load |
| UD-IQ4_XS | 73.1 | 5.61 | 33–40 | routed type rejected at load |
| UD-Q4_K_XL | 93.1 | 6.21 | **30–35** | rejected by the GLM TP gate |

The ladder is much flatter than file sizes suggest: real per-rank decode bytes
vary only **26%** across it, because a ~3.5 GB replicated/head-split floor is
quant-insensitive under dynamic quantisation. Routed experts are 97% of the file
but only 28–43% of per-rank decode bytes.

The range on each row is the stall factor. DS4F runs 1.84× off its byte floor;
GLM should run worse — 45 layers rather than 43, and 34 of them are KDA with
depthwise convolutions, two low-rank gate chains, a gated fp32 norm and a
recurrent state update, all small dispatches on a path where marginal dispatch
cost is 1–2.5 µs and stalls are already inside kernels. Expect 2.0–2.3×.

Prefill, scaling by active FLOPs (validated: DS4F prefill runs at 59% of the
21 TFLOP/s roof, so it is genuinely compute-bound):

| ctx | GLM | DS4F |
|---|---|---|
| 2k | ~340 | 423 |
| 8k | ~410 | 500 |
| 32k | ~410 | 461 |
| 131k | ~370 | 367 |

## Context scaling

DS4F decode falls 41.2 → 29.4 (−29%) from 2k to 131k. About 40% of that happens
below 4k, before the indexed path is even reachable — it is dense-window growth,
not indexer cost. Above the crossover the cost is the indexer scoring every
compressed row each token: at 131k, 21 ratio-4 layers × 32,768 rows × 128 × 64
heads ≈ 11.6 GFLOP/token, roughly 7.4 ms at the measured scoring rate.

GLM's advantage is **not** IndexPool — its 4:1 pooling matches DS4F's ratio-4
compression, so per-layer scoring rows are the same. The saving is **11 layers
instead of 21 and 32 indexer heads instead of 64**, about 4× less scoring work,
so roughly +2.5–3 ms at 131k against DS4F's +9.7.

That is a genuine long-context advantage, but part of the flatter curve is the
2k figure being *depressed*: KDA writes and reads ~285 MB/token of recurrent
state at every context, a constant tax. A flatter curve earned by a lower
starting point is not a win.

## Quantisation is the blocker

`tensor_is_routed_expert_type` accepts Q8_0, IQ2_XXS, Q2_K, Q4_K, Q5_K, Q6_K and
MXFP4, so IQ1_S, IQ1_M, IQ3_XXS, Q3_K and IQ4_XS are rejected at load.

**On the GLM path under tensor parallelism the gate is much narrower**:
`glm_tp_validate_ownership_kernels` accepts a routed gate type of **IQ2_XXS or
Q2_K only** and aborts startup otherwise, because anything else silently
computes the full expert set. So of seven published quantisations, **only
UD-Q2_K_XL is viable**, and only if every layer's gate tensor is Q2_K — which
dynamic quantisation does not guarantee.

Two related traps: Q5_K and Q6_K are whitelisted at load but have no Metal
decode kernel; and an MXFP4 GLM layer is accepted by the loader and rejected
silently by the Metal side.

**Widening that TP gate is the cheapest unblock available** and it gates the
whole port, not just a quality tier.

## Multi-token prediction

Structurally dead on this rig, for a reason that has nothing to do with the
drafter.

A one-token draft needs a two-row verify. The measured verify cost is
`V(k) = 34.83 + 20.64k`, so V(2) = 76.1 ms against a 24.26 ms token — break-even
mean commit **3.14**, against a *maximum possible* commit of 2.0 at k=1. Even a
perfect single-token drafter cannot pay.

GLM improves two terms and worsens two. Better: the drafter is one in-model
layer reusing the trunk rather than three stages that suspend the expert split
while they run, and a longer token time lowers the threshold. Worse: the routed
union grows faster in absolute terms (288 experts top-8 gives 15.8 distinct at
two rows against 256 top-6's 11.7), and the replicated per-rank floor is larger.
New: speculative reject must snapshot and restore ~142 MiB of KDA recurrent
state per cycle, roughly 1.2–2.5 ms added to the fixed term.

**The lever is the fixed verify term, not the drafter.** That term is known to be
layer-encode overhead inside the batch — 99.97% of verify time — so cheaper
attention does not touch it.

## Port effort

Already in the tree and directly reusable: mHC (the hyper-connection multiplier
and Sinkhorn iteration count already match), MLA with q/kv-LoRA on the GLM path
including shape validators at these ranks, the sparse indexer, MoE with a
sigmoid router and shared expert, per-layer heterogeneous block types, and the
entire GLM frontend — tokenizer, chat template, tool format, server and agent
plumbing. Shape maxima already accommodate the model.

Genuinely new: the **KDA recurrent kernels** — no linear-attention, gated-delta
or short-convolution code exists anywhere in the tree — and, more awkward,
**unbounded recurrent-state session semantics**. Chunked prefill, rewind,
speculative reject, exact replay and tensor-parallel lockstep all assume
position-indexed state that can be recomputed from a position; recurrent state
cannot. That, not the kernel, is the real gate.

Existing design work applies: the Qwen gated-delta-net kernel design and port
plan already in the tree describe the same gated delta rule. Qwen and GLM differ
only in decay parameterisation — per-head scalar versus low-rank per-channel —
so it is one kernel family behind one flag, not two ports.
