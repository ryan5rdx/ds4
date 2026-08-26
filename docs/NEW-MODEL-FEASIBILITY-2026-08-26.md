# Porting Qwen3.8-Flash-Next and GLM-5.3-Flash to ds4 — feasibility

Date: 2026-08-26. Rig: 2× M2 Ultra 60-core 128GB, TP2 over Thunderbolt RDMA, Metal.
Both models are ~2 days old (Qwen repo created 2026-08-24, GLM 2026-08-25).

## Verdict

Both are **feasible on this hardware and a major program, not a port.** Memory is
not the blocker for either. The blocker is that both are **hybrid
linear-attention** models, and ds4's session model is built end-to-end on the
assumption that state is position-indexed or replayable from a bounded tail.

**Recommendation: commit to neither yet.** Run a bounded spike against the one
question that gates both (§6), then start **GLM-5.3-Flash** if it passes.

## 1. The cost anchor

Adding **GLM-5.2** — architecturally *close* to DeepSeek-V4 (MLA + DSA indexer +
MoE, same BPE algorithm) — cost, in commit `005afed` plus ~40 follow-ups:

| | lines added |
|---|---|
| `ds4.c` | 20,718 |
| `ds4_metal.m` | 18,575 |
| `metal/dsv4_misc.metal` + `metal/moe.metal` | 7,743 |
| hotlist `.inc` | 6,509 |
| everything else | ~4,900 |
| **Metal-only subtotal** | **~52,500 across 25 files** |
| ROCm (`ef8d923`) | +8,870 |
| CUDA | ~4,300 |

That is the **easy** case. Both targets here are strictly harder. There is no
`model_ops` vtable and no per-family registry — GLM is a from-scratch parallel
implementation (192 `glm_graph_*` functions, ~12.7k lines) that shares only the
kernel-dispatch library. `AGENT.md:3-5` states the intent: *"not a generic GGUF
runner."*

## 2. Both models, side by side

They are strikingly convergent — this looks like the 2026 frontier-MoE recipe.

| | Qwen3.8-Flash-Next | GLM-5.3-Flash |
|---|---|---|
| total / active params | 180B / ~6B | 321B / ~18B |
| layers | 48 | 45 |
| linear : full | **3:1** — 36 GDN : 12 QSA | **~3:1** — 34 KDA : 11 DSA |
| linear attention | Gated DeltaNet | Kimi Delta Attention |
| full attention | QSA, GQA 24:2, hd 256, partial RoPE | MLA **NoPE**, 64 heads, hd 256 |
| indexer | 4 heads, pooled **4:1**, top-512 of 2048 | 32 heads, pooled **4:1**, top-512 of 2048 |
| residual | 4-branch Gated Residual, no layernorms | 4-stream mHC + 20-iter Sinkhorn |
| experts | 512, top-10, +1 shared, ffn 640 | 288, top-8, +1 shared, ffn 2048 |
| MTP | 1 layer (full QSA + MoE) | 1 layer (full DSA + MoE) |
| context | 262k native, 1M YaRN | 1M native |
| extras | 51B n-gram hash table, mRoPE, vision | clamped SwiGLU (limit 10), vision |
| official GGUF | **none** | **none** (all repos are empty placeholders) |

Both linear-attention variants are the **same gated delta rule**, differing only
in decay parameterisation (Qwen: scalar α per head; GLM: low-rank per-channel
gate with a −5.0 lower bound). **One kernel family covers both.**

## 3. Memory — both fit

Budget: `iogpu.wired_limit_mb=120000` → ~117 GiB/node; current proven shard is
76.73 GiB/rank; ceiling ~110 GiB/rank → **~200 GiB total** at ds4's quant mix.

| | as released | at ds4 quant mix | per rank |
|---|---|---|---|
| Qwen3.8-Flash-Next | FP8 172.8 GiB | experts MXFP4 ~60 + PLE Q8 ~48 + rest ~11 = **~119 GiB** | **~62 GiB** |
| GLM-5.3-Flash | FP8 **305.8 GiB** | 304.4B of 321.3B are routed experts → **~170 GiB** | **~90–95 GiB** |

- Qwen is *below* our current shard. The 51B n-gram table splits cleanly by hash
  head (8 of 16 per rank, one extra gate at one layer) and is a pure gather —
  ~2560 values from 16 rows per token — so it is a natural fit for lazy faulting
  rather than wired residency.
- GLM fits but is tight, and **requires MXFP4 routed experts**. Relevant: the
  `{IQ2_XXS, Q2_K}` TP quant gate at `ds4.c:41697` is **stale** — Metal already
  accepts Q4_K/Q5_K/Q6_K for GLM routed MoE (`ds4_metal.m:36945-36959`) and
  already threads `tp_expert_base`/`tp_rank`/`tp_world`. That gate is a
  predicate to widen, not kernels to write.
- Neither ships a GGUF, and `gguf-tools/deepseek4-quantize.c` **cannot emit one
  without a donor template GGUF** (it copies the KV block as an opaque blob,
  `:1963`). Either write a real converter or take a third-party quant
  dependency, which is what GLM-5.2 does today.

## 4. What is already reusable — more than it first appears

| Needed | Status in ds4 |
|---|---|
| 4-wide residual + Sinkhorn mixing | **Exists.** `metal/dsv4_hc.metal:113` `kernel_dsv4_hc_split_sinkhorn`; shape fields `n_hc=4`, `n_hc_sinkhorn_iter=20`. GLM's mHC *is* this mechanism — the HF reference says so: *"Unlike DeepSeek-V4, this is an unweighted mean."* |
| GQA attention | **Exists.** `metal/flash_attn.metal:381` carries the ggml GQA broadcast `ikv2 = iq2/(ne02/ne_12_2)`, and both extremes run in production — MQA at 16 DeepSeek sites, true MHA at `ds4_metal.m:34338`. Plain GQA is the interpolation, a host parameter. |
| Sparse indexer + top-k | **Exists** (DSA). Both models pool keys 4:1 — a modification of the existing scorer, not a new mechanism. |
| MLA with q/kv-LoRA | **Exists** on the GLM path. |
| MoE: sigmoid router, bias-corrected top-k, shared expert | **Exists.** |
| Per-layer heterogeneous block types | **Exists.** `g_ds4_compress_ratios[]` already selects *different attention micro-architectures per layer* (0/4/128) and drives per-layer cache sizing, RoPE base and state allocation. This is the hook for `LAYER_LINEAR_ATTN`. |
| Per-layer recurrent state + snapshot/restore | **Partially exists.** `layer_attn_state_kv/score`, `layer_index_state_kv/score` are non-position-indexed per-layer state, with working `spec_frontier_snapshot/restore/commit_prefix` (`ds4.c:53141+`) and disk serialisation. |

**External prior art for Qwen:** llama.cpp PRs #27739 (closed) and #27742 (open
draft). The decisive line — *"`git diff master --stat -- ggml/` is empty: no new
ggml op, and no change to any existing one"* — validated at ppl 4.0068 vs 4.0126
reference, 98.0% top-1. A GGUF-lineage engine expressed this whole model in
existing primitives.

And a free correctness oracle: **QSA is bit-identical to dense attention below
`indexer_top_k + compress_ratio − 1` = 2051 cached tokens.** Ship dense first,
validate everything else against it, write the top-k last. That is the same
incremental-with-an-oracle structure that made R5–R8 tractable.

## 5. What is genuinely new

1. **Recurrent linear-attention kernels** (GDN / KDA). Zero prior art in ds4 —
   an exhaustive sweep for `mamba|ssm|delta_rule|gated_delta|linear_attn|
   conv1d|short_conv|A_log|dt_bias|recurrent` returns nothing. Needed per
   backend; Metal-only is a defensible first scope.
2. **Unbounded recurrent state semantics.** See §6 — this is the gate.
3. **A third parallel `*_graph_*` implementation.** ~12–20k lines, no seam to
   inherit from.
4. **Conversion tooling**, per §3.
5. **The frontend family matrix** — ~35 hardcoded two-way branches across
   `ds4_server.c` / `ds4_agent.c` / `ds4_cli.c` / `ds4_distributed.c`, with
   **three independent family enums**, hand-written tool parsers and two
   hardcoded pre-tokenizers. Individually trivial, collectively a long tail —
   and demonstrably where things get missed (two latent GLM defects found at
   `ds4_distributed.c:2861` and `ds4_server.c:11050`).

## 6. The gate: unbounded recurrent state vs ds4's session model

ds4 assumes state is **position-indexed** (truncate by integer) or **replayable
from a bounded tail**. A DeltaNet/SSM state is neither. Five paths depend on it:

| Path | Anchor | What breaks |
|---|---|---|
| chunked prefill | `metal_graph_refresh_ratio4_compressor_state` `ds4.c:28450` | rebuilds the frontier from the last 4 tokens at every chunk boundary. O(1) for DSA; **impossible** for a recurrent state. |
| rewind | `ds4_session_rewind` `ds4.c:70505` | is essentially `checkpoint.len = pos`. |
| speculative reject | `ds4.c:37010` | rollback needs the pre-draft state. |
| exact replay | `ds4_session_greedy_splitkv_replay_exact` `ds4.c:54582` | re-executes tokens out of order. |
| TP lockstep | `ds4.c:12231` | state must advance **bit-identically** on both ranks. |

Each becomes either a full-prefix replay or a mandatory O(state) copy. State is
**108 MiB/sequence** (Qwen, fp32) or **142.6 MiB/sequence** (GLM) — copyable, but
not free, and it bounds concurrency long before KV does (crossover ~2,400 tokens).

Two mitigations already in the codebase: `spec_frontier_*` is the right template
for snapshot/restore, and multi-slot batching is friendly (each slot owns a fully
separate session/graph, `ds4_server.c:8484`). **PP is friendly too** — a
recurrent layer's state lives wholly on one rank, and PP needs only two numbers
from a new family versus seven workstreams for TP.

### Live defect found while checking this — fix before anything else

`ds4_session_rewind()` (`ds4.c:70505-70544`) rewinds `checkpoint.len`, the DSpark
cache/history and the MTP draft flag. It does **not** touch `layer_n_comp[]`,
`layer_n_index_comp[]`, or the compressor frontier state.

Verified control flow: the resumed prefill takes `else if (ok && ratio != 0)` →
`if (zero_prefix)`. After a rewind, `pos0 = checkpoint.len ≠ 0`, so the
**absolute** recompute `g->layer_n_comp[il] = n_comp` (`ds4.c:29838`) is skipped
and the **incremental** branch runs instead, reading `comp_before =
g->layer_n_comp[il]` — the stale pre-rewind value — and writing
`comp_before + comp_chunk` (`ds4.c:29951`). Compressed rows are appended past the
correct frontier and attention reads across the gap.

Reachable from both rewind paths this branch added: `request_cancel_rollback()`
and `live_prefix_rewind_target()`. DeepSeek-only (GLM has ratio 0 everywhere).
The dedicated restore paths that *do* reset these counters (`ds4.c:53067`,
`:53194`, `:53234`, `:54261`) are the fix template.

This is very likely the residual of the compaction corruption seen earlier on
this branch, which was attributed to the SWA ring and fixed only with a rewind
*budget* — the budget bounds the raw ring and does nothing for `layer_n_comp`.
It is also the hybrid-state hazard in miniature: exactly the bug class that
recurrent layers would make pervasive.

## 7. Plans

### Phase 0 — de-risk the gate (do this first, ~2–3 weeks, no model)

1. ~~Fix the rewind defect in §6~~ — **done.** `ds4_session_rewind()` now snaps
   down to `lcm(non-zero ratios)` (128 on both DeepSeek variants, 1 on GLM) and
   rebuilds `layer_n_comp[]` / `layer_n_index_comp[]` as `pos / ratio` with the
   frontiers reset to empty, which is exactly the state at such a boundary.
   Costs at most 127 tokens of re-prefill. Unit-tested in
   `tests/test_engine_mgpu_placement.c::test_compressor_rewind_alignment`.
   **Still owed: the on-rig end-to-end check** — rewind across a compressed-row
   boundary, resume, and compare logits against a cold prefill of the same
   prefix. The unit test pins the arithmetic, not the cache contents.
2. Prototype recurrent-state lifecycle against the **existing** bounded state:
   extend `spec_frontier_*` to snapshot/restore `layer_n_comp` and the
   compressor frontier, and make `ds4_session_rewind` use it.
3. Decide the chunk-boundary policy for an unbounded state: carry it forward
   across chunks (breaking the deliberate replay-for-determinism at
   `ds4.c:28466`) or snapshot per chunk.
4. Confirm TP bit-identity for a recurrent update under the gate schedule.

**Gate:** if (2)–(4) do not come out clean, neither port is viable and we stop
here having fixed a real bug.

### Phase 1 — GLM-5.3-Flash (recommended first)

Chosen over Qwen because the **entire frontend matrix is already done** — GLM
tokenizer, chat template, tool format, reasoning-effort handling, server/agent
syntax enums — and because mHC, MLA and the DSA indexer all exist.

| Step | Work | Notes |
|---|---|---|
| 1 | Shape entry + `glm5-next` arch detector + validators | ~200 lines, the easy part |
| 2 | Quantise BF16 → MXFP4-expert GGUF | needs a converter or a donor template; widen the stale TP quant gate |
| 3 | 11 DSA layers | existing GLM DSA **minus RoPE**, **plus** IndexPool 4:1 + `k_norm` LayerNorm-with-bias |
| 4 | mHC | reuse `dsv4_hc.metal`; delta is the unweighted final mean |
| 5 | Clamped SwiGLU (limit 10.0) | trivial, but silent quality drift if missed |
| 6 | **KDA recurrent kernel** | the real work; prefill chunked + decode fused-recurrent |
| 7 | Graph, session, TP gate schedule, tests | the §1 bulk |

Traps, all from the reference implementation: L2-norm is `x/sqrt(Σx²+1e-6)` not
`F.normalize`; the forget gate takes the `gate_lower_bound = -5.0` branch;
`A_log`/`dt_bias` stay F32; `o_norm` is a **sigmoid-gated** RMSNorm in strict
fp32; the checkpoint stores three separate conv1d tensors where HF fuses one.

### Phase 2 — Qwen3.8-Flash-Next

Cheaper *after* Phase 1, because the recurrent substrate and the pooled indexer
are shared. Order, following llama.cpp's own sequencing:

1. MoE + GQA trunk on the 4-wide residual (**no final `output_norm`**).
2. GDN — port the two reference bodies verbatim; sigmoid output gate, not SiLU.
3. **QSA as dense attention** — correct model immediately, free oracle < 2051.
4. The indexer — ReLU **per head before** summing is the documented trap
   (jaccard 0.794 → 0.975 when fixed).
5. PLE last — host-side 64-bit hash (multipliers reach 2⁴⁵), dilated depthwise
   conv (dilation 3, state length 9).
6. MTP dense, if at all — llama.cpp measured no throughput gain from QSA there.

New-to-ds4 beyond the shared substrate: Qwen2 BPE pre-tokenizer, the XML-ish
`<function=…><parameter=…>` tool format, interleaved mRoPE with partial rotary,
Gated Residual (4-wide plumbing reusable, but a *different* formulation from
mHC — elementwise gated read, per-branch scalar write, no Sinkhorn), and the
n-gram table. Note llama.cpp measured **mmap making PLE load 10× slower**
(22 s → 224 s); ds4 is mmap-throughout, so verify — it may invert for us, since
we want lazy faulting on a table where each token reads 16 rows.

## 8. Honest scoping

- Phase 0 alone: worth doing regardless — it fixes a live correctness bug.
- Phase 1, Metal-only, third-party quants: **on the order of GLM-5.2's ~52k
  lines**, less the frontend and mHC, plus the KDA kernel and the recurrent
  session work. Multi-month.
- Phase 2 after Phase 1: materially cheaper, but not small.
- All three backends + own conversion tooling: a different scale of commitment
  again.

The cheapest structural de-risking would be a per-family ops seam *before* the
second port, so family #4 does not cost what family #3 did. There is no seam
today, and `AGENT.md` says that is deliberate — worth an explicit decision
rather than drift.
