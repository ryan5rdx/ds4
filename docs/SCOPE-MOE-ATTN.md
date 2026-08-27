# Scoping: `routed_moe_folded` and `attn_inv_rope` (34% of the decode token)

Status: **IN PROGRESS** (written incrementally; sections are filled as they are concluded).

Rig: 2 x Mac Studio M2 Ultra 60-core, DeepSeek V4 Flash MXFP4, TP world 2 over
Thunderbolt RDMA. Branch `upstream-metal-wins`. All stage times quoted **net of
the ~0.18 ms/marker profiler tax**.

| stage | net ms @2k | % of 2k token |
|---|---:|---:|
| `routed_moe_folded` | 4.72 | 19.4% |
| `attn_inv_rope` | 3.63 | 14.9% |

---

## 0. Method, and what a "stage" physically is

**A stage boundary commits a command buffer.** `DS4_METAL_PROFILE_DECODE_STAGE`
(`ds4.c:22206`) calls `metal_graph_layer_stage_profile_boundary`
(`ds4.c:28954`), which under `DS4_METAL_GPU_STAGE_TIMESTAMPS=1` calls
`ds4_gpu_stage_flush` (`ds4_metal.m:10822`). That function *flushes the batch
command buffer in flight*, tags it, and retains it; `ds4_gpu_stage_report`
(`ds4_metal.m:10838`) later sums `GPUEndTime - GPUStartTime` per tag.

Three consequences that govern everything below:

1. A stage's number is **GPU busy time of command buffers that contain only
   that stage's dispatches** — not CPU wall, not a bubble, and not overlapped
   with any neighbour. It is a clean per-kernel measurement.
2. The ~0.18 ms/marker tax is **43 extra command-buffer commits** (one per
   layer) at roughly 4 us each. That is why it is a fixed ~0.18 ms per stage
   *row*, not per marker instance.
3. Because each stage runs in its own command buffer, the profiled run has
   **less inter-stage overlap** than production. The net figures are therefore
   a slight *over*-estimate of isolated kernel time and the right thing to
   roofline against.

**Byte model.** Every figure below is reconciled against
`speed-bench/tp_decode_investigation.md` §3 before a rate is quoted, per §14.6.
The two anchors used are:

- routed experts **13.369 MB per expert per layer** (gate+up+down, MXFP4)
- attention projections **60.17 MB/layer**, of which the attention *core* is
  explicitly **not** part (§3's own caveat)

**Roofs.** 760 GB/s measured streaming (U6), ~21 TFLOP/s FP32 (60 cores x 128
lanes x 2 flop x 1.398 GHz = 21.5). A third roof turns out to matter here and
is derived in §3.3: **instruction issue**, ~335 G SIMD-instruction issues/s
(60 cores x 4 SIMD pipes x 1.398 GHz).

---

## 1. What is actually in each span

### 1.1 `routed_moe_folded` — `ds4.c:25291` -> `ds4.c:25319`

The span is the narrowest in the whole decode layer. Between the `shared_down`
marker (`ds4.c:25291`) and the `routed_moe_folded` marker (`ds4.c:25319`) there
is one `metal_graph_debug_dump_tensor` (inert) and **exactly one call**:
`ds4_gpu_routed_moe_one_tensor(...)` at `ds4.c:25293-25317`, entered only under
`tp_fold_ffn`, with `add_in = metal_graph_shared_out(g)`.

Inside that call (`ds4_metal.m:39309`), on the TP-2 MXFP4 decode path, exactly
**two dispatches** are encoded per layer:

| # | encoder | kernel | grid | threads/tg |
|---|---|---|---|---|
| 1 | `ds4_gpu_encode_mul_mv_id_pair_swiglu` (`ds4_metal.m:31397`, called at `41537`) | `kernel_mul_mv_id_mxfp4_pair_swiglu_f32` (`metal/moe.metal:4553`) | **(1024, 1, 6)** | (32, 1) |
| 2 | `ds4_gpu_encode_mul_mv_id_sum6` (`ds4_metal.m:31788`, called at `41860`) | `kernel_mul_mv_id_mxfp4_sum6_f32` (`metal/moe.metal:6375`) or its `_r4` twin (`6430`) | **(2048, 1, 1)** or (1024,1,1) with r4 | (32, 1) |

Grid derivation: `row_groups = ne01 / (nr0 * nsg)`; `nr0 = N_R0_MXFP4 = 2`
(`metal/moe.metal:17`), `nsg = 1` because `ds4_gpu_mxfp4_moe_decode_nsg1_enabled`
(`ds4_metal.m:39284`) is true on pre-M5 Apple silicon. Gate/up `ne01 = 2048` ->
1024 row groups, `z = nei0*nei1 = 6`. Down `ne01 = 4096` -> 2048 row groups.

**So `routed_moe_folded` = 86 dispatches/token = the entire routed MoE, and
nothing else.** There is no gate wait, no shared expert, no HC, no router
inside it. This is the cleanest span in the layer and its number can be taken
at face value.

**Which specialisation runs (this matters, and contradicts an easy assumption).**
Every one of the five MXFP4 decode specialisations
(`tg_multiple`, `fixed_route_pair`, `fixed_route_sum6`, `sum6_full_rows`,
`static_trip`) is gated on `add_in == NULL` and `tp_world == 1`
(`ds4_metal.m:39515-39582`). On the folded TP path `add_in != NULL` and
`g_tp_split_world == 2`, so **none of them can ever fire here.** T8 killed the
ladder on measurement; the code says it was never reachable in this
configuration in the first place. The live kernels are the plain
`_nsg1` forms, plus the `_r4` down twin if
`ds4_gpu_mxfp4_moe_decode_down_r4_enabled` (`ds4_metal.m:39304`) holds.

### 1.2 `attn_inv_rope` — `ds4.c:23330` -> `ds4.c:23582`

Confirmed mis-named. The previous marker is `compressor_indexer`
(`ds4.c:23330`); the span therefore contains the whole of the attention
execution block at `ds4.c:23331-23581`. On the Metal TP-2 path (all
`cuda_tp_*` branches are the CUDA multi-GPU path and are inactive) the span is:

| # | call | site | when |
|---|---|---|---|
| 1 | `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` | `ds4.c:23508` | `indexed_attention` layers |
| 2 | `ds4_gpu_attention_decode_heads_tensor` | `ds4.c:23539` | all other layers |
| 3 | `ds4_gpu_rope_tail_tensor` (standalone inverse RoPE) | `ds4.c:23570` | unless the fuse was armed *and* consumed |

**At ctx 2k, branch 1 cannot fire.** `indexed_attention` requires
`comp_selected != NULL` (`ds4.c:23335`), which is only set at `ds4.c:23294`
inside a guard requiring `g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K`
(= 512) and `g->layer_n_comp[il] > decode_sparse_threshold`
(`ds4.c:23198-23202`). A ratio-4 layer at pos 2048 has ~512 compressed rows,
not more. So **at 2k every one of the 43 layers takes the gathered/raw
FlashAttention at `ds4.c:23539`**, and the standalone RoPE at `ds4.c:23570` is
skipped whenever the fuse is armed and consumed (`fuse_attn_inv_rope`,
`ds4.c:22093`, default on for pre-M5 Apple silicon).

That is why the stage is context-*invariant-ish* and dips at 32k: at 2k it is
43 gathered-attention dispatches over a short key range; at 131k it is 21
indexed + 22 gathered plus 21 unavoidable standalone RoPE dispatches (the
indexed branch never arms the fuse — see U13). **The 2k figure of 3.63 ms is
essentially 43 x gathered decode FlashAttention.**

---

(sections 2-5 follow)
