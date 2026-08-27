# Scoping: `routed_moe_folded` and `attn_inv_rope` (34% of the decode token)

Status: **COMPLETE** — 2026-08-27. Scoping/analysis only; no source file was
modified. Written incrementally against transient-failure risk.

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
lanes x 2 flop x 1.398 GHz = 21.5). A third roof turns out to matter for both
stages and is used in §3: **instruction issue**, ~335 G SIMD-instruction
issues/s (60 cores x 4 SIMD pipes x 1.398 GHz, 1 instruction/cycle/pipe).
That figure is derived from the published core geometry, not measured here —
treat it as a bound, not a datum.

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
(`ds4_metal.m:39541-39608`). On the folded TP path `add_in != NULL` and
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

**What the gathered path actually dispatches** (`ds4_gpu_attention_decode_heads_tensor`,
`ds4_metal.m:30494` -> `ds4_gpu_encode_flash_attention_gathered_heads`,
`ds4_metal.m:28965`), per layer:

| # | kernel | grid (threadgroups) | threads/tg |
|---|---|---|---|
| 1 | `ds4_gpu_encode_flash_kv_stage_f16` — gathers the F32 raw ring + F16 compressed cache into one contiguous F16 KV buffer (`ds4_metal.m:29172`) | shape-dependent | — |
| 2 | `kernel_flash_attn_ext_vec_f16_dk512_dv512` (`metal/flash_attn.metal:970/1391`), dispatched at `ds4_metal.m:29307` | **(1, n_head=32, nwg)** | **(32, nsg=1)** |
| 3 | `kernel_flash_attn_ext_..._reduce` (+ fused inverse RoPE), dispatched at `ds4_metal.m:29334` | **(32, 1, 1)** — always exactly `n_head` | (32·nwg, 1) |

`nsg = ds4_gpu_flash_attn_vec_nsg(n_keys, 32, 32)` (`ds4_metal.m:3730`) is **1**
for every n_keys < 2048. `nwg = ds4_gpu_flash_attn_decode_nwg` (`ds4_metal.m:3705`)
under TP-2 picks the smallest bucket in {4,5,12,16,24,32} that covers
`ceil(n_keys/32)` chunks. So each vec threadgroup is **one simdgroup that
processes exactly one 32-key chunk** and then exits.

Layer taxonomy at ctx 2k (`raw_window = DS4_N_SWA = 128`, `ds4.c:17163`;
ratios from `ds4.c:1120-1124`):

| class | layers | n_raw | n_comp | n_keys | nwg | vec threadgroups |
|---|---:|---:|---:|---:|---:|---:|
| ratio 0 (raw path) | 2 | 128 | 0 | 128 | 4 | 128 |
| ratio 128 | 20 | 128 | ~16 | ~144 | 5 | **160** |
| ratio 4 | 21 | 128 | ~512 | ~640 | 24 | 768 |

---

## 2. Bytes and FLOPs per token per rank

### 2.1 `routed_moe_folded`

**Weights.** Per owned expert per layer, gate + up + down, MXFP4:
`gate_row_bytes = 2176` = 4096 x 0.53125, x 2048 rows = 4,456,448 B per matrix,
x 3 = **13,369,344 B = 13.369 MB** — exactly §3's per-expert figure, and the
`down_expert_bytes == 4456448` literal in `ds4_metal.m:39576` confirms it from
the other side.

Under contiguous 128/128 expert sharding, a rank owns `k ~ Binomial(6, 1/2)`
of the 6 selected experts, `E[k] = 3.0`. **Per-rank weight bytes = 3.0 x
13.369 MB x 43 layers = 1.7246 GB/token.** This is exactly §4's "1.725 GB"
figure for the 3.0-expert case, so it reconciles. (§4's 2.264 GB uses
`E[max(k,6-k)] = 3.9375` — that is the *critical-path* number for the pair of
ranks and is the right one for the straggler analysis; it is the **wrong** one
for a single rank's own kernel time, which is what a stage timestamp measures.
Using 2.264 here would overstate the rate by 31%.)

Activations are trivial in DRAM terms (x 16 KB, mid 48 KB, out 16 KB per
layer). **Total DRAM ~= 1.735 GB/token/rank.**

**FLOPs.** 3.0 experts x 3 matrices x 4096 x 2048 x 2 flop x 43 layers =
**6.49 GFLOP/token/rank**.

### 2.2 `attn_inv_rope`

§3's 60.17 MB/layer models *projection weights*; §3 says explicitly it is not
an attention-core traffic invariant, so the anchor here is the addendum's
partial-buffer accounting instead.

**Chunk count.** 32 heads x `ceil(n_keys/32)`:
21 x 32 x 20 + 20 x 32 x 5 + 2 x 32 x 4 = **16,896 32-key chunks/token/rank**.
The same formula at ctx 512 (nwg 12/5/4) gives 11,520, which is the addendum's
independently derived "11,520/token". **Reconciled.**

**KV loads.** K and V bind the *same* buffer (`ds4_metal.m:29300-29301`) because
MLA stores one 512-wide row; the kernel reads it once for QK
(`metal/flash_attn.metal:1116-1128`) and again for PV (`1216-1225`). So per
chunk: 32 keys x 512 x 2 B x 2 = 65,536 B.
**16,896 x 64 KB = 1.107 GB of KV loads/token/rank.**

**Partial buffer.** `tmp_bytes = n_head x head_dim x nwg x 4 + ...`
(`ds4_metal.m:29046`). Written by vec, read by reduce:
2 x (21 x 32·512·24·4 + 20 x 32·512·5·4 + 2 x 32·512·4·4) = **80.2 MB/token**.
The same formula at ctx 512 gives 47.2 MB, against the addendum's "about
45.2 MiB/token". **Reconciled.**

**KV staging** ~39 MB; **Q re-reads** (each of the 32·nwg threadgroups loads the
full 2 KB Q row, `metal/flash_attn.metal:1030-1037`) ~40 MB.

**DRAM-resident working set is tiny.** The staged KV is `n_keys x 1024` B =
655 KB (ratio-4) / 147 KB (ratio-128) per layer, and the 1.107 GB of vec loads
is that buffer re-read **64 times** (32 heads x K-then-V). Dispatch order
(`x` fastest, then head, then iwg) means the 32 heads reading chunk *c* are
adjacent in the grid, so the reuse is cache-friendly by construction.
**True DRAM traffic is ~17 MB of KV plus at most 80 MB of partials
= 20-100 MB/token.**

**FLOPs.** 16,896 chunks x 32 keys x 512 dims x 4 flop (QK fma + PV fma) =
**1.107 GFLOP/token/rank**.

---

## 3. Achieved rate against the roofs

### 3.1 `routed_moe_folded` — bandwidth-classified, at ~half the roof

| quantity | value | vs roof |
|---|---:|---|
| DRAM bytes | 1.7246 GB | — |
| net stage time @2k | 4.72 ms | — |
| **achieved streaming rate** | **365 GB/s** | **48% of 760 GB/s** |
| | | **51% of the 713 GB/s stride-17-adjusted roof (U8)** |
| FLOPs | 6.49 GFLOP | — |
| **achieved FLOP rate** | **1.38 TFLOP/s** | **6.4% of ~21 TFLOP/s** |

Context-invariant: 131k reported 4.95 -> net 4.77 -> **361 GB/s**; 32k -> 358
GB/s. The three agree to 2%.

**Classification: bandwidth-side, and it is the clearer of the two stages.**
Note this is the *live* rate, computed from the stage timestamp and the
reconciled byte model. It is lower than the ~410 GB/s the plan has been
quoting, because 410 came from the **isolated** `bench_moe_mxfp4_decode`
harness (U1), which runs world-1 with `add_in == NULL` and therefore a
different kernel selection (see §1.1). The live folded TP kernel is ~11%
slower than the isolated one, and the isolated one is itself at 54%.

U8 already settled 5.9% of the shortfall as 17-byte stride. **That leaves a
1.95x factor between 365 GB/s and the 713 GB/s a pure streaming kernel gets at
the same stride.** Since it is not stride and it is not FLOPs, it is
per-byte *work* inside the kernel. Two structural facts about that work:

**(a) The 4-bit payload is read one byte at a time.** `block_mxfp4` is
`{uchar e; uchar qs[16]}` (`metal/moe.metal:412`) — 17 bytes, **alignment 1**,
with `qs` at offset 1. The kernels take `device const uchar *qg = bg.qs + 8*it`
and then dereference `qg[0] ... qg[7]` individually
(`metal/moe.metal:4501-4519` for gate/up, `6261-6271` for down). At alignment 1
the compiler cannot legally widen these into a vector load, so each 8-byte
half-block costs **8 scalar byte loads**, plus one more for `bg.e`.

**(b) Dequant is 16 threadgroup-memory lookups per 8 bytes.** Each nibble is
converted through `lut[q[i] & 15]` / `lut[q[i] >> 4]`
(`metal/moe.metal:4504-4517`), a 32-float threadgroup array
(`ds4_gpu_routed_mv_smem` returns `32 * sizeof(float)` = 128 B,
`ds4_metal.m:31159-31161`) with a data-dependent index.

Per 17-byte block the kernel therefore issues **18 device byte loads + 32
threadgroup loads = 50 memory instructions to consume 17 bytes**, i.e. ~2.9
memory instructions per weight byte. The Q8_0 dense matvec that measures
541-581 GB/s on the same rig (`tp_decode_investigation.md` §5) issues, per 34-byte block, 4 x 9 = 36 device
loads plus 32/`N_R0_Q8_0` = 16 amortised activation loads = 52 for 34 bytes,
i.e. ~1.5 per byte, **and no threadgroup lookup at all**. The ratio of
memory-instructions-per-byte (1.9x) is the same order as the ratio of byte
rates (541/365 = 1.48x).

*Marked as inference:* static instruction counting cannot settle the exact
apportionment between byte-load count, LUT lookups and cache-line gather
divergence — Apple does not publish per-class issue rates and the compiler's
codegen is not inspected here. What it does establish is that the kernel does
**~2x the memory-instruction work per byte** of a kernel that hits 73% of the
roof at the same k, and that both of the excess terms are removable. §4 gives
the falsification tests.

**Three mechanisms are ruled out here:**

1. **The k-curve does not apply.** The brief asks whether the MoE sits at a bad
   point on the k-curve. It does not: gate/up run at **k=4096** and down at
   **k=2048**, which are the rig's *two best* Q8_0 points (541 and 581 GB/s of
   the 283/413/581/541/517 curve in `tp_decode_investigation.md` §5). Unlike `shared_down` (k-sliced from
   2048 to 1024), the MoE has no bad geometry. **This is a clean negative.**
2. **Underfill does not apply.** The pair grid is 6144 threadgroups and the
   sum6 grid 2048, on 60 cores. Threadgroup memory is 128 B, so residency is
   not capped.
3. **Activation re-reads are second-order.** Every pair threadgroup reads the
   whole 16 KB `x` and every sum6 threadgroup reads 3 x 8 KB of `mid`, so
   activation *bytes* are ~2.5x the weight bytes per layer (100 MB vs 40 MB).
   But in *instructions* they are 4 vector loads against 36 byte loads per
   thread-iteration — ~10%. Raising `nr0` amortises them, and that was already
   tried: `a12e73d` measured `nr0=4` on gate/up as a **regression** (35.13 t/s)
   and `nr0=4` on down as a win (36.41 -> 36.78), which is why only the down
   kernel has an r4 twin (`ds4_metal.m:39290-39307`).

### 3.2 `attn_inv_rope` — near **neither** roof, by 1.5 orders of magnitude

| quantity | value | vs roof |
|---|---:|---|
| DRAM bytes | 20-100 MB | — |
| net stage time @2k | 3.63 ms | — |
| **achieved DRAM rate** | **6-28 GB/s** | **0.8-3.7% of 760 GB/s** |
| aggregate load traffic (mostly cache) | 1.27 GB | 350 GB/s |
| FLOPs | 1.107 GFLOP | — |
| **achieved FLOP rate** | **305 GFLOP/s** | **1.4% of ~21 TFLOP/s** |

**Classification: neither. It is latency / occupancy / launch bound.** A pure
instruction-count lower bound is ~1,900 lane-ops per chunk x 16,896 chunks /
(240 SIMD pipes x 1.398 GHz) ~= **0.10 ms**; the stage takes 3.63 ms, i.e.
**~35x its own instruction-issue floor**. Nothing in the byte or FLOP model
comes within an order of magnitude of explaining the time.

That classification is not a dead end — it names the lever. Four specific
sources of lost parallelism, all visible in the code:

**(i) The reduce is hard-wired to exactly 32 threadgroups.**
`dispatchThreadgroups:MTLSizeMake(nrows,1,1)` with `nrows = n_head`
(`ds4_metal.m:29334`). Under TP-2, `n_head = 32` local heads on a 60-core GPU:
**28 cores are idle in the reduce of every layer, unconditionally.** This is
the sixth underfill sighting and the only *structural* one — the other five
(packed32, T2's 112-threadgroup turnover, the indexer LLT smem cap, the
head-norm/RoPE at 32, `q_lora_norm` at 2) are shape-dependent.

**(ii) The vec kernel is one simdgroup per threadgroup and capped at ~9
resident per core.** `threadsPerThreadgroup = (32, nsg=1)` and
`shared_bytes = align16((PAD2(512,128) + 4·32 + 2·PAD2(512,128)) · nsg · 2)`
= **3,328 B** (`ds4_metal.m:29268-29271`). Against Apple's 32 KB per-core
threadgroup memory that is at most **9 threadgroups = 9 simdgroups per core =
2.25 per SIMD pipe** — too few to hide L2 latency behind only 4 independent
loads per inner iteration. *(Inference: the 32 KB per-core figure is the
documented Apple limit, not measured here.)*

**(iii) 22 of 43 layers do not even reach that cap.** The ratio-128 and raw
layers dispatch **160 and 128** vec threadgroups on 60 cores — 2.1-2.7 per
core, i.e. ~0.6 simdgroups per SIMD pipe. At 2k those 22 layers carry only 20%
of the chunks but they run at the worst occupancy in the decode.

**(iv) KV is re-read 64x.** 32 heads x (K for QK, V for PV) over the same
655 KB staged buffer. `packed32` exists precisely to batch heads, and it was
reverted (`fe4674f`) because it *also* collapses split-K, dropping the grid to
32 threadgroups at 32 heads. The two knobs have never been varied
independently.

---

## 4. Ranked candidates

End-to-end share is against the **24.34 ms @2k token**. "Prize" is ms saved on
the token, not on the kernel.

### Rank 1 — M3: planar MXFP4 repack (scale plane + 16-byte-aligned qs plane)

- **Mechanism.** Removes both excess terms at once: `qs` becomes 16-byte
  aligned so a thread's half-block is **one `uint2`/`uint4` load instead of 8
  byte loads**, the scale stream becomes contiguous and vector-loadable, and
  the effective stride becomes 16, which U8 measured at **758 GB/s ~= roof**.
- **Prize.** 4.72 ms -> 2.42 ms at 713 GB/s, or 2.27 ms at 758. **~2.3-2.45 ms
  = 9.5-10.1% of the token, ~+4.3-4.6 t/s.** Largest item in this document.
- **Cost.** High: a GGUF conversion tool and a repacked model file. **But it is
  the same tool §7 option C needs for the routed-expert rebalance**, and that
  item is independently worth 1.47 ms. Scope them as one deliverable; together
  they are ~3.8 ms / 15% of the token.
- **Falsified by.** Build a `bench_membw`-style arm that streams a planar
  layout and dequantises with vector loads. If it does not exceed ~600 GB/s the
  premise is wrong and the cost is unjustifiable. Cheap, model-free, and it can
  run before any conversion tool is written.
- **Correctness.** Bit-identical if the value order and accumulate order are
  preserved; the repack is a pure permutation of bytes on disk.

### Rank 2 — M1+M2: two in-place, bit-identical kernel edits

- **M2 (payload).** Replace `qg[0]..qg[7]` with two `packed_uchar4` loads.
  Metal's `packed_uchar4` has scalar alignment, so it is legal at offset 1;
  the idiom is already used in this codebase (`metal/cpy.metal:67,171`).
  Sites: `metal/moe.metal:4501-4519` (pair gate/up), `6261-6271`
  (`ds4_mxfp4_accumulate_rows`). All five sibling bodies must move together or
  they will diverge: `metal/moe.metal:4501, 4682, 6261, 6312, 6358, 6599`
  (pair, pair-static, sum6, sum6-r4, sum6-full-rows, slots6) plus the dense
  MXFP4 matvec at `2905`.
- **M1 (dequant).** Replace the 16-lookup nibble LUT with an 8-lookup
  byte-indexed `float2` LUT (256 entries, 2 KB of threadgroup memory against
  today's 128 B at `ds4_metal.m:31159`). Halves the threadgroup-load count.
- **Prize.** If the two together recover half the 1.95x residual: 4.72 -> 3.45
  ms, **1.27 ms = 5.2% of the token, +2.2 t/s.** A 10% kernel gain alone is
  0.47 ms = 1.9%.
- **Cost.** Low — ~30 lines, no model change, no host change.
- **Falsified by.** `tests/bench_moe_mxfp4_decode` on the rig, three arms
  (baseline / M2 / M2+M1). If M2 alone moves under 3% the byte-load hypothesis
  is dead and only M1 is worth landing. Must be run **on the rig** — the M1 Max
  dev box peaks at 400 GB/s and cannot discriminate (U1's own note).
- **Risk to watch.** M1 raises per-threadgroup shared memory 16x. At 2 KB, a
  32 KB per-core budget still allows 16 resident threadgroups, so it should not
  bind — but this is the U7 smem-cap family and the arm must report occupancy,
  not just GB/s.

### Rank 3 — A0: decompose `attn_inv_rope` (prerequisite, not a win)

- **Mechanism.** `ds4_gpu_flash_attn_stage_profile_boundary`
  (`ds4_metal.m:10883`) already exists and is wired into all four **prefill**
  encoders (`ds4_metal.m:27935, 28218, 28567, 28821`). The **decode** gathered,
  raw and indexed encoders have none. Add three boundaries inside
  `ds4_gpu_encode_flash_attention_gathered_heads` (`ds4_metal.m:28965`) —
  after KV staging, after vec, after reduce.
- **Why it is ranked above the attention patches.** 14.9% of the token is one
  undifferentiated number. Ranks 4-6 below have prize ranges spanning 0 to 8%
  of the token *because* of that. §14.6 and the epoch rule forbid getting the
  split by subtracting across runs; this is the instrumentation that produces
  it in one run.
- **Cost.** Low, and inert unless `DS4_METAL_FLASH_ATTN_STAGE_PROFILE` is set.

### Rank 4 — A2: head-batched vec that keeps split-K

- **Mechanism.** Today grid `(1, 32 heads, nwg)` with 1 simdgroup each, and the
  same 32-key chunk is loaded once per head. Batch H heads per threadgroup
  while keeping the `nwg` split-K axis: H=4, nwg=24 gives **8 x 24 = 192
  threadgroups of 4 simdgroups = 768 simdgroups — identical simdgroup count to
  today — at 1/4 the KV loads and 1/4 the Q loads.** This is the axis
  `packed32` conflated: it batched heads *and* removed split-K, and lost on the
  second.
- **Prize.** 0-2 ms (0-8%). Wide, and deliberately so — see Rank 3. If the vec
  kernel is load-issue-bound it is near the top of that range; if it is pure
  latency it is near the bottom.
- **Cost.** Medium. `metal/flash_attn.metal`'s vec kernel already has an `iq2`
  head index and a `sgitg` simdgroup index; batching heads means mapping
  `sgitg -> head` and giving each simdgroup its own `sq4`/`so4`/`ss` slice,
  which raises shared memory from 3,328 B to H x 3,328 B. **At H=4 that is
  13,312 B, cutting per-core residency from 9 threadgroups to 2** — the same
  trap as packed32. The design must instead share the *KV* across simdgroups
  (that is the point) and keep only Q/O per-head, which is a genuine rewrite,
  not a parameter change.
- **Falsified by.** `speed-bench/metal_flash_attn_decode_bench` already runs
  the production 32-head x 512 shape model-free and has a `--correctness-only`
  mode. Add an H sweep there before touching the engine.
- **Gate.** Do not start before Rank 3 reports.

### Rank 5 — A1: split the flash reduce over the DV axis

- **Mechanism.** `MTLSizeMake(nrows,1,1)` -> `MTLSizeMake(nrows, DV/128, 1)`
  (`ds4_metal.m:29334`), so 32 -> **128 threadgroups** and all 60 cores get
  work. Each threadgroup reduces `nwg` partials for 128 of the 512 output dims;
  the softmax max/sum tree over `nwg` lanes is 2·nwg floats and is simply
  recomputed per dim-chunk.
- **Prize.** 80.2 MB of partial traffic currently confined to 53% of the
  machine. **~0.1-0.3 ms = 0.4-1.2% of the token.** Small but cheap and certain
  in direction.
- **Cost.** Low, and the fused-RoPE variant splits cleanly.
  `kernel_flash_attn_ext_vec_reduce_rope` (`metal/dsv4_rope.metal:594`) does the
  full 512-dim reduce, then a device barrier, then rotates only
  `[n_nope, head_dim)` = **dims 448..511** in *adjacent* pairs
  (`ds4_rope_tail_pair_affine_row`, `metal/dsv4_rope.metal:435`, pairing at
  `468-469`). With
  128-wide dim chunks the entire rotated range lives in the last chunk, so
  chunks 0-2 skip the tail entirely and chunk 3 does it locally — no
  cross-threadgroup dependency and no extra barrier. **Safe at 128; would break
  at a chunk boundary that lands inside 448..511.**
- **Falsified by.** Rank 3's decomposition: if the reduce is under ~0.3 ms of
  the 3.63, drop this.

### Rank 6 — A4: fuse KV staging into the vec kernel

- **Mechanism.** Removes 43 dispatches/token and 39 MB of staging round-trip.
- **Prize.** 43 x 1.9-4.4 us = **0.08-0.19 ms = 0.3-0.8%.**
- **Cost.** Medium; §6 of the investigation record already concludes dispatch
  removal is not a productive strategy at this scale. Listed for completeness,
  not recommended.

### Not recommended

- **Re-tuning `nwg` buckets.** The compact schedule already removed ~74% of the
  empty grid at ctx 512, and at 2k only 4 of 24 ratio-4 workgroups are empty
  (17%). The remaining headroom is under 0.1 ms.
- **`nr0` widening on the MoE gate/up.** Measured regression (`a12e73d`).
- **Anything in the five MXFP4 specialisations.** Dead on measurement (T8) and
  unreachable in the TP-fold configuration anyway (§1.1).

---

## 5. Summary table

| item | mechanism | prize (ms) | share of token | cost | falsifier |
|---|---|---:|---:|---|---|
| M3 planar MXFP4 repack | aligned vector payload + 16 B stride | 2.30-2.45 | **9.5-10.1%** | high (shared with §7-C) | `bench_membw` planar arm < 600 GB/s |
| M1+M2 kernel edits | halve LUT loads, vectorise payload | 0.47-1.27 | **1.9-5.2%** | low | `bench_moe_mxfp4_decode` 3-arm on the rig |
| A0 decode FA boundaries | instrumentation | 0 | — | low | n/a — prerequisite |
| A2 head-batched split-K vec | 4x fewer KV/Q loads at equal simdgroups | 0-2.0 | **0-8.2%** | medium-high | `metal_flash_attn_decode_bench` H sweep |
| A1 reduce DV split | 32 -> 128 threadgroups | 0.1-0.3 | 0.4-1.2% | low | A0 says reduce < 0.3 ms |
| A4 fuse KV staging | -43 dispatches | 0.08-0.19 | 0.3-0.8% | medium | — |

**Both stages have real headroom, but of completely different kinds.**
`routed_moe_folded` is a clean bandwidth story at 48% of roof with two named,
removable per-byte costs and a large repack option behind them.
`attn_inv_rope` is 35x off its own instruction floor and is not a roofline
problem at all — it is a parallelism problem, and it cannot be attacked
responsibly until the 3.63 ms is split three ways, which is a one-run change to
an instrument that already exists for prefill.

---

## 6. Negatives established here (do not re-derive)

- **The k-curve does not explain the MoE.** gate/up k=4096 and down k=2048 are
  the rig's two *best* Q8_0 points (541 / 581 GB/s). The `shared_down`
  mechanism does not transfer.
- **The MoE is not underfilled.** 6144 and 2048 threadgroups; 128 B of
  threadgroup memory; no residency cap.
- **The five MXFP4 decode specialisations cannot fire under TP-fold** — all are
  gated on `add_in == NULL` and `tp_world == 1` (`ds4_metal.m:39541-39608`),
  and the folded path always has both false.
- **~50% of the MoE pair grid exits on the first instruction** (non-owned
  experts, `metal/moe.metal:4572`). 132k trivial threadgroups/token, order
  30 us. This is the straggler item's shadow, not a separate cost; do not
  double-count it.
- **The live MoE rate is 365 GB/s, not 410.** 410 is the isolated world-1
  harness (U1), which selects different kernels. Quote 365 for the stage.
- **`attn_inv_rope` cannot be decomposed today.** The flash-attn stage profiler
  is wired only into prefill encoders.

