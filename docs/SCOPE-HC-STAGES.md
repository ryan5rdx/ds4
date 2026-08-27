# Scoping the four hyper-connection (HC) decode stages

Branch: `upstream-metal-wins` (HEAD `26ef09f`). **Analysis only — no source file changed.**

Status: **COMPLETE**.

---

## 0. Summary / verdict

**Short version: the 4.56 ms is real but it is not 4.56 ms of hyper-connection
work, and what *is* HC work is not near either roof. Expect ~0.6–1.0 ms
(2.5–4.1% of the 2k token, +1.1 to +1.8 t/s) from a full campaign here, not the
18.7% the headline number suggests. Do the three zero-code measurements first.**

Five findings, in order of how much they change the plan:

1. **`ffn_hc_post` is 75% FFN TP gate.** Its span
   (`ds4.c:25319`→`ds4.c:25377`) contains `ds4_gpu_tp_gate_encode(il,
   DS4_TP_GATE_FFN)`, whose release fence is a **1-thread GPU spin loop**
   (`metal/dsv4_misc.metal:7333`) that blocks the command buffer for the whole
   RDMA round trip. `attn_hc_post` runs the *identical* expand kernel at the
   *identical* grid with no gate. Same run, so the subtraction is legitimate:
   **1.812 − 0.461 = 1.351 ms = 43 gates × 31.4 µs**, matching the independently
   measured ~38 µs/gate. **The real HC kernel pool is 3.21 ms (13.2%), not
   4.56 ms (18.7%).** (Corollary: the ATTN gate is likewise inside
   `attn_output`, so that stage is mis-attributed too.)

2. **None of the four stages is near either roof — not within 10×.**
   `attn_hc_pre` runs at **4.4% of the 760 GB/s roof** and **0.18% of 21
   TFLOP/s**; `attn_hc_post` at **2.0%** and **0.07%**. The 786,432 B of HC mix
   weight a producer must move would take **1.03 µs** at the roof; the stage
   takes **26.9 µs per layer**. These are **occupancy/latency bound** — the
   underfill family — not bandwidth or ALU bound.

3. **The grids are tiny, but the obvious fix is already known not to work.**
   The HC-pre producer is **6 threadgroups × 512 threads on 60 cores** with
   672 B of threadgroup memory; the HC-post expand is **16 × 256** with none.
   However, `d81a28f` on branch `pr-778` (**not in this branch**) already
   implements the widening — 6 → 10 threadgroups, bit-identical — and measured
   only **+0.12% on M5 Max**. The diff shows why: the producer's **TG0 alone**
   carries 4 matvec rows *and* the entire 4096-wide collapse + RMS epilogue, and
   the spread does not touch it. Widening without unloading TG0 is not the win.

4. **The measurement itself is suspect and cheap to fix.** Under
   `DS4_METAL_GPU_STAGE_TIMESTAMPS` each stage becomes its own command buffer
   (~900/token instead of 3). U12's stage sum at 32k is **32.70 ms** against
   **28.91 ms** unprofiled on the same build `b99dfa3` — **+13%**, ≈4.2 µs per
   boundary, which is **~39% of `attn_hc_post`'s entire 10.7 µs/layer**.
   Separately, `DS4_TP_ABLATE=hcpre` once measured the two HC-pre producers at
   **0.85 ms end-to-end** where the stage profile attributes **2.276 ms** — a
   2.7× disagreement that has never been resolved. **Re-running that ablation is
   the cheapest and highest-value action on this page.**

5. **A well-supported "not much headroom".** Even a *perfect* HC implementation
   — every byte at 760 GB/s plus launch — is ~0.4 ms, so the absolute ceiling is
   2.8 ms (11.5%, +5.3 t/s), and nothing found here suggests it is reachable.
   The credible slice is **0.6–1.0 ms**. **HC alone cannot supply the 4.34 ms
   the test plan needs for 50 t/s at 2k.**

Top three actions, all zero-code, all single-run:
`DS4_TP_ABLATE=hcpre` · `DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE=1` ·
a three-line stage marker after `ds4.c:25335` to split the gate out of
`ffn_hc_post`. See §7.

---

## 1. Method and ground rules

### 1.1 The numbers being scoped and where they come from

| stage | 32k ms/tok | 131k ms/tok | source |
|---|---|---|---|
| `ffn_hc_post` | 1.834 | 1.812 | U12, `BENCHMARKS-TP-PP.md:259` |
| `attn_hc_pre` | 1.142 | 1.155 | `BENCHMARKS-TP-PP.md:260` |
| `ffn_hc_pre` | 1.119 | 1.121 | `BENCHMARKS-TP-PP.md:261` |
| `attn_hc_post` | 0.467 | 0.461 | `BENCHMARKS-TP-PP.md:265` |

Epoch: **U12**, `DS4_METAL_GPU_STAGE_TIMESTAMPS=1`, rig = mat worker / lanfear
coordinator, TP2 over RDMA, build `b99dfa3`, gen 128, `DS4_METAL_FAST_SYNC=1`
(the standard procedure, `speed-bench/tp_decode_investigation.md:497`).
Per-token = sum of stage GPU-busy over 128 decode tokens ÷ 128, over **all 43
layers**. So per-layer cost is the table value ÷ 43.

### 1.2 How the marker actually works — and what it does to the numbers

`DS4_METAL_PROFILE_DECODE_STAGE` (`ds4.c:22206`) calls
`metal_graph_layer_stage_profile_boundary` (`ds4.c:28946`). Under
`DS4_METAL_GPU_STAGE_TIMESTAMPS` that routes to `ds4_gpu_stage_flush`
(`ds4_metal.m:10822`), which **commits the in-flight command buffer, tags it,
and opens a new one**. `ds4_gpu_stage_report` (`ds4_metal.m:10839`) then sums
`GPUEndTime − GPUStartTime` per tag.

Three consequences that must be carried through every number below:

1. **Each stage is one whole command buffer.** Its reported time therefore
   includes that CB's launch/drain envelope, not just kernel execution. Counting
   the markers a TP2 decode layer actually reaches (`attn_hc_pre` … `ffn_hc_post`,
   `ds4.c:22417`-`25377`) gives 20 unconditional plus the compressor/indexer
   conditionals, so a decode token in this mode is **~900 command buffers**
   instead of the normal 3 (`tp_decode_investigation.md` §6: "3 command buffers
   per token at pos >= 128"). All 43 layers are profiled: with
   `DS4_METAL_GPU_STAGE_TIMESTAMPS=1` and no `_LAYER` filter,
   `metal_graph_profile_layer_value_match(NULL, il)` returns true for every
   layer (`ds4.c:28846`).
2. **The mode measurably inflates the token.** U12's stage sum at 32k is
   **32.70 ms** (`BENCHMARKS-TP-PP.md:262`-region total row). The unprofiled
   steady decode at 32k on the *same build* `b99dfa3` is **34.59 t/s = 28.91
   ms** (U11 table, `docs/TP-A0-ROWSPLIT-TEST-PLAN.md`, "32k TIGHT on"). The
   profile therefore attributes **+3.79 ms (+13%)** of token time that the real
   run does not spend — ≈ **4.2 µs per stage boundary**. Small stages carry a
   proportionally larger share of this fixed artifact.
3. **The `t0`/marker span is per-layer-invocation, not per-token.**
   `decode_stage_t0` is initialised at `ds4.c:22158`, at the top of
   `metal_graph_encode_decode_layer_phase`. So `attn_hc_pre`'s span begins at
   the start of the layer function, **not** at the previous layer's
   `ffn_hc_post`. This is the one place the "marker to marker" rule needs
   qualifying, and it works in our favour: the four spans are clean.

**These stage numbers are an upper bound on real cost, and the four HC stages
are all small, so they are the most inflated rows in the table.** Section 5
carries this as an explicit error bar; §6 makes it the top falsification test.

### 1.3 Configuration that the profile forces

Stage profiling is *itself* an admission gate for two large fusions:

- `parallel_ffn_route_eligible` requires `!decode_stage_profile` (`ds4.c:22180`).
- `decode_graphs_common_ok` requires `!decode_stage_profile` (`ds4.c:22230`).

Both are **already disabled under TP2 anyway** (`g->tp_world < 2` at
`ds4.c:22177` and `g->tp_world <= 1` at `ds4.c:22226`), so for the U12 rig
config the profile does not change which kernels run. That is worth knowing:
the U12 stage decomposition is faithful *in kernel selection*, and only
distorted in *timing*.

### 1.4 Roofs used

- **Streaming memory roof: 760 GB/s measured** (U6/U-membw, `BENCHMARKS-TP-PP.md:200`
  region — seven allocation arms all 750–790 GB/s on two hosts). Not 800, not 400.
- **FP32 ALU peak ~21 TFLOP/s** (60-core M2 Ultra @ 1.398 GHz).
- **TP2:** each rank holds half the heads and half the routed experts. The HC
  tensors are **not** split — see §2.

### 1.5 Byte-model reconciliation (mandatory per §14.6)

`tp_decode_investigation.md` §3's verified **60.17 MB/layer** is the *attention*
weight total (`attn_q_a` + `attn_q_b` + `attn_kv` + `attn_output_a` +
`attn_output_b`, per rank) — it does **not** include HC weights, router, shared
expert, or routed experts. So HC bytes are *additive to* 60.17, not a slice of
it, and the reconciliation test for this document is different: the HC weight
figures must match `tensor_expect_layout`. They do — see §2.1.

---

## 2. What the HC machinery actually is

### 2.1 Shapes and constants (read, not assumed)

`DS4_SHAPE_FLASH` (`ds4.c:581`): `n_embd 4096`, `n_layer 43`, **`n_hc 4`**,
**`n_hc_sinkhorn_iter 20`**, `hc_eps = DS4_DEFAULT_HC_EPS`.

Derived (`ds4.c:22109`-`22110`):

- `hc_dim = n_hc × n_embd = ` **16384**
- `mix_hc = 2·n_hc + n_hc² = ` **24**

Per-layer HC weights (`ds4.c:5132`-`5134`, `ds4.c:5164`):

| tensor | type | shape | bytes/layer |
|---|---|---|---|
| `hc_attn_fn` | F16 | [16384, 24] | **786,432** (768 KiB) |
| `hc_attn_scale` | F32 | [3] | 12 |
| `hc_attn_base` | F32 | [24] | 96 |
| `hc_ffn_fn` | F16 | [16384, 24] | **786,432** (768 KiB) |
| `hc_ffn_scale` / `hc_ffn_base` | F32 | [3] / [24] | 108 |
| **total** | | | **1,573,080 B ≈ 1.500 MiB/layer** |

Over 43 layers: **64.5 MiB = 0.0676 GB of HC weight per token**, replicated on
both ranks (no TP split anywhere in the HC path — every call site passes the
full `hc_*_fn` offset).

For scale: HC weights are **2.5% of the 60.17 MB/layer attention weight
total**. Any claim that an HC stage is bandwidth-bound has to survive that.

### 2.2 The four sites in the layer

Per decode layer the HC state is a 4×4096 F32 "residual bundle"
(`cur_hc` → `after_attn_hc` → `after_ffn_hc`, 64 KiB each). Around each block:

- **HC-pre** (before attention, before FFN): RMSNorm the flat 16384 row → F16
  matvec to the 24-wide mixer row → Sinkhorn-split it into `pre[4]`, `post[4]`,
  `comb[4×4]` → collapse the 4 streams to one 4096 row with `pre` → RMSNorm +
  weight for the block's input norm.
- **HC-post** (after attention, after FFN): expand the single 4096 block output
  back to 4 streams: `out[h] = block·post[h] + Σ_s comb[h][s]·residual[s]`.

The Sinkhorn normalisation (20 iterations, row-then-column, on a 4×4 matrix) is
computed **by a single thread** in every variant — `tid == 0` in
`kernel_dsv4_hc_split_weighted_sum` (`metal/dsv4_hc.metal:307`),
`kernel_dsv4_hc_split_weighted_sum_norm4` (`:495`), and one lane of one
threadgroup in the fused producer (`:1537` via `ds4_hc_comb_weights4_exact`,
`:400`).

---

## 3. Span contents — marker to marker

Config assumed throughout: **DS4 Flash, TP2 (`tp_world == 2`), decode
(`n_tokens == 1`), `phase == METAL_DECODE_LAYER_FULL`, pre-M5 Apple silicon
(M2 Ultra), no steering, no reference kernels, `tp_fold_ffn` on** (U12 reported
`routed_moe_folded`, not `routed_moe`, so the folded path was live).

Summary of the four spans (per layer):

| stage | dispatches | kernel(s) | grid (TG × threads) | tg-mem |
|---|---|---|---|---|
| `attn_hc_pre` | **1** | `kernel_dsv4_hc_rms_norm_mix_f16_cluster2_pre_norm` | **6 × 512** (32×16) | 672 B |
| `attn_hc_post` | **1** | `kernel_dsv4_hc_expand4` | **16 × 256** | 0 |
| `ffn_hc_pre` | **1** | `kernel_dsv4_hc_rms_norm_mix_f16_cluster2_pre_norm` | **6 × 512** | 672 B |
| `ffn_hc_post` | **3** | `kernel_dsv4_tp_flag_set_coherent`, `kernel_dsv4_tp_fence_wait`, `kernel_dsv4_hc_expand4` | **1×1, 1×1, 16×256** | 0 |

Every grid here is far below the 60-core machine. This is the underfill family.

### 3.1 `attn_hc_pre` — span `ds4.c:22158` → `ds4.c:22417`

The span begins where `decode_stage_t0` is set (`ds4.c:22158`), i.e. at the very
top of the layer encode. Everything before the marker that could dispatch:

- `ds4_gpu_decode_dispatch_ballast()` (`ds4.c:22155`) — no-op unless
  `DS4_METAL_DISPATCH_BALLAST` is set.
- Decode-graph island A (`ds4.c:22242`) — **not taken**: `decode_graphs_common_ok`
  requires `g->tp_world <= 1` (`ds4.c:22226`).
- `ds4.c:22311`-`22376`: the HC-pre producer.
- `ds4.c:22377`-`22407`: skipped, plus `metal_graph_check_hc_norm_fusion`, which
  returns immediately unless `metal_graph_hc_norm_fusion_check_enabled()`
  (`ds4.c:20102`).

Which producer runs: `fuse_norm_mix` is true (`hc_dim == 16384 && mix_hc == 24`,
`hc_attn_fn` is F16, device is pre-M5 — `ds4.c:22314`-`22321`); `fuse_hc_norm`
is true (`ds4.c:22285`); `fuse_producer_pre_norm` is true because
`metal_graph_ported_m5_decode_feature_enabled` returns `pre_m5 || m5`
(`ds4.c:22070`) and no disable env is set. So the whole span is the single
compound dispatch `ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor`
(`ds4_metal.m:43779`).

**Grid:** `dispatchThreadgroups:MTLSizeMake(6,1,1)` ×
`threadsPerThreadgroup:MTLSizeMake(32,16,1)` (`ds4_metal.m:43935`-`43936`) =
**6 threadgroups × 512 threads = 3072 threads on a 60-core GPU.**

**Threadgroup memory:** `(32 + 4·32 + 4 + 32)` floats = 168 × 4 = **672 B**
(`ds4_metal.m:43911`). Nowhere near an occupancy limit — the limiter is grid
size, not smem (unlike U7's indexer LLT scorer).

**What the kernel does** (`metal/dsv4_hc.metal:1324`):

1. *Phase A* (`:1360`-`:1376`): RMS reduction over the flat 16384-float HC row,
   emulating the original 1024-virtual-thread tree. **All 6 threadgroups compute
   this redundantly** — each reads the whole 64 KiB row.
2. *Phase B* (`:1380`-`:1436`): F16 matvec 16384 → 24. Each TG holds 2 clusters
   of 8 simdgroups; each cluster does `NR0 = 2` output rows. 6 TG × 2 clusters ×
   2 rows = 24 rows exactly. **Each cluster streams the whole x row again** for
   the matvec operand (`x4[...]*scale` at `:1398`), so x is read 12 more times.
3. *Fence + fan-out* (`:1442`): a `mem_device_and_threadgroup` barrier, then
   work is split by `tgpig.x`:
   - **TG0** (`:1447`-`:1509`): lane 0 computes `pre[4]`; all 512 threads collapse
     the 4 HC streams into the 4096 row (2 float4 per thread), reduce `dot(v,v)`,
     and write both `collapse_dst` and the RMS-normalised, weighted `norm_dst`.
   - **TG1 lane 0** (`:1510`-`:1516`): the 4 `post` gates. **511 of 512 threads idle.**
   - **TG2..TG5** (`:1525`-`:1543`): each contributes one `atomic_fetch_add` from
     lane 0; the **last arriver alone** runs `ds4_hc_comb_weights4_exact` — the
     entire 20-iteration Sinkhorn on the 4×4 matrix, **single-threaded**.
     Threadgroups 2–5 are 512 threads each and do essentially nothing after
     Phase B; 2047 of their 2048 threads exit at `:1526`.

So after Phase B the kernel degenerates into a device-memory-ordered rendezvous
among 6 threadgroups where one lane does the Sinkhorn.

### 3.2 `attn_hc_post` — span `ds4.c:23870` → `ds4.c:23900`

- `ds4.c:23872`, `:23875` — debug dumps, no-op.
- `ds4.c:23877` — directional steering, off by default.
- `ds4.c:23880` — `!fuse_attn_out_hc` is **true under TP2**: the attention-output
  ⊕ HC fusion (`kernel_dsv4_q8_hc_expand4_q8_0`) requires `g->tp_world < 2`
  (`ds4.c:23598`). Under TP the rank partials are handed to the expand instead
  (`tp_attn_a` / `tp_attn_b`, set at `ds4.c:23865`-`23866`), so the branch taken
  is `ds4_gpu_hc_expand_add_tensor` (`ds4.c:23882`, `ds4_metal.m:44240`).

**Grid:** `n_elem = n_embd × n_tokens = 4096`, `nth = min(256, n_elem) = 256`,
`n_tg = 16` (`ds4_metal.m:44328`-`44329`) → **16 threadgroups × 256 threads,
4096 threads total, 16 of 60 cores.** No threadgroup memory.

**Kernel** `kernel_dsv4_hc_expand4` (`metal/dsv4_hc.metal:652`): one thread per
embedding dim; loads `block_out[d]`, `block_add[d]`, the 4 residual streams
`r0..r3`, then writes 4 outputs
`acc = block·post[h] + Σ_s comb[h][s]·r_s`. The `post`/`comb` scalars are
re-loaded from device memory inside the `dst_hc` loop by every thread
(`:684`-`:689`) — 20 scalar device loads per thread of the *same* 20 values,
4096 threads deep. There is a vectorised sibling that hoists them into
`float4`s (`kernel_dsv4_q8_hc_expand4_q8_0_vec_hc`, `:1029`-`:1038`) but it is
only wired into the *fused* single-node path, not this one.

**Note for the caller:** the ATTN TP gate (`ds4_gpu_tp_gate_encode(il,
DS4_TP_GATE_ATTN)`, `ds4.c:23856`) sits **before** the `attn_output` marker, so
it is charged to `attn_output` (3.746/3.804 ms), not to `attn_hc_post`.

### 3.3 `ffn_hc_pre` — span `ds4.c:23900` → `ds4.c:24009`

Structurally identical to §3.1 with `hc_ffn_*` weights: debug dump (no-op), then
the same compound producer (`ds4.c:23920`-`23947`, same wrapper
`ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor`), then a skipped standalone and
a disabled check. **1 dispatch, 6 × 512.**

That `attn_hc_pre` (1.155) and `ffn_hc_pre` (1.121) agree to **3%** is the
strongest available confirmation that both spans really are exactly this one
kernel: same shape, same weights size, different weights.

### 3.4 `ffn_hc_post` — span `ds4.c:25319` → `ds4.c:25377`

**This span is mostly not HC work.** With `tp_fold_ffn` true (it is:
`tp_split_shared && !keep_ffn_out && !steering`, `ds4.c:24975`; and U12 reports
a `routed_moe_folded` row, which only exists on that path), the previous marker
is `routed_moe_folded` at `ds4.c:25319`. Between it and `ffn_hc_post`:

1. `ds4.c:25331` — the `ds4_gpu_add_tensor` is **skipped** (folded).
2. `ds4.c:25335` — **`ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_FFN)`**
   (`ds4_metal.m:10449`). With `DS4_METAL_FAST_SYNC=1` (the standard rig
   procedure) this encodes, into the same command buffer:
   - `kernel_dsv4_tp_flag_set_coherent`, **1 threadgroup × 1 thread**
     (`ds4_metal.m:10495`) — publishes arrival to the peer;
   - `kernel_dsv4_tp_fence_wait`, **1 threadgroup × 1 thread**
     (`ds4_metal.m:10431`) — a bounded **spin loop on a system-coherent word**
     (`metal/dsv4_misc.metal:7333`-`7350`) that blocks the GPU until the service
     thread completes the 16 KB RDMA exchange.
3. `ds4.c:25347`, `:25356`, `:25359` — all skipped (no `keep_ffn_out`, no steering).
4. `ds4.c:25368` — `ds4_gpu_hc_expand_add_split_tensor` (`ds4_metal.m:44471`).
   **Byte-for-byte the same kernel and grid as §3.2**: `kernel_dsv4_hc_expand4`,
   `n_elem = 4096`, `nth = 256`, **16 threadgroups × 256 threads**
   (`ds4_metal.m:44555`-`44556`). The only difference from `hc_expand_add` is
   that `post`/`comb` are taken as offset views into the single `split` buffer
   (`:44566`-`:44567`) instead of separate tensors.

**So the GPU-busy time of `ffn_hc_post` = one 16×256 expand + the entire FFN TP
gate round trip, spun on the GPU.**

---

## 4. Bytes and FLOPs per token per rank

All figures: DS4 Flash, decode (1 token), one TP2 rank, 43 layers. HC tensors
are **replicated**, so TP2 does not halve any of this.

### 4.1 `attn_hc_pre` / `ffn_hc_pre` (each)

Per layer:

| traffic | unique bytes | requested bytes | note |
|---|---|---|---|
| `hc_*_fn` F16 weight | 786,432 | 786,432 | 24 rows × 16384 × 2 B; each cluster owns 2 rows, no redundancy |
| HC row `x` (16384 F32) | 65,536 | **1,245,184** | 6× for Phase A + 12× for Phase B + 1× for the TG0 collapse = 19 passes |
| block norm weight | 16,384 | 16,384 | `attn_norm` / `ffn_norm`, F32 4096 |
| `hc_*_base` + `scale` | 108 | 108 | |
| writes: `mix`, `split` | 192 | 192 | 24 F32 each |
| writes: `collapse_dst`, `norm_dst` | 32,768 | 32,768 | 4096 F32 each |
| **per layer** | **901,420 ≈ 880 KiB** | **2,081,068 ≈ 1.98 MiB** | |
| **per token (×43)** | **0.0388 GB** | **0.0895 GB** | |

Reconciliation (§1.5): 786,432 B/layer is exactly
`hc_dim × mix_hc × sizeof(F16) = 16384 × 24 × 2`, matching
`tensor_expect_layout(l->hc_attn_fn, DS4_TENSOR_F16, 2, hc_dim, hc_mix_dim, 0)`
at `ds4.c:5132`. Both HC producers together are 1.50 MiB/layer = **2.5% of the
60.17 MB/layer attention weight total**, so they are a rounding error in the
model's byte budget.

FLOPs per layer:

| term | flops |
|---|---|
| Phase A RMS, ×6 redundant | 196,608 |
| Phase B matvec 16384×24 | 786,432 |
| TG0 collapse + `dot(v,v)` + norm·weight | 49,152 |
| Sinkhorn (20 iters, 4×4, **1 thread**) | ~2,000 |
| **per layer** | **≈ 1.034 M** |
| **per token (×43)** | **≈ 44.5 MFLOP** |

### 4.2 `attn_hc_post`, and the expand kernel inside `ffn_hc_post`

Per layer (`kernel_dsv4_hc_expand4`, 4096 dims):

| traffic | bytes |
|---|---|
| `block_out` 4096 F32 | 16,384 |
| `block_add` 4096 F32 | 16,384 |
| `residual` 4×4096 F32 | 65,536 |
| `post` (4 F32) + `comb` (16 F32) | 80 (× re-read 4096 times from cache) |
| write `dst` 4×4096 F32 | 65,536 |
| **per layer** | **163,920 ≈ 160 KiB** |
| **per token (×43)** | **0.00705 GB** |

FLOPs per layer: 1 (add) + 4 dst_hc × (1 mul + 4 FMA = 9) = **37 per dim** ×
4096 = **151,552**; per token **6.52 MFLOP**.

### 4.3 The TP gate portion of `ffn_hc_post`

Payload: **16 KB per gate in one RDMA message**
(`tp_decode_investigation.md` §10 "RDMA nuances"). 43 FFN gates/token =
**0.688 MB/token** — 0.0007 GB. This is a *latency* item, not a bytes item;
pricing it against a bandwidth roof is meaningless.

---

## 5. Achieved rate vs both roofs

Roofs: **760 GB/s** streaming, **~21 TFLOP/s** FP32.

| stage | ms/tok (131k) | GB/tok (unique) | achieved GB/s | **% of 760** | GFLOP/tok | achieved GFLOP/s | **% of 21 T** |
|---|---|---|---|---|---|---|---|
| `attn_hc_pre` | 1.155 | 0.0388 | 33.6 | **4.4%** | 0.0445 | 38.5 | **0.18%** |
| `ffn_hc_pre` | 1.121 | 0.0388 | 34.6 | **4.6%** | 0.0445 | 39.7 | **0.19%** |
| `attn_hc_post` | 0.461 | 0.00705 | 15.3 | **2.0%** | 0.00652 | 14.1 | **0.07%** |
| `ffn_hc_post` (whole span) | 1.812 | 0.00775 | 4.3 | **0.6%** | 0.00652 | 3.6 | **0.02%** |
| `ffn_hc_post` (expand only, see §5.2) | ~0.461 | 0.00705 | 15.3 | **2.0%** | 0.00652 | 14.1 | **0.07%** |

If the HC-pre stages are instead priced on *requested* (redundant) bytes rather
than unique bytes, they rise to 77.5 / 79.9 GB/s = **10.2% / 10.5% of the roof**
— still an order of magnitude off.

### 5.1 Which roof is each stage near? — **none of them**

**Not one of the four HC stages is within 10× of either roof.** The best case is
an HC-pre stage at 10% of the memory roof counting every redundant re-read, and
0.2% of ALU. Concretely: the 786,432 B of HC mix weight that `attn_hc_pre` must
move would take **1.03 µs** at 760 GB/s; the stage takes **26.9 µs per layer**
(1.155 ms ÷ 43). That is **26× the bandwidth floor**.

The classification these stages fall into is the third one this project keeps
finding: **occupancy / latency bound**, i.e. the underfill family. Evidence
lines up with all four prior sightings:

- `attn_hc_pre` / `ffn_hc_pre`: **6 threadgroups on 60 cores** — 10% of the
  machine, and the last phase runs on 1 threadgroup (TG0) plus two single lanes.
- `attn_hc_post` / the expand in `ffn_hc_post`: **16 threadgroups on 60 cores** —
  27% of the machine, one dispatch, ~11 µs per layer for 160 KiB.
- Threadgroup memory is *not* the limiter anywhere here (672 B and 0 B), unlike
  U7's indexer scorer — so the constraint is purely grid size and the serial
  tail, and the fixes are different from U7's.

But note the counter-evidence the brief calls out: the packed32 flash reduce and
the split-K sweep both show that **simply making the grid bigger is not
automatically a win**. See §6 for why that matters to the specific candidates.

### 5.2 `ffn_hc_post` is 75% TP gate, not HC

`attn_hc_post` and the expand inside `ffn_hc_post` are the **same kernel with the
same grid and the same argument shapes** (§3.2, §3.4). Both rows come from the
**same U12 run**, so the difference is a within-epoch decomposition, not an
epoch mix:

```
ffn_hc_post 1.812  −  attn_hc_post 0.461  =  1.351 ms/token
1.351 ms / 43 FFN gates = 31.4 us per gate
```

Independent corroboration from a *different* epoch (quoted as corroboration
only, not subtracted into any figure): `tp_decode_investigation.md` §4 measured
the TP gate at **38 µs each, 86 gates = 3.30 ms/token** with
`DS4_METAL_FAST_SYNC` (commit `5adc371`), and §10 notes gates cost **~180 µs
each without it**. 43 gates × 38 µs = 1.63 ms — the same order, slightly higher,
consistent with a different build.

**Therefore: of the 4.56 ms attributed to "the four HC stages", ~1.35 ms
(29.6% of the HC pool, 5.6% of the 2k token) is the FFN TP gate.** The real HC
kernel pool is **~3.21 ms, 13.2% of the 2k token.** Any candidate aimed at
`ffn_hc_post` as if it were HC arithmetic is aimed at the wrong 75%.

The same correction applies to `attn_output` (3.746 ms), the other unpriced
stage in the test plan's table: the ATTN gate is inside it (`ds4.c:23856`), so
roughly another ~1.35 ms of that 3.75 ms is gate, not matvec. Priced on §4's
verified 1.533 GB for the attention output pair, the *kernel* part of
`attn_output` would then be ~2.4 ms → ~640 GB/s → **84% of the 760 roof**, which
is consistent with §8's warning that `attnout` is a **phantom target**.

### 5.3 The measurement error bar, which is large here

Two independent facts say these four numbers are inflated:

1. **The stage-profile artifact** (§1.2): U12's stage sum at 32k is 32.70 ms
   against 28.91 ms unprofiled on the same build — **+13%**, ≈4.2 µs per stage
   boundary. `attn_hc_post` is only 10.7 µs/layer *total*, so on this estimate
   **~39% of it is command-buffer envelope**, not kernel. The HC-pre stages are
   26.9 and 26.1 µs/layer, so ~16% each.
2. **The `hcpre` ablation disagrees by 2.7×.** `DS4_TP_ABLATE=hcpre`
   (`ds4.c:22211`, gating `ds4.c:22311` and `ds4.c:23905` — i.e. exactly the
   two HC-pre producers) measured **0.85 ms/token** end-to-end for *both* sites
   (`tp_decode_investigation.md` §4, 43.14 vs 41.56 t/s). U12's stage profile
   attributes **2.276 ms** to the same two sites.

   These are different epochs and different code (the ablation predates the
   fused compound producer), so **they must not be subtracted or averaged**.
   What they jointly license is only this: the end-to-end value of removing the
   HC-pre producers has been measured once, in a real run, at 0.85 ms — and the
   stage profile's 2.276 ms has never been corroborated end-to-end. **Re-running
   `DS4_TP_ABLATE=hcpre` on the current build is the single cheapest thing on
   this whole page and it gates every candidate in §6.**

### 5.4 Where the 26.9 µs/layer of an HC-pre stage plausibly goes

**This subsection is inference, not measurement** — flagged as required. But it
is the model that the candidates in §6 are built on, so it is stated explicitly
so it can be attacked.

The compound producer's threadgroups are **not symmetric**. Reading
`metal/dsv4_hc.metal:1447`-`1543`:

| group | Phase A (RMS) | Phase B (matvec) | epilogue |
|---|---|---|---|
| **TG0** | full 64 KiB row | 4 rows = 128 KiB weight + 2×64 KiB x | **the entire collapse**: reads 4×4096 x (64 KiB) + `norm_weight` (16 KiB), RMS-reduces, writes `collapse_dst` + `norm_dst` (32 KiB) |
| TG1 | full 64 KiB | 4 rows, same | 1 lane writes 4 `post` gates |
| TG2–TG5 | full 64 KiB | 4 rows, same | 1 lane of the last arriver runs the 20-iteration Sinkhorn |

**TG0 is the critical path and it is one threadgroup on one core.** Its serial
byte budget is ~320 KiB (A+B) + ~112 KiB (epilogue) = ~432 KiB. A single M2
core streaming at an assumed 30–60 GB/s gives **~7–14 µs**. Against the observed
**26.9 µs**, the residual ~13–20 µs is: the stage-profile CB envelope (~4.2 µs,
§1.2), the dispatch launch (1.9–4.4 µs, §6 of the investigation), and the
**three device-scope `seq_cst` fences** the fusion needs to order the split
behind the matvec inside one dispatch (`:1442`, `:1522`, `:1534`, `:1540`) —
which is precisely the cost the fusion traded a dispatch away for.

The theoretical floor, for contrast: the 880 KiB/layer of unique traffic at
760 GB/s is **1.19 µs**. So an idealised HC-pre is ~4 µs including launch, vs
26.9 µs today.

Two pieces of hard evidence bear on which term dominates, and they point in
**opposite** directions — which is why §6 leads with measurement:

- **For "grid-limited":** the grid is 6 threadgroups on 60 cores.
- **Against "grid-limited":** commit `d81a28f` on branch `pr-778` (**not in this
  branch** — verified: `git merge-base --is-ancestor d81a28f HEAD` fails) already
  implements exactly the widening fix — "HC producer spread: give every comb
  threadgroup one live NR0=2 cluster instead of packing both clusters on groups
  2..5, raising the producer grid from **6 to 10 threadgroups**", bit-identical
  — and it measured only **+0.12% decode on M5 Max (40-core)**. Reading the
  diff explains why: the spread unloads TG2..TG9, and **TG0's 4 matvec rows and
  its whole collapse epilogue are untouched.** It does not shorten the critical
  path.

---

## 6. Ranked candidate wins

Ranked by **expected end-to-end share of the 2k token (24.34 ms)**, not by
kernel multiple. Baseline 41.09 t/s at ctx 2048 (`BENCHMARKS-TP-PP.md:730`).

Conversion table used below: −0.5 ms → +0.86 t/s; −0.9 ms → +1.57 t/s;
−2.0 ms → +3.7 t/s.

### The ceiling, stated first

- The four stages sum to 4.56 ms = **18.7%** of the 2k token *as attributed by
  the stage profile*.
- **1.35 ms of that is the FFN TP gate, not HC** (§5.2). The HC kernel pool is
  **3.21 ms = 13.2%**.
- The stage profile over-attributes by ~13% on average and more for small
  stages (§1.2, §5.3), and the one end-to-end measurement that exists
  (`hcpre` ablation, 0.85 ms) is 2.7× below the profile's 2.276 ms for the same
  two sites.
- A *perfect* HC implementation (bandwidth floor + launch) would be ~0.4 ms, so
  the absolute ceiling is **~2.8 ms = 11.5% = +5.3 t/s**, and nothing found here
  suggests that is reachable.
- **HC alone cannot deliver the 4.34 ms the test plan needs for 50 t/s.**

### H1 — Re-price the pool before building anything (measurement, not a win)

**Mechanism.** Two zero-code arms, both already wired:

1. `DS4_TP_ABLATE=hcpre` on both ranks (`ds4.c:22211`, gating `ds4.c:22311` and
   `ds4.c:23905`). Caveat #3 of `tp_decode_investigation.md` §9 still holds in
   the current code: with the ablation on, `attn_hc_producer_pre_norm_fused`
   stays false, so `ds4_gpu_hc_split_weighted_sum_norm_tensor` runs instead —
   **the dispatch count is unchanged**, only the RMS+mix work disappears. Clean
   arm.
2. `DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE=1` (`ds4.c:22326`,
   `ds4.c:23918`). This is the **fusion-cost probe**: it replaces one 6×512
   fenced dispatch with `kernel_dsv4_hc_rms_norm_mix_f16` at **12 TG × 256**
   (`ds4_metal.m:43765`-`43769`, non-cluster2 because
   `ds4_gpu_device_is_m5_apple_silicon()` is false on M2 Ultra) **plus**
   `kernel_dsv4_hc_split_weighted_sum_norm4` at **1 TG × 1024, 16,528 B of
   threadgroup memory** (`ds4_metal.m:43637`, `:43645`, `:43670`).

**Expected saving.** Arm 1: none (it is destructive). Arm 2: **unknown, and
that is the point** — it is the only cheap way to learn whether the three
device-scope `seq_cst` fences cost more than the dispatch they removed. If arm 2
is faster, it is a **default-flip win at zero implementation cost**.

**Expected end-to-end share.** Arm 2 upside plausibly 0 to 0.4 ms
(0 to 1.6%, 0 to +0.7 t/s). Arm 1 re-prices 2.276 ms of attribution.

**Implementation cost.** Zero. Two env vars, one interleaved A/B each.

**Falsified by.** Arm 2 measuring flat-to-negative — which would say the fusion
is paying for itself and the stage is launch/critical-path bound, killing H2 and
most of H3.

**Note the confound:** arm 2 changes *three* things at once (fence removal,
grid 6→12 for the matvec, and the epilogue moving to a 1×1024 smem-heavy group).
It is a screening arm, not an attribution.

### H2 — Unload TG0: the producer's critical path

**Mechanism.** §5.4: TG0 alone does 4 matvec rows **and** the full 4096-wide
collapse + RMS + weighted-norm epilogue. Every other group is idle after Phase B
except for one lane. The fix is not a wider grid (that is `d81a28f`, which
measured +0.12% because it left TG0 alone) but **moving work off TG0**:

- Assign TG0 only the 2 rows it strictly needs (`mix[0..1]`) and move rows 2–3
  to a spare group — but `pre[0:4]` is `mix[0..3]`, so TG0 needs all four before
  it can start the collapse. The mix values would have to arrive via the same
  device fence the comb protocol already uses, adding a second rendezvous.
- Or: let TG0..TG_N share the collapse after the fence, each taking a slice of
  the 4096 row, with a second atomic rendezvous for the RMS sum.

**Expected saving.** If the collapse epilogue is ~2–4 µs/layer of the 26.9 and
it spreads over 8 groups: ~2–3.5 µs/layer × 2 sites × 43 = **0.17–0.30 ms**.

**Expected end-to-end share.** **0.7–1.2% of the token, +0.3 to +0.5 t/s.**

**Implementation cost.** **High**, and worse: cross-threadgroup RMS reduction
changes the reduction tree, so it is **not bit-exact**. This project's fusion
discipline is bit-exactness (every kernel comment in `dsv4_hc.metal` says so;
`ds4_test --metal-tensor-equivalence` and the frontier hash enforce it). A
non-exact change needs a different acceptance argument entirely.

**Falsified by.** H1 arm 2 showing the fenced/fused shape is already the best
of the two available shapes; or by a ballast calibration showing the residual
~13–20 µs is launch/CB envelope rather than TG0 serialisation.

### H3 — Cherry-pick the HC producer spread from `pr-778`

**Mechanism.** `d81a28f` templatises the producer on `COMB_TGS` and adds
`kernel_dsv4_hc_rms_norm_mix_f16_cluster2_pre_norm_spread` (`COMB_TGS = 8`,
grid 6 → 10), **verified bit-identical**, kill switch
`DS4_METAL_DISABLE_M5_HC_SPREAD`, default on M5 only. On this branch it does not
exist at all. M2 Ultra has 60 cores vs M5 Max's 40, so the underfill it targets
is *worse* here.

**Expected saving.** Measured **+0.12% on M5 Max**. Scaling by core count and
by the stage's larger share here, optimistically 0.15–0.3% → **0.04–0.07 ms**.

**Expected end-to-end share.** **0.15–0.3%, +0.06 to +0.12 t/s.** Marginal.

**Implementation cost.** **Low** — a clean cherry-pick of one hunk plus a
pre-M5 gate; already bit-exactness-verified upstream (8-run abort-on-diff).

**Falsified by.** §5.4's reading of the diff — it does not touch TG0 — which
already predicts it will be small. Include it only as a rider on a bigger change.

### H4 — Widen the HC expand grid (`nth` 256 → 128/64)

**Mechanism.** `kernel_dsv4_hc_expand4` runs at `nth = MIN(256, n_elem)` →
**16 threadgroups** on 60 cores, at two call sites (`ds4_metal.m:44328`-`44329`
and `:44555`-`:44556`). Setting `nth = 128` gives 32 TGs; `nth = 64` gives 64
TGs (2 simdgroups each), which is ~1 TG/core. Both stay well under T2's
turnover point (~112 TGs on 60 cores), and the kernel has **zero threadgroup
memory**, so residency is not smem-capped as in U7.

**Expected saving.** The kernel moves 160 KiB/layer — a **0.22 µs** bandwidth
floor against **10.7 µs** measured, so ~98% of it is not bandwidth. If the
non-envelope half of that (~6 µs after removing the ~4.2 µs CB artifact) scales
with active cores, 16→48 cores could recover ~3–4 µs/layer × 2 sites × 43 =
**0.26–0.34 ms**.

**Expected end-to-end share.** **1.1–1.4%, +0.45 to +0.6 t/s.**

**Implementation cost.** **Lowest of any real change on this page** — one
constant, two call sites, no kernel edit, trivially bit-exact (the kernel is one
thread per `d`, so the partition is irrelevant to the arithmetic).

**Falsified by.** A three-point sweep {256, 128, 64} coming back flat — which
would prove the expand is dispatch-envelope bound and that *both* HC-post rows
are essentially irreducible. Also falsified if 64 regresses, matching the
packed-32 / split-K family's "smaller threadgroups are not automatically
better".

### H5 — Retarget: `ffn_hc_post`'s 1.35 ms is the FFN TP gate

**Mechanism.** Not an HC candidate at all — it is the correction that stops
1.35 ms (5.6% of the token) being mis-attributed. Two prior results bound what
can be done with it:

- R11 (`ds4.c:22138`-`22142`): replicating the three attention splits to **halve
  the gate count** measured **−8.4% to −11.8%** — "the gate wait is the window
  the peer's half runs in, not idle time."
- `32ef898`: three TP gate-exclusion removals, **+0.019 ms, nothing**
  (`tp_decode_investigation.md` §8).

**Expected saving from HC work inside this stage: at most ~0.46 ms** (the
expand), and H4 already covers that.

**Expected end-to-end share of retargeting itself:** 0% — it is bookkeeping.
Its value is preventing an engineering campaign aimed at the wrong 75%.

**Implementation cost.** Zero (see H7 for the three-line marker that proves it).

### H6 — Fuse the HC expand into the adjacent HC-pre producer

**Mechanism.** `attn_hc_post`'s expand writes `after_attn_hc` (64 KiB) and
`ffn_hc_pre`'s producer immediately reads all of it. Same for `ffn_hc_post` →
next layer's `attn_hc_pre`. Fusing each pair removes one dispatch and one
64 KiB store/load round trip. The codebase already has the technique — the
`completion` atomic rendezvous at `metal/dsv4_hc.metal:1529`.

Note this dispatch **exists only because of TP2**: single-node already fuses
both expands into their producing Q8 matvecs (`fuse_attn_out_hc` requires
`g->tp_world < 2`, `ds4.c:23598`; `kernel_dsv4_q8_hc_expand4_q8_0` and
`kernel_dsv4_shared_down_hc_expand4_q8_0`). Under TP the expand must follow the
gate, so it cannot fuse *backwards* — only forwards into the next producer.

**Expected saving.** 2 sites × 43 × (1.9–4.4 µs) = **0.16–0.38 ms**, plus
5.5 MB/token of avoided round trip ≈ 7 µs (negligible).

**Expected end-to-end share.** **0.7–1.6%, +0.28 to +0.65 t/s.**

**Implementation cost.** **High.** The intra-layer pair is feasible; the
cross-layer pair crosses the FFN gate, the tier switch (`ds4.c:22105`) and the
decode-graph island boundary. And §6 of the investigation is explicit:
"**Dispatch removal is not a productive strategy here**" — 185 dispatches were
scoped at 0.35 ms. This is 86 dispatches for 0.16–0.38 ms: exactly consistent,
and exactly as unattractive.

**Falsified by.** A `DS4_METAL_DISPATCH_BALLAST` sweep re-confirming the ~1.9 µs
marginal dispatch cost — which caps this at 0.16 ms and kills it.

### Not proposed, with reasons

- **Parallelising the 20-iteration Sinkhorn.** Already a recorded negative:
  "20 Sinkhorn iterations — register-only loop inside the producer
  (`metal/dsv4_hc.metal:550`), zero dispatches. **Dead hypothesis**"
  (`tp_decode_investigation.md` §8). Independently confirmed here by arithmetic:
  the chain is ~20 iterations × ~15 dependent float ops ≈ 1200 cycles ≈
  **0.9 µs at 1.398 GHz**, ~3% of the stage, and it is off TG0's path.
- **Vectorising `post`/`comb` loads in `kernel_dsv4_hc_expand4`.** The
  `_vec_hc` sibling exists (`metal/dsv4_hc.metal:1029`-`1038`) but is only
  selected for M5 (`ds4_metal.m:44797`-`44800`) and only inside the *fused*
  kernel, where **one lane** does the expand and the 20 scalar loads are serial.
  In `hc_expand4` all 4096 threads do it in parallel from L1. Saving <1 µs/layer.
- **Splitting the HC mix matvec across TP ranks.** `hc_*_fn` is replicated and
  only 768 KiB/layer; splitting 24 output rows over 2 ranks would add a gate
  (≈31–38 µs) to save ≈0.5 µs of bandwidth. Strictly negative.
- **Reducing HC weight precision.** F16 → F8 halves 768 KiB/layer = 0.5 µs at
  the roof. The stage is 26× off its bandwidth floor; precision is irrelevant
  here and costs exactness.
- **Anything justified by "it's 2× on a 1.1 ms stage".** Per the brief: 2× on
  `attn_hc_pre` is 0.58 ms = **2.4% of the token = +1.0 t/s**, not "2×".

### Ranked summary

| # | candidate | expected ms | **% of 2k token** | Δ t/s | cost | confidence |
|---|---|---|---|---|---|---|
| H1 | re-price (2 env A/Bs) | 0 to −0.4 | 0–1.6% | 0 to +0.7 | **zero** | — (it is the gate on the rest) |
| H4 | widen expand grid, `nth` sweep | −0.26 to −0.34 | 1.1–1.4% | +0.45–0.6 | **very low** | medium |
| H6 | fuse expand into next producer | −0.16 to −0.38 | 0.7–1.6% | +0.28–0.65 | high | low |
| H2 | unload TG0 | −0.17 to −0.30 | 0.7–1.2% | +0.3–0.5 | high, **not bit-exact** | low |
| H3 | cherry-pick `pr-778` HC spread | −0.04 to −0.07 | 0.15–0.3% | +0.06–0.12 | low | medium-high (measured upstream) |
| H5 | retarget `ffn_hc_post` | 0 | 0% | 0 | zero | high |
| | **all of H2–H4+H6, no double-count** | **≈ −0.6 to −1.0** | **2.5–4.1%** | **+1.1 to +1.8** | | |

---

## 7. Falsification tests and needed instrumentation

Ordered by value per unit effort. Every one is a **single run** — no
cross-epoch subtraction anywhere.

1. **Split `ffn_hc_post` in one run — three lines, zero risk.** Add
   `DS4_METAL_PROFILE_DECODE_STAGE("ffn_tp_gate")` immediately after
   `ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_FFN)` at `ds4.c:25335`. This turns
   §5.2's 1.351 ms *inference* into a measured row from the same run. Add the
   mirror after `ds4.c:23856` to split the ATTN gate out of `attn_output`, which
   also re-prices the other unpriced stage in the test plan's table.
   **Falsifies §5.2 outright if the gate row is small.**
2. **`DS4_TP_ABLATE=hcpre`, interleaved, current build** (H1 arm 1). Answers:
   is the HC-pre pool worth 2.276 ms (stage profile) or 0.85 ms (last
   end-to-end measurement)? Everything in §6 scales with the answer.
3. **`DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE=1`, interleaved**
   (H1 arm 2). The fusion-cost probe. If positive, it is a free default flip;
   if negative, H2 dies.
4. **`nth` sweep on `kernel_dsv4_hc_expand4`** (H4): {256, 128, 64} at the two
   call sites. Watch for the T2-family turnover.
5. **`DS4_METAL_DISPATCH_BALLAST` sweep {0,1,2,4}, on the current build.**
   The §6 marginal-dispatch figure (1.9–4.4 µs) is old and it is the divisor in
   H6's whole case; it also gives the CB-envelope estimate that §1.2 currently
   derives indirectly from U11-vs-U12.
6. **A 2k stage profile (U15).** Everything in this document uses 32k/131k
   stage numbers to reason about a 2k token. The HC stages are context-invariant
   (1.142→1.155, 1.119→1.121, 1.834→1.812, 0.467→0.461 — all within 1.2%), so
   the inference is sound, but it has never been checked directly.

**What no instrumentation can give you here:** a breakdown *inside* the compound
producer. It is one dispatch, so a marker cannot split it. The only way to
attribute its 26.9 µs among Phase A / Phase B / TG0 epilogue / device fences is
**kernel-variant A/B** — build stripped variants behind an env (e.g. a variant
that skips the epilogue and writes garbage) and difference them within one build.
That is real work, and step 2 should decide whether it is worth doing at all.
