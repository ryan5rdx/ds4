# Qwen3.8-Flash-Next — Gated DeltaNet Metal Kernel Design

Status: **DESIGN ONLY — no code written.** Companion to `docs/QWEN38-FLASH-NEXT-PORT-PLAN.md`.
Target: ds4 inference engine, branch `tp-multi-slot-batching`, 2x Apple M2 Ultra 60-core over
Thunderbolt RDMA (TP2).

Primary sources used, all read directly (not from memory):

| what | path |
|---|---|
| HF reference for **this** model | `/private/tmp/tfm/modeling_qwen4_exp.py` (`Qwen4ExpTextGatedDeltaNet`, L403–563) |
| its config | `/private/tmp/qwen38_config.json` |
| sibling reference (Qwen3-Next 80B) | `.../transformers/models/qwen3_next/modeling_qwen3_next.py` |
| FLA canonical chunked kernel | `/private/tmp/fla_src/ops_gated_delta_rule_{chunk,chunk_fwd,wy_fast,fused_recurrent}.py` |
| port plan | `docs/QWEN38-FLASH-NEXT-PORT-PLAN.md` |

> Sections are written incrementally as each is concluded. Anything still marked `TODO`
> was not reached.

---

## 0. Shape audit — the 75 MiB figure in the task brief is wrong; 108 MiB is right

**Verdict: the port plan's 108 MiB is correct. The `16k×128 + 32v×128 = 6144` reading is a
numerical coincidence and does not hold for this model.**

Read straight out of `/private/tmp/qwen38_config.json`:

```
"linear_key_head_dim":     128
"linear_value_head_dim":   128
"linear_num_key_heads":     16
"linear_num_value_heads":   48      <-- not 32
"linear_conv_kernel_dim":    4
"hidden_size":            2560
"output_gate_type":  "sigmoid"
"rms_norm_eps":           1e-06
```

So, from `modeling_qwen4_exp.py:411-412` (`key_dim = head_k_dim * num_k_heads`,
`value_dim = head_v_dim * num_v_heads`):

| quantity | value | derivation |
|---|---|---|
| `key_dim` | **2048** | 16 × 128 |
| `value_dim` | **6144** | 48 × 128 |
| `conv_dim` | **10240** | `key_dim*2 + value_dim` = 2048+2048+6144 = 4 × n_embd |
| v-heads per k-head | **3** | 48 / 16 (`repeat_interleave(3)`, L520-522) |

`ssm.inner_size = 6144` in the GGUF is therefore **`value_dim` alone**, not `key_dim +
value_dim`. Three independent confirmations that this is the right reading:

1. `ssm.time_step_rank = 48`. GDN has no time-step low-rank projection at all; the only
   length-48 objects in the module are `dt_bias` and `A_log`, both `nn.Parameter(num_v_heads)`
   (`modeling_qwen4_exp.py:432,435`). llama.cpp's GGUF writer has repurposed
   `time_step_rank` as `num_v_heads`. Under the "32 v-heads" reading, 48 is unexplained.
2. `ssm.group_count = 16` = `num_k_heads`, and `ssm.state_size = 128` = `head_k_dim`.
   Together with `inner_size = num_v_heads * head_v_dim` these four keys fully determine
   the module — and only under the 48-head reading.
3. The config file itself says `linear_num_value_heads: 48`.

### Consequences for memory

Recurrent state is `[num_v_heads, head_k_dim, head_v_dim]` fp32
(`modeling_qwen4_exp.py:376`, and `"mamba_ssm_dtype": "float32"` in the config):

```
per head   : 128 × 128 × 4 B  =  65,536 B  = 64 KiB
per layer  : 48 × 64 KiB      =   3 MiB      (3,145,728 B)
36 GDN lyr : 36 × 3 MiB       = 108 MiB      per sequence, single node
TP2 (24 v-heads/rank)         =  54 MiB      per rank
```

**108 MiB / 54 MiB per rank. The brief's 75 MiB / 37.8 MiB is off by exactly the
48/32 head-count error (108 × 32/48 = 72 ≈ 75).**

Do not forget the **conv state**, which the brief omits entirely and which is a second
per-sequence recurrent buffer with all the same rewind hazards:

```
conv state = conv_dim × (conv_kernel - 1) = 10240 × 3
  fp32 : 122,880 B = 120 KiB/layer → 4.22 MiB over 36 layers (4.22 MiB/seq)
  TP2  : 60 KiB/layer/rank         → 2.11 MiB/rank
```

Total GDN recurrent footprint **112.2 MiB per sequence, 56.1 MiB per rank at TP2.** At 4
concurrent slots on the multi-slot branch that is 449 MiB / 224 MiB per rank — small
against 117 GiB usable, but it is *per slot and context-independent*, which is the
opposite profile from a KV cache and matters for slot admission accounting.

### Weight shapes per GDN layer (for the dispatch design in §6)

Four separate input projections — note this model has **`in_proj_qkv` / `in_proj_z` /
`in_proj_b` / `in_proj_a` as four distinct `nn.Linear`s** (`modeling_qwen4_exp.py:444-447`),
*unlike* Qwen3-Next which fuses them into `in_proj_qkvz` + `in_proj_ba` with a
per-kv-group interleaved layout requiring `fix_query_key_value_ordering`. **We inherit a
plain contiguous `[q | k | v]` layout and no de-interleave step.** That is a real
simplification worth ~150 lines.

| tensor | shape | notes |
|---|---|---|
| `in_proj_qkv.weight` | [10240, 2560] | output split `[2048, 2048, 6144]` |
| `in_proj_z.weight` | [6144, 2560] | output gate pre-activation |
| `in_proj_b.weight` | [48, 2560] | → `beta = sigmoid(b)` |
| `in_proj_a.weight` | [48, 2560] | → `g = -exp(A_log)·softplus(a + dt_bias)` |
| `conv1d.weight` | [10240, 1, 4] | depthwise, `groups = conv_dim`, **`bias=False`** (L424) |
| `dt_bias` | [48] | |
| `A_log` | [48] | |
| `norm.weight` | [128] | per-head, `ones` init, **not** zero-centred (see below) |
| `out_proj.weight` | [2560, 6144] | |

**Trap: the two RMSNorm classes in this model differ.** `Qwen4ExpTextRMSNorm`
(L158-178) is zero-centred: `out = normed * (1.0 + w)`, `w` init `zeros`.
`Qwen4ExpTextRMSNormGated` (L185-201), which is the one GDN uses, is **standard**:
`out = w * normed`, `w` init `ones`. Applying the `(1+w)` form to the GDN output norm is
a silent ~correctness bug. The port plan's blanket "zero-centred RMSNorm throughout"
(§2) is wrong for this one norm.

**Trap: `conv1d` has `bias=False`.** `causal_conv1d_fn(..., self.conv1d.bias, ...)` is
passed `None`. Do not allocate or read a conv bias.

---

## 1. Algorithm restated, exactly

### 1.1 Per-layer forward (host-level order)

For hidden states `x ∈ R^{T×2560}`:

```
qkv   = x @ Wqkv^T                       # [T, 10240]
z     = x @ Wz^T                         # [T,  6144]
b     = x @ Wb^T                         # [T,    48]
a     = x @ Wa^T                         # [T,    48]

qkv   = silu( depthwise_causal_conv1d(qkv, Wconv, k=4) )      # channelwise, over T
q, k, v = split(qkv, [2048, 2048, 6144], dim=-1)              # q,k: 16 heads; v: 48 heads

q = l2norm(q, per 128-dim head)          # x * rsqrt(sum(x^2) + 1e-6)   NOT F.normalize
k = l2norm(k, per 128-dim head)
q = q * (1/sqrt(128))                    # scale folded into q  (L295-296)
q = repeat_interleave(q, 3, head axis)   # 16 -> 48   (L520-522)
k = repeat_interleave(k, 3, head axis)

beta = sigmoid(b)                                              # [T, 48]
g    = -exp(A_log) * softplus(a + dt_bias)                     # [T, 48], <= 0, unbounded below

o, S' = gated_delta_rule(q, k, v, g, beta, S)                  # o: [T, 48, 128]

o = rmsnorm_head(o) * norm_w                                   # per 128-dim head, eps 1e-6
o = o * sigmoid(z)                                             # <-- SIGMOID, per config
out = o.reshape(T, 6144) @ Wo^T                                # [T, 2560]
```

Order inside the gated norm matters and is easy to get backwards
(`Qwen4ExpTextRMSNormGated.forward`, L192-201): **normalise first in fp32, then multiply
by `weight`, then multiply by `act(gate)`.** `act` is `sigmoid` because
`config.output_gate_type == "sigmoid"`; the class *default* is `silu`. The
`z` gate is **not** normalised.

`softplus(u) = log1p(exp(u))`; for `u > 20` use `u` directly to avoid overflow. `a` and
`dt_bias` are promoted to fp32 before the softplus (L519 `.float()`), and `A_log` is
exponentiated in fp32. Because `A ∈ [0.01, 16]` at init and `softplus > 0`, `g < 0`
always, so `exp(g) ∈ (0,1)` — every decay below is a contraction and cannot overflow.

### 1.2 The recurrence (ground truth)

Per head `h ∈ [0,48)`, state `S ∈ R^{128×128}` indexed `S[dk, dv]`. For `t = 0..T-1`
(`torch_recurrent_gated_delta_rule`, L381-392):

```
S      <- S * exp(g_t)                        # scalar decay, whole state
m      <- k_t^T S            ∈ R^{128}        # "kv_mem", read of the state along dk
d      <- (v_t - m) * beta_t ∈ R^{128}        # delta
S      <- S + k_t ⊗ d                         # rank-1 update
o_t    <- q_t^T S            ∈ R^{128}        # read AFTER the update  (note!)
```

`o_t` reads the *post-update* state — the diagonal of the intra-chunk attention is live.
Getting this off by one is the classic GDN porting bug.

Equivalently, in one line: `S_t = (I - β_t k_t k_t^T) · exp(g_t) · S_{t-1} + β_t k_t v_t^T`
— a generalised Householder/delta step with a scalar forget gate. `k_t` is unit-norm
(l2norm above), so `I - β_t k_t k_t^T` with `β_t ∈ (0,1)` is a contraction: eigenvalues
`{1-β_t} ∪ {1}`. No growth mode exists. This is what makes the fp32 state stable over
262k tokens.

### 1.3 The chunked form (chunk size C = 64)

From `torch_chunk_gated_delta_rule` (L266-344), which is what we implement. Within a
chunk, with `g` **cumulatively summed inside the chunk** (`g = g.cumsum(-1)`, L308) so
`g_i = Σ_{j≤i} g_raw_j` and `g_{-1} := 0`:

```
D[i,j]      = exp(g_i - g_j)  for i >= j, else 0          # decay_mask, L309, all in (0,1]
K_β         = β ⊙ K                                        # [C,128]
V_β         = β ⊙ V                                        # [C,128]
M           = -strict_tril( (K_β K^T) ⊙ D )                # [C,C], strictly lower
A           = (I - M)^{-1} = I + M + M^2 + ... + M^{C-1}   # UT/WY transform, exact in C steps
U           = A V_β                                        # [C,128]
W           = A (K_β ⊙ exp(g))                             # [C,128]   "k_cumdecay"
```

then, serially over chunks with carried `S`:

```
V'          = W S                                          # [C,128]  inter-chunk correction
Ṽ           = U - V'                                       # [C,128]  "pseudo-values"
O_intra     = ( (Q K^T) ⊙ D ) Ṽ                            # [C,128]  D already masks j>i
O_inter     = ( Q ⊙ exp(g) ) S                             # [C,128]
O           = O_inter + O_intra
S           <- S·exp(g_{C-1}) + ( K ⊙ exp(g_{C-1} - g) )^T Ṽ
```

Two details worth calling out because they differ from the Qwen3-Next file and from
several public ports:

- **No separate `masked_fill` on the intra attention.** L329 is
  `attn = q_i @ k_i.T * decay_mask[i]` with no mask, because `D` is already
  `tril()`-ed twice (L309) so the strict upper triangle is exactly `0`. The
  Qwen3-Next version's `masked_fill_(triu(diag=1), 0)` was redundant. Our kernel must
  apply `D` and must **not** additionally zero the diagonal.
- **`Q` carries the `1/sqrt(128)` scale, `K` does not** (L295-296, applied before
  chunking). `K` in `M` is the *unscaled, l2-normed* key; `K_β` likewise.

`exp(g_i - g_j) ≤ 1` for `i ≥ j` and `exp(g_{C-1} - g_i) ≤ 1`, so every exponential in
the chunked form is bounded by 1. There is **no** `exp` of a positive number anywhere,
hence no overflow path and no need for a max-subtraction trick. This is a meaningful
simplification versus flash-attention-style kernels.

### 1.4 Where the per-head RMSNorm goes

**After** the delta-rule output `o_t` and **before** `out_proj`, over the 128-element
`head_v_dim` axis only (L556-559: `core_attn_out.reshape(-1, self.head_v_dim)`).
It is *not* over 6144. This is the fact that makes the whole layer head-parallel with a
single cross-rank exchange after `W_o` — see §4. It also means the norm can be **fused
into the tail of the GDN kernel itself**, since each head's 128 outputs are already
resident in one threadgroup at that point.

### 1.5 The determinism contract this design commits to

Restating the constraints from `QWEN38-FLASH-NEXT-PORT-PLAN.md` §7a (commit 672378d)
as they bind this kernel:

1. Chunked ≠ sequential bitwise (~1.2e-4 relative). Accepted. The chunked form is the
   *reference* for our engine; the sequential recurrent kernel is only ever used for
   the ragged residue, from an aligned state.
2. Bitwise reproducibility across different prefill splits requires (a) every carried
   state boundary at a multiple of 64, (b) fp32 handoff, (c) no length- or
   batch-row-keyed kernel selection. §2, §5 and §6 respectively.
3. The inter-chunk loop is **strictly serial** over 64-blocks. No tree scan, no
   parallel prefix over chunks. This costs nothing here (see §2.6) and buys
   split-invariance outright.
4. `T mod 64` residue tokens go through the **recurrent** kernel (§3), starting from the
   last 64-aligned state — never through a partially-padded chunk. Padding a partial
   chunk to 64 with zeros is *not* bit-identical to running 64-aligned chunks later,
   because the zero-padded `k` rows contribute `β·k k^T` terms with `β = sigmoid(0) =
   0.5` unless separately masked, and even masked they perturb the reduction order.

---

## 2. Prefill kernel: chunked scan

### 2.0 Hardware envelope (verified, not assumed)

ds4 already queries the real limit rather than hardcoding it —
`[g_device maxThreadgroupMemoryLength]` at `ds4_metal.m:28943`, `:36183`, `:36363`,
`:43504`, `:43771`. On M2 Ultra (Apple8) that returns **32768 B**. The largest static
allocation anywhere in the tree today is the indexer scorer at
`ds4_metal.m:18278-18279`: `(64*128 + 8*128)*2 + (8+512)*4 = 20,512 B`. **Every budget
below must land under 32768 B and I hold them under ~25 KiB so the pipeline is not
occupancy-capped by smem.**

Threads per threadgroup: ds4 uses `MTLSizeMake(32, nsg, 1)` (simd-major) for the mv
family and `MTLSizeMake(128|256, 1, 1)` for the tiled kernels. I use **128 threads
(4 simdgroups)** for the two scan kernels and 256 for the elementwise pre/post kernels.

### 2.1 The decomposition, and why it is three kernels not one

The chunked form in §1.3 has exactly one serial dependency: `S`. Everything else is
independent per `(head, chunk)`. Split on that seam:

| kernel | parallel over | serial over | why it is separate |
|---|---|---|---|
| **K1 `gdn_pre`** | (channel, token) | conv taps only | conv1d is causal in `t`, elementwise in channel |
| **K2 `gdn_chunk_prep`** | head × chunk | 64 rows of the UT inverse | touches no state at all |
| **K3 `gdn_scan`** | head × dv-group | **chunks** | carries `S` |
| **K4 `gdn_post`** | (token, head) | — | per-head RMSNorm + `sigmoid(z)` gate |

K2 and K3 must be separate dispatches because K3's chunk `i` needs K2's output for chunk
`i` while K3 is *serial* in `i` — you cannot interleave them inside one threadgroup
without also serialising K2, which would collapse the only large parallel axis the
prefill has.

### 2.2 The key structural fact: the whole thing is dv-column-separable

Look again at the per-chunk body in §1.3 and mark which quantities depend on the `dv`
(value/output) axis:

```
A_intra = (Q K^T) ⊙ D          # [C,C]  -- NO dv dependence at all
V'      = W S[:, cols]         # dv-local
Ṽ       = U[:, cols] - V'      # dv-local
O_inter = (Q ⊙ exp g) S[:,cols]# dv-local
O[:,cols] = O_inter + A_intra Ṽ[:, cols]
S[:,cols] <- S[:,cols]·exp(g_{C-1}) + (K ⊙ exp(g_{C-1}-g))^T Ṽ[:, cols]
```

**Every state-touching operation is independent per `dv` column, and the only shared
term (`A_intra`) is state-independent.** So if `A_intra` is materialised by K2, the
serial scan K3 can be split into `G` independent column groups per head with **no
communication and no change to any reduction order** — i.e. bit-identically to `G = 1`.

This is what rescues occupancy. A serial-over-chunks kernel is otherwise only
`n_head_local = 24` threadgroups on a 60-core GPU (40% of the machine). With `G = 4` it
is 96, which covers 60 cores with a second partial wave.

`G` is a **compile-time function constant, fixed at 4, never derived from `T` or from
the token count** — that is constraint 2(c) of the determinism contract, applied here.

### 2.3 K1 — `gdn_pre` (conv1d + silu + split + l2norm + β + g)

One dispatch. Reads the four projection outputs and the conv state; writes `q`, `k`,
`v`, `g`, `β` in the layout K2/K3 want, and the rolled conv state.

- Grid: `(ceil(T/64), conv_dim_local/64)` threadgroups × 256 threads. At T=2048, TP2:
  `conv_dim_local = 1024 + 1024 + 3072 = 5120`, so `32 × 80 = 2560` threadgroups.
- The conv is depthwise with `kernel = 4` and **no bias** (§0). Channel `c` at token `t`
  needs `x[c, t-3..t]`, so a threadgroup owning a 64-token × 64-channel tile loads a
  67-token halo. Halo smem: `67 × 64 × 4 B = 17,152 B`. Under budget.
  For `t < 3` inside the *first* chunk of a prefill continuation, the taps come from the
  persistent conv state (`conv_dim × 3` fp32); pass it as a separate buffer and select
  by `t + pos0 < 3`.
- `l2norm` is over the 128 contiguous channels of one `q`/`k` head, so a 128-aligned
  channel tile does it with one 128-lane reduction. Use `x * rsqrt(sum(x²) + 1e-6)`,
  **not** a normalize intrinsic (§0 trap).
- `β = sigmoid(b)` and `g = -exp(A_log)·softplus(a + dt_bias)` are per-`(token, v-head)`
  scalars; compute them in the threadgroups that own `b`/`a` (48 wide, 24/rank) and write
  `[T, 48]` fp32. `softplus(u) = u > 20 ? u : log1p(exp(u))`.
- Output layout: `q`, `k` as `[n_kv_head_local, T, 128]` (head-major, so K2/K3 read
  contiguous chunk tiles); `v` as `[n_v_head_local, T, 128]`. The `repeat_interleave(3)`
  is **not materialised** — head `h` reads k-head `h/3`. That saves 3× on `q`/`k`
  traffic and 4 MB/layer of scratch at T=2048.

### 2.4 K2 — `gdn_chunk_prep` (UT/WY transform)

One dispatch. Grid `(n_v_head_local, ceil(T/64))` = `24 × 32 = 768` threadgroups × 128
threads at T=2048/TP2. Fully parallel; this is where the GPU is saturated.

Per `(head h, chunk c)`:

```
1. gc[i]   = Σ_{j<=i} g_raw[j]                    (inclusive scan, 64 elements, 1 simd)
2. M       = -strict_tril( (K_β K^T) ⊙ D )        [64,64]
3. A_ut    = (I - M)^{-1}                         blocked forward substitution
4. U       = A_ut  (β ⊙ V)                        [64,128]
5. W       = A_ut  (β ⊙ K ⊙ exp gc)               [64,128]
6. A_intra = (Q K^T) ⊙ D                          [64,64]
```

**Shared-memory budget (bytes, fp32 throughout):**

| buffer | shape | bytes |
|---|---|---|
| `sA` — holds `M`, then `A_ut` in place | 64 × 64 | 16,384 |
| `sSlab` — streaming `[64,32]` operand tile | 64 × 32 | 8,192 |
| `sgc`, `sexp_gc`, `sbeta` | 3 × 64 | 768 |
| **total** | | **25,344 B** |

25,344 < 32,768 ✓, with 7 KiB of slack for padding/bank-conflict avoidance.

`D[i,j] = exp(gc[i] - gc[j])` for `i ≥ j` is **recomputed on the fly** from `sgc` rather
than materialised — a `[64,64]` fp32 `D` would be another 16 KiB and blow the budget,
and one `exp` per element is cheap next to the MMA.

`K`, `V`, `Q` are never resident whole: `[64,128]` fp32 is 32 KiB on its own. They are
streamed in `[64,32]` slabs (4 passes over `dk`/`dv`), accumulating into
`simdgroup_float8x8` registers. Step 2's `K_β K^T` is `[64,128]×[128,64]`: 8×8 output
tiles = 64 tiles over 4 simdgroups = 16 tiles/simdgroup = 32 registers/lane.

**The UT inverse.** The reference is 63 serial row updates (`modeling_qwen4_exp.py:311-314`).
Done literally that is 63 threadgroup barriers per head-chunk. Instead use the standard
blocked unit-triangular inversion FLA uses in `fla/ops/gated_delta_rule/wy_fast.py`:
split 64 into four 16-blocks, solve each 16×16 diagonal block with 15 serial steps inside
**one simdgroup** (no threadgroup barrier — `simd_shuffle` suffices), then two levels of
block back-substitution `A21 <- -A22⁻¹ A21 A11⁻¹` via MMA. Barrier count drops from ~63
to ~9.

This is *algebraically* the same inverse — `(I-M)⁻¹` is unique — but **not bitwise** the
same as the reference's row loop, because the summation order differs. That is fine and
already inside the accepted ~1.2e-4 envelope (§1.5 item 1). What matters is that the
blocking is **fixed at 16 and never varies with `T`, chunk index, or head**, so it is
bit-identical across prefill splits (§1.5 item 2c). Say this in the kernel comment; a
future "optimise short chunks" patch that varies the block size silently breaks
split-reproducibility.

**Register pressure.** Peak is step 2/6: 16 accumulator tiles (32 regs) + 2 operand tiles
(4 regs) + addressing ≈ **~45 registers/lane**. Apple8 gives 128 regs/lane at full
occupancy, so K2 runs at full occupancy.

**Outputs to device** per `(h, c)`: `U [64,128] fp32` 32 KiB, `W [64,128] fp32` 32 KiB,
`A_intra [64,64] fp32` 16 KiB = 80 KiB. At T=2048, TP2: `768 × 80 KiB = 60 MiB` of
scratch per layer. Allocate **one** such scratch buffer and reuse it across all 36 GDN
layers — it is transient within a layer. 60 MiB against 117 GiB is free; but note it
scales with the prefill chunk `T`, so size it from `g->prefill_cap`
(`ds4.c:17171`), not from `ctx_size`.

*Guess, flagged:* `A_intra` could instead be recomputed inside each of the `G=4` column
groups, trading 16 KiB × 5 of traffic for 3× redundant `[64,128]×[128,64]` MMA. I
estimate materialising wins (~100 ns of traffic vs ~300 ns of redundant fp32 MMA per
head-chunk) but I have not measured it and the crossover depends on `G`.

### 2.5 K3 — `gdn_scan` (the serial inter-chunk loop)

One dispatch. Grid `(n_v_head_local, G) = (24, 4) = 96` threadgroups × 128 threads
(4 simdgroups). Threadgroup `(h, cg)` owns `S[h][:, 32·cg : 32·cg+32]` and loops
`c = 0 .. nchunks-1` **strictly serially**.

**State in registers, not memory.** `S` slice is `128 dk × 32 dv = 4096` fp32. Split by
`dv` across the 4 simdgroups: simdgroup `sg` owns `S[:, 8sg : 8sg+8]` = 16 `float8x8`
tiles = **32 registers/lane, 128 B/lane**. Because the split is again on `dv`, *no
cross-simdgroup reduction occurs anywhere in the chunk body* — the four simdgroups never
talk except at the slab-load barriers.

Per-chunk body, with every `[64,128]` operand streamed in `[64,32]` slabs:

```
load U[:, my 8 cols]  -> registers (16 f/lane)
for kb in 0..3:  sSlab <- W[:, 32kb:32kb+32];  V' += sSlab_tiles · S_tiles[32kb..]
Ṽ = U - V'                                                (registers)
for kb in 0..3:  sSlab <- (Q ⊙ exp gc)[:, 32kb:..];  O += sSlab · S_tiles[..]
for kb in 0..1:  sSlab <- A_intra[:, 32kb:..];       O += sSlab · Ṽ_tiles
S *= exp(gc[63])
for kb in 0..3:  sSlab <- (K ⊙ exp(gc[63]-gc))[:, 32kb:..];  S += sSlab^T · Ṽ
store O[:, my 8 cols]
```

**Shared-memory budget:**

| buffer | shape | bytes |
|---|---|---|
| `sSlab0` (streaming operand) | 64 × 32 fp32 | 8,192 |
| `sSlab1` (double buffer, hides the load behind the MMA) | 64 × 32 fp32 | 8,192 |
| `sgc`, `sexp_gc` | 2 × 64 | 512 |
| **total** | | **16,896 B** |

16,896 < 32,768 ✓ — comfortably, and small enough that two K3 threadgroups co-reside per
core, which matters because K3's grid (96) does not evenly cover 60 cores.

**Register pressure:** `S` 32 + `Ṽ`/`U` 16 + `O` 16 + `V'` 16 + operand tiles 4 ≈
**~90 registers/lane**. That is near the 128-reg full-occupancy ceiling; if the compiler
spills, drop to `G = 8` (16 dv per threadgroup, 4 dv per simdgroup) which halves `S` to
16 regs at the cost of 192 threadgroups and more redundant `W`/`Q`/`K` slab loads.
Decide this from a `-frecord-command-line` register report, not from taste.

**`K`/`Q`/`W`/`A_intra` are re-read by all `G` column groups** — that is the cost of the
split. Per layer at T=2048/TP2: `4 × (W 3 MiB + Qe 3 MiB + K 3 MiB + A 1.5 MiB) = 42 MiB`
re-read vs 10.5 MiB at `G=1`. 31.5 MiB extra per layer × 36 = 1.1 GiB per 2048-token
prefill chunk = **~1.6 ms at 700 GB/s**, against the ~2.5× occupancy win on the dominant
kernel. Clear win, and it is the reason to prefer `G=4` over `G=8`.

**Arithmetic.** Per head-chunk (all `G` groups together, excluding `A_intra` which K2
computes): `W S`, `(Q⊙e) S`, `A Ṽ`, `Kd^T Ṽ` = `3 × (64·128·128) + 64·64·128` = 3.67M
MACs = 7.34 MFLOP. Over 48 heads × 32 chunks × 36 layers = **406 GFLOP per 2048-token
prefill chunk**, i.e. **198 MFLOP/token**. Against ~12.4 GFLOP/token for the 6.2B active
params, the GDN scan is **~1.6% of prefill FLOPs.** That is the justification for keeping
everything in fp32 `simdgroup_float8x8` operands rather than reaching for half: the
numerical risk is not worth 1.6%.

At 203 GFLOP/rank and a pessimistic 25% of the 60-core M2 Ultra's ~21.5 TFLOP/s fp32
peak, K3 costs **~38 ms per 2048-token prefill chunk per rank**; at 50%, ~19 ms. A
2048-token chunk on this rig runs on the order of 1–2 s, so K3 lands at **1–4% of prefill
wall time.** *This is an estimate from a peak-FLOPs ratio, not a measurement.*

### 2.6 Why the serial loop is free

The alternative — a parallel scan over chunks — would need the associative operator
`(S, A, b) ↦ (A₂A₁, A₂b₁ + b₂)` where `A` is the `128×128` transition matrix. That is a
`128×128 × 128×128` matmul per combine, i.e. **2.1 MFLOP per combine against 7.3 MFLOP
for the whole serial chunk body.** A Blelloch scan does `2n` combines for `n` chunks, so
a tree scan costs ~57% *more* arithmetic than the serial loop and needs `n × 64 KiB` of
transition-matrix storage. It would only pay if `n_head × G` were too small to fill the
GPU — and §2.2 already fills it on the `dv` axis for free. **The serial loop is not a
determinism concession; it is also the faster choice.** (Constraint 3 of §1.5 is
therefore free, exactly as the brief states.)

### 2.7 K4 — `gdn_post`

One dispatch, 256 threads, grid `ceil(T·n_v_head_local / 8)`. Per `(token, head)`:
`o = w ⊙ (o · rsqrt(mean(o²) + 1e-6))`, then `o *= sigmoid(z)`. 128 elements, one simd
reduction. Cannot fuse into K3 because K3's threadgroups each hold only `32` of the
`128` dv values of a head (§2.2); with `G=1` it would fuse, but §2.5 shows `G=1` is the
wrong trade for prefill.

### 2.8 Prefill dispatch ledger

Per GDN layer, per prefill chunk of `T` tokens:

| # | dispatch | threadgroups @ T=2048, TP2 |
|---|---|---|
| 1 | `in_proj` (qkv‖z‖b‖a, one multi-slab matmul — see §6) | existing `mul_mm` path |
| 2 | K1 `gdn_pre` | 2,560 |
| 3 | K2 `gdn_chunk_prep` | 768 |
| 4 | K3 `gdn_scan` | 96 |
| 5 | K4 `gdn_post` | 6,144 |
| 6 | `out_proj` | existing `mul_mm` path |

**6 dispatches per GDN layer per prefill chunk** = 216 per prefill chunk over 36 layers,
i.e. **0.105 dispatches per token at T=2048**. Dispatch count is irrelevant for prefill;
it is entirely an occupancy and FLOP question, which is why the prefill design optimises
`G` and the smem budget and the decode design (§3) optimises something completely
different.

---

## 3. Decode kernel: fused recurrent step

### 3.1 What the dispatch budget is actually worth — calibrating against ds4's own numbers

The brief instructs me to treat dispatch count as the primary objective. Before designing
to that I have to reconcile two numbers in this tree, because they disagree by 6.7×, and
the design changes depending on which is right.

**ds4's own in-situ measurement** (`speed-bench/tp_decode_investigation.md:281`, and the
re-derivation at `docs/TP-A0-ROWSPLIT-TEST-PLAN.md:713-723`):

- **1021 dispatches per decode token** at ctx 512 on DS4-Flash (43 layers) ≈ 23.7/layer.
- **Marginal cost ~1.9 µs/dispatch**, from the cleanest arm (the `kv` ablation removes
  exactly 43 dispatches for 0.081 ms → 1.88 µs). A ballast instrument gives 3.74 µs and
  a mask arm 4.4 µs. The doc's own verdict: *"Use 1.9–4.4 µs, not 8.6."*
- Total attributable dispatch cost **1021 × 1.9 µs = 1.94 ms = 8.1%** of a 24.06 ms
  token, and the doc concludes *"dispatch removal is not a productive strategy here."*

**The new attribution battery** reports a ~13 ms flat residual. `13 ms / 1021 = 12.7 µs`
per dispatch — 6.7× the measured marginal cost. **These cannot both be a per-dispatch
effect.** The reconciliation the tree already points at is in
`TP-A0-ROWSPLIT-TEST-PLAN.md:731-740`: the ballast instrument emits its no-ops *inside an
already-open encoder*, so it measures marginal in-encoder dispatch cost and is **blind to
encoder boundaries**. There are **172 encoder close/reopen events per token** (86 gates ×
2, `ds4_metal.m:10412-10434`, `:10477-10506`), never measured. `13 ms / 172 = 75.6 µs per
boundary`, and the external datapoint in that section — upstream #590 measuring
53.4 → 55.3 t/s for removing *one* close/reopen — is consistent with a per-boundary cost
in the tens of µs.

**Design conclusion, and it is not the same as "minimise dispatches":**

1. **Do not add encoder boundaries or gate exchanges.** This is the expensive axis. The
   GDN design in §4 adds **zero** gates per layer (the layer's single exchange sits
   exactly where ds4's ATTN gate already is), and none of K1–K4 needs a fence, an
   `ds4_gpu_synchronize()`, or a command-buffer split. That is the load-bearing property.
2. **In-encoder dispatch count is worth 1.9–4.4 µs each**, which over 36 GDN layers is
   **68–158 µs per removed per-layer dispatch per token**. Real, worth engineering for,
   but roughly 0.24%–0.56% of a ~28 ms token each. I design to it, and I quantify it,
   but I decline to pay large bandwidth costs for it (see §3.4).
3. **I flag the 13 ms residual as unattributed rather than assume it is per-dispatch.**
   If it turns out to be per-*layer* rather than per-boundary, Qwen's 48 layers vs DS4's
   43 puts the port 1.5 ms in the hole regardless of what the GDN kernels do, and no
   kernel design fixes it. **Falsifier: run the encoder-boundary sibling instrument that
   `TP-A0-ROWSPLIT-TEST-PLAN.md:736-740` specifies (~20 lines, N extra close/reopen pairs
   per decode layer, fit `d(ms/token)/d(43N)`) before committing to the GDN port.** That
   is a half-day of work that determines whether this port's decode target is reachable.

### 3.2 The fusion analysis — what fuses and what cannot

| stage | fuses into | why / why not |
|---|---|---|
| `in_proj_qkv` [10240←2560] | **D1** | |
| `in_proj_z` [6144←2560] | **D1** | same input vector, different weight slab |
| `in_proj_b` [48←2560] | **D1** | 48-wide matvec; as its own dispatch it is 1.9 µs of overhead for 0.25 MFLOP |
| `in_proj_a` [48←2560] | **D1** | ditto |
| **conv1d** (4-tap depthwise, `bias=False`) | **D1** | depthwise ⇒ output channel `c` needs only input channel `c`. A D1 threadgroup that produced a complete 64-channel slab of `qkv` can convolve **that same slab** immediately, from its own registers plus the 3-deep conv state. No cross-threadgroup dependency. |
| `silu` after conv | **D1** | elementwise |
| `l2norm(q)`, `l2norm(k)` | **D1** | reduction over the 128 contiguous channels of one head — inside a 128-aligned channel slab |
| `q *= 1/√128` | **D1** | elementwise |
| `β = sigmoid(b)`, `g = -e^{A_log}·softplus(a+dt_bias)` | **D1** | per-v-head scalars; have the threadgroup owning v-head `h`'s 128 channels also compute rows `h` of `Wb`/`Wa` |
| conv state roll | **D1** | in place, same threadgroup |
| **recurrence** (`S*=e^g`; `m=k^ᵀS`; `d=(v-m)β`; `S+=k⊗d`; `o=q^ᵀS`) | **D2** | **cannot** fuse into D1 — see §3.4 |
| per-head RMSNorm | **D3** | cannot fuse into D2 when `G>1` (D2 holds 32 of 128 dv) |
| `× sigmoid(z)` | **D3** | follows the norm |
| `out_proj` [2560←6144] | **D3** | a matvec threadgroup reads the whole 6144 input anyway, so it can apply norm+gate on the fly |

**Result: 3 dispatches per GDN layer per decode token.**

### 3.3 The three dispatches

**D1 `gdn_decode_pre`** — 4 projections ‖ conv ‖ silu ‖ split ‖ l2norm ‖ scale ‖ β ‖ g ‖
conv-state roll. Grid: one threadgroup per 128-channel slab of the local
`conv_dim = 5120`, so **40 threadgroups** (plus the b/a rows folded into the 24 v-head
slabs). 128 threads each. Weight-bandwidth bound: reads
`(10240+6144+96)/2 × 2560 × 0.6175 B ≈ 13.0 MB/rank/layer`.

This is the direct analogue of `kernel_dsv4_comp_row_finalize_f32`
(`ds4.c:22966-22971`): *"seven tiny single-row dispatches per layer … run as one
two-threadgroup dispatch with each standalone kernel's reduction tree preserved
bit-exactly."* D1 collapses **ten** logical kernels the same way, and inherits the same
obligation — each fused stage must keep the reduction tree it would have had standalone,
because the l2norm and the `β`/`g` values feed the persistent state and a different
reduction order there is a permanent divergence, not a rounding blip.

**D2 `gdn_decode_recur`** — the fused recurrent step. Grid `(24 heads, G=4)` =
**96 threadgroups** × 128 threads, exactly the K3 geometry minus the chunk loop.
Threadgroup `(h, cg)` holds `S[h][:, 32cg:32cg+32]`; simdgroup `sg` within it holds
`S[:, 8sg:8sg+8]` = 16 `float8x8` tiles = 32 regs/lane.

```
load S slice            (16 KiB per threadgroup, streaming, one pass)
S *= exp(g[h])
m[my 8 cols]  = k^T S                     # 128×8 MACs per simdgroup
d = (v[my cols] - m) * beta[h]
S += k ⊗ d                                # rank-1, register-local
o[my 8 cols] = q^T S                      # reads the POST-update S (§1.2)
store S slice
```

Shared memory: `sq[128] + sk[128]` fp32 = 1,024 B, plus a 32-float staging row = **1,152 B**.
Trivially under budget. Register pressure: `S` 32 + scratch ~10 = **~42 regs/lane**,
full occupancy.

**D3 `gdn_decode_out`** — per-head RMSNorm + `sigmoid(z)` gate + `out_proj` matvec, in
one kernel. Each threadgroup computes the 24 local per-head RMS values redundantly
(24 × 128 = 3072 fp32 = 12 KiB re-read; the threadgroup reads the 6144-element input
anyway, so this is at most a 2× read of a 24 KiB vector — **~50 ns**, versus 1.9 µs for a
separate dispatch). Then the standard row-split matvec, writing the partial into the TP
gate out-slab.

### 3.4 Why the recurrence cannot fold into D1 — quantified

It *is* structurally fusable: give threadgroup `h` the 128 `q` channels of k-head `h/3`,
the 128 `k` channels, the 128 `v` channels and rows `h` of `Wb`/`Wa`, and it has
everything head `h` needs. The problem is that `q` and `k` would then be projected
**3× redundantly** (three v-heads share one k-head) — and with `G=4`, **12×**.

Cost at `G=1`: the `q+k` half of `in_proj_qkv` is `2048/10240 = 20%` of that projection's
weight traffic. Tripling it adds `2 × 0.20 × 13.0 MB = 5.2 MB` per layer per rank →
`187 MB/token/rank` → **~0.97 ms/token at 192 GB/s, ~0.37 ms at 500 GB/s.** Removing one
dispatch across 36 layers saves 68–158 µs. **Fusion loses by 3–14×.** Rejected, with
numbers.

Fusing D2 into D3 is impossible for a different reason: D3 needs the *complete* 6144-wide
`o`, which D2 produces across 96 threadgroups. Metal has no device-scope barrier inside a
dispatch. A spin-on-atomic-counter would deadlock because 96 threadgroups are not
guaranteed co-resident on 60 cores. A persistent-threadgroup kernel launched at exactly
the core count would work in practice but relies on a co-residency guarantee that is not
in the Metal spec — **rejected as a correctness hazard, not a performance one.**

### 3.5 The occupancy/`G` trade at decode, quantified

D2 is almost pure state streaming. Per threadgroup: 16 KiB read + 16 KiB write of `S`,
against `q`+`k` 1 KiB, `v` 128 B, `o` 128 B — **97% of the bytes are state.**
Arithmetic intensity: `(128·32)·3 = 12,288` MACs per `32,768 B` = **0.75 FLOP/byte.**
There is no compute question here at all; the only lever is bandwidth utilisation, and
that is set by how many cores are busy.

| `G` | threadgroups | cores busy (of 60) | est. state bandwidth | state time/token/rank |
|---|---|---|---|---|
| 1 | 24 | 24 (40%) | ~320 GB/s | ~354 µs |
| 2 | 48 | 48 (80%) | ~640 GB/s | ~177 µs |
| **4** | **96** | **60 (100%, 1.6 waves)** | **~700 GB/s** | **~162 µs** |
| 8 | 192 | 60 | ~700 GB/s | ~162 µs, +redundant q/k reads |

(Bandwidth column is a **linear scaling guess** off the per-core share of 800 GB/s, capped
by the measured whole-token 192 GB/s → per-stage ~530 GB/s ceiling seen at
`tp_decode_investigation.md:290`. Treat the absolute numbers as ±40%; the *ordering* is
what the design rests on.)

`G=1` would let D3's norm fuse into D2, saving one dispatch = 68–158 µs, but costs
`354 − 162 = 192 µs` of bandwidth. **`G=4` with 3 dispatches beats `G=1` with 2
dispatches.** This is the one place in the design where the dispatch-count objective is
correctly overridden, and it is overridden by a factor of ~1.2–2.8, not by taste.

### 3.6 Decode ledger and achieved-bandwidth estimate

**Dispatches: 3 per GDN layer per decode token → 108 across the 36 GDN layers.**
A naive one-kernel-per-operation port is 18/layer (4 projections, conv, silu, split,
2 l2norms, scale, β, g, decay, kv_mem, delta, S-update, o-read, norm, gate, out_proj =
18 counting the split as free) → 648. **Saving 540 dispatches/token = 1.03 ms at 1.9 µs,
2.38 ms at 4.4 µs**, i.e. **3.5%–8.5%** of a ~28 ms token.

Bytes per rank per decode token, all 36 GDN layers:

| | per layer/rank | × 36 |
|---|---|---|
| `in_proj` weights (16,480→8,240 rows × 2560 @ ~4.94 bit) | 13.0 MB | 469 MB |
| `out_proj` weights (2560 × 3072 @ ~4.94 bit) | 4.86 MB | 175 MB |
| conv weights + `A_log`/`dt_bias`/`norm.w` | 0.02 MB | 0.7 MB |
| **recurrent state, read + write, fp32** | **3.15 MB** | **113 MB** |
| conv state r/w | 0.12 MB | 4.3 MB |
| activations (q,k,v,z,o and the gate partial) | 0.09 MB | 3.2 MB |
| **total** | **21.2 MB** | **765 MB** |

At the per-stage rate ds4 actually achieves on well-formed decode kernels
(**532 GB/s**, measured for `attn output` at `tp_decode_investigation.md:290`), the whole
GDN half of the model costs **~1.44 ms/token/rank**. At the pessimistic whole-token
average of 192 GB/s it is 3.98 ms. Add 108 × 1.9 µs = 0.21 ms of dispatch. **Estimate:
~1.6–4.2 ms/token/rank for 36 GDN layers**, of which the recurrent state is 113 MB =
**14.8% of the bytes** and ~0.21 ms.

Two properties worth stating because they are the strategic case for this architecture:

- **The state cost is flat in context.** 113 MB/token/rank at 1k tokens and at 262k
  tokens. DS4's compressed-KV attention stage grows; this does not. That is where the
  "36 of 48 layers do not degrade with context" claim in the port plan §1 cashes out.
- **Halving the state to bf16 would save 56 MB/token/rank (~0.1 ms).** It is explicitly
  forbidden by §1.5 constraint 2(b) and it is not worth relitigating for 0.1 ms.

## 4. TP2 head split

### 4.1 The split axis: v-head, and it divides cleanly

**Split on the 48 value heads: 24 per rank.** The 16 key heads follow at `24 / 3 = 8` per
rank, so rank `r` owns v-heads `[24r, 24r+24)` and k-heads `[8r, 8r+8)` with the
`repeat_interleave(3)` relation intact inside each rank. **No k-head is shared across
ranks** — that is the property that makes the split free, and it holds because
`48 / 2 = 24` is divisible by the 3:1 ratio.

(For TP4, `48/4 = 12 = 4 × 3` also works, and TP8 gives `6 = 2 × 3`. TP16 would give
`3 = 1 × 3`, still legal. The split only breaks at TP > 16.)

| tensor | full | per rank (TP2) | how |
|---|---|---|---|
| `in_proj_qkv.weight` | [10240, 2560] | [5120, 2560] | row-split, **non-contiguous**: rows `[1024r,+1024)` ∪ `[2048+1024r,+1024)` ∪ `[4096+3072r,+3072)` |
| `in_proj_z.weight` | [6144, 2560] | [3072, 2560] | row-split, contiguous |
| `in_proj_b.weight`, `in_proj_a.weight` | [48, 2560] | [24, 2560] | row-split, contiguous |
| `conv1d.weight` | [10240, 4] | [5120, 4] | same 3-segment split as `qkv` |
| `A_log`, `dt_bias` | [48] | [24] | contiguous |
| `norm.weight` | [128] | [128] | **replicated** (per-head, dim is `head_v_dim`) |
| `out_proj.weight` | [2560, 6144] | [2560, 3072] | **k-split**, columns `[3072r, +3072)` |
| recurrent state | [48,128,128] | [24,128,128] | never crosses the wire |
| conv state | [10240, 3] | [5120, 3] | never crosses the wire |

The `in_proj_qkv` split being three disjoint row ranges rather than one is worth calling
out for the loader: a naive `rows/tp_world` split would give rank 1 the tail of `k` and
part of `v`, silently. Either build a 3-segment descriptor, or **re-lay the tensor at
load time into `[q0..q15 | k0..k15 | v0..v47]` grouped by *k-head*** so that rank `r`'s
slice is one contiguous range. The latter is ~40 lines in the loader and removes the
hazard permanently; I recommend it.

### 4.2 What crosses the wire: exactly one `n_embd` partial, at the existing gate

`out_proj` is the only k-split matmul, so each rank produces a **full 2560-wide fp32
partial** and the two are summed. Everything upstream of it — conv, l2norm, `β`, `g`, the
entire recurrence, the per-head RMSNorm, the `sigmoid(z)` gate — is head-local, because
the output norm's reduction axis is `head_v_dim = 128` and not `value_dim = 6144`
(`modeling_qwen4_exp.py:557`). §1.4.

Payload: `ds4_tp.h:126` — *"vec_bytes = n_embd * 4 (f32 partials, never quantized on the
wire)"* → **2560 × 4 = 10,240 B per gate**, against DS4-Flash's `4096 × 4 = 16,384 B`,
which the port plan notes sits *exactly* at the 16 KB single-WR cap. Qwen has **6 KB of
slack per work request.**

### 4.3 The gate schedule needs no change at all

`ds4_engine_tp_gate_schedule()` (`ds4.c:60420-60437`) has two branches. GLM takes the
special one; **everything else takes**

```c
*start = 0;  *step = 1;  *per_token = DS4_N_LAYER * DS4_TP_GATES_PER_LAYER;
```

With `DS4_TP_GATES_PER_LAYER = 2` (`ds4_tp.h:32`) and 48 layers that is the affine triple
`(0, 1, 96)` — the identity mapping, i.e. **every layer fires ATTN + FFN, including the
36 GDN layers.** `tp_gate_slot()` (`ds4_tp.c:941-946`) computes
`start + ((seq-1) % per_token) * step` unchanged. **Zero lines change in the transport.**

This is the entire reason the head split is mandatory rather than merely nice. If GDN
were replicated and only the 12 QSA layers fired an ATTN gate, the firing set would be
`{FFN on all 48} ∪ {ATTN on 3,7,11,…}` — not an arithmetic progression, and
`tp_rdma_post_gate_recv()` (`ds4_tp.c:975`) computes an unchecked slab address from the
schedule. A schedule that cannot be expressed as `(start, step, per_token)` is a silent
out-of-bounds write, not a compile error. (Port plan §3 makes the same point; this
section confirms it against the transport code.)

The GDN layer's ATTN-gate combine reuses the existing site verbatim
(`ds4.c:23838-23847`): encode the gate, then rebuild the canonical sum **rank 0 first,
then rank 1** so both machines evaluate an identical expression and stay bit-exact.

### 4.4 Bit-exactness across TP configurations — one honest caveat

Per-head GDN state evolution is completely independent across heads, so **the recurrent
state and the pre-`out_proj` output `o` are bit-identical between single-node and TP2**
for every head. Good.

`out_proj` is not. Single-node reduces over 6144 in one kernel; TP2 reduces over
`3072 + 3072` and adds. Different summation order ⇒ different last bits. **This is a
pre-existing, accepted property of ds4's TP mode** — `attn_output_b` already k-splits
8192 into 4096+4096 the same way (`speed-bench/tp_decode_investigation.md:255-258`) — so
GDN introduces no *new* class of divergence. But it means "TP2 == single node bitwise" is
false for this model as it is false today, and the S2 logits oracle must be run
per-topology, not once.

### 4.5 Prefill

Same structure, with `ds4_tp_big_gate_exchange()` (`ds4_tp.h:145`) carrying
`T × 2560 × 4` bytes instead of one vector. At `T = 2048` that is 20.97 MB per gate ×
96 gates = 2.01 GB per prefill chunk per direction — identical in shape to what DS4-Flash
already moves, and 37% smaller per gate (2560 vs 4096 embed). No new transport work.

### 4.6 What does *not* cross the wire, and why that matters

**The 54 MiB/rank of recurrent state never moves.** There is no all-gather, no
all-reduce, no state replication. Consequences:

- The two ranks hold **different, non-redundant** state. Losing a rank loses half the
  model's memory, unrecoverably. `ds4_tp_mark_failed()` must therefore imply session
  invalidation for a GDN model, not degraded operation.
- Session save/restore must persist **both** ranks' states (§5.4).
- Nothing in the GDN layer needs a second gate, a fence, or a command-buffer split. Per
  §3.1 that is the single most valuable property of this design, because encoder
  boundaries — not dispatches — are the plausible home of the flat decode residual.

---

## 5. State layout and lifecycle

### 5.1 Buffers

Two new per-layer tensors, both fp32, both allocated only on the 36 GDN layers.

**`layer_gdn_state[il]`** — shape `[n_v_head_local][G][head_k_dim][head_v_dim / G]`.

```
TP2:  24 heads × 4 groups × 128 dk × 32 dv × 4 B = 1,572,864 B = 1.5 MiB / layer / rank
      × 36 GDN layers                            = 54 MiB / rank
single node: 3 MiB / layer, 108 MiB
```

The **`G`-major layout is deliberate**: D2's threadgroup `(h, cg)` then reads one fully
contiguous 16,384 B block instead of 128 strided 128-B reads. On a kernel that is 97%
state bandwidth (§3.5) that is the difference between a streaming and a gather access
pattern. The cost is that **`G` is baked into the on-disk state layout**, so the session
payload version must encode it (§5.4) and changing `G` must bump it.

Each `(head, group)` block is 16 KiB, so every block is 16-KiB aligned inside a
page-aligned `ds4_gpu_tensor_alloc_ptr_on()` allocation — no alignment work needed.

**`layer_gdn_conv_state[il]`** — shape `[conv_dim_local][conv_kernel-1]` = `[5120][3]`
fp32 = **61,440 B = 60 KiB / layer / rank**, 2.11 MiB over 36 layers (120 KiB/layer and
4.22 MiB single-node). Tap-minor so a 128-channel D1 slab reads 1,536 contiguous bytes.

**Total new per-sequence state: 56.1 MiB / rank at TP2, 112.2 MiB single-node.**
Context-independent. At 4 concurrent slots on `tp-multi-slot-batching`: 224 MiB/rank.

Zero is the correct initial value for both (`initial_state=None → zeros`,
`modeling_qwen4_exp.py:318-322`), so allocation follows the existing pattern at
`ds4.c:17358-17360`: `metal_tensor_fill_f32(..., 0.0f, ...)`.

### 5.2 THE POLARITY TRAP — `ratio == 0` means the opposite thing

This is the most dangerous finding in this document and it is not a kernel issue at all.

Per the port plan §4, Qwen's GGUF reuses `compress_ratios` with `0` = GDN layer and `4` =
QSA layer. But **every state-lifecycle site in ds4 today treats `ratio == 0` as "this
layer has no cache state to manage"**:

| site | code | Qwen consequence |
|---|---|---|
| `ds4.c:17322` | `if (ratio != 0) { …allocate state… }` | **GDN layers get no state buffers** |
| `ds4.c:35718` | `if (ratio == 0) return true;` (reset frontier) | GDN state never reset |
| `ds4.c:35764` | `if (ratio == 0) { n_comp=0; continue; }` (rewind) | GDN state never rewound |
| `ds4.c:51992` | `if (ratio == 0) continue;` (payload size) | **GDN state silently omitted from session save** |
| `ds4.c:53237` | `if (ratio == 0) continue;` (frontier snapshot) | **spec decode silently corrupts GDN state** |
| `ds4.c:53277` | `if (ratio == 0) continue;` (frontier restore) | ditto |
| `ds4.c:1135` | `if (ratio == 0 …) continue;` (rewind align) | see §5.3 |

Under the reinterpretation, `ratio == 0` marks the layers with **by far the most** state.
Six of these seven produce **silent wrong answers**, not crashes. Do not paper over this
with a family `if` at each site — introduce one predicate,
`ds4_layer_has_recurrent_state(il)`, and one `ds4_layer_state_bytes(il)`, and convert all
seven call sites in a single mechanical commit *before* any GDN kernel is written, with
the DeepSeek behaviour preserved exactly. That commit is independently reviewable and
independently testable against the existing model.

### 5.3 Rewind — the recurrent state is destructive, so alignment is not enough

`ds4_session_rewind()` (`ds4.c:70812-70874`) is a **metadata** operation: it clamps and
aligns `pos`, mirrors it to the TP worker, truncates `checkpoint.len`, and resets counters
and frontiers. It works because the compressed caches are *append-only* — as the comment
at `ds4.c:35757-35759` puts it, rows beyond the prefix "can remain as invisible garbage".

**A recurrent state has no such property.** `S` at position `p` is destroyed the moment
position `p+1` is processed. Truncating a counter does not bring it back.

Two separate defects follow.

**(a) `ds4_compressor_rewind_align()` returns 4, not 64.** `ds4.c:1135-1145` takes the
lcm over **non-zero** ratios only. Qwen's ratio set is `{0, 4}`, so it returns **4**. The
determinism contract (§1.5 item 2a) requires every carried-state boundary to be a
multiple of **64**. `lcm(4, 64) = 64`, so the fix is arithmetically trivial —
`align = lcm(align, gdn_chunk)` for a family with recurrent layers — but if it is missed,
the failure mode is a rewind to a non-64 boundary, a re-prefill that starts a chunk
mid-stream, and **loss of split-reproducibility with no visible symptom.**

**(b) Alignment alone is insufficient; you need checkpoints.** Even landing on a 64-
multiple, `S` still holds the *post-rewind-point* value. Options:

| option | cost | verdict |
|---|---|---|
| Reset `S = 0` and force a full re-prefill | rewind becomes O(context) | correct but destroys the live-prefix reuse the branch exists for |
| Invert the recurrence (Sherman–Morrison on `S ← (I-βkkᵀ)e^g S + βkvᵀ`) | O(1)/token | **reject.** Algebraically possible, numerically catastrophic after a few steps (`1/(1-β)` blows up as `β→1`), and definitely not bit-exact. |
| **Periodic 64-aligned state checkpoints** | 56.1 MiB/rank each | **recommended** |

**Recommended design.** Keep a ring of `K = 2` GDN state checkpoints taken at
`CHKPT_SPACING`-aligned positions during prefill, `CHKPT_SPACING = 4096` (a multiple of
64 and of the smallest prefill chunk, `DS4_PREFILL_CHUNK_MIN = 512` at `ds4.c:12266`).
Cost `2 × 56.1 = 112 MiB/rank`; rewind replays at most `2 × 4096 = 8192` tokens.

This plugs into the existing API contract without inventing anything.
`ds4_session_rewind_align()` (`ds4.h:487-490`, `ds4.c:70899`) is already documented as
*"Granularity ds4_session_rewind() snaps to … A rewind can therefore land up to this many
tokens below the requested"*. For a GDN model it returns **`CHKPT_SPACING`**, and
`ds4_session_raw_rewind_budget()` gains a sibling `ds4_session_gdn_rewind_budget()`
returning `K × CHKPT_SPACING`. Callers already have to read the landing position back
with `ds4_session_pos()` rather than assume their request (`ds4.c:70827-70830`), so
nothing above the API changes shape.

*Guess, flagged:* 4096 is picked so a rewind costs at most ~8k tokens of re-prefill
(~8 s at 1000 t/s) for 112 MiB. I have not tuned it against the actual rewind
distribution the chat/tool-binding path produces (see the recent
`b5c5a24`/`ed63073` fixes, which are exactly about rewind targets). Measure that
distribution first.

### 5.4 Session save / restore

`session_payload_live_tensor_bytes()` (`ds4.c:51986-52003`) sizes the payload. It must
add, for GDN layers, `layer_gdn_state` + `layer_gdn_conv_state` = **56.1 MiB per rank,
flat, context-independent.** Three consequences:

1. **Both ranks must save.** §4.6: the two ranks' states are disjoint, not replicas. A
   leader-only save silently restores half a model. The payload is therefore
   112.2 MiB of GDN state total, and load must reject a payload whose `tp_world` or rank
   assignment differs from the current topology.
2. **The payload version must encode `G`** (§5.1) and `chunk_size = 64`. A payload written
   with `G=4` and read by a `G=8` build is a transposed state — a plausible-looking model
   that generates garbage.
3. **Short sessions get much bigger payloads.** DS4's payload scales with context; a
   1k-token GDN session still carries 112 MiB. If the kvstore has a per-entry size policy
   (`ds4_kvstore.h`), it will need revisiting.

`ds4_session_save_layer_payload()` (`ds4.h:555`, `ds4.c:52482`) already streams per layer,
so the GDN tensors slot in as two more per-layer records.

### 5.5 Speculative decode / frontier snapshot — the expensive interaction

`spec_frontier_snapshot()` (`ds4.c:53223`) and `spec_frontier_restore()` (`ds4.c:53260`)
currently copy, per `ratio != 0` layer, `layer_attn_state_{kv,score}` at
`attn_width × attn_rows × 4` bytes — 32 KiB/layer for ratio 4, 256 KiB for ratio 128.
Tiny. They are called on every speculative verify block (`ds4.c:55064`, `:65873`,
`:66485`, `:66748`, `:69797`, `:70255`).

Adding GDN makes each snapshot **56.1 MiB/rank** — roughly **100× more expensive**. At
~800 GB/s a snapshot is ~70 µs and a restore another ~70 µs, so a rejected verify block
costs ~140 µs of pure copy. Over a decode stream that is a direct tax on the MTP
acceptance economics.

**Do not do it that way.** Use **shadow-and-swap** instead:

> Allocate one extra `layer_gdn_state_shadow[il]` per layer (+54 MiB/rank). The verify
> pass runs D2 with `S_out = shadow`, `S_in = live` — it already writes `S` once, so the
> write simply goes elsewhere; **zero extra bytes moved.** On accept, swap the two
> pointers (free). On reject, do nothing (free). `conv_state` gets the same treatment
> (+2.1 MiB).

That converts 140 µs/block of copying into 56 MiB of memory and a pointer swap. The one
constraint it imposes is that a verify block must be ≤ the shadow's capacity, which it is
(one state, any number of steps).

Note that `DS4_SPEC_PREFIX_SLOTS = 4` (`ds4.c:2577`) multiplies the *prefix1* snapshot
buffers at `ds4.c:17344-17356`. Naively extending that to GDN is `4 × 56.1 = 224 MiB/rank`
per slot. **Do not extend `spec_prefix1_*` to GDN**; gate the prefix1 optimisation off for
recurrent-state models until someone measures that it pays.

### 5.6 Multi-slot batching

On this branch each slot needs its own `layer_gdn_state` + `layer_gdn_conv_state` +
shadow: `56.1 + 54 ≈ 110 MiB/rank/slot`. Four slots = **440 MiB/rank**, flat in context.
That is affordable against 117 GiB but it is *unconditional* — unlike a KV cache it does
not shrink for short sessions, so slot admission must reserve it up front rather than
grow into it. Also: D2's grid is `(24 heads, G=4)` **per slot**; batching `B` slots gives
`96B` threadgroups, which above `B ≈ 1` is already core-saturated, so multi-slot GDN
decode is pure bandwidth scaling with no dispatch-count growth — one D2 dispatch can
serve all slots by adding a slot axis to the grid. **That is a real win: GDN decode
dispatch count is O(1) in batch size.**

## 6. Dispatch pinning

### 6.1 The hazard, located exactly

I traced the brief's constraint 5 to the specific line that changes the summation order.

`ds4_gpu_mv_ext_nxpsg()` — `ds4_metal.m:5372-5376`:

```c
static int16_t ds4_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok) {
    if ((in_dim % 256u) == 0 && n_tok < 3) return 16;
    if ((in_dim % 128u) == 0) return 8;
    return 4;
}
```

`nxpsg` is not a cosmetic tile parameter. In `dense.metal`:

- `:1607` `const short nxpsg = FC_mul_mv_nxpsg;` — a function constant, so it is baked
  into the pipeline.
- `:1613-1614` `tx = tiisg % nxpsg; ty = tiisg / nxpsg;` — `tx` is the lane's index
  **along the reduction axis**.
- `:1638` `for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg)` — **the `k`-loop
  stride is `chpt*nxpsg`.** Each lane accumulates a different, `nxpsg`-dependent subset
  of `k`.
- `:1668-1681` — the cross-lane combine is a shuffle tree of depth `log2(nxpsg)`.

**`nxpsg` *is* the k-reduction tree.** `in_dim = 2560` for every GDN input projection,
and `2560 % 256 == 0`, so this function returns **16 at `n_tok < 3` and 8 at
`n_tok >= 3`** — two genuinely different summation orders for the same weights and the
same input row, selected purely by how many tokens happen to be in the batch.

On top of that, `ds4_gpu_matmul_f16_tensor()` (`ds4_metal.m:20631`) switches between
**three unrelated kernel families**: `n_tok == 1` → plain `mul_mv` (`:20676`),
`n_tok <= 8 && in_dim % 128 == 0` → `mul_mv_ext` (`:20708`), otherwise → the TensorOps
`mul_mm` path. The Q8 (`:19134`), Q4 (`:19622`), MoE (`:20467`) and f32 (`:21576`) entry
points all repeat the same shape.

For DS4 today this is harmless: the perturbation lands in one KV row and dies. For GDN it
lands in `k`, `v`, `β`, `g` and therefore in `S`, where it persists for the rest of the
sequence.

### 6.2 The necessary-and-sufficient condition (narrower than the stated rule)

§1.5 item 2(c) says *"nothing selects a kernel by sequence length or batch row count."*
That is sufficient but **stronger than necessary**, and the difference is worth 8× of
padded arithmetic at decode, so it is worth stating precisely.

In a matmul, `out[r,t] = Σ_k W[r,k]·x[t,k]`. **Token rows never enter each other's
reduction.** Therefore:

> A matmul kernel's output for row `t` is bit-invariant under any change to the token-axis
> tiling, and depends only on the partition and combine order of the **`k` axis**.

Checking the two knobs against this:

| knob | axis | affects bits? |
|---|---|---|
| `ds4_gpu_mv_ext_nxpsg(in_dim, n_tok)` `ds4_metal.m:5372` | **k** (via `dense.metal:1638,1668`) | **YES — must be pinned** |
| `ds4_gpu_mv_ext_r1ptg(n_tok)` `ds4_metal.m:5378` | token (`r1ptg` = rows/threadgroup) | no |
| `mul_mm` `NR1` template param (host picks "the widest token tile that evenly divides the batch", `dense.metal:1913-1914`) | token | no |
| `mul_mm` `NK` — `constexpr int NK = 32` (`dense.metal:1933`, loop at `:2002`) | k | already n_tok-independent ✓ |
| choice **between** `mul_mv` / `mul_mv_ext` / `mul_mm` | k (different trees entirely) | **YES — must be pinned** |

So the rule to enforce is: **the k-partition and the kernel family must be pinned; the
token tile may vary freely.** That is what makes a pinned GDN projection affordable — a
literal "one kernel, one tile, all `n_tok`" reading would force either 32× padded
arithmetic at decode (~1.1 ms/token/rank, roughly doubling GDN decode) or 256× weight
re-reads at prefill.

### 6.3 The design

**`kernel_gdn_in_proj`** — one new kernel, in `metal/gdn.metal`, used for all five GDN
input projections (`qkv`, `z`, `b`, `a`, and `out_proj` on the way out) at every `n_tok`.

```
constant short   FC_gdn_proj_nsg    [[function_constant(FC_GDN_PROJ + 0)]];  // pinned 4
constant short   FC_gdn_proj_nxpsg  [[function_constant(FC_GDN_PROJ + 1)]];  // pinned 8
constant short   FC_gdn_proj_ntile  [[function_constant(FC_GDN_PROJ + 2)]];  // 1 | 8 | 32
constant short   FC_gdn_proj_qtype  [[function_constant(FC_GDN_PROJ + 3)]];  // weight quant
```

Use `FC_GDN_PROJ = 1600` — the currently occupied bases are 100/200/300/400/500
(`flash_attn.metal:1-5`), 600 (`mul_mv`, `dense.metal:3-4`), 700 (`mul_mm`,
`dense.metal:1900-1901`), 1200 (`unary.metal:1`), 1400 (`sum_rows.metal:3`), and the
`bin.metal` base. 1600 and 1700 are free; reserve 1600 for the projection and 1700 for
the scan/recurrence kernels.

Rules, to be written into the kernel header comment as an invariant:

1. **`FC_gdn_proj_nxpsg` and `FC_gdn_proj_nsg` are constants of the *model*, not of the
   call.** They are set from `in_dim` alone (here always 2560) and never from `n_tok`.
   The pipeline memo key (following the `ds4_gpu_get_flash_attn_pipeline` pattern at
   `ds4_metal.m:3447-3456`) is `"gdn_in_proj_nsg=%d_nxpsg=%d_ntile=%d_q=%d"`.
2. **`FC_gdn_proj_ntile` may be chosen from `n_tok`** (1 for decode, 8 for verify blocks,
   32 for prefill) because it is a token-axis parameter and cannot perturb bits. This is
   the single documented exception to §1.5 2(c), and it is *guarded by a test*, not by an
   argument — see 6.4.
3. `out_proj` gets the same treatment even though its output does not enter `S`, because
   it determines the logits and hence the next sampled token.

**Enforcement, not convention.** Tag the GDN weight tensors with a distinct role at load
(`DS4_TENSOR_ROLE_GDN`) and add, under `DS4_DEBUG`, an assertion at the top of
`ds4_gpu_matmul_f16_tensor()`, `ds4_gpu_matmul_q8_tensor()`, `ds4_gpu_matmul_q4_tensor()`
and `ds4_gpu_matmul_f32_tensor()` that the weight is not GDN-roled. Otherwise the first
person who adds a "just use the generic path here" line reintroduces the bug invisibly,
and it will not show up as a crash or even as a bad benchmark — only as a session whose
state stops matching a re-prefill of the same prompt.

**A second, separate pin:** ds4 must guarantee prefill chunk boundaries are multiples of
64. The default ladder is powers of two ≥ `DS4_PREFILL_CHUNK_MIN = 512` (`ds4.c:12266`,
`ds4_prefill_watchdog_chunk()` at `:12272`) — all fine. But `DS4_METAL_PREFILL_CHUNK`
(`ds4.c:12293-12300`) and the `requested_chunk` argument accept **any** value. For a GDN
model, round both down to a multiple of 64 (and reject < 64). Without that, a user setting
`DS4_METAL_PREFILL_CHUNK=1000` silently breaks §1.5 item 2(a).

### 6.4 The oracle that makes this real

Assertions and comments do not survive refactors; a `memcmp` does.

**Split-invariance test** (`tests/test_gdn_split_invariance.c`): prefill the same 4096
token sequence four ways and `memcmp` `layer_gdn_state[il]` and
`layer_gdn_conv_state[il]` for all 36 GDN layers:

```
A: one chunk  {4096}
B: two chunks {2048, 2048}
C: 64-chunks  {64 × 64}
D: {4032, 64}          -- forces the n_tok=1 kernel family adjacent to n_tok=4032
E: {4095, 1}           -- forces the decode path for the last token
```

**A, B, C, D must be bit-identical.** E is the sharpest: the last token goes through the
`n_tok == 1` plain-`mul_mv` path in the unfixed engine and through
`kernel_gdn_in_proj(ntile=1)` in the fixed one. If `nxpsg` is not pinned, E fails and A–D
pass — which is exactly the regression that would otherwise ship.

`{4095, 1}` also crosses a non-64 boundary (4095), so per §1.5 item 4 the 63-token residue
runs through the recurrent kernel and the final single token continues from it. That is
*allowed to differ from A* at the 1.2e-4 level, so **E is asserted against a second run of
E, not against A** — bit-identical to itself, tolerance-equal to A. Write that distinction
into the test or it will be "fixed" wrongly later.

---

## 7. Line-count estimate and build order

### 7.1 Line counts

| file | component | lines |
|---|---|---|
| `metal/gdn.metal` (new) | K1 `gdn_pre` — conv1d + silu + split + l2norm + β + g | 350 |
| | K2 `gdn_chunk_prep` — cumsum, `M`, blocked UT inverse, `U`, `W`, `A_intra` | 450 |
| | K3 `gdn_scan` — serial inter-chunk loop, dv-group split | 400 |
| | K4 `gdn_post` — per-head RMSNorm + `sigmoid(z)` | 120 |
| | D1 `gdn_decode_pre` — 10-stage fused pre-step | 300 |
| | D2 `gdn_decode_recur` — fused recurrent step | 200 |
| | D3 `gdn_decode_out` — norm + gate + `out_proj` | 250 |
| | `kernel_gdn_in_proj` — pinned projection (§6), ×2 quant specialisations | 350 |
| | shared: `l2norm`, `softplus`, tile loaders, `float8x8` helpers | 200 |
| | **metal subtotal** | **2,620** |
| `ds4_metal.m` | 8 pipeline getters + memo structs | 400 |
| | 8 dispatch functions (arg fill, smem, grid) | 850 |
| | tensor-role tagging + debug assertions (§6.3) | 150 |
| | **host-Metal subtotal** | **1,400** |
| `ds4_gpu_args.h` | `ds4_metal_args_gdn_{pre,prep,scan,recur,out,proj}` | 120 |
| `ds4.c` | the `ratio == 0` polarity commit (7 sites, §5.2) | 250 |
| | state alloc/free/zero, shadow buffers | 200 |
| | prefill layer encode (6 dispatches) | 220 |
| | decode layer encode (3 dispatches) | 180 |
| | rewind + GDN checkpoint ring (§5.3) | 200 |
| | frontier shadow-swap (§5.5) | 120 |
| | session save/load payload records (§5.4) | 180 |
| | **`ds4.c` subtotal** | **1,350** |
| `ds4.h` | `ds4_session_gdn_rewind_budget`, align semantics | 30 |
| `tests/` | split-invariance oracle (§6.4) | 250 |
| | reference-vector oracle + generator | 200 |
| | `bench_gdn_shapes` | 180 |
| | **tests subtotal** | **630** |
| | **TOTAL** | **≈ 6,150** |

The port plan §1 budgets *"~3.5k lines"* for "the GDN kernels". My 2,620 lines of `.metal`
plus 1,400 of dispatch = 4,020 is in that ballpark; the remaining ~2,100 is state
lifecycle, the polarity commit, and tests, which the port plan appears to account for
under its integration total rather than under GDN. **I do not think the plan is wrong; I
think 3.5k is kernels-only and the honest all-in number is ~6.1k.**

### 7.2 Build order, with a correctness oracle for every step

Each step is independently mergeable and independently falsifiable. Do not proceed past a
step whose oracle is red.

| # | step | oracle | tolerance |
|---|---|---|---|
| **S0** | Read the real UD-Q4_K_XL GGUF header; assert `48 v / 16 k / 128 / 128 / conv 4`, dump every `blk.*.linear_attn.*` tensor name, shape and quant type | matches `/private/tmp/qwen38_config.json` and §0 | exact |
| **S1** | The `ratio == 0` polarity commit (§5.2) — `ds4_layer_has_recurrent_state()` + 7 call sites, **no GDN code yet** | existing DS4-Flash decode + prefill regression, and `spec_frontier` round-trip | **bit-identical logits** |
| **S2** | Scalar C reference for one GDN layer (sequential recurrence, fp32), driven from the real weights | vs `torch_recurrent_gated_delta_rule` (`modeling_qwen4_exp.py:348`) on 64 random tokens | 1e-6 rel |
| **S3** | **D2** decode recurrent kernel, standalone harness | vs S2, single step, random `S`/`k`/`v`/`g`/`β`; plus run-to-run | 1e-5 rel; **bit-identical run-to-run** |
| **S4** | **K2 + K3** prefill chunked scan | vs `torch_chunk_gated_delta_rule` with `chunk_size=64` (same algorithm) | 1e-6 rel |
| | | vs S2 sequential (different algorithm — §1.5 item 1) | **2e-4** rel, and record the actual value |
| **S5** | `kernel_gdn_in_proj` + the pin (§6) | split-invariance A–D (§6.4) | **`memcmp` == 0** |
| **S6** | **K1 / D1** fusion of conv + l2norm + β + g | vs the same stages run as separate kernels — the `kernel_dsv4_comp_row_finalize_f32` contract (`ds4.c:22966-22971`): *each standalone kernel's reduction tree preserved bit-exactly* | **bit-identical** |
| **S7** | **D3** norm + gate + `out_proj` fusion | vs unfused; same math, same reduction tree | **bit-identical** |
| **S8** | Whole-layer, then whole-model logits | the port plan's S2 logits oracle vs the HF reference on a fixed prompt | per port plan |
| **S9** | TP2 head split | per-head pre-`out_proj` `o` vs single-node | **bit-identical** (§4.4) |
| | | full logits vs single-node | accepted `out_proj`-split envelope only |
| **S10** | State lifecycle | save → load → decode 64 tokens vs uninterrupted decode | **bit-identical** |
| | | rewind to a checkpoint → replay → compare `S` | **bit-identical** |
| | | shadow-swap accept/reject vs snapshot/restore | **bit-identical** |
| **S11** | Perf | `bench_gdn_shapes`; and **the encoder-boundary instrument from `TP-A0-ROWSPLIT-TEST-PLAN.md:736-740`** | see R1 |

**Run S11's encoder-boundary instrument first, out of order, before S2.** It is ~20 lines,
it needs no Qwen code at all, and it decides whether the decode target in §3 is reachable.
See R1.

---

## 8. Risks, ranked

### R1 — The 13 ms flat decode residual is per-*layer*, not per-encoder-boundary
**Severity: kills the port's decode case.** If the floor scales with layer count, Qwen's
48 layers vs DS4's 43 start the port **~1.5 ms/token in the hole** before GDN issues a
single dispatch, and no kernel design in this document recovers it. Note that ds4's own
ballast measurement (1.9 µs/dispatch, 1021 dispatches = 1.94 ms total,
`tp_decode_investigation.md:281`) **cannot explain 13 ms**, so the residual is something
the existing instrument is blind to — most plausibly the 172 encoder close/reopen events
per token that `TP-A0-ROWSPLIT-TEST-PLAN.md:731-740` identifies as never measured
(13 ms / 172 = 75.6 µs/boundary, consistent with upstream #590's 53.4→55.3 t/s for
removing one).
**Falsifier:** build the sibling instrument specified at
`TP-A0-ROWSPLIT-TEST-PLAN.md:736-740` — N extra encoder close/reopen pairs per decode
layer, fit `d(ms/token)/d(43N)` — and separately re-run the dispatch ballast at N ∈ {0,2,4}
at 131k. If the boundary slope explains the residual, §3's design is right and GDN adds
zero boundaries. If the *layer* slope explains it, stop and re-plan.
**Do this before writing any GDN code.** ~20 lines, half a day.

### R2 — The Unsloth UD-Q4_K_XL quant is wrong
The port plan §4 records that the repo was **one day old with zero downloads** and that
`UD-Q4_K_XL` is a heterogeneous per-tensor mix, with known gaps on ds4's read side
(Q5_K/Q6_K are MoE-`mm`-only with no `mv`; a Q8_K routed prefill requests kernels that
exist in no `.metal` file, `ds4_metal.m:31044`). A GDN layer's `in_proj_qkv` at a quant
ds4 has no `mv` kernel for is a hard blocker for the pinned-projection design in §6,
which needs one kernel family for all `n_tok`.
**Falsifier:** S0. Dump the quant type of every `blk.*.linear_attn.*` tensor and
cross-check against ds4's implemented `(type × kernel-family)` matrix. Do this on day one;
it is a 30-minute check that can invalidate the plan.

### R3 — The `ratio == 0` polarity trap fires somewhere I did not enumerate
I found seven sites (§5.2). Six of them fail **silently**. I found them by grepping
`ratio == 0` / `ratio != 0`, so any site that expresses the same idea differently
(`layer_attn_state_kv[il] != NULL`, `layer_raw_cache[il]`, a cached bool) is still live.
Note `ds4.c:35717` already mixes idioms: `if (!g->layer_raw_cache[il]) return true;` then
`if (ratio == 0) return true;`.
**Falsifier:** after the S1 commit, grep for `layer_attn_state`, `layer_raw_cache`,
`layer_n_comp`, `layer_comp_cap` and audit every use for an implicit "GDN layers have no
state" assumption. Plus the S10 save/load round-trip, which catches omission from the
payload.

### R4 — The GDN output-norm weight convention in the GGUF
`Qwen4ExpTextRMSNormGated` is **standard** (`w * normed`, `ones` init,
`modeling_qwen4_exp.py:188,198`) while `Qwen4ExpTextRMSNorm` is **zero-centred**
(`normed * (1.0 + w)`, `zeros` init, `:162,177`). The port plan §2 says "zero-centred
RMSNorm throughout", which is wrong for this one tensor. Worse, the GGUF converter may
have normalised one convention into the other.
**Falsifier:** read `blk.N.linear_attn.norm.weight` out of the GGUF. `mean(w) ≈ 1` ⇒
standard, apply `w * normed`. `mean(w) ≈ 0` ⇒ the converter subtracted one, apply
`(1+w) * normed`. This is a 5-line check and it is the difference between a correct model
and one that is subtly, unfalsifiably worse.

### R5 — K3 register pressure causes spills
§2.5 estimates **~90 registers/lane** for the prefill scan (S 32 + Ṽ 16 + O 16 + V' 16 +
operands + addressing). Apple8's full-occupancy ceiling is 128. If the compiler spills to
threadgroup or device memory, the dominant prefill kernel loses several×.
**Falsifier:** compile and read the register report before writing the host side. If it
spills, `G = 8` halves the `S` slice to 16 regs at the cost of 192 threadgroups and 2× the
redundant `W`/`Q`/`K` slab reads (§2.5 quantifies that at ~1.6 ms/prefill-chunk for
`G=4`→`G=8` roughly doubling to 3.2).

### R6 — fp32 `simdgroup_float8x8` MMA is slow on Apple8
The whole design uses fp32 operands rather than half, justified in §2.5 by the GDN scan
being only ~1.6% of prefill FLOPs. If Apple8's fp32 `simdgroup_multiply_accumulate` runs
at ¼ rather than ½ of the half rate, that 1.6% becomes ~5%, which is still tolerable, but
if it is emulated it could be 10×.
**Falsifier:** a 30-line microbench of `simdgroup_float8x8` vs `simdgroup_half8x8`
`multiply_accumulate` throughput. Fallback if needed: half operands + fp32 accumulate for
`A_intra = (QKᵀ)⊙D` and `A_intra Ṽ` only, keeping every state-touching product in fp32.
That halves the numerical exposure relative to a blanket half conversion.

### R7 — Blocked UT inverse diverges more than the accepted envelope
§2.4 replaces the reference's 63 serial row updates with a 16-blocked inversion, cutting
barriers from ~63 to ~9. Algebraically identical, numerically not. If `(I-M)` is
ill-conditioned for some head — `M = -strict_tril((K_β Kᵀ)⊙D)` with `β → 1` and highly
correlated keys — the blocked form could amplify.
**Falsifier:** S4 against `torch_chunk_gated_delta_rule` at 1e-6, run over real
activations from a long prompt, not random data. Record the worst-case `‖A‖` per head. If
it fails, fall back to the serial-63 form and eat the barriers (§2.4 estimates ~1.4 ms per
2048-token prefill chunk for the serial version — annoying, not fatal).

### R8 — Frontier snapshot cost makes MTP/spec decode unprofitable
§5.5: naively extending `spec_frontier_snapshot` (`ds4.c:53223`) to GDN costs
**~140 µs per verify block** in pure copy. The shadow-and-swap design avoids it, but it
must compose with `DS4_SPEC_PREFIX_SLOTS = 4` (`ds4.c:2577`) and the `spec_prefix1_*`
buffers at `ds4.c:17344-17356`, which I have read but not traced through their call sites.
**Falsifier:** trace `spec_prefix1_attn_state_kv` consumers and check whether any of them
needs a *value* copy rather than a pointer. If they do, gate the prefix1 optimisation off
for GDN models (my recommendation regardless: `4 × 56.1 = 224 MiB/rank` per slot is not
worth it unmeasured).

### R9 — The `in_proj_qkv` three-segment TP split is done wrong
§4.1: rank `r`'s slice is rows `[1024r,+1024) ∪ [2048+1024r,+1024) ∪ [4096+3072r,+3072)`,
**not** a contiguous half. A naive `rows/tp_world` split gives rank 1 the tail of `k` plus
part of `v` and produces a model that still generates fluent text.
**Falsifier:** S9's per-head bit-identity test between TP2 and single-node. A wrong split
fails it loudly. **Mitigation:** re-lay the tensor at load into k-head-grouped order so
each rank's slice is contiguous (~40 lines).

### R10 — Prefill scratch sized from `ctx_size` instead of `prefill_cap`
§2.4: K2's output scratch is **60 MiB per layer at T=2048** and scales linearly with `T`.
`ds4.c:12290-12300` shows the tree already distinguishes the *runtime* chunk ladder from
the *allocation* cap and warns explicitly *"RUNTIME ONLY — never use this to size the
prefill workspace"*. Getting it backwards here allocates from `ctx_size` (262144) and asks
for **7.7 GiB**.
**Falsifier:** assert `scratch_bytes == f(g->prefill_cap)` at `ds4.c:17171` and log it at
startup.

### R11 — Multi-slot admission does not account for flat per-slot state
§5.6: **~110 MiB/rank/slot** including the shadow, unconditional and context-independent.
Four slots is 440 MiB/rank. That is affordable, but slot admission on this branch
presumably grows into a KV budget rather than reserving a flat block, so a 4-slot
configuration could over-commit.
**Falsifier:** arithmetic against the branch's actual slot budget, plus a 4-slot smoke
test at 131k.

### R12 — `ds4_compressor_rewind_align()` silently returns 4
§5.3(a). `ds4.c:1135-1145` takes the lcm over **non-zero** ratios; Qwen's set is `{0, 4}`,
so it returns 4 where GDN needs 64. Low severity only because it is trivially fixable —
**high** severity in the sense that the symptom is invisible: rewinds land on non-64
boundaries, chunk phase shifts, and split-reproducibility quietly stops holding.
**Falsifier:** the S5 split-invariance test run *after* a rewind, not only from a cold
session. Add that case explicitly; the four cases in §6.4 all start cold and would miss it.

