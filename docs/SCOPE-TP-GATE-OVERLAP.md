# SCOPE: Can the TP gate wait be overlapped with useful work?

Status: **IN PROGRESS** (incremental — sections filled as concluded)
Branch: `upstream-metal-wins`
Date: 2026-08-27

## 0. Question

The TP gate release fence is a 1-thread GPU spin loop (`kernel_dsv4_tp_fence_wait`,
`metal/dsv4_misc.metal:7333`) that blocks the command buffer for an RDMA round trip while
counting as GPU-busy. Measured at ~1.75–2.7 ms/token (7–11% of the 2k token) across 86 gates.
Can the GPU do useful work while waiting, or can the wait be shortened?

## 1. Wire vs peer-compute decomposition — reconciling M0's 24.8 µs against the ~31–38 µs gate

### The two readings are not in conflict. M0's "6–7%" divides by the wrong denominator.

`DS4_TP_GATE_PROFILE` reports two numbers per gate, from the service thread
(`ds4_metal.m:10072-10200`):

- **`gpu-wait` = `t1 − t0`** (`ds4_metal.m:10088`, `:10123`) — the service thread pops a
  request off `g_tp_queue` and then *spins until the GPU reaches that gate's arrival flag*
  (`ds4_metal.m:10100-10121`).
- **`exchange` = `t2 − t1`** (`ds4_metal.m:10125-10141`, `:10173`) — post send, wait for the
  peer's recv completion, re-arm (`tp_rdma_gate_exchange`, `ds4_tp.c:1051-1126`).

The requests are enqueued by `ds4_gpu_tp_gate_encode` **at encode time**
(`ds4_metal.m:10513-10525`), not at execute time — the mutex/condvar push happens on the
CPU encoding thread, immediately after the fence dispatch is recorded into the tape. The
GPU may not have started that command buffer yet. So `t1 − t0` is *encode-to-execute lead
plus every kernel the GPU runs between the previous gate and this one*. It is not a stall of
anything. M0 says this itself — "the wait *contains* the compute, so 86 × 375 µs ≈ the token
by construction" (`docs/TP-A0-ROWSPLIT-TEST-PLAN.md:701-706`) — and then still reports
`exchange / (gpu-wait + exchange)` as "wire is 6–7% of gate time". **That ratio is
exchange-over-token, not exchange-over-gate.** It should never have been used to close the
link question.

### Corrected accounting: the exchange is ~65–80% of the gate, not 6–7%

The GPU is parked in `kernel_dsv4_tp_fence_wait` from the moment the arrival flag store
retires until it observes the release word. On the service-thread clock that interval is
**exactly `t2 − t1` plus release-store propagation plus poll granularity** — i.e. it is
`exchange`, not `gpu-wait`.

| quantity | value | source |
|---|---|---|
| `exchange`, pinned shard, 131k | **24.8 µs** | M0, `TP-A0-ROWSPLIT-TEST-PLAN.md:661` |
| `exchange`, pinned shard, 2k | **23.6 µs** | same line |
| gate, from `ffn_hc_post − attn_hc_post` differential | **31.4 µs** | brief; two independent passes |
| gate, independent measurement | **~38 µs** | `tp_decode_investigation.md:181` (`5adc371`) |
| one-way wire, 16 KB single WR, p50 | **14.5–15.5 µs** | M3, `TP-A0-ROWSPLIT-TEST-PLAN.md:826-831` |
| one-way wire, 4 KB | **8.0 µs** | same |

24.8 µs exchange + 2 extra dispatches (`kernel_dsv4_tp_flag_set_coherent` and
`kernel_dsv4_tp_fence_wait`, ~1.9 µs marginal each) + two forced encoder breaks
(`ds4_gpu_close_batch_encoder()` at `ds4_metal.m:10498` and `:10434`) lands at ~30 µs.
That reconciles with the 31.4 µs differential to within the noise of the method.
**The two readings measure the same thing and agree.** The 38 µs figure predates the M0
pinning fix and should be retired in favour of ~31 µs.

### Inside the 24.8 µs: wire is the majority, and the residual is small

The gate payload is `n_embd * sizeof(float)` = **16,384 B**, one WR, no chunking
(`DS4_TP_RDMA_MAX_MSG` ≥ 16 KB; the loop at `ds4_tp.c:1074-1101` runs once). M3's p50 for
exactly that shape is 14.5–15.5 µs one-way. So of 24.8 µs:

- **~15 µs is irreducible wire** for this payload on this fabric (60% of the exchange).
- **~10 µs is software + peer skew**, and T1 (`DS4_TP_GATE_FASTPATH`, `ds4_tp.c:1010-1049`)
  already attacked the software half and came back a **wash, ±0.6%**.

T1's post-mortem — "the recv re-arm hoist cannot help a wait that is dominated by local GPU
completion, not wire exchange" — **is wrong for the reason above**, but its *result* still
stands: the recv re-arm and unsignalled sends are worth ~nothing. That leaves peer skew as
the only place the remaining ~10 µs can hide, and it is the only part of the gate that is
not a hardware constant.

### Note on symmetry — why the rank-averaged exchange under-reports skew

Both ranks post their sends and then block in `while (r->recv_done < seq)`
(`ds4_tp.c:1108-1120`). If rank A arrives `s` microseconds before rank B, A's `exchange`
is `s + w` and B's is `w` (its peer's bytes are already in flight or landed). Rank-averaged
exchange is therefore `w + E|s|/2`, so **a symmetric straggler is halved by the averaging**.
With `w ≈ 15 µs` and a rank-averaged 24.8 µs, `E|s| ≈ 2 × (24.8 − 15 − software)`. At
software ≈ 3 µs that is **E|s| ≈ 13.6 µs of per-gate skew** — a real number, and the
dominant non-hardware term.

**This is inference, not measurement.** It rests on M3's 15 µs being the true one-way
in-situ latency and on a software estimate. Two one-run instruments settle it (§3.4).

### Verdict on question 1

**Roughly half the gate is hardware wire, and most of the rest is peer skew — not local
GPU completion.** The 24.8 µs and the 31.4 µs are the same measurement seen from two
clocks. The "exchange is 6–7% of gate time" conclusion in M0 is an artifact of dividing by
a denominator that contains the whole token, and it should be struck: it is the reason the
link question was closed prematurely and the reason T1 was mis-diagnosed after the fact.

## 2. Is there independent work to overlap with?

### The residual chain is strictly serial — there is no *existing* filler

Per-layer stage order (marker line numbers in `ds4.c`):

```
attn_hc_pre 22417 → attn_norm 22431 → q_a_kv_proj 22612 → q_lora_norm 22732
→ q_path 22790 → kv_path 22849 → compressor_* 22977..23195 → compressor_indexer 23330
→ attn_inv_rope 23582 → [ATTN GATE 23856] → attn_output 23870 → attn_hc_post 23900
→ ffn_hc_pre 24009 → ffn_norm 24023 → router 24137 → shared_gate_up 25143
→ routed_moe 25030 → shared_down 25291 → [FFN GATE 25335] → ffn_tp_gate 25343
→ ffn_hc_post 25385
```

Checked each consumer downstream of each gate:

- **ATTN gate → `attn_hc_post`**: `ds4_gpu_hc_expand_add_tensor(after_attn_hc, tp_attn_a,
  tp_attn_b, …)` (`ds4.c:23884-23887`) consumes both rank partials directly.
- **`attn_hc_post` → `ffn_hc_pre`**: `metal_graph_decode_hc_pre(..., after_attn_hc, ...)`
  (`ds4.c:24000-24007`) — reads the gated residual.
- **`ffn_hc_pre` → `ffn_norm` → `router` → shared/routed**: all chained off `ffn_norm`.
- **FFN gate → `ffn_hc_post`**: `ds4_gpu_hc_expand_add_split_tensor(after_ffn_hc, tp_ffn_a,
  tp_ffn_b, after_attn_hc, hc_split, …)` (`ds4.c:25376-25383`).
- **Layer L → layer L+1**: `after_ffn_hc` becomes `cur_hc` by pointer swap
  (`ds4.c:28432-28434`).

**Every dispatch in the token is in the residual chain.** There is no rank-local
side-computation, no cross-layer independence, and no lookahead work. The claim "under TP2
each rank owns half the heads/experts, so some downstream work may be rank-local" is false
for *downstream* work specifically: the rank-local work is all **upstream** of its gate,
by construction — that is what the gate is for.

So overlap cannot be found. **It has to be manufactured**, from one of exactly three
sources.

### Source 1 — duplicated work (the only in-token filler)

Take a stage that is currently *split* across ranks and *replicate* it on both ranks
instead, moving it into the gate's shadow. The only stage where the arithmetic works is
the **shared expert**.

Today (`ds4.c:24580` `tp_split_shared = g->tp_world == 2`):
- gate/up: lane-sliced, `tp_half = shared_dim/2` (`ds4.c:25060-25092`)
- down: k-sliced, `metal_graph_matmul_dense_quant_kslice(..., rank*(shared_dim/2), shared_dim/2, ...)` (`ds4.c:25272-25281`)
- the resulting **partial** is folded into the gate payload (`ds4.c:25295-25333`), so the
  shared expert is *necessarily upstream* of the FFN gate.

U14 prices the per-rank shared-expert share at **0.0191 ms/layer**
(`TP-A0-ROWSPLIT-TEST-PLAN.md:1730-1735`). Replicating it makes it 0.0382 ms/layer of
gate-*independent* work per rank, which can be encoded concurrently with the fence:

| | per layer | per token (×43) |
|---|---|---|
| removed from the pre-gate critical path | −19.1 µs | **−0.82 ms** |
| added, hidden in a 31.4 µs shadow | +38.2 µs, of which 31.4 free | +6.8 µs exposed | +0.29 ms |
| **net** | **−12.3 µs** | **−0.53 ms (−2.2% of the 2k token)** |

Both ranks' shared weights are already resident — the shared expert is in the ~8.2 GiB
replicated part of the shard (`tp_decode_investigation.md:243-247`), not the routed 68.5 GiB —
so this costs **no memory and no repack**.

**Not bit-exact.** Today the shared expert is a k=2048 reduction summed as two k=1024
halves in canonical rank order; replicated it is one k=2048 reduction. Different summation
order. Same T2 bar as U14.

**It is mutually exclusive with U14's shared-shift** (§4): shared-shift spends the shared
expert as *ballast to rebalance the ranks*; shadow-hiding spends it as *filler*. You cannot
do both with the same 0.0382 ms.

### Source 2 — another sequence's work (the only unbounded filler)

The gate shadow is ~2.0–2.7 ms/token of GPU with one thread resident. Work from an
independent sequence has no dependency on this sequence's gates at all. Two existing
mechanisms already reach for this:

- **Speculative verify blocks.** `ds4_gpu_tp_batch_gate_encode` (`ds4_metal.m:10537`)
  amortises one gate over `rows` rows, and `ds4_gpu_tp_keepalive_pause` exists specifically
  because "the GPU is genuinely busy there" during a verify block
  (`ds4_metal.m:10008-10014`). This is **dilution, not overlap** — it divides the per-token
  gate cost by the accepted-token count.
- **Multi-slot batching** — the sibling branch `tp-multi-slot-batching`. Slot B's layer-L
  compute is unconditionally independent of slot A's gate.

Both are large structural programs, out of scope here, but they are the only route to
*filling* the shadow rather than nibbling at it.

### Source 3 — restructuring an existing kernel so part of it becomes gate-independent

`ffn_hc_post` computes `after_ffn_hc = combine(after_attn_hc, hc_split) + expand(a+b, hc_post/hc_comb)`.
The first term is gate-independent. Splitting it out would let the residual carry run in the
shadow. Ceiling is the whole stage: `attn_hc_post` net is **0.281 ms**
(`TP-A0-ROWSPLIT-TEST-PLAN.md:210-215`), i.e. **6.5 µs/layer** against a 31.4 µs shadow, and
you pay one extra dispatch (1.9 µs) and an extra 4×4096 write. **Net at best ~0.15 ms.**
Not worth the bit-exactness risk on the HC path.

### Verdict on question 2

**There is no free filler.** The in-token answer is Source 1 — replicate the shared expert
into the FFN gate's shadow, worth **~0.5 ms/token (2.2%)**, not bit-exact, and mutually
exclusive with U14. The large answer is Source 2, which is a batching program, not a gate
fix.

## 3. Can the fence be replaced or restructured?

_TBD_

## 4. Peer lateness: cost and mitigations

_TBD_

## 5. Recommendation

_TBD_
