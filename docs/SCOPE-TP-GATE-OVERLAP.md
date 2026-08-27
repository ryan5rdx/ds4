# SCOPE: Can the TP gate wait be overlapped with useful work?

Status: **COMPLETE**
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
is `s + w` and B's is `w` (its peer's bytes are already in flight or landed). **The
rank-average of the skew term is therefore `E|s|/2`, not `E|s|`** — a symmetric straggler is
halved by the averaging. And because `DS4_TP_GATE_PROFILE` pools ATTN and FFN gates, and only
the FFN gate carries a straggler, the reported average is halved *again*. §4.2 does this
arithmetic properly and turns it into a bound.

**Everything in this subsection below the wire figure is inference, not measurement.** It
rests on M3's 15 µs being the true one-way in-situ latency and on an unmeasured software
constant. Three one-run instruments settle it — §4.6.

### Verdict on question 1

**About half the gate is hardware wire; almost none of it is "local GPU completion".**
Of the 31.4 µs gate: ~15 µs one-way wire, ~4 µs dispatch and encoder breaks, and ~10 µs
that splits between RDMA software and peer skew — with skew bounded at ≤ 8.6 µs of the
*averaged* exchange (§4.2). The 24.8 µs and the 31.4 µs are the same interval seen from two
clocks.

The "exchange is 6–7% of gate time" conclusion in M0 is an artifact of dividing by a
denominator that contains the whole token, and it should be struck: it is why the link
question was closed prematurely and why T1 was mis-diagnosed after the fact.

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
| added, hidden in a 31.4 µs shadow (38.2 µs of work, 31.4 free) | +6.8 µs exposed | **+0.29 ms** |
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

### 3.1 §8's "concurrent decode encoder" dismissal is factually wrong — the machinery is production code

`tp_decode_investigation.md:435` records: *"Concurrent decode encoder — only one
`memoryBarrierWithResources` exists in 43k lines; dependent dispatches cannot overlap.
Argued down, not tested."*

The observation is true and the conclusion inverts it. There **is** a working
`MTLDispatchTypeConcurrent` path in the decode encoder:

| piece | location |
|---|---|
| encoder created concurrent when armed | `ds4_metal.m:1028-1030` |
| arm (shared expert vs routed MoE) | `ds4_gpu_parallel_ffn_start`, `ds4_metal.m:9526` |
| encode stage 1 + the one barrier | `ds4_gpu_parallel_q8_matvec_encode_pending`, `ds4_metal.m:9652-9705` |
| encode stage 2 | `ds4_gpu_parallel_ffn_encode_second_stage`, `ds4_metal.m:9709` |
| join | `ds4_gpu_parallel_ffn_finish`, `ds4_metal.m:9727` |
| live call sites | `ds4.c:24985` (start), `ds4.c:25222` (finish) |

The single `memoryBarrierWithResources` (`ds4_metal.m:9705`) is not evidence that overlap is
impossible — it is the *one level break inside the one concurrent region that exists*, and
its own comment states the semantics correctly: "Metal defines either barrier form as an
execution barrier for every earlier dispatch in this concurrent encoder. The resource list
narrows visibility, not completion."

**But it never runs on the TP rig.** `parallel_full_ffn` requires `fuse_shared_down_hc`
(`ds4.c:24979-24981`), and `fuse_shared_down_hc` requires `g->tp_world < 2`
(`ds4.c:24182-24187`). Under TP2 the shared expert takes the split path instead. So the
concurrent encoder is proven, general, and **switched off exactly where the gate lives**.

Recommendation: strike the §8 row. The correct statement is "untested under TP2, and the
mechanism it needs already ships."

### 3.2 The strongest evidence that overlap would work is already in the source

`ds4_metal.m:10504-10507`, justifying the 1×1 fence grid:

> "One threadgroup of one thread: two threadgroups cost **21% of a saturated GPU** against
> **4% for one**, so the shape is not tunable."

Read forward instead of backward: **the spinning fence only costs ~4% of a saturated GPU.**
It is not monopolising the machine; it is merely *ordered ahead of* everything else in a
`MTLDispatchTypeSerial` encoder. The keep-alive thread (`ds4_metal.m:10029-10068`) is the
second existence proof — an entire second command queue runs FMA busywork concurrently with
the fence, every gate, in production.

Corollary: **a wider fence kernel is the wrong direction** and is already measured harmful
(21% vs 4%). It also cannot shorten the wait, which is a remote event, not a computation.

### 3.3 Options evaluated

| option | verdict |
|---|---|
| **Wider fence kernel** | **Closed.** 2 TGs cost 21% vs 4% (`ds4_metal.m:10504-10507`), and width cannot shorten a remote wait. |
| **`MTLSharedEvent` release / `encodeWaitForEvent`** | **Closed, measured.** "Resuming the command processor from `g_tp_cpu_event` costs **~186 µs**" (`ds4_metal.m:9950-9952`; same figure at `metal/dsv4_misc.metal:7317`). This is *why* `DS4_METAL_FAST_SYNC` and the spin exist. Any event-based release is a 6× regression. |
| **`waitUntilCompleted` on a second queue** | **Closed.** Needs one command-buffer commit per gate (86/token), each paying the same command-processor resume. Worse: §8 records "hazard tracking serialises *across queues*", so a second queue carrying real work that touches main-queue resources is serialised by Metal anyway. The keep-alive overlaps only because it touches a private buffer (`g_tp_keepalive_buffer`). |
| **Encoder split so the wait is not inside a compute encoder** | **No effect.** A Metal command buffer is an ordered list of encoders; moving the fence to its own encoder does not let a later encoder start earlier. This is already what happens — `ds4_gpu_tp_release_fence_encode` calls `ds4_gpu_close_batch_encoder()` at `ds4_metal.m:10434`, and `ds4_gpu_tp_gate_encode` calls it again at `:10498`. |
| **Concurrent encoder around the fence** | **The only live option.** Requires: (a) do not close the encoder around the fence, (b) create it with `MTLDispatchTypeConcurrent`, (c) an explicit `memoryBarrierWithResources` on `tp_out` *before* the arrival-flag dispatch, and another on `{tp_in, filler_out}` after the fence. Shape is identical to `ds4_metal.m:9652-9705`. |
| **Issue the next layer's rank-local work before the wait** | **Nothing to issue** — §2. |

### 3.4 The enabling refactor already exists, for prefill

`ds4_gpu_tp_gate_encode` does arrival and wait back to back (`ds4_metal.m:10496-10512`).
The **big prefill gate does not**: `ds4_gpu_tp_big_gate_kick` (`ds4_metal.m:10620`) publishes
arrival and queues the exchange, and `ds4_gpu_tp_big_gate_wait` (`ds4_metal.m:10661`)
encodes the release later — the comment at `ds4_metal.m:10603-10613` says exactly why:
"which lets it interleave more GPU work with the wire exchange."

**So the kick/wait split for the decode row gate is a mechanical refactor with a working
precedent 100 lines away.**

**But read the rest of that comment — it is the hardest problem in this whole proposal, and
it is a measured failure, not a theoretical one:**

> "Arrival always uses the batch shared event, NOT the flag word: a flag write carries no
> memory-visibility guarantee for the payload buffer, and once the GPU keeps running past
> the kick (no event wait right behind it) the service thread can observe the flag before
> the producing kernels' stores reach CPU-visible memory (**measured: stale rows in the
> first sub-kick**)."

Today the decode row gate is safe from this only because the fence is immediately behind the
flag and stops the command processor. **Remove that and the same stale-payload bug is in
scope.** Two things differ in the row gate's favour and must be verified, not assumed:

- The row gate's arrival kernel is `kernel_dsv4_tp_flag_set_coherent`
  (`metal/dsv4_misc.metal:7403-7414`), which brackets the store with system-scope `seq_cst`
  device fences. The big-gate path had no such kernel available.
- An explicit `memoryBarrierWithResources` on `tp_out` before the flag dispatch is an
  *execution* barrier for every earlier dispatch in the encoder (`ds4_metal.m:9697-9700`) —
  which is precisely the completion guarantee the big gate had to buy with an event.

A dispatch-level execution barrier guarantees the producing threads have *retired*; whether
their device stores are visible to a CPU spin-reader on the other side of the fabric
interface is a separate (and Apple-undocumented) question. **This is the go/no-go risk.**
It is testable cheaply: byte-verify the peer's received rows against a serial-encoder
control over ≥2000 gates, the same protocol M3 used.

### 3.5 The bounded-wait invariant survives

`kernel_dsv4_tp_fence_wait` bounds itself at `max_iters`
(default 200,000,000, `ds4_metal.m:10271-10272`, env `DS4_TP_FENCE_MAX_ITERS`) and latches
`timeout[0] = 1` on expiry (`metal/dsv4_misc.metal:7349`), read host-side by
`ds4_gpu_tp_fence_timed_out` (`ds4_metal.m:10677`) after the command buffer completes.
Nothing about running the fence inside a concurrent encoder changes any of that: the kernel
body is untouched, the latch is still per-step and still cleared by
`ds4_gpu_tp_clear_fence_timeout` (`ds4_metal.m:10682`). The filler dispatches would run and
produce garbage on a peer death, but the host already rejects the whole graph in that case.

### 3.6 One cheap, unrelated arm: is 16 KB the right message shape?

M3: **8.0 µs at 4 KB, 14.5–15.5 µs p50 at 16 KB.** The R10b streaming ceiling is 4.4 GB/s,
so 12 KB of extra payload should cost `12288/4.4e9 = 2.8 µs`, predicting ~10.8 µs at 16 KB.
The observed 14.5–15.5 is **~4 µs above** that. Posting the gate as **4 chained 4 KB WRs in
one `ibv_post_send`** may recover part of it — the send loop at `ds4_tp.c:1074-1101` already
chunks on `DS4_TP_RDMA_MAX_MSG` (`ds4_tp.c:129`, currently 16384), so this is a constant
change plus a matching recv-window arithmetic change.

Worth **≤ 4 µs × 86 = 0.34 ms/token (1.4%)**, low confidence, and **settled with no engine
change at all** by re-running M3's `uc_pingpong` with 4×4 KB chained against 1×16 KB.
Minutes. Do this before anything else in this report.

## 4. Peer lateness: cost and mitigations

### 4.1 The right accounting for a straggler that resynchronises every layer

Both ranks leave a gate together. Within a layer, rank *r* does `work_r`, then waits. So

```
token = Σ_layers [ max(work_A, work_B) + w + sw + ovh ]
```

and the *excess over perfect balance* is `Σ_layers E|s| / 2`, where `s = work_A − work_B`.
Per rank, the same quantity shows up in the profiler as
`exchange_FFN − exchange_ATTN = E|s| / 2` — the early rank pays `s`, the late rank pays 0,
so the rank-average of the extra wait is `E|s|/2`, and every other term (`w`, `sw`, `ovh`)
is identical between the two gate types. **That difference *is* the straggler's per-layer
cost to the token.** Nothing needs to be estimated or subtracted across runs.

### 4.2 Bounding the straggler from what is already measured (all post-M0-pinning)

`DS4_TP_GATE_PROFILE`'s row-gate `exchange` averages over **both** gate types, and the ATTN
gate is symmetric (each rank owns 32 of 64 heads, identical shapes, `attn_output_a` is
block-diagonal 4-of-8 groups and `attn_output_b` k-slices 4096 of 8192 —
`tp_decode_investigation.md:262-273`). So the ATTN gate carries no straggler and:

```
exchange_avg = w + sw + E|s|/4
23.6 µs (2k)  = 14.5–15.5 + sw + E|s|/4
```

With `sw ≥ 0`, this gives **E|s| ≤ 34.4 µs**, hence

> **Routed-expert straggler ≤ 43 × E|s|/2 = 0.74 ms/token ≤ 3.0% of the 24.34 ms 2k token.**

At a plausible `sw ≈ 4–5 µs` it drops to **E|s| ≈ 14 µs → 0.30 ms/token (1.2%)**.

### 4.3 This contradicts the uniform-router model, and the model loses

U14/§7 model the straggler as `E[max(k, 6−k)] = 3.9375` vs an ideal 3.0
(`TP-A0-ROWSPLIT-TEST-PLAN.md:1691-1735`, `tp_decode_investigation.md:346-352`). Their own
per-layer table implies:

| | heavy rank | light rank | max | mean | excess |
|---|---|---|---|---|---|
| U14 @131k, shared 50/50 | 0.1256 + 0.0191 | 0.0658 + 0.0191 | 0.1447 | 0.1148 | **0.0299 ms/layer** |

`0.0299 × 43 = 1.29 ms/token` of excess, i.e. **E|s| = 59.8 µs**. The measured exchange caps
E|s| at 34.4 µs. **The uniform model over-predicts the straggler by ≥ 1.7×**, and it
over-predicts it in a way the gate measurement cannot accommodate at any value of `sw`.

This is not a surprise — `tp_decode_investigation.md`'s own audit addendum flags it: "The
uniform-router `E[max]` assumption is also unverified against real selected-expert traces."
Real routers are not uniform; expert popularity is skewed and correlated across tokens, and
a contiguous 128/128 shard of a *popularity-skewed* router has lower per-token variance than
the uniform case if the popular experts happen to straddle the split. **The straggler is
real but smaller than modelled.**

Corroborating, from the other direction: `tp_decode_investigation.md:452-457` records that
the `router` ablation, which freezes the top-6 and therefore maximally unbalances the
contiguous shard, cost **+0.574 ms net of 92 removed dispatches ≈ 0.77 ms of added
straggler**. That is the *degenerate* case, and it is the same order as the measured
bound on the *normal* case — which is what you would expect if the normal case is
0.15–0.74 ms, and not what you would expect if it were 1.29 ms.

### 4.4 Re-sizing U14, and improving its design

U14 proposes shifting **all** of the shared expert to the routed-light rank. With
`S = 38.2 µs/layer` of total shared work and `δ = E|s|`:

| regime | best shift | resulting excess | win |
|---|---|---|---|
| `δ ≥ S` (δ ≥ 38.2 µs) | all of S to the light rank | `δ/2 − S/2` | `S/2` = **0.82 ms** |
| `δ < S` | **`x* = (δ+S)/2` to the light rank** | **0** | `δ/2` per layer = **43·δ/2** |

**The measured bound puts us in the second regime** (`δ ≤ 34.4 < 38.2`). Two consequences:

1. **U14's 0.82 ms is not achievable** — it exceeds the entire imbalance
   (`43·δ/2 ≤ 0.74 ms`). Re-size it to **0.15–0.74 ms, best estimate ~0.3 ms**, pending §4.5.
2. **But the correct design is strictly better than U14's**, because in this regime a
   *variable* shared split can drive the FFN-gate imbalance to **exactly zero** rather than
   merely reducing it. Same implementation problem U14 identifies — `k` is only known
   device-side after the router runs, so it needs either an indirect command buffer or both
   ranks dispatching the full shared grid with each threadgroup predicated on a device-side
   `k` — but the payoff is the whole straggler instead of a fraction of it.

Also note this competes with §2's Source 1: you can spend the shared expert as **ballast**
(this section, ≤ 0.74 ms, needs a device-predicated dispatch) or as **shadow filler**
(§2, ~0.53 ms, needs the concurrent encoder). Neither is bit-exact. **Ballast wins on
size and does not depend on §3.4's stale-payload risk.**

### 4.5 Does the ATTN gate cost less than the FFN gate? Currently unresolved — and it is the whole question

| | value | source |
|---|---|---|
| FFN gate | **31.4 µs** (1.351 ms / 43) | `ffn_hc_post − attn_hc_post`, same run, identical kernel and grid |
| ATTN gate | **9.3–31.4 µs** (0.4–1.35 ms / 43) | `docs/SCOPE-ATTNOUT-ROUTER-SHARED.md`, an unresolved range |

The ATTN gate's range is an *assumption band*, not a measurement — its upper end simply
assumes the FFN gate's value. The prediction from §4.1 is that the ATTN gate should be
`31.4 − E|s|/2` = **14–29 µs**, i.e. genuinely shorter, and the shortfall is exactly the
straggler. **If the ATTN and FFN gates measure equal, the straggler is zero and every
load-balance item in the queue — U14, §7 designs A/B/C, C3 — is dead.** That is a large
enough consequence to be worth resolving before anything else is built.

### 4.6 Two instruments, one run, no cross-epoch subtraction

**I1 — split the existing gate profiler by gate index (preferred; zero GPU perturbation).**
`req.gate` is already carried through the queue (`ds4_metal.m:10518`, values from
`ds4_tp.h:30-32`: `ATTN = 0`, `FFN = 1`). Add two buckets alongside the existing
`DS4_TP_STAT_ROW/VERIFY/BIG` kinds (`ds4_metal.m:9987-9996`) and print
`exchange` for each. ~6 lines. Yields `E|s|/2` directly, **with no stage timestamping and
therefore no 0.18 ms/marker tax and no command-buffer explosion.** Run on **both** ranks and
compare — a systematic difference between hosts is a different problem from a per-layer
straggler, and this separates them.

**I2 — add the ATTN gate stage marker** (`DS4_METAL_PROFILE_DECODE_STAGE("attn_tp_gate")`
immediately after the gate encode at `ds4.c:23856` and before the `attn_output` marker at `ds4.c:23870`, mirroring `ffn_tp_gate` at `ds4.c:25343`). Confirms I1 from the GPU
side and closes the 0.4–1.35 ms range. The marker tax is identical on both gates and
cancels in the difference.

**I3 — per-layer exchange histogram** (optional, same run). If the straggler is real, the
FFN exchange distribution should be bimodal per layer (this rank early / this rank late)
with a mode separation of `E|s|`, while the ATTN distribution should be unimodal. This
distinguishes "straggler" from "one host is just slower" definitively.

Run all three in one session with `DS4_TP_GATE_PROFILE=1` on both ranks and no stage
timestamping for the headline.

## 5. Where the gate budget actually goes

Against the **24.34 ms 2k token** (implied by the standing "4.34 ms needed for 50 t/s at 2k",
`TP-A0-ROWSPLIT-TEST-PLAN.md:311`). 86 gates/token (`ds4.c:60458`).

| component | ms/token | % of 2k token | status |
|---|---|---|---|
| **wire, one-way, 86 × 14.5–15.5 µs** | **1.25–1.33** | **5.1–5.5%** | **irreducible** at 16 KB on this fabric |
| dispatch + encoder breaks, 86 × ~4 µs | ~0.34 | ~1.4% | 2 dispatches/gate, both structural |
| routed-expert straggler (FFN gates only) | ≤ 0.74, est. ~0.3 | ≤ 3.0%, est. 1.2% | §4 — attackable, and over-modelled today |
| verbs/CQ/mutex software | residual | — | T1 already tried; a wash |
| **total gate spin** | **1.75–2.70** | **7.2–11.1%** | matches both independent measurements |

**More than half the gate spin is the hardware one-way latency of a 16 KB exchange that has
to happen 86 times.** The four levers on that line are: fewer gates (structurally impossible
— see below), a smaller payload (not bit-exact), a faster fabric, or **more tokens per
gate**.

**Fewer gates is closed.** Two reductions per layer is the structural minimum for this
sharding: the FFN cannot start before the attention sum because `ffn_hc_pre` → RMSNorm →
router are all nonlinear in the residual (`ds4.c:24000-24023`), and layer L+1 cannot start
before the FFN sum. Replicating attention to delete the ATTN gate is possible — the
attention weights are in the replicated ~8.2 GiB, not the routed shard — but it doubles
`q_b` + attn core + attn output, **+7.1 ms** against a ~0.9 ms saving. Recorded so it is not
re-proposed.

## 6. Recommendation

**Answer to the headline question: mostly no, and the part that is yes is small.**

The wait cannot be usefully overlapped with existing work, because there is none — the
decode token is a single dependency chain and every rank-local computation is upstream of
its own gate by construction (§2). Overlap can only be *manufactured*, by duplicating work
or by introducing a second sequence. And ~5% of the token — the wire floor — is irreducible
under any of these.

Ranked, cheapest-first:

| # | action | cost | value | gate |
|---|---|---|---|---|
| **0** | **I1/I2/I3 — split the gate profiler by ATTN/FFN and add the `attn_tp_gate` marker** (§4.6) | ~10 lines, one rig session | decides every item below; resolves M0-vs-31.4 µs on the instrument rather than by argument | none — do this first |
| **1** | **M3 `uc_pingpong`: 4 × 4 KB chained vs 1 × 16 KB** (§3.6) | zero engine change, minutes | ≤ 0.34 ms (1.4%) | none |
| **2** | **Variable shared-expert split to equalise rank arrival** (§4.4) — U14 corrected from "shift all" to "shift `(δ+S)/2`" | device-predicated dispatch or an ICB; not bit-exact (T2 bar) | ≤ 0.74 ms, est. **~0.3 ms**; drives FFN-gate imbalance to **zero** in the measured regime | I1 must show `exchange_FFN > exchange_ATTN` |
| **3** | **Shared expert into the FFN gate shadow via the concurrent encoder** (§2 Source 1, §3.1) | new TP2 wiring of existing machinery; not bit-exact; carries the §3.4 stale-payload risk, which is a *measured* failure mode | ~0.53 ms (2.2%) | only if 2 is blocked; mutually exclusive with 2 |
| **4** | **Amortise gates over more tokens** — speculative acceptance rate, multi-slot batching | large structural program | the only lever on the 1.3 ms wire floor | separate workstream |

**Do not do:** wider fence kernel (measured 21% vs 4%, `ds4_metal.m:10504-10507`);
`MTLSharedEvent`/`waitUntilCompleted` release (measured ~186 µs/gate,
`ds4_metal.m:9950-9952`); a bare encoder split (no semantic effect); replicating attention to
delete the ATTN gate (+7.1 ms for −0.9 ms).

## 7. Corrections to standing documents

1. **`TP-A0-ROWSPLIT-TEST-PLAN.md:701-706` — "wire is 6–7% of gate time" is wrong.** The
   denominator (`gpu-wait + exchange`) contains the whole token; the document says so itself
   two lines later and then uses the ratio anyway. Corrected: the exchange is **~65–80% of
   the gate**, and wire is ~60% of the exchange. §1.

2. **`TP-A0-ROWSPLIT-TEST-PLAN.md:826-834` — T1's post-mortem reason is wrong**, though its
   result stands. "The recv re-arm hoist cannot help a wait that is dominated by local GPU
   completion, not wire exchange" — the wait is *not* dominated by local GPU completion.
   T1 was a wash because the software component of the exchange is genuinely small, not
   because the exchange is a small part of the gate.

3. **`tp_decode_investigation.md:435` — strike the "concurrent decode encoder" row.** A
   `MTLDispatchTypeConcurrent` path exists and ships (`ds4_metal.m:1028-1030`, `:9526-9733`,
   live at `ds4.c:24985`/`:25222`); it is simply **disabled under TP2** because
   `fuse_shared_down_hc` requires `g->tp_world < 2` (`ds4.c:24182-24187`). §3.1.

4. **`tp_decode_investigation.md:346-352` / U14 — the uniform `E[max(k, 6−k)] = 3.9375` model
   over-predicts the straggler by ≥ 1.7×.** Its implied `E|s| = 59.8 µs/layer` is
   incompatible with the measured 23.6 µs rank-averaged exchange at any non-negative software
   cost. Re-size U14 from 0.82 ms to **0.15–0.74 ms** and change its design from "shift all"
   to "shift to equalise". §4.3–§4.4.

5. **The gate spin does *not* explain the 30 W decode / 60 W prefill reading.** 2.0–2.7 ms of
   a 24.34 ms token at ~zero power accounts for at most 8–11% of the envelope: it predicts
   ~53 W, not 30 W. The gate spin is a real correction to "GPU busy ⇒ useful work", and it is
   *a* contributor, but the bulk of the power gap is decode being latency-bound matvec
   against prefill's GEMM — which `tp_decode_investigation.md:151` already calls "normal".
   Do not use the power reading as evidence for the size of the gate spin; the two
   direct measurements are the evidence.

## 8. Provenance of every number used

All rig figures below are post-`iogpu.wired_limit_mb` pinning (M0, 2026-08-26). Nothing is
subtracted across that boundary.

| number | value | origin |
|---|---|---|
| FFN gate | 31.4 µs | `ffn_hc_post − attn_hc_post`, same run, `TP-A0-ROWSPLIT-TEST-PLAN.md:186-192` |
| ATTN gate | 9.3–31.4 µs | `docs/SCOPE-ATTNOUT-ROUTER-SHARED.md`, range not resolved |
| row-gate `exchange` | 23.6 µs @2k, 24.8 µs @131k | M0, `TP-A0-ROWSPLIT-TEST-PLAN.md:661` |
| one-way 16 KB | 14.5–15.5 µs p50 | M3, `TP-A0-ROWSPLIT-TEST-PLAN.md:826-831` |
| one-way 4 KB | 8.0 µs | M3, same |
| per-direction link ceiling | 4.4 / 4.1 GB/s | R10b |
| marginal dispatch | 1.9 µs | `tp_decode_investigation.md:337-341` |
| stage-marker tax | ~0.18 ms/marker | `TP-A0-ROWSPLIT-TEST-PLAN.md:203-215` |
| shared expert, per rank | 0.0191 ms/layer | U14, `TP-A0-ROWSPLIT-TEST-PLAN.md:1730-1735` |
| 2k token | 24.34 ms | implied by `TP-A0-ROWSPLIT-TEST-PLAN.md:311` |
| gates/token | 86 | `ds4.c:60458`, `ds4_tp.h:30-32` |
| gate payload | 16,384 B, one WR | `n_embd`(4096) × 4, `DS4_TP_RDMA_MAX_MSG = 16384` (`ds4_tp.c:129`) |
| event-release cost | ~186 µs/gate | `ds4_metal.m:9950-9952`, `metal/dsv4_misc.metal:7317` |
| fence grid cost | 4% (1 TG) / 21% (2 TG) | `ds4_metal.m:10504-10507` |

**Inference, marked as such:** the `E|s| ≤ 34.4 µs` bound (§4.2) assumes the ATTN gate is
straggler-free and that M3's probe latency transfers in situ; the ~4 µs gate overhead
estimate (§1, §5) is `2 × 1.9 µs` plus unmeasured encoder-break cost; the §2 Source-1
arithmetic assumes the replicated shared expert costs exactly 2× the split one.
**Instrument I1 replaces the first of these with a measurement and costs ~6 lines.**
