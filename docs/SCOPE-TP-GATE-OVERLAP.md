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

_TBD_

## 3. Can the fence be replaced or restructured?

_TBD_

## 4. Peer lateness: cost and mitigations

_TBD_

## 5. Recommendation

_TBD_
