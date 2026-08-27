# BUG: TP worker "Metal model range ... is not covered by mapped model views" (2026-08-27)

Status: DIAGNOSED — root cause identified, fix applied (section 9).

All `file:line` references below are against the tree **with the section 9 fix applied**
(`ds4_metal.m` gained 10 lines at 42064, so pre-fix numbers in that file are 10 lower after that
point).

Branch: `upstream-metal-wins`. Rig: 2x M2 Ultra, TP2 over Thunderbolt RDMA, 50/50 routed-expert
split, DeepSeek V4 **Flash** (`DS4_SHAPE_FLASH`, `ds4.c:581`), MXFP4 routed experts.

Unrelated to `docs/TP-A0-ROWSPLIT-TEST-PLAN.md` (performance work) — nothing there is touched.

---

## 1. One-line answer

`ds4_gpu_routed_moe_batch_tensor()` rebases the routed-expert model offsets onto the rank's owned
expert half (`ds4_metal.m:42071-42073`) and then, when `n_tokens == 1`, delegates to
`ds4_gpu_routed_moe_one_tensor()` (`ds4_metal.m:42193`) — **before this change it passed the
already-rebased offsets** — and that callee rebases them a second time
(`ds4_metal.m:39457-39459`).

On rank 1 the rebase is `+128 * expert_bytes = +0.53125 GiB` per tensor, so a double rebase lands
`+1.0625 GiB` past the tensor base: exactly one whole expert blob too far, inside memory rank 1
does not map. On rank 0 `first_expert == 0`, so the double rebase is a no-op and the coordinator is
unaffected — which is why only the worker fails, and why the coordinator then sits on a TP gate
until the GPU watchdog fires.

The `n_tokens == 1` batch encode is reached because a **resumed** prefill aligns its first chunk to
a `g->prefill_cap` boundary (`ds4.c:36981-36988`); `cached=90111` is `4096*22 - 1`, so
`to_boundary == 1` and the first chunk is exactly **one token**. That is the intermittency.

---

## 2. Observed failure

Worker (rank 1):

```
tensor parallelism bound: rank 1, 50/50 expert split, rdma transport
tp worker ready for mirrored sessions
ds4: Metal model range 1.08..1.61 GiB is not covered by mapped model views
ds4: Metal model range 3.20..3.73 GiB is not covered by mapped model views
ds4: Metal model range 2.14..2.67 GiB is not covered by mapped model views
ds4: gpu layer 0 ffn batch encode failed
ds4: gpu whole-prefill layer 0 encode failed
tp worker sync: metal resumed prefill failed while extending checkpoint (session invalidated, worker continuing)
```

Coordinator (rank 0), same moment:

```
chat live continuation match=tool-output-ids ids=1 cached=90111 prompt=90598
chat ctx=90111..90598:487 TOOLS prompt start
chat ctx=90111..90598:487 TOOLS prefill chunk 0/487 (0.0%) ...
ds4: Metal command batch failed: Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)
tp: transport marked failed; this pair can no longer stay in sync and both ranks must be restarted
```

Message sources:
- `ds4: Metal model range ... not covered` — `ds4_metal.m:12287-12291`
  (`ds4_gpu_wrap_model_range`, coverage test at `ds4_metal.m:12281`).
- `ds4: gpu layer %u ffn batch encode failed` — `ds4.c:31937`.
- `ds4: gpu whole-prefill layer %u encode failed` — `ds4.c:36231`
  (the `!split_commands` single-command-buffer prefill branch, `ds4.c:36214`).
- `%s resumed prefill failed while extending checkpoint` — `ds4.c:63107`.

---

## 3. Model view geometry: how coverage is supposed to work

`ds4_gpu_add_model_view_range()` (`ds4_metal.m:1912-2055`):

- `page_model_offset = map_offset & ~(page-1)`, `mapped_model_size = round_up(leading + map_size, page)`
  (`ds4_metal.m:1931-1940`).
- `overlap = round_up(max_tensor_bytes, page) + page` (`ds4_metal.m:1967-1972`).
- `view_limit` = `maxBufferLength`, optionally capped by `DS4_METAL_MODEL_VIEW_MAX_GIB` or by a
  128 GiB default when a *single* span exceeds `maxBufferLength` (`ds4_metal.m:2002-2011`).
- Views step by `step = view_limit - overlap` (`ds4_metal.m:2014`).
- Lookup succeeds only if `[offset, offset+len)` lies wholly inside one view (`ds4_metal.m:12281`).

The overlap invariant is sound: with `V_k = [k*step, k*step + view_limit)` and
`k = floor(o/step)`, we get `o + L < k*step + view_limit - overlap + L`, so **any** `L <= overlap`
is covered. So the design guarantees coverage for spans up to `max_tensor_bytes`, inclusive.

Under TP the worker does **not** use one big range. `weights_model_map_sharded_spans()`
(`ds4.c:6485-6521`) emits, per layer, the replicated dense/attention/router tensors plus **three
isolated spans** — one per routed-expert blob — each covering only this rank's contiguous expert
half:

```c
const uint64_t low_experts   = x->dim[2] / 2;
const uint64_t first_expert  = rank == 1 ? low_experts : 0;
const uint64_t owned_experts = rank == 1 ? x->dim[2] - low_experts : low_experts;
const uint64_t owned_bytes   = owned_experts * expert_bytes;
const uint64_t lo            = x->abs_offset + first_expert * expert_bytes;
model_map_span_vec_append(spans, lo, lo + owned_bytes, true);   /* isolate = true */
if (owned_bytes > spans->max_tensor_bytes) spans->max_tensor_bytes = owned_bytes;
```

Those spans reach Metal through `ds4_gpu_set_model_map_spans()` (`ds4.c:60237`,
`ds4_metal.m:12130`), which clamps the per-span `effective_max` to `sizes[i]`
(`ds4_metal.m:12167-12168`). Each isolated 0.53 GiB span therefore becomes exactly **one** view of
its own size. A request for exactly that span is covered by construction.

---

## 4. The 0.53 GiB / 1.06 GiB arithmetic

DS4 Flash: `n_embd = 4096`, `n_ff_exp = 2048`, `n_expert = 256` (`ds4.c:585-597`).
Routed expert tensors are `[4096, 2048, 256]` MXFP4; MXFP4 is 32 elements per 17 bytes
(`gguf_types[39] = {"mxfp4", 32, 17}`, `ds4.c:2120`; `sizeof(block_mxfp4) == 17`, `ds4.c:859`).

```
row_bytes    = 4096/32 * 17            =         2176
expert_bytes = 2048    * 2176          =    4,456,448
blob         = 256     * expert_bytes  = 1,140,850,688  = 1.0625 GiB
rank-1 half  = 128     * expert_bytes  =   570,425,344  = 0.53125 GiB
```

So `0.53 GiB` = one rank's expert half and `1.06 GiB` = one whole expert blob. The three failing
ranges are each 0.53 GiB and start 1.06 GiB apart, i.e. three consecutive expert blobs of one layer
(layer 0, since encode aborts at the first layer).

Two important cross-checks:

- Both the mapper and the consumer compute `expert_bytes` identically:
  `tensor_expert_bytes()` uses `info->block_bytes` (`ds4.c:7972-7976`) and
  `routed_expert_row_bytes()` uses `routed_expert_block_bytes()` (`ds4.c:4519-4524`,
  `ds4.c:4505-4517`); for MXFP4 both are 17. There is **no** mapper/consumer stride mismatch.
- `ds4_gpu_tp_expert_range()` (`ds4_metal.m:9888-9901`) uses the same `n/2` split with rank 1
  taking the high half, matching `weights_model_map_sharded_spans` exactly.

**Therefore a single rebase produces a request byte-identical to the mapped span and cannot fail.**
The failure requires the request to be somewhere else. The three messages come from one site — `ds4_metal.m:40925-40927`, inside
`ds4_gpu_routed_moe_one_tensor`, which binds the whole owned expert range in gate, up, down order:

```c
gate_buf = ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, gate_tensor_bytes, &gate_inner);
up_buf   = ds4_gpu_wrap_model_range(model_map, model_size, up_offset,   gate_tensor_bytes, &up_inner);
down_buf = ds4_gpu_wrap_model_range(model_map, model_size, down_offset, down_tensor_bytes, &down_inner);
```

with `gate_tensor_bytes = down_tensor_bytes = n_bind_expert * expert_bytes = 0.53125 GiB`
(`ds4_metal.m:39509-39510`). So the log maps to `gate -> 1.08`, `up -> 3.20`, `down -> 2.14`.
Solving for the tensor bases under a **double** rebase (`base + 2*0.53125`):

```
gate base ~= 0.0175 GiB      request = base + 1.0625 = 1.08 .. 1.61
down base ~= 1.0775 GiB      request = base + 1.0625 = 2.14 .. 2.67
up   base ~= 2.1375 GiB      request = base + 1.0625 = 3.20 .. 3.73
```

Consecutive bases are 1.06 GiB apart (file order gate, down, up), and each request lands exactly on
the **rank-0 half of the next blob** — memory rank 1 never maps. Every number in the log is
reproduced.

Corroboration that the delegation really was taken: the only three sites that request a full
`n_bind_expert * expert_bytes` span are `ds4_metal.m:40925-40927` (inside
`ds4_gpu_routed_moe_one_tensor`) and `ds4_metal.m:42727-42741` (inside
`ds4_gpu_routed_moe_batch_tensor`). Both bind in gate/up/down order, so the message order alone
does not discriminate — but the batch site's offsets are single-rebased and therefore byte-equal to
the mapped span, so it cannot produce this message. The one_tensor site reached *through the
delegation* is the only one whose offsets are double-rebased. (`ds4_gpu_routed_moe_one_tensor`
called directly from the per-row TP loop at `ds4.c:31710` is also single-rebased and fine.)

---

## 5. Verdict on the `max_tensor_bytes` hypothesis: REFUTED

The lead was that `max_tensor_bytes` passed to view setup is smaller than the largest contiguous
span the routed-MoE path requests, making `overlap` too small so a 0.53 GiB span straddles a view
boundary. It does not hold:

1. Under TP the worker never goes through the stepped/overlapping-view geometry for expert data at
   all. Each owned expert half is an **isolated span** (`ds4.c:6516`, `isolate = true`, preserved
   through `model_map_span_vec_finish` at `ds4.c:6435-6452`) and becomes exactly one view sized to
   that span (`ds4_metal.m:2020-2027`). There is no boundary to straddle.
2. `max_tensor_bytes` for the sharded map is raised to `owned_bytes` (`ds4.c:6515-6517`), i.e. the
   0.53 GiB shard span itself, and is then clamped per span to `sizes[i]`
   (`ds4_metal.m:12167-12168`). It is not too small.
3. Even on the non-sharded fallback (`ds4_gpu_set_model_map_range`, `ds4.c:60247`),
   `max_tensor_bytes = model.max_tensor_bytes` is the largest single tensor = the full 1.0625 GiB
   blob, so `overlap` >= 1.0625 GiB and a 0.53 GiB request is covered with a full blob to spare.
4. A geometry bug would be deterministic in the tensor offsets, and the engine demonstrably runs
   layer-0 batch prefill successfully all day.

Widening `overlap` would have been exactly the papering-over the brief warned against: it would
have made the *wrong* read succeed silently, giving rank 1 the wrong experts.

The "sub-window" alternative is the right *shape* — rank 1 does map only sub-windows and the
request does fall outside this rank's window — but the cause is not a full-file/window offset
confusion in the resumed-prefill path. It is the double rebase.

---

## 6. Root cause

`ds4_gpu_routed_moe_batch_tensor()` (`ds4_metal.m:42023`), as it stood **before** the fix:

```c
/* ds4_metal.m:42057-42063, pre-fix */
uint32_t first_expert = 0, n_bind_expert = 0;
ds4_gpu_tp_expert_range(n_total_expert, &first_expert, &n_bind_expert);
const int32_t tp_expert_base_host = (int32_t)first_expert;
gate_offset += (uint64_t)first_expert * gate_expert_bytes;
up_offset   += (uint64_t)first_expert * gate_expert_bytes;
down_offset += (uint64_t)first_expert * down_expert_bytes;
```

then, for a single-token MXFP4 batch (`ds4_metal.m:42177-42190`: `gate_type == MXFP4 &&
down_type == MXFP4 && n_tokens == 1 && n_expert == 6 && n_total_expert >= 128 && !quality &&
pipelines present`):

```c
/* ds4_metal.m:42191-42220, pre-fix */
if (use_single_token_q4_one_tensor || use_single_token_mxfp4_one_tensor) {
    ...
    return ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, experts,
                                         model_map, model_size,
                                         gate_offset, up_offset, down_offset,   /* ALREADY REBASED */
                                         ...);
}
```

and `ds4_gpu_routed_moe_one_tensor()` (`ds4_metal.m:39410`) rebases again:

```c
/* ds4_metal.m:39455-39459 */
ds4_gpu_tp_expert_range(n_total_expert, &first_expert, &n_bind_expert);
const int32_t tp_expert_base_host = (int32_t)first_expert;
gate_offset += (uint64_t)first_expert * gate_expert_bytes;
up_offset   += (uint64_t)first_expert * gate_expert_bytes;
down_offset += (uint64_t)first_expert * down_expert_bytes;
```

Confirmations:

- The rebased offsets are **not read** between the rebase (`ds4_metal.m:42071-42073`) and the
  delegation at `:42193` — the delegation is the first consumer, so the fix is local and safe.
- All external callers pass raw `layer->ffn_*_exps->abs_offset` (`ds4.c:31757-31759`,
  `ds4.c:31670-31672`, `ds4.c:31714-31716`), so the only double rebase is this internal delegation.
- The defect was **known**. `ds4.c:31650-31657`:
  > "Restrict this to resident MXFP4: ... The n=1 batch wrapper also rebases rank-1 model offsets
  > twice, so all other cases retain the proven row loop."

  The TP DSpark verify branch guards itself with `n_tokens >= 2u` (`ds4.c:31658`) and falls back to
  a per-row `ds4_gpu_routed_moe_one_tensor` loop otherwise. The **plain prefill fallback** at
  `ds4.c:31750-31778` (`else if (ok)`) has no such guard, and that is the branch normal prefill
  takes, because `g->tp_batch_rows` is only non-zero inside the DSpark verify block
  (`ds4.c:37202-37206`, zeroed again at `ds4.c:37279`), so `tp_split_batch_moe`
  (`ds4.c:31639-31642`) is false during ordinary prefill.

Rank asymmetry: for rank 0 `first_expert == 0`, so `2*0 == 0` and the double rebase is invisible.
Only rank 1 is corrupted. (Inference, but forced by `ds4_gpu_tp_expert_range`'s definition.)

---

## 7. Why it only shows up here (the intermittency)

Three conditions must coincide:

**(a) The MoE call must be a *batch* call with `n_tokens == 1`.**
Decode goes straight to `ds4_gpu_routed_moe_one_tensor` via `metal_graph_encode_decode_layer`, so
decode is single-rebase and correct. Prefill with `n_tokens >= 2` never takes the delegation
(`n_tokens == 1` is required at `ds4_metal.m:42180`). So only a *one-token batch prefill encode*
trips it.

**(b) The prefill chunker must actually emit a 1-token chunk.**
`metal_graph_prefill_chunked_range()` (`ds4.c:36902`) aligns a *resumed* prefill to the
`prefill_cap` grid:

```c
/* ds4.c:36981-36988 */
const uint32_t remaining = end - pos0;
uint32_t local_cap = chunk_cap;
if (start != 0 && g->prefill_cap != 0) {
    const uint32_t mod = pos0 % g->prefill_cap;
    if (mod != 0) {
        const uint32_t to_boundary = g->prefill_cap - mod;
        if (to_boundary < local_cap) local_cap = to_boundary;
    }
}
const uint32_t chunk = remaining < local_cap ? remaining : local_cap;
```

For Flash, `g->prefill_cap == 4096` (`ds4.c:12282-12310`: any prompt > 4096 sizes the workspace at
the 4096 base for non-PRO). The failing session resumed at `start = cached = 90111`:

```
4096 * 22 = 90112      90111 % 4096 = 4095      to_boundary = 1      chunk = 1
```

So the very first chunk of that 487-token resume is **exactly one token**, which is precisely what
the coordinator log shows: it prints `prefill chunk 0/487 (0.0%)` and then dies. The schedule is
`[1, 486]`.

This is a 1-in-4096 alignment: a resumed prefill only emits a 1-token chunk when
`checkpoint.len % 4096 == 4095`. The same hazard exists for a fresh prefill whose *last* chunk is
one token (`n_tokens % chunk_cap == 1`), also 1-in-4096. That is the whole explanation for
"deterministic offsets, yet the engine runs normally most of the time": the offsets are fixed but
the *call shape* almost never occurs.

(`g->prefill_cap` is set from `metal_graph_prefill_cap_for_prompt(ctx_size, prefill_chunk)` ->
`ds4_prefill_cap_for_prompt` (`ds4.c:12282`), stored at `ds4.c:17171`. With `--prefill-chunk` or
`DS4_METAL_PREFILL_CHUNK` set, the modulus changes but the 1-in-`cap` hazard is identical. The
runtime watchdog ladder (`ds4_prefill_watchdog_chunk`, `ds4.c:12273`) only shrinks `chunk_cap`; the
boundary alignment is computed against the unclamped `g->prefill_cap`, so it is unaffected — for
this prompt the ladder also returns 4096.)

**(c) The rank must own a non-zero first expert**, i.e. rank 1 only.

Note also a real (separate, benign-here) divergence: the worker takes the `!split_commands`
whole-prefill branch (`ds4.c:36214`) while the coordinator takes the chunked branch, because
`callback_split = display_progress != NULL && n_tokens >= 32` (`ds4.c:36183`) is true only on the
coordinator. With `n_tokens == 1` both ranks take the whole-prefill branch anyway, so this did not
contribute — but it does mean the two ranks routinely build different command-buffer shapes.

---

## 8. Reproduction

**Smallest faithful repro (no 90k chat, no tools, no second box).** The trigger is purely the
1-token resumed chunk, so any prompt that resumes at `len % 4096 == 4095` reproduces it:

1. Start the TP pair as usual.
2. Prefill any prompt of exactly **4095** tokens and let it produce a checkpoint.
3. Continue the same session with >= 1 more token (so `suffix >= metal_graph_resume_prefill_min_tokens()`
   and the resumed-prefill branch at `ds4.c:63074` is taken).
4. The first chunk is 1 token; rank 1 prints the three `not covered by mapped model views` lines
   and fails; rank 0 hits the GPU watchdog.

Any `checkpoint.len ≡ 4095 (mod 4096)` works; 4095 is just the cheapest.

**Single-box repro, no TP hardware.** `ds4_gpu_test_set_tp_expert_shard(1, 2)`
(`ds4_metal.m:9904`, declared `ds4_gpu.h:214`) forces rank-1 expert ownership in-process. Then a
single `ds4_gpu_routed_moe_batch_tensor(..., n_tokens = 1, ...)` with MXFP4 gate/down,
`n_expert == 6`, `n_total_expert >= 128` reads from `base + 2*first_expert*expert_bytes` instead of
`base + first_expert*expert_bytes`, which is directly observable as wrong output values (or, with a
sharded model map, as the "not covered" message).

**Unit-test gap that let this through.** `tests/test_mxfp4_metal.c:856-925` already exercises
`ds4_gpu_routed_moe_batch_tensor` under both TP ranks and reconstructs the full result from the two
partials — but only with `TP_TEST_ROWS = 5` (`tests/test_mxfp4_metal.c:23`), and its fixture uses
`N_TOTAL_EXPERT = 8` (`:16`), which fails the `n_total_expert >= 128` guard at `ds4_metal.m:42182`.
So the harness could never have taken the single-token delegation. **Closing the gap requires a
fixture with `n_total_expert >= 128`, `n_expert == 6`, and `n_tokens == 1`, run at
`ds4_gpu_test_set_tp_expert_shard(1, 2)`** — recommended follow-up; it is a new fixture rather than
a tweak, so it is called out here rather than bundled into the minimal fix.

---

## 9. Fix (APPLIED)

Applied to `ds4_metal.m` only, two hunks, ~10 lines including the comment. `make ds4` is clean
(no new warnings under `-Wall -Wextra`).

`ds4_gpu_routed_moe_batch_tensor`: capture the caller-supplied (un-rebased) offsets
before the TP rebase and hand *those* to `ds4_gpu_routed_moe_one_tensor`, whose contract is to
rebase them itself.

```c
/* the callee applies the same TP rebase, so it must receive the caller's
 * un-rebased tensor bases -- see the double-rebase note in ds4.c */
const uint64_t caller_gate_offset = gate_offset;
const uint64_t caller_up_offset   = up_offset;
const uint64_t caller_down_offset = down_offset;
gate_offset += (uint64_t)first_expert * gate_expert_bytes;
...
```

and at the delegation, pass `caller_gate_offset / caller_up_offset / caller_down_offset`.

Why this shape rather than moving the rebase below the delegation: everything after the delegation
(the Q4 expert-table path at `ds4_metal.m:42673-42687`, the `wrap_model_range` binds at
`:42727-42741`) legitimately needs the rebased values, and the rebase feeds
`tp_expert_base_host` which is also used later. Saving the originals is the minimal change with no
reordering risk.

**Effect on rank 0:** none — `first_expert == 0`, so the saved and rebased values are identical.
**Effect on rank 1, `n_tokens >= 2`:** none — the delegation is not taken.
**Effect on rank 1, `n_tokens == 1`:** the one-token kernel now binds the owned expert half instead
of reading 1.0625 GiB past it.

**What would falsify this being correct:** if any caller of `ds4_gpu_routed_moe_batch_tensor`
passed *pre-rebased* offsets, the fix would under-rebase the `n_tokens == 1` path. Checked: all
call sites pass raw `layer->ffn_*_exps->abs_offset` (`ds4.c:31670-31672`, `ds4.c:31714-31716`,
`ds4.c:31757-31759`; test harness `tests/test_mxfp4_metal.c:830, 887, 947`). Also
falsified if `ds4_gpu_routed_moe_one_tensor` ever stops rebasing internally — it must keep the
rebase, since every other caller relies on it.

**Not fixed here (deliberately):** the `n_tokens == 1` case is now correct, but the deeper design
smell is that "offsets are raw at the API boundary, rebased internally" is enforced only by
convention. A follow-up worth considering is to have `ds4_gpu_routed_moe_batch_tensor` delegate
*before* it rebases anything, so there is one rebase site per call chain by construction.

---

## 10. Second, separable defect: worker failure escalates to a pair-fatal GPU timeout

Sequence:

1. Worker: layer-0 FFN encode fails, session invalidated, **worker continues its loop**
   (`ds4_tp.c`, "tp worker sync: ... session invalidated, worker continuing").
2. Coordinator: still inside its own command buffer, waiting on the per-layer TP gate for a
   contribution that will never arrive.
3. macOS GPU watchdog fires: `kIOGPUCommandBufferCallbackErrorTimeout`.
4. `tp: transport marked failed; this pair can no longer stay in sync and both ranks must be
   restarted` — an operator-visible full restart for what was a single recoverable request.

**Assessment: yes, the worker can and should report this over the control plane.** The pieces
already exist:

- `ds4_tp_control_lock` / `ds4_tp_control_unlock` (`ds4.h`) already serialize the control channel —
  "at most one command is in flight per rank" — so a worker-originated `EVAL_FAILED` frame has a
  well-defined slot: the worker holds the control lock, sends the failure, and the coordinator's
  next control read gets it instead of blocking.
- `ds4_tp_mark_failed` already exists to move the pair into a failed state deliberately rather than
  via a watchdog.
- `metal/dsv4_misc.metal:7333` already implements the right pattern on the GPU side: the bounded
  fence latches a timeout into shared memory instead of wedging the shader. The coordinator's
  host-side gate wait needs the same treatment — a bounded wait that returns "peer failed" rather
  than an unbounded wait that becomes a 5-second command-buffer timeout.

Sketch of the clean failure path (not implemented in this change):

1. Worker encode fails -> worker publishes a failure code into the shared TP slab flag word for the
   current `(layer, gate, seq)` **and** sends a control-plane `EVAL_FAILED{seq}` frame.
2. Coordinator's gate wait is bounded and checks the failure word; on failure it aborts encoding,
   ends the command buffer without submitting the stalled work, and fails **the request**.
3. Both ranks resynchronise at the next `ds4_session_sync()` — the worker already invalidates its
   session, so the coordinator only needs to invalidate its own checkpoint and let the next sync
   replay a cold prefill.
4. `ds4_tp_mark_failed` is then reserved for genuine transport faults, not for recoverable
   per-request compute failures.

**Scoping.** This is a separate change: it touches the TP wire protocol (a new control frame), the
host-side gate wait, and the session-invalidation contract on both ranks, and it needs its own
tests (induced worker-side encode failure; assert the coordinator fails the request and the pair
stays usable). It should not ride along with the one-line offset fix, which is independently
correct and removes the only known trigger.

Worth noting: with the section 9 fix in place, the specific escalation observed here no longer has
a trigger — but any future worker-side encode failure will escalate the same way, so the second
defect is real and independent.
