# Review: live-KV / tool-binding / cancel-rewind changes on `tp-multi-slot-batching`

Date: 2026-08-24
Scope reviewed: commits from 2026-08-23 (`22568ab`, `b8cb074`, `f8b73c6`, `b5c5a24`,
`ed63073`, `c1d4597`), the adjacent TP batching commits the branch is named for
(`a58f787`, `4bbfa88`, `e8cc8cd`), and the uncommitted working-tree doc changes.
Code delta: `ds4_server.c` +218/-6, two new docs, ~790 lines of uncommitted doc edits.

Method: seven focused reviewers over disjoint dimensions, adversarial verification of
every bug-shaped claim, plus independent spot-verification of each finding cited below.
45 findings survived; 0 were refuted.

## Status (2026-08-24, uncommitted)

**Fixed:** P0-1..6, P1-7..9, P1-12, P3-20, the TP row-cap item from P3-24, the stale TP
comment, and P4-25/26/29/30 (plus the dangling hashes across all six docs, not just the
committed ones). Tests added: `test_chat_live_tail_renders_tool_results_only`,
`test_chat_live_match_requires_visible_prefix`,
`test_chat_render_is_append_only_across_tool_turn`, and two asserts on the
`should_canonicalize_tool_checkpoint` gate. Build is warning-clean; `./ds4_test --server`,
`ds4_agent_test` and the eval extractor self-tests pass. (`make test` still stops at the
long-context stage, which needs `ds4flash.gguf` — not present in this tree, pre-existing.)

**Not done, and why:**
- P1-10 (cache-hit counters), P1-11 (stop-string rewind), P2-14..16 (head-stable
  rendering, generalized rewind-to-common, evict-store suppression), P2-13
  (`job_required_slot_locked` chat arm) — each is a design decision about behaviour, not a
  mechanical correction. P2-14 is the only one that actually restores hits under
  context-mode.
- The rest of P3-24 (TP fence latch, orphaned control frames, service-thread deadlines) —
  GPU/RDMA synchronisation semantics; wrong guesses there hang the pair.
- P3-22/23 (e2e cache-hit script, ASan target) — new build/test infrastructure.
- P4-27/28/31 — the PP-branch contradiction and the `rdma_en6`/`rdma_en7` host mapping
  live in ~790 lines of uncommitted in-progress doc edits, and the device→host binding is
  a physical fact about the rig that cannot be settled from the source. See the note at
  the end of §7.

---

## 1. The `pi context-mode` prefix problem — root cause

**Confirmed, and it is an architectural property of the cache, not a bug in the new code.**

What the extension does (from its own docs at https://pi.dev/packages/context-mode):

- Injects a "routing block" into the **system prompt** at session start.
- Re-injects a ~250-token guidance block on a **cadence of every N tool calls**
  (default 10, env-tunable to every call).
- After compaction, injects a `<session_knowledge>` XML snapshot (≤2 KB) rebuilt from
  **git state, cwd, active files, todo state, the user's last prompt, and per-tool call
  counts** — i.e. content that changes run to run.

Why that costs a full prefill on every request:

1. `render_deepseek_chat_prompt_text` concatenates all system/developer content into a
   single block at the **front** of the rendered prompt, right after the tool schemas
   (`ds4_server.c:2464`). So the volatile bytes land in the leading region.
2. Every cache tier is **all-or-nothing on the prefix**:
   - Live token path: `cached = common == old_pos && prompt.len >= old_pos ? common : 0;`
     (`ds4_server.c:11422`). If the prompt diverges *anywhere* before the end of the live
     checkpoint, `cached` is **0** — the valid `[0, common)` tokens are thrown away too.
   - Engine level: `ds4_session_sync_internal` reuses the checkpoint only when
     `ds4_tokens_starts_with(prompt, &s->checkpoint)`; otherwise `s->checkpoint.len = 0`
     and it prefills from token 0 (`ds4.c:61890`, `ds4.c:61900`).
   - Live text path: `byte_prefix_match` from byte 0 (`ds4_server.c:9797`).
   - Disk path: SHA1 over `prompt_text[0..text_bytes)` — keyed from byte 0.
3. The one partial-reuse escape hatch, `live_prefix_rewind_target`
   (`ds4_server.c:10167`), is gated on `ds4_engine_is_glm_dsa()` **and** only fires when
   the new prompt is a strict *prefix* of the live state (`common == prompt_len &&
   prompt_len < old_pos`). It does nothing for divergence-in-the-middle.

Net: a 500-token volatile block at the head of a 40k-token conversation invalidates
**100%** of the KV, every request. A constant prefix is harmless; a varying one is total.

### The part that is new and is a correctness risk, not just a perf risk

`f8b73c6` step 4 added `chat_live_*` for plain `/v1/chat/completions`. On a tool-result
turn it binds purely on `(tool_call ids set equal) && (live_tokens == old_pos)`
(`chat_live_matches_request`, `ds4_server.c:8957`) and then **replaces `j->req.prompt`
with `live_tokens + rendered tail`** (`ds4_server.c:11386`). The rendered tail
(`render_deepseek_live_tool_tail`, `ds4_server.c:2658`) **skips system/developer messages
entirely**.

So with context-mode active there are now two failure modes depending on turn shape:

| Turn shape | Outcome |
|---|---|
| Tool-result turn, ids match | `chat_live` binds → **cache hits**, but the model never sees the refreshed routing block / session guide. Silent staleness. |
| Any other turn | `common` ≈ 0 → `cached = 0` → **full re-prefill**. |

The Responses and thinking paths do not have this hole — they require
`byte_prefix_match(req->prompt_text, ..., live.visible_text, ...)` before reusing the KV
(`ds4_server.c:9928`, `9971`). The Anthropic path shares the hole but predates this branch.

### Diagnosability

Both cases log the same thing. `trace_cache_miss_reason` (`ds4_server.c:10051`) has no
band for head divergence, so

```
live kv cache miss live=40000 prompt=40500 common=300   reason=token-mismatch ...
live kv cache miss live=40000 prompt=40500 common=39998 reason=token-mismatch ...
```

are indistinguishable in shape — 99.3% loss and 0.005% loss read identically. And the
`chat_live` hit path emits **no log line at all**: the PREFILL chain at
`ds4_server.c:11517-11535` has arms for responses / anthropic / thinking but none for
chat, and `chat_live_match_ids` is computed at 11382 and never read.

### Second-order cost

On `cached == 0` with `--kv-disk-dir` set, the server writes an `"evict"` snapshot of the
doomed checkpoint (`ds4_server.c:11465`) and then a fresh cold snapshot — both keyed by
text containing that request's unique prefix, so neither can ever be hit again. A varying
prefix turns the disk cache into a self-evicting write amplifier.

---

## 2. Cancel-rewind (`22568ab` + `b5c5a24`): correct, but applied at 1 of 5 exits

`b5c5a24` is right: `committed_frontier = ds4_session_pos()` captured at
`ds4_server.c:11642` after the prompt sync, rewound at 12049. Every path that reaches
decode passes through the capture, and the two `goto decode_again` re-entries leave it at
the pre-recovery frontier, which is conservative and correct (the tool-error suffix is
server-injected and never replayed by the client). `ds4_session_rewind` mirrors to the TP
worker with an identical clamp, so leader/worker stay symmetric.

**The gap:** `generate_job_inner` has five post-generation cancel exits. Only 12049
rewinds. The exits at **12173, 12295, 12334 and 12480** call `request_live_state_clear`
(which only zeroes the four `*_live` binding structs — it never touches the session) and
return with the whole abandoned completion still in the live KV. That is exactly the
condition `22568ab` set out to remove.

Reachable **deterministically**, no race needed: a plain streaming chat request finishes
decode normally, passes the 12040 check, then the final held-tail `sse_chunk` at 12167
fails because the client is gone → `job_mark_cancelled` → falls into 12173 → no rewind.
Racily at the other three via the 100 ms disconnect poller (`ds4_server.c:13043`).

Note the stop-string path at 12024 is *not* affected — it already calls
`ds4_session_invalidate`. (Though it could rewind to `committed_frontier` instead of
wiping, which would preserve the prompt KV; see action item P1-3.)

**Separately (`ds4_server.c:11630`):** a *cancelled* (not failed) prefill takes the
`server_session_sync != 0` branch, where `kv_cache_discard_failed_disk_entry` runs
**before** the `job_cancelled(j)` check. It unlinks the disk snapshot the session was
loaded from, zeroes `continued_last_store_tokens`, and calls `ds4_session_invalidate` —
destroying both cache tiers, even though `ds4.h:356-359` documents an interrupted sync as
leaving a valid token prefix. ESC on a long prefill therefore makes the *next* request
strictly worse than if the cancel had never happened. (The sibling branch at 11593 does
not have this bug — it is reachable only when `cached == 0`, so `disk_cache_path` is
always NULL there.)

---

## 3. `trunc_raw_block` (`f8b73c6` step 2, `ed63073`)

**`ed63073` is complete.** The other `goto decode_again` at 12140 cannot carry a live
block: it runs only when `completed_truncation == false`, and that is the same condition
that gates the allocation. Ownership transfer at 12320-12322 is a genuine move
(`tool_memory_put_locked` copies), so no double free. `effective_prompt` is freed exactly
once on all 16 exits. All allocators are `x*` wrappers that `die()` on OOM.

**Two real defects in the captured *span*,** both from re-deriving the block boundary with
a raw scan instead of reusing the parser's:

1. **Missing `\n\n` separator.** `find_any_tool_start(text.ptr)` (`ds4_server.c:12091`)
   starts at the `<` of the tag; the DSML parser sets `raw_block_start = start - 2` when
   preceded by `\n\n` (`ds4_server.c:5136-5138`). The remembered block is two bytes short,
   so on the next tool-result request the prompt renders `OK<｜DSML｜tool_calls>` instead
   of `OK\n\n<｜DSML｜tool_calls>`, the byte-prefix check in `live_text_prefix_prompt`
   fails, and the turn re-prefills — the exact miss step 2 was written to prevent.
2. **`</think>` boundary ignored.** `try_repair_dsml` (5282-5287) and both parsers restrict
   the tool scan to after the last `</think>`; line 12091 scans from offset 0. In think
   mode, a model that mentions an unclosed tool tag inside its reasoning anchors
   `trunc_raw_block` inside the reasoning section.

Both are DeepSeek/DSML-path only — `try_repair_dsml` does not recognise GLM's `<tool_call>`.

---

## 4. Multi-slot batching gaps (this is the branch's own theme)

- `job_required_slot_locked` (`ds4_server.c:12592`) pins `responses_live` and
  `anthropic_live` continuations to their owning slot. There is **no `chat_live` arm** and
  no `chat_requires_live_tool_state` flag, so with `--resident-sessions`/`--batched-sessions`
  a chat tool-result request is routed by generic prefix score and routinely lands on a
  slot with no binding — the new fast path is effectively single-slot-only.
- Fallback scoring uses `ds4_session_common_prefix`, which a front-injected varying prefix
  zeroes out, so routing also degrades to LRU under context-mode.
- No log line carries the slot id, so on a multi-slot server a routing miss and a rendering
  miss produce byte-identical log output.

---

## 5. Ordering: `chat_live_remember` runs before canonicalization

`chat_live_remember` snapshots `ds4_session_pos()` at 12390; `canonicalize_tool_checkpoint`
at 12413 then rewrites the session to a different length. The recorded `live_tokens` is
stale. Consequence is mostly benign (the binding self-clears at 11656 and the canonicalized
session is a full prefix hit on the plain token path), but if the rewrite happens to land
on the same token count the stale binding matches and the suffix renderer prepends a second
EOS. Cheapest fix: `chat_live_clear` after any canonicalization that rewrote the session.

---

## 6. TP batching (`a58f787`, `4bbfa88`) — pre-window but this branch rides on it

- **Sticky fence latch (`ds4_metal.m:10609`).** `g_tp_fence_timeout_word` is set to 1 by
  `kernel_dsv4_tp_fence_wait` on a bounded-wait miss and is cleared *only* in
  `ds4_gpu_tp_init` (10215, once at engine bind). After one transient stall — a long peer
  prefill chunk, a page fault, an RDMA retransmit — `ds4_gpu_tp_failed()` returns true
  forever and every subsequent batched decode invalidates all rows in the batch. It is a
  process-wide latch shared by the row gate (10435) and the batch gate (10509). This
  presents to an operator as a permanent prompt-cache regression, not as a TP fault.
- **Orphaned control frames (`ds4.c:64651`, `64896`).** Any leader-local failure *after*
  `EVAL_BATCH`/`MIXED_BATCH` has been sent skips the ack and logits reads while the healthy
  worker has already written them, leaving the control stream offset by whole frames.
- **No deadline on service-thread arrival waits (`ds4_metal.m:10035`).** All four loops exit
  only on shutdown; a failed command buffer wedges the gate service with up to `86*N` queue
  entries undrained.
- **Row cap derived from the wrong shape (`ds4_server.c:13600`).** The `--batched-session`
  validation hardcodes 86 gates/row (43-layer Flash). The 61-layer Pro shape is 122, so a
  32-row step queues 3904 into a 4096-deep queue — the real ceiling is 33 rows, not ~47.
- **No automated coverage.** `test_metal_session_batch` is not in `make test`, needs
  `DS4_TEST_TP_MODE` plus a second host, caps at 16 rows, and never sets
  `DS4_METAL_FAST_SYNC` — so the entire content of `4bbfa88` is untested.
- Stale comment at `ds4_server.c:13717` still says batched mode is refused under TP.

---

## 7. Docs

New (`b8cb074`):

- **`docs/TP-PREFILL-LONG-CTX-INVESTIGATION.md:14`** — asserts TP splits attention *heads*
  during prefill. It does not. `tp_split_attn` is the decode encoder; prefill splits *rows*
  via `tp_row_split_attn`, and only when `pos0 == 0` (`ds4.c:28905`, `const bool zero_prefix
  = pos0 == 0;`). For a 131072-token prefill chunked at 4096, only chunk 0 row-splits — the
  other 31 chunks run the whole attention path replicated on both ranks. This changes the
  ranking of the fix options: lifting the `zero_prefix` restriction is likely a larger and
  cheaper win than Option A (split the indexer top-k). The doc's claim that the indexer
  top-k *is* replicated is correct (`ds4.c:28916`).
- **`ds4.c:28893` citation is off by 13-26 lines** — that line is
  `if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;`. The quoted comment is at
  28906-28919. Anchor on the function name instead.
- **`BENCHMARKS-TP-PP.md:40`** — the TP command uses `--ctx-start N --ctx-max N` with no
  `--step-mul`, so it cannot produce the documented "2048→131072, step ×2" table; with those
  flags `step_incr` defaults to 2048 (`ds4_bench.c:214`), a 64-point linear sweep. The PP
  command directly below *does* carry `--step-mul 2`.
- **`BENCHMARKS-TP-PP.md:123`** — power figures quoted at ~180k context, but no documented
  sweep exceeds 131072, and no measurement command / per-node-vs-pair basis is stated.
- **`docs/TP-PREFILL-LONG-CTX-INVESTIGATION.md:92`** — the verification step points at
  `ds4.c:13532`, which holds a different format string; the `indexer=`/`attn_rows=` line is
  13536 and is gated on `DS4_PREFILL_PROFILE_TOKEN`, which the doc never mentions.

Cross-document:

- **Nine cited commit hashes are dangling.** `40caa71`, `2fa324b`, `762886a`, `4349dbf`,
  `7960eaa` and four more resolve locally (reflog keeps them alive) but
  `git branch -a --contains` returns nothing for any of them — they were rebased away and do
  not exist in a fresh clone. `gguf-tools/README.md:195` ("use only with `tp-mtp-hunt` commit
  `40caa71` or a descendant") is unverifiable.
- **Opposite PP branch instructions.** `BENCHMARKS-TP-PP.md:34` says `apple-rdma-pp` is
  54 commits behind and must not be used, pointing at `pp-rdma-new`. The uncommitted
  `speed-bench/README.md` (65/75/83/221), `README.md` (457/662) and `QA_BEFORE_RELEASES.md`
  (573) all send PP work to `apple-rdma-pp` and never mention `pp-rdma-new`.
- **Opposite RDMA device→host mapping.** `BENCHMARKS-TP-PP.md` puts `rdma_en6` on the
  coordinator and `rdma_en7` on the worker; `speed-bench/README.md:131` reverses both, for
  both TP and PP. `--rdma-device` names the *local* verbs device (`ds4_tp.c:394`) and Apple
  RDMA is point-to-point per cable, so this is not interchangeable.
- **`speed-bench/README.md:54`** is dated 2026-08-19 and its "authoritative" branch topology
  never mentions `tp-multi-slot-batching`, sending readers to `tp-mtp-hunt` — an ancestor
  that lacks every fix in this window.

Hygiene: `.gitignore` covers neither `*.trace/` (11 Instruments bundles, **712 MB** at repo
root) nor `.claude/` nor the two `gguf-tools/tests/` binaries that the newly-documented
validation checklist tells people to build. One `git add -A` publishes all of it.

**Two doc items need a human decision and were deliberately left alone:**
1. *Which PP branch is current.* `pp-rdma-new` demonstrably contains the newer commits
   (`180e853`, `ecea7be`) on top of `4bbfa88`, so `BENCHMARKS-TP-PP.md` is almost certainly
   right and the four other docs are stale — but they are mid-edit in the working tree and
   rewriting that prose is the author's call.
2. *The `rdma_en6`/`rdma_en7` → host mapping.* `BENCHMARKS-TP-PP.md` is internally
   consistent (host table and both launch commands agree: `rdma_en6` = lanfear/coordinator,
   `rdma_en7` = mat/worker); `speed-bench/README.md` reverses both. One of them is wrong,
   but which is a fact about the physical cabling and cannot be determined from the source.
   Confirm on the rig, then fix the loser and the stray `rdma_en1` at `README.md:745`.

---

## 8. Tests

Zero new tests landed with 218 lines of state-machine logic. Concretely:

- `chat_prepare_live_continuation` is a line-for-line clone of
  `anthropic_prepare_live_continuation`, which **has** a unit test
  (`test_anthropic_live_tail_renders_tool_results_only`). The clone shipped with none.
- `test_batched_live_continuation_slot_binding` asserts exactly the pinning behaviour that
  `chat_live` is missing from, and was not extended.
- `should_canonicalize_tool_checkpoint` is a pure function with an existing test; the new
  `dsml_recovery_attempted` input was OR'd in at the call site (12404) instead of added to
  the predicate, putting the new decision on the untestable side of the seam.
- Both bugs already found in-window (`b5c5a24` rewind-to-0, `ed63073` stale block) were
  caught by human review. There is no ASan/LSan target (`grep fsanitize Makefile` → nothing)
  and no CI, so neither would have been caught mechanically.
- No metric for the headline number: `grep cache_hit|hit_rate|metrics` in `ds4_server.c`
  returns zero. The miss WARNING is suppressed when `old_pos == 0`, so cold-slot misses —
  the most expensive kind — produce no log line and cannot even be counted by scraping.
- The only server-level script (`tests/test_server_batching.py`) issues no tool calls and
  injects a fresh nonce specifically to *defeat* caching.

---

## Action items

### P0 — do before the next release

1. **Rewind at all five cancel exits.** Factor `request_live_state_clear` +
   mutex-guarded `ds4_session_rewind(slot->session, committed_frontier)` into one helper;
   call it from 12040, 12173, 12295, 12334, 12480. (`ds4_server.c`)
2. **Don't destroy caches on a cancelled prefill.** Move the `job_cancelled(j)` /
   `rc == DS4_SESSION_SYNC_INTERRUPTED` test *above* `kv_cache_discard_failed_disk_entry`
   at `ds4_server.c:11630`. Keep the disk file, keep `continued_last_store_tokens`, keep the
   partial checkpoint.
3. **Close the `chat_live` correctness hole.** Give it a visible-text key like
   `responses_live`/`thinking_live`: record `prompt_text + suffix` at
   `chat_live_remember`, and require `byte_prefix_match(req->prompt_text, ...)` in
   `chat_live_continuation_prompt` before reusing the KV. Keeps the hit for a genuine
   append-only continuation; degrades correctly the moment a client rewrites the head.
4. **Fix the `trunc_raw_block` span.** Extend backwards over `\n\n` and start the scan after
   the last `</think>` — or better, extract the parser's `raw_block_start` computation into a
   shared helper called from 4958, 5136 and 12091 so they cannot drift again.
5. **Add `live_tool_state_free(&slot->chat_live);`** to `server_close_resources`
   (`ds4_server.c:13290`).
6. **`.gitignore`**: `*.trace/`, `.claude/`, the two `gguf-tools/tests/` binaries. 712 MB is
   one `git add -A` away from being permanent.

### P1 — makes the next incident diagnosable in one glance

7. **Classify head divergence in `trace_cache_miss_reason`** (`ds4_server.c:10051`): return
   `leading-block-divergence` when `common * 8 < old_pos` (reusing the threshold
   `job_slot_score` already uses at 12641), `mid-prefix-divergence` otherwise, keep
   `token-mismatch` for near-tail. On that band, log the decoded text around `common` —
   `trace_write_escaped_bytes` (10066) already exists. This one change would have identified
   context-mode from a single log line.
8. **Add the missing chat arm to the PREFILL log chain** (`ds4_server.c:11527`) and make
   `chat_live_match_ids` meaningful. One line.
9. **Add `slot=%d` to the miss WARNING (11449) and the prompt-start line (11549).** On a
   multi-slot server this is the only discriminator between a routing miss and a rendering
   miss, and it is free.
10. **Aggregate cache stats.** `{requests, hits, cached_tokens, prefilled_tokens,
    by_source[]}` bumped at 11491, dumped on SIGUSR1 or at shutdown. Turns "did cache-hit
    rate improve" from log-scraping into a number, and immediately shows whether
    `chat-tool-output` ever fires.
11. **Rewind instead of invalidate on stop-string hit** (`ds4_server.c:12024`) —
    `committed_frontier` now exists and gives the same safety without wiping the prompt.
12. **Clamp `slot->continued_last_store_tokens` at every rewind site** (12049 and the GLM
    one at 11413), or fold it into the rewind helper from P0-1.

### P2 — mitigations for the injected-prefix class

13. **Add a chat arm to `job_required_slot_locked`** (`ds4_server.c:12592`), or at minimum a
    soft affinity band in `job_slot_score` keyed on
    `live_state_contains_all(&slot->chat_live, &r->chat_live_call_ids)`. Without it the new
    fast path does not work on the branch that adds multi-slot batching.
14. **Head-stable rendering (the only fix that actually restores hits).** An opt-in flag that
    relocates client system/developer content emitted after the first user turn to a trailing
    turn, keeping the leading region byte-stable. Pure rendering change in
    `render_deepseek_chat_prompt_text` / `render_glm_chat_prompt_text`, zero prefill cost.
    Document that the existing `--kv-cache-boundary-*` knobs are **tail-only** and cannot help
    here.
15. **Generalize rewind-to-common** for `common > 0 && common < old_pos` on backends where
    `ds4_session_rewind` is sound (the cancel path already relies on it for non-GLM). Honest
    caveat: when the divergence is at the *head* this saves ~1% — it is worth doing for
    mid-transcript edits and compaction, but it is not the context-mode fix.
16. **Suppress the doomed `"evict"` disk store** when `common * 8 < old_pos`
    (`ds4_server.c:11465`) — a leading-divergence miss is direct evidence the snapshot will
    never be hit.
17. **Client-side, worth telling users:** set the context-mode re-injection cadence as high as
    tolerable, and prefer a stable routing block over the dynamic session guide, until 14
    lands.

### P3 — tests and TP

18. Clone `test_anthropic_live_tail_renders_tool_results_only` for the chat path (~35 lines);
    add negative cases for a partial parallel-result tail and a trailing developer message.
19. Extend `test_batched_live_continuation_slot_binding` with a `chat_live` stanza — asserting
    `-1` with a comment if the decision is "do not pin".
20. Change `should_canonicalize_tool_checkpoint` to take `bool recovery_attempted` and add two
    asserts to its existing test.
21. Extract `cancel_rewind_target(...)` as a pure function next to `live_prefix_rewind_target`
    and table-test the `b5c5a24` case explicitly.
22. Add `tests/test_server_tool_cache.py`: a 4-turn scripted agent loop asserting
    `usage.prompt_tokens_details.cached_tokens >= 0.9 * prev_prompt_tokens` on turns 2-4, plus
    three variants — constant injected prefix (must hit), abort-then-resend (must hit), and
    varying injected prefix (documents the expected behaviour).
23. Add a `make test-asan` target. Both in-window bugs were lifecycle bugs found by eye.
24. **TP:** clear or escalate `g_tp_fence_timeout_word`; drain the worker's ack/logits (or
    `ds4_tp_mark_failed`) on the `!ok` path at `ds4.c:64651`/`64896`; give the four
    service-thread arrival waits a deadline; derive the row cap from
    `ds4_engine_tp_gate_schedule` after engine open instead of hardcoding 86; raise
    `MAX_SESSION_COUNT` to 32 and add a `DS4_METAL_FAST_SYNC=1` QA arm.

### P4 — docs, before committing the working tree

25. Fix the prefill row-split vs head-split claim and re-rank the improvement options.
26. Replace all nine dangling hashes with reachable ones; prefer branch/tag or a feature-detect
    over a hash in `gguf-tools/README.md`.
27. Pick one PP branch name and update all four docs together; replace the hardcoded
    "54 commits behind" with a `git rev-list --count` instruction.
28. Reconcile the `rdma_en6`/`rdma_en7` host mapping and label commands by hostname, not role.
29. Add `--step-mul 2` to the TP sweep command; replace the `--gen-tokens N` placeholder.
30. Anchor line citations on function names; fix the profiling instruction and name
    `DS4_PREFILL_PROFILE_TOKEN`.
31. Add `tp-multi-slot-batching` to the `speed-bench/README.md` topology as the current tip and
    bump its date; state the power measurement command and context.
