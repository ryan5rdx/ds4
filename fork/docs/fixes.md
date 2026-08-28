# Fixes

Correctness, robustness and operability changes this fork carries that make no
throughput claim. Speedups are in [`optimizations.md`](optimizations.md); the
measurement record is in [`benchmarks.md`](benchmarks.md).

Each entry states the symptom first, because most of these were found from a
symptom rather than from reading the code.

---

## Serving: live KV cache

The single largest practical improvement in this fork does not appear in any
benchmark. A fixed-prompt sweep re-prefills once and measures steady-state
decode; an interactive session with tool calls re-prefills constantly. These
changes address the second case.

### A tool call discarded the whole live cache

*Symptom.* Every tool call in an agent loop paid a full prompt re-prefill. At
long context this dominated the turn.

*Cause.* A chat turn that ended in a tool call invalidated the live cache and
started a new session for the tool result, rather than continuing the existing
one.

*Fix.* A tool result continues the existing session. The turn rewinds to the
committed prefill frontier instead of to token zero.

### Cancellation rewound to the wrong length

*Symptom.* A cancelled turn re-prefilled its entire prompt on the next request,
even though the cache was intact.

*Cause.* The rewind targeted the requested prompt length rather than the
committed prefill frontier. Those differ whenever prefill is chunked, which is
always at long context, so the rewind landed short of what had actually been
committed and the next request could not match a prefix.

*Fix.* Cancellation rewinds to the committed frontier. The same correction
applies to the truncation path.

### Tool-error recovery re-entry reused stale truncation state

*Symptom.* A turn that hit a tool error and re-entered recovery could mismatch
its cache.

*Cause.* The truncation raw-block marker was not reset on re-entry, so the
second pass reasoned about a block boundary from the first.

*Fix.* Reset the marker on recovery re-entry.

### A cache miss said nothing about why

*Symptom.* Misses were visible only as latency. Two very different causes — a
client injecting per-request state into the system prompt, and a genuine
mid-prefix divergence — were indistinguishable in a log, and they have different
fixes.

*Fix.* The log now reports how many tokens were lost, where the divergence
began, and the token ids on each side, and classifies a leading-block divergence
separately from a mid-prefix mismatch. A leading-block divergence discards
everything, so it is worth calling out by name.

### Requests did not route to the slot holding their cache

*Fix.* Requests route to a slot by prefix match, so a follow-up turn lands on
the session that already holds its cache rather than evicting another one.

### A long prompt blocked the queue

*Fix.* Prefill is issued in bounded quanta, so a long prompt yields between
chunks instead of occupying the server for the duration.

---

## Session state

### Rewind moved the checkpoint without the state that depends on it

*Symptom.* A rewound session could produce output inconsistent with its visible
token history.

*Cause.* Rewinding moved the checkpoint length but did not restore the per-layer
compressed-KV state derived from it. The rows survived the rewind while the
tokens in front of them did not, so a reused row no longer described the token
at its position.

*Fix.* Restore the per-layer compressed-KV state along with the checkpoint
length.

---

## Tensor-parallel lifecycle

No throughput claim: this is what makes a pair usable outside a benchmark
harness.

### Only the benchmark harness could lead a pair

*Fix.* Any frontend can lead — server, agent, CLI and eval all bind one, and the
flags and help text cover them.

### Teardown dropped the socket under the worker

*Symptom.* The worker exited on a dropped connection rather than cleanly,
producing spurious errors at the end of a normal run.

*Fix.* Teardown is ordered: the leader announces before it closes.

### A recoverable worker failure stalled the pair

*Symptom.* The pair hung until the leader's watchdog fired and restarted it,
turning a recoverable condition into a restart.

*Fix.* The worker reports recoverable failures over the control plane. The gate
exchange carries a timeout word, so a stalled peer surfaces as an error rather
than as a hang.

---

## GPU watchdog

### A long prefill chunk could be killed mid-flight

*Symptom.* Long-context prefill under tensor parallelism could die in a way
that was not recoverable.

*Cause.* One chunked-prefill command buffer could run long enough for the GPU
watchdog to kill it. Under tensor parallelism that kill aborts the in-flight
bulk gate, and there is no path back from that — the peer is left waiting on an
exchange that will never complete.

*Fix.* The runtime chunk shrinks as the prompt grows, on a work budget both
ranks compute identically so the split stays symmetric. Allocation deliberately
does *not* follow the ladder: workspaces are sized from the context and stay at
the variant ceiling, because sizing them from a shrinking runtime chunk would
reallocate mid-prefill.

Pinned by `test_prefill_watchdog_bound` in
`tests/test_engine_mgpu_placement.c`: allocation flat, runtime bound monotone,
bounded, power-of-two, floored, and matching the boundary values observed on
hardware. Its ladder extends past the point where the work budget alone would
select a chunk below the floor, so the floor assertion is exercised rather than
vacuously true.

---

## Kernel and dispatch correctness

### The argsort tie-break comparator is required, not optional

*Symptom.* Removing what looked like a redundant comparator produced 27,533
differing indices against a CPU reference.

*Cause.* The beneficiary is not either selector on its own — it is *agreement
between the two paths*. The streaming top-512 orders by packing score and index
into one word; the block argsort needs the matching total order at the same `k`
or the two disagree on ties.

*Fix.* The canon comparator is selected for `top_k == 512`, which is exactly
where the streaming selector can be chosen, so the two paths cannot disagree.
Measured neutral for throughput (+0.06% / −0.13% / −0.06% across arms) — this is
a correctness change, not a speedup.

### Host row count and kernel row stride are one quantity

*Symptom.* Raising the host row count alone leaves the upper half of every
routed MXFP4 down projection unwritten.

*Cause.* The dispatch computes the grid from the row count and the kernel
strides by it. Raising one without the other quarters the grid without widening
the kernel.

*Fix.* The host sets `nr0 = 4` exactly when it selects the four-row pipeline, and
the invariant is stated at both sites.

### Split-K reduced over work groups that held no keys

Covered in [`optimizations.md`](optimizations.md) — it is a correctness tidy that
happens to save work, not a measured speedup.

---

## Test coverage

Gaps found by audit rather than by failure. Every default-on kernel change on
this branch now has a model-free gate that runs on any Metal device.

### Two tests passed vacuously

*Symptom.* Both indexer A/B tests reported "bit-exact". Forcing both arms onto
the same kernel still reported bit-exact.

*Cause.* The reference arm did not actually select the reference kernel, so each
test compared a path against itself. A test that cannot fail is worse than no
test, because it is reported as coverage.

*Fix.* Both tests call `ds4_gpu_last_indexer_scorer()` /
`ds4_gpu_last_indexer_topk()`, print which kernel each arm selected, and assert
the two differ before comparing output. Verified by mutation.

`test_topk_stream512` additionally compares both GPU paths against a CPU
reference in the same total order, and all three comparisons are fatal.
Comparing two GPU paths alone cannot catch an error they share. Its scores are
quantised coarsely so exact ties are common — ties are what the key packing has
to get right — and each row's visible prefix varies so rows with fewer live
candidates than `k` are covered.

### The chunk ladder's floor was never exercised

The ladder in the watchdog test stopped at 1M, but the floor binds at ~1.31M, so
the floor assertion was vacuously true. Extended to 2,000,000.

### DSpark argmax had no shape coverage

`test_mxfp4_metal` gains a vocabulary sized past the capped dispatch, so the
strided traversal runs in more than one threadgroup, plus the tensor-parallel row
shape.

### Orphan build rules

Fifteen Makefile rules pointed at harnesses that no longer exist in the tree and
one was advertised in `make help`. Removed. The two new test targets are in
`.PHONY` and in `make help`.

---

## Removed

Investigation scaffolding that shipped and should not have, or that was
superseded:

- **n-gram speculation** (~300 lines): measured at 1.03–1.10× with 91–97% of
  steps committing nothing. Removed rather than left default-off.
- **Encoder-timestamp instrumentation**: the batch reuses one encoder across
  many dispatches, so per-stage labels collapse onto a single span. Forcing a
  boundary per stage resolves them but perturbs throughput by 20–30% and
  inflates long-context spans by 50×, so the facility could not measure what it
  was built to measure.
- **Trace tag plumbing**: wrote a per-layer tag that only the removed encoder
  labeller ever read, leaving write-only state on the decode path. The debug
  groups, which do work under Metal System Trace, are kept.
- **`metal_graph_ablate_indexer`**, **`DS4_TP_LOGITS_PROBE_DIV`**, and dead
  DSpark test hooks: one-off ablation switches from closed investigations.
- **tiled4 indexer scorer** (~190 lines) and four dead LLT instantiations:
  superseded by the register-resident scorer.
