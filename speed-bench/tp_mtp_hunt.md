# TP DSpark performance hunt

This note records the `tp-mtp-hunt` branch changes and the two-node test
matrix.  It is a companion to `tp_decode_investigation.md`; all estimates are
pre-measurement until the M2 Ultra pair has run the matrix below.

## What changed

The default DSpark TP path now:

- uses the fast Metal release fence for verifier batch gates as well as normal
  decode gates (expected saving: 4.1--7.7 ms per 43-layer verify block);
- commits partial verifier prefixes on both ranks instead of replaying each
  accepted token through ordinary decode (about 23.8 ms avoided per replayed
  token at the measured 41.96 t/s baseline);
- evaluates every verifier row in one resident-MXFP4 routed-MoE batch per
  layer, reducing that stage from `2*N` kernel dispatches to two per layer;
- performs the DSpark Q8 Markov correction and argmax on Metal, avoiding a
  full vocabulary-row readback and CPU Q8 matvec for every proposal step
  (including the confidence-zero diagnostic path), and skips unused confidence
  readback when that path also disables the adaptive scheduler;
- submits verifier command buffers every four layers on both ranks when the
  init-fixed batch-fence configuration makes cross-buffer gates safe;
- declines a one-token proposal after its free first-token check, because a
  one-row target verify cannot eliminate any target evaluation;
- reports proposal, verifier, replay, and acceptance economics by draft
  length; and
- lets `ds4-bench` load `--mtp ... --dspark` and time speculative blocks.

`DS4_DSPARK_TP_VERIFY_ATTN_OUT_SPLIT=1` is an experimental, default-off
projection split.  It halves each rank's verifier attention-output weights but
adds one TP batch gate per layer, so it must win an A/B before becoming a
default.

Correctness guards added with the performance work:

- TP DSpark blocks are capped at five rows, matching four captured proper
  prefixes; larger support-model blocks remain available outside TP.
- Prefix captures carry per-cycle validity, EOS truncates the mirrored block,
  and a raw cache without `SWA window + verify block` slack refuses speculative
  mutation.
- The TP protocol is version 8 and carries verifier feature flags, rejecting
  mixed binaries at bind time and mismatched gate-changing options on verify.
- Verifier decisions complete transactionally: the worker ACKs a successful
  full/prefix/replay commit before optional replay logits are transferred.
- Row and verifier fence words occupy separate banks; fence timeouts latch a
  host-visible failure, and a transport failure invalidates the verifier before
  either rank can commit stale peer rows.

### Official-reference semantic correction

The first two-node measurements on this branch exposed a more fundamental
proposer mismatch.  The runtime has now been reconciled with DeepSeek's pinned
[`Transformer.forward_spec`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py),
`DSparkBlock.forward_embed`, and `DSparkAttention.forward`:

- captured target hidden `h_P` is paired with the target-predicted anchor
  `x_{P+1}`, not the token `x_P` that produced the hidden state;
- the five DSpark outputs are proper proposals `x_{P+2}...`, excluding the
  anchor.  They remain pending for one cycle: the target evaluates the anchor
  normally, then its new logits provide the free check for proposal zero and
  the existing five-row verifier checks the pending block;
- every support stage persists `kv_norm(wkv(main_x))` directly at position
  `P`.  The anchor/noise block alone passes through support HC/attention norm,
  and its KVs at `P+1...` are transient lookahead rows;
- verifier commits materialize direct support KVs for committed intermediate
  target captures while leaving the newest capture for the next DSpark call;
- confidence consumes the unnormalized post-HC-head state, while only the
  vocabulary head consumes the final RMS-normalized state; and
- the logical support-attention window is 128 rows.  The physical ring remains
  larger so the 128 persistent rows plus five transient draft rows never alias.

Chunked prefill carries the last 128 target-hidden rows across chunk boundaries,
including a short final remainder.  Capture batches are tagged by provenance so
rejected verifier rows can never be mistaken for prefill history.  Reused
sessions also keep the support ring aligned while extending a prompt or using
ordinary decode; an exact single-node snapshot hit without serialized support
state falls back to a cold target prefill instead of silently disabling DSpark.

Results produced before this correction used the wrong anchor, transformed the
main KV through the support block, and exposed a much larger history than the
checkpoint was trained for.  Do not use their acceptance rates to judge the
support model; rerun both the forced-five and production arms.

## Required setup

Build and deploy the same `tp-mtp-hunt` binary on both nodes.  The worker needs
the new binary and the same Metal gate environment, but it does **not** need the
DSpark support GGUF or `--dspark`: proposal work is coordinator-only.  The
coordinator needs a checkpoint-matched Flash 0731 support file, normally:

```sh
./download_model.sh ds4f-dspark
```

Keep `DS4_METAL_FAST_SYNC=1` on both nodes in every comparison.  Do not enable
dispatch/GPU-busy profiling for throughput runs.

DSpark requires the effective prefill chunk to be at least
`min(session_context, 128)`.  The default benchmark allocation satisfies this;
an explicitly smaller `--prefill-chunk` is rejected rather than silently
shortening the support model's trained attention history.

## First measurement set

First run a short semantic smoke with the command below changed to
`--gen-tokens 32` and with `DS4_DSPARK_SPEC_LOG=1`.  The first external cycle is
bootstrap: it creates a pending proposal but does not charge a no-draft
scheduler result or launch a verifier.  Starting with the next matching anchor,
proposal zero is checked against post-anchor target logits.  The smoke must
finish with `errors=0`, `replay_fallbacks=0`, and `replayed=0` before timing a
full run.

Also run one reused-session smoke before the 512-token timing arm.  Keep the
same forced-five environment and add `DS4_DSPARK_SPEC_LOG=1`, but use:

```sh
--ctx-start 256 --ctx-max 512 --step-incr 256 --gen-tokens 16
```

Under TP, `ds4-bench` restores the first frontier by replaying its prefix, then
extends the same session to 512.  This covers pending invalidation, cold
support-ring reconstruction, and batched suffix maintenance.  Require two CSV
rows, no hang, proposer stages reported as `ok`, and zero `invalid`,
`verifier_unavailable`, `errors`, `replay_fallbacks`, and `replayed` counters.

Use the existing worker command, rebuilt from this branch.  On the coordinator,
append the DSpark options to the same command used for the 41.96 t/s baseline:

```sh
MODEL=~/Downloads/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf
DSPARK=gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf

DS4_METAL_FAST_SYNC=1 \
DS4_DSPARK_STATS=1 \
DS4_DSPARK_SCHEDULER=0 \
./ds4-bench \
  -m "$MODEL" \
  --mtp "$DSPARK" --dspark-confidence 0 \
  --role coordinator --tensor-parallel --transport rdma \
  --listen 0.0.0.0 1234 --rdma-device rdma_en7 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 512 --ctx-max 512 --gen-tokens 512
```

This deliberately requests five proper proposal tokens and disables adaptive
skipping.  It measures the raw proposer/verifier economics; the `outcomes`
matrix tells us whether that prompt actually covered full and partial commits.
Unlike the pre-correction run, `miss_first` now compares `x_{P+2}` against
target logits obtained after evaluating the `x_{P+1}` anchor.
Save the CSV row plus the final `ds4: DSpark stats` and `ds4: DSpark detail`
lines.

Then run the production-policy arm by removing `DS4_DSPARK_SCHEDULER=0` and
replacing `--dspark-confidence 0` with `--dspark`.  Finally remove
`DS4_DSPARK_STATS=1` for the clean throughput number.

The target-only control is the same command without `--mtp`/`--dspark` and
without DSpark environment variables.  Alternate control and treatment if
multiple samples are taken so temperature and DVFS drift do not all favor one
arm.

### Support-model precision follow-up

The published 5.99 GB support GGUF is not numerically equivalent to the
official checkpoint: its routed experts are IQ2_XXS/Q2_K and its Markov and
confidence heads are Q8_0, whereas the source uses FP4 experts, BF16 Markov W1,
and F32 Markov W2/confidence.  First establish corrected-runtime acceptance
with the published file.  If production confidence still prunes too heavily,
build a confidence-preserving sidecar from the original HF shards:

```sh
gguf-tools/deepseek4-quantize \
  --hf /path/DeepSeek-V4-Flash-0731 \
  --dspark-support \
  --out DSpark-confidence-f32.gguf \
  --tensor-type mtp.2.confidence_head.proj.weight=f32 \
  --threads 8
```

For the highest-quality proposal A/B, add `--experts mxfp4`; this losslessly
repacks the source FP4 experts and produces an approximately 10.8 GB support
file.  F32 Markov W1/W2 overrides are useful as a quality diagnostic, but they
disable the current Q8-only GPU Markov fast path and therefore are not a clean
throughput comparison.

## Isolation matrix

Run these only after the forced-five optimized arm succeeds.  Apply an option
on both ranks unless the table says coordinator-only.

| Arm | Added environment | What it isolates | Expected direction |
|---|---|---|---|
| Optimized | none | All default branch changes | Reference |
| Old batch release | `DS4_METAL_FAST_BATCH_SYNC=0` | Fast verifier fence only; ordinary fast fence stays on | 4.1--7.7 ms/block slower |
| Row MoE | `DS4_DSPARK_TP_VERIFY_ROW_MOE=1` | Batched resident-MXFP4 verifier MoE | Slower; magnitude unmeasured |
| Replay partials | `DS4_DSPARK_TP_REPLAY_PARTIAL=1` | Distributed prefix snapshots | ~23.8 ms per replayed token slower |
| No verifier pipeline | `DS4_DSPARK_DISABLE_VERIFY_PIPELINE=1` | Host encode/GPU overlap | ~0.5--2 ms/block slower |
| CPU Markov | `DS4_DSPARK_NO_GPU_MARKOV=1` (coordinator only) | Metal Markov argmax | Slower; inspect proposal timing |
| Split attention output | `DS4_DSPARK_TP_VERIFY_ATTN_OUT_SPLIT=1` | Default-off projection split | Unknown; likely -1 to +2 ms/block |
| Verify one-row drafts | `DS4_DSPARK_VERIFY_SINGLE=1` (coordinator only) | One-row economic guard | Always slower when exercised |

For any arm, the high-value diagnostics are `gen_tps`, `gen_steady_tps`,
`proposed`, `accepted_draft`, `direct_partial`, `replay_fallbacks`, `replayed`,
`net_saved`, `by_draft`, and `outcomes`.  A performance result with verifier
errors or a nonzero replay count in the optimized arm is a correctness or
coverage result first, not a speed result.

The CPU-Markov ablation may change proposal IDs and therefore acceptance near
ties: its Q8 path requantizes the W1 state, while the Metal kernel consumes the
dequantized W1 row directly.  The target verifier still decides every committed
token, so compare both proposal timing and acceptance rather than treating this
arm as a proposal-parity test.

## Read-only RDMA/synchronization follow-ups

No RDMA hot-path optimization is applied on this branch.  The fast release
fence itself relies on the undocumented Apple Metal system-scope mechanism
demonstrated in [MLX PR #1773](https://github.com/ml-explore/mlx/pull/1773),
and retains shared-event fallback when it is unavailable.  The current measured
~38 us gate is already much closer to a CPU-posted two-sided Apple/RDMA floor
than to the link's raw ~10 us latency.  The following are the remaining options
from a fresh hot-path audit; gains are estimated against the 23.83 ms/token,
41.96 t/s control and are not additive.

| Option | Estimated gain | Complexity | Risk | Notes |
|---|---:|---|---|---|
| Fuse GPU arrival publish and release polling into one persistent gate kernel | 0.16--0.38 ms/token, +0.28--0.68 t/s; 0.08--0.19 ms/DSpark block | Medium | Medium | Removes one tiny dispatch per gate and its encoder boundary |
| Clean the RDMA hot path | 0.15--0.43 ms/token, +0.27--0.77 t/s; 0.05--0.20 ms/block | Low--medium | Low | Cache trace env, delay disconnect peeks, signal every 8--16 sends, prepost/chained receives, prebuild WRs, remove uncontended lock work |
| Dedicated verifier QP/CQ | 0.6--1.4 ms/DSpark block | Medium | Medium | Removes per-layer TCP header/barrier and the first-verifier dummy-window drain |
| NIC-delivered flag polled directly by GPU | 0.2--0.6 ms/token, +0.36--1.08 t/s; 0.1--0.4 ms/block | High | High | Must prove NIC-to-GPU coherence; `coherent(system)` only establishes CPU/GPU behavior |
| Pipeline verifier row exchanges | 0.5--1.8 ms/block | High | Medium | Overlap adjacent verifier-layer communication/compute with multiple safe slab slots |
| Exchange exact greedy `(max,id)` instead of half logits | 0.12--0.18 ms/token, +0.21--0.32 t/s; 0.5--0.8 ms/block | Medium | Low in greedy mode | Requires exact tie/NaN semantics and is not a sampled-decoding protocol |
| Probe >16 KiB messages on macOS 26.3+ | 0.1--0.4 ms/block | Medium | Low | MLX/JACCL suggests up to 512 KiB can work; runtime capability probe required |
| Pin/realtime-tune the gate service thread | 0--0.15 ms/token, up to +0.27 t/s | Low | Medium | Mostly reduces tail jitter and can harm the rest of the system |
| FP16 verifier communication | 0.2--0.5 ms/token; 1--2 ms/block | Medium | High | Numerical/argmax risk; not an exact transport optimization |

A conservative endpoint is about 42.5--43.4 t/s with software hot-path work,
43.5--44.4 t/s if a direct NIC marker is reliable, and roughly 45--46 t/s only
under near-ideal gate elimination.  These are ceilings to prioritize work, not
promised benchmark results.
