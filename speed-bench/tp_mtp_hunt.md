# TP DSpark performance hunt

This note records the `tp-mtp-hunt` branch changes and the two-node test
matrix.  It is a companion to `tp_decode_investigation.md`.  Measurements
explicitly listed below are from the M2 Ultra pair; remaining gains are
estimates until their corresponding A/B runs complete.

For the branch graph, target-only control, PP/RDMA status, and prioritized
cross-branch roadmap, start with the central
[`speed-bench` handoff](README.md).

## 2026-08-19 current verdict

The major proposer semantics, capture/cache lifecycle, compact-view, and F32-HC
runtime bugs are fixed.  DSpark is nevertheless **not a throughput win** on the
current context-512 Promessi sposi fixture:

| Arm | Steady decode | Result |
|---|---:|---|
| Target only | **41.98 t/s** | Current control |
| Published 5.99 GB support, forced d5 | **21.29--21.35 t/s** | 189/1,596 accepted (11.84%), no `d5>a5` |
| Current low-yield production policy | **41.57 t/s** | No verifier launched; about 112 ms total DSpark overhead |
| Published support + verifier head split | **20.18 t/s** | 151/1,785 accepted; verifier timing slightly lower but headline worse |
| Full-fat MXFP4 support after the F32-HC fix, forced d5 | **19.91 t/s** | 157/1,767 accepted (8.89%); no full d5 |

The production policy is a successful stop-loss: it backs off unproductive
probes and returns close to target-only speed.  It does not demonstrate a
speculative gain.  Further work should establish proposal parity and acceptance
on matched code and prose fixtures before spending more time on verifier
micro-optimizations.

The public llama.cpp/GaelicThunder headline of about 50% acceptance and mean
3.5 is not an equivalent fixture.  It is predictable code generation on a
single GB10 with a different target quant; the same card reports roughly 25%
for literary prose.  Llama's mean also includes the unconditional target token,
so 3.5 means about 2.5 accepted draft tokens plus one target token per verify
cycle, not 3.5 accepted drafts out of five.  The referenced GaelicThunder GGUF
is also a Q2_K/Q8 presentation rather than the 10.8 GB full-fat artifact.

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
- keeps the verifier's Apple Q8 vocabulary head on one weight pass for draft
  lengths two through five: lengths two through four use the existing exact
  four-row tile and length five uses the bit-checked native five-row tile,
  instead of padding every block to eight rows (`DS4_DSPARK_VERIFY_HEAD_PAD8=1`
  restores the old path);
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

`DS4_DSPARK_TP_VERIFY_HEAD_SPLIT=1` is the end-to-end form of that experiment.
It automatically enables the output split, slices `attn_q_b` to this rank's 32
heads, keeps those heads compact through raw/indexed attention, and reconstructs
the full 4K attention row with the same one batch gate per layer.  The option
changes verifier gate participation and therefore must be set identically on
both ranks.  The design estimate was 6--9 ms per five-row block, but the first
remote run measured about 106 versus 108 ms/verifier on different generated
paths and reduced headline throughput to 20.18 t/s.  It remains default-off and
is not a promotion candidate without a controlled same-path A/B.

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

## First corrected two-node results

The corrected proposer and compact-capture fix produced these 512-token
Promessi sposi measurements with the published 5.99 GB support GGUF.  Throughput
runs used `DS4_METAL_FAST_SYNC=1` on both ranks and no dispatch/GPU-busy
profiling.

| Arm | Steady decode | Key result |
|---|---:|---|
| Target-only control | 41.96 t/s | 23.83 ms/token baseline |
| Production policy, clean | 38.08 t/s | Working proposals, but still a large low-yield tax |
| Production policy, stats | 38.82 t/s | 13 proposed, 2 accepted; 490/502 cycles produced no draft |
| Forced five, stats | 21.35 t/s | 1,596 proposed, 189 accepted; zero full `d5>a5` blocks |

The forced arm had a 35.5% first-proposal hit rate and accepted 1.658 drafts per
actual verifier call.  Proposal work cost 3,411.8 ms, while 114 five-row
verifiers cost 12,332.9 ms, or 108.18 ms each.  A normal target cycle cost
about 24.53 ms.  Even a perfect five-token block was therefore marginal before
the new verifier cuts; the observed average acceptance was far below break-even.

The production stats arm paid 873.0 ms of proposal work plus one 68.8 ms d2
verifier to save only 48.1 ms of target work (`net_saved=-896.1 ms`).  That
result motivated the default Apple TP policy below: make cold probing sparse,
batch its support-KV catch-up, and reserve the verifier for d5.  The policy,
one-pass Q8 verifier head, and optional attention-head split were implemented
after these measurements.  Their later outcomes are summarized in the
current-verdict table above; retain this section for the detailed economics
that motivated them.

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
finish with `proposed>0`, `errors=0`, `replay_fallbacks=0`, and `replayed=0`
before timing a full run.

Also run one reused-session smoke before the 512-token timing arm.  Keep the
same forced-five environment and add `DS4_DSPARK_SPEC_LOG=1`, but use:

```sh
--ctx-start 256 --ctx-max 512 --step-incr 256 --gen-tokens 16
```

Under TP, `ds4-bench` restores the first frontier by replaying its prefix, then
extends the same session to 512.  This covers pending invalidation, cold
support-ring reconstruction, and batched suffix maintenance.  Require two CSV
rows, no hang, `proposed>0`, and zero `invalid`, `verifier_unavailable`,
`errors`, `replay_fallbacks`, and `replayed` counters.

If a smoke reports `proposed=0`, shorten it to four generated tokens and add
`DS4_DSPARK_PROBE=1`.  That diagnostic changes the proposer execution path, so
use it only to capture the `DSpark proposer pending` stage-status line; remove
it again for acceptance and throughput measurements.

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

On Apple TP, that production arm now admits only full five-token proposals to
the verifier.  A short proposal or any verifier result below `a5` backs the
next proposal off by 64, then 128, then 256 target cycles; `d5>a5` immediately
returns to hot probing.  Skipped cycles retain target-hidden rows in a 128-row
GPU ring and batch-convert the missing support KVs at the next probe, avoiding
the old per-token support-ring maintenance tax.  For an old-policy A/B, add
`DS4_DSPARK_TP_LOW_YIELD_POLICY=0` on the coordinator only.  This is distinct
from `DS4_DSPARK_SCHEDULER=0`, which remains the forced diagnostic mode and
bypasses all adaptive policy.

The target-only control is the same command without `--mtp`/`--dspark` and
without DSpark environment variables.  Alternate control and treatment if
multiple samples are taken so temperature and DVFS drift do not all favor one
arm.

### Support-model precision follow-up

The published 5.99 GB support GGUF is not numerically equivalent to the
official checkpoint: its routed experts are IQ2_XXS/Q2_K and its Markov and
confidence heads are Q8_0, whereas the checkpoint uses FP4 experts and BF16
Markov/confidence weights (the official confidence path computes in F32).
First establish corrected-runtime acceptance with the published file.

If the full-fat llama.cpp `DeepseekV4-Flash-20260731-DSpark.gguf` is already
available, adapt it directly without downloading the HF shards:

```sh
gguf-tools/deepseek4-quantize \
  --import-dflash-gguf /path/DeepseekV4-Flash-20260731-DSpark.gguf \
  --out DSpark-support-full.gguf
```

This preserves the MXFP4 experts, expands the BF16 router gates and confidence
projection to F32, and converts only the two Markov matrices to Q8_0 for the
Metal GPU fast path.  Otherwise, build a support sidecar from the original HF
shards:

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

The first full-fat run produced zero first hits because the runtime incorrectly
sent F32 HC projection weights through an F16-only fused helper.  That result is
invalid and led to `242642c`.  After the type-aware F32 fix, the forced run was
still only 19.91 t/s with 157/1,767 accepted drafts (8.89%), below the published
support file on this prompt.  Full-fat weights therefore did not resolve the
acceptance problem.  Direct native `dflash` loading would preserve additional
precision only for the two Markov matrices; the other 79 imported tensors are
copied losslessly or expanded exactly.

## Isolation matrix

Run these only after the forced-five optimized arm succeeds.  Apply an option
on both ranks unless the table says coordinator-only.

| Arm | Added environment | What it isolates | Expected direction |
|---|---|---|---|
| Optimized | none | All default branch changes | Reference |
| Old production scheduler | `DS4_DSPARK_TP_LOW_YIELD_POLICY=0` (coordinator only) | d5-only admission, fixed backoff, and deferred support-KV catch-up | Slower on low-yield prompts; useful rollback/A-B |
| Old batch release | `DS4_METAL_FAST_BATCH_SYNC=0` | Fast verifier fence only; ordinary fast fence stays on | 4.1--7.7 ms/block slower |
| Row MoE | `DS4_DSPARK_TP_VERIFY_ROW_MOE=1` | Batched resident-MXFP4 verifier MoE | Slower; magnitude unmeasured |
| Replay partials | `DS4_DSPARK_TP_REPLAY_PARTIAL=1` | Distributed prefix snapshots | ~23.8 ms per replayed token slower |
| No verifier pipeline | `DS4_DSPARK_DISABLE_VERIFY_PIPELINE=1` | Host encode/GPU overlap | ~0.5--2 ms/block slower |
| CPU Markov | `DS4_DSPARK_NO_GPU_MARKOV=1` (coordinator only) | Metal Markov argmax | Slower; inspect proposal timing |
| Pad verifier head to eight | `DS4_DSPARK_VERIFY_HEAD_PAD8=1` | Native/four-row Apple Q8 verifier head | ~0.5--1.1 ms/block slower; no dispatch-count change |
| Split attention output | `DS4_DSPARK_TP_VERIFY_ATTN_OUT_SPLIT=1` | Default-off projection split | Unknown; likely -1 to +2 ms/block |
| Split verifier heads | `DS4_DSPARK_TP_VERIFY_HEAD_SPLIT=1` | q_b + attention core + output projection, 32 heads/rank | About 106 vs 108 ms/verifier on different generated paths; headline 20.18 t/s, so inconclusive/not promotable |
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

## RDMA/synchronization experiment outcome

No later RDMA hot-path optimization is applied on this branch.  The fast
release fence itself remains essential: it reduces the ordinary 86 TP gates
from roughly 180 us to 38 us and relies on the undocumented Apple Metal
system-scope mechanism demonstrated in
[MLX PR #1773](https://github.com/ml-explore/mlx/pull/1773).

The proposed software cleanup and fused gate were implemented together in
commit `fa47b1c` on `backup/pre-shared-base-20260816/tp-frontends-phase1` and
then tested separately.  They did not produce a
reproducible gain.  One combined sample was 41.50 steady t/s with both flags
versus 41.86 with both off, and isolated arms were also indistinguishable from
noise.  `DS4_TP_RDMA_HOTPATH` and `DS4_METAL_FUSED_TP_GATE` are therefore not
present in the active branch stack.

The result is technically plausible.  A Flash gate is already one 16 KiB WR;
prebuilt/chained WRs do not reduce the steady verbs-call count, receive CQEs are
still polled every gate, and the removed second Metal dispatch was largely
hidden underneath the exchange.  The fused kernel may also delay CPU
observation of a coherent store made in the middle of a dispatch.

Remaining DSpark-specific communication ideas are lower priority than proposal
quality:

- a dedicated verifier QP/CQ could save roughly 0.6--1.4 ms/block;
- pipelining adjacent verifier-layer exchanges may save 0.5--1.8 ms/block;
- exact distributed greedy `(max,id)` could reduce output-head traffic; and
- a direct NIC-delivered marker is interesting only after NIC-to-GPU coherence
  is proved.

None can compensate for the current 108 ms verifier accepting only about 1.66
draft tokens.  Revisit them after a matched code fixture reaches high, stable
acceptance.
