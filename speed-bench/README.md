# Performance work handoff

This is the authoritative handoff for the active Metal, tensor-parallel (TP),
pipeline-parallel (PP), RDMA, and DSpark work as of **2026-08-19**.  It is meant
to let a new reader choose the right branch, reproduce the important numbers,
and avoid repeating experiments that have already been settled.

The long-form experiment records remain useful, but some of their early
conclusions are intentionally historical:

- [`tp_decode_investigation.md`](tp_decode_investigation.md) contains the full
  two-node TP decomposition, ablations, and kernel evidence.
- [`tp_mtp_hunt.md`](tp_mtp_hunt.md) contains the DSpark protocol, proposer,
  verifier, scheduler, and support-GGUF details.
- [`../gguf-tools/README.md`](../gguf-tools/README.md) documents support-model
  conversion and the full-fat `dflash` importer.

## Executive summary

| Area | Current outcome | Recommendation |
|---|---|---|
| Two-node Flash TP decode | **41.5--42.0 t/s** at context 512 on the M2 Ultra pair | Stable reference configuration; keep `DS4_METAL_FAST_SYNC=1` on both ranks |
| TP Metal release fence | About **180 us -> 38 us per gate**; the direct 86-gate delta is **12.2 ms/token**, while full-run estimates were roughly 12--15 ms | Essential; this is the RDMA/synchronization optimization that mattered |
| Later RDMA hot-path cleanup and fused gate | No reproducible gain; a combined sample was 41.50 vs 41.86 t/s and isolated arms were also neutral | Deliberately excluded from the current branch stack; do not restore without new evidence |
| Apple PP over RDMA | WORK/RESULT inference traffic uses RDMA end to end; TCP remains for bootstrap/control and snapshots | Useful for capacity and long-prefill pipelining, not single-stream decode acceleration |
| PP fast activation fence | Decode-only and qualitatively neutral in current 128 KiB RDMA framing; no numeric A/B is recorded and prefill is unaffected | Leave available as an experiment, but do not expect a speedup |
| TP DSpark/DSpark correctness | Major official-semantics, cache, lifecycle, and F32-HC bugs are fixed | Correct enough to investigate, but not production-fast on the current prompt |
| TP DSpark performance | Target-only is **41.98 t/s**; forced DSpark is **19--21 t/s**; low-yield production policy recovers **41.57 t/s** by mostly not drafting | Keep DSpark off for maximum throughput; next work is proposal acceptance, not verifier micro-optimization |
| Four-node direction | Native **TP4 on a full RDMA mesh**, using one-round flat all-gather for 16 KiB decode gates | First build a 3-peer collective microbenchmark; realistic target **50--56 t/s**, aggressive ceiling about **60--63 t/s** |

## Hardware and benchmark contract

The authoritative TP/DSpark measurements use:

- 2 x Mac Studio M2 Ultra, 60-core GPU, 128 GiB RAM;
- a direct Thunderbolt RDMA link;
- DeepSeek V4 Flash 0731 with native MXFP4 routed experts, F16 HC/
  compressor/indexer, and Q8 attention/shared/output tensors;
- `speed-bench/promessi_sposi.txt`;
- `--ctx-start 512 --ctx-max 512 --gen-tokens 512`;
- greedy decoding; and
- `gen_steady_tps`, the eighth CSV field, as the headline decode number.

The run-to-run noise floor is about **1%** and larger drift has occurred across
distant thermal/time windows.  For small changes, run A/B/B/A, warm the same
pipelines, use the same binaries and environment on both ranks, and compare
medians.  Profiling, trace, dispatch, and GPU-busy flags are diagnostic arms,
not headline throughput arms.

The main model is about 145.26 GiB.  Each TP rank maps a 76.73 GiB shard, so a
single-node control for this exact model is not available.  The target-only
two-rank run is the control.

## Branch topology

The intended branch ancestry is shown below.  The hashes identify the key code
capability commits before this documentation change, not immutable future tips:

```text
metal-tp-fast-fence-1  0dcbfbc
  `- metal-pp-fast-fence-1  3a651eb
       `- metal-optimizations  01fc03c
            |- tp-frontends-phase1  c8d9100
            |    `- tp-mtp-hunt  242642c
            `- apple-rdma-pp  f7d3837
```

| Branch | Use it for | Important scope |
|---|---|---|
| `metal-tp-fast-fence-1` | Minimal TP fast-fence baseline | Optional polled Metal release fence only |
| `metal-pp-fast-fence-1` | Shared fast-fence base | Adds the experimental one-token PP activation handoff |
| `metal-optimizations` | Common optimized Metal base | Current upstream merge, exact pre-M5 decode/prefill paths, indexer/host cleanup, build isolation |
| `tp-frontends-phase1` | Two-node TP performance work | TP frontends, compact decode FlashAttention, instrumentation, tests, and investigation record |
| `tp-mtp-hunt` | TP DSpark and support-GGUF work | All TP work plus DSpark proposer/verifier/scheduler fixes and full-fat importer |
| `apple-rdma-pp` | Apple PP data-plane work | Common base plus PP WORK/RESULT RDMA transport |

The `backup/pre-shared-base-*` branches preserve the older branch layout.  In
particular, the no-impact RDMA cleanup/fused-gate experiment lives only in that
history and is not part of `tp-frontends-phase1` or `tp-mtp-hunt`.

Start new generic Metal work from `metal-optimizations`, two-/four-node TP work
from `tp-frontends-phase1`, DSpark proposal/verifier work from `tp-mtp-hunt`,
and PP transport work from `apple-rdma-pp`.  Do not merge the MTP and PP sibling
branches merely to start an unrelated experiment: both substantially touch
distributed/session code, while hybrid TP+PP is not implemented.
The PP RDMA commit also factors common verbs setup out of `ds4_tp.c`, so a
future merge into the TP branch should expect semantic conflict resolution even
when Git reports a small textual conflict set.

## Common Metal optimization base

The common base combines current upstream exact Metal work with the pre-M5
ports and campaign fixes needed by this M2 Ultra pair.  The most relevant local
commits are:

| Commit | Change | Evidence/status |
|---|---|---|
| `b757e6b` | Collapse indexer decode score barriers from 33 per row to 2 | Long-context improvement; no effect at context 512 before the indexer engages |
| `4427c3e` | Remove three per-token host costs found from trace symbolication | Host time fell, but this did not produce a measurable decode gain by itself |
| `bc27975` | Keep fixed-row Q8 kernels off the wide `nr0` path | Correct dispatch selection/hardening |
| `e8349a9` | Admit pre-M5 to decode pair/affine RoPE | Historical TP A/B: about **-1.16 ms/token** |
| `e33ced3` | Admit pre-M5 to gathered KV and persistent zero-mask paths | Historical TP A/B: about **-1.13 ms/token** |
| `5697d9c` | Fix compressor pair state-store failures | Correctness companion for the pre-M5 path |
| `09fadf9` | Admit pre-M5 to output-HC weights4 | Small, about **0.02 ms/token** in the original campaign |
| `c1c2a5d` | Admit pre-M5 to six prefill fusion gates | Prefill-only; retain exactness tests |
| `01fc03c` | Isolate distributed CPU objects from Metal | Build/branch hygiene |

`tp-frontends-phase1` restores the TP-specific frontends, compact decode
FlashAttention, phase instrumentation, and the standalone kernel harnesses in
one commit (`c8d9100`).  Across the original campaign, TP decode moved from
roughly **37 t/s to 41.2 t/s** at short context.  Later context-512 controls
settled at **41.5--42.0 t/s**.

The phase-1 bundle also restores the routed-MXFP4 down-projection R4 path
(historically about **-0.53 ms/token**), safe per-slot TP command-buffer splits,
ablation/dispatch/logits/gate profiling, and the exact compact-attention
harness.  The token split was estimated at about **-1.62 ms/token / +3.3%** at
roughly 190k context; it is not part of the context-512 headline.

## Two-node tensor parallelism

### Reproduction commands

Start the worker first:

```sh
DS4_METAL_FAST_SYNC=1 \
./ds4-bench -m "$MODEL" \
  --role worker --tensor-parallel --transport rdma \
  --coordinator <coordinator-ip> 1234 \
  --rdma-device rdma_en6
```

Then run the coordinator:

```sh
DS4_METAL_FAST_SYNC=1 \
./ds4-bench -m "$MODEL" \
  --role coordinator --tensor-parallel --transport rdma \
  --listen 0.0.0.0 1234 \
  --rdma-device rdma_en7 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 512 --ctx-max 512 --gen-tokens 512
```

Confirm both `transport=rdma` and `TP fast release fence enabled`.  Do not pass
`--layers` in TP mode: this implementation is a 50/50 expert/head split, not a
layer pipeline.

Useful diagnostic-only controls are:

```sh
DS4_TP_GATE_PROFILE=1
DS4_TP_LOGITS_PROFILE=1
DS4_METAL_DISPATCH_PROFILE=1
DS4_METAL_GPU_BUSY_PROFILE=1
DS4_TP_ABLATE=...
```

Gate-changing or graph-changing options must match on both ranks.

### Current token budget

At the 41.56 t/s investigation point, one token cost 24.06 ms.  These rows are
stage attribution, not a perfectly additive model:

| Stage | Approximate cost | Notes |
|---|---:|---|
| Routed MoE | 6.17 ms | Dominated by per-layer rank imbalance from top-6 expert routing |
| 86 TP gates | 3.30 ms | 38 us/gate with the fast release fence |
| Attention output | 2.88 ms | Already near the isolated Q8 bandwidth result |
| Attention core | 2.68 ms | Compact short-context scheduling removed much empty work; retain exactness checks |
| `q_b` | 1.56 ms | Near isolated kernel rate |
| Shared expert | about 1.15 ms | TP-sliced |
| HC mix | 0.85 ms | Small remaining stage |
| Output head | about 0.52 ms | Logits exchange is not the principal bottleneck |

The compact TP FlashAttention schedule has model-free bit-exact coverage and
reduces the context-512 vec grid from 44,032 to 11,520 workgroups/token, but no
authoritative full two-node A/B was recorded.  Keep that distinction between a
validated work reduction and a measured end-to-end gain.  Use
`DS4_METAL_DECODE_NWG=32` as the legacy scheduling control.

The best remaining two-node compute opportunity is expert load balance.  The
contiguous 128/128 ownership gives an expected critical load of 3.938 selected
experts per layer instead of the ideal 3.0.  Perfect balance would recover
about **1.47 ms/token**, putting the pair near **44.3 t/s**.  Practical options
are overlapping expert ownership (about +27 GiB/node) or a TP-specific repacked
GGUF that stores contiguous intra-expert slices.

### Synchronization and RDMA conclusions

The polled Metal release fence is the large win.  Without it, shared-event
resume was about 180 us/gate; with it, the measured gate was about 38 us.  That
quoted per-gate delta is 12.2 ms over 86 gates; broader full-run comparisons
put the recovered path in the roughly 12--15 ms/token range.  The mechanism
uses an undocumented Apple `coherent(system)` behavior and retains a fallback,
so timeout/failure handling and identical rank configuration matter.

The later `DS4_TP_RDMA_HOTPATH` and `DS4_METAL_FUSED_TP_GATE` experiments did
not move throughput.  The cleanup saved software operations, but the 16 KiB
gate is already one WR and receive CQ polling still occurs for every gate.  The
fused kernel also removed a dispatch that was mostly hidden underneath the
exchange, while making the CPU observe a mid-dispatch coherent store.  These
flags are intentionally absent from the current branch tips.

Other settled RDMA facts:

- AppleThunderboltRDMA accepts one-sided WRITE WRs but does not execute them;
  the production path must remain SEND/RECV.
- Plain volatile or device-scope polling is not reliable; system scope is
  required.
- More polling threadgroups and backoff made the GPU cost worse.
- The 16 KiB gate is latency/software bound, so inline quantization and other
  bandwidth-only collective tricks are the wrong lever.
- Half-logit exchange is only about 1.3% of the token.  A distributed greedy
  `(max,id)` protocol remains possible, but is not a general sampling path.

## Pipeline parallelism and Apple RDMA

Checkout `apple-rdma-pp` for the PP RDMA data plane.  In a two-node route,
steady WORK and RESULT messages use RDMA.  TCP remains for HELLO/QP bootstrap,
liveness/control, and KV snapshot traffic.  A failed RDMA handshake does not
silently use TCP when RDMA was explicitly requested.

Select the transport and local Thunderbolt RDMA device with PP-specific
options:

```sh
# Worker.
DS4_METAL_FAST_SYNC=1 \
./ds4 -m "$MODEL" \
  --role worker --layers 22:output \
  --coordinator <coordinator-ip> 1234 \
  --dist-transport rdma \
  --dist-rdma-adj-devices rdma_en6

# Coordinator benchmark.
DS4_METAL_FAST_SYNC=1 \
./ds4-bench -m "$MODEL" \
  --role coordinator --layers 0:21 \
  --listen 0.0.0.0 1234 \
  --dist-transport rdma \
  --dist-rdma-adj-devices rdma_en7 \
  --dist-activation-bits 32 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 512 --ctx-max 512 --gen-tokens 512
```

For a point-to-point multi-hop mesh, `--dist-rdma-adj-devices` also accepts a
comma-separated `peerhost=device` map.  Use `DS4_DIST_RDMA_TRACE=1` for a short
transport proof, `DS4_DIST_CONNECT_TRACE=1` for bring-up, and
`DS4_DIST_DECODE_PROFILE=1` plus `--debug` for phase timing; remove all tracing
and profiling for throughput.

Middle workers should name both adjacent peers explicitly.  The current
downstream fallback uses the last device-map entry when a route address is not
directly matched, so ordering is part of the experimental interface.
Each adjacency consumes two UC QPs, one per direction; Apple's documented
ten-QP/device ceiling therefore matters when several neighbours share one
Thunderbolt RDMA device.

PP is primarily a capacity and long-prefill feature.  Historical **TCP**
two-M5/Q4 measurements reached 582--674 prefill t/s on 9k--64k prompts versus
a 421--354 t/s single-process Q2 reference.  The model quant differs, so those
prefill figures are directional rather than an apples-to-apples speedup.
Decode fell from 30.59 t/s local to 24.67 t/s distributed because one token
still traverses every stage serially.  No checked-in TCP-vs-RDMA PP benchmark
has yet established an RDMA headline gain.

The PP fast fence, also selected by `DS4_METAL_FAST_SYNC=1`, is narrower than
the TP fence:

- it applies only to one-token, position>0, F32 activation handoffs;
- it cannot accelerate prefill;
- the startup log proves local capability, not that every request used it; and
- the current RDMA channel sends fixed 128 KiB strides.

Flash's decode activation is only 64 KiB.  Without the fast fence, metadata and
activation fit in one 128 KiB segment.  The route-first fast layout must flush
the prefix, creating two 128 KiB sends.  It therefore adds a segment while
only overlapping worker command-buffer setup and one host copy.  The user's
qualitative A/B found no noticeable change, which is expected from this
geometry.  No quantitative PP fast-fence A/B has been preserved yet.

PP already supports arbitrary A -> B -> C -> D routes and worker-to-worker
RDMA.  It is not a physical ring: the final RESULT currently relays back
through upstream workers.  A direct final-worker -> coordinator result edge can
reduce relay overhead, but it cannot parallelize a single autoregressive token.
PP+DSpark is not qualified: distributed evaluation does not prepare the normal
support-model proposal, and hybrid TP+PP is explicitly rejected.

## DSpark/MTP on `tp-mtp-hunt`

### What has been implemented and fixed

| Commit | Main content |
|---|---|
| `46034ae` | Batched TP verifier MoE, direct accepted-state commits, fast batch gates, GPU Markov, verifier economics, Q8 head tiles, benchmark support |
| `11758f4` | Official anchor/proposal semantics, direct `main_x` KV, transient draft KV, 128-row window, pending lifecycle, prefill/session support-cache maintenance |
| `98f25f0` | Accept compact row-offset target-hidden capture views across Metal/CUDA/ROCm |
| `ccf4de2` | Apple TP d5-only economic admission, 64/128/256 backoff, deferred 128-row hidden history and batched support-KV catch-up |
| `644d00d` | MXFP4 DSpark planning fix and atomic full-fat `dflash` GGUF importer |
| `242642c` | Type-aware F32 DSpark HC projections instead of interpreting F32 payloads as F16 |

The corrected proposer now matches the official 0731 architecture in the
important structural details:

- captured `h_P` is paired with the target-predicted anchor `x_{P+1}`;
- the pending proposals are proper future tokens beginning at `x_{P+2}`;
- each stage persists direct `kv_norm(wkv(main_x))` at P;
- anchor/noise KVs are transient;
- support attention sees the last 128 target rows;
- confidence reads the unnormalized post-HC-head state; and
- accepted verifier captures advance the support ring without admitting
  rejected speculative rows.

Chunked prefill preserves the last 128 captures across chunk boundaries.
Prompt extension, rewind, ordinary sampled decode, tail exits, and reused
sessions repair or invalidate pending/support state rather than silently
disabling speculation.  The compact-capture and F32-HC fixes were both real
runtime bugs found by the remote two-node smoke tests.

### Support GGUFs

The published 5.99 GB support GGUF is compatible, but aggressively compressed:
IQ2_XXS/Q2_K routed experts and Q8 Markov/confidence heads.  The full-fat
llama.cpp `dflash` GGUF cannot be passed directly to `--mtp`; import it with:

```sh
gguf-tools/deepseek4-quantize \
  --import-dflash-gguf ~/Downloads/DeepseekV4-Flash-20260731-DSpark.gguf \
  --out ~/Downloads/DeepSeek-V4-Flash-DSpark-support-full.gguf
```

For the pinned artifact, the input is 10,896,057,568 bytes and the DS4 output
is 10,835,055,488 bytes.  The importer stream-copies 75 tensors, expands three
BF16 router gates plus confidence to F32, converts two BF16 Markov matrices to
Q8_0 for the Metal fast path, and preserves nine MXFP4 expert tensors.  The
output inventory is 45 F32, 27 Q8_0, and 9 MXFP4 tensors.

Direct native `dflash` loading would mostly be a convenience feature, not an
acceptance fix: 75/81 payloads are already copied byte-for-byte, and BF16 -> F32
router/confidence expansion is numerically exact.  Only the two Markov Q8
conversions lose precision; native BF16 Markov would require new kernels and
would read more data.

### Measured results

All rows below are context 512 / generation 512 on the same M2 Ultra pair.

| Arm | Steady t/s | Acceptance/economics |
|---|---:|---|
| Target only, no support model | **41.98** | Control |
| Published 5.99 GB, old production policy | 38.82 | 13 proposed, 2 accepted; net saved **-896 ms** |
| Published 5.99 GB, forced d5 | **21.35** | 1,596 proposed, 189 accepted (11.84%); 35.5% first hit; no `d5>a5` |
| New low-yield production policy | **41.57** | 499 scheduler skips, 4 backoffs, 3 history flushes; no verifier; net **-112 ms** |
| Forced d5 + verifier head split | **20.18** | 1,785 proposed, 151 accepted (8.46%); no speedup from the split |
| Full-fat MXFP4 sidecar before F32-HC fix | 28.05 | 0/510 first hits; invalid result that exposed the F32-as-F16 bug |
| Full-fat MXFP4 sidecar after F32-HC fix, forced | **19.91** | 1,767 proposed, 157 accepted (8.89%); one short full block, no `d5>a5`; net **-13.09 s** |

The published-sidecar forced run spent 3.41 s proposing and 12.33 s in 114
verifiers.  Each five-row verifier cost about 108.2 ms while a normal target
token cost about 24.5 ms.  It accepted only 1.66 draft tokens per launched
verifier.  At that cost, even a perfect five-token block is only marginal; the
observed acceptance is nowhere near break-even.

The low-yield policy is a stop-loss, not a speculative speedup.  It admits only
full d5 proposals to TP verification, backs off 64 -> 128 -> 256 cycles after a
bad probe, and batches support-KV catch-up at the next probe.  On this prose
fixture it recovers target-only speed by doing almost no draft work.

The often-cited llama.cpp result of about 50% acceptance and mean 3.5 is not a
direct contradiction.  It used a predictable code fixture, and llama's mean
includes the unconditional target token: about 50% means roughly 2.5 accepted
draft tokens out of 5 and 3.5 output tokens per verification cycle.  Even so,
the current 8--12% Promessi acceptance remains too low and deserves a matched
code-prompt/runtime comparison before further verifier tuning.  The referenced
GaelicThunder drafter is also a Q2_K/Q8 presentation, not evidence that the
10.8 GB full-fat sidecar should independently reach the headline.

### Current DSpark recommendation

- For maximum throughput, omit `--mtp`/`--dspark`.
- For production-policy characterization, use `--dspark` and
  `DS4_DSPARK_STATS=1`; the low-yield policy should stay near target-only on
  unproductive text.
- For raw acceptance/economics, use `DS4_DSPARK_SCHEDULER=0` and
  `--dspark-confidence 0`.  This is intentionally slow.
- `DS4_DSPARK_TP_VERIFY_HEAD_SPLIT=1` is still experimental and did not improve
  the measured forced run.  It changes verifier gate participation and must be
  set identically on both ranks when tested.
- Do not optimize verifier kernels again until a matched predictable-code
  fixture demonstrates much higher acceptance.

The forced coordinator arm is the normal TP command plus:

```sh
DS4_METAL_FAST_SYNC=1 \
DS4_DSPARK_SCHEDULER=0 \
DS4_DSPARK_STATS=1 \
./ds4-bench ... \
  --mtp "$DSPARK" --dspark-confidence 0
```

For production policy, remove `DS4_DSPARK_SCHEDULER=0` and replace
`--dspark-confidence 0` with `--dspark`.  Only the coordinator loads the
support GGUF; both ranks still need the same binary and fast-fence environment.

## Settled negative, neutral, or inconclusive experiments

| Experiment | Result |
|---|---|
| TP RDMA hot-path cleanup | No measurable throughput gain; excluded from current branches |
| Fused TP gate kernel | No measurable gain and plausible mid-dispatch visibility cost; excluded |
| PP fast activation fence with fixed 128 KiB segments | Qualitatively neutral; no recorded numeric A/B and no prefill effect |
| DSpark TP verifier head split | Inconclusive/not promotable: forced headline fell from about 21.3 to 20.2 t/s, but acceptance and generated paths changed |
| Full-fat MXFP4 DSpark sidecar | Fixed the zero-hit F32-HC bug, but did not improve Promessi acceptance |
| Packed32 FlashAttention reduction at 32 TP heads | About 1.35 t/s slower; reverted |
| Fused router projection/select under TP | Broke output; reverted |
| Three TP gate-exclusion removals | About 0.019 ms total, effectively zero |
| Model-untracked hazard mode | Noise |
| More GPU polling threadgroups/backoff | Materially worse GPU cost |
| One-sided Apple RDMA WRITE | Provider accepts WRs but never executes them |

## Prioritized future work

### 1. Re-establish controls before every campaign

Record target-only TP, production DSpark, and forced DSpark using the same
binary and ABBA ordering.  Keep a predictable code fixture in addition to
Promessi so workload sensitivity is visible.  Save the CSV row and complete
`DSpark stats/detail` lines.

### 2. Balance routed experts on two nodes

This is the best measured two-node compute opportunity: about 1.47 ms/token,
or a path from roughly 41.6 to **44 t/s**.  Start by collecting real route
co-occurrence traces; then compare partial replication against a repacked
intra-expert layout.  Do not assume uniform routing without checking held-out
prompts.

### 3. Build native four-node TP, not a decode ring

The four-node choices serve different goals:

| Topology | One-stream decode | Aggregate behavior | Verdict |
|---|---:|---:|---|
| Two independent TP2 replicas | 41--42 t/s per stream | **82--84 t/s** with two streams | Best low-risk serving option |
| Existing PP4 chain | Roughly 25--32 t/s projected | Can pipeline independent work/prefill | Capacity and long-prefill option |
| Hybrid TP2 x PP2 | Roughly 40--42 t/s projected | Roughly 75--82 t/s with at least two streams | Highest complexity; no compelling one-stream gain |
| Native TP4 mesh | **47--58 t/s** depending on expert layout | One faster stream | Best single-stream topology |

For one active sequence, the preferred four-node topology is one TP group on a
true six-link full mesh.  Each rank owns 16 attention heads, two of eight output
groups, one vocabulary quarter, one shared-expert quarter, and initially 64
routed experts.

Decode gates are only 16 KiB and occur 86 times/token.  Use a one-round flat
all-gather: every rank posts three receives and three sends concurrently, then
sums rank 0,1,2,3 in canonical order, ideally folded into the existing HC
consumer.  A ring reduce-scatter/all-gather takes six serial phases at N=4 and
is the wrong latency tradeoff for this payload.

| TP4 design | Realistic single-stream decode | Notes |
|---|---:|---|
| Disjoint 64-expert quarters | **47--51 t/s** | Simplest first implementation; route imbalance limits MoE scaling |
| Two-copy cyclic expert placement | **52--56 t/s** | Likely best practical design; about 75--77 GiB/node |
| Repacked intra-expert slicing | **53--58 t/s** | Best balance, larger model/tooling change |
| Aggressive ideal ceiling | **60--63 t/s** | Assumes near-pair gate latency and excellent quarter-width kernels |

The first go/no-go is a model-free 3-peer 16 KiB all-gather benchmark.  Target
**40--60 us/gate p50** with acceptable p99.  At 70 us, even the two-copy design
falls toward 49 t/s; at 100 us it falls to roughly 44 t/s.

Implementation work includes group rendezvous and rank/world metadata; three
QPs/CQs and receive windows per rank; concurrent control broadcast/ACK; an
aggregate arrival bitmask and one GPU fence; deterministic add4/add4+HC;
quarter model views and kernels; output-quarter assembly or an exact greedy
max/id protocol; a bandwidth-oriented bulk collective for prefill; and
four-rank failure/session/rewind testing.  Disable DSpark initially.

The main risks are provider or CQ serialization across nominally independent
links, quarter-width kernels underfilling the GPU, routed-expert hotspots and
slowest-rank jitter, UC loss without retransmission, system-coherent GPU polling
across three NIC DMA sources, reduction-order drift, and the many pair-only
assumptions in TP session/verifier/snapshot protocols.

Three-node TP is almost the same engineering effort with worse geometry: 64
heads, eight output groups, 256 experts, and the vocabulary do not divide
cleanly by three.  Its rough projection is only 45--48 t/s.  Implement world
sizes two and four first.

### 4. Resolve DSpark acceptance before chasing verifier milliseconds

Use the same code prompt, target model/quant, tokenization, draft length, and
metric definitions as the public llama.cpp result.  Capture proposal IDs and
stage0/base logits for the published and full-fat sidecars.  If Markov precision
is still suspected, an F32 Markov diagnostic measures its maximum quality
benefit, but it will disable the fast Q8 GPU path and is not a throughput arm.

### 5. Rework PP transport only for a concrete use case

To make PP fast sync useful, avoid the extra 128 KiB prefix segment.  Candidate
designs are a small control/prefix ring, variable-length SEND geometry, or a
pre-registered Metal activation slab receiving directly from RDMA.  A direct
final-worker -> coordinator RESULT edge is also reasonable on a mesh.  These
changes improve handoff overhead but still do not parallelize single-stream
autoregressive decode.

For aggregate throughput, two independent TP2 replicas are much simpler than
hybrid TP2 x PP2 and give about **82--84 t/s aggregate** across two active
streams.  Hybrid mode requires a new `(pp_stage,tp_rank)` topology, activation
broadcast inside each stage, intersected layer/expert ownership, composed
session/KV/snapshot lifecycle, and a multi-session scheduler.  Build it only
when aggregate serving or model capacity justifies that complexity.

## Validation checklist

For changes touching the common/TP/MTP stack:

```sh
make -j4 ds4-bench ds4_test
./ds4_test --server
./ds4_test --metal-kernels
make test-mxfp4-metal
make check-dispatch-count
git diff --check
```

For support tooling:

```sh
make -C gguf-tools all test
```

The local development Mac is suitable for correctness and kernel tests, not
for quoting M2 Ultra pair performance.  Remote two-node validation must include
a target-only run, output/stream sanity, clean shutdown, and the exact transport
and fast-fence log lines.

## Generic benchmark and plotting commands

Run a normal local benchmark as:

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Name CSV files after the hardware.  To generate an SVG using only the Python
standard library:

```sh
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The output defaults to a sibling `_ts.svg` file.

### Metal decode schedule A/B

```sh
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates variant order and
variant-to-session assignment.  It aborts unless full-vocabulary logits are
bit-identical and, with `--include-selection`, selection matches.  Use
`--candidate-env NAME` for a rollback control.

For example, compare the default pre-M5 ratio-4 compressor pack/transpose
fusion with its legacy rollback using:

```sh
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### Metal prefill variant A/B

```sh
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

The harness warms both variants, alternates ABBA/BAAB order, poisons host logit
buffers, and requires bit-identical final logits.  Use `--help` for prompt,
repeat, and candidate controls.

To isolate the routed-down tail-SIMDgroup cull from the retained pair default:

```sh
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The default prefill fixture is an 8192-token prefix with an automatically sized
8193-token context and two repeats.  Every run uses fresh sessions on one Metal
engine and aborts on any final-logit mismatch.
