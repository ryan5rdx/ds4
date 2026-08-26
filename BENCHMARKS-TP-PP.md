# DS4 — Full-Sweep (128k) Benchmark Reference, 2× M2 Ultra 128GB

Reference for running and reading the standard **full sweep** (ctx 2048→131072,
step ×2) of TP and PP distributed inference on the Apple Metal pair.

Model: **DeepSeek V4 Flash** (MXFP4 experts, F16 HC/compressor/indexer, Q8
attn/shared/output).

- Model file (both hosts): `~/Downloads/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf`
- Size: 155,976,458,848 bytes (~145 GiB), 43 layers (0:42 + output)

## The standard sweep

- ctx 2048 → 131072, `--step-mul 2` (7 points), 128 gen tokens per point.
- **Sweep semantics**: each frontier *advances* from the previous one
  (`ds4_bench.c`), so the `131072` row is the 65536→131072 *increment*, not a
  cold 131k prefill. A row's mean attended position is ~3N/4. Use the sweep for
  A/B between configs on the same rig; use a cold single point
  (`--ctx-start 131072 --ctx-max 131072`) for an honest cross-machine number.
- Prompt file: a text file with **≥ `--ctx-max` tokens** (the bench rejects
  shorter prompts). Standard prompt since 2026-08-26: **`speed-bench/promessi_sposi.txt`**
  (Manzoni, *I promessi sposi*; 1,329,139 bytes, md5 `2edade70f1d2d24c8c34c3861170fa9d`,
  >200k tokens — verified against the bench at `--ctx-max 200000`). Persistent
  copies at `~/Downloads/promessi_sposi.txt` on both hosts (survive reboot;
  re-copy from the repo after a fresh checkout).
  - Earlier runs (Runs 1–3, R1–R11) used a seeded word-soup `/tmp/bench_long.txt`
    (~135k tokens, lost on reboot): in-campaign A/B stays comparable, but
    cross-campaign numbers across the prompt switch are not 1:1, and the
    word-soup's degenerate repetition regime inflated control-vs-control
    logit deltas (see R11b).
  - One observed worker GPU timeout (`kIOGPUCommandBufferCallbackErrorTimeout`)
    during a 200k-token prefill attempt on 2026-08-26 — an artifact of that
    size check, no conclusion drawn; benchmark contexts stay ≤131072.
- `--step-mul 2` is what makes it 7 points; without it the step defaults to a
  2048-token linear sweep (64 points, many hours at 128k).
- `prefill_tps` in a row = incremental prefill of the added tokens.
- Quote `gen_steady_tps` (not first-token-affected `gen_tps`) for decode.

## Hosts and rig

| Host | Role | TB IP | RDMA dev | Metal |
|------|------|-------|----------|-------|
| `moiraine@lanfear.local` | coordinator | 192.168.0.6 (`en6`) | `rdma_en6` | M2 Ultra 128GB |
| `moiraine@mat.local` | worker | 192.168.0.5 (`en7`) | `rdma_en7` | M2 Ultra 128GB |

- Both repos at `~/Downloads/ds4r/ds4`; local (Linux, no Metal) repo at
  `/home/moiraine/Projects/ds4`. **Builds must happen on the Mac hosts.**
- Passwordless ssh + sudo to both.
- Only test with **mat + lanfear**. rand/dashiva are read-only (dashiva tmux for
  server debugging).
- Use `ps aux` (NOT `pgrep -f "ds4-bench --role"`) to check liveness — the model
  path sits between args so that pgrep pattern is a false negative.
- **Any hang → restart BOTH ranks.** `tp->failed` is sticky; a wedged pair does
  not recover in-process.

### RDMA link setup (re-run after every reboot)

Config is runtime-only. On **both** nodes:

```
cd ~/Downloads/rdma-tb4/tests && ./setup-rdma-net.sh
```

- Un-bridges the TB ports and gives each cable its own /30 pair
  (mat:en7=192.168.0.5 ↔ lanfear:en6=192.168.0.6), which creates the
  IPv4-mapped RoCE-v2 GID at index 1 (RTR needs `sgid_index=1`).
- Verify with `./check-roce-v2-gid.sh` (expect `GID index 1 present` on the
  active ifaces) and `./uc_pingpong` (listener on one end, client on the other:
  `./uc_pingpong -d rdma_en6 -l` / `./uc_pingpong -d rdma_en7 192.168.0.6`).
- The script's inline "GID[1] missing" warning is a display quirk — trust
  `check-roce-v2-gid.sh`.
- If pingpong/RTR starts failing errno=1 (EPERM, "inet_arp_lookup failed") after
  working: run `./setup-rdma-net.sh --reset` on BOTH ends of the cable.

## Prerequisites for a comparable run

1. **Same commit on both hosts**: `git rev-parse HEAD` must match. `git pull`,
   then `make -j ds4-bench ds4-server ds4` on both. Do not reuse stale binaries.
2. **Env symmetry**: flags that change per-layer gate counts
   (`DS4_TP_PREFILL_SPLIT_NONZERO`, `DS4_TP_FORCE_DENSE_ATTN_OUT`,
   `DS4_TP_SUBGATE_PIPELINE`, …) **must be set on both ranks or neither** —
   asymmetric settings deadlock the gate exchange, they do not degrade.
3. **`DS4_METAL_FAST_SYNC=1` on both ranks** for all published numbers.
4. RDMA device names are host-local (lanfear=`rdma_en6`, mat=`rdma_en7` per the
  table above); a wrong name is a failed QP bring-up, not a silent slowdown.

## Launch commands

### TP pair (tensor-parallel over RDMA) — the coordinator runs the sweep

**lanfear (coordinator):**
```
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role coordinator \
  --tensor-parallel --transport rdma --listen 0.0.0.0 1234 \
  --rdma-device rdma_en6 --prompt-file ~/Downloads/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 131072 --step-mul 2 \
  --gen-tokens 128 --csv /tmp/tp_sweep.csv
```
**mat (worker):** (no sweep flags; it mirrors the leader's session)
```
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role worker \
  --coordinator 192.168.0.6 1234 --transport rdma --tensor-parallel \
  --rdma-device rdma_en7
```

Notes:
- Launch the coordinator first (background, log to a file), then the worker.
- Under `nohup` stdout is fully buffered — watch the `--csv` file for progress,
  not the log.
- Correctness/bits: add `--show-output` (prints generated text) and/or
  `--dump-frontier-logits-dir DIR` (per-frontier logits JSON) to the
  coordinator; compare dirs between arms. Plain `cmp` of the bench log is
  useless — it contains timing lines that always differ.

### PP pair (pipeline-parallel over RDMA)

- **lanfear (coordinator / bench):** layers 0:21
  ```
  DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role coordinator \
    --layers 0:21 --listen 0.0.0.0 9000 --dist-transport rdma \
    --dist-rdma-adj-devices rdma_en6 --prompt-file ~/Downloads/promessi_sposi.txt \
    --ctx-start 2048 --ctx-max 131072 --step-mul 2 \
    --gen-tokens 128 --csv /tmp/pp_sweep.csv
  ```
- **mat (worker):** layers 22:output — **must use `./ds4`** (serving binary),
  not `ds4-bench` (rejects `--role worker`)
  ```
  DS4_METAL_FAST_SYNC=1 ./ds4 -m <MODEL> --role worker --layers 22:output \
    --coordinator 192.168.0.6 9000 --dist-transport rdma \
    --dist-rdma-adj-devices rdma_en7 --ctx 132096
  ```
- **`--ctx` on the PP worker must be ≥ coordinator's `ctx_alloc`**
  (`ctx_max + gen_tokens + 1`). Worker default is 32768 — with `--ctx-max 131072`
  the coordinator allocates 131201 and rejects the worker with "worker context
  too small" (`DS4_DIST_MSG_ERROR`, type 2) → reconnect loop. Use `--ctx 132096`.

## Results

Format: one table per (branch, commit, date). Newest first. Every table should
name the branch + HEAD commit + date + the exact env flags that differ from
defaults; the sweep semantics above apply to all rows.

### Current state — `upstream-metal-wins` @ `f3668a1`/`627ccc1`, 2026-08-25

Consolidated, from the two full sweeps above (TP = all three splits + nsg4
default; PP = same-commit TCP-PP). `627ccc1` is server-only
(`--prefill-quantum`) and does not touch `ds4-bench`; the sweep numbers are
from `f3668a1` and carry over.

| ctx | TP prefill | PP prefill | prefill winner | TP dec | PP dec | decode winner |
|---|---|---|---|---|---|---|
| 2048 | 423.03 | 316.98 | TP +34% | 40.91 | 27.67 | TP +48% |
| 4096 | 417.54 | 310.46 | TP +35% | 36.47 | 26.29 | TP +39% |
| 8192 | 500.16 | 348.21 | TP +44% | 35.98 | 26.08 | TP +38% |
| 16384 | 480.79 | 459.38 | TP +5% | 35.43 | 25.54 | TP +39% |
| 32768 | 461.20 | 530.00 | PP +15% | 33.73 | 24.11 | TP +40% |
| 65536 | 427.08 | 520.42 | PP +22% | 31.52 | 22.52 | TP +40% |
| 131072 | 367.29 | 444.38 | PP +21% | 28.13 | 20.57 | TP +37% |

Cold 131k prefill (TP): 402.64 — +54.9% over the pre-workstream flag-off cold
baseline (259.90); sweep +65.8% (221.50 → 367.29). TP wins prefill at
≤16k and decode at every point; PP wins prefill from 32k up.

### R11 — decode gate count halved via replicated attention — `upstream-metal-wins` @ `a0cf853` build, 2026-08-26 — **NEGATIVE: decode −8.5 to −11.8%, avenue closed**

Flag: `DS4_TP_DECODE_REPLICATE_ATTN=1` on both ranks (`a0cf853`). Replicates
the whole decode attention block on both ranks; the per-layer ATTN row-gate
disappears and the decode schedule becomes `start=DS4_TP_GATE_FFN, step=2,
per_token=43` (43 gates/token instead of 86). Prefill code is untouched
(code review + measurement below). The build also carries `dedc830`
(compressor-frontier rewind); the flag-off A arms reproduce the R10-era
baseline (2k decode 40.86 vs 40.91; 131k cold prefill 399.99 vs ~399–402;
131k decode 28.27 vs 28.13–28.31), so the new binary is inert and every A/B
delta below isolates the flag.

**R11b correctness (16384 greedy, 7 frontiers 4096→16384; control / candidate /
control-2): PASS, with a baseline note.** Frontier dumps cover the first decode
position; the 128-token texts were compared per frontier.

| frontier | B vs A max\|Δlogit\| | A2 vs A max\|Δlogit\| |
|---|---|---|
| 4096 | 0.0448 | 0.0000 |
| 6144 | 0.0000 | **0.6876** |
| 8192 | 0.3255 | 0.0000 |
| 10240 | **0.6651** | 0.0000 |
| 12288 | 0.0000 | 0.0000 |
| 14336 | 0.6593 | 0.0000 |
| 16384 | 0.0000 | 0.0000 |

argmax: 7/7 identical in both comparisons. The in-session control-vs-control
baseline (0.688) is ~125× the R2-era 0.0055 — this prompt plus the
restore/incremental frontier chain amplifies run-to-run nondeterminism far
more; the candidate (0.665) sits *inside* the in-session baseline (0.97×),
which is the gate the protocol actually defines. Generated text: 2/7
frontiers identical over all 128 tokens (4096, 8192 — including the 0.3255
frontier: the first-position Δ never flipped a greedy choice), 5/7 diverge
within 1–2 tokens of a degenerate "of/and" repetition regime. No O(1) logit
spike, no collapse → summation-order effect as predicted, not a gate-slot bug
(a slot carrying a partial nobody sums would diverge ~10× at logit magnitude
~27). Dense-attn-out b-arms cross-check: decode 36.4→35.5 (A) vs
33.1→32.3 (B), same ~−9% shape.

**R11c throughput — the verdict.** Same flag set otherwise; flag off vs on:

| ctx | decode off | decode on | Δ |
|---|---|---|---|
| 2048 (cold) | 40.86 | 36.05 | **−11.8%** |
| 131072 (cold) | 28.27 | 25.88 | **−8.5%** |

Same-protocol 7-point warm sweep, flag off (run same day) vs on:

| ctx | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|---|
| decode off (A_sweep) | 40.86 | 36.40 | 35.95 | 35.42 | 33.70 | 31.56 | 28.13 |
| decode on (B_sweep) | 36.03 | 33.18 | 32.80 | 32.34 | 30.87 | 28.80 | 25.78 |
| B/A | 0.882 | 0.912 | 0.912 | 0.913 | 0.916 | 0.913 | 0.916 |

Uniform −8.4 to −11.8% at all 7 contexts, worst at the short end.

**Prefill is not moved by the flag — but this rig's sweep prefill is noisier
than the doc had suggested.** Per-point sweep prefill, B vs A:
+0.6 / −0.3 / +0.3 / +5.5 / +14.1 / −12.7 / +1.7%. The two extreme points
(32768: B 460.95 vs A 404.00; 65536: B 371.31 vs A 425.21) are each a
~12–13% single-point dip in *one* sweep, and at both points the *other* flag
state sits on the historical baseline (461.20 / 427.08). The flag does not
enter prefill code, the cold A/B prefill agrees to 0.4% (2k: 443.73/444.80;
131k: 399.99/401.47), and the dips sit in opposite runs — per-sweep rig
noise, not a leak. Treat warm-sweep prefill rows as ±5% with occasional
single-point ~13% dips; same-day A/B is the only clean comparison.

**Why the bet lost (interpretation, labeled).** R10c was right that decode is
stall-bound (30 W, ~54% of prefill power) and wrong about the *content* of the
stall. A TP gate wait is not idle: it is the window in which the peer rank's
half of the attention runs, and at 2k each rank's GPU is already ~54% busy, so
that work hides inside the waits. Replication removes 43 waits (43 × 293 µs ≈
12.6 ms/token at 2k) but also removes the peer compute that overlapped them,
and adds real traffic — full `q_b`/`o_a` weights (+1.6 GB/token/node) plus
full-head KV reads, which only bite at long context. Measured net: a uniform
+2.9 to +3.2 ms/token *cost*. The bubble the lever assumed was not free space;
it was the peer's attention.

**Verdict: clean negative, avenue closed.** As the request predicted: "I would
not be surprised by a negative result — record it either way, because a clean
negative closes the last lever and says the 86-gate structure is load
bearing." Reducing the gate count has to come from a structure that does not
add per-token work on the ranks (replication does); what remains of decode
work is per-gate fixed cost (R10e), which is transport/engine design, not a
knob.

**Ops notes.** (1) `--dump-frontier-logits-dir` does not mkdir its target; a
missing dir aborts the arm with a header-only CSV — the driver now
pre-creates it and validates row count. (2) The worker redirect targets the
results tree on the *worker* host; the driver now mkdirs it there too — the
campaign's first launch silently never started a worker, and the coordinator
sits in "waiting for worker" with no timeout (arm timeout is the only
resolution). (3) A fixed-timeout driver pkill kills a live sweep mid-run
(65536→131072 is ~150 s): the same-day A_sweep control was killed once by its
own 420 s timer and re-run at 780 s — wait on process exit, not a sleep.

### R10 — gate path + first decode profiling — `upstream-metal-wins` @ `f3668a1`/`627ccc1` build, 2026-08-26

Binary: the same `ds4-bench` build as Runs 1–3 and R1–R9 (built Aug 26 00:16;
only docs-only commits landed on the branch since). All arms carry the current
flag set (all three splits + `DS4_METAL_FAST_SYNC`); per-arm additions as
noted. A/B arms are single replicates unless stated.

**R10a — `DS4_TP_SUBGATE_PIPELINE=1` re-test (wash; the sub-gate avenue is
closed).** A/B/A battery, control (A) vs subgate-on (B), full 7-point sweep +
cold 131k. The A2/B2 replicate rows were lost in an ops incident (see notes
below); the verdict stands on the A1/B1 pair, which is well calibrated against
the pre-workstream baseline:

| ctx | A1 prefill | B1 prefill | ΔB1-A1 | baseline¹ prefill | A1 dec | B1 dec | baseline¹ dec |
|---|---|---|---|---|---|---|---|
| 2048 | 387.98 | 404.04 | +4.1% | 423.03 | 36.66 | 40.54 | 40.91 |
| 4096 | 378.67 | 415.86 | +9.8% | 417.54 | 32.32 | 36.35 | 36.47 |
| 8192 | 492.46 | 497.88 | +1.1% | 500.16 | 35.94 | 35.94 | 35.98 |
| 16384 | 478.28 | 473.37 | −1.0% | 480.79 | 35.39 | 35.34 | 35.43 |
| 32768 | 459.13 | 456.16 | −0.6% | 461.20 | 33.69 | 33.65 | 33.73 |
| 65536 | 424.30 | 419.90 | −1.0% | 427.08 | 31.53 | 31.31 | 31.52 |
| 131072 | 364.29 | 363.74 | −0.2% | 367.29 | 28.31 | 28.32 | 28.13 |
| cold 131k | 399.34 | 400.22 | +0.2% | 402.64 | 28.30 | 28.22 | 28.13 |

¹ "Current state" table above, same build, earlier sweeps.

The +4/+10% "gains" at 2k/4k are an artifact of the *control* arm, not the
sub-gate flag: A1 (run first) sits 8–9% below baseline on 2k/4k prefill and
10–11% below on decode, while B1 matches baseline to within −4.5%/−0.4%
prefill and −0.9%/−0.3% decode. At every context ≥8k the two arms agree within
±1.1% prefill and ≤0.7% decode — inside run-to-run noise, and both track
baseline. So the flag's effect is **indistinguishable from zero at every
context**, including the 131k point where the 18.5% BIG-wire target lives.
The R9-era rejection ("net-negative on the M5 Max pair") is superseded by
"no measurable effect here even at wait:wire 4.3:1" — sub-chunk overlap buys
nothing on this rig, and the sub-gate avenue closes for good: the hoped-for
lever on the 18.5% BIG-wire target was a hypothesis about *where* the wait
lives, and the measurement says it does not live where sub-gating can reach.

**R10b — link ceiling (4.4 / 4.1 GB/s per direction, not 5).** The request
asked for a 64 MiB message; that is structurally impossible on this provider —
Apple TB UC SEND is capped at 4096 B per WR (anything larger never completes;
>4096 posts EPERM), and `-s` is clamped to MTU. The valid proxy is sustained
4 KiB SEND streaming (50k messages ≈ 205 MB), `uc_bench --bw --send`, both
directions, twice for stability:

| direction | run 1 | run 2 |
|---|---|---|
| mat → lanfear | 35.08 Gb/s (4.39 GB/s) | 35.58 Gb/s (4.45 GB/s) |
| lanfear → mat | 30.42 Gb/s (3.80 GB/s) | 33.03 Gb/s (4.13 GB/s) |

Effective BIG-gate wire is 2.8–3.1 GB/s against this 3.8–4.4 GB/s ceiling,
i.e. **~65–75% of the link** — ~25–35% is on the floor, not the ~40% the
5 GB/s assumption implied, and the two directions are asymmetric (lanfear→mat
is the weaker one, and it varied more run-to-run). The ceiling is real and
stable; closing the remaining 25–35% is an overlap/staging-window problem, not
a "the link can simply go faster" one. R10b's decision criterion ("if the
link delivers ~5 GB/s, staging tuning is back on the table") resolves as:
staging/window tuning stays on the table, against a 4.4 GB/s ceiling.

**R10c — decode residency and power @2048 and @131072 (decode is not
compute-saturated; the gate chain, not the kernels, is the target).**
`powermetrics --samplers gpu_power` (5 s samples). The sampler on these boxes
reports GPU *power*, not residency %, so the 86–91% prefill figure cannot be
reproduced directly; power is the proxy. At 131k the phases are cleanly
separable (gen 1024 → ~36 s decode window):

| phase | @2048 | @131072 |
|---|---|---|
| prefill | ~32 W (merged into the window) | ~55.5 W (55.0–56.9) |
| decode (window) | ~32 W (31.2–32.5, 12 samples) | **~30 W** (30.2–30.4, 7 samples) |
| idle | ~0.1 W | ~0.1 W |

decode sits at ~54–58% of prefill power at both contexts and is *flat*
through the whole window — a stable stall-bound regime, not a decaying one.
The R10c prior ("low, 30–50%") is confirmed in power terms: decode is now
measured to be the weaker half in every sense, and the lever order is settled
— the 86 gates/token (R10e) before the kernels. Combined with R10e, the
implication is unambiguous: decode's bottleneck is the gate chain, so the
link speed measured in R10b buys decode nothing, and kernel tuning is second
queue behind reducing the gate count.

**R10d — `DS4_METAL_DECODE_NWG` sweep @2048 and @131072 (plateau at 8–32 on
both contexts; small NWG costs real decode; the default is already at the top
of the plateau).** 7 arms each, gen 128:

| NWG | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| decode @2048 t/s | 35.21 | 38.18 | 40.41 | 40.90 | 40.69 | 40.82 | **40.91** |
| decode @131072 t/s | 25.32 | 27.11 | 27.92 | 28.12 | **28.18** | 28.08 | 28.09 |

Both contexts show the same shape: a flat plateau from NWG 8 up (2k: 40.4–40.9,
131k: 28.08–28.18, within noise) and a monotonic climb out of small NWG —
NWG 2 costs 14% at 2k and 10.6% at 131k (25.32 t/s), NWG 4 costs 6.7% and 3.8%.
At 131k the first-token latency tracks it too (966 ms at NWG 2 vs 570–618 at
≥4). Prefill is flat across the sweep (401–402 t/s at 131k), confirming the
knob only touches decode, as designed. The TP default (32) sits at the top of
the plateau on **both** contexts — 40.91 @2k (exactly the pre-sweep number)
and 28.09 @131k (0.03% off the best arm). The "coarse bucket heuristic" the
request suspected of being a tuned-once constant is in fact already optimal:
no change warranted, and the knob stays — it is a real lever (small-NWG
degrades up to 14%), just one with nothing left to tune on this model.
(One scatter point: the NWG=32 arm's 131k prefill read 373.49 vs 401–402 in
the other six arms; decode in that arm is on-plateau, so it is a prefill
run-to-run dip, not an NWG effect — the knob cannot reach prefill.)

**R10e — decode gate profile @2048 and @131072 (the floor is per-gate fixed
cost; the link is not decode's problem at either context).**
`DS4_TP_GATE_PROFILE=1`, gen 128, same flags; @131k is a fresh re-profile of
the R9 run, not the old numbers:

| | @2048 | @131072 (this run) | @131072 (R9) |
|---|---|---|---|
| BIG gates | 86 @ 40,681.8 µs wait / 13,073.8 µs wire | 2,752 @ 93,516.4 / 22,530.9 | 2,752 @ 93,314.9 / 21,841.2 |
| BIG wait:wire | 3.1 : 1 | 4.2 : 1 | 4.3 : 1 |
| row (decode) gates, final cumulative | 10,234 @ 292.6 µs wait / 34.8 µs wire | 11,008 @ 401.4 / 49.7 | 11,008 @ 418.0 / 29.9 |

- The 86 BIG gates are the single 2048 prefill chunk (43 layers × 2), and
  they dominate short-context prefill in a way the 131k number hides: 86 ×
  53.75 ms = 4.62 s of a 4.96 s prefill — **93% of 2048 prefill wall time is
  BIG-gate wait+wire**, versus 18.5% at 131k where 2,752 gates are spread over
  325 s. At 2048 there is no other prefill work: the gate *is* the prefill.
- The @131k re-profile reproduces R9 on the wait side: 401.4 µs vs 418 µs
  (−4%), and the BIG gates are indistinguishable (93,516/22,531 vs
  93,315/21,841 — sub-2%). The one discordant number is the row-gate wire:
  49.7 µs this run vs 29.9 µs in R9 (+66%). It does not change any conclusion
  below (wire stays a 7–12% minority of gate time either way), but it does
  mean the 131k wire figure should be treated as 30–50 µs with a wide bar.
- Wire as a fraction of wait (the informative quantity): 34.8/292.6 =
  **11.9% @2048** vs 49.7/401.4 = **12.4% @131k** (R9's 29.9 µs would be
  7.2%) — a minority of gate time at **both** contexts under either 131k
  measurement. The link is not decode's problem at either context — R10b will
  not help decode, confirmed.
- Wait scaling: 292.6 µs/gate @2048 vs 401.4 µs @131k — a 1.37× growth
  between the two contexts, while wire moves 34.8 → 49.7 µs (+43%, and see the
  bar above). Most of the growth is on the wait side, i.e. compute, as the
  request hypothesized. Extrapolating the @2048 gate cost: 86 × ~284 µs ≈
  24.4 ms/token ≈ 41 t/s — precisely the measured 2048 decode floor.
  **Short-context decode is entirely per-gate fixed cost**, and the only
  lever with real headroom is reducing the 86 gates/token (2 per layer) — a
  design change, not a knob, as the request anticipated.
- One caution, per the request's own warning: Σwait @2048 = 86 × 292.6 µs =
  25.2 ms vs the 24.8 ms/token measured (closes to within 2%), but adding the
  wire (86 × 34.8 µs) puts Σgate at 28.2 ms > token time — so even in decode
  the gate waits are not all on the critical path at short context; some
  overlap with GPU work. At 131k Σwait alone is 34.5 ms vs 35.5 ms/token
  (the near-tautology the request flagged), and Σgate is 38.8 ms > token
  time — the same overlap, a larger margin. Neither context is a clean
  "wait sums to the token" story; the wait-side scaling is the signal.

**Ops notes (2026-08-26, cost the campaign ~40 min).** (1) One arm
(`r10d_nwg8_2k`, first pass) finished its work — CSV written, worker exited
"leader finished" — but the coordinator deadlocked in teardown (uninterruptible
kernel wait, state U) on RDMA/Metal release; `kill -9` cannot reap a U-state
process, so the driver's timeout fired against a corpse. Resolution: rebooted
lanfear. (2) The reboot wiped `/tmp`: the prompt file was restored from mat
(verified by md5), the TB /30 IPs had to be re-applied
(`setup-rdma-net.sh` — runtime-only by design), and the already-collected
arm CSVs were lost. Everything through A1/B1 sweep+cold was recovered from
the session transcript; the A2/B2 replicate rows and the first-pass nwg
2/4/8 @2048 rows were not, and the three 2k NWG arms were re-run (this
section's nwg @2048 table is the re-run). (3) The PP-worker-doesn't-exit note
from R9 remains valid; a *TP* coordinator can also wedge in teardown as above
— the driver now kills both ranks on timeout and the stale-proc guard aborts
the next arm rather than double-launching, which is exactly what it should do.

### TP — `upstream-metal-wins` @ `3746eae1`, 2026-08-25 (A0 off = baseline)

Full sweep, `DS4_METAL_FAST_SYNC=1` only, all new flags unset (this is the A0
inertness run; see `docs/TP-A0-ROWSPLIT-TEST-PLAN.md`). Startup hang from the
previous session did **not** reproduce: both ranks bound and completed the
sweep cleanly, zero faults.

| ctx | prefill t/s | decode t/s (steady) |
|-----|-------------|---------------------|
| 2048 | 378.94 | 40.88 |
| 4096 | 346.12 | 36.01 |
| 8192 | 379.73 | 35.76 |
| 16384 | 357.48 | 35.06 |
| 32768 | 328.48 | 33.52 |
| 65536 | 282.83 | 31.38 |
| 131072 | **221.50** | 28.06 |

- 131072 = 221.5 t/s vs 183.4 on the old TP base — the upstream indexer stack
  (tiled5 scorer / stream512 topk) landed, as predicted (~228 expected).
- Short-ctx sanity (2048 ctx, 16 gen): 390.80 t/s prefill, 40.37 steady decode.

Correctness (Run 2, 16384-token prompt crossing chunk boundaries, both arms with
`DS4_TP_FORCE_DENSE_ATTN_OUT=1`, temp 0, `--dump-frontier-logits-dir`):
- The test plan's **byte-identical criterion is unsatisfiable on this rig**: two
  identical *control* runs (flag off, no A0) already differ in all 129,280
  dumped logits (max abs diff 0.0055, argmax identical). Cross-run bit
equality does not hold for the unsplit path.
- Split arm vs control: 128 generated tokens **byte-identical**, frontier argmax
  identical (305), all logits differ with max abs diff 0.049 — ≈ F16 ULP at the
  observed logit magnitude (~27). Consistent with accumulation-order change from
  row redistribution, not a token-level defect.
- Verdict: pass as numeric-equality-within-noise. Replace the plan's
  byte-identical gate with: top-1 identity + max logit diff bounded against a
  control-vs-control baseline (baseline 0.0055 vs split 0.049, ~9× — same ULP
  class). Throughput (Run 3) gated on this: proceed.

A0 throughput (Run 3, full sweep, `DS4_TP_PREFILL_SPLIT_NONZERO=1` on both
ranks, `DS4_TP_FORCE_DENSE_ATTN_OUT` unset, 2026-08-25):

| ctx | Run 1 (off) | Run 3 (on) | Δ |
|-----|-------------|------------|----|
| 2048 | 378.94 | 384.72 | +1.5% |
| 4096 | 346.12 | 353.77 | +2.2% |
| 8192 | 379.73 | 422.05 | +11.1% |
| 16384 | 357.48 | 385.54 | +7.9% |
| 32768 | 328.48 | 363.75 | +10.7% |
| 65536 | 282.83 | 309.28 | +9.3% |
| 131072 | 221.50 | 237.44 | +7.2% |

- Gain is real but **not the predicted shape**: expected ~0% at 2048 rising to
  ~1.5× at 131072; observed a mid-context peak (~+11% at 8k/32k) falling back
  to +7.2% at 131k. 2048 is ~0% as constructed. Decode untouched.
- 131072 target was ~1.5× (≈332 t/s); measured 237.4 (+7.2%). The low long-ctx
  gain is mechanistic, not noise: A0 splits attention *consumption* (constant
  4.19e7 MACs/token/layer) but not indexer *scoring* (n_comp × 64 × 128, grows
  linearly; 2.68e8 MACs/token/layer at 131k, 6.4× larger). Confirmed by R1
  below, and fixed by the indexer split (R5) and static-mixed split (R7),
  then the FlashAttention simdgroup fix (R8): cold 131k 259.90 → 402.64
  (+54.9% total), sweep 221.50 → 367.29 (+65.8%). Against **same-commit** PP
  (re-measured, 444.38 @131k) TP wins prefill at ≤16k and PP wins from 32k up;
  TP keeps the decode lead everywhere. (The earlier "TP above PP at 131k"
  headline was against the stale pp-rdma-new table — see the correction below.)

**R5 — indexer split** (`DS4_TP_PREFILL_SPLIT_INDEXER=1` + A0, both ranks;
`589e93e`, 2026-08-25)

R5a — correctness (16384, `DS4_TP_FORCE_DENSE_ATTN_OUT=1` both arms, greedy):

| | A0 only | A0 + indexer |
|---|---|---|
| prefill t/s | 417.81 | 432.56 |
| tokens | 128/128 **byte-identical** to A0 arm | |
| frontier argmax | 305 (logit 27.2699318) | 305 (27.2699318) |
| max \|Δlogit\| | **0.0 — bit-identical, 0 of 129,280 logits differ** | |

Stronger than the ULP-class criterion expected: the split moves no
FlashAttention block geometry, so per-row score/top-k is bit-identical
regardless of which rank computes which rows. **Pass.**

R5c — stage profile re-run (cold 131k, split +
`DS4_METAL_INDEXER_STAGE_PROFILE=1`; profile-run throughput not comparable,
but it also rose 279.81 → 320.30 t/s):

| layer (comp=32768) | score ms | topk ms | attention ms |
|---|---|---|---|
| 2 | 138.20 (was 317.68) | 5.17 (was 10.00) | 41.33 (was 41.33) |
| 4 | 137.96 (was 317.67) | 5.61 (was 10.33) | 39.43 (was 39.43) |
| 6 | 137.44 (was 317.14) | 5.59 (was 11.12) | 39.54 (was 39.54) |

**Mechanism confirmed:** score nearly halved (predicted ~160, got ~138), top-k
halved, attention untouched. score:attention fell 8:1 → 3.4:1.

R5b — throughput (sweep, A0+indexer vs A0-only arm):

| ctx | A0 only | A0 + indexer | Δ |
|---|---|---|---|
| 2048 | 384.72 | 395.36 | +2.8% |
| 4096 | 353.77 | 352.31 | −0.4% |
| 8192 | 422.05 | 427.45 | +1.3% |
| 16384 | 385.54 | 410.85 | +6.6% |
| 32768 | 363.75 | 387.11 | +6.4% |
| 65536 | 309.28 | 344.55 | +11.4% |
| **131072** | **237.44** | **283.64** | **+19.5%** |

Cold single point: 281.20 → **322.64** (+14.8%). Against the flag-off
baseline: sweep 221.50 → 283.64 (**+28.1%**), cold 259.90 → 322.64
(**+24.2%**). The gain now rises with context exactly as the mechanism
predicts (splittable share grows with `comp`). Decode untouched
(28.09 steady).

**R6 — sizing the ratio-128 (odd) layers** (`DS4_METAL_LAYER_STAGE_PROFILE`,
both splits on, cold 131k, 2026-08-25; profile-run throughput not comparable,
but 322.09/322.07 t/s vs 322.64 clean — per-layer profiling is cheap here.
Late chunk pos=126976, tokens=4096.)

| stage | odd `il`=3 (ratio 128) | even `il`=4 (ratio 4) |
|---|---|---|
| `hc_pre` (attn) | 1.935 | 3.307 |
| `norm` (attn) | 0.026 | 0.071 |
| `q_path` | 21.069 | 12.353 |
| `kv_path` | 0.437 | 0.491 |
| `compressor` | 3.070 | 5.027 |
| `indexer_setup` | — (no indexer) | 7.484 |
| `attention` | **255.293** | **182.807** |
| `inv_rope` | 1.959 | 1.101 |
| `output_proj` | **34.857** | 17.190 |
| `hc_post` (attn) | 2.993 | 16.691 |

FFN part (already TP-sharded by the 50/50 expert split, not row-split):
routed_moe 52.3/65.0, hc_post 50.2/23.9, shared 4.5/2.4 both layers.

- Odd-layer attention **grows with context** (192.0 ms at pos 81920 → 255.3 ms
  at pos 126976) and is *larger* than the ratio-4 layer's (255 vs 183 ms):
  these layers carry the bulk of the replicated attention work, not the
  indexer layers.
- **The prize** (`q_path` + `attention` + `output_proj` on one odd layer) =
  **311.2 ms/layer-chunk → × 20 layers ≈ 6.2 s of every chunk pass**, ~71% of
  the odd layer's own ~435 ms chunk time. Large — the mask-slicing work on the
  static-mixed path is justified.
- **The hard floor** (stages that must stay full-width): `kv_path` +
  `compressor` ≈ 3.5–5.5 ms/layer. Row-splitting's ceiling on these layers is
  essentially the whole attention block.
- Note the two profilers measure different groupings: this `attention` stage
  (182.8 ms on the even layer) spans the full attention consumption, while R1's
  indexer-stage `attention` (39.4 ms on the same layer) covers only the
  top-k-selected rows sub-stage.

**R7 — static-mixed (ratio-128) row split** (`DS4_TP_PREFILL_SPLIT_STATIC_MIXED=1`
+ the two split flags, both ranks; `e672c22`, 2026-08-25)

R7a — inertness: full sweep with the R5 flags only on the new binary.
Reproduces the R5 arm within ~1–2% at every point (131k: 284.03 vs 283.64,
4k: 359.24 vs 352.31, 8k: 432.06 vs 427.45). **Pass.**

R7b — correctness (16384, `DS4_TP_FORCE_DENSE_ATTN_OUT=1` both arms, greedy):

| | A0+indexer (control) | + static-mixed (candidate) |
|---|---|---|
| prefill t/s (dense) | 430.86 | 472.61 |
| tokens | 128/128 byte-identical | |
| frontier argmax | 305 (27.2699318) | 305 (27.2699318) |
| max \|Δlogit\| | **0.0 — bit-identical, 0 of 129,280 differ** | |

As predicted: the mask is rebuilt per call from the origin, so a row sub-range
reproduces its own rows exactly. **Pass.**

R7c — throughput, all three flags, vs the R5 arm (projection from R6's
numbers, recorded before the run: sweep 131k → ~340, cold → ~380):

| ctx | R5 arm (A0+idx) | + static-mixed | Δ | PP |
|---|---|---|---|---|
| 2048 | 391.77 | 391.51 | ~0% | 393.90 |
| 4096 | 359.24 | 405.11 | +12.8% | 412.78 |
| 8192 | 432.06 | 485.69 | +12.4% | 454.20 |
| 16384 | 412.35 | 467.73 | +13.4% | 444.70 |
| 32768 | 386.93 | 444.99 | +15.0% | 438.98 |
| 65536 | 345.37 | 401.92 | +16.4% | 373.38 |
| **131072** | **284.03** | **342.25** | **+20.5%** (proj. ~340) | 334.53 |

Cold single point: 322.64 → **380.27** (+18.2%; projection ~380). Full arc,
cold 131k: 259.90 → 281.20 (A0) → 322.64 (A0+indexer) → **380.27** (A0+indexer+
static-mixed), **+46.4%** over the flag-off baseline; sweep +54.4% (221.50 →
342.25). Decode untouched (28.28 steady).

**TP sweep prefill now exceeds PP at 131k (342.25 vs 334.53) for the first
time**, while keeping the ~40% decode lead (28.28 vs 20.09). Both projections
landed within model error; no CPU-mask undershoot observed, so R6's
single-threaded mask-fill concern did not bite at this size.

**R8 — FlashAttention simdgroup count** (`DS4_METAL_FA_NSG=4` + the three
split flags, both ranks; `fa94db7`, 2026-08-25). First change that speeds the
kernel itself instead of removing replicated work.

R8a — inertness: R7-flag sweep on the new binary with the knob unset.
Reproduces the R7 arm at every point (131k: 342.18 vs 342.25, 8k: 484.71 vs
485.69, 4k: 411.25 vs 405.11). **Pass.**

R8b — correctness (16384, DENSE both arms, greedy): **bit-identical** —
0/129,280 logits differ, tokens identical, argmax 305 both (control 469.44 /
candidate 481.12 t/s dense). As the standalone runs predicted: partitioning is
accumulation-neutral. **Pass.**

R8c — throughput, all four flags, vs the R7 arm (projection recorded before
the run: sweep 131k ~375–395, cold ~415–440):

prefill t/s; decode t/s (steady, 127 tok) in parentheses:

| ctx | R7 arm (NSG=8) | NSG=4 | Δ prefill |
|---|---|---|---|
| 2048 | 389.99 (40.91) | 423.03 (40.91) | +8.5% |
| 4096 | 411.25 (36.43) | 417.54 (36.47) | +1.5% |
| 8192 | 484.71 (35.93) | 500.16 (35.98) | +3.2% |
| 16384 | 467.50 (35.42) | 480.79 (35.43) | +2.8% |
| 32768 | 443.12 (33.67) | 461.20 (33.73) | +4.1% |
| 65536 | 400.90 (31.46) | 427.08 (31.52) | +6.5% |
| **131072** | **342.18 (28.31)** | **367.29 (28.13)** | **+7.3%** (proj. ~375–395) |

Decode is untouched by the kernel change (±0.2%) — as expected, the nsg knob
only touches prefill FlashAttention.

Cold single point: 380.27 → **402.64** (+5.9%; projection ~415–440). Gain is
positive at every point and grows with context (1.5% @4k → 7.3% @131k), as
the mechanism predicts, but the magnitude is about half the M1 Max standalone
2.2× — the M1→M2 extrapolation was the flagged risk, and it is the part that
came in soft. Decode untouched (28.18 steady).

Full arc, cold 131k: 259.90 → 281.20 (A0) → 322.64 (+indexer) → 380.27
(+static-mixed) → **402.64** (+NSG4) = **+54.9%**; sweep 221.50 → **367.29**
= **+65.8%**.

**Power/residency capture (this run, both nodes, powermetrics 5 s × 40):**
lanfear 86–89% active residency, 54.5–58.1 W; mat 89–91%, 54.1–57.3 W.
Residency has been flat (~89–91%) at **both 8k and 131k in every arm so far**,
regardless of stage mix (R3: same flatness at 8k off/on; R7/R8: same at 131k).
Since attention's share of layer time dropped substantially across R5–R8 while
residency did not rise, the ceiling is consistent with a fixed per-iteration
overhead (gates/boundaries), not with any single GPU stage. Recorded for the
follow-up residency investigation. Suggested follow-up per the plan: if NSG=4
stays, flip the `ds4_gpu_flash_attn_nonvec_nsg()` default and drop the env
knob. (Default flipped in `f3668a1`; the env knob now only forces a value.)

**PP column re-measured** (R7 carry-over; `upstream-metal-wins` @ `f3668a1`,
2026-08-25). The R7c PP column cited the old `pp-rdma-new` table; re-measured
PP on the *same commit* as the TP column. Transport note: this branch's PP
hidden-state path is plain low-latency TCP pinned to the TB interface
(`DS4_DIST_CONNECT_BIND_IF=en6/en7`); the old `--dist-transport rdma`
flag/path no longer exists in this tree, so these are same-tree numbers, not
the old RDMA-PP ones.

Full data, same commit, same prompt, same day (TP = all splits + nsg4;
decode shown as overall/steady over 128/127 tok):

| ctx | PP prefill | PP dec (overall) | PP dec (steady) | TP prefill | TP dec (overall) | TP dec (steady) |
|---|---|---|---|---|---|---|
| 2048 | 316.98 | 24.47 | 27.67 | 423.03 | 34.93 | 40.91 |
| 4096 | 310.46 | 23.38 | 26.29 | 417.54 | 31.74 | 36.47 |
| 8192 | 348.21 | 23.20 | 26.08 | 500.16 | 30.98 | 35.98 |
| 16384 | 459.38 | 22.82 | 25.54 | 480.79 | 30.76 | 35.43 |
| 32768 | 530.00 | 21.68 | 24.11 | 461.20 | 29.26 | 33.73 |
| 65536 | 520.42 | 20.33 | 22.52 | 427.08 | 27.87 | 31.52 |
| 131072 | **444.38** | **18.70** | **20.57** | 367.29 | 25.16 | 28.13 |

First-token latency (ms) for reference: PP 614–649 (rises with ctx),
TP 545–604.

**Headline correction:** the R7 "first TP sweep prefill above PP at 131k" was
against the stale pp-rdma-new number (334.53). Against same-commit PP
(444.38), **PP wins prefill from 32k up** (530/520/444 vs TP 461/427/367) and
TP wins at 8k and below (500 vs 348 at 8k; 481 vs 459 at 16k). TP keeps the
decode lead at every point (28.13 vs 20.57 at 131k, +37%). The PP column is
now sourced to this run; the old table is kept for history only.

**R9 — `nqptg = 8` ceiling (scoping, no runs, 2026-08-25).** No code, no
bench: the request is a sizing/decision input. Decision arithmetic using this
rig's own numbers: attention is ~27–28% of a late 131k chunk post-R7/R8
(~128 ms/odd-layer + ~39 ms/even-layer of ~12.5 s). A standalone 2× on the
kernel, discounted by R8's measured ~0.63 transfer factor, lands ~1.27× on the
stage → roughly **+6–8% end-to-end** on 131k prefill — same order as the flat
~10% residency lever, so a prototype is only worth it if the M1 Max
restructure shows ≥ ~1.5–2×. The MQA head-sharing prize (64× K/V re-read) is
in the same workstream and is not separately quantifiable from here.

**R9 gate profile — current flags** (`DS4_TP_GATE_PROFILE=1`, all three splits
on, cold 131k, 403.47 t/s; flag is `DS4_TP_GATE_PROFILE`, not
`DS4_METAL_GATE_PROFILE` — the first launch used the wrong name and produced
no data). Final cumulative rows:

| | this run (all splits + nsg4) | R2 ON arm (A0 only) |
|---|---|---|
| total gates | 13,760 | — |
| row gates | 11,008 @ 418.0 µs wait / 29.9 µs wire | 10,768 @ 423 / 26 |
| BIG gates | **2,752 @ 93,314.9 µs wait / 21,841.2 µs wire** | 2,132 @ 190,223 / 23,741 |
| wait:wire ratio (BIG) | **4.3 : 1** | 8.0 : 1 |

- BIG count rose exactly +620 = 20 odd layers × 31 split chunks — the
  `attn_out` swaps static-mixed adds, as designed (no other traffic).
- Per-BIG-gate GPU-wait **halved** (190.2 → 93.3 ms): smaller stages mean
  shorter encoder drains per gate. Wire per gate is unchanged (~22 ms, ~60 MB
  of attn_out at ~2.8 GB/s).
- Totals for the 325 s cold prefill: BIG wire = 2752 × 21.8 ms ≈ **60 s
  (18.5% of wall time)**; row-gate wire ≈ 0.3 s (negligible). The rest of the
  gate time is GPU-wait — the CPU sitting in the gate until the GPU reaches
  the boundary. So the flat ~10% residency ceiling is a serialization story:
  the GPU cannot run ahead of the encoder past a gate, and even the wire
  portion (18.5%) is currently on the critical path rather than overlapped.

Ops note (hit this round): the PP worker (`./ds4`) **does not exit** when the
PP coordinator finishes — it sat idle holding the model for ~20 min and
blocked the next TP launch with the single-instance guard. Kill the PP worker
explicitly after PP runs.

**Run 3b — cold single point (the honest number), 2026-08-25**

| | flag off | flag on | Δ |
|---|---|---|---|
| cold 131072 prefill t/s | 259.90 | 281.20 | +8.2% |

**R1 — indexer stage profile @131k** (`DS4_METAL_INDEXER_STAGE_PROFILE=1`,
split on; throughput under the flag not comparable — stage-boundary command
buffer splits. Relative split is the signal.)

Late-chunk prefill rows (pos=126976, tokens=4096, comp=32768), even layers:

| layer | score ms | topk ms | attention ms |
|---|---|---|---|
| 2 | 317.68 | 10.00 | 41.33 |
| 4 | 317.67 | 10.33 | 39.43 |
| 6 | 317.14 | 11.12 | 39.54 |

- **score : attention ≈ 8 : 1** (predicted 6.4× from the MAC count) — score
  dominates, exactly as the model predicted. The indexer split is the next
  change; its ceiling is roughly the score share (~61% of the ~519 ms/layer-chunk).
- Score grows linearly with `comp` (layer 2): 20.0 ms @ comp=1024 (pos 0) →
  141.0 @ 15360 → 237.7 @ 24576 → 317.7 @ 32768.

**R2 — gate profile @131k cold** (`DS4_TP_GATE_PROFILE=1`, both arms)

| BIG | A0 off | A0 on |
|---|---|---|
| gates | 1419 | 2132 |
| avg gpu-wait µs | 318,073 | 190,223 |
| avg exchange µs | 30,503 | 23,741 |
| eff. per-direction bandwidth (67,108,864 / exchange) | 2.20 GB/s | 2.83 GB/s |

ROW gates: off 10,621 @ 427/28 µs, on 10,768 @ 423/26 µs (wait/exchange); VERIFY
0 in both. A0 adds ~50% more big gates but each swaps a smaller row range
(per-gate time falls); the gpu-wait bubble (0.2–0.3 s per big gate) dwarfs the
wire time (0.024–0.031 s) — the cost is encoder drain / GPU wait, not the
link. Effective 2.2–2.8 GB/s per direction is far below TB4 line rate.

**R3 — GPU utilisation shape** (`powermetrics --samplers gpu_power`, during
131k cold prefills, 2026-08-25)

| arm / ctx | lanfear residency / power | mat residency / power |
|---|---|---|
| off, 131k | 87–90% / 59.9–62.7 W | 92% / 59.7–60.7 W |
| on, 131k | 90–91% / 55.1–55.9 W | 91% / 54.5–54.7 W |
| off, 8k | 87–91% / 51.2–63.6 W | 88–93% / 55.6–63.1 W |
| on, 8k | 85–87% / 52.7–57.3 W | 88–91% / 56.4–56.5 W |

(peaks over the run; 8k rows from background powermetrics during cold 8192
prefills — 8k prefill is ~20 s, so capture the whole window.)

**Answer: the idle is flat (~10%), not growing with context, in both arms.**
8k and 131k sit in the same ~90% class. A constant idle fraction implicates a
fixed per-iteration cost (gate/pipeline overhead, encoder drain — see R2's
gpu-wait ≫ exchange), not context-dependent work.

**R4 — short-context regression** (`DS4_METAL_DISABLE_ARGSORT_CANON=1` both
ranks, 2048/4096/8192 points, 2026-08-25)

| ctx | Run 1 (canon on) | canon off | Δ | old base |
|---|---|---|---|---|
| 2048 | 378.94 | 400.19 | +5.6% | 381.3 |
| 4096 | 346.12 | 354.65 | +2.5% | 353.8 |
| 8192 | 379.73 | 378.92 | ~0% | 377.1 |

Hypothesis supported: the canonical argsort comparator (not token-count gated
like its siblings) costs ~2.5% at 4k and ~5% at 2k; disabling it recovers the
4096 point to old-base level. Fixed in `6fa977c` (gated on `n_tokens >= 32`;
re-verify on the next short-sweep).

---

## Indexer split — `589e93e` (upstream-metal-wins, 2026-08-25)

Opt-in via `DS4_TP_PREFILL_SPLIT_INDEXER=1` (requires
`DS4_TP_PREFILL_SPLIT_NONZERO=1`; both ranks or neither). Splits the indexer
score + top-k by query row at `pos0 > 0` — the term R1 measured at ~61% of the
layer-chunk at 131k. No extra gate traffic. Full results in the R5 section of
the `upstream-metal-wins` entry above: **cold 131k 259.90 → 281.20 → 322.64
t/s** (off → A0 → A0+indexer), sweep +19.5% @131k, correctness
bit-identical (R5a), score stage 317.7 → 138.0 ms (R5c).

### TP — `tp-multi-slot-batching` (old base, superseded)

| ctx | prefill t/s | decode t/s |
|-----|-------------|------------|
| 2048 | 381.3 | 40.87 |
| 4096 | 353.8 | 36.76 |
| 8192 | 377.1 | 36.07 |
| 16384 | 346.3 | 35.04 |
| 32768 | 310.4 | 32.98 |
| 65536 | 252.6 | 29.80 |
| 131072 | 183.4 | 25.33 |

**Do not** use the old 131072 = 183.4 figure as the A0 baseline; the A0
baseline is the branch-under-test with the flag off (table above).

### PP-RDMA — `pp-rdma-new` (rebuilt off tp-multi-slot-batching)

| ctx | prefill t/s | decode t/s |
|-----|-------------|------------|
| 2048 | 303.04 | 28.97 |
| 4096 | 295.60 | 27.49 |
| 8192 | 323.72 | 27.11 |
| 16384 | 419.59 | 26.39 |
| 32768 | 462.14 | 24.83 |
| 65536 | 426.59 | 22.84 |
| 131072 | 334.53 | 20.21 |

## TP vs PP takeaways

- **TP wins decode at every context**: 41% faster at 2k (40.9 vs 29.0), narrowing
  to **25% at 128k** (25.3 vs 20.2). TP splits compute per-layer; PP serializes
  full ~131 KB activations over RDMA on the decode critical path.
- **PP degrades less with context**: decode drops 38% (TP 40.9→25.3) vs 30%
  (PP 29.0→20.2) — PP's per-node KV is halved.
- **PP prefill is much faster at scale**: 462 vs 310 t/s at 32k; 335 vs 183 t/s
  at 128k. PP pipelines prefill and each node only touches its layer block's KV.

## TP regression on pp-rdma-new (verbs-layer extraction check)

2048: prefill 382.9 t/s, steady decode 40.86 t/s — matches baseline. The verbs
RDMA transport extraction did not change TP behavior.

## Power observation (investigation target)

During **TP prefill**, node power drops as context grows:
- ~120 W prefill at < 50k context
- ~100 W prefill at ~180k+ context

Caveat before reproducing: these two readings predate the sweep tables above and
are not from them — the documented sweep stops at 131072, so the ~180k point is
not reachable by re-running it. The measurement command, per-node vs pair basis,
and host were not recorded. Re-measure with
`sudo powermetrics --samplers gpu_power` on a named host and record the context
of each reading before treating the delta as a quantitative result.

Interpretation: lower power = GPU under-utilization / more waiting as context
grows. See investigation notes in `docs/TP-PREFILL-LONG-CTX-INVESTIGATION.md`
(if present) for root-cause analysis and improvement options.

## Key TP internals (from code read)

- Transport: RDMA over AppleThunderboltRDMA. Decode uses per-token gate schedule
  (86 gates/token fixed order); prefill uses **bulk "big gate" row swaps** —
  one full-batch hidden-state exchange per layer over the latency QP, chunked
  into 16 KiB messages with a 1 MiB receive window (`tp_rdma_big_gate_exchange`,
  `ds4_tp.c`).
- Attention **head** splitting (`tp_split_attn = g->tp_world==2`) is the **decode**
  path only. **Prefill** splits attention **rows** (`tp_row_split_attn` in
  `metal_graph_encode_layer_attention_batch`), and only for `pos0 == 0` chunks —
  `const bool zero_prefix = pos0 == 0` gates every split shape. With the default
  4096-token chunking, a 131072-token prefill row-splits chunk 0 and runs the
  whole attention path replicated on both ranks for the other 31 chunks.
  (A0 on `upstream-metal-wins` extends row-splitting to `pos0 > 0`; do not read
  the decode fact as a prefill fact; it changes which fix is worth doing.)
- Prefill chunking is layer-major; chunk cap from `ds4_prefill_cap_for_prompt`
  (env `DS4_METAL_PREFILL_CHUNK`, default 4096 for prompt > 4096).
- Compressed KV cache grows with context: `comp_cap = ctx_size/4 + 2`
  (per-layer ratio). Raw SWA cache is bounded by sliding window (`DS4_N_SWA`).
- **Indexer top-k scoring over the compressed cache is replicated on both TP
  ranks** — see the TP row-split comment at the top of
  `metal_graph_encode_layer_attention_batch()` in `ds4.c`: score/top-k selection
  "stays replicated while the attention consumption splits by rows", and both
  ranks update the compressor/indexer from full rows. This replicated work grows
  linearly with context and does not benefit from TP. (Anchor on the function
  name, not a line number — line citations here rot within a few commits.)
- CUDA TP has cache-duplication/peer-read infra (`cuda_tp_attn_cache_dup`,
  `cuda_tp_attn_peer_read`, `layer_attn_comp_cache_tp[]`, `copy_xdev`) that could
  be a template for splitting the Metal TP compressed cache / indexer.

## Useful env knobs seen in code

- `DS4_METAL_FAST_SYNC=1` — fast release fence; required for published numbers.
- `DS4_METAL_PREFILL_CHUNK` — prefill chunk size (tokens).
- `DS4_METAL_GRAPH_RAW_CAP` — raw KV cache cap override (cap 8192).
- `DS4_TP_PREFILL_SPLIT_MIN` — min tokens to row-split replicated shared expert
  in TP prefill (default 32).
- `DS4_TP_PREFILL_SPLIT_NONZERO=1` — A0: row-split attention for `pos0 > 0`
  prefill chunks (opt-in; must be set on BOTH ranks).
- `DS4_TP_PREFILL_SPLIT_INDEXER=1` — also row-split the indexer score + top-k at
  `pos0 > 0` (requires the above; BOTH ranks). R5: sweep +19.5% @131k.
- `DS4_TP_PREFILL_SPLIT_STATIC_MIXED=1` — also row-split the ratio-128
  (static-mixed) layers at `pos0 > 0` (requires `..._NONZERO`; BOTH ranks).
  R7, untested on the rig.
- `DS4_TP_FORCE_DENSE_ATTN_OUT=1` — dense attention output projection on both
  arms of a correctness comparison (costs throughput; unset for speed runs).
- `DS4_TP_GATE_PROFILE=1` — per-gate-kind (ROW/VERIFY/BIG) counts, avg GPU-wait
  µs, avg exchange µs; bandwidth = 67,108,864 / avg_exchange_us.
- `DS4_METAL_GPU_STAGE_TIMESTAMPS=1` (+`_LAYER=<il>`, `_DETAIL`) — per-stage
  layer timing; pick a ratio-4 layer (even `il` ≥ 2) to include the indexer.
- `DS4_TP_SUBGATE_PIPELINE` — opt-in sub-chunk gate pipelining for TP prefill
  row swaps; **default off (measured net-negative on M5 Max pair)**. Must be set
  on BOTH ranks.
- `DS4_TP_SPEC_F_ATTN_OUT_SPLIT` — TP verifier attention output split path.
- `DS4_METAL_ABLATE_INDEXER_SCORE` / `_TOPK` — ablate indexer stages (diagnostic,
  changes output).
