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
- **The GID index is not stable across link flaps.** A reset/flap cycle can
  leave a stale hole at index 1 and commit the new IPv4-mapped entry at index 2
  (the table is not compacted). `check-roce-v2-gid.sh` and the
  index-1-hardcoded probes (`uc_pingpong`, `jaccl`) then fail even though the
  link is fine; ds4 itself is immune (it scans the GID table, `ds4_tp.c:780-790`).
  Check `ibv_devinfo -d rdma_enX -v` for the real index before chasing the
  network. Full failure-mode table: the rig runbook in
  `docs/TP-A0-ROWSPLIT-TEST-PLAN.md`.

### GPU wired limit (re-run after every reboot)

Also runtime-only, and unlike the RDMA setup nothing in the launch path warns
loudly if it is missing. On **both** nodes:

```
sudo sysctl iogpu.wired_limit_mb=120000
```

- 120 GB of the 128 GB box, leaving ~8 GB for the OS. The in-tree suggestion is
  the same value (`ds4.c:59891`).
- **Why it matters:** with the sysctl at 0, `ds4.c:59886` takes
  `ds4_gpu_model_residency_skip(1)` and TP **skips the residency set entirely**
  — the 76.7 GiB shard is then paged in lazily instead of pinned. That produces
  a flat, stall-shaped, neither-bandwidth-nor-compute-bound decode profile,
  i.e. it looks exactly like the thing several runs here have been trying to
  explain.
- **Verify:** `sysctl iogpu.wired_limit_mb` on both hosts *and* grep the run log
  for `wired_limit_mb is 0`. The engine prints that warning once at startup; it
  is easy to lose in a `nohup` log, so check for it explicitly rather than
  assuming a clean run.
- Any arm whose log contains that warning should be discarded and re-run.

## Prerequisites for a comparable run

1. **Same commit on both hosts**: `git rev-parse HEAD` must match. `git pull`,
   then `make -j ds4-bench ds4-server ds4` on both. Do not reuse stale binaries.
2. **Env symmetry**: flags that change per-layer gate counts
   (`DS4_TP_FORCE_DENSE_ATTN_OUT`, `DS4_TP_SUBGATE_PIPELINE`, …) **must be set
   on both ranks or neither** — asymmetric settings deadlock the gate exchange,
   they do not degrade. The three prefill splits are now **default on**
   (`f45b535`), so a flag-off baseline means explicitly setting
   `DS4_TP_PREFILL_SPLIT_{NONZERO,INDEXER,STATIC_MIXED}=0` on both ranks.
3. **`DS4_METAL_FAST_SYNC=1` on both ranks** for all published numbers. It is
   default-off (`ds4_metal.m:10263`) and nothing in the tree sets it. Beyond the
   fast release fence, `ds4_gpu_tp_split_safe()` returns 0 without it, which
   makes the decode command-buffer split a no-op under TP — so any split-schedule
   experiment run without it returns a null result for the wrong reason.
   **Confirm it is in the `ds4-server` launch path too**, not just the bench.
4. **`sudo sysctl iogpu.wired_limit_mb=120000` on both hosts** — see the section
   above. Runtime-only, lost on reboot, and silently degrades to a lazily-paged
   shard rather than failing.
5. RDMA device names are host-local (lanfear=`rdma_en6`, mat=`rdma_en7` per the
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

### U6 — the roof is ~760 GB/s, not 400 — allocation path costs nothing — 2026-08-27

`tests/bench_membw` on mat (M2 Ultra, 800 GB/s spec), one rank, no model — a
self-contained grid-stride `uint4` streaming kernel (no dependent chains, 2M
threads, a compiler-cannot-fold store sink) over seven allocation arms of
identical size, `BENCH_MEMBW_FILE` pointed at the real GGUF so the `mmap-file`
arm is literally the engine's own load path. Best-of-20 GPU time; 16 GiB and
32 GiB per arm.

| arm | allocation | 16 GiB | 32 GiB |
|---|---|---|---|
| metal-shared | `newBufferWithLength:` Shared | 760.9 | 756.5 |
| mmap-anon | NoCopy MAP_ANON | 760.0 | 757.1 |
| mmap-file | NoCopy MAP_SHARED (**ds4 today**) | 759.9 | 756.7 |
| mmap-untracked | as above + untracked | 760.2 | 755.7 |
| mmap-parallel | as above, 16-thread fault | 760.3 | 756.6 |
| metal-private | `newBufferWithLength:` Private | 761.7 | 757.4 |
| two-queue | 2× Private, 2 queues, overlapped | 754.7 (agg) | 752.1 (agg) |

**Verdict: the platform reads at ~760 GB/s — 94–95% of the 800 GB/s spec —
and every arm is within 1%.** The M2 Ultra saturates *both* dies with a plain
streaming kernel on ds4's *own* `mmap` path. Allocation path (Shared / Private
/ mmap / untracked / parallel-fault) costs nothing; concurrency (`two-queue`)
costs nothing.

**This reverses U1.** U1's ~408–410 GB/s plateau is not the platform and not
placement — it is the **MoE matvec kernel**, which achieves only ~54% of the
~760 GB/s the part can stream. The matvec is bound by its own access pattern
(MXFP4 17-byte blocks, gather/scatter, dequant interleave), not by the chip.

**Correction to U1 and T8.** U1's "platform/placement — escalate, do not tune
kernels" is wrong; the headroom is in the kernel's access pattern and the
loader is exonerated. T8's "matvec is bandwidth-bound at ~400 — near the M2
Ultra ceiling" is doubly wrong: the ceiling is 800 GB/s, and the matvec is
not even bandwidth-bound — it sits at ~54% of the achievable streaming rate.

**Consequence:** matvec tuning re-opens. The MoE stage (5.4 ms at 131k in the
stage profile) is the largest single decode stage and the one whose ~2× is
closest to reachable — if its access pattern can be made to stream.

**Validated on lanfear (second M2 Ultra), 2026-08-27** — same command, both
16 and 32 GiB per arm, `BENCH_MEMBW_FILE` at the real GGUF:

| arm | lanfear 16G | lanfear 32G |
|---|---|---|
| metal-shared | 791.5 | 778.3 |
| mmap-anon | 764.6 | 760.1 |
| mmap-file (ds4) | 760.5 | 758.2 |
| mmap-untracked | 760.9 | 753.6 |
| mmap-parallel | 758.6 | 757.6 |
| metal-private | 791.2 | 784.7 |
| two-queue (agg) | 789.3 | 775.2 |

The core verdict holds on both hosts: every arm is ~750–790 GB/s (94–99% of
the 800 spec), far above U1's ~410 plateau. **One host difference:** on
lanfear the Metal-allocated arms (shared/private/two-queue) run ~2–4% faster
(~778–791 GB/s) than the mmap arms (~753–764), whereas on mat all seven arms
were within 1%. The mmap arms match mat's numbers almost exactly; lanfear's
Metal allocations are the faster ones. It is a small (~2–4%), repeatable
host-to-host effect — Metal's allocator placing buffers slightly better on
that host — and it does **not** change the verdict (roof ~760+, not 400).

### U11 — U10 (TIGHT, default-on) does not regress prefill — neutral — 2026-08-27

Prefill A/B on the rig (mat worker / lanfear coord, TP over RDMA, build
`b99dfa3`), default (TIGHT on) vs `DS4_METAL_INDEXER_LLT_TIGHT=0`, 32k/65k/131k
with `promessi_sposi.txt`, gen 128. CSV `prefill_tps` / `gen_first_ms`
(first-token latency — the number T3 caught nsg4 on).

| ctx | arm | prefill t/s | first-token ms | steady t/s |
|---|---|---|---|---|
| 32k | TIGHT on | 501.92 | 272.9* | 34.59 |
| 32k | TIGHT=0 | 517.19 | 31.1 | 33.92 |
| 65k | TIGHT on | 460.94 | 32.8 | 32.19 |
| 65k | TIGHT=0 | 461.72 | 32.5 | 31.68 |
| 131k | TIGHT on | 393.03 | 36.5 | 29.15 |
| 131k | TIGHT=0 | 393.37 | 35.5 | 28.42 |

*\*272.9 ms at 32k TIGHT-on is a cold-start/warm-up artifact (first frontier of
the first arm includes model load + shader compile); the same arm's 65k/131k
first-token is 32.8/36.5 ms — normal. Not a regression.*

**Verdict: prefill neutral.** 393.03 vs 393.37 t/s @131k (−0.1%), 460.94 vs
461.72 @65k (−0.2%); first-token 36.5 vs 35.5 ms @131k (within noise). **No
prefill-catastrophic spike** — nothing like T3's nsg4's 189 ms (U10 changes
only the allocation, not the grid). Decode steady-state slightly *better* with
TIGHT on (29.15 vs 28.42 t/s @131k). **U10 stays default-on.**

### U10 — TIGHT alias confirmed on the rig: +17.4%, bit-identical — 2026-08-27

`DS4_METAL_INDEXER_LLT_TIGHT=1` (opt-in, NSG=8) on mat (M2 Ultra),
n_comp=32768, `DS4_METAL_GPU_BUSY_PROFILE=1 tests/bench_indexer_score 32768
300`, 3 runs per arm. The implemented U10 is the cheap intermediate — alias
`sq`/`sw`/`sqk` over the under-used `sk` buffer for residency 1 → 2 at
unchanged NK — not the full F16-cache drop (that is U10c).

| arm | GFLOP/s (3 runs) | mean | vs default |
|---|---|---|---|
| default NSG=8 | 1646.8 / 1715.2 / 1789.6 | 1717.2 | — |
| **U10 TIGHT** | **1924.3 / 2089.0 / 2033.6** | **2015.6** | **+17.4%** |
| NSG=4 | 1877.2 / 1677.7 / 1720.7 | 1758.5 | +2.4% |

**U10 TIGHT is the clear winner on the rig (+17.4% vs default), and it also
beats NSG=4 (+14.6%).** It is **bit-identical** to default (32750/32768 exact,
same worst-rel row 22878), so no correctness gate is needed. This confirms the
dev-box +19.3% and is the cheap half of the residency hypothesis confirmed at
Ultra scale. End-to-end against the token this is a ~2–4% item (score kernel =
3.84 ms of 36.36 ms = 10.9%; see plan U10b).

**Consequence:** U10 confirmed → **U10c** (F16 index cache, 2 → 7 resident) is
gated on, per the sequencing table. Note per U10b the honest end-to-end ceiling
is ~2.5–5.4%, worth doing but not before U9's larger algorithmic prize.

### U10a — NSG sweep on the rig: residency beats NK — U10 is ON — 2026-08-27

`DS4_METAL_INDEXER_LLT_NSG` ∈ {8, 4, 2} with `DS4_METAL_GPU_BUSY_PROFILE=1
tests/bench_indexer_score 32768 300` on mat (M2 Ultra), repeated for
stability. Zero code — the knob was already built and bit-identical across
arms. This is the direct test of U10's premise: is the Ultra short of latency
hiding, so that residency is worth more than the NK it trades for?

| NSG | GFLOP/s (runs) | vs default |
|---|---|---|
| **8** (default) | 1525.2 / 1607.4 / 1683.0 | — |
| **4** | **1720.7 / 1813.8 / 1704.4** | **+12.8% / +12.8% / +1.3%** |
| 2 | 1342.2 | −12% |

**NSG=4 beats NSG=8 in every run (~+5–13%); NSG=2 is worse.** This is the
plan's *first* outcome — the M1 Max ranking (NSG=4 −1.4%) does **not**
transfer to the Ultra, so residency is worth more than NK there. The Ultra's
72% scaling efficiency / 7.3% of ALU peak is consistent with a latency-hiding
shortfall. **Consequence: U10 is ON.** Dropping the `sk` staging buffer (F16
index cache, `ds4.c:17373`) buys 7× residency at *unchanged* NK — strictly
better than what NSG=4 trades for (NSG=4 confounds halved NK with doubled
residency; U10 gets the residency without the NK cost).

### U7 — indexer LLT scoring: 1565 GFLOP/s on the rig, 72% core scaling, occupancy is the lever — 2026-08-27

`DS4_METAL_GPU_BUSY_PROFILE=1 tests/bench_indexer_score 32768 300` on mat
(M2 Ultra), freshly built binary. The decode path is
`kernel_dsv4_indexer_scores_llt <NBPTG=8,T_NSG=8>` (64 keys/threadgroup), not
the per-row fallback.

| metric | value |
|---|---|
| GFLOP/s | **1565.2** |
| K-cache GB/s | 48.9 (16.78 MB/dispatch) |
| vs M1 Max LLT (1075.9) | **1.45×** — cores×clock predicts 2.02× → **72% efficient** |
| ALU peak | 7.3% (Ultra) vs 10.1% (M1 Max) |
| scoring ×21 layers | 7.20 ms/token (harness, over-predicts ~1.9× vs M2's 3.84 ms) |

**Scaling is real but 72% efficient, and the bigger part is the less
efficient one.** U2's 1015 was the *direct* fallback path (its own note
records the (32,4) shape), not LLT — the non-scaling worry was an artefact.
The deficit is memory latency: threadgroup memory allows exactly one
threadgroup resident per core, so there is almost nothing to hide latency
with, and the Ultra has more latency to hide. Consistent with the NSG sweep
(more residency is worse — bigger NK wins), T2's turnover at ~112
threadgroups, and U9/U10's constraint recall.

**Calibration:** at 1565 the scoring half is 7.20 ms across 21 layers, but
M2's in-situ ablation attributed 3.84 ms — this harness over-predicts by
**1.88×**. Standalone indexer numbers should be divided by ~1.9 before
projecting engine impact.

**Direction (U10):** `sk[64*128]half` is 16,384 of the 20,512 B budget and
exists only because the index cache is F32. Storing it F16 lets the kernel
`simdgroup_load` keys straight from device memory → threadgroup memory drops
to 4,128 B → **residency 1 → 7 threadgroups/core** at unchanged NK. T9/U3
returns for a fourth reason with a real mechanism: 7× residency, not
bandwidth. Cheapest-first experiment: halve `sk` to 32 keys (no format
change) to isolate whether residency helps at all, then do the F16 cache only
if it does. See test plan U9/U10.

### U8 — the 17-byte MXFP4 block granularity is ~6%, not the 46% gap — 2026-08-27

`bench_membw` block-stride arms on mat (M2 Ultra), 16 GiB buffer, 20 iters —
`stream_blocks` consumes a 16 B payload from each block, stride 16 (aligned)
vs stride 17 (MXFP4 layout, every load straddling a 16-byte boundary),
isolating layout cost from dequant arithmetic and from the expert gather:

| arm | GB/s |
|---|---|
| blk-16 aligned | 758.0 |
| blk-17 mxfp4 | 713.2 |

**Gap: 5.9%** (713.2/758.0), versus 4.7% on the M1 Max pre-screen. The
17-byte MXFP4 block granularity is worth ~6%, **not** the ~46% that separates
the matvec's ~410 from the ~760 GB/s roof. So the block layout is not what
puts `routed_moe_folded` at 54% of achievable. **Remaining suspects:** the
expert gather/scatter and the dequant/accumulate path — the gather is 6
experts × 3 tensors = 18 large contiguous regions per token, which should
stream, so dequant and accumulation are now the leading candidates. See
test plan U8 (diagnosis deliverable, not a patch; T8's specialisations stay
dead).

### U2 — indexer-score roofline: latency-bound, U3 will not pay — 2026-08-26

`tests/bench_indexer_score` on lanfear (M2 Ultra), model-free, one rank, no
TP. `n_comp` swept with `DS4_METAL_GPU_BUSY_PROFILE=1 tests/bench_indexer_score
$n 300`; 300 dispatches per arm.

| n_comp | GPU busy (ms/dispatch) | GFLOP/s | K-cache (GB/s) |
|---|---|---|---|
| 4096 | 0.066 | 287 | 9.0 |
| 8192 | 0.097 | 495 | 15.5 |
| 16384 | 0.184 | 731 | 22.9 |
| 32768 | 0.352 | 1015 | 31.7 |
| 65536 | 0.693 | 1271 | 39.7 |

Kernel config: the single-token score kernel
(`ds4_gpu_indexer_score_one_tensor`, direct path at n_head=32/head_dim=128)
dispatches n_rows threadgroups of (32,4) = 128 threads — one threadgroup per
compressed row, per-thread row count 1.

**Verdict: latency-bound.** GPU-busy scales ~linearly in `n_comp` (each
doubling ~1.9–2.0× past 16k; the 4096→8192 step is sublinear because small
arms are overhead-dominated). Achieved bandwidth stays far below any roof:
39.7 GB/s at 65536 is ~10% of the 400 GB/s platform (5% of the 800 GB/s
spec), and 1271 GFLOP/s is ~12% of the ~10.4 TFLOP/s FP32 peak. That is the
signature of a latency-bound kernel — one threadgroup per row, dependent
loads, poor K-cache reuse.

**Consequence for U3: it will not pay.** Per the plan's U2 decision fork, a
kernel that is latency-bound per byte buys little from halving its bytes:
halving bytes of a kernel at ~10% of bandwidth is a near-zero prize. Skip
U3 (indexer cache F32→F16); go instead to the restructure (batching layers
per dispatch / K-cache reuse).

**Correctness, resolving the plan's open item.** The harness's own gate
flags worst relative error 7.6e-3 at `n_comp` ≥ 32768 (row 17391), over its
1e-3 threshold ("kernel looks wrong"). Per the harness header the CPU
reference sums lanes sequentially while the Metal `simd_sum` reduces as a
tree, so exact agreement was never expected. The worst row is the same
(17391) at both 32768 and 65536 — one near-cancellation row where FP32
tree-vs-sequential ordering diverges. Verdict: expected FP32
reduction-association tolerance, benign for ranking; not a real defect, but
watch that row if scores are used for exact tie-breaking.

### U1 — streaming-read ceiling: ~400 GB/s is the platform, not our kernels — 2026-08-26

`tests/bench_moe_mxfp4_decode` on lanfear (M2 Ultra, 800 GB/s spec), one
rank, no TP, no model — T8's conditions with the working set pushed 8× via
`n_total_expert` (harness caps `n_sel=6`). Every arm streams 80.22 MB/iter;
3 runs each.

| n_experts | model map | GB/s (3 runs) | mean |
|---|---|---|---|
| 256 | 3.19 GiB | 374.5 / 387.2 / 383.5 | ~382 |
| 512 | 6.38 GiB | 409.1 / 407.7 / 410.4 | ~409 |
| 1024 | 12.75 GiB | 408.6 / 406.6 / 410.2 | ~408 |
| 2048 | 25.50 GiB | 410.8 / 406.8 / 412.0 | ~410 |

**Verdict: nothing exceeds ~410 GB/s even with the working set pushed 8×.**
The ~408–410 plateau was initially read as exactly one M2 Max die's worth of
the 800 GB/s Ultra spec (400/die), i.e. a platform/placement characteristic
(single-die UltraFusion locality), not a kernel limit. **This verdict was
superseded by U6 (2026-08-27): the part streams at ~760 GB/s on ds4's own
`mmap` path, so U1's plateau is the MoE matvec kernel's access pattern, not
the platform. The kernel headroom (~2×) re-opens; do not escalate as a
platform bug.**

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

### T8 — MoE decode specialization pricing (standalone harness) — 2026-08-26

`tests/bench_moe_mxfp4_decode 256 256 6` on lanfear (M2 Ultra 60-core, same
GPU as mat), no model, no TP. 256 routed experts, 6 distinct streamed per
iter, 80.22 MB/iter, 43-layer projection. Cumulative-disable ladder prices
each of the five MXFP4 routed-MoE decode specializations
(`ds4_metal.m:39358`+): static_trip → sum6_full_rows → fixed_route_sum6 →
fixed_route_pair → tg_multiple.

| arm | disabled | wall ms/iter | GB/s | 43L ms/token |
|---|---|---|---|---|
| t8_full | none | 0.2076 / 0.2143 / 0.2057 | ~383 | 9.08 |
| t8_no_trip | static_trip | 0.2033 | 394.5 | 8.74 |
| t8_no_fullrows | +sum6_full_rows | 0.2086 | 384.6 | 8.97 |
| t8_no_sum6 | +fixed_route_sum6 | 0.1932 | 415.2 | 8.31 |
| t8_no_pair | +fixed_route_pair | 0.1970 | 407.1 | 8.47 |
| t8_no_tgmult | +tg_multiple (all generic) | 0.1988 / 0.2046 / 0.1929 | ~403 | 8.54 |

**Conclusion: the specialization ladder is worth nothing to −5%; the generic
path is consistently slightly faster** (fully-disabled mean 0.1988 ms/iter vs
full-ladder 0.2092, ~403 vs ~383 GB/s, non-overlapping across 3 runs each).
The routed MoE matvec is **bandwidth-bound at ~400 GB/s** — near the M2 Ultra
ceiling — so kernel specializations cannot buy what is not there. **T8 pricing
argues against a MoE specialization port.**

### Stage profile — the ~13 ms floor is real compute, not stall — 2026-08-26

`DS4_METAL_GPU_STAGE_TIMESTAMPS=1` at 32k and 131k, single ctx each, gen
128, both ranks (per-token decode stage report added to the main TP decode
path — the report was previously only wired on the speculative `_top`
variant; `ds4.c` `metal_graph_eval_token_raw_swa`). Steady-state decode
stage gpu_ms (coordinator rank):

| stage | 32k | 131k | Δ |
|---|---|---|---|
| compressor_indexer | 5.45 | **10.51** | **+5.06** |
| q_path | 5.48 | 5.47 | 0 |
| routed_moe_folded | 4.67 | 5.40 | +0.73 |
| attn_inv_rope | 3.42 | 4.27 | +0.85 |
| attn_output | 3.73 | 3.80 | 0 |
| ffn_hc_post | 2.19 | 1.48 | −0.71 |
| attn_hc_pre | 1.16 | 1.14 | 0 |
| ffn_hc_pre | 1.13 | 1.11 | 0 |
| router | 1.11 | 1.09 | 0 |
| shared_gate_up | 0.99 | 0.98 | 0 |
| shared_down | 0.66 | 0.66 | 0 |
| attn_hc_post | 0.46 | 0.46 | 0 |
| **total gpu_busy** | **30.43** | **36.36** | +5.93 |
| span | 30.73 | 36.67 | |
| **gap (stall)** | **0.307** | **0.308** | |

**The decode GPU is ~99% busy — the stall gap is ~0.31 ms at both contexts.**
The stage sum (30.427/36.360) equals `total gpu_busy` exactly, so every
microsecond is attributed to a stage; there is no hidden idle time.

**The ~13 ms M2 "residual" is real compute, not stall.** M2's ablation set
did not cover several stages that the stage profile now prices directly:
q_path's un-ablated remainder, compressor_indexer (the compressed-KV index
lookup — distinct from the indexer score/topk M2 ablated), attn_inv_rope,
router, shared_gate_up/down, ffn_hc_pre/post, attn_hc_pre/post. Summing the
stage profile at 131k accounts for the 12.45 ms residual M2 could not
attribute. The GPU is busy the whole token; there is no dispatch/idle floor
left to find.

**The long-context term is `compressor_indexer`, essentially in full.**
Measured growth 32k→131k is +5.93 ms; compressor_indexer alone is +5.06 ms,
with a small attention tail (inv_rope +0.85, routed_moe_folded +0.73,
ffn_hc_post −0.71). This is a finer attribution than M2's indexer
score/topk — the dominant indexer cost is the compressed-KV lookup, not the
scoring or top-k selection.

**Consequence for the plan: R12b (dispatch ballast) and the encoder-boundary
instrument are now moot — they probe stall, and there is no stall.** The
floor is compute. The two levers that remain are (a) making the big stages
cheaper — compressor_indexer (10.5 ms @131k), q_path (5.5), routed_moe
(5.4), attn_inv_rope (4.3) — and (b) T2's ~1% on attncore+attnout.

### M2 — attribute the 11.1 ms — ablation battery at 32k/65k/131k — 2026-08-26

Re-ran the `DS4_TP_ABLATE` chain battery at 32k and 131k plus the never-run
indexer ablations (`DS4_METAL_ABLATE_INDEXER_SCORE`/`_TOPK`), sweep
`32768→131072 --step-mul 2`, gen 128, in-session control `m2_ctrl` (tracks
M0 within 0.4%). Deltas are decode steady t/s gain from *removing* each chain
(semantically-wrong output, t/s delta = in-situ cost; both ranks agree).

| chain | 32k | 65k | 131k |
|---|---|---|---|
| control t/s | 33.92 | 31.69 | 28.29 |
| hcpre | +2.6% | +2.8% | +2.2% |
| qb | +5.5% | +5.3% | +5.3% |
| attnout | +10.5% | +9.7% | +9.0% |
| moe | +25% | +25% | +22% |
| attncore | +4.5% | +5.5% | +5.7% |
| indexer score | +5.9% | +8.1% | +12.2% |
| indexer topk | +7.9% | +10.9% | +17.7% |

**The routed MoE is the dominant decode stage (~22–25%)** — larger than T8's
isolated ~12% because production carries the 128/128 shard exchange plus the
real stage shape. Consistent with `DS4_TP_ABLATE=moe` removing ~1/4 of the
token.

**The indexer is the long-context story.** Both indexer ablations grow with
context: score +5.9% → +12.2%, topk +7.9% → +17.7% as ctx goes 32k→131k. The
indexer path is inactive at ctx 512 (per M2's framing) and becomes a
**17.7% / ~6 ms of the 35.5 ms token** cost at 131k — this is the largest
single attributed slice of the 11.1 ms long-context growth. (Caveat: ablating
the indexer also removes its contribution to the routed-MoE top-6 selection,
so some of that delta may be double-counted with `moe`; the two are not
disjoint.)

**attnout ~9–10%** is the largest single attention stage; qb/attncore ~5% each;
hcpre ~2%. No chain is free — every one shows a positive in-situ cost.

### T1 — row-gate fastpath A/B — 2026-08-26

`DS4_TP_GATE_FASTPATH=1` (recv re-arm hoisted before the gate wait,
signal-every-16th-send), built default-off, A/B against the M0 baseline and
m0_gate profiles.

| ctx | m0 baseline | t1 fastpath | Δ |
|---|---|---|---|
| 2048 | 41.09 | 41.32 | +0.6% |
| 4096 | 36.58 | 36.55 | −0.1% |
| 8192 | 36.04 | 36.06 | +0.1% |
| 16384 | 35.39 | 35.52 | +0.4% |
| 32768 | 33.71 | 33.72 | +0.0% |
| 65536 | 31.56 | 31.63 | +0.2% |
| 131072 | 28.34 | 28.26 | −0.3% |

**T1 is a wash** — all deltas within ±0.6%, no consistent direction. The gate
profile confirms why: fastpath does not change the row-gate exchange at all
(23.0 vs 23.6 µs @2k; 24.2 vs 24.8 @131k vs m0_gate). The recv re-arm hoist
cannot help a wait that is dominated by local GPU completion, not wire
exchange. Consistent with M0's finding that exchange is 6–7% of gate time.

**Correctness: top-1 preserved (7/7 steps) but NOT bit-identical** — 2/7 steps
differ, logits shift up to 2.3 (mean 0.27–0.35). The reorder perturbs float
accumulation order enough to matter for sampling, though greedy argmax is
unchanged. Since the perf is a wash and the logits are perturbed, **the
fastpath should stay default-off; do not enable.**

### T2 follow-up — decode split-K peaked at 24 — 2026-08-26

The env battery's T2 curve was monotonic 12→24 with 24 the top of the
tested range (the code allows 2..31). This follow-up sweeps 28 and 31 in
the same session shape (32k/65k/131k decode rows, gen 128,
`DS4_NGRAM_SPEC` off). Decode steady t/s:

| splits | 32k | 65k | 131k |
|---|---|---|---|
| 12 (control) | 33.79 | 31.71 | 28.43 |
| **24 (peak)** | 34.24 | 31.99 | **28.68** |
| 28 | 34.19 | 31.99 | 28.48 |
| 31 | 34.24 | 31.94 | 28.45 |

**T2 peaked at 24.** 28 and 31 both regress at 131k (28.48 and 28.45 vs
24's 28.68); the trend is no longer monotonic — it turns over at 28. At
65k, 28 ties 24 and 31 is within 0.2%; the 24/28/31 arms are flat within
0.15% at 32k (34.19–34.24).

**Decision per the plan: set the default to the measured peak (24) and
close T2 — do not move to 28/31.** The comment at `ds4_metal.m:29988`
reasons from exact fill: under TP each rank holds 32 heads, so 4 × 24 = 96
threadgroups on 60 cores. Gains out to 24 mean oversubscription for
latency hiding, not occupancy, and the turnover at 28 (4 × 28 = 112)
caps it — 24 is the sweet spot. The code default and the comment move
with this decision.

Interaction caveat to keep with any future n-gram work: with
`DS4_NGRAM_SPEC` on, verify steps set `decode_splits = 1`
(`ds4_metal.m:30001`) and T2 is inert on those steps, so T2's value shrinks
by the n-gram acceptance rate if n-gram ever defaults on. Measured with
n-gram off.

### Env battery — T2 + T3 + T4 at 32k/65k/131k — 2026-08-26

One rig session, `DS4_NGRAM_SPEC` off, sweep `32768→131072 --step-mul 2`
(covers 32k/65k/131k decode rows), gen 128, control = in-session `t2_12`
(= the decode-split default, also tracking M0). Decode steady t/s:

| arm | 32k | 65k | 131k | vs control @131k |
|---|---|---|---|---|
| **t2_12** (split=12, control) | 33.79 | 31.71 | 28.43 | — |
| t2_8 | 33.49 | 31.40 | 28.08 | −1.2% |
| t2_15 | 34.01 | 31.78 | 28.40 | −0.1% |
| t2_16 | 33.93 | 31.85 | 28.49 | +0.2% |
| t2_20 | 34.22 | 31.93 | 28.59 | +0.6% |
| **t2_24** | 34.24 | 31.99 | 28.68 | **+0.9%** |
| t3_1 (LLT nsg4) | 34.27 | 31.11 | 28.26 | −0.6% |
| t4_1 (nsg=1) | 33.68 | 31.18 | 28.17 | −0.9% |
| **t4_2** (nsg=2) | 33.89 | 31.65 | 28.40 | −0.1% |
| t4_3 (nsg=3) | 33.33 | 31.12 | 27.98 | −1.6% |
| t4_4 (nsg=4) | 32.64 | 30.66 | 27.58 | −3.0% |
| t4_6 (nsg=6) | 32.32 | 30.17 | 27.25 | −4.1% |

Prefill was flat across every arm (393.5–393.7 @131k; 461.6–462.0 @65k;
495–518 @32k) — none of these knobs touch prefill.

**T2 — decode split-K: monotonic gain from 12→24, default is not optimal.**
+0.9% @131k / +1.3% @32k at split=24 vs the default 12. Larger splits win
(24 is the top of the tested range; the code allows 2..31).

**T3 — indexer LLT nsg4 (presence flag): context-dependent, roughly neutral.**
Helps 32k steady (+1.4%), hurts 65k/131k (−1.7%/−0.6%). First-token latency
spikes to 189 ms @32k vs 31 ms control — the nsg4 LLT scorer is slower on the
first decode step but faster in steady state. Mixed; not a clear win.

**T4 — Q8_MV_NSG: default 2 is optimal; higher is monotonically worse.**
n=1 −0.9%, n=3 −1.6%, n=4 −3.0%, n=6 −4.1% @131k. The `parallel_full_ffn`
confound (setting the env var disables it, `ds4.c:22199`) is isolated by t4_2
(nsg=2, env set) vs control: −0.1% — negligible, that path is not on this
model anyway (it is IQ2_XXS/Q2_K only).

### M0 — re-baseline decode on a pinned shard — `upstream-metal-wins` @ `5866100` tree build, 2026-08-26

Step 0 came back positive: `iogpu.wired_limit_mb` was `0` on both hosts for
the entire R10/R11 era, so the 76.7 GiB shard paged lazily. M0 re-runs the
decode baseline with it set to `120000` (prefill splits default-on since
`f45b535`, `DS4_NGRAM_SPEC` off, `DS4_METAL_FAST_SYNC=1` both ranks, standard
prompt, no other TP flags).

**Headline: the decode numbers survive; the gate-exchange numbers do not.**
Decode t/s moved at most ~1% vs the wired=0 era — lazy paging cost almost
nothing at the throughput level. The gate exchange was the lazy-paging
casualty: it halved at 131k.

Sweep (incremental, `--ctx-start 2048 --ctx-max 131072 --step-mul 2`, gen 128):

| ctx | prefill t/s | decode t/s | first ms |
|---|---|---|---|
| 2048 | 448.69 | 41.09 | 26.9 |
| 4096 | 522.62 | 36.58 | 26.2 |
| 8192 | 555.47 | 36.04 | 29.1 |
| 16384 | 532.93 | 35.39 | 29.3 |
| 32768 | 506.71 | 33.71 | 30.2 |
| 65536 | 461.88 | 31.56 | 32.1 |
| 131072 | 393.53 | 28.34 | 35.2 |

| arm | prefill t/s | decode t/s | first ms |
|---|---|---|---|
| cold 131k | 435.67 | 28.40 | 36.0 |
| gate 2048 (`DS4_TP_GATE_PROFILE=1`) | 452.30 | 41.15 | 26.3 |
| gate 131k | 435.87 | 28.24 | 36.1 |

Gate profile (final cumulative snapshot, `DS4_TP_GATE_PROFILE=1`):

| ctx | row gates | row wait µs | row exchange µs | BIG gates | BIG wait µs | BIG exchange µs |
|---|---|---|---|---|---|---|
| 2048 | 10,234 | 247.1 | 23.6 | 86 | 35,197 | 10,350 |
| 131072 | 11,008 | 375.0 | 24.8 | 2,752 | 89,595 | 17,091 |

Vs R10e (wired=0, word-soup prompt, `f3668a1` build):

| quantity | R10e | M0 | Δ |
|---|---|---|---|
| decode @2048 t/s | 40.91 | 41.15 | +0.6% |
| decode @131k t/s | 28.09 | 28.24–28.40 | +0.5–1.1% |
| row gate exchange @2k µs | 34.8 | 23.6 | −32% |
| row gate exchange @131k µs | 49.7 | 24.8 | **−50%** |
| row gate wait @2k µs | 292.6 | 247.1 | −15.5% |
| row gate wait @131k µs | 401.4 | 375.0 | −6.6% |
| BIG exchange @131k µs | 22,531 | 17,091 | −24% |

Consequences:

1. **R10/R11 conclusions survive the re-baseline.** The sub-gate wash, the
   4.4/4.1 GB/s link ceiling, and the replicated-attention negative none
   hinged on the ≤1% decode shift (R11's A/B was in-session anyway).
2. **T1 re-sizes, downward.** Its "86 × 49.7 µs = 4.27 ms/token, 12%" sizing
   used a lazy-paging exchange. At 24.8 µs the exchange sits within ~9–10 µs
   of M3's probe one-way p50 (14.5–15.5 µs for the same 16 KB single WR) —
   no longer 3–4× above the fabric. The recoverable gap is at most
   ~9 µs/gate × 86 = **0.8 ms/token ≈ +2% at 131k**, and that upper bound
   assumes the entire delta to the probe RTT is recoverable, which it is not
   (probe ping-pong and in-engine exchange are not the same measurement).
   T1 stays on the sequence — it is a cheap A/B — but it is no longer "the
   largest sized item".
3. **The decode critical path is the gpu-wait, not the link.** A row gate
   waits 375 µs @131k to exchange 24.8 µs — wire is 6–7% of gate time, the
   rest is waiting on local GPU completion. 86 gates × 375 µs ≈ 32 ms ≈ the
   whole 35.4 ms token. That is the regime M2/R13 are aimed at, and the
   reason R10c's "stall-bound, ~30 W" profile was real all along.

### M3 — RDMA ping-pong latency, the T1 gate — probe battery, 2026-08-26

**Question (plan step 1):** is T1 (reclaim the per-gate software cost) viable?
Threshold: half-RTT ≲20 µs → viable (~30 µs/gate ≈ 2.5 ms/token at 131k, ~+8%);
~45 µs → T1 closes as a documented hard floor.

**Probe.** `uc_lat2`, built on `uc_bench`'s proven skeleton (OOB metadata over
TCP with the socket kept open and a `barrier()` before the first ping — closing
it early is a documented UC first-packet race; targeted CQE polling; negotiated
QP depth; GID-table scan exactly as `ds4_tp.c` does). Payload is a per-byte
pattern, re-rotated each iteration, and **verified byte-by-byte on every reply
in both directions** (server echo), because `uc_pingpong`'s 16 KB check is a
documented false positive (it only checked the first 64 B). n = 2000/arm.
`uc_pingpong` itself was unusable here: it hardcodes GID index 1, and after
the TB-net reset mat's IPv4-mapped GID sits at index 2 (index 1 is a stale
hole) — the scan makes the probe immune, as `ds4` is.

| arm | RTT min/p50/mean/p99/max (µs) | one-way p50 (µs) | bad bytes |
|---|---|---|---|
| 4 KB mat→lanfear | 11.0 / 16.0 / 16.4 / 25.0 / 789 | 8.0 | 0 |
| 4 KB lanfear→mat | 14.0 / 16.0 / 16.0 / 19.0 / 93 | 8.0 | 0 |
| 16 KB mat→lanfear | 19.0 / 29.0 / 40.6 / 113 / 1608 | 14.5 | 0 |
| 16 KB lanfear→mat | 20.0 / 31.0 / 38.1 / 115 / 120 | 15.5 | 0 |

Raw (unverified) `uc_bench` 4 KB: 11.55 / 10.93 µs RTT per direction — the
verified numbers carry a few µs of per-byte check overhead on both sides.

**Findings.**

- **Both sizes clear the threshold with ~2.5× margin. T1 is open.** 4 KB
  half-RTT ~8 µs; 16 KB (the gate's actual payload) ~14.5–15.5 µs p50 one-way.
- **A single 16,384 B UC SEND WR works on this stack** — the ds4 gate shape
  (`DS4_TP_RDMA_MAX_MSG = 16384`, one WR, no chunking): no EPERM at post, all
  2000 × 16 KB deliveries byte-verified. The `uc_bench` note ("UC SEND >4096 B
  EPERMs at post") does not apply here; ds4's all-day R10e runs are
  independent proof.
- 16 KB p50 RTT (29–31 µs) ≈ 4 KB (16 µs) + ~14 µs: consistent with
  per-TB-frame processing (~3–4 µs × 3 extra frames) plus ~2.7 µs of pure wire
  at the R10b ceiling (4.4 GB/s). Payload size is not what makes the gate
  slow — hardware one-way is ~15 µs against ds4's observed 49.7 µs wire
  (R9→R10e), corroborating T1's ~35 µs/gate software estimate.
- One transient first-ping UC drop occurred across the whole battery (4 KB
  lanfear→mat; retry clean). UC has no retransmit: a T1 implementation needs a
  re-arm/retry path for a dropped gate (`timeout_sec` covers correctness, not
  latency).

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
asked for a 64 MiB message; that is structurally impossible on this provider.

> **Correction (M3, 2026-08-26):** the stated reason — *"Apple TB UC SEND is
> capped at 4096 B per WR (anything larger never completes; >4096 posts
> EPERM)"* — is **wrong**. M3 posted 2000 single **16,384 B** UC SEND WRs per
> arm and byte-verified every delivery, and ds4 itself runs
> `DS4_TP_RDMA_MAX_MSG = 16384` as one WR in production. The EPERM was a
> property of `uc_bench`'s configuration, not the stack. The bandwidth numbers
> below stand (they were measured, not inferred); only the explanation for why
> 64 MiB was not attempted does not.

`-s` is clamped to MTU. The valid proxy is sustained
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

### PP-RDMA — `pp-rdma-new` (old base, **superseded** by the same-commit PP re-measure above; kept for history only — do not quote)

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

Numbers in the **Current state** table above; it is same-commit and supersedes
everything here. Summary:

- **TP wins decode at every context**, +37–48%. TP splits compute per layer; PP
  serialises full activations over the link on the decode critical path.
- **TP wins prefill at ≤16k** (+34% @2k, +44% @8k, +5% @16k); **PP wins from 32k
  up** (+15/22/21%). PP moves activations twice per pipeline pass while TP moves
  them once per layer per chunk, so TP's transfer cost scales with layers ×
  chunks and PP's does not.
- Score cross-architecture claims **only against a same-commit PP run**. Several
  wins in this series (the upstream indexer stack, `nsg=4`) sit on the shared
  Metal path and lifted PP too, so TP-only deltas do not move the comparison by
  the same amount. The earlier "TP now beats PP at 131k" headline was drawn
  against a stale PP figure and was wrong.

## Key TP internals (from code read)

- Transport: RDMA over AppleThunderboltRDMA. Decode uses a per-token gate
  schedule (86 gates/token = 43 layers × 2, derived not hardcoded); prefill uses
  bulk "big gate" row swaps — one full-batch hidden-state exchange per layer,
  chunked into 16 KiB messages (`tp_rdma_big_gate_exchange`, `ds4_tp.c`).
- Attention **head** splitting is the **decode** path; **prefill** splits
  attention **rows** (`tp_row_split_attn` in
  `metal_graph_encode_layer_attention_batch`). Row-splitting now covers
  `pos0 > 0` chunks and the ratio-128 layers, and the indexer score/top-k splits
  too — all three default-on since `f45b535`. (This section previously described
  the pre-A0 state; that is what the whole R5–R7 arc changed.)
- Prefill chunking is layer-major; chunk cap from `ds4_prefill_cap_for_prompt`
  (env `DS4_METAL_PREFILL_CHUNK`, default 4096 for prompt > 4096). The **server**
  additionally slices prefill into `--prefill-quantum` steps in batched mode
  (default 2048), and the smaller of the two is what actually runs.
- Compressed KV grows with context: `comp_cap = ctx_size/ratio + 2`. Raw SWA
  cache is a ring bounded by `DS4_N_SWA`; `ds4_session_rewind()` snaps to a
  compressor-window boundary (`lcm` of the non-zero ratios, 128 here).
- CUDA TP has cache-duplication/peer-read infra (`cuda_tp_attn_cache_dup`,
  `cuda_tp_attn_peer_read`, `layer_attn_comp_cache_tp[]`, `copy_xdev`) that could
  template a Metal TP compressed-cache split.

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
