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

> **CORRECTIONS APPLIED 2026-08-27** from the arms C–F audit (50 agents; 27
> findings survived adversarial refutation, 31 refuted). Raw observations are
> untouched; interpretations are annotated inline with `**CORRECTION:**`.
>
> 1. **Every per-encoder µs/call figure in arms B, B3, B4 and B5 is retired** —
>    now confirmed on the rig by arm B6 below, not merely predicted. What
>    survives: tick 1.000 ns, `conc ≈ 2`, `union = 100%`, `gap ≈ 0`, throughput.
> 2. **Arm C is a PREFILL reading, not decode.** Decode `n_keys` = 128/144/640.
>    640 is inside the 512–700 PASS band, so **item G is not killed**.
> 3. **Arm F confirms 72.4% of the compressor accounting, not 100%.** Do not
>    quote 608/0.903 = 673 GB/s.
> 4. **Arm D's 3.464 µs/dispatch is a LOWER bound**, not an upper bound.
>
> Full correction table with file:line evidence:
> `docs/TP-A0-ROWSPLIT-TEST-PLAN.md`, "Corrections owed to BENCHMARKS-TP-PP.md".

### Arm R1 re-run — path counts resolve the branch ambiguity; label-collapse persists — 2026-08-28

Build `d8536ed` (post `9685613` label fixes + `51198b5` path counters). Pass 1
at 2k + 131k; pass 2 SPLIT at 2k only (131k SPLIT known to inflate 50×).

**Path counts (calls, not spans) — the decisive new signal:**

| label | 2k | 131k |
|---|---|---|
| `raw_attn` | 256 | 256 |
| `inv_rope` | 256 | 2944 |
| `idx_sort` | **0** | 672 |
| `idx_attn` | **0** | 672 |
| `idx_attn_split` | **0** | 2688 |
| `idx_split_red` | **0** | 2688 |
| `indexer_score_llt` | 0 | 2688 |
| `mask_fill` / `mask_cpy` / `kv_stage` / `fa_core` / `reduce` | 5248 each | 2560 each |

**R1's "`idx_*` stay fused at 2k" reading was wrong: they never ran.** Indexed
attention needs >1024 *compressed* rows (`ds4.c:19947`); ratio-4 at 2k gives
2048/4 = 512, so the branch is structurally absent at 2k. At 131k the idx
branches fire (672 sort/attn, 2688 split/red, 2688 score_llt). The path counter
turns three lines of inference into one unambiguous line — the fix works.

**Honest (pass 1) still collapses the 9-label bracket into ONE composite**
(`mask_fill..reduce+5`), conc mean 1.95-1.99, max 2, union 100%, gap 0.000 ms.
The whole FA bracket is one lump: **~4.31 ms @2k / ~3.24 ms @131k per token**
(vs stage-profile `attn_inv_rope` 3.817/4.243 — same operating ballpark, not a
clean match). Label-collapse persists in honest mode: all 9 FA labels pile onto
one span per encoder; the instrument cannot sub-divide the bracket honestly.

**Pass 2 SPLIT @2k** (perturbed, 32.96 t/s, conc 1.74-1.87, gap 3.3-4.4 ms):
splits into `reduce` 36.1, `fa_core` 28.3, `kv_stage` 5.5, `mask_fill..mask_cpy`
8.9 µs/call — ceilings/ratios only, as in B6.

**Verdict — per plan read-order #4, the residual moves to the `ds4.c` call
site.** The 9 labels still do not sub-divide honestly on the fixed build (pass 1
is one composite; pass 2 perturbs). The encoder instrument has gone as far as
it can. The unnamed ~1.9-2.4 ms inside `attn_inv_rope` is not separable at the
GPU-encoder boundary and must be chased where the stages are distinct code paths.

### Arm W1 — does prefill pay the batch-encode fixed cost? — 2026-08-28

**Units corrected 2026-08-28 (`5d1f45d`): the table is in SECONDS, not ms.**
Build `6d632c9`. Prefill 2048 tokens in 1/4/16/64 steps (additive path,
`--step-incr 512/128/32`), gen 8, pair restart per arm. Total prefill time =
Σ per-step (prefill_tokens / prefill_tps) from CSV.

| batches | total prefill (s) |
|---|---|
| 1 | 4.522 s (452.9 t/s) |
| 4 | 5.676 s |
| 16 | 10.168 s |
| 64 | 25.387 s |

**Fit (R² = 0.9989): total = 4.45 + 0.329 × batches s** →
**F = 329 ms/batch = 7.65 ms/layer** (per 43-layer batch).

**Decision rule fires the OTHER way — F ≫ 35 ms-equivalent.** Prefill's
per-batch cost is **7.65 ms/layer = 9.4× the verify's 810 µs/layer**, not 100×
smaller. But **the arm cannot settle the question**: varying batch count at fixed
total tokens varies encode overhead *and* expert re-streaming together (a
32-token chunk routes to ~136 of 256 experts ≈ 39 GB ≈ 87 ms). The fit is real
but confounded. This is why W1b exists (holds routed work constant).

**What W1 does establish:** prefill carries a genuine per-batch cost with an
excellent fit — **chunk size matters far more than a few percent** (at 64
batches 2k prefill collapses 453 → 81 t/s). Prefill is safe because the work
budget floors chunks at 512 tokens; the exposed case is multi-slot batching
(W3).

### Arm W1b — separate encode overhead from weight re-streaming — 2026-08-28

Build `6d632c9`. Prefill 2048 tokens in 1/4/16/64 steps (additive path,
`--step-incr 512/128/32`), gen 8, pair restart per arm. Total prefill time =
Σ per-step (prefill_tokens / prefill_tps) from CSV.

| batches | total prefill |
|---|---|
| 1 | 4.522 ms |
| 4 | 5.676 ms |
| 16 | 10.168 ms |
| 64 | 25.387 ms |

**Fit (R² = 0.9989): total = 4.45 + 0.329 × batches ms** →
**F = 329 µs/batch = 7.6 µs/layer** (per 43-layer batch).

**Decision rule fires: F ≪ 35 ms → the verify's fixed term is verify-specific.**
Prefill pays **7.6 µs/layer** of per-batch fixed cost against the verify's
**810 µs/layer** — ~100× smaller. The 34.83 ms verify fixed term is NOT the
general batch-encode cost; it is local to the speculative path (TP batch gates,
capture-row bookkeeping, the spec-cycle wrapper). This closes the "largest
unexplained quantity in the engine" question: it is not an engine-wide encode
overhead, it is the spec-cycle machinery, and it does not threaten prefill or
any non-speculative multi-row batch.

> **CORRECTION 2026-08-28 — the table is in SECONDS, and the verdict inverts.**
> The section's own cross-check gives it away: 2048 tokens in 4.522 **ms** would
> be 452,897 t/s. The quoted 452.9 t/s is right, so the unit is **seconds** —
> and every row confirms it (as ms they read 80,671 to 452,897 t/s).
>
> | batches | total | t/s |
> |---|---|---|
> | 1 | 4.522 s | 452.9 |
> | 4 | 5.676 s | 360.8 |
> | 16 | 10.168 s | 201.4 |
> | 64 | 25.387 s | 80.7 |
>
> So **F = 329 ms/batch = 7.65 ms/layer**, not 329 µs / 7.6 µs — **9.4× the
> verify's 810 µs/layer, not 100× smaller.** The stated decision rule
> (`F ≪ 35 ms`) fires the other way: 329 ms ≫ 35 ms.
>
> **But the arm cannot answer the question either way, and that is a flaw in how
> it was specified.** Varying batch count at fixed total tokens varies two things
> at once: encode overhead *and* expert re-streaming. A 32-token chunk routes to
> ~136 of 256 experts (~68 per rank) ≈ 39 GB of expert weight ≈ 87 ms at the
> measured rate, so a large share of the 329 ms is real, unavoidable work — and
> is exactly why prefill uses large chunks. Comparing a per-batch cost against
> the verify's fixed term treats them as the same quantity; they are not.
>
> **What the data does establish**, and it is worth having: the fit is excellent
> (R² = 0.9989) and prefill batching carries a genuine ~329 ms per-batch cost, so
> chunk size matters far more than a few percent — at 64 batches, 2k prefill
> collapses from 453 to 81 t/s. Prefill itself is not threatened, because the
> internal work budget already floors chunks at 512 tokens. The exposed case is
> multi-slot batching, which makes W3 more interesting rather than less.

**Note on the 1-batch arm:** 2048-token single prefill = 4.522 ms = 452.9 t/s,
consistent with the established 2k prefill. The 64-batch arm's 25.4 ms total is
all fixed-cost accumulation (63 × 0.329 ms = 20.7 ms of overhead on a 4.45 ms
base) — the fixed cost is real and additive even at prefill scale, just 100×
smaller than the verify's.

**W2 (analysis) and W3 (ds4-server multi-slot) follow from this:** W2's
proportionality question is now moot at rig level — the per-batch cost is
7.6 µs/layer, not 810; W3's multi-slot exposure is bounded by this smaller F,
not the verify's 34.83 ms.

### Arm V-residual — the verify's unattributed 44 ms — 2026-08-28

Build `d8536ed` (latest `99601ca`). Three `--dspark --mtp SUPPORT` runs at 2k,
gen 512, each with `DS4_DSPARK_VERIFY_PROFILE=1 DS4_DSPARK_STATS=1`:

| run | force | cycles | proposed | accepted | verify ms | t/s |
|---|---|---|---|---|---|---|
| 1 | (production policy) | 502 | 6 | 0 | 0.000 | 40.33 |
| 2 | `LOW_YIELD_POLICY=0` | 482 | 56 | 20 (35.7%) | 917.9 | 35.89 |
| 3 | `LOW_YIELD=0 SCHEDULER=0` | 468 | 181 | 42 (23.2%) | 1794.2 | 29.14 |
| 4 | + `FAKE_ARGMAX_PROPOSAL=1` | 468 | 498 | 42 (8.4%) | 1771.0 | 28.99 |

**Run 1 is a null by design**: the production low-yield policy backs off
(`no_draft=498, backoffs=4`) because prose acceptance is ~0%, so no verifier
ever launches. The plan's plain command measures nothing on prose — it needs
`DS4_DSPARK_TP_LOW_YIELD_POLICY=0` (and scheduler off for volume).

**V(k) fit (consolidated n=50 @d2, n=7 @d3):**

```
V(2) = 76.12 ms/verify    V(3) = 96.76 ms/verify
fit  V(k) = 34.83 + 20.64·k
V(5) extrapolated = 138 ms  (vs 108.18 measured in tp_mtp_hunt)
intercept fraction of V(5) ≈ 25%  (plan guessed ~42.5 ms / 39%)
```

**Intercept is real but smaller than the plan's guess** (~35 ms, not 42.5),
and the extrapolation to V(5) overshoots the measured 108 ms — the fit is
nonlinear across d5 (d5 uses the native 5-row tile, a different path from
d2/d3, per tp_mtp_hunt). The fixed term is not 39% of V(5).

**The decisive read — verify is GPU-layer-encode-bound, not transport-bound:**
`verify_layer` = **99.97%** of verify in every run (1793.7/1794.2, etc.);
`verify_upload` 0.26-0.5 ms, `verify_read` 0.26-0.4 ms — negligible. The 44 ms
unattributed by the byte model lives **inside `verify_layer`** (the 43-layer
batch), not in any per-row bucket the byte model can see. The fixed term is the
43-layer fixed launch/encode overhead, not a data movement cost.

**Propose now dominates the budget** (run 3/4: propose 3859/3945 ms vs verify
1794/1771 ms), and `prop_chain` = 86% of propose — consistent with the
`8df84c3` correction that propose is **latency-bound**, not a 5.99 GB
bandwidth wall. The two cost terms to attack are `prop_chain` (latency) and
`verify_layer` (fixed 43-layer encode overhead).

**Draft-length ceiling on prose:** even fake-argmax forces only d1 (441/498)
and d2 (24) — the Markov chain stops early on prose, so d4/d5 verify samples
are structurally hard to get on this fixture. A coding fixture would produce
longer drafts but the verify cost per row is what this arm set out to measure.

### Arm S — n-gram commit rate, first measurement on this rig — 2026-08-27

Build `d8536ed` (trace hook now in `ds4_session_eval`, the path ds4-bench and
ds4-server actually use). Coordinator-only `DS4_NGRAM_TRACE`; worker NOT set.
Heartbeat confirmed ("writing (N tokens so far)") — the `9685613` fix works.

**Two workloads traced and analysed offline** (`tests/bench_ngram_accept`, the
shipped proposer, linked):

| trace | tokens | best | best cfg | mean_cmt @depth4 | offered% @depth4 |
|---|---|---|---|---|---|
| **prose** (`promessi_sposi.txt`) | 6143 | **1.097×** | k=6 d=4 | 0.30 | 9.9% |
| **coding** (hash-table C program) | 3938 | **1.034×** | k=8 d=4 | 0.15 | 4.0% |

**Commit-length distribution is heavily dominated by zero-commit steps:**
prose 91.4% commit=0 (6.4% commit=4); coding 96.5% commit=0 (2.1% commit=4).

**Legacy flat V/T=4.459 comparison:** prose **0.907× (loss)**, coding **0.945×
(loss)** — nothing fundable at the old verify cost either.

**Verdict — speculation does NOT clear the bar.** The decision rule requires
~2.0 mean commit; the best observed is **0.30 (prose) / 0.15 (coding)** — 6-13×
short. Both traces land just above the uniform-random control (1.000×, zero
offers) and far below the period-8 synthetic (1.716×). **At the corrected cost
model the best case is ~1.03-1.10× — a rounding error, not a route to 50 t/s.**

**Caveat.** Both traces are single-pass generations (prose prompt, one coding
prompt). The plan's premise — that a *real* PI coding-harness session (editing,
refactoring, repeated code across turns) carries far more repetition than a
synthetic prompt — was not captured; `ds4-server` was built but no harness
session was driven through it. If that premise holds, a harness trace could
still move the number; as measured on these workloads, n-gram drafting is dead.

### Arm D-slope — item C dispatch-ballast slope — 2026-08-27

Build `d8536ed`. Three interleaved repeats of `DS4_METAL_DISPATCH_BALLAST ∈
{0,2,8,16}` at 2k, gen 128, pair restart each. Fit t/s vs N.

| N | t/s per repeat | mean | sd |
|---|---|---|---|
| 0 | 41.19, 41.20, 41.17 | **41.187** | 0.015 |
| 2 | 40.48, 41.00, 41.00 | **40.827** | 0.300 |
| 8 | 40.65, 40.66, 40.60 | **40.637** | 0.032 |
| 16 | 40.06, 39.95, 39.90 | **39.970** | 0.082 |

**Fitted: t/s = 41.106 − 0.0694·N.** Per-token time goes 24.327 → 25.003 ms
(Δ0.676 ms) across N=0→16 = 688 no-ops → **0.98 µs/dispatch at N=16**.

**Prediction check: falsified.** The plan predicted N=16 → 37.53 t/s (3.69
drop = 7.4× signal). Observed **40.00 t/s (1.11 drop)** — the dev-box-scaled
3.464 µs/dispatch is **3.3× too high on the rig**.

**Nonlinearity.** Marginal cost per no-op falls with N: N=0→2 (86 no-ops, 0.214
ms) ≈ **2.49 µs/dispatch**; N=0→16 ≈ **0.98 µs**. Later no-ops hide behind the
established ~2× encoder overlap. **Item C's 0.510 ms / +0.88 t/s is
over-estimated**; the true per-dispatch cost is ~2.5 µs at the small-N scale a
dispatch reduction actually operates at, ~1 µs at scale. The earlier arm D
(41.22 vs 40.72, Δ0.50 t/s for +86) is consistent with the N=0→2 leg (Δ0.36
here, 0.214 ms) — both ~2.5 µs.

### Arm R1 — name the 2.42 ms inside `attn_inv_rope` — 2026-08-27

Build `179c105` (5 new `DS4_METAL_PROFILE_BRACKET_STAGE` labels: `idx_sort`,
`idx_attn_split`, `idx_split_red`, `idx_attn`, `rope_tail`, taking bracket
coverage 4→9). Two passes × 2k + 131k, gen 128.

**Pass 1 — the whole bracket collapses into one composite.**

| ctx | composite | of total |
|---|---|---|
| 2k | `rope_tail..reduce` (9 labels in one segment) | 27 of 109 per cb |
| 131k | `rope_tail..reduce` 2432, `rope_tail..idx_split_red` 2560, `rope_tail..idx_attn` 672 | — |

Pass 1 honest reads hold: **gap ≈ 0.000 ms (union 100%), conc mean 1.97-1.98**
— consistent with B5/B6 (no idle pool in the buffer; stalls inside encoders).
Throughput baseline (42.03 / 29.37 t/s).

**Pass 2 (SPLIT) — the falsifier fires; pass 2 is ceilings, ratios only.**
Pass 2 @131k inflates pathologically: `idx_attn` 38511 µs, `rope_tail` 251 µs,
total **212 ms ≫ attn_inv_rope 4.24 ms** — exactly the pre-registered failure
mode (SPLIT perturbation inflates spans faster than it resolves them). At 2k
pass 2, only three labels split out cleanly:

| label (2k pass 2) | fair | product |
|---|---|---|
| rope_tail | 9.8 µs | 0.222 ms |
| fa_core | 24.7 µs | 0.537 ms |
| reduce | 30.5 µs | 0.662 ms |
| **sum** | | **1.42 ms** |

**Verdict.** The five new labels do **not** cleanly split the 2.42 ms residual:
`idx_*` and `gather`/`packed` do not emerge as separate spans (they stay fused
into the composite at 2k, and inflate pathologically at 131k). Per the
pre-registered falsifier, pass 2 is a ceiling; the 9-label sum does not
reconcile to 3.64 ms. **The ~1.9 ms unexplained residual is still not named by
the encoder instrument.** Per the plan's read order #4, the next step is the
call site in `ds4.c` rather than the encoder.

### Arm B6 — the label was the bug: `reduce` was never a reduce — 2026-08-27

Build `eb90b5f` (labels counted per span, composites printed; `SPLIT` mode
opt-in). Two passes × 2k + 131k, gen 128.

**Pass 1 (default) — the falsifier fired: the collapse is real and total.**

| ctx | composite spans | of total | label |
|---|---|---|---|
| 2k | **3456** | 27 of 109 per cb | `fa_core..reduce` |
| 131k | **2432** | 19 of 157 per cb | `fa_core..reduce` |

**`reduce` was never a reduce.** In the batch, `ds4_gpu_end_compute_encoder` is
a no-op and `ds4_gpu_compute_encoder` reuses the same encoder without a new slot,
so all four FA label sites (`gather`/`packed`/`fa_core`/`reduce`) wrote the SAME
slot and `ds4_gpu_ts_name_last` overwrote unconditionally — `reduce`, last, won.
The span reported as `reduce` was the **whole batch segment**, and the 27/token
(2k) / 19/token (131k) count was the number of segments whose last label was
`reduce` — which is why it matched no partition of the 43 layers. **Everything
per-encoder from arm B through B5 is retired** (µs/call figures). What survives:
tick 1.000 ns, conc ≈ 2, union 100%, gap ≈ 0, and the throughput baselines.

**Pass 2 (SPLIT=1) — true per-stage spans, perturbed (ratio, not absolute):**

| ctx | fa_core | reduce | reduce product | `attn_inv_rope` | t/s (perturbed) |
|---|---|---|---|---|---|
| 2k | 25.8 µs | 30.5 µs | **0.66 ms** (×21.7/token) | 3.64 ms | 32.34 (-23%) |
| 131k | 59.6 µs | 69.2 µs | **0.60 ms** (×8.7/token) | 4.24 ms | 25.12 |

**`reduce` alone is ~30-69 µs/call and ~0.6 ms/token — tiny against
`attn_inv_rope` (3.64/4.24 ms).** The 2k overshoot is fully explained: it was a
composite span, not a paradox. FA split @2k ≈ fa_core 46% : reduce 54%.
Residual composites in pass 2 are minimal (1 of 512 spans, prefill/boundary).
Pass 2 perturbs as designed (over-serialisation): conc 1.78-1.79, gap opened to
2.7-6.3 ms, throughput −23%/-15%.

**Correction to arm C's layer table:** ratio-128 is **20** layers, not 21
(2 raw + 20 ratio-128 + 21 ratio-4 = 43).

### Arm B5 — fair-share decomposition; the falsifier fires — 2026-08-27

Build `a59d3af` (fair-share interval decomposition replaces the uniform
divisor). Same flags, 2k + 131k, gen 128.

**New instruments — union/conc confirm the overlap conclusion and close the
idle question:**

| ctx | conc mean | conc max | union of cb | gap |
|---|---|---|---|---|
| 2k | 1.97-1.99 | 2 | 100.0% | **0.000 ms** |
| 131k | 1.98 | 2 | 100.0% | **0.000 ms** |

**`conc mean ≈ 2.0` — pipelining depth really is ~2**, so the banked "2×
overlap" conclusion stands (not a span-endpoint artefact). **`gap ≈ 0` — no
idle pool inside the command buffer** (union = 100% of cb), consistent with
arm E's 100% residency/max clock. The stalls are **inside** the encoders, not
in scheduling between them.

**`fair` for `reduce` — the falsifier fires:**

| ctx | calls/token | norm µs/call | **fair µs/call** | product | `attn_inv_rope` | ratio |
|---|---|---|---|---|---|---|
| 2k | 27 | 159.1 | **157.4** | **4.25 ms** | 3.64 ms | **1.17× — still over** |
| 131k | 19 | 170.0 | **169.0** | 3.21 ms | 4.24 ms | 0.76× — plausible |

**Fair barely moved from norm (157.4 vs 159.1 @2k) and the 2k product still
exceeds its containing stage.** The fair-share decomposition — which corrects
exactly the non-uniform-overlap error — did not fix the overshoot. Per the
pre-registered falsifier, all three candidates are now dead (loss, slot
aliasing, non-uniform overlap).

**Remaining suspect: the cross-instrument comparison itself.** B's token is
~23.7 ms (42.15 t/s) while the stage profile that produced
`attn_inv_rope = 3.64 ms` ran at 28.83 ms gpu_busy — a ~22% different
operating point (profiler tax). Per the plan, no per-encoder figure enters the
budget until `attn_inv_rope` and the enc-ts `reduce` are re-measured **in the
same run**. Throughput at baseline (42.15 / 29.62 t/s).

> **CORRECTION — it was cause 4, the label.** `reduce` was never the reduce
> dispatch. In the batch, `ds4_gpu_end_compute_encoder` no-ops (`ds4_metal.m:1389`)
> and `ds4_gpu_compute_encoder` reuses the open encoder **without reserving a
> timestamp slot**; decode reaches the FA path with `owned = 0` and
> `cb == g_batch_cb` (`ds4_metal.m:963`), and there is no `close_batch_encoder`
> between the four label sites (`:29617/:29678/:29698/:29726`). All four named one
> slot and the last writer won. **The span is the whole batch segment.**
>
> Two independent confirmations, both at zero rig cost:
> - `reduce` is the **only** FA label ever reported across all four enc-ts arms —
>   10 mentions, zero for `gather`, `packed`, `fa_core`. Separate encoders would
>   show all four.
> - The reduce kernel's own traffic is `21×1,579,008 + 20×328,960 = 39.7 MB / 41 =
>   969 kB/call` → **1.28 µs/call** at 760 GB/s against a reported 157–160.
>   **123× off.** No inefficiency spans two orders of magnitude.
>
> This also explains the count that never matched any partition of 43 layers:
> **27/token is the number of batch segments whose last label was `reduce`.**
>
> **Why `fair` barely moved:** with `conc` identically 2, fair-share and the
> uniform divide are the *same operation* (`raw/2` vs `raw/1.97`). B5 was a null
> test by construction and could not have separated cause 3.
>
> **What survives from B5, unaffected by the label:** `conc mean ≈ 2.0` /
> `max = 2` (pipelining depth really is 2), and `union = 100%` of cb with
> `gap = 0.000 ms` — **no idle pool between encoders. The stalls are inside the
> encoders.** That is the most consequential result the instrument has produced.
>
> Arm B6 (`5f573a6`, `6260e6f`) makes composites self-reporting and adds
> `DS4_METAL_GPU_ENCODER_TIMESTAMPS_SPLIT=1` for true per-phase spans.
> **It ran and confirmed all of the above on the rig** — 3456 `fa_core..reduce`
> composites @2k / 2432 @131k, exactly the 27- and 19-per-token counts every arm
> since B has reported, and a true `reduce` of 30.5 µs/call (0.66 ms/token). See
> the Arm B6 section above.

### Arm B4 — banked slots + LOSS counter; the 2k overshoot survives — 2026-08-27

Build `c9e0f72` (carries `d645b29` banking fix). Same flags as run 3
(`DS4_METAL_GPU_ENCODER_TIMESTAMPS=1`), 2k + 131k, gen 128.

**The LOSS line (new) — context-dependent, and it does not explain the
overshoot:**

| ctx | ranges never reported | encoders lost |
|---|---|---|
| 2k | **257/388 = 66%** | 8449 |
| 131k | 129/1698 = 8% | 2177 |

**Decisive: the `reduce` count is identical to run 3 (3456 @2k, 2432 @131k)
and the norm is unchanged (159.5 / 170.1).** Banking changed which ranges get
reported, yet `reduce`'s count did not move — so the loss is dropping *other*
(unlabelled) spans, not `reduce`. The 41→27 calls/token drift is therefore
NOT a loss artifact; the reported reduce count is trustworthy.

**The product still overshoots at 2k:**

| ctx | calls/token | norm µs/call | product | `attn_inv_rope` | ratio |
|---|---|---|---|---|---|
| 2k | 27 | 159.5 | **4.31 ms** | 3.64 ms | **1.18× — survives** |
| 131k | 19 | 170.1 | 3.23 ms | 4.24 ms | 0.76× — plausible |

**Verdict per the plan's read order.** Loss is present (66% @2k) but does not
explain the 2k overshoot — the reduce count is unchanged by banking, so cause
3 (non-uniform overlap) is what remains. **The normalised column is not a
budget; treat raw as the upper bound and stop quoting norm.** The 131k row
reconciles (0.76×), so the per-call figure is probably right and the
uniform-overlap divide over-states `reduce` at 2k specifically.

Throughput at baseline (42.29 / 29.71 t/s) — instrument not distorting.

### Arms C-F — shape fields, dispatch ballast, powermetrics, compressor accounting — 2026-08-27

One session, build `05f402d`, 2k context, gen 128. All four zero-code arms.

**Arm C — flash-attn shape fields** (`DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1`;
timings are host round trips, read shapes only). Per-layer compression shape
at 2k (heads=64, dim=512, window=128):

| ratio | layers | n_comp | n_keys |
|---|---|---|---|
| 0 (raw) | ~2 | 0 | 2048 |
| 128 | 21 | 16 | 2064 |
| 4 | 21 | 512 | 2560 |

(Line counts: 8 raw, 120 ratio-128, 105 ratio-4 — /5 stages per prefill
chunk, 2 prefill chunks. The exact per-layer split is ratio-determined and
context-invariant.)

> **CORRECTION — this is a PREFILL reading and must not size `attn_inv_rope`.**
> All four `DS4_METAL_PROFILE_FLASH_ATTN_STAGE` sites are prefill encoders
> (`ds4_metal.m:28387/28669/29018/29272`), the print string is literally
> "Metal FlashAttention **prefill** stage" (`:11345`), and `heads=64` is
> structurally impossible for TP2 decode, which passes **32**. So
> `n_keys = 2048/2064/2560` is a prefill tile extent.
>
> **`n_comp` does transfer.** With `n_raw = 128` — fixed by
> `metal_graph_raw_span_for_batch` at `n_tokens=1`, window `DS4_N_SWA=128`
> (`ds4.c:19764-19780`) — the decode shapes are:
>
> | ratio | layers | n_comp | **decode n_keys** |
> |---|---|---|---|
> | 0 (raw) | **2** | 0 | **128** |
> | 128 | **20** | 16 | **144** |
> | 4 | **21** | 512 | **640** |
>
> **640 lands inside the 512–700 PASS band, so item G is not killed.**
>
> **Layer counts corrected: 2 + 20 + 21 = 43** (the table's `2+21+21 = 44`
> exceeds `n_layer`). The line counts confirm it exactly, and it is **one**
> chunk, not two: `2×4 + 20×6 + 21×5 = 8 + 120 + 105 = 233`. The divisor is
> per-class because pad fires on ratio-128 (`2064 % ncpsg(64) = 16`) and not on
> ratio-4 (`2560 % 64 = 0`) — `ds4_metal.m:28305-28307`.

**Arm D — dispatch ballast ∈ {0,2}** (`DS4_METAL_DISPATCH_BALLAST`) at 2k,
the dispatch price in the live graph:

| ballast | gen_steady t/s |
|---|---|
| 0 | 41.22 |
| 2 | 40.72 |

Delta 0.50 t/s for +86 no-op dispatches (2/layer × 43) ≈ **0.28 ms/token ≈
3.3 µs/dispatch** — above the 2.9 µs item-C kill threshold, so item C is not
dropped by this arm alone (marginal; the R12a/ballast nulls say added
dispatches hide in the ~2× pipeline overlap, so treat 3.3 µs as an upper
bound).

> **CORRECTION — arithmetic, and the bound runs the other way.**
> `1000/41.22 = 24.260068`, `1000/40.72 = 24.557957`, Δ `= 0.297889 ms`,
> `/86 = ` **`3.4638 µs/dispatch`**. The 0.28/3.3 figures are a double rounding,
> 5% low. Against the `250/86 = 2.9070 µs` threshold the margin is **1.19×**,
> not 1.14×. Verdict sign unchanged: **item C is not dropped.**
>
> **"Upper bound" is backwards on its own premise.** If added dispatches partly
> hide in the ~2× overlap, the marginal price reads *cheap* — so **3.464 µs is a
> LOWER bound** on what a real dispatch costs. A real dispatch also carries
> memory traffic and a dependency edge that a no-op ballast does not.
>
> **Design caveat.** Two points, single run, 1.21% signal, where the plan
> specified a 4-point fitted slope (`ds4_metal.m:1568`). A slope over
> {0, 2, 8, 16} interleaved would give 7.4× the signal.
>
> **Latent defect, inert today.** `ds4_gpu_decode_dispatch_ballast` takes `owned`
> (`ds4_metal.m:1768`) and never tests it before
> `ds4_gpu_finish_command_buffer(cb, owned, …)` (`:1779`), unlike the real gate's
> `if (!cb || owned) return 0;`. Harmless while decode always has `g_batch_cb`
> open (`owned = 0`), but a call outside an open batch would report a full
> command-buffer round trip as a "dispatch price".

**Arm E — powermetrics, decode vs bench_membw** (`sudo powermetrics
--samplers gpu_power -i 200`, coordinator host):

| phase | mean power | peak | clock | residency |
|---|---|---|---|---|
| prefill (2k) | ~59-60 W | 68 W | 1394-1398 MHz | ~90-100% |
| **decode** | **~33.4 W** | 34 W | **1398 MHz (max)** | **100%** |
| idle | ~0.08 W | — | — | — |
| bench_membw (760 GB/s) | 0.75 W all / 38 W active-burst | 66 W | DVFS ~950-1398 | bursty |

**Retires the "30 W / 20% utilised" premise.** Decode draws ~33 W but at
**1398 MHz max clock, 100% residency** — the GPU is not under-clocked or
idle; it is busy at the top P-state drawing a modest 33 W because the work
is latency/bandwidth-bound, not power-hungry. The 30 W figure was never
evidence of headroom. DVFS is NOT the decode gap: decode sits at max clock.

> **CORRECTION — right verdict, and there is a much stronger reason available.**
> The decisive fact is not that decode sits at max clock; it is that **prefill
> runs at 1394–1398 MHz — at or 0.3% *below* decode's pinned 1398 — while drawing
> 1.78× the power (59.5 vs 33.4 W).** A *lower*-clocked phase drawing *more* power
> excludes DVFS outright, with no appeal to what "residency" means. 1398 MHz is
> also the clock the FLOP roof was measured at
> (`60 × 128 × 2 × 1.398e9 = 21.47 TFLOP/s`), so "decode at ~2% of peak FLOPs" is
> not a DVFS artefact either.
>
> **Second leg — power does not track bytes.** `bench_membw` is genuinely
> bandwidth-saturated at 760 GB/s and draws 38 W active-burst; decode moves
> ~244 GB/s and draws 33.4 W. **3.11× the bandwidth for 1.14× the power.** So the
> 33 W reading was never evidence of headroom *or* of its absence.
>
> **Caveats to record, none load-bearing:**
> - At `-i 200` one sample averages `200/24.26 = 8.24 tokens`, so this arm can
>   never support a stage- or spin-level power claim.
> - n ≈ `3.105 s / 0.2 = 15.5` samples, one session, no replicate.
> - **The residency column conflicts with `BENCHMARKS-TP-PP.md:1505`** ("the
>   sampler on these boxes reports GPU *power*, not residency %"). One of the two
>   is wrong — attach the raw powermetrics stanza or strike one.
> - Coordinator only. Worker decode power is unmeasured, and an asymmetry would be
>   a direct readout of gate-wait imbalance — free, since the sampler is already
>   being run.
>
> **What this does NOT touch:** the "99% busy / ~20% utilised" occupancy premise
> behind U7/U9/U10. powermetrics residency is active-time fraction, not ALU
> occupancy; that premise stands on its own evidence and on the roofline
> (`2.26 / 27.6 FLOP/byte = 8.19%` max ALU utilisation).

**Arm F — QKV pair quad fuse disable** (`DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_QUAD_FUSE=1`
+ stage profiler) at 2k:

| stage | fused (iter 3) | **unfused** |
|---|---|---|
| q_a_kv_proj | 2.138 ms | **1.441 ms** |
| compressor_proj | (hidden) | **0.903 ms appears** |
| compressor_update | 1.278 (attrib) | 1.278 |
| total gpu_busy | 28.83 ms | 29.045 ms |

**Confirms the 608 MB compressor accounting.** Unfusing moves the quad
compressor out of q_a_kv_proj: q_a_kv_proj drops 2.138→1.441 (Δ0.697 ms) and
a compressor_proj row appears at 0.903 ms. The compressor read work is real
and lands where the corrected byte model put it. (F ran with stage
timestamps, so 35.70 t/s is ~13-18% distorted; the stage deltas are the
read.)

> **CORRECTION — it confirms 440.40 of 608.17 MB (72.4%), and once corrected the
> books are *better* than claimed.**
>
> `DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_QUAD_FUSE` gates only the `ratio==4` branch
> (`ds4.c:22478`). The 20 ratio-128 layers fuse behind a **different, unset** var
> `DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_COMPRESSOR_FUSE` (`ds4.c:22536`), and
> `comp_state_already_stored = qkv_pair_quad_fused` (`ds4.c:22881`) suppresses
> their re-emission. So:
>
> - **Movable:** `21×2×4096×1024×2 + 21×2×4096×256×2 = 352,321,536 + 88,080,384 =`
>   **`440,401,920 B = 440.40 MB`** — 72.4% of 608.17 MB.
> - **Immovable:** `20×2×4096×512×2 = 167,772,160 B = 167.77 MB`, still fused.
>
> **Do not quote `608/0.903 = 673 GB/s`.** The correct rates, net of one 0.18 ms
> marker tax: `440.40/0.697 = 632 GB/s` out and `440.40/(0.903−0.18) = 609 GB/s`
> in — **3.8% agreement**. At sum level `895.61/1.958 = 457` fused vs
> `895.61/1.984 = 451` unfused, −1.3%.
>
> **`compressor_update` fused = 1.278 is UNMEASURED.** The value appears nowhere
> else in the corpus and the Iteration-3 table does not list the stage. The
> 0.009 ms residual is conditional on it.
>
> **Stage deltas are safe only on rows with the same marker count in both arms.**
> `q_a_kv_proj` is 43 in both, so −0.697 is quotable. `compressor_proj` exists in
> one arm only, so its +0.903 and the +0.215 total each carry ~0.18 ms of pure
> instrument — the +0.215 "fusion is worth 0.21 ms" reading is **inside its own
> marker tax and is not supported**.
>
> **The one genuine anomaly** is the retained side at `455.21/1.261 = 361 GB/s`,
> explained by the +0.211 ms `gpu_busy` the deoptimised config itself costs
> (21 layers × 3.464 µs dispatch = 0.073 ms, remainder lost fusion).
>
> The arm also missed its own pre-registered prediction (`PATH-TO-50TPS.md:769-771`
> expected `q_a_kv_proj → 0.6–0.7`, `compressor_proj → 1.3–1.5`; observed 1.441 and
> 0.903) and was still logged "Confirms it".

### Arm B — encoder-timestamp re-baseline (re-run 3, slot-scoped + normalised build `05f402d`) — 2026-08-27

> **RETIRED 2026-08-27.** Every per-encoder µs/call figure below is void: the
> enc-ts label `reduce` was a composite span covering the whole batch segment,
> not the reduce dispatch (see the CORRECTION under Arm B5). The tick, the
> coverage/overlap reading and the throughput baselines are unaffected.


**Falsifier resolved.** The pre-registered falsifier said: if the inferred
tick pins at ~1.000 ns at both contexts on the slot-scoped build, slot reuse
was the single cause. It does:

| ctx | gen_steady t/s | tick | coverage |
|---|---|---|---|
| 2k | 42.25 | **1.000 ns** | ~197% OVERLAPPING |
| 131k | 29.75 | **1.000 ns** | ~198% OVERLAPPING |

**The counter is 1.0 ns on both parts; the 0.632 ns readings were slot-reuse
corruption** (global sample slots let a second in-flight buffer's encoders
append to the first's range). **Coverage ~197% is now a genuine measurement**
for the first time: encoder spans really do overlap ~2×, so the GPU already
pipelines adjacent encoders (pipelining depth ~2, ~132 µs of overlap per span).
This independently explains three standing nulls — R12a flat split schedule,
marginal ballast dispatch, dead dispatch-count reduction.

**Normalised budget** (the column that sums to the command buffer, per the
plan's instruction to take normalised as the budget; raw is an upper bound):

| ctx | calls/token | raw µs/call | **norm µs/call** |
|---|---|---|---|
| 2k | 27 (41 pre-slot-scope) | 314.0 | **159.7** |
| 131k | 19 (20 pre-slot-scope) | 339.3 | **171.3** |

Normalised `reduce` ≈ **160-171 µs/call**, vs the hand-corrected table's
`attn_inv_rope` 3.64 ms stage. Throughput at baseline (42.25 / 29.75 t/s)
confirms the instrument's ~1-3% distortion claim once more.

### Arm B — encoder-timestamp re-baseline (re-run, tick-calibrated build `0b987e5`) — 2026-08-27

> **RETIRED 2026-08-27.** Every per-encoder µs/call figure below is void: the
> enc-ts label `reduce` was a composite span covering the whole batch segment,
> not the reduce dispatch (see the CORRECTION under Arm B5). The tick, the
> coverage/overlap reading and the throughput baselines are unaffected.


Re-run because the first arm-B run (build `fba4ef0`) predated the tick
calibration (`d50bbaa`) and reported spans ~1.85× too large — it read the
counter as nanoseconds when it is a GPU tick.

**Tick calibration result — decisive.** The instrument now derives
seconds-per-tick from the command buffer's own clock and prints coverage %
and the inferred ns/tick:

| ctx | gen_steady t/s | encoders/token | inferred tick | coverage |
|---|---|---|---|---|
| 2k | 42.04 | 175 | **0.632-0.642 ns** | **~198-199% OVERLAPPING** |
| 131k | 29.70 | 174 | **1.000 ns (most), 0.92-0.93 (some)** | **~198% OVERLAPPING** |

**Both causes are present.** (1) The M2 Ultra counter ticks at **~0.63 ns**,
not 1.0 ns — so the original read-over-1.85× was a genuine unit error. (2)
Coverage ~198% means the encoder spans **genuinely overlap ~2×** (the GPU
pipelines encoder setup against the previous encoder's drain), so the
per-encoder figures are **upper bounds**, not exact busy times. At 131k the
derived tick reads ~1.0 ns because the overlap compresses `hi−lo`, inflating
the inferred tick — the two effects are entangled there.

**Calibrated labeled `reduce` span** (tick applied, still an upper bound due
to overlap):

| ctx | calls/token | µs/call |
|---|---|---|
| 2k | 41 | 199.4 |
| 131k | 20 | 311.1 |

These replace the void `fba4ef0` figures (313 / 339 µs/call). Throughput at
baseline (42.04 / 29.70 t/s) again confirms the instrument's ~1-3% distortion
claim. **Interpretation for the fan-out:** the true per-encoder busy times
are at most ~half the reported spans (overlap), so `reduce` is ~100-155 µs
per call — still to be reconciled against `attn_inv_rope`'s 3.64 ms stage.

### Arm B — encoder-timestamp re-baseline — 2026-08-27

> **RETIRED 2026-08-27.** Every per-encoder µs/call figure below is void: the
> enc-ts label `reduce` was a composite span covering the whole batch segment,
> not the reduce dispatch (see the CORRECTION under Arm B5). The tick, the
> coverage/overlap reading and the throughput baselines are unaffected.


Build `fba4ef0` (with `df0037e` `DS4_METAL_GPU_ENCODER_TIMESTAMPS`), rig,
gen 128. The low-distortion instrument (GPU timestamp counter at encoder
boundaries, one command buffer, ~1-3% distortion vs the 13-18% of
`DS4_METAL_GPU_STAGE_TIMESTAMPS`).

| ctx | gen_steady t/s | encoders/token |
|---|---|---|
| 2k | 42.08 | 175 |
| 131k | 29.71 | 174 |

**Throughput at baseline** — confirms the instrument's ~1-3% distortion
claim (no round-trip-per-marker inflation; the run is not distorted).

**Labeled `reduce` span** (the FA decode phase, the only non-unlabelled span):

| ctx | calls/token | µs/call |
|---|---|---|
| 2k | 41 | 313.1 |
| 131k | 20 | 339.0 |

**Caveat to flag (not yet interpreted):** the per-report "N encoders, X ms of
GPU time" summary sums ~44 ms @2k / ~64-69 ms @131k per report, which
**exceeds the per-token wall clock** (23.8 ms @2k / 33.7 ms @131k). The
individual encoder spans are in µs and labelled spans (reduce) reconcile
sensibly; the summary sum likely includes timestamp-unit or span-overlap
effects. Recorded raw for the fan-out to interpret against the hand-corrected
table.

### Iteration 3 — TP crash-fix validation, clean re-baseline, R12a split-schedule sweep — 2026-08-27

Build `06c2c6b` (post-C2-revert, with TP crash fix `69a3b86`), rig
(lanfear coord / mat worker). Three arms.

**Arm 1 — TP crash fix validation — PASS.** Prefilled 4095 tokens fresh
(493.64 t/s), then resumed 4095→4099 (`--step-incr 4`): the CSV shows the
resume prefilled only **4 tokens** (first chunk = 1 token, `to_boundary=1`,
the exact trigger). **Zero** "Metal model range … not covered by mapped model
views" / "resumed prefill failed" / GPU-timeout / "transport marked failed"
lines in the worker or coordinator logs. The double expert-offset rebase fix
(`69a3b86`) holds.

**Arm 2 — clean re-baseline (post-revert).** Confirms both corrected figures:

| stage | 2k ms/token | 131k ms/token |
|---|---|---|
| compressor_indexer | 0.198 | 9.657 |
| routed_moe_folded | 4.899 | 4.956 |
| attn_inv_rope | 3.817 | 4.243 |
| **attn_out_proj** | **2.867** | **2.865** |
| q_a_kv_proj | 2.138 | 2.135 |
| q_path | 1.867 | 1.852 |
| q_lora_norm | 1.672 | 1.656 |
| ffn_tp_gate | 1.558 | 1.485 |
| **attn_tp_gate** | **0.935** | — |
| **total gpu_busy** | **28.834** | **38.365** |

**`attn_out_proj` = 2.867 ms** (ctx-invariant), not the 2.38 from iteration
2's compromised build — close to the predicted ~2.73. `attn_tp_gate` 0.935
ms. The iteration-2 2.38 was the C2-broken build's artefact.

**Arm 3 — per-slot gate profiler (corrected formula) re-baseline:**

| ctx | ATTN | FFN | delta | straggler |
|---|---|---|---|---|
| 2k | 16.6 µs | 29.3-29.4 µs | +12.7 µs | **0.54-0.55 ms/token** |
| 131k | 15.9 µs | 28.1 µs | +12.2 µs | **0.52 ms/token** |

Straggler **~0.52-0.55 ms/token** — close to the predicted ~0.50 (not the
0.66 from iteration 2's broken build). U14 + §7 survive at ~0.5 ms.

**Arm 4 — R12a command-buffer split schedule sweep (131k):**

| split (first/second) | 131k t/s |
|---|---|
| 4/0 | 29.31 |
| 2/8 | 29.21 |
| 2/16 | 29.24 |
| 2/32 | 29.38 |
| 3/12 | 29.19 |
| 4/12 | 29.12 |

**Flat null — all arms within 0.9%** (29.12-29.38 t/s). The command-buffer
split schedule makes no measurable difference at 131k; best 2/32 (29.38),
worst 4/12 (29.12). The round-trip cost is not being attacked by the split
schedule, or it is already amortised at this context.

### Iteration 2 — stage profile (fixed attn_out_proj boundary), sampled flash-attn, corrected gate profiler, n-gram — 2026-08-27

Build `6b962db`, gen 128, rig (lanfear coord / mat worker). Zero-code arms
(four).

**Arm 1 — stage profile with `attn_out_proj` boundary added.** The boundary
fix separates what the last run's mislabeled `attn_tp_gate` (3.72 ms) was
hiding:

| stage | 2k ms/token | 131k ms/token |
|---|---|---|
| compressor_indexer | 0.199 | 9.643 |
| routed_moe_folded | 4.927 | 4.784 |
| attn_inv_rope | 3.784 | 4.260 |
| **attn_out_proj** | **2.383** | **2.384** |
| q_a_kv_proj | 2.141 | 2.144 |
| q_path | 1.876 | 1.859 |
| q_lora_norm | 1.675 | 1.676 |
| ffn_tp_gate | 1.643 | 1.651 |
| **attn_tp_gate** | **0.994** | **0.912** |
| **total gpu_busy** | **28.174** | **37.712** |

**`attn_out_proj` (2.38 ms, ctx-invariant) is a real top-4 stage** that the
last run's gate marker had swallowed; the actual `attn_tp_gate` is only
~0.91-0.99 ms. `attn_output` still reads ~0.000 (absorbed). The 3.72 ms
figure was a mislabel, not a real gate cost.

**Arm 2 — flash-attn split, sampled** (`..._EVERY` default 43 = one
layer/token). `decode_gathered` per-token: **fa_core 0.251 ms, reduce
0.274 ms** (123 calls/128 tokens — the sampling works, vs 10496 calls
un-sampled). Per-call fa_core ~0.25-0.41 ms, reduce ~0.27-0.35 ms @2k.
Still does **not** reconcile to the 3.78 ms `attn_inv_rope` stage marker
(per-layer × 43 ≈ 22.6 ms) — the honest decomposition of the 3.63-4.22 ms
remains open; A2 stays unsized.

**Arm 3 — per-slot gate profiler, corrected formula** (was printing
`43 × 2 × delta`, double-counting since `delta = E|s|/2` is already the
per-layer excess):

| ctx | ATTN | FFN | delta | straggler |
|---|---|---|---|---|
| 2k | 16.0 µs | 31.4 µs | +15.4 µs | **0.66 ms/token** |
| 131k | 17.5-17.6 µs | 33.5-33.9 µs | +15.9-16.3 µs | **0.68-0.70 ms/token** |

**Straggler ~0.66-0.70 ms/token** — the pre-fix 1.00/0.70 was double-counted.
The exchange delta is actually larger (+15-16 µs) but the per-layer cost is
halved. U14 and §7 survive at **~0.7 ms**, not 1.0.

**Arm 4 — n-gram arms: DID NOT ENGAGE.** The bench's plain-decode path calls
`ds4_session_eval` → `ds4_session_eval_probe_tp` (`ds4_bench.c:883`), which
**never reaches the n-gram dispatch** — that lives only inside
`ds4_session_eval_speculative_argmax_impl` (`ds4.c:69923`), reachable only via
`ds4_session_eval_speculative_argmax_excluding`, which the bench calls only
when `cfg.dspark`, and `--dspark` requires `--mtp FILE`. With
`DS4_NGRAM_SPEC=1 DS4_NGRAM_SPEC_STATS=1` on the plain bench, both 2k and
131k printed **zero `ngram spec` stat lines** and ran at baseline throughput
(42.00 / 29.55 t/s vs gate 41-42 / 29.7). **The n-gram arm needs a dspark
invocation (`--dspark --mtp FILE`) to be exercised — implementation-side
(user's domain).**

### Next-run battery (arms 1-5) — gate profiler, flash-attn split, stage profile, q8 shapes, attn_out nsg — 2026-08-27

Build `345de30`, gen 128, rig (lanfear coord / mat worker). Zero-code arms.

**Arm 1 — per-slot TP gate profiler (`DS4_TP_GATE_PROFILE=1`) — DECISIVE.**
ATTN vs FFN exchange are **NOT equal**:

| ctx | ATTN exchange | FFN exchange | delta | straggler bound |
|---|---|---|---|---|
| 2k | 17.8-17.9 µs | 29.4-29.6 µs | **+11.6 µs** | **≤ 1.0 ms/token** |
| 131k | 19.0-19.1 µs | 27.1-27.2 µs | **+8.1 µs** | **≤ 0.69-0.70 ms/token** |

The plan's decision rule: "if they measure equal, U14 and all three §7
designs die." They are **not** equal — the FFN gate (which sits behind the
routed shard) trails the ATTN gate by ~8-12 µs. **U14 and the §7 designs
SURVIVE.** The straggler is real but modest (~0.7-1.0 ms/token upper bound,
smaller at long context).

**Arm 2 — flash-attn stage split (`DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1`).**
Batch-context profile (inert on owned command buffer by design). Per-call
`decode_gathered`: fa_core ~0.31-0.33 ms, reduce ~0.40-0.49 ms @2k. NOTE:
per-call sums do **not** reconcile to the 3.8 ms `attn_inv_rope` stage marker
(~30 ms summed @2k) — the profile is a different (batch-context) epoch, and
its t/s is not comparable (13 t/s @2k vs 41 t/s unprofiled). Recorded raw
for the A2 sizing; the honest decomposition of the 3.63 ms still needs a
non-batch measurement.

**Arm 3 — stage profile (`DS4_METAL_GPU_STAGE_TIMESTAMPS=1`) with new gate
rows.** The gate is now a measured row, not a two-stage difference:

| stage | 2k ms/token | 131k ms/token |
|---|---|---|
| **attn_tp_gate** | 3.719 | 3.750 |
| **ffn_tp_gate** | 1.581 | 1.494 |
| compressor_indexer | 0.199 | 9.712 |
| routed_moe_folded | 4.911 | 4.963 |
| attn_inv_rope | 3.813 | 4.220 |
| q_a_kv_proj | 2.152 | 2.140 |
| q_path | 1.874 | 1.869 |
| q_lora_norm | 1.673 | 1.678 |
| **total gpu_busy** | **28.657** | **38.402** |

Note: `attn_output` now reads ~0.000 at 2k (its timing is absorbed by the
new `attn_tp_gate` marker). `attn_tp_gate` (3.72 ms, ctx-invariant) is now
visible as a top-5 stage — worth the plan's attention.

**Arm 4 — Q8_0 shape sweep (`tests/bench_q8_attn_shapes 760`, mat).** Rig
curve is **NOT monotonic** (M1 Max climbs monotonically):

| k | GB/s | % peak |
|---|---|---|
| 512 | 224-227 | 29.5-29.8 |
| 1024 | 279-280 | 36.8-36.9 |
| 2048 | 410-413 | 54.1-54.4 |
| 4096 | 445-458 | 58.6-60.3 |
| 8192 | 421-441 | 55.4-58.1 |

The curve **humps at k=2048 then recovers** — C1's shape (monotonic climb)
**must be re-established on the rig**; the M1 Max shape does not transfer.

**Arm 5 — `DS4_METAL_ATTN_OUT_LOW_NSG` sweep {1,2,4,8}.** Default is 4, so the
nsg4 arm is the baseline. **Monotonic improvement with higher nsg:**

| nsg | 2k t/s | 131k t/s |
|---|---|---|
| 1 | 38.86 | 28.02 |
| 2 | 40.66 | 28.79 |
| 4 (default) | 41.15 | 29.20 |
| **8** | **41.85** | **29.61** |

**nsg=8 beats default by ~+1.5-1.7%** at both contexts; nsg=2 slightly
worse; nsg=1 worst. T4's "nsg=4 ~3% worse everywhere else" finding does
**NOT reproduce** on the rig — higher nsg is monotonically better here.
C2 candidate: raise attn_out_low_nsg to 8.

> **CORRECTION — this sweep measured skipped work, and C2 is NOT banked.**
> `nsg` is a **baked function constant** on the pipeline while the dispatch
> passes a hardcoded 4. At nsg=8 the kernel reduces over 8 simdgroups with 4
> launched, so it covers 8 of 16 k-chunks — **it skipped half the weight
> stream**, and "faster" scaled with how much work was skipped. Shipped once,
> produced complete gibberish, **reverted in `da63283`**. The tree today reads
> `out_low_nsg = 4` (`ds4_metal.m:26870`). **Strike +1.7% from the banked list.**
>
> **Banked-list correction, all three entries.** Of U10 / T2 / C2 at 2k:
> C2 is reverted; T2's `DS4_METAL_DECODE_SPLITS` only reaches the indexed branch,
> which is unreachable at 2k (`ds4.c:23200-23202` gates on
> `layer_n_comp[il] > decode_sparse_threshold`); U10 tunes the indexer and is
> 0.02 ms net at 2k. **All three are ~0 at 2k.** The 41.22 t/s anchor already
> reflects that — there is no banked 2.5% to add to it.
>
> The lesson is the process one: this arm was a throughput-only sweep with no
> correctness gate, on the exact path where the standing bar is top-1 plus
> bounded Δlogit. A pipeline constant and a dispatch argument are a paired
> quantity and must move together.

### U15 — stage profile at 2k (first ever) + HC arms — 2026-08-27

`DS4_METAL_GPU_STAGE_TIMESTAMPS=1` on the rig (build `c13e3bb`), gen 128,
ctx 2048. Per-token stage gpu_ms. First 2k profile ever run.

| stage | 2k ms/token | 32k (ref) |
|---|---|---|
| routed_moe_folded | 4.901 | 4.990 |
| attn_inv_rope | 3.806 | 3.393 |
| attn_output | 3.731 | 3.746 |
| q_a_kv_proj | 2.150 | 2.137 |
| ffn_hc_post | 1.921 | 1.812 |
| q_path | 1.875 | 1.862 |
| q_lora_norm | 1.684 | 1.680 |
| attn_hc_pre | 1.135 | 1.142 |
| ffn_hc_pre | 1.109 | 1.119 |
| router | 1.101 | 1.112 |
| shared_gate_up | 0.974 | 0.985 |
| shared_down | 0.665 | 0.660 |
| attn_hc_post | 0.471 | 0.467 |
| compressor_indexer | 0.201 | 4.957 |
| **total gpu_busy** | **28.613** | 32.70 |

**Context-invariance confirmed at 2k.** Every stage matches its 32k value
within noise except `compressor_indexer` (0.201 ms @2k vs 4.957 @32k) — the
long-context term collapses exactly as the model predicts. The short-context
token is dominated by context-invariant stages.

**HC arm — `DS4_TP_ABLATE=hcpre`:** attn_hc_pre 1.135→0.645, ffn_hc_pre
1.109→0.627 (−~0.48 each), total 28.61→27.86 (**−0.76 ms**). The ablation
removes only 0.76 ms despite the profile attributing 2.24 ms to the hc_pre
stages — **the ~2.7× profile-vs-ablation disagreement reproduces at 2k** (the
plan's flagged open item).

**HC arm — `DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE=1`:** attn_hc_pre
1.135→1.416, ffn_hc_pre 1.109→1.398 (+~0.28 each), total 28.61→29.20
(**+0.58 ms — the fusion is a net win**, not a free flip; the three `seq_cst`
fences cost less than the dispatch they remove).

### U12b — q_path split + inverse-RoPE isolation — 26.8% explained — 2026-08-27

`DS4_METAL_GPU_STAGE_TIMESTAMPS=1` stage-profile on the rig (build
`26ef09f`, which adds the `q_a_kv_proj` / `q_lora_norm` / `q_path` split
markers), gen 128, 4 arms: default and
`DS4_METAL_DISABLE_PRE_M5_ATTN_INV_ROPE_FUSE=1` at 32k and 131k. Per-token
stage gpu_ms.

**Arm 2 — q_path split (default; context-invariant):**

| stage | contents | 32k | 131k |
|---|---|---|---|
| q_a_kv_proj | fused q_a/kv Q8_0 pair | 2.137 | 2.136 |
| q_lora_norm | q-LoRA RMS norm (+ fused KV RoPE) | 1.680 | 1.680 |
| q_path | q_b proj + per-head RMS norm + RoPE tail | 1.862 | 1.863 |

Total ≈ 5.68 ms, matching U12's 5.47 (marker overhead ~0.2). **q_path is the
largest of the three at 1.86 ms, slightly above the ~1.5 ms q_b-alone
estimate** — so q_b is most of it and the per-head norm/RoPE tail is a small
residual. q_a_kv_proj (2.14) and q_lora_norm (1.68) are the two larger
pieces and are the place to look next.

**Arm 1 — standalone inverse-RoPE isolation (norope = gathered branch forced
onto the standalone RoPE; delta prices the dispatch across the 20 gathered
layers):**

| ctx | default | norope | delta |
|---|---|---|---|
| 32k | 3.393 | 3.526 | **+0.133 ms** |
| 131k | 4.252 | 4.764 | **+0.512 ms** |

**The standalone inverse-RoPE dispatch is ~0.13 ms at 32k and ~0.51 ms at
131k** (per the plan's gate, the 131k delta is the relevant number for the
long-context indexed layers). U13's prize is roughly this × 21/20 for the
indexed layers — a real but modest win at 131k, not the 2–4 ms hoped for;
most of `attn_inv_rope` is attention doing its job. Per the plan's gate
(under ~0.3 ms → U13 dead; 0.51 ms at 131k is over it), the U13 call is
marginal — see test plan.

### U12 — price q_path (5.47 ms) and attn_inv_rope (4.27 ms) — stage profile, 2026-08-27

`DS4_METAL_GPU_STAGE_TIMESTAMPS=1` stage-profile run on the rig (mat worker
/ lanfear coord, TP over RDMA, build `b99dfa3`), gen 128, at 32k and 131k.
Per-token stage gpu_ms (sum over 128 decode tokens ÷ 128).

| stage | 32k ms/token | 131k ms/token |
|---|---|---|
| **q_path** | **5.474** | **5.472** |
| **attn_inv_rope** | **3.389** | **4.258** |
| routed_moe_folded | 4.990 | 4.950 |
| compressor_indexer | 4.957 | 9.668 |
| attn_output | 3.746 | 3.804 |
| ffn_hc_post | 1.834 | 1.812 |
| attn_hc_pre | 1.142 | 1.155 |
| ffn_hc_pre | 1.119 | 1.121 |
| router | 1.112 | 1.108 |
| shared_gate_up | 0.985 | 0.965 |
| shared_down | 0.660 | 0.657 |
| attn_hc_post | 0.467 | 0.461 |
| total gpu_busy | 32.70 | 37.99 |

**q_path is context-invariant** (5.474 → 5.472 ms from 32k to 131k) — it is
the fixed per-token query projection cost (q_a/q_b Q8_0 matvecs + head RMS
norm + RoPE tail), 15.0% of the token, and the single largest non-indexer
stage. **attn_inv_rope grows with context** (3.389 → 4.258 ms, +26%) — the
standalone 64-threadgroup inverse-RoPE dispatch on the attention output, paid
per layer. Together they are 9.7–9.7 ms, 26.8% of the token.

Model shape (from `speed-bench/tp_decode_investigation.md`): n_head 64,
n_head_kv 1, n_head_dim 512, n_rot 64, n_lora_q 1024, n_key_mla 256, n_embd
4096, 43 layers (21 ratio-4).

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
