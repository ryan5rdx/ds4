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
  shorter prompts). `/tmp/bench_long.txt` is ~135k tokens; it is lost on reboot
  — regenerate (seeded python word-soup, see conversation 2026-08-25) and copy
  to both hosts.
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
  --rdma-device rdma_en6 --prompt-file /tmp/bench_long.txt \
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
    --dist-rdma-adj-devices rdma_en6 --prompt-file /tmp/bench_long.txt \
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
  below, and fixed by the indexer split (R5): sweep +19.5% @131k, cold
  +14.8% over the A0 arm; +28.1% over the flag-off sweep baseline.

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
