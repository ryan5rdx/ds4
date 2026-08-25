# DS4 Flash — TP/PP Benchmark Reference (2× M2 Ultra 128GB)

Reference notes for distributed inference benchmarks on the Apple Metal pair.
Model: **DeepSeek V4 Flash** (MXFP4 experts, F16 HC/compressor/indexer, Q8
attn/shared/output).

- Model file: `~/Downloads/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf`
- Size: 155,976,458,848 bytes (~145 GiB)
- Layers: 43 (0:42 + output)

## Hosts

| Host | Role | IP | RDMA dev | Metal |
|------|------|-----|----------|-------|
| `moiraine@lanfear.local` | coordinator | 192.168.0.6 | `rdma_en6` | M2 Ultra 128GB |
| `moiraine@mat.local` | worker | 192.168.0.5 | `rdma_en7` | M2 Ultra 128GB |

- Both repos at `~/Downloads/ds4r/ds4`; local (Linux, no Metal) repo at
  `/home/moiraine/Projects/ds4`. **Builds must happen on the Mac hosts.**
- Passwordless ssh + sudo to both.
- Only test with **mat + lanfear**. rand/dashiva are read-only (dashiva tmux for
  server debugging).
- Use `ps aux` (NOT `pgrep -f "ds4-bench --role"`) to check liveness — the model
  path sits between args so `pgrep -f "ds4-bench --role"` is a false negative.

## Branches

- `tp-multi-slot-batching` — current TP base. Includes ESC-cancel live-KV fix
  (`22568ab`: rewind live session to prefill frontier on job cancel).
- `pp-rdma-new` — PP-RDMA rebuilt off `tp-multi-slot-batching` base:
  - `180e853` cherry-pick of PP-RDMA transport (from `apple-rdma-pp`'s `f7d3837`)
  - `ecea7be` PP RDMA connection lifecycle fix (persistent worker chan, borrowed
    by route plans) — eliminated per-generation teardown/reconnect.
- Old `apple-rdma-pp` branch — do not use. Check the real gap with
  `git rev-list --count apple-rdma-pp..tp-multi-slot-batching` rather than
  trusting a number written here; it goes stale on every commit.

## Launch commands

### TP pair (tensor-parallel over RDMA)
- **lanfear (coordinator):**
  ```
  DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role coordinator \
    --tensor-parallel --transport rdma --listen 0.0.0.0 1234 \
    --rdma-device rdma_en6 --prompt-file /tmp/bench_long.txt \
    --ctx-start 2048 --ctx-max 131072 --step-mul 2 \
    --gen-tokens 128 --csv /tmp/tp_sweep_full.csv
  ```
  `--step-mul 2` is what produces the ×2 sweep table below. Without it
  `step_incr` defaults to 2048 (`ds4_bench.c`), i.e. a 64-point linear sweep —
  many hours at 128k rather than the seven points published here.
- **mat (worker):**
  ```
  DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role worker \
    --coordinator 192.168.0.6 1234 --transport rdma --tensor-parallel \
    --rdma-device rdma_en7
  ```

### PP pair (pipeline-parallel over RDMA)
- **lanfear (coordinator / bench):** layers 0:21
  ```
  DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role coordinator \
    --layers 0:21 --listen 0.0.0.0 9000 --dist-transport rdma \
    --dist-rdma-adj-devices rdma_en6 --prompt-file /tmp/bench_long.txt \
    --ctx-start 2048 --ctx-max 131072 --step-mul 2 \
    --gen-tokens 128 --csv /tmp/pp_sweep_full.csv
  ```
- **mat (worker):** layers 22:output — **must use `./ds4`** (serving binary), not
  `ds4-bench` (rejects `--role worker`)
  ```
  DS4_METAL_FAST_SYNC=1 ./ds4 -m <MODEL> --role worker --layers 22:output \
    --coordinator 192.168.0.6 9000 --dist-transport rdma \
    --dist-rdma-adj-devices rdma_en7 --ctx 132096
  ```
- **`--ctx` on the PP worker must be ≥ coordinator's `ctx_alloc`**
  (`ctx_max + gen_tokens + 1`). Worker default is 32768 — with `--ctx-max 131072`
  the coordinator allocates 131201 and rejects the worker with "worker context
  too small" (`DS4_DIST_MSG_ERROR`, type 2) → reconnect loop. Use `--ctx 132096`.

## TP-RDMA sweep — decode (gen t/s) & prefill

Run: ctx 2048→131072, step ×2, 128 gen tokens, 2× M2 Ultra, RDMA.

| ctx | prefill t/s | decode t/s |
|-----|-------------|------------|
| 2048 | 381.3 | 40.87 |
| 4096 | 353.8 | 36.76 |
| 8192 | 377.1 | 36.07 |
| 16384 | 346.3 | 35.04 |
| 32768 | 310.4 | 32.98 |
| 65536 | 252.6 | 29.80 |
| 131072 | 183.4 | 25.33 |

CSV: `/tmp/tp_sweep_full.csv` on lanfear.

## PP-RDMA sweep — decode & prefill

Run: ctx 2048→131072, step ×2, 128 gen tokens, 2× M2 Ultra, RDMA.
(`prefill_tps` is the incremental prefill of the added tokens each frontier.)

| ctx | prefill t/s | decode t/s |
|-----|-------------|------------|
| 2048 | 303.04 | 28.97 |
| 4096 | 295.60 | 27.49 |
| 8192 | 323.72 | 27.11 |
| 16384 | 419.59 | 26.39 |
| 32768 | 462.14 | 24.83 |
| 65536 | 426.59 | 22.84 |
| 131072 | 334.53 | 20.21 |

CSV: `/tmp/pp_sweep_full.csv` on lanfear.

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

Caveat before reproducing: these two readings predate the sweep table above and
are not from it — the documented sweep stops at 131072, so the ~180k point is
not reachable by re-running it. The measurement command, per-node vs pair basis,
and host were not recorded. Re-measure with
`sudo powermetrics --samplers gpu_power` on a named host and record the context
of each reading before treating the delta as a quantitative result.

Interpretation: lower power = GPU under-utilization / more waiting as context
grows. See investigation notes in `docs/TP-PREFILL-LONG-CTX-INVESTIGATION.md`
(if present) / the conversation for root-cause analysis and improvement options.

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
  whole attention path replicated on both ranks for the other 31 chunks. Do not
  read the decode fact as a prefill fact; it changes which fix is worth doing.
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

- `DS4_METAL_PREFILL_CHUNK` — prefill chunk size (tokens).
- `DS4_METAL_GRAPH_RAW_CAP` — raw KV cache cap override (cap 8192).
- `DS4_TP_PREFILL_SPLIT_MIN` — min tokens to row-split replicated shared expert
  in TP prefill (default 32).
- `DS4_TP_SUBGATE_PIPELINE` — opt-in sub-chunk gate pipelining for TP prefill
  row swaps; **default off (measured net-negative on M5 Max pair)**. Must be set
  on BOTH ranks.
- `DS4_TP_SPEC_F_ATTN_OUT_SPLIT` — TP verifier attention output split path.
- `DS4_METAL_ABLATE_INDEXER_SCORE` / `_TOPK` — ablate indexer stages (diagnostic,
  changes output).
