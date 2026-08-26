# TP2 prefill: row-split-at-pos0>0 (A0) rig test plan

Branch: `upstream-metal-wins`
Rig: 2× M2 Ultra 60-core 128GB, tensor-parallel over Thunderbolt RDMA
Model: DeepSeek V4 Flash MXFP4, 145.26 GiB (76.73 GiB shard per rank)

## Status (2026-08-25)

This doc is the **request** side of a loop: results go into
`BENCHMARKS-TP-PP.md`, asks go here.

| run | state | outcome |
|---|---|---|
| 1 — inertness / baseline | done | 221.50 t/s @131k prefill, 28.06 decode. Indexer stack +20.8%, LLT decode +10.8% vs old base. Startup hang did not reproduce at `3746eae`. |
| 2 — correctness | done | **Pass.** 128/128 tokens identical, 305/305 argmax identical, max Δlogit 0.049 vs a 0.0055 control-vs-control baseline (~2 f16 ULP). Criterion rewritten below. |
| 3 — A0 throughput | done | +7.2% @131k (221.50 → 237.44), peaking +11.1% at 8k. Real, correct, decode untouched — but **not** the predicted ~1.5×. |
| 3b — cold single point | in progress | |
| 4 — gate + stage profile | **requested below** | |

**Why Run 3 came in low, and it is not noise.** A0 splits the attention
*consumption*, which `top_k` caps at a constant `(512 + 128) × 64 × 512 × 2`
= 4.19e7 MACs/token/layer. It does **not** split the indexer *scoring*, which is
`n_comp × 64 × 128` and grows linearly with context — 2.68e8 MACs/token/layer at
131k, i.e. **6.4× larger**. The measured gain falls exactly as that ratio rises
(8k: 0.4× → +11.1%; 131k: 6.4× → +7.2%), which is the mechanism, not scatter.

Consequence: **A0 structurally cannot fix long-context degradation** — the term
it splits does not grow with context. The term that does is still replicated on
both ranks. Splitting the indexer is the actual fix, and unlike A0 it adds **no
gate traffic**, because score/top-k are per-query-row: each rank scores its own
rows, feeds its own top-k into its own attention rows, and the existing
`attn_out` swap already recombines everything downstream.

| | compute removed /token/layer | gates added | GPU idle |
|---|---|---|---|
| A0 (landed) | 4.19e7 MACs | +23/chunk (43 → 66) | worse (~90% util observed) |
| indexer split (next) | 2.68e8 MACs | **0** | unchanged |

## Open requests

Ordered by value. R1 is the one that unblocks the next change.

### R1 — indexer stage breakdown at 131k (highest value)

Sizes the indexer split before it is built. Everything above is arithmetic; this
is the measurement that replaces it.

```
DS4_METAL_FAST_SYNC=1 DS4_TP_PREFILL_SPLIT_NONZERO=1 \
DS4_METAL_INDEXER_STAGE_PROFILE=1 \
  ./ds4-bench ... --ctx-start 131072 --ctx-max 131072
```

Emits per layer per chunk:
```
ds4: metal indexer stage layer=<il> pos=<pos0> tokens=<n> comp=<n_comp> score=<ms> ms
                                                                        topk=<ms> ms
                                                                   attention=<ms> ms
```

Record the three stage times for **a few even layers ≥ 2** (they carry the
indexer) at a late chunk, plus `comp=`. Paste a representative handful of lines
rather than the whole log.

**Caveat:** the boundary helper calls `ds4_gpu_end_commands()` /
`ds4_gpu_begin_commands()` around each stage, so it forces a command-buffer
split per stage. Throughput under this flag is **not** comparable to a clean
run — only the relative score : topk : attention split is meaningful.

What it decides: if `score` dominates `attention` by the predicted ~6×, the
indexer split is the next change and its ceiling is roughly the `score` share.
If it does not, the model above is wrong and I want to know before building.

### R2 — gate profile, both arms

Turns "~90% GPU util" into a number, and gives the bandwidth figure that decides
whether the RDMA workstream is worth anything.

```
DS4_TP_GATE_PROFILE=1 ...  --ctx-start 131072 --ctx-max 131072
```
once with `DS4_TP_PREFILL_SPLIT_NONZERO=1`, once without.

From the **BIG** line record, for each arm:

| | A0 off | A0 on |
|---|---|---|
| `gates` (expect 43 → 66 per chunk) | | |
| avg **gpu wait** µs — the pipeline bubble | | |
| avg **exchange** µs | | |

Effective per-direction bandwidth = `67,108,864 / avg_exchange_us`.

The gpu-wait delta between arms is the true cost of A0's added gates, and is the
number I got wrong by estimating wire time only.

### R3 — GPU utilisation shape

Was the ~90% seen with A0 **on**, **off**, or both? And is the idle flat across
the sweep or growing with context? Flat implicates gate count; growing
implicates something else. A rough `powermetrics --samplers gpu_power` reading
at 8k and at 131k in each arm is enough.

### R4 — short-context regression (cheap, low priority)

Run 1 showed −2.2% prefill and −2.0% decode at 4096 vs the old base, above the
~1% noise floor. Suspect is #832's canonical argsort comparator: unlike the rest
of the indexer stack it is **not** token-count gated, so it applies where the
sort is a large fraction.

```
DS4_METAL_DISABLE_ARGSORT_CANON=1     # 2048 and 4096 points only
```

If that recovers the 2%, it should be gated on `n_tokens >= 32` like its
siblings and I will patch it.

## What is being measured

Two independent changes landed on this branch. **They must be separated**, or the
A0 result will be contaminated by the upstream prefill stack.

| group | commits | expected effect |
|---|---|---|
| upstream indexer stack | `4760333`, `557ebf4`, `4624e5f`, `1718151`, `ead8786`, `5321d86`, `b7dc56c` | prefill +~24% at 131k, bit-exact |
| LLT decode scorer | `4333606` | decode +6–7% at long ctx only |
| **A0 row split** | `309c0e2`, `534648e`, `82d9e9a` | **prefill, opt-in, the thing under test** |
| server/correctness | `958b248`, `92c4a5a`, `4cc4497`, `471d11b`, `5994f7d`, `d2367a3`, `318e6eb` | no prefill throughput effect |

**The old 183.4 t/s at 131072 is from `tp-multi-slot-batching` and is NOT the A0
baseline.** A0's baseline is this branch with the flag off, which should already
be faster because of the indexer stack.

## Hard prerequisites

1. **Same commit on both hosts.** `git rev-parse HEAD` must match on lanfear and
   mat. Rebuild both; do not reuse a stale binary.
2. **Env symmetry.** `DS4_TP_PREFILL_SPLIT_NONZERO` changes the per-layer gate
   count. An asymmetric setting **deadlocks the gate exchange**, it does not
   degrade. Set it on both ranks or neither. Same for
   `DS4_TP_FORCE_DENSE_ATTN_OUT`.
3. **RDMA device names are host-local and the docs disagree.** `BENCHMARKS-TP-PP.md`
   has lanfear=`rdma_en6`, mat=`rdma_en7`; `speed-bench/README.md` has them
   reversed. Confirm against the actual hardware before the first run — a wrong
   name is a failed QP bring-up, not a silent slowdown.
4. **The sweep is incremental.** `ds4_bench.c:780-795` advances the frontier from
   the previous ctx point, so the `131072` row measures the 65536→131072
   increment (65536 tokens), not a 131k prefill from scratch. Every chunk in that
   row has `pos0 > 0`, which is why it is the cleanest A0 signal in the sweep —
   but it also means the row's mean attended position is ~3N/4, not N/2, so it is
   **not comparable** to any single-node number whose protocol we do not know.
   Use the sweep for A/B against itself, and Run 3b below for anything
   cross-machine.

## Run 0 — build and sanity

On **both** hosts:

```
git fetch && git checkout upstream-metal-wins && git rev-parse HEAD
make -j ds4-bench ds4-server ds4
```

Sanity, short prompt, all new flags unset. Expect normal output and no new
warnings in stderr.

## Run 1 — inertness (do this first, it is the cheapest failure)

Flag unset. Confirms A0 is a true no-op when off, and establishes the A0 baseline.

**lanfear (coordinator):**
```
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role coordinator \
  --tensor-parallel --transport rdma --listen 0.0.0.0 1234 \
  --rdma-device rdma_en6 --prompt-file /tmp/bench_long.txt \
  --ctx-start 2048 --ctx-max 131072 --step-mul 2 \
  --gen-tokens 128 --csv /tmp/a0_baseline.csv
```

**mat (worker):**
```
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m <MODEL> --role worker \
  --coordinator 192.168.0.6 1234 --transport rdma --tensor-parallel \
  --rdma-device rdma_en7
```

Pass criteria: completes with zero faults; prefill at 131072 is **≥ the old
183.4 t/s**. If the indexer stack is working it should be nearer ~228.

## Run 2 — correctness (must pass before any throughput claim)

Bit-equality of split vs unsplit. **`DS4_TP_FORCE_DENSE_ATTN_OUT=1` on both arms
and both ranks** — without it the arms differ by the output-projection kernel as
well as by the split, and the comparison is meaningless.

Use a fixed prompt long enough to cross a chunk boundary (>8k tokens), temp 0.

**Arm A (control):**
```
DS4_METAL_FAST_SYNC=1 DS4_TP_FORCE_DENSE_ATTN_OUT=1 \
  ./ds4-bench ... --gen-tokens 128 > /tmp/a0_ctrl.txt
```

**Arm B (candidate):** identical plus `DS4_TP_PREFILL_SPLIT_NONZERO=1` on both
ranks.

```
cmp -s /tmp/a0_ctrl.txt /tmp/a0_cand.txt && echo IDENTICAL || echo MISMATCH
```

**Superseded criterion (2026-08-25).** The original gate here was
"byte-identical logits". That is unsatisfiable on this rig and always was:
**two identical control runs** (flag off) already differ in all 129,280 dumped
logits, max abs 0.0055, argmax identical. The unsplit TP path is not
bit-deterministic run to run, so cross-run bit equality cannot be a gate for
anything.

The earlier reasoning — "each output row is an independent accumulation over the
same key sequence, so the split cannot change how a row is computed" — is right
about the per-row math and wrong about the surrounding blocking. The split
changes `n_raw` and `raw_start`, so the SWA ring is linearised from a different
offset and the FlashAttention block geometry moves with it (`has_kvpad` from
`n_keys`, `bc_mask` from `n_tokens % 8`). Same keys, same values, different
block boundaries, different rounding.

**Actual pass criteria — all three:**

1. Generated tokens **byte-identical** to the control at temp 0. This is the
   strong one: it means every sampling decision matched.
2. Frontier **argmax identical at every dumped frontier**.
3. `max |Δlogit|` within a few ULP of f16 at the observed logit magnitude, and
   within ~10× the control-vs-control baseline measured in the same session.
   Record both numbers; the baseline is the scale, not zero.

Measured 2026-08-25: control-vs-control 0.0055, split-vs-control 0.049 at logit
magnitude ~27 (f16 ULP there is ~0.026, so ~2 ULP). 128/128 tokens identical,
305/305 argmax identical. **Pass.**

**If criteria 1 or 2 fail, stop.** Criterion 3 alone drifting means investigate
before trusting the throughput numbers, not necessarily abort.

## Run 3 — A0 throughput sweep

`DS4_TP_PREFILL_SPLIT_NONZERO=1` on both ranks, `DS4_TP_FORCE_DENSE_ATTN_OUT`
**unset** (it costs throughput).

```
DS4_METAL_FAST_SYNC=1 DS4_TP_PREFILL_SPLIT_NONZERO=1 \
  ./ds4-bench ... --csv /tmp/a0_split.csv
```

### Results table

| ctx | old branch | Run 1 (flag off) | Run 3 (flag on) | Δ vs Run 1 |
|---|---|---|---|---|
| 2048 | 381.3 | | | |
| 4096 | 353.8 | | | |
| 8192 | 377.1 | | | |
| 16384 | 346.3 | | | |
| 32768 | 310.4 | | | |
| 65536 | 252.6 | | | |
| **131072** | **183.4** | | | |

Expected shape: **~0% at 2048** (chunk 0 only, already split), rising with
context, largest at 131072. A0 does nothing at short context by construction —
if you see a gain at 2048 something else changed.

Target at 131072: **~1.5× over Run 1.** Derivation: the pair executes ~8.9e10
FLOPs/token of which ~45% duplicates the peer; halving the splittable part
predicts 1.51×, and the measured TP2-vs-single-node ratio after discounting the
M2/M3 gap independently gives 1.52.

## Run 3b — cold single-point prefill (do this, it is the honest number)

The sweep rows are incremental suffixes. For a number that means something on
its own — and for any comparison against a single-node figure — run one **cold**
point, flag off then flag on:

```
--ctx-start 131072 --ctx-max 131072      # no --step-mul; one cold prefill
```

This is the number to quote. The sweep tells you the *shape* of the gain across
context; this tells you the *size* of it at the context you care about, without
the 3N/4 mean-position artefact.

| | flag off | flag on | Δ |
|---|---|---|---|
| cold 131072 prefill t/s | | | |

## Run 4 — free diagnostics while the rig is up

Costs one extra run and decides whether the RDMA workstream is worth anything.

```
DS4_TP_GATE_PROFILE=1 DS4_METAL_FAST_SYNC=1 ... --ctx-start 131072 --ctx-max 131072
```

The profile prints per gate kind (ROW / VERIFY / BIG): gate count, average GPU
wait µs, average exchange µs. From the **BIG** line record:

- `gates` — sanity check: should be ~16 chunks × 43 layers = 688 with the flag
  off, roughly double with it on (A0 adds the attn_out row swap per layer)
- `avg exchange us` → **effective per-direction bandwidth =
  67,108,864 / avg_exchange_us** (bytes/µs = MB/s)
- `avg gpu wait us` → the pipeline bubble per gate. This is the number I could
  only estimate statically; if it is large relative to the exchange, the cost is
  the encoder drain and not the wire.

The bandwidth number decides WS3+4 on its own: two of its three proposed gain
arms required bandwidth above TB4 line rate, i.e. were impossible.

Also worth capturing, same run:
```
DS4_METAL_GPU_STAGE_TIMESTAMPS=1 DS4_METAL_GPU_STAGE_TIMESTAMPS_LAYER=<il> ...
```
(`_LAYER` restricts to one layer, `_DETAIL` adds sub-stages). Pick a ratio-4
layer — even `il` ≥ 2 — since those carry the indexer. This gives the per-stage
split of the ~519 ms/layer-chunk: indexer scoring vs attention core vs
projections. That sizes the *next* change — splitting indexer score/top-k, which
needs no cross-rank merge because it is per-query-row, and is the largest
remaining win once A0 lands.

## Failure signatures

| symptom | meaning |
|---|---|
| Gate wait hangs, or `tp: worker sync send failed` | **asymmetric env.** The flag is set on one rank only. |
| `kIOGPUCommandBufferCallbackErrorTimeout` on the worker | watchdog kill; check whether it also happens with the flag off (then it is pre-existing, see #852) |
| Row-range view rejected / prefill aborts | a bounds case the even-chunk guard did not cover — capture `n_tokens`, `pos0`, `il` |
| Run 2 mismatch | real correctness defect in the split |
| Run 3 gain ≈ 0 at all ctx | flag not reaching the predicate — confirm with `DS4_LOG` that split chunks are firing |
| Run 3 gain at 2048 | contamination; something other than A0 changed |

## Abort criteria

- Run 2 mismatch → revert to Run 1 config, report the prompt and ctx.
- Any hang → both ranks must be restarted (`tp->failed` is sticky and never
  cleared; a wedged pair does not recover in-process).
- Worker exits with a GPU command-buffer error → expected behaviour as of
  `03cbf99`; it now exits loudly rather than lingering as a zombie.

## Rollback

Everything under test is opt-in and off by default. Unsetting
`DS4_TP_PREFILL_SPLIT_NONZERO` restores current behaviour with no rebuild. To
drop A0 entirely: `git revert 82d9e9a 534648e 309c0e2`.
