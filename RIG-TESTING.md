# Rig testing — operational guide (engine-agnostic)

This is the *how to actually run things* doc for the two-node Apple Silicon rig.
It is deliberately **not** engine-specific: use it to bench/debug any inference
engine (llama.cpp, ds4, whatever). Engine-specific launch examples live at the
bottom. The workflow it captures: **run prereqs → update code on both machines →
build → launch harness → check logs → document results → push docs.**

Keep this doc concise. If a section grows past ~20 lines, that detail belongs in
a dated notes file, not here.

---

## 1. Topology & access

Two Macs, 2× M2 Ultra, 128 GB RAM each, connected by two networks:

| Role | Hostname | Management net | TB link IP | RDMA device |
|---|---|---|---|---|
| coordinator | `lanfear.local` | via hostname | `192.168.0.6` | `rdma_en6` |
| worker | `mat.local` | via hostname | `192.168.0.5` | `rdma_en7` |

- **Reach the machines by hostname** (`ssh moiraine@lanfear.local`). The
  `192.168.0.x` addresses are the **Thunderbolt RDMA link, NOT management** —
  use them only for high-speed data transfer or RDMA, never for routine shell.
- SSH password is provided via an **ephemeral askpass helper** in each session.
  Never write the password into docs, commits, or persistent files.
- `moiraine` is the user on both hosts. Both are macOS.

**Model / fixture locations** (identical on both hosts unless noted):
```
~/Downloads/<model>.gguf                      # target model, e.g. the 145 GiB DeepSeek-V4
~/Downloads/*-DSpark-support-*.gguf           # support/drafter GGUF (ds4 DSpark only)
~/Downloads/promessi_sposi.txt                # canonical prose prompt
```

---

## 2. Prereqs (run after every reboot)

1. **Wired memory limit** (both hosts):
   ```
   sudo -n sysctl iogpu.wired_limit_mb=120000
   ```
2. **TB RDMA link is runtime-only** — it does not survive reboot. Re-run the
   setup on **both** hosts:
   ```
   ~/Downloads/rdma-tb4/tests/setup-rdma-net.sh
   ```
   Verify with `ifconfig` that `rdma_en6`/`rdma_en7` are up on the expected IPs.
3. Confirm the management net works both directions (`ssh mat.local` from
   lanfear and vice versa) and that SSH keys / askpass are set up.

---

## 3. Code sync & build (per engine)

Repos live at `~/Downloads/<name>/<repo>` on each host. For ds4:
`~/Downloads/ds4r/ds4`.

**The worker (`mat.local`) frequently accumulates divergent local changes.**
Before any round, force both to the branch HEAD:
```
# on BOTH hosts:
git fetch origin
git reset --hard origin/<branch>
git clean -fd
```

Build **on the Mac hosts only** — Linux/this dev box has no Metal. macOS has no
`timeout` command; don't rely on it in scripts.

Typical ds4 build (see §6 for the full command):
```
make -j ds4-bench            # the measurement harness
make -j ds4 ds4-server       # CLI + HTTP server, only if the arm needs them
make -j tests/bench_ngram_accept tests/bench_cpu_draft_cost   # offline harnesses
```

---

## 4. Launching a distributed run

**Standard pattern (ds4 TP2):** a driver script on the coordinator launches the
coordinator with `nohup`, then SSHes to the worker and launches it too. Use
`grep -v "zsh -c"` when filtering `ps aux` (macOS wraps commands).

Two gotchas that cost real time:
- **`nohup` stdout is fully buffered** on macOS — `tail -f` shows nothing until
  the buffer flushes. Write logs to files and `grep` them; don't watch the pipe.
- **The worker needs the same model file (and support GGUF, if used) on its own
  disk.** If a file is only on the coordinator, copy it over the **TB link**
  (fast), not the management net (≈1 MB/s for a 5.9 GB file). E.g.:
  `scp ~/Downloads/<file> moiraine@192.168.0.5:~/Downloads/`

**Pair restart between arms.** Kill coordinator + worker, `sleep ~8`, relaunch
both. Do not reuse a running pair for a different arm.

---

## 5. Gotchas / lessons (read before a session)

- **Reboot loses the TB link and the wired limit** — see §2. The most common
  "worker won't connect" cause is the TB link being down.
- **macOS quirks:** no `timeout`; `sudo -n` works passwordless; `nohup` output
  fully buffered; `ps aux` lines are wrapped in `zsh -c` — always
  `grep -v "zsh -c"`.
- **`powermetrics`**: killing its pid (`kill -9 $ppid`) hits the sudo wrapper,
  not the root process. Use `sudo -n kill <root_pid>` instead.
- **ds4 locates `metal/*.metal` relative to the cwd** — always `cd` into the
  repo before launching.
- **The coordinator/worker model must match** on both hosts (same GGUF path).
- **The production scheduler/backoff can silently skip the thing you're
  measuring** (e.g. ds4 DSpark backs off on prose so the verifier never runs).
  If a measurement comes back exactly at baseline, check for a "backoff"/"skip"
  counter in the stats before trusting it.
- **Interleave repeats** across arms (A,B,A,B,...) rather than running
  sequential blocks — sequential repeats of the same arm drift (56-124 ms
  across repeats on the same config).
- **Small command-buffer / prefill chunks can break TP** on ds4: the per-layer
  gate exchange requires both ranks chunk identically, and sub-512-token chunks
  fail with `TP gate exchange failed`. Don't design a TP arm that needs tiny
  chunks.

---

## 6. Harnesses

### ds4 TP2 (the main rig harness)
`ds4-bench` is the measurement harness; `ds4` is the interactive CLI;
`ds4-server` is the HTTP server. All three support
`--role coordinator/worker --tensor-parallel --transport rdma` (the CLI/server
undocumented args).

Sample ds4 TP2 launch (keep this as the starting point for future ds4 work):

```
# coordinator (lanfear.local), from the repo dir:
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m ~/Downloads/<MODEL>.gguf \
  --role coordinator --tensor-parallel --transport rdma \
  --listen 0.0.0.0 1234 --rdma-device rdma_en6 \
  --prompt-file ~/Downloads/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --gen-tokens 128 \
  --csv /tmp/out.csv

# worker (mat.local), from the repo dir:
DS4_METAL_FAST_SYNC=1 ./ds4-bench -m ~/Downloads/<MODEL>.gguf \
  --role worker --coordinator 192.168.0.6 1234 \
  --transport rdma --tensor-parallel --rdma-device rdma_en7
```

- `DS4_METAL_FAST_SYNC=1` enables the TP decode split (default off).
- CSV columns: `ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,
  gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes`.
- Multi-step prefill: `--step-mul 1 --step-incr <N>` (additive path) or
  `--step-mul 2` (doubling). Sum per-step `prefill_tokens/prefill_tps` for total.
- Useful env flags (all optional): `DS4_METAL_PROFILE_DECODE_STAGE` (stage
  profile), `DS4_METAL_GPU_ENCODER_TIMESTAMPS=1` (+`..._SPLIT=1` for forced
  boundaries), `DS4_METAL_PATH_COUNTS=1`, `DS4_METAL_DISPATCH_BALLAST=N`,
  `DS4_METAL_DISABLE_ARGSORT_CANON=1`, `DS4_NGRAM_TRACE=<file>` (coordinator
  only), `DS4_DSPARK_STATS=1 DS4_DSPARK_VERIFY_PROFILE=1`.

### ds4 offline harnesses (no GPU / dev box)
- `tests/bench_ngram_accept <trace>` — analyze an n-gram token trace
  (`--vt 4.459` for legacy cost model).
- `tests/bench_cpu_draft_cost` — model-free CPU draft-cost probe. Note the
  numbers are machine-specific (M1 Max vs x86 dev box differ ~10×).

### Other engines (llama.cpp, etc.)
Use the same rig workflow: §2 prereqs, §3 sync/build on the Macs, §4
launch-with-driver + pair-restart, then the engine's own bench binary. The
topology, TB link, wired limit, and pair-restart discipline all apply unchanged.

---

## 7. The measurement workflow (how to resume a session)

1. **Run prereqs** (§2) if the boxes rebooted.
2. **Sync + build** both hosts (§3).
3. **Write a driver script** in `/tmp` on the coordinator (§4): launches the
   pair per arm, pair-restarts between arms, interleaves repeats, writes logs +
   CSVs to `/tmp/<arm>_results/`.
4. **Launch, then poll the log files** (not the pipe) for the heartbeat/expected
   output. Verify within the first ~30s that the run is actually measuring what
   you think (check for backoff/skip counters).
5. **Compute the read** (fit, delta, etc.) from the CSV/stats.
6. **Document** results newest-first in the benchmark markdown, and record the
   result + verdict in the plan doc. Commit + push docs (the plan author pushes
   code; you push docs).
7. **Leave the rig idle** (kill coordinator + worker) when done.

Rule of thumb that has saved many sessions: **do not over-analyse before
recording.** Put the numbers in the doc and push; the plan author consults them.
