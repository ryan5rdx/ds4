<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

> ## Fork: tuned for a two-node Apple Silicon pair
>
> This branch is a fork of [antirez/ds4](https://github.com/antirez/ds4) tuned
> for **DeepSeek V4 Flash across two Mac Studios over Thunderbolt RDMA**, in
> tensor-parallel mode. Reference rig: 2× M2 Ultra, 60 GPU cores, 128 GB each.
> Everything here also runs unchanged on a single machine and on the CUDA and
> ROCm backends; the work is additive.
>
> ### Where it stands
>
> Full sweep against pipeline-parallel at the same commit, 128 generated tokens
> per point. Prefill and decode are tokens/second.
>
> | ctx | TP prefill | PP prefill | TP decode | PP decode |
> |---:|---:|---:|---:|---:|
> | 2048 | **423.0** | 317.0 | **40.9** | 27.7 |
> | 4096 | **417.5** | 310.5 | **36.5** | 26.3 |
> | 8192 | **500.2** | 348.2 | **36.0** | 26.1 |
> | 16384 | **480.8** | 459.4 | **35.4** | 25.5 |
> | 32768 | 461.2 | **530.0** | **33.7** | 24.1 |
> | 65536 | 427.1 | **520.4** | **31.5** | 22.5 |
> | 131072 | 367.3 | **444.4** | **28.1** | 20.6 |
>
> **GLM 5.3 Flash Q4_K** full TP2 sweep, same commit, 128 greedy tokens per
> point, two M2 Ultra 128 GB Macs (lanfear coordinator + mat worker, RDMA,
> 50/50 expert split). Prefill and decode are tokens/second.
>
> | ctx | TP prefill | TP decode |
> |---:|---:|---:|
> | 2048 | 248.09 | 18.37 |
> | 4096 | 255.74 | 18.08 |
> | 8192 | 263.16 | 17.99 |
> | 16384 | 260.98 | 17.87 |
> | 32768 | 256.56 | 17.70 |
> | 65536 | 248.33 | 17.37 |
> | 131072 | 233.55 | 16.70 |
>
> Cold 131k prefill under tensor parallelism: **402.6 t/s**. Prefill at 131k is
> **+65.8%** over the same branch with the splits disabled, and **+100%** over
> the pre-fork baseline. Tensor parallelism wins prefill to 16k and decode at
> every context; pipeline parallelism wins prefill from 32k up, so both are
> worth keeping.
>
> ### What this fork changes
>
> **Prefill — tensor-parallel row splitting.** A prefill chunk's rows are
> divided across the pair instead of replicated:
>
> * split attention rows at non-zero positions, not just the first chunk of a
>   prompt — **+7.2% at 131k**
> * split the indexer score and top-k alongside the rows each rank already owns,
>   adding no cross-rank merge — **+19.5%, bit-identical**
> * extend the split to chunks carrying a compressed-key mask — **+20.5%,
>   bit-identical**
> * four simdgroups rather than eight for the non-vec dk512 flash-attention
>   kernel — **+7.3%, bit-identical**
>
> **Prefill — kernels.**
>
> * exact single-pass streaming top-512 for wide score rows, replacing a block
>   argsort and merge cascade
> * register-resident indexer scorer: the staged K tile stays in simdgroup
>   registers across the head loop, halving threadgroup loads per matrix multiply
> * chunk size bounded by a work budget so one prefill command buffer stays clear
>   of the GPU watchdog, which under tensor parallelism is not survivable
>
> **Decode.** Honest summary: the individually measured decode changes are at or
> below the measurement floor at short context, and the gains that do exist are
> long-context. Decode is latency-bound here — it runs at maximum clock and full
> residency at roughly 8% of peak arithmetic throughput.
>
> * shared-memory alias in the decode indexer scorer, lifting residency from one
>   threadgroup per core to two — **+17.4% on the kernel**, worth little at short
>   context because the indexed path is unreachable below ~4k
> * four rows per thread in the routed MXFP4 down projection
> * split-K reduce sized to the work groups that actually hold keys
> * five decode fast paths admitted on earlier Apple silicon, where the gate was
>   a device-family check rather than a capability check
>
> **Serving.**
>
> * a chat turn ending in a tool call, truncation or cancellation rewinds to the
>   committed prefill frontier instead of discarding the live KV cache and
>   re-prefilling from token zero. GLM-5.3 gets there by a different route: its
>   KDA recurrence and DSA indexer pool cannot be truncated, so a snapshot of
>   both is taken at each sync frontier and restored when a rewind lands on it
> * cache-miss diagnostics that say how many tokens were lost, where the
>   divergence began, and the token ids on each side
> * slot routing by prefix match, and prefill issued in bounded quanta
> * every frontend can lead a tensor-parallel pair, with ordered teardown and a
>   worker that reports recoverable failures instead of stalling
>
> **Correctness.** Each default-on kernel change has a model-free gate that runs
> on any Metal device: `make test-topk-stream512`, `make test-indexer-scorer`,
> `make test-mxfp4-metal`, and the prefill chunk ladder inside
> `tests/test_engine_mgpu_placement.c`.
>
> ### Credit
>
> Several of the kernels this fork builds on came from open pull requests
> against [antirez/ds4](https://github.com/antirez/ds4) rather than from work
> done here. They are carried, extended and measured on the two-node pair, but
> the kernels are theirs:
>
> * **PR #831** and **PR #832**, Adrian Galilea — the register-resident prefill
>   indexer scorer, and the streaming top-512 selector with its tie-break
>   comparator. Between them these are the largest prefill kernel wins on this
>   branch.
> * **PR #778**, david — the M5 decode encoder and occupancy work, which this
>   fork extends to earlier Apple silicon by replacing device-family gates with
>   capability checks.
> * **PR #846**, Tiziano Arena — M1-class decode tuning and the batch-verifier
>   groundwork the tensor-parallel speculative path is built on. The n-gram
>   speculation from that PR was measured on this rig and removed; the rest is
>   carried.
>
> The tensor-parallel row splitting, the RDMA transport and fence, the prefill
> watchdog ladder, and the serving-side live-KV work are this fork's.
>
> ### Documentation
>
> * [`fork/docs/benchmarks.md`](fork/docs/benchmarks.md) — the measurement
>   record, including the closed avenues, so they are not re-investigated
>
> Everything below is the upstream README.

---

**DwarfStar** is a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model).
It also supports **GLM 5.2 and 5.3**, **GLM 5.3 Flash**, and
**DeepSeek V4 PRO**. It is self-contained and
deliberately narrow, not a general GGUF runner. Model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

Supported backends:

* **Metal**, the primary target, on Macs with 96 GB or more. Smaller machines
  can use SSD streaming.
* **NVIDIA CUDA**, including multi-GPU systems and DGX Spark.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

Model support is intentionally opportunistic. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 512 GB
workstations. A model may be removed when a better replacement arrives.

The project has first class support for SSD streaming of weights, so it is
possible to run models bigger than RAM while often still getting decent
performances, and even running very large models (like the full GLM 5.3 or DeepSeek v4 PRO)
on systems with just 128GB of RAM at a slower speed, but fast enough for
QA-style chats.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from GPT 5.5, 5.6, Claude Fable** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash adds vision to the same text model.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Speculative decoding is opt-in. GLM uses `--mtp`; Flash DSpark needs a matching
support GGUF. It can improve generation, but not every workload benefits.
Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for setup and the
difference between default opportunistic sampling and `--mtp-exact-sampling`.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek, `--power N` trades throughput for lower sustained GPU load.
The default is 100. GLM currently requires `--power 100`.

DeepSeek Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.
### Full DeepSeek V4 PRO Q4 on two Mac Studios

The full-size PRO Q4 GGUF can be run across two 512 GB Mac Studio M3 Ultra
machines by giving the coordinator layers `0:30` and the worker
`31:output`. Use the split GGUF files so each side maps only the tensors it
needs:

```sh
# Coordinator machine.
./download_model.sh pro-q4-layers00-30

# Worker machine.
./download_model.sh pro-q4-layers31-output
```

The two files are:

```text
gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf
gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf
```

This is a capacity use case: each process maps only its own half of the model,
while the worker owns the output head and returns logits.

The current PRO Q4 Metal path uses queue-resident exact expert tables for the
large routed experts. This avoids the broad multi-GiB routed-tensor bindings
that made early distributed PRO Q4 attempts either run very slowly or hit Metal
memory accounting limits. In a short greedy smoke test over the direct
`192.168.0.182` / `192.168.0.183` link, the model generated coherent text and
measured 11.47 t/s generation after startup. Per-token telemetry was balanced:
local layers were around 39-43 ms, remote layers around 44-49 ms, for total
token times around 84-92 ms. Expect a slow startup while each side maps and
makes its half of the model resident. Long-context PRO Q4 prefill and decode
performance still needs separate benchmarking.

The measurements above use a Thunderbolt 5 cable. The implementation is plain
TCP and also works over slower links, including WiFi, but fast Ethernet or
Thunderbolt networking is strongly recommended. Slow links mostly hurt
generation latency and short prefills; large prefills can still benefit when
the layer split is balanced. In the normal performance path, the last worker
owns the output head and returns logits directly.

Minimal two-host configuration:

```sh
# Machine A: coordinator, owns tokenization, sampling, the prompt, and layers 0..30.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf \
  --role coordinator \
  --layers 0:30 \
  --listen 169.254.43.68 1234

# Machine B: worker, connects to A and owns layers 31..output.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf \
  --role worker \
  --layers 31:output \
  --coordinator 169.254.43.68 1234
```

Normally the final worker should own the output head too, for example
`--layers 20:output`. This avoids returning a full final hidden-state batch
after prefill and lets the final worker produce the logits directly. On very
slow or metered links, `--layers 20:42` is also supported: the coordinator will
load the output head and compute logits locally, trading extra coordinator work
for smaller per-token replies.

### Network Link Comparison

The table below shows the same two M5 Max hosts, the same 91 GB Flash quant,
coordinator `--layers 0:19`, worker `--layers 20:output`, an 8192-token prompt
from `speed-bench/promessi_sposi.txt`, and 128 generated tokens. WiFi and
Internet numbers vary with local conditions, but the shape is the important
part: high latency hurts generation directly, while lower bandwidth also pulls
down long-prefill speed.

| Link | Addresses | Ping avg | Prefill | Generation |
| --- | --- | ---: | ---: | ---: |
| Thunderbolt 5 | `169.254.43.68` -> `169.254.12.245` | 0.45 ms | 582.99 t/s | 25.09 t/s |
| WiFi | `192.168.1.57` -> `192.168.1.95` | 77.20 ms | 250.70 t/s | 10.70 t/s |
| Internet / VPN | `10.77.0.4` -> `10.77.0.3` | 152.10 ms | 114.88 t/s | 3.63 t/s |

The Internet/VPN case is not meant to be a good interactive experience. It is
still useful for collective testing: multiple people can temporarily combine
machines to run a larger model that would not fit on any single host, accepting
slow decode in exchange for being able to inspect the model at all.

Use the coordinator exactly like normal `./ds4`: interactive chat, `/read`,
and ordinary generation go through the same high-level session API. The same
distributed options are also wired into `ds4-agent`, `ds4-eval`, and
`ds4-bench`. For benchmarks, workers should already be running; `ds4-bench`
waits until a complete route is available.

Useful tuning and diagnostics:

```sh
./ds4-bench \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 \
  --ctx-max 65536 \
  --step-incr 32768 \
  --gen-tokens 0 \
  --role coordinator \
  --layers 0:19 \
  --listen 169.254.43.68 1234 \
  --debug
```

`--debug` on the coordinator prints route formation and per-hop telemetry:
layer range, token span, local evaluation time, downstream wait time, socket
send time, and input/output byte counts. This is the current profiling tool for
deciding whether a split is balanced. `--dist-prefill-window N` controls how
many prefill chunks may be in flight end-to-end; the default is conservative
and bounded. `--dist-prefill-chunk N` exists for experiments, but the default
4096-token chunk is the canonical setting and should be used unless you are
explicitly validating a different chunk size.

By default DwarfStar sends hidden-state activations as 32-bit floats. To reduce
traffic, pass `--dist-activation-bits 16` or `--dist-activation-bits 8` on the
coordinator. This changes only the transport format between machines, not the
model weights or KV cache. 16-bit transport halves activation traffic and is the
first option to try on Ethernet or WiFi. 8-bit transport is more aggressive and
should be treated as an approximate/experimental mode unless you have validated
the output for your use case. However experimentally reduction activation
size didn't provide a significant improvement, so this option may be removed
in the future.

**If a worker disconnects, the coordinator removes that worker from the active
route**. The request already in flight can fail, and later calls report an
incomplete route until a compatible worker reconnects and sends a new
registration. For live sessions, the coordinator keeps the token history and can
rebuild worker KV state by replaying the prefix when the route is available
again. Workers also validate a rolling 64-bit token-prefix hash on every work
item, so a restarted worker at position 0 cannot silently accept work for
position N; it reports the mismatch and the coordinator replays the current
transcript. Ctrl+C in the CLI and agent is cooperative: DwarfStar waits for the
current distributed token or prefill chunk to drain before returning control,
which avoids coordinator-caused KV splits. Saved agent/server sessions use the
same KV file format as single-machine sessions: during save the coordinator
fetches worker-owned layer tensors and serializes one normal payload; during
load it splits that payload over the currently registered route.

### Distributed protocol overview

At the protocol level there are two kinds of connections. Workers keep a
control TCP connection open to the coordinator and send a `HELLO` with their
model ID, model family, quant profile, layer slice, context capacity, and data
port. The coordinator uses these registrations to build a route that covers all
layers. Work then moves over low-latency TCP data connections: the coordinator
computes the first slice, sends a `WORK` frame with session ID, token positions,
rolling token-prefix hashes before and after the span, route information, and
hidden-state payload, and each worker computes its slice. Middle workers can
forward directly to the next worker. The final worker returns logits to the
coordinator, or ACKs for non-final prefill chunks so the prefill pipeline can
stay full. `RESULT` frames echo the request ID and the post-span hash. A worker
status error is handled differently from a socket failure: KV/hash mismatch can
be recovered by replaying the token history on the same route, while transport
failure drops the route and waits for a replacement worker. For persistent KV,
the coordinator opens worker data connections and sends snapshot save/load
messages for each worker-owned layer range; the disk payload remains a single
agent/server cache file. The protocol has no
encryption or authentication, and is not release-stable yet; coordinator and
workers should be built from the same commit and used on trusted machines and
trusted networks.

## Tensor Parallelism over RDMA

Tensor parallelism runs a single decode across two Macs connected with a
Thunderbolt 5 cable, splitting the heavy per-layer work between the two
GPUs and exchanging 16-24KB partial sums at synchronization gates inside the
graph (RDMA over Thunderbolt when available, a dedicated TCP socket
otherwise). Unlike the pipelined distributed mode above, both
machines work on the *same token at the same time*, so it reduces
per-token latency instead of just fitting a bigger model.

Each machine keeps one contiguous half of the routed experts resident. Dense,
attention, shared-expert, embedding, and output weights remain replicated.
This lets a model whose routed experts do not fit on one machine run fully
resident across the pair; routed kernels never touch the peer's expert half.

### Running GLM 5.2 or GLM 5.3 across two 128 GB MacBooks

One-time setup per boot, on **both** machines:

```sh
# Let the GPU wire ~117 GB (default cap is ~75% of RAM; the resident
# expert shard needs ~97.5 GiB plus KV/scratch).
sudo sysctl iogpu.wired_limit_mb=120000

# RDMA over Thunderbolt needs an IPv4 address directly on the cabled
# member interface (the bridge IP does not count). Use the interface
# that is 'active' in ifconfig, e.g. en1 on one side and en6 on the
# other. Skip this if you are fine with the TCP fallback.
sudo ifconfig en1 inet 10.99.0.2/30 alias     # machine A
sudo ifconfig en6 inet 10.99.0.1/30 alias     # machine B
```

Check the verbs device before loading the model:

```sh
rdma_ctl status
ibv_devinfo -v
```

The device must be active and expose the IPv4-mapped GID for the address above,
for example `::ffff:10.99.0.2`. A working IP ping does not prove that RDMA is
active.

Both machines need the same tree, commit, and GGUF path. Tensor parallelism is
always a 50/50 split with one worker, so do not pass `--layers`. Start the worker
first; it retries while the coordinator loads. The worker must dial the address
on the Thunderbolt member interface, not the bridge address:

```sh
MODEL=gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
# For GLM 5.3 Q4, use MODEL=gguf/GLM-5.3-Flash-Q4_K.gguf instead.

# Machine B: worker.
./ds4 -m "$MODEL" --tensor-parallel --role worker \
  --coordinator 10.99.0.2 9911 --transport rdma

# Machine A: coordinator.
./ds4 -m "$MODEL" --tensor-parallel --role coordinator \
  --listen 10.99.0.2 9911 --transport rdma -c 8192 \
  -p "Tell me something about the sea."
```

The active verbs device and IPv4-mapped GID are selected automatically. If that
is ambiguous, add `--rdma-device rdma_en6 --rdma-gid-index 1` on the worker and
the matching `rdma_en1` flags on the coordinator. Use `--transport tcp` on both
sides to force TCP. Run workers with `ds4`; the coordinator may be `ds4`,
`ds4-agent`, or `ds4-server`.

Startup takes about 9 seconds per machine: each rank pre-faults its
~100 GiB shard from SSD and pins it through a Metal residency set.
DeepSeek V4 Flash works the same way with its own GGUF on both machines.
DeepSeek gate vectors are 16 KB and ride as one RDMA message. GLM 5.2's
6144-wide 24 KB vectors are split into two ordered RDMA messages. GLM 5.3 uses
its own KDA/DSA gate schedule, exchanged and checked during TP startup.

Measured on two M5 Max 128 GB MacBooks (GLM 5.2, IQ2_XXS, 188 GiB):

| | two Macs, tensor parallel | one Mac, SSD streaming |
|---|---|---|
| decode | ~16.8 t/s (15.4 at 4k context) | ~4.8 t/s |
| prefill (4096 tokens) | ~94 t/s | ~3-5 t/s |
| residency | fully memory-resident | streams experts from SSD |

**GLM 5.3 Flash Q4_K full sweep** measured on two 2x M2 Ultra 128 GB Macs
(this rig: lanfear coordinator + mat worker, TP2 over RDMA, 50/50 expert
split), 128 greedy tokens per frontier:

| ctx | prefill t/s | decode t/s |
|---:|---:|---:|
| 2048 | 248.1 | 18.4 |
| 4096 | 255.7 | 18.1 |
| 8192 | 263.2 | 18.0 |
| 16384 | 261.0 | 17.9 |
| 32768 | 256.6 | 17.7 |
| 65536 | 248.3 | 17.4 |
| 131072 | 233.6 | 16.7 |

Notes: the coordinator mirrors every prompt sync and eval to the worker, so
both KV caches stay in lockstep; prompt processing splits both the
routed-expert GEMMs (by expert ownership) and the attention heads (a
contiguous half per machine) with one bulk partial-sum exchange per
layer per stage (`--tensor-parallel-token-prefill` selects a slower
token-by-token prefill that exactly matches the single-machine arithmetic).
The split graph is deterministic, but its changed floating-point reduction
order is not generally byte-identical to single-machine execution.

## Tensor Parallelism across CUDA GPUs

On a single CUDA server, `--cuda-tensor-parallel` splits DeepSeek V4 Flash
tensor and routed-expert work across an even number of GPUs. This is separate
from the Mac-to-Mac mode above: it does not use `--role`, RDMA, or the
distributed layer pipeline. GPU placement and memory budgets are selected with
the normal `--gpu-devices` and `--gpu-vram` options.

The device order is significant. With `N` devices, the first `N/2` logical
tiers are contiguous layer-pipeline homes and the second `N/2` tiers are their
tensor-parallel partners. Specify all homes first and then all partners, with
the closest P2P pair at matching positions. For example, the tested L40S host
uses physical pairs `(0,1)`, `(2,3)`, `(4,5)`, and `(6,7)`, expressed as
`0,2,4,6,1,3,5,7`. Each pair stores a 50/50 split of the routed experts, and
the vocabulary head is row-sharded across the participating output tiers.
Those large tensors are not duplicated. Dense attention, router, and shared
expert weights are replicated within each pair.

For maximum throughput on eight 48 GB L40S cards, use the imatrix Q4 model.
Its routed `Q4_K` layout has the native grouped multi-session kernels; the Q2
model is the lower-memory choice (including tested four-card runs), but its
unsupported grouped routed shapes use the exact fallback and have lower
aggregate serving throughput. Download and build the L40S target with:

```sh
./download_model.sh ds4f-q4
make cuda CUDA_ARCH=sm_89
```

This is the interactive-agent setup used on the eight-L40S server:

```sh
MODEL=gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf

./ds4-agent --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000
```

For serving, keep multiple KV sessions resident so decode rows can be grouped
across requests. The tested host is configured for up to 16 resident sessions:

```sh
./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000 \
  --batched-session 16 \
  --host 0.0.0.0
```

The equivalent local launchers are `./run-nvidia-tp-agent.sh` and
`./run-nvidia-tp-server.sh`. The server launcher also enables the on-disk KV
cache and defaults to the native 0731 MXFP4 GGUF. Set `DS4_MODEL` to use the Q4
file above instead. Reduce the session count or context size if the requested
resident KV caches do not fit after model loading. CUDA TP, half-resident expert
ownership, output sharding, pipelined prefill, and compatible grouped decode are
selected by `--cuda-tensor-parallel`; no `DS4_CUDA_*` environment tuning is required.
Without an explicit `--prefill-chunk`, this mode uses 2048-token chunks so the
tested 16-session, 100k-context layout retains enough VRAM for resident KV
caches. An explicit `--prefill-chunk` remains an override for other topologies.

Any even card count that can hold the selected model and graph scratch is a
valid topology. On this class of 48 GB card, the useful measured endpoints are
Q2 on four cards (two pipeline stages) and Q4 on eight cards (four stages).
For a four-card PIX-paired subset such as physical GPUs `0,1,4,5`, the ordered
list is `0,4,1,5`. Two cards do not have enough memory for these Flash models.

This mode currently requires DeepSeek V4 Flash and an even multi-GPU
placement. GLM 5.2 instead uses normal layer placement across the selected
CUDA devices. DGX Spark is a single-GPU target and must not be started with
`--cuda-tensor-parallel`.

## Reducing heat, power usage and fan noise

Long local inference runs can keep the GPU busy for extended periods. If you
care more about heat, fan noise, battery life on MacBooks, or reducing thermal
stress on the hardware than about maximum throughput, use `--power N`.

`--power 100` is the default and means full speed. Lower values ask DwarfStar to target
that percentage of GPU usage: `--power 70` targets about 70%, `--power 50`
targets about half usage, and so forth. DwarfStar does this by measuring GPU work time
and inserting small sleeps between work units: during prefill it sleeps between
layers, and during generation it sleeps between decoded tokens. This reduces
sustained load without changing model output.

The option is available on the CLI, server, agent, eval, and benchmark tools
for DeepSeek models. GLM 5.2 currently accepts only `--power 100`. For example:

```sh
./ds4 --power 50
./ds4-agent --power 70
./ds4-server --power 40 --ctx 100000
```

## Native agent

DwarfStar features a native coding agent that works in a different way
than most other systems: the inference is controlled from within the agent
itself, without socket/API boundaries, so the session is represented
by the on-disk KV cache itself. Moreover the tools and the system prompt
are all designed vertically for DeepSeek v4 Flash and PRO. This provides a
few advantages:

* Low latency experience, bounded mainly by the prefill speed limits. Displaying of generated text, tool calling, start of a new session are always instantaneous.
* Live progress bar during prefill time.
* No DSML tool calling conversion, the tools are handled natively in the LLM format.
* KV cache mismatch are impossible by construction, the current state is always the truth.
* Everything is tuned for this model.
* Ability to switch saved sessions with `/list` and `/switch`; full KV sessions resume without a prefill stage.

Agent sessions are stored in `~/.ds4/kvcache`. Use `/save` to persist the
current session, `/list` to show saved sessions sorted by recent update time,
and `/switch <sha>` to resume one of them. The session ID is stable across
future saves and is derived from the first user prompt and creation time.
`/del <sha>` removes a saved session. `/strip <sha>` keeps the rendered
conversation text and title but removes the heavy KV payload; switching to a
stripped session rebuilds the KV cache by prefilling the saved text.

Use `--chdir /path/to/ds4` when launching `ds4-agent` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

However while the system already works, there is a lot of work to do
in order to make it ready for prime time. When finally the agent will reach
the wanted shape, we will *likely* split the server and the client creating a stateful
session-based protocol that can recreate all that in a client-server way.

## Benchmarking

`ds4-bench` measures instantaneous prefill and generation throughput at context
frontiers instead of reporting one whole-run average. It loads the model once,
walks a fixed token sequence to frontiers such as 2048, 4096, 6144, and uses
incremental prefill so each row measures only the newly-added token interval.
After each frontier it saves the live KV state to memory, generates a fixed
greedy non-EOS probe, restores the memory snapshot, and continues prefill.

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

The example file is a cleaned public-domain Project Gutenberg text of
Alessandro Manzoni's *I Promessi Sposi* (ebook #45334), with the Gutenberg
header and footer removed: <https://www.gutenberg.org/ebooks/45334>.

Use `--step-incr N` for different linear spacing, or `--step-mul F` for
exponential sweeps. Output is CSV with one row per frontier: latest prefill
interval tokens/sec, generation tokens/sec at that frontier, and
`kvcache_bytes`.

Sessions prefill long prompts in 4096-token chunks by default. Use
`--prefill-chunk 2048`, for example, to match the strict official-vector
checkpoint path. Changing the chunk changes the KV checkpoint/logit path, so
compare it as an explicit run configuration.
Chunked Metal prefill reuses the same range-capable layer-major graph for each
chunk, preserving absolute compressor/indexer boundaries while avoiding the old
per-layer chunk dispatch path.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, and matching encoders.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
