/* U16 — is the q-LoRA norm stage the kernel, or the serialisation around it?
 *
 * The `q_lora_norm` decode stage costs 1.50 ms/token net of profiler tax --
 * 6.2% of the 2k token -- across 43 layers, so ~35 us per layer.  On Metal the
 * span resolves to ONE dispatch of TWO threadgroups
 * (`ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor`, grid
 * MTLSizeMake(1,2,1) at ds4_metal.m:22402), already a triple fusion of q-norm,
 * kv-norm + RoPE, and the FP8 cache store.  It moves roughly 6 KB.
 *
 * 35 us cannot be explained from first principles for that much work: the tree
 * reduction is ~10 barrier rounds and the reads are one DRAM round trip.  Two
 * hypotheses, and they imply completely different fixes:
 *
 *   A. the kernel really costs ~35 us  -> optimise the kernel (width, reduction
 *      shape, split the q and kv halves across more threadgroups)
 *   B. the kernel costs a few us and the rest is pipeline drain/refill around a
 *      2-of-60-core dispatch that everything downstream depends on
 *      -> fusion into the consumer, or concurrent scheduling; optimising the
 *         kernel itself would buy nothing
 *
 * Guessing between these is how this project has repeatedly burned a day, so
 * measure first.  This harness calls the kernel directly at the production
 * shapes with no model loaded, so its GPU time is the kernel alone.
 *
 * Interpretation:
 *   isolated ~= 35 us  -> hypothesis A
 *   isolated <<  35 us -> hypothesis B, and the gap is the serialisation cost
 *
 * Per `tests/bench_indexer_score.c`'s header, ALWAYS one dispatch per command
 * buffer: batching repeats that write the same buffer serialises them behind
 * write-after-write hazards and inflated that measurement 267x.
 *
 *   make tests/bench_qkv_norm
 *   DS4_METAL_GPU_BUSY_PROFILE=1 ./tests/bench_qkv_norm
 *   DS4_METAL_GPU_BUSY_PROFILE=1 ./tests/bench_qkv_norm 1024 512 400
 */

#define _DARWIN_C_SOURCE

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); return 1; } } while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const uint32_t q_n   = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 1024u;  /* q-LoRA rank */
    const uint32_t kv_n  = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 512u;   /* DS4_N_HEAD_DIM */
    const int      iters = argc > 3 ? atoi(argv[3]) : 400;
    const uint32_t n_rot = 64u;
    const uint32_t layers = 43u;

    CHECK(ds4_gpu_init(), "gpu init");

    /* The kernel reads its RMS weights straight out of the model map, so stand
     * in a synthetic one: two f32 weight vectors at known offsets. */
    const uint64_t q_w_off  = 0;
    const uint64_t kv_w_off = (uint64_t)q_n * sizeof(float);
    /* The kernel reads its RMS weights through the registered Metal model
     * views, not a bare host pointer, so the map has to be mmap'd and handed
     * to ds4_gpu_set_model_map -- a plain malloc trips "model range is not
     * covered by mapped model views". */
    const uint64_t map_size = 1u << 20;   /* page-multiple, room for both vectors */
    float *map = mmap(NULL, (size_t)map_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(map != MAP_FAILED, "map mmap");
    for (uint32_t i = 0; i < q_n; i++)  map[i] = 1.0f + (float)(i % 7) * 0.01f;
    for (uint32_t i = 0; i < kv_n; i++) map[q_n + i] = 1.0f + (float)(i % 5) * 0.02f;
    CHECK(ds4_gpu_set_model_map(map, map_size), "set_model_map");

    /* One dispatch per command buffer measures command-buffer overhead, not the
     * kernel: it came out at 110 us/cb, three times the in-engine figure, for a
     * kernel the engine runs inside a shared buffer.  Batch instead, and rotate
     * outputs across NSETS so repeats do not serialise behind write-after-write
     * hazards -- the failure mode `bench_indexer_score`'s header records as
     * inflating a measurement 267x. */
#define NSETS 8
    ds4_gpu_tensor *q      = ds4_gpu_tensor_alloc((size_t)q_n * sizeof(float));
    ds4_gpu_tensor *kv     = ds4_gpu_tensor_alloc((size_t)kv_n * sizeof(float));
    ds4_gpu_tensor *q_out_s[NSETS], *kv_out_s[NSETS], *raw_s[NSETS];
    ds4_gpu_tensor *q_out, *kv_out;
    /* raw_cache is the KV ring the store writes into.  The host sizes it as
     * raw_cap * kv_n * sizeof(float) (ds4_metal.m, raw_bytes) -- not a packed
     * FP8 row, which is what an earlier version of this harness assumed and
     * why it tripped the undersized-buffer guard. */
    const uint32_t raw_cap = 256u;
    ds4_gpu_tensor *raw;
    for (int i = 0; i < NSETS; i++) {
        q_out_s[i]  = ds4_gpu_tensor_alloc((size_t)q_n * sizeof(float));
        kv_out_s[i] = ds4_gpu_tensor_alloc((size_t)kv_n * sizeof(float));
        raw_s[i]    = ds4_gpu_tensor_alloc((size_t)raw_cap * kv_n * sizeof(float));
        CHECK(q_out_s[i] && kv_out_s[i] && raw_s[i], "set alloc");
    }
    q_out = q_out_s[0]; kv_out = kv_out_s[0]; raw = raw_s[0];
    CHECK(q && kv, "tensor alloc");

    float *hq  = malloc((size_t)q_n * sizeof(float));
    float *hkv = malloc((size_t)kv_n * sizeof(float));
    CHECK(hq && hkv, "host alloc");
    /* Signed, bounded, no underflow: a previous harness produced NaN from an
     * unsigned modulo that wrapped. */
    for (uint32_t i = 0; i < q_n; i++)  hq[i]  = (float)((int)(i % 31) - 15) * 0.03f;
    for (uint32_t i = 0; i < kv_n; i++) hkv[i] = (float)((int)(i % 23) - 11) * 0.05f;
    CHECK(ds4_gpu_tensor_write(q, 0, hq, (size_t)q_n * sizeof(float)), "write q");
    CHECK(ds4_gpu_tensor_write(kv, 0, hkv, (size_t)kv_n * sizeof(float)), "write kv");

    printf("qkv rms norm + kv rope + fp8 store: q_n=%u kv_n=%u n_rot=%u\n",
           q_n, kv_n, n_rot);

    int fired = ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
            q_out, q, map, map_size, q_w_off, q_n,
            kv_out, kv, kv_w_off, kv_n,
            raw, raw_cap, 0u, n_rot, 0u, 0u,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f);
    if (!fired) {
        printf("  kernel unavailable on this device (fp8 fuse gate) -- nothing to measure\n");
        return 0;
    }
    ds4_gpu_synchronize();

    /* One dispatch per command buffer; wall includes submit, so read GPU busy
     * under DS4_METAL_GPU_BUSY_PROFILE=1 for the kernel time itself. */
    const int per_cb = getenv("BENCH_PER_CB") ? atoi(getenv("BENCH_PER_CB")) : 16;
    const double t0 = now_ms();
    int done = 0;
    while (done < iters) {
        const int n = (iters - done) < per_cb ? (iters - done) : per_cb;
        ds4_gpu_begin_commands();
        for (int j = 0; j < n; j++) {
            const int i = done + j;
            (void)ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
                    q_out_s[i % NSETS], q, map, map_size, q_w_off, q_n,
                    kv_out_s[i % NSETS], kv, kv_w_off, kv_n,
                    raw_s[i % NSETS], raw_cap, (uint32_t)(i % raw_cap),
                    n_rot, (uint32_t)i, 0u,
                    10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f);
        }
        ds4_gpu_end_commands();
        done += n;
    }
    ds4_gpu_synchronize();
    const double wall = now_ms() - t0;

    const double us = wall / iters * 1000.0;
    printf("  %d dispatches, %d per command buffer, %d rotated output sets\n", iters, per_cb, NSETS);
    printf("  wall %.2f us/dispatch (includes submit; see gpu busy below)\n", us);
    printf("  x%u layers: %.3f ms/token at this rate\n", layers, us * layers / 1000.0);
    printf("\n  in-engine q_lora_norm is 1.50 ms/token net = %.1f us/layer.\n",
           1500.0 / layers);
    printf("  if gpu busy here is far below that, the stage is serialisation,\n"
           "  not kernel cost -- fix the scheduling, not the kernel.\n");

    munmap(map, (size_t)map_size); free(hq); free(hkv);
    return 0;
}
