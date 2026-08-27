#define _DARWIN_C_SOURCE

/* Standalone harness for the DS4-Flash routed-MoE decode matvec pair
 * (kernel_mul_mv_id_mxfp4_pair_swiglu_f32 + kernel_mul_mv_id_mxfp4_sum6_f32).
 *
 * Purpose: get an *achieved bytes/second* number for the routed MXFP4 matvec
 * in isolation, so that claims of the form "the MoE matvec runs at N GB/s"
 * can be checked rather than assumed.
 *
 * Follows tests/bench_indexer_score.c: links only ds4_metal.o, one dispatch
 * per command buffer, reads GPU busy time rather than wall clock.
 *
 * The "model" is an anonymous mmap filled with valid MXFP4 blocks; nothing
 * here needs a real GGUF.  Numerics are irrelevant, only traffic is.
 *
 *   cc -O3 -I. -c tests/bench_moe_mxfp4_decode.c && \
 *   cc -O3 -o tests/bench_moe_mxfp4_decode bench_moe_mxfp4_decode.o ds4_metal.o \
 *      -lm -pthread -framework Foundation -framework Metal
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
void ds4_log_line(int type, const char *fmt, ...) { (void)type; (void)fmt; }

#define N_EMBD      4096u
#define N_FF_EXP    2048u
#define N_USED      6u
#define QK_MXFP4    32u
#define BLK_BYTES   17u

/* must match ds4_gpu.h's tensor type enum for MXFP4 */
#ifndef DS4_METAL_TENSOR_MXFP4
#define DS4_METAL_TENSOR_MXFP4 39u
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const uint32_t n_total_expert = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 128u;
    const int iters = argc > 2 ? atoi(argv[2]) : 128;
    const uint32_t n_sel = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : N_USED;

    if (n_total_expert < N_USED || iters <= 0 || n_sel == 0 || n_sel > N_USED) {
        fprintf(stderr, "usage: %s [n_total_expert] [iters] [n_distinct_sel<=6]\n", argv[0]);
        return 2;
    }

    if (!ds4_gpu_init()) { fprintf(stderr, "bench: Metal init failed\n"); return 1; }

    const uint64_t gate_row_bytes = (uint64_t)(N_EMBD / QK_MXFP4) * BLK_BYTES;   /* 2176 */
    const uint64_t down_row_bytes = (uint64_t)(N_FF_EXP / QK_MXFP4) * BLK_BYTES; /* 1088 */
    const uint64_t gate_expert_bytes = gate_row_bytes * N_FF_EXP;                /* 4456448 */
    const uint64_t down_expert_bytes = down_row_bytes * N_EMBD;                  /* 4456448 */

    const uint64_t seg = gate_expert_bytes * n_total_expert;
    const uint64_t model_size = seg * 3ull;
    printf("bench: %u experts, model map %.2f GiB, per-expert %.3f MB (gate+up+down)\n",
           n_total_expert, (double)model_size / (1024.0 * 1024.0 * 1024.0),
           (double)(gate_expert_bytes * 2 + down_expert_bytes) / 1e6);

    void *map = mmap(NULL, (size_t)model_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }

    /* Valid MXFP4: e = 127 (scale 1.0), nibbles arbitrary. */
    {
        uint8_t *p = map;
        uint32_t rng = 0x12345678u;
        const uint64_t nblk = model_size / BLK_BYTES;
        for (uint64_t b = 0; b < nblk; b++) {
            uint8_t *blk = p + b * BLK_BYTES;
            blk[0] = 127;
            for (int i = 1; i < 17; i++) {
                rng = rng * 1664525u + 1013904223u;
                blk[i] = (uint8_t)(rng >> 24);
            }
        }
    }

    if (!ds4_gpu_set_model_map(map, model_size)) {
        fprintf(stderr, "bench: set_model_map failed\n");
        return 1;
    }

    /* Rotate over several output/scratch sets so back-to-back dispatches do
     * not serialize on write-after-write on the same buffers. */
#define NSETS 4
    ds4_gpu_tensor *out[NSETS], *g_t[NSETS], *u_t[NSETS], *m_t[NSETS], *d_t[NSETS];
    for (int s = 0; s < NSETS; s++) {
        out[s] = ds4_gpu_tensor_alloc((uint64_t)N_EMBD * sizeof(float));
        g_t[s] = ds4_gpu_tensor_alloc((uint64_t)N_USED * N_FF_EXP * sizeof(float));
        u_t[s] = ds4_gpu_tensor_alloc((uint64_t)N_USED * N_FF_EXP * sizeof(float));
        m_t[s] = ds4_gpu_tensor_alloc((uint64_t)N_USED * N_FF_EXP * sizeof(float));
        d_t[s] = ds4_gpu_tensor_alloc((uint64_t)N_USED * N_EMBD * sizeof(float));
        if (!out[s] || !g_t[s] || !u_t[s] || !m_t[s] || !d_t[s]) {
            fprintf(stderr, "bench: tensor alloc failed\n"); return 1;
        }
    }
    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc((uint64_t)N_EMBD * sizeof(float));
    if (!x_t) { fprintf(stderr, "bench: x alloc failed\n"); return 1; }
    {
        float *x = ds4_gpu_tensor_contents(x_t);
        uint32_t rng = 0xdeadbeefu;
        for (uint32_t i = 0; i < N_EMBD; i++) {
            rng = rng * 1664525u + 1013904223u;
            x[i] = (float)((int32_t)(rng >> 16) % 2048 - 1024) / 4096.0f;
        }
    }

    /* Pre-built selection buffers: rotating distinct expert sets so the
     * weights are not simply served from the system level cache. */
#define NSEL 32
    ds4_gpu_tensor *sel[NSEL], *wgt[NSEL];
    for (int s = 0; s < NSEL; s++) {
        sel[s] = ds4_gpu_tensor_alloc((uint64_t)N_USED * sizeof(int32_t));
        wgt[s] = ds4_gpu_tensor_alloc((uint64_t)N_USED * sizeof(float));
        if (!sel[s] || !wgt[s]) { fprintf(stderr, "bench: sel alloc failed\n"); return 1; }
        int32_t *ids = ds4_gpu_tensor_contents(sel[s]);
        float *w = ds4_gpu_tensor_contents(wgt[s]);
        for (uint32_t k = 0; k < N_USED; k++) {
            /* n_sel distinct experts, remaining slots repeat the last one so
             * the *number of distinct experts streamed* is n_sel. */
            const uint32_t kk = k < n_sel ? k : n_sel - 1;
            ids[k] = (int32_t)(((uint32_t)s * 6u + kk * 7u + 1u) % n_total_expert);
            w[k] = 0.25f;
        }
        /* de-duplicate so distinct count is exactly n_sel for k<n_sel */
        for (uint32_t k = 1; k < n_sel; k++) {
            for (uint32_t j = 0; j < k; j++) {
                if (ids[k] == ids[j]) ids[k] = (int32_t)(((uint32_t)ids[k] + 1u) % n_total_expert);
            }
        }
    }

    const uint64_t gate_off = 0;
    const uint64_t up_off = seg;
    const uint64_t down_off = 2ull * seg;

    /* warmup */
    for (int i = 0; i < 8; i++) {
        if (!ds4_gpu_begin_commands()) { fprintf(stderr, "begin failed\n"); return 1; }
        if (!ds4_gpu_routed_moe_one_tensor(out[i % NSETS], g_t[i % NSETS], u_t[i % NSETS],
                                           m_t[i % NSETS], d_t[i % NSETS],
                                           map, model_size, gate_off, up_off, down_off,
                                           DS4_METAL_TENSOR_MXFP4, DS4_METAL_TENSOR_MXFP4,
                                           gate_expert_bytes, gate_row_bytes,
                                           down_expert_bytes, down_row_bytes,
                                           N_EMBD, N_FF_EXP, N_EMBD,
                                           sel[i % NSEL], wgt[i % NSEL],
                                           n_total_expert, N_USED, 7.0f, x_t, NULL,
                                           0, false)) {
            fprintf(stderr, "bench: routed_moe dispatch failed\n");
            return 1;
        }
        if (!ds4_gpu_end_commands()) { fprintf(stderr, "end failed\n"); return 1; }
    }
    ds4_gpu_synchronize();

    /* [cb computeCommandEncoder] is MTLDispatchTypeSerial, so N dispatches in
     * one command buffer execute back to back; batching amortizes the very
     * large per-command-buffer residency cost of a no-copy model mapping. */
    const int per_cb = getenv("BENCH_PER_CB") ? atoi(getenv("BENCH_PER_CB")) : 16;
    const double t0 = now_ms();
    int done = 0;
    while (done < iters) {
        const int n = (iters - done) < per_cb ? (iters - done) : per_cb;
        ds4_gpu_begin_commands();
        for (int j = 0; j < n; j++) {
            const int i = done + j;
            ds4_gpu_routed_moe_one_tensor(out[i % NSETS], g_t[i % NSETS], u_t[i % NSETS],
                                          m_t[i % NSETS], d_t[i % NSETS],
                                          map, model_size, gate_off, up_off, down_off,
                                          DS4_METAL_TENSOR_MXFP4, DS4_METAL_TENSOR_MXFP4,
                                          gate_expert_bytes, gate_row_bytes,
                                          down_expert_bytes, down_row_bytes,
                                          N_EMBD, N_FF_EXP, N_EMBD,
                                          sel[i % NSEL], wgt[i % NSEL],
                                          n_total_expert, N_USED, 7.0f, x_t, NULL,
                                          0, false);
        }
        ds4_gpu_end_commands();
        done += n;
    }
    ds4_gpu_synchronize();
    const double wall = now_ms() - t0;
    printf("  per_cb=%d\n", per_cb);

    const double bytes = (double)n_sel *
        (double)(gate_expert_bytes * 2ull + down_expert_bytes);
    printf("  %d iterations, %u distinct experts/iter\n", iters, n_sel);
    printf("  wall %.4f ms/iter  (includes submit; see gpu busy under "
           "DS4_METAL_GPU_BUSY_PROFILE=1)\n", wall / iters);
    printf("  weight bytes/iter %.2f MB -> wall-derived %.1f GB/s\n",
           bytes / 1e6, bytes / (wall / iters / 1000.0) / 1e9);
    printf("  (43 layers at this rate: %.2f ms/token of routed MoE)\n",
           wall / iters * 43.0);

    const float *o = ds4_gpu_tensor_contents(out[(iters - 1) % NSETS]);
    printf("  out[0]=%g out[1]=%g (liveness only)\n", (double)o[0], (double)o[1]);
    /* Bit-exact fingerprint of the whole output, so a kernel edit that is meant
     * to change only how bytes are fetched -- not what they decode to -- can be
     * proved identical rather than assumed.  FNV-1a over the raw float bits;
     * any single-bit change moves it. */
    {
        uint64_t h = 1469598103934665603ull;
        const unsigned char *b = (const unsigned char *)o;
        const size_t n = (size_t)N_EMBD * sizeof(float);
        for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
        printf("  output fnv1a=%016llx over %zu bytes\n",
               (unsigned long long)h, n);
    }
    return 0;
}
