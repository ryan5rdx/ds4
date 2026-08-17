#define _DARWIN_C_SOURCE

/* Standalone harness for the DeepSeek V4 compressor pair projection.
 *
 * Dispatches kernel_mul_mv_f16_f32_pair_4 (via
 * ds4_gpu_matmul_f16_pair_tensor) at the decode shapes used by
 * ds4.c:22301 (attn compressor) and ds4.c:22408 (indexer compressor):
 * in_dim = 4096, out_dim in {1024, 512, 256}.
 *
 * Weight offsets rotate through a large fake model map so the stream is not
 * served out of the system level cache, which is what happens in the engine
 * where GBs move per token.
 *
 * Same caveats as tests/bench_indexer_score.c: this measures a kernel in
 * isolation and DOES NOT PREDICT ENGINE GAINS. One dispatch per command
 * buffer; read GPU busy time, not the wall clock.
 *
 *   make tests/bench_f16_pair_compressor
 *   DS4_METAL_GPU_BUSY_PROFILE=1 ./tests/bench_f16_pair_compressor 1024 640
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

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static float sample(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)((int32_t)(*state >> 16) % 2048 - 1024) / 1024.0f;
}

int main(int argc, char **argv) {
    const uint64_t in_dim = 4096;
    uint64_t out_dim = argc > 1 ? strtoull(argv[1], NULL, 10) : 1024;
    int iters = argc > 2 ? atoi(argv[2]) : 640;
    uint64_t map_bytes = argc > 3 ? strtoull(argv[3], NULL, 10) * 1024ull * 1024ull
                                  : 1024ull * 1024ull * 1024ull;

    if (!ds4_gpu_init()) {
        fprintf(stderr, "bench: Metal init failed\n");
        return 1;
    }

    void *map = mmap(NULL, (size_t)map_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "bench: mmap failed\n");
        return 1;
    }

    /* One MiB of plausible half-precision weights, replicated. */
    const size_t pat_bytes = 1024 * 1024;
    __fp16 *pat = malloc(pat_bytes);
    uint32_t rng = 0x9e3779b9u;
    for (size_t i = 0; i < pat_bytes / sizeof(__fp16); i++) {
        pat[i] = (__fp16)(sample(&rng) * 0.05f);
    }
    for (size_t off = 0; off < (size_t)map_bytes; off += pat_bytes) {
        size_t n = pat_bytes;
        if (off + n > (size_t)map_bytes) n = (size_t)map_bytes - off;
        memcpy((char *)map + off, pat, n);
    }

    if (!ds4_gpu_set_model_map(map, map_bytes)) {
        fprintf(stderr, "bench: set_model_map failed\n");
        return 1;
    }

    const uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    const uint64_t pair_bytes = 2 * weight_bytes;

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *a_t = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *b_t = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    if (!x_t || !a_t || !b_t) {
        fprintf(stderr, "bench: tensor alloc failed\n");
        return 1;
    }
    float *x = ds4_gpu_tensor_contents(x_t);
    for (uint64_t i = 0; i < in_dim; i++) x[i] = sample(&rng);

    const uint64_t slots = map_bytes / pair_bytes;
    if (slots == 0) {
        fprintf(stderr, "bench: map too small for out_dim %llu\n",
                (unsigned long long)out_dim);
        return 1;
    }

    for (int i = 0; i < 8; i++) {
        if (!ds4_gpu_matmul_f16_pair_tensor(a_t, b_t, map, map_bytes,
                                            0, weight_bytes,
                                            in_dim, out_dim, x_t, 1)) {
            fprintf(stderr, "bench: warm dispatch failed\n");
            return 1;
        }
    }

    double best = 1e30, total = 0.0;
    for (int i = 0; i < iters; i++) {
        const uint64_t slot = (uint64_t)i % slots;
        const uint64_t off_a = slot * pair_bytes;
        const double t0 = now_ms();
        if (!ds4_gpu_matmul_f16_pair_tensor(a_t, b_t, map, map_bytes,
                                            off_a, off_a + weight_bytes,
                                            in_dim, out_dim, x_t, 1)) {
            fprintf(stderr, "bench: dispatch failed\n");
            return 1;
        }
        const double dt = now_ms() - t0;
        total += dt;
        if (dt < best) best = dt;
    }

    const float *a = ds4_gpu_tensor_contents(a_t);
    printf("f16 pair compressor projection: in=%llu out=%llu (%.2f MB/dispatch)\n",
           (unsigned long long)in_dim, (unsigned long long)out_dim,
           (double)pair_bytes / 1e6);
    printf("  slots=%llu (%.0f MB map), out[0]=%.6f out[%llu]=%.6f\n",
           (unsigned long long)slots, (double)map_bytes / 1e6,
           (double)a[0], (unsigned long long)out_dim - 1, (double)a[out_dim - 1]);
    printf("  %d dispatches, 1 per command buffer\n", iters);
    printf("  wall best %.4f ms  mean %.4f ms  -> %.1f GB/s (wall best)\n",
           best, total / iters, (double)pair_bytes / (best / 1000.0) / 1e9);
    printf("  run under DS4_METAL_GPU_BUSY_PROFILE=1 for GPU busy time\n");
    return 0;
}
