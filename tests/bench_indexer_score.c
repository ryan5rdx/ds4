#define _DARWIN_C_SOURCE

/* Standalone harness for the sparse indexer's decode score kernel.
 *
 * The kernel dominates long-context decode: at 155k of context it is roughly
 * 12 ms of a 47.6 ms token on a 2x M2 Ultra pair, running at ~2.6% of peak.
 * Iterating through a real model costs a ~12 minute prefill per data point,
 * so this links only ds4_metal.o and dispatches the kernel directly.
 *
 * STATUS: the harness is self-consistent but DOES NOT PREDICT ENGINE GAINS.
 * Use it to compare kernel variants, not to forecast tokens/s.
 *
 * It scales linearly and cleanly (M1 Max, GPU busy via
 * DS4_METAL_GPU_BUSY_PROFILE=1): 0.138 ms at n_comp=4096, 0.253 at 8192,
 * 0.497 at 16384, 1.163 at 38947 - within 1.5% of the linear fit. Scaled to
 * an M2 Ultra by core count that is 0.367 ms/layer against the engine's
 * ~0.57 ms deflated, so the absolute level is sane.
 *
 * But calibrating it against a change with a known engine result found a 10x
 * discrepancy. Collapsing the score kernel's per-head barriers measured 7.934
 * -> 1.163 ms/dispatch here, 6.8x, which would imply +27% decode. On the real
 * pair it delivered +2.7%. Run the kernel alone and every stall becomes wall
 * time; run it inside the graph and other work hides them. Either the harness
 * exaggerates latency-bound fixes, or the engine's score share is far below
 * the ~12 ms deflated estimate. Those are distinguishable: re-run the engine
 * under DS4_METAL_INDEXER_STAGE_PROFILE=1 and see whether per-layer score fell
 * 6.8x or barely moved.
 *
 * ALWAYS dispatch one kernel per command buffer (per_cb=1). Batching repeat
 * dispatches that write the same buffer serializes them behind write-after-
 * write hazards and inflated the measurement 267x here.
 *
 * Also note the CPU reference sums lanes sequentially while simd_sum reduces
 * them as a tree, so exact agreement is not expected and its absence is not
 * evidence of a kernel bug. For comparing kernel versions use BENCH_DUMP and
 * diff GPU output against GPU output.
 *
 *   make tests/bench_indexer_score
 *   DS4_METAL_GPU_BUSY_PROFILE=1 ./tests/bench_indexer_score 38947 640 10
 *   BENCH_DUMP=/tmp/scores.bin ./tests/bench_indexer_score 38947 1 1
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_HEAD    64u
#define HEAD_DIM  128u

/* ds4_metal.o expects these from the engine; the harness never logs through
 * them but the linker needs the symbols. */
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
void ds4_log_line(int type, const char *fmt, ...) { (void)type; (void)fmt; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Deterministic, and bounded well away from the f32 rounding cliff so the
 * reference and the kernel agree exactly rather than approximately. */
static float sample(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)((int32_t)(*state >> 16) % 2048 - 1024) / 1024.0f;
}

static void reference_scores(const float *q, const float *w, const float *comp,
                             float *out, uint32_t n_comp, float scale) {
    for (uint32_t row = 0; row < n_comp; row++) {
        const float *k = comp + (size_t)row * HEAD_DIM;
        float acc = 0.0f;
        for (uint32_t head = 0; head < N_HEAD; head++) {
            const float *qh = q + (size_t)head * HEAD_DIM;
            float s = 0.0f;
            /* float4 lanes: the kernel's dot() pairs element j with j, and
             * simd_sum reduces 32 lane partials as a tree. Sum in the same
             * lane-major order so the reference tracks it. */
            for (uint32_t lane = 0; lane < 32u; lane++) {
                for (uint32_t c = 0; c < 4u; c++) {
                    const uint32_t j = lane * 4u + c;
                    s += qh[j] * k[j];
                }
            }
            acc += fmaxf(s, 0.0f) * (w[head] * scale);
        }
        out[row] = acc;
    }
}

int main(int argc, char **argv) {
    uint32_t n_comp = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 38947u;
    int iters = argc > 2 ? atoi(argv[2]) : 640;
    /* Keep this at 1: repeat dispatches writing the same scores buffer
     * serialize behind write-after-write hazards, which inflated the measured
     * time 267x. Read GPU time, not the wall clock. */
    int per_cb = argc > 3 ? atoi(argv[3]) : 1;
    if (n_comp == 0 || iters <= 0) {
        fprintf(stderr, "usage: %s [n_comp] [iters]\n", argv[0]);
        return 2;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "bench: Metal init failed\n");
        return 1;
    }

    const size_t q_n = (size_t)N_HEAD * HEAD_DIM;
    const size_t comp_n = (size_t)n_comp * HEAD_DIM;
    const float scale = 1.0f / sqrtf((float)(HEAD_DIM * N_HEAD));

    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(q_n * sizeof(float));
    ds4_gpu_tensor *w_t = ds4_gpu_tensor_alloc(N_HEAD * sizeof(float));
    ds4_gpu_tensor *c_t = ds4_gpu_tensor_alloc(comp_n * sizeof(float));
    ds4_gpu_tensor *s_t = ds4_gpu_tensor_alloc((size_t)n_comp * sizeof(float));
    if (!q_t || !w_t || !c_t || !s_t) {
        fprintf(stderr, "bench: allocation failed (n_comp=%u, %.2f GiB)\n",
                n_comp, (double)(comp_n * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
        return 1;
    }

    float *q = ds4_gpu_tensor_contents(q_t);
    float *w = ds4_gpu_tensor_contents(w_t);
    float *c = ds4_gpu_tensor_contents(c_t);
    float *s = ds4_gpu_tensor_contents(s_t);

    uint32_t rng = 0x9e3779b9u;
    for (size_t i = 0; i < q_n; i++) q[i] = sample(&rng);
    for (size_t i = 0; i < N_HEAD; i++) w[i] = sample(&rng);
    for (size_t i = 0; i < comp_n; i++) c[i] = sample(&rng);

    /* Correctness first: a fast wrong kernel is worthless. */
    float *ref = malloc((size_t)n_comp * sizeof(float));
    if (!ref) return 1;
    reference_scores(q, w, c, ref, n_comp, scale);

    memset(s, 0, (size_t)n_comp * sizeof(float));
    if (!ds4_gpu_begin_commands() ||
        !ds4_gpu_indexer_score_one_tensor(s_t, q_t, w_t, c_t,
                                          n_comp, N_HEAD, HEAD_DIM, scale) ||
        !ds4_gpu_end_commands() || !ds4_gpu_synchronize()) {
        fprintf(stderr, "bench: dispatch failed\n");
        return 1;
    }

    uint32_t exact = 0, close = 0;
    double worst = 0.0;
    uint32_t worst_row = 0;
    for (uint32_t i = 0; i < n_comp; i++) {
        if (s[i] == ref[i]) { exact++; continue; }
        const double denom = fabs((double)ref[i]) > 1e-6 ? fabs((double)ref[i]) : 1e-6;
        const double rel = fabs((double)s[i] - (double)ref[i]) / denom;
        if (rel < 1e-5) close++;
        if (rel > worst) { worst = rel; worst_row = i; }
    }

    printf("indexer decode score: n_comp=%u  n_head=%u head_dim=%u\n",
           n_comp, N_HEAD, HEAD_DIM);
    printf("  vs CPU reference: %u/%u bit-exact, %u within 1e-5, worst rel %.3e (row %u)\n",
           exact, n_comp, close, worst, worst_row);
    /* The reference sums lanes sequentially; simd_sum reduces them as a tree,
     * so exact agreement is not expected and its absence proves nothing. This
     * is a sanity bound. The real oracle is --dump: write the GPU scores and
     * diff them across kernel versions, which compares like with like. */
    if (worst > 1e-3) {
        printf("  WARNING: worst relative error %.3e exceeds 1e-3 -- kernel looks wrong.\n", worst);
    }
    const char *dump = getenv("BENCH_DUMP");
    if (dump) {
        FILE *fp = fopen(dump, "wb");
        if (fp) {
            fwrite(s, sizeof(float), n_comp, fp);
            fclose(fp);
            printf("  scores written to %s (%u floats) for cross-version diff\n", dump, n_comp);
        }
    }

    /* Warm the pipeline and the residency set before timing. */
    for (int i = 0; i < 5; i++) {
        ds4_gpu_begin_commands();
        ds4_gpu_indexer_score_one_tensor(s_t, q_t, w_t, c_t,
                                         n_comp, N_HEAD, HEAD_DIM, scale);
        ds4_gpu_end_commands();
    }

    /* Batch the dispatches into one command buffer so the measurement is the
     * kernel, not per-submit latency -- this is how the engine encodes it. */
    const int batches = (iters + per_cb - 1) / per_cb;
    double best = 1e30, total = 0.0;
    for (int b = 0; b < batches; b++) {
        const double t0 = now_ms();
        ds4_gpu_begin_commands();
        for (int i = 0; i < per_cb; i++) {
            ds4_gpu_indexer_score_one_tensor(s_t, q_t, w_t, c_t,
                                             n_comp, N_HEAD, HEAD_DIM, scale);
        }
        ds4_gpu_end_commands();
        const double dt = (now_ms() - t0) / per_cb;
        total += dt;
        if (dt < best) best = dt;
    }
    const double mean = total / batches;

    const double flop = (double)n_comp * N_HEAD * HEAD_DIM * 2.0;
    const double kbytes = (double)n_comp * HEAD_DIM * sizeof(float);
    printf("  %d dispatches (%d per command buffer)\n", batches * per_cb, per_cb);
    printf("  wall %.4f ms/dispatch (best) -- CPU-dominated, see GPU busy below\n", best);
    printf("  run with DS4_METAL_GPU_BUSY_PROFILE=1 for true GPU time; divide by %d\n",
           per_cb * 64);
    (void)mean;
    printf("  %.1f GFLOP/s   K-cache %.1f GB/s (%.2f MB/dispatch)\n",
           flop / (best / 1000.0) / 1e9,
           kbytes / (best / 1000.0) / 1e9,
           kbytes / 1e6);
    printf("  q re-read if uncached: %.2f GB/dispatch\n",
           (double)n_comp * q_n * sizeof(float) / 1e9);
    printf("  extrapolated to 21 ratio-4 layers: %.2f ms/token\n", best * 21.0);

    free(ref);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(w_t);
    ds4_gpu_tensor_free(c_t);
    ds4_gpu_tensor_free(s_t);
    return exact == n_comp ? 0 : 1;
}
