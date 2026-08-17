/* CPU reference for the sparse indexer score, via Accelerate (which dispatches
 * to the AMX coprocessor).
 *
 * The point is not to offload anything. It is to have an independent answer to
 * "what should this arithmetic cost", which every attempt to predict a GPU
 * kernel speedup in this engine has lacked. The indexer score is
 *
 *     S = K(n_comp x 128) * Qt(128 x 64)
 *     score[r] = sum_h relu(S[r][h]) * w[h]
 *
 * i.e. a plain f32 GEMM plus a cheap epilogue, with no MXFP4 anywhere.
 *
 * Measured on an M1 Max at n_comp = 47530 (190k of context):
 *     sgemm 0.572 ms at 1361 GFLOP/s = 76% of AMX's ~1.78 TFLOP/s peak,
 *     plus a 0.200 ms scalar epilogue, so 0.772 ms/layer = 16.2 ms/token.
 * The Metal kernel needs ~10.3 ms/token for the same 16.3 GFLOP across 21
 * ratio-4 layers, which is 3% of the GPU's ~27 TFLOP/s. A coprocessor with a
 * fifteenth of the peak throughput lands within 1.6x of the GPU, which says
 * the GPU kernel is dispatch-bound rather than doing hard work.
 *
 * Accelerate is the sanctioned entry point; none of the undocumented AMX
 * instructions are needed. Note that cblas is f32 only - reaching the bf16
 * modes M2 added would mean BNNS or raw instructions.
 *
 *   make tests/bench_amx_indexer && ./tests/bench_amx_indexer [n_comp]
 */

#define ACCELERATE_NEW_LAPACK

#include <Accelerate/Accelerate.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_INDEXER_HEAD      64
#define N_INDEXER_HEAD_DIM  128
#define RATIO4_LAYERS       21   /* DS4 Flash: even layers 2..42 */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const int n_comp = argc > 1 ? atoi(argv[1]) : 47530;
    if (n_comp <= 0) {
        fprintf(stderr, "usage: %s [n_comp]   (47530 ~ 190k context)\n", argv[0]);
        return 2;
    }
    const int H = N_INDEXER_HEAD, D = N_INDEXER_HEAD_DIM;

    /* aligned_alloc requires a size that is a multiple of the alignment. */
    #define ALIGN64(n) ((((size_t)(n)) + 63u) & ~(size_t)63u)
    float *K = aligned_alloc(64, ALIGN64((size_t)n_comp * D * sizeof(float)));
    float *Q = aligned_alloc(64, ALIGN64((size_t)D * H * sizeof(float)));
    float *S = aligned_alloc(64, ALIGN64((size_t)n_comp * H * sizeof(float)));
    float *w = aligned_alloc(64, ALIGN64((size_t)H * sizeof(float)));
    float *out = aligned_alloc(64, ALIGN64((size_t)n_comp * sizeof(float)));
    if (!K || !Q || !S || !w || !out) {
        fprintf(stderr, "allocation failed for n_comp=%d\n", n_comp);
        return 1;
    }
    for (size_t i = 0; i < (size_t)n_comp * D; i++)
        K[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < D * H; i++)
        Q[i] = (float)((i * 40503u) % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < H; i++) w[i] = 0.01f * (float)(i % 7);

    double best_gemm = 1e30, best_total = 1e30;
    for (int it = 0; it < 12; it++) {
        const double t0 = now_ms();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n_comp, H, D, 1.0f, K, D, Q, H, 0.0f, S, H);
        const double t1 = now_ms();
        for (int r = 0; r < n_comp; r++) {
            const float *s = S + (size_t)r * H;
            float acc = 0.0f;
            for (int h = 0; h < H; h++) acc += (s[h] > 0.0f ? s[h] : 0.0f) * w[h];
            out[r] = acc;
        }
        const double t2 = now_ms();
        if (t1 - t0 < best_gemm) best_gemm = t1 - t0;
        if (t2 - t0 < best_total) best_total = t2 - t0;
    }

    const double flop = 2.0 * (double)n_comp * H * D;
    const double kbytes = (double)n_comp * D * sizeof(float);
    printf("indexer score, CPU/AMX via Accelerate: n_comp=%d (~%dk context)\n",
           n_comp, n_comp * 4 / 1000);
    printf("  sgemm        %7.3f ms   %7.1f GFLOP/s   %6.1f GB/s over K\n",
           best_gemm, flop / (best_gemm / 1000.0) / 1e9,
           kbytes / (best_gemm / 1000.0) / 1e9);
    printf("  + epilogue   %7.3f ms\n", best_total - best_gemm);
    printf("  per layer    %7.3f ms      x%d layers = %6.2f ms/token\n",
           best_total, RATIO4_LAYERS, best_total * RATIO4_LAYERS);
    printf("  for scale: the Metal kernel needs ~10.3 ms/token for score at 190k\n");

    free(K); free(Q); free(S); free(w); free(out);
    return 0;
}
