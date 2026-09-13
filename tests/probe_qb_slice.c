/*
 * Byte-identity gate for the sliced Q8_0 projection.
 *
 * Prefill's TP attention head split computes the full q width on both ranks
 * and reads only its own half — q_b produces 64 heads where 32 are consumed,
 * about 6 ms of a 12 ms layer across 11 DSA layers. Slicing it is only worth
 * doing if the owned half comes out BIT-IDENTICAL to the full-width result:
 * "close" would be a silent accuracy change in the attention path, which is
 * the worst place in the model to take one.
 *
 * That identity is not obvious. The weight offset advances by whole Q8_0 rows,
 * so each output row's arithmetic is unchanged — but the kernel selected can
 * differ with out_dim (row count drives the dispatch grid and the tiling), and
 * a different kernel is a different reduction order. This measures rather than
 * assumes, at the production shape and at a realistic stripe.
 *
 * No GGUF: synthetic Q8_0 weights at 1536->16384 are the right shape, and
 * identity does not depend on the values.
 *
 *   probe_qb_slice [--tokens N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "ds4_gpu.h"

int ds4_gpu_begin_commands(void);
int ds4_gpu_end_commands(void);

#define IN_DIM   1536u            /* DS4_N_LORA_Q */
#define OUT_DIM  16384u           /* DS4_N_HEAD * DS4_N_HEAD_DIM */
#define QK       32u
#define BLK      34u

int main(int argc, char **argv) {
    uint32_t n_tok = 512;         /* a realistic prefill stripe */
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--tokens") && i + 1 < argc) n_tok = (uint32_t)atoi(argv[++i]);

    if (!ds4_gpu_init()) { puts("VOID: no GPU"); return 1; }

    const uint64_t row_bytes = (uint64_t)(IN_DIM / QK) * BLK;
    const uint64_t wbytes = (uint64_t)OUT_DIM * row_bytes;
    uint8_t *w = NULL;
    if (posix_memalign((void **)&w, 16384, (size_t)wbytes) != 0 || !w) {
        puts("VOID: weights"); return 1;
    }
    uint32_t st = 2463534242u;
    for (uint64_t r = 0; r < OUT_DIM; ++r) {
        uint8_t *row = w + r * row_bytes;
        for (uint64_t b = 0; b < IN_DIM / QK; ++b) {
            uint8_t *p = row + b * BLK;
            _Float16 d = (_Float16)(0.008f + (float)(st % 11) * 0.0013f);
            memcpy(p, &d, 2);
            for (uint32_t j = 0; j < QK; ++j) {
                st ^= st << 13; st ^= st >> 17; st ^= st << 5;
                p[2 + j] = (uint8_t)(int8_t)((int)(st >> 24) - 128);
            }
        }
    }

    /* The matmul resolves its weights through the registered model map, so a
     * bare malloc is invisible to it. Page-align: the registration wraps the
     * pointer in an MTLBuffer with no copy. */
    if (!ds4_gpu_set_model_map(w, wbytes)) {
        puts("VOID: could not register the synthetic weights as a model map");
        return 1;
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * IN_DIM * sizeof(float));
    ds4_gpu_tensor *full = ds4_gpu_tensor_alloc((uint64_t)n_tok * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *slic = ds4_gpu_tensor_alloc((uint64_t)n_tok * OUT_DIM * sizeof(float));
    if (!x || !full || !slic) { puts("VOID: tensors"); return 1; }
    float *xp = ds4_gpu_tensor_contents(x);
    for (uint64_t i = 0; i < (uint64_t)n_tok * IN_DIM; ++i)
        xp[i] = sinf((float)(i % 9973) * 0.0037f) * 0.8f;

    /* Full width, the reference. */
    ds4_gpu_begin_commands();
    const int okf = ds4_gpu_matmul_q8_0_cols_tensor(full, w, wbytes, 0, IN_DIM,
                                                    OUT_DIM, OUT_DIM, 0, x, n_tok);
    ds4_gpu_end_commands();
    if (!okf) { puts("VOID: full-width call failed"); return 1; }

    const uint32_t half = OUT_DIM / 2;
    int bad_total = 0;
    for (uint32_t rank = 0; rank < 2; ++rank) {
        const uint32_t lo = rank * half;
        /* Poison, so an untouched region is distinguishable from a correct
         * zero and a short write cannot pass. */
        float *sp = ds4_gpu_tensor_contents(slic);
        for (uint64_t i = 0; i < (uint64_t)n_tok * OUT_DIM; ++i) sp[i] = -123456.0f;

        ds4_gpu_begin_commands();
        const int oks = ds4_gpu_matmul_q8_0_cols_tensor(slic, w, wbytes,
                                                        0, IN_DIM, half,
                                                        OUT_DIM, lo, x, n_tok);
        ds4_gpu_end_commands();
        if (!oks) { printf("rank %u: VOID -- sliced call refused\n", rank); return 1; }

        const float *fp = ds4_gpu_tensor_contents(full);
        uint64_t diff = 0, touched_outside = 0;
        double worst = 0.0;
        for (uint32_t t = 0; t < n_tok; ++t) {
            const float *fr = fp + (uint64_t)t * OUT_DIM;
            const float *sr = sp + (uint64_t)t * OUT_DIM;
            for (uint32_t c = 0; c < OUT_DIM; ++c) {
                if (c >= lo && c < lo + half) {
                    /* memcmp semantics: identical bits, not merely close. */
                    if (memcmp(&fr[c], &sr[c], sizeof(float)) != 0) {
                        diff++;
                        const double d = fabs((double)fr[c] - (double)sr[c]);
                        if (d > worst) worst = d;
                    }
                } else if (sr[c] != -123456.0f) {
                    touched_outside++;
                }
            }
        }
        printf("rank %u cols [%u,%u): %llu/%llu bits differ (worst %.9g), "
               "%llu writes outside the slice\n",
               rank, lo, lo + half,
               (unsigned long long)diff,
               (unsigned long long)((uint64_t)n_tok * half), worst,
               (unsigned long long)touched_outside);
        if (diff || touched_outside) bad_total++;
    }

    printf("\n%s\n", bad_total
           ? "FAIL: the sliced projection is not byte-identical"
           : "PASS: owned half byte-identical, nothing written outside the slice");
    free(w);
    return bad_total ? 1 : 0;
}

/* Stubs: ds4_metal.o references these from ds4.c, which this probe omits. */
int ds4_log_is_tty(void) { return 0; }
int ds4_deepseek4_attention_bounds(void *a, void *b, void *c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 0;
}
