/* Bridge-kernel exactness probe. The transpose is the part that fails
 * silently: a wrong index still produces a full, plausible tensor. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "ds4_gpu.h"
int ds4_gpu_begin_commands(void); int ds4_gpu_end_commands(void);
int main(void) {
    if (!ds4_gpu_init()) { puts("VOID: no GPU"); return 1; }
    const uint32_t D = 256, M = 64;      /* small, same index math */
    void *in = NULL, *out = NULL;
    if (!ds4_gpu_ane_stage_alloc(D, M, &in, &out)) { puts("VOID: stage alloc"); return 1; }
    ds4_gpu_tensor *src = ds4_gpu_tensor_alloc((uint64_t)D * M * sizeof(float));
    ds4_gpu_tensor *dst = ds4_gpu_tensor_alloc((uint64_t)D * M * sizeof(float));
    if (!src || !dst) { puts("VOID: tensors"); return 1; }
    float *sp = ds4_gpu_tensor_contents(src), *dp = ds4_gpu_tensor_contents(dst);
    for (uint32_t t = 0; t < M; ++t)
        for (uint32_t d = 0; d < D; ++d)
            sp[t * D + d] = (float)((int)(t * 7 + d * 13) % 101) * 0.03125f - 1.5f;
    for (uint32_t i = 0; i < D * M; ++i) dp[i] = -999.0f;

    ds4_gpu_begin_commands();
    if (!ds4_gpu_ane_pack(src, D, M))   { puts("VOID: pack"); return 1; }
    ds4_gpu_end_commands();

    /* 1. the staged surface must be the TRANSPOSE, in f16 */
    const uint16_t *h = (const uint16_t *)in;
    int bad_t = 0; double worst = 0;
    for (uint32_t t = 0; t < M; ++t) for (uint32_t d = 0; d < D; ++d) {
        _Float16 v; __builtin_memcpy(&v, &h[(size_t)d * M + t], 2);
        double e = fabs((double)(float)v - (double)sp[t * D + d]);
        if (e > worst) worst = e;
        if (e > 1e-3) bad_t++;
    }
    printf("pack   : transposed f16, %d/%u mismatches, worst %.6f\n", bad_t, D * M, worst);

    /* 2. round trip back must reproduce the source */
    ds4_gpu_begin_commands();
    for (uint32_t i = 0; i < D * M; ++i) ((uint16_t *)out)[i] = ((const uint16_t *)in)[i];
    if (!ds4_gpu_ane_unpack(dst, D, M, 0)) { puts("VOID: unpack"); return 1; }
    ds4_gpu_end_commands();
    int bad_r = 0; worst = 0;
    for (uint32_t i = 0; i < D * M; ++i) {
        double e = fabs((double)dp[i] - (double)sp[i]);
        if (e > worst) worst = e;
        if (e > 1e-3) bad_r++;
    }
    printf("unpack : round trip, %d/%u mismatches, worst %.6f\n", bad_r, D * M, worst);

    /* 3. accumulate must ADD, not overwrite -- dropping the routed-MoE
     *    contribution is exactly the bug that still produces fluent text */
    for (uint32_t i = 0; i < D * M; ++i) dp[i] = 1.0f;
    ds4_gpu_begin_commands();
    ds4_gpu_ane_unpack(dst, D, M, 1);
    ds4_gpu_end_commands();
    int bad_a = 0;
    for (uint32_t i = 0; i < D * M; ++i)
        if (fabs((double)dp[i] - (1.0 + (double)sp[i])) > 1e-3) bad_a++;
    printf("accum  : += semantics, %d/%u mismatches\n", bad_a, D * M);

    /* 4. compare must report ~0 against a matching reference, and non-zero
     *    against a perturbed one */
    double ma = -1, rr = -1;
    ds4_gpu_begin_commands();
    ds4_gpu_ane_compare(src, D, M, &ma, &rr);
    printf("compare: identical   max_abs %.6f rel_rms %.6f\n", ma, rr);
    sp[0] += 4.0f; sp[1] -= 3.0f;
    ds4_gpu_ane_compare(src, D, M, &ma, &rr);
    printf("compare: perturbed   max_abs %.6f rel_rms %.6f\n", ma, rr);
    ds4_gpu_end_commands();

    printf("\n%s\n", (bad_t || bad_r || bad_a) ? "FAIL" : "PASS: bridge kernels exact");
    return (bad_t || bad_r || bad_a) ? 1 : 0;
}
/* Stubs: ds4_metal.o references these from ds4.c, which the probe omits. */
int ds4_log_is_tty(void) { return 0; }
int ds4_deepseek4_attention_bounds(void *a, void *b, void *c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 0;
}
