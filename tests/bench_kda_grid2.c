/* Second probe: attribute GLM 5.3 KDA prefill cost between the three kernels
 * by varying the dimensions each one is dispatched over.
 *
 *   prepare    : grid (n_heads),            serial loop over n_tokens
 *   recurrence : grid (n_heads, 32),        serial loop over n_tokens
 *   output     : grid (n_tokens, n_heads),  no loop
 *
 * If total time is flat in n_heads, the serial token loops dominate.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
int ds4_gpu_begin_commands(void);
int ds4_gpu_end_commands(void);

static void req(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "%s failed\n", what); exit(1); }
}
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

enum { D = 128, HEADS_TOTAL = 64 };
enum {
    Q_CONV_OFFSET = 0, K_CONV_OFFSET = 131072, V_CONV_OFFSET = 262144,
    A_LOG_OFFSET = 393216, DT_BIAS_OFFSET = 393728, NORM_OFFSET = 426496,
    MODEL_BYTES = 524288,
};

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    req(model != MAP_FAILED, "mmap");
    float *q_conv = (float *)(model + Q_CONV_OFFSET);
    float *k_conv = (float *)(model + K_CONV_OFFSET);
    float *v_conv = (float *)(model + V_CONV_OFFSET);
    float *a_log  = (float *)(model + A_LOG_OFFSET);
    float *dt     = (float *)(model + DT_BIAS_OFFSET);
    float *norm   = (float *)(model + NORM_OFFSET);
    for (uint32_t c = 0; c < HEADS_TOTAL * D; c++) {
        q_conv[c*4+3] = 1.0f; k_conv[c*4+3] = 1.0f; v_conv[c*4+3] = 1.0f;
        dt[c] = 0.0f;
    }
    for (uint32_t h = 0; h < HEADS_TOTAL; h++) a_log[h] = 0.0f;
    for (uint32_t i = 0; i < D; i++) norm[i] = 1.0f;
    req(ds4_gpu_init(), "gpu init");
    req(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map");

    ds4_gpu_tensor *conv = ds4_gpu_tensor_alloc(9ull * HEADS_TOTAL * D * sizeof(float));
    ds4_gpu_tensor *st   = ds4_gpu_tensor_alloc((uint64_t)HEADS_TOTAL * D * D * sizeof(float));
    req(conv && st, "alloc state");
    req(ds4_gpu_tensor_fill_f32(conv, 0.0f, 9ull * HEADS_TOTAL * D), "fill conv");
    req(ds4_gpu_tensor_fill_f32(st, 0.01f, (uint64_t)HEADS_TOTAL * D * D), "fill st");

    const uint32_t hs[] = { 8, 16, 32, 64 };
    const uint32_t ts[] = { 64, 128, 256 };
    for (uint32_t ti = 0; ti < 3; ti++) {
      for (uint32_t hi = 0; hi < 4; hi++) {
        const uint32_t heads = hs[hi], toks = ts[ti];
        const uint64_t act = (uint64_t)heads * D * toks;
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *g = ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *og= ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *b = ds4_gpu_tensor_alloc((uint64_t)toks*heads*sizeof(float));
        ds4_gpu_tensor *o = ds4_gpu_tensor_alloc(act * sizeof(float));
        req(q&&k&&v&&g&&og&&b&&o, "alloc act");
        req(ds4_gpu_tensor_fill_f32(q, 0.3f, act), "fq");
        req(ds4_gpu_tensor_fill_f32(k, 0.2f, act), "fk");
        req(ds4_gpu_tensor_fill_f32(v, 0.1f, act), "fv");
        req(ds4_gpu_tensor_fill_f32(g, 0.0f, act), "fg");
        req(ds4_gpu_tensor_fill_f32(og,0.0f, act), "fog");
        req(ds4_gpu_tensor_fill_f32(b, 0.0f, (uint64_t)toks*heads), "fb");
        const uint32_t reps = 20;
        for (int w = 0; w < 3; w++)
            req(ds4_gpu_glm53_kda_prefill(o, conv, st, q,k,v,g,b,og, model,
                MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
                A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
                heads, toks, HEADS_TOTAL, 0u, -5.0f, 1e-5f), "warm");
        req(ds4_gpu_begin_commands(), "begin");
        const double t0 = now_ms();
        for (uint32_t i = 0; i < reps; i++)
            req(ds4_gpu_glm53_kda_prefill(o, conv, st, q,k,v,g,b,og, model,
                MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
                A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
                heads, toks, HEADS_TOTAL, 0u, -5.0f, 1e-5f), "run");
        req(ds4_gpu_end_commands(), "end");
        const double us = (now_ms() - t0) * 1000.0 / reps;
        printf("prefill toks=%4u heads=%2u  %9.1f us  (%6.3f us/token, "
               "prep_tgs=%u rec_tgs=%u out_tgs=%u)\n",
               toks, heads, us, us / toks, heads, heads*32u, toks*heads);
        ds4_gpu_tensor_free(o); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(og);
        ds4_gpu_tensor_free(g); ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(k);
        ds4_gpu_tensor_free(q);
      }
    }
    ds4_gpu_cleanup();
    return 0;
}
