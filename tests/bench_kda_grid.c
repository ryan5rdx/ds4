/* Probe: is kernel_glm53_kda_decode grid-starved at the GLM 5.3 TP2 shape?
 *
 * Production decode dispatch is MTLSizeMake(n_rows=1, n_heads=32, 1) with
 * 128 threads/threadgroup -> 32 threadgroups.  Compare against:
 *   - the same kernel at n_heads = 64 (unsplit): if the time is flat, the
 *     kernel is grid/latency bound, not work bound.
 *   - kernel_glm53_kda_prefill_recurrence at n_rows = 1, which is the SAME
 *     recurrence math on a (n_heads, 32) grid = 1024 threadgroups.
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
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

enum { D = 128, HEADS_TOTAL = 64 };
enum {
    Q_CONV_OFFSET = 0,        /* 64*128 channels * 4 taps * 4B = 131072 */
    K_CONV_OFFSET = 131072,
    V_CONV_OFFSET = 262144,
    A_LOG_OFFSET  = 393216,   /* 64 floats */
    DT_BIAS_OFFSET= 393728,   /* 8192 floats = 32768 B */
    NORM_OFFSET   = 426496,   /* 128 floats */
    MODEL_BYTES   = 524288,
};

int main(int argc, char **argv) {
    const uint32_t iters = argc > 1 ? (uint32_t)atoi(argv[1]) : 200;

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
        q_conv[c * 4 + 3] = 1.0f; k_conv[c * 4 + 3] = 1.0f;
        v_conv[c * 4 + 3] = 1.0f; dt[c] = 0.0f;
    }
    for (uint32_t h = 0; h < HEADS_TOTAL; h++) a_log[h] = 0.0f;
    for (uint32_t i = 0; i < D; i++) norm[i] = 1.0f;

    req(ds4_gpu_init(), "gpu init");
    req(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map");

    const uint32_t PROJ_MAX = HEADS_TOTAL * D;
    ds4_gpu_tensor *q  = ds4_gpu_tensor_alloc(PROJ_MAX * sizeof(float));
    ds4_gpu_tensor *k  = ds4_gpu_tensor_alloc(PROJ_MAX * sizeof(float));
    ds4_gpu_tensor *v  = ds4_gpu_tensor_alloc(PROJ_MAX * sizeof(float));
    ds4_gpu_tensor *g  = ds4_gpu_tensor_alloc(PROJ_MAX * sizeof(float));
    ds4_gpu_tensor *og = ds4_gpu_tensor_alloc(PROJ_MAX * sizeof(float));
    ds4_gpu_tensor *b  = ds4_gpu_tensor_alloc(HEADS_TOTAL * sizeof(float));
    ds4_gpu_tensor *o  = ds4_gpu_tensor_alloc(PROJ_MAX * sizeof(float));
    ds4_gpu_tensor *conv = ds4_gpu_tensor_alloc(9ull * PROJ_MAX * sizeof(float));
    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc((uint64_t)HEADS_TOTAL * D * D * sizeof(float));
    req(q && k && v && g && og && b && o && conv && st, "alloc");
    req(ds4_gpu_tensor_fill_f32(q, 0.3f, PROJ_MAX), "fill q");
    req(ds4_gpu_tensor_fill_f32(k, 0.2f, PROJ_MAX), "fill k");
    req(ds4_gpu_tensor_fill_f32(v, 0.1f, PROJ_MAX), "fill v");
    req(ds4_gpu_tensor_fill_f32(g, 0.0f, PROJ_MAX), "fill g");
    req(ds4_gpu_tensor_fill_f32(og, 0.0f, PROJ_MAX), "fill og");
    req(ds4_gpu_tensor_fill_f32(b, 0.0f, HEADS_TOTAL), "fill b");
    req(ds4_gpu_tensor_fill_f32(conv, 0.0f, 9ull * PROJ_MAX), "fill conv");
    req(ds4_gpu_tensor_fill_f32(st, 0.01f, (uint64_t)HEADS_TOTAL * D * D), "fill st");

    const uint32_t head_counts[] = { 8, 16, 32, 64 };
    for (uint32_t hi = 0; hi < 4; hi++) {
        const uint32_t heads = head_counts[hi];
        /* warm */
        for (int w = 0; w < 5; w++) {
            req(ds4_gpu_glm53_kda_decode(o, conv, st, q, k, v, g, b, og,
                    model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET,
                    V_CONV_OFFSET, A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
                    heads, 1, HEADS_TOTAL, 0u, -5.0f, 1e-5f), "warm decode");
        }
        req(ds4_gpu_begin_commands(), "begin");
        const double t0 = now_ms();
        for (uint32_t i = 0; i < iters; i++) {
            req(ds4_gpu_glm53_kda_decode(o, conv, st, q, k, v, g, b, og,
                    model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET,
                    V_CONV_OFFSET, A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
                    heads, 1, HEADS_TOTAL, 0u, -5.0f, 1e-5f), "decode");
        }
        req(ds4_gpu_end_commands(), "end");
        const double dt_ms = now_ms() - t0;
        /* state bytes touched: heads * D * D * 4, read+written */
        const double bytes = (double)heads * D * D * 4.0 * 2.0;
        printf("kda_decode  heads=%2u  tgs=%3u  %8.4f us/call  %7.1f GB/s(state)\n",
               heads, heads, dt_ms * 1000.0 / iters,
               bytes / (dt_ms * 1e-3 / iters) / 1e9);
    }

    /* Same recurrence math on the wide grid: prefill with n_tokens = 1.
     * grid = (n_heads, 32) threadgroups. Includes prepare + output kernels. */
    ds4_gpu_tensor *pconv = ds4_gpu_tensor_alloc(9ull * PROJ_MAX * sizeof(float));
    ds4_gpu_tensor *pst = ds4_gpu_tensor_alloc((uint64_t)HEADS_TOTAL * D * D * sizeof(float));
    req(pconv && pst, "alloc prefill");
    req(ds4_gpu_tensor_fill_f32(pconv, 0.0f, 9ull * PROJ_MAX), "fill pconv");
    req(ds4_gpu_tensor_fill_f32(pst, 0.01f, (uint64_t)HEADS_TOTAL * D * D), "fill pst");

    for (uint32_t hi = 0; hi < 4; hi++) {
        const uint32_t heads = head_counts[hi];
        for (int w = 0; w < 5; w++) {
            req(ds4_gpu_tensor_fill_f32(q, 0.3f, PROJ_MAX), "refill q");
            req(ds4_gpu_tensor_fill_f32(k, 0.2f, PROJ_MAX), "refill k");
            req(ds4_gpu_tensor_fill_f32(v, 0.1f, PROJ_MAX), "refill v");
            req(ds4_gpu_tensor_fill_f32(g, 0.0f, PROJ_MAX), "refill g");
            req(ds4_gpu_glm53_kda_prefill(o, pconv, pst, q, k, v, g, b, og,
                    model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET,
                    V_CONV_OFFSET, A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
                    heads, 1, HEADS_TOTAL, 0u, -5.0f, 1e-5f), "warm prefill1");
        }
        req(ds4_gpu_begin_commands(), "begin2");
        const double t0 = now_ms();
        for (uint32_t i = 0; i < iters; i++) {
            req(ds4_gpu_glm53_kda_prefill(o, pconv, pst, q, k, v, g, b, og,
                    model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET,
                    V_CONV_OFFSET, A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
                    heads, 1, HEADS_TOTAL, 0u, -5.0f, 1e-5f), "prefill1");
        }
        req(ds4_gpu_end_commands(), "end2");
        const double dt_ms = now_ms() - t0;
        const double bytes = (double)heads * D * D * 4.0 * 2.0;
        printf("prefill n=1 heads=%2u  tgs=%3u(rec) %8.4f us/call  %7.1f GB/s(state)\n",
               heads, heads * 32u, dt_ms * 1000.0 / iters,
               bytes / (dt_ms * 1e-3 / iters) / 1e9);
    }

    /* Prefill token scaling: is prepare (n_heads threadgroups, serial token
     * loop) the bottleneck for long chunks? */
    const uint32_t tok_counts[] = { 1, 16, 64, 256, 512 };
    for (uint32_t ti = 0; ti < 5; ti++) {
        const uint32_t toks = tok_counts[ti];
        const uint32_t heads = 64;
        const uint64_t act = (uint64_t)heads * D * toks;
        ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *tv = ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *tg = ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *tog = ds4_gpu_tensor_alloc(act * sizeof(float));
        ds4_gpu_tensor *tb = ds4_gpu_tensor_alloc((uint64_t)toks * heads * sizeof(float));
        ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(act * sizeof(float));
        req(tq && tk && tv && tg && tog && tb && to, "alloc tok");
        req(ds4_gpu_tensor_fill_f32(tq, 0.3f, act), "f tq");
        req(ds4_gpu_tensor_fill_f32(tk, 0.2f, act), "f tk");
        req(ds4_gpu_tensor_fill_f32(tv, 0.1f, act), "f tv");
        req(ds4_gpu_tensor_fill_f32(tg, 0.0f, act), "f tg");
        req(ds4_gpu_tensor_fill_f32(tog, 0.0f, act), "f tog");
        req(ds4_gpu_tensor_fill_f32(tb, 0.0f, (uint64_t)toks * heads), "f tb");
        const uint32_t reps = toks >= 256 ? 10 : (toks >= 64 ? 30 : 100);
        for (int w = 0; w < 2; w++) {
            req(ds4_gpu_glm53_kda_prefill(to, pconv, pst, tq, tk, tv, tg, tb, tog,
                    model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET,
                    V_CONV_OFFSET, A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
                    heads, toks, HEADS_TOTAL, 0u, -5.0f, 1e-5f), "warm prefillN");
        }
        req(ds4_gpu_begin_commands(), "begin3");
        const double t0 = now_ms();
        for (uint32_t i = 0; i < reps; i++) {
            req(ds4_gpu_glm53_kda_prefill(to, pconv, pst, tq, tk, tv, tg, tb, tog,
                    model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET,
                    V_CONV_OFFSET, A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
                    heads, toks, HEADS_TOTAL, 0u, -5.0f, 1e-5f), "prefillN");
        }
        req(ds4_gpu_end_commands(), "end3");
        const double dt_ms = now_ms() - t0;
        printf("prefill toks=%4u heads=64  %9.1f us/call  %7.3f us/token\n",
               toks, dt_ms * 1000.0 / reps, dt_ms * 1000.0 / reps / toks);
        ds4_gpu_tensor_free(to); ds4_gpu_tensor_free(tb);
        ds4_gpu_tensor_free(tog); ds4_gpu_tensor_free(tg);
        ds4_gpu_tensor_free(tv); ds4_gpu_tensor_free(tk);
        ds4_gpu_tensor_free(tq);
    }

    ds4_gpu_cleanup();
    return 0;
}
