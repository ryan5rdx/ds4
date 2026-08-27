#define _DARWIN_C_SOURCE

/*
 * KV re-read amplification probe (model-free).
 *
 * The decode FlashAttention vec kernel is dispatched as grid (1, n_head, NWG)
 * and binds the SAME staged KV buffer to both the K and the V argument
 * (ds4_metal.m:29375-29376).  So one staged KV row is read
 *   n_head (32 under TP2)  x  2 (once as K, once as V)  = 64 times per layer.
 *
 * Question this answers: does that amplification cost real time, or is it
 * absorbed by cache?  Method: hold n_keys fixed and sweep n_head.  If time is
 * linear in n_head the re-reads are paid; if it saturates they are cached.
 *
 *   cc -O3 -I. -c -o /tmp/kvamp.o speed-bench/kv_amp_probe.c
 *   cc -O3 -o /tmp/kvamp /tmp/kvamp.o ds4_metal.o -lm -pthread \
 *      -framework Foundation -framework Metal
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    MAX_HEAD = 64,
    HEAD_DIM = 512,
    N_RAW = 128,
    RAW_CAP = 4097,
    MAX_COMP = 1024,
};

bool ds4_log_is_tty(FILE *fp);
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint16_t f2h(float f) {
    union { float f; uint32_t u; } v = {.f = f};
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float rnd(uint32_t *s, float scale) {
    *s = *s * 1664525u + 1013904223u;
    return (float)((int32_t)((*s >> 8) & 0xffffu) - 32768) * (scale / 32768.0f);
}

static int noop_tp(void *ud, uint32_t l, uint32_t g, uint64_t s) {
    (void)ud; (void)l; (void)g; (void)s; return 1;
}

static void *g_map;
static ds4_gpu_tensor *g_q, *g_raw, *g_comp, *g_heads;

static int encode(uint32_t n_head, uint32_t n_comp) {
    return ds4_gpu_attention_decode_heads_tensor(
        g_heads, g_map, (uint64_t)getpagesize(), 0, g_q, g_raw,
        N_RAW, RAW_CAP, 0,
        n_comp ? g_comp : NULL, 1, n_comp, NULL, 0, n_head, HEAD_DIM);
}

int main(int argc, char **argv) {
    int tokens = 64;
    int samples = 8;
    if (argc > 1) tokens = atoi(argv[1]);
    if (argc > 2) samples = atoi(argv[2]);

    (void)setenv("DS4_METAL_NO_RESIDENCY", "1", 1);
    (void)setenv("DS4_METAL_NO_MODEL_WARMUP", "1", 1);
    (void)setenv("DS4_TP_NO_KEEPALIVE", "1", 1);
    if (!ds4_gpu_init()) { fprintf(stderr, "init failed\n"); return 1; }

    const size_t page = (size_t)getpagesize();
    if (posix_memalign(&g_map, page, page) != 0) return 1;
    memset(g_map, 0, page);
    float *sinks = g_map;
    for (int h = 0; h < MAX_HEAD; h++) sinks[h] = (float)((h * 17) % 23 - 11) / 32.0f;

    const uint64_t qn = (uint64_t)MAX_HEAD * HEAD_DIM;
    const uint64_t rn = (uint64_t)RAW_CAP * HEAD_DIM;
    const uint64_t cn = (uint64_t)MAX_COMP * HEAD_DIM;
    g_q = ds4_gpu_tensor_alloc(qn * sizeof(float));
    g_raw = ds4_gpu_tensor_alloc(rn * sizeof(float));
    g_comp = ds4_gpu_tensor_alloc(cn * sizeof(uint16_t));
    g_heads = ds4_gpu_tensor_alloc(qn * sizeof(float));
    if (!g_q || !g_raw || !g_comp || !g_heads) return 1;

    float *q = malloc(qn * sizeof(float));
    float *raw = malloc(rn * sizeof(float));
    uint16_t *comp = malloc(cn * sizeof(uint16_t));
    uint32_t s = 0x4d595df4u;
    for (uint64_t i = 0; i < qn; i++) q[i] = rnd(&s, 0.75f);
    for (uint64_t i = 0; i < rn; i++) raw[i] = rnd(&s, 0.60f);
    for (uint64_t i = 0; i < cn; i++) comp[i] = f2h(rnd(&s, 0.60f));
    ds4_gpu_tensor_write(g_q, 0, q, qn * sizeof(float));
    ds4_gpu_tensor_write(g_raw, 0, raw, rn * sizeof(float));
    ds4_gpu_tensor_write(g_comp, 0, comp, cn * sizeof(uint16_t));
    free(q); free(raw); free(comp);
    ds4_gpu_set_quality(false);
    if (!ds4_gpu_set_model_map(g_map, page)) return 1;
    if (!ds4_gpu_tp_init(0, NULL, 0, noop_tp, NULL)) return 1;

    const uint32_t comps[] = {0, 128, 256, 384, 512, 896};
    const uint32_t heads[] = {4, 8, 16, 32};

    printf("tokens/sample=%d samples=%d\n", tokens, samples);
    printf("%-8s %-6s %-6s %10s %12s %12s\n",
           "n_keys", "heads", "nwg", "ms/call", "us/head", "impl GB/s");

    for (size_t ci = 0; ci < sizeof(comps)/sizeof(comps[0]); ci++) {
        const uint32_t n_comp = comps[ci];
        const uint32_t n_keys = N_RAW + n_comp;
        double base = 0.0;
        for (size_t hi = 0; hi < sizeof(heads)/sizeof(heads[0]); hi++) {
            const uint32_t nh = heads[hi];
            for (int w = 0; w < 2; w++) {
                if (!ds4_gpu_begin_commands()) return 1;
                for (int t = 0; t < 43; t++) if (!encode(nh, n_comp)) return 1;
                if (!ds4_gpu_end_commands()) return 1;
            }
            double best = 1e30;
            for (int it = 0; it < samples; it++) {
                const double t0 = now_ms();
                for (int t = 0; t < tokens; t++) {
                    if (!ds4_gpu_begin_commands()) return 1;
                    for (int l = 0; l < 43; l++) if (!encode(nh, n_comp)) return 1;
                    if (!ds4_gpu_end_commands()) return 1;
                }
                const double dt = (now_ms() - t0) / (double)tokens / 43.0;
                if (dt < best) best = dt;
            }
            /* KV bytes the current geometry actually issues: every head reads
             * every staged key row twice (K arg and V arg are the same bind). */
            const double kv_bytes = (double)n_keys * HEAD_DIM * 2.0 /*f16*/
                                  * 2.0 /*K then V*/ * (double)nh;
            if (hi == 0) base = best;
            printf("%-8u %-6u %-6s %10.4f %12.3f %12.1f\n",
                   n_keys, nh, "auto", best, best * 1000.0 / (double)nh,
                   kv_bytes / (best * 1.0e-3) / 1.0e9);
            (void)base;
        }
        printf("\n");
    }
    ds4_gpu_tp_shutdown();
    return 0;
}
