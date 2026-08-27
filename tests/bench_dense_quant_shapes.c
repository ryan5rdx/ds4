#define _DARWIN_C_SOURCE

/* Q8_0 vs Q4_K vs Q4_0 decode-matvec WEIGHT throughput at the DS4-Flash
 * attention-projection shapes, per rank under TP=2.
 *
 * Why: switching attn_q_b / attn_output_a / attn_output_b / shared / output
 * from Q8_0 (1.0625 B/w) to a 4-bit family (0.5625 B/w) removes 47% of those
 * bytes.  Whether that removes time depends entirely on the format's
 * weights/second, not its bits/weight -- MXFP4's measured 687 Gw/s against
 * Q8_0's ~547 Gw/s on the rig is only a 1.26x, not 2x.  This harness measures
 * the same quantity for the formats the loader already accepts for these
 * tensors (tensor_expect_dense_quant_layout, ds4.c:4427 -> q8_0/q4_K/q4_0).
 *
 * Derived from tests/bench_q8_attn_shapes.c and keeps its three traps:
 * distinct destination per repeat, distinct weight region per repeat,
 * file-backed mmap.
 *
 *   cc -O3 -ffast-math -mcpu=native -I. -c tests/bench_dense_quant_shapes.c \
 *      -o tests/bench_dense_quant_shapes.o
 *   cc -O3 -o tests/bench_dense_quant_shapes tests/bench_dense_quant_shapes.o \
 *      ds4_metal.o -lm -pthread -framework Foundation -framework Metal
 *   DS4_BENCH_MAP_GB=16 ./tests/bench_dense_quant_shapes 800
 */

#include "ds4_gpu.h"

#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
void ds4_log_line(int type, const char *fmt, ...) { (void)type; (void)fmt; }

#define TY_Q4_0 2u
#define TY_Q8_0 8u
#define TY_Q4_K 12u

typedef struct { const char *name; uint32_t ty; int blk; int tsz; } fmt;
static const fmt g_fmts[] = {
    {"q8_0", TY_Q8_0,  32,  34},
    {"q4_0", TY_Q4_0,  32,  18},
    {"q4_K", TY_Q4_K, 256, 144},
};

static double g_peak = 800.0;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static int cmpd(const void *x, const void *y) {
    const double a = *(const double *)x, b = *(const double *)y;
    return a < b ? -1 : (a > b ? 1 : 0);
}

/* Fill the whole map with valid blocks of one format.  Numerics are
 * irrelevant here; only the traffic and the unpack instruction stream are. */
static void fill(void *map, uint64_t size, const fmt *f) {
    uint8_t *p = map;
    uint32_t s = 12345;
    for (uint64_t b = 0; b + (uint64_t)f->tsz <= size; b += (uint64_t)f->tsz) {
        if (f->ty == TY_Q8_0) {
            p[b] = 0x00; p[b + 1] = 0x3C;                 /* d = half 1.0 */
            for (int i = 0; i < 32; i++) {
                s = s * 1664525u + 1013904223u;
                p[b + 2 + i] = (uint8_t)((s >> 20) & 0x3F);
            }
        } else if (f->ty == TY_Q4_0) {
            p[b] = 0x00; p[b + 1] = 0x3C;                 /* d = half 1.0 */
            for (int i = 0; i < 16; i++) {
                s = s * 1664525u + 1013904223u;
                p[b + 2 + i] = (uint8_t)((s >> 20) & 0xFF);
            }
        } else {                                          /* q4_K, 144 B/256 */
            p[b] = 0x00; p[b + 1] = 0x3C;                 /* d    = half 1.0 */
            p[b + 2] = 0x00; p[b + 3] = 0xB8;             /* dmin = half -0.5 */
            for (int i = 0; i < 12; i++) p[b + 4 + i] = 0x21;
            for (int i = 0; i < 128; i++) {
                s = s * 1664525u + 1013904223u;
                p[b + 16 + i] = (uint8_t)((s >> 20) & 0xFF);
            }
        }
    }
}

static void bench(void *map, uint64_t map_size, const fmt *f,
                  const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (in_dim % (uint64_t)f->blk) { printf("   %-6s %-20s SKIP (k%%blk)\n", f->name, label); return; }
    const uint64_t wbytes = out_dim * (in_dim / (uint64_t)f->blk) * (uint64_t)f->tsz;
    const double gweights = (double)in_dim * (double)out_dim / 1e9;
    uint64_t regions = map_size / wbytes;
    if (regions == 0) { printf("   %-6s %-20s SKIPPED\n", f->name, label); return; }
    if (regions > 1024ull) regions = 1024ull;
    const int N = (int)regions;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    float *xf = ds4_gpu_tensor_contents(x);
    for (uint64_t i = 0; i < in_dim; i++) xf[i] = 0.01f * (float)(i % 13);
    ds4_gpu_tensor **out = malloc((size_t)N * sizeof(*out));
    for (int i = 0; i < N; i++) out[i] = ds4_gpu_tensor_alloc(out_dim * sizeof(float));

    for (int w = 0; w < 3; w++) {
        ds4_gpu_begin_commands();
        for (int i = 0; i < N; i++)
            ds4_gpu_matmul_quant_tensor(out[i], map, map_size,
                                        ((uint64_t)i % regions) * wbytes,
                                        f->ty, in_dim, out_dim, x, 1);
        ds4_gpu_end_commands();
    }
    double t[11];
    for (int r = 0; r < 11; r++) {
        const double t0 = now_ms();
        ds4_gpu_begin_commands();
        for (int i = 0; i < N; i++)
            ds4_gpu_matmul_quant_tensor(out[i], map, map_size,
                                        ((uint64_t)i % regions) * wbytes,
                                        f->ty, in_dim, out_dim, x, 1);
        ds4_gpu_end_commands();
        t[r] = now_ms() - t0;
    }
    qsort(t, 11, sizeof(double), cmpd);
    const double per = t[0] / N;
    const double gbs = (double)wbytes / (per / 1000.0) / 1e9;
    const double gws = gweights / (per / 1000.0);
    printf("   %-6s %-20s in=%5llu out=%6llu %7.2f MB (%4d reps) | "
           "%8.4f ms %6.1f GB/s (%4.1f%% peak) %6.1f Gweights/s\n",
           f->name, label,
           (unsigned long long)in_dim, (unsigned long long)out_dim,
           (double)wbytes / 1e6, N, per, gbs, gbs / g_peak * 100.0, gws);
    fflush(stdout);
    for (int i = 0; i < N; i++) ds4_gpu_tensor_free(out[i]);
    free(out);
    ds4_gpu_tensor_free(x);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) g_peak = atof(argv[1]);
    if (!ds4_gpu_init()) { fprintf(stderr, "init failed\n"); return 1; }

    double map_gb = 2.0;
    const char *e = getenv("DS4_BENCH_MAP_GB");
    if (e && e[0]) { const double v = atof(e); if (v >= 0.25 && v <= 96.0) map_gb = v; }
    const uint64_t map_size = (uint64_t)(map_gb * 1073741824.0) & ~(uint64_t)0xFFFFF;

    const char *path = "/tmp/ds4_bench_dense_quant.bin";
    int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) { perror("open"); return 1; }
    if (ftruncate(fd, (off_t)map_size) != 0) { perror("ftruncate"); return 1; }
    void *map = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }

    printf("\n== dense-quant decode matvec, DS4-Flash attention shapes ==\n");
    printf("   map %.2f GiB, peak reference %.0f GB/s\n\n", map_gb, g_peak);

    for (size_t fi = 0; fi < sizeof(g_fmts) / sizeof(g_fmts[0]); fi++) {
        const fmt *f = &g_fmts[fi];
        fprintf(stderr, "filling %llu MB with %s...\n",
                (unsigned long long)(map_size >> 20), f->name);
        fill(map, map_size, f);
        msync(map, map_size, MS_SYNC);
        if (!ds4_gpu_set_model_map(map, map_size)) { fprintf(stderr, "map failed\n"); return 1; }
        for (int pass = 0; pass < 2; pass++) {
            bench(map, map_size, f, "out_b  k=4096 n=4096",  4096, 4096);
            bench(map, map_size, f, "out_a  k=4096 n=4096",  4096, 4096);
            bench(map, map_size, f, "q_b    k=1024 n=16384", 1024, 16384);
            bench(map, map_size, f, "shexp  k=4096 n=1024",  4096, 1024);
            bench(map, map_size, f, "head   k=4096 n=64640", 4096, 64640);
        }
        printf("\n");
    }
    return 0;
}
