#define _DARWIN_C_SOURCE

/* Q8_0 decode-matvec bandwidth at the exact DS4-Flash attention-projection
 * shapes, per rank under the 2-way tensor-parallel split.
 *
 * Why: the routed MXFP4 MoE moves 2.01 GB per rank per token, but the Q8_0
 * attention projections move more (q_a + q_b/2 + kv + out_a/2 + out_b/2 =
 * 60.2 MB per layer, 2.59 GB over 43 layers).  Nothing in tree measures how
 * close that dense stream runs to the memory ceiling.
 *
 * Two traps this harness avoids:
 *   - every repeat writes a DIFFERENT destination tensor (write-after-write
 *     hazards serialise otherwise; see tests/bench_indexer_score.c);
 *   - every repeat reads a DIFFERENT weight region, so the working set stays
 *     far larger than the system level cache (48 MB on M1 Max).  Re-reading
 *     one 17.8 MB matrix measures the SLC, not DRAM.
 *   - the weights are FILE-backed.  An anonymous mmap handed to
 *     newBufferWithBytesNoCopy took 42 s to warm 320 MB here and then read at
 *     a pathological rate; the engine mmaps the GGUF, so do the same.
 *
 *   cc -O3 -ffast-math -mcpu=native -I. -c tests/bench_q8_attn_shapes.c
 *   cc -O3 -o b bench_q8_attn_shapes.o ds4_metal.o -lm -pthread \
 *      -framework Foundation -framework Metal
 *   ./b 400        # argv[1] = peak GB/s of this machine
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

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int cmpd(const void *x, const void *y) {
    const double a = *(const double *)x, b = *(const double *)y;
    return a < b ? -1 : (a > b ? 1 : 0);
}

static double g_peak_gbs = 400.0;
static int g_sweep_all;

/* DS4_BENCH_CHAIN=1.
 *
 * The default loop issues N matvecs that all read the same x and write
 * distinct outputs, so they are mutually independent and the GPU overlaps
 * them freely. Decode is the opposite: a strict dependency chain where each
 * kernel consumes the previous one's output, so every dispatch must drain
 * before the next begins and the tail wave of threadgroups runs with most of
 * the machine idle.
 *
 * That difference is the leading explanation for the ~5.4 ms/token by which
 * in-engine kernel time exceeds what these isolated rates predict. Chaining
 * the harness reproduces the dependency structure at the same shapes: if
 * chained throughput lands near the in-engine figure, the isolated rate was
 * never reachable in a decode graph and the gap is structural rather than
 * recoverable.
 *
 * Only square shapes can chain (out feeds in), so the others fall back to the
 * independent loop and say so. */
static int g_chain_mode;

static void bench(void *map, uint64_t map_size,
                  const char *label, uint64_t in_dim, uint64_t out_dim,
                  int N, double per_token_count) {
    const uint64_t wbytes = out_dim * (in_dim / 32) * 34;
    uint64_t regions = map_size / wbytes;
    if (regions == 0) { printf("   %-24s SKIPPED\n", label); return; }
    /* The default 512 MB map with N=28 gives a ~499 MB working set, which is
     * far smaller than the ~76 GiB resident shard the engine streams. Page
     * table and TLB behaviour at 16 KiB pages differs by two orders of
     * magnitude between the two, so this harness can flatter the kernel.
     * DS4_BENCH_MAP_GB grows the map; N then follows the region count so the
     * whole map is actually swept rather than just its first 28 slices. */
    const uint64_t region_cap = g_sweep_all ? 8192ull : 64ull;
    if (regions > region_cap) regions = region_cap;
    if (g_sweep_all) N = (int)regions;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    float *xf = ds4_gpu_tensor_contents(x);
    for (uint64_t i = 0; i < in_dim; i++) xf[i] = 0.01f * (float)(i % 13);

    ds4_gpu_tensor **out = malloc((size_t)N * sizeof(*out));
    for (int i = 0; i < N; i++) out[i] = ds4_gpu_tensor_alloc(out_dim * sizeof(float));

    /* Chaining needs out to be a valid input, so the shape must be square. */
    const int chained = g_chain_mode && in_dim == out_dim;

    for (int w = 0; w < 4; w++) {
        ds4_gpu_begin_commands();
        for (int i = 0; i < N; i++)
            ds4_gpu_matmul_q8_0_tensor(out[i], map, map_size,
                                       ((uint64_t)i % regions) * wbytes,
                                       in_dim, out_dim,
                                       (chained && i > 0) ? out[i - 1] : x, 1);
        ds4_gpu_end_commands();
    }

    double t[15];
    for (int r = 0; r < 15; r++) {
        const double t0 = now_ms();
        ds4_gpu_begin_commands();
        for (int i = 0; i < N; i++)
            ds4_gpu_matmul_q8_0_tensor(out[i], map, map_size,
                                       ((uint64_t)i % regions) * wbytes,
                                       in_dim, out_dim,
                                       (chained && i > 0) ? out[i - 1] : x, 1);
        ds4_gpu_end_commands();
        t[r] = now_ms() - t0;
    }
    qsort(t, 15, sizeof(double), cmpd);
    const double per_min = t[0] / N;
    const double gbs = (double)wbytes / (per_min / 1000.0) / 1e9;
    printf("   %-24s in=%5llu out=%6llu %7.2f MB (%2llu regions) %-6s | "
           "%8.4f ms  %6.1f GB/s = %4.1f%% peak | x%.0f = %6.3f ms/token\n",
           label,
           (unsigned long long)in_dim, (unsigned long long)out_dim,
           (double)wbytes / 1e6, (unsigned long long)regions,
           g_chain_mode ? (chained ? "CHAIN" : "indep*") : "indep",
           per_min, gbs, gbs / g_peak_gbs * 100.0,
           per_token_count, per_min * per_token_count);
    fflush(stdout);

    for (int i = 0; i < N; i++) ds4_gpu_tensor_free(out[i]);
    free(out);
    ds4_gpu_tensor_free(x);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) g_peak_gbs = atof(argv[1]);
    if (!ds4_gpu_init()) { fprintf(stderr, "init failed\n"); return 1; }

    double map_gb = 0.5;
    const char *map_env = getenv("DS4_BENCH_MAP_GB");
    if (map_env && map_env[0]) {
        const double v = atof(map_env);
        if (v >= 0.125 && v <= 96.0) { map_gb = v; g_sweep_all = 1; }
    }
    const uint64_t map_size =
        (uint64_t)(map_gb * 1024.0 * 1024.0 * 1024.0) & ~(uint64_t)0xFFFFF;
    g_chain_mode = getenv("DS4_BENCH_CHAIN") != NULL;
    const char *path = "/tmp/ds4_bench_q8_weights.bin";
    fprintf(stderr, "map %.2f GiB, sweep_all=%d, chain=%d\n",
            map_gb, g_sweep_all, g_chain_mode);
    int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) { perror("open"); return 1; }
    if (ftruncate(fd, (off_t)map_size) != 0) { perror("ftruncate"); return 1; }
    void *map = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }

    fprintf(stderr, "filling %llu MB of q8_0 blocks...\n",
            (unsigned long long)(map_size >> 20));
    uint8_t *p = map;
    uint32_t s = 12345;
    for (uint64_t b = 0; b + 34 <= map_size; b += 34) {
        p[b] = 0x00; p[b + 1] = 0x3C;              /* half 1.0 */
        for (int i = 0; i < 32; i++) {
            s = s * 1664525u + 1013904223u;
            p[b + 2 + i] = (uint8_t)((s >> 20) & 0x3F);
        }
    }
    msync(map, map_size, MS_SYNC);
    fprintf(stderr, "mapping...\n");
    if (!ds4_gpu_set_model_map(map, map_size)) { fprintf(stderr, "map failed\n"); return 1; }
    fprintf(stderr, "mapped\n");

    printf("\n== Q8_0 decode matvec, DS4-Flash shapes, per rank under TP=2 ==\n");
    printf("   peak reference %.0f GB/s\n\n", g_peak_gbs);
    /* Characterise the k dimension at a fixed 17.83 MB matrix size: the
     * attention q_b projection under TP is k=1024, which is only 32 q8_0
     * blocks per row -- one block per simd lane, then a full simd_sum. */
    for (int pass = 0; pass < 3; pass++) {
        printf("   --- pass %d ---\n", pass);
        bench(map, map_size, "k=4096  4096->4096",    4096,  4096, 28, 43);
        bench(map, map_size, "k=2048  2048->8192",    2048,  8192, 28, 43);
        bench(map, map_size, "k=1024  1024->16384",   1024, 16384, 28, 43);
        bench(map, map_size, "k=512    512->32768",    512, 32768, 28, 43);
        bench(map, map_size, "k=8192  8192->2048",    8192,  2048, 28, 43);
        printf("\n");
    }
    printf("\n");
    return 0;
}
