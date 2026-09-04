/* Probe: kernel_mul_mv_q8_0_f32_pair vs two standalone Q8_0 matvecs at the
 * GLM 5.3 decode shapes.
 *   DSA: 4096 -> 1536 (attn_q_a) + 4096 -> 512 (attn_kv_a_mqa)
 *   KDA: 4096 -> 8192 (kda_q)    + 4096 -> 8192 (kda_k)
 * Also a bare launch-count probe (4096 -> 32) to price one dispatch.
 * Set DS4_METAL_Q8_MV_NSG=2 to emulate the TP2 nsg the rig uses. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
#include "../ds4_gpu.h"

#define IN_DIM 4096u
#define DS4_TENSOR_Q8_0 8u

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static size_t q8_bytes(uint64_t in_dim, uint64_t out_dim) {
    return (size_t)out_dim * (in_dim / 32u) * 34u;
}

static void fill_q8(unsigned char *p, size_t nblocks, unsigned seed) {
    for (size_t b = 0; b < nblocks; b++) {
        unsigned short d = 0x3000 + (unsigned short)((b + seed) & 0xFF); /* small fp16 */
        memcpy(p + b * 34, &d, 2);
        for (int i = 0; i < 32; i++)
            p[b * 34 + 2 + i] = (unsigned char)(signed char)(((int)(b + seed + i) % 17) - 8);
    }
}

static void run_case(const char *name, void *map, size_t msize,
                     uint64_t off0, uint64_t off1,
                     uint64_t out0_dim, uint64_t out1_dim,
                     ds4_gpu_tensor *x, int iters, int reps) {
    ds4_gpu_tensor *o0 = ds4_gpu_tensor_alloc(out0_dim * sizeof(float));
    ds4_gpu_tensor *o1 = ds4_gpu_tensor_alloc(out1_dim * sizeof(float));
    ds4_gpu_tensor *p0 = ds4_gpu_tensor_alloc(out0_dim * sizeof(float));
    ds4_gpu_tensor *p1 = ds4_gpu_tensor_alloc(out1_dim * sizeof(float));
    if (!o0 || !o1 || !p0 || !p1) { fprintf(stderr, "alloc\n"); exit(1); }

    /* correctness first */
    if (!ds4_gpu_begin_commands()) { fprintf(stderr, "begin\n"); exit(1); }
    if (!ds4_gpu_matmul_quant_tensor(o0, map, msize, off0, DS4_TENSOR_Q8_0,
                                     IN_DIM, out0_dim, x, 1) ||
        !ds4_gpu_matmul_quant_tensor(o1, map, msize, off1, DS4_TENSOR_Q8_0,
                                     IN_DIM, out1_dim, x, 1)) {
        fprintf(stderr, "%s: split matvec failed\n", name); exit(1);
    }
    int pair_ok = ds4_gpu_matmul_q8_0_pair_tensor(p0, p1, map, msize, off0, off1,
                                                  IN_DIM, out0_dim, out1_dim, x, 1);
    if (!ds4_gpu_end_commands()) { fprintf(stderr, "end\n"); exit(1); }
    if (!pair_ok) { printf("%-10s  PAIR REFUSED (predicate rejected)\n", name); return; }

    float *a = malloc(out0_dim * 4), *b = malloc(out0_dim * 4);
    ds4_gpu_tensor_read(o0, 0, a, out0_dim * 4);
    ds4_gpu_tensor_read(p0, 0, b, out0_dim * 4);
    double maxrel = 0;
    for (uint64_t i = 0; i < out0_dim; i++) {
        double d = fabs(a[i] - b[i]) / (fabs(a[i]) + 1e-6);
        if (d > maxrel) maxrel = d;
    }
    ds4_gpu_tensor_read(o1, 0, a, out1_dim * 4);
    ds4_gpu_tensor_read(p1, 0, b, out1_dim * 4);
    for (uint64_t i = 0; i < out1_dim; i++) {
        double d = fabs(a[i] - b[i]) / (fabs(a[i]) + 1e-6);
        if (d > maxrel) maxrel = d;
    }
    free(a); free(b);

    double best[2] = {1e30, 1e30};
    for (int mode = 0; mode < 2; mode++) {
        for (int r = 0; r < reps; r++) {
            if (!ds4_gpu_begin_commands()) exit(1);
            double t0 = now_ms();
            for (int i = 0; i < iters; i++) {
                if (mode == 0) {
                    ds4_gpu_matmul_quant_tensor(o0, map, msize, off0, DS4_TENSOR_Q8_0,
                                                IN_DIM, out0_dim, x, 1);
                    ds4_gpu_matmul_quant_tensor(o1, map, msize, off1, DS4_TENSOR_Q8_0,
                                                IN_DIM, out1_dim, x, 1);
                } else {
                    ds4_gpu_matmul_q8_0_pair_tensor(p0, p1, map, msize, off0, off1,
                                                    IN_DIM, out0_dim, out1_dim, x, 1);
                }
            }
            if (!ds4_gpu_end_commands()) exit(1);
            double dt = now_ms() - t0;
            if (r >= reps / 4 && dt < best[mode]) best[mode] = dt;
        }
    }
    printf("%-10s out %5llu+%5llu  split %7.2f us/pair   pair %7.2f us/pair   "
           "delta %+6.2f us (%+5.1f%%)  maxrel %.2e\n",
           name, (unsigned long long)out0_dim, (unsigned long long)out1_dim,
           best[0] * 1000.0 / iters, best[1] * 1000.0 / iters,
           (best[1] - best[0]) * 1000.0 / iters,
           100.0 * (best[1] - best[0]) / best[0], maxrel);
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    const int reps  = argc > 2 ? atoi(argv[2]) : 12;
    if (!ds4_gpu_init()) { fprintf(stderr, "no gpu\n"); return 1; }

    /* layout: [0]=1536 [A]=512 [B]=8192 [C]=8192 [D]=32 [E]=32 */
    size_t s1536 = q8_bytes(IN_DIM, 1536), s512 = q8_bytes(IN_DIM, 512);
    size_t s8192 = q8_bytes(IN_DIM, 8192), s32 = q8_bytes(IN_DIM, 32);
    size_t off_1536 = 0;
    size_t off_512  = (off_1536 + s1536 + 0xFFFF) & ~(size_t)0xFFFF;
    size_t off_8192a= (off_512  + s512  + 0xFFFF) & ~(size_t)0xFFFF;
    size_t off_8192b= (off_8192a+ s8192 + 0xFFFF) & ~(size_t)0xFFFF;
    size_t off_32a  = (off_8192b+ s8192 + 0xFFFF) & ~(size_t)0xFFFF;
    size_t off_32b  = (off_32a  + s32   + 0xFFFF) & ~(size_t)0xFFFF;
    size_t msize    = (off_32b + s32 + 0xFFFFF) & ~(size_t)0xFFFFF;

    void *map = mmap(NULL, msize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    fill_q8((unsigned char *)map + off_1536, s1536 / 34, 1);
    fill_q8((unsigned char *)map + off_512,  s512  / 34, 2);
    fill_q8((unsigned char *)map + off_8192a,s8192 / 34, 3);
    fill_q8((unsigned char *)map + off_8192b,s8192 / 34, 4);
    fill_q8((unsigned char *)map + off_32a,  s32   / 34, 5);
    fill_q8((unsigned char *)map + off_32b,  s32   / 34, 6);

    if (!ds4_gpu_set_model_map_range(map, msize, 0, msize, msize)) {
        fprintf(stderr, "map views failed\n"); return 1;
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)IN_DIM * sizeof(float));
    static float xh[IN_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++) xh[i] = 0.01f * (float)((i % 13) + 1);
    ds4_gpu_tensor_write(x, 0, xh, sizeof(xh));

    printf("nsg env DS4_METAL_Q8_MV_NSG=%s\n",
           getenv("DS4_METAL_Q8_MV_NSG") ? getenv("DS4_METAL_Q8_MV_NSG") : "(unset)");
    run_case("DSA q+kv", map, msize, off_1536, off_512, 1536, 512, x, iters, reps);
    run_case("KDA q+k",  map, msize, off_8192a, off_8192b, 8192, 8192, x, iters, reps);
    run_case("tiny",     map, msize, off_32a, off_32b, 32, 32, x, iters * 4, reps);
    return 0;
}
