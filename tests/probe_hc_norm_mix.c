/* Probe: cost of the GLM 5.3 hc_pre front half (flat RMSNorm over 16384 +
 * BF16 hc_mix matvec 16384->24) versus the matvec alone, batched in one
 * command buffer the way the decode graph runs it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
#include "../ds4_gpu.h"

#define HC_DIM  16384u
#define HC_MIX  24u

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 90;
    const int reps  = argc > 2 ? atoi(argv[2]) : 20;

    if (!ds4_gpu_init()) { fprintf(stderr, "no gpu\n"); return 1; }

    size_t wbytes = (size_t)HC_DIM * HC_MIX * sizeof(unsigned short);
    size_t msize  = (wbytes + 0xFFFFF) & ~(size_t)0xFFFFF;
    void *map = mmap(NULL, msize, PROT_READ|PROT_WRITE,
                     MAP_PRIVATE|MAP_ANON, -1, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    unsigned short *w = (unsigned short *)map;
    for (size_t i = 0; i < (size_t)HC_DIM * HC_MIX; i++) w[i] = 0x3F00; /* bf16 ~0.5 */

    if (!ds4_gpu_set_model_map_range(map, msize, 0, msize, msize)) {
        fprintf(stderr, "map views failed\n"); return 1;
    }

    ds4_gpu_tensor *x    = ds4_gpu_tensor_alloc((uint64_t)HC_DIM * sizeof(float));
    ds4_gpu_tensor *flat = ds4_gpu_tensor_alloc((uint64_t)HC_DIM * sizeof(float));
    ds4_gpu_tensor *mix  = ds4_gpu_tensor_alloc((uint64_t)HC_MIX * sizeof(float));
    if (!x || !flat || !mix) { fprintf(stderr, "alloc\n"); return 1; }
    static float xh[HC_DIM];
    for (uint32_t i = 0; i < HC_DIM; i++) xh[i] = 0.01f * (float)((i % 17) + 1);
    ds4_gpu_tensor_write(x, 0, xh, sizeof(xh));

    for (int mode = 0; mode < 3; mode++) {
        const char *name = mode == 0 ? "norm only" :
                           mode == 1 ? "matvec only" : "norm + matvec";
        double best = 1e30, sum = 0;
        for (int r = 0; r < reps; r++) {
            if (!ds4_gpu_begin_commands()) { fprintf(stderr, "begin\n"); return 1; }
            double t0 = now_ms();
            for (int i = 0; i < iters; i++) {
                if (mode != 1) {
                    if (!ds4_gpu_rms_norm_plain_tensor(flat, x, HC_DIM, 1e-5f)) {
                        fprintf(stderr, "norm failed\n"); return 1;
                    }
                }
                if (mode != 0) {
                    if (!ds4_gpu_glm53_matmul_bf16(mix, map, msize, 0,
                                                   HC_DIM, HC_MIX,
                                                   mode == 1 ? x : flat, 1)) {
                        fprintf(stderr, "mv failed\n"); return 1;
                    }
                }
            }
            if (!ds4_gpu_end_commands()) { fprintf(stderr, "end\n"); return 1; }
            double dt = now_ms() - t0;
            if (r >= reps/4) { sum += dt; if (dt < best) best = dt; }
        }
        int counted = reps - reps/4;
        printf("%-14s  %d calls: best %.3f ms (%.2f us/call)   mean %.3f ms (%.2f us/call)\n",
               name, iters, best, best*1000.0/iters,
               sum/counted, sum/counted*1000.0/iters);
    }
    return 0;
}
