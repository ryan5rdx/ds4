/*
 * WQ8 async staging (10d) correctness gate.
 *
 * The arms must be BIT-IDENTICAL. The staged path reads the same bytes in the
 * same order and scales them the same way -- only the address space changes --
 * so any divergence is a bug in the staging, never a tradeoff. Nothing at
 * runtime can check this: production runs one arm.
 *
 * It also reports kernel time per arm, which is the other half of 10d. The
 * occupancy cost is priced separately by 10c (DS4_WQ8_TG_PROBE); neither
 * number means much alone, because the async arms pay that cost themselves.
 *
 * Build: see the Makefile target test-wq8-async.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { int ne00, ne01, ne02; unsigned long nb00, nb01, nb02, nb03;
                 int ne10, ne11, ne12; unsigned long nb10, nb11, nb12, nb13;
                 int ne0, ne1, nr0; short r2, r3; } mv_args;

enum { QK = 32, BLK = 34, NSG = 2, NR0 = 2, REPS = 25 };

static id<MTLComputePipelineState> mk(id<MTLDevice> d, id<MTLLibrary> lib,
                                      const char *n, short nsg) {
    MTLFunctionConstantValues *c = [[MTLFunctionConstantValues alloc] init];
    [c setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    BOOL compact = NO; [c setConstantValue:&compact type:MTLDataTypeBool atIndex:602];
    NSError *e = nil;
    id<MTLFunction> f = [lib newFunctionWithName:@(n) constantValues:c error:&e];
    if (!f) return nil;
    return [d newComputePipelineStateWithFunction:f error:&e];
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y);
}

int main(int argc, const char **argv) { @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: %s <modern.metal> <private.metallib>\n", argv[0]); return 2; }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *e = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&e];
    id<MTLLibrary> mod = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&e];
    if (!mod) { printf("VOID: modern compile failed: %s\n", e.localizedDescription.UTF8String); return 2; }
    id<MTLLibrary> priv = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&e];
    if (!priv) { printf("VOID: private metallib failed: %s\n", e.localizedDescription.UTF8String); return 2; }

    /* Production wide shape: K = 4096 (nb = 128 blocks), 2048 output rows. */
    const int ne00 = 4096, ne01 = 16384;
    const int nb = ne00 / QK;
    const size_t w_bytes = (size_t)ne01 * nb * BLK;

    id<MTLBuffer> w = [dev newBufferWithLength:w_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> x = [dev newBufferWithLength:(size_t)ne00 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> o = [dev newBufferWithLength:(size_t)ne01 * sizeof(float) options:MTLResourceStorageModeShared];
    unsigned char *wp = w.contents; float *xp = x.contents;
    /* Deterministic, and deliberately NOT smooth: a pattern that varies per
     * block and per lane is what catches an off-by-one in the staged index,
     * where a constant fill would pass on any of them. */
    unsigned s = 12345u;
    for (size_t i = 0; i < w_bytes; i++) { s = s * 1664525u + 1013904223u; wp[i] = (unsigned char)(s >> 24); }
    for (int i = 0; i < ne00; i++) { s = s * 1664525u + 1013904223u; xp[i] = (float)((int)(s >> 20) % 2001 - 1000) / 250.0f; }

    mv_args a = { .ne00 = ne00, .ne01 = ne01, .ne02 = 1,
                  .nb00 = BLK, .nb01 = (unsigned long)nb * BLK,
                  .nb02 = (unsigned long)ne01 * nb * BLK, .nb03 = (unsigned long)ne01 * nb * BLK,
                  .ne10 = ne00, .ne11 = 1, .ne12 = 1,
                  .nb10 = 4, .nb11 = (unsigned long)ne00 * 4,
                  .nb12 = (unsigned long)ne00 * 4, .nb13 = (unsigned long)ne00 * 4,
                  .ne0 = ne01, .ne1 = 1, .nr0 = NR0, .r2 = 1, .r3 = 1 };

    /* The middle two arms are the SHIPPING kernel carrying the async arms'
     * threadgroup footprint and never touching it. They separate the two things
     * the async arms pay for at once -- lost residency, and the staging work
     * itself -- which is 10c's question answered here for this kernel. */
    struct { const char *name; id<MTLLibrary> lib; size_t extra; } arms[] = {
        { "kernel_mul_mv_q8_0_f32",            mod,  0 },
        { "kernel_mul_mv_q8_0_f32",            priv, 0 },
        { "kernel_mul_mv_q8_0_f32",            mod,  (size_t)2*NSG*NR0*2*(8*BLK) },
        { "kernel_mul_mv_q8_0_f32",            mod,  (size_t)2*NSG*NR0*4*(8*BLK) },
        /* The PIVOT: device -> register, no threadgroup memory, shipping
         * corpus. Separate flags so the four-arm run attributes the gain. */
        { "kernel_mul_mv_q8_0_f32_packed",     mod,  0 },
        { "kernel_mul_mv_q8_0_f32_spec",       mod,  0 },
        { "kernel_mul_mv_q8_0_f32_both",       mod,  0 },
        /* 2 buffers x NSG simdgroups x NR0 rows x C runs x (NQ blocks x 34 B).
         * Getting this wrong by a factor of two is what hung the GPU the first
         * time -- the kernel wrote past the end of the threadgroup allocation.
         * It must match ds4_gpu_wq8_async_stage_bytes() exactly. */
        { "kernel_mul_mv_q8_0_f32_sgasync2",   priv, (size_t)2*NSG*NR0*2*(8*BLK) },
        { "kernel_mul_mv_q8_0_f32_sgasync4",   priv, (size_t)2*NSG*NR0*4*(8*BLK) },
    };
    const char *label[] = { "shipping (modern)", "14.2 clone, non-async",
                            "shipping +4352 B unused", "shipping +8704 B unused",
                            "packed loads", "shape-specialised", "packed+spec",
                            "async C=2", "async C=4" };
    const int n_arms = 9;
    float *ref = malloc((size_t)ne01 * sizeof(float));
    double ref_ms = 0.0;
    int fails = 0;

    id<MTLCommandQueue> q = [dev newCommandQueue];

    /* INTERLEAVED, not arm-at-a-time. Each dispatch here is a fraction of a
     * millisecond and the GPU's clock state moves over the run, so measuring
     * all of arm 0 then all of arm 1 aliases that drift straight onto the
     * result -- the first version of this probe did exactly that and reported
     * the staged arms both 150% slower and 24% FASTER on consecutive runs.
     * Round-robin puts every arm in every thermal position. */
    id<MTLComputePipelineState> pso[12];
    NSUInteger smem[12];
    for (int arm = 0; arm < n_arms; arm++) {
        pso[arm] = mk(dev, arms[arm].lib, arms[arm].name, NSG);
        smem[arm] = ((32u*2u*sizeof(float) + arms[arm].extra) + 15) & ~(NSUInteger)15;
        if (!pso[arm]) { printf("  %-24s MISSING\n", label[arm]); fails++; }
    }
    static double t[12][REPS];
    float *got[12];
    for (int arm = 0; arm < n_arms; arm++) got[arm] = malloc((size_t)ne01 * sizeof(float));

    for (int r = 0; r < REPS; r++) {
        /* CYCLIC, not merely interleaved. Round-robin in a fixed order still
         * pins each arm to the same position within every repetition, so a
         * within-rep ramp lands on the same arm each time. Rotating the start
         * puts every arm in every position. */
        for (int k = 0; k < n_arms; k++) {
            const int arm = (k + r) % n_arms;
            if (!pso[arm]) { t[arm][r] = 1e9; continue; }
            memset(o.contents, 0, (size_t)ne01 * sizeof(float));
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso[arm]];
            [enc setBytes:&a length:sizeof(a) atIndex:0];
            [enc setBuffer:w offset:0 atIndex:1];
            [enc setBuffer:x offset:0 atIndex:2];
            [enc setBuffer:o offset:0 atIndex:3];
            [enc setThreadgroupMemoryLength:smem[arm] atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((ne01 + NR0 - 1) / NR0, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * NSG, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                printf("  %-24s DISPATCH FAILED: %s\n", label[arm],
                       cb.error.localizedDescription.UTF8String);
                fails++; pso[arm] = nil; t[arm][r] = 1e9; continue;
            }
            t[arm][r] = (cb.GPUEndTime - cb.GPUStartTime) * 1.0e3;
            if (r == 0) memcpy(got[arm], o.contents, (size_t)ne01 * sizeof(float));
        }
    }

    for (int arm = 0; arm < n_arms; arm++) {
        if (!pso[arm]) continue;
        qsort(t[arm], REPS, sizeof(double), cmpd);
    }
    ref_ms = t[0][REPS/2];
    printf("  %-24s %8.4f ms   (reference, median of %d interleaved)\n",
           label[0], ref_ms, REPS);
    memcpy(ref, got[0], (size_t)ne01 * sizeof(float));
    for (int arm = 1; arm < n_arms; arm++) {
        if (!pso[arm]) continue;
        int bad = 0, first = -1;
        for (int i = 0; i < ne01; i++) {
            if (memcmp(&got[arm][i], &ref[i], sizeof(float)) != 0) {
                if (first < 0) first = i; bad++;
            }
        }
        printf("  %-24s %8.4f ms   %+7.2f%%   %s\n", label[arm], t[arm][REPS/2],
               100.0 * (ref_ms - t[arm][REPS/2]) / ref_ms,
               bad == 0 ? "bit-identical" : "DIVERGES");
        if (bad) {
            printf("      %d/%d differ, first at %d: got %.9g want %.9g\n",
                   bad, ne01, first, got[arm][first], ref[first]);
            fails++;
        }
    }
    for (int arm = 0; arm < n_arms; arm++) free(got[arm]);
    free(ref);
    printf("\n%s\n", fails == 0
           ? "PASS: every wide-Q8 arm is bit-identical to shipping"
           : "FAILED");
    return fails == 0 ? 0 : 1;
} }
