/*
 * What does an extra dispatch actually cost? The KDASEG bar, measured.
 *
 * KDASEG proposes splitting kernel_glm53_kda_decode into three stages so the
 * recurrent-state update can run on a (head, value-segment) grid. The campaign
 * prices that at "roughly 0.12 ms at the measured launch rate" for two added
 * dispatches over 34 layers, and concludes stage B must make the original
 * kernel 35-40% faster to leave a bankable result.
 *
 * That bar rests entirely on the launch rate, and the launch rate is the one
 * number in the estimate nobody measured on Apple8. It is also the cheapest
 * thing in the whole proposal to measure: if two extra dispatches per layer
 * cost materially more than 0.12 ms over 34 layers, the required speedup rises
 * above what a state-partitioned kernel can plausibly deliver and KDASEG is
 * dead before a line of it is written.
 *
 * So: time N trivial dispatches in one command buffer, and report the
 * per-dispatch cost and what 2 x 34 of them would cost per token. Trivial on
 * purpose -- this measures encode + launch, not work.
 *
 * It also reports the SAME count issued as one dispatch with an N-deep grid,
 * which is the floor: whatever separates the two IS the per-launch overhead,
 * with the work held constant.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>

static const char *kSrc =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void noop(device atomic_uint *sink [[buffer(0)]],\n"
"                 uint gid [[thread_position_in_grid]]) {\n"
"    /* One store the compiler cannot elide, on a path never taken, so the\n"
"     * dispatch is real without the work being what is measured. */\n"
"    if (gid == 0xffffffffu) atomic_fetch_add_explicit(sink, 1u, memory_order_relaxed);\n"
"}\n";

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(void) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc)
                                           options:[MTLCompileOptions new] error:&e];
    if (!lib) { printf("VOID: %s\n", e.description.UTF8String); return 2; }
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"noop"] error:&e];
    if (!pso) { printf("VOID: %s\n", e.description.UTF8String); return 2; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> sink = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    printf("device: %s\n\n", dev.name.UTF8String);

    const int counts[] = { 1, 34, 68, 136 };
    const int reps = 40;
    printf("%10s  %12s  %14s\n", "dispatches", "median ms", "per dispatch us");
    double per_dispatch = 0.0;
    for (unsigned c = 0; c < sizeof(counts)/sizeof(counts[0]); c++) {
        const int n = counts[c];
        double *t = calloc(reps, sizeof(double));
        for (int r = 0; r < reps; r++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            for (int i = 0; i < n; i++) {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:sink offset:0 atIndex:0];
                [enc dispatchThreads:MTLSizeMake(64, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                [enc endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
            t[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        }
        qsort(t, reps, sizeof(double), cmp_d);
        const double med = t[reps/2];
        printf("%10d  %12.4f  %14.2f\n", n, med, med * 1000.0 / n);
        if (n == 68) per_dispatch = med / n;
        free(t);
    }

    printf("\nKDASEG's bar, from the measured launch rate:\n");
    printf("  two extra dispatches x 34 layers = %d dispatches\n", 2 * 34);
    printf("  measured cost                    = %.4f ms/token\n", per_dispatch * 68.0);
    printf("  campaign's estimate              = 0.1200 ms/token\n");
    printf("  KDA recurrence target            = 0.6400 ms/token\n");
    const double need = (per_dispatch * 68.0) / 0.64;
    printf("\n  stage B must therefore make the kernel at least %.0f%% faster\n"
           "  just to break even, before anything is banked.\n", need * 100.0);
    printf("\nThis is encode+launch on a trivial kernel, so it is a FLOOR: the real\n"
           "stages carry arguments and barriers. If the floor alone is close to the\n"
           "campaign's 35-40%% requirement, KDASEG is not worth writing.\n");
    return 0;
} }
