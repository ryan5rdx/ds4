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
 * thing in the whole proposal to measure.
 *
 * What it can and cannot settle. It fixes the CONSTANT in the bar; it does not
 * decide the arm. Apple8 measured 0.130-0.131 ms/token, which essentially
 * CONFIRMS the campaign's 0.120 estimate -- break-even near 20%, banking
 * 0.10 ms near 36%, i.e. the 35-40% requirement that was already written down.
 * Reading that as "KDASEG is dead" was an overreach on this probe's part: the
 *32-threadgroup kernel underfills 60 GPU cores, so a segmented stage can
 * plausibly reach that range if the parallelizable update dominates. The next
 * question is a SHARE question -- roughly >=72% parallelizable for a two-way
 * split, >=48% for four-way, before barrier costs -- and this probe does not
 * answer it.
 *
 * So: time N trivial dispatches in one command buffer, and report the
 * per-dispatch cost and what 2 x 34 of them would cost per token. Trivial on
 * purpose -- this measures encode + launch, not work.
 *
 * It also reports the SAME total threadgroup count issued as ONE dispatch with
 * an N-deep grid. Whatever separates the two IS the per-launch overhead with
 * the work held constant -- without it, the per-dispatch figure includes the
 * work itself and overstates the launch cost.
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

    /* The deep-grid control: same threadgroups, one launch. */
    printf("\n%10s  %12s  %14s\n", "deep grid", "median ms", "vs N dispatches");
    for (unsigned c = 0; c < sizeof(counts)/sizeof(counts[0]); c++) {
        const int n = counts[c];
        double *t = calloc(reps, sizeof(double));
        for (int r = 0; r < reps; r++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:sink offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(64 * n, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            t[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        }
        qsort(t, reps, sizeof(double), cmp_d);
        printf("%10d  %12.4f  %14s\n", n, t[reps/2],
               n == 1 ? "-" : "(one launch)");
        free(t);
    }

    printf("\nKDASEG's bar, from the measured launch rate:\n");
    printf("  two extra dispatches x 34 layers = %d dispatches\n", 2 * 34);
    printf("  measured cost                    = %.4f ms/token\n", per_dispatch * 68.0);
    printf("  campaign's estimate              = 0.1200 ms/token\n");
    printf("  KDA recurrence target            = 0.6400 ms/token\n");
    const double breakeven = (per_dispatch * 68.0) / 0.64;
    const double bankable   = (per_dispatch * 68.0 + 0.10) / 0.64;
    printf("\n  break-even:            stage B >= %.0f%% faster\n", breakeven * 100.0);
    printf("  to bank 0.10 ms/token: stage B >= %.0f%% faster\n", bankable * 100.0);
    printf("\n  Those are the campaign's own 35-40%% requirement, not a new\n"
           "  obstacle -- this probe CONFIRMS the estimate rather than\n"
           "  refuting it. What it settles is that the launch cost is real and\n"
           "  the margin is thin, so the next question is whether stage B is a\n"
           "  large enough share of the 0.64 ms: a two-way split needs roughly\n"
           "  >=72%% parallelizable, a four-way >=48%%, before barrier costs.\n");
    printf("\nThis is encode+launch on a trivial kernel, so it is a FLOOR: the real\n"
           "stages carry arguments and barriers.\n");
    return 0;
} }
