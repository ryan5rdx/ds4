/* Stage-marker command-buffer tax probe (scouting instrument, model-free).
 *
 * WHY THIS EXISTS
 *
 * DS4_METAL_GPU_STAGE_TIMESTAMPS attributes a stage's cost as the sum of
 * (GPUEndTime - GPUStartTime) over the command buffers flushed at that stage
 * boundary (ds4_gpu_stage_flush / ds4_gpu_stage_report, ds4_metal.m:11275).
 * ds4_gpu_flush_commands (ds4_metal.m:9813) commits UNCONDITIONALLY, even when
 * g_batch_has_work == NO -- so a stage that encoded zero dispatches still gets
 * one command buffer per layer charged to it.
 *
 * Question this answers: what does an EMPTY command buffer report as GPU busy?
 * If it is ~25-30 us, then any decode stage whose marker fires 41-43 times per
 * token has a ~1.0-1.3 ms floor made entirely of instrument.
 *
 *   ARM E  : N empty command buffers
 *   ARM N1 : N command buffers, 1 no-op dispatch each
 *   ARM P  : N command buffers, 1 pool-shaped dispatch (512 TG x 32 thr) each
 *   ARM B  : 1 command buffer with N pool-shaped dispatches (the un-profiled
 *            equivalent: same work, no marker)
 *
 *   make tests/bench_stage_marker_tax && ./tests/bench_stage_marker_tax
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void k_noop(device float *out [[buffer(0)]],
                   uint tgid [[threadgroup_position_in_grid]]) {
    if (tgid == 0xffffffffu) out[0] = 1.0f;
}

/* Shape-faithful stand-in for kernel_dsv4_compressor_exact_pool_ratio4_decode_ggml
 * (metal/dsv4_kv.metal:439): grid = head_dim threadgroups of 32 threads, each
 * threadgroup gathers 8 scores + 8 kv values strided by 2*head_dim, does an
 * 8-wide softmax on 2 active lanes, a product on 4 active lanes and an 8-lane
 * two-stage simd_sum, with three device-scoped threadgroup barriers. */
kernel void k_pool(device const float *state_kv    [[buffer(0)]],
                   device const float *state_score [[buffer(1)]],
                   device float *softmax           [[buffer(2)]],
                   device float *product           [[buffer(3)]],
                   device float *dst               [[buffer(4)]],
                   constant uint &head_dim         [[buffer(5)]],
                   threadgroup float *sum_scratch  [[threadgroup(0)]],
                   uint col [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]]) {
    const uint64_t stride = 2ull * head_dim;
    float4 score_values = -INFINITY;
    if (tid < 2u) {
        const uint row0 = 4u * tid;
        for (uint j = 0; j < 4u; ++j) {
            const uint row = row0 + j;
            const uint64_t src = (uint64_t)row * stride + (row >= 4u ? head_dim : 0u) + col;
            score_values[j] = state_score[src];
        }
    }
    const uint64_t base = (uint64_t)col * 8u;
    device float4 *softmax4 = (device float4 *)(softmax + base);
    float4 lmax4 = -INFINITY;
    for (int i = (int)tid; i < 2; i += 32) lmax4 = fmax(lmax4, score_values);
    const float lmax = max(max(lmax4[0], lmax4[1]), max(lmax4[2], lmax4[3]));
    const float mx = simd_max(lmax);
    float4 lsum4 = 0.0f;
    for (int i = (int)tid; i < 2; i += 32) {
        const float4 e = exp(score_values - mx);
        lsum4 += e;
        softmax4[i] = e;
    }
    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];
    threadgroup_barrier(mem_flags::mem_none);
    const float s = simd_sum(lsum);
    const float inv = 1.0f / s;
    for (int i = (int)tid; i < 2; i += 32) softmax4[i] *= inv;
    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float *rs = (device volatile const float *)(softmax + base);
    device float *prow = product + base;
    if (tid < 4u) {
        for (uint i0 = tid; i0 < 8u; i0 += 4u) {
            const uint64_t src = (uint64_t)i0 * stride + (i0 >= 4u ? head_dim : 0u) + col;
            prow[i0] = state_kv[src] * rs[i0];
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float *rp = (device volatile const float *)prow;
    sum_scratch[tid] = 0.0f;
    float rsum = 0.0f;
    if (tid < 8u) { rsum += rp[tid]; rsum = simd_sum(rsum); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) sum_scratch[0] = rsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
        rsum = sum_scratch[tid];
        rsum = simd_sum(rsum);
        if (tid == 0u) dst[col] = rsum;
    }
}
)MSL";

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmpd(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        int reps = 9;
        int N = 41;               /* compressed layers per decode token */
        uint32_t head_dim = 512;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--n") && i + 1 < argc) N = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--head-dim") && i + 1 < argc) head_dim = (uint32_t)atoi(argv[++i]);
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no metal device\n"); return 1; }
        printf("device: %s  N=%d head_dim=%u reps=%d\n", [dev.name UTF8String], N, head_dim, reps);

        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                               options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLComputePipelineState> pNoop = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_noop"] error:&err];
        id<MTLComputePipelineState> pPool = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_pool"] error:&err];
        if (!pNoop || !pPool) { fprintf(stderr, "pipeline: %s\n", [[err localizedDescription] UTF8String]); return 1; }

        id<MTLCommandQueue> q = [dev newCommandQueue];
        const NSUInteger stateBytes = (NSUInteger)16 * head_dim * sizeof(float);
        id<MTLBuffer> bKv    = [dev newBufferWithLength:stateBytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bSc    = [dev newBufferWithLength:stateBytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bSm    = [dev newBufferWithLength:(NSUInteger)8 * head_dim * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bPr    = [dev newBufferWithLength:(NSUInteger)8 * head_dim * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bDst   = [dev newBufferWithLength:(NSUInteger)head_dim * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bScrap = [dev newBufferWithLength:4096 options:MTLResourceStorageModePrivate];

        typedef enum { ARM_EMPTY, ARM_NOOP, ARM_POOL, ARM_BATCH_POOL, ARM_BATCH_NOOP, ARM_COUNT } arm_t;
        const char *names[ARM_COUNT] = {
            "E  empty cb x N          ",
            "N1 cb+1 noop dispatch x N",
            "P  cb+1 pool dispatch x N",
            "B  1 cb, N pool dispatches",
            "B0 1 cb, N noop dispatches",
        };
        double busy[ARM_COUNT][64], wall[ARM_COUNT][64];

        for (int arm = 0; arm < ARM_COUNT; arm++) {
            for (int r = 0; r < reps + 1; r++) {   /* rep 0 = warmup, discarded */
                NSMutableArray<id<MTLCommandBuffer>> *cbs = [NSMutableArray array];
                const double t0 = now_s();
                if (arm == ARM_BATCH_POOL || arm == ARM_BATCH_NOOP) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    for (int i = 0; i < N; i++) {
                        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                        if (arm == ARM_BATCH_NOOP) {
                            [e setComputePipelineState:pNoop];
                            [e setBuffer:bScrap offset:0 atIndex:0];
                            [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                        } else {
                            [e setComputePipelineState:pPool];
                            [e setBuffer:bKv offset:0 atIndex:0];
                            [e setBuffer:bSc offset:0 atIndex:1];
                            [e setBuffer:bSm offset:0 atIndex:2];
                            [e setBuffer:bPr offset:0 atIndex:3];
                            [e setBuffer:bDst offset:0 atIndex:4];
                            [e setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
                            [e setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];
                            [e dispatchThreadgroups:MTLSizeMake(head_dim,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                        }
                        [e endEncoding];
                    }
                    [cb commit];
                    [cbs addObject:cb];
                } else {
                    for (int i = 0; i < N; i++) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        if (arm == ARM_NOOP) {
                            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                            [e setComputePipelineState:pNoop];
                            [e setBuffer:bScrap offset:0 atIndex:0];
                            [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                            [e endEncoding];
                        } else if (arm == ARM_POOL) {
                            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                            [e setComputePipelineState:pPool];
                            [e setBuffer:bKv offset:0 atIndex:0];
                            [e setBuffer:bSc offset:0 atIndex:1];
                            [e setBuffer:bSm offset:0 atIndex:2];
                            [e setBuffer:bPr offset:0 atIndex:3];
                            [e setBuffer:bDst offset:0 atIndex:4];
                            [e setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
                            [e setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];
                            [e dispatchThreadgroups:MTLSizeMake(head_dim,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                            [e endEncoding];
                        }
                        [cb commit];
                        [cbs addObject:cb];
                    }
                }
                [cbs.lastObject waitUntilCompleted];
                const double t1 = now_s();
                double sum = 0.0;
                for (id<MTLCommandBuffer> cb in cbs) {
                    const double s = cb.GPUStartTime, e = cb.GPUEndTime;
                    sum += (e > s) ? (e - s) : 0.0;
                }
                if (r > 0) { busy[arm][r-1] = sum * 1000.0; wall[arm][r-1] = (t1 - t0) * 1000.0; }
            }
            qsort(busy[arm], (size_t)reps, sizeof(double), cmpd);
            qsort(wall[arm], (size_t)reps, sizeof(double), cmpd);
        }

        printf("\n%-27s  %12s  %12s  %12s\n", "arm", "busy_sum_ms", "per_cb_us", "wall_ms");
        for (int arm = 0; arm < ARM_COUNT; arm++) {
            const double b = busy[arm][reps/2], w = wall[arm][reps/2];
            printf("%-27s  %12.4f  %12.3f  %12.4f\n", names[arm], b, b*1000.0/N, w);
        }
        /* Interleaved arm: alternate (busy cb, empty cb) N times, the realistic
         * stage-profiler shape.  Report the empty cbs' busy separately: this is
         * exactly what a stage marker that encoded nothing gets charged. */
        {
            double e_busy[64], b_busy[64];
            for (int r = 0; r < reps + 1; r++) {
                NSMutableArray<id<MTLCommandBuffer>> *busycbs = [NSMutableArray array];
                NSMutableArray<id<MTLCommandBuffer>> *emptycbs = [NSMutableArray array];
                for (int i = 0; i < N; i++) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    [e setComputePipelineState:pPool];
                    [e setBuffer:bKv offset:0 atIndex:0];
                    [e setBuffer:bSc offset:0 atIndex:1];
                    [e setBuffer:bSm offset:0 atIndex:2];
                    [e setBuffer:bPr offset:0 atIndex:3];
                    [e setBuffer:bDst offset:0 atIndex:4];
                    [e setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
                    [e setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];
                    [e dispatchThreadgroups:MTLSizeMake(head_dim,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                    [e endEncoding];
                    [cb commit];
                    [busycbs addObject:cb];
                    id<MTLCommandBuffer> cb2 = [q commandBuffer];
                    [cb2 commit];
                    [emptycbs addObject:cb2];
                }
                [emptycbs.lastObject waitUntilCompleted];
                double se = 0.0, sb = 0.0;
                for (id<MTLCommandBuffer> cb in emptycbs) { double a=cb.GPUStartTime,b=cb.GPUEndTime; se += b>a?b-a:0.0; }
                for (id<MTLCommandBuffer> cb in busycbs)  { double a=cb.GPUStartTime,b=cb.GPUEndTime; sb += b>a?b-a:0.0; }
                if (r > 0) { e_busy[r-1] = se*1000.0; b_busy[r-1] = sb*1000.0; }
            }
            qsort(e_busy, (size_t)reps, sizeof(double), cmpd);
            qsort(b_busy, (size_t)reps, sizeof(double), cmpd);
            printf("\nINTERLEAVED (busy cb, empty cb) x N:\n");
            printf("  empty cbs busy_sum_ms = %.4f  (%.3f us/cb)   [min %.4f max %.4f]\n",
                   e_busy[reps/2], e_busy[reps/2]*1000.0/N, e_busy[0], e_busy[reps-1]);
            printf("  busy  cbs busy_sum_ms = %.4f  (%.3f us/cb)   [min %.4f max %.4f]\n",
                   b_busy[reps/2], b_busy[reps/2]*1000.0/N, b_busy[0], b_busy[reps-1]);
        }

        printf("\nraw medians min/med/max busy_ms:\n");
        for (int arm = 0; arm < ARM_COUNT; arm++) {
            printf("  %-27s %.4f / %.4f / %.4f\n", names[arm],
                   busy[arm][0], busy[arm][reps/2], busy[arm][reps-1]);
        }
    }
    return 0;
}
