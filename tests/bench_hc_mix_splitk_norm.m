/* HC mix matvec: hc_attn_fn is F16 [n=16384, out=24] = 786,432 B per layer,
 * and kernel_dsv4_hc_rms_norm_mix_f16_cluster2 dispatches exactly 6
 * threadgroups because each covers NCLUSTER*NR0 = 4 of the 24 output rows
 * (metal/dsv4_hc.metal:1230-1233, ds4_metal.m:44059).  The grid is pinned by
 * out_dim, not chosen.  How much is left on the table?
 *
 * A = 6 threadgroups x 512 threads, 4 rows each, full k (the shipped shape)
 * B = split-K: SK threadgroups per row-group, each covering k/SK, plus a
 *     trivial second pass.  Only the k-split is timed here; the reduce is
 *     24*SK floats and is priced separately as one extra dispatch.
 *
 *   cc -O3 -fobjc-arc -o tests/bench_hc_mix_splitk tests/bench_hc_mix_splitk.m \
 *      -framework Foundation -framework Metal && ./tests/bench_hc_mix_splitk
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

static const char *kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void mv(device const half4 *w [[buffer(0)]],
               device const float4 *x [[buffer(1)]],
               device float *out [[buffer(2)]],
               constant uint &n4 [[buffer(3)]],      /* float4 count of k */
               constant uint &sk [[buffer(4)]],      /* k splits */
               constant uint &donorm [[buffer(5)]],
               threadgroup float *shm [[threadgroup(0)]],
               uint2 tgpig [[threadgroup_position_in_grid]],
               ushort tiisg [[thread_index_in_simdgroup]],
               ushort sgitg [[simdgroup_index_in_threadgroup]],
               ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint rowgrp = tgpig.x;            /* 0..5, 4 rows each */
    const uint split  = tgpig.y;            /* 0..sk-1 */
    const uint chunk  = n4 / sk;
    const uint lo = split * chunk, hi = lo + chunk;
    /* Faithful copy of the shipped kernel's REDUNDANT full-k RMS reduction:
     * every threadgroup reads all of x (16384 floats) regardless of its k
     * slice (metal/dsv4_hc.metal:1246-1264, VTHREADS=1024 over NSG_TOTAL=16). */
    float scale = 1.0f;
    if (donorm) {
        const uint VTHREADS = 1024u;
        for (short v = 0; v < 2; ++v) {
            const uint vt = (uint)(sgitg + 16*v)*32u + tiisg;
            float sumf = 0.0f;
            for (uint i00 = vt; i00 < n4; i00 += VTHREADS) sumf += dot(x[i00], x[i00]);
            sumf = simd_sum(sumf);
            if (tiisg == 0) shm[sgitg + 16*v] = sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = shm[tiisg];
        total = simd_sum(total);
        scale = 1.0f/sqrt(total/(float)(n4*4) + 1e-6f);
    }
    for (uint r = 0; r < 4; r++) {
        const uint row = rowgrp * 4 + r;
        device const half4 *wr = w + (uint64_t)row * n4;
        float acc = 0.0f;
        for (uint i = lo + sgitg * 32u + tiisg; i < hi; i += nsg * 32u)
            acc += dot(float4(wr[i]), x[i]*scale);
        acc = simd_sum(acc);
        if (tiisg == 0 && acc == 12345.678f) out[row * 64 + split] = acc;
    }
}
)MSL";

int main(void) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&e];
    if (!lib) { printf("compile: %s\n", [[e localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mv"] error:&e];
    if (!p) { printf("pso: %s\n", [[e localizedDescription] UTF8String]); return 1; }

    const uint32_t n = 16384, n4 = n / 4;
    /* 43 layers x 2 (attn+ffn) distinct weight blocks so nothing is SLC resident */
    const int NW = 86;
    id<MTLBuffer> w = [dev newBufferWithLength:(size_t)NW * 24 * n * 2 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> x = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> o = [dev newBufferWithLength:24 * 64 * 4 options:MTLResourceStorageModePrivate];
    printf("shape: F16 [k=16384, out=24] = %.0f KB per call, %d distinct weight blocks (%.0f MB)\n",
           24.0 * n * 2 / 1024.0, NW, (double)NW * 24 * n * 2 / 1048576.0);

    const uint32_t sks[] = {1, 2, 4, 8, 16, 32};
    for (uint32_t donorm = 0; donorm <= 1; donorm++) {
    printf("--- redundant RMS norm: %s ---\n", donorm ? "ON (as shipped)" : "OFF (scout harness)");
    for (int si = 0; si < 6; si++) {
        uint32_t sk = sks[si];
        const int ns[] = {43, 172, 430};
        double y[3];
        for (int k = 0; k < 3; k++) {
            double best = 1e18;
            for (int r = 0; r < 7; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int i = 0; i < ns[k]; i++) {
                    [enc setComputePipelineState:p];
                    [enc setBuffer:w offset:(NSUInteger)(i % NW) * 24 * n * 2 atIndex:0];
                    [enc setBuffer:x offset:0 atIndex:1];
                    [enc setBuffer:o offset:0 atIndex:2];
                    [enc setBytes:&n4 length:4 atIndex:3];
                    [enc setBytes:&sk length:4 atIndex:4];
                    [enc setBytes:&donorm length:4 atIndex:5];
                    [enc setThreadgroupMemoryLength:32*4 atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(6, sk, 1)
                        threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
                if (t < best) best = t;
            }
            y[k] = best;
        }
        double us = (y[2] - y[0]) / (430 - 43);
        printf("split-K %-3u -> %4u threadgroups   %7.3f us/call   %6.1f GB/s\n",
               sk, 6 * sk, us, 786432.0 / (us * 1e-6) / 1e9);
    }
    }
    return 0;
} }
