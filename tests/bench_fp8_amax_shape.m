/* Is the flat ~22 us of kernel_dsv4_qkv_rms_norm_kv_rope_fp8_store_f32 the
 * dispatch, or the serialized FP8 amax loop inside it?
 *
 * A = the shipped shape: 7 sequential 64-element blocks, each a 6-round
 *     threadgroup tree-max with a barrier per round, 64 of 256 threads active.
 * B = same arithmetic, one simdgroup per block: 7 independent simd_max, no
 *     threadgroup barriers at all.  Bit-identical amax per block (a 64-lane
 *     tree max and a 2x32-lane simd max both return the exact maximum; max is
 *     associative and commutative on finite floats, and abs() of a float is
 *     exact), so the quantized output is bit-identical.
 *
 *   cc -O3 -fobjc-arc -o tests/bench_fp8_amax_shape tests/bench_fp8_amax_shape.m \
 *      -framework Foundation -framework Metal && ./tests/bench_fp8_amax_shape
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <math.h>

static const char *kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float e4m3(float x) {           /* stand-in for dsv4_e4m3fn_dequant */
    return float(half(x));
}

/* A: shipped shape */
kernel void k_serial(device float *kv [[buffer(0)]],
                     device float *raw [[buffer(1)]],
                     constant uint &n_nope [[buffer(2)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint tid  [[thread_position_in_threadgroup]]) {
    threadgroup float scratch[64];
    device float *row = kv + tgid * 512;
    device float *out = raw + tgid * 512;
    for (uint off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64u && off + tid < n_nope) { v = row[off + tid]; scratch[tid] = fabs(v); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float amax = max(scratch[0], 1.0e-4f);
        const float s = exp2(ceil(log2(amax / 448.0f)));
        if (tid < 64u && off + tid < n_nope) {
            const float q = e4m3(clamp(v / s, -448.0f, 448.0f)) * s;
            row[off + tid] = q;
            out[off + tid] = float(half(q));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* B: one simdgroup per 64-element block, two lanes-worth folded per simd */
kernel void k_parallel(device float *kv [[buffer(0)]],
                       device float *raw [[buffer(1)]],
                       constant uint &n_nope [[buffer(2)]],
                       uint tgid  [[threadgroup_position_in_grid]],
                       uint tid   [[thread_position_in_threadgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]],
                       ushort tiisg [[thread_index_in_simdgroup]]) {
    device float *row = kv + tgid * 512;
    device float *out = raw + tgid * 512;
    const uint nblk = (n_nope + 63u) / 64u;
    /* 8 simdgroups (256 threads); simdgroup s owns blocks s, s+8, ... */
    for (uint b = sgitg; b < nblk; b += 8u) {
        const uint off = b * 64u;
        float v0 = (off + tiisg      < n_nope) ? row[off + tiisg]      : 0.0f;
        float v1 = (off + tiisg + 32 < n_nope) ? row[off + tiisg + 32] : 0.0f;
        float amax = simd_max(max(fabs(v0), fabs(v1)));
        amax = max(amax, 1.0e-4f);
        const float s = exp2(ceil(log2(amax / 448.0f)));
        if (off + tiisg < n_nope) {
            const float q = e4m3(clamp(v0 / s, -448.0f, 448.0f)) * s;
            row[off + tiisg] = q; out[off + tiisg] = float(half(q));
        }
        if (off + tiisg + 32 < n_nope) {
            const float q = e4m3(clamp(v1 / s, -448.0f, 448.0f)) * s;
            row[off + tiisg + 32] = q; out[off + tiisg + 32] = float(half(q));
        }
    }
}
)MSL";

int main(void) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&e];
    if (!lib) { printf("compile: %s\n", [[e localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> pa = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_serial"] error:&e];
    id<MTLComputePipelineState> pb = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_parallel"] error:&e];
    if (!pa || !pb) { printf("pso: %s\n", [[e localizedDescription] UTF8String]); return 1; }

    const int NROW = 64;
    id<MTLBuffer> kvA = [dev newBufferWithLength:NROW*512*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kvB = [dev newBufferWithLength:NROW*512*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> rwA = [dev newBufferWithLength:NROW*512*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> rwB = [dev newBufferWithLength:NROW*512*4 options:MTLResourceStorageModeShared];
    float *a = kvA.contents, *b = kvB.contents;
    for (int i = 0; i < NROW*512; i++) { float v = sinf(i*0.017f)*(1.0f+(i%13)); a[i]=v; b[i]=v; }

    uint32_t n_nope = 448;
    /* correctness: one dispatch each, compare bitwise */
    for (int which = 0; which < 2; which++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:which ? pb : pa];
        [enc setBuffer:(which?kvB:kvA) offset:0 atIndex:0];
        [enc setBuffer:(which?rwB:rwA) offset:0 atIndex:1];
        [enc setBytes:&n_nope length:4 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
    int diff = 0;
    for (int i = 0; i < 512; i++) if (memcmp(&a[i], &b[i], 4)) diff++;
    printf("bitwise mismatch over the first row: %d of 512 floats %s\n",
           diff, diff ? "-- NOT bit identical" : "-- bit identical");

    /* timing: N dispatches in one cb, slope */
    const int ns[] = {43, 172, 430};
    for (int which = 0; which < 2; which++) {
        double y[3];
        for (int k = 0; k < 3; k++) {
            double best = 1e18;
            for (int r = 0; r < 9; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int i = 0; i < ns[k]; i++) {
                    [enc setComputePipelineState:which ? pb : pa];
                    [enc setBuffer:(which?kvB:kvA) offset:0 atIndex:0];
                    [enc setBuffer:(which?rwB:rwA) offset:0 atIndex:1];
                    [enc setBytes:&n_nope length:4 atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
                if (t < best) best = t;
            }
            y[k] = best;
        }
        printf("%-10s per dispatch %6.3f us   [%.0f %.0f %.0f us @ N=43/172/430]\n",
               which ? "PARALLEL" : "SERIAL", (y[2]-y[0])/(430-43), y[0], y[1], y[2]);
    }
    return 0;
} }
