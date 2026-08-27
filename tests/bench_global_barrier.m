/* Megakernel feasibility: what does a device-scope global barrier cost on
 * Apple Metal, and across how many threadgroups does it still complete?
 *
 * A persistent/megakernel decode layer replaces K dispatches with 1 dispatch
 * and K global barriers.  That only pays if a global barrier is cheaper than
 * the ~5 us dispatch tax AND the resident grid is wide enough to feed memory.
 * Metal gives no forward-progress guarantee across threadgroups, so the
 * barrier deadlocks once the grid exceeds what is co-resident; the spin here
 * is bounded so that shows up as a huge number instead of a hang.
 *
 *   cc -O3 -fobjc-arc -o tests/bench_global_barrier tests/bench_global_barrier.m \
 *      -framework Foundation -framework Metal && ./tests/bench_global_barrier
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void k_gbar(device atomic_uint   *ctr   [[buffer(0)]],
                   volatile device uint *sense [[buffer(1)]],
                   device const float   *src   [[buffer(2)]],
                   device float         *out   [[buffer(3)]],
                   constant uint &nbar    [[buffer(4)]],
                   constant uint &ngroups [[buffer(5)]],
                   constant uint &work    [[buffer(6)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint tid  [[thread_position_in_threadgroup]],
                   uint nthr [[threads_per_threadgroup]]) {
    float acc = 0.0f;
    for (uint b = 0; b < nbar; b++) {
        for (uint i = tid; i < work; i += nthr) acc += src[tgid * work + i];
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
            uint my = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
            if (my + 1u == (b + 1u) * ngroups) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
                sense[0] = b + 1u;
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
            } else {
                uint spins = 0;
                for (; spins < 20000u; spins++) {
                    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
                    if (sense[0] >= b + 1u) break;
                }
                if (spins >= 20000u) { out[ngroups] = 999.0f; }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    if (acc == 1234.5f) out[tgid] = acc;
}

/* control: identical work, no barrier at all */
kernel void k_nobar(device const float *src [[buffer(0)]],
                    device float *out [[buffer(1)]],
                    constant uint &nbar [[buffer(2)]],
                    constant uint &work [[buffer(3)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint tid  [[thread_position_in_threadgroup]],
                    uint nthr [[threads_per_threadgroup]]) {
    float acc = 0.0f;
    for (uint b = 0; b < nbar; b++)
        for (uint i = tid; i < work; i += nthr) acc += src[tgid * work + i];
    if (acc == 1234.5f) out[tgid] = acc;
}
)MSL";

int main(void) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&e];
    if (!lib) { printf("compile: %s\n", [[e localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> p  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_gbar"] error:&e];
    id<MTLComputePipelineState> pn = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_nobar"] error:&e];
    if (!p || !pn) { printf("pso: %s\n", [[e localizedDescription] UTF8String]); return 1; }

    id<MTLBuffer> ctr   = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> sense = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> src   = [dev newBufferWithLength:64u<<20 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> out   = [dev newBufferWithLength:1u<<20 options:MTLResourceStorageModePrivate];

    const uint32_t Gs[] = {8, 12, 16, 20, 24, 32, 48};
    const uint32_t nbars[] = {1, 4, 16};
    printf("nbar sweep 16/64/256, 256 threads/tg, work=64 floats/tg/round, best of 7\n");
    for (int gi = 0; gi < 7; gi++) {
        uint32_t G = Gs[gi];
        double yb[3], yn[3];
        for (int k = 0; k < 3; k++) {
            double bb = 1e18, bn = 1e18;
            for (int r = 0; r < 2; r++) {
                for (int which = 0; which < 2; which++) {
                    memset(ctr.contents, 0, 4); memset(sense.contents, 0, 4);
                    uint32_t nb = nbars[k], work = 64;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    if (which == 0) {
                        [enc setComputePipelineState:p];
                        [enc setBuffer:ctr offset:0 atIndex:0];
                        [enc setBuffer:sense offset:0 atIndex:1];
                        [enc setBuffer:src offset:0 atIndex:2];
                        [enc setBuffer:out offset:0 atIndex:3];
                        [enc setBytes:&nb length:4 atIndex:4];
                        [enc setBytes:&G length:4 atIndex:5];
                        [enc setBytes:&work length:4 atIndex:6];
                    } else {
                        [enc setComputePipelineState:pn];
                        [enc setBuffer:src offset:0 atIndex:0];
                        [enc setBuffer:out offset:0 atIndex:1];
                        [enc setBytes:&nb length:4 atIndex:2];
                        [enc setBytes:&work length:4 atIndex:3];
                    }
                    [enc dispatchThreadgroups:MTLSizeMake(G,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                    [enc endEncoding];
                    [cb commit]; [cb waitUntilCompleted];
                    if (cb.status != MTLCommandBufferStatusCompleted) { printf("G=%u nb=%u which=%d FAILED\n", G, nb, which); return 1; }
                    double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
                    if (which == 0) { if (t < bb) bb = t; } else { if (t < bn) bn = t; }
                }
            }
            yb[k] = bb; yn[k] = bn;
        }
        double sb = (yb[2]-yb[0])/(double)(nbars[2]-nbars[0]);
        double sn = (yn[2]-yn[0])/(double)(nbars[2]-nbars[0]);
        printf("G=%-4u  per-round WITH gbar %8.3f us   no-bar control %7.3f us   -> gbar costs %8.3f us\n",
               G, sb, sn, sb - sn);
    }
    return 0;
} }
