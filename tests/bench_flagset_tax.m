/* Is DS4_METAL_DISPATCH_BALLAST representative of the item-C flag_set dispatch?
 *
 * Arm D prices item C (86 kernel_dsv4_tp_flag_set(_coherent) dispatches/token)
 * with ds4_gpu_decode_dispatch_ballast(), whose kernel is
 * kernel_ds4_dispatch_ballast (metal/unary.metal:320) -- a pure no-op.
 *
 * The real dispatch under DS4_METAL_FAST_SYNC is
 * kernel_dsv4_tp_flag_set_coherent (metal/dsv4_misc.metal:7403), which brackets
 * a single 4-byte store with two system-scope seq_cst device fences.
 *
 * Slope of GPU busy vs dispatch count, one command buffer, same shape
 * (1 threadgroup x 1 thread) for every arm, plus an arm that closes and
 * reopens the encoder after each dispatch (the live gate does
 * ds4_gpu_close_batch_encoder() immediately after the flag_set,
 * ds4_metal.m:10765).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

static const char *kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void k_ballast(device uint *sink [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid == 0xFFFFFFFFu) sink[0] = tid;
}

kernel void k_flag_relaxed(device atomic_uint &flag [[buffer(0)]],
                           constant uint &value [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {
    if (tid == 0) atomic_store_explicit(&flag, value, memory_order_relaxed);
}

#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}
kernel void k_flag_coherent(volatile coherent(system) device uint *flag [[buffer(0)]],
                            constant uint &value [[buffer(1)]]) {
    metal::atomic_thread_fence(metal::mem_flags::mem_device,
                               metal::memory_order_seq_cst,
                               metal::thread_scope_system);
    flag[0] = value;
    metal::atomic_thread_fence(metal::mem_flags::mem_device,
                               metal::memory_order_seq_cst,
                               metal::thread_scope_system);
}
#pragma METAL internals : disable
)MSL";

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLBuffer> buf;

static double run_once(id<MTLComputePipelineState> p, int n, int new_encoder) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        uint32_t value = 7;
        if (new_encoder) {
            for (int i = 0; i < n; i++) {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p];
                [e setBuffer:buf offset:0 atIndex:0];
                [e setBytes:&value length:4 atIndex:1];
                [e dispatchThreadgroups:MTLSizeMake(1,1,1)
                  threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                [e endEncoding];
            }
        } else {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBuffer:buf offset:0 atIndex:0];
            [e setBytes:&value length:4 atIndex:1];
            for (int i = 0; i < n; i++)
                [e dispatchThreadgroups:MTLSizeMake(1,1,1)
                  threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [e endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) * 1e6;   /* us */
    }
}

static void measure(const char *label, id<MTLComputePipelineState> p,
                    int new_encoder, int reps) {
    const int ns[4] = {43, 129, 258, 430};
    double best[4];
    for (int k = 0; k < 4; k++) best[k] = 1e18;
    for (int r = 0; r < reps; r++)
        for (int k = 0; k < 4; k++) {
            double t = run_once(p, ns[k], new_encoder);
            if (t < best[k]) best[k] = t;
        }
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int k = 0; k < 4; k++) {
        sx += ns[k]; sy += best[k];
        sxx += (double)ns[k]*ns[k]; sxy += (double)ns[k]*best[k];
    }
    double slope = (4*sxy - sx*sy) / (4*sxx - sx*sx);
    double icpt  = (sy - slope*sx) / 4;
    printf("%-42s slope %7.3f us/dispatch  fixed %7.1f us   [%.0f %.0f %.0f %.0f us]\n",
           label, slope, icpt, best[0], best[1], best[2], best[3]);
}

int main(int argc, char **argv) {
    int reps = (argc > 2 && strcmp(argv[1], "--reps") == 0) ? atoi(argv[2]) : 7;
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        NSError *err = nil;
        MTLCompileOptions *o = [MTLCompileOptions new];

        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                               options:o error:&err];
        if (!lib) { printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        buf = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        id<MTLComputePipelineState> pb =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_ballast"] error:&err];
        id<MTLComputePipelineState> pr =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_flag_relaxed"] error:&err];
        id<MTLComputePipelineState> pc =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_flag_coherent"] error:&err];
        if (!pb || !pr || !pc) { printf("pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }

        printf("device: %s   reps %d, best-of, slope over N=43..430 in ONE command buffer\n",
               dev.name.UTF8String, reps);
        /* warm-up: the first arm in a process is systematically distorted */
        measure("(warmup, discard) ballast no-op", pb, 0, 3);
        printf("\n--- same encoder (the DS4_METAL_DISPATCH_BALLAST shape) ---\n");
        measure("ballast no-op", pb, 0, reps);
        measure("flag_set relaxed atomic store", pr, 0, reps);
        measure("flag_set COHERENT (system seq_cst x2)", pc, 0, reps);
        printf("\n--- encoder closed+reopened after each dispatch (the live gate) ---\n");
        measure("ballast no-op, encoder per dispatch", pb, 1, reps);
        measure("flag_set relaxed, encoder per dispatch", pr, 1, reps);
        measure("flag_set COHERENT, encoder per dispatch", pc, 1, reps);
    }
    return 0;
}
