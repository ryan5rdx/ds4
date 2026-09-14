/*
 * Does setThreadgroupMemoryLength actually reduce co-residency on this GPU?
 *
 * Every occupancy result in this campaign -- the wide-Q8 -47%, and now the
 * MOETGOCC null on both the rig and this box -- rests on the assumption that
 * reserving N bytes per threadgroup caps residency at floor(32768 / N)
 * threadgroups per core. That assumption has never been checked directly. It
 * has only ever been INFERRED from timing, and timing has confounds:
 *
 *   - a bandwidth-bound kernel is residency-insensitive by construction
 *   - a dependent-chain kernel can be capped by outstanding-miss slots rather
 *     than by resident threads, which also flattens it
 *
 * Both of those produce the same flat table as "the knob does nothing", so no
 * amount of careful timing distinguishes them. This does not time anything.
 * It COUNTS.
 *
 * Each threadgroup's thread 0 increments a device counter on entry, pushes the
 * running value into a peak, spins, then decrements. The peak is the largest
 * number of threadgroups the hardware ever had in flight at once. If the
 * reservation binds, that number must fall roughly in proportion to
 * floor(32768 / bytes); if it does not move, the knob is not a residency knob
 * and every sweep built on it measured nothing.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>

/* TWO kernels in ONE library, differing ONLY in whether `scratch` is reachable.
 * Same threads, same grid, same spin, same reservation call. If the flat/cliff
 * split tracks that single difference, the mechanism is not in doubt. */
#define CENSUS_BODY(TOUCH) \
"    if (tid == 0u) {\n" \
"        uint n = atomic_fetch_add_explicit(cur, 1u, memory_order_relaxed) + 1u;\n" \
"        atomic_fetch_max_explicit(peak, n, memory_order_relaxed);\n" \
"    }\n" \
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n" \
"    uint idx = ((tgpig * 64u + tid) * 2654435761u) & 0x3ffffu;\n" \
"    uint acc = 0u;\n" \
"    for (uint i = 0; i < spin; ++i) { idx = tbl[idx] & 0x3ffffu; acc += idx; }\n" \
    TOUCH \
"    if (acc == 0xffffffffu) sink[0] = acc;\n" \
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n" \
"    if (tid == 0u) atomic_fetch_sub_explicit(cur, 1u, memory_order_relaxed);\n" \
"}\n"

#define CENSUS_SIG(NAME) \
"kernel void " NAME "(\n" \
"        device atomic_uint *cur   [[buffer(0)]],\n" \
"        device atomic_uint *peak  [[buffer(1)]],\n" \
"        device const uint *tbl    [[buffer(2)]],\n" \
"        device uint *sink         [[buffer(3)]],\n" \
"        constant uint &spin       [[buffer(4)]],\n" \
"        threadgroup float *scratch [[threadgroup(0)]],\n" \
"        uint tgpig [[threadgroup_position_in_grid]],\n" \
"        uint tid   [[thread_position_in_threadgroup]]) {\n"

static const char *kSrc =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
/* DEAD: exactly what metal/moe.metal's pair impl and the down _tgprobe clones
 * do today -- declare the argument, then (void) it. */
CENSUS_SIG("census_dead")
CENSUS_BODY("    (void)scratch;\n")
/* LIVE: the same kernel with the pointer reachable. */
CENSUS_SIG("census_live")
CENSUS_BODY("    scratch[tid] = (float)acc;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    acc += (uint)scratch[63u - tid];\n");

int main(void) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc)
                                           options:[MTLCompileOptions new]
                                             error:&err];
    if (!lib) { printf("VOID: %s\n", err.description.UTF8String); return 2; }
    const char *knames[2] = { "census_dead", "census_live" };
    id<MTLComputePipelineState> psos[2];
    for (int k = 0; k < 2; k++) {
        psos[k] = [dev newComputePipelineStateWithFunction:
                       [lib newFunctionWithName:@(knames[k])] error:&err];
        if (!psos[k]) { printf("VOID: %s\n", err.description.UTF8String); return 2; }
    }
    id<MTLComputePipelineState> pso = psos[0];

    const uint32_t n_tbl = 256u * 1024u;
    id<MTLBuffer> tbl  = [dev newBufferWithLength:n_tbl * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> cur  = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> peak = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> sink = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    { uint32_t *t = tbl.contents, st = 0x9e3779b9u;
      for (uint32_t i = 0; i < n_tbl; i++) { st ^= st<<13; st ^= st>>17; st ^= st<<5; t[i] = st; } }

    id<MTLCommandQueue> q = [dev newCommandQueue];
    const NSUInteger bytes[] = { 0, 2304, 4352, 4608, 8704, 9216, 16384, 18432, 32768 };
    const uint32_t spin = 20000;
    const NSUInteger grid = 8192;   /* heavily oversubscribed, so the peak is set
                                       by what the hardware allows, not by supply */

    printf("device: %s   maxTGMem: %lu B   maxTPT(kernel): %lu\n",
           dev.name.UTF8String, (unsigned long)dev.maxThreadgroupMemoryLength,
           (unsigned long)pso.maxTotalThreadsPerThreadgroup);
    printf("grid: %lu threadgroups x 64 threads, spin %u dependent loads\n\n",
           (unsigned long)grid, spin);
    for (int kk = 0; kk < 2; kk++) {
    printf("\n--- %s ---\n", kk == 0
           ? "census_dead: `(void)scratch;` -- what the shipping MoE kernels do"
           : "census_live: the same kernel, scratch reachable");
    printf("%10s  %14s  %14s  %10s\n",
           "reserved", "predicted TG/core", "PEAK resident", "ms");

    for (unsigned i = 0; i < sizeof(bytes)/sizeof(bytes[0]); i++) {
        *(uint32_t *)cur.contents = 0;
        *(uint32_t *)peak.contents = 0;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psos[kk]];
        [enc setBuffer:cur offset:0 atIndex:0];
        [enc setBuffer:peak offset:0 atIndex:1];
        [enc setBuffer:tbl offset:0 atIndex:2];
        [enc setBuffer:sink offset:0 atIndex:3];
        [enc setBytes:&spin length:sizeof(spin) atIndex:4];
        if (bytes[i]) [enc setThreadgroupMemoryLength:bytes[i] atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            printf("%10lu  DISPATCH FAILED: %s\n", (unsigned long)bytes[i],
                   cb.error.localizedDescription.UTF8String);
            continue;
        }
        char pred[16];
        if (bytes[i]) snprintf(pred, sizeof(pred), "%lu",
                               (unsigned long)(dev.maxThreadgroupMemoryLength / bytes[i]));
        else          snprintf(pred, sizeof(pred), "unbounded");
        printf("%10lu  %14s  %14u  %10.2f\n", (unsigned long)bytes[i], pred,
               *(uint32_t *)peak.contents,
               (cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
    }
    }
    return 0;
} }
