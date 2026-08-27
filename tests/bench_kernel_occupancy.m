/* Residency of the REAL decode Q8_0 matvec, measured rather than assumed.
 *
 * `tests/bench_occupancy.m` measures the machine's occupancy governors with a
 * synthetic kernel.  This one takes the production body of
 * `kernel_mul_mv_q8_0_f32_impl` (metal/dense.metal:115-180) verbatim, adds only
 * a live/peak threadgroup counter around it, and reports how many threadgroups
 * of it the GPU actually holds resident per core at each NSG.
 *
 * That matters because under TP the host picks NSG=2 (ds4_metal.m:5275), i.e.
 * 64-thread threadgroups, for every dense Q8_0 projection in decode -- roughly
 * 39% of the token.  Whether that costs occupancy depends on which governor
 * binds (threadgroup count, thread count, or registers), and nobody has
 * measured which.
 *
 * The kernel body is a copy, so it must be kept in sync by hand; it is here for
 * a register-allocation-faithful occupancy read, not for correctness.
 *
 *   cc -O2 -fobjc-arc -o tests/bench_kernel_occupancy tests/bench_kernel_occupancy.m \
 *      -framework Foundation -framework Metal
 *   ./tests/bench_kernel_occupancy [cores] [k] [out_dim]
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static NSString *const kSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"#define QK8_0 32\n"
"#define N_SIMDWIDTH 32\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n"
"struct args_t { int ne00; int ne01; ulong nb01; int ne0; };\n"
"\n"
"template<short NR0, short NSG>\n"
"static void mv_body(constant args_t &args,\n"
"                    device const char *src0, device const char *src1,\n"
"                    device char *dst, threadgroup char *shmem,\n"
"                    uint3 tgpig, ushort tiisg, ushort sgitg) {\n"
"    constexpr short NW = N_SIMDWIDTH;\n"
"    constexpr short NQ = 8;\n"
"    const int nb = args.ne00/QK8_0;\n"
"    const int r0 = tgpig.x*NR0;\n"
"    device const float * y = (device const float *) src1;\n"
"    device const block_q8_0 * ax[NR0];\n"
"    FOR_UNROLL (short row = 0; row < NR0; ++row) {\n"
"        ax[row] = (device const block_q8_0 *) ((device char *) src0 + (r0 + row)*args.nb01);\n"
"    }\n"
"    float sumf[NR0] = { 0.f };\n"
"    const short ix = tiisg/(NW/NQ);\n"
"    const short il = tiisg%(NW/NQ);\n"
"    const int ib0 = sgitg*NQ + ix;\n"
"    float yl[NQ];\n"
"    device const float * yb = y + ib0*QK8_0 + il*NQ;\n"
"    for (int ib = ib0; ib < nb; ib += NSG*NQ) {\n"
"        for (short i = 0; i < NQ; ++i) { yl[i] = yb[i]; }\n"
"        for (short row = 0; row < NR0; row++) {\n"
"            device const int8_t * qs = ax[row][ib].qs + il*NQ;\n"
"            float sumq = 0.f;\n"
"            FOR_UNROLL (short i = 0; i < NQ; ++i) { sumq += qs[i] * yl[i]; }\n"
"            sumf[row] += sumq*ax[row][ib].d;\n"
"        }\n"
"        yb += NSG*NQ*QK8_0;\n"
"    }\n"
"    threadgroup float (*shmem_f32)[NW] = (threadgroup float (*)[NW]) shmem;\n"
"    device float * dst_f32 = (device float *) dst;\n"
"    for (short row = 0; row < NR0; ++row) {\n"
"        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;\n"
"        sumf[row] = simd_sum(sumf[row]);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (short row = 0; row < NR0; ++row) {\n"
"        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {\n"
"        float tot = simd_sum(shmem_f32[row][tiisg]);\n"
"        if (tiisg == 0 && sgitg == 0) dst_f32[r0 + row] = tot;\n"
"    }\n"
"}\n"
"\n"
"#define MV(NAME, NSG) \\\n"
"kernel void NAME(constant args_t &args [[buffer(0)]], \\\n"
"                 device const char *src0 [[buffer(1)]], \\\n"
"                 device const char *src1 [[buffer(2)]], \\\n"
"                 device char *dst        [[buffer(3)]], \\\n"
"                 device atomic_uint *live [[buffer(4)]], \\\n"
"                 device atomic_uint *peak [[buffer(5)]], \\\n"
"                 threadgroup char *shmem [[threadgroup(0)]], \\\n"
"                 uint3 tgpig [[threadgroup_position_in_grid]], \\\n"
"                 ushort tiisg [[thread_index_in_simdgroup]], \\\n"
"                 ushort sgitg [[simdgroup_index_in_threadgroup]]) { \\\n"
"    if (tiisg == 0 && sgitg == 0) { \\\n"
"        uint n = atomic_fetch_add_explicit(live, 1u, memory_order_relaxed) + 1u; \\\n"
"        atomic_fetch_max_explicit(peak, n, memory_order_relaxed); \\\n"
"    } \\\n"
"    mv_body<2, NSG>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); \\\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
"    if (tiisg == 0 && sgitg == 0) \\\n"
"        atomic_fetch_sub_explicit(live, 1u, memory_order_relaxed); \\\n"
"}\n"
"MV(mv_nsg1, 1)\n"
"MV(mv_nsg2, 2)\n"
"MV(mv_nsg4, 4)\n"
"MV(mv_nsg8, 8)\n"
"MV(mv_nsg16, 16)\n";

int main(int argc, char **argv) {
    @autoreleasepool {
        const uint32_t cores = (uint32_t)(argc > 1 ? atoi(argv[1]) : 24);
        const int k    = argc > 2 ? atoi(argv[2]) : 4096;
        const int outd = argc > 3 ? atoi(argv[3]) : 4096;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.description.UTF8String); return 1; }

        const size_t row_bytes = (size_t)(k / 32) * sizeof(short) + (size_t)k; /* half d + 32 int8 per block */
        const size_t w_bytes = row_bytes * (size_t)outd;
        id<MTLBuffer> w = [dev newBufferWithLength:w_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> x = [dev newBufferWithLength:(size_t)k * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> o = [dev newBufferWithLength:(size_t)outd * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> live = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> peak = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];

        struct { int ne00; int ne01; unsigned long nb01; int ne0; } args = {
            k, outd, (unsigned long)row_bytes, outd };

        printf("device %s  cores %u  k=%d out=%d  rowbytes=%zu  weights=%.1f MiB\n",
               dev.name.UTF8String, cores, k, outd, row_bytes, w_bytes / 1048576.0);
        printf("%6s %8s %8s %10s %10s %12s %10s\n",
               "nsg", "threads", "grid", "peak tg", "tg/core", "threads/core", "ms/iter");

        const int nsgs[] = {1, 2, 4, 8, 16};
        NSString *names[] = {@"mv_nsg1", @"mv_nsg2", @"mv_nsg4", @"mv_nsg8", @"mv_nsg16"};
        for (int i = 0; i < 5; ++i) {
            id<MTLFunction> fn = [lib newFunctionWithName:names[i]];
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!p) { fprintf(stderr, "pso: %s\n", err.description.UTF8String); return 1; }
            const uint32_t threads = 32u * (uint32_t)nsgs[i];
            const uint32_t grid = (uint32_t)(outd / 2);
            uint32_t bestpeak = 0;
            double best_ms = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                memset(live.contents, 0, 4);
                memset(peak.contents, 0, 4);
                @autoreleasepool {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:p];
                    for (int rep2 = 0; rep2 < 64; ++rep2) {
                        [enc setBytes:&args length:sizeof(args) atIndex:0];
                        [enc setBuffer:w offset:0 atIndex:1];
                        [enc setBuffer:x offset:0 atIndex:2];
                        [enc setBuffer:o offset:0 atIndex:3];
                        [enc setBuffer:live offset:0 atIndex:4];
                        [enc setBuffer:peak offset:0 atIndex:5];
                        [enc setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                    }
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 / 64.0;
                    if (ms < best_ms) best_ms = ms;
                }
                uint32_t v = *(uint32_t *)peak.contents;
                if (v > bestpeak) bestpeak = v;
            }
            printf("%6d %8u %8u %10u %10.2f %12.1f %10.4f   %6.1f GB/s\n",
                   nsgs[i], threads, grid, bestpeak, (double)bestpeak / cores,
                   (double)bestpeak * threads / cores, best_ms,
                   w_bytes / (best_ms * 1e-3) / 1e9);
        }

        /* Residency throttle: allocating extra (unused) threadgroup memory
         * changes NOTHING about the arithmetic -- the kernel is bit-identical --
         * but it caps how many threadgroups fit per core.  If throughput rises
         * as residency falls, the matvec is over-subscribing the memory system,
         * and the fix is a one-line setThreadgroupMemoryLength. */
        printf("\nresidency throttle at nsg=2 (64 threads), padding smem only:\n");
        printf("%8s %10s %10s %12s %10s %10s\n", "smem B", "peak tg", "tg/core",
               "threads/core", "ms/iter", "GB/s");
        id<MTLFunction> fn2 = [lib newFunctionWithName:@"mv_nsg2"];
        id<MTLComputePipelineState> p2 = [dev newComputePipelineStateWithFunction:fn2 error:&err];
        const uint32_t smd[] = {256, 4096, 8192, 12288, 16384, 22528, 32768};
        for (unsigned i = 0; i < sizeof(smd)/sizeof(smd[0]); ++i) {
            const uint32_t grid = (uint32_t)(outd / 2);
            uint32_t bestpeak = 0; double best_ms = 1e30;
            for (int rep = 0; rep < 5; ++rep) @autoreleasepool {
                memset(live.contents, 0, 4); memset(peak.contents, 0, 4);
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:p2];
                const int n = 8;
                for (int r2 = 0; r2 < n; ++r2) {
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:w offset:0 atIndex:1];
                    [enc setBuffer:x offset:0 atIndex:2];
                    [enc setBuffer:o offset:0 atIndex:3];
                    [enc setBuffer:live offset:0 atIndex:4];
                    [enc setBuffer:peak offset:0 atIndex:5];
                    [enc setThreadgroupMemoryLength:smd[i] atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 / n;
                if (ms < best_ms) best_ms = ms;
                uint32_t v = *(uint32_t *)peak.contents;
                if (v > bestpeak) bestpeak = v;
            }
            printf("%8u %10u %10.2f %12.1f %10.4f %10.1f\n", smd[i], bestpeak,
                   (double)bestpeak / cores, (double)bestpeak * 64 / cores,
                   best_ms, w_bytes / (best_ms * 1e-3) / 1e9);
        }

        /* Small-out_dim arm: the production q_a/kv pair (out 1024), shared
         * gate/up (out 1024) and router (out 256) dispatch out_dim/2
         * threadgroups.  On 60 cores that is 8.5 and 2.1 per core against the
         * ~23 the kernel can hold.  nsg splits k WITHIN a threadgroup, so it
         * raises threads-in-flight at an unchanged grid.  Weights are read from
         * a rotating window so each repeat is DRAM-cold. */
        printf("\nsmall-grid nsg sweep (cold weights, rotating window):\n");
        printf("%8s %6s %8s %8s %10s %10s %12s %10s %10s\n", "out_dim", "nsg",
               "threads", "grid", "peak tg", "tg/core", "threads/core", "us", "GB/s");
        const uint32_t outs[] = {256, 512, 1024, 2048};
        for (unsigned oi = 0; oi < 4; ++oi) {
            const uint32_t od = outs[oi];
            const size_t slice = row_bytes * od;
            const int nwin = (int)(w_bytes / slice);
            struct { int ne00; int ne01; unsigned long nb01; int ne0; } a2 = {
                k, (int)od, (unsigned long)row_bytes, (int)od };
            for (int i = 0; i < 5; ++i) {
                id<MTLFunction> fn3 = [lib newFunctionWithName:names[i]];
                id<MTLComputePipelineState> p3 =
                    [dev newComputePipelineStateWithFunction:fn3 error:&err];
                const uint32_t threads = 32u * (uint32_t)nsgs[i];
                const uint32_t grid = od / 2;
                uint32_t bestpeak = 0; double best_us = 1e30;
                for (int rep = 0; rep < 5; ++rep) @autoreleasepool {
                    memset(live.contents, 0, 4); memset(peak.contents, 0, 4);
                    const int n = nwin > 64 ? 64 : nwin;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:p3];
                    for (int d = 0; d < n; ++d) {
                        [enc setBytes:&a2 length:sizeof(a2) atIndex:0];
                        [enc setBuffer:w offset:(NSUInteger)((size_t)d * slice) atIndex:1];
                        [enc setBuffer:x offset:0 atIndex:2];
                        [enc setBuffer:o offset:0 atIndex:3];
                        [enc setBuffer:live offset:0 atIndex:4];
                        [enc setBuffer:peak offset:0 atIndex:5];
                        [enc setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                    }
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    double us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / n;
                    if (us < best_us) best_us = us;
                    uint32_t v = *(uint32_t *)peak.contents;
                    if (v > bestpeak) bestpeak = v;
                }
                printf("%8u %6d %8u %8u %10u %10.2f %12.1f %10.2f %10.1f\n",
                       od, nsgs[i], threads, grid, bestpeak,
                       (double)bestpeak / cores, (double)bestpeak * threads / cores,
                       best_us, slice / (best_us * 1e-6) / 1e9);
            }
            printf("\n");
        }
    }
    return 0;
}
