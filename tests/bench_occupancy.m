/* Occupancy probe — what actually governs resident threadgroups per core.
 *
 * WHY THIS EXISTS
 *
 * The decode investigation has six recorded "underfill" sightings and exactly
 * one banked occupancy win (U10: 20512 -> 16384 B of threadgroup memory took
 * the indexer LLT scorer from 1 to 2 resident threadgroups per core, +17.4%).
 * Every residency number quoted in that work was DERIVED from an assumed
 * 32 KiB per-core threadgroup-memory pool.  Nobody ever measured residency.
 *
 * This harness measures it directly, and it returns integers, not timings, so
 * it is valid on a busy dev box where `tests/bench_*` timing is not.
 *
 * METHOD
 *
 * Launch G threadgroups of a kernel that (a) atomically bumps a live counter
 * and records its running max, (b) burns a fixed dependent FMA chain long
 * enough that the whole first residency wave overlaps, (c) decrements.  The
 * recorded max IS the number of threadgroups the GPU held in flight.  Divide
 * by the core count for per-core residency.
 *
 * Three axes, each isolating one candidate governor:
 *   --threads    threads per threadgroup, no smem, low register pressure
 *   --smem       dynamic threadgroup memory at fixed 256 threads
 *   --regs       live-register pressure at fixed 128 threads
 *
 *   cc -O2 -fobjc-arc -o tests/bench_occupancy tests/bench_occupancy.m \
 *      -framework Foundation -framework Metal
 *   ./tests/bench_occupancy
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static NSString *const kSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"/* R independent accumulators -> R live registers the compiler cannot fold. */\n"
"template <uint R>\n"
"static void burn(thread float *acc, uint spin, uint tid) {\n"
"    for (uint r = 0; r < R; ++r) acc[r] = (float)(tid + r) * 1e-3f;\n"
"    for (uint i = 0; i < spin; ++i) {\n"
"        for (uint r = 0; r < R; ++r) acc[r] = fma(acc[r], 1.0000001f, 1.0f);\n"
"    }\n"
"}\n"
"\n"
"template <uint R>\n"
"static void probe_impl(device atomic_uint *live,\n"
"                       device atomic_uint *peak,\n"
"                       device uint *out,\n"
"                       constant uint &spin,\n"
"                       threadgroup uchar *scratch,\n"
"                       uint tgid, uint tid, uint ntg) {\n"
"    if (tid == 0) {\n"
"        uint n = atomic_fetch_add_explicit(live, 1u, memory_order_relaxed) + 1u;\n"
"        atomic_fetch_max_explicit(peak, n, memory_order_relaxed);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float acc[R];\n"
"    burn<R>(acc, spin, tid);\n"
"    float s = 0.0f;\n"
"    for (uint r = 0; r < R; ++r) s += acc[r];\n"
"    /* touch the dynamic threadgroup allocation so it is really reserved */\n"
"    scratch[tid & 63u] = (uchar)((uint)s & 0xffu);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        out[tgid % ntg] = (uint)scratch[0];\n"
"        atomic_fetch_sub_explicit(live, 1u, memory_order_relaxed);\n"
"    }\n"
"}\n"
"\n"
"#define PROBE(NAME, R) \\\n"
"kernel void NAME(device atomic_uint *live [[buffer(0)]], \\\n"
"                 device atomic_uint *peak [[buffer(1)]], \\\n"
"                 device uint *out         [[buffer(2)]], \\\n"
"                 constant uint &spin      [[buffer(3)]], \\\n"
"                 constant uint &ntg       [[buffer(4)]], \\\n"
"                 threadgroup uchar *scratch [[threadgroup(0)]], \\\n"
"                 uint tgid [[threadgroup_position_in_grid]], \\\n"
"                 uint tid  [[thread_position_in_threadgroup]]) { \\\n"
"    probe_impl<R>(live, peak, out, spin, scratch, tgid, tid, ntg); \\\n"
"}\n"
"\n"
"PROBE(probe_r4,   4)\n"
"PROBE(probe_r8,   8)\n"
"PROBE(probe_r16, 16)\n"
"PROBE(probe_r32, 32)\n"
"PROBE(probe_r64, 64)\n"
"PROBE(probe_r96, 96)\n"
"PROBE(probe_r128, 128)\n"
"\n"
"/* Arm D: streaming read whose ONLY variable is residency.  The dynamic\n"
" * threadgroup allocation is never touched in the inner loop, so instruction\n"
" * mix and register pressure are identical across arms; only how many\n"
" * threadgroups fit per core changes.  This isolates bandwidth-vs-occupancy. */\n"
"kernel void stream_read(device const uint4 *src [[buffer(0)]],\n"
"                        device uint4 *sink      [[buffer(1)]],\n"
"                        constant uint &n_vec    [[buffer(2)]],\n"
"                        constant uint &store    [[buffer(3)]],\n"
"                        device atomic_uint *live [[buffer(4)]],\n"
"                        device atomic_uint *peak [[buffer(5)]],\n"
"                        threadgroup uchar *scratch [[threadgroup(0)]],\n"
"                        uint gid [[thread_position_in_grid]],\n"
"                        uint gsz [[threads_per_grid]],\n"
"                        uint tid [[thread_position_in_threadgroup]]) {\n"
"    if (tid == 0) {\n"
"        uint n = atomic_fetch_add_explicit(live, 1u, memory_order_relaxed) + 1u;\n"
"        atomic_fetch_max_explicit(peak, n, memory_order_relaxed);\n"
"    }\n"
"    scratch[tid & 63u] = (uchar)tid;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    uint4 acc = uint4(scratch[0]);\n"
"    for (uint i = gid; i < n_vec; i += gsz) acc ^= src[i];\n"
"    if (store != 0u) sink[gid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) atomic_fetch_sub_explicit(live, 1u, memory_order_relaxed);\n"
"}\n"
"\n"
"/* Arm E: cost of one dispatch in a SERIAL encoder as a function of grid size.\n"
" * ds4 encodes decode with [cb computeCommandEncoder] (MTLDispatchTypeSerial,\n"
" * ds4_metal.m:1028 region), so every dispatch is a barrier.  If a 2-threadgroup\n"
" * dispatch and a 2048-threadgroup dispatch cost the same, grid underfill is\n"
" * free and widening grids cannot pay. */\n"
"kernel void tiny(device float *a [[buffer(0)]],\n"
"                 device float *b [[buffer(1)]],\n"
"                 constant uint &work [[buffer(2)]],\n"
"                 uint gid [[thread_position_in_grid]]) {\n"
"    float v = a[gid & 1023u];\n"
"    for (uint i = 0; i < work; ++i) v = fma(v, 1.000001f, 1.0f);\n"
"    b[gid & 1023u] = v;\n"
"}\n"
"\n"
"/* Arm F: FIXED total bytes, spread over G threadgroups, in a serial encoder.\n"
" * This is the underfill penalty curve: how much does a small-grid dispatch\n"
" * cost against the same work spread over the whole machine?  ds4 has several\n"
" * production dispatches at G=2, G=6 and G=32 on a 60-core part. */\n"
"kernel void fixed_work(device const float4 *src [[buffer(0)]],\n"
"                       device float *out        [[buffer(1)]],\n"
"                       constant uint &vec_per_tg [[buffer(2)]],\n"
"                       uint tgid [[threadgroup_position_in_grid]],\n"
"                       uint tid  [[thread_position_in_threadgroup]],\n"
"                       uint tsz  [[threads_per_threadgroup]]) {\n"
"    const uint base = tgid * vec_per_tg;\n"
"    float4 acc = float4(0);\n"
"    for (uint i = tid; i < vec_per_tg; i += tsz) acc += src[base + i];\n"
"    float s = acc.x + acc.y + acc.z + acc.w;\n"
"    s = simd_sum(s);\n"
"    if ((tid & 31u) == 0u) out[tgid * 32u + (tid >> 5)] = s;\n"
"}\n";

typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLLibrary> lib;
} ctx_t;

static id<MTLComputePipelineState> pso(ctx_t *c, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [c->lib newFunctionWithName:name];
    if (!fn) { fprintf(stderr, "no function %s\n", name.UTF8String); exit(1); }
    id<MTLComputePipelineState> p = [c->dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "pso %s: %s\n", name.UTF8String, err.description.UTF8String); exit(1); }
    return p;
}

/* Returns the peak number of concurrently resident threadgroups. */
static uint32_t measure(ctx_t *c, id<MTLComputePipelineState> p,
                        uint32_t threads, uint32_t smem, uint32_t grid,
                        uint32_t spin) {
    __block uint32_t best = 0;
    for (int rep = 0; rep < 3; ++rep) {
        @autoreleasepool {
            id<MTLBuffer> live = [c->dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> peak = [c->dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> out  = [c->dev newBufferWithLength:4 * grid options:MTLResourceStorageModeShared];
            memset(live.contents, 0, 4);
            memset(peak.contents, 0, 4);
            id<MTLCommandBuffer> cb = [c->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:p];
            [enc setBuffer:live offset:0 atIndex:0];
            [enc setBuffer:peak offset:0 atIndex:1];
            [enc setBuffer:out  offset:0 atIndex:2];
            [enc setBytes:&spin length:4 atIndex:3];
            [enc setBytes:&grid length:4 atIndex:4];
            [enc setThreadgroupMemoryLength:(smem ? smem : 64) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            uint32_t v = *(uint32_t *)peak.contents;
            if (v > best) best = v;
        }
    }
    return best;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        ctx_t c;
        c.dev = MTLCreateSystemDefaultDevice();
        c.q = [c.dev newCommandQueue];
        NSError *err = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new];
        c.lib = [c.dev newLibraryWithSource:kSource options:opt error:&err];
        if (!c.lib) { fprintf(stderr, "compile: %s\n", err.description.UTF8String); return 1; }

        const uint32_t cores = (uint32_t)(argc > 1 ? atoi(argv[1]) : 24);
        const uint32_t spin  = (uint32_t)(argc > 2 ? atoi(argv[2]) : 30000);
        printf("device: %s   assumed cores: %u   spin: %u\n",
               c.dev.name.UTF8String, cores, spin);
        printf("maxThreadgroupMemoryLength: %lu B\n",
               (unsigned long)c.dev.maxThreadgroupMemoryLength);

        id<MTLComputePipelineState> p8 = pso(&c, @"probe_r8");
        printf("probe_r8: maxTotalThreadsPerThreadgroup=%lu threadExecutionWidth=%lu staticSmem=%lu\n",
               (unsigned long)p8.maxTotalThreadsPerThreadgroup,
               (unsigned long)p8.threadExecutionWidth,
               (unsigned long)p8.staticThreadgroupMemoryLength);

        printf("\n== A: threads/threadgroup (smem=64B, 8 live regs) ==\n");
        printf("%8s %8s %10s %12s %12s\n", "threads", "grid", "peak tg", "tg/core", "threads/core");
        const uint32_t thr[] = {32, 64, 128, 192, 256, 384, 512, 768, 1024};
        for (unsigned i = 0; i < sizeof(thr)/sizeof(thr[0]); ++i) {
            if (thr[i] > p8.maxTotalThreadsPerThreadgroup) continue;
            uint32_t grid = cores * 64;
            uint32_t pk = measure(&c, p8, thr[i], 64, grid, spin);
            printf("%8u %8u %10u %12.2f %12.1f\n", thr[i], grid, pk,
                   (double)pk / cores, (double)pk * thr[i] / cores);
        }

        printf("\n== B: dynamic threadgroup memory (256 threads, 8 live regs) ==\n");
        printf("%8s %10s %12s %14s\n", "smem B", "peak tg", "tg/core", "32768/smem");
        const uint32_t sm[] = {64, 1024, 2048, 4096, 6144, 6688, 8192, 11296,
                               16384, 16385, 20512, 24576, 32768};
        for (unsigned i = 0; i < sizeof(sm)/sizeof(sm[0]); ++i) {
            if (sm[i] > c.dev.maxThreadgroupMemoryLength) continue;
            uint32_t grid = cores * 32;
            uint32_t pk = measure(&c, p8, 256, sm[i], grid, spin);
            printf("%8u %10u %12.2f %14.2f\n", sm[i], pk, (double)pk / cores,
                   32768.0 / sm[i]);
        }

        printf("\n== C: register pressure (128 threads, smem=64B) ==\n");
        printf("%8s %10s %12s %12s %10s\n", "live regs", "peak tg", "tg/core",
               "threads/core", "maxTPTG");
        NSString *names[] = {@"probe_r4", @"probe_r8", @"probe_r16", @"probe_r32",
                             @"probe_r64", @"probe_r96", @"probe_r128"};
        const uint32_t rr[] = {4, 8, 16, 32, 64, 96, 128};
        for (unsigned i = 0; i < 7; ++i) {
            id<MTLComputePipelineState> p = pso(&c, names[i]);
            uint32_t grid = cores * 64;
            uint32_t pk = measure(&c, p, 128, 64, grid, spin);
            printf("%8u %10u %12.2f %12.1f %10lu\n", rr[i], pk, (double)pk / cores,
                   (double)pk * 128 / cores,
                   (unsigned long)p.maxTotalThreadsPerThreadgroup);
        }

        printf("\n== D: streaming bandwidth vs residency (256 threads, smem is the only knob) ==\n");
        const size_t gib = (size_t)(argc > 3 ? atoi(argv[3]) : 8);
        const size_t bytes = gib << 30;
        id<MTLBuffer> src = [c.dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        if (!src) { printf("  (allocation of %zu GiB failed)\n", gib); return 0; }
        id<MTLBuffer> sink = [c.dev newBufferWithLength:1 << 20 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> live = [c.dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> peak = [c.dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        id<MTLComputePipelineState> ps = pso(&c, @"stream_read");
        const uint32_t n_vec = (uint32_t)(bytes / 16);
        const uint32_t store = 0;
        printf("%8s %10s %12s %12s %10s %10s\n", "smem B", "peak tg", "tg/core",
               "threads/core", "ms", "GB/s");
        const uint32_t smd[] = {64, 8192, 16384, 22528, 32768};
        for (unsigned i = 0; i < sizeof(smd)/sizeof(smd[0]); ++i) {
            uint32_t grid = cores * 64;
            double best = 1e30; uint32_t pk = 0;
            for (int rep = 0; rep < 4; ++rep) @autoreleasepool {
                memset(live.contents, 0, 4); memset(peak.contents, 0, 4);
                id<MTLCommandBuffer> cb = [c.q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:src offset:0 atIndex:0];
                [enc setBuffer:sink offset:0 atIndex:1];
                [enc setBytes:&n_vec length:4 atIndex:2];
                [enc setBytes:&store length:4 atIndex:3];
                [enc setBuffer:live offset:0 atIndex:4];
                [enc setBuffer:peak offset:0 atIndex:5];
                [enc setThreadgroupMemoryLength:smd[i] atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                if (ms < best) best = ms;
                uint32_t v = *(uint32_t *)peak.contents;
                if (v > pk) pk = v;
            }
            printf("%8u %10u %12.2f %12.1f %10.2f %10.1f\n", smd[i], pk,
                   (double)pk / cores, (double)pk * 256 / cores, best,
                   bytes / (best * 1e-3) / 1e9);
        }

        printf("\n== E: serial-encoder per-dispatch cost vs grid (128 threads) ==\n");
        id<MTLComputePipelineState> pt = pso(&c, @"tiny");
        id<MTLBuffer> ba = [c.dev newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bb = [c.dev newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
        printf("%10s %10s %14s %14s\n", "grid tg", "work", "us/dispatch", "us/dispatch(w=0)");
        const uint32_t grids[] = {1, 2, 4, 8, 16, 32, 60, 120, 256, 1024, 4096};
        for (unsigned i = 0; i < sizeof(grids)/sizeof(grids[0]); ++i) {
            double row[2];
            const uint32_t works[2] = {200, 0};
            for (int wi = 0; wi < 2; ++wi) {
                double best = 1e30;
                for (int rep = 0; rep < 5; ++rep) @autoreleasepool {
                    const int n = 256;
                    id<MTLCommandBuffer> cb = [c.q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pt];
                    for (int d = 0; d < n; ++d) {
                        [enc setBuffer:(d & 1) ? bb : ba offset:0 atIndex:0];
                        [enc setBuffer:(d & 1) ? ba : bb offset:0 atIndex:1];
                        [enc setBytes:&works[wi] length:4 atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(grids[i], 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    double us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / n;
                    if (us < best) best = us;
                }
                row[wi] = best;
            }
            printf("%10u %10u %14.2f %14.2f\n", grids[i], 200u, row[0], row[1]);
        }

        printf("\n== F: fixed total bytes vs grid width (serial encoder) ==\n");
        id<MTLComputePipelineState> pf = pso(&c, @"fixed_work");
        printf("%10s %10s %12s %12s %12s %10s\n", "total KiB", "grid tg",
               "threads/tg", "KiB/tg", "us/dispatch", "GB/s");
        const uint32_t totals_kib[] = {192, 768, 1536, 6144};
        const uint32_t gs[] = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 60, 120, 240, 480};
        for (unsigned t = 0; t < sizeof(totals_kib)/sizeof(totals_kib[0]); ++t) {
            const uint32_t total_vec = totals_kib[t] * 1024u / 16u;
            id<MTLBuffer> sb = [c.dev newBufferWithLength:(size_t)total_vec * 16
                                                  options:MTLResourceStorageModeShared];
            memset(sb.contents, 1, (size_t)total_vec * 16);
            id<MTLBuffer> ob = [c.dev newBufferWithLength:1 << 20
                                                  options:MTLResourceStorageModePrivate];
            for (unsigned i = 0; i < sizeof(gs)/sizeof(gs[0]); ++i) {
                if (total_vec % gs[i]) continue;
                const uint32_t vpt = total_vec / gs[i];
                const uint32_t thr = vpt >= 256 ? 256 : (vpt >= 64 ? 64 : 32);
                double best = 1e30;
                for (int rep = 0; rep < 5; ++rep) @autoreleasepool {
                    const int n = 128;
                    id<MTLCommandBuffer> cb = [c.q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pf];
                    [enc setBuffer:sb offset:0 atIndex:0];
                    [enc setBuffer:ob offset:0 atIndex:1];
                    [enc setBytes:&vpt length:4 atIndex:2];
                    for (int d = 0; d < n; ++d) {
                        [enc dispatchThreadgroups:MTLSizeMake(gs[i], 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
                    }
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    double us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / n;
                    if (us < best) best = us;
                }
                printf("%10u %10u %12u %12.1f %12.2f %10.1f\n", totals_kib[t], gs[i],
                       thr, totals_kib[t] / (double)gs[i], best,
                       totals_kib[t] * 1024.0 / (best * 1e-6) / 1e9);
            }
            printf("\n");
        }

        printf("\n== G: delivered bandwidth of ONE threadgroup vs its width ==\n");
        printf("(432 KiB is the HC producer TG0 serial byte budget, SCOPE-HC-STAGES 5.4)\n");
        printf("%10s %10s %12s %12s %10s\n", "KiB", "grid tg", "threads/tg",
               "us", "GB/s");
        const uint32_t kib_g[] = {96, 432, 768};
        const uint32_t thr_g[] = {32, 64, 128, 256, 512, 1024};
        const uint32_t grid_g[] = {1, 6, 60};
        for (unsigned t = 0; t < 3; ++t) {
            const uint32_t total_vec = kib_g[t] * 1024u / 16u;
            id<MTLBuffer> sb = [c.dev newBufferWithLength:(size_t)total_vec * 16
                                                  options:MTLResourceStorageModeShared];
            memset(sb.contents, 1, (size_t)total_vec * 16);
            id<MTLBuffer> ob = [c.dev newBufferWithLength:1 << 20
                                                  options:MTLResourceStorageModePrivate];
            for (unsigned gi = 0; gi < 3; ++gi) {
                if (total_vec % grid_g[gi]) continue;
                for (unsigned i = 0; i < 6; ++i) {
                    const uint32_t vpt = total_vec / grid_g[gi];
                    double best = 1e30;
                    for (int rep = 0; rep < 5; ++rep) @autoreleasepool {
                        const int n = 128;
                        id<MTLCommandBuffer> cb = [c.q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pf];
                        [enc setBuffer:sb offset:0 atIndex:0];
                        [enc setBuffer:ob offset:0 atIndex:1];
                        [enc setBytes:&vpt length:4 atIndex:2];
                        for (int d = 0; d < n; ++d) {
                            [enc dispatchThreadgroups:MTLSizeMake(grid_g[gi], 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(thr_g[i], 1, 1)];
                        }
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        double us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / n;
                        if (us < best) best = us;
                    }
                    printf("%10u %10u %12u %12.2f %10.1f\n", kib_g[t], grid_g[gi],
                           thr_g[i], best, kib_g[t] * 1024.0 / (best * 1e-6) / 1e9);
                }
            }
            printf("\n");
        }
    }
    return 0;
}
