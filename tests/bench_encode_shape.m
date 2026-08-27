/* LENS: why prefill fills the machine and decode does not.
 *
 * Prefill and decode call the SAME Metal kernels.  The only mechanical
 * differences are (a) the grid: prefill dispatches `rows` threadgroups where
 * decode dispatches 1-2, and (b) how many dispatches sit between two
 * dependent stages.  Example, verbatim from the engine:
 *
 *   prefill  ds4_metal.m:22299  dispatchThreadgroups:MTLSizeMake(rows, 2, 1)
 *   decode   ds4_metal.m:22433  dispatchThreadgroups:MTLSizeMake(1,    2, 1)
 *
 * same pipeline, same threadsPerThreadgroup, same threadgroup memory.
 *
 * This harness reproduces that shape with a synthetic RMS-norm-like kernel and
 * answers three questions that the engine profiler cannot separate:
 *
 *   ROWS     what does one dispatch cost as a function of grid rows?  If it is
 *            flat from 1 to ~N rows, prefill's whole advantage is amortising a
 *            FIXED per-dispatch cost, not "occupancy" in any deeper sense, and
 *            the crossover row count tells you the size of that fixed cost.
 *
 *   SERIAL   N independent 1-row dispatches in a serial encoder
 *            (MTLDispatchTypeSerial, what ds4_gpu_compute_encoder builds at
 *            ds4_metal.m:1028) versus the same N in a concurrent encoder.
 *            The delta is the recoverable part of the serialisation tax.
 *
 *   CHAIN    N dispatches where each reads the previous one's output: the real
 *            decode graph.  This is the number that cannot be beaten by
 *            scheduling alone.
 *
 * The final arm checks the concurrent schedule against the serial one float by
 * float.  A throughput number from a schedule that skipped work is worthless
 * (this project shipped exactly that bug once).
 *
 *   make bench-encode-shape
 *
 * Measured on the M1 Max dev box 2026-08-27 (structural, not a rig number):
 *   ROWS      1 row 6.0 us, 32 rows 8.7, 128 rows 12.2, 512 rows 16.1,
 *             4096 rows 64.1 (523 GB/s).  Fixed cost ~6 us; it equals the work
 *             at about 385 rows.  Prefill's 4096-token quantum pays ~9%
 *             overhead, decode's one row pays ~100%.  Grid width alone buys
 *             nothing below a few hundred rows -- which is why pr-778's
 *             6->10 threadgroup widening measured +0.12%.
 *   ENCODER   serial 3.55 us/dispatch, concurrent 0.15 at N=64.
 *   CHAIN     3.50 us -- a real RAW dependency costs the same as none, so the
 *             cost is the encoder's uniform barrier, not any data stall.
 *   GROUPS    concurrent with a barrier at EVERY edge = 3.59 vs serial 3.55,
 *             i.e. free; 2 independent siblings 1.79 (1.99x), 4 -> 0.95,
 *             8 -> 0.50.  Output bit-identical.
 *   ENCBOUND  splitting the same 64 dispatches across 1..64 encoders in one
 *             command buffer leaves us/dispatch flat at 3.3-3.7, so a
 *             compute-encoder close/reopen is free.  ds4 does 172 of them per
 *             decoded token (two per TP gate, ds4_metal.m:10465,:10529,:10604);
 *             they cost nothing -- but only because the encoder is serial and
 *             already barriers at every dispatch.  Under a concurrent encoder
 *             each close becomes a forced full barrier and they stop being
 *             free.  This settles the "genuinely unmeasured half" left open at
 *             docs/TP-A0-ROWSPLIT-TEST-PLAN.md:2936.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shape-matched to ds4's fused q/kv RMS norm decode dispatch:
 *   q_n = 1024 floats in, 1024 floats out, 256 threads, one f32 weight row,
 *   a 32-float threadgroup reduction scratch.
 * One threadgroup handles one row.  grid.x = rows exactly as the engine does. */
static NSString *const kSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void rowsnorm(device const float4 *src  [[buffer(0)]],\n"
"                     device const float4 *w    [[buffer(1)]],\n"
"                     device float4       *dst  [[buffer(2)]],\n"
"                     constant uint       &n4   [[buffer(3)]],\n"
"                     threadgroup float   *sc   [[threadgroup(0)]],\n"
"                     uint tgid [[threadgroup_position_in_grid]],\n"
"                     uint tid  [[thread_position_in_threadgroup]],\n"
"                     uint ntg  [[threads_per_threadgroup]],\n"
"                     uint sgid [[simdgroup_index_in_threadgroup]],\n"
"                     uint lane [[thread_index_in_simdgroup]]) {\n"
"    device const float4 *row = src + (ulong)tgid * n4;\n"
"    device float4 *out = dst + (ulong)tgid * n4;\n"
"    float acc = 0.0f;\n"
"    for (uint i = tid; i < n4; i += ntg) { float4 v = row[i]; acc += dot(v, v); }\n"
"    acc = simd_sum(acc);\n"
"    if (lane == 0) sc[sgid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float tot = 0.0f;\n"
"    for (uint i = 0; i < ntg / 32u; i++) tot += sc[i];\n"
"    float s = rsqrt(tot / (float)(n4 * 4u) + 1e-6f);\n"
"    for (uint i = tid; i < n4; i += ntg) out[i] = row[i] * w[i] * s;\n"
"}\n";

int main(int argc, char **argv) {
    @autoreleasepool {
        const uint32_t n = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 1024u;
        const int reps   = argc > 2 ? atoi(argv[2]) : 9;
        const uint32_t n4 = n / 4u;
        const uint32_t nth = 256u;      /* == ds4_gpu_rms_norm_threads(1024) */

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no device\n"); return 1; }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rowsnorm"] error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        printf("device %s  cores(hint) maxThreadsPerTG=%lu  n=%u nth=%u\n",
               dev.name.UTF8String, (unsigned long)pso.maxTotalThreadsPerThreadgroup, n, nth);

        const uint32_t MAXROWS = 8192u;
        const size_t rowb = (size_t)n * sizeof(float);
        id<MTLBuffer> src = [dev newBufferWithLength:rowb * MAXROWS options:MTLResourceStorageModeShared];
        id<MTLBuffer> w   = [dev newBufferWithLength:rowb options:MTLResourceStorageModeShared];
        enum { NSETS = 4 };
        id<MTLBuffer> dst[NSETS];
        for (int i = 0; i < NSETS; i++)
            dst[i] = [dev newBufferWithLength:rowb * MAXROWS options:MTLResourceStorageModeShared];
        float *hs = (float *)src.contents, *hw = (float *)w.contents;
        for (size_t i = 0; i < (size_t)n * MAXROWS; i++) hs[i] = (float)((int)(i % 31) - 15) * 0.03f;
        for (uint32_t i = 0; i < n; i++) hw[i] = 1.0f + (float)(i % 7) * 0.01f;
        /* First touch every destination page on the host: an untouched shared
         * buffer faults in on first GPU write and that fault cost lands inside
         * cb.GPUEndTime - cb.GPUStartTime, which inflated the small-grid arms
         * of the first version of this harness by ~8x. */
        for (int i = 0; i < NSETS; i++) memset(dst[i].contents, 0, rowb * MAXROWS);

        /* Warm the GPU-side mapping of every buffer before the first timed arm.
         * A host memset does not fault the pages into the GPU's tables, and the
         * first GPU touch of a 32 MiB shared buffer landed inside
         * GPUEndTime-GPUStartTime and inflated the first arm ~2.5x. */
        for (int wu = 0; wu < 3; wu++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int i = 0; i < NSETS; i++) {
                [enc setComputePipelineState:pso];
                [enc setBuffer:src offset:0 atIndex:0];
                [enc setBuffer:w offset:0 atIndex:1];
                [enc setBuffer:dst[i] offset:0 atIndex:2];
                [enc setBytes:&n4 length:sizeof(n4) atIndex:3];
                [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(MAXROWS,1,1)
                    threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        /* ---------------- ROWS: one dispatch, grid.x = rows ---------------- */
        printf("\nROWS  one dispatch per row-count, %d in flight per cb, serial encoder\n", 64);
        printf("  %8s %10s %10s %12s\n", "rows", "us/disp", "us/row", "GB/s");
        const uint32_t rowsv[] = {1,2,4,8,16,32,60,64,120,128,240,256,512,1024,2048,4096,8192};
        double us_at_1 = 0;
        for (size_t k = 0; k < sizeof(rowsv)/sizeof(rowsv[0]); k++) {
            const uint32_t rows = rowsv[k];
            const int ndisp = 64;
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int d = 0; d < ndisp; d++) {
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:src offset:0 atIndex:0];
                    [enc setBuffer:w offset:0 atIndex:1];
                    [enc setBuffer:dst[d % NSETS] offset:0 atIndex:2];
                    [enc setBytes:&n4 length:sizeof(n4) atIndex:3];
                    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(rows,1,1)
                        threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double busy = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / ndisp;
                if (busy > 0 && busy < best) best = busy;
            }
            if (rows == 1) us_at_1 = best;
            double bytes = (double)rows * (2.0 * rowb) + rowb; /* read row + weight, write row */
            printf("  %8u %10.2f %10.4f %12.1f\n", rows, best, best / rows, bytes / (best * 1e-6) / 1e9);
        }

        /* ------- SERIAL vs CONCURRENT: N independent 1-row dispatches ------- */
        printf("\nENCODER  N independent 1-row dispatches, distinct outputs\n");
        printf("  %8s %14s %14s %8s\n", "N", "serial us/d", "concur us/d", "ratio");
        for (int N = 8; N <= 64; N *= 2) {
            double bs = 1e30, bc = 1e30;
            for (int r = 0; r < reps; r++) {
                for (int mode = 0; mode < 2; mode++) {   /* interleaved A/B */
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = mode
                        ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                        : [cb computeCommandEncoder];
                    for (int d = 0; d < N; d++) {
                        [enc setComputePipelineState:pso];
                        [enc setBuffer:src offset:(NSUInteger)(d * rowb) atIndex:0];
                        [enc setBuffer:w offset:0 atIndex:1];
                        [enc setBuffer:dst[0] offset:(NSUInteger)(d * rowb) atIndex:2];
                        [enc setBytes:&n4 length:sizeof(n4) atIndex:3];
                        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
                            threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
                    }
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / N;
                    if (t > 0) { if (mode == 0) { if (t < bs) bs = t; } else { if (t < bc) bc = t; } }
                }
            }
            printf("  %8d %14.2f %14.2f %8.2fx\n", N, bs, bc, bs / bc);
        }

        /* --- CHAIN: N dependent 1-row dispatches (the real decode graph) --- */
        printf("\nCHAIN  N dependent 1-row dispatches (each reads the previous output)\n");
        printf("  %8s %14s\n", "N", "us/disp");
        for (int N = 8; N <= 64; N *= 2) {
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int d = 0; d < N; d++) {
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:(d == 0 ? src : dst[(d-1) % 2]) offset:0 atIndex:0];
                    [enc setBuffer:w offset:0 atIndex:1];
                    [enc setBuffer:dst[d % 2] offset:0 atIndex:2];
                    [enc setBytes:&n4 length:sizeof(n4) atIndex:3];
                    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
                        threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / N;
                if (t > 0 && t < best) best = t;
            }
            printf("  %8d %14.2f\n", N, best);
        }

        /* --- GROUPS: concurrent encoder with an explicit barrier every G ---
         * This is the only arm that maps onto a real conversion of the decode
         * graph.  G = 1 is "concurrent encoder, barrier at every edge" and must
         * be compared against SERIAL: if it is not cheaper, converting costs
         * nothing but also gains nothing where dispatches are dependent.
         * G = k is "k independent siblings between dependency edges", which is
         * what a decode layer actually has (q_a || kv, router || shared gate/up,
         * the four out_a groups, ...). */
        printf("\nGROUPS  concurrent encoder, explicit barrier every G dispatches, N=64\n");
        printf("  %8s %12s %12s %10s\n", "G", "us/disp", "vs serial", "speedup");
        double serial_ref = 0;
        for (int gi = 0; gi < 7; gi++) {
            const int G = (gi == 0) ? 0 : (1 << (gi - 1));   /* 0 = plain serial */
            const int N = 64;
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = (G == 0)
                    ? [cb computeCommandEncoder]
                    : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                for (int d = 0; d < N; d++) {
                    if (G > 0 && d > 0 && (d % G) == 0)
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:src offset:(NSUInteger)(d * rowb) atIndex:0];
                    [enc setBuffer:w offset:0 atIndex:1];
                    [enc setBuffer:dst[0] offset:(NSUInteger)(d * rowb) atIndex:2];
                    [enc setBytes:&n4 length:sizeof(n4) atIndex:3];
                    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
                        threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / N;
                if (t > 0 && t < best) best = t;
            }
            if (G == 0) serial_ref = best;
            if (G == 0) printf("  %8s %12.2f %12s %10s\n", "serial", best, "-", "1.00x");
            else printf("  %8d %12.2f %12.2f %9.2fx\n", G, best, serial_ref - best, serial_ref / best);
        }

        /* --- ENCBOUND: cost of a compute-encoder close/reopen -----------
         * ds4 closes and reopens the batch encoder twice per TP gate
         * (ds4_metal.m:10465 after the fence-wait spin, :10529 and :10604
         * after the flag set), i.e. 172 close/reopen events per decoded token
         * at 86 gates.  Ballast dispatches cannot see this cost because they
         * are emitted inside the open encoder.  This arm splits N identical
         * tiny dispatches across 1, 2, 4 ... N encoders in ONE command buffer
         * and fits the slope. */
        printf("\nENCBOUND  N=64 identical 1-row dispatches split across E encoders, one cb\n");
        printf("  %10s %12s %14s\n", "encoders", "us/disp", "us/boundary");
        double enc1 = 0;
        for (int E = 1; E <= 64; E *= 2) {
            const int N = 64;
            const int per = N / E;
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                for (int e = 0; e < E; e++) {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    for (int j = 0; j < per; j++) {
                        const int d = e * per + j;
                        [enc setComputePipelineState:pso];
                        [enc setBuffer:src offset:(NSUInteger)(d * rowb) atIndex:0];
                        [enc setBuffer:w offset:0 atIndex:1];
                        [enc setBuffer:dst[0] offset:(NSUInteger)(d * rowb) atIndex:2];
                        [enc setBytes:&n4 length:sizeof(n4) atIndex:3];
                        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
                            threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
                    }
                    [enc endEncoding];
                }
                [cb commit];
                [cb waitUntilCompleted];
                double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / N;
                if (t > 0 && t < best) best = t;
            }
            if (E == 1) enc1 = best;
            printf("  %10d %12.2f %14.2f\n", E, best,
                   E > 1 ? (best - enc1) * N / (double)(E - 1) : 0.0);
        }

        /* --- BINDING: does swapping the bound MTLBuffer object between two
         * otherwise identical 1-row dispatches cost more than swapping the
         * offset inside one buffer?  The engine rebinds distinct buffers on
         * every dispatch (model views, activation tensors), the ENCODER arm
         * above rebinds only offsets, and the two arms disagreed by ~2x. */
        printf("\nBINDING  serial encoder, 1-row dispatches, N=64\n");
        for (int arm = 0; arm < 3; arm++) {
            const int N = 64;
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int d = 0; d < N; d++) {
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:src offset:(arm == 2 ? 0 : (NSUInteger)(d * rowb)) atIndex:0];
                    [enc setBuffer:w offset:0 atIndex:1];
                    if (arm == 0) [enc setBuffer:dst[0] offset:(NSUInteger)(d * rowb) atIndex:2];
                    else          [enc setBuffer:dst[d % NSETS] offset:(NSUInteger)((arm == 2) ? 0 : d * rowb) atIndex:2];
                    [enc setBytes:&n4 length:sizeof(n4) atIndex:3];
                    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
                        threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / N;
                if (t > 0 && t < best) best = t;
            }
            const char *names[] = {"one dst buffer, distinct offsets",
                                   "4 dst buffers, distinct offsets",
                                   "4 dst buffers, SAME offset (WAW)"};
            printf("  %-36s %8.2f us/disp\n", names[arm], best);
        }

        /* Correctness: the concurrent arms must produce the reference output.
         * A scheduling change that silently drops work measures as a win. */
        {
            const int N = 64;
            float *ref = (float *)malloc(rowb * N);
            for (int mode = 0; mode < 2; mode++) {
                memset(dst[0].contents, 0, rowb * N);
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = mode
                    ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                    : [cb computeCommandEncoder];
                for (int d = 0; d < N; d++) {
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:src offset:(NSUInteger)(d * rowb) atIndex:0];
                    [enc setBuffer:w offset:0 atIndex:1];
                    [enc setBuffer:dst[0] offset:(NSUInteger)(d * rowb) atIndex:2];
                    [enc setBytes:&n4 length:sizeof(n4) atIndex:3];
                    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
                        threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (mode == 0) memcpy(ref, dst[0].contents, rowb * N);
                else {
                    const float *got = (const float *)dst[0].contents;
                    size_t bad = 0;
                    for (size_t i = 0; i < (size_t)n * N; i++)
                        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) bad++;
                    printf("\nCORRECTNESS concurrent vs serial: %s (%zu/%zu floats differ)\n",
                           bad ? "MISMATCH" : "bit-identical", bad, (size_t)n * N);
                }
            }
            free(ref);
        }

        printf("\n  1-row dispatch floor was %.2f us; x1021 decode dispatches = %.2f ms/token\n",
               us_at_1, us_at_1 * 1021.0 / 1000.0);
        return 0;
    }
}
