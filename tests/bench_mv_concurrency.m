/* LENS: why prefill fills the machine and decode does not — part 2.
 *
 * Part 1 (tests/bench_encode_shape.m) showed the launch/barrier side: a serial
 * dispatch costs ~3.5 us on M1 Max no matter how little it does, and a
 * concurrent encoder recovers all of it for independent siblings.
 *
 * This harness asks the OTHER half of the same question, which is worth much
 * more: prefill's GEMMs saturate DRAM (bench_membw: 752-762 GB/s on the rig,
 * 357-365 on M1 Max, ~90-95% of spec).  ds4's decode matvecs do not — the MoE
 * matvec measures ~408-480 GB/s against a 760 GB/s roof (U1/U6), and the dense
 * Q8_0 matvecs 445-530.  Nobody has asked whether that shortfall is a property
 * of ONE matvec dispatch that TWO concurrent matvec dispatches would not have.
 *
 * If two independent Q8_0 matvecs in a concurrent encoder reach materially
 * more aggregate GB/s than the same two run back-to-back in a serial encoder,
 * then a decode layer's independent siblings (router || shared gate/up,
 * routed MoE gate/up || shared expert, q_a || kv before they were fused) are
 * leaving memory bandwidth on the floor purely because MTLDispatchTypeSerial
 * forbids them from overlapping.  That is a scheduling fix, not a kernel
 * rewrite, and it is bit-identical by construction.
 *
 * If aggregate bandwidth does NOT rise, the matvec shortfall is intrinsic to
 * the kernel's access pattern and concurrency buys only the ~3.5 us launch.
 * Either answer closes the question.
 *
 * The kernel is a transcription of ds4's kernel_mul_mv_q8_0_f32_impl
 * (metal/dense.metal:110-183) at the production decode shape: NR0 = 2,
 * NSG = 2 (ds4_gpu_make_q8_0_mv_dispatch picks 2 under TP world 2,
 * ds4_metal.m:5275), NQ = 8, 32-wide simdgroups, grid = out_dim/2.
 *
 *   clang -O2 -fobjc-arc -o tests/bench_mv_concurrency tests/bench_mv_concurrency.m \
 *       -framework Foundation -framework Metal
 *   ./tests/bench_mv_concurrency [k] [out_dim] [n_matvec] [reps]
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static NSString *const kSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"#define QK8_0 32\n"
"typedef struct { half d; char qs[QK8_0]; } block_q8_0;\n"   /* 34 bytes, as ds4 */
"kernel void mv_q8_0(device const char  *src0 [[buffer(0)]],\n"
"                    device const float *src1 [[buffer(1)]],\n"
"                    device float       *dst  [[buffer(2)]],\n"
"                    constant uint      &ne00 [[buffer(3)]],\n"
"                    constant uint      &nb01 [[buffer(4)]],\n"
"                    threadgroup float  *shmem [[threadgroup(0)]],\n"
"                    uint3  tgpig [[threadgroup_position_in_grid]],\n"
"                    ushort tiisg [[thread_index_in_simdgroup]],\n"
"                    ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    const short NSG = 2, NR0 = 2, NW = 32, NQ = 8;\n"
"    const int nb = ne00 / QK8_0;\n"
"    const int r0 = tgpig.x * NR0;\n"
"    device const block_q8_0 *ax[2];\n"
"    for (short row = 0; row < NR0; ++row)\n"
"        ax[row] = (device const block_q8_0 *)(src0 + (ulong)(r0 + row) * nb01);\n"
"    float sumf[2] = {0.f, 0.f};\n"
"    const short ix = tiisg / (NW / NQ);\n"
"    const short il = tiisg % (NW / NQ);\n"
"    const int ib0 = sgitg * NQ + ix;\n"
"    float yl[8];\n"
"    device const float *yb = src1 + ib0 * QK8_0 + il * NQ;\n"
"    for (int ib = ib0; ib < nb; ib += NSG * NQ) {\n"
"        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];\n"
"        for (short row = 0; row < NR0; row++) {\n"
"            device const char *qs = ax[row][ib].qs + il * NQ;\n"
"            float sumq = 0.f;\n"
"            for (short i = 0; i < NQ; ++i) sumq += (float)qs[i] * yl[i];\n"
"            sumf[row] += sumq * (float)ax[row][ib].d;\n"
"        }\n"
"        yb += NSG * NQ * QK8_0;\n"
"    }\n"
"    for (short row = 0; row < NR0; ++row) {\n"
"        float t = simd_sum(sumf[row]);\n"
"        if (tiisg == 0) shmem[sgitg * 2 + row] = t;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sgitg == 0 && tiisg < 2) {\n"
"        float t = 0.f;\n"
"        for (short s = 0; s < 2; ++s) t += shmem[s * 2 + tiisg];\n"
"        dst[r0 + tiisg] = t;\n"
"    }\n"
"}\n";

int main(int argc, char **argv) {
    @autoreleasepool {
        const uint32_t k       = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 4096u;
        const uint32_t out_dim = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 4096u;
        const int      NMV     = argc > 3 ? atoi(argv[3]) : 48;   /* distinct weight slices */
        const int      reps    = argc > 4 ? atoi(argv[4]) : 9;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mv_q8_0"] error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        const uint32_t nb01 = (k / 32u) * 34u;                 /* Q8_0 row bytes */
        const size_t   wbytes_one = (size_t)nb01 * out_dim;    /* one matvec's weights */
        const size_t   wbytes = wbytes_one * (size_t)NMV;
        printf("device %s\n", dev.name.UTF8String);
        printf("k=%u out=%u  %d matvecs x %.1f MiB = %.2f GiB of distinct weights\n",
               k, out_dim, NMV, wbytes_one / 1048576.0, wbytes / 1073741824.0);
        printf("grid %u threadgroups x 64 threads (NR0=2, NSG=2), as ds4 decode under TP2\n",
               out_dim / 2u);

        id<MTLBuffer> W = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> X = [dev newBufferWithLength:(size_t)k * sizeof(float) * NMV
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:(size_t)out_dim * sizeof(float) * NMV
                                           options:MTLResourceStorageModeShared];
        if (!W || !X || !Y) { fprintf(stderr, "alloc failed (need %.1f GiB)\n", wbytes/1073741824.0); return 1; }
        memset(W.contents, 0x11, wbytes);
        float *hx = (float *)X.contents;
        for (size_t i = 0; i < (size_t)k * NMV; i++) hx[i] = (float)((int)(i % 17) - 8) * 0.1f;
        memset(Y.contents, 0, (size_t)out_dim * sizeof(float) * NMV);

        /* Warm the GPU-side page tables for the whole weight region. */
        for (int wu = 0; wu < 2; wu++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int m = 0; m < NMV; m++) {
                [enc setComputePipelineState:pso];
                [enc setBuffer:W offset:(NSUInteger)(m * wbytes_one) atIndex:0];
                [enc setBuffer:X offset:(NSUInteger)(m * k * sizeof(float)) atIndex:1];
                [enc setBuffer:Y offset:(NSUInteger)(m * out_dim * sizeof(float)) atIndex:2];
                [enc setBytes:&k length:sizeof(k) atIndex:3];
                [enc setBytes:&nb01 length:sizeof(nb01) atIndex:4];
                [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(out_dim / 2u, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
            }
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        float *ref = (float *)malloc((size_t)out_dim * sizeof(float) * NMV);
        memcpy(ref, Y.contents, (size_t)out_dim * sizeof(float) * NMV);

        /* G = 0 serial encoder; G = k means concurrent with a barrier every k
         * dispatches, i.e. k independent matvecs allowed to overlap. */
        printf("\n  %-26s %10s %12s %10s\n", "schedule", "ms/matvec", "agg GB/s", "vs serial");
        double serial_gbs = 0;
        const int gv[] = {0, 1, 2, 3, 4, 6, 8, 16};
        for (size_t gi = 0; gi < sizeof(gv)/sizeof(gv[0]); gi++) {
            const int G = gv[gi];
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                memset(Y.contents, 0, (size_t)out_dim * sizeof(float) * NMV);
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = (G == 0)
                    ? [cb computeCommandEncoder]
                    : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                for (int m = 0; m < NMV; m++) {
                    if (G > 0 && m > 0 && (m % G) == 0)
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:W offset:(NSUInteger)(m * wbytes_one) atIndex:0];
                    [enc setBuffer:X offset:(NSUInteger)(m * k * sizeof(float)) atIndex:1];
                    [enc setBuffer:Y offset:(NSUInteger)(m * out_dim * sizeof(float)) atIndex:2];
                    [enc setBytes:&k length:sizeof(k) atIndex:3];
                    [enc setBytes:&nb01 length:sizeof(nb01) atIndex:4];
                    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(out_dim / 2u, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                double t = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;   /* ms for all NMV */
                if (t > 0 && t < best) best = t;
            }
            /* Correctness: every schedule must reproduce the serial output. */
            size_t bad = 0;
            const float *got = (const float *)Y.contents;
            for (size_t i = 0; i < (size_t)out_dim * NMV; i++)
                if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) bad++;
            const double gbs = (double)wbytes / (best * 1e-3) / 1e9;
            if (G == 0) serial_gbs = gbs;
            char name[64];
            if (G == 0) snprintf(name, sizeof(name), "serial encoder");
            else snprintf(name, sizeof(name), "concurrent, %d-wide group", G);
            printf("  %-26s %10.4f %12.1f %9.2fx%s\n", name, best / NMV, gbs,
                   gbs / serial_gbs, bad ? "  ** OUTPUT MISMATCH **" : "");
        }
        free(ref);
        return 0;
    }
}
