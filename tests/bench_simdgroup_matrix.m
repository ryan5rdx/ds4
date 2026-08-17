/* Is simdgroup_matrix actually faster arithmetic than plain FMA on this GPU?
 *
 * The question matters because decode is a batch-1 matvec and never touches
 * the matrix path (the routed MoE switches to kernel_mul_mm_id only at
 * n_tokens >= 32, ds4_metal.m). Before asking whether decode SHOULD use it, it
 * is worth knowing whether the instruction is a throughput win at all.
 *
 * Apple GPUs before M3 have no dedicated matrix unit in the GPU -- AMX is a
 * CPU-side coprocessor -- so simdgroup_multiply_accumulate is expected to lower
 * onto the ordinary FP32 ALUs. If that is right, it is a code-shape
 * convenience for GEMM, not a multiplier, and no amount of using it can help a
 * kernel that is not FLOP-bound.
 *
 * Both kernels are pure arithmetic with enough independent accumulators to
 * hide latency, so this measures issue throughput, not memory.
 *
 *   clang -O3 -fobjc-arc -framework Foundation -framework Metal \
 *       -o tests/bench_simdgroup_matrix tests/bench_simdgroup_matrix.m
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>

static const char *kSrc =
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
"\n"
"/* 8 independent chains so the FMA latency (~3 cycles) is fully hidden and we\n"
" * measure issue rate. */\n"
"kernel void bench_fma(device float *out, constant uint &iters,\n"
"                      uint tid [[thread_position_in_grid]]) {\n"
"    float a = (float)(tid & 7) * 0.5f + 1.0f;\n"
"    float b = (float)(tid & 3) * 0.25f + 1.0f;\n"
"    float c0=0,c1=1,c2=2,c3=3,c4=4,c5=5,c6=6,c7=7;\n"
"    for (uint i = 0; i < iters; i++) {\n"
"        c0 = fma(a,b,c0); c1 = fma(a,b,c1); c2 = fma(a,b,c2); c3 = fma(a,b,c3);\n"
"        c4 = fma(a,b,c4); c5 = fma(a,b,c5); c6 = fma(a,b,c6); c7 = fma(a,b,c7);\n"
"    }\n"
"    out[tid] = ((c0+c1)+(c2+c3))+((c4+c5)+(c6+c7));\n"
"}\n"
"\n"
"/* 4 independent 8x8 accumulators, same role as the 8 scalar chains. */\n"
"kernel void bench_mat(device float *out, constant uint &iters,\n"
"                      threadgroup float *scratch [[threadgroup(0)]],\n"
"                      uint tgid [[threadgroup_position_in_grid]],\n"
"                      ushort tiitg [[thread_index_in_threadgroup]],\n"
"                      ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    simdgroup_float8x8 A = simdgroup_float8x8(1.0009765625f);\n"
"    simdgroup_float8x8 B = simdgroup_float8x8(0.9990234375f);\n"
"    simdgroup_float8x8 C0 = simdgroup_float8x8(0.0f);\n"
"    simdgroup_float8x8 C1 = simdgroup_float8x8(1.0f);\n"
"    simdgroup_float8x8 C2 = simdgroup_float8x8(2.0f);\n"
"    simdgroup_float8x8 C3 = simdgroup_float8x8(3.0f);\n"
"    for (uint i = 0; i < iters; i++) {\n"
"        simdgroup_multiply_accumulate(C0, A, B, C0);\n"
"        simdgroup_multiply_accumulate(C1, A, B, C1);\n"
"        simdgroup_multiply_accumulate(C2, A, B, C2);\n"
"        simdgroup_multiply_accumulate(C3, A, B, C3);\n"
"    }\n"
"    threadgroup float *dst = scratch + (uint)sgitg * 64u;\n"
"    simdgroup_store(C0, dst, 8);\n"
"    simdgroup_store(C1, dst, 8);\n"
"    simdgroup_store(C2, dst, 8);\n"
"    simdgroup_store(C3, dst, 8);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tiitg == 0) out[tgid] = scratch[0];\n"
"}\n";

typedef struct { double ms; double gflops; } result;

static result run(id<MTLDevice> dev, id<MTLCommandQueue> q,
                  id<MTLComputePipelineState> pso, id<MTLBuffer> out,
                  NSUInteger threadgroups, NSUInteger tg_size,
                  NSUInteger tg_mem, uint32_t iters, double flop_per_iter_total) {
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBytes:&iters length:sizeof(iters) atIndex:1];
        if (tg_mem) [enc setThreadgroupMemoryLength:tg_mem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        if (ms > 0 && ms < best) best = ms;
    }
    result r;
    r.ms = best;
    r.gflops = flop_per_iter_total * (double)iters / (best / 1000.0) / 1e9;
    return r;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                               options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> pf =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bench_fma"] error:&err];
        id<MTLComputePipelineState> pm =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bench_mat"] error:&err];
        if (!pf || !pm) {
            fprintf(stderr, "pipeline failed: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        fprintf(stderr, "compiled ok, running...\n");

        /* Enough simdgroups to fill the machine several times over. */
        const NSUInteger TG = 512, TG_SIZE = 256;
        const NSUInteger simds = TG * TG_SIZE / 32;
        const uint32_t iters = 2000;

        id<MTLBuffer> out = [dev newBufferWithLength:TG * TG_SIZE * sizeof(float)
                                             options:MTLResourceStorageModeShared];

        /* scalar: per thread per iter, 8 FMA = 16 FLOP */
        const double fma_flop = (double)TG * TG_SIZE * 8.0 * 2.0;
        /* matrix: per simdgroup per iter, 4 x (8x8x8 MAC) = 4 * 1024 FLOP */
        const double mat_flop = (double)simds * 4.0 * 8.0 * 8.0 * 8.0 * 2.0;

        result rf = run(dev, q, pf, out, TG, TG_SIZE, 0, iters, fma_flop);
        result rm = run(dev, q, pm, out, TG, TG_SIZE,
                        (TG_SIZE / 32) * 64 * sizeof(float), iters, mat_flop);

        printf("%s\n", dev.name.UTF8String); fflush(stdout);
        printf("  %-26s %8.2f ms   %8.1f GFLOP/s\n", "plain FMA (f32)", rf.ms, rf.gflops);
        printf("  %-26s %8.2f ms   %8.1f GFLOP/s\n", "simdgroup_matrix (f32)", rm.ms, rm.gflops);
        printf("\n  matrix / scalar throughput ratio: %.2fx\n", rm.gflops / rf.gflops);
        printf("  (>1 means the matrix instruction is a real arithmetic win;\n"
               "   ~1 or less means it lowers onto the same ALUs and only\n"
               "   changes code shape, so it cannot help a memory-bound kernel)\n");
        return 0;
    }
}
