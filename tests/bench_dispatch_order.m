/*
 * bench_dispatch_order -- model-free measurement of the cost of ONE enforced
 * ordering point in a Metal compute graph, on this machine.
 *
 * Why this exists.  The decode graph is 1021 fully serialized dispatches.  The
 * existing estimate of what one dispatch costs (1.9 us, tp_decode_investigation
 * section 6) came from a marginal-ballast increment, which cannot separate
 * "launch overhead" from "pipeline drain forced by ordering".  This harness
 * separates them: it runs the SAME dispatch stream with N ordering points and
 * with N/G ordering points, interleaved best-of inside one process, and fits
 *
 *     T = D * c + K * w        D = ordering points, K = dispatches
 *
 * c is the quantity we care about.  On an M1 Max (Apple7, 32 core, 400 GB/s)
 * c = 3.2-3.5 us and is flat across a 64x range of per-dispatch work
 * (32 KB / 8 tgs, 256 KB / 64 tgs, 2 MB / 128 tgs).  w is the real kernel time
 * and hits the streaming roof in every arm.
 *
 * All three ways of expressing an ordering point measure the same:
 *   - MTLDispatchTypeSerial dispatch boundary
 *   - one compute encoder per dispatch (either dispatch type)
 *   - memoryBarrierWithScope: inside a concurrent encoder
 * and unordered dispatches inside a concurrent encoder cost 0.06-0.7 us.
 *
 * Build (no model, no ds4 objects needed):
 *   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
 *         -o tests/bench_dispatch_order tests/bench_dispatch_order.m
 * Run:
 *   ./tests/bench_dispatch_order            # sweeps three shapes
 *   ./tests/bench_dispatch_order 512 32768 8
 *
 * The last section is a correctness check, because a throughput measurement
 * cannot detect wrong output: it verifies that Metal's inter-encoder hazard
 * tracking really does order dependent dispatches that live in different
 * encoders, and that dependent dispatches inside one concurrent encoder really
 * do produce garbage.  If the first three lines are not OK, do not trust any
 * number above them.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void stream_reduce(device const uint4 *s [[buffer(0)]],\n"
"                          device const float *dep [[buffer(1)]],\n"
"                          device float *o [[buffer(2)]],\n"
"                          constant uint &n [[buffer(3)]],\n"
"                          uint tid [[thread_position_in_grid]],\n"
"                          uint ntg [[threadgroups_per_grid]],\n"
"                          uint tgid [[threadgroup_position_in_grid]],\n"
"                          uint lid [[thread_position_in_threadgroup]],\n"
"                          uint tgsz [[threads_per_threadgroup]]) {\n"
"    uint4 a = uint4(0); uint g = ntg * tgsz;\n"
"    for (uint i = tid; i < n; i += g) a += s[i];\n"
"    float v = float((a.x + a.y + a.z + a.w) & 255u) + dep[0];\n"
"    float r = simd_sum(v);\n"
"    if (lid == 0) o[tgid] = r;\n"
"}\n"
"kernel void inc(device const float *in [[buffer(0)]],\n"
"                device float *out [[buffer(1)]],\n"
"                uint t [[thread_position_in_grid]]) {\n"
"    if (t == 0) out[0] = in[0] + 1.0f;\n"
"}\n";

static void sweep(id<MTLDevice> d, id<MTLCommandQueue> q,
                  id<MTLComputePipelineState> pso,
                  int K, size_t BYTES, int TGS) {
    const int TPT = 256;
    const size_t POOL = 256u << 20;
    id<MTLBuffer> s = [d newBufferWithLength:POOL options:MTLResourceStorageModePrivate];
    NSMutableArray *B = [NSMutableArray array];
    for (int i = 0; i <= K + 1; i++) {
        [B addObject:[d newBufferWithLength:65536 options:MTLResourceStorageModePrivate]];
    }
    uint32_t n = (uint32_t)(BYTES / 16);
    const int Gs[] = {1, 2, 3, 4, 6, 8, 16};
    const int nG = 7;
    double best[7];
    for (int i = 0; i < nG; i++) best[i] = 1e30;

    /* Islands of G INDEPENDENT dispatches in one concurrent encoder; islands
     * are ordered against each other by Metal's inter-encoder hazard tracking
     * (island t reads what island t-1 wrote).  D = ceil(K/G). */
    for (int rep = 0; rep < 15; rep++) {
        for (int gi = 0; gi < nG; gi++) {
            const int G = Gs[gi];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            int t = 0;
            for (int st = 0; st < K; st += G, t++) {
                const int cnt = (st + G <= K) ? G : (K - st);
                id<MTLComputeCommandEncoder> enc =
                    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [enc setComputePipelineState:pso];
                for (int j = 0; j < cnt; j++) {
                    size_t off = (((size_t)(st + j) * BYTES) % (POOL - BYTES)) & ~(size_t)255;
                    [enc setBuffer:s offset:off atIndex:0];
                    [enc setBuffer:B[t] offset:0 atIndex:1];
                    [enc setBuffer:B[t + 1] offset:(NSUInteger)j * 256 atIndex:2];
                    [enc setBytes:&n length:4 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(TGS, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(TPT, 1, 1)];
                }
                [enc endEncoding];
            }
            [cb commit];
            [cb waitUntilCompleted];
            const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            if (rep >= 3 && ms < best[gi]) best[gi] = ms;
        }
    }
    printf("K=%d  %.0f KB/dispatch  %d threadgroups\n", K, BYTES / 1024.0, TGS);
    for (int i = 0; i < nG; i++) {
        printf("  islands of %2d : %8.3f ms  %6.2f us/dispatch  D=%d ordering points\n",
               Gs[i], best[i], best[i] * 1000 / K, (K + Gs[i] - 1) / Gs[i]);
    }
    const int D0 = K, D1 = (K + 15) / 16;
    printf("  --> cost of ONE ordering point: %.2f us   (kernel work %.2f us/dispatch)\n\n",
           (best[0] - best[6]) * 1000 / (D0 - D1),
           (best[6] * 1000 - (double)D1 * (best[0] - best[6]) * 1000 / (D0 - D1)) / K);
}

int main(int argc, char **argv) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [d newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                         options:nil error:&e];
    if (!lib) { printf("compile: %s\n", [[e localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> pso =
        [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream_reduce"] error:&e];
    id<MTLComputePipelineState> pinc =
        [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"inc"] error:&e];
    if (!pso || !pinc) { printf("pso: %s\n", [[e localizedDescription] UTF8String]); return 1; }
    printf("device: %s\n\n", [d.name UTF8String]);

    if (argc > 3) {
        sweep(d, q, pso, atoi(argv[1]), (size_t)atol(argv[2]), atoi(argv[3]));
    } else {
        sweep(d, q, pso, 512, 32768,   8);
        sweep(d, q, pso, 256, 262144,  64);
        sweep(d, q, pso, 128, 2097152, 128);
    }

    /* Correctness of the ordering primitives themselves. */
    const int K = 2000;
    const char *nm[4] = {
        "one serial encoder, 2000 dependent dispatches   ",
        "one encoder per dispatch (serial type)          ",
        "one CONCURRENT encoder per dispatch             ",
        "concurrent encoders of 4, chain INSIDE an island"
    };
    const int expect_ok[4] = {1, 1, 1, 0};
    for (int arm = 0; arm < 4; arm++) {
        NSMutableArray *B = [NSMutableArray array];
        for (int i = 0; i <= K; i++) {
            id<MTLBuffer> b = [d newBufferWithLength:16 options:MTLResourceStorageModeShared];
            ((float *)b.contents)[0] = 0.0f;
            [B addObject:b];
        }
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = nil;
        for (int i = 0; i < K; i++) {
            if (arm == 0) {
                if (!enc) { enc = [cb computeCommandEncoder]; [enc setComputePipelineState:pinc]; }
            } else if (arm == 1) {
                enc = [cb computeCommandEncoder]; [enc setComputePipelineState:pinc];
            } else if (arm == 2) {
                enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [enc setComputePipelineState:pinc];
            } else if (i % 4 == 0) {
                enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [enc setComputePipelineState:pinc];
            }
            [enc setBuffer:B[i] offset:0 atIndex:0];
            [enc setBuffer:B[i + 1] offset:0 atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            if (arm == 1 || arm == 2) { [enc endEncoding]; enc = nil; }
            else if (arm == 3 && i % 4 == 3) { [enc endEncoding]; enc = nil; }
        }
        if (enc) [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float got = ((float *)((id<MTLBuffer>)B[K]).contents)[0];
        const int ok = (got == (float)K);
        printf("%s final=%6.0f expect=%d  %s\n", nm[arm], (double)got, K,
               ok == expect_ok[arm] ? "as expected" : "*** UNEXPECTED ***");
    }
    return 0;
}}
