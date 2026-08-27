/* Dispatch-tax distribution probe (scouting instrument, model-free).
 *
 * WHY THIS EXISTS
 *
 * The decode graph is 1021 serialized dispatches/token.  Two numbers are on
 * record for what one costs and they differ by 6x:
 *
 *   ~1.9-3.7 us   marginal, from DS4_METAL_DISPATCH_BALLAST no-ops inserted
 *                 among real work (tp_decode_investigation.md section 6)
 *   ~22 us        from tests/bench_qkv_norm, flat across a 64x range of q_n
 *
 * Both were single points.  Nobody has measured the *distribution*: what a
 * dispatch costs as a function of (a) how many threadgroups it launches,
 * (b) how many bytes it must reach DRAM for, (c) whether it depends on its
 * predecessor, (d) serial vs concurrent encoder.  Without that you cannot
 * price a fusion campaign, because "remove 185 dispatches" is worth 0.35 ms
 * at 1.9 us and 4.1 ms at 22 us.
 *
 * Method: everything is measured as a SLOPE.  Each arm is run at several
 * dispatch counts inside one command buffer and the per-dispatch cost is the
 * least-squares slope of GPU busy vs N.  That cancels command-buffer fixed
 * cost, which is what made the two prior numbers incomparable.
 *
 * GPU busy is cb.GPUEndTime - cb.GPUStartTime, the same clock the engine's
 * stage profiler uses.
 *
 *   make tests/bench_dispatch_tax
 *   ./tests/bench_dispatch_tax
 *   ./tests/bench_dispatch_tax --reps 9
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

/* A. no-op: touches nothing.  The ballast shape. */
kernel void k_noop(device float *out [[buffer(0)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint tid  [[thread_position_in_threadgroup]]) {
    if (tgid == 0xffffffffu) out[0] = 1.0f;   /* never taken; keeps `out` live */
}

/* B. one dependent scalar: reads a single float the previous dispatch wrote,
 *    writes a single float.  Minimum possible DRAM round trip. */
kernel void k_dep1(device const float *in [[buffer(0)]],
                   device float *out [[buffer(1)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint tid  [[thread_position_in_threadgroup]]) {
    float v = in[0];
    if (tid == 0) out[tgid] = v + 1.0f;
}

/* C. small strided read: each threadgroup reduces `per_tg` floats out of a
 *    buffer far larger than any cache, then writes one float.  This is the
 *    shape of q_lora_norm / hc_pre: a handful of threadgroups, a few KB each,
 *    a tree reduction, and everything downstream waiting. */
kernel void k_read(device const float *src [[buffer(0)]],
                   device float *out [[buffer(1)]],
                   constant uint &per_tg [[buffer(2)]],
                   constant uint &base [[buffer(3)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint tid  [[thread_position_in_threadgroup]],
                   uint nthr [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    uint off = base + tgid * per_tg;
    float acc = 0.0f;
    for (uint i = tid; i < per_tg; i += nthr) acc += src[off + i];
    sh[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthr / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[tgid] = sh[0];
}

/* D. streaming read: many threadgroups, lots of bytes.  Models a real matvec
 *    -- the dispatches that are supposed to be filling the machine. */
kernel void k_stream(device const float4 *src [[buffer(0)]],
                     device float *out [[buffer(1)]],
                     constant uint &vec_per_tg [[buffer(2)]],
                     constant uint &base [[buffer(3)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint tid  [[thread_position_in_threadgroup]],
                     uint nthr [[threads_per_threadgroup]]) {
    uint off = base + tgid * vec_per_tg;
    float4 acc = 0.0f;
    for (uint i = tid; i < vec_per_tg; i += nthr) acc += src[off + i];
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 12345.678f) out[tgid] = s;   /* sink; never taken */
}
)MSL";

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLLibrary> g_lib;

static id<MTLComputePipelineState> pso(const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { fprintf(stderr, "no function %s\n", name); exit(1); }
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "pso %s: %s\n", name, [[err localizedDescription] UTF8String]); exit(1); }
    return p;
}

typedef struct {
    int   kind;        /* 0 noop, 1 dep1, 2 read, 3 stream */
    int   tgs;         /* threadgroups */
    int   threads;     /* threads per threadgroup */
    uint32_t per_tg;   /* elements per threadgroup for kind 2/3 */
    int   chain;       /* 1 = each dispatch reads the previous one's output */
    int   concurrent;  /* 1 = MTLDispatchTypeConcurrent encoder */
    int   interleave;  /* >0 = insert one k_stream fat dispatch every N tiny ones */
    int   new_encoder; /* 1 = end and reopen the compute encoder after each dispatch */
} arm_cfg;

static id<MTLBuffer> g_src;       /* big cold source */
static id<MTLBuffer> g_out[64];   /* rotated outputs */
static id<MTLComputePipelineState> g_noop, g_dep1, g_read, g_stream;

/* One command buffer containing `n` dispatches of the arm.  Returns GPU busy. */
static double run_once(const arm_cfg *a, int n, uint32_t *seed) {
    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> enc = a->concurrent
        ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
        : [cb computeCommandEncoder];
    const uint32_t src_vec4 = (uint32_t)(g_src.length / 16);
    for (int i = 0; i < n; i++) {
        if (a->interleave > 0 && i > 0 && (i % a->interleave) == 0) {
            /* fat streaming dispatch: 4096 threadgroups x 256 float4 = 16 MB */
            uint32_t vpt = 256, base = 0;
            [enc setComputePipelineState:g_stream];
            [enc setBuffer:g_src offset:0 atIndex:0];
            [enc setBuffer:g_out[0] offset:0 atIndex:1];
            [enc setBytes:&vpt length:4 atIndex:2];
            [enc setBytes:&base length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(4096, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            if (a->concurrent == 1) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        switch (a->kind) {
        case 0:
            [enc setComputePipelineState:g_noop];
            [enc setBuffer:g_out[i % 64] offset:0 atIndex:0];
            break;
        case 1:
            [enc setComputePipelineState:g_dep1];
            [enc setBuffer:(a->chain ? g_out[(i + 63) % 64] : g_src) offset:0 atIndex:0];
            [enc setBuffer:g_out[i % 64] offset:0 atIndex:1];
            break;
        case 2: {
            /* walk the source so nothing is ever cache-resident */
            *seed = *seed * 1664525u + 1013904223u;
            uint32_t span = (uint32_t)a->tgs * a->per_tg;
            uint32_t base = span >= (g_src.length / 4) ? 0
                          : (*seed % ((uint32_t)(g_src.length / 4) - span)) & ~255u;
            [enc setComputePipelineState:g_read];
            [enc setBuffer:g_src offset:0 atIndex:0];
            [enc setBuffer:g_out[i % 64] offset:0 atIndex:1];
            [enc setBytes:&a->per_tg length:4 atIndex:2];
            [enc setBytes:&base length:4 atIndex:3];
            break;
        }
        case 3: {
            *seed = *seed * 1664525u + 1013904223u;
            uint32_t span = (uint32_t)a->tgs * a->per_tg;
            uint32_t base = span >= src_vec4 ? 0 : (*seed % (src_vec4 - span)) & ~255u;
            [enc setComputePipelineState:g_stream];
            [enc setBuffer:g_src offset:0 atIndex:0];
            [enc setBuffer:g_out[i % 64] offset:0 atIndex:1];
            [enc setBytes:&a->per_tg length:4 atIndex:2];
            [enc setBytes:&base length:4 atIndex:3];
            break;
        }
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)a->tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)a->threads, 1, 1)];
        if (a->concurrent == 1) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        if (a->new_encoder) {
            [enc endEncoding];
            enc = a->concurrent
                ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                : [cb computeCommandEncoder];
        }
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e6;   /* us */
}

/* Best-of `reps` at each N, then least-squares slope in us/dispatch. */
static void measure(const char *label, const arm_cfg *a, int reps) {
    const int ns[] = {43, 129, 258, 430};
    const int nn = 4;
    double y[4];
    uint32_t seed = 12345u;
    for (int k = 0; k < nn; k++) {
        double best = 1e18;
        for (int r = 0; r < reps; r++) {
            double t = run_once(a, ns[k], &seed);
            if (t < best) best = t;
        }
        y[k] = best;
    }
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int k = 0; k < nn; k++) { sx += ns[k]; sy += y[k]; sxx += (double)ns[k]*ns[k]; sxy += (double)ns[k]*y[k]; }
    double slope = (nn*sxy - sx*sy) / (nn*sxx - sx*sx);
    double icept = (sy - slope*sx) / nn;
    printf("%-42s slope %7.3f us/dispatch   cb_fixed %7.1f us   [",
           label, slope, icept);
    for (int k = 0; k < nn; k++) printf("%s%.0f", k ? " " : "", y[k]);
    printf(" us @ N=43/129/258/430]\n");
}

int main(int argc, char **argv) {
    int reps = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    }
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) { fprintf(stderr, "no metal device\n"); return 1; }
    g_q = [g_dev newCommandQueue];
    NSError *err = nil;
    MTLCompileOptions *opt = [MTLCompileOptions new];
    g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                options:opt error:&err];
    if (!g_lib) { fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    g_noop = pso("k_noop"); g_dep1 = pso("k_dep1");
    g_read = pso("k_read"); g_stream = pso("k_stream");

    const size_t src_bytes = 1024ull * 1024 * 1024;   /* 1 GiB, far past any cache */
    g_src = [g_dev newBufferWithLength:src_bytes options:MTLResourceStorageModePrivate];
    for (int i = 0; i < 64; i++)
        g_out[i] = [g_dev newBufferWithLength:1 << 20 options:MTLResourceStorageModePrivate];

    printf("device: %s  (%lu cores reported by Metal: n/a)\n",
           [[g_dev name] UTF8String], (unsigned long)0);
    printf("source buffer %.1f GiB private; %d reps, best-of; slope over N=43..430 in ONE command buffer\n\n",
           src_bytes / 1073741824.0, reps);

    printf("--- 1. what does an empty dispatch cost? (the ballast shape) ---\n");
    { arm_cfg a = {0, 1,   32, 0, 0, 0, 0, 0}; measure("noop, 1 threadgroup", &a, reps); }
    { arm_cfg a = {0, 64,  32, 0, 0, 0, 0, 0}; measure("noop, 64 threadgroups", &a, reps); }
    { arm_cfg a = {0, 1024,32, 0, 0, 0, 0, 0}; measure("noop, 1024 threadgroups", &a, reps); }

    printf("\n--- 2. one dependent scalar: does the dependency itself cost? ---\n");
    { arm_cfg a = {1, 2, 32, 0, 0, 0, 0, 0}; measure("dep1, 2 tg, INDEPENDENT (reads src)", &a, reps); }
    { arm_cfg a = {1, 2, 32, 0, 1, 0, 0, 0}; measure("dep1, 2 tg, CHAINED (reads prev out)", &a, reps); }
    { arm_cfg a = {1, 64,32, 0, 1, 0, 0, 0}; measure("dep1, 64 tg, CHAINED", &a, reps); }

    printf("\n--- 3. small cold read: the q_lora_norm / hc_pre shape ---\n");
    /* per_tg floats per threadgroup out of a 1 GiB cold buffer */
    { arm_cfg a = {2, 2,  256, 1024, 0, 0, 0, 0}; measure("read 2 tg x 4 KB   (8 KB total)", &a, reps); }
    { arm_cfg a = {2, 6,  256, 1024, 0, 0, 0, 0}; measure("read 6 tg x 4 KB   (24 KB total)", &a, reps); }
    { arm_cfg a = {2, 32, 256, 1024, 0, 0, 0, 0}; measure("read 32 tg x 4 KB  (128 KB total)", &a, reps); }
    { arm_cfg a = {2, 64, 256, 1024, 0, 0, 0, 0}; measure("read 64 tg x 4 KB  (256 KB total)", &a, reps); }
    { arm_cfg a = {2, 256,256, 1024, 0, 0, 0, 0}; measure("read 256 tg x 4 KB (1 MB total)", &a, reps); }

    printf("\n--- 4. same bytes, fewer/more threadgroups (pure underfill curve) ---\n");
    /* 768 KB total in every arm; only the grid shape changes */
    { arm_cfg a = {2, 2,   256, 98304, 0, 0, 0, 0}; measure("768 KB over 2 tg", &a, reps); }
    { arm_cfg a = {2, 6,   256, 32768, 0, 0, 0, 0}; measure("768 KB over 6 tg", &a, reps); }
    { arm_cfg a = {2, 24,  256, 8192,  0, 0, 0, 0}; measure("768 KB over 24 tg", &a, reps); }
    { arm_cfg a = {2, 96,  256, 2048,  0, 0, 0, 0}; measure("768 KB over 96 tg", &a, reps); }
    { arm_cfg a = {2, 192, 256, 1024,  0, 0, 0, 0}; measure("768 KB over 192 tg", &a, reps); }
    { arm_cfg a = {2, 768, 256, 256,   0, 0, 0, 0}; measure("768 KB over 768 tg", &a, reps); }

    printf("\n--- 5. a fat streaming dispatch, for scale ---\n");
    { arm_cfg a = {3, 4096, 256, 256, 0, 0, 0, 0}; measure("stream 16 MB, 4096 tg", &a, reps); }
    { arm_cfg a = {3, 1024, 256, 256, 0, 0, 0, 0}; measure("stream 4 MB, 1024 tg", &a, reps); }

    printf("\n--- 6. does a tiny dispatch cost less when big work surrounds it? ---\n");
    { arm_cfg a = {2, 2, 256, 1024, 0, 0, 4, 0}; measure("read 2 tg x 4 KB, 1 fat per 4", &a, reps); }
    { arm_cfg a = {0, 1, 32,  0,    0, 0, 4, 0}; measure("noop, 1 fat per 4", &a, reps); }

    printf("\n--- 7. concurrent encoder, explicit barrier after each dispatch ---\n");
    { arm_cfg a = {2, 2, 256, 1024, 0, 1, 0, 0}; measure("read 2 tg x 4 KB, CONCURRENT+barrier", &a, reps); }
    { arm_cfg a = {1, 2, 32,  0,    1, 1, 0, 0}; measure("dep1 chained, CONCURRENT+barrier", &a, reps); }

    printf("\n--- 8. concurrent encoder, NO barrier (upper bound on overlap) ---\n");
    { arm_cfg a = {2, 2, 256, 1024, 0, 2, 0, 0}; measure("read 2 tg x 4 KB, CONCURRENT no barrier", &a, reps); }
    { arm_cfg a = {0, 1, 32,  0,    0, 2, 0, 0}; measure("noop, CONCURRENT no barrier", &a, reps); }

    printf("\n--- 9. encoder-boundary tax (ds4 reopens the encoder 172x/token) ---\n");
    { arm_cfg a = {2, 2, 256, 1024, 0, 0, 0, 0}; measure("read 2 tg x 4 KB, ONE encoder", &a, reps); }
    { arm_cfg a = {2, 2, 256, 1024, 0, 0, 0, 1}; measure("read 2 tg x 4 KB, encoder PER dispatch", &a, reps); }
    { arm_cfg a = {0, 1, 32,  0,    0, 0, 0, 0}; measure("noop, ONE encoder", &a, reps); }
    { arm_cfg a = {0, 1, 32,  0,    0, 0, 0, 1}; measure("noop, encoder PER dispatch", &a, reps); }

    printf("\n--- 10. fusion payoff: K tiny dispatches vs 1 dispatch of K x work ---\n");
    { arm_cfg a = {2, 2,  256, 1024, 0, 0, 0, 0}; measure("1x: 2 tg x 4 KB", &a, reps); }
    { arm_cfg a = {2, 4,  256, 1024, 0, 0, 0, 0}; measure("fused 2x: 4 tg x 4 KB", &a, reps); }
    { arm_cfg a = {2, 8,  256, 1024, 0, 0, 0, 0}; measure("fused 4x: 8 tg x 4 KB", &a, reps); }
    { arm_cfg a = {2, 14, 256, 1024, 0, 0, 0, 0}; measure("fused 7x: 14 tg x 4 KB", &a, reps); }
    { arm_cfg a = {2, 48, 256, 1024, 0, 0, 0, 0}; measure("fused 24x: 48 tg x 4 KB", &a, reps); }
    return 0;
}
