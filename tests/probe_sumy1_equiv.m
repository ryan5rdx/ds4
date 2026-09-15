/*
 * SUMY1 exactness: does the producer reproduce the in-kernel sums BIT FOR BIT?
 *
 * The pair kernel rebuilds four activation-group sums for every output tile and
 * every selected expert, though they depend only on the activation and the
 * lane's (ix, iq, ir). SUMY1 produces them once. The saving is real -- an
 * intentionally wrong stub took 5.0% off the 42-layer pair+down chain on
 * Apple8 -- but the saving is worthless unless the sums are identical.
 *
 * "Same mathematical sum" is NOT the bar. Each sum is eight strict
 * left-to-right float additions; any reassociation gives a different float,
 * which changes the dequantised value and therefore the token. This codebase
 * has already been bitten by reassociation in the routed logits, so the
 * producer copies the consumer's loop verbatim rather than computing the same
 * quantity a tidier way -- and this probe is what proves it did.
 *
 * It compares the producer against a GPU REFERENCE kernel running the
 * consumer's loop verbatim -- not against a CPU replica. The CPU is the wrong
 * oracle: Metal flushes denormals to zero and the host does not, so a
 * denormal-bearing activation makes them disagree on ~80% of sums by ~1e-42,
 * which says nothing about the producer. The first version of this probe used a
 * CPU reference and failed exactly that way.
 *
 * Activations are chosen to expose ordering: catastrophic cancellation, wide
 * exponent spreads, denormals, and signed zero.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256

typedef struct {
    uint32_t in_dim, mid_dim, out_dim, n_total_expert, n_expert_used, n_tokens;
    uint32_t mid_token_stride, down_type;
    float    swiglu_clamp;
    int32_t  tp_rank, tp_world, tp_expert_base;
    uint64_t gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes,
             down_expert_bytes, down_row_bytes;
} moe_args;

static int fails;

int main(int argc, const char **argv) { @autoreleasepool {
    const char *src_path = argc > 1 ? argv[1] : "/tmp/m.metal";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *e = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(src_path)
                                              encoding:NSUTF8StringEncoding error:&e];
    if (!src) { printf("VOID: cannot read %s\n", src_path); return 2; }
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&e];
    if (!lib) { printf("VOID: %s\n", e.description.UTF8String); return 2; }
    id<MTLComputePipelineState> pso[2];
    const char *kn[2] = { "kernel_glm53_sumy1_produce", "kernel_glm53_sumy1_reference" };
    for (int i = 0; i < 2; i++) {
        id<MTLFunction> f = [lib newFunctionWithName:@(kn[i])];
        if (!f) { printf("VOID: %s missing\n", kn[i]); return 2; }
        pso[i] = [dev newComputePipelineStateWithFunction:f error:&e];
        if (!pso[i]) { printf("VOID: pipeline %s: %s\n", kn[i], e.description.UTF8String); return 2; }
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    printf("device: %s\n\n", dev.name.UTF8String);

    const uint32_t in_dim = 4096, n_tokens = 4;
    const uint32_t nb = in_dim / QK_K;                 /* 16 */
    const size_t n_sums = (size_t)n_tokens * nb * 32;  /* 512 floats/token */

    /* Activations designed so ordering MATTERS. A uniform random vector sums
     * the same in almost any order; these do not. */
    const char *case_name[] = { "uniform random", "catastrophic cancellation",
                                "wide exponent spread", "denormals + signed zero" };
    for (int c = 0; c < 4; c++) {
        float *xs = malloc((size_t)n_tokens * in_dim * sizeof(float));
        uint32_t st = 0x9E3779B9u ^ (uint32_t)c;
        for (size_t i = 0; i < (size_t)n_tokens * in_dim; i++) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            switch (c) {
            case 0: xs[i] = (float)((int32_t)st) / 1.0e9f; break;
            case 1: /* alternating +/- huge with a tiny residue: the sum depends
                     * entirely on the order the cancellations happen in */
                xs[i] = (i & 1) ? 1.0e7f : -1.0e7f + 1.0e-3f; break;
            case 2: xs[i] = ldexpf(1.0f + (float)(st & 0xffff) / 65536.0f,
                                   (int)(st % 60) - 30); break;
            default:
                xs[i] = (st & 3u) == 0 ? 0.0f
                      : (st & 3u) == 1 ? -0.0f
                      : (st & 3u) == 2 ? ldexpf(1.0f, -140)   /* denormal */
                                       : -ldexpf(1.0f, -140);
            }
        }

        id<MTLBuffer> bx = [dev newBufferWithBytes:xs
                length:(size_t)n_tokens * in_dim * sizeof(float)
               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bs = [dev newBufferWithLength:n_sums * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> br = [dev newBufferWithLength:n_sums * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        memset(bs.contents, 0xA5, n_sums * sizeof(float));   /* poison */
        memset(br.contents, 0x5A, n_sums * sizeof(float));

        moe_args a = { .in_dim = in_dim, .mid_dim = 2048, .out_dim = 4096,
                       .n_total_expert = 8, .n_expert_used = 8,
                       .n_tokens = n_tokens, .tp_world = 1 };
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (int k = 0; k < 2; k++) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso[k]];
            [enc setBytes:&a length:sizeof(a) atIndex:0];
            [enc setBuffer:bx offset:0 atIndex:1];
            [enc setBuffer:(k ? br : bs) offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(8, nb, n_tokens)
                  threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];
            [enc endEncoding];
        }
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            printf("FAIL dispatch: %s\n", cb.error.localizedDescription.UTF8String);
            fails++; free(xs); continue;
        }

        const float *got = (const float *)bs.contents;
        const float *ref = (const float *)br.contents;
        size_t bad = 0; float worst = 0.0f;
        for (size_t i = 0; i < n_sums; i++) {
            uint32_t a32, b32;
            memcpy(&a32, &ref[i], 4); memcpy(&b32, &got[i], 4);
            if (a32 != b32) {   /* BITS, so NaN compares as itself */
                bad++;
                const float d = fabsf(ref[i] - got[i]);
                if (d > worst) worst = d;
            }
        }
        printf("%s   %-26s %zu/%zu sums differ%s\n", bad ? "FAIL" : "ok  ",
               case_name[c], bad, n_sums, bad ? "" : " (bit-identical)");
        if (bad) { printf("       worst abs delta %g\n", (double)worst); fails++; }
        free(xs);
    }

    printf("\n%s\n", fails == 0
           ? "PASS: the SUMY1 producer reproduces the consumer's sums bit for bit"
           : "FAILED -- a reassociated sum changes the token; this is not a tolerance gate");
    return fails == 0 ? 0 : 1;
} }
