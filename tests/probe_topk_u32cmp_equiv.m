/*
 * CMPSEL2 C4: does the _u32cmp top-k arm agree with shipping, and is it live?
 *
 * TWO QUESTIONS, and the second is the one that keeps biting this campaign.
 *
 * 1. EXACTNESS. The arm changes how a packed key is compared, not what order it
 *    implies, so every output must be bit-identical to shipping. These indices
 *    feed attention; a single swapped tie is a wrong token.
 *
 * 2. LIVENESS. The arm has already shipped twice in a state where it could
 *    announce itself and execute the shipping comparison anyway -- first by
 *    arming a kernel the tiled path returns before reaching, then by leaving a
 *    bare `buf[i] < buf[ixj]` in the decode merge. Bit-identical output cannot
 *    distinguish "correctly armed" from "not armed at all": both give zero
 *    differences. So this probe also MUTATES the comparator, via a second
 *    library compiled with ds4_topk_key_gt_split deliberately inverted, and
 *    requires that the _u32cmp kernel's output CHANGES while shipping's does
 *    not. That is the only check that proves the template parameter reaches the
 *    comparison.
 *
 * idxsplit_merge_expand is the DECODE arm and is self-contained enough to
 * dispatch here: two sorted 512-runs in, expanded selection out. The tiled
 * prefill kernels need the indexer's surrounding buffers and are not covered.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

typedef struct { uint32_t top_k, pool_size, index_topk, output_width, pos0; } idxsplit_args;

static uint64_t pack_key(float v, uint32_t idx) {
    uint32_t u; memcpy(&u, &v, 4);
    const uint32_t ordered = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
    return ((uint64_t)ordered << 32) | (uint64_t)(0xffffffffu - idx);
}

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? 1 : x > y ? -1 : 0;     /* descending */
}

/* Two sorted-descending runs of `k` keys, with ties across the boundary so the
 * index half of the key actually decides something. */
static void build_runs(uint64_t *a, uint64_t *b, uint32_t k, int tie_heavy) {
    uint32_t st = 0xC0FFEEu;
    for (uint32_t i = 0; i < k; i++) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        const float va = tie_heavy ? (float)(i % 8) : (float)((int32_t)st) / 1.0e7f;
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        const float vb = tie_heavy ? (float)(i % 8) : (float)((int32_t)st) / 1.0e7f;
        a[i] = pack_key(va, i * 2u);
        b[i] = pack_key(vb, i * 2u + 1u);
    }
    qsort(a, k, sizeof(uint64_t), cmp_u64);
    qsort(b, k, sizeof(uint64_t), cmp_u64);
}

static id<MTLLibrary> build_lib(id<MTLDevice> dev, NSString *src, int mutate) {
    NSError *e = nil;
    if (mutate) {
        /* Invert the split comparator. Only the _u32cmp instantiations route
         * through it, so shipping must be unaffected. */
        NSString *from = @"return (uint)((ah > bh) | ((ah == bh) & (al > bl))) != 0u;";
        NSString *to   = @"return (uint)((ah < bh) | ((ah == bh) & (al < bl))) != 0u;";
        if ([src rangeOfString:from].location == NSNotFound) {
            printf("VOID: could not find the split comparator to mutate\n");
            return nil;
        }
        src = [src stringByReplacingOccurrencesOfString:from withString:to];
    }
    id<MTLLibrary> l = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&e];
    if (!l) printf("VOID: compile failed: %s\n", e.description.UTF8String);
    return l;
}

static int run_kernel(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib,
                      const char *name, const uint64_t *ka, const uint64_t *kb,
                      idxsplit_args a, uint32_t out_n, uint32_t *out) {
    NSError *e = nil;
    id<MTLFunction> f = [lib newFunctionWithName:@(name)];
    if (!f) { printf("FAIL no kernel %s\n", name); return 0; }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
    if (!p) { printf("FAIL pipeline %s: %s\n", name, e.description.UTF8String); return 0; }

    const size_t kb_bytes = (size_t)a.top_k * sizeof(uint64_t);
    id<MTLBuffer> ba = [dev newBufferWithBytes:ka length:kb_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithBytes:kb length:kb_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:out_n * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    memset(bo.contents, 0xA5, out_n * sizeof(uint32_t));   /* poison */

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:ba offset:0 atIndex:1];
    [enc setBuffer:bb offset:0 atIndex:2];
    [enc setBuffer:bo offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:2u * a.top_k * sizeof(uint64_t) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("FAIL dispatch %s: %s\n", name, cb.error.localizedDescription.UTF8String);
        return 0;
    }
    memcpy(out, bo.contents, out_n * sizeof(uint32_t));
    return 1;
}

int main(int argc, const char **argv) { @autoreleasepool {
    const char *src_path = argc > 1 ? argv[1] : "/tmp/m.metal";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *e = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(src_path)
                                              encoding:NSUTF8StringEncoding error:&e];
    if (!src) { printf("VOID: cannot read %s\n", src_path); return 2; }
    id<MTLLibrary> good = build_lib(dev, src, 0);
    id<MTLLibrary> mut  = build_lib(dev, src, 1);
    if (!good || !mut) return 2;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    printf("device: %s\n\n", dev.name.UTF8String);

    const uint32_t k = 512, pool = 4, width = 2048;
    idxsplit_args a = { k, pool, k, width, 0 };
    const uint32_t out_n = width;
    uint64_t *ka = malloc(k * 8), *kb = malloc(k * 8);
    uint32_t *o_ship = malloc(out_n * 4), *o_arm = malloc(out_n * 4),
             *o_mut_ship = malloc(out_n * 4), *o_mut_arm = malloc(out_n * 4);

    for (int tie = 0; tie <= 1; tie++) {
        build_runs(ka, kb, k, tie);
        const char *label = tie ? "tie-heavy (index decides)" : "unique scores";
        int ok = run_kernel(dev, q, good, "kernel_glm53_idxsplit_merge_expand", ka, kb, a, out_n, o_ship)
              && run_kernel(dev, q, good, "kernel_glm53_idxsplit_merge_expand_u32cmp", ka, kb, a, out_n, o_arm)
              && run_kernel(dev, q, mut,  "kernel_glm53_idxsplit_merge_expand", ka, kb, a, out_n, o_mut_ship)
              && run_kernel(dev, q, mut,  "kernel_glm53_idxsplit_merge_expand_u32cmp", ka, kb, a, out_n, o_mut_arm);
        if (!ok) { fails++; continue; }

        const int same_arm  = memcmp(o_ship, o_arm, out_n * 4) == 0;
        const int ship_moved = memcmp(o_ship, o_mut_ship, out_n * 4) != 0;
        const int arm_moved  = memcmp(o_arm,  o_mut_arm,  out_n * 4) != 0;

        printf("## %s\n", label);
        printf("%s   arm output is bit-identical to shipping\n", same_arm ? "ok  " : "FAIL");
        printf("%s   shipping is UNAFFECTED by mutating the split comparator\n",
               !ship_moved ? "ok  " : "FAIL");
        printf("%s   arm IS affected by it -- the template parameter reaches the compare\n",
               arm_moved ? "ok  " : "FAIL");
        if (!same_arm || ship_moved || !arm_moved) fails++;
        printf("\n");
    }

    free(ka); free(kb); free(o_ship); free(o_arm); free(o_mut_ship); free(o_mut_arm);
    printf("%s\n", fails == 0
           ? "PASS: the decode arm agrees with shipping AND is demonstrably live"
           : "FAILED");
    return fails == 0 ? 0 : 1;
} }
