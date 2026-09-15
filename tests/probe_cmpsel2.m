/*
 * CMPSEL2-U32KEY host probe: exactness first, then both comparator shapes.
 *
 * Reopens what CMPSEL-AUDIT set aside. That audit showed ternary / max() /
 * guarded store agree to 0.2% on ulong -- the compiler already emits
 * compare-select -- and then recorded, without acting on it, that ulong
 * compare-select costs ~2.5x uint. It was M1-only and measured no production
 * kernel. Our packed keys are lexicographic (score<<32 | ~idx), so the ordering
 * needs no 64-bit arithmetic at all; only the storage is 64-bit.
 *
 * DO NOT KILL THIS FROM M1. Apple7's 64-bit ALU path is not Apple8's, and the
 * campaign has already been wrong once in each direction about that (WQ8
 * residency was -47% on M1 and -61.5% on Apple8). The dev box establishes
 * EXACTNESS, which is chip-independent, and produces M1 timings as data. The
 * verdict needs both rig ranks.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

/* Mirrors ds4_topk_pack_key / ds4_top1_pack_key exactly. */
static uint64_t pack_key(float v, uint32_t idx) {
    uint32_t u;
    memcpy(&u, &v, 4);
    const uint32_t ordered = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
    return ((uint64_t)ordered << 32) | (uint64_t)(0xffffffffu - idx);
}

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* The adversarial key set. Every case here has produced a real bug in this
 * campaign or is one comparison away from one. */
static uint32_t build_keys(uint64_t *ka, uint64_t *kb, uint32_t cap) {
    const float specials[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 1e-30f, -1e-30f, 3.4e38f, -3.4e38f,
        (float)INFINITY, (float)-INFINITY, 1e30f, -1e30f,   /* the DS4 floor */
    };
    const uint32_t ns = (uint32_t)(sizeof(specials) / sizeof(specials[0]));
    uint32_t n = 0;

    /* 1. every special against every special, at equal and unequal indices --
     *    signed zero and the infinities live here, and so does the exact
     *    tie-on-score/differ-on-index case the index half exists to break. */
    for (uint32_t i = 0; i < ns && n + 4 < cap; i++) {
        for (uint32_t j = 0; j < ns && n + 4 < cap; j++) {
            ka[n] = pack_key(specials[i], 7);   kb[n] = pack_key(specials[j], 7);   n++;
            ka[n] = pack_key(specials[i], 7);   kb[n] = pack_key(specials[j], 9);   n++;
            ka[n] = pack_key(specials[i], 9);   kb[n] = pack_key(specials[j], 7);   n++;
            ka[n] = pack_key(specials[i], 0);   kb[n] = pack_key(specials[j], 0xfffffffeu); n++;
        }
    }
    /* 2. NaN. Not in the total order, but it reaches the packer in practice --
     *    a TP rank whose half is all NaN produced a real wrong-token bug. */
    const float nan_v = (float)NAN;
    for (uint32_t i = 0; i < ns && n + 2 < cap; i++) {
        ka[n] = pack_key(nan_v, 3);        kb[n] = pack_key(specials[i], 3); n++;
        ka[n] = pack_key(specials[i], 3);  kb[n] = pack_key(nan_v, 3);       n++;
    }
    /* 3. TIE-HEAVY: identical scores, differing indices only. This is the case
     *    the rare-tie branch is betting against, so it must be measured with
     *    ties both rare and dominant. */
    for (uint32_t i = 0; n + 1 < cap && i < 4096; i++) {
        ka[n] = pack_key(0.5f, i); kb[n] = pack_key(0.5f, i ^ 1u); n++;
    }
    /* 4. PADDED ZEROS -- the slots a partially filled 2048-buffer carries. */
    for (uint32_t i = 0; n + 2 < cap && i < 512; i++) {
        ka[n] = 0ull;              kb[n] = pack_key(0.0f, i); n++;
        ka[n] = pack_key(0.0f, i); kb[n] = 0ull;              n++;
    }
    /* 5. bulk random, unique scores. */
    uint32_t st = 0x12345678u;
    while (n + 1 < cap) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        const float va = (float)((int32_t)st) / 1.0e6f;
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        const float vb = (float)((int32_t)st) / 1.0e6f;
        ka[n] = pack_key(va, st & 0xffffu);
        kb[n] = pack_key(vb, (st >> 16) & 0xffffu);
        n++;
    }
    return n;
}

int main(int argc, const char **argv) { @autoreleasepool {
    const char *src_path = argc > 1 ? argv[1] : "metal/cmpsel2.metal";
    const int reps = argc > 2 ? atoi(argv[2]) : 15;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(src_path)
                                              encoding:NSUTF8StringEncoding error:&err];
    if (!src) { printf("VOID: cannot read %s\n", src_path); return 2; }
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    if (!lib) { printf("VOID: %s\n", err.description.UTF8String); return 2; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    printf("device: %s\n\n", dev.name.UTF8String);

    /* ---------------- exactness ---------------- */
    const uint32_t cap = 65536;
    uint64_t *ka = malloc(cap * 8), *kb = malloc(cap * 8);
    const uint32_t n = build_keys(ka, kb, cap);
    id<MTLBuffer> ba = [dev newBufferWithBytes:ka length:cap * 8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithBytes:kb length:cap * 8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc = [dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
    memset(bc.contents, 0, 32);

    id<MTLComputePipelineState> ex =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cs2_exact"] error:&err];
    if (!ex) { printf("VOID: cs2_exact: %s\n", err.description.UTF8String); return 2; }
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ex];
        [e setBuffer:ba offset:0 atIndex:0];
        [e setBuffer:bb offset:0 atIndex:1];
        [e setBuffer:bc offset:0 atIndex:2];
        [e setBytes:&n length:4 atIndex:3];
        [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
    const uint32_t *c = (const uint32_t *)bc.contents;
    const char *rep_name[5] = { "native (reference)", "branchless hi/lo",
                                "rare-tie branch", "uint2 vector",
                                "split-in-registers (ulong storage)" };
    printf("## Exactness over %u adversarial pairs "
           "(signed zero, +/-inf, NaN, ties, padded zeros, the -1e30 floor)\n\n", n);
    for (int i = 0; i < 5; i++) {
        printf("%s   %-20s %u disagreement(s)\n",
               c[i] == 0 ? "ok  " : "FAIL", rep_name[i], c[i]);
        if (c[i] != 0) fails++;
    }
    if (fails) {
        printf("\nFAILED: a representation that is not bit-identical is not a "
               "candidate, whatever it times at.\n");
        return 1;
    }

    /* ---------------- timing ---------------- */
    struct { const char *name; const char *kern; int bitonic; } arms[] = {
        { "chain native    ", "cs2_chain_native",     0 },
        { "chain branchless", "cs2_chain_branchless", 0 },
        { "chain raretie   ", "cs2_chain_raretie",    0 },
        { "chain vector    ", "cs2_chain_vector",     0 },
        { "bitonic native    ", "cs2_bitonic_native",     1 },
        { "bitonic branchless", "cs2_bitonic_branchless", 1 },
        { "bitonic raretie   ", "cs2_bitonic_raretie",    1 },
        { "bitonic vector    ", "cs2_bitonic_vector",     1 },
        { "bitonic split-u64 ", "cs2_bitonic_split",      1 },
    };
    const int n_arms = 9;
    id<MTLComputePipelineState> pso[9];
    for (int i = 0; i < n_arms; i++) {
        pso[i] = [dev newComputePipelineStateWithFunction:
                     [lib newFunctionWithName:@(arms[i].kern)] error:&err];
        if (!pso[i]) { printf("VOID: %s: %s\n", arms[i].kern, err.description.UTF8String); return 2; }
    }

    const uint32_t chain_iters = 4096, bitonic_reps = 4, keyn = cap;
    id<MTLBuffer> sink = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    static double t[9][64];

    for (int r = 0; r < reps; r++) {
        for (int k = 0; k < n_arms; k++) {
            /* Serpentine: a fixed order gives every arm the same predecessor in
             * every repetition, which manufactured a spurious 5% gap in the
             * MOETGOCC probe before it was corrected. */
            const int fwd = (k + r) % n_arms;
            const int a = (r & 1) ? (n_arms - 1 - fwd) : fwd;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso[a]];
            [e setBuffer:ba offset:0 atIndex:0];
            [e setBuffer:sink offset:0 atIndex:1];
            if (arms[a].bitonic) {
                [e setBytes:&bitonic_reps length:4 atIndex:2];
                [e setThreadgroupMemoryLength:16384 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(256, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
            } else {
                [e setBytes:&keyn length:4 atIndex:2];
                [e setBytes:&chain_iters length:4 atIndex:3];
                [e dispatchThreads:MTLSizeMake(65536, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            t[a][r] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        }
    }

    printf("\n## Comparator cost, %d serpentine reps (median ms)\n\n", reps);
    printf("%-20s %10s %10s   %s\n", "arm", "median", "min", "vs native of its shape");
    double base_chain = 0, base_bit = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (int a = 0; a < n_arms; a++) {
            qsort(t[a], reps, sizeof(double), pass ? cmp_d : cmp_d);
            const double med = t[a][reps / 2];
            if (pass == 0) {
                if (a == 0) base_chain = med;
                if (a == 4) base_bit = med;
                continue;
            }
            const double base = arms[a].bitonic ? base_bit : base_chain;
            printf("%-20s %10.4f %10.4f   %+7.2f%%\n", arms[a].name, med, t[a][0],
                   (med - base) / base * 100.0);
        }
    }
    printf("\nM1 numbers are DATA, not a verdict: this is a 64-bit ALU question "
           "and Apple7 is not Apple8.\nExactness above is chip-independent and "
           "is the gate.\n");
    free(ka); free(kb);
    return 0;
} }
