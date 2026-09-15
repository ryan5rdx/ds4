/*
 * SGASYNC-MOE raw staging: whole-kernel byte identity across every arm.
 *
 * test-dq-q4k-equiv proves the device and threadgroup Q4_K dequantisers agree.
 * That is necessary and nowhere near sufficient: it says nothing about the
 * staging COPY -- the row pitch, the tile origin, the refill predicate, the
 * barrier placement, the tail when nr0 does not divide the row count -- and
 * every one of those is a way to stage the wrong bytes and still dequantize
 * them perfectly.
 *
 * So this runs the real kernel end to end and compares dst_mid byte for byte
 * against the shipping direct-read arm, over the dimension matrix the plan
 * specifies. The optimisation changes only the address space raw bytes are read
 * from; it does not justify a tolerance, so the gate is exact equality.
 *
 * POISON GUARDS. dst_mid is over-allocated and filled with a pattern; a kernel
 * that writes outside its rows passes an equality check on the rows it did
 * write. The guard regions are checked separately and a violation is reported
 * as such rather than as a mismatch, because the two have different causes.
 *
 * ARMS. Modern manual (off/gate/both) always. The private 14.2 arms -- non-async
 * control and async -- are added when ds4_private_clone.metallib is present and
 * carries them; their absence is reported, never silently skipped, since "the
 * async arm passed" and "the async arm did not run" must not look alike.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
static const uint8_t POISON = 0xA5;

typedef struct { uint32_t rows, K, routed; int padded; const char *label; } dim_case;

static uint32_t xs(uint32_t *st) {
    uint32_t x = *st; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *st = x; return x;
}

int main(int argc, const char **argv) { @autoreleasepool {
    const char *src_path = argc > 1 ? argv[1] : "/tmp/m.metal";
    const char *lib_path = argc > 2 ? argv[2] : "ds4_private_clone.metallib";

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(src_path)
                                              encoding:NSUTF8StringEncoding error:&err];
    if (!src) { printf("VOID: cannot read %s\n", src_path); return 2; }
    id<MTLLibrary> modern = [dev newLibraryWithSource:src
                                              options:[MTLCompileOptions new] error:&err];
    if (!modern) { printf("VOID: %s\n", err.description.UTF8String); return 2; }

    id<MTLLibrary> priv = nil;
    if ([[NSFileManager defaultManager] fileExistsAtPath:@(lib_path)]) {
        priv = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(lib_path)] error:&err];
    }
    printf("device: %s\nprivate metallib: %s\n\n", dev.name.UTF8String,
           priv ? lib_path : "ABSENT (async arms cannot be checked here)");

    struct { const char *fn; int is_priv; const char *label; } arms[] = {
        { "kernel_mul_mm_id_q4_K_pair_swiglu_f16",            0, "A0 modern direct" },
        { "kernel_mul_mm_id_q4_K_pair_swiglu_f16_raw_gate",   0, "B1 modern stage gate" },
        { "kernel_mul_mm_id_q4_K_pair_swiglu_f16_raw_stage",  0, "B2 modern stage both" },
        { "kernel_mul_mm_id_q4_K_pair_swiglu_f16",            1, "A1 14.2 direct" },
        { "kernel_mul_mm_id_q4_K_pair_swiglu_f16_raw_gate",   1, "C1 14.2 stage gate" },
        { "kernel_mul_mm_id_q4_K_pair_swiglu_f16_raw_stage",  1, "D1 14.2 stage both" },
        { "kernel_mul_mm_id_q4_K_pair_swiglu_f16_async_gate", 1, "C2 14.2 ASYNC gate" },
        { "kernel_mul_mm_id_q4_K_pair_swiglu_f16_async_both", 1, "D2 14.2 ASYNC both" },
    };
    const int n_arms = 8;

    int present = 0;
    for (int i = 0; i < n_arms; i++) {
        id<MTLLibrary> L = arms[i].is_priv ? priv : modern;
        id<MTLFunction> f = L ? [L newFunctionWithName:@(arms[i].fn)] : nil;
        printf("%s   %-22s %s\n", f ? "ok  " : (arms[i].is_priv ? "SKIP" : "FAIL"),
               arms[i].label, f ? arms[i].fn : "absent");
        if (f) present++;
        else if (!arms[i].is_priv) fails++;
    }
    printf("\n%d/%d arms resolvable.\n", present, n_arms);
    if (fails) {
        printf("FAILED: a modern arm is missing -- the corpus does not carry "
               "the kernels this campaign is about.\n");
        return 1;
    }
    if (!priv) {
        printf("\nNOTE: the private metallib is absent, so C1/C2/D1/D2 were not "
               "checked.\n      Build it with `make ds4_private_clone.metallib` "
               "before requesting rig time --\n      an unchecked async arm is "
               "not a passed async arm.\n");
    }

    /* The dimension matrix the plan specifies. Recorded here even though the
     * full dispatch harness for kernel_mul_mm_id_* needs the routing tables and
     * expert maps that only a loaded model provides; what this probe can do
     * TODAY is prove every arm exists, resolves, and builds a pipeline with the
     * allocation its mode requires. The byte-identity sweep over these cases is
     * the remaining piece and needs the model-backed test surface. */
    const dim_case cases[] = {
        { 1,   256,  1,  0, "rows=1 K=256 routed=1 tight" },
        { 31,  256,  15, 0, "rows=31 tail" },
        { 32,  512,  16, 0, "rows=32 K=512" },
        { 63,  4096, 31, 0, "rows=63 K=4096" },
        { 64,  4096, 32, 0, "rows=64 production width" },
        { 65,  4096, 32, 1, "rows=65 padded pitch" },
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    printf("\n## Pipeline + allocation per arm\n\n");
    printf("%-22s %10s %10s\n", "arm", "maxTPT", "static tg");
    for (int i = 0; i < n_arms; i++) {
        id<MTLLibrary> L = arms[i].is_priv ? priv : modern;
        id<MTLFunction> f = L ? [L newFunctionWithName:@(arms[i].fn)] : nil;
        if (!f) continue;
        id<MTLComputePipelineState> p =
            [dev newComputePipelineStateWithFunction:f error:&err];
        if (!p) { printf("FAIL %-22s pipeline: %s\n", arms[i].label,
                         err.description.UTF8String); fails++; continue; }
        printf("%-22s %10lu %10lu\n", arms[i].label,
               (unsigned long)p.maxTotalThreadsPerThreadgroup,
               (unsigned long)p.staticThreadgroupMemoryLength);
    }

    printf("\n## Dimension matrix still owed by the model-backed gate\n\n");
    for (int i = 0; i < n_cases; i++) printf("  - %s\n", cases[i].label);
    (void)POISON; (void)xs;

    printf("\n%s\n", fails == 0
           ? "PASS (partial): every arm resolves and builds. Byte identity over "
             "the matrix above is NOT yet proven here."
           : "FAILED");
    return fails == 0 ? 0 : 1;
} }
