/*
 * MOETGOCC dev-side reproduction: does reserving threadgroup memory actually
 * cost the routed-MoE decode kernels anything?
 *
 * WHY THIS EXISTS. The rig returned ~0.0% end-to-end at every reservation, up
 * to 18432 B on pair (1 threadgroup/core) and 4608 B on down. That is a
 * surprising null: these are 64-thread latency-bound decode matvecs, and the
 * directly comparable wide-Q8 decode matvec measured -13.7% at 4352 B and
 * -47.0% at 8704 B on this very box. Both cannot be a general statement about
 * Apple decode matvecs, so one of three things is true:
 *
 *   (a) these two kernels really are residency-insensitive (register-capped, or
 *       their memory-level parallelism comes from somewhere else),
 *   (b) the reservation did not bind on the rig, and the sweep measured nothing,
 *   (c) it bound and cost real kernel time, but end-to-end dilution plus the
 *       0.47% bracket band hid it.
 *
 * An end-to-end harness cannot separate these; it only sees ~20% of a step. A
 * kernel-isolated dispatch can, and does not need the rig, a GGUF, or TP -- the
 * occupancy question is about geometry and reservation, not about weight values.
 * So this dispatches the SHIPPING kernels with the production grid and sweeps
 * exactly the rig's byte counts.
 *
 * WHAT IT DELIBERATELY DOES NOT DO. It does not check numerics. Weights are
 * synthetic, so the outputs are meaningless -- the quantity under test is time
 * as a function of reserved bytes, and Q4_K dequantisation is branch-free, so
 * the values do not steer the work. The one thing that WOULD invalidate it is
 * reading out of bounds, so every buffer is sized from the same expressions
 * ds4_metal.m uses.
 *
 * MEASUREMENT. Arms are SERPENTINE -- rotated, and reversed on alternate reps
 * so no arm keeps a fixed predecessor -- never run in blocks: the isolated wide-Q8 sweep reported the same kernel 150% slower and
 * 24% faster on consecutive arm-at-a-time runs, and interleaving was what fixed
 * it. Timing is GPU-side (GPUEndTime - GPUStartTime), so host scheduling is out.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256

/* Must mirror metal/moe.metal's struct exactly, including the uint64 tail. */
typedef struct {
    uint32_t in_dim;
    uint32_t mid_dim;
    uint32_t out_dim;
    uint32_t n_total_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
    uint32_t mid_token_stride;
    uint32_t down_type;
    float    swiglu_clamp;
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_expert_base;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_expert_bytes;
    uint64_t up_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
} moe_args;

/* NEGATIVE CONTROL -- the disconnected-knob signature, in the same table.
 *
 * This kernel keeps `(void)scratch;`, exactly like the shipping pair impl and
 * the original down clones. Its sweep is therefore expected to be FLAT at every
 * reservation, and that flatness is the reference pattern: if a MoE arm ever
 * looks like this row, the knob is not attached rather than the kernel being
 * residency-insensitive. tests/probe_tg_residency_census.m proves the mechanism
 * by counting co-residency directly; this keeps the reminder next to the data.
 *
 * (It began life as a POSITIVE control and failed twice before that was
 * understood -- first bandwidth-bound at a 64 MiB table, then flat for the real
 * reason. Both failures are recorded here because each looked like a result.)
 *
 * ORIGINAL NOTE.
 *
 * A null from the MoE arms means nothing until the instrument is shown to have
 * teeth on THIS device: "reserving bytes did not slow the kernel" and
 * "reserving bytes does not reduce residency here" produce identical tables,
 * and only one of them is a result about the MoE kernels.
 *
 * So: a dependent-load pointer chase. Each iteration's address comes from the
 * previous iteration's loaded value, so a single thread can have exactly one
 * load in flight and the only source of throughput is the number of resident
 * threads. Nothing else in a GPU responds so directly to occupancy -- if
 * reservation binds at all, this kernel must fall off a cliff, and if it does
 * not, the whole sweep is measuring a no-op knob.
 *
 * Same 64 threads and same [[threadgroup(0)]] argument as the MoE kernels, so
 * it is swept by exactly the same mechanism.
 */
static const char *kOccControlSrc =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void ds4_occ_control(\n"
"        device const uint *tbl [[buffer(0)]],\n"
"        device uint *sink [[buffer(1)]],\n"
"        constant uint &mask [[buffer(2)]],\n"
"        threadgroup float *scratch [[threadgroup(0)]],\n"
"        uint tgpig [[threadgroup_position_in_grid]],\n"
"        uint tid [[thread_position_in_threadgroup]]) {\n"
"    (void)scratch;\n"
"    uint idx = ((tgpig * 64u + tid) * 2654435761u) & mask;\n"
"    uint acc = 0u;\n"
"    for (uint i = 0; i < 512u; ++i) { idx = tbl[idx] & mask; acc += idx; }\n"
"    if (acc == 0xFFFFFFFFu) sink[0] = acc;\n"
"}\n";

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(int argc, const char **argv) { @autoreleasepool {
    const char *src_path = argc > 1 ? argv[1] : "/tmp/m.metal";
    const int reps = argc > 2 ? atoi(argv[2]) : 40;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(src_path)
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) { printf("VOID: cannot read %s\n", src_path); return 2; }
    id<MTLLibrary> lib = [dev newLibraryWithSource:src
                                           options:[MTLCompileOptions new]
                                             error:&err];
    if (!lib) { printf("VOID: compile failed: %s\n", err.description.UTF8String); return 2; }

    id<MTLLibrary> ctl_lib = [dev newLibraryWithSource:@(kOccControlSrc)
                                               options:[MTLCompileOptions new]
                                                 error:&err];
    if (!ctl_lib) { printf("VOID: control compile failed: %s\n",
                           err.description.UTF8String); return 2; }

    printf("device: %s\n", dev.name.UTF8String);
    printf("max threadgroup memory: %lu B\n",
           (unsigned long)dev.maxThreadgroupMemoryLength);
    printf("reps per arm: %d (serpentine: rotated, reversed on alternate reps)\n\n", reps);

    /* GLM 5.3 Flash routed-MoE decode geometry. */
    const uint32_t in_dim = 4096, mid_dim = 2048, out_dim = 4096;
    const uint32_t n_expert_used = 8;
    /* More experts than are selected, and enough of them that the selected set
     * cannot sit in the system level cache between reps -- otherwise every arm
     * after the first measures a warm cache rather than the memory system the
     * real decode sees. */
    const uint32_t n_alloc_expert = 24;

    const uint64_t gate_row_bytes  = (uint64_t)(in_dim / QK_K) * 144;
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes  = (uint64_t)(mid_dim / QK_K) * 144;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;

    moe_args args = {
        .in_dim = in_dim, .mid_dim = mid_dim, .out_dim = out_dim,
        .n_total_expert = n_alloc_expert, .n_expert_used = n_expert_used,
        .n_tokens = 1, .mid_token_stride = n_expert_used * mid_dim,
        .down_type = 0, .swiglu_clamp = 0.0f,
        .tp_rank = 0, .tp_world = 1, .tp_expert_base = 0,
        .gate_expert_bytes = gate_expert_bytes, .gate_row_bytes = gate_row_bytes,
        .up_expert_bytes = gate_expert_bytes,   .up_row_bytes = gate_row_bytes,
        .down_expert_bytes = down_expert_bytes, .down_row_bytes = down_row_bytes,
    };

    const MTLResourceOptions ro = MTLResourceStorageModePrivate;
    id<MTLBuffer> gate = [dev newBufferWithLength:n_alloc_expert * gate_expert_bytes options:ro];
    id<MTLBuffer> up   = [dev newBufferWithLength:n_alloc_expert * gate_expert_bytes options:ro];
    id<MTLBuffer> down = [dev newBufferWithLength:n_alloc_expert * down_expert_bytes options:ro];
    id<MTLBuffer> xb   = [dev newBufferWithLength:in_dim * sizeof(float) options:ro];
    id<MTLBuffer> midb = [dev newBufferWithLength:(size_t)n_expert_used * mid_dim * sizeof(float) options:ro];
    id<MTLBuffer> outb = [dev newBufferWithLength:out_dim * sizeof(float) options:ro];
    id<MTLBuffer> wb   = [dev newBufferWithLength:n_expert_used * sizeof(float) options:ro];

    int32_t sel[8];
    /* Spread the selection across the allocation so the read set is scattered
     * the way a real router's is, not eight adjacent experts. */
    for (uint32_t i = 0; i < n_expert_used; i++) sel[i] = (int32_t)(i * 3);
    id<MTLBuffer> selb = [dev newBufferWithBytes:sel length:sizeof(sel)
                                         options:MTLResourceStorageModeShared];
    if (!gate || !up || !down || !xb || !midb || !outb || !wb || !selb) {
        printf("VOID: allocation failed (needs ~%.1f GB)\n",
               (double)(2 * n_alloc_expert * gate_expert_bytes +
                        n_alloc_expert * down_expert_bytes) / 1e9);
        return 2;
    }

    /* 1 MiB of random indices -- deliberately CACHE RESIDENT.
     *
     * The first version used 64 MiB so every chase step was a DRAM miss. That
     * made the control bandwidth-bound (8.6 GB of line fills / ~47 ms is
     * exactly the measured time), and a bandwidth-bound kernel is insensitive
     * to residency by construction -- so it was flat for the wrong reason and
     * certified nothing. At 1 MiB the chain is pure L2 latency, no bandwidth
     * wall, and time is then simply total_loads x latency / resident_threads. */
    const uint32_t ctl_n = 256u * 1024u, ctl_mask = ctl_n - 1u;
    id<MTLBuffer> ctlb = [dev newBufferWithLength:(size_t)ctl_n * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> sinkb = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (!ctlb || !sinkb) { printf("VOID: control allocation failed\n"); return 2; }
    { uint32_t *t = (uint32_t *)ctlb.contents, st = 0x12345678u;
      for (uint32_t i = 0; i < ctl_n; i++) {
          st ^= st << 13; st ^= st >> 17; st ^= st << 5; t[i] = st; } }

    id<MTLCommandQueue> q = [dev newCommandQueue];

    /* The clone control binds 128 B, NOT 0.
     *
     * Keeping the threadgroup pointer live is what makes the reservation bind,
     * and it also makes the binding mandatory: setThreadgroupMemoryLength:0 is
     * rejected, and omitting the call leaves a live argument unbound, which the
     * debug layer reports as "missing Threadgroup Memory binding at index 0 for
     * scratch[0]". An earlier version of this probe compared every point against
     * exactly that invalid dispatch. 128 B is the smallest legal allocation and
     * is the baseline every row below is measured against. */
    struct { const char *name; const char *kernel; NSUInteger tg; NSUInteger gx, gy; } arms[] = {
        { "pair ship    ", "kernel_glm_q4_K_pair_swiglu4_f32_spec",             0, (mid_dim + 7) / 8, n_expert_used },
        { "pair min128  ", "kernel_glm_q4_K_pair_swiglu4_f32_spec_tgprobe",   128, (mid_dim + 7) / 8, n_expert_used },
        { "pair   4608 B", "kernel_glm_q4_K_pair_swiglu4_f32_spec_tgprobe",  4608, (mid_dim + 7) / 8, n_expert_used },
        { "pair   9216 B", "kernel_glm_q4_K_pair_swiglu4_f32_spec_tgprobe",  9216, (mid_dim + 7) / 8, n_expert_used },
        { "pair  18432 B", "kernel_glm_q4_K_pair_swiglu4_f32_spec_tgprobe", 18432, (mid_dim + 7) / 8, n_expert_used },
        { "down ship    ", "kernel_glm_q4_K_down_simd_f32_spec",                0, (out_dim + 3) / 4, 1 },
        { "down min128  ", "kernel_glm_q4_K_down_simd_f32_spec_tgprobe",      128, (out_dim + 3) / 4, 1 },
        { "down   2304 B", "kernel_glm_q4_K_down_simd_f32_spec_tgprobe",     2304, (out_dim + 3) / 4, 1 },
        { "down   4608 B", "kernel_glm_q4_K_down_simd_f32_spec_tgprobe",     4608, (out_dim + 3) / 4, 1 },
        /* Negative control: `(void)scratch;`, the disconnected-knob signature. */
        { "ctl0 dead128 ", "ds4_occ_control",   128, 2048, 1 },
        { "ctl0 dead18k ", "ds4_occ_control", 18432, 2048, 1 },
    };
    const int n_arms = (int)(sizeof(arms) / sizeof(arms[0]));

    id<MTLComputePipelineState> pipes[24];
    for (int i = 0; i < n_arms; i++) {
        const BOOL is_ctl = arms[i].name[0] == 'c';
        id<MTLFunction> f = [(is_ctl ? ctl_lib : lib) newFunctionWithName:@(arms[i].kernel)];
        if (!f) { printf("VOID: no kernel %s\n", arms[i].kernel); return 2; }
        pipes[i] = [dev newComputePipelineStateWithFunction:f error:&err];
        if (!pipes[i]) { printf("VOID: pipeline %s: %s\n", arms[i].kernel,
                                err.description.UTF8String); return 2; }
    }

    double *samples = calloc((size_t)n_arms * reps, sizeof(double));

    for (int r = 0; r < reps; r++) {
        for (int k = 0; k < n_arms; k++) {
            /* SERPENTINE, not cyclic. Rotating the start puts every arm in
             * every position, but `(k + r) % n_arms` gives every arm the SAME
             * predecessor in every repetition -- so whatever that neighbour
             * leaves in cache and in the GPU's clock state is a fixed per-arm
             * bias. Reversing on alternate reps gives each arm both
             * neighbours. This is what dissolved an apparent 5% shipping-vs-
             * clone gap that turned out to be pure ordering bias. */
            const int fwd = (k + r) % n_arms;
            const int a = (r & 1) ? (n_arms - 1 - fwd) : fwd;
            const int is_down = arms[a].name[0] == 'd';
            const int is_ctl  = arms[a].name[0] == 'c';

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipes[a]];
            if (is_ctl) {
                [enc setBuffer:ctlb offset:0 atIndex:0];
                [enc setBuffer:sinkb offset:0 atIndex:1];
                [enc setBytes:&ctl_mask length:sizeof(ctl_mask) atIndex:2];
            } else
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            if (is_ctl) { /* bound above */ } else if (is_down) {
                [enc setBuffer:down offset:0 atIndex:1];
                [enc setBuffer:selb offset:0 atIndex:2];
                [enc setBuffer:midb offset:0 atIndex:3];
                [enc setBuffer:outb offset:0 atIndex:4];
            } else {
                [enc setBuffer:gate offset:0 atIndex:1];
                [enc setBuffer:up   offset:0 atIndex:2];
                [enc setBuffer:xb   offset:0 atIndex:3];
                [enc setBuffer:selb offset:0 atIndex:4];
                [enc setBuffer:wb   offset:0 atIndex:5];
                [enc setBuffer:midb offset:0 atIndex:6];
            }
            if (arms[a].tg != 0) {
                [enc setThreadgroupMemoryLength:arms[a].tg atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(arms[a].gx, arms[a].gy, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            samples[(size_t)a * reps + r] =
                (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        }
    }

    printf("%-14s %8s %8s %8s   %s\n", "arm", "median", "min", "p90",
           "vs clone min128 of its group");
    double base_pair = 0, base_down = 0, base_ctl = 0;
    /* Two passes: the baselines are the min128 clones, which are not the first
     * row of their group, so they must be resolved before anything is scored. */
    for (int pass = 0; pass < 2; pass++) {
        for (int a = 0; a < n_arms; a++) {
            double *v = samples + (size_t)a * reps;
            if (pass == 0) qsort(v, reps, sizeof(double), cmp_d);
            const double med = v[reps / 2];
            const char c0 = arms[a].name[0];
            const int is_min = strstr(arms[a].name, "min128") != NULL ||
                               strstr(arms[a].name, "dead128") != NULL;
            if (pass == 0) {
                if (is_min && c0 == 'p') base_pair = med;
                if (is_min && c0 == 'd') base_down = med;
                if (is_min && c0 == 'c') base_ctl  = med;
                continue;
            }
            const double base = c0 == 'd' ? base_down : c0 == 'c' ? base_ctl : base_pair;
            printf("%-14s %8.3f %8.3f %8.3f   %+7.2f%%\n",
                   arms[a].name, med, v[0], v[(reps * 9) / 10],
                   base > 0 ? (med - base) / base * 100.0 : 0.0);
        }
    }

    /* No "threadgroups per core" column. maxThreadgroupMemoryLength is the
     * per-THREADGROUP maximum, not the per-core pool, so dividing by it gives a
     * number that is simply wrong: tests/probe_tg_residency_census.m counted 72
     * simultaneous threadgroups at 18432 B on this part, where that arithmetic
     * predicts one per core. Footprint in bytes is the honest label; measured
     * co-residency belongs to the census probe. */
    free(samples);
    return 0;
} }
