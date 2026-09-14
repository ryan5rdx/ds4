#include <metal_stdlib>
using namespace metal;

/* Exact GPU top-1 over a logit row, as a packed total order.
 *
 * This exists to replace a per-rank half-logit GPU->CPU read plus a
 * vocabulary-half TP exchange with an 8-byte key: under vocabulary TP each rank
 * reduces its half to one key using GLOBAL token ids, and rank 0 takes the
 * unsigned maximum of two keys. It is only correct where the sampler contract
 * says a raw argmax is sufficient (see ds4_sampler_can_use_raw_argmax()).
 *
 * TWO IMPLEMENTATIONS, ONE COMPARATOR. The conventional two-pass reducer is the
 * shipping candidate and works on every family; the UInt64 atomic is an Apple8
 * experiment. They must agree bit for bit, so both call ds4_top1_pack_key() and
 * both use the same local reducer -- otherwise an A/B measures two algorithms
 * rather than two publication mechanisms.
 */

/* ds4_top1_pack_key() is defined in ds4_top1_key.h, shared verbatim with the C
 * host -- see that header for why a second copy would be a silent correctness
 * hazard rather than duplication. It is CONCATENATED ahead of this file by
 * ds4_gpu_full_source(); a quoted #include does not resolve in a runtime source
 * string, so the loader list is the mechanism. */

struct ds4_metal_args_top1 {
    uint32_t n_cols;        /* columns in this rank's slice                 */
    uint32_t row_stride;    /* elements between rows of `logits`            */
    uint32_t global_base;   /* token id of column 0 -- the rank's offset    */
    uint32_t n_rows;
    uint32_t n_groups;      /* threadgroups per row (pass 1 / partials)     */
    uint32_t n_shards;      /* atomic variant only: winner slots per row    */
};

#define DS4_TOP1_NT 256u

/* The shared local reducer. Both mechanisms use it so an A/B isolates only how
 * the per-group result is published. */
static inline ulong ds4_top1_reduce_tg(threadgroup ulong *scratch,
                                       ulong mine,
                                       uint tid) {
    scratch[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = DS4_TOP1_NT / 2u; s != 0u; s >>= 1) {
        if (tid < s) {
            const ulong rhs = scratch[tid + s];
            if (rhs > scratch[tid]) scratch[tid] = rhs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
}

static inline ulong ds4_top1_scan_slice(device const float *logits,
                                        constant ds4_metal_args_top1 &args,
                                        uint row, uint gx, uint tid) {
    const ulong row_off = (ulong)row * (ulong)args.row_stride;
    const uint  stride  = DS4_TOP1_NT * args.n_groups;
    /* Seed at the SAMPLER FLOOR, not 0. Reducing from 0 makes the winner the
     * largest actual value even when every value is at or below DS4_NEG_INF,
     * where sample_argmax returns its seed index instead -- a masked vocabulary
     * is exactly that shape. See ds4_top1_seed_key(). */
    ulong best = ds4_top1_seed_key(args.global_base);
    for (uint col = gx * DS4_TOP1_NT + tid; col < args.n_cols; col += stride) {
        const ulong k = ds4_top1_pack_key(logits[row_off + col],
                                          args.global_base + col);
        if (k > best) best = k;
    }
    return best;
}

/* ---- conventional two-pass: the shipping candidate ---------------------- */

kernel void kernel_ds4_top1_scan(
        constant ds4_metal_args_top1 & args,
        device const float * logits   [[buffer(1)]],
        device       ulong * partials [[buffer(2)]],
        uint  tid [[thread_index_in_threadgroup]],
        uint3 tg  [[threadgroup_position_in_grid]]) {
    threadgroup ulong scratch[DS4_TOP1_NT];
    const ulong mine = ds4_top1_scan_slice(logits, args, tg.y, tg.x, tid);
    const ulong best = ds4_top1_reduce_tg(scratch, mine, tid);
    if (tid == 0u) partials[(ulong)tg.y * args.n_groups + tg.x] = best;
}

kernel void kernel_ds4_top1_merge(
        constant ds4_metal_args_top1 & args,
        device const ulong * partials [[buffer(1)]],
        device       ulong * out      [[buffer(2)]],
        uint  tid [[thread_index_in_threadgroup]],
        uint3 tg  [[threadgroup_position_in_grid]]) {
    threadgroup ulong scratch[DS4_TOP1_NT];
    const ulong base = (ulong)tg.y * args.n_groups;
    ulong mine = ds4_top1_seed_key(args.global_base);
    for (uint i = tid; i < args.n_groups; i += DS4_TOP1_NT) {
        const ulong k = partials[base + i];
        if (k > mine) mine = k;
    }
    const ulong best = ds4_top1_reduce_tg(scratch, mine, tid);
    if (tid == 0u) out[tg.y] = best;
}

/* ---- Apple8 native UInt64 atomic max ------------------------------------
 *
 * relaxed ordering is sufficient: these are competing maxima on one address and
 * the visibility edge is command-buffer completion, not anything in-kernel.
 *
 * Shards exist because M2 Ultra spans two dies and every group hammering one
 * address may serialise across the interconnect. Each group publishes into
 * `group % n_shards`; the merge kernel below folds the shards. Sweeping shards
 * is the point -- "native atomics plus a few shards" may beat both extremes.
 */
/* MEASURED ON THE DEV BOX 2026-09-13: THIS GUARD DOES NOT DISCRIMINATE.
 *
 * `__HAVE_ATOMIC_ULONG_MIN_MAX__` is defined by the TOOLCHAIN, not by the
 * device. On M1 Max / Apple7 under Xcode 26.6 it is defined, both kernels below
 * compile into the metallib, and it is PIPELINE CREATION that rejects the
 * atomic one:
 *
 *     Metal kernel_ds4_top1_atomic pipeline failed:
 *     Unsupported float atomic operation for given target.
 *
 * (Note the message says "float" for a ulong atomic; do not match on its text.)
 *
 * The campaign brief says not to gate only on supportsFamily. It is stronger
 * than that: the macro does not gate either, and neither does the function
 * being present in the library. Only newComputePipelineStateWithFunction tells
 * the truth, so that is layer 2 of the probe and it is the load-bearing one.
 * The guard is kept for older SDKs and as documentation, not as protection.
 */
#if defined(__HAVE_ATOMIC_ULONG_MIN_MAX__)
#define DS4_TOP1_HAVE_U64_ATOMIC 1

kernel void kernel_ds4_top1_atomic(
        constant ds4_metal_args_top1 & args,
        device const float         * logits  [[buffer(1)]],
        device       atomic_ulong  * winners [[buffer(2)]],
        uint  tid [[thread_index_in_threadgroup]],
        uint3 tg  [[threadgroup_position_in_grid]]) {
    threadgroup ulong scratch[DS4_TOP1_NT];
    const ulong mine = ds4_top1_scan_slice(logits, args, tg.y, tg.x, tid);
    const ulong best = ds4_top1_reduce_tg(scratch, mine, tid);
    if (tid == 0u) {
        const uint shard = args.n_shards > 1u ? (tg.x % args.n_shards) : 0u;
        atomic_max_explicit(winners + (ulong)tg.y * args.n_shards + shard,
                            best, memory_order_relaxed);
    }
}


#endif   /* __HAVE_ATOMIC_ULONG_MIN_MAX__ */

/* Folds the per-row shards. Trivial, but it is part of the atomic path's cost
 * and the gate requires reset and merge to be counted against it. */
kernel void kernel_ds4_top1_shard_merge(
        constant ds4_metal_args_top1 & args,
        device const ulong * winners [[buffer(1)]],
        device       ulong * out     [[buffer(2)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tg [[threadgroup_position_in_grid]]) {
    if (tid != 0u) return;
    const ulong base = (ulong)tg.y * args.n_shards;
    ulong best = ds4_top1_seed_key(args.global_base);
    for (uint i = 0u; i < args.n_shards; ++i) {
        const ulong k = winners[base + i];
        if (k > best) best = k;
    }
    out[tg.y] = best;
}

/* Reset as a kernel, so the sweep can price it against a CPU write and a blit
 * without any of the three sitting outside the timed region.
 *
 * The slots reset to the SEED KEY, not to zero. atomic_max against a zeroed
 * slot lets the largest actual value win even when every value is at or below
 * the sampler floor, where the CPU returns its seed index -- so a zeroed slot
 * is not a neutral starting point, it is a different algorithm. This is also
 * why reset mode 2 ("slots already clear, skip the reset") is no longer
 * available: "clear" now means a specific nonzero value that depends on
 * global_base, and a recycled buffer does not have it. */
kernel void kernel_ds4_top1_reset(
        constant ds4_metal_args_top1 & args,
        device ulong * slots [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const uint n = args.n_rows * (args.n_shards > 0u ? args.n_shards : 1u);
    if (gid < n) slots[gid] = ds4_top1_seed_key(args.global_base);
}
