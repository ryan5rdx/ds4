/*
 * CMPSEL2-U32KEY -- is the 64-bit packed key the wrong representation?
 *
 * WHAT THE FIRST AUDIT DID AND DID NOT SETTLE. CMPSEL-AUDIT compared a ternary,
 * a max() and a guarded store on ulong and found them within 0.2%, concluding
 * that MSL already lowers all three to compare-select. That is true and it
 * closes the question it asked. It then recorded, and explicitly set aside,
 * the datum that matters here: **ulong compare-select costs ~2.5x uint or
 * float** (12.3 ms against 4.8/5.1). It called that "not a CMPSEL question --
 * it is the width of the key". The width of the key is a question we can act
 * on, and nobody asked it. It was also M1/Apple7 only, and measured no
 * production kernel.
 *
 * Our packed keys are lexicographic by construction:
 *
 *     key = ((ulong)ordered_score << 32) | (0xffffffff - idx)
 *
 * so hi and lo are both uint32 and `a > b` on the ulong is exactly
 * `(hi > hi2) || (hi == hi2 && lo > lo2)`. Nothing about the ordering requires
 * 64-bit arithmetic -- only the STORAGE is 64-bit, and the comparison could be
 * done in 32-bit lanes. This file asks whether that is cheaper on hardware
 * whose 64-bit ALU path is evidently narrower.
 *
 * FOUR REPRESENTATIONS, one ordering:
 *   A native   -- `a > b` on ulong, what ships today
 *   B branchless -- (hi>hi2) | ((hi==hi2) & (lo>lo2)), no control flow
 *   C raretie  -- hi != hi2 ? hi > hi2 : lo > lo2, betting ties are rare
 *   D vector   -- uint2 storage, compared with vector ops
 *
 * EXACTNESS IS THE GATE, NOT THE TIMING. These keys are compared across ranks
 * and across kernels; a representation that disagrees on one tie, one signed
 * zero or one infinity emits a token no single-node run produces. So the
 * exactness kernel runs every representation over the same pairs and counts
 * disagreements against the native result. A faster comparator that is not
 * bit-identical is not a candidate.
 *
 * TWO SHAPES, because the comparator sits in two different kernels:
 *   chain    -- a dependent running max. The top-1 reducer's shape; comparator
 *               LATENCY is on the critical path.
 *   pairwise -- independent compare-and-swap. The bitonic sort's shape;
 *               comparator THROUGHPUT is what matters and the compares can
 *               overlap.
 * A representation can win one and lose the other, which a single number would
 * hide.
 */
#include <metal_stdlib>
using namespace metal;

/* ---- the four comparators, all returning "a orders before b" (a > b) ---- */

static inline bool cs2_gt_native(ulong a, ulong b) {
    return a > b;
}
static inline bool cs2_gt_branchless(uint2 a, uint2 b) {
    /* a.y is the high half. Written as bitwise | and & rather than || and &&
     * so no short-circuit branch is expressible. */
    return (uint)((a.y > b.y) | ((a.y == b.y) & (a.x > b.x))) != 0u;
}
static inline bool cs2_gt_raretie(uint2 a, uint2 b) {
    return a.y != b.y ? (a.y > b.y) : (a.x > b.x);
}
static inline bool cs2_gt_vector(uint2 a, uint2 b) {
    const bool2 gt = a > b, eq = a == b;
    return gt.y || (eq.y && gt.x);
}

static inline bool cs2_gt_split(ulong a, ulong b) {
    const uint ah = (uint)(a >> 32), bh = (uint)(b >> 32);
    const uint al = (uint)a,         bl = (uint)b;
    return (uint)((ah > bh) | ((ah == bh) & (al > bl))) != 0u;
}
/* ---- exactness: every representation against the native one -------------
 *
 * counts[0..3] = disagreements for native/branchless/raretie/vector. Native is
 * the reference, so counts[0] must be 0 by construction -- it is kept as a
 * self-check that the harness is comparing what it thinks it is. */
kernel void cs2_exact(
        device const ulong *a_keys [[buffer(0)]],
        device const ulong *b_keys [[buffer(1)]],
        device atomic_uint *counts [[buffer(2)]],
        constant uint      &n      [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    const ulong a = a_keys[gid], b = b_keys[gid];
    const uint2 av = uint2((uint)(a & 0xffffffffu), (uint)(a >> 32));
    const uint2 bv = uint2((uint)(b & 0xffffffffu), (uint)(b >> 32));

    const bool ref = cs2_gt_native(a, b);
    if (cs2_gt_native(a, b)     != ref) atomic_fetch_add_explicit(&counts[0], 1u, memory_order_relaxed);
    if (cs2_gt_branchless(av, bv) != ref) atomic_fetch_add_explicit(&counts[1], 1u, memory_order_relaxed);
    if (cs2_gt_raretie(av, bv)    != ref) atomic_fetch_add_explicit(&counts[2], 1u, memory_order_relaxed);
    if (cs2_gt_vector(av, bv)     != ref) atomic_fetch_add_explicit(&counts[3], 1u, memory_order_relaxed);
    if (cs2_gt_split(a, b)        != ref) atomic_fetch_add_explicit(&counts[4], 1u, memory_order_relaxed);
}

/* ---- shape 1: dependent running max (top-1 reducer) ---------------------
 *
 * Each iteration's compare depends on the previous accumulator, so this
 * measures comparator latency. The salt rotation keeps the loop from being
 * hoisted and keeps every iteration reading a different key. */
#define CS2_CHAIN(NAME, LOAD, GT, ACCT, INIT)                                  \
kernel void NAME(                                                              \
        device const ulong *keys [[buffer(0)]],                                \
        device uint        *sink [[buffer(1)]],                                \
        constant uint      &n    [[buffer(2)]],                                \
        constant uint      &iters[[buffer(3)]],                                \
        uint gid [[thread_position_in_grid]]) {                                \
    ACCT acc = INIT;                                                           \
    uint idx = gid;                                                            \
    for (uint it = 0; it < iters; ++it) {                                      \
        idx = (idx * 1664525u + 1013904223u) & (n - 1u);                       \
        ACCT k = LOAD(keys[idx]);                                              \
        if (GT(k, acc)) acc = k;                                               \
    }                                                                          \
    if (as_type<uint2>((ulong)0).x == 0xffffffffu) sink[0] = (uint)acc.x;      \
}

static inline ulong cs2_load_u64(ulong k) { return k; }
static inline uint2 cs2_load_u32(ulong k) {
    return uint2((uint)(k & 0xffffffffu), (uint)(k >> 32));
}

/* The ulong variant cannot use the uint2 macro body verbatim (acc.x), so it is
 * written out rather than made to fit. */
kernel void cs2_chain_native(
        device const ulong *keys [[buffer(0)]],
        device uint        *sink [[buffer(1)]],
        constant uint      &n    [[buffer(2)]],
        constant uint      &iters[[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    ulong acc = 0ul;
    uint idx = gid;
    for (uint it = 0; it < iters; ++it) {
        idx = (idx * 1664525u + 1013904223u) & (n - 1u);
        const ulong k = keys[idx];
        if (cs2_gt_native(k, acc)) acc = k;
    }
    if (acc == 0xfffffffffffffffful) sink[0] = 1u;
}

#define CS2_CHAIN_U32(NAME, GT)                                                \
kernel void NAME(                                                              \
        device const ulong *keys [[buffer(0)]],                                \
        device uint        *sink [[buffer(1)]],                                \
        constant uint      &n    [[buffer(2)]],                                \
        constant uint      &iters[[buffer(3)]],                                \
        uint gid [[thread_position_in_grid]]) {                                \
    uint2 acc = uint2(0u, 0u);                                                 \
    uint idx = gid;                                                            \
    for (uint it = 0; it < iters; ++it) {                                      \
        idx = (idx * 1664525u + 1013904223u) & (n - 1u);                       \
        const uint2 k = cs2_load_u32(keys[idx]);                               \
        if (GT(k, acc)) acc = k;                                               \
    }                                                                          \
    if (acc.x == 0xffffffffu && acc.y == 0xffffffffu) sink[0] = 1u;            \
}
CS2_CHAIN_U32(cs2_chain_branchless, cs2_gt_branchless)
CS2_CHAIN_U32(cs2_chain_raretie,    cs2_gt_raretie)
CS2_CHAIN_U32(cs2_chain_vector,     cs2_gt_vector)

/* ---- shape 2: independent compare-and-swap (bitonic) --------------------
 *
 * A threadgroup bitonic pass over 2048 slots, the exact geometry
 * ds4_topk_bitonic_desc_2048 uses, so the barrier and threadgroup-traffic
 * structure is production's rather than a microbenchmark's. This is the shape
 * where the comparator has to beat memory and barriers to show up at all --
 * which is the reason to measure it rather than extrapolate from the chain. */
#define CS2_BITONIC(NAME, TYPE, GT, PACK, UNPACK, TGBYTES)                     \
kernel void NAME(                                                              \
        device const ulong *src  [[buffer(0)]],                                \
        device uint        *sink [[buffer(1)]],                                \
        constant uint      &reps [[buffer(2)]],                                \
        threadgroup TYPE   *buf  [[threadgroup(0)]],                           \
        uint  tgid [[threadgroup_position_in_grid]],                           \
        ushort tid [[thread_position_in_threadgroup]]) {                       \
    for (uint r = 0; r < reps; ++r) {                                          \
        for (uint i = tid; i < 2048u; i += 512u) {                             \
            buf[i] = PACK(src[(tgid * 2048u + i + r) & 65535u]);               \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        for (uint k = 2; k <= 2048u; k <<= 1) {                                \
            for (uint j = k >> 1; j > 0; j >>= 1) {                            \
                for (uint i = tid; i < 2048u; i += 512u) {                     \
                    const uint ixj = i ^ j;                                    \
                    if (ixj > i) {                                             \
                        const bool desc = (i & k) == 0u;                       \
                        const TYPE a = buf[i], b = buf[ixj];                   \
                        if (desc ? GT(b, a) : GT(a, b)) {                      \
                            buf[i] = b; buf[ixj] = a;                          \
                        }                                                      \
                    }                                                          \
                }                                                              \
                threadgroup_barrier(mem_flags::mem_threadgroup);               \
            }                                                                  \
        }                                                                      \
        if (tid == 0 && UNPACK(buf[0]) == 0xfffffffffffffffful) sink[0] = 1u;  \
    }                                                                          \
}

static inline ulong cs2_pack_id(ulong k)   { return k; }
static inline ulong cs2_unpack_id(ulong k) { return k; }
static inline uint2 cs2_pack_v(ulong k)    { return cs2_load_u32(k); }
static inline ulong cs2_unpack_v(uint2 k)  { return ((ulong)k.y << 32) | (ulong)k.x; }

/* THE ARM THAT DECIDES HOW INVASIVE THE PRODUCTION CHANGE IS.
 *
 * Storage stays ulong -- the threadgroup buffer, the packing, the compaction
 * and every inter-kernel interface are untouched -- and ONLY the comparison is
 * done in 32-bit halves, split in registers at the point of use. If this
 * captures the uint2 arms' win then the production edit is three lines inside
 * one comparator; if it does not, the win is in the storage and the change
 * reaches ds4_topk_stream_core's buffer type, the threshold word and the merge
 * kernel's interface, which is a different proposition entirely. */
CS2_BITONIC(cs2_bitonic_split,      ulong, cs2_gt_split,      cs2_pack_id, cs2_unpack_id, 16384)
CS2_BITONIC(cs2_bitonic_native,     ulong, cs2_gt_native,     cs2_pack_id, cs2_unpack_id, 16384)
CS2_BITONIC(cs2_bitonic_branchless, uint2, cs2_gt_branchless, cs2_pack_v,  cs2_unpack_v,  16384)
CS2_BITONIC(cs2_bitonic_raretie,    uint2, cs2_gt_raretie,    cs2_pack_v,  cs2_unpack_v,  16384)
CS2_BITONIC(cs2_bitonic_vector,     uint2, cs2_gt_vector,     cs2_pack_v,  cs2_unpack_v,  16384)
