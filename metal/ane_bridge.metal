#include <metal_stdlib>
using namespace metal;

/*
 * GPU <-> ANE staging for the shared-expert sidecar.
 *
 * Two layouts have to meet here and they disagree on both order and width.
 * ds4 keeps batch activations row-major per token -- x[tok * dim + d], f32 --
 * while a Core ML conv graph wants NCHW with the contraction on C and the
 * tokens on W, i.e. (1, dim, 1, n_tok) in f16 -- y[d * n_tok + tok]. So the
 * bridge is a transpose AND a narrowing, and it is worth doing on the GPU: at
 * 4096x2048 that is 8.4M elements per layer per direction, which is ~50 us of
 * bandwidth here and tens of milliseconds on the CPU. Against a ~67 ms
 * per-layer routed-MoE window the GPU cost disappears; a CPU transpose would
 * eat the entire margin the sidecar exists to exploit.
 *
 * Both kernels are dispatched over (dim, n_tok) with no threadgroup memory:
 * the access pattern is a pure gather on one side and contiguous on the other,
 * and a tiled variant is not worth the complexity until this shows up in a
 * profile.
 */

typedef struct {
    uint dim;        /* DS4_N_EMBD */
    uint n_tok;      /* tokens in this chunk (the model's M) */
    uint accumulate; /* unpack only: 1 = += into dst, 0 = overwrite */
} ds4_ane_bridge_args;

/* TILED, and the reason is worth stating because the first version was not.
 *
 * The naive form -- one thread per element, dst[d*n_tok + t] with d on the
 * fast axis -- has consecutive threads writing 4096 bytes apart. Every 2-byte
 * store touches its own 128-byte cache line: 64x write amplification, about
 * 1 GB of line traffic per layer per direction where 16.8 MB is useful. At 42
 * layers and two directions that is ~90 GB per chunk of pure waste, which is
 * what put the GPU at 100% and 86 W during the shadow arm while the ANE sat at
 * its correct ~10% duty. It is also, almost certainly, the +8% layout step
 * that made `bridge` cost 2.3x the fence.
 *
 * So: stage a 32x32 tile in threadgroup memory, read coalesced along one axis
 * and write coalesced along the other. The +1 pad breaks the bank conflict
 * that a 32-wide tile would otherwise hit on the transposed access.
 */
#define ANE_TILE 32
#define ANE_ROWS 8          /* 32x8 threads, four rows each */

/* ds4 [tok][dim] f32  ->  Core ML (1, dim, 1, n_tok) f16 */
kernel void kernel_ds4_ane_pack_f32_to_f16(
        device const float        *src  [[buffer(0)]],
        device half               *dst  [[buffer(1)]],
        constant ds4_ane_bridge_args &a  [[buffer(2)]],
        uint2                      tgid [[threadgroup_position_in_grid]],
        uint2                      tid  [[thread_position_in_threadgroup]]) {
    threadgroup half tile[ANE_TILE][ANE_TILE + 1];

    const uint d0 = tgid.x * ANE_TILE, t0 = tgid.y * ANE_TILE;
    for (uint j = 0; j < ANE_TILE; j += ANE_ROWS) {
        const uint d = d0 + tid.x, t = t0 + tid.y + j;
        tile[tid.y + j][tid.x] = (d < a.dim && t < a.n_tok)
                ? (half)src[(ulong)t * a.dim + d] : (half)0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < ANE_TILE; j += ANE_ROWS) {
        const uint d = d0 + tid.y + j, t = t0 + tid.x;
        if (d < a.dim && t < a.n_tok)
            dst[(ulong)d * a.n_tok + t] = tile[tid.x][tid.y + j];
    }
}

/* Core ML (1, dim, 1, n_tok) f16  ->  ds4 [tok][dim] f32
 *
 * `accumulate` exists because the shared-expert output is added into the
 * residual alongside the routed-MoE output. Writing through it would silently
 * drop the routed contribution, which is the kind of error that still produces
 * fluent text. */
kernel void kernel_ds4_ane_unpack_f16_to_f32(
        device const half         *src  [[buffer(0)]],
        device float              *dst  [[buffer(1)]],
        constant ds4_ane_bridge_args &a  [[buffer(2)]],
        uint2                      tgid [[threadgroup_position_in_grid]],
        uint2                      tid  [[thread_position_in_threadgroup]]) {
    threadgroup half tile[ANE_TILE][ANE_TILE + 1];

    const uint d0 = tgid.x * ANE_TILE, t0 = tgid.y * ANE_TILE;
    for (uint j = 0; j < ANE_TILE; j += ANE_ROWS) {
        const uint d = d0 + tid.y + j, t = t0 + tid.x;
        tile[tid.y + j][tid.x] = (d < a.dim && t < a.n_tok)
                ? src[(ulong)d * a.n_tok + t] : (half)0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < ANE_TILE; j += ANE_ROWS) {
        const uint d = d0 + tid.x, t = t0 + tid.y + j;
        if (d < a.dim && t < a.n_tok) {
            const float v = (float)tile[tid.x][tid.y + j];
            const ulong o = (ulong)t * a.dim + d;
            dst[o] = a.accumulate ? dst[o] + v : v;
        }
    }
}

/* Max-abs and sum-of-squares between an ANE result and the GPU's own, so
 * `shadow` mode can report divergence without a CPU readback of 33 MB.
 * One atomic pair per threadgroup; exactness is not the point, the magnitude
 * is. Values are scaled by 1e6 and kept as integers because Metal has no
 * float atomics that are portable across the families this runs on. */
kernel void kernel_ds4_ane_compare(
        device const half         *ane  [[buffer(0)]],
        device const float        *gpu  [[buffer(1)]],
        constant ds4_ane_bridge_args &a  [[buffer(2)]],
        device atomic_uint        *out  [[buffer(3)]],   /* [0]=max_abs_ppm [1]=se [2]=sref */
        uint2                      gid  [[thread_position_in_grid]]) {
    const uint d = gid.x, t = gid.y;
    if (d >= a.dim || t >= a.n_tok) return;
    const float v = (float)ane[(ulong)d * a.n_tok + t];
    const float r = gpu[(ulong)t * a.dim + d];
    const float e = fabs(v - r);
    atomic_fetch_max_explicit(&out[0], (uint)(e * 1.0e6f), memory_order_relaxed);
    atomic_fetch_add_explicit(&out[1], (uint)(e * e * 1.0e3f), memory_order_relaxed);
    atomic_fetch_add_explicit(&out[2], (uint)(r * r * 1.0e3f), memory_order_relaxed);
}

/* The naive form, kept so the tiled one can be proved rather than asserted.
 * Identical arithmetic, one thread per element, dst on the strided axis. If
 * this is not dramatically slower at production shape then the diagnosis of
 * the +8% layout step was wrong and the tiling bought nothing. */
kernel void kernel_ds4_ane_pack_f32_to_f16_naive(
        device const float        *src  [[buffer(0)]],
        device half               *dst  [[buffer(1)]],
        constant ds4_ane_bridge_args &a  [[buffer(2)]],
        uint2                      gid  [[thread_position_in_grid]]) {
    const uint d = gid.x, t = gid.y;
    if (d >= a.dim || t >= a.n_tok) return;
    dst[(ulong)d * a.n_tok + t] = (half)src[(ulong)t * a.dim + d];
}

kernel void kernel_ds4_ane_unpack_f16_to_f32_naive(
        device const half         *src  [[buffer(0)]],
        device float              *dst  [[buffer(1)]],
        constant ds4_ane_bridge_args &a  [[buffer(2)]],
        uint2                      gid  [[thread_position_in_grid]]) {
    const uint d = gid.x, t = gid.y;
    if (d >= a.dim || t >= a.n_tok) return;
    const float v = (float)src[(ulong)d * a.n_tok + t];
    const ulong o = (ulong)t * a.dim + d;
    dst[o] = a.accumulate ? dst[o] + v : v;
}
