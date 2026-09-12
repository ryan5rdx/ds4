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

/* ds4 [tok][dim] f32  ->  Core ML (1, dim, 1, n_tok) f16 */
kernel void kernel_ds4_ane_pack_f32_to_f16(
        device const float        *src  [[buffer(0)]],
        device half               *dst  [[buffer(1)]],
        constant ds4_ane_bridge_args &a  [[buffer(2)]],
        uint2                      gid  [[thread_position_in_grid]]) {
    const uint d = gid.x, t = gid.y;
    if (d >= a.dim || t >= a.n_tok) return;
    dst[(ulong)d * a.n_tok + t] = (half)src[(ulong)t * a.dim + d];
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
        uint2                      gid  [[thread_position_in_grid]]) {
    const uint d = gid.x, t = gid.y;
    if (d >= a.dim || t >= a.n_tok) return;
    const float v = (float)src[(ulong)d * a.n_tok + t];
    const ulong o = (ulong)t * a.dim + d;
    dst[o] = a.accumulate ? dst[o] + v : v;
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
