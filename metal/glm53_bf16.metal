// BF16 model-weight kernels used by GLM-5.3 Flash.

static inline float glm53_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

struct glm53_bf16_matmul_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

kernel void kernel_glm53_embedding_bf16(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const int                *tokens,
        device float                    *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= args.in_dim || row >= args.n_rows) return;
    const int token = tokens[row];
    out[(ulong)row * args.in_dim + d] =
        token >= 0 && (uint)token < args.out_dim
            ? glm53_bf16_to_f32(weights[(ulong)(uint)token * args.in_dim + d])
            : 0.0f;
}

static inline void glm53_mul_mv_bf16_f32_row(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *out,
        uint2                            tgpig,
        ushort                           lane,
        ushort                           sg,
        ushort                           nsg) {
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 224u < args.in_dim; k += 256u) {
        const ushort w0 = w[k];
        const ushort w1 = w[k + 32u];
        const ushort w2 = w[k + 64u];
        const ushort w3 = w[k + 96u];
        const ushort w4 = w[k + 128u];
        const ushort w5 = w[k + 160u];
        const ushort w6 = w[k + 192u];
        const ushort w7 = w[k + 224u];
        const float x0 = xr[k];
        const float x1 = xr[k + 32u];
        const float x2 = xr[k + 64u];
        const float x3 = xr[k + 96u];
        const float x4 = xr[k + 128u];
        const float x5 = xr[k + 160u];
        const float x6 = xr[k + 192u];
        const float x7 = xr[k + 224u];
        sum = fma(glm53_bf16_to_f32(w0), x0, sum);
        sum = fma(glm53_bf16_to_f32(w1), x1, sum);
        sum = fma(glm53_bf16_to_f32(w2), x2, sum);
        sum = fma(glm53_bf16_to_f32(w3), x3, sum);
        sum = fma(glm53_bf16_to_f32(w4), x4, sum);
        sum = fma(glm53_bf16_to_f32(w5), x5, sum);
        sum = fma(glm53_bf16_to_f32(w6), x6, sum);
        sum = fma(glm53_bf16_to_f32(w7), x7, sum);
    }
    for (; k < args.in_dim; k += 32u) {
        sum = fma(glm53_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * args.out_dim + out_row] = sum;
}

/* One simdgroup owns one output row. Eight independent loads expose enough
 * memory-level parallelism for decode without changing the reduction tree. */
kernel void kernel_glm53_mul_mv_bf16_f32(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    glm53_mul_mv_bf16_f32_row(args, weights, x, out,
                              tgpig, lane, sg, nsg);
}

/* B1: split-K BF16 matvec.
 *
 * The row kernel above gives ONE SIMDGROUP an entire in_dim-long row, so the
 * threadgroup count is ceil(out_dim / nsg). At the hc_mix shape -- in_dim
 * 16384, out_dim 24 -- that is 3 threadgroups on a 60-76 core GPU. The F16
 * sibling for the identical shape already takes a split-K plain-mv path.
 *
 * Here each threadgroup owns NR0 output rows and its nsg simdgroups split the
 * contraction, reducing through threadgroup memory. At out_dim 24, nsg 8:
 * 12 threadgroups with an 8-way K split, against 3.
 *
 * NOT bit-identical to the row kernel: the contraction is summed in a
 * different order (per-simdgroup partials combined in a tree, rather than one
 * simdgroup accumulating the whole row through an 8-way unrolled fma chain).
 * results.md C4 already retired byte-identity for this kernel's mv/mm fork, so
 * a band is expected -- but this arm still needs a quality gate, unlike the
 * other three in its bundle. */
kernel void kernel_glm53_mul_mv_bf16_f32_splitk(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *out,
        threadgroup float               *shmem [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort nsg  [[simdgroups_per_threadgroup]]) {
    constexpr uint NR0 = 2u;
    const uint row0  = tgpig.x * NR0;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_rows) return;
    device const float *xr = x + (ulong)token * args.in_dim;

    for (uint r = 0; r < NR0; r++) {
        const uint out_row = row0 + r;
        float sum = 0.0f;
        if (out_row < args.out_dim) {
            device const ushort *w = weights + (ulong)out_row * args.in_dim;
            /* Simdgroup sg walks lanes [32*sg, 32*sg+32) of each 32*nsg-wide
             * chunk: coalesced inside a simdgroup, and the union over
             * simdgroups covers the row exactly once.
             *
             * Keep the row kernel's EIGHT independent loads per iteration. A
             * first cut used one load per iteration and measured 0.77x -- i.e.
             * SLOWER than the 3-threadgroup kernel it replaces -- because the
             * memory-level parallelism the row kernel's comment calls out is
             * worth more here than the extra threadgroups. */
            const uint stride = 32u * (uint)nsg;
            uint k = lane + 32u * (uint)sg;
            for (; k + 7u * stride < args.in_dim; k += 8u * stride) {
                const ushort w0 = w[k];
                const ushort w1 = w[k + stride];
                const ushort w2 = w[k + 2u * stride];
                const ushort w3 = w[k + 3u * stride];
                const ushort w4 = w[k + 4u * stride];
                const ushort w5 = w[k + 5u * stride];
                const ushort w6 = w[k + 6u * stride];
                const ushort w7 = w[k + 7u * stride];
                const float x0 = xr[k];
                const float x1 = xr[k + stride];
                const float x2 = xr[k + 2u * stride];
                const float x3 = xr[k + 3u * stride];
                const float x4 = xr[k + 4u * stride];
                const float x5 = xr[k + 5u * stride];
                const float x6 = xr[k + 6u * stride];
                const float x7 = xr[k + 7u * stride];
                sum = fma(glm53_bf16_to_f32(w0), x0, sum);
                sum = fma(glm53_bf16_to_f32(w1), x1, sum);
                sum = fma(glm53_bf16_to_f32(w2), x2, sum);
                sum = fma(glm53_bf16_to_f32(w3), x3, sum);
                sum = fma(glm53_bf16_to_f32(w4), x4, sum);
                sum = fma(glm53_bf16_to_f32(w5), x5, sum);
                sum = fma(glm53_bf16_to_f32(w6), x6, sum);
                sum = fma(glm53_bf16_to_f32(w7), x7, sum);
            }
            for (; k < args.in_dim; k += stride) {
                sum = fma(glm53_bf16_to_f32(w[k]), xr[k], sum);
            }
        }
        sum = simd_sum(sum);
        if (lane == 0u) shmem[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = lane < nsg ? shmem[lane] : 0.0f;
            tot = simd_sum(tot);
            if (lane == 0u && out_row < args.out_dim) {
                out[(ulong)token * args.out_dim + out_row] = tot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* HCMIX-WIDE -- split the contraction across THREADGROUPS, not just simdgroups.
 *
 * WHAT THE PREVIOUS ATTEMPT GOT WRONG.  A queued sweep proposed turning nsg up
 * on the kernel above, on the belief that its grid is ceil(out_dim/nsg).  It is
 * not: the split-K branch dispatches (out_dim + 1)/2 threadgroups INDEPENDENT of
 * nsg -- ceil(out_dim/nsg) is the NON-split branch's formula.  Every nsg
 * launches the same 12 threadgroups at the hc_mix shape, so nsg changes only
 * threads-per-group and the within-threadgroup K partition, and a null would
 * have said nothing about the question.  That sweep was withdrawn.
 *
 * THE ACTUAL DEFECT.  hc_mix is 16384 -> 24.  The row kernel gives one
 * simdgroup a whole row: 3 threadgroups.  B1 splits K across simdgroups: 12.
 * On a 60-core part 12 threadgroups is still a fifth of the machine, and the
 * contraction is 16384 long -- there is far more K to split than there are
 * rows to spread.  B1's own 3 -> 12 win is the positive evidence for going
 * further in the same direction.
 *
 * THE FIX.  A third grid axis: threadgroup (r, token, ks) sums only the K
 * slice belonging to ks and writes a partial; a second, trivial kernel sums the
 * KSPLIT partials.  At hc_mix with KSPLIT 8 that is 96 threadgroups against 12.
 *
 * The K partition is INTERLEAVED, not blocked: lane l of simdgroup sg in slice
 * ks starts at ks*32*nsg + 32*sg + l and strides by 32*nsg*KSPLIT.  Blocking
 * would have been simpler to describe but would give the last slice a ragged
 * tail and break the coalescing the eight independent loads depend on.  The
 * union over (ks, sg, lane) covers [0, in_dim) exactly once.
 *
 * Keep the EIGHT independent loads per iteration.  The first cut of B1 used one
 * and measured 0.77x -- slower than the kernel it replaced -- because at this
 * shape memory-level parallelism is worth more than threadgroup count.  Adding
 * a grid axis does not change that; it multiplies it.
 *
 * NOT bit-identical -- to the row kernel or to B1.  The contraction is summed
 * in a different order again, so this arm needs a quality gate exactly as B1
 * did.  The reduce kernel sums slices in index order so the result is at least
 * deterministic run to run.
 */
kernel void kernel_glm53_matmul_bf16_mv_2d_partial(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *partials,
        constant uint                   &ksplit,
        threadgroup float               *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort nsg  [[simdgroups_per_threadgroup]]) {
    constexpr uint NR0 = 2u;
    const uint row0  = tgpig.x * NR0;
    const uint token = tgpig.y;
    const uint ks    = tgpig.z;
    if (row0 >= args.out_dim || token >= args.n_rows || ks >= ksplit) return;
    device const float *xr = x + (ulong)token * args.in_dim;

    for (uint r = 0; r < NR0; r++) {
        const uint out_row = row0 + r;
        float sum = 0.0f;
        if (out_row < args.out_dim) {
            device const ushort *w = weights + (ulong)out_row * args.in_dim;
            const uint stride = 32u * (uint)nsg * ksplit;
            uint k = ks * 32u * (uint)nsg + lane + 32u * (uint)sg;
            for (; k + 7u * stride < args.in_dim; k += 8u * stride) {
                const ushort w0 = w[k];
                const ushort w1 = w[k + stride];
                const ushort w2 = w[k + 2u * stride];
                const ushort w3 = w[k + 3u * stride];
                const ushort w4 = w[k + 4u * stride];
                const ushort w5 = w[k + 5u * stride];
                const ushort w6 = w[k + 6u * stride];
                const ushort w7 = w[k + 7u * stride];
                const float x0 = xr[k];
                const float x1 = xr[k + stride];
                const float x2 = xr[k + 2u * stride];
                const float x3 = xr[k + 3u * stride];
                const float x4 = xr[k + 4u * stride];
                const float x5 = xr[k + 5u * stride];
                const float x6 = xr[k + 6u * stride];
                const float x7 = xr[k + 7u * stride];
                sum = fma(glm53_bf16_to_f32(w0), x0, sum);
                sum = fma(glm53_bf16_to_f32(w1), x1, sum);
                sum = fma(glm53_bf16_to_f32(w2), x2, sum);
                sum = fma(glm53_bf16_to_f32(w3), x3, sum);
                sum = fma(glm53_bf16_to_f32(w4), x4, sum);
                sum = fma(glm53_bf16_to_f32(w5), x5, sum);
                sum = fma(glm53_bf16_to_f32(w6), x6, sum);
                sum = fma(glm53_bf16_to_f32(w7), x7, sum);
            }
            for (; k < args.in_dim; k += stride) {
                sum = fma(glm53_bf16_to_f32(w[k]), xr[k], sum);
            }
        }
        sum = simd_sum(sum);
        if (lane == 0u) shmem[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = lane < nsg ? shmem[lane] : 0.0f;
            tot = simd_sum(tot);
            if (lane == 0u && out_row < args.out_dim) {
                /* KSPLIT partials for one output land contiguously, so the
                 * reduce below reads a single cache line per output. */
                partials[(((ulong)token * args.out_dim) + out_row) * ksplit + ks] = tot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_glm53_matmul_bf16_mv_2d_reduce(
        constant glm53_bf16_matmul_args &args,
        device const float              *partials,
        device float                    *out,
        constant uint                   &ksplit,
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.out_dim * args.n_rows;
    if (gid >= total) return;
    device const float *p = partials + (ulong)gid * ksplit;
    /* Index order, so the result is deterministic across runs even though it
     * differs from both the row kernel and B1. */
    float sum = 0.0f;
    for (uint i = 0; i < ksplit; i++) sum += p[i];
    out[gid] = sum;
}

kernel void kernel_glm53_mul_mv_bf16_f32_qkv(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights_q,
        device const ushort             *weights_k,
        device const ushort             *weights_v,
        device const float              *x,
        device float                    *out_q,
        device float                    *out_k,
        device float                    *out_v,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const ushort *weights = tgpig.z == 0u ? weights_q :
                                     (tgpig.z == 1u ? weights_k : weights_v);
    device float *out = tgpig.z == 0u ? out_q :
                            (tgpig.z == 1u ? out_k : out_v);
    glm53_mul_mv_bf16_f32_row(args, weights, x, out,
                              tgpig.xy, lane, sg, nsg);
}

struct glm53_bf16_block16 {
    ushort v[16];
};

template <typename type4x4>
void glm53_dequantize_bf16(
        device const glm53_bf16_block16 *src,
        short il,
        thread type4x4 &reg) {
    (void)il;
    float4x4 values;
    for (short i = 0; i < 16; i++) {
        values[i / 4][i % 4] = glm53_bf16_to_f32(src->v[i]);
    }
    reg = (type4x4)values;
}

typedef decltype(kernel_mul_mm<
        half, half4x4, simdgroup_half8x8,
        half, half2x4, simdgroup_half8x8,
        glm53_bf16_block16, 1, glm53_dequantize_bf16,
        float, float4x4, float, float2x4>) glm53_mul_mm_bf16_t;

template [[host_name("kernel_glm53_mul_mm_bf16_f32")]]
kernel glm53_mul_mm_bf16_t kernel_mul_mm<
        half, half4x4, simdgroup_half8x8,
        half, half2x4, simdgroup_half8x8,
        glm53_bf16_block16, 1, glm53_dequantize_bf16,
        half, half4x4, float, float2x4>;
