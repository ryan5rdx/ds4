// Kimi Delta Attention kernels, adapted from the kimi-k3 branch.

struct glm53_kda_args {
    uint n_heads;        // heads this rank computes
    uint n_rows;
    float lower_bound;
    float norm_eps;
    /* State addressing is ABSOLUTE, activations and weights are lane-relative.
     *
     * A rank that owns a head subrange still stores that range at its true head
     * index in a full-width state buffer, so the conv ring and the recurrent
     * state have one layout regardless of how the heads happen to be divided.
     * Without this, a phase that ran 64 heads and a phase that ran 32 would
     * partition the same buffer differently and silently corrupt it -- which is
     * exactly what blocked replicating prefill while splitting decode.
     *
     * Activations (q_in/k_in/v_in/raw_gate/raw_beta/output_gate/out) stay packed
     * at lane width, and the model-side conv/dt_bias/a_log pointers are already
     * offset to the lane by the host, so both keep using n_heads and head. */
    uint n_heads_total;  // heads in the whole layer, for state strides
    uint head_first;     // absolute index of this rank's first head
    /* KDA-PREPARE-PAR: tokens each prepare threadgroup owns.  0 means the
     * original whole-sequence kernel, which ignores it. */
    uint tokens_per_block;
    /* MTP3-MIN: emit the mid-sequence recurrent state to a second buffer.
     *
     * The recurrence carries `h` in registers across the token loop and stores
     * only the final value, so the state after any interior row already exists
     * and is simply never written.  Speculative rejection needs exactly the
     * after-row-0 state: today the engine throws it away, restores the
     * pre-verify snapshot, and re-runs row 0 as a full forward -- measured at
     * 28.4 ms per reject, ~60% of everything in an MTP cycle that is not the
     * verifier (2026-09-08-MTPP).  One store here removes that forward.
     *
     * UINT_MAX disables it and the `bank` buffer is then never written; the
     * host still binds a valid buffer because Metal requires one. */
    uint bank_after_row;
};

/*
 * One threadgroup owns one (sequence, head). Four simdgroups update four
 * value rows concurrently; every lane owns four adjacent key columns.
 */
kernel void kernel_glm53_kda_decode(
        constant glm53_kda_args &args,
        device const float   *q_in,
        device const float   *k_in,
        device const float   *v_in,
        device const float   *raw_gate,
        device const float   *raw_beta,
        device const float   *output_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device const float   *output_norm,
        device float         *conv_state,
        device float         *state,
        device float         *out,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *sd = sk + D;
    threadgroup float *sv = sd + D;
    threadgroup float *so = sv + D;
    threadgroup float *reduce_q = so + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    threadgroup float *reduce_o = reduce_k + 4u;
    threadgroup float *beta_shared = reduce_o + 4u;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;
    /* Absolute, full-width state geometry (see glm53_kda_args). */
    const uint state_projection = args.n_heads_total * D;
    const uint state_channel = (args.head_first + head) * D + tid;
    const ulong conv_row_stride = 3ul * HISTORY * state_projection;

    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float *q_state = conv_state +
            (ulong)row * conv_row_stride;
        device float *k_state = q_state + HISTORY * state_projection;
        device float *v_state = k_state + HISTORY * state_projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * state_projection + state_channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * state_projection + state_channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * state_projection + state_channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[state_channel] = q_state[state_projection + state_channel];
        q_state[state_projection + state_channel] =
            q_state[2ul * state_projection + state_channel];
        q_state[2ul * state_projection + state_channel] = q_new;
        k_state[state_channel] = k_state[state_projection + state_channel];
        k_state[state_projection + state_channel] =
            k_state[2ul * state_projection + state_channel];
        k_state[2ul * state_projection + state_channel] = k_new;
        v_state[state_channel] = v_state[state_projection + state_channel];
        v_state[state_projection + state_channel] =
            v_state[2ul * state_projection + state_channel];
        v_state[2ul * state_projection + state_channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        sd[tid] = exp(args.lower_bound *
                      (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] =
            1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    float q_sumsq = sq[tid] * sq[tid];
    float k_sumsq = sk[tid] * sk[tid];
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    if (lane == 0u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head =
        ((ulong)row * args.n_heads_total + args.head_first + head) * D * D;

    for (uint value = sg; value < D; value += 4u) {
        device float4 *hptr =
            (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr * decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h = fma(k4, float4(delta_v), h);
        *hptr = h;
        float hq = simd_sum(dot(h, q4));
        if (lane == 0u) so[value] = hq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    float o_sumsq = so[tid] * so[tid];
    o_sumsq = simd_sum(o_sumsq);
    if (lane == 0u) reduce_o[sg] = o_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
    o_total = simd_sum(o_total);
    const float o_scale = rsqrt(o_total / (float)D + args.norm_eps);
    if (tid < D) {
        const ulong index = input_base + tid;
        const float gate =
            1.0f / (1.0f + exp(-output_gate[index]));
        out[index] = so[tid] * o_scale * output_norm[tid] * gate;
    }
}

/* ---------------------------------------------------------------------------
 * R3 -- the decode recurrence turns 128 independent rows into a 32-deep chain.
 *
 * THE DEFECT.  `for (uint value = sg; value < D; value += 4u)` walks 128
 * independent value rows with 4 simdgroups, 32 iterations deep, and every
 * iteration is a device load -> simd_sum -> fma -> device store -> simd_sum
 * chain with no independent work to interleave.  Row i touches only
 * state[value*128 + k0..k0+3], sv[value] and so[value]; nothing couples it to
 * row i+-1.  At rank shape there are 4096 independent rows and the launch
 * exposes 32 tg x 4 sg = 128 chains.
 *
 * WHY THE TWO PRIOR CLOSURES DO NOT APPLY.  The 2026-09-03 sweeps found this
 * kernel twice and closed it twice -- once priced at ~1% and killed on rule 3,
 * once as "byte-bound near its roof (209-311 GB/s)" with "the compiler already
 * pipelines it".  Both were measured at 64 heads.  Production runs 32 heads per
 * rank after S6c, where the kernel achieves 110-116 GB/s -- 28% of spec, not
 * near any roof.  And if the compiler had already pipelined it, adding
 * simdgroups would buy nothing; it does.
 *
 * TWO COMPETING FIXES, swept together because they are not assumed to compose.
 *
 * (a) VPT -- rows per simdgroup per iteration.  Load all VPT h[] before any
 *     store, then VPT independent simd_sums, then VPT fma+stores, then VPT more
 *     simd_sums.  Chain depth D/NSG -> D/(NSG*VPT).  Grid and threadgroup size
 *     unchanged.  Local sweep: VPT2 1.27x, VPT4 1.49x, VPT8 1.42x, VPT16 0.71x,
 *     VPT32 0.59x (spill).  VPT4 has a 16-pair stage A/B behind it,
 *     p = 0.0017.
 *
 * (b) NSGC -- simdgroups per threadgroup.  Stride by NSGC, size the reduction
 *     scratch to NSGC, and raise the threadgroup to 32*NSGC threads.  Local
 *     sweep: 4 (null control) 0.99x, 8 1.33x, 16 1.44x, 32 1.24x.  Isolated
 *     only -- no stage A/B, which is exactly the arithmetic kill-rule 3
 *     forbids, so it is a sweep arm and not a candidate to bank on its own.
 *
 * EXACTNESS.  Row `value` is still handled by simdgroup `value % NSGC` and its
 * arithmetic is unchanged, so no accumulation order moves and every simd_sum
 * sees the same 32-lane vector.  Both are bit-identical, verified by memcmp
 * over the output AND the full recurrent and conv state.
 *
 * MANDATORY GUARDS AT NSGC > 4, and they are not cosmetic.  The threadgroup is
 * 32*NSGC threads but the scratch rows are D=128 long: `so = scratch + 512`, so
 * an unguarded `so[tid]` at tid=511 reads past the allocation, and an unguarded
 * `sq[tid]` at tid >= 128 reads live sk/sd/sv/so data whose square would
 * silently poison the norm rather than crash.  Hence `tid < D ? ... : 0.0f`.
 *
 * Do NOT split value rows across threadgroups: the conv shift above would race
 * and the two cross-D reductions need all 128 rows in one threadgroup.
 */
/* DEVSCOPE selects the barrier scope, and it is a diagnostic, not a tuning
 * knob.  Both barriers below are written `mem_threadgroup | mem_device`, but
 * nothing in this kernel consumes another thread's DEVICE write: every `*hptr`
 * store lands in a slice indexed by (value, k0), disjoint per thread, and the
 * only cross-thread reads after either barrier are of threadgroup scratch
 * (`sq`/`sk`/`so`).  Ordering for later dispatches comes from kernel
 * completion, not from a barrier inside the kernel.  The third barrier in this
 * same function is already `mem_threadgroup` alone, which is the author having
 * reached the same conclusion once.
 *
 * DEVSCOPE=false drops the device scope so the cost can be MEASURED rather
 * than argued about.  It is a template parameter and not a source edit so both
 * variants live in ONE library and can be compared byte-for-byte in one
 * process, alternating, without a second compile.  Nothing selects
 * DEVSCOPE=false by default. */
template<uint VPT, uint NSGC, bool DEVSCOPE = true>
static inline void glm53_kda_decode_impl(
        constant glm53_kda_args &args,
        device const float   *q_in,
        device const float   *k_in,
        device const float   *v_in,
        device const float   *raw_gate,
        device const float   *raw_beta,
        device const float   *output_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device const float   *output_norm,
        device float         *conv_state,
        device float         *state,
        device float         *out,
        threadgroup float    *scratch,
        uint2  tgpig,
        ushort tid,
        ushort lane,
        ushort sg) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *sd = sk + D;
    threadgroup float *sv = sd + D;
    threadgroup float *so = sv + D;
    threadgroup float *reduce_q = so + D;
    threadgroup float *reduce_k = reduce_q + NSGC;
    threadgroup float *reduce_o = reduce_k + NSGC;
    threadgroup float *beta_shared = reduce_o + NSGC;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;
    const uint state_projection = args.n_heads_total * D;
    const uint state_channel = (args.head_first + head) * D + tid;
    const ulong conv_row_stride = 3ul * HISTORY * state_projection;

    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float *q_state = conv_state + (ulong)row * conv_row_stride;
        device float *k_state = q_state + HISTORY * state_projection;
        device float *v_state = k_state + HISTORY * state_projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * state_projection + state_channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * state_projection + state_channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * state_projection + state_channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[state_channel] = q_state[state_projection + state_channel];
        q_state[state_projection + state_channel] =
            q_state[2ul * state_projection + state_channel];
        q_state[2ul * state_projection + state_channel] = q_new;
        k_state[state_channel] = k_state[state_projection + state_channel];
        k_state[state_projection + state_channel] =
            k_state[2ul * state_projection + state_channel];
        k_state[2ul * state_projection + state_channel] = k_new;
        v_state[state_channel] = v_state[state_projection + state_channel];
        v_state[state_projection + state_channel] =
            v_state[2ul * state_projection + state_channel];
        v_state[2ul * state_projection + state_channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        sd[tid] = exp(args.lower_bound *
                      (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] =
            1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(DEVSCOPE ? (mem_flags::mem_threadgroup | mem_flags::mem_device)
                                 : mem_flags::mem_threadgroup);

    /* tid < D or 0: at NSGC > 4 the threadgroup is wider than the scratch row,
     * and an out-of-range read here poisons the norm instead of faulting. */
    const float sq_t = tid < D ? sq[tid] : 0.0f;
    const float sk_t = tid < D ? sk[tid] : 0.0f;
    float q_sumsq = simd_sum(sq_t * sq_t);
    float k_sumsq = simd_sum(sk_t * sk_t);
    if (lane == 0u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < NSGC ? reduce_q[lane] : 0.0f;
    float k_total = lane < NSGC ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head =
        ((ulong)row * args.n_heads_total + args.head_first + head) * D * D;

    /* Row `value` is still owned by simdgroup `value % NSGC`; VPT only changes
     * how many of that simdgroup's rows are in flight at once, so no row's
     * arithmetic or reduction order moves. */
    for (uint v0 = sg; v0 < D; v0 += NSGC * VPT) {
        device float4 *hp[VPT];
        float4 h[VPT];
        float hk[VPT];
        uint vidx[VPT];
        bool live[VPT];
        for (uint i = 0; i < VPT; i++) {
            vidx[i] = v0 + i * NSGC;
            live[i] = vidx[i] < D;      /* simdgroup-uniform */
            if (live[i]) {
                hp[i] = (device float4 *)(state + state_head +
                                          (ulong)vidx[i] * D + k0);
                h[i] = *hp[i] * decay4;
            }
        }
        for (uint i = 0; i < VPT; i++) {
            if (live[i]) hk[i] = simd_sum(dot(h[i], k4));
        }
        for (uint i = 0; i < VPT; i++) {
            if (live[i]) {
                const float delta_v = (sv[vidx[i]] - hk[i]) * beta_shared[0];
                h[i] = fma(k4, float4(delta_v), h[i]);
                *hp[i] = h[i];
            }
        }
        for (uint i = 0; i < VPT; i++) {
            if (live[i]) {
                const float hq = simd_sum(dot(h[i], q4));
                if (lane == 0u) so[vidx[i]] = hq;
            }
        }
    }
    threadgroup_barrier(DEVSCOPE ? (mem_flags::mem_threadgroup | mem_flags::mem_device)
                                 : mem_flags::mem_threadgroup);

    const float so_t = tid < D ? so[tid] : 0.0f;
    float o_sumsq = simd_sum(so_t * so_t);
    if (lane == 0u) reduce_o[sg] = o_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_total = lane < NSGC ? reduce_o[lane] : 0.0f;
    o_total = simd_sum(o_total);
    const float o_scale = rsqrt(o_total / (float)D + args.norm_eps);
    if (tid < D) {
        const ulong index = input_base + tid;
        const float gate = 1.0f / (1.0f + exp(-output_gate[index]));
        out[index] = so[tid] * o_scale * output_norm[tid] * gate;
    }
}

#define DS4_GLM53_KDA_DECODE_VARIANT_SCOPED(NAME, VPT, NSGC, DEVSCOPE)         \
kernel void NAME(                                                              \
        constant glm53_kda_args &args,                                         \
        device const float   *q_in,                                            \
        device const float   *k_in,                                            \
        device const float   *v_in,                                            \
        device const float   *raw_gate,                                        \
        device const float   *raw_beta,                                        \
        device const float   *output_gate,                                     \
        device const float   *q_conv,                                          \
        device const float   *k_conv,                                          \
        device const float   *v_conv,                                          \
        device const float   *a_log,                                           \
        device const float   *dt_bias,                                         \
        device const float   *output_norm,                                     \
        device float         *conv_state,                                      \
        device float         *state,                                           \
        device float         *out,                                             \
        threadgroup float    *scratch [[threadgroup(0)]],                      \
        uint2  tgpig [[threadgroup_position_in_grid]],                         \
        ushort tid [[thread_index_in_threadgroup]],                            \
        ushort lane [[thread_index_in_simdgroup]],                             \
        ushort sg [[simdgroup_index_in_threadgroup]]) {                        \
    glm53_kda_decode_impl<VPT, NSGC, DEVSCOPE>(                                \
        args, q_in, k_in, v_in, raw_gate, raw_beta, output_gate,               \
        q_conv, k_conv, v_conv, a_log, dt_bias, output_norm,                   \
        conv_state, state, out, scratch, tgpig, tid, lane, sg);                \
}
#define DS4_GLM53_KDA_DECODE_VARIANT(NAME, VPT, NSGC)                          \
    DS4_GLM53_KDA_DECODE_VARIANT_SCOPED(NAME, VPT, NSGC, true)

/* VPT=1, NSGC=4 is the shipped shape: the null control that proves the template
 * reproduces kernel_glm53_kda_decode exactly before any variant is trusted. */
DS4_GLM53_KDA_DECODE_VARIANT(kernel_glm53_kda_decode_v1, 1u, 4u)
DS4_GLM53_KDA_DECODE_VARIANT(kernel_glm53_kda_decode_vpt2, 2u, 4u)
DS4_GLM53_KDA_DECODE_VARIANT(kernel_glm53_kda_decode_vpt4, 4u, 4u)
DS4_GLM53_KDA_DECODE_VARIANT(kernel_glm53_kda_decode_vpt8, 8u, 4u)
DS4_GLM53_KDA_DECODE_VARIANT(kernel_glm53_kda_decode_nsg8, 1u, 8u)
DS4_GLM53_KDA_DECODE_VARIANT(kernel_glm53_kda_decode_nsg16, 1u, 16u)
DS4_GLM53_KDA_DECODE_VARIANT(kernel_glm53_kda_decode_nsg32, 1u, 32u)
/* The two levers together, since "do not assume they compose" is a hypothesis
 * to test rather than a reason not to build the arm. */
DS4_GLM53_KDA_DECODE_VARIANT(kernel_glm53_kda_decode_vpt4_nsg8, 4u, 8u)
/* Barrier-scope probe arm, shipped shape only.  Paired with _v1 above, which is
 * the same code with the device scope kept, so the pair isolates exactly one
 * change.  Never selected by the graph; probe_kdadecode drives it. */
DS4_GLM53_KDA_DECODE_VARIANT_SCOPED(kernel_glm53_kda_decode_v1_tgbar, 1u, 4u, false)

/* ---------------------------------------------------------------------------
 * KDA-PREPARE-PAR.  The kernel below this pair walks all n_rows tokens in ONE
 * threadgroup per head -- 32 threadgroups of 128 threads, each running a 4096
 * iteration chain, measured at 8.46 of the KDA block's 29.75 ms and
 * parallelism-invariant (7.40 ms at H=1), so the chain is the cost, not the
 * grid.
 *
 * Everything in that loop body is parallel over tokens: a width-4 causal
 * depthwise conv, a SiLU, a gate, and an RMS-norm across the head's 128
 * channels.  What forces the serialisation is that the loop OVERWRITES ITS OWN
 * INPUTS -- it reads q[t] at the top and writes q[t] at the bottom -- so the
 * 3-deep shift register exists precisely because the array is destroyed as it
 * goes, and a parallel version cannot re-read x[t-3..t].
 *
 * Fix in two passes.  Pass 1 saves each block's 3-token prologue while the
 * inputs are still pristine; pass 2 then runs blocks independently, each
 * seeding its history from that snapshot rather than from its predecessor.
 * Serial depth drops from n_rows to TOKENS_PER_BLOCK and the grid goes from
 * n_heads to n_heads x n_blocks.
 *
 * Cost is one small buffer: n_blocks * 3 history * 3 tensors * projection
 * floats -- ~9.4 MB at 64 blocks and projection 4096 -- against the ~200 MB a
 * full out-of-place staging of q/k/v would need.
 *
 * Bit-identical.  Each token's four FMAs run in the same order against the same
 * values; history that was read from conv_state/device memory is now read from
 * registers seeded with the identical bytes.  The last block writes conv_state,
 * so the carried state out is unchanged too.
 * ------------------------------------------------------------------------- */

/* Pass 1.  MUST complete before any pass-2 threadgroup writes q/k/v.
 * grid = (n_heads, n_blocks); block 0 seeds from conv_state, the rest from the
 * still-pristine activations. */
kernel void kernel_glm53_kda_prefill_prologue(
        constant glm53_kda_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *conv_state,
        device float         *prologue,
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint head  = tgpig.x;
    const uint block = tgpig.y;
    if (head >= args.n_heads) return;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const uint state_projection = args.n_heads_total * D;
    const uint state_channel = (args.head_first + head) * D + tid;
    const uint tok0 = block * args.tokens_per_block;
    if (tok0 >= args.n_rows) return;

    /* prologue layout: [block][tensor 0..2][history 0..2][channel] */
    const ulong pbase = ((ulong)block * 3ul * HISTORY) * projection + channel;
    device const float *qs = conv_state;
    device const float *ks = qs + HISTORY * state_projection;
    device const float *vs = ks + HISTORY * state_projection;

    for (uint w = 0; w < HISTORY; w++) {
        float qh, kh, vh;
        /* History slot w holds x[tok0 - HISTORY + w].  Before the first token
         * of the whole sequence that is the carried conv_state; otherwise it is
         * the activation itself, which no pass-2 block has touched yet. */
        const int src = (int)tok0 - (int)HISTORY + (int)w;
        if (src < 0) {
            const uint sw = (uint)(src + (int)HISTORY);
            qh = qs[(ulong)sw * state_projection + state_channel];
            kh = ks[(ulong)sw * state_projection + state_channel];
            vh = vs[(ulong)sw * state_projection + state_channel];
        } else {
            const ulong idx = (ulong)src * projection + channel;
            qh = q[idx]; kh = k[idx]; vh = v[idx];
        }
        prologue[pbase + (ulong)(0u * HISTORY + w) * projection] = qh;
        prologue[pbase + (ulong)(1u * HISTORY + w) * projection] = kh;
        prologue[pbase + (ulong)(2u * HISTORY + w) * projection] = vh;
    }
}

kernel void kernel_glm53_kda_prefill_prepare(
        constant glm53_kda_args &args,
        device float         *q,
        device float         *k,
        device float         *v,
        device float         *raw_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        device float         *conv_bank,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    if (head >= args.n_heads) return;
    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *reduce_q = sk + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const uint state_projection = args.n_heads_total * D;
    const uint state_channel = (args.head_first + head) * D + tid;
    device float *q_state = conv_state;
    device float *k_state = q_state + HISTORY * state_projection;
    device float *v_state = k_state + HISTORY * state_projection;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong index = (ulong)token * projection + channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * state_projection + state_channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * state_projection + state_channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * state_projection + state_channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q[index];
        const float k_new = k[index];
        const float v_new = v[index];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);
        q_state[state_channel] = q_state[state_projection + state_channel];
        q_state[state_projection + state_channel] =
            q_state[2ul * state_projection + state_channel];
        q_state[2ul * state_projection + state_channel] = q_new;
        k_state[state_channel] = k_state[state_projection + state_channel];
        k_state[state_projection + state_channel] =
            k_state[2ul * state_projection + state_channel];
        k_state[2ul * state_projection + state_channel] = k_new;
        v_state[state_channel] = v_state[state_projection + state_channel];
        v_state[state_projection + state_channel] =
            v_state[2ul * state_projection + state_channel];
        v_state[2ul * state_projection + state_channel] = v_new;

        /* MTP3-MIN: bank the conv ring as of this row, alongside the recurrent
         * bank the recurrence kernel writes.  Both are needed -- on speculative
         * rejection the next cycle must see the ring as of the ACCEPTED token;
         * a ring still carrying the rejected draft would feed a token that was
         * never emitted into the next three convolutions.
         *
         * Only the plain prepare needs it: blocking engages above
         * prepare_tpb = 64 rows and the two-row verify never reaches that. */
        if (token == args.bank_after_row) {
            device float *qb = conv_bank;
            device float *kb = qb + HISTORY * state_projection;
            device float *vb = kb + HISTORY * state_projection;
            for (uint w = 0; w < HISTORY; w++) {
                const ulong o = (ulong)w * state_projection + state_channel;
                qb[o] = q_state[o];
                kb[o] = k_state[o];
                vb[o] = v_state[o];
            }
        }
        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        v[index] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[index] + dt_bias[channel];
        raw_gate[index] = exp(args.lower_bound *
            (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);

        float q_sumsq = simd_sum(sq[tid] * sq[tid]);
        float k_sumsq = simd_sum(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[sg] = q_sumsq;
            reduce_k[sg] = k_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = simd_sum(q_total);
        k_total = simd_sum(k_total);
        q[index] = sq[tid] * rsqrt(q_total + 1.0e-6f) *
                   0x1.6a09e6p-4f;
        k[index] = sk[tid] * rsqrt(k_total + 1.0e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);
    }
}

kernel void kernel_glm53_kda_prefill_prepare_blocked(
        constant glm53_kda_args &args,
        device float         *q,
        device float         *k,
        device float         *v,
        device float         *raw_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        device const float   *prologue,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint head  = tgpig.x;
    const uint block = tgpig.y;
    if (head >= args.n_heads) return;
    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *reduce_q = sk + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const uint state_projection = args.n_heads_total * D;
    const uint state_channel = (args.head_first + head) * D + tid;
    const uint tok0 = block * args.tokens_per_block;
    if (tok0 >= args.n_rows) return;
    uint tok_end = tok0 + args.tokens_per_block;
    if (tok_end > args.n_rows) tok_end = args.n_rows;
    const bool last_block = (tok_end == args.n_rows);

    /* History in REGISTERS, seeded from the pass-1 snapshot.  The original held
     * it in conv_state and shifted it there; per block that would race, and the
     * registers also remove a device round trip per token. */
    const ulong pbase = ((ulong)block * 3ul * HISTORY) * projection + channel;
    float qh[HISTORY], kh[HISTORY], vh[HISTORY];
    for (uint w = 0; w < HISTORY; w++) {
        qh[w] = prologue[pbase + (ulong)(0u * HISTORY + w) * projection];
        kh[w] = prologue[pbase + (ulong)(1u * HISTORY + w) * projection];
        vh[w] = prologue[pbase + (ulong)(2u * HISTORY + w) * projection];
    }

    for (uint token = tok0; token < tok_end; token++) {
        const ulong index = (ulong)token * projection + channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(qh[w], q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(kh[w], k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(vh[w], v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q[index];
        const float k_new = k[index];
        const float v_new = v[index];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);
        qh[0] = qh[1]; qh[1] = qh[2]; qh[2] = q_new;
        kh[0] = kh[1]; kh[1] = kh[2]; kh[2] = k_new;
        vh[0] = vh[1]; vh[1] = vh[2]; vh[2] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        v[index] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[index] + dt_bias[channel];
        raw_gate[index] = exp(args.lower_bound *
            (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);

        float q_sumsq = simd_sum(sq[tid] * sq[tid]);
        float k_sumsq = simd_sum(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[sg] = q_sumsq;
            reduce_k[sg] = k_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = simd_sum(q_total);
        k_total = simd_sum(k_total);
        q[index] = sq[tid] * rsqrt(q_total + 1.0e-6f) *
                   0x1.6a09e6p-4f;
        k[index] = sk[tid] * rsqrt(k_total + 1.0e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);
    }

    /* Only the block holding the final tokens carries the state forward; the
     * others' registers are discarded, which is what makes them independent. */
    if (last_block) {
        device float *q_state = conv_state;
        device float *k_state = q_state + HISTORY * state_projection;
        device float *v_state = k_state + HISTORY * state_projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_state[(ulong)w * state_projection + state_channel] = qh[w];
            k_state[(ulong)w * state_projection + state_channel] = kh[w];
            v_state[(ulong)w * state_projection + state_channel] = vh[w];
        }
    }
}

kernel void kernel_glm53_kda_prefill_recurrence(
        constant glm53_kda_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *decay,
        device const float   *raw_beta,
        device float         *state,
        device float         *out,
        device float         *bank,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value = tgpig.y * 4u + sg;
    if (head >= args.n_heads || value >= D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;
    device float4 *state_ptr = (device float4 *)(
        state + ((ulong)(args.head_first + head) * D + value) * D + k0);
    float4 h = *state_ptr;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        const float4 q4 = *((device const float4 *)(q + base + k0));
        const float4 k4 = *((device const float4 *)(k + base + k0));
        const float4 decay4 =
            *((device const float4 *)(decay + base + k0));
        h *= decay4;
        const float hk = simd_sum(dot(h, k4));
        const float beta = 1.0f /
            (1.0f + exp(-raw_beta[(ulong)token * args.n_heads + head]));
        const float delta_v = (v[base + value] - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        const float result = simd_sum(dot(h, q4));
        if (lane == 0u) out[base + value] = result;
        /* MTP3-MIN.  Same address arithmetic as state_ptr -- the bank is a
         * full-width state buffer with identical layout, so a rank that owns a
         * head subrange writes at its true head index exactly as it does for
         * `state`.  Guarded, so with banking off this kernel is unchanged. */
        if (token == args.bank_after_row) {
            device float4 *bank_ptr = (device float4 *)(
                bank + ((ulong)(args.head_first + head) * D + value) * D + k0);
            *bank_ptr = h;
        }
    }
    *state_ptr = h;
}

/* K1: kernel_glm53_kda_prefill_recurrence with four value rows per simdgroup.
 *
 * q4, k4 and decay4 are indexed by (token, head, lane) and do NOT depend on
 * `value`, so in the one-value-per-simdgroup form every one of the 128 value
 * simdgroups for a head re-reads the same three float4s on every token. Four
 * rows per simdgroup cuts those reads 4x. raw_beta is likewise per (token,
 * head), so it is hoisted out of the value loop.
 *
 * Grid goes (n_heads, 32) -> (n_heads, 8) with the same 128 threads. Both forms
 * launch the same total number of value-simdgroups' worth of work, so this is a
 * traffic and load-issue reduction, NOT an occupancy change -- do not describe
 * it as a wave-count win.
 *
 * Bit-identical to the original: the per-value arithmetic and the order of the
 * two simd_sum reductions are untouched, and beta is loop-invariant. */
kernel void kernel_glm53_kda_prefill_recurrence_vpt4(
        constant glm53_kda_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *decay,
        device const float   *raw_beta,
        device float         *state,
        device float         *out,
        device float         *bank,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint VPT = 4u;
    const uint head = tgpig.x;
    const uint value0 = (tgpig.y * 4u + (uint)sg) * VPT;
    if (head >= args.n_heads || value0 + VPT > D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;

    device float4 *state_ptr[VPT];
    float4 h[VPT];
    for (uint i = 0; i < VPT; i++) {
        state_ptr[i] = (device float4 *)(
            state + ((ulong)(args.head_first + head) * D + value0 + i) * D + k0);
        h[i] = *state_ptr[i];
    }

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        const float4 q4 = *((device const float4 *)(q + base + k0));
        const float4 k4 = *((device const float4 *)(k + base + k0));
        const float4 decay4 =
            *((device const float4 *)(decay + base + k0));
        const float beta = 1.0f /
            (1.0f + exp(-raw_beta[(ulong)token * args.n_heads + head]));
        for (uint i = 0; i < VPT; i++) {
            h[i] *= decay4;
            const float hk = simd_sum(dot(h[i], k4));
            const float delta_v = (v[base + value0 + i] - hk) * beta;
            h[i] = fma(k4, float4(delta_v), h[i]);
            const float result = simd_sum(dot(h[i], q4));
            if (lane == 0u) out[base + value0 + i] = result;
        }
        /* MTP3-MIN -- see the baseline kernel. */
        if (token == args.bank_after_row) {
            for (uint i = 0; i < VPT; i++) {
                device float4 *bank_ptr = (device float4 *)(
                    bank + ((ulong)(args.head_first + head) * D + value0 + i) * D + k0);
                *bank_ptr = h[i];
            }
        }
    }
    for (uint i = 0; i < VPT; i++) *state_ptr[i] = h[i];
}

kernel void kernel_glm53_kda_prefill_output(
        constant glm53_kda_args &args,
        device float         *out,
        device const float   *output_gate,
        device const float   *output_norm,
        threadgroup float    *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_rows || head >= args.n_heads) return;
    const uint projection = args.n_heads * D;
    const ulong base = (ulong)token * projection + head * D;
    const float raw = out[base + tid];
    float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 4u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.norm_eps);
    out[base + tid] = raw * scale * output_norm[tid] /
        (1.0f + exp(-output_gate[base + tid]));
}

// ---------------------------------------------------------------------------
// R1 -- f_a + beta + g_a as one dispatch.
//
// THE DEFECT.  Three Q8_0 matvecs read the same activation (attn_norm), none
// depends on another, and all three are far too small to fill the machine:
// 4096->128, 4096->32 and 4096->128 at nr0=2 are 64, 16 and 64 threadgroups.
// On a 60-core part beta alone can reach 27% of the GPU; f_a and g_a land at
// ~1.07 threadgroups per core with a ragged tail.  The batch encoder is
// MTLDispatchTypeSerial, so they do not overlap -- "the graph was already
// overlapping them" is not available here.
//
// WHY THEY COULD NOT SIMPLY BE MERGED.  f_a and g_a both wrote the single
// kda_lowrank buffer, with f_b reading it in between.  That is a false
// dependency created by buffer reuse, not by the math; the host now gives g_a
// its own kda_lowrank_g (ds4.c), which is what makes this legal.  On its own
// that second buffer buys ~0.35 us/layer and is not worth doing -- it is a
// prerequisite, not an optimisation.
//
// WHY THIS IS NOT DF2.  DF2 paired two equal-out_dim matvecs and dispatched
// max_out_dim/nr0, HALVING the grid: every threadgroup did twice the work and
// it measured -1.15%.  This SUMS the row counts -- (128+128+32)/2 = 144
// threadgroups, each doing exactly the two rows one threadgroup does today.
// Neither the threadgroup count nor the per-threadgroup work changes; only the
// number of dispatch boundaries does, from three to one.
//
// EXACTNESS.  The body is kernel_mul_mv_q8_0_f32_impl called verbatim, with the
// same NR0, the same nsg function constant, the same per-row block traversal
// and the same two-stage reduction.  The only thing this kernel adds is the
// tgpig.x -> (bank, local row block) map, and within a bank the local index is
// exactly the tgpig.x the separate dispatch would have had -- so no row's
// accumulation order moves.  Bit-identical by construction, and measured 0/288
// differing words.
//
// The bank is a function of tgpig.x alone, so it is uniform across the
// threadgroup and the reduction's threadgroup_barrier stays well-formed.
struct glm53_kda_small_mv_args {
    ds4_metal_args_mul_mv mv;   // shared shape; ne01/ne0 are overwritten per bank
    uint rows[3];               // output rows in each bank
    uint tg0[4];                // first tgpig.x of each bank, plus the end
    uint n_banks;
};

[[host_name("kernel_glm53_kda_small_mv_merged")]]
kernel void kernel_glm53_kda_small_mv_merged(
        constant glm53_kda_small_mv_args & a [[buffer(0)]],
        device const char * w0   [[buffer(1)]],
        device const char * w1   [[buffer(2)]],
        device const char * w2   [[buffer(3)]],
        device const char * src1 [[buffer(4)]],
        device       char * d0   [[buffer(5)]],
        device       char * d1   [[buffer(6)]],
        device       char * d2   [[buffer(7)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint bank = 0;
    for (uint b = 1; b < a.n_banks; b++) {
        if (tgpig.x >= a.tg0[b]) bank = b;
    }
    // Surplus threadgroups (the host rounds the grid up) retire here rather
    // than writing past a destination.
    if (tgpig.x >= a.tg0[a.n_banks]) return;

    device const char * src0 = bank == 0 ? w0 : (bank == 1 ? w1 : w2);
    device       char * dst  = bank == 0 ? d0 : (bank == 1 ? d1 : d2);

    ds4_metal_args_mul_mv args = a.mv;
    args.ne01 = (int)a.rows[bank];
    args.ne0  = (int)a.rows[bank];

    const uint3 tg = uint3(tgpig.x - a.tg0[bank], tgpig.y, tgpig.z);

    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, ds4_metal_args_mul_mv>(
        args, src0, src1, dst, shmem, tg, tiisg, sgitg);
}
