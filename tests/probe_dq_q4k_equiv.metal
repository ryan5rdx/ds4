// Appended to the full corpus: compares the device and threadgroup Q4_K
// dequantisers on identical bytes. Any difference is a transcription error in
// dequantize_q4_K_tg / ds4_get_scale_min_k4_just2_tg.
kernel void ds4_probe_dq_q4k_equiv(
        device const block_q4_K *blocks [[buffer(0)]],
        device uint *mismatches         [[buffer(1)]],
        device float *first_bad         [[buffer(2)]],
        threadgroup block_q4_K *stage   [[threadgroup(0)]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 tgs3 [[threads_per_threadgroup]],
        uint3 tg   [[threadgroup_position_in_grid]]) {
    const uint tid = tid3.x, nthreads = tgs3.x;
    // Stage one block per threadgroup, byte-wise, exactly as the MoE arm does.
    threadgroup uchar *d = (threadgroup uchar *)stage;
    device const uchar *srcb = (device const uchar *)(blocks + tg.x);
    for (uint i = tid; i < sizeof(block_q4_K); i += nthreads) d[i] = srcb[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short il = (short)tid; il < 16; il += (short)nthreads) {
        half4x4 a, b;
        dequantize_q4_K(blocks + tg.x, il, a);
        dequantize_q4_K_tg(stage, il, b);
        for (int i = 0; i < 16; ++i) {
            const half x = a[i/4][i%4], y = b[i/4][i%4];
            if (as_type<ushort>(x) != as_type<ushort>(y)) {
                const uint slot = atomic_fetch_add_explicit(
                    (device atomic_uint *)mismatches, 1u, memory_order_relaxed);
                if (slot == 0u) { first_bad[0] = (float)x; first_bad[1] = (float)y;
                                  first_bad[2] = (float)il; first_bad[3] = (float)i; }
            }
        }
    }
}
