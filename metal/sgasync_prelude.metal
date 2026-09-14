/*
 * SGASYNC prelude -- concatenated FIRST, so every later .metal file can use the
 * private async-copy helpers.
 *
 * It was previously inside dsv4_misc.metal, which works only for that file and
 * for the ones after it in ds4_gpu_full_source()'s order. dsv4_rope.metal comes
 * before it, so a helper there was unreachable. Ordering in the loader list is
 * the mechanism -- see ds4_top1_key.h for the same pattern.
 *
 * Everything here is behind DS4_PRIVATE_CLONE, which only
 * tests/make_private_clone_source.py defines, so the modern runtime corpus
 * never sees a line of it.
 */
#ifdef DS4_PRIVATE_CLONE
#include <metal_simdgroup_async>
#if !defined(__HAVE_SIMDGROUP_ASYNC_COPY__)
#error "DS4_PRIVATE_CLONE set but the toolchain has no simdgroup async copy"
#endif

/* A run of `n_rows` CONSECUTIVE cache rows is one contiguous copy.
 *
 * GLM53 selection is not an arbitrary gather: kernel_glm53_expand_pool_selection
 * emits `pool * pool_size + slot % pool_size`, so each pool contributes
 * pool_size consecutive raw rows. A 16-row stage at pool_size 4 is therefore
 * four independent contiguous 4 KiB copies -- which is the only reason the copy
 * engine applies to the indexed-attention kernels at all.
 *
 * That is a property of the producer, not a guarantee, so it is CHECKED per run
 * and the caller falls back to the scalar path when it does not hold (short
 * tails, invalid rows, a different pool_size). Assuming it would be a silent
 * wrong-rows bug, and attention output is exactly the kind of thing that
 * degrades without failing.
 */
static inline bool ds4_sgasync_run_is_contiguous(
        device const uint32_t *selected, uint base, uint n_rows, uint cache_cap) {
    const uint first = selected[base];
    if (first >= cache_cap || first + n_rows > cache_cap) return false;
    for (uint i = 1; i < n_rows; ++i) {
        if (selected[base + i] != first + i) return false;
    }
    return true;
}

/* SMALL CONTIGUOUS STAGE: one simdgroup issues the whole thing.
 *
 * For the 512 B - 2 KiB stages this is the right shape. Partitioning across
 * simdgroups the way the 16 KiB stages do would hand each one ~128 bytes, which
 * is below anything the copy engine was ever measured on, and it would need a
 * per-run ownership argument for no benefit. One simdgroup issues, wait()
 * orders it, and the CALLER'S EXISTING BARRIER publishes it to the rest -- so
 * this adds no barrier and no threadgroup memory, which is what makes these
 * sites byte-identical and free rather than a tradeoff.
 *
 * The caller must already have a threadgroup_barrier between this and the first
 * read of `dst`. Every site this is applied to already did.
 */
template <typename T>
static inline void ds4_sgasync_stage_1d(threadgroup T *dst,
                                        device const T *src,
                                        uint n_elements,
                                        ushort sg) {
    if (sg == 0u) {
        simdgroup_future<void> c = simdgroup_async_copy(dst, src,
                                                        (ulong)n_elements);
        c.wait();
    }
}
#endif  /* DS4_PRIVATE_CLONE */
