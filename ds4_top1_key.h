#ifndef DS4_TOP1_KEY_H
#define DS4_TOP1_KEY_H

/* THE packed-key definition -- shared verbatim by the Metal reducer
 * (metal/top1.metal) and the C host (ds4.c's compact TP path).
 *
 * Two ranks exchange these keys over the wire and rank 0 takes an unsigned max.
 * If the two producers disagreed on any float case the pair would emit a token
 * that neither rank, and no single-node run, would ever produce -- and nothing
 * downstream could see it, because a wrong key is still a well-formed key. The
 * only defence that holds under later edits is a single definition, so this
 * header is the definition and neither side may keep a copy.
 *
 * Pack (score, index) so that UNSIGNED MAX reproduces the CPU sampler exactly.
 *
 * The CPU is argmax_f32_unrolled8_range()'s `if (x > v)`: strictly greater, so
 * ties keep the LOWEST index. `0xffffffff - idx` inverts the index into the low
 * word, which makes a lower index a higher key, so unsigned max keeps it too.
 *
 * Two float cases are not handled by the bit trick alone and both would produce
 * a different token from the CPU:
 *
 *   -0.0 vs +0.0 -- `>` says neither is greater, so the CPU keeps the first.
 *     The raw bits disagree: +0 maps to 0x80000000 and -0 to 0x7fffffff, so a
 *     later +0 would beat an earlier -0. Canonicalised on the bits, because
 *     `v == 0.0f ? 0.0f : v` is exactly what fast-math is licensed to delete.
 *
 *   NaN -- `NaN > v` is false, so the CPU never selects one; the bit trick
 *     makes NaN the LARGEST key and would always select it. Mapped to
 *     -infinity's key WITH ITS INDEX RETAINED, which reproduces the CPU in both
 *     directions: a NaN loses to any finite value, and an all-NaN row falls
 *     back to index 0 exactly as the CPU's `best = 0` initialiser does.
 */

#ifdef __METAL_VERSION__
#define DS4_TOP1_U64  ulong
#define DS4_TOP1_U32  uint32_t
#define DS4_TOP1_INLINE static inline
#define DS4_TOP1_BITS(v) as_type<uint32_t>(v)
#else
#include <stdint.h>
#include <string.h>
#define DS4_TOP1_U64  uint64_t
#define DS4_TOP1_U32  uint32_t
#define DS4_TOP1_INLINE static inline
static inline uint32_t ds4_top1_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    return u;
}
#define DS4_TOP1_BITS(v) ds4_top1_bits(v)
#endif

/* THE SAMPLER FLOOR. sample_argmax_unrolled8() seeds best_v with DS4_NEG_INF
 * and only replaces it on a strict `>`, so a value at or below the floor never
 * wins -- the seed index does. This is a FINITE sentinel, not -inf, and the
 * difference is observable: a half whose values are all <= -1e30 must yield the
 * seed, not the largest of them.
 *
 * Every producer of a key must therefore start from this floor, or it will
 * disagree with the CPU sampler on exactly the inputs a masked vocabulary
 * produces. Defined here rather than read from ds4.c because the Metal reducer
 * needs the same number and cannot include ds4.c; a static assert in ds4.c ties
 * the two together. */
#define DS4_TOP1_NEG_INF (-1.0e30f)

DS4_TOP1_INLINE DS4_TOP1_U64 ds4_top1_pack_key(float v, DS4_TOP1_U32 idx) {
    DS4_TOP1_U32 u = DS4_TOP1_BITS(v);
    if ((u & 0x7fffffffu) == 0u) u = 0u;                    /* -0.0 -> +0.0 */
    if ((u & 0x7fffffffu) > 0x7f800000u) u = 0xff800000u;   /* NaN -> -inf  */
    const DS4_TOP1_U32 ordered = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
    return ((DS4_TOP1_U64)ordered << 32) | (DS4_TOP1_U64)(0xffffffffu - idx);
}

/* The key a reduction must START from: the floor paired with the first column
 * it is responsible for. Reducing from 0 instead -- which is below every packed
 * key including -inf's -- makes the winner the largest actual value even when
 * every value is beneath the floor, and the CPU would have returned the seed
 * index there. Under the vocabulary split `base` is the rank's first global id,
 * so a rank that sees nothing above the floor claims its own first column and
 * never a column it does not own. */
DS4_TOP1_INLINE DS4_TOP1_U64 ds4_top1_seed_key(DS4_TOP1_U32 base) {
    return ds4_top1_pack_key(DS4_TOP1_NEG_INF, base);
}

/* The index a key decodes to. The score half is deliberately NOT recoverable as
 * a float by the consumer: a compact exchange carries a winner, not a logit. */
DS4_TOP1_INLINE DS4_TOP1_U32 ds4_top1_key_index(DS4_TOP1_U64 k) {
    return 0xffffffffu - (DS4_TOP1_U32)(k & 0xffffffffu);
}

#endif /* DS4_TOP1_KEY_H */
