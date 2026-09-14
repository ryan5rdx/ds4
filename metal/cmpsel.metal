#include <metal_stdlib>
using namespace metal;

/* CMPSEL-AUDIT — does the compiler emit compare/select, or a branch?
 *
 * fcmpsel/icmpsel is an AGX compare-and-select ALU instruction. It is NOT a
 * separately dispatchable unit and there is no throughput to unlock by
 * rewriting inference maths; ordinary MSL min/max/select/ternary should already
 * lower to it where profitable. The only credible lead the campaign brief
 * allows is a hot compare/swap where the compiler accidentally emitted
 * divergent control flow -- and it names one production candidate: the
 * packed-ulong compare/swap in the top-k / argmax reducer.
 *
 * So this measures that, in the shape it actually appears in
 * (ds4_top1_reduce_tg's `if (rhs > scratch[tid]) scratch[tid] = rhs`), against
 * the alternatives, plus float and uint for context.
 *
 * DEPENDENT chains measure latency: each step needs the previous result.
 * INDEPENDENT streams measure throughput: four accumulators break the chain.
 * A divergent predicate is included because if-conversion's whole advantage
 * disappears when lanes disagree -- and its disadvantage (executing both sides)
 * does not.
 */

struct ds4_cmpsel_args { uint n_iter; uint divergent; };

#define CS_KERNEL(NAME, TYPE, INIT, BODY)                                     \
kernel void NAME(constant ds4_cmpsel_args & a  [[buffer(0)]],                 \
                 device TYPE             * out [[buffer(1)]],                 \
                 uint tid [[thread_position_in_grid]]) {                      \
    TYPE x = (TYPE)(INIT + tid);                                              \
    TYPE y = (TYPE)(INIT + (tid ^ 7u));                                       \
    TYPE a0 = x, a1 = y, a2 = x + (TYPE)1, a3 = y + (TYPE)1;                  \
    const bool div = a.divergent != 0u;                                       \
    for (uint i = 0; i < a.n_iter; ++i) {                                     \
        const bool p = div ? ((tid + i) & 1u) != 0u : (i & 1u) != 0u;         \
        BODY                                                                  \
    }                                                                         \
    out[tid] = a0 + a1 + a2 + a3;                                             \
}

/* --- packed ulong: the production candidate --- */
CS_KERNEL(kernel_ds4_cs_ulong_ternary, ulong, 1000000ul,
    a0 = (a0 > a1) ? a0 : a1; a1 = p ? a1 + 1ul : a1 + 3ul;
    a2 = (a2 > a3) ? a2 : a3; a3 = p ? a3 + 1ul : a3 + 3ul;)

CS_KERNEL(kernel_ds4_cs_ulong_max, ulong, 1000000ul,
    a0 = max(a0, a1); a1 = p ? a1 + 1ul : a1 + 3ul;
    a2 = max(a2, a3); a3 = p ? a3 + 1ul : a3 + 3ul;)

/* The exact shape ds4_top1_reduce_tg uses: a guarded store, not a ternary. */
CS_KERNEL(kernel_ds4_cs_ulong_branch, ulong, 1000000ul,
    if (a1 > a0) a0 = a1; a1 = p ? a1 + 1ul : a1 + 3ul;
    if (a3 > a2) a2 = a3; a3 = p ? a3 + 1ul : a3 + 3ul;)

/* Dependent chain: every step needs the previous, so this is latency. */
CS_KERNEL(kernel_ds4_cs_ulong_dep, ulong, 1000000ul,
    a0 = (a0 > a1) ? a0 : a1; a1 = a0 + (p ? 1ul : 3ul);
    a2 = a0; a3 = a1;)

/* --- float and uint, for context --- */
CS_KERNEL(kernel_ds4_cs_float_ternary, float, 1.0f,
    a0 = (a0 > a1) ? a0 : a1; a1 = p ? a1 + 1.0f : a1 + 3.0f;
    a2 = (a2 > a3) ? a2 : a3; a3 = p ? a3 + 1.0f : a3 + 3.0f;)

CS_KERNEL(kernel_ds4_cs_float_max, float, 1.0f,
    a0 = max(a0, a1); a1 = p ? a1 + 1.0f : a1 + 3.0f;
    a2 = max(a2, a3); a3 = p ? a3 + 1.0f : a3 + 3.0f;)

CS_KERNEL(kernel_ds4_cs_uint_ternary, uint, 1u,
    a0 = (a0 > a1) ? a0 : a1; a1 = p ? a1 + 1u : a1 + 3u;
    a2 = (a2 > a3) ? a2 : a3; a3 = p ? a3 + 1u : a3 + 3u;)

CS_KERNEL(kernel_ds4_cs_uint_max, uint, 1u,
    a0 = max(a0, a1); a1 = p ? a1 + 1u : a1 + 3u;
    a2 = max(a2, a3); a3 = p ? a3 + 1u : a3 + 3u;)
