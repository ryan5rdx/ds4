/*
 * U64TOP1-TP merge gate: does the two-key exchange pick the SAME token as a
 * full-vector argmax?
 *
 * The compact path never materialises the whole logit vector on either rank, so
 * nothing at runtime can compare the two answers -- a divergence would show up
 * as slightly different text, months later, with no error anywhere. This is the
 * only place the comparison is possible, so it is where the equivalence has to
 * be pinned.
 *
 * The reference is sample_argmax's rule as transcribed in probe_top1.c: strict
 * `>` scanning upward from index 0, so ties keep the LOWEST index. The cases
 * that matter are the ones where the bit trick alone disagrees with `>`:
 * NaN (never selected), -0.0 vs +0.0 (neither is greater), exact ties across
 * the split boundary, and a half that is entirely NaN or -inf.
 *
 * Build: make tests/test_top1_merge && ./tests/test_top1_merge
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#include "ds4_top1_key.h"

#define VOCAB  2048u
#define VHALF  (VOCAB / 2u)

static int fails;

/* The CPU sampler's rule, verbatim in intent: seed (0, -inf), strict `>`. */
static uint32_t ref_argmax(const float *v, uint32_t n) {
    uint32_t best = 0;
    float best_v = -INFINITY;
    for (uint32_t i = 0; i < n; i++) {
        if (v[i] > best_v) { best_v = v[i]; best = i; }
    }
    return best;
}

/* What ds4_session_glm_top1_key() does for one rank: argmax over [base, base+n)
 * seeded at `base`, then pack the logit AT the winner with its global id. */
static uint64_t rank_key(const float *v, uint32_t base, uint32_t n) {
    uint32_t best = base;
    float best_v = -INFINITY;
    for (uint32_t i = base; i < base + n; i++) {
        if (v[i] > best_v) { best_v = v[i]; best = i; }
    }
    return ds4_top1_pack_key(v[best], best);
}

static void check(const char *name, const float *v) {
    const uint32_t want = ref_argmax(v, VOCAB);
    const uint64_t k0 = rank_key(v, 0, VHALF);
    const uint64_t k1 = rank_key(v, VHALF, VHALF);
    const uint64_t merged = k0 > k1 ? k0 : k1;   /* rank 0's unsigned max */
    const uint32_t got = ds4_top1_key_index(merged);
    if (got != want) {
        fails++;
        printf("FAIL %-28s merged idx %u (%.9g) != argmax %u (%.9g)\n",
               name, got, got < VOCAB ? v[got] : 0.0f, want, v[want]);
    }
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint32_t rng32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}
static float rnd_logit(void) {
    return ((float)(rng32() % 20001u) - 10000.0f) / 500.0f;   /* -20 .. +20 */
}

int main(void) {
    static float v[VOCAB];
    char name[64];

    /* 1. Random vectors -- the ordinary case, both halves live. */
    for (int trial = 0; trial < 4000; trial++) {
        for (uint32_t i = 0; i < VOCAB; i++) v[i] = rnd_logit();
        snprintf(name, sizeof(name), "random[%d]", trial);
        check(name, v);
    }

    /* 2. Winner planted at every structurally interesting index. The boundary
     *    indices are the ones a base-offset bug moves by exactly vhalf. */
    const uint32_t spots[] = {0, 1, VHALF - 1, VHALF, VHALF + 1, VOCAB - 1};
    for (size_t s = 0; s < sizeof(spots) / sizeof(spots[0]); s++) {
        for (uint32_t i = 0; i < VOCAB; i++) v[i] = -5.0f;
        v[spots[s]] = 42.0f;
        snprintf(name, sizeof(name), "planted@%u", spots[s]);
        check(name, v);
    }

    /* 3. Exact ties. `>` keeps the lowest index, so a tie spanning the split
     *    must resolve to rank 0's -- this is the case the inverted index term
     *    in the key exists for, and the one a plain (score, index) pack gets
     *    wrong half the time. */
    for (uint32_t i = 0; i < VOCAB; i++) v[i] = 3.25f;
    check("all-tied", v);
    for (uint32_t i = 0; i < VOCAB; i++) v[i] = -1.0f;
    v[VHALF - 1] = 7.0f; v[VHALF] = 7.0f;
    check("tie-across-split", v);
    v[VHALF - 1] = -1.0f; v[7] = 7.0f; v[VOCAB - 3] = 7.0f;
    check("tie-far-apart", v);

    /* 4. Signed zero. +0.0 and -0.0 are not ordered by `>`, so the first wins;
     *    their raw bits say the opposite. */
    for (uint32_t i = 0; i < VOCAB; i++) v[i] = -3.0f;
    v[3] = -0.0f; v[VHALF + 3] = 0.0f;
    check("neg-zero-first", v);
    v[3] = 0.0f; v[VHALF + 3] = -0.0f;
    check("pos-zero-first", v);
    for (uint32_t i = 0; i < VOCAB; i++) v[i] = (i & 1u) ? 0.0f : -0.0f;
    check("all-zeros-mixed-sign", v);

    /* 5. NaN never wins -- including when it is the only thing in a half. */
    for (uint32_t i = 0; i < VOCAB; i++) v[i] = -2.0f;
    v[10] = NAN; v[VHALF + 10] = NAN; v[VOCAB - 1] = 1.0f;
    check("nan-scattered", v);
    for (uint32_t i = 0; i < VHALF; i++) v[i] = NAN;
    for (uint32_t i = VHALF; i < VOCAB; i++) v[i] = -2.0f;
    v[VHALF + 5] = 1.0f;
    check("rank0-all-nan", v);
    for (uint32_t i = 0; i < VHALF; i++) v[i] = -2.0f;
    for (uint32_t i = VHALF; i < VOCAB; i++) v[i] = NAN;
    v[5] = 1.0f;
    check("rank1-all-nan", v);
    for (uint32_t i = 0; i < VOCAB; i++) v[i] = NAN;
    check("all-nan", v);              /* falls back to index 0, as the CPU does */

    /* 6. -inf everywhere: nothing beats the seed on either side, and rank 1
     *    must not claim a token it does not own. */
    for (uint32_t i = 0; i < VOCAB; i++) v[i] = -INFINITY;
    check("all-neg-inf", v);
    for (uint32_t i = 0; i < VHALF; i++) v[i] = -INFINITY;
    for (uint32_t i = VHALF; i < VOCAB; i++) v[i] = -9.0f;
    check("rank0-all-neg-inf", v);

    /* 7. +inf, and a tie between two +inf across the split. */
    for (uint32_t i = 0; i < VOCAB; i++) v[i] = 0.5f;
    v[VHALF + 2] = INFINITY;
    check("pos-inf-rank1", v);
    v[2] = INFINITY;
    check("pos-inf-both", v);

    /* 8. Sparse random with mostly -inf, the masked-vocabulary shape. */
    for (int trial = 0; trial < 2000; trial++) {
        for (uint32_t i = 0; i < VOCAB; i++) v[i] = -INFINITY;
        const uint32_t live = 1u + rng32() % 8u;
        for (uint32_t j = 0; j < live; j++) v[rng32() % VOCAB] = rnd_logit();
        snprintf(name, sizeof(name), "sparse[%d]", trial);
        check(name, v);
    }

    /* 9. The index term must survive the full production id range: 77440 is
     *    GLM's per-rank half, so rank 1's ids run to 154879. */
    {
        const uint32_t ids[] = {0, 1, 77439, 77440, 154878, 154879, 0xfffffffeu};
        for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); i++) {
            const uint64_t k = ds4_top1_pack_key(1.5f, ids[i]);
            if (ds4_top1_key_index(k) != ids[i]) {
                fails++;
                printf("FAIL roundtrip id %u -> %u\n",
                       ids[i], ds4_top1_key_index(k));
            }
        }
        /* Equal score, lower id must produce the HIGHER key. */
        if (!(ds4_top1_pack_key(1.5f, 77439) > ds4_top1_pack_key(1.5f, 77440))) {
            fails++;
            printf("FAIL tie ordering across the split boundary\n");
        }
    }

    printf("%s\n", fails == 0
           ? "PASS: compact two-key merge matches full-vector argmax on every case"
           : "FAILED");
    return fails == 0 ? 0 : 1;
}
