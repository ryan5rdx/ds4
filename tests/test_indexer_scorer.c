/* Bit-exactness gate for the register-resident indexer prefill scorer.
 *
 * PURPOSE
 *
 * The default scorer keeps the staged K tile in simdgroup registers across the
 * head loop instead of reloading it from threadgroup memory per head.  That is
 * a data-movement change only -- staging, barriers and the per-pair reduction
 * order are unchanged -- so it must be bit-identical to the threadgroup-staged
 * scorer it replaced.  This test asserts exactly that: the same inputs through
 * the public entry point twice, once with the default and once with the
 * rollback, byte-compared.  Any difference is fatal.
 *
 * A CPU reference in the same arithmetic order is computed for the first
 * differing cell only, as a diagnostic: it says whether the default or the
 * fallback drifted, which a two-way GPU comparison cannot.
 *
 * WHY THESE SHAPES
 *
 * n_tokens is 61 -- above the 32-token gate that admits the tiled scorers, and
 * deliberately not a multiple of the 8- or 16-row tile heights, so the ragged
 * final tile is exercised.  n_comp is likewise off a tile boundary.  The causal
 * -INFINITY cut is included because it is the one input the reduction has to
 * treat specially.
 *
 * USAGE
 *
 *   make test-indexer-scorer
 *   ./tests/test_indexer_scorer
 *
 * Requires a Metal device.  Needs no model weights.
 */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } \
} while (0)

static float half_round(float v) {
    /* mirror the f32->f16 rounding the pack pass applies */
    _Float16 h = (_Float16)v;
    return (float)h;
}

int main(void) {
    CHECK(ds4_gpu_init(), "gpu init");

    const uint32_t n_head = 64u;
    const uint32_t head_dim = 128u;
    const uint32_t ratio = 4u;
    const uint32_t n_tokens = 61u;   /* >=32 (tiled2 gate), not a multiple of 8 */
    const uint32_t pos0 = 731u;
    const uint32_t n_comp = 197u;    /* not a multiple of 64: edge comp tile */
    const float scale = 1.0f / 11.3137f;

    const size_t q_count = (size_t)n_tokens * n_head * head_dim;
    const size_t w_count = (size_t)n_tokens * n_head;
    const size_t k_count = (size_t)n_comp * head_dim;
    const size_t s_count = (size_t)n_tokens * n_comp;

    float *hq = malloc(q_count * sizeof(float));
    float *hw = malloc(w_count * sizeof(float));
    float *hk = malloc(k_count * sizeof(float));
    float *sa = malloc(s_count * sizeof(float));
    float *sb = malloc(s_count * sizeof(float));
    CHECK(hq && hw && hk && sa && sb, "host alloc");

    srand(1337);
    for (size_t i = 0; i < q_count; i++)
        hq[i] = (float)(rand() % 2001 - 1000) / 512.0f;
    for (size_t i = 0; i < w_count; i++)
        hw[i] = (float)(rand() % 2001 - 1000) / 1024.0f;
    for (size_t i = 0; i < k_count; i++)
        hk[i] = (float)(rand() % 2001 - 1000) / 512.0f;

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *w = ds4_gpu_tensor_alloc(w_count * sizeof(float));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(k_count * sizeof(float));
    ds4_gpu_tensor *s = ds4_gpu_tensor_alloc(s_count * sizeof(float));
    CHECK(q && w && k && s, "gpu alloc");
    CHECK(ds4_gpu_tensor_write(q, 0, hq, q_count * sizeof(float)), "write q");
    CHECK(ds4_gpu_tensor_write(w, 0, hw, w_count * sizeof(float)), "write w");
    CHECK(ds4_gpu_tensor_write(k, 0, hk, k_count * sizeof(float)), "write k");

    /* Reference arm: the threadgroup-staged scorer.  It is no longer the
     * default, so it has to be asked for -- leaving both arms on the default
     * would compare a path against itself and pass vacuously. */
    setenv("DS4_METAL_DISABLE_INDEXER_SCORES_TILED5", "1", 1);
    CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(
              s, q, w, k, n_comp, n_tokens, pos0, n_head, head_dim, ratio,
              scale), "threadgroup-staged scorer compute");
    CHECK(ds4_gpu_tensor_read(s, 0, sa, s_count * sizeof(float)), "read a");
    const char *arm_a = ds4_gpu_last_indexer_scorer();

    /* Candidate arm: the shipped default. */
    unsetenv("DS4_METAL_DISABLE_INDEXER_SCORES_TILED5");
    CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(
              s, q, w, k, n_comp, n_tokens, pos0, n_head, head_dim, ratio,
              scale), "register-resident scorer compute");
    CHECK(ds4_gpu_tensor_read(s, 0, sb, s_count * sizeof(float)), "read b");
    const char *arm_b = ds4_gpu_last_indexer_scorer();
    /* Prove two different kernels ran.  Without this the test passes whenever
     * the arms silently resolve to the same path -- an env rename, a moved
     * gate, or a shape that misses the selector all produce a green vacuous
     * pass, which is the failure this test exists to prevent. */
    fprintf(stderr, "arms: reference=%s candidate=%s\n", arm_a, arm_b);
    CHECK(strcmp(arm_a, arm_b) != 0, "both arms selected the same kernel");

    size_t diff = 0, first = (size_t)-1;
    for (size_t i = 0; i < s_count; i++) {
        if (memcmp(&sa[i], &sb[i], sizeof(float)) != 0) {
            if (first == (size_t)-1) first = i;
            diff++;
        }
    }
    if (diff) {
        const uint32_t t = (uint32_t)(first / n_comp);
        const uint32_t c = (uint32_t)(first % n_comp);
        fprintf(stderr,
                "FAIL: register-resident scorer differs at %zu/%zu floats; first token=%u "
                "comp=%u tiled2=%a tiled3=%a\n",
                diff, s_count, t, c, sa[first], sb[first]);
        /* dump the first differing pair's per-head picture on the CPU */
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            float dot = 0.0f;
            const float *qh = hq + ((size_t)t * n_head + h) * head_dim;
            const float *kr = hk + (size_t)c * head_dim;
            for (uint32_t d = 0; d < head_dim; d++)
                dot += half_round(qh[d]) * half_round(kr[d]);
            if (dot > 0.0f) acc += dot * (hw[(size_t)t * n_head + h] * scale);
        }
        fprintf(stderr, "  (cpu f16-ish ref for that cell: %a)\n", acc);
        return 1;
    }

    /* sanity: causal cut present */
    size_t ninf = 0;
    for (size_t i = 0; i < s_count; i++) if (isinf(sa[i])) ninf++;
    fprintf(stderr, "scorer A/B bit-exact: %zu floats identical (%zu -inf)\n",
            s_count, ninf);
    ds4_gpu_cleanup();
    return 0;
}
