/* Correctness gate for the streaming top-512 indexer selector.
 *
 * PURPOSE
 *
 * The selector is default-on for wide prefill rows and replaces a block
 * argsort plus merge cascade.  It is only admissible if it returns exactly the
 * same index list, so this test pins that: for each row it compares
 *
 *   1. the selector's output,
 *   2. the block-argsort fallback's output (DS4_METAL_DISABLE_TOPK_STREAM512),
 *   3. a CPU reference sorted by (score descending, index ascending),
 *
 * and fails on any difference.  The CPU reference is what makes the test
 * meaningful: comparing the two GPU paths alone cannot catch an error they
 * share, and cannot catch a configuration in which both arms resolve to the
 * same kernel.
 *
 * WHY THE INPUTS LOOK LIKE THIS
 *
 * Ties are the interesting case, because the selector orders keys by packing
 * score and index into one word rather than by comparing scores.  Scores are
 * therefore quantised coarsely so exact ties are common; set FINE=1 for a
 * near-tie-free distribution as a control.  Each row also has a visible prefix
 * followed by -inf, with the prefix length varying per row and dipping below
 * top_k, so the test covers rows with fewer live candidates than the requested
 * k as well as full rows.
 *
 * USAGE
 *
 *   make test-topk-stream512
 *   ./tests/test_topk_stream512
 *
 *   NC=<n_comp>    compressed rows per token   (default 16384)
 *   NT=<n_tokens>  tokens per batch            (default 64; must be >= 32 for
 *                  the selector to engage, so a lower value tests the
 *                  fallback against the CPU reference only)
 *   VIS=<n>        pin the visible prefix for every row instead of varying it
 *   FINE=1         fine-grained scores, few ties
 *
 * Requires a Metal device.  Needs no model weights.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } \
} while (0)

static const float *g_row;

/* The total order the selector implements: score descending, index ascending.
 * Index breaks ties so the expected output is unique even with duplicate
 * scores -- without that the comparison would be ambiguous rather than wrong. */
static int cmp_desc_idx(const void *x, const void *y) {
    const uint32_t ix = *(const uint32_t *)x, iy = *(const uint32_t *)y;
    if (g_row[ix] > g_row[iy]) return -1;
    if (g_row[ix] < g_row[iy]) return 1;
    return ix < iy ? -1 : 1;
}

static uint32_t env_u32(const char *name, uint32_t dflt) {
    const char *v = getenv(name);
    return (v && v[0]) ? (uint32_t)strtoul(v, NULL, 10) : dflt;
}

int main(void) {
    CHECK(ds4_gpu_init(), "gpu init");

    const uint32_t n_comp = env_u32("NC", 16384u);
    const uint32_t n_tokens = env_u32("NT", 64u);
    const uint32_t top_k = 512u;
    const size_t n_scores = (size_t)n_comp * n_tokens;
    const size_t n_sel = (size_t)top_k * n_tokens;

    float *scores_host = malloc(n_scores * sizeof(float));
    int32_t *sel_stream = malloc(n_sel * sizeof(int32_t));
    int32_t *sel_argsort = malloc(n_sel * sizeof(int32_t));
    uint32_t *order = malloc((size_t)n_comp * sizeof(uint32_t));
    CHECK(scores_host && sel_stream && sel_argsort && order, "host alloc");

    srand(99);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t visible = getenv("VIS") ? env_u32("VIS", n_comp)
                                               : 300u + t * 251u;
        for (uint32_t c = 0; c < n_comp; c++) {
            scores_host[(size_t)t * n_comp + c] = c < visible
                ? (float)(rand() % (getenv("FINE") ? 1000000 : 97)) / 8.0f
                : -INFINITY;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(n_scores * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(n_sel * sizeof(int32_t));
    CHECK(scores && sel, "gpu alloc");
    CHECK(ds4_gpu_tensor_write(scores, 0, scores_host, n_scores * sizeof(float)),
          "write scores");

    /* Reference arm first: the selector is default-on, so the fallback has to
     * be asked for explicitly.  Getting this backwards is how an earlier
     * version of this test passed vacuously -- both arms took the selector and
     * it compared a path against itself. */
    setenv("DS4_METAL_DISABLE_TOPK_STREAM512", "1", 1);
    CHECK(ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k),
          "argsort fallback topk");
    CHECK(ds4_gpu_tensor_read(sel, 0, sel_argsort, n_sel * sizeof(int32_t)),
          "read fallback");

    unsetenv("DS4_METAL_DISABLE_TOPK_STREAM512");
    CHECK(ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k),
          "stream512 topk");
    CHECK(ds4_gpu_tensor_read(sel, 0, sel_stream, n_sel * sizeof(int32_t)),
          "read stream512");

    uint64_t diff_ab = 0, diff_cpu_stream = 0, diff_cpu_argsort = 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        g_row = scores_host + (size_t)t * n_comp;
        for (uint32_t c = 0; c < n_comp; c++) order[c] = c;
        qsort(order, n_comp, sizeof(order[0]), cmp_desc_idx);

        const int32_t *st = sel_stream + (size_t)t * top_k;
        const int32_t *as = sel_argsort + (size_t)t * top_k;
        for (uint32_t k = 0; k < top_k; k++) {
            /* A row with fewer visible candidates than top_k has no defined
             * expectation past the live prefix, so compare only entries whose
             * reference score is finite. */
            if (!isfinite(g_row[order[k]])) break;
            if (st[k] != as[k]) diff_ab++;
            if ((uint32_t)st[k] != order[k]) diff_cpu_stream++;
            if ((uint32_t)as[k] != order[k]) diff_cpu_argsort++;
        }
    }

    printf("n_comp=%u n_tokens=%u top_k=%u%s\n", n_comp, n_tokens, top_k,
           getenv("FINE") ? " (fine scores, few ties)" : " (coarse scores, many ties)");
    printf("  stream512 vs argsort fallback : %llu differing indices\n",
           (unsigned long long)diff_ab);
    printf("  stream512 vs CPU reference    : %llu differing indices\n",
           (unsigned long long)diff_cpu_stream);
    printf("  fallback  vs CPU reference    : %llu differing indices\n",
           (unsigned long long)diff_cpu_argsort);

    free(scores_host); free(sel_stream); free(sel_argsort); free(order);

    /* Every one of these is fatal.  An earlier version printed the CPU-reference
     * count and returned success regardless, which made the only arm that can
     * catch a shared error advisory. */
    CHECK(diff_ab == 0, "stream512 and the argsort fallback disagree");
    CHECK(diff_cpu_stream == 0, "stream512 disagrees with the CPU reference");
    CHECK(diff_cpu_argsort == 0, "argsort fallback disagrees with the CPU reference");
    printf("PASSED\n");
    return 0;
}
