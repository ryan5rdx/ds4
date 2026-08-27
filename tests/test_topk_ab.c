/* Standalone A/B for the indexer top-k: the canonical argsort path and the
 * stream512 selector must produce byte-identical index lists (same
 * (score desc, idx asc) total order).  Random scores with duplicate values
 * and a -inf tail exercise ties and the visibility pattern. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include "ds4_gpu.h"

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } \
} while (0)

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static const float *g_row;
static int cmp_desc_idx(const void *x, const void *y) {
    const uint32_t ix = *(const uint32_t *)x, iy = *(const uint32_t *)y;
    if (g_row[ix] > g_row[iy]) return -1;
    if (g_row[ix] < g_row[iy]) return 1;
    return ix < iy ? -1 : 1;
}

int main(void) {
    CHECK(ds4_gpu_init(), "init");
    #define ENVU(name, dflt) (getenv(name) ? (uint32_t)strtoul(getenv(name), NULL, 10) : (dflt))
    const uint32_t n_comp = ENVU("NC", 16384u), n_tokens = ENVU("NT", 64u), top_k = 512u;
    const size_t sc = (size_t)n_comp * n_tokens;
    float *hs = malloc(sc * 4);
    int32_t *a = malloc((size_t)top_k * n_tokens * 4);
    int32_t *b = malloc((size_t)top_k * n_tokens * 4);
    CHECK(hs && a && b, "host alloc");
    srand(99);
    for (uint32_t t = 0; t < n_tokens; t++) {
        /* visible prefix grows with t; some tokens have visible < top_k */
        const uint32_t visible = getenv("VIS")
            ? (uint32_t)strtoul(getenv("VIS"), NULL, 10)
            : 300u + t * 251u;
        for (uint32_t c = 0; c < n_comp; c++) {
            /* coarse quantized scores -> plenty of exact ties */
            hs[(size_t)t * n_comp + c] = c < visible
                ? (float)(rand() % (getenv("FINE") ? 1000000 : 97)) / 8.0f
                : -INFINITY;
        }
    }
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(sc * 4);
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((size_t)top_k * n_tokens * 4);
    CHECK(scores && sel, "gpu alloc");
    CHECK(ds4_gpu_tensor_write(scores, 0, hs, sc * 4), "write");

    setenv("DS4_METAL_ARGSORT_CANON", "1", 1);
    unsetenv("DS4_METAL_TOPK_STREAM512");
    /* Reference arm is always the canonical argsort: pin the U9 hierarchical
     * selector off, or at NT < 32 both arms would take it and compare a path
     * against itself. */
    setenv("DS4_METAL_TOPK_PARTITIONS", "0", 1);
    CHECK(ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k), "canon topk");
    CHECK(ds4_gpu_tensor_read(sel, 0, a, (size_t)top_k * n_tokens * 4), "read a");

    setenv("DS4_METAL_TOPK_STREAM512", "1", 1);
    /* NT >= 32 exercises stream512; NT < 32 (the decode shape) exercises the
     * U9 two-level select, which is default-off and so must be asked for. */
    setenv("DS4_METAL_TOPK_PARTITIONS", getenv("PARTS") ? getenv("PARTS") : "8", 1);
    CHECK(ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k), "stream topk");
    CHECK(ds4_gpu_tensor_read(sel, 0, b, (size_t)top_k * n_tokens * 4), "read b");

    /* Timing (ITERS=n): both arms, GPU-side, so the U9 two-level select can be
     * priced against the argsort it replaces at the same shape. */
    if (getenv("ITERS")) {
        const int it = atoi(getenv("ITERS"));
        /* ARM=a|b times a single arm in its own process, so
         * DS4_METAL_GPU_BUSY_PROFILE=1 attributes GPU time to one path.  Wall
         * clock here is dominated by per-call command-buffer round trip
         * (~0.5 ms), which swamps the kernel at this shape. */
        const char *arm = getenv("ARM");
        if (arm && arm[0] == 'b') setenv("DS4_METAL_TOPK_PARTITIONS",
                                         getenv("PARTS") ? getenv("PARTS") : "8", 1);
        if (arm) {
            if (arm[0] == 'a') setenv("DS4_METAL_TOPK_PARTITIONS", "0", 1);
            ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k);
            ds4_gpu_synchronize();
            for (int i = 0; i < it; i++)
                ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k);
            ds4_gpu_synchronize();
            fprintf(stderr, "arm=%s iters=%d done\n", arm, it);
            return 0;
        }
        struct timespec t0, t1;
        /* Interleave the arms and keep the minimum of each.  Sequential
         * back-to-back blocks are not comparable on this box: the same arm
         * measured 56-124 ms of GPU-busy across repeats (thermal drift), which
         * is far wider than the effect.  Alternating shares that drift, and the
         * minimum is the least-interfered sample. */
        {
            const int rounds = it < 20 ? it : 20;
            const int inner = it / rounds > 0 ? it / rounds : 1;
            double best_a = 1e30, best_b = 1e30;
            for (int r = 0; r < rounds; r++) {
                for (int arm2 = 0; arm2 < 2; arm2++) {
                    if (arm2 == 0) setenv("DS4_METAL_TOPK_PARTITIONS", "0", 1);
                    else setenv("DS4_METAL_TOPK_PARTITIONS",
                                getenv("PARTS") ? getenv("PARTS") : "8", 1);
                    ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k);
                    ds4_gpu_synchronize();
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                    for (int i = 0; i < inner; i++)
                        ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k);
                    ds4_gpu_synchronize();
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    const double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 +
                                       (t1.tv_nsec - t0.tv_nsec) / 1e6) / inner;
                    if (arm2 == 0) { if (ms < best_a) best_a = ms; }
                    else           { if (ms < best_b) best_b = ms; }
                }
            }
            fprintf(stderr,
                    "interleaved best-of-%d n_comp=%u: argsort %.4f ms  new %.4f ms  %.2fx\n",
                    rounds, n_comp, best_a, best_b, best_a / best_b);
        }
        setenv("DS4_METAL_TOPK_PARTITIONS", "0", 1);
        ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k);
        ds4_gpu_synchronize();
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < it; i++)
            ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k);
        ds4_gpu_synchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms_a = ((t1.tv_sec - t0.tv_sec) * 1e3 +
                             (t1.tv_nsec - t0.tv_nsec) / 1e6) / it;
        unsetenv("DS4_METAL_TOPK_PARTITIONS");
        ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k);
        ds4_gpu_synchronize();
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < it; i++)
            ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens, top_k);
        ds4_gpu_synchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms_b = ((t1.tv_sec - t0.tv_sec) * 1e3 +
                             (t1.tv_nsec - t0.tv_nsec) / 1e6) / it;
        fprintf(stderr,
                "timing n_comp=%u n_tokens=%u: argsort %.4f ms  new %.4f ms  "
                "%.2fx   (x21 layers: %.2f -> %.2f ms/token)\n",
                n_comp, n_tokens, ms_a, ms_b, ms_a / ms_b, ms_a * 21.0, ms_b * 21.0);
    }

    /* CPU ground truth: strict (score desc, idx asc) top-k per row */
    int32_t *ref = malloc((size_t)top_k * n_tokens * 4);
    uint32_t *ord = malloc((size_t)n_comp * 4);
    for (uint32_t t = 0; t < n_tokens; t++) {
        g_row = hs + (size_t)t * n_comp;
        for (uint32_t c = 0; c < n_comp; c++) ord[c] = c;
        qsort(ord, n_comp, 4, cmp_desc_idx);
        for (uint32_t k = 0; k < top_k; k++)
            ref[(size_t)t * top_k + k] = (int32_t)ord[k];
    }
    size_t da = 0, db_ = 0;
    for (size_t i = 0; i < (size_t)top_k * n_tokens; i++) {
        if (a[i] != ref[i]) da++;
        if (b[i] != ref[i]) db_++;
    }
    fprintf(stderr, "vs cpu-ref: canon-argsort diffs=%zu stream512 diffs=%zu\n", da, db_);

    if (getenv("DUMP")) {
        fprintf(stderr, "t0 canon:  ");
        for (int k = 0; k < 8; k++) fprintf(stderr, "%d ", a[k]);
        fprintf(stderr, "\nt0 stream: ");
        for (int k = 0; k < 8; k++) fprintf(stderr, "%d ", b[k]);
        fprintf(stderr, "\nt0 stream scores: ");
        for (int k = 0; k < 8; k++) fprintf(stderr, "%.1f ", b[k] >= 0 && (uint32_t)b[k] < n_comp ? hs[b[k]] : -1.0f);
        fprintf(stderr, "\n");
    }
    size_t diff = 0, first = (size_t)-1;
    for (size_t i = 0; i < (size_t)top_k * n_tokens; i++)
        if (a[i] != b[i]) { if (first == (size_t)-1) first = i; diff++; }
    if (diff) {
        const size_t ft = first / top_k;
        fprintf(stderr, "FAIL: %zu/%zu differ; first@%zu (t=%zu k=%zu) canon=%d stream=%d\n",
                diff, (size_t)top_k * n_tokens, first, ft, first % top_k,
                a[first], b[first]);
        /* set diagnosis for the failing token */
        size_t missing = 0, extra = 0;
        for (uint32_t k = 0; k < top_k; k++) {
            const int32_t want = a[ft * top_k + k];
            int found = 0;
            for (uint32_t j = 0; j < top_k; j++)
                if (b[ft * top_k + j] == want) { found = 1; break; }
            if (!found) {
                if (missing < 4)
                    fprintf(stderr, "  missing from stream: idx=%d score=%f rank=%u\n",
                            want, hs[ft * n_comp + want], k);
                missing++;
            }
        }
        for (uint32_t k = 0; k < top_k; k++) {
            const int32_t got = b[ft * top_k + k];
            int found = 0;
            for (uint32_t j = 0; j < top_k; j++)
                if (a[ft * top_k + j] == got) { found = 1; break; }
            if (!found) {
                if (extra < 4)
                    fprintf(stderr, "  extra in stream: idx=%d score=%f rank=%u\n",
                            got, got >= 0 && (uint32_t)got < n_comp ? hs[ft * n_comp + got] : -999.0f, k);
                extra++;
            }
        }
        fprintf(stderr, "  token %zu: missing=%zu extra=%zu (of %u)\n", ft, missing, extra, top_k);
        /* duplicate detection in the stream list */
        size_t dups = 0;
        for (uint32_t k = 0; k < top_k; k++)
            for (uint32_t j = k + 1; j < top_k; j++)
                if (b[ft * top_k + k] == b[ft * top_k + j]) {
                    if (dups < 5)
                        fprintf(stderr, "  dup: idx=%d at ranks %u and %u\n",
                                b[ft * top_k + k], k, j);
                    dups++;
                }
        fprintf(stderr, "  dups=%zu\n", dups);
        return 1;
    }
    fprintf(stderr, "topk stream512 vs canon: %u x %u identical\n", n_tokens, top_k);
    ds4_gpu_cleanup();
    return 0;
}
