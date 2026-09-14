/*
 * TOP1-GPU correctness gate: does the packed-key GPU top-1 pick the SAME token
 * as the CPU sampler, on every shape and value class the campaign brief names?
 *
 * "Same token", not "same score". The compact TP protocol sends one 8-byte key
 * instead of a half-logit vector, so a single disagreeing tie-break changes a
 * generated token with nothing downstream able to notice. Exactness is the
 * whole gate; timing is meaningless until it passes.
 *
 * The reference is a transcription of argmax_f32_unrolled8_range()'s rule --
 * strict `>`, first index wins -- not a re-derivation. Three value classes are
 * where the bit trick and the CPU disagree unless the packing corrects for it,
 * so they are tested explicitly rather than hoped for:
 *
 *   +0/-0   the CPU keeps the first; raw bits rank +0 above -0
 *   NaN     the CPU never selects one; raw bits make NaN the largest key
 *   ties    the CPU keeps the lowest index
 *
 * No GGUF and no model: this is a reducer over a float row.
 *
 *   probe_top1 [--impl 0|1] [--groups N] [--shards N]
 *   probe_top1 --sweep          the U64TOP1-CAP timing sweep
 *
 * The first version of this was exactness-only, with no clocks anywhere, and
 * the campaign gate is a TIMING claim -- "the atomic beats the two-pass reducer
 * INCLUDING reset and merge". So the rig could confirm the capability and prove
 * exactness and then not run the sweep it had been sent to run. --sweep closes
 * that.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "ds4_gpu.h"

int ds4_log_is_tty(void) { return 0; }
int ds4_deepseek4_attention_bounds(void *a, void *b, void *c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 0;
}

/* The CPU rule, transcribed. */
static uint32_t cpu_argmax(const float *v, uint32_t n) {
    uint32_t best = 0;
    float best_v = -INFINITY;
    for (uint32_t i = 0; i < n; i++) if (v[i] > best_v) { best_v = v[i]; best = i; }
    return best;
}

static uint32_t key_to_idx(uint64_t k) { return 0xffffffffu - (uint32_t)k; }

static uint32_t rng_state = 2463534242u;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5; return rng_state;
}

static int failures;

static void check(const char *what, uint32_t n_cols, uint32_t base,
                  const float *row, uint32_t got, uint32_t rows_done) {
    const uint32_t want = cpu_argmax(row, n_cols) + base;
    if (got == want) return;
    failures++;
    /* The reported index must be RANGE-CHECKED before it is dereferenced. A
     * wrong key decodes to an arbitrary index -- inverting the index term makes
     * it 0xffffffff -- and printing row[got - base] then segfaults inside the
     * diagnostic, so the probe detects the defect and dies before saying so.
     * That is how the tie-break negative control first appeared to pass. */
    const int in_range = (got >= base) && (got - base) < n_cols;
    printf("  FAIL %-22s n=%-6u base=%-6u row%u: gpu=%u cpu=%u (cpu v=%.9g",
           what, n_cols, base, rows_done, got, want, (double)row[want - base]);
    if (in_range) printf(", gpu v=%.9g)\n", (double)row[got - base]);
    else          printf(", gpu index OUT OF RANGE)\n");
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

static int cmp_dbl(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* One timed configuration. Every dispatch the implementation makes is inside
 * the measured region -- for the atomic that is reset + atomic + shard merge,
 * because the gate requires it to win with those included and leaving the reset
 * outside is exactly how a primitive gets undeserved credit. */
static double time_top1(ds4_gpu_tensor *out, ds4_gpu_tensor *lg,
                        ds4_gpu_tensor *scr, uint32_t n, uint32_t rows,
                        uint32_t groups, uint32_t shards, int impl,
                        int iters, int cold, double *p50) {
    double *samp = (double *)malloc((size_t)iters * sizeof(double));
    if (!samp) return -1.0;
    /* Warm: the first dispatch pays pipeline binding, which is not what the
     * sweep is comparing. `cold` skips it so the difference is visible. */
    if (!cold) {
        ds4_gpu_top1(out, lg, scr, n, n, 0, rows, groups, shards, impl);
        ds4_gpu_synchronize();
    }
    for (int i = 0; i < iters; i++) {
        const double t0 = now_ms();
        const int ok = ds4_gpu_top1(out, lg, scr, n, n, 0, rows, groups, shards, impl);
        ds4_gpu_synchronize();
        samp[i] = ok ? now_ms() - t0 : -1.0;
    }
    qsort(samp, (size_t)iters, sizeof(double), cmp_dbl);
    *p50 = samp[iters / 2];
    const double best = samp[0];
    free(samp);
    return best;
}

static void sweep(ds4_gpu_tensor *out, ds4_gpu_tensor *lg, ds4_gpu_tensor *scr,
                  float *lp) {
    const uint32_t widths[] = { 77440, 154880 };
    const uint32_t groupset[] = { 16, 32, 64, 128, 256, 512 };
    const uint32_t shardset[] = { 1, 2, 4, 8, 16 };
    const uint32_t rowset[] = { 1, 8 };
    const int iters = 50;

    printf("\n=== U64TOP1-CAP timing sweep ===\n");
    printf("u64_atomic_available=%d  (impl=1 silently falls back when 0, so a\n"
           "flat twopass/u64 comparison there means the fallback, not a tie)\n",
           ds4_gpu_top1_u64_available());
    printf("\n%-7s %-5s %-7s %-7s %-9s %-9s %-9s\n",
           "n_cols", "rows", "groups", "shards", "twopass", "u64", "ratio");

    for (size_t wi = 0; wi < sizeof(widths)/sizeof(*widths); wi++) {
        const uint32_t n = widths[wi];
        for (uint32_t c = 0; c < n * 8; c++)
            lp[c] = (float)((int32_t)(rnd() >> 8) - 8388608) * 1e-4f;
        for (size_t ri = 0; ri < sizeof(rowset)/sizeof(*rowset); ri++) {
            const uint32_t rows = rowset[ri];
            for (size_t gi = 0; gi < sizeof(groupset)/sizeof(*groupset); gi++) {
                const uint32_t g = groupset[gi];
                double p50a = 0;
                time_top1(out, lg, scr, n, rows, g, 1, 0, iters, 0, &p50a);
                for (size_t si = 0; si < sizeof(shardset)/sizeof(*shardset); si++) {
                    const uint32_t sh = shardset[si];
                    double p50b = 0;
                    time_top1(out, lg, scr, n, rows, g, sh, 1, iters, 0, &p50b);
                    printf("%-7u %-5u %-7u %-7u %-9.4f %-9.4f %-9.3f\n",
                           n, rows, g, sh, p50a, p50b,
                           p50b > 0 ? p50a / p50b : 0.0);
                }
            }
        }
    }
    /* Cold is reported separately rather than folded in: a rotating buffer is a
     * different question from steady state and mixing them hides both. */
    double p50c = 0, p50w = 0;
    time_top1(out, lg, scr, 77440, 1, 64, 1, 0, 20, 1, &p50c);
    time_top1(out, lg, scr, 77440, 1, 64, 1, 0, 20, 0, &p50w);
    printf("\ncold-first vs warm (twopass, 77440, groups 64): %.4f / %.4f ms\n",
           p50c, p50w);
    printf("\nGate: the atomic must beat twopass INCLUDING its reset and shard\n"
           "merge, both of which are inside the timed region above.\n");
}

int main(int argc, char **argv) {
    int impl = 0, do_sweep = 0;
    uint32_t groups = 64, shards = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--impl") && i + 1 < argc) impl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--groups") && i + 1 < argc)
            groups = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shards") && i + 1 < argc)
            shards = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sweep")) do_sweep = 1;
    }
    if (!ds4_gpu_init()) { puts("VOID: no GPU"); return 1; }
    printf("impl=%s groups=%u shards=%u  u64_atomic_available=%d\n",
           impl ? "u64-atomic" : "twopass", groups, shards,
           ds4_gpu_top1_u64_available());
    if (impl == 1 && !ds4_gpu_top1_u64_available()) {
        puts("NOTE: the native atomic is unavailable here (expected on Apple7);"
             " ds4_gpu_top1 falls back to two-pass, which is the fail-open"
             " contract. Exactness below still applies.");
    }

    /* 77440 is the production per-rank half of GLM's 154880 vocabulary; the
     * full width is kept as a general reducer stress case. 288 and the odd
     * tails catch grid-stride remainder bugs. */
    const uint32_t widths[] = { 288, 4096, 32768, 77440, 154880,
                                255, 257, 1023, 4097, 77441 };
    const uint32_t rowcounts[] = { 1, 2, 4, 8 };
    /* Nonzero rank offsets: a key must already carry the global id. */
    const uint32_t bases[] = { 0, 77440, 12345 };

    const uint32_t maxw = 154880, maxr = 8;
    ds4_gpu_tensor *lg = ds4_gpu_tensor_alloc((uint64_t)maxw * maxr * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)maxr * sizeof(uint64_t));
    ds4_gpu_tensor *scr = ds4_gpu_tensor_alloc((uint64_t)maxr * 4096 * sizeof(uint64_t));
    if (!lg || !out || !scr) { puts("VOID: tensors"); return 1; }
    float *lp = ds4_gpu_tensor_contents(lg);

    for (size_t wi = 0; wi < sizeof(widths)/sizeof(*widths); wi++) {
        const uint32_t n = widths[wi];
        for (size_t ri = 0; ri < sizeof(rowcounts)/sizeof(*rowcounts); ri++) {
            const uint32_t rows = rowcounts[ri];
            for (size_t bi = 0; bi < sizeof(bases)/sizeof(*bases); bi++) {
                const uint32_t base = bases[bi];
                for (uint32_t r = 0; r < rows; r++) {
                    float *row = lp + (size_t)r * n;
                    for (uint32_t c = 0; c < n; c++)
                        row[c] = (float)((int32_t)(rnd() >> 8) - 8388608) * 1e-4f;
                }
                if (!ds4_gpu_top1(out, lg, scr, n, n, base, rows, groups, shards, impl)) {
                    printf("  FAIL dispatch refused n=%u rows=%u\n", n, rows);
                    failures++; continue;
                }
                ds4_gpu_synchronize();
                const uint64_t *ok = ds4_gpu_tensor_contents(out);
                for (uint32_t r = 0; r < rows; r++)
                    check("random", n, base, lp + (size_t)r * n,
                          key_to_idx(ok[r]), r);
            }
        }
    }

    /* Value classes. One row, production half-width. */
    const uint32_t n = 77440;
    struct { const char *name; void (*fill)(float *, uint32_t); } cases[] = {
        { "all-equal",     NULL }, { "ties-at-boundary", NULL },
        { "plus-minus-0",  NULL }, { "infinities",       NULL },
        { "all-neg-inf",   NULL }, { "nan-mixed",        NULL },
        { "all-nan",       NULL },
    };
    for (size_t ci = 0; ci < sizeof(cases)/sizeof(*cases); ci++) {
        const char *nm = cases[ci].name;
        for (uint32_t c = 0; c < n; c++) lp[c] = -1.0f;
        if (!strcmp(nm, "all-equal")) {
            for (uint32_t c = 0; c < n; c++) lp[c] = 3.5f;
        } else if (!strcmp(nm, "ties-at-boundary")) {
            /* The maximum repeated exactly on threadgroup-slice edges: the CPU
             * keeps the lowest index and so must the packed order. */
            lp[0] = 9.0f; lp[255] = 9.0f; lp[256] = 9.0f;
            lp[n - 1] = 9.0f; lp[n / 2] = 9.0f;
        } else if (!strcmp(nm, "plus-minus-0")) {
            for (uint32_t c = 0; c < n; c++) lp[c] = -2.0f;
            lp[100] = -0.0f; lp[200] = 0.0f;   /* CPU keeps 100 */
        } else if (!strcmp(nm, "infinities")) {
            lp[5] = INFINITY; lp[6] = INFINITY; lp[7] = -INFINITY;
        } else if (!strcmp(nm, "all-neg-inf")) {
            for (uint32_t c = 0; c < n; c++) lp[c] = -INFINITY;
        } else if (!strcmp(nm, "nan-mixed")) {
            lp[10] = NAN; lp[11] = 4.0f; lp[12] = NAN;
        } else if (!strcmp(nm, "all-nan")) {
            for (uint32_t c = 0; c < n; c++) lp[c] = NAN;
        }
        if (!ds4_gpu_top1(out, lg, scr, n, n, 0, 1, groups, shards, impl)) {
            printf("  FAIL dispatch refused (%s)\n", nm); failures++; continue;
        }
        ds4_gpu_synchronize();
        const uint64_t *ok = ds4_gpu_tensor_contents(out);
        check(nm, n, 0, lp, key_to_idx(ok[0]), 0);
    }

    /* Guard zone: the reducer must not read or write past its slice. */
    for (uint32_t c = 0; c < maxw; c++) lp[c] = -5.0f;
    lp[4095] = 7.0f;                      /* the answer, inside  */
    lp[4096] = 99.0f;                     /* a trap, just outside */
    if (ds4_gpu_top1(out, lg, scr, 4096, maxw, 0, 1, groups, shards, impl)) {
        ds4_gpu_synchronize();
        const uint64_t *ok = ds4_gpu_tensor_contents(out);
        const uint32_t got = key_to_idx(ok[0]);
        if (got != 4095) {
            printf("  FAIL guard-zone: gpu=%u expected 4095 "
                   "(read past n_cols into the trap)\n", got);
            failures++;
        }
    }

    if (do_sweep && !failures) sweep(out, lg, scr, lp);
    else if (do_sweep) puts("\nsweep SKIPPED: exactness failed, so timing is meaningless");

    printf("\n%s\n", failures ? "FAIL: the GPU top-1 does not match the CPU sampler"
                              : "PASS: GPU top-1 matches sample_argmax on every "
                                "shape and value class");
    return failures ? 1 : 0;
}
