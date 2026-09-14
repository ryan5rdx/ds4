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
 *   probe_top1 [--impl 0|1] [--groups N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
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

int main(int argc, char **argv) {
    int impl = 0;
    uint32_t groups = 64;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--impl") && i + 1 < argc) impl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--groups") && i + 1 < argc)
            groups = (uint32_t)atoi(argv[++i]);
    }
    if (!ds4_gpu_init()) { puts("VOID: no GPU"); return 1; }
    printf("impl=%s groups=%u  u64_atomic_available=%d\n",
           impl ? "u64-atomic" : "twopass", groups, ds4_gpu_top1_u64_available());
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
                if (!ds4_gpu_top1(out, lg, scr, n, n, base, rows, groups, 1, impl)) {
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
        if (!ds4_gpu_top1(out, lg, scr, n, n, 0, 1, groups, 1, impl)) {
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
    if (ds4_gpu_top1(out, lg, scr, 4096, maxw, 0, 1, groups, 1, impl)) {
        ds4_gpu_synchronize();
        const uint64_t *ok = ds4_gpu_tensor_contents(out);
        const uint32_t got = key_to_idx(ok[0]);
        if (got != 4095) {
            printf("  FAIL guard-zone: gpu=%u expected 4095 "
                   "(read past n_cols into the trap)\n", got);
            failures++;
        }
    }

    printf("\n%s\n", failures ? "FAIL: the GPU top-1 does not match the CPU sampler"
                              : "PASS: GPU top-1 matches sample_argmax on every "
                                "shape and value class");
    return failures ? 1 : 0;
}
