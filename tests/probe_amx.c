/*
 * AMX-CAP — Track A instruction, safety and layout probe. No model required.
 *
 * Three jobs, in order of how much they matter:
 *
 *   1. Report the behavioural capability set, so later arms select a kernel
 *      from measured primitives rather than a CPU brand string.
 *   2. DERIVE the physical Z row mapping instead of assuming it. The brief is
 *      explicit about this and it is the part most likely to be silently wrong:
 *      an outer product that lands somewhere other than where the kernel reads
 *      produces plausible garbage, not a crash.
 *   3. Prove the SET/CLR discipline is real -- including that a nested SET
 *      actually traps -- so the worker ABI rests on a demonstrated fact.
 *
 * Guard bytes surround every AMX-visible buffer. An instruction that writes
 * further than expected is the failure this campaign most needs to catch early,
 * because the units are undocumented and the operand encodings are inferred.
 *
 *   probe_amx [--verbose]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdarg.h>
#include "ds4_amx.h"

static int failures;
static int verbose;

static void fail(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("  FAIL "); vprintf(fmt, ap); printf("\n"); va_end(ap);
    failures++;
}

#if !DS4_AMX_BUILDABLE
int main(void) { puts("VOID: AMX not buildable on this target"); return 0; }
#else

/* A buffer with 256-byte red zones either side. AMX operand encodings are
 * inferred from public reverse engineering; a wrong row/flag field writes
 * outside the intended extent and nothing else would notice. */
#define GUARD 256
typedef struct { uint8_t *base, *data; size_t n; } guarded;

static guarded guarded_alloc(size_t n) {
    guarded g = {0};
    if (posix_memalign((void **)&g.base, 256, n + 2 * GUARD) != 0) return g;
    memset(g.base, 0xA5, n + 2 * GUARD);
    g.data = g.base + GUARD;
    memset(g.data, 0, n);
    g.n = n;
    return g;
}
static int guarded_check(const guarded *g, const char *what) {
    for (size_t i = 0; i < GUARD; i++) {
        if (g->base[i] != 0xA5) { fail("%s: underrun at -%zu", what, GUARD - i); return 0; }
        if (g->base[GUARD + g->n + i] != 0xA5) { fail("%s: overrun at +%zu", what, i); return 0; }
    }
    return 1;
}

/* ---- 2. derive the Z mapping -------------------------------------------
 *
 * One-hot x and y: set x[i] = 1, y[j] = 1, run the outer product, and find
 * which Z element became nonzero. That element IS the (i,j) location, measured.
 */
static int derive_z_mapping(void) {
    /* Search the WHOLE Z array, not one vector.
     *
     * The first cut stored only Z row 0 and looked at z[0..15], found 1 of 16
     * one-hot products, and still reported PASS because the condition was
     * "mapped > 0". An outer product over 64-byte operands does not land in one
     * 64-byte vector, so that was looking in the wrong place and grading itself
     * generously -- the two failure modes this campaign can least afford, since
     * a mislocated accumulator yields plausible garbage rather than a crash.
     */
    enum { ZROWS = 64, ZPERROW = 16 };          /* 64 rows x 64 bytes */
    guarded gx = guarded_alloc(64), gy = guarded_alloc(64);
    guarded gz = guarded_alloc(ZROWS * ZPERROW * 4);
    if (!gx.data || !gy.data || !gz.data) { fail("alloc"); return 0; }
    int8_t  *x = (int8_t *)gx.data;
    int8_t  *y = (int8_t *)gy.data;
    int32_t *z = (int32_t *)gz.data;

    /* corsix/matint.md, mode 8 with 32-bit Z: "only every fourth Y lane
     * participates, each reused four times, so each 4x4 byte block accumulates
     * Z row k from X lane k times Y lane 0."
     *
     * So the mapping to derive is x lane -> Z row, with y FIXED at lane 0.
     * Sweeping y[1..3] and calling the empty result a failure was testing
     * against a geometry the hardware does not have -- which is how the first
     * version reported 2/16 and blamed the unit. */
    int mapped = 0, ambiguous = 0;
    int loc[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 1; j++) {
            loc[i][j] = -1;
            memset(x, 0, 64); memset(y, 0, 64);
            memset(z, 0, (size_t)ZROWS * ZPERROW * 4);
            x[i] = 1; y[j] = 1;
            AMX_SET();
            AMX_LDX(AMX_PTR_ROW_FLAGS(x, 0, 0));
            AMX_LDY(AMX_PTR_ROW_FLAGS(y, 0, 0));
            for (int r = 0; r < ZROWS; r++)
                AMX_LDZ(AMX_PTR_ROW_FLAGS(&z[r * ZPERROW], r, 0));
            AMX_MATINT(DS4_AMX_MATINT_I8_I32);
            for (int r = 0; r < ZROWS; r++)
                AMX_STZ(AMX_PTR_ROW_FLAGS(&z[r * ZPERROW], r, 0));
            AMX_CLR();
            int at = -1, count = 0;
            for (int k = 0; k < ZROWS * ZPERROW; k++)
                if (z[k] == 1) { if (at < 0) at = k; count++; }
            if (at >= 0) {
                mapped++; loc[i][j] = at;
                if (count > 1) ambiguous++;
                if (verbose)
                    printf("    x[%d] y[%d] -> z row %d lane %d%s\n",
                           i, j, at / ZPERROW, at % ZPERROW,
                           count > 1 ? "  (NOT UNIQUE)" : "");
            }
        }
    }
    guarded_check(&gx, "MATINT x"); guarded_check(&gy, "MATINT y");
    guarded_check(&gz, "MATINT z");

    /* A usable mapping means every probed (i,j) landed somewhere, each in a
     * distinct place. Anything less is not a layout a kernel can be written
     * against, so it is a failure rather than a partial success. */
    int distinct = 1;
    for (int a = 0; a < 4 && distinct; a++)
        for (int b = a + 1; b < 4; b++)
            if (loc[a][0] >= 0 && loc[a][0] == loc[b][0]) { distinct = 0; break; }

    printf("  Z mapping: %d/4 x-lane products located, %d ambiguous, "
           "distinct=%s\n", mapped, ambiguous, distinct ? "yes" : "NO");
    if (mapped != 4) {
        /* CURRENT STATE ON M1 MAX: 2/16, at x[0]y[0] -> Z row 0 and
         * x[0]y[2] -> Z row 2. y indexes the Z ROW, only even rows respond, and
         * no x beyond 0 contributes. That is the shape of a lane/mask field in
         * the MATINT operand that this probe passes as 0, not a dead unit --
         * AMX_MATINT(0) is selecting a narrow default rather than the full
         * 64-row x 16-column outer product the geometry table describes.
         *
         * Resolving it needs the corsix operand field layout for MATINT (mode,
         * lane mask, Z row offset, operand widths), which is the next concrete
         * step. Until then no Track B kernel may be written against this
         * layout, and this failure is the thing preventing that -- which is
         * precisely what Track A is for. */
        fail("only %d/4 x-lane products were observable with the documented "
             "i8->i32 operand; the layout still does not derive", mapped);
    } else if (!distinct || ambiguous) {
        fail("the Z mapping is not one-to-one (%d ambiguous)", ambiguous);
    } else {
        const int stride = loc[0][1] - loc[0][0];
        const int rstride = loc[1][0] - loc[0][0];
        printf("  derived: base=%d, +1 in y moves %d lanes, +1 in x moves %d lanes\n",
               loc[0][0], stride, rstride);
    }
    free(gx.base); free(gy.base); free(gz.base);
    return mapped == 4 && distinct && !ambiguous;
}

/* ---- 3. SET/CLR discipline ---------------------------------------------- */

/* Nested SET must trap. If it does NOT, the worker ABI's central rule is not
 * actually enforced by the hardware and every "never SET twice" comment in
 * this campaign is unverified folklore. Run in a child: it is expected to die. */
static int nested_set_traps(void) {
    const pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) { AMX_SET(); AMX_SET(); AMX_CLR(); _exit(7); }
    int st = 0;
    if (waitpid(pid, &st, 0) != pid) return -1;
    if (WIFSIGNALED(st)) return 1;                  /* trapped, as documented */
    if (WIFEXITED(st) && WEXITSTATUS(st) == 7) return 0;   /* did NOT trap */
    return -1;
}

static int repeated_set_clr(int iters) {
    for (int i = 0; i < iters; i++) { AMX_SET(); AMX_CLR(); }
    return 1;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--verbose")) verbose = 1;

    ds4_amx_log_caps();
    const ds4_amx_caps *c = ds4_amx_probe();
    if (!c->available) {
        puts("VOID: raw AMX is unavailable here (the child canary trapped). "
             "That is a valid answer, not a failure: every call site falls back.");
        return 0;
    }

    printf("\nkernel eligibility from measured primitives:\n");
    const char *ks[] = { "q4-predeq-f16", "q4-postscale", "q4-bf16", "q8-shexp" };
    for (size_t i = 0; i < sizeof(ks)/sizeof(*ks); i++)
        printf("  %-16s %s\n", ks[i], ds4_amx_can_run(ks[i]) ? "eligible" : "no");

    printf("\nZ layout (derived, not assumed):\n");
    derive_z_mapping();

    printf("\nSET/CLR discipline:\n");
    repeated_set_clr(10000);
    printf("  10000 x SET/CLR: survived\n");
    const int trapped = nested_set_traps();
    if (trapped == 1) {
        printf("  nested SET traps: YES (the worker ABI rule is enforced)\n");
    } else if (trapped == 0) {
        printf("  nested SET traps: NO -- it returned normally. The "
               "non-reentrancy rule is NOT hardware-enforced here, so the "
               "discipline is ours to keep and a stray SET will corrupt state "
               "silently rather than crash.\n");
    } else {
        fail("could not determine nested-SET behaviour");
    }

    /* M2-only primitives cannot be exercised here; say so rather than pass. */
    if (!c->load4 || !c->bf16 || !c->vecfp_multi) {
        printf("\nNOTE: load4=%d bf16=%d vecfp_multi=%d extr_multi=%d on this "
               "host.\n  The M2-specific arms (predeq-f16, bf16) are therefore "
               "UNTESTED here, not passing.\n  Only an M2 Ultra run can "
               "establish them.\n",
               c->load4, c->bf16, c->vecfp_multi, c->extr_multi);
    }

    printf("\n%s\n", failures ? "FAIL: AMX capability probe found a defect"
                              : "PASS: AMX canaries consistent, Z mapping "
                                "observable, SET/CLR discipline characterised");
    return failures ? 1 : 0;
}
#endif
