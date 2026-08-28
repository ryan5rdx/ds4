/* Unit test for ds4_gpu_ts_fair_share -- the interval decomposition behind the
 * `fair` column of DS4_METAL_GPU_ENCODER_TIMESTAMPS.
 *
 * Why this exists.  The column it replaces divided every encoder span by one
 * cb-wide factor, i.e. assumed all encoders overlap equally.  Arm B4 caught that
 * assumption failing only because the resulting figure made a sub-phase cost more
 * than the stage containing it -- 27 `reduce` calls x 159.5 us = 4.31 ms against
 * an `attn_inv_rope` of 3.64 ms.  A budget error that can only be detected by a
 * downstream impossibility is one that will be believed the next time the
 * impossibility is less obvious, so the replacement gets checked against
 * hand-computed interval sets instead.
 *
 * Runs on the CPU, needs no GPU and no model.
 *
 *   make tests/test_ts_fairshare && ./tests/test_ts_fairshare
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static int g_fail;

#define EPS 1e-6

static void check(const char *what, double got, double want) {
    if (fabs(got - want) > EPS) {
        printf("  FAIL %-44s got %.6f want %.6f\n", what, got, want);
        g_fail++;
    } else {
        printf("  ok   %-44s %.6f\n", what, got);
    }
}

static void check_u32(const char *what, uint32_t got, uint32_t want) {
    if (got != want) {
        printf("  FAIL %-44s got %u want %u\n", what, got, want);
        g_fail++;
    } else {
        printf("  ok   %-44s %u\n", what, got);
    }
}

/* spt = 1e-6 s/tick makes 1 tick == 1 us, so every expected value below is just
 * the tick arithmetic and can be read off the diagram in the comment. */
#define SPT 1e-6

static void run(const char *name, const uint64_t *s, const uint64_t *e, uint32_t n,
                double *fair, double *uni, double *conc, uint32_t *peak) {
    printf("%s\n", name);
    ds4_gpu_ts_fair_share(s, e, n, SPT, fair, uni, conc, peak);
}

int main(void) {
    double fair[8], uni, conc;
    uint32_t peak;

    /* 1. Disjoint spans -- no overlap, so fair == raw and conc == 1.
     *      A: [0,10)      B: [20,30)
     *      union 20, gap between them is NOT counted. */
    {
        const uint64_t s[] = {0, 20}, e[] = {10, 30};
        run("disjoint [0,10) [20,30)", s, e, 2, fair, &uni, &conc, &peak);
        check("fair[A]", fair[0], 10.0);
        check("fair[B]", fair[1], 10.0);
        check("union", uni, 20.0);
        check("conc mean", conc, 1.0);
        check_u32("conc max", peak, 1);
    }

    /* 2. Exactly coincident spans -- the pathological case the uniform divisor
     *    gets right by luck and everything else gets wrong.
     *      A: [0,10)  B: [0,10)   union 10, each credited 5. */
    {
        const uint64_t s[] = {0, 0}, e[] = {10, 10};
        run("coincident [0,10) x2", s, e, 2, fair, &uni, &conc, &peak);
        check("fair[A]", fair[0], 5.0);
        check("fair[B]", fair[1], 5.0);
        check("union", uni, 10.0);
        check("conc mean", conc, 2.0);
        check_u32("conc max", peak, 2);
    }

    /* 3. Partial overlap -- the real shape, and the one the uniform divisor
     *    misprices.  A: [0,10)  B: [6,20)
     *      [0,6)   A alone   -> A += 6
     *      [6,10)  both      -> A += 2, B += 2
     *      [10,20) B alone   -> B += 10
     *    fair A = 8, B = 12, union = 20, sum(fair) == union.
     *    Note the uniform divisor would have given A 10*(20/24)=8.33 and
     *    B 14*(20/24)=11.67 -- wrong in both directions. */
    {
        const uint64_t s[] = {0, 6}, e[] = {10, 20};
        run("partial [0,10) [6,20)", s, e, 2, fair, &uni, &conc, &peak);
        check("fair[A]", fair[0], 8.0);
        check("fair[B]", fair[1], 12.0);
        check("sum(fair) == union", fair[0] + fair[1], 20.0);
        check("union", uni, 20.0);
        check("conc mean", conc, 24.0 / 20.0);   /* raw sum 24 over union 20 */
        check_u32("conc max", peak, 2);
    }

    /* 4. One long span with two short ones nested inside it -- the decode shape:
     *    a big flash-attn encoder with small neighbours pipelined against it.
     *      A: [0,100)   B: [10,20)   C: [10,20)
     *      [0,10)    A          -> A += 10
     *      [10,20)   A,B,C      -> each += 10/3
     *      [20,100)  A          -> A += 80
     *    fair A = 90 + 10/3, B = C = 10/3, union = 100. */
    {
        const uint64_t s[] = {0, 10, 10}, e[] = {100, 20, 20};
        run("nested [0,100) + [10,20) x2", s, e, 3, fair, &uni, &conc, &peak);
        check("fair[A]", fair[0], 90.0 + 10.0 / 3.0);
        check("fair[B]", fair[1], 10.0 / 3.0);
        check("fair[C]", fair[2], 10.0 / 3.0);
        check("sum(fair) == union", fair[0] + fair[1] + fair[2], 100.0);
        check("union", uni, 100.0);
        check_u32("conc max", peak, 3);
        /* The point of the test: A really costs ~93 of the 100, but a uniform
         * divide by coverage (raw sum 120 over union 100) would bill A only
         * 100*(100/120) = 83.3 and each short span 8.3 -- overstating the cheap
         * encoders by 2.5x and understating the expensive one. */
        check("uniform divisor would misprice A", 100.0 * (100.0 / 120.0), 83.333333);
    }

    /* 5. Touching but not overlapping -- [0,10) and [10,20) must read as
     *    concurrency 1, not 2.  This is why the comparator closes before it
     *    opens on a tie. */
    {
        const uint64_t s[] = {0, 10}, e[] = {10, 20};
        run("touching [0,10) [10,20)", s, e, 2, fair, &uni, &conc, &peak);
        check("fair[A]", fair[0], 10.0);
        check("fair[B]", fair[1], 10.0);
        check("union", uni, 20.0);
        check("conc mean", conc, 1.0);
        check_u32("conc max", peak, 1);
    }

    /* 6. Degenerate inputs must not crash or contribute: an empty span and a
     *    reversed one are both dropped.  This mirrors the `b <= a` guard the
     *    report has always had for slots a command buffer never wrote. */
    {
        const uint64_t s[] = {0, 5, 30}, e[] = {10, 5, 20};   /* B empty, C reversed */
        run("degenerate: empty and reversed spans", s, e, 3, fair, &uni, &conc, &peak);
        check("fair[A]", fair[0], 10.0);
        check("fair[B] (empty)", fair[1], 0.0);
        check("fair[C] (reversed)", fair[2], 0.0);
        check("union", uni, 10.0);
        check_u32("conc max", peak, 1);
    }

    /* 7. n == 0 and NULL must be safe -- the report calls this before it knows
     *    whether resolveCounterRange returned anything. */
    {
        run("n=0", NULL, NULL, 0, NULL, &uni, &conc, &peak);
        check("union", uni, 0.0);
        check("conc mean", conc, 0.0);
        check_u32("conc max", peak, 0);
    }

    /* 8. The invariant that makes the column trustworthy, on a messy set:
     *    sum(fair) == union, always, for any arrangement. */
    {
        const uint64_t s[] = {0, 3, 3, 7, 50, 51}, e[] = {10, 4, 30, 8, 60, 55};
        double sum = 0.0;
        run("invariant sweep, 6 messy spans", s, e, 6, fair, &uni, &conc, &peak);
        for (int i = 0; i < 6; i++) sum += fair[i];
        check("sum(fair) == union", sum, uni);
        /* union by hand: [0,30) from A/C plus [50,60) from E  = 30 + 10 = 40 */
        check("union", uni, 40.0);
        check_u32("conc max", peak, 3);          /* [3,4): A, B, C */
    }

    printf("\n%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED",
           g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
