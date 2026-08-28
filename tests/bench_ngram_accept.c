/* What commit rate would n-gram drafting actually achieve on this workload?
 *
 * WHY THIS EXISTS
 *
 * Speculation is the only route to 50 t/s left in this corpus, and whether it
 * is worth weeks turns on a single number that has never been measured on the
 * rig, on any fixture: the sustained commit length.  A drafted step emits
 * (1 + commit) tokens for (T + V), so it wins iff
 *
 *     commit > V / T
 *
 * -- but only if V were a constant, and it is not.  V(k) = V_fixed + k*V_marg:
 * dense weights are read once for the whole batch and amortize, while the MoE
 * expert union grows ~linearly with rows and does not.  And a cycle that
 * proposes nothing never verifies at all, so V is charged at the OFFER RATE,
 * not on every step.  Both corrections are in the cost model below; the flat
 * --vt path is kept only for comparison with older numbers.
 *
 * ds4_ngram_propose (ds4.c) is a PURE function of token history, so that
 * question is answerable offline, with no GPU and no model.  Offline keeps the
 * measurement clear of the defects in the live speculative CYCLE (s->logits
 * never refreshed on exit, drafts[0] pushed without an eval) -- those break the
 * implementation, not the opportunity.
 *
 * It does NOT sidestep the shipped proposer's own look-ahead bound, because it
 * calls that proposer deliberately: the question is what OUR drafter achieves,
 * and a reimplementation would drift.  An earlier version of this comment
 * claimed otherwise and was wrong.  To separate "is there opportunity" from
 * "does our drafter capture it", the bound would need a second arm with an
 * idealised proposer -- not yet built.
 *
 * WHAT IT SIMULATES
 *
 * The real cycle, exactly: at position i the drafter proposes d[1..n]; the
 * target verifies them in one batch and commits the longest matching prefix c;
 * the cycle restarts at i + 1 + c.  Reporting mean tokens-per-step over that
 * loop -- not a raw per-token hit rate -- is the point, because a per-token
 * rate silently credits acceptances that a real cycle would never have reached
 * (once a draft mismatches, the rest of it is discarded).
 *
 * INPUT: a token-id trace, one integer per line, from a real rig session:
 *     DS4_NGRAM_TRACE=/tmp/toks.txt ./ds4-bench ...      (or ds4-server)
 *
 * The trace is the SESSION CHECKPOINT, prompt included, because that is what
 * production searches -- a decode-only trace under-reports matches.
 *
 *     make tests/bench_ngram_accept
 *     ./tests/bench_ngram_accept /tmp/toks.txt
 *     ./tests/bench_ngram_accept /tmp/toks.txt --vfixed 19.85 --vmarg 6.62
 *     ./tests/bench_ngram_accept /tmp/toks.txt --vt 4.459    # legacy flat model
 *
 * Depth is swept only to 4: production caps the draft at DS4_SPEC_PREFIX_SLOTS
 * (ds4.c:2577), and reporting depth 5 or 8 would rank a config that cannot run.
 */

#define _GNU_SOURCE

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Exposed by ds4.c purely so this harness can reach the real proposer rather
 * than a reimplementation of it -- a copy would drift and would not be
 * evidence about the shipped drafter. */
int ds4_test_ngram_propose(const int *hist, int hist_len, int k,
                           int max_draft, int *out);

#define MAX_DRAFT_CAP 16

static int *load_trace(const char *path, int *out_n) {
    FILE *fp = strcmp(path, "-") ? fopen(path, "re") : stdin;
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    int cap = 1 << 16, n = 0;
    int *v = malloc((size_t)cap * sizeof(*v));
    if (!v) { if (fp != stdin) fclose(fp); return NULL; }
    char line[64];
    while (fgets(line, sizeof line, fp)) {
        char *end = NULL;
        const long t = strtol(line, &end, 10);
        if (end == line) continue;            /* blank or non-numeric: skip */
        if (n == cap) {
            cap *= 2;
            int *nv = realloc(v, (size_t)cap * sizeof(*v));
            if (!nv) { free(v); if (fp != stdin) fclose(fp); return NULL; }
            v = nv;
        }
        v[n++] = (int)t;
    }
    if (fp != stdin) fclose(fp);
    *out_n = n;
    return v;
}

typedef struct {
    long steps;          /* speculative cycles run */
    long offered;        /* cycles where the drafter proposed anything */
    long emitted;        /* tokens emitted across all cycles (1 + commit each) */
    long verified;       /* cycles that actually ran a verify (offered > 0) */
    long commit_hist[MAX_DRAFT_CAP + 2];
} accept_stats;

/* One pass over the trace, running the real cycle: propose at i, commit the
 * longest matching prefix, jump past it. */
/* `skip` is the first position to speculate FROM.  Production runs prefill as one
 * batched sync and only speculates during decode, so counting prompt positions as
 * cycles dilutes both offered% and mean commit -- on a 2048-prompt/4096-gen trace
 * a third of the steps are ones the engine would never have run.  The prompt stays
 * in the array either way: it is history the proposer searches (ds4.c:69856-69862),
 * so trimming the file instead of skipping would remove real matches. */
static accept_stats simulate(const int *tok, int n, int k, int depth, int skip) {
    accept_stats st;
    memset(&st, 0, sizeof st);
    int draft[MAX_DRAFT_CAP];
    int i = skip > 1 ? skip : 1;              /* need >=1 token of history */
    while (i < n) {
        /* drafts[0] is the target's own argmax for the position just decoded --
         * free, always correct by construction.  The n-gram continuation starts
         * after it, which is why history includes tok[i-1] and the proposals are
         * compared against tok[i] onward. */
        /* The live path pushes drafts[0] -- the free argmax, i.e. tok[i] -- onto
         * the probe BEFORE proposing (ds4.c:69812-69819), so history is
         * tok[0..i] inclusive and the proposals are compared against tok[i+1]
         * onward.  Proposing from tok[0..i) instead shifts every match by one
         * position and silently changes the answer. */
        const int hist_len = i + 1;
        const int want = depth < MAX_DRAFT_CAP ? depth : MAX_DRAFT_CAP;
        const int avail = n - hist_len;
        if (avail <= 0) break;
        st.steps++;                           /* only cycles that actually ran */
        const int nd = ds4_test_ngram_propose(tok, hist_len, k,
                                              want < avail ? want : avail, draft);
        int c = 0;
        while (c < nd && tok[hist_len + c] == draft[c]) c++;
        st.commit_hist[c]++;
        st.emitted += 1 + c;                  /* the free argmax plus the commit */
        /* A cycle that proposes nothing never runs a verify (ds4.c:69838 commits
         * the free argmax and returns), so it must not be charged V.  Counting
         * those steps as verified is what made a zero-offer random trace report
         * 0.327x instead of the 1.0x baseline it actually is. */
        if (nd > 0) { st.offered++; st.verified++; }
        i += 1 + c;
    }
    return st;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <token-trace|-> [--vt BREAKEVEN] [--k K] [--depth D]\n"
                "  trace: one token id per line (DS4_NGRAM_TRACE=<path> on a rig run)\n"
                "  --vt : legacy flat verify/decode ratio (prefer --vfixed/--vmarg)\n"
                "  --skip-first N : start measuring at position N; use the prompt length,\n"
                "                   since production never speculates during prefill\n",
                argv[0]);
        return 2;
    }
    /* V is NOT a single ratio.  Dense weights are read once for the whole batch
     * and amortize; the MoE expert union grows ~linearly with rows
     * (256*(1-(250/256)^k): k=1 -> 6.00, k=5 -> 28.63, per-row multiplier
     * 1.000 -> 0.954) and does not.  So V(k) = V_fixed + k*V_marginal, and a
     * single --vt collapses two very different terms into one.  Defaults are
     * the corrected floor: dense 15.9 + gates 2.7 + head 1.25 = 19.85 fixed,
     * MoE 4.44 + attention 2.18 = 6.62 per row. */
    double t_ms = 24.260, v_fixed = 19.85, v_marg = 6.62;
    double vt = 0.0;                          /* legacy single-ratio override */
    int only_k = 0, only_depth = 0, skip_first = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--vt") && i + 1 < argc) vt = atof(argv[++i]);
        else if (!strcmp(argv[i], "--t") && i + 1 < argc) t_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--vfixed") && i + 1 < argc) v_fixed = atof(argv[++i]);
        else if (!strcmp(argv[i], "--vmarg") && i + 1 < argc) v_marg = atof(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) only_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth") && i + 1 < argc) only_depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--skip-first") && i + 1 < argc) skip_first = atoi(argv[++i]);
    }

    int n = 0;
    int *tok = load_trace(argv[1], &n);
    if (!tok) return 1;
    if (n < 64) {
        fprintf(stderr, "trace has only %d tokens; need a few thousand to mean anything\n", n);
        free(tok);
        return 1;
    }
    if (skip_first < 0) skip_first = 0;
    if (skip_first >= n - 64) {
        fprintf(stderr, "--skip-first %d leaves under 64 positions of a %d-token trace\n",
                skip_first, n);
        free(tok);
        return 1;
    }
    printf("trace: %d tokens from %s\n", n, argv[1]);
    if (skip_first) {
        printf("measuring positions %d..%d (%d cycles); the first %d are prompt and stay as "
               "proposer history\n", skip_first, n - 1, n - skip_first, skip_first);
    }
    if (vt > 0.0) {
        printf("cost model: FLAT override, V/T = %.3f (use --vfixed/--vmarg instead)\n\n", vt);
    } else {
        printf("cost model: T = %.2f ms, V(k) = %.2f + k*%.2f ms, charged only on offered cycles\n"
               "  break-even mean commit at depth k is (V(k)/T) * offer_rate\n\n",
               t_ms, v_fixed, v_marg);
    }

    /* Production caps the draft at DS4_SPEC_PREFIX_SLOTS = 4 (ds4.c:2577).
     * Sweeping 5 and 8 reports configurations the engine cannot run, and the
     * previous version happily picked one as "best". */
    const int ks[] = {2, 3, 4, 5, 6, 8};
    const int depths[] = {1, 2, 3, 4};
    printf("%3s %6s  %8s  %8s  %10s  %8s  %9s  %s\n",
           "k", "depth", "offered%", "mean_cmt", "tok/step", "V(k)ms", "speedup", "verdict");
    double best = -1.0; int best_k = 0, best_d = 0;
    for (size_t a = 0; a < sizeof ks / sizeof *ks; a++) {
        if (only_k && ks[a] != only_k) continue;
        for (size_t b = 0; b < sizeof depths / sizeof *depths; b++) {
            if (only_depth && depths[b] != only_depth) continue;
            const accept_stats st = simulate(tok, n, ks[a], depths[b], skip_first);
            if (!st.steps) continue;
            const double mean_commit = (double)(st.emitted - st.steps) / (double)st.steps;
            const double tok_per_step = (double)st.emitted / (double)st.steps;
            /* Cost per step = T, plus V only on the cycles that actually offered
             * a draft, and V itself scales with the rows verified:
             *     V(k) = v_fixed + k * v_marg
             * Averaging V over ALL steps (offered or not) and using one flat
             * ratio were the two things that made a zero-offer trace look like a
             * 3x slowdown instead of the exact baseline it is. */
            const double offer_rate = (double)st.verified / (double)st.steps;
            /* Production verifies 1 + n_proposals rows: drafts[0] rides the batch
             * even though its logits are already known (ds4.c:69853, :69900-69903),
             * which is why the measured verify was a FIVE-row one at depth 4. */
            const double v_step = vt > 0.0
                    ? t_ms * vt                       /* legacy flat override */
                    : v_fixed + (double)(depths[b] + 1) * v_marg;
            const double cost_per_step = t_ms + offer_rate * v_step;
            const double speedup = (tok_per_step * t_ms) / cost_per_step;
            if (speedup > best) { best = speedup; best_k = ks[a]; best_d = depths[b]; }
            printf("%3d %6d  %7.1f%%  %8.3f  %10.3f  %8.2f  %8.3fx  %s\n",
                   ks[a], depths[b], 100.0 * offer_rate,
                   mean_commit, tok_per_step, v_step, speedup,
                   speedup > 1.0 ? "WIN" : "loss");
        }
    }
    if (best_k) {
        printf("\nbest: k=%d depth=%d -> %.3fx  (%s)\n", best_k, best_d, best,
               best > 1.0 ? "fundable at this verify cost" : "NOT fundable at this verify cost");
        printf("the bar is NOT a flat mean commit: V is charged only on cycles that offer,\n"
               "so break-even is (V(k)/T) x offered%%, which at 10%% offer is 0.20, not 2.0.\n"
               "Read the speedup column -- >1.000x is fundable at this verify cost.\n");
    }

    /* The distribution matters as much as the mean: a bimodal trace (long runs
     * inside repeated code, nothing elsewhere) behaves very differently under a
     * fixed depth than a uniform one with the same mean. */
    const accept_stats st = simulate(tok, n, best_k ? best_k : 3, best_d ? best_d : 4, skip_first);
    printf("\ncommit-length distribution at k=%d depth=%d:\n",
           best_k ? best_k : 3, best_d ? best_d : 4);
    for (int c = 0; c <= MAX_DRAFT_CAP; c++) {
        if (!st.commit_hist[c]) continue;
        printf("  commit=%-2d  %8ld  %5.1f%%\n", c, st.commit_hist[c],
               100.0 * (double)st.commit_hist[c] / (double)st.steps);
    }

    free(tok);
    return 0;
}
