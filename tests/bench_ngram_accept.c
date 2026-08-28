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
 * With today's 5-row verify (V = 108.18 ms, T = 24.260) that is commit > 4.459
 * out of a maximum 5 -- near-perfect acceptance of four drafts every step, i.e.
 * dead.  With the corrected verify floor (~50 ms, once the row split is
 * re-enabled for batch rows) it is commit > 2.06, which is plausible on code.
 * The whole decision is therefore "is the sustained commit above ~2?".
 *
 * ds4_ngram_propose (ds4.c) is a PURE function of token history, so that
 * question is answerable offline, with no GPU and no model.  Doing it offline
 * also keeps the measurement clear of the four known defects in the live
 * speculative path (s->logits never refreshed on exit, drafts[0] pushed without
 * an eval, the min-match bound returning 0 on the ideal periodic case) -- those
 * break the *implementation*, not the *opportunity*, and we want the
 * opportunity measured before anyone spends a week on the implementation.
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
 *     make tests/bench_ngram_accept
 *     ./tests/bench_ngram_accept /tmp/toks.txt
 *     ./tests/bench_ngram_accept /tmp/toks.txt --vt 2.06     # break-even to beat
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
    long commit_hist[MAX_DRAFT_CAP + 2];
} accept_stats;

/* One pass over the trace, running the real cycle: propose at i, commit the
 * longest matching prefix, jump past it. */
static accept_stats simulate(const int *tok, int n, int k, int depth) {
    accept_stats st;
    memset(&st, 0, sizeof st);
    int draft[MAX_DRAFT_CAP];
    int i = 1;                                /* need >=1 token of history */
    while (i < n) {
        /* drafts[0] is the target's own argmax for the position just decoded --
         * free, always correct by construction.  The n-gram continuation starts
         * after it, which is why history includes tok[i-1] and the proposals are
         * compared against tok[i] onward. */
        st.steps++;
        const int want = depth < MAX_DRAFT_CAP ? depth : MAX_DRAFT_CAP;
        const int avail = n - i;
        const int nd = ds4_test_ngram_propose(tok, i, k,
                                              want < avail ? want : avail, draft);
        if (nd > 0) st.offered++;
        int c = 0;
        while (c < nd && tok[i + c] == draft[c]) c++;
        st.commit_hist[c]++;
        st.emitted += 1 + c;                  /* the free argmax plus the commit */
        i += 1 + c;
    }
    return st;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <token-trace|-> [--vt BREAKEVEN] [--k K] [--depth D]\n"
                "  trace: one token id per line (DS4_NGRAM_TRACE=<path> on a rig run)\n"
                "  --vt : verify/decode cost ratio; a cycle wins iff mean commit > this\n",
                argv[0]);
        return 2;
    }
    double vt = 2.06;                         /* corrected ~50 ms verify floor / 24.26 */
    int only_k = 0, only_depth = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--vt") && i + 1 < argc) vt = atof(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) only_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth") && i + 1 < argc) only_depth = atoi(argv[++i]);
    }

    int n = 0;
    int *tok = load_trace(argv[1], &n);
    if (!tok) return 1;
    if (n < 64) {
        fprintf(stderr, "trace has only %d tokens; need a few thousand to mean anything\n", n);
        free(tok);
        return 1;
    }
    printf("trace: %d tokens from %s\n", n, argv[1]);
    printf("break-even mean commit (V/T) = %.3f   [today's 5-row verify = 4.459; "
           "corrected ~50 ms floor = 2.06]\n\n", vt);

    const int ks[] = {2, 3, 4, 5, 6, 8};
    const int depths[] = {2, 3, 4, 5, 8};
    printf("%3s %6s  %8s  %8s  %10s  %9s  %s\n",
           "k", "depth", "offered%", "mean_cmt", "tok/step", "speedup", "verdict");
    double best = -1.0; int best_k = 0, best_d = 0;
    for (size_t a = 0; a < sizeof ks / sizeof *ks; a++) {
        if (only_k && ks[a] != only_k) continue;
        for (size_t b = 0; b < sizeof depths / sizeof *depths; b++) {
            if (only_depth && depths[b] != only_depth) continue;
            const accept_stats st = simulate(tok, n, ks[a], depths[b]);
            if (!st.steps) continue;
            const double mean_commit = (double)(st.emitted - st.steps) / (double)st.steps;
            const double tok_per_step = (double)st.emitted / (double)st.steps;
            /* A step costs T + V = T(1 + vt) and emits tok_per_step tokens, so
             * throughput scales by tok_per_step / (1 + vt) against 1 token per T. */
            const double speedup = tok_per_step / (1.0 + vt);
            if (speedup > best) { best = speedup; best_k = ks[a]; best_d = depths[b]; }
            printf("%3d %6d  %7.1f%%  %8.3f  %10.3f  %8.3fx  %s\n",
                   ks[a], depths[b], 100.0 * (double)st.offered / (double)st.steps,
                   mean_commit, tok_per_step, speedup,
                   speedup > 1.0 ? "WIN" : "loss");
        }
    }
    if (best_k) {
        printf("\nbest: k=%d depth=%d -> %.3fx  (%s)\n", best_k, best_d, best,
               best > 1.0 ? "fundable at this verify cost" : "NOT fundable at this verify cost");
        printf("re-run with --vt to test other verify costs; the decision is whether\n"
               "mean commit clears V/T, so a cheaper verify moves the bar, not the trace.\n");
    }

    /* The distribution matters as much as the mean: a bimodal trace (long runs
     * inside repeated code, nothing elsewhere) behaves very differently under a
     * fixed depth than a uniform one with the same mean. */
    const accept_stats st = simulate(tok, n, best_k ? best_k : 3, best_d ? best_d : 5);
    printf("\ncommit-length distribution at k=%d depth=%d:\n",
           best_k ? best_k : 3, best_d ? best_d : 5);
    for (int c = 0; c <= MAX_DRAFT_CAP; c++) {
        if (!st.commit_hist[c]) continue;
        printf("  commit=%-2d  %8ld  %5.1f%%\n", c, st.commit_hist[c],
               100.0 * (double)st.commit_hist[c] / (double)st.steps);
    }

    free(tok);
    return 0;
}
