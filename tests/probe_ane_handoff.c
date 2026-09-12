/*
 * Fast-handoff protocol poison test.
 *
 * The FAST path replaces two command-buffer round trips per layer with a pair
 * of system-coherent words: the GPU publishes READY after packing, a CPU
 * thread spins on it and produces the output, stores DONE, and a GPU fence
 * spins on DONE before unpacking. Nothing is committed in between.
 *
 * Every failure mode of that arrangement is a STALE READ, and a stale read
 * does not crash -- it produces a complete, plausible tensor from the previous
 * iteration. A hundred iterations will not find it; the window is small and
 * only opens when the GPU and CPU happen to interleave the wrong way. So this
 * runs ten thousand, with a nonce that changes every iteration, and checks
 * that what comes out is THIS iteration's value rather than the last one's.
 *
 * This deliberately does not involve Core ML. It tests the protocol, and the
 * protocol is what is new; substituting a CPU transform for the prediction
 * makes the rendezvous tighter, not looser, because the producer finishes
 * sooner and widens the window in which the GPU could run ahead.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include "ds4_gpu.h"

int ds4_gpu_begin_commands(void);
int ds4_gpu_end_commands(void);

#define DIM  256u
#define NTOK  64u
#define ITERS 10000u

static volatile uint32_t *g_w;
static void *g_in, *g_out;
static volatile int g_stop;
static volatile uint32_t g_served;
static uint32_t g_ahead;   /* handoffs served by the run-ahead pass */

/* Stands in for the sidecar thread: wait for READY, transform in -> out,
 * release DONE. */
static void *producer(void *ud) {
    (void)ud;
    uint32_t last = 0;
    while (!g_stop) {
        uint32_t seq = __atomic_load_n(&g_w[0], __ATOMIC_ACQUIRE);
        if (seq == last || seq == 0u) continue;
        const uint16_t *src = (const uint16_t *)g_in;
        uint16_t *dst = (uint16_t *)g_out;
        for (uint32_t i = 0; i < DIM * NTOK; ++i) {
            _Float16 v;
            __builtin_memcpy(&v, &src[i], 2);
            v = (_Float16)((float)v + 1.0f);
            __builtin_memcpy(&dst[i], &v, 2);
        }
        last = seq;
        g_served++;
        __atomic_store_n(&g_w[1], seq, __ATOMIC_RELEASE);
    }
    return NULL;
}

int main(void) {
    if (!ds4_gpu_init()) { puts("VOID: no GPU"); return 1; }
    if (!ds4_gpu_ane_stage_alloc(DIM, NTOK, &g_in, &g_out)) {
        puts("VOID: stage alloc"); return 1;
    }
    g_w = ds4_gpu_ane_sync_words();
    if (!g_w) { puts("VOID: no sync words"); return 1; }

    ds4_gpu_tensor *src = ds4_gpu_tensor_alloc((uint64_t)DIM * NTOK * sizeof(float));
    ds4_gpu_tensor *dst = ds4_gpu_tensor_alloc((uint64_t)DIM * NTOK * sizeof(float));
    if (!src || !dst) { puts("VOID: tensors"); return 1; }
    float *sp = ds4_gpu_tensor_contents(src), *dp = ds4_gpu_tensor_contents(dst);

    uint32_t seq = 0;
    pthread_t th;
    if (pthread_create(&th, NULL, producer, NULL) != 0) { puts("VOID: thread"); return 1; }

    /* PASS 1: all 42 handoffs encoded before ONE completion.
     *
     * This is the case the per-iteration loop below cannot reach, and it is
     * the case that found the single-slot bug: the host encodes far ahead of
     * what the GPU has executed, so with one (seq, model) slot layer N+1's
     * request overwrote layer N's before the sidecar saw READY(N). A ring plus
     * in-order drain is what makes this correct, and correctness here rests on
     * the GPU being unable to pack N+1 before it has fenced on DONE(N) --
     * encoder boundaries, program order, one command buffer. */
    {
        const uint32_t L = 42;
        g_ahead = L;
        ds4_gpu_begin_commands();
        for (uint32_t l = 0; l < L; ++l) {
            for (uint32_t i = 0; i < DIM * NTOK; ++i) sp[i] = (float)l * 0.5f - 8.0f;
            seq++;
            ds4_gpu_ane_order_boundary();
            if (!ds4_gpu_ane_pack(src, DIM, NTOK))      { puts("VOID: pack/ahead");    return 1; }
            if (!ds4_gpu_ane_publish_ready(seq))        { puts("VOID: publish/ahead"); return 1; }
            if (!ds4_gpu_ane_fence_done(seq))           { puts("VOID: fence/ahead");   return 1; }
            if (!ds4_gpu_ane_unpack(dst, DIM, NTOK, 0)) { puts("VOID: unpack/ahead");  return 1; }
        }
        ds4_gpu_end_commands();
        /* Only the LAST layer's output survives in dst, but every layer had to
         * be served in order for the GPU to have got there at all: a missed
         * request stalls its fence until timeout. */
        int bad = 0;
        const float want = (float)(L - 1u) * 0.5f - 8.0f + 1.0f;
        for (uint32_t i = 0; i < DIM * NTOK; ++i)
            if (fabsf(dp[i] - want) > 1e-3f) bad++;
        printf("run-ahead pass: %u handoffs encoded before one completion, "
               "served %u, %s\n", L, g_served,
               (bad || ds4_gpu_ane_sync_timed_out()) ? "FAIL" : "ok");
        if (bad || ds4_gpu_ane_sync_timed_out()) {
            printf("  %d/%u wrong, timeout=%d -- a request was dropped or "
                   "reordered\n", bad, DIM * NTOK, ds4_gpu_ane_sync_timed_out());
            g_stop = 1; pthread_join(th, NULL);
            return 1;
        }
    }

    uint32_t stale = 0, wrong = 0;
    for (uint32_t it = 0; it < ITERS; ++it) {
        /* A nonce that is exact in f16 and changes every iteration, so a
         * result carried over from the previous one is unmistakable. */
        const float nonce = (float)(it % 512u) * 0.125f - 32.0f;
        for (uint32_t i = 0; i < DIM * NTOK; ++i) sp[i] = nonce + (float)(i % 5) * 0.25f;
        for (uint32_t i = 0; i < DIM * NTOK; ++i) dp[i] = -9999.0f;

        ds4_gpu_begin_commands();
        seq++;
        ds4_gpu_ane_order_boundary();
        if (!ds4_gpu_ane_pack(src, DIM, NTOK))      { puts("VOID: pack");    return 1; }
        if (!ds4_gpu_ane_publish_ready(seq))        { puts("VOID: publish"); return 1; }
        if (!ds4_gpu_ane_fence_done(seq))           { puts("VOID: fence");   return 1; }
        if (!ds4_gpu_ane_unpack(dst, DIM, NTOK, 0)) { puts("VOID: unpack");  return 1; }
        ds4_gpu_end_commands();

        if (ds4_gpu_ane_sync_timed_out()) {
            printf("FAIL: fence timed out at iteration %u\n", it);
            g_stop = 1; pthread_join(th, NULL);
            return 1;
        }
        const float want_prev = (float)((it ? it - 1u : 0u) % 512u) * 0.125f - 32.0f;
        int bad = 0, looks_prev = 0;
        for (uint32_t i = 0; i < DIM * NTOK; ++i) {
            const float want = nonce + (float)(i % 5) * 0.25f + 1.0f;
            if (fabsf(dp[i] - want) > 1e-3f) {
                bad++;
                if (it && fabsf(dp[i] - (want_prev + (float)(i % 5) * 0.25f + 1.0f)) < 1e-3f)
                    looks_prev++;
            }
        }
        if (bad) { wrong++; if (looks_prev > (int)(DIM * NTOK) / 2) stale++; }
    }
    g_stop = 1; pthread_join(th, NULL);

    printf("\niterations      %u\n", ITERS);
    printf("producer served %u (%u run-ahead + %u loop)\n",
           g_served, g_ahead, ITERS);
    printf("wrong results   %u\n", wrong);
    printf("  of which stale (previous iteration's value) %u\n", stale);
    printf("fence timeouts  %d\n", ds4_gpu_ane_sync_timed_out());
    const int ok = !wrong && g_served == ITERS + g_ahead;
    printf("\n%s\n", ok ? "PASS: race-free over 10k iterations AND 42-deep run-ahead"
                        : "FAIL: the handoff drops or reorders");
    return ok ? 0 : 1;
}

/* Stubs: ds4_metal.o references these from ds4.c, which this probe omits. */
int ds4_log_is_tty(void) { return 0; }
int ds4_deepseek4_attention_bounds(void *a, void *b, void *c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 0;
}
