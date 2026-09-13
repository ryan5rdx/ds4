#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#include "ds4_ane.h"

/*
 * The Core ML half of the shared-expert sidecar. Metal, IOSurface and the
 * command buffer all live in ds4_metal.m; this file sees two base pointers and
 * a set of models, and never links against Metal at all.
 */

/* Provided by ds4_metal.m. */
int  ds4_gpu_ane_fence_stats(uint64_t *iters, uint64_t *execs, uint64_t *hit0,
                             uint64_t *max_iters, double *ns_per_iter);
void ds4_gpu_ane_fence_stats_reset(void);
int  ds4_gpu_ane_stage_alloc(uint32_t dim, uint32_t n_tok,
                             void **in_ptr, void **out_ptr);
void ds4_gpu_ane_stage_free(void);
volatile uint32_t *ds4_gpu_ane_sync_words(void);
int  ds4_gpu_ane_sync_timed_out(void);
int  ds4_gpu_synchronize(void);

#define DS4_ANE_MAX_LAYERS 64

static int       g_mode = -1;
static int       g_ready;
static uint32_t  g_n_layers, g_n_tok, g_dim;
static MLModel  *g_models[DS4_ANE_MAX_LAYERS];
static MLMultiArray *g_in_array, *g_out_array;
static dispatch_queue_t g_queue;
static dispatch_semaphore_t g_done[DS4_ANE_MAX_LAYERS];
static int       g_started[DS4_ANE_MAX_LAYERS];
static uint32_t  g_seq;
static int       g_fast_stop;
static pthread_t g_fast_thread;
static int       g_fast_running;

/* REQUEST RING.
 *
 * The first cut had ONE (seq, model) slot, and the host encodes far ahead of
 * what the GPU has executed -- that is the entire point of FAST, since nothing
 * blocks the encoding thread. So layer N+1's request overwrote layer N's
 * before the sidecar had seen READY(N): the sidecar would then wait for
 * READY(N+1) while the GPU published READY(N), and the GPU's fence on DONE(N)
 * would spin to timeout and consume a stale surface. The 10k test never caught
 * it because it waits for completion every iteration, so the host was never
 * more than one layer ahead.
 *
 * Draining IN ORDER is correct, and the reason is worth stating: within a
 * command buffer the GPU cannot pack layer N+1 before it has fenced on DONE(N)
 * -- pack, publish, fence and unpack are separated by encoder boundaries and
 * execute in program order. So a single staging surface is safe, and the only
 * thing that had to be queued was the host's INTENT. */
#define DS4_ANE_RING 128u
/* `il` rides along because the timeline is keyed by LAYER and the sequence is
 * global: seq counts every prediction of the run, so (seq-1) % n_layers only
 * coincides with il during the first chunk. The consumer must be told which
 * layer it is serving rather than inferring it. */
typedef struct { uint32_t seq; uint32_t il; __unsafe_unretained MLModel *model; } ane_req;
static ane_req   g_ring[DS4_ANE_RING];
static volatile uint64_t g_ring_head, g_ring_tail;
static dispatch_semaphore_t g_ring_sem;
static uint64_t  g_backpressure;
static uint32_t  g_expect_layers;
static uint64_t  g_chunks;
static uint64_t  g_cancelled;
/* Run-level totals. The per-chunk counters are reset every boundary, so the
 * harness needs something that survives to the end to assert on. */
static uint64_t  g_engaged_total, g_skipped_total, g_failed_total, g_short_chunks;

/* ANESCHED1: the per-layer timeline, CPU side.
 *
 * Four numbers per layer answer the scheduling half of the ~16% question:
 *   enq->obs   how long the sidecar waited for the GPU to publish READY
 *   predict    Core ML's own duration in situ
 *   obs->done  everything the sidecar did between seeing READY and releasing
 *   slack      enq->done, i.e. how much of the routed window the sidecar used
 * The GPU half -- whether it then had to wait at all -- comes from the fence
 * spin counters, which the kernel already keeps and which are now calibrated.
 * Off unless DS4_ANE_SCHED_TRACE=1: this is a diagnostic arm, not a feature. */
static int g_sched_trace = -1;
static int ds4_ane_sched_trace(void) {
    if (g_sched_trace < 0) {
        const char *e = getenv("DS4_ANE_SCHED_TRACE");
        g_sched_trace = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return g_sched_trace;
}
typedef struct { double enq, obs, p0, p1, done; int valid; } ane_tl;
static ane_tl g_tl[DS4_ANE_MAX_LAYERS];

/* TEST HOOK (DS4_ANE_TEST_HOOK=1, never set in production).
 *
 * probe_ane_handoff exercised the GPU fence protocol with its own producer
 * thread -- which means it would have passed against the SINGLETON request
 * slot the ring replaced, because it never drove this file at all. The hook
 * lets FAST run with no Core ML models so the real ring, the real ordered
 * drain and the real READY/DONE words are under test, and records the sequence
 * the sidecar actually served so the test can assert it arrived in order with
 * nothing dropped. A singleton fails that assertion; a ring passes it. */
static int      g_test_hook = -1;
#define DS4_ANE_TEST_LOG 256u
static uint32_t g_test_served[DS4_ANE_TEST_LOG];
static uint32_t g_test_n;

static int ds4_ane_test_hook(void) {
    if (g_test_hook < 0) {
        const char *e = getenv("DS4_ANE_TEST_HOOK");
        g_test_hook = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return g_test_hook;
}

uint32_t ds4_ane_test_served(uint32_t *out, uint32_t max) {
    const uint32_t n = g_test_n < max ? g_test_n : max;
    for (uint32_t i = 0; i < n; ++i) out[i] = g_test_served[i];
    return g_test_n;
}

/* Counters. Reported per run rather than per layer: a per-layer print at 42
 * layers x 32 chunks would itself perturb what it measures. */
static double    g_ns_predict;
static uint64_t  g_engaged, g_skipped, g_failed;

static double ds4_ane_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1.0e9 + (double)ts.tv_nsec;
}

int ds4_ane_mode(void) {
    if (g_mode >= 0) return g_mode;
    const char *e = getenv("DS4_METAL_ANE_SHEXP");
    g_mode = DS4_ANE_OFF;
    if (e && e[0]) {
        if      (!strcmp(e, "probe"))  g_mode = DS4_ANE_PROBE;
        else if (!strcmp(e, "fast"))   g_mode = DS4_ANE_FAST;
        else if (!strcmp(e, "fastnull")) g_mode = DS4_ANE_FASTNULL;
        else if (!strcmp(e, "bridge")) g_mode = DS4_ANE_BRIDGE;
        else if (!strcmp(e, "shadow")) g_mode = DS4_ANE_SHADOW;
        else if (!strcmp(e, "0") || !strcmp(e, "off")) g_mode = DS4_ANE_OFF;
        else {
            fprintf(stderr, "ds4: DS4_METAL_ANE_SHEXP=%s unrecognised "
                            "(probe|bridge|shadow|fast|fastnull|off) -- sidecar off\n", e);
        }
    }
    if (g_mode != DS4_ANE_OFF) {
        fprintf(stderr, "ds4: ANE shared-expert sidecar mode=%s\n", e);
    }
    return g_mode;
}

static void *ds4_ane_sidecar_thread(void *ud);
static void ds4_ane_atexit(void);

/* A sidecar that silently declines to run is the most dangerous outcome under
 * TP2, and ANESIDE3 is what that looks like: the models existed only on the
 * coordinator, so the worker's init failed, fell back to the GPU path and ran
 * at full speed while the coordinator stalled ~14 ms per layer in Core ML.
 * Roughly 590 ms of ONE-SIDED skew per chunk against a 750 ms gate budget ends
 * as "big gate window barrier failed" -- which reads as a transport fault and
 * is actually a missing directory on one host.
 *
 * So an explicitly requested sidecar that cannot start is fatal by default.
 * Failing loudly on one rank is recoverable; running on one rank is not. */
static void ds4_ane_required_abort(const char *why) {
    if (getenv("DS4_ANE_OPTIONAL")) {
        fprintf(stderr, "ds4: ANE sidecar unavailable (%s) -- continuing on the "
                        "GPU path because DS4_ANE_OPTIONAL is set. Under TP this "
                        "is only safe if EVERY rank is also without it.\n", why);
        return;
    }
    fprintf(stderr,
            "ds4: FATAL: ANE sidecar was requested but cannot start (%s).\n"
            "ds4:   Refusing to fall back silently: under TP2 a sidecar running\n"
            "ds4:   on one rank and not the other skews the gate window and\n"
            "ds4:   surfaces as 'big gate window barrier failed', which looks\n"
            "ds4:   like a transport fault and is not one.\n"
            "ds4:   Set DS4_ANE_OPTIONAL=1 to allow the fallback.\n", why);
    exit(1);
}

static void ds4_ane_teardown(void) {
    if (g_fast_running) {
        __atomic_store_n(&g_fast_stop, 1, __ATOMIC_RELEASE);
        if (g_ring_sem) dispatch_semaphore_signal(g_ring_sem);
        pthread_join(g_fast_thread, NULL);
        g_fast_running = 0;
    }
    g_ring_sem = nil;
    for (uint32_t i = 0; i < DS4_ANE_MAX_LAYERS; ++i) {
        g_models[i] = nil;
        g_done[i] = nil;
        g_started[i] = 0;
    }
    g_in_array = nil; g_out_array = nil; g_queue = nil;
    ds4_gpu_ane_stage_free();
    g_ready = 0; g_n_layers = 0; g_n_tok = 0; g_dim = 0;
}

/* One MLMultiArray per staging surface, built ONCE. Rebuilding per prediction
 * was worth 2.2x in the rig harness: Core ML will happily allocate a fresh
 * output every call if you let it, and that allocation dominates at this
 * size. */
static MLMultiArray *ds4_ane_wrap(void *base, uint32_t dim, uint32_t n_tok) {
    NSArray<NSNumber *> *shape   = @[@1, @(dim), @1, @(n_tok)];
    NSArray<NSNumber *> *strides = @[@((NSInteger)dim * n_tok),
                                     @((NSInteger)n_tok), @((NSInteger)n_tok), @1];
    NSError *err = nil;
    MLMultiArray *a = [[MLMultiArray alloc] initWithDataPointer:base
                                                          shape:shape
                                                       dataType:MLMultiArrayDataTypeFloat16
                                                        strides:strides
                                                    deallocator:nil
                                                          error:&err];
    if (!a) {
        fprintf(stderr, "ds4: ANE MLMultiArray failed: %s\n",
                [[err localizedDescription] UTF8String]);
    }
    return a;
}

int ds4_ane_init(uint32_t n_layers, uint32_t dim, uint32_t n_tokens) {
    if (ds4_ane_mode() == DS4_ANE_OFF) return 0;
    if (n_layers == 0 || n_layers > DS4_ANE_MAX_LAYERS || n_tokens == 0 || dim == 0)
        return 0;
    g_expect_layers = n_layers;
    if (g_ready && g_n_layers == n_layers && g_n_tok == n_tokens && g_dim == dim)
        return 1;
    ds4_ane_teardown();

    void *in_ptr = NULL, *out_ptr = NULL;
    if (!ds4_gpu_ane_stage_alloc(dim, n_tokens, &in_ptr, &out_ptr)) {
        char why[128];
        snprintf(why, sizeof(why), "staging alloc failed (dim=%u tok=%u)",
                 dim, n_tokens);
        ds4_ane_required_abort(why);
        return 0;
    }

    @autoreleasepool {
        g_in_array  = ds4_ane_wrap(in_ptr,  dim, n_tokens);
        g_out_array = ds4_ane_wrap(out_ptr, dim, n_tokens);
        if (!g_in_array || !g_out_array) { ds4_ane_teardown(); return 0; }

        /* PROBE and BRIDGE deliberately load nothing: they price the fence and
         * the layout conversion, and a Core ML load would contaminate both. */
        /* FASTNULL loads nothing on purpose: it is FAST's seam -- pack, ring,
         * release-word fence, unpack -- with the prediction removed, so
         * fast-null vs off is the fixed cost and fast vs fast-null is Core ML.
         * ANESIDE5B could not separate those and the k2 projection depends
         * entirely on which one the 4.8 ms/layer belongs to. */
        if (ds4_ane_mode() >= DS4_ANE_SHADOW &&
            ds4_ane_mode() != DS4_ANE_FASTNULL && !ds4_ane_test_hook()) {
            const char *dir = getenv("DS4_ANE_MODEL_DIR");
            if (!dir || !dir[0]) {
                ds4_ane_teardown();
                ds4_ane_required_abort("DS4_ANE_MODEL_DIR is unset");
                return 0;
            }
            /* Which graph shape to load. ANEIO3 measured _k2 at -30% against
             * _fused on ANE-side latency with both ALL-ANE, and found _fk2
             * fails on-device compilation outright -- concatenation and
             * K-split are alternatives, not stackable. The default stays
             * _fused because k2 is a change to the ARITHMETIC and ANEIO3's
             * divergence column contradicted itself (see the review), so the
             * numerical case is not yet made. Selectable so the rig can price
             * either without a rebuild. */
            const char *variant = getenv("DS4_ANE_MODEL_VARIANT");
            if (!variant || !variant[0]) variant = "fused";
            MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
            cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            uint32_t loaded = 0;
            for (uint32_t il = 0; il < n_layers; ++il) {
                NSString *path = [NSString stringWithFormat:
                        @"%s/shexp_L%02u_%s.mlpackage", dir, il, variant];
                NSURL *url = [NSURL fileURLWithPath:path];
                NSError *err = nil;
                NSURL *compiled = [MLModel compileModelAtURL:url error:&err];
                if (!compiled) continue;
                g_models[il] = [MLModel modelWithContentsOfURL:compiled
                                                 configuration:cfg
                                                         error:&err];
                if (g_models[il]) loaded++;
            }
            /* All or nothing. A partial load used to mean the missing layers
             * silently fell back to the GPU while their fences waited on a
             * publisher that never came -- and the run still reported a
             * plausible per-prediction cost, measured over the layers that
             * happened to work. */
            if (loaded != n_layers) {
                char why[512];
                snprintf(why, sizeof(why), "loaded %u of %u %s models under %s",
                         loaded, n_layers, variant, dir);
                ds4_ane_teardown();
                ds4_ane_required_abort(why);
                return 0;
            }
            fprintf(stderr, "ds4: ANE READY loaded=%u/%u variant=%s M=%u dir=%s\n",
                    loaded, n_layers, variant, n_tokens, dir);
            fprintf(stderr, "ds4: ANE shadow weights are SYNTHETIC -- the "
                            "divergence below is expected to be large and the "
                            "GPU stays authoritative\n");
        }

        g_queue = dispatch_queue_create("ds4.ane.shexp", DISPATCH_QUEUE_SERIAL);
        if (ds4_ane_mode() >= DS4_ANE_FAST && !g_fast_running) {
            __atomic_store_n(&g_fast_stop, 0, __ATOMIC_RELEASE);
            g_ring_head = g_ring_tail = 0; g_backpressure = 0;
            g_ring_sem = dispatch_semaphore_create(0);
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
            if (pthread_create(&g_fast_thread, &attr, ds4_ane_sidecar_thread, NULL) == 0) {
                g_fast_running = 1;
            } else {
                fprintf(stderr, "ds4: ANE sidecar thread failed to start\n");
                pthread_attr_destroy(&attr);
                ds4_ane_teardown();
                return 0;
            }
            pthread_attr_destroy(&attr);
        }
        for (uint32_t i = 0; i < n_layers; ++i) g_done[i] = dispatch_semaphore_create(0);
    }
    g_n_layers = n_layers; g_n_tok = n_tokens; g_dim = dim; g_ready = 1;
    static int atexit_armed;
    if (!atexit_armed) { atexit_armed = 1; atexit(ds4_ane_atexit); }
    return 1;
}

uint32_t ds4_ane_next_seq(void) { return ++g_seq; }

/* ANESIDE5B emitted ZERO of these lines on either rank, and the reason is that
 * the only caller sat in metal_graph_prefill_chunked_range -- which
 * metal_graph_prefill_raw_swa and ds4_session_eval_layer_slice_impl both
 * bypass by calling metal_graph_prefill_layer_major directly. Adding the call
 * to each of those would work until the next path appears.
 *
 * So the sidecar reports itself. It sees every layer it serves, so it knows
 * when logical layer 0 comes round again, and that IS a chunk boundary no
 * matter which loop above produced it. */
static void ds4_ane_chunk_boundary(void) {
    if (g_engaged || g_skipped || g_failed) { ds4_ane_report(); ds4_ane_reset(); }
}

/* Count COMPLETIONS, not encoder-side layer-0 arrivals.
 *
 * Reporting at the next layer 0 left the final chunk of a run unreported
 * unless the process exited cleanly -- and the harness SIGKILLs the worker, so
 * the worker's last chunk was always lost. It also read counters the sidecar
 * thread was concurrently writing. Now the thread that owns the counters is
 * the one that decides a chunk is done, at every n-th completion, so there is
 * no cross-thread read and no dependence on exit. */
static uint32_t g_completed;
static void ds4_ane_note_completion(void) {
    if (++g_completed < g_expect_layers || g_expect_layers == 0) return;
    g_completed = 0;
    ds4_ane_chunk_boundary();
}

/* The authoritative line, via atexit so it does not depend on a caller either.
 * Drains the ring first: in FAST the sidecar is asynchronous, so a prediction
 * from the final chunk can still be in flight and would otherwise be counted
 * as a shortfall. */
static void ds4_ane_atexit(void) {
    if (ds4_ane_mode() == DS4_ANE_OFF) return;
    if (g_fast_running) {
        const double t0 = ds4_ane_now_ns();
        while (__atomic_load_n(&g_ring_head, __ATOMIC_ACQUIRE) !=
               __atomic_load_n(&g_ring_tail, __ATOMIC_ACQUIRE)) {
            if (ds4_ane_now_ns() - t0 > 5.0e9) break;      /* 5 s, then give up */
            struct timespec ts = { 0, 100000 };
            nanosleep(&ts, NULL);
        }
    }
    /* Let the GPU finish before the final fence read: the counters are written
     * by the fence kernel, and anything still queued would be missed. This is
     * the one place a sync is free, because the run is over. */
    ds4_gpu_synchronize();
    ds4_ane_chunk_boundary();
    {
        uint64_t it = 0, ex = 0, h0 = 0, mx = 0; double nspi = 0;
        if (ds4_ane_sched_trace() &&
            ds4_gpu_ane_fence_stats(&it, &ex, &h0, &mx, &nspi)) {
            fprintf(stderr,
                    "ds4: ANE FENCE TOTAL execs=%llu hit0=%llu iters=%llu "
                    "max=%llu ns_per_iter=%.2f (after GPU sync)\n",
                    (unsigned long long)ex, (unsigned long long)h0,
                    (unsigned long long)it, (unsigned long long)mx, nspi);
        }
    }
    fprintf(stderr,
            "ds4: ANE TOTAL chunks=%llu engaged=%llu expect_per_chunk=%u "
            "shortfall_chunks=%llu skipped=%llu failed=%llu cancelled=%llu "
            "backpressure=%llu timeout=%d\n",
            (unsigned long long)g_chunks, (unsigned long long)g_engaged_total,
            g_expect_layers, (unsigned long long)g_short_chunks,
            (unsigned long long)g_skipped_total, (unsigned long long)g_failed_total,
            (unsigned long long)g_cancelled, (unsigned long long)g_backpressure,
            ds4_gpu_ane_sync_timed_out());
    fflush(stderr);
}

/* The FAST rendezvous, run on a dedicated thread so the encoding thread never
 * blocks. Spin rather than sleep: the wait is a few hundred microseconds at
 * most (the GPU only has to finish the pack), and a condition variable would
 * add a wakeup latency of the same order as the thing being saved. */
static void *ds4_ane_sidecar_thread(void *ud) {
    (void)ud;
    /* USER_INTERACTIVE: this thread gates the GPU -- a fence is spinning on its
     * result -- so letting the scheduler treat it as background work would
     * stall the GPU at E-core latency. */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    volatile uint32_t *w = ds4_gpu_ane_sync_words();
    if (!w) return NULL;
    for (;;) {
        /* Block rather than spin when there is nothing queued. The first cut
         * burned a core continuously whenever the sidecar was idle, which on a
         * machine whose CPU is also running the TP service is not free. */
        if (dispatch_semaphore_wait(g_ring_sem,
                dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC)) != 0) {
            if (__atomic_load_n(&g_fast_stop, __ATOMIC_ACQUIRE)) return NULL;
            continue;
        }
        if (__atomic_load_n(&g_fast_stop, __ATOMIC_ACQUIRE)) return NULL;
        const uint64_t t = __atomic_load_n(&g_ring_tail, __ATOMIC_ACQUIRE);
        const ane_req req = g_ring[t % DS4_ANE_RING];
        const uint32_t seq = req.seq;
        /* >= not ==: the GPU publishes monotonically and may already be past
         * this seq by the time we look. Equality would hang on a skipped one.
         *
         * Requests are queued while the HOST is encoding, often tens of
         * milliseconds before the command buffer reaches READY, so a pure spin
         * holds a performance core for most of the chunk. Spin briefly for the
         * case where the GPU is already there, then back off: a 10-50 us wake
         * is cheap against a multi-millisecond overlap window, and the core is
         * worth more to the TP service than to this loop. */
        uint32_t spins = 0;
        while ((int32_t)(__atomic_load_n(&w[0], __ATOMIC_ACQUIRE) - seq) < 0) {
            if (__atomic_load_n(&g_fast_stop, __ATOMIC_ACQUIRE)) return NULL;
            if (++spins < 2000u) continue;
            struct timespec ts = { 0, 20000 };          /* 20 us */
            nanosleep(&ts, NULL);
        }
        const uint32_t tl_i = req.il < DS4_ANE_MAX_LAYERS ? req.il : 0u;
        if (ds4_ane_sched_trace()) g_tl[tl_i].obs = ds4_ane_now_ns();
        MLModel *m = req.model;
        const double t0 = ds4_ane_now_ns();
        if (m) {
            @autoreleasepool {
                NSError *err = nil;
                MLDictionaryFeatureProvider *inp = [[MLDictionaryFeatureProvider alloc]
                        initWithDictionary:@{ @"x": [MLFeatureValue
                                              featureValueWithMultiArray:g_in_array] }
                                     error:&err];
                MLPredictionOptions *opts = [[MLPredictionOptions alloc] init];
                NSString *on = m.modelDescription.outputDescriptionsByName.allKeys.firstObject;
                if (on) opts.outputBackings = @{ on: g_out_array };
                if (inp && [m predictionFromFeatures:inp options:opts error:&err]) g_engaged++;
                else g_failed++;
            }
        } else if (ds4_ane_test_hook() || ds4_ane_mode() == DS4_ANE_FASTNULL) {
            if (g_test_n < DS4_ANE_TEST_LOG) g_test_served[g_test_n] = seq;
            g_test_n++;
            g_engaged++;
        } else {
            g_skipped++;
        }
        const double t1 = ds4_ane_now_ns();
        g_ns_predict += t1 - t0;
        /* Release-store: everything Core ML wrote to the output surface must be
         * visible to the GPU before it sees DONE.
         *
         * FIRST, before any bookkeeping. The trace used to run ahead of this
         * and delayed the store it was trying to measure -- and `done` was set
         * to p1, so `release` was unconditionally zero and the instrument
         * could not have shown its own cost. DONE, then timestamp, then
         * report. */
        __atomic_store_n(&w[1], seq, __ATOMIC_RELEASE);
        const double tdone = ds4_ane_now_ns();
        if (ds4_ane_sched_trace() && tl_i < DS4_ANE_MAX_LAYERS) {
            g_tl[tl_i].p0 = t0;
            g_tl[tl_i].p1 = t1;
            g_tl[tl_i].done = tdone;
            g_tl[tl_i].valid = 1;
        }
        __atomic_store_n(&g_ring_tail, t + 1u, __ATOMIC_RELEASE);
        ds4_ane_note_completion();
    }
}

int ds4_ane_begin_layer(uint32_t il) {
    if (!g_ready || il >= g_n_layers) return 0;
    if (ds4_ane_mode() < DS4_ANE_SHADOW) return 0;
    MLModel *m = g_models[il];
    if (!m && !ds4_ane_test_hook() && ds4_ane_mode() != DS4_ANE_FASTNULL) {
        g_skipped++; return 0;
    }

    if (ds4_ane_mode() >= DS4_ANE_FAST) {
        /* Enqueue. Backpressure rather than overwrite: the ring holds 128 and
         * a chunk is 42 layers, so this should never spin -- if it does, the
         * host has run further ahead than the design assumed, and that is
         * worth knowing rather than silently dropping a request. */
        uint64_t h = __atomic_load_n(&g_ring_head, __ATOMIC_ACQUIRE);
        while (h - __atomic_load_n(&g_ring_tail, __ATOMIC_ACQUIRE) >= DS4_ANE_RING) {
            g_backpressure++;
            if (__atomic_load_n(&g_fast_stop, __ATOMIC_ACQUIRE)) return 0;
        }
        g_ring[h % DS4_ANE_RING].seq = g_seq;
        g_ring[h % DS4_ANE_RING].model = m;
        g_ring[h % DS4_ANE_RING].il = il;
        if (ds4_ane_sched_trace() && il < DS4_ANE_MAX_LAYERS) {
            g_tl[il].enq = ds4_ane_now_ns();
        }
        __atomic_store_n(&g_ring_head, h + 1u, __ATOMIC_RELEASE);
        /* NO signal here -- see ds4_ane_commit_layer(). */
        g_started[il] = 1;
        return 1;
    }

    g_started[il] = 1;
    MLMultiArray *in = g_in_array, *out = g_out_array;
    dispatch_semaphore_t done = g_done[il];
    /* Asynchronous on purpose: the whole point is for this to run while the
     * GPU is inside routed-MoE. A synchronous call here would measure the ANE
     * and the GPU back to back and show a slowdown where the design predicts
     * an overlap. */
    dispatch_async(g_queue, ^{
        @autoreleasepool {
            const double t0 = ds4_ane_now_ns();
            NSError *err = nil;
            MLDictionaryFeatureProvider *inp = [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{ @"x": [MLFeatureValue
                                                  featureValueWithMultiArray:in] }
                                 error:&err];
            MLPredictionOptions *opts = [[MLPredictionOptions alloc] init];
            NSString *outName = m.modelDescription.outputDescriptionsByName.allKeys.firstObject;
            if (outName) opts.outputBackings = @{ outName: out };
            if (inp && [m predictionFromFeatures:inp options:opts error:&err]) {
                g_engaged++;
            } else {
                /* Counted, never swallowed: a failed prediction returns
                 * instantly and would otherwise read as a very fast ANE. */
                g_failed++;
            }
            g_ns_predict += ds4_ane_now_ns() - t0;
            ds4_ane_note_completion();
        }
        dispatch_semaphore_signal(done);
    });
    return 1;
}

/* Undo the most recent enqueue. Called when the GPU publish that would have
 * released it could not be encoded: the alternative is a sidecar blocked
 * forever on a READY nobody will write. Safe because the producer is single
 * -- only the encoding thread enqueues. */
/* Wake the sidecar for the request enqueued by the last begin_layer. Split
 * from the enqueue because cancel used to rewind g_ring_head AFTER signalling,
 * which races a consumer that may already have taken the slot. With the signal
 * held back until the publish is encoded, a cancel is a pure single-producer
 * rewind and cannot race anything. */
void ds4_ane_commit_layer(void) {
    if (!g_ready || ds4_ane_mode() < DS4_ANE_FAST) return;
    dispatch_semaphore_signal(g_ring_sem);
}

void ds4_ane_cancel_layer(uint32_t il) {
    if (!g_ready || ds4_ane_mode() < DS4_ANE_FAST) return;
    const uint64_t h = __atomic_load_n(&g_ring_head, __ATOMIC_ACQUIRE);
    if (h == __atomic_load_n(&g_ring_tail, __ATOMIC_ACQUIRE)) return;
    __atomic_store_n(&g_ring_head, h - 1u, __ATOMIC_RELEASE);
    g_cancelled++;
    if (il < g_n_layers) g_started[il] = 0;
}

int ds4_ane_wait(uint32_t il) {
    if (!g_ready || il >= g_n_layers || !g_started[il]) return 0;
    if (ds4_ane_mode() >= DS4_ANE_FAST) {
        /* Nothing to wait for on this thread: the GPU fence does it. Waiting
         * here would reintroduce exactly the host round trip FAST removes. */
        g_started[il] = 0;
        return 1;
    }
    dispatch_semaphore_wait(g_done[il], DISPATCH_TIME_FOREVER);
    g_started[il] = 0;
    return 1;
}

void ds4_ane_reset(void) {
    /* Deliberately NOT clearing g_cancelled or the timeout word: those are
     * run-level faults, and a per-chunk reset would let one bad chunk vanish
     * from the record. */
    g_ns_predict = 0.0; g_engaged = g_skipped = g_failed = 0;
}

void ds4_ane_report(void) {
    if (ds4_ane_mode() == DS4_ANE_OFF) return;
    if (g_engaged == 0 && g_skipped == 0 && g_failed == 0) return;
    g_engaged_total += g_engaged;
    g_skipped_total += g_skipped;
    g_failed_total  += g_failed;
    if (g_engaged != g_expect_layers) g_short_chunks++;
    if (ds4_ane_sched_trace()) {
        /* PER LAYER, as promised -- averages hid the shape, and the shape is
         * the question. Buffered and emitted once at the chunk boundary so the
         * I/O is not inside the per-layer path it measures.
         *
         * Fence counters are reported CUMULATIVE and never reset here. The old
         * code read and zeroed them from the 42nd completion, which runs when
         * the CPU stores the last DONE -- before the GPU has necessarily
         * executed the matching fence. That is why n=42 but execs=41, and the
         * reset could land while fence 42 was still running. Cumulative counts
         * have no reset to race; the per-chunk delta is the reader's
         * subtraction, and the atexit total is taken after a GPU sync. */
        uint64_t it = 0, ex = 0, h0 = 0, mx = 0; double nspi = 0;
        ds4_gpu_ane_fence_stats(&it, &ex, &h0, &mx, &nspi);
        for (uint32_t i = 0; i < g_expect_layers && i < DS4_ANE_MAX_LAYERS; ++i) {
            const ane_tl *t = &g_tl[i];
            if (!t->valid) continue;
            fprintf(stderr,
                    "ds4: ANE layer chunk=%llu il=%u wait_ready=%.3f "
                    "predict=%.3f release=%.3f span=%.3f ms\n",
                    (unsigned long long)(g_chunks + 1u), i,
                    (t->obs - t->enq) / 1e6, (t->p1 - t->p0) / 1e6,
                    (t->done - t->p1) / 1e6, (t->done - t->enq) / 1e6);
        }
        fprintf(stderr,
                "ds4: ANE sched chunk=%llu fence_cum execs=%llu hit0=%llu "
                "iters=%llu max=%llu ns_per_iter=%.2f%s\n",
                (unsigned long long)(g_chunks + 1u),
                (unsigned long long)ex, (unsigned long long)h0,
                (unsigned long long)it, (unsigned long long)mx, nspi,
                nspi > 0.0 ? "" : "  (raw iterations; set DS4_ANE_SCHED_CALIBRATE"
                                  " outside a timed run to convert)");
        fflush(stderr);
        for (uint32_t i = 0; i < DS4_ANE_MAX_LAYERS; ++i) g_tl[i].valid = 0;
    }
    /* One grep-able line per chunk. The harness asserts on every field:
     * engaged must equal the sparse-layer count, and skipped, failed,
     * cancelled and timeout must all be zero. Anything else means some layers
     * quietly ran on the GPU while the numbers described the ones that did
     * not. */
    g_chunks++;
    fprintf(stderr,
            "ds4: ANE chunk %llu: engaged=%llu/%u skipped=%llu failed=%llu "
            "cancelled=%llu timeout=%d predict=%.1f ms (%.3f ms/engaged)\n",
            (unsigned long long)g_chunks, (unsigned long long)g_engaged,
            g_expect_layers, (unsigned long long)g_skipped,
            (unsigned long long)g_failed, (unsigned long long)g_cancelled,
            ds4_gpu_ane_sync_timed_out(), g_ns_predict / 1.0e6,
            g_engaged ? g_ns_predict / 1.0e6 / (double)g_engaged : 0.0);
    /* The worker is SIGKILLed by the harness, so atexit never runs there and
     * anything still in the stdio buffer is lost. Per-chunk lines are the
     * primary record and must be on disk when they are written. */
    fflush(stderr);
    if (ds4_ane_mode() >= DS4_ANE_FAST && g_backpressure) {
        fprintf(stderr, "ds4: ANE ring backpressure %llu spins -- the host ran "
                        "further ahead than %u layers\n",
                (unsigned long long)g_backpressure, DS4_ANE_RING);
    }
    if (ds4_ane_mode() >= DS4_ANE_FAST && ds4_gpu_ane_sync_timed_out()) {
        fprintf(stderr, "ds4: ANE FAST fence TIMED OUT -- the GPU gave up "
                        "waiting on DONE, so at least one layer consumed a "
                        "stale output surface. Raise DS4_ANE_FENCE_MAX_ITERS; "
                        "do not read the timings from this run.\n");
    }
}
