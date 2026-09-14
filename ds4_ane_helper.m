/*
 * ds4-ane-helper -- the ANEPROC sidecar, as its own process.
 *
 * Attaches to three IOSurfaces by ID, loads the shared-expert Core ML models,
 * and serves predictions off a shared release word. It never links Metal and
 * never touches the GPU: the point of a separate process is a separate Core ML
 * scheduling context, and adding a Metal device here would put a second GPU
 * client on the machine for no reason.
 *
 * Protocol, all of it, per prediction:
 *   parent  writes layer index into ring[seq % RING]   (host, at enqueue)
 *   GPU     stores READY = seq                          (after the input pack)
 *   helper  sees READY >= seq, reads the layer, predicts in-place
 *   helper  stores DONE = seq
 *   GPU     fence spins on DONE before the unpack
 *
 * There is no message and no copy. See ds4_aneproc.h.
 *
 * --null skips Core ML entirely and copies input to output. That arm exists so
 * the cross-process ring can be exercised on a machine with no models and no
 * usable ANE -- it prices the handoff alone, the same separation FASTNULL makes
 * in-process, and it is the only part of this that can be tested off the rig.
 */
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ds4_aneproc.h"

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static IOSurfaceRef attach(const char *what, uint32_t id, void **base) {
    IOSurfaceRef s = IOSurfaceLookup((IOSurfaceID)id);
    if (!s) {
        fprintf(stderr, "ds4-ane-helper: cannot attach %s surface %u\n", what, id);
        return NULL;
    }
    IOSurfaceLock(s, 0, NULL);
    *base = IOSurfaceGetBaseAddress(s);
    IOSurfaceUnlock(s, 0, NULL);
    if (!*base) {
        fprintf(stderr, "ds4-ane-helper: %s surface %u has no base address\n", what, id);
        CFRelease(s);
        return NULL;
    }
    return s;
}

/* One MLMultiArray per staging surface, built once. Core ML will hold the
 * pointer for the process lifetime; rebuilding per prediction showed up as
 * allocator time in the in-process sidecar and there is no reason to repeat it
 * here. */
static MLMultiArray *wrap(void *base, uint32_t dim, uint32_t n_tok) {
    /* [1, dim, 1, n_tok], NOT [1, n_tok, dim]. This must match what
     * ane-gen-io.py converts and what ds4_ane_wrap() uses in-process; the first
     * version here had it transposed, which real models reject at shape
     * validation and which null mode -- copying bytes -- could not notice. */
    NSError *e = nil;
    MLMultiArray *a = [[MLMultiArray alloc]
        initWithDataPointer:base
                      shape:@[@1, @(dim), @1, @(n_tok)]
                   dataType:MLMultiArrayDataTypeFloat16
                    strides:@[@((NSInteger)dim * n_tok), @((NSInteger)n_tok),
                              @((NSInteger)n_tok), @1]
                deallocator:nil
                      error:&e];
    if (!a) {
        fprintf(stderr, "ds4-ane-helper: MLMultiArray failed: %s\n",
                e.localizedDescription.UTF8String);
    }
    return a;
}

int main(int argc, const char **argv) { @autoreleasepool {
    uint32_t in_id = 0, out_id = 0, ctl_id = 0;
    const char *model_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc)      in_id  = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_id = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ctl") && i + 1 < argc) ctl_id = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--models") && i + 1 < argc) model_dir = argv[++i];
        else {
            fprintf(stderr, "usage: %s --in <id> --out <id> --ctl <id> "
                            "[--models <dir>]\n", argv[0]);
            return 2;
        }
    }
    if (!in_id || !out_id || !ctl_id) {
        fprintf(stderr, "ds4-ane-helper: --in, --out and --ctl are required\n");
        return 2;
    }

    void *in_base = NULL, *out_base = NULL, *ctl_base = NULL;
    IOSurfaceRef si = attach("input", in_id, &in_base);
    IOSurfaceRef so = attach("output", out_id, &out_base);
    IOSurfaceRef sc = attach("control", ctl_id, &ctl_base);
    if (!si || !so || !sc) return 2;

    _Atomic uint32_t *w = (_Atomic uint32_t *)ctl_base;
    /* Validate before trusting a single field. A stale or mismatched control
     * block would otherwise be read as a plausible geometry and the helper
     * would predict on the wrong shape -- wrong numbers, no error. */
    const uint32_t magic = atomic_load_explicit(&w[DS4_ANEPROC_W_MAGIC], memory_order_acquire);
    const uint32_t ver   = atomic_load_explicit(&w[DS4_ANEPROC_W_VERSION], memory_order_relaxed);
    if (magic != DS4_ANEPROC_MAGIC || ver != DS4_ANEPROC_VERSION) {
        fprintf(stderr, "ds4-ane-helper: control block is magic %08x version %u, "
                        "expected %08x/%u -- refusing\n",
                magic, ver, DS4_ANEPROC_MAGIC, DS4_ANEPROC_VERSION);
        atomic_store_explicit(&w[DS4_ANEPROC_W_FAULT], 1u, memory_order_release);
        return 2;
    }
    const uint32_t dim      = atomic_load_explicit(&w[DS4_ANEPROC_W_DIM], memory_order_relaxed);
    const uint32_t n_tok    = atomic_load_explicit(&w[DS4_ANEPROC_W_NTOK], memory_order_relaxed);
    const uint32_t n_layers = atomic_load_explicit(&w[DS4_ANEPROC_W_NLAYERS], memory_order_relaxed);
    const uint32_t null_mode = atomic_load_explicit(&w[DS4_ANEPROC_W_NULLMODE], memory_order_relaxed);
    const pid_t parent_pid = (pid_t)atomic_load_explicit(&w[DS4_ANEPROC_W_PARENT_PID],
                                                         memory_order_relaxed);
    if (!dim || !n_tok || !n_layers || n_layers > 256u) {
        fprintf(stderr, "ds4-ane-helper: implausible geometry dim=%u n_tok=%u "
                        "n_layers=%u\n", dim, n_tok, n_layers);
        atomic_store_explicit(&w[DS4_ANEPROC_W_FAULT], 2u, memory_order_release);
        return 2;
    }

    MLMultiArray *in = wrap(in_base, dim, n_tok);
    MLMultiArray *out = wrap(out_base, dim, n_tok);
    if (!in || !out) {
        atomic_store_explicit(&w[DS4_ANEPROC_W_FAULT], 3u, memory_order_release);
        return 2;
    }

    /* Load EVERY model before announcing ALIVE. The in-process sidecar's
     * biggest measurement defect was per-model init landing inside a timed
     * region -- 0.3675 s/model, R^2 0.9954 against model count, which voided
     * the whole ANEI8 projection. A separate process makes that structural:
     * nothing is timed until loading is finished. */
    const size_t bytes_for_gate = (size_t)dim * n_tok * sizeof(uint16_t);
    NSMutableArray *models = [NSMutableArray array];
    uint32_t loaded = 0;
    if (!null_mode) {
        if (!model_dir) {
            fprintf(stderr, "ds4-ane-helper: --models is required unless the "
                            "control block requests null mode\n");
            atomic_store_explicit(&w[DS4_ANEPROC_W_FAULT], 4u, memory_order_release);
            return 2;
        }
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        /* CPUAndNeuralEngine, matching the in-process control. MLComputeUnitsAll
         * lets Core ML fall back to the GPU, which would contaminate exactly
         * the GPU-contention measurement this whole arm exists to make -- the
         * sidecar would be competing with the routed MoE it is supposed to run
         * beside. */
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        const uint64_t t0 = now_ns();
        for (uint32_t il = 0; il < n_layers; il++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/shexp_L%02u_fused.mlpackage", model_dir, il);
            NSURL *u = [NSURL fileURLWithPath:@(path)];
            NSError *e = nil;
            NSURL *c = [MLModel compileModelAtURL:u error:&e];
            MLModel *m = c ? [MLModel modelWithContentsOfURL:c configuration:cfg error:&e] : nil;
            if (!m) {
                /* EVERY model, or none. A partially loaded set means some
                 * layers silently do nothing while DONE still advances, which
                 * makes FAST look cheap because no computation happened. */
                fprintf(stderr, "ds4-ane-helper: layer %u failed to load (%s): %s\n",
                        il, path, e.localizedDescription.UTF8String);
                atomic_store_explicit(&w[DS4_ANEPROC_W_FAULT], 5u, memory_order_release);
                return 2;
            }
            [models addObject:m];
            loaded++;
        }
        fprintf(stderr, "ds4-ane-helper: %u/%u models loaded in %.2f s "
                        "(OUTSIDE any timed region, by construction)\n",
                loaded, n_layers, (double)(now_ns() - t0) / 1e9);

        /* WARM EVERY MODEL, AND GATE ZERO-COPY ON POINTER EQUALITY.
         *
         * Both in one pass, because both have to be true before the first
         * timed prediction. The first call per model carries initialisation
         * that would otherwise land inside the measurement -- the defect that
         * voided ANEI8 -- and if Core ML declines the output backing it
         * allocates its own buffer, which at 4096x2048 is a 16 MiB copy per
         * layer and 672 MiB per 42-layer chunk. That is not a warning to print
         * and then memcpy past; it is the premise of the design, so it is a
         * startup gate. */
        const uint64_t tw = now_ns();
        for (uint32_t il = 0; il < n_layers; il++) {
            @autoreleasepool {
                MLModel *m = models[il];
                NSError *e = nil;
                MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{ @"x": [MLFeatureValue featureValueWithMultiArray:in] }
                                 error:&e];
                MLPredictionOptions *o = [[MLPredictionOptions alloc] init];
                NSString *okey = m.modelDescription.outputDescriptionsByName.allKeys.firstObject;
                if (okey) o.outputBackings = @{ okey: out };
                id<MLFeatureProvider> r = fp ? [m predictionFromFeatures:fp options:o error:&e] : nil;
                if (!r) {
                    fprintf(stderr, "ds4-ane-helper: layer %u warm prediction "
                            "failed: %s\n", il, e.localizedDescription.UTF8String);
                    atomic_store_explicit(&w[DS4_ANEPROC_W_FAULT], 6u, memory_order_release);
                    return 2;
                }
                MLMultiArray *ov = [r featureValueForName:okey].multiArrayValue;
                if (!ov || ov.dataPointer != out_base) {
                    fprintf(stderr, "ds4-ane-helper: layer %u did NOT honour the "
                            "output backing (%p vs %p). Zero copy is the premise "
                            "of ANEPROC -- refusing rather than copying %zu bytes "
                            "per layer.\n", il, ov ? ov.dataPointer : NULL,
                            out_base, bytes_for_gate);
                    atomic_store_explicit(&w[DS4_ANEPROC_W_FAULT], 7u, memory_order_release);
                    return 2;
                }
            }
        }
        fprintf(stderr, "ds4-ane-helper: %u models warmed in %.2f s; output "
                        "backing honoured by all (zero copy confirmed)\n",
                n_layers, (double)(now_ns() - tw) / 1e9);
    } else {
        fprintf(stderr, "ds4-ane-helper: NULL MODE %u (%s) -- no Core ML.\n",
                null_mode,
                null_mode == DS4_ANEPROC_NULL_NOOP ? "noop: control/fence only"
                                                   : "echo: shared-surface correctness");
    }

    const size_t bytes = (size_t)dim * n_tok * sizeof(uint16_t);
    MLPredictionOptions *popt = [[MLPredictionOptions alloc] init];

    if (parent_pid == 0) {
        /* Not fatal, but said out loud: a parent that does not publish its pid
         * gets no orphan detection, and an orphaned helper holds every model
         * and polls forever. Silently skipping the check would be the worse
         * half of both options. */
        fprintf(stderr, "ds4-ane-helper: parent published no pid -- ORPHAN "
                        "DETECTION DISABLED for this run\n");
    }
    fprintf(stderr, "ds4-ane-helper: serving dim=%u n_tok=%u n_layers=%u%s "
                    "(parent %d)\n",
            dim, n_tok, n_layers, null_mode ? " (null)" : "", (int)parent_pid);
    atomic_store_explicit(&w[DS4_ANEPROC_W_ALIVE], 1u, memory_order_release);

    uint32_t served_seq = 0, served = 0;
    /* Bounded spin, then back off. A dedicated process must not burn a
     * performance core for the ~70% of the window it is idle -- on a machine
     * whose CPU also runs the TP service that core is worth more elsewhere.
     * The same reasoning, and the same numbers, as the in-process loop. */
    for (;;) {
        uint32_t spins = 0;
        for (;;) {
            if (atomic_load_explicit(&w[DS4_ANEPROC_W_STOP], memory_order_acquire)) goto drained;
            const uint32_t ready = atomic_load_explicit(&w[DS4_ANEPROC_W_READY],
                                                        memory_order_acquire);
            if ((int32_t)(ready - served_seq) > 0) break;
            if (++spins < 2000u) continue;
            /* ORPHAN CHECK, on the backoff path only -- never in the spin, so
             * it costs nothing while serving. Neither GPU cleanup nor the
             * harness's `pkill ds4-bench` stops this process, so without it a
             * killed parent leaves a helper holding every model and polling
             * forever, and the next arm measures against it. */
            if (parent_pid != 0 && getppid() != parent_pid) {
                fprintf(stderr, "ds4-ane-helper: parent %d is gone (now %d); "
                        "exiting rather than orphaning\n",
                        (int)parent_pid, (int)getppid());
                goto drained;
            }
            struct timespec ts = { 0, 20000 };   /* 20 us */
            nanosleep(&ts, NULL);
        }

        /* Serve every seq the GPU has published, in order. `>` not `==`: the
         * GPU publishes monotonically and may already be several ahead. */
        const uint32_t ready = atomic_load_explicit(&w[DS4_ANEPROC_W_READY],
                                                    memory_order_acquire);
        while ((int32_t)(ready - served_seq) > 0) {
            const uint32_t seq = served_seq + 1u;
            const uint32_t il = atomic_load_explicit(
                &w[DS4_ANEPROC_W_RING + (seq % DS4_ANEPROC_RING)],
                memory_order_acquire);
            const uint64_t t0 = now_ns();
            int ok = 1;
            /* Per-prediction pool. The process-lifetime pool was retaining
             * every Core ML temporary for the whole run -- at this size that is
             * not a leak you notice at the end, it is memory pressure during
             * the measurement. */
            @autoreleasepool {
                if (null_mode == DS4_ANEPROC_NULL_ECHO) {
                    memcpy(out_base, in_base, bytes);
                } else if (null_mode == DS4_ANEPROC_NULL_NOOP) {
                    /* Nothing at all: control and fence overhead alone. */
                } else if (il >= n_layers) {
                    fprintf(stderr, "ds4-ane-helper: seq %u published layer %u "
                            ">= %u\n", seq, il, n_layers);
                    ok = 0;
                } else {
                    MLModel *m = models[il];
                    NSError *e = nil;
                    MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
                        initWithDictionary:@{ @"x": [MLFeatureValue
                                                      featureValueWithMultiArray:in] }
                                     error:&e];
                    MLPredictionOptions *o = [[MLPredictionOptions alloc] init];
                    NSString *okey = m.modelDescription.outputDescriptionsByName.allKeys.firstObject;
                    /* The output backing, which the first version omitted --
                     * Core ML then allocated its own buffer and the result was
                     * memcpy'd, 16 MiB per layer. Startup already proved every
                     * model honours it. */
                    if (okey) o.outputBackings = @{ okey: out };
                    id<MLFeatureProvider> r = fp ? [m predictionFromFeatures:fp
                                                                     options:o
                                                                       error:&e] : nil;
                    if (!r) {
                        fprintf(stderr, "ds4-ane-helper: seq %u layer %u "
                                "prediction failed: %s\n", seq, il,
                                e.localizedDescription.UTF8String);
                        ok = 0;
                    }
                }
            }
            atomic_store_explicit(&w[DS4_ANEPROC_W_PREDICT_NS],
                                  (uint32_t)(now_ns() - t0), memory_order_relaxed);
            atomic_store_explicit(&w[DS4_ANEPROC_W_LAST_LAYER], il,
                                  memory_order_relaxed);
            if (!ok) {
                /* DO NOT publish DONE. The first version stored it
                 * unconditionally, so a missing model or a failed prediction
                 * looked like a completed one: FAST measured a handoff with no
                 * computation behind it and PERFONLY consumed stale output,
                 * with nothing in the run able to tell success from failure.
                 *
                 * There is no safe fallback at this point either -- the GPU is
                 * already fenced on this seq and the output surface holds the
                 * previous layer's values. The run is invalid; say so and let
                 * the fence time out against a FAULT the parent can report. */
                atomic_fetch_add_explicit(&w[DS4_ANEPROC_W_FAILED], 1u,
                                          memory_order_relaxed);
                atomic_store_explicit(&w[DS4_ANEPROC_W_FAULT], 8u,
                                      memory_order_release);
                fprintf(stderr, "ds4-ane-helper: FATAL at seq %u -- not "
                        "publishing DONE; this run is invalid\n", seq);
                goto drained;
            }
            served_seq = seq;
            served++;
            /* EVERY telemetry store BEFORE the DONE release.
             *
             * SERVED used to be written after it, and DONE is what unblocks the
             * GPU fence -- so the parent could complete the chunk and run its
             * validation while SERVED still held the previous value, reporting
             * a shortfall that never happened. DONE is the release store and
             * nothing may follow it. */
            atomic_store_explicit(&w[DS4_ANEPROC_W_SERVED], served, memory_order_relaxed);
            atomic_store_explicit(&w[DS4_ANEPROC_W_DONE], seq, memory_order_release);
        }
    }

drained:
    fprintf(stderr, "ds4-ane-helper: stopping after %u predictions\n", served);
    atomic_store_explicit(&w[DS4_ANEPROC_W_ALIVE], 0u, memory_order_release);
    CFRelease(si); CFRelease(so); CFRelease(sc);
    return 0;
} }
