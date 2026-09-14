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
    NSError *e = nil;
    MLMultiArray *a = [[MLMultiArray alloc]
        initWithDataPointer:base
                      shape:@[@1, @(n_tok), @(dim)]
                   dataType:MLMultiArrayDataTypeFloat16
                    strides:@[@(n_tok * dim), @(dim), @1]
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
        cfg.computeUnits = MLComputeUnitsAll;
        const uint64_t t0 = now_ns();
        for (uint32_t il = 0; il < n_layers; il++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/shexp_L%02u_fused.mlpackage", model_dir, il);
            NSURL *u = [NSURL fileURLWithPath:@(path)];
            NSError *e = nil;
            NSURL *c = [MLModel compileModelAtURL:u error:&e];
            MLModel *m = c ? [MLModel modelWithContentsOfURL:c configuration:cfg error:&e] : nil;
            [models addObject:(m ? (id)m : (id)[NSNull null])];
            if (m) loaded++;
        }
        fprintf(stderr, "ds4-ane-helper: %u/%u models loaded in %.2f s "
                        "(OUTSIDE any timed region, by construction)\n",
                loaded, n_layers, (double)(now_ns() - t0) / 1e9);
        if (loaded == 0) {
            atomic_store_explicit(&w[DS4_ANEPROC_W_FAULT], 5u, memory_order_release);
            return 2;
        }
    } else {
        fprintf(stderr, "ds4-ane-helper: NULL MODE -- no Core ML, input copied "
                        "to output. This prices the handoff alone.\n");
    }

    const size_t bytes = (size_t)dim * n_tok * sizeof(uint16_t);
    MLPredictionOptions *popt = [[MLPredictionOptions alloc] init];

    fprintf(stderr, "ds4-ane-helper: serving dim=%u n_tok=%u n_layers=%u%s\n",
            dim, n_tok, n_layers, null_mode ? " (null)" : "");
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
            if (!null_mode && il < n_layers) {
                id entry = models[il];
                if (entry != [NSNull null]) {
                    MLModel *m = (MLModel *)entry;
                    NSError *e = nil;
                    NSString *ikey = m.modelDescription.inputDescriptionsByName.allKeys.firstObject;
                    NSString *okey = m.modelDescription.outputDescriptionsByName.allKeys.firstObject;
                    MLDictionaryFeatureProvider *fp =
                        [[MLDictionaryFeatureProvider alloc]
                            initWithDictionary:@{ ikey: in } error:&e];
                    id<MLFeatureProvider> r = fp ? [m predictionFromFeatures:fp
                                                                     options:popt
                                                                       error:&e] : nil;
                    MLMultiArray *ov = r ? [r featureValueForName:okey].multiArrayValue : nil;
                    if (ov && ov.dataPointer != out_base) {
                        /* Core ML returned its own buffer rather than writing
                         * in place. Copying is correct but it is exactly the
                         * cost this design exists to avoid, so say so once. */
                        static int warned;
                        if (!warned) {
                            warned = 1;
                            fprintf(stderr, "ds4-ane-helper: prediction is NOT "
                                    "writing in place; copying %zu bytes per "
                                    "layer. The zero-copy path is not engaged.\n",
                                    bytes);
                        }
                        memcpy(out_base, ov.dataPointer, bytes);
                    }
                }
            } else if (null_mode) {
                memcpy(out_base, in_base, bytes);
            }
            atomic_store_explicit(&w[DS4_ANEPROC_W_PREDICT_NS],
                                  (uint32_t)(now_ns() - t0), memory_order_relaxed);
            /* Echo the resolved layer so the ring mapping is observable even in
             * null mode, where nothing else depends on it. */
            atomic_store_explicit(&w[DS4_ANEPROC_W_LAST_LAYER], il,
                                  memory_order_relaxed);
            served_seq = seq;
            served++;
            /* RELEASE: the output bytes must be visible before DONE is. */
            atomic_store_explicit(&w[DS4_ANEPROC_W_DONE], seq, memory_order_release);
            atomic_store_explicit(&w[DS4_ANEPROC_W_SERVED], served, memory_order_relaxed);
        }
    }

drained:
    fprintf(stderr, "ds4-ane-helper: stopping after %u predictions\n", served);
    atomic_store_explicit(&w[DS4_ANEPROC_W_ALIVE], 0u, memory_order_release);
    CFRelease(si); CFRelease(so); CFRelease(sc);
    return 0;
} }
