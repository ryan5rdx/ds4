#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ds4_ane.h"

/*
 * The Core ML half of the shared-expert sidecar. Metal, IOSurface and the
 * command buffer all live in ds4_metal.m; this file sees two base pointers and
 * a set of models, and never links against Metal at all.
 */

/* Provided by ds4_metal.m. */
int  ds4_gpu_ane_stage_alloc(uint32_t dim, uint32_t n_tok,
                             void **in_ptr, void **out_ptr);
void ds4_gpu_ane_stage_free(void);

#define DS4_ANE_MAX_LAYERS 64
#define DS4_ANE_DIM        4096u

static int       g_mode = -1;
static int       g_ready;
static uint32_t  g_n_layers, g_n_tok;
static MLModel  *g_models[DS4_ANE_MAX_LAYERS];
static MLMultiArray *g_in_array, *g_out_array;
static dispatch_queue_t g_queue;
static dispatch_semaphore_t g_done[DS4_ANE_MAX_LAYERS];
static int       g_started[DS4_ANE_MAX_LAYERS];

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
        else if (!strcmp(e, "bridge")) g_mode = DS4_ANE_BRIDGE;
        else if (!strcmp(e, "shadow")) g_mode = DS4_ANE_SHADOW;
        else if (!strcmp(e, "0") || !strcmp(e, "off")) g_mode = DS4_ANE_OFF;
        else {
            fprintf(stderr, "ds4: DS4_METAL_ANE_SHEXP=%s unrecognised "
                            "(probe|bridge|shadow|off) -- sidecar off\n", e);
        }
    }
    if (g_mode != DS4_ANE_OFF) {
        fprintf(stderr, "ds4: ANE shared-expert sidecar mode=%s\n", e);
    }
    return g_mode;
}

static void ds4_ane_teardown(void) {
    for (uint32_t i = 0; i < DS4_ANE_MAX_LAYERS; ++i) {
        g_models[i] = nil;
        g_done[i] = nil;
        g_started[i] = 0;
    }
    g_in_array = nil; g_out_array = nil; g_queue = nil;
    ds4_gpu_ane_stage_free();
    g_ready = 0; g_n_layers = 0; g_n_tok = 0;
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

int ds4_ane_init(uint32_t n_layers, uint32_t n_tokens) {
    if (ds4_ane_mode() == DS4_ANE_OFF) return 0;
    if (n_layers == 0 || n_layers > DS4_ANE_MAX_LAYERS || n_tokens == 0) return 0;
    if (g_ready && g_n_layers == n_layers && g_n_tok == n_tokens) return 1;
    ds4_ane_teardown();

    void *in_ptr = NULL, *out_ptr = NULL;
    if (!ds4_gpu_ane_stage_alloc(DS4_ANE_DIM, n_tokens, &in_ptr, &out_ptr)) {
        fprintf(stderr, "ds4: ANE staging alloc failed (dim=%u tok=%u)\n",
                DS4_ANE_DIM, n_tokens);
        return 0;
    }

    @autoreleasepool {
        g_in_array  = ds4_ane_wrap(in_ptr,  DS4_ANE_DIM, n_tokens);
        g_out_array = ds4_ane_wrap(out_ptr, DS4_ANE_DIM, n_tokens);
        if (!g_in_array || !g_out_array) { ds4_ane_teardown(); return 0; }

        /* PROBE and BRIDGE deliberately load nothing: they price the fence and
         * the layout conversion, and a Core ML load would contaminate both. */
        if (ds4_ane_mode() >= DS4_ANE_SHADOW) {
            const char *dir = getenv("DS4_ANE_MODEL_DIR");
            if (!dir || !dir[0]) {
                fprintf(stderr, "ds4: ANE shadow needs DS4_ANE_MODEL_DIR\n");
                ds4_ane_teardown();
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
            if (loaded == 0) {
                fprintf(stderr, "ds4: ANE shadow found no models under %s\n", dir);
                ds4_ane_teardown();
                return 0;
            }
            fprintf(stderr, "ds4: ANE loaded %u/%u shared-expert models "
                            "(variant=%s M=%u) from %s\n",
                    loaded, n_layers, variant, n_tokens, dir);
            fprintf(stderr, "ds4: ANE shadow weights are SYNTHETIC -- the "
                            "divergence below is expected to be large and the "
                            "GPU stays authoritative\n");
        }

        g_queue = dispatch_queue_create("ds4.ane.shexp", DISPATCH_QUEUE_SERIAL);
        for (uint32_t i = 0; i < n_layers; ++i) g_done[i] = dispatch_semaphore_create(0);
    }
    g_n_layers = n_layers; g_n_tok = n_tokens; g_ready = 1;
    return 1;
}

int ds4_ane_begin_layer(uint32_t il) {
    if (!g_ready || il >= g_n_layers) return 0;
    if (ds4_ane_mode() < DS4_ANE_SHADOW) return 0;
    MLModel *m = g_models[il];
    if (!m) { g_skipped++; return 0; }

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
        }
        dispatch_semaphore_signal(done);
    });
    return 1;
}

int ds4_ane_wait(uint32_t il) {
    if (!g_ready || il >= g_n_layers || !g_started[il]) return 0;
    dispatch_semaphore_wait(g_done[il], DISPATCH_TIME_FOREVER);
    g_started[il] = 0;
    return 1;
}

void ds4_ane_reset(void) {
    g_ns_predict = 0.0; g_engaged = g_skipped = g_failed = 0;
}

void ds4_ane_report(void) {
    if (ds4_ane_mode() == DS4_ANE_OFF) return;
    if (g_engaged == 0 && g_skipped == 0 && g_failed == 0) return;
    fprintf(stderr,
            "ds4: ANE sidecar: engaged=%llu skipped=%llu failed=%llu "
            "predict=%.1f ms total (%.3f ms/engaged)\n",
            (unsigned long long)g_engaged, (unsigned long long)g_skipped,
            (unsigned long long)g_failed, g_ns_predict / 1.0e6,
            g_engaged ? g_ns_predict / 1.0e6 / (double)g_engaged : 0.0);
}
