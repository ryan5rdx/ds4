#ifndef DS4_ANE_H
#define DS4_ANE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ANE shared-expert sidecar (GLM 5.3 Flash prefill).
 *
 * The rig established that the Neural Engine will take the shared-expert graph
 * at production shape and that running it flat out next to a routed-MoE prefill
 * produces no contention detectable above the noise floor. What none of that
 * establishes is the only number that decides the branch: what it costs to put
 * the handoff INSIDE ds4's prefill loop. The shared expert is ~417 ms of a
 * ~10.5 s chunk, about 4%, so an integration cost of a few milliseconds per
 * layer erases the entire win. 42 layers x 32 chunks is 1344 opportunities to
 * spend it.
 *
 * Hence the mode ladder. Each rung prices one thing and nothing else:
 *
 *   DS4_ANE_PROBE    fence only. No Core ML, no models, no data movement --
 *                    just the command-buffer flush the ANE would need in order
 *                    to read a completed batch_ffn_norm. If this alone costs
 *                    more than 417 ms/chunk the branch is dead and no amount of
 *                    Neural Engine throughput can save it.
 *   DS4_ANE_BRIDGE   fence + the f32->f16 transpose in and out. Prices the
 *                    layout conversion without involving Core ML.
 *   DS4_ANE_SHADOW   fence + bridge + a real prediction whose result is
 *                    DISCARDED, with the GPU still authoritative. Prices full
 *                    integration, and reports divergence against the GPU's own
 *                    shared-expert output so correctness is observed before it
 *                    is ever relied on.
 *
 * There is deliberately no "on". Replacing the GPU path needs Core ML models
 * built from the model's OWN Q8 weights; the models this loads are the
 * synthetic fp16 set from the rig harness, which are shape-correct and
 * value-meaningless. Shipping a mode that silently substituted random weights
 * for the shared expert would produce fluent, wrong text -- the worst possible
 * failure. The divergence figure `shadow` reports is therefore expected to be
 * enormous, and that is the point: it proves the comparison is wired up, and it
 * turns green only once real weights arrive.
 *
 * Enable with DS4_METAL_ANE_SHEXP=probe|bridge|shadow (default off), and point
 * DS4_ANE_MODEL_DIR at a directory of shexp_L{NN}_fused.mlpackage.
 */
enum {
    DS4_ANE_OFF    = 0,
    DS4_ANE_PROBE  = 1,
    DS4_ANE_BRIDGE = 2,
    DS4_ANE_SHADOW = 3,
    /* Same work as SHADOW, rendezvoused on ds4's system-coherent release words
     * instead of on command-buffer completion. The safe path costs two full
     * round trips per layer -- 2688 over a 131k prefill -- and the earlier
     * claim that this was "unavoidable" was too strong: it is the public-API
     * path, and ds4 already owns a cheaper one. Here the GPU publishes READY
     * after the pack, the sidecar thread spins on it and runs Core ML while
     * the GPU is inside routed-MoE, stores DONE on completion, and a GPU fence
     * spins on DONE before the unpack. Nothing is committed in between. */
    DS4_ANE_FAST   = 4,
};

/* Parsed once from the environment. 0 when unset or when this build has no
 * Core ML support. */
int ds4_ane_mode(void);

/* Prepare for `n_layers` layers at exactly `n_tokens` tokens per prediction.
 * Returns 0 on failure, which the caller must treat as "run the GPU path" --
 * never as a reason to abort the prefill. Safe to call repeatedly; a change of
 * n_tokens tears down and rebuilds, because a Core ML model's M is fixed at
 * conversion time and a mismatched shape must skip rather than reshape. */
int ds4_ane_init(uint32_t n_layers, uint32_t n_tokens);

/* Start layer `il` asynchronously. The caller must already have fenced the GPU
 * work producing the input. Returns 0 if this layer is not eligible (shape
 * mismatch, model missing, mode too low) -- not an error. */
int ds4_ane_begin_layer(uint32_t il);

/* Block until layer `il`'s prediction has landed. Returns 0 if nothing was
 * started for it. */
int ds4_ane_wait(uint32_t il);

/* One line per run to stderr: per-stage cost, engaged/skipped layers, and the
 * shadow divergence. Cheap and idempotent. */
void ds4_ane_report(void);

/* Reset the accumulated counters (used by the per-chunk reporting path). */
void ds4_ane_reset(void);

/* FAST only: the per-layer sequence rendezvous. begin_layer publishes nothing
 * itself -- ds4.c encodes the GPU publish -- so the caller needs the sequence
 * number it must hand to ds4_gpu_ane_publish_ready/ds4_gpu_ane_fence_done. */
uint32_t ds4_ane_next_seq(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_ANE_H */
