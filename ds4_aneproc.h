#ifndef DS4_ANEPROC_H
#define DS4_ANEPROC_H

#include <stdint.h>

/*
 * ANEPROC -- the shared-expert sidecar as a SEPARATE PROCESS.
 *
 * ANEDIES2R settled the topology: two processes each get an ANE die at full
 * speed (1.997x, 99-100% verified overlap), while two streams inside ONE
 * process serialise. The in-process sidecar therefore cannot use the second
 * die at all, and that is a property of Core ML's scheduling rather than of
 * the silicon.
 *
 * WHAT THIS COSTS AT RUNTIME: nothing. There is no IPC in the steady state.
 *
 * The staging buffers were already IOSurface-backed, so making them global and
 * looking them up by ID in the helper gives both processes the SAME physical
 * pages -- the GPU writes the input, the ANE reads it, the ANE writes the
 * output, the GPU reads it, and no byte is copied or messaged. The existing
 * FAST-mode rendezvous then works across the process boundary unchanged: the
 * GPU publishes READY into a shared word, the helper spins on it, and the GPU
 * fence spins on DONE. Measured round trip through a shared IOSurface word
 * between two processes on this machine: 0.07 us. It is a cache line.
 *
 * So the only IPC is SETUP: three IOSurface IDs and a model directory, passed
 * on the helper's command line once.
 *
 * SECURITY NOTE, stated because it is a real tradeoff and not an oversight:
 * IOSurfaceLookup() by ID requires kIOSurfaceIsGlobal, which is deprecated and
 * lets any process of the same user attach to the surface by guessing a small
 * integer. The supported alternative is passing a mach port, which needs XPC or
 * a bootstrap-registered service -- materially more machinery for a private
 * research fork on a dedicated rig. The surfaces are only created global when
 * the sidecar is explicitly enabled, and they hold activations, not weights or
 * keys. This is a deliberate private-fork decision; it must not be carried
 * upstream as-is.
 *
 * ONE HELPER, not two. A second helper only pays if the shared expert is
 * sharded by HIDDEN DIMENSION -- layers are serially dependent, so splitting by
 * layer range leaves one die idle at any instant -- and that is a separate
 * experiment. The control block carries a helper index so a second can be added
 * without a format change.
 */

/* Layout of the shared control block. Words 0 and 1 keep the meanings the
 * in-process FAST path already gave them, so the GPU-side publish and fence
 * kernels are untouched: the buffer they were handed is now simply backed by an
 * IOSurface instead of a private allocation. */
#define DS4_ANEPROC_MAGIC    0x414E4550u   /* 'ANEP' */
#define DS4_ANEPROC_VERSION  1u
#define DS4_ANEPROC_RING     512u          /* power of two; seq % RING -> layer */

/* Word indices inside the control surface. */
enum {
    DS4_ANEPROC_W_READY = 0,   /* GPU    -> helper: highest seq whose input is live */
    DS4_ANEPROC_W_DONE  = 1,   /* helper -> GPU:    highest seq whose output is live */
    DS4_ANEPROC_W_RSV2  = 2,
    DS4_ANEPROC_W_RSV3  = 3,
    /* Header begins at word 4 so the four words above keep their offsets. */
    DS4_ANEPROC_W_MAGIC = 4,
    DS4_ANEPROC_W_VERSION,
    DS4_ANEPROC_W_DIM,
    DS4_ANEPROC_W_NTOK,
    DS4_ANEPROC_W_NLAYERS,
    DS4_ANEPROC_W_STOP,        /* parent -> helper: drain and exit */
    DS4_ANEPROC_W_ALIVE,       /* helper -> parent: models loaded, serving */
    DS4_ANEPROC_W_SERVED,      /* helper -> parent: predictions completed */
    DS4_ANEPROC_W_FAULT,       /* helper -> parent: nonzero = gave up, see stderr */
    DS4_ANEPROC_W_NULLMODE,    /* parent -> helper: skip Core ML (handoff-only) */
    DS4_ANEPROC_W_HELPER_IDX,  /* which helper this is; 0 today */
    DS4_ANEPROC_W_PREDICT_NS,  /* helper -> parent: last prediction, nanoseconds */
    /* helper -> parent: the layer index the helper resolved for the last seq.
     * Exists because null mode does not USE the layer, so without echoing it
     * the seq -> layer mapping is untested -- and an off-by-one there runs the
     * wrong layer's model in production, silently. Also useful live. */
    DS4_ANEPROC_W_LAST_LAYER,
    DS4_ANEPROC_W_HDR_END,
    /* The seq -> layer ring starts here, 16-word aligned. */
    DS4_ANEPROC_W_RING = 32,
};

#define DS4_ANEPROC_CTL_WORDS (DS4_ANEPROC_W_RING + DS4_ANEPROC_RING)
#define DS4_ANEPROC_CTL_BYTES (DS4_ANEPROC_CTL_WORDS * 4u)

#endif /* DS4_ANEPROC_H */
