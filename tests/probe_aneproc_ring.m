/*
 * ANEPROC cross-process ring probe.
 *
 * The sidecar's VALUE needs the rig: a real prefill, real models, and both ANE
 * dies. Its CORRECTNESS and its handoff cost do not, and those are the parts a
 * dev box can settle -- so this exercises the whole shared-surface protocol
 * with the helper in --null mode: no Core ML, no models, no ANE.
 *
 * What it checks:
 *   - the helper attaches to all three surfaces by ID and validates the header
 *   - every published seq is served exactly once, in order
 *   - the data path is genuinely shared: the parent writes the input surface,
 *     the helper copies in->out, and the parent reads the result WITHOUT any
 *     message passing. A per-seq pattern catches a stale or aliased surface.
 *   - the round-trip cost of the handoff itself
 *   - a mismatched control block is REFUSED rather than served
 *
 * It does not use Metal: the GPU's role here is only to store a word, which the
 * host can do identically. What it cannot check is whether Core ML writes its
 * output in place; the helper reports that at runtime instead.
 */
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "ds4_aneproc.h"

static int fails;
extern char **environ;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static IOSurfaceRef make_surface(size_t bytes, void **base) {
    const size_t bpr = (bytes + 4095u) & ~(size_t)4095u;
    NSDictionary *p = @{
        (id)kIOSurfaceWidth:           @(bytes / 2),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement: @2,
        (id)kIOSurfaceBytesPerRow:     @(bpr),
        (id)kIOSurfaceAllocSize:       @(bpr),
        (id)kIOSurfacePixelFormat:     @(0x4C303136),
        (id)kIOSurfaceIsGlobal:        @YES,
    };
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)p);
    if (!s) return NULL;
    IOSurfaceLock(s, 0, NULL);
    *base = IOSurfaceGetBaseAddress(s);
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static pid_t spawn_helper(const char *path, IOSurfaceRef in, IOSurfaceRef out,
                          IOSurfaceRef ctl) {
    char a[16], b[16], c[16];
    snprintf(a, sizeof(a), "%u", (unsigned)IOSurfaceGetID(in));
    snprintf(b, sizeof(b), "%u", (unsigned)IOSurfaceGetID(out));
    snprintf(c, sizeof(c), "%u", (unsigned)IOSurfaceGetID(ctl));
    const char *argv[] = { path, "--in", a, "--out", b, "--ctl", c, NULL };
    pid_t pid = 0;
    if (posix_spawn(&pid, path, NULL, NULL, (char *const *)argv, environ) != 0) {
        fprintf(stderr, "spawn %s: %s\n", path, strerror(errno));
        return 0;
    }
    return pid;
}

int main(int argc, const char **argv) { @autoreleasepool {
    const char *helper = argc > 1 ? argv[1] : "./ds4-ane-helper";
    /* Past TWO full ring wraps. 400 was under the 512-slot ring, so slot reuse --
     * the one thing the ring index can get wrong at scale -- was never
     * exercised. n_layers is deliberately not a divisor of the ring size, so a
     * wrapped slot holding a stale layer shows up as a mismatch. */
    const uint32_t dim = 256, n_tok = 8, n_layers = 7;
    const uint32_t n_iter = DS4_ANEPROC_RING * 2 + 37;
    const size_t bytes = (size_t)dim * n_tok * sizeof(uint16_t);

    void *in_base = NULL, *out_base = NULL, *ctl_base = NULL;
    IOSurfaceRef si = make_surface(bytes, &in_base);
    IOSurfaceRef so = make_surface(bytes, &out_base);
    IOSurfaceRef sc = make_surface(DS4_ANEPROC_CTL_BYTES, &ctl_base);
    if (!si || !so || !sc || !in_base || !out_base || !ctl_base) {
        printf("VOID: could not create the shared surfaces\n");
        return 2;
    }
    _Atomic uint32_t *w = (_Atomic uint32_t *)ctl_base;

    /* --- refusal case first: a bad magic must not be served ---------------
     * Checked BEFORE the good case because the helper writes ALIVE into the
     * same block; running it second would not distinguish "refused" from
     * "served then stopped". */
    memset((void *)w, 0, DS4_ANEPROC_CTL_BYTES);
    atomic_store(&w[DS4_ANEPROC_W_MAGIC], 0xDEADBEEFu);
    atomic_store(&w[DS4_ANEPROC_W_VERSION], DS4_ANEPROC_VERSION);
    atomic_store(&w[DS4_ANEPROC_W_DIM], dim);
    atomic_store(&w[DS4_ANEPROC_W_NTOK], n_tok);
    atomic_store(&w[DS4_ANEPROC_W_NLAYERS], n_layers);
    atomic_store(&w[DS4_ANEPROC_W_NULLMODE], DS4_ANEPROC_NULL_ECHO);
    pid_t bad = spawn_helper(helper, si, so, sc);
    if (!bad) return 2;
    int st = 0;
    waitpid(bad, &st, 0);
    if (atomic_load(&w[DS4_ANEPROC_W_ALIVE]) != 0 ||
        atomic_load(&w[DS4_ANEPROC_W_FAULT]) == 0) {
        printf("FAIL bad magic was not refused (alive=%u fault=%u)\n",
               atomic_load(&w[DS4_ANEPROC_W_ALIVE]),
               atomic_load(&w[DS4_ANEPROC_W_FAULT]));
        fails++;
    } else {
        printf("ok   mismatched control block refused (fault=%u)\n",
               atomic_load(&w[DS4_ANEPROC_W_FAULT]));
    }

    /* --- the real ring ---------------------------------------------------- */
    memset((void *)w, 0, DS4_ANEPROC_CTL_BYTES);
    atomic_store(&w[DS4_ANEPROC_W_VERSION], DS4_ANEPROC_VERSION);
    atomic_store(&w[DS4_ANEPROC_W_DIM], dim);
    atomic_store(&w[DS4_ANEPROC_W_NTOK], n_tok);
    atomic_store(&w[DS4_ANEPROC_W_NLAYERS], n_layers);
    atomic_store(&w[DS4_ANEPROC_W_NULLMODE], DS4_ANEPROC_NULL_ECHO);
    atomic_store_explicit(&w[DS4_ANEPROC_W_MAGIC], DS4_ANEPROC_MAGIC,
                          memory_order_release);

    pid_t pid = spawn_helper(helper, si, so, sc);
    if (!pid) return 2;
    for (int i = 0; i < 5000 && !atomic_load(&w[DS4_ANEPROC_W_ALIVE]); i++) usleep(1000);
    if (!atomic_load(&w[DS4_ANEPROC_W_ALIVE])) {
        printf("FAIL helper never reported ALIVE (fault=%u)\n",
               atomic_load(&w[DS4_ANEPROC_W_FAULT]));
        kill(pid, SIGKILL); waitpid(pid, NULL, 0);
        return 1;
    }
    printf("ok   helper attached by ID and is serving\n");

    uint16_t *in16 = (uint16_t *)in_base, *out16 = (uint16_t *)out_base;
    uint64_t total = 0, worst = 0;
    for (uint32_t seq = 1; seq <= n_iter; seq++) {
        const uint32_t il = seq % n_layers;
        /* A pattern that is distinct per seq AND per element: a stale surface,
         * an off-by-one on the ring, or an output that was never written all
         * show up, where a constant fill would pass on any of them. */
        for (uint32_t e = 0; e < dim * n_tok; e++) {
            in16[e] = (uint16_t)((seq * 2654435761u + e) & 0xffffu);
        }
        memset(out16, 0, bytes);
        atomic_store_explicit(&w[DS4_ANEPROC_W_RING + (seq % DS4_ANEPROC_RING)],
                              il, memory_order_release);
        const uint64_t t0 = now_ns();
        /* This is exactly what the GPU publish kernel does: one store. */
        atomic_store_explicit(&w[DS4_ANEPROC_W_READY], seq, memory_order_release);
        while ((int32_t)(atomic_load_explicit(&w[DS4_ANEPROC_W_DONE],
                                              memory_order_acquire) - seq) < 0) {
            if (now_ns() - t0 > 5000000000ull) {
                printf("FAIL seq %u never completed\n", seq);
                fails++;
                goto done;
            }
        }
        const uint64_t dt = now_ns() - t0;
        total += dt;
        if (dt > worst) worst = dt;

        /* The layer the helper actually resolved for this seq. Null mode does
         * not use it, so without this echo an off-by-one in the ring index
         * passes -- verified: shifting the helper's slot by one used to leave
         * this probe green. */
        const uint32_t got_il = atomic_load(&w[DS4_ANEPROC_W_LAST_LAYER]);
        if (got_il != il) {
            printf("FAIL seq %u resolved layer %u, published %u\n", seq, got_il, il);
            fails++;
            goto done;
        }
        for (uint32_t e = 0; e < dim * n_tok; e++) {
            const uint16_t want = (uint16_t)((seq * 2654435761u + e) & 0xffffu);
            if (out16[e] != want) {
                printf("FAIL seq %u element %u: got %04x want %04x\n",
                       seq, e, out16[e], want);
                fails++;
                goto done;
            }
        }
    }
    printf("ok   %u predictions served in order, layer mapping and data correct\n", n_iter);
    printf("     handoff round trip: mean %.2f us, worst %.2f us "
           "(publish -> helper -> done, %zu KiB each way)\n",
           (double)total / n_iter / 1000.0, (double)worst / 1000.0, bytes / 1024);
    if (atomic_load(&w[DS4_ANEPROC_W_SERVED]) != n_iter) {
        printf("FAIL helper served %u, expected %u\n",
               atomic_load(&w[DS4_ANEPROC_W_SERVED]), n_iter);
        fails++;
    }

done:
    atomic_store_explicit(&w[DS4_ANEPROC_W_STOP], 1u, memory_order_release);
    for (int i = 0; i < 3000; i++) {
        if (waitpid(pid, &st, WNOHANG) == pid) { pid = 0; break; }
        usleep(1000);
    }
    if (pid) { printf("FAIL helper did not stop on request\n"); fails++;
               kill(pid, SIGKILL); waitpid(pid, NULL, 0); }
    else     { printf("ok   helper drained and exited on STOP\n"); }

    printf("\n%s\n", fails == 0
           ? "PASS: the ANEPROC cross-process ring is correct and the handoff "
             "needs no IPC"
           : "FAILED");
    return fails == 0 ? 0 : 1;
} }
