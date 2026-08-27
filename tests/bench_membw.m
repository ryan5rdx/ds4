/* U6 — memory-bandwidth ceiling probe.
 *
 * WHY THIS EXISTS
 *
 * On a 2x M2 Ultra rig every bandwidth-bound decode stage tops out near
 * 400 GB/s.  U1 swept the working set 8x with `bench_moe_mxfp4_decode` and
 * found a hard 408-410 GB/s plateau, which rules out cache effects and size --
 * but every arm shared one kernel and one allocation path, so it could not
 * distinguish "the part cannot do better" from "our buffers are placed badly".
 *
 * The M2 Ultra is two M2 Max dies over UltraFusion, 400 GB/s of LPDDR5 each.
 * Reaching the 800 GB/s spec requires traffic spread across both dies'
 * controllers.  ds4 never lets Metal place the weights: the GGUF is a
 * file-backed MAP_SHARED mmap (ds4.c:2544) wrapped with
 * newBufferWithBytesNoCopy as StorageModeShared (ds4_metal.m:2023, :1377).
 * Physical placement is decided by the VM's fault path.  If those pages land
 * predominantly on one die, all GPU cores contend for that die's 400 GB/s and
 * half the memory system idles -- which is exactly the number we measure.
 *
 * This harness runs ONE streaming-read kernel against several allocation
 * paths so the three candidate explanations separate cleanly:
 *
 *   placement      -> `private` and `parallel-fault` beat `mmap-file`
 *   concurrency    -> `two-queue` beats every single-dispatch arm
 *   architectural  -> everything lands on the same number
 *
 * WHAT IT IS NOT
 *
 * Not a model benchmark.  It measures how fast the GPU can read bytes, and
 * nothing else.  Read `tests/bench_indexer_score.c`'s header before drawing
 * engine conclusions from any standalone harness -- that one over-predicted a
 * latency fix by 10x.  A bandwidth ceiling is a much safer thing to measure
 * than a latency win, but the same caution applies to extrapolating.
 *
 * The M1 Max is a single 400 GB/s die and CANNOT answer the two-die question.
 * It is still useful as a pre-screen: if the allocation arms differ on a
 * single-die part, the penalty is paging or hazard tracking rather than die
 * locality, and that is worth knowing before spending rig time.
 *
 *   make tests/bench_membw
 *   ./tests/bench_membw                      # 4 GiB, all arms
 *   ./tests/bench_membw 16                   # 16 GiB
 *   ./tests/bench_membw 16 20                # 16 GiB, 20 timed iterations
 *   BENCH_MEMBW_FILE=/path/to/model.gguf ./tests/bench_membw 16
 *
 * Arms are freed as they finish, so peak footprint is about 2x the arm size
 * (the two-queue arm holds two buffers at once).  Size accordingly: 16 GiB per
 * arm wants ~32 GiB free, and note iogpu.wired_limit_mb on the rig.
 *
 * BENCH_MEMBW_FILE points the file-backed arms at a real file -- use the GGUF
 * on the rig, which makes `mmap-file` literally the engine's own path.  With
 * no file set, a temp file is created and removed.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Grid-stride read of uint4 lanes.  Consecutive threads touch consecutive
 * 16-byte lanes so each simdgroup issues one coalesced transaction per step,
 * and the per-thread stride keeps many loads outstanding without any
 * dependent chain between them.
 *
 * `store_flag` is passed as 0 from the host every run.  The compiler cannot
 * prove that, so it cannot sink the loads -- which a plain unused accumulator
 * would let it do.  Do not "simplify" this away. */
static NSString *const kSource = @"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void stream_read(device const uint4 *src   [[buffer(0)]],\n"
"                        device uint4       *sink  [[buffer(1)]],\n"
"                        constant uint      &n_vec [[buffer(2)]],\n"
"                        constant uint      &store [[buffer(3)]],\n"
"                        uint gid [[thread_position_in_grid]],\n"
"                        uint gsz [[threads_per_grid]]) {\n"
"    uint4 a = uint4(0);\n"
"    for (uint i = gid; i < n_vec; i += gsz) a ^= src[i];\n"
"    if (store) sink[gid & 1023u] = a;\n"
"}\n"
"kernel void fill_buf(device uint4 *dst   [[buffer(0)]],\n"
"                     constant uint &n_vec [[buffer(1)]],\n"
"                     uint gid [[thread_position_in_grid]],\n"
"                     uint gsz [[threads_per_grid]]) {\n"
"    for (uint i = gid; i < n_vec; i += gsz) dst[i] = uint4(i, i + 1u, i + 2u, i + 3u);\n"
"}\n"
/* U8 bracket: walk fixed-size blocks instead of a flat vector stream.  MXFP4
 * packs 32 4-bit values plus one E8M0 scale into 17 bytes, so block b starts
 * at 17b and a simdgroup's 32 threads span 544 bytes at 32 different odd
 * offsets -- every load straddles a 16-byte boundary and touches more cache
 * lines than the same payload would at stride 16.  Running the identical
 * kernel at blk=16 and blk=17 isolates that layout cost from dequant
 * arithmetic and from the expert gather. */
"kernel void stream_blocks(device const uchar *src  [[buffer(0)]],\n"
"                          device uint4       *sink [[buffer(1)]],\n"
"                          constant uint      &n_blk [[buffer(2)]],\n"
"                          constant uint      &store [[buffer(3)]],\n"
"                          constant uint      &blk  [[buffer(4)]],\n"
"                          uint gid [[thread_position_in_grid]],\n"
"                          uint gsz [[threads_per_grid]]) {\n"
"    uint a = 0;\n"
"    for (uint b = gid; b < n_blk; b += gsz) {\n"
"        device const uchar *p = src + (ulong)b * blk;\n"
"        for (uint k = 0; k < 16u; k++) a ^= (uint)p[k];\n"
"    }\n"
"    if (store) sink[gid & 1023u] = uint4(a);\n"
"}\n";

typedef struct {
    const char *name;
    const char *what;
} arm_desc;

static id<MTLDevice>              g_dev;
static id<MTLCommandQueue>        g_queue;
static id<MTLCommandQueue>        g_queue2;
static id<MTLComputePipelineState> g_stream;
static id<MTLComputePipelineState> g_fill;
static id<MTLComputePipelineState> g_blocks;
static id<MTLBuffer>              g_sink;

/* One dispatch covering `bytes` of `buf`, timed by the command buffer's own
 * GPU clock.  Returns seconds of GPU time. */
static double run_stream(id<MTLCommandQueue> q, id<MTLBuffer> buf,
                         uint64_t bytes, uint64_t offset, bool wait) {
    const uint32_t n_vec = (uint32_t)(bytes / 16u);
    const uint32_t store = 0u;

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_stream];
    [enc setBuffer:buf offset:(NSUInteger)offset atIndex:0];
    [enc setBuffer:g_sink offset:0 atIndex:1];
    [enc setBytes:&n_vec length:sizeof(n_vec) atIndex:2];
    [enc setBytes:&store length:sizeof(store) atIndex:3];

    /* Enough threads that every core has many simdgroups resident; the
     * grid-stride loop absorbs whatever the total turns out to be. */
    const NSUInteger tg = 256;
    const NSUInteger threads = tg * 8192;
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    if (!wait) return 0.0;
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "  dispatch failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return -1.0;
    }
    return cb.GPUEndTime - cb.GPUStartTime;
}

static void gpu_fill(id<MTLBuffer> buf, uint64_t bytes) {
    const uint32_t n_vec = (uint32_t)(bytes / 16u);
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_fill];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&n_vec length:sizeof(n_vec) atIndex:1];
    [enc dispatchThreads:MTLSizeMake(256 * 4096, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

/* Best-of-N GPU time, reported as GB/s.  Best rather than mean: we are after
 * the ceiling, and a slow sample only ever means interference. */
static void report(const char *name, const char *what,
                   id<MTLBuffer> buf, uint64_t bytes, int iters) {
    if (!buf) {
        printf("  %-16s  %-34s  SKIPPED (allocation failed)\n", name, what);
        return;
    }
    (void)run_stream(g_queue, buf, bytes, 0, true); /* warm: page-in, clocks up */

    double best = 1e30, sum = 0.0;
    for (int i = 0; i < iters; i++) {
        const double t = run_stream(g_queue, buf, bytes, 0, true);
        if (t < 0.0) return;
        if (t < best) best = t;
        sum += t;
    }
    const double gbs_best = (double)bytes / best / 1e9;
    const double gbs_mean = (double)bytes / (sum / iters) / 1e9;
    printf("  %-16s  %-34s  %7.1f GB/s (mean %7.1f)\n",
           name, what, gbs_best, gbs_mean);
}

/* Fault a mapping in from `nthreads` threads over interleaved stripes.
 * If the OS places pages by first touch, spreading the touches spreads the
 * pages -- and on a two-die part that is the difference between one memory
 * controller and both.  This arm is the cheap fix if it works: the loader
 * would prefault the same way. */
typedef struct {
    const volatile char *base;
    uint64_t bytes;
    uint64_t stride;
    uint64_t start;
    uint64_t acc;
} fault_arg;

static void *fault_stripe(void *p) {
    fault_arg *a = (fault_arg *)p;
    uint64_t acc = 0;
    for (uint64_t off = a->start; off < a->bytes; off += a->stride) acc += (uint64_t)a->base[off];
    a->acc = acc;
    return NULL;
}

static void fault_in(void *base, uint64_t bytes, int nthreads) {
    const uint64_t page = 16384; /* Apple silicon */
    if (nthreads <= 1) {
        volatile uint64_t acc = 0;
        for (uint64_t off = 0; off < bytes; off += page) acc += ((volatile char *)base)[off];
        (void)acc;
        return;
    }
    pthread_t th[64];
    fault_arg args[64];
    if (nthreads > 64) nthreads = 64;
    for (int i = 0; i < nthreads; i++) {
        args[i] = (fault_arg){ (const volatile char *)base, bytes, page * (uint64_t)nthreads,
                               page * (uint64_t)i, 0 };
        pthread_create(&th[i], NULL, fault_stripe, &args[i]);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const double want_gib = argc > 1 ? atof(argv[1]) : 4.0;
        const int iters = argc > 2 ? atoi(argv[2]) : 10;
        if (want_gib <= 0.0 || iters <= 0) {
            fprintf(stderr, "usage: %s [size_gib] [iters]\n", argv[0]);
            return 2;
        }

        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { fprintf(stderr, "no Metal device\n"); return 1; }

        uint64_t bytes = (uint64_t)(want_gib * 1024.0 * 1024.0 * 1024.0);
        bytes &= ~(uint64_t)(16384 - 1);
        if (bytes > (uint64_t)g_dev.maxBufferLength) {
            bytes = (uint64_t)g_dev.maxBufferLength & ~(uint64_t)(16384 - 1);
            printf("note: clamped to device maxBufferLength\n");
        }
        /* n_vec is a uint in the kernel: 16 B per lane caps one dispatch at 64 GiB. */
        if (bytes / 16u > 0xFFFFFFFFull) bytes = 0xFFFFFFFFull * 16ull;

        printf("ds4 U6 membw: %s, %.2f GiB per arm, %d timed iters\n",
               [[g_dev name] UTF8String], (double)bytes / (1024.0 * 1024.0 * 1024.0), iters);
        printf("  maxBufferLength %.2f GiB, unified=%d\n",
               (double)g_dev.maxBufferLength / (1024.0 * 1024.0 * 1024.0),
               (int)g_dev.hasUnifiedMemory);

        NSError *err = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:kSource
                                                 options:[MTLCompileOptions new]
                                                   error:&err];
        if (!lib) {
            fprintf(stderr, "shader compile failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            return 1;
        }
        g_stream = [g_dev newComputePipelineStateWithFunction:
                        [lib newFunctionWithName:@"stream_read"] error:&err];
        g_fill = [g_dev newComputePipelineStateWithFunction:
                      [lib newFunctionWithName:@"fill_buf"] error:&err];
        g_blocks = [g_dev newComputePipelineStateWithFunction:
                        [lib newFunctionWithName:@"stream_blocks"] error:&err];
        if (!g_stream || !g_fill) {
            fprintf(stderr, "pipeline failed: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }
        g_queue = [g_dev newCommandQueue];
        g_queue2 = [g_dev newCommandQueue];
        g_sink = [g_dev newBufferWithLength:1024 * 16 options:MTLResourceStorageModeShared];

        printf("\n  %-16s  %-34s  %s\n", "arm", "allocation", "read bandwidth");
        printf("  %-16s  %-34s  %s\n", "----------------",
               "----------------------------------", "--------------");

        /* ---- Metal-allocated: the driver chooses placement ---- */
        id<MTLBuffer> shared = [g_dev newBufferWithLength:(NSUInteger)bytes
                                                  options:MTLResourceStorageModeShared];
        if (shared) gpu_fill(shared, bytes);
        report("metal-shared", "newBufferWithLength, Shared", shared, bytes, iters);

        shared = nil; /* free before the next arm: peak stays ~2x arm size */

        /* ---- Wrapped host memory: the VM chooses placement (ds4's path) ---- */
        void *anon = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
        id<MTLBuffer> anonbuf = nil;
        if (anon != MAP_FAILED) {
            memset(anon, 1, (size_t)bytes);
            anonbuf = [g_dev newBufferWithBytesNoCopy:anon
                                               length:(NSUInteger)bytes
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
        }
        report("mmap-anon", "NoCopy over MAP_ANON", anonbuf, bytes, iters);
        if (anon != MAP_FAILED) { anonbuf = nil; munmap(anon, (size_t)bytes); anon = MAP_FAILED; }

        /* The engine's actual path.  Prefer a real file (the GGUF on the rig)
         * so this arm is literally what ds4 does at load. */
        const char *path = getenv("BENCH_MEMBW_FILE");
        char tmpl[] = "/tmp/ds4_membw_XXXXXX";
        int fd = -1;
        bool made_temp = false;
        if (path && path[0]) {
            fd = open(path, O_RDONLY);
            if (fd < 0) fprintf(stderr, "  cannot open %s: %s\n", path, strerror(errno));
        } else {
            fd = mkstemp(tmpl);
            if (fd >= 0) {
                made_temp = true;
                if (ftruncate(fd, (off_t)bytes) != 0) {
                    fprintf(stderr, "  ftruncate failed: %s\n", strerror(errno));
                    close(fd); fd = -1;
                } else {
                    /* Write real bytes: a sparse file reads as shared zero
                     * pages and would measure nothing. */
                    void *w = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE,
                                   MAP_SHARED, fd, 0);
                    if (w == MAP_FAILED) { close(fd); fd = -1; }
                    else { memset(w, 2, (size_t)bytes); msync(w, (size_t)bytes, MS_SYNC);
                           munmap(w, (size_t)bytes); }
                }
            }
        }

        uint64_t fbytes = bytes;
        if (fd >= 0) {
            struct stat st;
            if (fstat(fd, &st) == 0 && (uint64_t)st.st_size < fbytes) {
                fbytes = (uint64_t)st.st_size & ~(uint64_t)(16384 - 1);
                printf("  note: file is %.2f GiB, file arms use that\n",
                       (double)fbytes / (1024.0 * 1024.0 * 1024.0));
            }
        }

        for (int pass = 0; pass < 3 && fd >= 0; pass++) {
            /* pass 0: single-threaded fault-in, hazard-tracked  (ds4 today)
             * pass 1: single-threaded fault-in, untracked       (DS4_METAL_MODEL_UNTRACKED)
             * pass 2: parallel fault-in, hazard-tracked         (the candidate fix) */
            void *m = mmap(NULL, (size_t)fbytes, PROT_READ, MAP_SHARED, fd, 0);
            if (m == MAP_FAILED) { fprintf(stderr, "  mmap failed\n"); break; }
            fault_in(m, fbytes, pass == 2 ? 16 : 1);

            MTLResourceOptions opt = MTLResourceStorageModeShared;
            if (pass == 1) opt |= MTLResourceHazardTrackingModeUntracked;
            id<MTLBuffer> fb = [g_dev newBufferWithBytesNoCopy:m
                                                        length:(NSUInteger)fbytes
                                                       options:opt
                                                   deallocator:nil];
            report(pass == 0 ? "mmap-file" : (pass == 1 ? "mmap-untracked" : "mmap-parallel"),
                   pass == 0 ? "NoCopy MAP_SHARED  <- ds4 today"
                             : (pass == 1 ? "NoCopy MAP_SHARED, untracked"
                                          : "NoCopy MAP_SHARED, 16-thread fault"),
                   fb, fbytes, iters);
            fb = nil;
            munmap(m, (size_t)fbytes);
        }
        if (fd >= 0) close(fd);
        if (made_temp) unlink(tmpl);

        /* ---- Metal-allocated, GPU-local: the driver chooses placement ---- */
        id<MTLBuffer> priv = [g_dev newBufferWithLength:(NSUInteger)bytes
                                                options:MTLResourceStorageModePrivate];
        if (priv) gpu_fill(priv, bytes);
        report("metal-private", "newBufferWithLength, Private", priv, bytes, iters);

        /* ---- Concurrency: does one dispatch simply not saturate? ---- */
        if (priv) {
            id<MTLBuffer> priv2 = [g_dev newBufferWithLength:(NSUInteger)bytes
                                                     options:MTLResourceStorageModePrivate];
            if (priv2) {
                gpu_fill(priv2, bytes);
                (void)run_stream(g_queue, priv, bytes, 0, true);
                double best = 1e30;
                for (int i = 0; i < iters; i++) {
                    /* Two independent buffers on two queues, overlapped; wall
                     * time bounds the pair, so aggregate GB/s is 2*bytes/wall. */
                    const double t0 = [[NSProcessInfo processInfo] systemUptime];
                    id<MTLCommandBuffer> a = [g_queue commandBuffer];
                    id<MTLComputeCommandEncoder> ea = [a computeCommandEncoder];
                    const uint32_t nv = (uint32_t)(bytes / 16u), st = 0u;
                    [ea setComputePipelineState:g_stream];
                    [ea setBuffer:priv offset:0 atIndex:0];
                    [ea setBuffer:g_sink offset:0 atIndex:1];
                    [ea setBytes:&nv length:4 atIndex:2];
                    [ea setBytes:&st length:4 atIndex:3];
                    [ea dispatchThreads:MTLSizeMake(256 * 8192, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [ea endEncoding];
                    [a commit];

                    id<MTLCommandBuffer> b = [g_queue2 commandBuffer];
                    id<MTLComputeCommandEncoder> eb = [b computeCommandEncoder];
                    [eb setComputePipelineState:g_stream];
                    [eb setBuffer:priv2 offset:0 atIndex:0];
                    [eb setBuffer:g_sink offset:0 atIndex:1];
                    [eb setBytes:&nv length:4 atIndex:2];
                    [eb setBytes:&st length:4 atIndex:3];
                    [eb dispatchThreads:MTLSizeMake(256 * 8192, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [eb endEncoding];
                    [b commit];

                    [a waitUntilCompleted];
                    [b waitUntilCompleted];
                    const double wall = [[NSProcessInfo processInfo] systemUptime] - t0;
                    if (wall < best) best = wall;
                }
                printf("  %-16s  %-34s  %7.1f GB/s (aggregate, wall)\n",
                       "two-queue", "2x Private on 2 queues, overlapped",
                       (double)(2 * bytes) / best / 1e9);
                priv2 = nil;
            }
        }
        priv = nil;

        /* ---- U8: block-stride layout cost, aligned vs MXFP4's 17 ---- */
        {
            id<MTLBuffer> b = [g_dev newBufferWithLength:(NSUInteger)bytes
                                                 options:MTLResourceStorageModePrivate];
            if (b) {
                gpu_fill(b, bytes);
                printf("\n  block-stride arms (payload is 16 B either way; only the stride differs)\n");
                for (int bi = 0; bi < 2; bi++) {
                    const uint32_t blk = bi == 0 ? 16u : 17u;
                    const uint32_t n_blk = (uint32_t)(bytes / blk);
                    const uint32_t store = 0u;
                    double best = 1e30;
                    for (int i = -1; i < iters; i++) {
                        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                        [e setComputePipelineState:g_blocks];
                        [e setBuffer:b offset:0 atIndex:0];
                        [e setBuffer:g_sink offset:0 atIndex:1];
                        [e setBytes:&n_blk length:4 atIndex:2];
                        [e setBytes:&store length:4 atIndex:3];
                        [e setBytes:&blk length:4 atIndex:4];
                        [e dispatchThreads:MTLSizeMake(256 * 8192, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [e endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        const double t = cb.GPUEndTime - cb.GPUStartTime;
                        if (i >= 0 && t < best) best = t;
                    }
                    /* Payload is what a dequant kernel would consume: 16 B/block. */
                    printf("  %-16s  %-34s  %7.1f GB/s\n",
                           blk == 16u ? "blk-16 aligned" : "blk-17 mxfp4",
                           blk == 16u ? "16 B payload, stride 16 (aligned)"
                                      : "16 B payload, stride 17 (straddles)",
                           (double)((uint64_t)n_blk * 16ull) / best / 1e9);
                }
                b = nil;
            }
        }

        printf("\nReading the result:\n"
               "  metal-private >> mmap-file      -> placement; fix is in the loader,\n"
               "                                     not the kernels.\n"
               "  mmap-parallel >> mmap-file      -> first-touch placement; the loader\n"
               "                                     should prefault in parallel.\n"
               "  two-queue     >> everything     -> one dispatch cannot saturate; split\n"
               "                                     work across queues.\n"
               "  all within a few percent        -> the ceiling is real. On an M2 Ultra\n"
               "                                     that means ~400 of a nominal 800.\n");
    }
    return 0;
}
