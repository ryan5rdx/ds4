/* LENS: why prefill fills the machine and decode does not — the row question,
 * asked of ds4's OWN production kernels rather than a transcription.
 *
 * `ds4_gpu_matmul_q8_0_tensor` picks the decode matvec at n_tok == 1
 * (kernel_mul_mv_q8_0_f32, ds4_metal.m:19276-19300), the tuned multi-row
 * mul_mv_ext family for 2 <= n_tok <= DS4_METAL_Q8_MV_EXT_MAX_TOKENS (16 by
 * default, ds4_metal.m:19303-19340), and the TensorOps/GEMM path above that.
 * So the whole prefill/decode row ladder is already implemented and reachable
 * from one call.
 *
 * The question: prefill amortises each weight byte over 4096 rows.  What does
 * the SECOND row cost a decode-shaped matvec, and the fourth, and the eighth?
 * If n_tok = 4 costs materially less than 4x n_tok = 1, then every mechanism
 * that manufactures rows for a single stream -- multi-token prediction,
 * speculative drafting, multi-slot batching -- is buying prefill's advantage
 * at the only price decode can pay, and the break-even acceptance rate follows
 * directly from this curve.
 *
 * Shapes are the real per-rank TP2 decode projections (speed-bench/
 * tp_decode_investigation.md section 3):
 *   attn_q_a        4096 -> 1024
 *   attn_kv         4096 ->  512
 *   attn_q_b        1024 -> 16384   (head half)
 *   attn_output_a   4096 -> 4096    (4 of 8 groups)
 *   attn_output_b   4096 -> 4096    (k-slice)
 *   ffn_gate_shexp  4096 -> 1024    (shared half)
 *
 * Weights are a scratch file so no GGUF is needed.  Each shape gets its own
 * region and the sweep is large enough to defeat cache; run twice and use the
 * second run, as with the other harnesses here (the first faults in the mmap).
 *
 *   make tests/bench_decode_rows
 *   ./tests/bench_decode_rows
 */

#define _DARWIN_C_SOURCE

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); return 1; } } while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

typedef struct { const char *name; uint32_t in_dim; uint32_t out_dim; } shape;

int main(int argc, char **argv) {
    const int reps  = argc > 1 ? atoi(argv[1]) : 7;
    const int nrep  = argc > 2 ? atoi(argv[2]) : 24;   /* matvecs per timed batch */
    const int maxt  = argc > 3 ? atoi(argv[3]) : 8;

    CHECK(ds4_gpu_init(), "gpu init");

    const shape shapes[] = {
        { "attn_q_a      4096->1024",  4096,  1024 },
        { "attn_kv       4096-> 512",  4096,   512 },
        { "attn_q_b      1024->16384", 1024, 16384 },
        { "attn_output_a 4096->4096",  4096,  4096 },
        { "shared_gate   4096->1024",  4096,  1024 },
    };
    const int nshape = (int)(sizeof(shapes) / sizeof(shapes[0]));

    /* One mmap holding nrep distinct weight copies of the widest shape, so the
     * timed batch never re-reads the same bytes and cache cannot flatter the
     * multi-row arms. */
    uint64_t widest = 0;
    for (int s = 0; s < nshape; s++) {
        const uint64_t rb = ((uint64_t)shapes[s].in_dim / 32u) * 34u;
        const uint64_t wb = rb * shapes[s].out_dim;
        if (wb > widest) widest = wb;
    }
    const uint64_t map_size = ((widest * (uint64_t)nrep) + 0xFFFFFull) & ~0xFFFFFull;
    void *map = mmap(NULL, (size_t)map_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(map != MAP_FAILED, "map mmap");
    memset(map, 0x11, (size_t)map_size);
    CHECK(ds4_gpu_set_model_map(map, map_size), "set_model_map");
    printf("weights: %.2f GiB of scratch, %d distinct copies per timed batch\n",
           map_size / 1073741824.0, nrep);

    printf("\n%-26s %6s %11s %12s %11s\n",
           "shape", "n_tok", "us/matvec", "us/token", "vs 1 token");
    for (int s = 0; s < nshape; s++) {
        const uint32_t in_dim = shapes[s].in_dim, out_dim = shapes[s].out_dim;
        const uint64_t rb = ((uint64_t)in_dim / 32u) * 34u;
        const uint64_t wb = rb * out_dim;
        ds4_gpu_tensor *x   = ds4_gpu_tensor_alloc((uint64_t)in_dim  * maxt * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * maxt * sizeof(float));
        CHECK(x && out, "tensor alloc");
        float *hx = malloc((size_t)in_dim * maxt * sizeof(float));
        CHECK(hx, "host alloc");
        for (uint32_t i = 0; i < in_dim * (uint32_t)maxt; i++)
            hx[i] = (float)((int)(i % 17) - 8) * 0.1f;
        CHECK(ds4_gpu_tensor_write(x, 0, hx, (size_t)in_dim * maxt * sizeof(float)), "write x");

        double base = 0;
        for (int nt = 1; nt <= maxt; nt *= 2) {
            /* warm: compiles the PSO for this n_tok and faults the pages */
            ds4_gpu_begin_commands();
            for (int i = 0; i < nrep; i++)
                (void)ds4_gpu_matmul_q8_0_tensor(out, map, map_size,
                                                 (uint64_t)i * wb, in_dim, out_dim, x, nt);
            ds4_gpu_end_commands();
            ds4_gpu_synchronize();

            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                const double t0 = now_ms();
                ds4_gpu_begin_commands();
                for (int i = 0; i < nrep; i++)
                    (void)ds4_gpu_matmul_q8_0_tensor(out, map, map_size,
                                                     (uint64_t)i * wb, in_dim, out_dim, x, nt);
                ds4_gpu_end_commands();
                ds4_gpu_synchronize();
                const double t = (now_ms() - t0) * 1000.0 / nrep;
                if (t > 0 && t < best) best = t;
            }
            if (nt == 1) base = best;
            printf("%-26s %6d %11.3f %12.3f %10.2fx\n",
                   nt == 1 ? shapes[s].name : "", nt, best, best / nt, best / base);
        }
        free(hx);
    }
    printf("\n'vs 1 token' below n_tok is prefill's row amortisation available to decode.\n");
    munmap(map, (size_t)map_size);
    return 0;
}
