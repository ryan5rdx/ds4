/* CPU cost of one DSpark propose call, measured with ds4's own CPU kernels.
 *
 * Not a proposal to offload anything.  It answers one question the corpus has
 * never measured: what would a DSpark propose call cost on the CPU cores?
 *
 * The propose call, per ds4.c:33805 metal_graph_eval_dspark_stage_block and
 * ds4.c:34475 metal_graph_eval_dspark_base_logits, is per stage:
 *   - MLA attention over 5 draft rows (small; ignored here)
 *   - one routed MoE over 5 rows: IQ2_XXS gate+up, Q2_K down
 * followed once by the TARGET model's Q8_0 output head over 5 rows.
 *
 * This harness fills an anonymous arena with pseudorandom bytes, describes it
 * with fake ds4_tensor headers, and calls the same static kernels decode uses:
 *   matvec_iq2_xxs_expert_pair_prequant   (ds4.c:8003)
 *   matvec_q2_k_expert                    (ds4.c:8313)
 *   matvec_q8_0                           (ds4.c:7665)
 *
 *   cc -O3 -I. -DDS4_NO_GPU -Wno-unused-function tests/bench_cpu_draft_cost.c \
 *      -o tests/bench_cpu_draft_cost -lm -pthread
 */

#include "../ds4.c"

static double ms_now(void) { return now_sec() * 1000.0; }

static void fill_random(uint8_t *p, size_t n) {
    uint64_t s = 0x243f6a8885a308d3ull;
    uint64_t *q = (uint64_t *)p;
    for (size_t i = 0; i < n / 8; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        q[i] = s;
    }
}

typedef struct { const uint8_t *base; size_t n; uint64_t *acc; } bw_ctx;

static void bw_worker(void *vctx, uint64_t r0, uint64_t r1) {
    bw_ctx *c = vctx;
    const uint64_t *p = (const uint64_t *)c->base;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    uint64_t i = r0;
    for (; i + 8 <= r1; i += 8) {
        a0 += p[i + 0]; a1 += p[i + 1]; a2 += p[i + 2]; a3 += p[i + 3];
        a4 += p[i + 4]; a5 += p[i + 5]; a6 += p[i + 6]; a7 += p[i + 7];
    }
    for (; i < r1; i++) a0 += p[i];
    c->acc[0] += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

int main(int argc, char **argv) {
    const uint32_t n_experts = argc > 1 ? (uint32_t)atoi(argv[1]) : 32;
    const uint32_t n_embd = 4096, n_ff = 2048, n_vocab = 129280;

    /* Q8_0 output head: n_vocab rows of n_embd/32 blocks of 34 bytes. */
    const size_t head_bytes  = (size_t)n_vocab * (n_embd / 32) * 34;
    /* IQ2_XXS: 66 bytes per 256 elems.  Q2_K: sizeof(block_q2_K) per 256. */
    const size_t iq2_row     = (size_t)(n_embd / QK_K) * sizeof(block_iq2_xxs);
    const size_t gate_bytes  = (size_t)n_ff * iq2_row * n_experts;
    const size_t up_bytes    = gate_bytes;
    const size_t q2k_row     = (size_t)(n_ff / QK_K) * sizeof(block_q2_K);
    const size_t down_bytes  = (size_t)n_embd * q2k_row * n_experts;

    const size_t total = head_bytes + gate_bytes + up_bytes + down_bytes;
    printf("arena: head %.1f MB  gate %.1f MB  up %.1f MB  down %.1f MB  total %.2f GB\n",
           head_bytes / 1e6, gate_bytes / 1e6, up_bytes / 1e6, down_bytes / 1e6,
           total / 1e9);
    printf("per-expert bytes: gate %.3f MB  up %.3f MB  down %.3f MB  sum %.3f MB\n",
           gate_bytes / 1e6 / n_experts, up_bytes / 1e6 / n_experts,
           down_bytes / 1e6 / n_experts,
           (gate_bytes + up_bytes + down_bytes) / 1e6 / n_experts);

    uint8_t *arena = mmap(NULL, total, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (arena == MAP_FAILED) { perror("mmap"); return 1; }
    fill_random(arena, total);

    ds4_model m = {0};
    m.fd = -1;
    m.map = arena;
    m.size = total;

    ds4_tensor head = {0};
    head.type = DS4_TENSOR_Q8_0; head.ndim = 2;
    head.dim[0] = n_embd; head.dim[1] = n_vocab;
    head.abs_offset = 0;

    ds4_tensor gate = {0};
    gate.type = DS4_TENSOR_IQ2_XXS; gate.ndim = 3;
    gate.dim[0] = n_embd; gate.dim[1] = n_ff; gate.dim[2] = n_experts;
    gate.abs_offset = head_bytes;

    ds4_tensor up = gate;
    up.abs_offset = head_bytes + gate_bytes;

    ds4_tensor down = {0};
    down.type = DS4_TENSOR_Q2_K; down.ndim = 3;
    down.dim[0] = n_ff; down.dim[1] = n_embd; down.dim[2] = n_experts;
    down.abs_offset = head_bytes + gate_bytes + up_bytes;

    float *x_embd = xmalloc(n_embd * sizeof(float));
    float *x_ff   = xmalloc(n_ff * sizeof(float));
    for (uint32_t i = 0; i < n_embd; i++) x_embd[i] = (float)((i % 17) - 8) * 0.01f;
    for (uint32_t i = 0; i < n_ff; i++)   x_ff[i]   = (float)((i % 13) - 6) * 0.01f;
    float *o_ff0 = xmalloc(n_ff * sizeof(float));
    float *o_ff1 = xmalloc(n_ff * sizeof(float));
    float *o_embd = xmalloc(n_embd * sizeof(float));
    float *logits = xmalloc((size_t)n_vocab * sizeof(float));

    block_q8_K *xq = xmalloc((size_t)(n_embd / QK_K) * sizeof(block_q8_K));
    ds4_quantize_row_q8_K(x_embd, xq, n_embd);

    ds4_threads_init();
    printf("threads: %u (DS4_THREADS overrides)\n", g_pool.n_threads);

    /* Arm 0: multithreaded streaming read of the whole arena. */
    {
        uint64_t acc = 0;
        bw_ctx c = { arena, total, &acc };
        const double t0 = ms_now();
        ds4_parallel_for(total / 8, bw_worker, &c);
        const double t1 = ms_now();
        printf("stream read  %8.2f ms  %7.1f GB/s   (checksum %llu)\n",
               t1 - t0, total / 1e6 / (t1 - t0), (unsigned long long)acc);
    }

    /* Arm 1: one routed expert (gate+up pair, then down). */
    {
        const int reps = 3;
        double best_pair = 1e18, best_down = 1e18;
        for (int r = 0; r < reps; r++) {
            double t0 = ms_now();
            for (uint32_t e = 0; e < n_experts; e++)
                matvec_iq2_xxs_expert_pair_prequant(o_ff0, o_ff1, &m, &gate, &up, xq, e);
            double t1 = ms_now();
            for (uint32_t e = 0; e < n_experts; e++)
                matvec_q2_k_expert(o_embd, &m, &down, x_ff, e);
            double t2 = ms_now();
            if (t1 - t0 < best_pair) best_pair = t1 - t0;
            if (t2 - t1 < best_down) best_down = t2 - t1;
        }
        const double per_expert = (best_pair + best_down) / n_experts;
        printf("gate+up x%u   %8.2f ms  %7.1f GB/s\n", n_experts, best_pair,
               (gate_bytes + up_bytes) / 1e6 / best_pair);
        printf("down    x%u   %8.2f ms  %7.1f GB/s\n", n_experts, best_down,
               down_bytes / 1e6 / best_down);
        printf("one expert   %8.3f ms  (%.3f MB)\n", per_expert,
               (gate_bytes + up_bytes + down_bytes) / 1e6 / n_experts);
    }

    /* Arm 2: the target model's Q8_0 vocabulary head, one row. */
    {
        double best = 1e18;
        for (int r = 0; r < 3; r++) {
            const double t0 = ms_now();
            matvec_q8_0(logits, &m, &head, x_embd);
            const double t1 = ms_now();
            if (t1 - t0 < best) best = t1 - t0;
        }
        printf("q8_0 head    %8.2f ms  %7.1f GB/s  (%.1f MB, 1 row)\n",
               best, head_bytes / 1e6 / best, head_bytes / 1e6);
    }

    ds4_threads_shutdown();
    return 0;
}
