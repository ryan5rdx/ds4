#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MXFP4_TYPE 39u
#define QK_MXFP4 32u
#define N_TOTAL_EXPERT 8u
#define N_EXPERT 6u
#define DIM 256u
#define BATCH_TOKENS 48u
#define QK_Q8_0 32u
#define ATTN_GROUPS 8u
#define ATTN_RANK 32u
#define TP_TEST_ROWS 5u
/* The first 32768 rows fill the capped 128 x 256 dispatch.  Another 257 rows
 * exercise the strided traversal in two different threadgroups. */
#define MARKOV_VOCAB 33025u
#define MARKOV_RANK 256u
#define MARKOV_STEPS 3u

typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2u];
} block_mxfp4;

typedef struct {
    _Float16 d;
    int8_t qs[QK_Q8_0];
} block_q8_0;

typedef char block_q8_0_must_be_34_bytes[
    sizeof(block_q8_0) == 34u ? 1 : -1];

static const float mxfp4_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
};

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static float e8m0_to_f32(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : (uint32_t)e << 23u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float dot_mxfp4(const block_mxfp4 *row, const float *x) {
    float sum = 0.0f;
    for (uint32_t block = 0; block < DIM / QK_MXFP4; block++) {
        const block_mxfp4 *b = row + block;
        const float scale = e8m0_to_f32(b->e);
        for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
            const uint8_t q = b->qs[i];
            sum += scale * mxfp4_values[q & 15u] *
                   x[block * QK_MXFP4 + i];
            sum += scale * mxfp4_values[q >> 4u] *
                   x[block * QK_MXFP4 + i + QK_MXFP4 / 2u];
        }
    }
    return sum;
}

static void fill_matrix(block_mxfp4 *matrix, uint32_t salt) {
    const uint32_t blocks_per_row = DIM / QK_MXFP4;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < DIM; row++) {
            block_mxfp4 *blocks = matrix +
                ((uint64_t)expert * DIM + row) * blocks_per_row;
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                block_mxfp4 *b = blocks + block;
                b->e = (uint8_t)(120u +
                    ((salt + expert * 3u + row + block * 5u) % 6u));
                for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
                    const uint8_t lo = (uint8_t)(
                        (salt + expert * 7u + row * 3u + block + i) & 15u);
                    const uint8_t hi = (uint8_t)(
                        (salt * 3u + expert + row * 5u + block * 7u + i * 3u) & 15u);
                    b->qs[i] = (uint8_t)(lo | (hi << 4u));
                }
            }
        }
    }
}

static void fill_q8_matrix(block_q8_0 *matrix,
                           uint32_t rows,
                           uint32_t cols,
                           uint32_t salt) {
    const uint32_t blocks_per_row = cols / QK_Q8_0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            block_q8_0 *b = matrix +
                (uint64_t)row * blocks_per_row + block;
            b->d = (_Float16)((float)(1u +
                ((salt + row * 3u + block * 5u) % 7u)) / 512.0f);
            for (uint32_t i = 0; i < QK_Q8_0; i++) {
                b->qs[i] = (int8_t)((int32_t)(
                    (salt + row * 11u + block * 7u + i * 5u) % 31u) - 15);
            }
        }
    }
}

static float dot_q8(const block_q8_0 *row,
                    const float *x,
                    uint32_t cols) {
    float sum = 0.0f;
    for (uint32_t block = 0; block < cols / QK_Q8_0; block++) {
        const block_q8_0 *b = row + block;
        const float d = (float)b->d;
        for (uint32_t i = 0; i < QK_Q8_0; i++) {
            sum += d * (float)b->qs[i] * x[block * QK_Q8_0 + i];
        }
    }
    return sum;
}

static int compare_values(const char *name, const float *actual,
                          const float *expected, uint64_t count,
                          float tolerance) {
    float max_abs = 0.0f;
    double sum_sq = 0.0;
    uint64_t max_index = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) {
            fprintf(stderr, "MXFP4 Metal %s non-finite value at %llu\n",
                    name, (unsigned long long)i);
            return 0;
        }
        const float error = fabsf(actual[i] - expected[i]);
        if (error > max_abs) {
            max_abs = error;
            max_index = i;
        }
        sum_sq += (double)error * error;
    }
    const double rms = sqrt(sum_sq / (double)count);
    fprintf(stderr,
            "MXFP4 Metal %-4s max_abs=%g rms=%g at=%llu\n",
            name, max_abs, rms, (unsigned long long)max_index);
    return max_abs <= tolerance;
}

/* The k-slice at prefill row counts.
 *
 * The existing TP case runs 5 rows, which is BELOW the mv/mm threshold
 * (use_mv = n_rows <= 8), so it never reaches the tiled path and passed
 * identically before and after that path existed. Prefill runs thousands of
 * rows, and taking the matvec there re-reads the weight per row -- which is what
 * made the S6b prefill split lose: full-width tiled GEMM versus half-width
 * matvec, with the lost cross-row reuse swamping the halved arithmetic.
 *
 * Straddle the threshold and check the property that matters: two k-slices must
 * sum to the unsplit result, at every row count, on whichever kernel each side
 * picks. */
static int run_kslice_rows(void) {
    enum { KIN = 512, KOUT = 128, KHALF = KIN / 2 };
    const uint64_t krow = (uint64_t)(KIN / QK_Q8_0) * sizeof(block_q8_0);
    const uint64_t kbytes = (uint64_t)KOUT * krow;
    uint8_t *kmodel = calloc(1, (size_t)kbytes + 4096);
    if (!kmodel) return 0;

    fill_q8_matrix((block_q8_0 *)kmodel, KOUT, KIN, 24601u);
    uint32_t seed = 24601u;

    int ok = ds4_gpu_set_model_map(kmodel, (uint64_t)kbytes + 4096) != 0;
    const uint32_t counts[] = { 1u, 8u, 9u, 32u, 128u };
    int failures = 0;
    /* Correctness alone cannot see a silent fallback -- both kernels give the
     * right answer, one just re-reads the weight per row. Assert the dispatch
     * actually crossed to the tiled path, or this test would keep passing while
     * the optimisation it guards quietly stopped happening. */
    const uint64_t tiled_before = ds4_gpu_kslice_tiled_count();

    for (size_t ci = 0; ok && ci < sizeof(counts) / sizeof(counts[0]); ci++) {
        const uint32_t R = counts[ci];
        float *xf = malloc((size_t)R * KIN * sizeof(float));
        float *xl = malloc((size_t)R * KHALF * sizeof(float));
        float *hf = malloc((size_t)R * KOUT * sizeof(float));
        float *hs = malloc((size_t)R * KOUT * sizeof(float));
        if (!xf || !xl || !hf || !hs) { failures++; free(xf); free(xl); free(hf); free(hs); break; }
        for (uint64_t i = 0; i < (uint64_t)R * KIN; i++) {
            seed = seed * 1664525u + 1013904223u;
            xf[i] = (float)((int32_t)(seed >> 8) - 8388608) / 8388608.0f;
        }

        ds4_gpu_tensor *tF = ds4_gpu_tensor_alloc((uint64_t)R * KIN * sizeof(float));
        ds4_gpu_tensor *tL = ds4_gpu_tensor_alloc((uint64_t)R * KHALF * sizeof(float));
        ds4_gpu_tensor *oF = ds4_gpu_tensor_alloc((uint64_t)R * KOUT * sizeof(float));
        ds4_gpu_tensor *p0 = ds4_gpu_tensor_alloc((uint64_t)R * KOUT * sizeof(float));
        ds4_gpu_tensor *p1 = ds4_gpu_tensor_alloc((uint64_t)R * KOUT * sizeof(float));
        int step = tF && tL && oF && p0 && p1;
        step = step && ds4_gpu_tensor_write(tF, 0, xf, (uint64_t)R * KIN * sizeof(float));
        step = step && ds4_gpu_matmul_quant_tensor(oF, kmodel, (uint64_t)kbytes + 4096,
                                                   0, 8u, KIN, KOUT, tF, R);
        for (int lane = 0; step && lane < 2; lane++) {
            for (uint32_t t = 0; t < R; t++) {
                memcpy(xl + (size_t)t * KHALF,
                       xf + (size_t)t * KIN + (size_t)lane * KHALF,
                       KHALF * sizeof(float));
            }
            step = step && ds4_gpu_tensor_write(tL, 0, xl,
                                                (uint64_t)R * KHALF * sizeof(float));
            /* tL is built COMPACT above -- row t is xf[t*KIN + lane*KHALF]
             * copied to xl[t*KHALF] -- so the activation geometry is
             * (KHALF, 0). k_off still offsets the WEIGHT. */
            step = step && ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                    lane ? p1 : p0, kmodel, (uint64_t)kbytes + 4096, 0,
                    KIN, KOUT, (uint64_t)lane * KHALF, KHALF, tL,
                    KHALF, 0, R);
        }
        step = step && ds4_gpu_add_tensor(p0, p0, p1, (uint32_t)((uint64_t)R * KOUT));
        step = step && ds4_gpu_tensor_read(oF, 0, hf, (uint64_t)R * KOUT * sizeof(float));
        step = step && ds4_gpu_tensor_read(p0, 0, hs, (uint64_t)R * KOUT * sizeof(float));

        if (!step) {
            fprintf(stderr, "  kslice rows=%u: dispatch failed\n", R);
            failures++;
        } else {
            double sa = 0.0, sd = 0.0;
            for (uint64_t i = 0; i < (uint64_t)R * KOUT; i++) {
                const double d = (double)hf[i] - (double)hs[i];
                sa += (double)hf[i] * hf[i];
                sd += d * d;
            }
            const double rel = sa > 0.0 ? sqrt(sd / sa) : 0.0;
            /* A reduction split, so last-bit only.  1e-5 is loose enough for
             * either kernel and far tighter than a wrong k offset or a
             * transposed dispatch grid, both of which land at O(1). */
            if (!(rel < 1e-5)) {
                fprintf(stderr,
                        "  kslice rows=%u: halves do not sum to the whole "
                        "(rms-rel %.3e)\n", R, rel);
                failures++;
            }
        }
        ds4_gpu_tensor_free(p1); ds4_gpu_tensor_free(p0);
        ds4_gpu_tensor_free(oF); ds4_gpu_tensor_free(tL); ds4_gpu_tensor_free(tF);
        free(xf); free(xl); free(hf); free(hs);
    }
    free(kmodel);
    if (!ok) return 0;
    /* rows 32 and 128 are above the mv/mm threshold with k_cnt = 256, which is
     * not a multiple of 128, so the generic rule sends them to the tiled path. */
    const uint64_t tiled = ds4_gpu_kslice_tiled_count() - tiled_before;
    if (tiled == 0) {
        fprintf(stderr,
                "  kslice: no tiled dispatch happened -- every row count fell "
                "back to the matvec\n");
        failures++;
    }
    return failures == 0;
}

int main(void) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes =
        (DIM / QK_MXFP4) * sizeof(block_mxfp4);
    const uint64_t expert_bytes = DIM * row_bytes;
    const uint64_t tensor_bytes = N_TOTAL_EXPERT * expert_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = align_up(tensor_bytes, page);
    const uint64_t down_offset = align_up(up_offset + tensor_bytes, page);
    const uint64_t q8_attn_row_bytes =
        (DIM / QK_Q8_0) * sizeof(block_q8_0);
    const uint64_t q8_low_dim = (uint64_t)ATTN_GROUPS * ATTN_RANK;
    const uint64_t attn_a_bytes = q8_low_dim * q8_attn_row_bytes;
    const uint64_t attn_b_bytes = DIM *
        (q8_low_dim / QK_Q8_0) * sizeof(block_q8_0);
    const uint64_t markov_row_bytes =
        (MARKOV_RANK / QK_Q8_0) * sizeof(block_q8_0);
    const uint64_t markov_bytes = MARKOV_VOCAB * markov_row_bytes;
    const uint64_t attn_a_offset =
        align_up(down_offset + tensor_bytes, page);
    const uint64_t attn_b_offset =
        align_up(attn_a_offset + attn_a_bytes, page);
    const uint64_t markov_w1_offset =
        align_up(attn_b_offset + attn_b_bytes, page);
    const uint64_t markov_w2_offset =
        align_up(markov_w1_offset + markov_bytes, page);
    const uint64_t model_size =
        align_up(markov_w2_offset + markov_bytes, page);
    void *model = NULL;
    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        fprintf(stderr, "MXFP4 Metal test model allocation failed\n");
        return 1;
    }
    memset(model, 0, (size_t)model_size);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + gate_offset), 1u);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + up_offset), 5u);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + down_offset), 9u);
    fill_q8_matrix(
        (block_q8_0 *)((uint8_t *)model + attn_a_offset),
        ATTN_GROUPS * ATTN_RANK, DIM, 13u);
    fill_q8_matrix(
        (block_q8_0 *)((uint8_t *)model + attn_b_offset),
        DIM, (uint32_t)q8_low_dim, 17u);
    fill_q8_matrix(
        (block_q8_0 *)((uint8_t *)model + markov_w1_offset),
        MARKOV_VOCAB, MARKOV_RANK, 19u);
    fill_q8_matrix(
        (block_q8_0 *)((uint8_t *)model + markov_w2_offset),
        MARKOV_VOCAB, MARKOV_RANK, 23u);

    float x[DIM];
    int32_t selected[N_EXPERT] = { 0, 2, 3, 5, 6, 7 };
    float weights[N_EXPERT] = { 0.24f, 0.20f, 0.18f, 0.16f, 0.12f, 0.10f };
    for (uint32_t i = 0; i < DIM; i++) {
        x[i] = (float)((int32_t)((i * 13u) % 31u) - 15) / 64.0f;
    }

    const uint64_t pair_count = (uint64_t)N_EXPERT * DIM;
    float *gate_ref = calloc((size_t)pair_count, sizeof(float));
    float *up_ref = calloc((size_t)pair_count, sizeof(float));
    float *mid_ref = calloc((size_t)pair_count, sizeof(float));
    float *out_ref = calloc(DIM, sizeof(float));
    float *gate_gpu = calloc((size_t)pair_count, sizeof(float));
    float *up_gpu = calloc((size_t)pair_count, sizeof(float));
    float *mid_gpu = calloc((size_t)pair_count, sizeof(float));
    float *experts_gpu = calloc((size_t)pair_count, sizeof(float));
    float *out_gpu = calloc(DIM, sizeof(float));
    float *gate_fast = calloc((size_t)pair_count, sizeof(float));
    float *up_fast = calloc((size_t)pair_count, sizeof(float));
    float *mid_fast = calloc((size_t)pair_count, sizeof(float));
    float *experts_fast = calloc((size_t)pair_count, sizeof(float));
    float *out_fast = calloc(DIM, sizeof(float));
    if (!gate_ref || !up_ref || !mid_ref || !out_ref ||
        !gate_gpu || !up_gpu || !mid_gpu || !experts_gpu || !out_gpu ||
        !gate_fast || !up_fast || !mid_fast || !experts_fast || !out_fast) {
        fprintf(stderr, "MXFP4 Metal host tensor allocation failed\n");
        return 1;
    }

    const block_mxfp4 *gate_matrix =
        (const block_mxfp4 *)((const uint8_t *)model + gate_offset);
    const block_mxfp4 *up_matrix =
        (const block_mxfp4 *)((const uint8_t *)model + up_offset);
    const block_mxfp4 *down_matrix =
        (const block_mxfp4 *)((const uint8_t *)model + down_offset);
    const uint64_t blocks_per_expert = expert_bytes / sizeof(block_mxfp4);
    const uint64_t blocks_per_row = row_bytes / sizeof(block_mxfp4);
    for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
        const uint32_t expert = (uint32_t)selected[slot];
        for (uint32_t row = 0; row < DIM; row++) {
            const uint64_t pair = (uint64_t)slot * DIM + row;
            gate_ref[pair] = dot_mxfp4(
                gate_matrix + (uint64_t)expert * blocks_per_expert +
                    (uint64_t)row * blocks_per_row,
                x);
            up_ref[pair] = dot_mxfp4(
                up_matrix + (uint64_t)expert * blocks_per_expert +
                    (uint64_t)row * blocks_per_row,
                x);
            const float g = fminf(gate_ref[pair], 7.0f);
            const float u = fmaxf(-7.0f, fminf(up_ref[pair], 7.0f));
            mid_ref[pair] = (g / (1.0f + expf(-g))) * u * weights[slot];
        }
    }
    for (uint32_t row = 0; row < DIM; row++) {
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            const uint32_t expert = (uint32_t)selected[slot];
            out_ref[row] += dot_mxfp4(
                down_matrix + (uint64_t)expert * blocks_per_expert +
                    (uint64_t)row * blocks_per_row,
                mid_ref + (uint64_t)slot * DIM);
        }
    }

    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_size);
    ok = ok && ds4_gpu_test_decode_pipeline_fast_lookup();
    if (ok) {
        fprintf(stderr,
                "MXFP4 Metal decode pipeline fast lookup inactive/populate/hit guards PASS\n");
    }
    ok = ok && ds4_gpu_test_decode_pipeline_fast_lookup_ext();
    if (ok) {
        fprintf(stderr,
                "MXFP4 Metal decode pipeline fast lookup ext (nsg+nxpsg) guards PASS\n");
    }
    uint16_t legacy_half_bits[256u * 16u];
    uint16_t lut_half_bits[256u * 16u];
    ok = ok && ds4_gpu_test_mxfp4_down_half_lut(
        legacy_half_bits, lut_half_bits);
    if (ok) {
        uint32_t mismatches = 0;
        for (uint32_t i = 0; i < 256u * 16u; i++) {
            if (legacy_half_bits[i] != lut_half_bits[i]) {
                if (mismatches < 16u) {
                    fprintf(stderr,
                            "MXFP4 Metal half-LUT raw-bit mismatch e=%u q=%u "
                            "legacy=0x%04x lut=0x%04x\n",
                            i >> 4u, i & 15u,
                            legacy_half_bits[i], lut_half_bits[i]);
                }
                mismatches++;
            }
        }
        if (mismatches != 0u) {
            fprintf(stderr,
                    "MXFP4 Metal half-LUT raw-bit mismatches=%u/4096\n",
                    mismatches);
            ok = 0;
        }
    }
    if (ok) {
        fprintf(stderr,
                "MXFP4 Metal half-LUT raw-bit A/B exact for all 4096 e/q pairs\n");
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_tensor *x_tensor = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *selected_tensor = ds4_gpu_tensor_alloc(sizeof(selected));
    ds4_gpu_tensor *weights_tensor = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *gate_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *up_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *mid_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *experts_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *out_tensor = ds4_gpu_tensor_alloc(DIM * sizeof(float));
    ok = ok && x_tensor && selected_tensor && weights_tensor && gate_tensor &&
         up_tensor && mid_tensor && experts_tensor && out_tensor;
    ok = ok && ds4_gpu_tensor_write(x_tensor, 0, x, sizeof(x));
    ok = ok && ds4_gpu_tensor_write(
        selected_tensor, 0, selected, sizeof(selected));
    ok = ok && ds4_gpu_tensor_write(
        weights_tensor, 0, weights, sizeof(weights));
    ok = ok && ds4_gpu_tensor_fill_f32(
        gate_tensor, -101.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(
        up_tensor, -102.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(
        mid_tensor, -103.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(
        experts_tensor, -104.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(out_tensor, -105.0f, DIM);
    ok = ok && ds4_gpu_routed_moe_one_tensor(
        out_tensor, gate_tensor, up_tensor, mid_tensor, experts_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_tensor, weights_tensor, N_TOTAL_EXPERT, N_EXPERT,
        7.0f, x_tensor, NULL, 0u, true);
    ok = ok && ds4_gpu_tensor_read(
        gate_tensor, 0, gate_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        up_tensor, 0, up_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        mid_tensor, 0, mid_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        experts_tensor, 0, experts_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        out_tensor, 0, out_gpu, DIM * sizeof(float));

    /* The routed-MXFP4 helper uses its precompiled pipeline globals, so it is
     * deliberately neutral to the fast mirror. Toggle the per-token hint
     * around two active runs and one inactive guard-tail run to prove unrelated
     * decode dispatches stay byte-exact. Poison and compare every writable
     * tensor byte-for-byte on all three runs. */
    bool fast_lookup_ok = ok;
    const int previous_fast_lookup =
        ds4_gpu_set_decode_pipeline_fast_lookup(1);
    for (uint32_t run = 0; run < 3u && fast_lookup_ok; run++) {
        if (run == 2u) {
            (void)ds4_gpu_set_decode_pipeline_fast_lookup(0);
        }
        fast_lookup_ok =
            ds4_gpu_tensor_fill_f32(gate_tensor, -101.0f, pair_count) &&
            ds4_gpu_tensor_fill_f32(up_tensor, -102.0f, pair_count) &&
            ds4_gpu_tensor_fill_f32(mid_tensor, -103.0f, pair_count) &&
            ds4_gpu_tensor_fill_f32(experts_tensor, -104.0f, pair_count) &&
            ds4_gpu_tensor_fill_f32(out_tensor, -105.0f, DIM) &&
            ds4_gpu_routed_moe_one_tensor(
                out_tensor, gate_tensor, up_tensor, mid_tensor, experts_tensor,
                model, model_size, gate_offset, up_offset, down_offset,
                MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
                expert_bytes, row_bytes, DIM, DIM, DIM,
                selected_tensor, weights_tensor, N_TOTAL_EXPERT, N_EXPERT,
                7.0f, x_tensor, NULL, 0u, true) &&
            ds4_gpu_tensor_read(
                gate_tensor, 0, gate_fast, pair_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                up_tensor, 0, up_fast, pair_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                mid_tensor, 0, mid_fast, pair_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                experts_tensor, 0, experts_fast,
                pair_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                out_tensor, 0, out_fast, DIM * sizeof(float));
        const bool gate_exact =
            memcmp(gate_fast, gate_gpu, pair_count * sizeof(float)) == 0;
        const bool up_exact =
            memcmp(up_fast, up_gpu, pair_count * sizeof(float)) == 0;
        const bool mid_exact =
            memcmp(mid_fast, mid_gpu, pair_count * sizeof(float)) == 0;
        const bool experts_exact =
            memcmp(experts_fast, experts_gpu,
                   pair_count * sizeof(float)) == 0;
        const bool out_exact =
            memcmp(out_fast, out_gpu, DIM * sizeof(float)) == 0;
        if (fast_lookup_ok &&
            (!gate_exact || !up_exact || !mid_exact ||
             !experts_exact || !out_exact)) {
            fprintf(stderr,
                    "MXFP4 Metal decode pipeline fast lookup byte-exact A/B mismatch "
                    "run=%u gate=%d up=%d mid=%d experts=%d out=%d\n",
                    run + 1u, gate_exact, up_exact, mid_exact,
                    experts_exact, out_exact);
            fast_lookup_ok = false;
        } else if (fast_lookup_ok) {
            fprintf(stderr,
                    "MXFP4 Metal decode pipeline fast lookup byte-exact A/B PASS run=%u mode=%s\n",
                    run + 1u, run < 2u ? "active" : "inactive");
        }
    }
    (void)ds4_gpu_set_decode_pipeline_fast_lookup(previous_fast_lookup);
    ok = ok && fast_lookup_ok;

    if (ok) {
        ok = compare_values("gate", gate_gpu, gate_ref, pair_count, 2.0e-5f) &&
             compare_values("up", up_gpu, up_ref, pair_count, 2.0e-5f) &&
             compare_values("mid", mid_gpu, mid_ref, pair_count, 2.0e-5f) &&
             compare_values("out", out_gpu, out_ref, DIM, 2.0e-4f);
    }

    /* The TP verifier keeps every attention row but splits the eight output
     * groups. Exercise both rank slices with a full-row-stride heads tensor,
     * compact low rows, and the matching out_b K window. The two partials
     * must reconstruct the unsplit projection for all five DSpark rows. */
    enum { TP_GROUPS = ATTN_GROUPS / 2u };
    const uint64_t heads_count =
        (uint64_t)TP_TEST_ROWS * ATTN_GROUPS * DIM;
    const uint64_t tp_low_count =
        (uint64_t)TP_TEST_ROWS * TP_GROUPS * ATTN_RANK;
    const uint64_t tp_heads_compact_count =
        (uint64_t)TP_TEST_ROWS * TP_GROUPS * DIM;
    const uint64_t tp_out_count = (uint64_t)TP_TEST_ROWS * DIM;
    float *tp_heads_host = calloc((size_t)heads_count, sizeof(float));
    float *tp_heads_compact_host = calloc(
        (size_t)tp_heads_compact_count, sizeof(float));
    float *tp_low_full_ref = calloc(
        (size_t)TP_TEST_ROWS * (size_t)q8_low_dim, sizeof(float));
    float *tp_low_expected = calloc((size_t)tp_low_count, sizeof(float));
    float *tp_low_actual = calloc((size_t)tp_low_count, sizeof(float));
    float *tp_out_expected = calloc((size_t)tp_out_count, sizeof(float));
    float *tp_out_actual = calloc((size_t)tp_out_count, sizeof(float));
    float *tp_out_sum = calloc((size_t)tp_out_count, sizeof(float));
    float *tp_out_full_ref = calloc((size_t)tp_out_count, sizeof(float));
    ds4_gpu_tensor *tp_heads_tensor = ds4_gpu_tensor_alloc(
        heads_count * sizeof(float));
    ds4_gpu_tensor *tp_heads_compact_tensor = ds4_gpu_tensor_alloc(
        tp_heads_compact_count * sizeof(float));
    ds4_gpu_tensor *tp_low_tensor = ds4_gpu_tensor_alloc(
        tp_low_count * sizeof(float));
    ds4_gpu_tensor *tp_out_tensor = ds4_gpu_tensor_alloc(
        tp_out_count * sizeof(float));
    ok = ok && tp_heads_host && tp_heads_compact_host &&
         tp_low_full_ref && tp_low_expected &&
         tp_low_actual && tp_out_expected && tp_out_actual && tp_out_sum &&
         tp_out_full_ref && tp_heads_tensor && tp_heads_compact_tensor &&
         tp_low_tensor && tp_out_tensor;
    for (uint64_t i = 0; i < heads_count; i++) {
        tp_heads_host[i] =
            (float)((int32_t)((i * 17u + i / DIM * 7u) % 53u) - 26) /
            128.0f;
    }
    const block_q8_0 *attn_a =
        (const block_q8_0 *)((const uint8_t *)model + attn_a_offset);
    const block_q8_0 *attn_b =
        (const block_q8_0 *)((const uint8_t *)model + attn_b_offset);
    const uint64_t attn_a_blocks_per_row = DIM / QK_Q8_0;
    const uint64_t attn_b_blocks_per_row = q8_low_dim / QK_Q8_0;
    for (uint32_t t = 0; t < TP_TEST_ROWS; t++) {
        for (uint32_t group = 0; group < ATTN_GROUPS; group++) {
            const float *head = tp_heads_host +
                ((uint64_t)t * ATTN_GROUPS + group) * DIM;
            for (uint32_t j = 0; j < ATTN_RANK; j++) {
                tp_low_full_ref[(uint64_t)t * q8_low_dim +
                                (uint64_t)group * ATTN_RANK + j] =
                    dot_q8(attn_a +
                               ((uint64_t)group * ATTN_RANK + j) *
                                   attn_a_blocks_per_row,
                           head,
                           DIM);
            }
        }
        for (uint32_t out = 0; out < DIM; out++) {
            tp_out_full_ref[(uint64_t)t * DIM + out] =
                dot_q8(attn_b + (uint64_t)out * attn_b_blocks_per_row,
                       tp_low_full_ref + (uint64_t)t * q8_low_dim,
                       (uint32_t)q8_low_dim);
        }
    }
    ok = ok && ds4_gpu_tensor_write(
        tp_heads_tensor, 0, tp_heads_host, heads_count * sizeof(float));
    for (uint32_t rank_id = 0; rank_id < 2u && ok; rank_id++) {
        const uint32_t group0 = rank_id * TP_GROUPS;
        const uint64_t k0 = (uint64_t)group0 * ATTN_RANK;
        for (uint32_t t = 0; t < TP_TEST_ROWS; t++) {
            memcpy(tp_heads_compact_host +
                       (uint64_t)t * TP_GROUPS * DIM,
                   tp_heads_host +
                       ((uint64_t)t * ATTN_GROUPS + group0) * DIM,
                   (size_t)TP_GROUPS * DIM * sizeof(float));
            memcpy(tp_low_expected +
                       (uint64_t)t * TP_GROUPS * ATTN_RANK,
                   tp_low_full_ref + (uint64_t)t * q8_low_dim + k0,
                   (size_t)TP_GROUPS * ATTN_RANK * sizeof(float));
            for (uint32_t out = 0; out < DIM; out++) {
                tp_out_expected[(uint64_t)t * DIM + out] =
                    dot_q8(attn_b +
                               (uint64_t)out * attn_b_blocks_per_row +
                               k0 / QK_Q8_0,
                           tp_low_full_ref +
                               (uint64_t)t * q8_low_dim + k0,
                           TP_GROUPS * ATTN_RANK);
            }
        }
        ok = ok && ds4_gpu_tensor_fill_f32(
            tp_low_tensor, -111.0f, tp_low_count);
        ok = ok && ds4_gpu_tensor_fill_f32(
            tp_out_tensor, -112.0f, tp_out_count);
        ok = ok && ds4_gpu_attention_output_low_q8_rows_exact_tensor(
            tp_low_tensor,
            model,
            model_size,
            attn_a_offset,
            DIM,
            ATTN_RANK,
            ATTN_GROUPS,
            group0,
            TP_GROUPS,
            tp_heads_tensor,
            TP_TEST_ROWS);
        ok = ok && ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            tp_out_tensor,
            model,
            model_size,
            attn_b_offset,
            q8_low_dim,
            DIM,
            k0,
            (uint64_t)TP_GROUPS * ATTN_RANK,
            tp_low_tensor,
            /* compact: tp_low_count is TP_TEST_ROWS * TP_GROUPS *
             * ATTN_RANK, so this rank's groups are packed at the base
             * and the window is the whole row. k_off offsets the WEIGHT. */
            (uint64_t)TP_GROUPS * ATTN_RANK, 0,
            TP_TEST_ROWS);
        ok = ok && ds4_gpu_tensor_read(
            tp_low_tensor, 0, tp_low_actual, tp_low_count * sizeof(float));
        ok = ok && ds4_gpu_tensor_read(
            tp_out_tensor, 0, tp_out_actual, tp_out_count * sizeof(float));
        if (ok) {
            char low_label[16];
            char out_label[16];
            snprintf(low_label, sizeof(low_label), "tp%u-low", rank_id);
            snprintf(out_label, sizeof(out_label), "tp%u-out", rank_id);
            ok = compare_values(low_label, tp_low_actual, tp_low_expected,
                                tp_low_count, 3.0e-4f) &&
                 compare_values(out_label, tp_out_actual, tp_out_expected,
                                tp_out_count, 3.0e-3f);
        }
        for (uint64_t i = 0; i < tp_out_count; i++) {
            tp_out_sum[i] += tp_out_actual[i];
        }
        /* The end-to-end verifier head split packs this rank's groups at the
         * start of every row. Point the weight range at the global group and
         * present both the logical group count and heads stride as compact. */
        const uint64_t attn_a_row_bytes =
            attn_a_blocks_per_row * sizeof(block_q8_0);
        const uint64_t compact_attn_a_offset = attn_a_offset +
            (uint64_t)group0 * ATTN_RANK * attn_a_row_bytes;
        ok = ok && ds4_gpu_tensor_write(
            tp_heads_compact_tensor, 0, tp_heads_compact_host,
            tp_heads_compact_count * sizeof(float));
        ok = ok && ds4_gpu_tensor_fill_f32(
            tp_low_tensor, -113.0f, tp_low_count);
        ok = ok && ds4_gpu_tensor_fill_f32(
            tp_out_tensor, -114.0f, tp_out_count);
        ok = ok && ds4_gpu_attention_output_low_q8_rows_exact_tensor(
            tp_low_tensor,
            model,
            model_size,
            compact_attn_a_offset,
            DIM,
            ATTN_RANK,
            TP_GROUPS,
            0,
            TP_GROUPS,
            tp_heads_compact_tensor,
            TP_TEST_ROWS);
        ok = ok && ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            tp_out_tensor,
            model,
            model_size,
            attn_b_offset,
            q8_low_dim,
            DIM,
            k0,
            (uint64_t)TP_GROUPS * ATTN_RANK,
            tp_low_tensor,
            /* compact: tp_low_count is TP_TEST_ROWS * TP_GROUPS *
             * ATTN_RANK, so this rank's groups are packed at the base
             * and the window is the whole row. k_off offsets the WEIGHT. */
            (uint64_t)TP_GROUPS * ATTN_RANK, 0,
            TP_TEST_ROWS);
        ok = ok && ds4_gpu_tensor_read(
            tp_low_tensor, 0, tp_low_actual, tp_low_count * sizeof(float));
        ok = ok && ds4_gpu_tensor_read(
            tp_out_tensor, 0, tp_out_actual, tp_out_count * sizeof(float));
        if (ok) {
            char low_label[24];
            char out_label[24];
            snprintf(low_label, sizeof(low_label), "tp%u-compact-low", rank_id);
            snprintf(out_label, sizeof(out_label), "tp%u-compact-out", rank_id);
            ok = compare_values(low_label, tp_low_actual, tp_low_expected,
                                tp_low_count, 3.0e-4f) &&
                 compare_values(out_label, tp_out_actual, tp_out_expected,
                                tp_out_count, 3.0e-3f);
        }
    }
    if (ok) {
        ok = compare_values("tp-sum", tp_out_sum, tp_out_full_ref,
                            tp_out_count, 4.0e-3f);
    }

    /* Keep the DSpark Markov correction resident too: validate a short greedy
     * chain (including previous-token feedback, production rank, and a
     * non-multiple-of-256 vocabulary tail) against a scalar Q8_0 reference. */
    float markov_logits[MARKOV_VOCAB];
    float markov_state[MARKOV_RANK];
    const block_q8_0 *markov_w1 =
        (const block_q8_0 *)((const uint8_t *)model + markov_w1_offset);
    const block_q8_0 *markov_w2 =
        (const block_q8_0 *)((const uint8_t *)model + markov_w2_offset);
    ds4_gpu_tensor *markov_logits_tensor = ds4_gpu_tensor_alloc(
        sizeof(markov_logits));
    ds4_gpu_tensor *markov_key_tensor = ds4_gpu_tensor_alloc(sizeof(uint64_t));
    ok = ok && markov_logits_tensor && markov_key_tensor;
    uint32_t markov_prev = 7u;
    for (uint32_t step = 0; ok && step < MARKOV_STEPS; step++) {
        const block_q8_0 *markov_w1_row = markov_w1 +
            (uint64_t)markov_prev * (MARKOV_RANK / QK_Q8_0);
        for (uint32_t i = 0; i < MARKOV_RANK; i++) {
            const block_q8_0 *b = markov_w1_row + i / QK_Q8_0;
            markov_state[i] = (float)b->d * (float)b->qs[i % QK_Q8_0];
        }
        uint32_t markov_expected = 0;
        float markov_best = -FLT_MAX;
        for (uint32_t token = 0; token < MARKOV_VOCAB; token++) {
            markov_logits[token] =
                (float)((int32_t)((token * 29u + step * 17u + 3u) % 41u) - 20) /
                32.0f;
            if (step + 1u == MARKOV_STEPS &&
                token + 1u == MARKOV_VOCAB) {
                /* Force a winner in the second strided threadgroup; otherwise
                 * a broken kernel that ignores rows >= 32768 can pass when
                 * the deterministic weights favor an early vocabulary row. */
                markov_logits[token] = 1.0e6f;
            }
            const float score = markov_logits[token] +
                dot_q8(markov_w2 +
                           (uint64_t)token * (MARKOV_RANK / QK_Q8_0),
                       markov_state,
                       MARKOV_RANK);
            if (score > markov_best ||
                (score == markov_best && token < markov_expected)) {
                markov_best = score;
                markov_expected = token;
            }
        }
        uint64_t markov_key = 0;
        ok = ds4_gpu_tensor_write(
            markov_logits_tensor, 0, markov_logits, sizeof(markov_logits));
        ok = ok && ds4_gpu_dspark_markov_argmax_tensor(
            markov_key_tensor,
            markov_logits_tensor,
            model,
            model_size,
            markov_w1_offset,
            markov_w2_offset,
            markov_prev,
            MARKOV_VOCAB,
            MARKOV_RANK);
        ok = ok && ds4_gpu_tensor_read(
            markov_key_tensor, 0, &markov_key, sizeof(markov_key));
        const uint32_t markov_actual = ~(uint32_t)markov_key;
        if (ok && markov_actual != markov_expected) {
            fprintf(stderr,
                    "MXFP4 Metal DSpark Markov step=%u mismatch expected=%u actual=%u key=%016llx\n",
                    step,
                    markov_expected,
                    markov_actual,
                    (unsigned long long)markov_key);
            ok = 0;
        } else if (ok) {
            fprintf(stderr,
                    "MXFP4 Metal DSpark Markov step=%u expected=%u actual=%u key=%016llx PASS\n",
                    step,
                    markov_expected,
                    markov_actual,
                    (unsigned long long)markov_key);
            markov_prev = markov_actual;
        }
    }
    ds4_gpu_tensor_free(markov_key_tensor);
    ds4_gpu_tensor_free(markov_logits_tensor);
    ds4_gpu_tensor_free(tp_out_tensor);
    ds4_gpu_tensor_free(tp_low_tensor);
    ds4_gpu_tensor_free(tp_heads_compact_tensor);
    ds4_gpu_tensor_free(tp_heads_tensor);
    free(tp_out_full_ref);
    free(tp_out_sum);
    free(tp_out_actual);
    free(tp_out_expected);
    free(tp_low_actual);
    free(tp_low_expected);
    free(tp_low_full_ref);
    free(tp_heads_compact_host);
    free(tp_heads_host);

    /* The production large-prefill path stores its fused SwiGLU result as
     * FP16 before the down projection. Compare that independent grouped-MMA
     * path against the same scalar reference with the documented rounding. */
    const uint64_t batch_pairs = (uint64_t)BATCH_TOKENS * pair_count;
    const uint64_t batch_out_count = (uint64_t)BATCH_TOKENS * DIM;
    float *x_batch = calloc((size_t)BATCH_TOKENS * DIM, sizeof(float));
    int32_t *selected_batch = calloc(
        (size_t)BATCH_TOKENS * N_EXPERT, sizeof(int32_t));
    float *weights_batch = calloc(
        (size_t)BATCH_TOKENS * N_EXPERT, sizeof(float));
    float *mid_batch_expected = calloc((size_t)batch_pairs, sizeof(float));
    float *mid_batch_actual = calloc((size_t)batch_pairs, sizeof(float));
    _Float16 *mid_batch_baseline = calloc(
        (size_t)batch_pairs, sizeof(_Float16));
    _Float16 *mid_batch_storage = calloc(
        (size_t)batch_pairs, sizeof(_Float16));
    float *out_batch_expected = calloc((size_t)batch_out_count, sizeof(float));
    float *out_batch_baseline = calloc(
        (size_t)batch_out_count, sizeof(float));
    float *out_batch_actual = calloc((size_t)batch_out_count, sizeof(float));
    float *experts_batch_baseline = calloc(
        (size_t)batch_out_count * N_EXPERT, sizeof(float));
    float *experts_batch_actual = calloc(
        (size_t)batch_out_count * N_EXPERT, sizeof(float));
    uint8_t *batch_poison = malloc(
        (size_t)batch_out_count * N_EXPERT * sizeof(float));
    for (uint32_t token = 0; token < BATCH_TOKENS; token++) {
        float *x_row = x_batch + (uint64_t)token * DIM;
        float *mid_row = mid_batch_expected + (uint64_t)token * pair_count;
        float *out_row = out_batch_expected + (uint64_t)token * DIM;
        for (uint32_t i = 0; i < DIM; i++) {
            const int32_t delta =
                (int32_t)((token * 7u + i * 3u) % 9u) - 4;
            x_row[i] = x[i] + (float)delta / 1024.0f;
        }
        memcpy(selected_batch + (uint64_t)token * N_EXPERT,
               selected, sizeof(selected));
        memcpy(weights_batch + (uint64_t)token * N_EXPERT,
               weights, sizeof(weights));
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            const uint32_t expert = (uint32_t)selected[slot];
            for (uint32_t row = 0; row < DIM; row++) {
                const uint64_t pair = (uint64_t)slot * DIM + row;
                const float gate_v = dot_mxfp4(
                    gate_matrix + (uint64_t)expert * blocks_per_expert +
                        (uint64_t)row * blocks_per_row,
                    x_row);
                const float up_v = dot_mxfp4(
                    up_matrix + (uint64_t)expert * blocks_per_expert +
                        (uint64_t)row * blocks_per_row,
                    x_row);
                const float g = fminf(gate_v, 7.0f);
                const float u = fmaxf(-7.0f, fminf(up_v, 7.0f));
                mid_row[pair] = (float)(_Float16)(
                    (g / (1.0f + expf(-g))) * u * weights[slot]);
            }
        }
        for (uint32_t row = 0; row < DIM; row++) {
            for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
                const uint32_t expert = (uint32_t)selected[slot];
                out_row[row] += dot_mxfp4(
                    down_matrix + (uint64_t)expert * blocks_per_expert +
                        (uint64_t)row * blocks_per_row,
                    mid_row + (uint64_t)slot * DIM);
            }
        }
    }

    ds4_gpu_tensor *x_batch_tensor = ds4_gpu_tensor_alloc(
        (uint64_t)BATCH_TOKENS * sizeof(x));
    ds4_gpu_tensor *selected_batch_tensor = ds4_gpu_tensor_alloc(
        (uint64_t)BATCH_TOKENS * sizeof(selected));
    ds4_gpu_tensor *weights_batch_tensor = ds4_gpu_tensor_alloc(
        (uint64_t)BATCH_TOKENS * sizeof(weights));
    ds4_gpu_tensor *gate_batch_tensor = ds4_gpu_tensor_alloc(
        batch_pairs * sizeof(float));
    ds4_gpu_tensor *up_batch_tensor = ds4_gpu_tensor_alloc(
        batch_pairs * sizeof(float));
    ds4_gpu_tensor *mid_batch_tensor = ds4_gpu_tensor_alloc(
        batch_pairs * sizeof(float));
    ds4_gpu_tensor *experts_batch_tensor = ds4_gpu_tensor_alloc(
        batch_out_count * N_EXPERT * sizeof(float));
    ds4_gpu_tensor *out_batch_tensor = ds4_gpu_tensor_alloc(
        batch_out_count * sizeof(float));
    bool mid_is_f16 = false;
    ok = ok && x_batch && selected_batch && weights_batch &&
         mid_batch_expected && mid_batch_actual && mid_batch_baseline &&
         mid_batch_storage && out_batch_expected && out_batch_baseline &&
         out_batch_actual && experts_batch_baseline &&
         experts_batch_actual && batch_poison &&
         x_batch_tensor && selected_batch_tensor && weights_batch_tensor &&
         gate_batch_tensor && up_batch_tensor && mid_batch_tensor &&
         experts_batch_tensor && out_batch_tensor;
    ok = ok && ds4_gpu_tensor_write(
        x_batch_tensor, 0, x_batch, BATCH_TOKENS * sizeof(x));
    ok = ok && ds4_gpu_tensor_write(
        selected_batch_tensor, 0, selected_batch,
        BATCH_TOKENS * sizeof(selected));
    ok = ok && ds4_gpu_tensor_write(
        weights_batch_tensor, 0, weights_batch,
        BATCH_TOKENS * sizeof(weights));

    /* DSpark verifies tiny suffixes rather than prefill-sized batches.  Keep
     * every production block length (2..5) on the fused pair-SwiGLU/direct
     * sum path covered by the model-free harness; in particular, five rows
     * used to fall back to a large per-expert scratch and a separate sum. */
    for (uint32_t tiny_tokens = 2u; tiny_tokens <= 5u && ok; tiny_tokens++) {
        mid_is_f16 = true;
        ok = ok && ds4_gpu_tensor_fill_f32(
            out_batch_tensor, -105.0f, batch_out_count);
        ok = ok && ds4_gpu_routed_moe_batch_tensor(
            out_batch_tensor, gate_batch_tensor, up_batch_tensor,
            mid_batch_tensor, experts_batch_tensor,
            model, model_size, gate_offset, up_offset, down_offset,
            MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
            expert_bytes, row_bytes, DIM, DIM, DIM,
            selected_batch_tensor, weights_batch_tensor,
            N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
            0u, tiny_tokens, &mid_is_f16, true);
        ok = ok && !mid_is_f16;
        ok = ok && ds4_gpu_tensor_read(
            out_batch_tensor, 0, out_batch_actual,
            (uint64_t)tiny_tokens * DIM * sizeof(float));
        for (uint32_t token = 0; token < tiny_tokens && ok; token++) {
            char label[32];
            snprintf(label, sizeof(label), "tiny%u.%u", tiny_tokens, token);
            ok = compare_values(label,
                                out_batch_actual + (uint64_t)token * DIM,
                                out_batch_expected + (uint64_t)token * DIM,
                                DIM,
                                4.0e-4f);
        }
    }

    /* The verifier batch path runs the same call with one expert half resident
     * on each rank.  Exercise both global-ID rebases and reconstruct the full
     * result from their partials across five distinct input rows. */
    const uint64_t tp_moe_count = (uint64_t)TP_TEST_ROWS * DIM;
    float *tp_moe_expected = calloc(2u * (size_t)tp_moe_count, sizeof(float));
    float *tp_moe_actual = calloc(2u * (size_t)tp_moe_count, sizeof(float));
    float *tp_moe_sum = calloc((size_t)tp_moe_count, sizeof(float));
    ok = ok && tp_moe_expected && tp_moe_actual && tp_moe_sum;
    for (uint32_t rank_id = 0; rank_id < 2u && ok; rank_id++) {
        const uint32_t expert0 = rank_id * (N_TOTAL_EXPERT / 2u);
        const uint32_t expert1 = rank_id == 0u ?
            N_TOTAL_EXPERT / 2u : N_TOTAL_EXPERT;
        for (uint32_t token = 0; token < TP_TEST_ROWS; token++) {
            const float *mid_row =
                mid_batch_expected + (uint64_t)token * pair_count;
            float *expected_row = tp_moe_expected +
                ((uint64_t)rank_id * TP_TEST_ROWS + token) * DIM;
            for (uint32_t row = 0; row < DIM; row++) {
                for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
                    const uint32_t expert = (uint32_t)selected[slot];
                    if (expert < expert0 || expert >= expert1) continue;
                    expected_row[row] += dot_mxfp4(
                        down_matrix + (uint64_t)expert * blocks_per_expert +
                            (uint64_t)row * blocks_per_row,
                        mid_row + (uint64_t)slot * DIM);
                }
            }
        }

        ds4_gpu_test_set_tp_expert_shard(rank_id, 2u);
        mid_is_f16 = true;
        ok = ok && ds4_gpu_tensor_fill_f32(
            out_batch_tensor, -105.0f, batch_out_count);
        ok = ok && ds4_gpu_routed_moe_batch_tensor(
            out_batch_tensor, gate_batch_tensor, up_batch_tensor,
            mid_batch_tensor, experts_batch_tensor,
            model, model_size, gate_offset, up_offset, down_offset,
            MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
            expert_bytes, row_bytes, DIM, DIM, DIM,
            selected_batch_tensor, weights_batch_tensor,
            N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
            0u, TP_TEST_ROWS, &mid_is_f16, true);
        ok = ok && !mid_is_f16;
        ok = ok && ds4_gpu_tensor_read(
            out_batch_tensor, 0,
            tp_moe_actual + (uint64_t)rank_id * tp_moe_count,
            tp_moe_count * sizeof(float));
        for (uint32_t token = 0; token < TP_TEST_ROWS && ok; token++) {
            char label[32];
            snprintf(label, sizeof(label), "tp-moe%u.%u", rank_id, token);
            ok = compare_values(
                label,
                tp_moe_actual +
                    ((uint64_t)rank_id * TP_TEST_ROWS + token) * DIM,
                tp_moe_expected +
                    ((uint64_t)rank_id * TP_TEST_ROWS + token) * DIM,
                DIM,
                4.0e-4f);
        }
    }
    ds4_gpu_test_set_tp_expert_shard(0u, 1u);
    if (tp_moe_sum && tp_moe_actual) {
        for (uint64_t i = 0; i < tp_moe_count; i++) {
            tp_moe_sum[i] =
                tp_moe_actual[i] + tp_moe_actual[tp_moe_count + i];
        }
    }
    if (ok) {
        ok = compare_values("tp-moe-sum", tp_moe_sum,
                            out_batch_expected, tp_moe_count, 6.0e-4f);
    }
    free(tp_moe_sum);
    free(tp_moe_actual);
    free(tp_moe_expected);

    if (batch_poison) {
        memset(batch_poison, 0xa5,
               (size_t)batch_out_count * N_EXPERT * sizeof(float));
    }
    /* With 48 identical routes, every occupied expert has a 32-row tile plus
     * a 16-row tail tile. Run the established, pair-culling, and down-culling
     * pipelines in the same process and require exact FP16 intermediates and
     * F32 outputs. */
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL);
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_baseline,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_baseline,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        experts_batch_tensor, 0, experts_batch_baseline,
        batch_out_count * N_EXPERT * sizeof(float));

    /* Force the exact half-result dequantization table only for the resident
     * MXFP4/F16-mid/single-rank down projection. Poison every writable output
     * and run twice in this process; the FP16 mid, expert-major F32 down
     * scratch, and summed F32 output must all match the established kernel
     * byte-for-byte on each repetition. */
    ds4_gpu_test_set_flags(
        DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL |
        DS4_GPU_TEST_MXFP4_DOWN_HALF_LUT);
    for (uint32_t run = 0; run < 2u && ok; run++) {
        mid_is_f16 = false;
        ok = ok && ds4_gpu_tensor_write(
            mid_batch_tensor, 0, batch_poison,
            batch_pairs * sizeof(_Float16));
        ok = ok && ds4_gpu_tensor_write(
            experts_batch_tensor, 0, batch_poison,
            batch_out_count * N_EXPERT * sizeof(float));
        ok = ok && ds4_gpu_tensor_write(
            out_batch_tensor, 0, batch_poison,
            batch_out_count * sizeof(float));
        ok = ok && ds4_gpu_routed_moe_batch_tensor(
            out_batch_tensor, gate_batch_tensor, up_batch_tensor,
            mid_batch_tensor, experts_batch_tensor,
            model, model_size, gate_offset, up_offset, down_offset,
            MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
            expert_bytes, row_bytes, DIM, DIM, DIM,
            selected_batch_tensor, weights_batch_tensor,
            N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
            0u, BATCH_TOKENS, &mid_is_f16, true);
        ok = ok && mid_is_f16;
        ok = ok && ds4_gpu_tensor_read(
            mid_batch_tensor, 0, mid_batch_storage,
            batch_pairs * sizeof(_Float16));
        ok = ok && ds4_gpu_tensor_read(
            experts_batch_tensor, 0, experts_batch_actual,
            batch_out_count * N_EXPERT * sizeof(float));
        ok = ok && ds4_gpu_tensor_read(
            out_batch_tensor, 0, out_batch_actual,
            batch_out_count * sizeof(float));
        if (ok &&
            (memcmp(mid_batch_storage, mid_batch_baseline,
                    batch_pairs * sizeof(_Float16)) != 0 ||
             memcmp(experts_batch_actual, experts_batch_baseline,
                    batch_out_count * N_EXPERT * sizeof(float)) != 0 ||
             memcmp(out_batch_actual, out_batch_baseline,
                    batch_out_count * sizeof(float)) != 0)) {
            fprintf(stderr,
                    "MXFP4 Metal down half-LUT poisoned A/B mismatch on repetition %u\n",
                    run + 1u);
            ok = 0;
        } else if (ok) {
            fprintf(stderr,
                    "MXFP4 Metal down half-LUT poisoned A/B exact on repetition %u\n",
                    run + 1u);
        }
    }
    ds4_gpu_test_set_flags(0);

    mid_is_f16 = false;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_PAIR_TAIL_CULL);
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_storage,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_actual,
        batch_out_count * sizeof(float));
    ds4_gpu_test_set_flags(0);
    if (ok && (memcmp(mid_batch_storage, mid_batch_baseline,
                      batch_pairs * sizeof(_Float16)) != 0 ||
               memcmp(out_batch_actual, out_batch_baseline,
                      batch_out_count * sizeof(float)) != 0)) {
        fprintf(stderr,
                "MXFP4 Metal pair tail-cull A/B mismatch for 16-row expert tails\n");
        ok = 0;
    } else if (ok) {
        fprintf(stderr,
                "MXFP4 Metal pair tail-cull A/B exact for 16-row expert tails\n");
    }

    /* Force the diagnostic-only 32x32/64-thread pair kernel. Its second
     * SIMDgroup is inactive on each 16-row expert tail, while both groups
     * remain in staging and at every threadgroup barrier. Require its FP16
     * intermediate and final F32 output to match the established 64x32 path
     * byte-for-byte. */
    mid_is_f16 = false;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE);
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_storage,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_actual,
        batch_out_count * sizeof(float));
    ds4_gpu_test_set_flags(0);
    if (ok && (memcmp(mid_batch_storage, mid_batch_baseline,
                      batch_pairs * sizeof(_Float16)) != 0 ||
               memcmp(out_batch_actual, out_batch_baseline,
                      batch_out_count * sizeof(float)) != 0)) {
        fprintf(stderr,
                "MXFP4 Metal compact pair A/B mismatch for 16-row expert tails\n");
        ok = 0;
    } else if (ok) {
        fprintf(stderr,
                "MXFP4 Metal compact pair A/B exact for 16-row expert tails\n");
    }

    mid_is_f16 = false;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL);
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_storage,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_actual,
        batch_out_count * sizeof(float));
    ds4_gpu_test_set_flags(0);
    if (ok && (memcmp(mid_batch_storage, mid_batch_baseline,
                      batch_pairs * sizeof(_Float16)) != 0 ||
               memcmp(out_batch_actual, out_batch_baseline,
                      batch_out_count * sizeof(float)) != 0)) {
        fprintf(stderr,
                "MXFP4 Metal down tail-cull A/B mismatch for 16-row expert tails\n");
        ok = 0;
    } else if (ok) {
        fprintf(stderr,
                "MXFP4 Metal down tail-cull A/B exact for 16-row expert tails\n");
    }
    for (uint64_t i = 0; i < batch_pairs; i++) {
        mid_batch_actual[i] = (float)mid_batch_storage[i];
    }
    if (ok) {
        ok = compare_values("bmid", mid_batch_actual, mid_batch_expected,
                            batch_pairs, 2.0e-3f) &&
             compare_values("bout", out_batch_actual, out_batch_expected,
                            batch_out_count, 2.0e-3f);
    }

    /* Exercise token-centric map construction with uneven expert occupancy.
     * Experts 0..4 receive 48 rows, expert 5 receives 33, expert 6 receives
     * 15, and expert 7 is empty. This produces 13 occupied descriptors against
     * a padded direct work capacity of 17, including 16-, 15-, and 1-row
     * tails. Keep the compact pair kernel fixed across the baseline and both
     * scatter repetitions, and require exact FP16 mid and F32 output bytes. */
    for (uint32_t token = 0; token < BATCH_TOKENS; token++) {
        int32_t * token_selected =
            selected_batch + (uint64_t)token * N_EXPERT;
        for (uint32_t slot = 0; slot < N_EXPERT - 1u; slot++) {
            token_selected[slot] = (int32_t)slot;
        }
        token_selected[N_EXPERT - 1u] = token < 33u ? 5 : 6;
    }
    ok = ok && ds4_gpu_tensor_write(
        selected_batch_tensor, 0, selected_batch,
        BATCH_TOKENS * sizeof(selected));
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE);
    mid_is_f16 = false;
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_baseline,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_baseline,
        batch_out_count * sizeof(float));

    ds4_gpu_test_set_flags(
        DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE |
        DS4_GPU_TEST_MXFP4_MAP_SCATTER);
    for (uint32_t run = 0; run < 2u && ok; run++) {
        mid_is_f16 = false;
        ok = ok && ds4_gpu_tensor_write(
            mid_batch_tensor, 0, batch_poison,
            batch_pairs * sizeof(_Float16));
        ok = ok && ds4_gpu_tensor_write(
            experts_batch_tensor, 0, batch_poison,
            batch_out_count * N_EXPERT * sizeof(float));
        ok = ok && ds4_gpu_tensor_write(
            out_batch_tensor, 0, batch_poison,
            batch_out_count * sizeof(float));
        ok = ok && ds4_gpu_routed_moe_batch_tensor(
            out_batch_tensor, gate_batch_tensor, up_batch_tensor,
            mid_batch_tensor, experts_batch_tensor,
            model, model_size, gate_offset, up_offset, down_offset,
            MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
            expert_bytes, row_bytes, DIM, DIM, DIM,
            selected_batch_tensor, weights_batch_tensor,
            N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
            0u, BATCH_TOKENS, &mid_is_f16, true);
        ok = ok && mid_is_f16;
        ok = ok && ds4_gpu_tensor_read(
            mid_batch_tensor, 0, mid_batch_storage,
            batch_pairs * sizeof(_Float16));
        ok = ok && ds4_gpu_tensor_read(
            out_batch_tensor, 0, out_batch_actual,
            batch_out_count * sizeof(float));
        if (ok && (memcmp(mid_batch_storage, mid_batch_baseline,
                          batch_pairs * sizeof(_Float16)) != 0 ||
                   memcmp(out_batch_actual, out_batch_baseline,
                          batch_out_count * sizeof(float)) != 0)) {
            fprintf(stderr,
                    "MXFP4 Metal map scatter A/B mismatch on repetition %u\n",
                    run + 1u);
            ok = 0;
        } else if (ok) {
            fprintf(stderr,
                    "MXFP4 Metal map scatter A/B exact on repetition %u\n",
                    run + 1u);
        }
    }
    ds4_gpu_test_set_flags(0);

    ds4_gpu_tensor_free(out_batch_tensor);
    ds4_gpu_tensor_free(experts_batch_tensor);
    ds4_gpu_tensor_free(mid_batch_tensor);
    ds4_gpu_tensor_free(up_batch_tensor);
    ds4_gpu_tensor_free(gate_batch_tensor);
    ds4_gpu_tensor_free(weights_batch_tensor);
    ds4_gpu_tensor_free(selected_batch_tensor);
    ds4_gpu_tensor_free(x_batch_tensor);
    free(out_batch_actual);
    free(out_batch_baseline);
    free(out_batch_expected);
    free(experts_batch_actual);
    free(experts_batch_baseline);
    free(batch_poison);
    free(mid_batch_storage);
    free(mid_batch_baseline);
    free(mid_batch_actual);
    free(mid_batch_expected);
    free(weights_batch);
    free(selected_batch);
    free(x_batch);

    ds4_gpu_tensor_free(out_tensor);
    ds4_gpu_tensor_free(experts_tensor);
    ds4_gpu_tensor_free(mid_tensor);
    ds4_gpu_tensor_free(up_tensor);
    ds4_gpu_tensor_free(gate_tensor);
    ds4_gpu_tensor_free(weights_tensor);
    ds4_gpu_tensor_free(selected_tensor);
    ds4_gpu_tensor_free(x_tensor);
    ds4_gpu_cleanup();
    free(out_fast);
    free(experts_fast);
    free(mid_fast);
    free(up_fast);
    free(gate_fast);
    free(out_gpu);
    free(experts_gpu);
    free(mid_gpu);
    free(up_gpu);
    free(gate_gpu);
    free(out_ref);
    free(mid_ref);
    free(up_ref);
    free(gate_ref);
    free(model);

    fprintf(stderr, "MXFP4 Metal fused MoE: %s\n", ok ? "PASS" : "FAIL");

    /* Runs last: it replaces the model map, so nothing above may depend on it. */
    const int kslice_ok = run_kslice_rows();
    fprintf(stderr, "Q8_0 k-slice across the mv/mm threshold: %s\n",
            kslice_ok ? "PASS" : "FAIL");

    return (ok && kslice_ok) ? 0 : 1;
}
