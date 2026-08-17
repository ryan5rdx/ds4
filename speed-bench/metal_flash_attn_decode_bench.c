#define _DARWIN_C_SOURCE

/*
 * Model-free benchmark for DS4 Flash decode attention on Metal.
 *
 * At ctx=512 the Flash model has two uncompressed layers, 21 ratio-4
 * compressor layers, and 20 ratio-128 layers.  The live 128-token raw window
 * therefore gives the three key counts below: 128, 128 + 128, and 128 + 4.
 * Tensor parallelism owns 32 of the 64 query heads on each rank.
 *
 * This harness binds a gate-free TP=2 Metal backend so that leaving
 * DS4_METAL_DECODE_NWG unset exercises the real per-shape default.  It never
 * emits a TP gate and needs neither a peer nor a GGUF.
 */

#include "ds4_gpu.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BENCH_NAME "metal-flash-attn-decode-bench"
#define NWG_ENV "DS4_METAL_DECODE_NWG"

enum {
    N_HEAD = 32,
    HEAD_DIM = 512,
    N_RAW = 128,
    /* The documented ctx=512/gen=512 ds4-bench allocates ctx=1025. */
    RAW_CAP = 1025,
    RAW_START_CTX512 = 385,
    MAX_COMP = 256,
    CASE_COUNT = 6,
    NWG_COUNT = 8,
    VARIANT_COUNT = 5,
    DEFAULT_WARMUP = 1,
    DEFAULT_SAMPLES = 8,
    DEFAULT_TOKENS = 32,
};

typedef struct {
    const char *name;
    uint32_t n_comp;
    uint32_t layers;
    uint32_t adaptive_nwg;
} attention_case;

static const attention_case g_cases[CASE_COUNT] = {
    {"raw-128",        0,   2, 4},
    {"ratio128-ctx512", 4, 20, 5},
    {"ratio4-ctx512", 128, 21, 12},
    {"keys-257",      129,  0, 12},
    {"keys-288",      160,  0, 12},
    {"keys-384",      256,  0, 12},
};

static const int g_nwg_values[NWG_COUNT] = {2, 4, 5, 8, 12, 16, 24, 32};

typedef struct {
    const char *name;
    int nwg; /* zero means the TP=2 adaptive default */
} timing_variant;

static const timing_variant g_variants[VARIANT_COUNT] = {
    {"exact-bucket-4/5/12", 0},
    {"fixed-32",      32},
    {"fixed-12",      12},
    {"fixed-16",      16},
    {"fixed-24",      24},
};

typedef struct {
    int warmup;
    int samples;
    int tokens;
    bool correctness_only;
} bench_config;

typedef struct {
    void *model_map;
    uint64_t model_size;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *raw;
    ds4_gpu_tensor *comp;
    ds4_gpu_tensor *heads;
} bench_state;

typedef struct {
    size_t bit_mismatches;
    size_t nonfinite;
    double max_abs;
    double max_rel;
    double rmse;
    double reference_scale;
} compare_stats;

/* ds4_metal.o uses this common logger query, but the standalone harness does
 * not otherwise link the application's logging implementation. */
bool ds4_log_is_tty(FILE *fp);

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

/* Decode-RoPE fusion is internal to the fixed graph, but is deliberately
 * exercised here because its reducer has a different pipeline specialization. */
extern void ds4_gpu_set_decode_attn_rope_fuse(
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig,
        bool inverse, float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow);
extern int ds4_gpu_decode_attn_rope_fuse_used(void);

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "\n"
            "  --warmup N   untimed 43-layer schedules per variant (default: %d)\n"
            "  --samples N  balanced timing samples per variant (default: %d)\n"
            "  --tokens N   schedules in each timing sample (default: %d)\n"
            "  --correctness-only  skip the timing sweep\n",
            argv0,
            DEFAULT_WARMUP,
            DEFAULT_SAMPLES,
            DEFAULT_TOKENS);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s: %s requires an argument\n", BENCH_NAME, opt);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int(const char *text, const char *opt, int minimum, int maximum) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || text[0] == '\0' || !end || *end != '\0' ||
        value < minimum || value > maximum) {
        fprintf(stderr, "%s: invalid %s value: %s\n", BENCH_NAME, opt, text);
        exit(2);
    }
    return (int)value;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config cfg = {
        .warmup = DEFAULT_WARMUP,
        .samples = DEFAULT_SAMPLES,
        .tokens = DEFAULT_TOKENS,
        .correctness_only = false,
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(arg, "--warmup")) {
            cfg.warmup = parse_int(need_arg(&i, argc, argv, arg), arg, 0, 1000);
        } else if (!strcmp(arg, "--samples")) {
            cfg.samples = parse_int(need_arg(&i, argc, argv, arg), arg, 2, 1000);
        } else if (!strcmp(arg, "--tokens")) {
            cfg.tokens = parse_int(need_arg(&i, argc, argv, arg), arg, 32, 10000);
        } else if (!strcmp(arg, "--correctness-only")) {
            cfg.correctness_only = true;
        } else {
            fprintf(stderr, "%s: unknown option: %s\n", BENCH_NAME, arg);
            usage(stderr, argv[0]);
            exit(2);
        }
    }
    if ((cfg.samples & 1) != 0) {
        fprintf(stderr, "%s: --samples must be even for balanced ordering\n",
                BENCH_NAME);
        exit(2);
    }
    return cfg;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint16_t float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } value = {.f = f};

    const uint32_t sign = (value.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((value.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = value.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1u)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float random_signed(uint32_t *state, float scale) {
    *state = *state * 1664525u + 1013904223u;
    const int32_t centered = (int32_t)((*state >> 8) & 0xffffu) - 32768;
    return (float)centered * (scale / 32768.0f);
}

static int noop_tp_exchange(void *ud, uint32_t layer, uint32_t gate, uint64_t seq) {
    (void)ud;
    (void)layer;
    (void)gate;
    (void)seq;
    return 1;
}

static int select_nwg(int nwg) {
    if (nwg == 0) return unsetenv(NWG_ENV) == 0;
    char text[16];
    snprintf(text, sizeof(text), "%d", nwg);
    return setenv(NWG_ENV, text, 1) == 0;
}

static int init_state(bench_state *state) {
    memset(state, 0, sizeof(*state));
    /* The mapped "model" is one page containing only attention sinks. Avoid
     * paying (or timing) whole-model residency machinery for that page. */
    (void)setenv("DS4_METAL_NO_RESIDENCY", "1", 1);
    (void)setenv("DS4_METAL_NO_MODEL_WARMUP", "1", 1);
    if (!ds4_gpu_init()) {
        fprintf(stderr, "%s: Metal initialization failed\n", BENCH_NAME);
        return 0;
    }

    const long page_long = getpagesize();
    if (page_long <= 0) return 0;
    const size_t page = (size_t)page_long;
    if (posix_memalign(&state->model_map, page, page) != 0) {
        fprintf(stderr, "%s: model-map allocation failed\n", BENCH_NAME);
        return 0;
    }
    state->model_size = page;
    memset(state->model_map, 0, page);
    float *sinks = state->model_map;
    for (uint32_t h = 0; h < N_HEAD; h++) {
        sinks[h] = (float)((int32_t)((h * 17u) % 23u) - 11) / 32.0f;
    }

    const uint64_t q_count = (uint64_t)N_HEAD * HEAD_DIM;
    const uint64_t raw_count = (uint64_t)RAW_CAP * HEAD_DIM;
    const uint64_t comp_count = (uint64_t)MAX_COMP * HEAD_DIM;
    state->q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    state->raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    state->comp = ds4_gpu_tensor_alloc(comp_count * sizeof(uint16_t));
    state->heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    if (!state->q || !state->raw || !state->comp || !state->heads) {
        fprintf(stderr, "%s: Metal tensor allocation failed\n", BENCH_NAME);
        return 0;
    }

    float *q = malloc((size_t)q_count * sizeof(float));
    float *raw = malloc((size_t)raw_count * sizeof(float));
    uint16_t *comp = malloc((size_t)comp_count * sizeof(uint16_t));
    if (!q || !raw || !comp) {
        free(comp);
        free(raw);
        free(q);
        fprintf(stderr, "%s: host tensor allocation failed\n", BENCH_NAME);
        return 0;
    }

    uint32_t rng = 0x4d595df4u;
    for (uint64_t i = 0; i < q_count; i++) q[i] = random_signed(&rng, 0.75f);
    for (uint64_t i = 0; i < raw_count; i++) raw[i] = random_signed(&rng, 0.60f);
    for (uint64_t i = 0; i < comp_count; i++) {
        comp[i] = float_to_f16(random_signed(&rng, 0.60f));
    }

    const int wrote =
        ds4_gpu_tensor_write(state->q, 0, q, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(state->raw, 0, raw, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(state->comp, 0, comp,
                             comp_count * sizeof(uint16_t));
    free(comp);
    free(raw);
    free(q);
    if (!wrote || !ds4_gpu_set_model_map(state->model_map, state->model_size)) {
        fprintf(stderr, "%s: tensor/model-map setup failed\n", BENCH_NAME);
        return 0;
    }

    ds4_gpu_set_quality(false);
    /* Do not let an inherited opt-in keepalive contaminate this local run. */
    (void)setenv("DS4_TP_NO_KEEPALIVE", "1", 1);
    if (!ds4_gpu_tp_init(0, NULL, 0, noop_tp_exchange, NULL)) {
        fprintf(stderr, "%s: local TP=2 binding failed\n", BENCH_NAME);
        return 0;
    }
    return 1;
}

static void cleanup_state(bench_state *state) {
    ds4_gpu_tp_shutdown();
    ds4_gpu_tensor_free(state->heads);
    ds4_gpu_tensor_free(state->comp);
    ds4_gpu_tensor_free(state->raw);
    ds4_gpu_tensor_free(state->q);
    ds4_gpu_cleanup();
    free(state->model_map);
    memset(state, 0, sizeof(*state));
}

static int encode_case(bench_state *state, const attention_case *shape) {
    return ds4_gpu_attention_decode_heads_tensor(
        state->heads,
        state->model_map,
        state->model_size,
        0,
        state->q,
        state->raw,
        N_RAW,
        RAW_CAP,
        RAW_START_CTX512,
        shape->n_comp ? state->comp : NULL,
        1,
        shape->n_comp,
        NULL,
        0,
        N_HEAD,
        HEAD_DIM);
}

static int encode_dynamic(bench_state *state,
                          uint32_t n_comp,
                          uint32_t raw_start) {
    return ds4_gpu_attention_decode_heads_tensor(
        state->heads,
        state->model_map,
        state->model_size,
        0,
        state->q,
        state->raw,
        N_RAW,
        RAW_CAP,
        raw_start,
        n_comp ? state->comp : NULL,
        1,
        n_comp,
        NULL,
        0,
        N_HEAD,
        HEAD_DIM);
}

static void arm_fused_inverse_rope(uint32_t pos) {
    const float freq_scale = 1.0f / 16.0f;
    const float attn_factor = 1.0f / (1.0f + 0.1f * logf(16.0f));
    ds4_gpu_set_decode_attn_rope_fuse(
        HEAD_DIM, 64, pos, 65536, true,
        160000.0f, freq_scale, 1.0f, attn_factor, 32.0f, 1.0f);
}

static int run_case(bench_state *state,
                    const attention_case *shape,
                    int nwg,
                    bool fuse_rope,
                    float *output) {
    if (!select_nwg(nwg)) return 0;
    uint32_t *poison = ds4_gpu_tensor_contents(state->heads);
    const size_t count = (size_t)N_HEAD * HEAD_DIM;
    for (size_t i = 0; i < count; i++) poison[i] = 0x7fc00000u;
    if (fuse_rope) arm_fused_inverse_rope(511);
    if (!encode_case(state, shape)) return 0;
    if (fuse_rope && !ds4_gpu_decode_attn_rope_fuse_used()) return 0;
    return ds4_gpu_tensor_read(state->heads, 0, output,
                               (uint64_t)count * sizeof(float));
}

static compare_stats compare_output(const float *reference,
                                    const float *observed,
                                    size_t count) {
    compare_stats stats = {0};
    double squared = 0.0;
    for (size_t i = 0; i < count; i++) {
        uint32_t ref_bits = 0;
        uint32_t got_bits = 0;
        memcpy(&ref_bits, &reference[i], sizeof(ref_bits));
        memcpy(&got_bits, &observed[i], sizeof(got_bits));
        if (ref_bits != got_bits) stats.bit_mismatches++;
        if (!isfinite(reference[i]) || !isfinite(observed[i])) {
            stats.nonfinite++;
            continue;
        }
        const double ref_abs = fabs((double)reference[i]);
        const double abs_err = fabs((double)observed[i] - (double)reference[i]);
        const double rel_err = abs_err / fmax(ref_abs, 1.0e-6);
        if (ref_abs > stats.reference_scale) stats.reference_scale = ref_abs;
        if (abs_err > stats.max_abs) stats.max_abs = abs_err;
        if (rel_err > stats.max_rel) stats.max_rel = rel_err;
        squared += abs_err * abs_err;
    }
    stats.rmse = sqrt(squared / (double)count);
    return stats;
}

static int run_correctness(bench_state *state) {
    const size_t count = (size_t)N_HEAD * HEAD_DIM;
    float *reference = malloc(CASE_COUNT * count * sizeof(float));
    float *observed = malloc(count * sizeof(float));
    if (!reference || !observed) {
        free(observed);
        free(reference);
        return 0;
    }

    printf("\nCorrectness versus NWG=32 (%u heads x %u dims)\n",
           N_HEAD, HEAD_DIM);
    printf("  %-18s %4s %7s %10s %10s %10s %10s %s\n",
           "shape", "keys", "NWG", "bit-diff", "max-abs", "rmse",
           "max-rel", "status");

    int ok = 1;
    for (size_t ci = 0; ci < CASE_COUNT; ci++) {
        float *case_reference = reference + ci * count;
        if (!run_case(state, &g_cases[ci], 32, false, case_reference)) {
            fprintf(stderr, "%s: NWG=32 reference failed for %s\n",
                    BENCH_NAME, g_cases[ci].name);
            ok = 0;
            break;
        }
        if (!run_case(state, &g_cases[ci], 0, false, observed)) {
            fprintf(stderr, "%s: adaptive NWG failed for %s\n",
                    BENCH_NAME, g_cases[ci].name);
            ok = 0;
        } else {
            const compare_stats stats =
                compare_output(case_reference, observed, count);
            const int acceptable =
                stats.nonfinite == 0 && stats.bit_mismatches == 0;
            char label[16];
            snprintf(label, sizeof(label), "auto/%u", g_cases[ci].adaptive_nwg);
            printf("  %-18s %4u %7s %10zu %10.3g %10.3g %10.3g %s\n",
                   g_cases[ci].name,
                   N_RAW + g_cases[ci].n_comp,
                   label,
                   stats.bit_mismatches,
                   stats.max_abs,
                   stats.rmse,
                   stats.max_rel,
                   acceptable ? "EXACT" : "FAIL");
            if (!acceptable) ok = 0;
        }
        for (size_t ni = 0; ni < NWG_COUNT; ni++) {
            const int nwg = g_nwg_values[ni];
            const uint32_t chunks =
                (N_RAW + g_cases[ci].n_comp + 31u) / 32u;
            const int exact_expected = (uint32_t)nwg >= chunks;
            if (!run_case(state, &g_cases[ci], nwg, false, observed)) {
                fprintf(stderr, "%s: NWG=%d failed for %s\n",
                        BENCH_NAME, nwg, g_cases[ci].name);
                if (exact_expected) ok = 0;
                continue;
            }
            const compare_stats stats = compare_output(case_reference, observed, count);
            const int exact = stats.nonfinite == 0 && stats.bit_mismatches == 0;
            const char *status = stats.nonfinite != 0
                ? "NONFINITE"
                : exact_expected ? (exact ? "EXACT" : "FAIL") : "NONEXACT";
            printf("  %-18s %4u %7d %10zu %10.3g %10.3g %10.3g %s\n",
                   g_cases[ci].name,
                   N_RAW + g_cases[ci].n_comp,
                   nwg,
                   stats.bit_mismatches,
                   stats.max_abs,
                   stats.rmse,
                   stats.max_rel,
                   status);
            if (stats.nonfinite != 0 || (exact_expected && !exact)) ok = 0;
        }
    }
    free(observed);
    free(reference);
    return ok;
}

static int run_fused_rope_correctness(bench_state *state) {
    const size_t count = (size_t)N_HEAD * HEAD_DIM;
    float *reference = malloc(count * sizeof(float));
    float *observed = malloc(count * sizeof(float));
    if (!reference || !observed) {
        free(observed);
        free(reference);
        return 0;
    }

    printf("\nFused inverse-RoPE adaptive reducer versus NWG=32\n");
    printf("  %-18s %4s %7s %10s %10s %10s %s\n",
           "shape", "keys", "NWG", "bit-diff", "max-abs", "rmse", "status");
    int ok = 1;
    static const size_t fused_cases[] = {1, 2, 3}; /* keys 132, 256, 257 */
    for (size_t fi = 0; fi < sizeof(fused_cases) / sizeof(fused_cases[0]); fi++) {
        const attention_case *shape = &g_cases[fused_cases[fi]];
        if (!run_case(state, shape, 32, true, reference) ||
            !run_case(state, shape, 0, true, observed)) {
            fprintf(stderr, "%s: fused-RoPE adaptive check failed for %s\n",
                    BENCH_NAME, shape->name);
            ok = 0;
            continue;
        }
        const compare_stats stats = compare_output(reference, observed, count);
        const int acceptable = stats.nonfinite == 0 && stats.bit_mismatches == 0;
        char label[16];
        snprintf(label, sizeof(label), "auto/%u", shape->adaptive_nwg);
        printf("  %-18s %4u %7s %10zu %10.3g %10.3g %s\n",
               shape->name,
               N_RAW + shape->n_comp,
               label,
               stats.bit_mismatches,
               stats.max_abs,
               stats.rmse,
               acceptable ? "EXACT" : "FAIL");
        if (!acceptable) ok = 0;
    }

    const attention_case *shape = &g_cases[2]; /* ratio4 ctx512, 256 keys */
    printf("\nFused inverse-RoPE forced-NWG sweep (%s)\n", shape->name);
    printf("  %-7s %10s %10s %10s %10s %s\n",
           "NWG", "bit-diff", "max-abs", "rmse", "max-rel", "status");
    const int have_reference = run_case(state, shape, 32, true, reference);
    if (!have_reference) {
        fprintf(stderr, "%s: fused-RoPE NWG=32 reference failed\n", BENCH_NAME);
        ok = 0;
    }
    for (size_t ni = 0; have_reference && ni < NWG_COUNT; ni++) {
        const int nwg = g_nwg_values[ni];
        const uint32_t chunks = (N_RAW + shape->n_comp + 31u) / 32u;
        const int exact_expected = (uint32_t)nwg >= chunks;
        if (!run_case(state, shape, nwg, true, observed)) {
            fprintf(stderr, "%s: fused-RoPE NWG=%d failed\n", BENCH_NAME, nwg);
            if (exact_expected) ok = 0;
            continue;
        }
        const compare_stats stats = compare_output(reference, observed, count);
        const int exact = stats.nonfinite == 0 && stats.bit_mismatches == 0;
        const char *status = stats.nonfinite != 0
            ? "NONFINITE"
            : exact_expected ? (exact ? "EXACT" : "FAIL") : "NONEXACT";
        printf("  %7d %10zu %10.3g %10.3g %10.3g %s\n",
               nwg,
               stats.bit_mismatches,
               stats.max_abs,
               stats.rmse,
               stats.max_rel,
               status);
        if (stats.nonfinite != 0 || (exact_expected && !exact)) ok = 0;
    }

    free(observed);
    free(reference);
    return ok;
}

static int encode_schedule(bench_state *state, uint32_t generation_offset) {
    const uint32_t raw_start = RAW_START_CTX512 + generation_offset;
    if (raw_start + N_RAW > RAW_CAP) return 0;
    /* The token at position 512 + offset has already passed through the
     * compressor when attention runs, hence the +1 emission boundary. */
    const uint32_t ratio4_comp = 128u + (generation_offset + 1u) / 4u;
    const uint32_t ratio128_comp = 4u + (generation_offset + 1u) / 128u;
    const uint32_t pos = 512u + generation_offset;
    if (!ds4_gpu_begin_commands()) return 0;
    for (uint32_t layer = 0; layer < 43; layer++) {
        uint32_t n_comp;
        if (layer < 2) {
            n_comp = 0;
        } else {
            n_comp = (layer & 1u) == 0 ? ratio4_comp : ratio128_comp;
        }
        if (n_comp != 0) arm_fused_inverse_rope(pos);
        if (!encode_dynamic(state, n_comp, raw_start)) {
            (void)ds4_gpu_end_commands();
            return 0;
        }
        if (n_comp != 0 && !ds4_gpu_decode_attn_rope_fuse_used()) {
            (void)ds4_gpu_end_commands();
            return 0;
        }
    }
    return ds4_gpu_end_commands();
}

static double time_variant(bench_state *state, int nwg, int tokens) {
    if (!select_nwg(nwg)) return -1.0;
    const double start = now_ms();
    for (int token = 0; token < tokens; token++) {
        /* Midpoints of equal bins avoid over-representing rare exact
         * compressor/block boundaries (especially the no-pad ratio-4 case). */
        const uint32_t offset =
            (uint32_t)(((uint64_t)(2u * (uint32_t)token + 1u) * 512u) /
                       (2u * (uint32_t)tokens));
        if (!encode_schedule(state, offset)) return -1.0;
    }
    return (now_ms() - start) / (double)tokens;
}

static int compare_double(const void *lhs, const void *rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return a < b ? -1 : (a > b ? 1 : 0);
}

static int run_timing(bench_state *state, const bench_config *cfg) {
    double *measurements = calloc((size_t)VARIANT_COUNT * (size_t)cfg->samples,
                                  sizeof(double));
    double *sorted = malloc((size_t)cfg->samples * sizeof(double));
    double medians[VARIANT_COUNT] = {0};
    if (!measurements || !sorted) {
        free(sorted);
        free(measurements);
        return 0;
    }

    for (size_t vi = 0; vi < VARIANT_COUNT; vi++) {
        for (int warm = 0; warm < cfg->warmup; warm++) {
            if (time_variant(state, g_variants[vi].nwg, 1) < 0.0) {
                fprintf(stderr, "%s: warmup failed for %s\n",
                        BENCH_NAME, g_variants[vi].name);
                free(sorted);
                free(measurements);
                return 0;
            }
        }
    }

    /* Forward/reverse blocks keep each variant equally exposed to drift. */
    for (int sample = 0; sample < cfg->samples; sample++) {
        for (size_t order = 0; order < VARIANT_COUNT; order++) {
            const size_t vi = (sample & 1)
                ? VARIANT_COUNT - 1u - order
                : order;
            const double ms = time_variant(state, g_variants[vi].nwg, cfg->tokens);
            if (ms < 0.0) {
                fprintf(stderr, "%s: timing failed for %s\n",
                        BENCH_NAME, g_variants[vi].name);
                free(sorted);
                free(measurements);
                return 0;
            }
            measurements[vi * (size_t)cfg->samples + (size_t)sample] = ms;
        }
    }

    for (size_t vi = 0; vi < VARIANT_COUNT; vi++) {
        memcpy(sorted,
               measurements + vi * (size_t)cfg->samples,
               (size_t)cfg->samples * sizeof(double));
        qsort(sorted, (size_t)cfg->samples, sizeof(double), compare_double);
        if (cfg->samples & 1) {
            medians[vi] = sorted[cfg->samples / 2];
        } else {
            medians[vi] =
                0.5 * (sorted[cfg->samples / 2 - 1] + sorted[cfg->samples / 2]);
        }
    }

    const double legacy_ms = medians[1];
    const uint32_t first_offset = 512u / (2u * (uint32_t)cfg->tokens);
    const uint32_t last_offset =
        (uint32_t)(((uint64_t)(2u * (uint32_t)cfg->tokens - 1u) * 512u) /
                   (2u * (uint32_t)cfg->tokens));
    const uint32_t ratio4_first = 256u + (first_offset + 1u) / 4u;
    const uint32_t ratio4_last = 256u + (last_offset + 1u) / 4u;
    const uint32_t ratio128_first = 132u + (first_offset + 1u) / 128u;
    const uint32_t ratio128_last = 132u + (last_offset + 1u) / 128u;
    printf("\nAttention-only 43-layer ctx=512->1024 sampled schedule\n");
    printf("  ctx range: 2 x raw-128, 21 x ratio4 keys=256..384, "
           "20 x ratio128 keys=132..136\n");
    printf("  midpoint samples exercise ratio4 keys=%u..%u, "
           "ratio128 keys=%u..%u\n",
           ratio4_first, ratio4_last, ratio128_first, ratio128_last);
    printf("  balanced samples=%d schedules/sample=%d warmup=%d\n",
           cfg->samples, cfg->tokens, cfg->warmup);
    printf("  caveat: this reuses shared q/KV/output buffers across layers; "
           "treat it as a kernel A/B, not an end-to-end TP throughput result\n");
    printf("  %-18s %10s %10s %10s %10s\n",
           "variant", "median-ms", "min-ms", "max-ms", "vs-NWG32");
    int unstable = 0;
    for (size_t vi = 0; vi < VARIANT_COUNT; vi++) {
        memcpy(sorted,
               measurements + vi * (size_t)cfg->samples,
               (size_t)cfg->samples * sizeof(double));
        qsort(sorted, (size_t)cfg->samples, sizeof(double), compare_double);
        const double delta = (legacy_ms / medians[vi] - 1.0) * 100.0;
        if (sorted[cfg->samples - 1] > sorted[0] * 1.20) unstable = 1;
        printf("  %-18s %10.3f %10.3f %10.3f %+9.2f%%\n",
               g_variants[vi].name,
               medians[vi],
               sorted[0],
               sorted[cfg->samples - 1],
               delta);
    }
    if (unstable) {
        printf("  WARNING: one or more sample ranges exceed 20%%; "
               "do not quote this timing run\n");
    }

    free(sorted);
    free(measurements);
    return 1;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const bench_config cfg = parse_options(argc, argv);
    bench_state state;
    if (!init_state(&state)) {
        cleanup_state(&state);
        return 1;
    }

    printf("\n== Model-free Metal FlashAttention decode NWG benchmark ==\n");
    printf("TP=2 geometry is enabled locally; no peer, gate, or GGUF is used.\n");

    const int base_correct = run_correctness(&state);
    const int fused_correct = run_fused_rope_correctness(&state);
    const int correct = base_correct && fused_correct;
    const int timed = cfg.correctness_only ? 1 : run_timing(&state, &cfg);
    (void)unsetenv(NWG_ENV);
    cleanup_state(&state);

    if (!correct || !timed) {
        fprintf(stderr, "%s: failed\n", BENCH_NAME);
        return 1;
    }
    printf("\n%s: OK\n", BENCH_NAME);
    return 0;
}
