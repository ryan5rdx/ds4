/* test_engine_mgpu_placement — wave-2 placement-classification regression.
 *
 * Exercises the engine-side classify path (tensor_to_entry,
 * engine_compute_entry_bytes, engine_classify_multi_tier) via the
 * DS4_TEST_HOOKS-gated public helpers. Compiles only when ds4.c is
 * built with -DDS4_TEST_HOOKS (the test target adds this flag).
 *
 * Scenarios:
 *  1. NULL config: no_op, multi_tier == 0, n_entries == 0.
 *  2. Tensor classifier: bounded ds4_str parsing (no NUL).
 *  3. Forced multi-tier no-CPU placement: 2 GPUs, both budgets force a
 *     transition without CPU spill. multi_tier == 1, monotonic, both
 *     tiers used.
 *  4. CPU-spill placement: 2 GPUs with tiny budgets so some layers
 *     spill. multi_tier == 1 and at least one DS4_LAYER_PACK_CPU entry.
 *  5. GLM compact-cache accounting: ordinary, indexed, and NextN layers.
 *  6. GLM batched-session placement scales independent cache allocations. */

#define DS4_TEST_HOOKS
#include "../ds4.h"
#include "../ds4_gpu_mgpu.h"
#include "../ds4_layer_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* These match the typedef in ds4.c under DS4_TEST_HOOKS. */
typedef struct {
    const char *name;
    uint64_t    bytes;
} ds4_test_fake_tensor;

int ds4_test_classify_multi_tier(const ds4_test_fake_tensor *tensors,
                                  int n_tensors,
                                  const ds4_gpu_config *cfg,
                                  int placement_out[],
                                  int *out_multi_tier,
                                  int *out_n_entries);
int ds4_test_tensor_to_entry(const char *name, int name_len);

/* Ctx-aware variants and calibration helpers. Declared here (not in
 * ds4.h) matching the existing DS4_TEST_HOOKS pattern. */
int ds4_test_classify_multi_tier_with_ctx(const ds4_test_fake_tensor *tensors,
                                           int n_tensors,
                                           const ds4_gpu_config *cfg,
                                           int placement_ctx_hint,
                                           int placement_out[],
                                           int *out_multi_tier,
                                           int *out_n_entries);
int ds4_test_classify_multi_tier_with_ctx_cuda_tp(
                                           const ds4_test_fake_tensor *tensors,
                                           int n_tensors,
                                           const ds4_gpu_config *cfg,
                                           int placement_ctx_hint,
                                           int placement_out[],
                                           int *out_multi_tier,
                                           int *out_n_entries);
void   ds4_test_seed_compress_ratios(void);
void   ds4_test_clear_compress_ratios(void);
size_t ds4_test_per_tier_graph_overhead_bytes(int placement_ctx_hint);
size_t ds4_test_per_tier_graph_overhead_bytes_with_prefill(
                                         int placement_ctx_hint,
                                         uint32_t prefill_chunk);
size_t ds4_test_compute_entry_bytes_sum(const ds4_test_fake_tensor *tensors,
                                         int n_tensors,
                                         int placement_ctx_hint);
size_t ds4_test_compute_entry_bytes_sum_with_prefill(
                                         const ds4_test_fake_tensor *tensors,
                                         int n_tensors,
                                         int placement_ctx_hint,
                                         uint32_t prefill_chunk);
uint32_t ds4_test_effective_prefill_chunk(bool cuda_tensor_parallel,
                                          uint32_t requested_chunk);
uint32_t ds4_test_planner_prefill_cap(int prompt_len,
                                      uint32_t prefill_chunk);
uint32_t ds4_test_planner_raw_cap(int ctx_size, uint32_t prefill_cap);
uint32_t ds4_test_prefill_watchdog_chunk(uint32_t prompt_len);
uint32_t ds4_test_glm53_prefill_chunk_ceiling(void);
uint32_t ds4_test_glm53_prefill_chunk_for_prompt(uint32_t prompt_len);
size_t ds4_test_glm_per_layer_kv_bytes(uint32_t layer, int ctx_size);
size_t ds4_test_compute_glm_entry_bytes_sum_with_sessions(
                                         const ds4_test_fake_tensor *tensors,
                                         int n_tensors,
                                         int placement_ctx_hint,
                                         int placement_session_count_hint);
uint64_t ds4_test_glm_memory_guard_default_budget(uint64_t host_bytes,
                                                   uint64_t model_bytes,
                                                   bool glm53);
int ds4_test_glm_memory_guard_disabled(void);
uint32_t ds4_test_glm53_tp_split_flags(void);
int ds4_test_glm53_kda_phase_splits(int mode, int phase);
int ds4_test_glm53_kda_split_mode_of(const char *env);
void ds4_test_glm53_layer_tp_gates(uint32_t il,
                                   uint32_t n_layer,
                                   uint32_t n_nextn,
                                   uint32_t n_leading_dense,
                                   int kda_split,
                                   int dense_ffn_split,
                                   int *fires_attn,
                                   int *fires_ffn);

/* DS4_N_LAYER constant is private to ds4.c; for the test we use
 * the same value. (The packer header doesn't expose it.) */
#define DS4_N_LAYER_LOCAL 43
#define DS4_N_VOCAB_LOCAL 129280
#define DS4_N_ENTRIES (DS4_N_LAYER_LOCAL + 2)

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
        g_failures++; \
    } \
} while (0)

static void test_tensor_to_entry(void) {
    fprintf(stderr, "RUN: test_tensor_to_entry\n");
    /* Bounded name buffer to confirm we never read past name_len. */
    char buf[64];

    /* "blk.0.attn_norm.weight" should map to entry 1 (layer 0 + 1). */
    memcpy(buf, "blk.0.attn_norm.weight", 22);
    CHECK(ds4_test_tensor_to_entry(buf, 22) == 1, "blk.0.* -> entry 1");

    /* "blk.42.ffn_norm.weight" -> entry 43 (layer 42 + 1). */
    memcpy(buf, "blk.42.ffn_norm.weight", 22);
    CHECK(ds4_test_tensor_to_entry(buf, 22) == 43, "blk.42.* -> entry 43");

    /* "blk.43.x" — layer 43 is out of range (DS4_N_LAYER=43, layers are 0..42) */
    memcpy(buf, "blk.43.x", 8);
    CHECK(ds4_test_tensor_to_entry(buf, 8) == 0, "blk.43.* out of range");

    /* "output.weight" -> entry 44 (head). */
    memcpy(buf, "output.weight", 13);
    CHECK(ds4_test_tensor_to_entry(buf, 13) == 44, "output.weight -> entry 44");

    /* "output_norm.weight" -> entry 44. */
    memcpy(buf, "output_norm.weight", 18);
    CHECK(ds4_test_tensor_to_entry(buf, 18) == 44, "output_norm.weight -> entry 44");

    /* "token_embd.weight" -> entry 0. */
    memcpy(buf, "token_embd.weight", 17);
    CHECK(ds4_test_tensor_to_entry(buf, 17) == 0, "token_embd.weight -> entry 0");

    /* "mtp.0.foo" -> entry 44. */
    memcpy(buf, "mtp.0.foo", 9);
    CHECK(ds4_test_tensor_to_entry(buf, 9) == 44, "mtp.* -> head");

    /* "output_hc_*.weight" -> entry 44 (head bucket). Regression for review
     * finding that the three output_hc_ tensors were falling through to
     * entry 0 (embedding tier) instead of the head tier. */
    memcpy(buf, "output_hc_base.weight", 21);
    CHECK(ds4_test_tensor_to_entry(buf, 21) == 44, "output_hc_base.weight -> head");
    memcpy(buf, "output_hc_fn.weight", 19);
    CHECK(ds4_test_tensor_to_entry(buf, 19) == 44, "output_hc_fn.weight -> head");
    memcpy(buf, "output_hc_scale.weight", 22);
    CHECK(ds4_test_tensor_to_entry(buf, 22) == 44, "output_hc_scale.weight -> head");
    /* "output.weight" / "output_norm.weight" still classified to head. */
    memcpy(buf, "output.weight", 13);
    CHECK(ds4_test_tensor_to_entry(buf, 13) == 44, "output.weight -> head");
    memcpy(buf, "output_norm.weight", 18);
    CHECK(ds4_test_tensor_to_entry(buf, 18) == 44, "output_norm.weight -> head");
    /* "token_embd.weight" stays at embedding (entry 0). */
    memcpy(buf, "token_embd.weight", 17);
    CHECK(ds4_test_tensor_to_entry(buf, 17) == 0, "token_embd.weight -> embedding");

    /* Bounded parsing: pass a long buffer with garbage past name_len. */
    const char with_trailing[] = "blk.5.attn_norm.weightTRAILINGGARBAGE";
    CHECK(ds4_test_tensor_to_entry(with_trailing, 22) == 6,
          "bounded parsing ignores trailing bytes");

    /* Empty name -> entry 0. */
    CHECK(ds4_test_tensor_to_entry("", 0) == 0, "empty name -> entry 0");
}

static void test_null_config(void) {
    fprintf(stderr, "RUN: test_null_config\n");
    int placement[DS4_N_ENTRIES];
    int multi_tier = 99;
    int n_entries = 99;

    /* A trivial fake tensor list. */
    ds4_test_fake_tensor tensors[] = {
        {"token_embd.weight", 4096},
        {"output.weight", 4096},
    };
    int rc = ds4_test_classify_multi_tier(tensors,
                                           (int)(sizeof(tensors)/sizeof(tensors[0])),
                                           NULL,
                                           placement, &multi_tier, &n_entries);
    CHECK(rc == 0, "NULL cfg returns success");
    CHECK(multi_tier == 0, "NULL cfg -> multi_tier 0");
    CHECK(n_entries == 0, "NULL cfg -> n_entries 0");
}

/* Build a synthetic, model-shaped tensor list: 1 embedding + 43 layers
 * (each with 2 tensors of equal size) + 1 output head. Used by the
 * multi-tier tests to drive a realistic placement decision. */
static int build_synthetic_model(ds4_test_fake_tensor *out, int cap) {
    int n = 0;
    static char names[1024][32];

    /* Embedding. */
    snprintf(names[n], 32, "token_embd.weight");
    out[n].name = names[n]; out[n].bytes = (uint64_t)8ull * 1024 * 1024;
    n++;

    /* Per-layer tensors. */
    for (int il = 0; il < DS4_N_LAYER_LOCAL; il++) {
        snprintf(names[n], 32, "blk.%d.attn_q.weight", il);
        out[n].name = names[n]; out[n].bytes = (uint64_t)256ull * 1024 * 1024;
        n++;
        snprintf(names[n], 32, "blk.%d.ffn_down.weight", il);
        out[n].name = names[n]; out[n].bytes = (uint64_t)768ull * 1024 * 1024;
        n++;
        if (n + 2 > cap) return -1;
    }

    /* Output head. */
    snprintf(names[n], 32, "output.weight");
    out[n].name = names[n]; out[n].bytes = (uint64_t)16ull * 1024 * 1024;
    n++;
    snprintf(names[n], 32, "output_norm.weight");
    out[n].name = names[n]; out[n].bytes = (uint64_t)1ull * 1024 * 1024;
    n++;
    return n;
}

static void test_forced_two_tier_no_spill(void) {
    fprintf(stderr, "RUN: test_forced_two_tier_no_spill\n");
    ds4_test_fake_tensor tensors[256];
    int n = build_synthetic_model(tensors, 256);
    CHECK(n > 0, "synthetic model built");
    if (n <= 0) return;

    /* Sum approx total weights:
     *   1 embed + 43 layers * 1024 MiB + 1 head ~ 43 GiB.
     * Pick budgets that force a transition. The packer also adds a
     * per-layer KV estimate that the engine computes; using equal
     * budgets sized below the total guarantees a transition without
     * CPU spill. */
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    /* Total synthetic weights ~ 44 GiB plus per-layer KV estimate from
     * ds4_context_memory_estimate(CUDA, 4096). Pick budgets near half
     * the total so the packer is forced to split across both tiers
     * but with enough headroom on each to avoid CPU spill. */
    cfg.vram_bytes[0] = (size_t)28ull * 1024ull * 1024ull * 1024ull;
    cfg.vram_bytes[1] = (size_t)40ull * 1024ull * 1024ull * 1024ull;
    cfg.safety_margin_bytes = 0;

    int placement[DS4_N_ENTRIES];
    int multi_tier = 0;
    int n_entries = 0;
    int rc = ds4_test_classify_multi_tier(tensors, n, &cfg,
                                           placement, &multi_tier, &n_entries);
    CHECK(rc == 0, "classify succeeded");
    CHECK(n_entries == DS4_N_ENTRIES, "n_entries == DS4_N_LAYER + 2");
    CHECK(multi_tier == 1, "multi_tier set");

    /* Monotonic-contiguous (wave-1 packer guarantee): each successive
     * entry's tier is >= previous, with CPU treated as a higher
     * "spill" tier. We assert no decrease. */
    int prev = placement[0];
    int saw_0 = 0, saw_1 = 0, saw_cpu = 0;
    for (int i = 0; i < n_entries; i++) {
        int cur = placement[i];
        CHECK(cur == prev || cur > prev || cur == DS4_LAYER_PACK_CPU,
              "monotonic (cur >= prev or CPU)");
        if (cur == 0) saw_0 = 1;
        else if (cur == 1) saw_1 = 1;
        else if (cur == DS4_LAYER_PACK_CPU) saw_cpu = 1;
        prev = cur;
    }
    CHECK(saw_0 && saw_1, "both tiers used");
    CHECK(!saw_cpu, "no CPU spill for this budget");
}

static void test_cpu_spill(void) {
    fprintf(stderr, "RUN: test_cpu_spill\n");
    ds4_test_fake_tensor tensors[256];
    int n = build_synthetic_model(tensors, 256);
    if (n <= 0) return;

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    /* Tiny budgets: ~5 GiB each, but total weights are ~43 GiB +
     * per-layer KV estimate, so most layers spill to CPU. */
    cfg.vram_bytes[0] = (size_t)5ull * 1024ull * 1024ull * 1024ull;
    cfg.vram_bytes[1] = (size_t)5ull * 1024ull * 1024ull * 1024ull;

    int placement[DS4_N_ENTRIES];
    int multi_tier = 0;
    int n_entries = 0;
    int rc = ds4_test_classify_multi_tier(tensors, n, &cfg,
                                           placement, &multi_tier, &n_entries);
    CHECK(rc == 0, "classify succeeded");
    CHECK(multi_tier == 1, "multi_tier set with CPU spill");
    int any_cpu = 0;
    for (int i = 0; i < n_entries; i++) {
        if (placement[i] == DS4_LAYER_PACK_CPU) { any_cpu = 1; break; }
    }
    CHECK(any_cpu, "at least one CPU spill entry");
}

static void test_zero_budget_guard(void) {
    fprintf(stderr, "RUN: test_zero_budget_guard\n");
    ds4_test_fake_tensor tensors[256];
    int n = build_synthetic_model(tensors, 256);
    if (n <= 0) return;

    /* Regression for review finding: zero-init ds4_gpu_config with only
     * n_gpus and device_indices populated must be rejected at classify
     * time, not silently classified as all-CPU. */
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    /* vram_bytes[] intentionally left at zero. */

    int placement[DS4_N_ENTRIES];
    int multi_tier = 0;
    int n_entries = 0;
    int rc = ds4_test_classify_multi_tier(tensors, n, &cfg,
                                           placement, &multi_tier, &n_entries);
    CHECK(rc != 0, "classify rejects all-zero vram_bytes");
}

/* Exercise the placement_ctx_hint path in engine_compute_entry_bytes:
 * the same layout at a larger ctx must produce more spill or refusal,
 * proving the hint actually flows into per-layer KV pricing. */
static void test_placement_ctx_hint_scales(void) {
    fprintf(stderr, "RUN: test_placement_ctx_hint_scales\n");
    ds4_test_fake_tensor tensors[256];
    int n = build_synthetic_model(tensors, 256);
    if (n <= 0) return;

    /* Seed FLASH compress ratios so the planner sees ratio==4 on half
     * the layers; without this, min_ratio==est_ctx in test mode and the
     * per-layer KV / per-tier overhead don't scale meaningfully with
     * ctx. */
    ds4_test_seed_compress_ratios();

    /* Two-GPU budgets sized so that ctx=4096 fits cleanly but ctx=131072
     * forces CPU spill (or refusal). */
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    cfg.vram_bytes[0] = (size_t)24ull * 1024ull * 1024ull * 1024ull;
    cfg.vram_bytes[1] = (size_t)24ull * 1024ull * 1024ull * 1024ull;
    cfg.safety_margin_bytes = 0;

    int placement_small[DS4_N_ENTRIES] = {0};
    int placement_big[DS4_N_ENTRIES]   = {0};
    int mt_small = 0, mt_big = 0;
    int ne_small = 0, ne_big = 0;

    int rc_s = ds4_test_classify_multi_tier_with_ctx(
        tensors, n, &cfg, 4096, placement_small, &mt_small, &ne_small);
    CHECK(rc_s == 0, "ctx=4096 classify ok");
    int spill_s = 0;
    for (int i = 0; i < ne_small; i++)
        if (placement_small[i] == DS4_LAYER_PACK_CPU) spill_s++;

    int rc_b = ds4_test_classify_multi_tier_with_ctx(
        tensors, n, &cfg, 131072, placement_big, &mt_big, &ne_big);
    /* rc_b may be 0 (with spill) or -1 (per-tier overhead refusal). */
    int spill_b = 0;
    for (int i = 0; i < ne_big; i++)
        if (placement_big[i] == DS4_LAYER_PACK_CPU) spill_b++;

    /* The discriminator: at the larger ctx hint the layout MUST be
     * different — more spill OR upfront refusal. */
    CHECK(rc_b != 0 || spill_b > spill_s,
          "placement_ctx_hint plumbs through to per-layer KV / per-tier "
          "overhead — larger ctx forces more spill (or refusal).");

    ds4_test_clear_compress_ratios();
}

/* Verifies the per-tier overhead pre-subtract actually changes a
 * packer decision: at a budget that fits WITHOUT the pre-subtract, the
 * layout must spill or refuse WITH it; at 1.5× the overhead headroom,
 * the layout must still fit (counter-control). */
static void test_pertier_overhead_pushes_to_spill(void) {
    fprintf(stderr, "RUN: test_pertier_overhead_pushes_to_spill\n");
    ds4_test_fake_tensor tensors[256];
    int n = build_synthetic_model(tensors, 256);
    if (n <= 0) return;

    /* Seed compress ratios so the per-tier overhead has its real
     * (non-collapsed) magnitude. */
    ds4_test_seed_compress_ratios();

    /* Query EXACT planner numbers at ctx=4096 — same code paths the real
     * classify will hit. No approximations. */
    const size_t entry_sum = ds4_test_compute_entry_bytes_sum(tensors, n, 4096);
    const size_t overhead  = ds4_test_per_tier_graph_overhead_bytes(4096);
    CHECK(entry_sum > 0, "planner entry-bytes sum > 0");
    CHECK(overhead > 0,  "per-tier overhead > 0 with seeded compress ratios");

    /* Budget = entry_sum + cublas + 0.6*overhead.
     * WITHOUT pre-subtract: pcfg.gpu_budget = entry_sum + 0.6*overhead
     *   → fits with 0.6*overhead spare.
     * WITH pre-subtract: pcfg.gpu_budget = entry_sum - 0.4*overhead
     *   → packer must spill 0.4*overhead worth of entries. */
    const size_t cublas_workspace = (size_t)64ull * 1024ull * 1024ull;
    const size_t headroom = overhead * 6 / 10;
    const size_t budget = entry_sum + cublas_workspace + headroom;

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    cfg.vram_bytes[0] = budget;
    cfg.safety_margin_bytes = 0;

    int placement[DS4_N_ENTRIES] = {0};
    int multi_tier = 0;
    int n_entries = 0;
    int rc = ds4_test_classify_multi_tier(tensors, n, &cfg,
                                           placement, &multi_tier, &n_entries);

    if (rc == 0) {
        int any_cpu = 0;
        for (int i = 0; i < n_entries; i++) {
            if (placement[i] == DS4_LAYER_PACK_CPU) { any_cpu = 1; break; }
        }
        CHECK(any_cpu,
              "per-tier overhead pre-subtract pushes layout to CPU spill");
    } else {
        CHECK(rc == -1,
              "per-tier overhead pre-subtract refuses upfront (budget < overhead)");
    }

    /* Counter-control: with budget = entry_sum + cublas + 1.5*overhead the
     * layout MUST fit even AFTER the pre-subtract — verifies the test
     * isn't asserting on noise. */
    cfg.vram_bytes[0] = entry_sum + cublas_workspace + overhead * 3 / 2;
    int placement2[DS4_N_ENTRIES] = {0};
    int mt2 = 0, ne2 = 0;
    int rc2 = ds4_test_classify_multi_tier(tensors, n, &cfg,
                                            placement2, &mt2, &ne2);
    CHECK(rc2 == 0, "1.5x-overhead budget classify ok");
    int spill2 = 0;
    for (int i = 0; i < ne2; i++)
        if (placement2[i] == DS4_LAYER_PACK_CPU) spill2++;
    CHECK(spill2 == 0,
          "1.5x-overhead budget fits without CPU spill (control)");

    ds4_test_clear_compress_ratios();
}

/* Per-tier scratch must not be charged BOTH per layer (in
 * engine_per_layer_kv_bytes_planner) AND per tier (in
 * engine_per_tier_graph_overhead_bytes). At large ctx, double-counting
 * inflates entry_sum by tens of GiB and falsely refuses valid layouts.
 * Per-layer math charges KV/index ONLY; per-tier scratch is reserved
 * separately by the overhead pre-subtract. */
static void test_no_per_layer_scratch_double_count(void) {
    fprintf(stderr, "RUN: test_no_per_layer_scratch_double_count\n");
    ds4_test_fake_tensor tensors[256];
    int n = build_synthetic_model(tensors, 256);
    if (n <= 0) return;

    ds4_test_seed_compress_ratios();

    /* Entry-bytes delta as ctx grows 4096 -> 65536 must be dominated by
     * per-layer KV growth, NOT by per-layer scratch growth.
     *
     *   KV growth per layer (after fix): bounded by per-layer comp_cap
     *   delta ~ (65536/4 - 4096/4) * (head_dim + indexer_head_dim) * 4
     *         ~ 15360 * 160 * 4 = ~9.4 MB per layer
     *         x DS4_N_LAYER ~ <1 GiB total.
     *
     *   Scratch growth per layer (under bug): 2 * comp_cap * prefill_cap * 4
     *         ~ 2 * 16386 * 4096 * 4 = ~537 MB per layer at ctx=65536
     *         minus ~33 MB at ctx=4096 = ~504 MB delta per layer
     *         x DS4_N_LAYER ~ ~21 GiB total.
     *
     * 5 GiB bound discriminates cleanly: passes after fix, fails before. */
    const size_t small = ds4_test_compute_entry_bytes_sum(tensors, n, 4096);
    const size_t large = ds4_test_compute_entry_bytes_sum(tensors, n, 65536);
    const size_t delta = large > small ? large - small : 0;
    const size_t bound = (size_t)5ull * 1024ull * 1024ull * 1024ull;
    CHECK(delta < bound,
          "per-layer entry-bytes delta 4096->65536 is KV-only (no scratch double-count)");

    ds4_test_clear_compress_ratios();
}

static void test_glm_per_layer_cache_accounting(void) {
    fprintf(stderr, "RUN: test_glm_per_layer_cache_accounting\n");
    const uint64_t ctx = 100000u;
#if defined(__APPLE__)
    const uint64_t elem_bytes = sizeof(uint16_t);
#else
    const uint64_t elem_bytes = sizeof(float);
#endif
    const size_t base =
        (size_t)(ctx * (512u + 64u) * elem_bytes);
    const size_t indexed =
        (size_t)(ctx * (512u + 64u + 128u) * elem_bytes);

    CHECK(ds4_test_glm_per_layer_kv_bytes(4, (int)ctx) == base,
          "GLM normal layer includes compact KV and RoPE cache");
    CHECK(ds4_test_glm_per_layer_kv_bytes(6, (int)ctx) == indexed,
          "GLM indexed layer also includes compact indexer cache");
    CHECK(ds4_test_glm_per_layer_kv_bytes(78, (int)ctx) == 0,
          "GLM NextN layer has no generation cache");
}

static void test_glm_session_count_accounting(void) {
    fprintf(stderr, "RUN: test_glm_session_count_accounting\n");
    ds4_test_fake_tensor tensors[256];
    const int n = build_synthetic_model(tensors, 256);
    if (n <= 0) return;

    size_t weights = 0;
    for (int i = 0; i < n; i++) weights += (size_t)tensors[i].bytes;

    const size_t one = ds4_test_compute_glm_entry_bytes_sum_with_sessions(
        tensors, n, 4096, 1);
    const size_t unset = ds4_test_compute_glm_entry_bytes_sum_with_sessions(
        tensors, n, 4096, 0);
    const size_t four = ds4_test_compute_glm_entry_bytes_sum_with_sessions(
        tensors, n, 4096, 4);
    CHECK(one > weights, "single GLM session includes compact-cache bytes");
    CHECK(unset == one, "unset GLM session hint preserves one-session accounting");
    CHECK(four > weights && four - weights == 4u * (one - weights),
          "four GLM sessions reserve four independent compact caches");
}

static char *save_env_value(const char *name) {
    const char *v = getenv(name);
    if (!v) return NULL;
    size_t n = strlen(v) + 1;
    char *copy = malloc(n);
    if (copy) memcpy(copy, v, n);
    return copy;
}

static void restore_env_value(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

static void test_glm_memory_guard_budget(void) {
    fprintf(stderr, "RUN: test_glm_memory_guard_budget\n");
    const uint64_t gib = 1024ull * 1024ull * 1024ull;

    CHECK(ds4_test_glm_memory_guard_default_budget(
                  128ull * gib, 90ull * gib, true) == 110ull * gib,
          "GLM-5.3 keeps 18 GiB free on a 128 GiB resident-Q2 host");
    CHECK(ds4_test_glm_memory_guard_default_budget(
                  112ull * gib, 90ull * gib, true) == 94ull * gib,
          "GLM-5.3 recognizes a 128 GB ROCm host by available GiB");
    CHECK(ds4_test_glm_memory_guard_default_budget(
                  256ull * gib, 178ull * gib, true) == 224ull * gib,
          "GLM-5.3 uses the host-sized budget on a 256 GiB host");
    CHECK(ds4_test_glm_memory_guard_default_budget(
                  256ull * gib, 178ull * gib, false) == 224ull * gib,
          "larger-host budget is model-variant independent");

    char *old_guard = save_env_value("DS4_GLM_MEMORY_GUARD");
    unsetenv("DS4_GLM_MEMORY_GUARD");
    CHECK(ds4_test_glm_memory_guard_disabled() == 0,
          "memory guard defaults to enabled");
    setenv("DS4_GLM_MEMORY_GUARD", "0", 1);
    CHECK(ds4_test_glm_memory_guard_disabled() == 1,
          "DS4_GLM_MEMORY_GUARD=0 disables the guard for every GLM variant");
    setenv("DS4_GLM_MEMORY_GUARD", "false", 1);
    CHECK(ds4_test_glm_memory_guard_disabled() == 1,
          "false spelling disables the memory guard");
    setenv("DS4_GLM_MEMORY_GUARD", "1", 1);
    CHECK(ds4_test_glm_memory_guard_disabled() == 0,
          "DS4_GLM_MEMORY_GUARD=1 keeps the guard enabled");
    restore_env_value("DS4_GLM_MEMORY_GUARD", old_guard);
}

static void test_cuda_tp_prefill_default_accounting(void) {
    fprintf(stderr, "RUN: test_cuda_tp_prefill_default_accounting\n");

    CHECK(ds4_test_effective_prefill_chunk(true, 0) == 2048,
          "CUDA TP defaults to a 2048-token prefill chunk");
    CHECK(ds4_test_effective_prefill_chunk(true, 4096) == 4096,
          "CUDA TP preserves an explicit prefill chunk");
    CHECK(ds4_test_effective_prefill_chunk(false, 0) == 0,
          "ordinary inference retains its model-specific default");

    ds4_test_fake_tensor tensors[256];
    const int n = build_synthetic_model(tensors, 256);
    if (n <= 0) return;

    char *old_chunk = save_env_value("DS4_METAL_PREFILL_CHUNK");
    char *old_raw = save_env_value("DS4_METAL_GRAPH_RAW_CAP");
    unsetenv("DS4_METAL_PREFILL_CHUNK");
    unsetenv("DS4_METAL_GRAPH_RAW_CAP");
    ds4_test_seed_compress_ratios();

    const uint32_t ordinary_prefill =
        ds4_test_planner_prefill_cap(100000, 0);
    const uint32_t cuda_tp_prefill =
        ds4_test_planner_prefill_cap(100000, 2048);
    CHECK(ordinary_prefill == 4096,
          "ordinary long-context prefill cap remains 4096");
    CHECK(cuda_tp_prefill == 2048,
          "CUDA TP long-context prefill cap is 2048");
    CHECK(ds4_test_planner_raw_cap(100000, cuda_tp_prefill) <
          ds4_test_planner_raw_cap(100000, ordinary_prefill),
          "CUDA TP prefill default reduces raw KV allocation");

    const size_t ordinary_entries =
        ds4_test_compute_entry_bytes_sum_with_prefill(tensors, n, 100000, 0);
    const size_t cuda_tp_entries =
        ds4_test_compute_entry_bytes_sum_with_prefill(tensors, n, 100000, 2048);
    const size_t ordinary_scratch =
        ds4_test_per_tier_graph_overhead_bytes_with_prefill(100000, 0);
    const size_t cuda_tp_scratch =
        ds4_test_per_tier_graph_overhead_bytes_with_prefill(100000, 2048);
    CHECK(cuda_tp_entries < ordinary_entries,
          "placement KV accounting uses the effective CUDA TP chunk");
    CHECK(cuda_tp_scratch < ordinary_scratch,
          "placement scratch accounting uses the effective CUDA TP chunk");

    ds4_test_clear_compress_ratios();
    restore_env_value("DS4_METAL_PREFILL_CHUNK", old_chunk);
    restore_env_value("DS4_METAL_GRAPH_RAW_CAP", old_raw);
}

static int build_output_tp_head_move_model(ds4_test_fake_tensor *out, int cap) {
    if (cap < DS4_N_LAYER_LOCAL + 2) return -1;
    int n = 0;
    static char names[DS4_N_LAYER_LOCAL + 2][32];
    const uint64_t mib = 1024ull * 1024ull;

    snprintf(names[n], sizeof(names[n]), "token_embd.weight");
    out[n].name = names[n];
    out[n].bytes = 1536ull * mib;
    n++;

    for (int il = 0; il < DS4_N_LAYER_LOCAL; il++) {
        snprintf(names[n], sizeof(names[n]), "blk.%d.ffn_gate_exps.weight", il);
        out[n].name = names[n];
        out[n].bytes = 3550ull * mib;
        n++;
    }

    snprintf(names[n], sizeof(names[n]), "output.weight");
    out[n].name = names[n];
    out[n].bytes = ((1536ull * mib) / DS4_N_VOCAB_LOCAL) * DS4_N_VOCAB_LOCAL;
    n++;
    return n;
}

static void test_cuda_tp_output_head_moves_to_lower_half(void) {
    fprintf(stderr, "RUN: test_cuda_tp_output_head_moves_to_lower_half\n");
    ds4_test_fake_tensor tensors[DS4_N_LAYER_LOCAL + 2];
    int n = build_output_tp_head_move_model(tensors,
                                            (int)(sizeof(tensors) / sizeof(tensors[0])));
    CHECK(n > 0, "output-head synthetic model built");
    if (n <= 0) return;

    char *old_pipe = save_env_value("DS4_CUDA_PREFILL_PIPELINE");
    char *old_chunk = save_env_value("DS4_METAL_PREFILL_CHUNK");
    unsetenv("DS4_CUDA_PREFILL_PIPELINE");
    unsetenv("DS4_METAL_PREFILL_CHUNK");

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 8;
    for (int i = 0; i < cfg.n_gpus; i++) {
        cfg.device_indices[i] = i;
        cfg.vram_bytes[i] = (size_t)42ull * 1024ull * 1024ull * 1024ull;
    }
    cfg.safety_margin_bytes = (size_t)512ull * 1024ull * 1024ull;

    int placement[DS4_N_ENTRIES] = {0};
    int multi_tier = 0;
    int n_entries = 0;
    int rc = ds4_test_classify_multi_tier_with_ctx_cuda_tp(tensors,
                                                           n,
                                                           &cfg,
                                                           4096,
                                                           placement,
                                                           &multi_tier,
                                                           &n_entries);
    CHECK(rc == 0, "CUDA TP output-head classify succeeds");
    CHECK(multi_tier == 1, "CUDA TP output-head model is multi-tier");
    CHECK(n_entries == DS4_N_ENTRIES, "CUDA TP output-head n_entries");
    const int last_layer_tier = placement[DS4_N_LAYER_LOCAL];
    CHECK(last_layer_tier >= 0 && last_layer_tier < cfg.n_gpus,
          "last layer remains on a GPU tier");
    CHECK(placement[DS4_N_LAYER_LOCAL + 1] >= 0 &&
          placement[DS4_N_LAYER_LOCAL + 1] < cfg.n_gpus / 2,
          "output head moved to a lower-half tier for output TP");

    restore_env_value("DS4_CUDA_PREFILL_PIPELINE", old_pipe);
    restore_env_value("DS4_METAL_PREFILL_CHUNK", old_chunk);
}

/* Regression for the prefill chunk ladder that keeps a chunked-prefill command
 * buffer clear of the GPU watchdog.
 *
 * The runtime bound shrinks as the prompt grows, on a work budget both ranks
 * compute identically -- they must chunk the same way or the per-layer gates
 * stop matching.  Allocation is deliberately NOT on that ladder: workspaces are
 * sized from the context, so they stay at the variant ceiling while only the
 * runtime bound moves.  This test pins both halves of that, plus the ladder's
 * shape (monotone, bounded, power-of-two, floored at 512) and the boundary
 * values observed on hardware.
 *
 * Model-free: it exercises the planner through its test hooks only. */
static void test_prefill_watchdog_bound(void) {
    fprintf(stderr, "RUN: test_prefill_watchdog_bound\n");

    char *old_chunk = save_env_value("DS4_METAL_PREFILL_CHUNK");
    unsetenv("DS4_METAL_PREFILL_CHUNK");

    /* cap(8192) is the variant's long-prompt ceiling (4096, or 8192 for PRO). */
    const uint32_t ceiling = ds4_test_planner_prefill_cap(8192, 0);

    static const int ladder[] = {
        8192, 65536, 131072, 154390, 163840, 232306, 262144, 327680,
        393216, 524288, 655360, 1000000,
        /* Past the point where the work budget alone would select a chunk
         * below the floor, so the floor assertion below is actually
         * exercised rather than vacuously true. */
        2000000,
    };
    const int n_ladder = (int)(sizeof(ladder) / sizeof(ladder[0]));

    /* ALLOCATION must NOT track the ladder: it is called with ctx_size. */
    bool alloc_flat = true;
    for (int i = 0; i < n_ladder; i++) {
        if (ds4_test_planner_prefill_cap(ladder[i], 0) != ceiling) {
            alloc_flat = false;
        }
    }
    CHECK(alloc_flat,
          "workspace sizing stays at the variant ceiling for every context");

    /* RUNTIME bound must shrink, monotonically, powers of two, floored. */
    bool monotone = true, bounded = true, floored = true, pow2 = true;
    uint32_t prev = ceiling;
    for (int i = 0; i < n_ladder; i++) {
        const uint32_t c = ds4_test_prefill_watchdog_chunk((uint32_t)ladder[i]);
        if (c > prev) monotone = false;
        if (c > ceiling) bounded = false;
        if (c < 512u) floored = false;
        if ((c & (c - 1u)) != 0u) pow2 = false;
        prev = c;
    }
    CHECK(monotone, "runtime bound never grows as the prompt grows");
    CHECK(bounded, "runtime bound never exceeds the variant ceiling");
    CHECK(floored, "runtime bound keeps its 512-token floor");
    CHECK(pow2, "runtime bound only selects power-of-two chunks");
    CHECK(prev < ceiling, "runtime bound has actually shrunk by 1M tokens");

    /* Boundary values observed to complete on hardware.  Recompute them from
     * DS4_PREFILL_CHUNK_WORK_BUDGET if the budget is retuned -- they are the
     * only part of this test that encodes a measurement rather than an
     * invariant. */
    CHECK(ds4_test_prefill_watchdog_chunk(154390) == 4096,
          "runtime bound keeps the hardware-proven 4096 at 154k tokens");
    CHECK(ds4_test_prefill_watchdog_chunk(232306) == 2048,
          "runtime bound reproduces the hardware-proven 2048 at 232k tokens");

    /* And the configuration observed to be fatal must be rejected: a 4096
     * chunk at ~271k tokens was killed on its first command buffer by the GPU
     * watchdog, which under tensor parallelism aborts the in-flight bulk gate
     * and is not recoverable. */
    CHECK(ds4_test_prefill_watchdog_chunk(271180) < 4096,
          "runtime bound rejects the 4096 chunk that was killed at 271k");

    /* GLM 5.3 runs the same two-level scheme: a fixed allocation ceiling, and a
     * runtime chunk that follows the shared ladder.  Before this, GLM had only
     * the constant, so a chunk proven at 131k would have been used unchanged at
     * every context above it. */
    {
        const uint32_t glm_ceiling = ds4_test_glm53_prefill_chunk_ceiling();
        CHECK(glm_ceiling == 4096,
              "GLM allocation ceiling matches the DeepSeek base");
        CHECK(ds4_test_glm53_prefill_chunk_for_prompt(0) == glm_ceiling,
              "an unknown prompt length sizes at the ceiling");
        CHECK(ds4_test_glm53_prefill_chunk_for_prompt(8192) == 4096,
              "GLM keeps the E2-measured 4096 chunk at 8k");
        CHECK(ds4_test_glm53_prefill_chunk_for_prompt(131072) == 4096,
              "GLM keeps the E2-measured 4096 chunk at 131k");
        CHECK(ds4_test_glm53_prefill_chunk_for_prompt(271180) < glm_ceiling,
              "GLM backs off past the context that killed a 4096 chunk");

        uint32_t glm_prev = glm_ceiling;
        bool glm_monotone = true, glm_bounded = true;
        for (size_t i = 0; i < sizeof(ladder) / sizeof(ladder[0]); i++) {
            const uint32_t c =
                ds4_test_glm53_prefill_chunk_for_prompt((uint32_t)ladder[i]);
            if (c > glm_prev) glm_monotone = false;
            if (c > glm_ceiling) glm_bounded = false;
            glm_prev = c;
        }
        CHECK(glm_monotone, "GLM runtime chunk never grows as the prompt grows");
        CHECK(glm_bounded, "GLM runtime chunk never exceeds its own ceiling");
        CHECK(glm_prev < glm_ceiling,
              "GLM runtime chunk has actually shrunk by 1M tokens");
    }

    /* GLM 5.3's TP gate firing rule.  The schedule mask and the graph encoders
     * must agree layer for layer: tp_gate_slot() walks the mask by ordinal, so
     * one unscheduled gate shifts every later gate onto the wrong slot and the
     * pair dies at its first exchange.  That is exactly how S6a failed --
     * the schedule started at layer 3 while the graph split KDA heads from
     * layer 0, so a layer-0 ATTN gate arrived on seq 1 with nothing to match.
     * GLM 5.3 shape: 46 layers, 1 nextn, 3 leading dense. */
    {
        const uint32_t N_LAYER = 46, N_NEXTN = 1, N_DENSE = 3;
        int attn = -1, ffn = -1;

        /* The leading three are dense in their FFN only; the attention is KDA
         * and gates as soon as the heads are split.  This is the S6a bug. */
        for (uint32_t il = 0; il < N_DENSE; il++) {
            ds4_test_glm53_layer_tp_gates(il, N_LAYER, N_NEXTN, N_DENSE, 0, 0,
                                          &attn, &ffn);
            CHECK(attn == 0 && ffn == 0,
                  "leading dense layer fires nothing with the KDA split off");
            ds4_test_glm53_layer_tp_gates(il, N_LAYER, N_NEXTN, N_DENSE, 1, 0,
                                          &attn, &ffn);
            CHECK(attn == 1 && ffn == 0,
                  "leading dense layer gates its KDA attention when split, "
                  "and still never gates its dense FFN");
        }

        /* Layer 3 is the first DSA layer and the first routed FFN. */
        ds4_test_glm53_layer_tp_gates(3, N_LAYER, N_NEXTN, N_DENSE, 0, 0,
                                      &attn, &ffn);
        CHECK(attn == 1 && ffn == 1,
              "a DSA layer gates attention regardless of the KDA split");
        ds4_test_glm53_layer_tp_gates(4, N_LAYER, N_NEXTN, N_DENSE, 0, 0,
                                      &attn, &ffn);
        CHECK(attn == 0 && ffn == 1,
              "a sparse KDA layer gates only its FFN with the split off");
        ds4_test_glm53_layer_tp_gates(4, N_LAYER, N_NEXTN, N_DENSE, 1, 0,
                                      &attn, &ffn);
        CHECK(attn == 1 && ffn == 1,
              "a sparse KDA layer gates both once the heads are split");

        /* The nextn layer runs on the coordinator alone and never gates. */
        ds4_test_glm53_layer_tp_gates(N_LAYER - 1, N_LAYER, N_NEXTN, N_DENSE, 1, 0,
                                      &attn, &ffn);
        CHECK(attn == 0 && ffn == 0, "the nextn layer fires no gate");

        /* S4 lights the FFN gate on the three leading dense layers, which fire
         * nothing today.  This is the same rule S6a got wrong twice, so pin it
         * from both directions. */
        for (uint32_t il = 0; il < N_DENSE; il++) {
            ds4_test_glm53_layer_tp_gates(il, N_LAYER, N_NEXTN, N_DENSE, 0, 1,
                                          &attn, &ffn);
            CHECK(attn == 0 && ffn == 1,
                  "S4 gates the leading dense FFN and nothing else");
            ds4_test_glm53_layer_tp_gates(il, N_LAYER, N_NEXTN, N_DENSE, 1, 1,
                                          &attn, &ffn);
            CHECK(attn == 1 && ffn == 1,
                  "S4 and the KDA split compose on a leading dense layer");
        }
        ds4_test_glm53_layer_tp_gates(N_LAYER - 1, N_LAYER, N_NEXTN, N_DENSE,
                                      1, 1, &attn, &ffn);
        CHECK(attn == 0 && ffn == 0,
              "S4 does not gate the nextn layer");
        {
            uint32_t on = 0, max_slot = 0;
            for (uint32_t il = 0; il < N_LAYER; il++) {
                int a1 = 0, f1 = 0;
                ds4_test_glm53_layer_tp_gates(il, N_LAYER, N_NEXTN, N_DENSE,
                                              1, 1, &a1, &f1);
                on += (uint32_t)(a1 + f1);
                if (a1) max_slot = il * 2u;
                if (f1) max_slot = il * 2u + 1u;
            }
            CHECK(on == 90, "KDA split plus S4 is 90 gates per token");
            CHECK(max_slot < N_LAYER * 2u,
                  "S4's extra gates still fit the 92-slot slab");
        }

        /* Totals, and that the mask still fits the slab.  34 KDA + 11 DSA
         * attention and 42 routed FFNs: 53 gates off, 87 on. */
        uint32_t off = 0, on = 0, max_slot = 0;
        for (uint32_t il = 0; il < N_LAYER; il++) {
            int a0 = 0, f0 = 0, a1 = 0, f1 = 0;
            ds4_test_glm53_layer_tp_gates(il, N_LAYER, N_NEXTN, N_DENSE, 0, 0,
                                          &a0, &f0);
            ds4_test_glm53_layer_tp_gates(il, N_LAYER, N_NEXTN, N_DENSE, 1, 0,
                                          &a1, &f1);
            off += (uint32_t)(a0 + f0);
            on += (uint32_t)(a1 + f1);
            if (a1) max_slot = il * 2u;
            if (f1) max_slot = il * 2u + 1u;
        }
        CHECK(off == 53, "53 gates per token with the KDA split off");
        CHECK(on == 87, "87 gates per token with the KDA split on");
        CHECK(on - off == 34,
              "the split adds one gate for each of the 34 KDA layers");
        CHECK(max_slot < N_LAYER * 2u,
              "every gated slot fits the n_layer * gates_per_layer slab");
    }

    /* Which KDA phases split, per mode.  Rig arm S6c died because the phase
     * was passed as a bare enum argument and the two call sites were swapped:
     * the decode encoder asked for the prefill lane and the prefill encoder for
     * the decode lane.  Under mode=decode that meant decode did NOT split while
     * the gate schedule still expected its ATTN gates, so the pair broke on the
     * first decode gate ("layer 3 gate 0 vs seq 1" -- layer 3 is the first DSA
     * layer, i.e. no KDA gate had fired at all).
     *
     * No local probe could catch it: probes drive the kernel directly and never
     * go through lane selection. This pins the policy the schedule and both
     * encoders now share. */
    {
        enum { OFF = 0, DECODE_ONLY = 1, BOTH = 2 };
        enum { PH_DECODE = 0, PH_PREFILL = 1 };

        CHECK(ds4_test_glm53_kda_split_mode_of(NULL) == OFF &&
              ds4_test_glm53_kda_split_mode_of("") == OFF &&
              ds4_test_glm53_kda_split_mode_of("0") == OFF,
              "unset/empty/0 leaves the KDA split off");
        CHECK(ds4_test_glm53_kda_split_mode_of("decode") == DECODE_ONLY,
              "\"decode\" selects the decode-only split");
        CHECK(ds4_test_glm53_kda_split_mode_of("both") == BOTH &&
              ds4_test_glm53_kda_split_mode_of("1") == BOTH,
              "\"both\" and legacy \"1\" split both phases");

        CHECK(!ds4_test_glm53_kda_phase_splits(OFF, PH_DECODE) &&
              !ds4_test_glm53_kda_phase_splits(OFF, PH_PREFILL),
              "off: neither phase splits");
        CHECK(ds4_test_glm53_kda_phase_splits(DECODE_ONLY, PH_DECODE),
              "decode mode splits DECODE -- the half S6c inverted");
        CHECK(!ds4_test_glm53_kda_phase_splits(DECODE_ONLY, PH_PREFILL),
              "decode mode leaves PREFILL replicated -- the other half");
        CHECK(ds4_test_glm53_kda_phase_splits(BOTH, PH_DECODE) &&
              ds4_test_glm53_kda_phase_splits(BOTH, PH_PREFILL),
              "both: each phase splits, i.e. S6b is unchanged");

        /* The decode gate schedule keys off the DECODE phase, so any mode whose
         * decode splits must light the KDA ATTN bits, and one whose decode does
         * not must leave them dark.  This is the schedule/graph agreement that
         * actually broke. */
        const uint32_t N_LAYER = 46, N_NEXTN = 1, N_DENSE = 3;
        int attn = -1, ffn = -1;
        for (int mode = OFF; mode <= BOTH; mode++) {
            const int decode_splits =
                ds4_test_glm53_kda_phase_splits(mode, PH_DECODE);
            /* layer 4: a sparse KDA layer */
            ds4_test_glm53_layer_tp_gates(4, N_LAYER, N_NEXTN, N_DENSE,
                                          decode_splits, 0, &attn, &ffn);
            CHECK(attn == decode_splits,
                  "KDA ATTN gate fires exactly when the decode phase splits");
            /* layer 3: DSA, always gates, whatever the mode */
            ds4_test_glm53_layer_tp_gates(3, N_LAYER, N_NEXTN, N_DENSE,
                                          decode_splits, 0, &attn, &ffn);
            CHECK(attn == 1, "DSA attention gates in every mode");
        }
    }

    /* The TP hello's split-capability word.  This is the field whose absence on
     * the worker would have refused every split-enabled bring-up -- including
     * the shipping DS4_GLM_TP_KDA_SPLIT=decode config -- while building clean
     * under -Wall -Wextra and reporting it as a model mismatch.
     *
     * The structural fix is that split_flags is an out-param of
     * ds4_engine_tp_gate_schedule(), so a site that forgets it does not compile.
     * These checks pin the payload: it must be a pure function of the
     * environment, so the two ranks agree by construction. */
    {
        const char *keys[] = { "DS4_GLM_TP_KDA_SPLIT", "DS4_GLM_TP_SHARED_SPLIT",
                               "DS4_GLM_TP_DENSE_FFN_SPLIT",
                               "DS4_GLM_TP_VOCAB_SPLIT",
                               "DS4_GLM_TP_EXACT_PREFILL_MAX" };
        char *saved[5];
        for (int i = 0; i < 5; i++) {
            const char *v = getenv(keys[i]);
            saved[i] = v ? strdup(v) : NULL;
            unsetenv(keys[i]);
        }

        /* On a non-GLM build the word is always 0, so nothing changes for
         * DeepSeek and an old peer still matches. */
        const uint32_t clean = ds4_test_glm53_tp_split_flags();
        CHECK(clean == 0, "no split env yields a zero capability word");

        setenv("DS4_GLM_TP_SHARED_SPLIT", "1", 1);
        CHECK(ds4_test_glm53_tp_split_flags() == clean,
              "capability word is shape-gated: zero on a non-GLM build");
        unsetenv("DS4_GLM_TP_SHARED_SPLIT");

        /* Determinism is the property that matters: both ranks run the same
         * binary and must compute the same word from the same environment. */
        setenv("DS4_GLM_TP_KDA_SPLIT", "decode", 1);
        setenv("DS4_GLM_TP_VOCAB_SPLIT", "1", 1);
        const uint32_t a1 = ds4_test_glm53_tp_split_flags();
        const uint32_t a2 = ds4_test_glm53_tp_split_flags();
        CHECK(a1 == a2, "capability word is a pure function of the environment");
        unsetenv("DS4_GLM_TP_KDA_SPLIT");
        unsetenv("DS4_GLM_TP_VOCAB_SPLIT");

        for (int i = 0; i < 5; i++) {
            if (saved[i]) { setenv(keys[i], saved[i], 1); free(saved[i]); }
            else unsetenv(keys[i]);
        }
    }

    /* Short prompts are untouched, and the clamp to prompt_len still holds. */
    CHECK(ds4_test_planner_prefill_cap(4096, 0) == 4096,
          "prompts at the whole-batch threshold are not chunked");
    CHECK(ds4_test_planner_prefill_cap(1000, 0) == 1000,
          "short prompts are capped by prompt length");
    CHECK(ds4_test_planner_prefill_cap(1000000, 1u << 30) == 1000000,
          "an oversized explicit chunk still clamps to prompt length");

    /* Precedence: an operator override must still win outright. */
    CHECK(ds4_test_planner_prefill_cap(1000000, 2048) == 2048,
          "an explicit prefill chunk wins over the default");
    setenv("DS4_METAL_PREFILL_CHUNK", "2048", 1);
    CHECK(ds4_test_planner_prefill_cap(1000000, 0) == 2048,
          "DS4_METAL_PREFILL_CHUNK wins over the default");
    CHECK(ds4_test_planner_prefill_cap(1000000, 4096) == 4096,
          "an explicit prefill chunk wins over DS4_METAL_PREFILL_CHUNK");
    unsetenv("DS4_METAL_PREFILL_CHUNK");

    restore_env_value("DS4_METAL_PREFILL_CHUNK", old_chunk);
}

int main(void) {
    test_tensor_to_entry();
    test_null_config();
    test_forced_two_tier_no_spill();
    test_cpu_spill();
    test_zero_budget_guard();
    test_placement_ctx_hint_scales();
    test_pertier_overhead_pushes_to_spill();
    test_no_per_layer_scratch_double_count();
    test_glm_per_layer_cache_accounting();
    test_glm_session_count_accounting();
    test_glm_memory_guard_budget();
    test_cuda_tp_prefill_default_accounting();
    test_prefill_watchdog_bound();
    test_cuda_tp_output_head_moves_to_lower_half();

    fprintf(stderr, "\ntest_engine_mgpu_placement: %d/%d checks passed (%d failed)\n",
            g_checks - g_failures, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
