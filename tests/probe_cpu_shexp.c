/*
 * CPU decode sidecar probe — can the shared expert fit in the routed-MoE
 * overlap window on CPU?
 *
 * The shared expert is the only compelling decode target because it is
 * independent of routed MoE after ffn_norm: each rank can compute its existing
 * 1024-lane S2 slice while the GPU runs its routed experts, then feed the
 * result into the FFN TP gate that already exists. No new exchange.
 *
 * The window is ~149 us/layer (6.25 ms/token over 42 layers). Three separate
 * Core ML CPU predictions measured ~201 us, which misses by ~52 us — but that
 * number overcounts launch overhead three times, and Core ML works in fp16.
 * The rank-local Q8 slice is only 12.75 MiB/layer against 24 MiB for fp16:
 *
 *     gate 4096->1024, up 4096->1024, down 1024->4096, Q8_0 at 34 B / 32 values
 *       = 3 x 4.456 MB = 13.37 MB/layer = 12.75 MiB
 *     13.37 MB / 149 us = ~90 GB/s
 *
 * ~90 GB/s is plausible on an M2 Ultra, so the question is real rather than
 * rhetorical, and it is a bandwidth question rather than a FLOP one.
 *
 * Arms:
 *   pairq8   the existing paired gate/up kernel: activation quantized to int8,
 *            then vdotq_s32. Transcribed from dot_q8_0_row_pair in ds4.c --
 *            it is static, so it cannot be linked, and a transcription that
 *            drifts would be worse than useless. `--check` verifies both arms
 *            against a scalar reference for exactly that reason.
 *   q8f32    Q8_0 weights against F32 activations directly, skipping the
 *            activation quantization the paired kernel needs. Costs more
 *            arithmetic (widen + convert instead of a dot-product
 *            instruction) and saves a pass over the activations plus the
 *            quantization itself.
 *
 * WEIGHTS ARE SYNTHETIC and that is fine here: this measures bandwidth and
 * instruction throughput at the production SHAPES, neither of which depends on
 * the values. Exactness is checked against a scalar reference, not against the
 * model.
 *
 * TIMING FROM THIS BOX IS NOT THE ANSWER. Standing rule on this campaign: the
 * dev box decides exactness, never timing. An M1 Max has roughly half an M2
 * Ultra's memory bandwidth and fewer cores, and this arm lives or dies on
 * bandwidth. Run it on the rig.
 *
 *   probe_cpu_shexp [--layers N] [--threads a,b,c] [--iters N] [--check]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sys/sysctl.h>
#include <pthread/qos.h>
#include <unistd.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define N_EMBD    4096u
#define N_LANE    1024u          /* n_ff_exp 2048 halved per rank under S2 */
#define N_FF_FULL 2048u          /* the down matrix is stored at FULL width */
#define QK        32u
#define BLK_BYTES 34u            /* fp16 scale + 32 int8 */
#define CLAMP     10.0f          /* DS4_SHAPE_GLM53.swiglu_clamp_exp */

static uint32_t g_layers  = 42;
static uint32_t g_iters   = 200;
static int      g_check   = 0;
/* Long-lived contention mode. The sweep is a benchmark; this is a LOAD, and it
 * has to outlive a decode run rather than finish during its setup -- the first
 * harness cut launched 100k iterations and then spent 19 seconds starting the
 * worker and loading the model, by which time the load was over and the
 * "concurrent" measurement had nothing running beside it. */
static double   g_duration = 0.0;      /* seconds; 0 = not a load run */
static double   g_duty     = 1.0;      /* 1.0 = continuous stress */
static const char *g_stopfile = NULL;
/* Idle policy between dispatches. Under a duty cycle the pool is idle most of
 * the time, and the two options are a real engineering choice rather than a
 * detail: SLEEP frees the cores the duty cycle is supposed to free but pays a
 * wake-up on every layer (measured: 157 -> 224 us at 8 threads), while SPIN
 * keeps the pool hot and holds the cores through the ~520 us gap between
 * layers that decode actually leaves. Production has to pick one; the probe
 * should price both rather than bake one in. */
static int      g_idle_spin = 0;

typedef struct { uint8_t *gate, *up, *down; } layer_w;

/* down is [n_embd][n_ff_exp] at FULL 2048 width; a rank reads a contiguous
 * 1024-lane slice out of each row. Packing that slice contiguously -- as the
 * first cut did -- turns a strided read into a streaming one and touches half
 * as many DRAM pages, which is exactly the sort of flattering the window
 * cannot afford. Rows are DN_FULL_RB apart and DN_RB wide. */
#define DN_FULL_RB ((uint64_t)(N_FF_FULL / QK) * BLK_BYTES)   /* 2176 */
#define DN_RB      ((uint64_t)(N_LANE / QK) * BLK_BYTES)      /* 1088 */
static uint64_t g_lane_off_bytes;   /* rank 1 starts half way along the row */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}
static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y);
}
static float f16d(uint16_t h) {           /* fp16 -> float, scales only */
    _Float16 v; memcpy(&v, &h, 2); return (float)v;
}
static uint16_t f16e(float f) {
    _Float16 v = (_Float16)f; uint16_t h; memcpy(&h, &v, 2); return h;
}

/* ------------------------------------------------------------ arm: pairq8
 * Transcribed VERBATIM from ds4.c's dot_q8_0_row_pair and dot_q8_0_row. The
 * first cut paraphrased: it reduced every block with vaddvq_s32 instead of
 * accumulating float32x4 lanes across block PAIRS and reducing once, and it
 * had no single-row kernel at all -- `down` called the paired one with the
 * same row twice and threw half the result away, roughly 4.2M wasted MACs per
 * layer, about a third extra arithmetic across the whole MLP. Both made the
 * 156 us figure pessimistic by an unknown amount, which is the worst kind.
 * --check verifies both against a scalar reference. */
static inline void dot_pair_q8(const uint8_t *r0, const uint8_t *r1,
                               const int8_t *xq, const float *xd,
                               uint64_t blocks, float *o0, float *o1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f);
    float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f);
    uint64_t b = 0;
    for (; b + 1 < blocks; b += 2) {
        uint16_t s00, s01, s10, s11;
        memcpy(&s00, r0 + b * BLK_BYTES, 2);
        memcpy(&s01, r0 + (b + 1) * BLK_BYTES, 2);
        memcpy(&s10, r1 + b * BLK_BYTES, 2);
        memcpy(&s11, r1 + (b + 1) * BLK_BYTES, 2);
        const int8_t *xq0 = xq + b * QK, *xq1 = xq + (b + 1) * QK;
        const int8x16_t xv00 = vld1q_s8(xq0),      xv01 = vld1q_s8(xq0 + 16);
        const int8x16_t xv10 = vld1q_s8(xq1),      xv11 = vld1q_s8(xq1 + 16);
        const int8_t *q00 = (const int8_t *)(r0 + b * BLK_BYTES + 2);
        const int8_t *q01 = (const int8_t *)(r0 + (b + 1) * BLK_BYTES + 2);
        const int8_t *q10 = (const int8_t *)(r1 + b * BLK_BYTES + 2);
        const int8_t *q11 = (const int8_t *)(r1 + (b + 1) * BLK_BYTES + 2);
        int32x4_t d00 = vdupq_n_s32(0);
        d00 = vdotq_s32(d00, vld1q_s8(q00),      xv00);
        d00 = vdotq_s32(d00, vld1q_s8(q00 + 16), xv01);
        int32x4_t d01 = vdupq_n_s32(0);
        d01 = vdotq_s32(d01, vld1q_s8(q01),      xv10);
        d01 = vdotq_s32(d01, vld1q_s8(q01 + 16), xv11);
        int32x4_t d10 = vdupq_n_s32(0);
        d10 = vdotq_s32(d10, vld1q_s8(q10),      xv00);
        d10 = vdotq_s32(d10, vld1q_s8(q10 + 16), xv01);
        int32x4_t d11 = vdupq_n_s32(0);
        d11 = vdotq_s32(d11, vld1q_s8(q11),      xv10);
        d11 = vdotq_s32(d11, vld1q_s8(q11 + 16), xv11);
        a00 = vfmaq_n_f32(a00, vcvtq_f32_s32(d00), f16d(s00) * xd[b]);
        a01 = vfmaq_n_f32(a01, vcvtq_f32_s32(d01), f16d(s01) * xd[b + 1]);
        a10 = vfmaq_n_f32(a10, vcvtq_f32_s32(d10), f16d(s10) * xd[b]);
        a11 = vfmaq_n_f32(a11, vcvtq_f32_s32(d11), f16d(s11) * xd[b + 1]);
    }
    if (b < blocks) {
        uint16_t s0, s1;
        memcpy(&s0, r0 + b * BLK_BYTES, 2);
        memcpy(&s1, r1 + b * BLK_BYTES, 2);
        const int8_t *xqb = xq + b * QK;
        const int8x16_t xv0 = vld1q_s8(xqb), xv1 = vld1q_s8(xqb + 16);
        const int8_t *q0 = (const int8_t *)(r0 + b * BLK_BYTES + 2);
        const int8_t *q1 = (const int8_t *)(r1 + b * BLK_BYTES + 2);
        int32x4_t d0 = vdupq_n_s32(0);
        d0 = vdotq_s32(d0, vld1q_s8(q0),      xv0);
        d0 = vdotq_s32(d0, vld1q_s8(q0 + 16), xv1);
        int32x4_t d1 = vdupq_n_s32(0);
        d1 = vdotq_s32(d1, vld1q_s8(q1),      xv0);
        d1 = vdotq_s32(d1, vld1q_s8(q1 + 16), xv1);
        a00 = vfmaq_n_f32(a00, vcvtq_f32_s32(d0), f16d(s0) * xd[b]);
        a10 = vfmaq_n_f32(a10, vcvtq_f32_s32(d1), f16d(s1) * xd[b]);
    }
    *o0 = vaddvq_f32(vaddq_f32(a00, a01));
    *o1 = vaddvq_f32(vaddq_f32(a10, a11));
#else
    float s0 = 0.0f, s1 = 0.0f;
    for (uint64_t b = 0; b < blocks; ++b) {
        uint16_t h0, h1;
        memcpy(&h0, r0 + b * BLK_BYTES, 2);
        memcpy(&h1, r1 + b * BLK_BYTES, 2);
        int a0 = 0, a1 = 0;
        for (uint32_t j = 0; j < QK; ++j) {
            const int xv = xq[b * QK + j];
            a0 += (int)((const int8_t *)(r0 + b * BLK_BYTES + 2))[j] * xv;
            a1 += (int)((const int8_t *)(r1 + b * BLK_BYTES + 2))[j] * xv;
        }
        s0 += f16d(h0) * xd[b] * (float)a0;
        s1 += f16d(h1) * xd[b] * (float)a1;
    }
    *o0 = s0; *o1 = s1;
#endif
}

/* The SINGLE-row kernel production uses for `down`. Its absence was the
 * expensive omission: calling the paired kernel with one row twice does two
 * dots and discards one. */
static inline float dot_row_q8(const uint8_t *r, const int8_t *xq,
                               const float *xd, uint64_t blocks) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    uint64_t b = 0;
    for (; b + 1 < blocks; b += 2) {
        uint16_t s0, s1;
        memcpy(&s0, r + b * BLK_BYTES, 2);
        memcpy(&s1, r + (b + 1) * BLK_BYTES, 2);
        const int8_t *q0 = (const int8_t *)(r + b * BLK_BYTES + 2);
        const int8_t *q1 = (const int8_t *)(r + (b + 1) * BLK_BYTES + 2);
        const int8_t *x0 = xq + b * QK, *x1 = xq + (b + 1) * QK;
        int32x4_t d0 = vdupq_n_s32(0);
        d0 = vdotq_s32(d0, vld1q_s8(q0),      vld1q_s8(x0));
        d0 = vdotq_s32(d0, vld1q_s8(q0 + 16), vld1q_s8(x0 + 16));
        int32x4_t d1 = vdupq_n_s32(0);
        d1 = vdotq_s32(d1, vld1q_s8(q1),      vld1q_s8(x1));
        d1 = vdotq_s32(d1, vld1q_s8(q1 + 16), vld1q_s8(x1 + 16));
        a0 = vfmaq_n_f32(a0, vcvtq_f32_s32(d0), f16d(s0) * xd[b]);
        a1 = vfmaq_n_f32(a1, vcvtq_f32_s32(d1), f16d(s1) * xd[b + 1]);
    }
    if (b < blocks) {
        uint16_t sb;
        memcpy(&sb, r + b * BLK_BYTES, 2);
        const int8_t *q = (const int8_t *)(r + b * BLK_BYTES + 2);
        const int8_t *x = xq + b * QK;
        int32x4_t d = vdupq_n_s32(0);
        d = vdotq_s32(d, vld1q_s8(q),      vld1q_s8(x));
        d = vdotq_s32(d, vld1q_s8(q + 16), vld1q_s8(x + 16));
        a0 = vfmaq_n_f32(a0, vcvtq_f32_s32(d), f16d(sb) * xd[b]);
    }
    return vaddvq_f32(vaddq_f32(a0, a1));
#else
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; ++b) {
        uint16_t h; memcpy(&h, r + b * BLK_BYTES, 2);
        const int8_t *q = (const int8_t *)(r + b * BLK_BYTES + 2);
        int a = 0;
        for (uint32_t j = 0; j < QK; ++j) a += (int)q[j] * (int)xq[b * QK + j];
        acc += f16d(h) * xd[b] * (float)a;
    }
    return acc;
#endif
}

/* ------------------------------------------------------------- arm: q8f32
 * Q8_0 weights against F32 activations. No activation quantization: no extra
 * pass over x, no per-token scale, and no quantization error on the
 * activation side. Pays for it by widening int8 to f32 instead of issuing a
 * dot-product instruction, so this is the arm that tests whether the window is
 * bound by bandwidth (where it should win, since the weights dominate traffic
 * either way) or by arithmetic (where it should lose). */
static inline void dot_pair_q8_f32(const uint8_t *r0, const uint8_t *r1,
                                   const float *x, uint64_t blocks,
                                   float *o0, float *o1) {
    float s0 = 0.0f, s1 = 0.0f;
#if defined(__ARM_NEON)
    for (uint64_t b = 0; b < blocks; ++b) {
        uint16_t h0, h1;
        memcpy(&h0, r0 + b * BLK_BYTES, 2);
        memcpy(&h1, r1 + b * BLK_BYTES, 2);
        const int8_t *q0 = (const int8_t *)(r0 + b * BLK_BYTES + 2);
        const int8_t *q1 = (const int8_t *)(r1 + b * BLK_BYTES + 2);
        const float *xp = x + b * QK;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        for (uint32_t j = 0; j < QK; j += 16) {
            const int8x16_t v0 = vld1q_s8(q0 + j), v1 = vld1q_s8(q1 + j);
            const int16x8_t l0 = vmovl_s8(vget_low_s8(v0)),  h0v = vmovl_s8(vget_high_s8(v0));
            const int16x8_t l1 = vmovl_s8(vget_low_s8(v1)),  h1v = vmovl_s8(vget_high_s8(v1));
            const float32x4_t xa = vld1q_f32(xp + j),      xb = vld1q_f32(xp + j + 4);
            const float32x4_t xc = vld1q_f32(xp + j + 8),  xd = vld1q_f32(xp + j + 12);
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l0))),  xa);
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l0))), xb);
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h0v))), xc);
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h0v))), xd);
            a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l1))),  xa);
            a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l1))), xb);
            a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h1v))), xc);
            a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h1v))), xd);
        }
        s0 += f16d(h0) * vaddvq_f32(a0);
        s1 += f16d(h1) * vaddvq_f32(a1);
    }
#else
    for (uint64_t b = 0; b < blocks; ++b) {
        uint16_t h0, h1;
        memcpy(&h0, r0 + b * BLK_BYTES, 2);
        memcpy(&h1, r1 + b * BLK_BYTES, 2);
        float a0 = 0.0f, a1 = 0.0f;
        for (uint32_t j = 0; j < QK; ++j) {
            const float xv = x[b * QK + j];
            a0 += (float)((const int8_t *)(r0 + b * BLK_BYTES + 2))[j] * xv;
            a1 += (float)((const int8_t *)(r1 + b * BLK_BYTES + 2))[j] * xv;
        }
        s0 += f16d(h0) * a0;
        s1 += f16d(h1) * a1;
    }
#endif
    *o0 = s0; *o1 = s1;
}

/* q8f32 has no production counterpart, so its single-row form is the paired
 * one with the second accumulator dropped rather than computed. */
static inline float dot_row_q8_f32(const uint8_t *r, const float *x,
                                   uint64_t blocks) {
    float s = 0.0f;
#if defined(__ARM_NEON)
    for (uint64_t b = 0; b < blocks; ++b) {
        uint16_t h; memcpy(&h, r + b * BLK_BYTES, 2);
        const int8_t *q = (const int8_t *)(r + b * BLK_BYTES + 2);
        const float *xp = x + b * QK;
        float32x4_t a = vdupq_n_f32(0.0f);
        for (uint32_t j = 0; j < QK; j += 16) {
            const int8x16_t v = vld1q_s8(q + j);
            const int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_s8(vget_high_s8(v));
            a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),  vld1q_f32(xp + j));
            a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(xp + j + 4));
            a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),  vld1q_f32(xp + j + 8));
            a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(xp + j + 12));
        }
        s += f16d(h) * vaddvq_f32(a);
    }
#else
    for (uint64_t b = 0; b < blocks; ++b) {
        uint16_t h; memcpy(&h, r + b * BLK_BYTES, 2);
        const int8_t *q = (const int8_t *)(r + b * BLK_BYTES + 2);
        float a = 0.0f;
        for (uint32_t j = 0; j < QK; ++j) a += (float)q[j] * x[b * QK + j];
        s += f16d(h) * a;
    }
#endif
    return s;
}


/* ------------------------------------------------------------ thread pool
 * Spawn-per-call is out of the question at 67 us a stage: pthread_create alone
 * would dominate. Persistent workers on a spin barrier, which is also what a
 * real integration would have to use. */
typedef void (*work_fn)(void *ud, uint32_t r0, uint32_t r1);
typedef struct {
    pthread_t th[32];
    uint32_t  n;
    volatile uint32_t gen, done, stop;
    work_fn   fn;
    void     *ud;
    uint32_t  rows;
    int       dynamic;                  /* 0 = equal partition, 1 = row tiles */
    volatile uint32_t next;             /* dynamic: the tile cursor */
    uint32_t  tile;
    qos_class_t qos;
} pool;
static pool g_pool;

static void pool_do(uint32_t id) {
    if (!g_pool.dynamic) {
        const uint32_t per = (g_pool.rows + g_pool.n - 1u) / g_pool.n;
        uint32_t r0 = id * per, r1 = r0 + per;
        if (r1 > g_pool.rows) r1 = g_pool.rows;
        if (r0 < r1) g_pool.fn(g_pool.ud, r0, r1);
        return;
    }
    /* Dynamic tiles. An equal partition finishes at the speed of its slowest
     * worker, so one E-core paces everyone; small tiles let a slow core take
     * fewer of them instead. Costs one atomic per tile. */
    for (;;) {
        const uint32_t r0 = __atomic_fetch_add(&g_pool.next, g_pool.tile,
                                               __ATOMIC_ACQ_REL);
        if (r0 >= g_pool.rows) return;
        uint32_t r1 = r0 + g_pool.tile;
        if (r1 > g_pool.rows) r1 = g_pool.rows;
        g_pool.fn(g_pool.ud, r0, r1);
    }
}

static void *pool_worker(void *vid) {
    const uint32_t id = (uint32_t)(uintptr_t)vid;
    /* macOS has no reliable P-core pinning -- THREAD_AFFINITY_POLICY is a
     * cache-affinity hint, not a binding -- so QoS is the only lever. It biases
     * placement rather than guaranteeing it, which is why the E-core arms below
     * are negative controls and not assumptions. */
    pthread_set_qos_class_self_np(g_pool.qos, 0);
    uint32_t seen = 0;
    for (;;) {
        /* Spin briefly, then back off. Under a duty cycle the workers are
         * idle most of the time by design, and a pure spin would keep every
         * core pinned through the nominal idle window -- which is the opposite
         * of what a 22%-duty load is supposed to represent, and would make the
         * contention arm measure a busy machine either way. */
        uint32_t idle = 0;
        while (__atomic_load_n(&g_pool.gen, __ATOMIC_ACQUIRE) == seen) {
            if (__atomic_load_n(&g_pool.stop, __ATOMIC_ACQUIRE)) return NULL;
            if (g_idle_spin || ++idle < 4000u) continue;
            struct timespec ts = { 0, 20000 };          /* 20 us */
            nanosleep(&ts, NULL);
        }
        seen = __atomic_load_n(&g_pool.gen, __ATOMIC_ACQUIRE);
        pool_do(id);
        __atomic_add_fetch(&g_pool.done, 1u, __ATOMIC_RELEASE);
    }
}
static void pool_start(uint32_t n, int dynamic, uint32_t tile, qos_class_t q) {
    g_pool.n = n; g_pool.gen = 0; g_pool.done = 0; g_pool.stop = 0;
    g_pool.dynamic = dynamic; g_pool.tile = tile ? tile : 64u; g_pool.qos = q;
    pthread_set_qos_class_self_np(q, 0);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, q, 0);
    for (uint32_t i = 1; i < n; ++i)
        pthread_create(&g_pool.th[i], &attr, pool_worker, (void *)(uintptr_t)i);
    pthread_attr_destroy(&attr);
}
static void pool_stop(void) {
    __atomic_store_n(&g_pool.stop, 1u, __ATOMIC_RELEASE);
    for (uint32_t i = 1; i < g_pool.n; ++i) pthread_join(g_pool.th[i], NULL);
}
static void pool_run(work_fn fn, void *ud, uint32_t rows) {
    g_pool.fn = fn; g_pool.ud = ud; g_pool.rows = rows;
    __atomic_store_n(&g_pool.next, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&g_pool.done, 0u, __ATOMIC_RELEASE);
    __atomic_add_fetch(&g_pool.gen, 1u, __ATOMIC_RELEASE);
    pool_do(0);                                      /* thread 0 is this one */
    while (__atomic_load_n(&g_pool.done, __ATOMIC_ACQUIRE) < g_pool.n - 1u) { }
}

/* ----------------------------------------------------------------- stages */
typedef struct {
    const layer_w *L;
    const float   *x;          /* f32 activation, N_EMBD */
    const int8_t  *xq;         /* int8 activation, N_EMBD */
    const float   *xd;          /* per-block activation scales */
    float         *mid;        /* N_LANE */
    const float   *midx;       /* f32 mid for the down stage */
    const int8_t  *midq;
    const float   *midd;
    float         *out;        /* N_EMBD */
    int            f32_path;
} stage_ctx;

static float silu_f(float v) { return v / (1.0f + expf(-v)); }

static void stage_gate_up(void *vud, uint32_t r0, uint32_t r1) {
    stage_ctx *c = vud;
    const uint64_t blocks = N_EMBD / QK;
    const uint64_t rb = blocks * BLK_BYTES;
    for (uint32_t r = r0; r < r1; ++r) {
        float g, u;
        if (c->f32_path)
            dot_pair_q8_f32(c->L->gate + r * rb, c->L->up + r * rb, c->x, blocks, &g, &u);
        else
            dot_pair_q8(c->L->gate + r * rb, c->L->up + r * rb, c->xq, c->xd,
                        blocks, &g, &u);
        /* Production clamps both before the SwiGLU (matvec_q8_k_mid_worker).
         * Two compares and two selects per row is not much, but leaving it out
         * makes the kernel cheaper than the one being proposed. */
        if (g > CLAMP) g = CLAMP;
        if (u > CLAMP) u = CLAMP;
        if (u < -CLAMP) u = -CLAMP;
        c->mid[r] = silu_f(g) * u;
    }
}
static void stage_down(void *vud, uint32_t r0, uint32_t r1) {
    stage_ctx *c = vud;
    const uint64_t blocks = N_LANE / QK;
    for (uint32_t r = r0; r < r1; ++r) {
        const uint8_t *row = c->L->down + (uint64_t)r * DN_FULL_RB + g_lane_off_bytes;
        c->out[r] = c->f32_path ? dot_row_q8_f32(row, c->midx, blocks)
                                : dot_row_q8(row, c->midq, c->midd, blocks);
    }
}

/* Q8_0 quantizes PER BLOCK OF 32, not once for the whole vector. The first cut
 * used a single scale, which understates both the work (one amax pass and one
 * reciprocal per block, not per vector) and the accuracy, and would have made
 * the 115 us figure not production-faithful. */
static void quantize_q8_0(const float *x, uint32_t n, int8_t *q, float *xd) {
    for (uint32_t b = 0; b < n / QK; ++b) {
        float amax = 0.0f;
        for (uint32_t j = 0; j < QK; ++j) {
            const float a = fabsf(x[b * QK + j]); if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        xd[b] = d;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        for (uint32_t j = 0; j < QK; ++j)
            q[b * QK + j] = (int8_t)lrintf(x[b * QK + j] * id);
    }
}

int main(int argc, char **argv) {
    /* P-oriented candidates first, then 20/24 as deliberate E-core negative
     * controls. An M2 Ultra is 16 P + 8 E, so 12-16 is the real range and the
     * last two exist to be shown worse. Production will also want 1-2 P-cores
     * left for the TP service and the Metal driver threads, so 12-14 beating 16
     * would not be a surprise even if 16 wins standalone. */
    uint32_t threads[12] = {8, 12, 14, 15, 16, 20, 24}; uint32_t n_threads = 7;
    int rank = 0;
    const char *only_qos = NULL, *only_part = NULL;
    /* Tile 64 over 1024 gate/up rows is only 16 tiles, so a 20- or 24-thread
     * arm cannot put every worker to work and the dynamic partition looks bad
     * for a reason that has nothing to do with E-cores. */
    uint32_t tiles[4] = {16, 32, 64}; uint32_t n_tiles = 3;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--layers")  && i + 1 < argc) g_layers = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) g_iters = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) g_check = 1;
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = atoi(argv[++i]);
        /* Pin one configuration -- the contention arm has to run the winner,
         * not sweep while a decode is trying to be measured next to it. */
        else if (!strcmp(argv[i], "--qos")  && i + 1 < argc) only_qos  = argv[++i];
        else if (!strcmp(argv[i], "--part") && i + 1 < argc) only_part = argv[++i];
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc) g_duration = atof(argv[++i]);
        else if (!strcmp(argv[i], "--duty") && i + 1 < argc) g_duty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--stop") && i + 1 < argc) g_stopfile = argv[++i];
        else if (!strcmp(argv[i], "--idle") && i + 1 < argc) g_idle_spin = !strcmp(argv[++i], "spin");
        else if (!strcmp(argv[i], "--tiles") && i + 1 < argc) {
            n_tiles = 0;
            for (char *t = strtok(argv[++i], ","); t && n_tiles < 4; t = strtok(NULL, ","))
                tiles[n_tiles++] = (uint32_t)atoi(t);
        }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            n_threads = 0;
            for (char *t = strtok(argv[++i], ","); t && n_threads < 8; t = strtok(NULL, ","))
                threads[n_threads++] = (uint32_t)atoi(t);
        }
    }

    const uint64_t gu_rb = (N_EMBD / QK) * BLK_BYTES;   /* 4352 */
    const uint64_t per_layer = N_LANE * gu_rb * 2u + N_EMBD * DN_RB;
    g_lane_off_bytes = rank ? DN_RB : 0u;

    printf("CPU decode shared-expert sidecar probe\n");
    printf("  shapes  gate/up %ux%u  down %ux%u   (S2: rank-local lane slice)\n",
           N_EMBD, N_LANE, N_LANE, N_EMBD);
    printf("  weights %.2f MB/layer, %.2f MB over %u layers\n",
           per_layer / 1e6, per_layer * g_layers / 1e6, g_layers);
    printf("  window  149 us/layer -> needs %.1f GB/s\n", per_layer / 149e-6 / 1e9);
    printf("  down is a STRIDED slice: %llu B read out of every %llu B row, "
           "rank %d offset %llu\n\n",
           (unsigned long long)DN_RB, (unsigned long long)DN_FULL_RB,
           rank, (unsigned long long)g_lane_off_bytes);

    layer_w *W = calloc(g_layers, sizeof(*W));
    for (uint32_t l = 0; l < g_layers; ++l) {
        W[l].gate = malloc(N_LANE * gu_rb);
        W[l].up   = malloc(N_LANE * gu_rb);
        W[l].down = malloc(N_EMBD * DN_FULL_RB);
        if (!W[l].gate || !W[l].up || !W[l].down) { puts("VOID: alloc"); return 1; }
        /* Distinct per layer so the rotation is a real rotation and not 42
         * views of one resident matrix -- the mistake ANE-CAP made. */
        uint32_t s = 12345u + l;
        uint8_t *bufs[3] = { W[l].gate, W[l].up, W[l].down };
        uint64_t sizes[3] = { N_LANE * gu_rb, N_LANE * gu_rb, N_EMBD * DN_FULL_RB };
        uint64_t rbs[3] = { gu_rb, gu_rb, DN_FULL_RB };
        for (int k = 0; k < 3; ++k) {
            for (uint64_t off = 0; off < sizes[k]; off += rbs[k])
                for (uint64_t b = 0; b < rbs[k] / BLK_BYTES; ++b) {
                    uint8_t *p = bufs[k] + off + b * BLK_BYTES;
                    const uint16_t h = f16e(0.01f + (float)(s % 7) * 0.002f);
                    memcpy(p, &h, 2);
                    for (uint32_t j = 0; j < QK; ++j) {
                        s = s * 1664525u + 1013904223u;
                        p[2 + j] = (uint8_t)(int8_t)((int)(s >> 24) - 128);
                    }
                }
        }
    }

    float *x = malloc(N_EMBD * sizeof(float));
    float *mid = malloc(N_LANE * sizeof(float));
    float *out = malloc(N_EMBD * sizeof(float));
    int8_t *xq = malloc(N_EMBD), *midq = malloc(N_LANE);
    float *xd = malloc((N_EMBD / QK) * sizeof(float));
    float *midd = malloc((N_LANE / QK) * sizeof(float));
    for (uint32_t i = 0; i < N_EMBD; ++i) x[i] = sinf((float)i * 0.01f) * 0.7f;

    if (g_check) {
        /* Both arms against a scalar reference. The pairq8 kernel is a
         * TRANSCRIPTION of a static function in ds4.c, so "it looks right" is
         * not evidence; and q8f32 is new. */
        quantize_q8_0(x, N_EMBD, xq, xd);
        const uint64_t blocks = N_EMBD / QK;
        double worst_pair = 0.0, worst_f32 = 0.0, ref_mag = 0.0;
        for (uint32_t r = 0; r < 64; ++r) {
            const uint8_t *row = W[0].gate + r * gu_rb;
            double ref = 0.0, refq = 0.0;
            for (uint64_t b = 0; b < blocks; ++b) {
                uint16_t h; memcpy(&h, row + b * BLK_BYTES, 2);
                const int8_t *q = (const int8_t *)(row + b * BLK_BYTES + 2);
                double a = 0.0, aq = 0.0;
                for (uint32_t j = 0; j < QK; ++j) {
                    a  += (double)q[j] * (double)x[b * QK + j];
                    aq += (double)q[j] * (double)xq[b * QK + j];
                }
                ref  += (double)f16d(h) * a;
                refq += (double)f16d(h) * (double)xd[b] * aq;
            }
            const float gf = dot_row_q8_f32(row, x, blocks);
            const float gp = dot_row_q8(row, xq, xd, blocks);
            worst_f32  = fmax(worst_f32,  fabs((double)gf - ref));
            worst_pair = fmax(worst_pair, fabs((double)gp - refq));
            ref_mag = fmax(ref_mag, fabs(ref));
        }
        printf("exactness (64 rows, |ref| up to %.3f):\n", ref_mag);
        printf("  q8f32  vs scalar f32 ref : max abs %.6f\n", worst_f32);
        printf("  pairq8 vs scalar int ref : max abs %.6f  (transcription faithful)\n\n",
               worst_pair);
        if (worst_f32 > 1e-2 || worst_pair > 1e-2) { puts("FAIL: kernel mismatch"); return 1; }
    }

    struct { const char *name; qos_class_t q; } qoss[2] = {
        { "ui", QOS_CLASS_USER_INTERACTIVE }, { "ud", QOS_CLASS_USER_INITIATED },
    };
    if (g_duration > 0.0) {
        /* One configuration, held for a wall-clock duration, at a duty cycle.
         * Production pacing is 42 x 149 us = 6.26 ms of CPU per 28.19 ms token,
         * about 22%; duty 1.0 is a deliberate upper-bound stress and is
         * labelled as such rather than quoted as the answer. */
        pool_start(threads[0], only_part && !strcmp(only_part, "dyn"),
                   tiles[0], (only_qos && !strcmp(only_qos, "ud"))
                             ? QOS_CLASS_USER_INITIATED : QOS_CLASS_USER_INTERACTIVE);
        stage_ctx c = {0};
        c.x = x; c.xq = xq; c.xd = xd; c.mid = mid; c.midx = mid;
        c.midq = midq; c.midd = midd; c.out = out; c.f32_path = 0;
        const double t_end = now_us() + g_duration * 1e6;
        double *samp = malloc(200000 * sizeof(double));
        uint32_t n = 0; uint64_t rots = 0;
        while (now_us() < t_end) {
            if (g_stopfile && access(g_stopfile, F_OK) == 0) break;
            for (uint32_t l = 0; l < g_layers; ++l) {
                c.L = &W[l];
                const double t0 = now_us();
                quantize_q8_0(x, N_EMBD, xq, xd);
                pool_run(stage_gate_up, &c, N_LANE);
                quantize_q8_0(mid, N_LANE, midq, midd);
                pool_run(stage_down, &c, N_EMBD);
                const double busy = now_us() - t0;
                if (n < 200000) samp[n++] = busy;
                /* Pace PER LAYER. Running all 42 flat out and then sleeping
                 * gave the right average and the wrong shape: production
                 * interleaves one layer of CPU with one layer of GPU, so a
                 * burst followed by an idle window is a different contention
                 * experiment from the one being claimed. */
                if (g_duty > 0.0 && g_duty < 1.0) {
                    const double idle = busy * (1.0 / g_duty - 1.0);
                    struct timespec ts = { (time_t)(idle / 1e6),
                                           (long)((idle - (long)(idle / 1e6) * 1e6) * 1e3) };
                    nanosleep(&ts, NULL);
                }
                if (g_stopfile && access(g_stopfile, F_OK) == 0) break;
            }
            rots++;
        }
        pool_stop();
        qsort(samp, n, sizeof(double), cmp_d);
        printf("load\tthreads\tduty\tidle\trotations\tlayers\tp50_us\tp95_us\trot42_med_us\n");
        printf("load\t%u\t%.2f\t%s\t%llu\t%u\t%.1f\t%.1f\t%.1f\n", threads[0], g_duty,
               g_idle_spin ? "spin" : "sleep",
               (unsigned long long)rots, n,
               n ? samp[n / 2] : 0.0, n ? samp[(n * 95) / 100] : 0.0,
               n ? samp[n / 2] * 42.0 : 0.0);
        free(samp);
        return 0;
    }

    printf("%-8s %-4s %-5s %-5s %-7s %9s %9s %9s %9s %9s\n",
           "arm", "qos", "part", "tile", "threads", "p50_us", "p95_us",
           "rot42_us", "rot_med", "GB/s");
    int green = 0, marginal = 0;
    int ncpu = 0, nperf = 0;
    size_t sz = sizeof(ncpu);
    sysctlbyname("hw.logicalcpu", &ncpu, &sz, NULL, 0);
    sz = sizeof(nperf);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &nperf, &sz, NULL, 0) != 0) nperf = ncpu;

    for (uint32_t ti = 0; ti < n_threads; ++ti) {
        /* A spin barrier waits for its slowest worker, so both of these are
         * cliffs rather than slopes. Oversubscribing measured 160x worse here
         * (12 threads on 10 cores: 116 us -> 18.8 ms) because spinning threads
         * get descheduled while spinning; and crossing into E-cores makes the
         * straggler roughly three times slower than the rest, which the
         * barrier then charges to everyone. */
        if (ncpu && (int)threads[ti] > ncpu) {
            printf("%-8s %-7u  SKIPPED: %u threads on %d logical cores -- a spin\n",
                   "--", threads[ti], threads[ti], ncpu);
            printf("%-8s %-7s  barrier collapses when oversubscribed, and the\n", "", "");
            printf("%-8s %-7s  number would be an artifact, not a measurement.\n", "", "");
            continue;
        }
        if (nperf && (int)threads[ti] > nperf) {
            printf("# %u threads exceeds %d performance cores: NEGATIVE CONTROL.\n",
                   threads[ti], nperf);
            printf("#   Equal partition should degrade (one E-core paces all);\n");
            printf("#   dynamic tiles are the arm that lets them contribute.\n");
        }
        for (int qi = 0; qi < 2; ++qi) {
        if (only_qos && strcmp(only_qos, qoss[qi].name)) continue;
        for (int dyn = 0; dyn < 2; ++dyn) {
        if (only_part && strcmp(only_part, dyn ? "dyn" : "eq")) continue;
        for (uint32_t tk = 0; tk < (dyn ? n_tiles : 1u); ++tk) {
        pool_start(threads[ti], dyn, tiles[tk], qoss[qi].q);
        for (int arm = 0; arm < 2; ++arm) {
            stage_ctx c = {0};
            c.x = x; c.xq = xq; c.xd = xd; c.mid = mid; c.midx = mid;
            c.midq = midq; c.midd = midd;
            c.out = out; c.f32_path = (arm == 1);
            double *samp = malloc(g_iters * sizeof(double));
            /* One cold pass is a single sample of the thing that actually
             * matters, and it is the noisiest. Collect EVERY full rotation and
             * report the median alongside, because a configuration is chosen
             * on 42-layer totals and tails, not on the best single layer. */
            const uint32_t n_rot = g_iters / g_layers;
            double *rots = calloc(n_rot ? n_rot : 1u, sizeof(double));
            double rot = 0.0, cur = 0.0;
            for (uint32_t it = 0; it < g_iters; ++it) {
                const uint32_t l = it % g_layers;     /* rotate: cold weights */
                c.L = &W[l];
                const double t0 = now_us();
                if (!c.f32_path) quantize_q8_0(x, N_EMBD, xq, xd);
                pool_run(stage_gate_up, &c, N_LANE);
                if (!c.f32_path) quantize_q8_0(mid, N_LANE, midq, midd);
                pool_run(stage_down, &c, N_EMBD);
                const double dt = now_us() - t0;
                samp[it] = dt;
                cur += dt;
                if ((it + 1u) % g_layers == 0u) {
                    const uint32_t ri = it / g_layers;
                    if (ri < n_rot) rots[ri] = cur;
                    cur = 0.0;
                }
                if (it >= g_iters - g_layers) rot += dt;   /* the last cold pass */
            }
            qsort(samp, g_iters, sizeof(double), cmp_d);
            const double p50 = samp[g_iters / 2], p95 = samp[(g_iters * 95) / 100];
            double rot_med = 0.0;
            if (n_rot) {
                qsort(rots, n_rot, sizeof(double), cmp_d);
                rot_med = rots[n_rot / 2];
            }
            printf("%-8s %-4s %-5s %-5u %-7u %9.1f %9.1f %9.1f %9.1f %9.1f\n",
                   arm ? "q8f32" : "pairq8", qoss[qi].name,
                   dyn ? "dyn" : "eq", dyn ? tiles[tk] : 0u, threads[ti],
                   p50, p95, rot, rot_med,
                   per_layer / (p50 * 1e-6) / 1e9);
            if (p50 <= 150.0) green = 1; else if (p50 < 200.0) marginal = 1;
            free(samp); free(rots);
        }
        pool_stop();
        }
        }
        }
    }

    printf("\ngate: <=150 us/layer green, 150-185 marginal, >=200 kills it\n");
    printf("%s\n", green ? "  some arm is inside the window HERE"
                         : (marginal ? "  marginal HERE" : "  over the window HERE"));
    puts("\nBUT NOT ON THIS BOX. The dev box decides exactness, never timing --\n"
         "this arm lives or dies on memory bandwidth, and an M1 Max has roughly\n"
         "half an M2 Ultra's. Read the exactness lines here and the microseconds\n"
         "from the rig.");
    return 0;
}
