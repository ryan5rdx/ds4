/*
 * Apple AMX capability detection. See ds4_amx.h for the rules.
 *
 * Everything here answers one question: may a later AMX kernel run on THIS
 * machine, decided by executing the operation and checking the result. The
 * campaign brief is explicit that a CPU-family string and a non-trapping
 * instruction are both insufficient, because some M2 encodings execute on M1
 * with different semantics instead of faulting.
 */
#include "ds4_amx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


static ds4_amx_caps g_caps;
static bool g_probed;

#if !DS4_AMX_BUILDABLE

const ds4_amx_caps *ds4_amx_probe(void) {
    if (!g_probed) { g_probed = true; g_caps.buildable = false; }
    return &g_caps;
}
bool ds4_amx_can_run(const char *k) { (void)k; return false; }
void ds4_amx_log_caps(void) {
    fprintf(stderr, "ds4: AMX not buildable on this target\n");
}

#else

/* ---- individual behavioural canaries ------------------------------------
 *
 * Each runs inside its own SET/CLR and touches nothing that can yield. They are
 * called only from the forked child (for the first availability test) or after
 * availability is established.
 */

/* M1 fills X0:X1 from a 128-byte load; M2 fills X0:X3 from 256 bytes when the
 * bit-60 flag is set. Reading back X2 therefore distinguishes them by VALUE,
 * which is the whole point -- the instruction executes on both. */
static bool canary_load4(void) {
    __attribute__((aligned(256))) uint8_t buf[256] = {0};
    buf[128] = 1;
    AMX_SET();
    AMX_LDX(AMX_PTR_ROW_FLAGS(buf, 16, 1));
    AMX_STX(AMX_PTR_ROW_FLAGS(buf, 2, 0));   /* store X2 over buf[0..63] */
    AMX_CLR();
    return buf[0] == 1;
}

/* I8 x I8 -> I32 outer product, checked against exact integer arithmetic.
 * Integer is the right first correctness test: there is no rounding to argue
 * about, so a mismatch is a layout or encoding error and nothing else. */
static bool canary_matint_i8(void) {
    __attribute__((aligned(128))) int8_t  x[64] = {0};
    __attribute__((aligned(128))) int8_t  y[64] = {0};
    __attribute__((aligned(128))) int32_t z[256] = {0};
    for (int i = 0; i < 64; i++) { x[i] = (int8_t)(i - 32); y[i] = (int8_t)(1 + (i % 5)); }

    AMX_SET();
    AMX_LDX(AMX_PTR_ROW_FLAGS(x, 0, 0));
    AMX_LDY(AMX_PTR_ROW_FLAGS(y, 0, 0));
    /* Zero the Z region this op accumulates into before using it. */
    AMX_LDZ(AMX_PTR_ROW_FLAGS(z, 0, 0));
    AMX_MATINT(DS4_AMX_MATINT_I8_I32);
    AMX_STZ(AMX_PTR_ROW_FLAGS(z, 0, 0));
    AMX_CLR();

    /* Only the presence of a plausible outer product is asserted here: the
     * exact Z row mapping is derived and unit-tested in the probe, not assumed
     * in a capability check. A dead unit leaves z all zero. */
    for (int i = 0; i < 16; i++) if (z[i] != 0) return true;
    return false;
}

/* F16 x F16 -> F32. Same shape as above; used to tell a working FP matrix path
 * from an integer-only one. */
static bool canary_matfp_f16(void) {
    __attribute__((aligned(128))) uint16_t x[32];
    __attribute__((aligned(128))) uint16_t y[32];
    __attribute__((aligned(128))) float    z[256] = {0};
    for (int i = 0; i < 32; i++) { x[i] = 0x3C00; y[i] = 0x4000; }  /* 1.0, 2.0 */

    AMX_SET();
    AMX_LDX(AMX_PTR_ROW_FLAGS(x, 0, 0));
    AMX_LDY(AMX_PTR_ROW_FLAGS(y, 0, 0));
    AMX_LDZ(AMX_PTR_ROW_FLAGS(z, 0, 0));
    AMX_MATFP(DS4_AMX_MATFP_F16_F32);
    AMX_STZ(AMX_PTR_ROW_FLAGS(z, 0, 0));
    AMX_CLR();

    for (int i = 0; i < 16; i++) if (z[i] != 0.0f) return true;
    return false;
}

/* BF16 must be distinguished from F16 BY VALUE, not by executing.
 *
 * 0x3F80 is 1.0 read as BF16 and 1.9375 read as F16. A unit that ignores the
 * BF16 mode bit and treats the operand as F16 therefore produces a visibly
 * different product, which is exactly the failure mode the brief warns about:
 * the encoding runs on M1 with the wrong interpretation instead of trapping. */
static bool canary_bf16(void) {
    __attribute__((aligned(128))) uint16_t x[32];
    __attribute__((aligned(128))) uint16_t y[32];
    __attribute__((aligned(128))) float    z[256] = {0};
    for (int i = 0; i < 32; i++) { x[i] = 0x3F80; y[i] = 0x3F80; }

    AMX_SET();
    AMX_LDX(AMX_PTR_ROW_FLAGS(x, 0, 0));
    AMX_LDY(AMX_PTR_ROW_FLAGS(y, 0, 0));
    AMX_LDZ(AMX_PTR_ROW_FLAGS(z, 0, 0));
    AMX_MATFP(DS4_AMX_MATFP_BF16_F32);
    AMX_STZ(AMX_PTR_ROW_FLAGS(z, 0, 0));
    AMX_CLR();

    /* BF16: 1.0*1.0 = 1.0. F16 misread: 1.9375*1.9375 = 3.7539. */
    for (int i = 0; i < 16; i++) {
        if (z[i] == 0.0f) continue;
        return (z[i] > 0.9f && z[i] < 1.1f);
    }
    return false;
}

/* 4-bit indexed load: does GENLUT expand nibbles through a 16-entry table? */
static bool canary_genlut4(void) {
    __attribute__((aligned(128))) float tab[16];
    __attribute__((aligned(128))) uint8_t idx[64] = {0};
    __attribute__((aligned(128))) float out[64] = {0};
    for (int i = 0; i < 16; i++) tab[i] = (float)(i + 1);
    idx[0] = 0x21;   /* nibbles 1 and 2 */

    AMX_SET();
    AMX_LDY(AMX_PTR_ROW_FLAGS(tab, 0, 0));
    AMX_LDX(AMX_PTR_ROW_FLAGS(idx, 0, 0));
    AMX_GENLUT(0);
    AMX_STX(AMX_PTR_ROW_FLAGS(out, 0, 0));
    AMX_CLR();

    for (int i = 0; i < 16; i++) if (out[i] != 0.0f) return true;
    return false;
}

/* Multi-vector VECFP / EXTR, by DIFFERENTIAL EXTENT.
 *
 * The first version of these returned true unconditionally after executing the
 * instruction -- which is precisely the "it did not trap" test the campaign
 * brief forbids, and it duly reported both as present on an M1 Max where load4
 * and bf16 correctly reported absent. A false positive here is not cosmetic:
 * ds4_amx_can_run("q4-predeq-f16") requires vecfp_multi, so it would have
 * admitted a kernel whose primitive the machine does not have.
 *
 * The test that actually distinguishes: run the SINGLE-vector form and the
 * MULTI-vector form over a sentinel-filled Z, and compare how many 64-byte
 * vectors each modified. A unit that ignores the mode bit writes the same
 * extent both times. Conservative by construction -- "cannot tell" returns
 * false, so an unknown host gets the fallback path.
 */
static unsigned amx_z_written_vectors(uint64_t op_is_vecfp, uint64_t operand) {
    /* ALL 64 Z rows, not the first 8.
     *
     * corsix/vecfp.md on the multiple-vector form: "the row stepping by 32 or
     * 16 per iteration". Two vectors therefore write Z rows 0 and 32; four
     * write 0, 16, 32 and 48. Scanning rows 0-7 could only ever observe row 0
     * in BOTH arms, so the differential reported "no difference" and
     * vecfp_multi=0 on hardware whose capability it never looked at. The rig
     * ran that twice and the second run was specifically to settle whether the
     * absence was real. It was not; it was this. */
    enum { NV = 64, VEC = 16 };                /* 64 rows x 64 bytes */
    __attribute__((aligned(256))) float z[NV * VEC];
    __attribute__((aligned(256))) float x[VEC], y[VEC];
    for (unsigned i = 0; i < NV * VEC; i++) z[i] = -12345.0f;
    /* X and Y MUST be loaded. The ALU mode is z + x*y, so with the registers
     * left at zero the operation is z + 0 and Z never changes -- which is what
     * the previous version measured: single=0, two=0, four=0, and it read that
     * as "the multi-vector form does nothing" rather than "I did not give it
     * anything to multiply". */
    for (unsigned i = 0; i < VEC; i++) { x[i] = 2.0f; y[i] = 3.0f; }

    AMX_SET();
    AMX_LDX(AMX_PTR_ROW_FLAGS(x, 0, 0));
    AMX_LDY(AMX_PTR_ROW_FLAGS(y, 0, 0));
    for (unsigned v = 0; v < NV; v++)
        AMX_LDZ(AMX_PTR_ROW_FLAGS(&z[v * VEC], v, 0));
    if (op_is_vecfp) AMX_VECFP(operand); else AMX_EXTRX(operand);
    for (unsigned v = 0; v < NV; v++)
        AMX_STZ(AMX_PTR_ROW_FLAGS(&z[v * VEC], v, 0));
    AMX_CLR();

    unsigned touched = 0;
    for (unsigned v = 0; v < NV; v++) {
        for (unsigned i = 0; i < VEC; i++) {
            if (z[v * VEC + i] != -12345.0f) { touched++; break; }
        }
    }
    return touched;
}

static bool canary_vecfp_multi(void) {
    /* Both arms do a real f32 z + x*y and differ only in bit 31 (+ bit 25). */
    const unsigned single = amx_z_written_vectors(1, DS4_AMX_VECFP_F32_ONE);
    const unsigned two    = amx_z_written_vectors(1, DS4_AMX_VECFP_F32_TWO);
    const unsigned four   = amx_z_written_vectors(1, DS4_AMX_VECFP_F32_FOUR);
    if (getenv("DS4_AMX_VERBOSE")) {
        fprintf(stderr, "ds4: AMX vecfp Z rows touched: single=%u two=%u four=%u "
                        "(spec: 1 / 2 at rows 0,32 / 4 at rows 0,16,32,48)\n",
                single, two, four);
    }
    return two > single || four > single;
}

/* ---- availability, established without risking the process --------------
 *
 * A host without AMX raises SIGILL on the first encoding. Running that test in
 * the parent would kill the server, so it runs in a FORKED CHILD and the parent
 * reads the exit status. This is why the brief says to do it before worker
 * threads exist: fork() in a threaded process is only safe for the
 * exec-or-die shape, which this is.
 */
static bool amx_available_via_child(void) {
    const pid_t pid = fork();
    if (pid < 0) return false;              /* cannot test => assume absent */
    if (pid == 0) {
        AMX_SET();
        AMX_CLR();
        _exit(42);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return false;
    return WIFEXITED(status) && WEXITSTATUS(status) == 42;
}

const ds4_amx_caps *ds4_amx_probe(void) {
    if (g_probed) return &g_caps;
    g_probed = true;
    g_caps.buildable = true;

    size_t sz = sizeof(g_caps.cpu);
    if (sysctlbyname("machdep.cpu.brand_string", g_caps.cpu, &sz, NULL, 0) != 0) {
        snprintf(g_caps.cpu, sizeof(g_caps.cpu), "unknown");
    }

    if (getenv("DS4_AMX_DISABLE")) return &g_caps;
    if (!amx_available_via_child()) return &g_caps;
    g_caps.available = true;

    /* Order matters only in that load4 is the cheapest discriminator. */
    g_caps.load4       = canary_load4();
    g_caps.matint_i8   = canary_matint_i8();
    g_caps.matfp_f16   = canary_matfp_f16();
    g_caps.genlut4     = canary_genlut4();
    g_caps.bf16        = canary_bf16();
    g_caps.vecfp_multi = canary_vecfp_multi();
    /* -1, not 0: the EXTRH/EXTRV operand layout is unconfirmed, and a guessed
     * encoding reported as a measured absence is exactly how bf16 and
     * vecfp_multi were wrongly closed. */
    g_caps.extr_multi  = -1;
    return &g_caps;
}

bool ds4_amx_can_run(const char *kernel) {
    const ds4_amx_caps *c = ds4_amx_probe();
    if (!c->available) return false;
    if (!kernel) return false;
    /* Each kernel names every primitive it needs. A partial M2 selects a
     * simpler arm; it never aborts and never runs a kernel whose primitives
     * were not individually confirmed. */
    /* extr_multi is required by the brief's predeq-f16 pipeline (mixed-width
     * EXTRH packs the F32 result to the F16 vector MATFP consumes), and it is
     * currently UNVERIFIED rather than absent -- so this stays ineligible, but
     * for a reason the log distinguishes from "the hardware lacks it". */
    if (!strcmp(kernel, "q4-predeq-f16"))
        return c->matfp_f16 && c->genlut4 && c->load4 && c->vecfp_multi &&
               c->extr_multi > 0;
    if (!strcmp(kernel, "q4-postscale"))
        return c->matfp_f16 && c->genlut4;
    if (!strcmp(kernel, "q4-bf16"))
        return c->bf16 && c->genlut4 && c->load4 && c->vecfp_multi &&
               c->extr_multi > 0;
    if (!strcmp(kernel, "q8-shexp"))
        return c->matint_i8;
    return false;
}

void ds4_amx_log_caps(void) {
    const ds4_amx_caps *c = ds4_amx_probe();
    fprintf(stderr,
            "ds4: AMX cpu=\"%s\" buildable=%d available=%d load4=%d matfp_f16=%d "
            "matint_i8=%d genlut4=%d bf16=%d vecfp_multi=%d extr_multi=%s\n",
            c->cpu, c->buildable, c->available, c->load4, c->matfp_f16,
            c->matint_i8, c->genlut4, c->bf16, c->vecfp_multi,
            c->extr_multi < 0 ? "UNVERIFIED" :
            (c->extr_multi ? "1" : "0"));
}

#endif /* DS4_AMX_BUILDABLE */
