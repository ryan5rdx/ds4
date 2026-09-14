/*
 * Apple AMX — isolated raw-opcode wrapper and capability canaries.
 *
 * AMX is an UNDOCUMENTED, UNSUPPORTED CPU matrix unit driven synchronously by
 * CPU instructions. Everything here is default-off and every call site must
 * have a conventional fallback.
 *
 * The instruction encodings are from corsix/amx (MIT licensed):
 *     https://github.com/corsix/amx
 * Copyright (c) 2022 Peter Cawley. Retained per that licence. They are kept in
 * this one header deliberately: raw .word opcodes must not spread through model
 * code, because a stray AMX instruction on a thread that already issued
 * AMX_SET traps, and the blast radius should be one file.
 *
 * THREE RULES THAT ARE NOT STYLE PREFERENCES:
 *
 *   1. AMX_SET is non-reentrant. Issuing it twice on one thread traps. Every
 *      region is SET ... CLR with nothing between them that can yield: no
 *      allocation, no logging, no condition variable, no Accelerate call.
 *   2. The state is ~5 KiB of architectural registers the OS must preserve
 *      across context switches. Never hold it live while a worker sleeps.
 *   3. "The instruction did not trap" is NOT an M2 test. Some M2 encodings
 *      execute on M1 with different semantics rather than faulting, so
 *      capability detection has to be BEHAVIOURAL -- run the operation and
 *      check the value.
 */
#ifndef DS4_AMX_H
#define DS4_AMX_H

#include <stdbool.h>
#include <stdint.h>

#if defined(__aarch64__) && defined(__APPLE__)
#define DS4_AMX_BUILDABLE 1
#else
#define DS4_AMX_BUILDABLE 0
#endif

#if DS4_AMX_BUILDABLE

/* --- corsix/amx encodings (MIT, Peter Cawley) ---------------------------- */
#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" \
            : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
    __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" \
            : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define AMX_LDX(gpr)     AMX_OP_GPR( 0, gpr)
#define AMX_LDY(gpr)     AMX_OP_GPR( 1, gpr)
#define AMX_STX(gpr)     AMX_OP_GPR( 2, gpr)
#define AMX_STY(gpr)     AMX_OP_GPR( 3, gpr)
#define AMX_LDZ(gpr)     AMX_OP_GPR( 4, gpr)
#define AMX_STZ(gpr)     AMX_OP_GPR( 5, gpr)
#define AMX_LDZI(gpr)    AMX_OP_GPR( 6, gpr)
#define AMX_STZI(gpr)    AMX_OP_GPR( 7, gpr)
#define AMX_EXTRX(gpr)   AMX_OP_GPR( 8, gpr)
#define AMX_EXTRY(gpr)   AMX_OP_GPR( 9, gpr)
#define AMX_FMA64(gpr)   AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr)   AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr)   AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr)   AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr)   AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr)   AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr)   AMX_OP_GPR(16, gpr)
#define AMX_SET()        AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()        AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(gpr)  AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr)   AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr)  AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr)   AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr)  AMX_OP_GPR(22, gpr)

/* Operand helper: the pointer carries row and flag bits in its high byte. */
#define AMX_PTR_ROW_FLAGS(ptr, row, flags) \
    (((uint64_t)(ptr)) + (((uint64_t)((row) + (flags) * 64)) << 56))

#endif /* DS4_AMX_BUILDABLE */

/* What a given host actually supports, established behaviourally. Every field
 * is measured, never inferred from a CPU brand string. */
typedef struct {
    bool buildable;      /* the encodings exist in this build at all         */
    bool available;      /* raw AMX executes without trapping (child-tested) */
    bool load4;          /* M2: LDX/LDY fill four registers, 256 bytes       */
    bool matfp_f16;      /* F16 x F16 -> F32 matches a scalar reference      */
    bool matint_i8;      /* I8 x I8 -> I32 exact                             */
    bool genlut4;        /* 4-bit indexed load expands a 16-entry table      */
    bool bf16;           /* M2: BF16 multiplicands, distinguished from F16   */
    bool vecfp_multi;    /* M2: two/four-vector VECFP                        */
    bool extr_multi;     /* M2: two/four-vector EXTRH/EXTRV                  */
    char cpu[64];
} ds4_amx_caps;

/* Probes once per process and caches. Safe to call from anywhere: the
 * availability test runs in a FORKED CHILD, so a SIGILL on a host without AMX
 * kills the child rather than the server. Never aborts startup. */
const ds4_amx_caps *ds4_amx_probe(void);

/* True only when every primitive the named kernel needs passed its canary. A
 * partial M2 result selects a simpler arm or normal code. */
bool ds4_amx_can_run(const char *kernel);

/* One line, once per process. */
void ds4_amx_log_caps(void);

#endif /* DS4_AMX_H */
