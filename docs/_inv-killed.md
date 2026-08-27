# Inventory: DO-NOT-RE-INVESTIGATE (killed avenues)

STATUS: IN PROGRESS — sources 3,4,5 extracted; SCOPE docs + test plan + benchmarks pending.

Scope: every avenue tried and abandoned across the nine TP/decode optimisation
sources. Grouped by theme. Provenance carried on every number.

## 0. Legend / conventions

- **Rig** = 2x Mac Studio M2 Ultra, 60-core GPU, 128 GB each, DeepSeek-V4-Flash
  MXFP4, TP2 over Thunderbolt RDMA. All quotable performance numbers come from
  here unless noted.
- **Dev box** = M1 Max, 64 GB. Build + `ds4_test --metal-kernels` only.
  *Never quote a timing number from it* (`tp_decode_investigation.md:191-195`:
  repeated runs of the same bench returned 0.2%-of-peak garbage rows).
- Kill classes:
  - **M** = killed by measurement (an A/B was run)
  - **A** = killed by argument / code reading only (never measured)
  - **S** = superseded (the claim was later shown wrong, so the kill itself is
    suspect — see §7)
  - **B** = blocked structurally (cannot be built as proposed)
- Noise floor on the rig is **~1%**; ~10% machine drift seen between distant
  windows (`tp_decode_investigation.md:531`). Anything under ~1% uninterleaved
  is not a result.

## 1. Dispatch & scheduling

TBD

## 2. Kernel tuning

TBD

## 3. Quantisation & format

TBD

## 4. Parallel decomposition (TP / PP / row-split / sharding)

TBD

## 5. Speculation (MTP / DSpark / draft)

TBD

## 6. Measurement & instrumentation

TBD

## 7. Killed, but the reasoning was later invalidated

TBD

## 8. Contamination register

TBD

## 9. Conditional kills (revisit if X changes)

TBD
