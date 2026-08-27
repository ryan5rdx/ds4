# Inventory: Measurement Methodology and Instrument Caveats

STATUS: IN PROGRESS — pass 1 (test plan lines 1-1330, tp_decode_investigation full)

Scope: every measurement rule, instrument caveat and contamination flag found in
the nine sources, each with the incident that produced it and a file:line cite.

Abbreviations used throughout:
- **rig** = 2x Mac Studio M2 Ultra 60-core 128 GB, TP2 over Thunderbolt RDMA
  (hosts named `lanfear` = coordinator, `mat` = worker).
- **dev box** = M1 Max 64 GB, single node. Cannot load the model (145.26 GiB).
- **TPLAN** = `docs/TP-A0-ROWSPLIT-TEST-PLAN.md`
- **BENCH** = `BENCHMARKS-TP-PP.md`
- **DECINV** = `speed-bench/tp_decode_investigation.md`
- **MTP** = `speed-bench/tp_mtp_hunt.md`
- **PREFILL** = `docs/TP-PREFILL-LONG-CTX-INVESTIGATION.md`
- **SCOPE-HC / SCOPE-ARS / SCOPE-GATE / SCOPE-MA** = the four `docs/SCOPE-*.md`

---

## 1. Profiler / instrument taxes

(filled in pass 2)

## 2. Epoch arithmetic
## 3. Byte-model reconciliation
## 4. Interleaved best-of A/B
## 5. Throughput cannot detect wrong output
## 6. Ablation vs profile
## 7. Standalone-to-rig transfer factor
## 8. End-to-end share, not kernel multiples
## 9. Environment knobs
