# Inventory: Hardware and Configuration Reference

STATUS: DRAFT 1 (2026-08-27) — sources mined so far: `speed-bench/tp_decode_investigation.md`,
`BENCHMARKS-TP-PP.md`. Remaining: `docs/TP-A0-ROWSPLIT-TEST-PLAN.md`, `speed-bench/tp_mtp_hunt.md`,
`docs/TP-PREFILL-LONG-CTX-INVESTIGATION.md`, the four SCOPE docs.

## 1. Rig topology and hosts

| Host | Role | TB IP | iface | RDMA dev | Machine |
|---|---|---|---|---|---|
| `moiraine@lanfear.local` | coordinator / leader | 192.168.0.6 | `en6` | `rdma_en6` | M2 Ultra 60-core GPU, 128 GB |
| `moiraine@mat.local` | worker | 192.168.0.5 | `en7` | `rdma_en7` | M2 Ultra 60-core GPU, 128 GB |

Source: `BENCHMARKS-TP-PP.md:39-54`; `speed-bench/tp_decode_investigation.md:3-4`.

## 2. RDMA link setup, latency, bandwidth
## 3. GPU wired limit prerequisite
## 4. Measured roofs
## 5. Model shape constants
## 6. Per-rank memory footprint
## 7. Reboot / setup procedure
## 8. TP gate mechanics
## 9. Superseded figures and corrections
## 10. Contamination flags
