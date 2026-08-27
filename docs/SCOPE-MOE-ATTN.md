# Scoping: `routed_moe_folded` and `attn_inv_rope` (34% of the decode token)

Status: **IN PROGRESS** (written incrementally; sections are filled as they are concluded).

Rig: 2 x Mac Studio M2 Ultra 60-core, DeepSeek V4 Flash MXFP4, TP world 2 over
Thunderbolt RDMA. Branch `upstream-metal-wins`. All stage times quoted **net of
the ~0.18 ms/marker profiler tax**.

| stage | net ms @2k | % of 2k token |
|---|---:|---:|
| `routed_moe_folded` | 4.72 | 19.4% |
| `attn_inv_rope` | 3.63 | 14.9% |

---

## 0. Method and ground rules

(to fill)

## 1. What is actually in each span

### 1.1 `routed_moe_folded`
(to fill)

### 1.2 `attn_inv_rope`
(to fill)

## 2. Bytes and FLOPs per token per rank

### 2.1 `routed_moe_folded`
(to fill)

### 2.2 `attn_inv_rope`
(to fill)

## 3. Achieved rate against both roofs

(to fill)

## 4. Ranked candidates

(to fill)

## 5. What was checked and rejected

(to fill)
