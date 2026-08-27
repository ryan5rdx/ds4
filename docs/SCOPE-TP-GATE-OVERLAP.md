# SCOPE: Can the TP gate wait be overlapped with useful work?

Status: **IN PROGRESS** (incremental — sections filled as concluded)
Branch: `upstream-metal-wins`
Date: 2026-08-27

## 0. Question

The TP gate release fence is a 1-thread GPU spin loop (`kernel_dsv4_tp_fence_wait`,
`metal/dsv4_misc.metal:7333`) that blocks the command buffer for an RDMA round trip while
counting as GPU-busy. Measured at ~1.75–2.7 ms/token (7–11% of the 2k token) across 86 gates.
Can the GPU do useful work while waiting, or can the wait be shortened?

## 1. Wire vs peer-compute decomposition — reconciling M0's 24.8 µs against the ~31–38 µs gate

_TBD_

## 2. Is there independent work to overlap with?

_TBD_

## 3. Can the fence be replaced or restructured?

_TBD_

## 4. Peer lateness: cost and mitigations

_TBD_

## 5. Recommendation

_TBD_
