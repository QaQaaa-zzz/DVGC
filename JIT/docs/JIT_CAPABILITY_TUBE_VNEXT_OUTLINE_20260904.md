# SUPERSEDED — Resolution-Aware Capability-Tube vNext Outline

Date: 2026-09-04

This file is retained only as method-history provenance for the introduction of physical-state resolutions and root/full capability-cell profiles.

It has been superseded twice:

```text
target-free physical-cell Tube
        ↓
trajectory-centered Jump Tube
        ↓
CURRENT: causal reachable Jump Capability Tube
```

Current authority:

```text
JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md
```

The physical-resolution contribution from this intermediate version remains active:

```text
position                     0.10 m
linear velocity              0.10 m/s
orientation                  0.50 deg
root angular velocity        2.0 deg/s
joint angle                  0.50 deg
joint angular velocity       2.0 deg/s
wheel tangential velocity    0.10 m/s
```

But the old scientific implication is retired. Neither target-free global Tube occupancy nor trajectory-centered RSI continuation support is sufficient to establish Jump Capability.

The current rule is:

```text
J_k = R_k^forward ∩ V_k^continuation
```

A state must first be physically reached from the natural ground reset using real `env.step` dynamics. Only then may RSI be used to evaluate continuation from that exact reached state.

Do not use this document for current workflow, frontier, paper, or capability claims.
