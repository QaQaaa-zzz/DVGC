# SUPERSEDED — Trajectory-Centered Jump-Tube Report

Date: 2026-09-04

This report represented an intermediate method revision: trajectory-centered, x-indexed, resolution-aware Tube identification.

It is **not the current JIT authority** because it still allowed a methodological gap between local real-dynamics perturbation and natural-start forward reachability. A state reached after restoring an RSI/Tube anchor is not necessarily reachable from the real ground start.

Current authority:

```text
JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md
```

Active correction:

```text
RSI continuation success != forward reachability

J_k = R_k^forward ∩ V_k^continuation
```

The following ideas from this intermediate version remain in the current method:

- one successful real jump provides a fixed centerline scaffold;
- nominal x support starts at 2.5 m, uses 0.1 m spacing, and stops at first valid landing or 4.2 m;
- downstream capability support must still be descending;
- late recovery is excluded from Jump-Capability frontier accounting;
- Actor observation is not the physical capability metric space;
- per-variable physical resolution is used for cell identity;
- Tube geometry is analyzed as `x -> cross-section` rather than raw snapshot count;
- no goal/intent variable is added to the current Actor.

The decisive addition in the causal version is that every new capability candidate must first be generated from the natural ground reset through real `env.step` dynamics, with auditable ground-reachability provenance. RSI is used only after that point for continuation evaluation and later training.

Historical details of this intermediate report remain available in Git history. Do not cite this file as the current method definition and do not place it in the active authority read order.
