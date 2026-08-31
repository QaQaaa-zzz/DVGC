# Downstream Strength-Extrapolation Implementation Plan

**Goal:** Add and execute one locked strength-extrapolation config variant that
uses the completed 3,045-label run as prior evidence without changing JIT
scientific semantics.

## Constraints

- Modify only `JIT/`; preserve all user-owned dirty paths and run evidence.
- Use the existing CLI/module; do not add a versioned production entrypoint.
- Keep policy, Tube, XML, physics, reward, actions, label horizon, readiness,
  terminal clipping, and claim boundary unchanged.
- Spend at most 3,600 acquisition and 48,000 labeling interactions.
- Do not train `C^0`, construct `Tube_1`, or launch `pi_1` unless downstream
  positive/negative readiness and the later group-disjoint gate both close.

## Tasks

- [x] Add RED config tests for the exact new search mode, strength grid,
  duration, prior counts, and rejection of near-miss variants.
- [x] Generalize the validator and protocol purposes while preserving the old
  locked mode exactly.
- [x] Add the exact config and prelaunch declaration using fresh seed bases and
  a new output directory.
- [x] Run static, focused CPU/GPU, full preflight, and enhanced audit gates
  (33 focused CPU, 4 focused GPU, 418 full CPU, and 14 full GPU passed).
- [ ] Explicitly stage/commit only validated JIT paths and re-audit the committed
  HEAD.
- [ ] Launch exactly one panel and delegate sparse monitoring/result analysis
  to Luna medium.
- [ ] If readiness closes, freeze Iteration-0 TRAIN evidence and design the
  group-disjoint expansion-validation gate; otherwise stop this acquisition
  family and report the remaining blocker.
