# Phase U Hip Exploration Preservation Design

## Decision

Change exactly one Phase U training hypothesis:

```text
policy_initial_action_std:
  [steer=0.05, drive=0.05, hip=0.50, knee=0.05]
→ [steer=0.05, drive=0.05, hip=0.25, knee=0.05]
```

Early airborne behavior remains neutral with respect to task progress: it is
not terminal, is not Phase U success, receives no jump/ascent/clearance/Apex
reward before the legal jump-window latch, and receives no new early-airborne
penalty.

## Evidence

The completed 192,000-transition channel-exploration run used hip standard
deviation 0.50. Across its 15 PPO blocks, stochastic training episodes had a
mean 97.1% physical-failure rate and a mean length of 17.34 control ticks. The
frozen 192,000-transition actor used the existing deployable root-distance
feature, but learned the wrong timing direction: deterministic hip action was
+0.413 at the natural start and -0.215 near the legal window.

The frozen normalizer placed legal-window distance values approximately
2.3--7.2 standard deviations below its running means. The problem is therefore
not that jump reward is active before the window, nor that the actor lacks a
distance signal. The current 0.50 hip exploration distribution terminates most
stochastic trajectories before they cover the legal timing region.

## Reward and Termination Invariants

This change does not modify reward code. The following existing behavior is
part of the design contract:

- `forward_propulsion` remains active before the window so the vehicle can
  learn to reach it;
- `jump_window_progress` is emitted only on the legal window-entry transition;
- `ascent_progress`, `clearance_progress`, `apex_approach`, and
  `apex_success_bonus` remain zero before legal window entry;
- early airborne alone is neither success nor failure;
- no vertical-velocity, airborne, or hip-action penalty is added before the
  window;
- roll, pitch, prohibited-contact, invalid-contact, backward-motion,
  platform-back-edge, and nonfinite safety failures remain unchanged.

Attitude, angular-rate, illegal-contact, action-smoothness, action-magnitude,
and physical-failure penalties remain physical/control regularizers. They are
not an early-airborne-specific penalty and are unchanged by this hypothesis.

## Scope

Change the stable Phase U smoke and formal configuration templates to the
ordered vector `[0.05, 0.05, 0.25, 0.05]`. Do not change:

- `dvgc/env.py`, environment observation fields, reset, or event latches;
- reward weights or reward implementation;
- PPO optimizer, network layers, horizon, or normalizer implementation;
- XML, payload, force limits, collision settings, or action mapping;
- interaction ceilings or checkpoint schedules;
- snapshot, continuation-label, feasibility, or Tube contracts.

The existing scalar-or-vector runtime support is already implemented and must
not be refactored for this change.

## Alternatives Considered

1. Add a pre-window early-airborne penalty. Rejected because the user requires
   early jumping to remain unpunished, and it would change reward semantics.
2. Add `jump_signal_latched` or a new observation feature. Rejected for this
   iteration because the frozen actor demonstrably uses its existing deployable
   distance feature.
3. Change observation normalization. Deferred because it is a broader
   preprocessing hypothesis. It becomes relevant only if reduced hip
   exploration still fails to cover the legal window.

## Red-Green Verification

Tests must first require both stable configuration files to resolve the exact
ordered vector `[0.05, 0.05, 0.25, 0.05]` and fail against the current 0.50
configuration. Existing pre-window reward tests must continue to prove that an
airborne, rising state before legal window entry receives zero window/ascent/
clearance/Apex progress while the independent physical/control penalties remain
unchanged.

After the configuration change, run targeted phase-expert and runtime tests,
static compilation, the full test suite, and `scripts/local_preflight.sh`.
Because this is a training configuration fingerprint change, a future PPO run
requires a fresh run-bound authorization. This design itself grants no
additional environment transitions and does not authorize smoke or formal PPO.

## Gate Outcome

Successful static implementation means only that the next single hypothesis is
encoded and validated. It does not create `pi_up_star`, snapshots,
continuation labels, `V_up`, a Soft Tube, Phase D training, or unified PPO.
