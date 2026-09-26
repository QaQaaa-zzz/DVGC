# Initial Three-Step Pulse Evaluation Design

## Purpose

Repeat the comparison for four policies under the corrected disturbance timing, using 10,000 episodes per policy. The comparison measures recovery from an action disturbance applied immediately after reset, before the jump develops, rather than disturbances distributed through the takeoff and flight.

## Frozen comparison contract

- Policies: `lineage_repair_0010`, fresh RSI `transition_8000000`, Phase U `transition_4988928`, and `lineage_repair_0070` from `neighborhood_reward005_safe256_20260916/nonfinite_recovery/attempt_0002/lineage/round_0070`.
- Episodes: 10,000 per policy, paired random draws, 40,000 new disturbed episodes total.
- Initial state, horizon 400, reward, four-channel action limits ±0.25, three-step duration, and `stable_forward_recovery` remain unchanged.
- The disturbance starts at control tick 0 and is active only at ticks 0, 1, and 2. With `CTRL_DT=0.020`, this is the first 0.06 seconds after initialization.
- The jump policy controls every tick. During ticks 0–2 the existing random action residual is added to the policy action. From tick 3 onward the requested residual is exactly zero and the policy alone controls the episode.
- Preserve the current random sampling semantics: each of the three disturbance ticks receives the existing independently sampled four-channel residual. This isolates timing as the only protocol change.
- The old mixed-onset experiment remains immutable and is labeled incompatible with this corrected timing question.

## Implementation

Extend the generic batched-comparison preparation API and CLI with an optional pulse-start schedule override and an optional additional-method manifest. Validate the schedule as a nonempty list of nonnegative control-tick indices and write the override into every method template and the common frozen contract. Each additional method identifies a unique key, label, frozen bank and policy name; its task XML and action/observation contract must match the source comparison. The corrected run passes `[0]` and adds the specified `lineage_repair_0070`; existing callers without overrides retain their source methods and schedule.

Generate one matched zero-disturbance nominal rollout for each added method inside the bounded run. Reuse the locked nominal trajectories for the three existing methods. Make the readable report layout depend on the method count so four policies receive four equally sized trajectory panels.

Batch verification will use one array-level timing assertion per saved policy tape: the pulse mask may only cover ticks 0–2, all live lanes must receive those three pulse ticks, and requested disturbance must be zero afterward. This is an internal integrity gate, not an additional per-episode plot or report.

## Outputs and stopping

Create `runs/experiments/four_policy_initial_pulse_10k_20260918` with a fresh seed, 40 bounded batches of 256 environments (last batch 16), at most 16,000,400 control interactions including one new nominal rollout, and no training. Stop on input drift, timing mismatch, nonfinite trajectory, child failure, timeout, or resource-wait timeout. Start and verify the normal JIT desktop watcher. After completion, generate the same readable x≤5.5 m density, success-rate, failure-composition, and machine-readable artifacts as the previous comparison.
