# Randomized Initial State and Fixed Pulse Evaluation

## Goal

Evaluate the five frozen policies under a randomized jump-start state followed by the existing three-step random action pulse at each of five fixed onset times.

## Protocol

- Preserve the world jump-start coordinate at `x=2.5 m`.
- Randomize only initial velocities, independently per episode and separately from the later pulse RNG:
  - forward velocity: uniform `[-0.1, 0.1] m/s` around the configured nominal forward velocity;
  - lateral velocity: uniform `[-0.1, 0.1] m/s`;
  - root angular velocities: uniform `[-0.05, 0.05] rad/s` for each axis;
  - steering, hip, and knee joint velocities: uniform `[-0.10, 0.10] rad/s`.
- Keep initial pose, joint positions, contact geometry, controller history, and root `x` deterministic.
- Apply the existing four-channel action residual independently as `U(-0.25,+0.25)` for three control ticks at onset `0`, `5`, `10`, `15`, or `25`; requested residuals must be zero outside that window.
- Run five policies × five onset conditions × 10,000 episodes, preserving every episode in the denominator.
- Record initialization noise and pulse requests as separate arrays and include both distributions in the report.

## Safety and comparability

The initial velocity ranges are explicit configuration fields. Reset validates finite values and the fixed `x=2.5 m` contract. No initial pose randomization is introduced in this experiment, so contact geometry and jump height remain interpretable. This is a new randomized-initial-state protocol; its results are not merged into the old fixed-reset results.

## Training boundary

This change first applies to the requested 250,000-episode evaluation. If an RL retraining stage is run, it must use the same declared reset randomization, a new run directory, and a separate training provenance record; it must not overwrite the frozen student or historical evaluation artifacts.
