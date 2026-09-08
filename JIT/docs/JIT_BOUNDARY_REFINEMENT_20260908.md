# Local TRAIN boundary refinement

The completed landing frontier found 12 no-witness states on one pi_1 trajectory
with a positive knee offset of 0.2. This run refines that specific gap using
existing Actors, not PPO. It does not restrict the overall JIT method to knee.

## Server command

```bash
cd /home/qy/DVGC &&
git switch agent/two-phase-soft-tube &&
git pull --ff-only origin agent/two-phase-soft-tube &&
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/refine_jump_boundary.py --gpu 0
```

Requires the existing `JIT/runs/discovery/landing_frontier_v1` and
`JIT/runs/policy_comparison/expanded_pi0_pi3_20260907`, including their referenced
checkpoint and source files. `--previous` and `--baseline` override those paths;
they do not rewrite absolute paths embedded in historical manifests.
Default output: `JIT/runs/discovery/knee_boundary_v1`.

## Locked experiment

- Proposer pi_1; evaluators pi_0, pi_1, pi_2, pi_3, same first-valid-landing endpoint.
- Nine trajectories: offsets 0.15, 0.175, 0.20 crossed with action-window endpoints
  x=2.85, 2.90, 2.95 m. Each window begins 0.15 m earlier.
- Add the offset to the normalized knee action, then clip actions to [-1,1].
  This is not a torque value or a percentage gain. Other action channels still
  follow the Actor. No position/velocity injection, force impulse or lift.
- Fixed complete near-ground start x=2.5 m; retain the original centerline.
  Capture real frames at 5 cm slices through landing/termination or x=8 m guard.
- At most 128 candidates per trajectory, 400 steps per forward/suffix rollout.
  First-attempt ceiling 1,847,700 interactions; default total budget 2,000,000,
  including retries and conservative charges for missing attempt costs.
- Frozen acquisition seed 9843101 and label seed 9843201. Different windows are
  local proposal groups, not independent experimental repetitions. The repeated
  nominal 0.20/2.90 condition is a new declared attempt, not exact replay proof.
- Serial execution, fresh processes per bounded shard, no device benchmarks.
  Maximum candidate count 1,152; actual landing/failure usually stops earlier.
  Nine versus 48 acquisition trajectories reduces scope; no runtime guarantee.

The prior report self-hash, TRAIN role and gap composition are checked before
execution. Its report and boundary table file hashes enter the new locked plan.
Changing inputs/recipe on resume refuses; unchanged completed tasks are reused.

## Read the result

Send the printed compact `results_to_send.zip`, including engineering failures.
Root summary has counts and costs; `figures/trajectory_comparison.csv` has each
window/strength, termination reason, and successful/disagreement/no-witness
counts. `boundary_refinement.png` compares real x-z support by strength, with
windows pooled within each panel. Per-window comparisons are in the table.

`boundary_candidates.csv` retains physical coordinates and four outcomes.
`training_review_candidates.csv` lists disagreements and missing witnesses for
review only: all rows have training_admitted=false. Nearby all-success states
remain available in the full table. Missing/failed engineering labels never
become negative outcomes. Empty arrivals remain an explicit completed-empty
child outcome; inspect receipts rather than inferring physical impossibility.

Afterward, decide whether the gap persists across nearby windows/strengths.
If it does, define a small complementary training support including nearby
witnessed states; lock initializer, reset mixture, budget, seed and stopping rule
before launching PPO. If it disappears, retain the evidence and choose another
observed gap instead of training to one correlated failure trajectory.
No automatic training admission, new replay gate or final TEST access occurs.
