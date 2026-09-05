# JIT agent and maintenance rules

The root `AGENTS.md` is primary authority. This file narrows it for `JIT/`.

## Active scientific contract

The active experiment is conditioned on the fixed jump start at `x = 2.5 m`:

```text
pi_0 jump-start prefix + bounded perturbation + env.step
-> exact arrived state A
-> frozen pi_0/pi_1/pi_2 first-landing rollouts
-> family witness E
-> observed positive physical cells J
-> TRAIN-only replay support S
-> one unified Actor realization r
```

Never collapse `A`, `E`, `J`, `S` and `r`:

- family OR is an offline witness, not one-Actor full-chain execution;
- Actor observation space is not the physical capability metric space;
- raw Tube rows are not automatically causally reached states;
- first valid landing is the active endpoint; recovery is not required;
- no natural-reset connectivity claim is active;
- no formal reachability, viability or safety claim is allowed.

The centerline and proposer remain frozen `pi_0`. The evaluator family remains
exactly `{pi_0, pi_1, pi_2}` unless a new protocol explicitly changes it. Final
runtime expert switching is forbidden.

## Current stop boundary

The first family round produced Tube3 and trained `pi_3`. The stored π3 core
gate mixed `stable_recovery` baseline labels with `first_valid_landing`
candidate labels. Therefore π3 is trained historical evidence, not valid
prospective selection authority. Do not train π4 from the historical selection.

The expanded predictor-audit round has completed acquisition and locked
pre-outcome scores for TRAIN/CALIBRATION/ACCEPTANCE. Family labeling is
incomplete because large single-process evaluators exhausted GPU allocation.
The next executable task is evaluator-by-evaluator independent-process
sharding, followed by strict merge and fresh predictor audit. No final
TEST/JCE/JEL interaction is permitted.

## Centerline and acquisition

- use `jit_dvgc.analysis.nominal_jump_centerline`;
- use only real captured frames, with no qpos/qvel interpolation;
- use the locked π0 artifact; do not recompute it per iteration;
- use 0.1 m x slices from 2.5 m through landing or 4.2 m;
- reach candidates only by `env.step` from the locked jump-start state;
- proposal anchors identify targets and are never reset states;
- record that RSI/injection/reset anchors were not used for arrival.

Changing a seed does not by itself prove a distinct physical trajectory.

## Labeling and predictor rules

Every exact candidate receives real rollouts from all three frozen evaluators.
The family label is the OR of first-valid-landing results. Sharding may change
only execution lifetime; preserve catalog ordering, global candidate index,
seed, horizon, evaluator identity and endpoint.

The predictor:

- is fit on TRAIN and calibrated on CALIBRATION;
- must lock model, normalization, threshold, candidate order and scores before
  reading fresh outcomes;
- reports ROC-AUC, PR-AUC where defined, recall, FPR/accepted negatives,
  class/group counts and uncertainty when supporting a claim;
- cannot establish arrival, create labels, filter Tube admission or support a
  safety claim;
- requires a controlled same-budget ablation before any sample-efficiency use.

## Physical resolution

Primary `root_geometry_v1` cells use 0.10 m root position and 0.10 m/s root
velocity bins. `full_physical_v1` additionally uses 0.50 degree joint/orientation
and 2 degree/s angular/joint-rate bins, 0.10 m/s wheel tangential speed and
discrete phase. Raw rows, causal root cells, control cells and semantic corridor
cells are different quantities and must be reported separately.

## Training and evaluation

- Actor-only warm start imports Actor and observation normalizer only;
- critic and optimizer remain fresh unless explicitly predeclared;
- compare baseline and candidate with the same endpoint, panel, horizon and
  remaining-time convention;
- retain every regression and improvement, not only net coverage;
- training-support realization is not final forward-task generalization;
- ACCEPTANCE used for a decision remains development data;
- use independent training seeds, not checkpoints from one seed.

## Code placement

- durable logic: `JIT/src/jit_dvgc/`;
- thin CLIs: `JIT/cli/`;
- tests: `JIT/tests/`;
- configurations: `JIT/configs/`;
- reports and handoffs: `JIT/docs/`;
- run evidence: `JIT/runs/`, with lightweight summaries indexed in Git and
  large checkpoints kept out of normal commits.

Modify existing modules for existing behavior. New modules require a genuinely
new durable capability. Never encode one iteration's names, seeds or paths as
general logic.

## Fixed runtime and safety

- branch `agent/two-phase-soft-tube`;
- XML `assets/orange_bike_4kg_horizontal.xml`, 2 kg payload;
- 0.005 s simulation step, 0.020 s control interval;
- actions `[steer, rear-wheel drive, hip, knee]`, hip/knee +/-30 N m;
- use `/home/qy/mujoco_playground/.venv/bin/python`;
- preserve unrelated work; never reset/clean/stash/rebase/force-push;
- never alter historical artifacts to change their meaning;
- do not repeatedly recompute locked hashes; retain automatic provenance checks.

## Read order

1. root `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. root `PROJECT.md`
5. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `JIT/docs/CODE_ORGANIZATION.md`
8. `JIT/docs/JIT_SCIENTIFIC_REVIEW_RESPONSE_20260905.md`
