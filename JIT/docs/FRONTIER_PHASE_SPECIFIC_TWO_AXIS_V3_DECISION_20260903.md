# JIT frontier v3 phase-specific two-axis decision — 2026-09-03

## Status

Iteration-1 frontier identification has now produced two preserved failed rounds:

1. legacy v1 (`4/8/16/32` tick single-axis panel): downstream TRAIN candidate support was zero;
2. `local_horizon_v2` (`1/2/4/8` tick single-axis panel): downstream candidates were recovered, but all downstream continuation labels were positive.

The v2 TRAIN result is therefore not sufficient to fit `C_down^1`, and it does not authorize `Tube_2` or `pi_2`.

The next allowed mainline attempt is a new **phase-specific boundary-bracketing v3**. It must be predeclared before any v3 environment interaction or continuation outcome is observed.

## Evidence that motivates v3

The completed v2 TRAIN evidence reported:

```text
upstream:
  candidates = 821
  positive   = 785
  negative   = 36
  parent groups = 9

downstream:
  candidates = 111
  positive   = 111
  negative   = 0
  parent groups = 3
```

Therefore upstream already brackets both continuation classes under the v2 single-axis panel, while downstream does not.

The read-only v2 probe-support diagnostic is an explicit input to v3 plan construction. The v3 CLI refuses to create the plan unless the diagnostic confirms:

- upstream has at least 20 positives and 20 negatives;
- downstream has meaningful positive support;
- downstream has exactly zero negatives;
- the diagnostic policy identity matches the v2 plan.

The v3 plan records the diagnostic file SHA-256 and the observed phase counts.

## Historical precedent

A closely related Iteration-1 repair-acceptance problem previously followed this sequence:

```text
single-axis probe
  -> insufficient boundary support
stronger single-axis probe
  -> still insufficient
sparse two-axis probe
  -> 3720 candidates
  -> monolithic continuation labeling OOM
  -> 4 independent processes x 930 candidates
  -> strict merge by global candidate index
  -> fresh acceptance bank succeeded
```

That history does not prove that two-axis perturbations must solve the current downstream frontier, but it is directly relevant evidence that a strongly coupled reachable frontier may not be exposed by one-axis action perturbations alone.

The historical method is therefore reused as a **new predeclared acquisition family**, not as a post-hoc label/gate relaxation.

## v3 probe contract

### Upstream — unchanged from v2

```text
active action dimensions = 1
strengths = 0.025 / 0.05 / 0.10
durations = 1 / 2 / 4 / 8 ticks
actions = steer / rear_wheel_drive / hip / knee
signs = -1 / +1
```

With four action dimensions this gives:

```text
8 one-axis directions
x 3 strengths
x 4 durations
= 96 variants per upstream anchor
```

No upstream acquisition parameter is strengthened in v3 because v2 already produced both continuation classes there.

### Downstream — sparse two-axis v3

```text
active action dimensions = 2
strengths = 0.15 / 0.30 / 0.50
durations = 2 / 4 / 8 ticks
actions = steer / rear_wheel_drive / hip / knee
signs = -1 / +1 for each active dimension
```

With four action dimensions:

```text
C(4,2) = 6 action pairs
4 sign combinations per pair
=> 24 sparse two-axis directions

24 directions
x 3 strengths
x 3 durations
= 216 variants per downstream anchor
```

Candidate states remain real-dynamics reachable states only. No qpos/qvel dilation, interpolation, geometric boundary fabrication, expert switching, or alternate physics is introduced.

## Contracts that do NOT change

The v3 revision must preserve exactly:

- selected engineering `pi_1 = repair02` identity;
- Tube_1 identity and all 3,119 existing Tube_1 states;
- newest-shell-only frontier parent pool;
- the already assigned parent groups and their TRAIN/CALIBRATION/ACCEPTANCE roles;
- parent-group disjointness across logical roles;
- role-level acquisition seeds and labeling seeds;
- deterministic frozen-policy continuation evaluation;
- 400-tick continuation horizon and success definition;
- TRAIN requirement: each phase `>=20 positive`, `>=20 negative`, `>=3 parent groups`;
- CALIBRATION and ACCEPTANCE isolation;
- Tube_2 rule: retain every Tube_1 state exactly plus qualifying new TRAIN expansion;
- pi_2 retained/newest replay contract: 75% / 25% inside Tube RSI;
- final TEST/JCE/JEL isolation.

The Soft Tube remains empirical training/curriculum support and is not a certified safe set, viability kernel, reachability proof, or invariant set.

## Acquisition implementation

The historical single-panel collector is intentionally left unchanged.

For a v3 role, the iteration adapter performs:

```text
same predeclared role anchors
  -> upstream phase collector with the unchanged v2 panel
  -> downstream phase collector with the new two-axis panel
  -> preserve both low-level phase protocols
  -> merge their candidate catalogs into one logical acquisition catalog
  -> rewrite only logical catalog routing/protocol identity
  -> continuation labeling consumes the merged catalog
```

Snapshot files stay inside their phase acquisition directories. The merged rows point to those exact snapshots through nested `source_bank` paths; the physical snapshot state is not copied or modified.

The merged catalog records each candidate's original phase-acquisition protocol SHA-256 as well as the one merged logical acquisition protocol SHA-256.

## Pre-label structural gate remains active

Before any expensive continuation labeling, the existing unlabeled support preflight still runs.

For TRAIN, each phase must have at least:

```text
40 acquired candidates
3 parent groups
```

This is only a necessary-condition pruning rule: fewer than 40 candidates cannot possibly later satisfy 20 positive + 20 negative labels.

It does not inspect outcomes and does not weaken the final TRAIN support contract.

## Independent-process continuation labeling

The previous 3720-candidate repair bank demonstrated that one long GPU process can fail from cumulative JAX/Warp memory pressure even when the scientific protocol is valid.

v3 therefore predeclares an execution-only limit:

```text
maximum logical candidates per continuation-label process = 930
```

If a completed role acquisition exceeds 930 candidates, the workflow deliberately stops **before labeling** and writes `label_shard_plan.json`.

The operator then runs:

```bash
python JIT/cli/run_frontier_label_shards.py run-all \
  --plan <v3-frontier-plan> \
  --role-root <frontier-role-root> \
  --role <train|calibration|acceptance>
```

The supervisor itself does not initialize JAX. It starts each contiguous shard as a separate Python process, waits for that process to exit, then starts the next one. This releases CUDA/Warp/JAX process memory between shards.

Scientific identity is unchanged by sharding:

```text
candidate_key = fold_in(PRNGKey(labeling_seed), global_candidate_index)
action_key    = fold_in(candidate_key, tick)
```

The merge requires exactly one completed shard for every contiguous range and restores labels in global candidate-index order. Thus sharding changes process lifetime only, not candidate order, continuation policy, horizon, seed, label definition, or outcomes.

For reference:

```text
3720 candidates / 930 = 4 independent processes
```

## Fresh v3 workflow

Do not mutate or continue either failed v1/v2 workflow root as the scientific mainline.

Use a new root/tag, for example:

```bash
cd ~/DVGC

git pull --ff-only origin agent/two-phase-soft-tube

export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python

# If it does not already exist, regenerate the read-only v2 diagnostic with no new rollouts.
$PY JIT/cli/analyze_frontier_support.py \
  --role-root JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_localhorizon_v2/frontier_train \
  --output JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_localhorizon_v2/frontier_train/support_diagnostics.json

TAG=20260903_phase_specific_twoaxis_v3
ROOT=JIT/runs/iteration_auto/pi_1_to_pi_2_${TAG}
CFG=${ROOT}/workflow.json

$PY JIT/cli/prepare_iterative_envelope_workflow.py \
  --selected-policy JIT/runs/iteration_selection/pi_1_repair02_selected_20260903/selected_policy.json \
  --source-tube JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901 \
  --tag ${TAG} \
  --work-root ${ROOT} \
  --config-out ${CFG}

$PY JIT/cli/run_iterative_frontier_protocol.py \
  revise-plan-phase-specific-two-axis-v3 \
  --source-plan JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_localhorizon_v2/frontier_plan.json \
  --v2-support-diagnostic JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_localhorizon_v2/frontier_train/support_diagnostics.json \
  --output ${ROOT}/frontier_plan.json

$PY JIT/cli/run_iteration_workflow.py --config ${CFG}
$PY JIT/cli/run_iteration_workflow.py --config ${CFG} --execute
```

The pre-created v3 `frontier_plan.json` is the completion artifact for the workflow's prepare stage, so the workflow must verify and skip its default v1 plan creation rather than overwrite the v3 plan.

## If the workflow stops for sharding

For example, if TRAIN acquisition exceeds 930 candidates:

```bash
$PY JIT/cli/run_frontier_label_shards.py run-all \
  --plan ${ROOT}/frontier_plan.json \
  --role-root ${ROOT}/frontier_train \
  --role train

$PY JIT/cli/run_iteration_workflow.py --config ${CFG} --execute
```

The rerun reuses the completed acquisition and merged labels and continues the same logical role. Do not delete the phase acquisition directories or shard evidence.

## Stopping rule after v3

If v3 TRAIN still cannot satisfy the fixed two-class support requirements, stop again.

Do **not** automatically:

- increase strength again;
- move to three/four action dimensions;
- extend durations again;
- move parent groups between roles;
- sample arbitrary full-Tube parents;
- lower the 20/20/3 support requirement;
- relabel using experts;
- change the continuation horizon;
- alter physics/domain randomization;
- touch TEST/JCE/JEL.

A second failed frontier family after the v2 local-horizon and v3 two-axis revisions must trigger a separate parent-generation/frontier-saturation decision rather than another silent probe retune.
