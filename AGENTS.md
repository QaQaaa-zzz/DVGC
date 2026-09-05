# DVGC repository authority — empirical jumping envelope

Updated 2026-09-05 after the user's paper-outline decisions. This is the active research direction. The original audit baseline was `bfc22f2e32cb78cb269b0e522c3bdd7c6e7a8d42`. Correctness fixes and a first probe-bank path now exist; [implementation status](JIT/docs/JIT_PROBE_BANK_IMPLEMENTATION_20260905.md) distinguishes CPU verification from pending production gates.

## Research objective

JIT studies **bootstrapping and budget-controlled discovery of an empirical jumping capability envelope** for the fixed bicycle–pendulum robot and operating condition. Phase-specific up/down learning supplies initial reset support; a successful frozen unified policy supplies a seed trajectory; complementary frozen probes discover additional forward-arrived states with successful landing continuations.

The primary result is valid new physical support versus total environment interactions. One Actor's realization is a separate diagnostic/application result. **Do not require a new Actor to cover the whole cumulative Tube or replace all previous policies.** A regression by a later policy does not erase a valid historical witness.

## User-confirmed scientific contract

- Task begins at the declared complete ground jump-start state at `x = 2.5 m`, including pose, velocities, controller/event history and time semantics. Earlier natural-reset approach is outside scope.
- A state enters empirical support only with real forward dynamics from that start (or a fully verified ancestor chain) and a successful continuation from the **same exact state and required context**.
- Success is `first_valid_landing` before declared failure/horizon; recovery is not required. One observed success is a witness, not a calibrated success probability or safety guarantee.
- Multiple frozen policies may provide forward proposals and continuation evaluations. Membership and roles are versioned; one forward rollout uses its declared frozen proposer plus bounded perturbations. Different prefix/suffix policies are allowed as offline witnesses, not claimed as one-Actor execution.
- Keep the real-frame pi_0 centerline fixed as longitudinal coordinates. It is not an Actor command, reward target, tracking trajectory or interpolated reachable corridor.
- All tried policies failing means `no_success_witness_under_declared_bank`, not physical infeasibility. Untested and incomplete engineering attempts are separate states.
- Never infer formal reachability, viability, safe invariance, continuous feasible volume, the complete physical limit, or universal Actor impossibility.

## Objects that must stay separate

| Object | Meaning |
| --- | --- |
| `R_hat` / arrival evidence | Exact states reached with auditable prefix provenance |
| landing witness | A declared frozen evaluator succeeded from that exact state |
| `T_hat` / empirical support | Exact arrival states with at least one valid landing witness |
| physical cells | Declared projection of witnessed states; a cell does not certify every state inside |
| `S` / training Tube | Reset/replay support, including historical rows without current arrival/landing evidence |
| `Pi` / probe bank | Immutable policy records with explicit proposer/evaluator roles and version |
| Actor realization | A single policy's results on a declared common panel |

Deduplicate physical coverage separately from storing witnesses. A row already in `S` may still need its first arrival or continuation witness. Equal qpos/qvel does not establish equal FIFO, event state, controller context or remaining time.

## Fixed runtime

- Repository target branch: `agent/two-phase-soft-tube`; isolated review branches/worktrees may be used to preserve concurrent work.
- XML: `assets/orange_bike_4kg_horizontal.xml`.
- Recorded XML identity: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`.
- Payload: 2 kg; simulation step: 0.005 s; control interval: 0.020 s.
- Actions: `[steer, rear-wheel drive, hip, knee]`; hip/knee limits: +/-30 N m.
- Production Python: `/home/qy/mujoco_playground/.venv/bin/python`.

Do not change physics, reward, endpoint, reset semantics or action order silently. If the production runtime is unavailable, report the limit; source/fixture checks are not GPU rollout validation.

## Historical evidence and migration boundary

The locked 2026-09-04/05 scans used pi_0 as proposer and exactly pi_0/pi_1/pi_2 as evaluators. Finish their missing labels under those identities; do not retrofit a larger bank or new seed into their outputs.

Legacy family and selected-policy workflows retain their original contracts. New `JIT/cli/probe_bank.py` supports versioned proposer/evaluator banks, separate causal catalogs, bounded fresh-process suffix jobs, attempt accounting and an observation index. CPU fixture tests pass; GPU equivalence, cumulative physical-cell accounting, cross-version isolation, probe admission and complementary training are still pending.

The historical pi_3 mixed-endpoint gate remains invalid as a fair comparison. Keep the trained checkpoint and valid underlying outcomes. pi_3 may be assessed as a prospective probe under a new identity/endpoint contract without requiring old full-Tube retention, but never reuse its old selected manifest as automatic authority.

New formal probe training is not ready: correctness guards and the first bank path are implemented, but production serial/shard and prefix/suffix equivalence, a full cost/coverage ledger, and the training recipe/budget must close first. Existing frozen-probe pilots should precede additional large PPO runs. A predictor is optional and must not block a predictor-free discovery experiment once essential gates pass.

## Data roles and evidence integrity

- TRAIN may guide exploration, train probes and supply reset support.
- CALIBRATION calibrates optional predictors; ACCEPTANCE is development evidence when used for decisions.
- Final TEST/JCE/JEL remains unopened for this work. Historical bootstrap files have their own splits named `test`; do not present those already-used splits as untouched final tests.
- Re-audit role isolation across every proposer, bank version, ancestor and training Tube. Shared ancestors/proposal groups imply correlated samples.
- Lock task, complete start/context, catalog, proposer/evaluator identities, endpoint, seed, horizon, remaining-time rule, role, cell resolution and budget before execution.
- A hash proves content identity, not when it was frozen. Preserve pre-outcome/pre-training records and history.
- Cache reuse and merges must verify the full requested contract before publishing; engineering failures never become physical negative labels.
- Keep raw runs immutable. Add new derived views and machine-readable eligibility records; never relabel historical results to manufacture a pass.

## Paper evidence and cost

Report raw candidates, witnessed exact states, novel physical cells, failed attempts, individual-probe contributions and single-Actor realization separately. Distinguish new arrival discovery from a new suffix witness on an old arrival.

Count expert/bootstrap training, proposal prefixes, all evaluator rollouts, unsuccessful/excluded acquisitions, PPO, development evaluations and failed retries. Shared bootstrap can be reported separately but belongs in the end-to-end total. Use matched-budget comparisons and independent seeds/group-aware uncertainty.

The existing Tube0 is value-weighted training support, not an all-success capability set: 222 rows include 42 historical negative labels. The successful pi_0 identity used by current scans is the later Round1 artifact; do not equate it with the first completed unified PPO run.

## Repository work

The user explicitly authorized subsequent routine commits and pushes to `agent/two-phase-soft-tube` within this project scope; do not repeatedly request permission. Preserve unrelated changes; never reset/clean/stash/rebase/force-push. Keep durable logic in `JIT/src/jit_dvgc/`, thin CLIs in `JIT/cli/`, tests in `JIT/tests/`, and research guidance in `JIT/docs/`. Extend existing capabilities rather than adding iteration-specific duplicate modules. Retain provenance checks; do not repeatedly recalculate locked hashes without a concrete identity question.

## Read order

1. This `AGENTS.md` and [JIT/AGENTS.md](JIT/AGENTS.md).
2. [PROJECT.md](PROJECT.md) and [CURRENT_STATUS.md](JIT/docs/CURRENT_STATUS.md).
3. [Paper outline](JIT/docs/JIT_PAPER_OUTLINE.md).
4. [Code/evidence review](JIT/docs/JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md).
5. [Iteration protocol](JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md).
6. [Training roadmap](JIT/docs/JIT_TRAINING_ROADMAP.md).
7. [Handoff](JIT/docs/CODEX_HANDOFF_20260904.md) and [code organization](JIT/docs/CODE_ORGANIZATION.md).

Dated older reports and run manifests remain historical evidence, not overrides of this user-confirmed direction.
