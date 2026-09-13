# Autonomous empirical-envelope campaign — 2026-09-09

Current action (2026-09-09): callback-fixed smoke completed pi_4 PPO (25,600 steps), freeze, 753 arrivals and all five evaluator panels. Final plotting failed with KeyError pi_4 because the old panel has only four policies. Use JIT/cli/analyze_envelope_campaign.py --source JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1 to recover reports with zero new simulation/PPO. It verifies source artifacts and labels, writes a separate sibling analysis directory, and preserves the original failure. Untested old-policy coverage is null, not zero. New panel union = 743 witnessed contexts / 734 root cells; historical novelty still needs this analysis. Do not retrain this smoke or run the old campaign command after source changes.


Production fix (2026-09-09): the first campaign smoke failed inside PPO because `JAX_PLATFORMS=cuda` hid the CPU device required by `jax.debug.callback`. Training children now use `cuda,cpu`: GPU stays the default, CPU is available for callbacks. No completed pi_4 or discovery round was reported. The 27,200 charged interactions are the failed attempt reservation, not a measured completed-transition count. Preserve the original directory and retry under `JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1`; code-identity locks intentionally reject in-place continuation after source changes. Both runs' costs must be included in total experiment accounting. See JIT/docs/JIT_AUTONOMOUS_ENVELOPE_20260909.md.


The user authorized automatic learning and discovery. Historical instructions to stop before PPO are superseded for this versioned entry point. No simulator training was run in the development environment.

## Latest production evidence

Read from `agent/jit-run-reports/reports/lower_boundary_v1-ad61c9d662`:

- 16 trajectories, 10 full forward landings, 260 reached states, all 260 witnessed by at least one frozen evaluator; no truncated trajectories.
- 20,735 charged interactions, zero PPO transitions.
- Lowest observed successful peak **root z = 0.6108724475 m**, on `dense_v1_train_x2.85_family2/variant_7`: positive knee offset 0.2 in the early window. This is a control-step-sampled root height, not wheel clearance, an exact continuous-time maximum, or a globally minimal trajectory.
- The six other trajectories terminated before landing; do not infer physical infeasibility from those outcomes. This run lacks a paired nominal-trajectory control, so it alone cannot quantify height reduction versus nominal.
- Per-evaluator suffix successes: pi_0 257, pi_1 260, pi_2 255, pi_3 258. Union root cells: 254. All four unique-root contributions are zero on this panel. These are local TRAIN observations, not a held-out ranking.

The compact root report is preserved in `JIT/review_evidence/lower_boundary_remote_summary_20260909.json`.

## Iteration method

1. Verify the completed knee-refinement and low-boundary TRAIN catalogs, full snapshot contexts, first-valid-landing labels and frozen identities. They seed this campaign. Existing historical evidence remains intact.
2. Construct a bounded training view from cumulative witnessed arrivals. Separate phases 50/50 inside the snapshot portion; round-robin over source trajectories, at most 512 samples per phase; trajectory-normalized weights, most recent source weighted twice. Keep unwitnessed arrivals in the source evidence, but never call them positive resets or physically impossible states.
3. Train pi_4 (then pi_5, etc.). Initializer is pi_2 for the first round and the preceding completed probe thereafter. Warm-start Actor and observation normalizer; reinitialize critic and optimizer. Normalizer continues updating in PPO. Reset mixture: 20% complete fixed near-ground x=2.5 m starts, 80% exact witnessed snapshots. Snapshot resets clear episode/phase clocks and return but preserve event/controller/history context, with the previously accepted numerical restoration limits.
4. Preserve original phase rewards, terminate at first valid landing instead of recovery. Use a separate versioned config; do not silently change old training definitions. First valid landing overrides timeout, but never overrides physical failure. The old `natural_reset_probability` field remains only as a compatibility field; the explicit new selection declares the fixed jump start.
5. Verify exact final training transitions and checkpoint restore, freeze parameters, then append the probe to the bank. No whole-Tube performance gate or sole-successor claim.
6. Let the new probe generate real trajectories from x=2.5 using clipped signed offsets on all four normalized action channels. No qpos/qvel injection, external lift, or height teleportation. Each round has 32 attempts with two spatial windows and two strengths; windows shift by 2.5 cm and strengths by 0.025 over a three-round cycle. Seeds differ per round. This is a declared finite exploration family, not exhaustive action optimization.
7. Sample real states at 5 cm x spacing; retain original physical cell resolution. All bank members evaluate the same new candidates to first valid landing, serially in separate GPU shard processes. Skip the previously failed device benchmarks. Add only fully verified witnesses to the cumulative empirical support.
8. Count novel successful root cells relative to **this campaign's seed panels plus prior rounds**, train again using the updated support, and repeat. Old rows are not re-evaluated by every later policy; per-round common-panel plots and historical union are different objects. Current seed scope is the two recent completed TRAIN panels, not the previously reported 3,463-cell all-history baseline. Do not add these two counts or label campaign novelty as global historical novelty.

A new probe that duplicates existing behavior can add zero cells. The method does not promise monotonic individual-policy ability, guaranteed learning progress, or a global physical limit. Stored witnessed union is monotonic.

## Execution and accounting

- Default: at most 3 rounds, 512,000 PPO transitions each, 128 environments, 3,200-transition blocks, total new-interaction budget 40,000,000. Four final TRAIN panel starts per training run add at most 1,600 interactions. No final TEST access.
- Stop after 2 consecutive completed rounds each adding fewer than 5 root cells, or at budget/round limits. Missing labels and engineering errors stop with `engineering_error`, never with stagnation or a negative capability claim.
- Every failed/interrupted training attempt is charged its full reserved PPO+panel ceiling unless completed evidence is present. Retries use new attempt directories and retain failures. Completed checkpoints are reused, not retrained. Discovery uses its existing per-attempt ledger. Full reservation is conservative when actual failed cost is unknown.
- Request limits, source code, support, input files, frozen policies and completed artifacts are immutable during resume. Repeat the exact command to resume after process failure. Changed code/limits require a new output directory. Running two supervisors on the same directory is locked out.
- Historical bootstrap/seed costs are not included in the new campaign cost axis. Paper total-cost accounting must add them separately.

## First production run

Use a short **one-cycle training/discovery run** first because this complete snapshot training adapter has not run on the production GPU. It trains a new Actor and explores with it automatically; it is not another numerical replay investigation.

```bash
cd /home/qy/DVGC
git switch agent/two-phase-soft-tube
git pull --ff-only origin agent/two-phase-soft-tube
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/run_envelope_campaign.py --gpu 0 \
  --output-dir JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1 \
  --max-rounds 1 --ppo-steps 25600 --budget 12000000
```

The first-cycle conservative ceiling is 25,600 PPO + 1,600 panel + 16,000 acquisition + 32×128×5×400 suffix = **8,235,200** interactions. Actual trajectories normally terminate far before the maximum. This small training budget checks the learning/exploration interface; it is not a sufficient budget to claim converged complementary learning. Existing seed raw files/checkpoints must remain on this server.

Once the first cycle runs successfully, the full three-round default is:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/run_envelope_campaign.py --gpu 0
```

For a longer bounded campaign use a new directory with `--max-rounds 20 --budget 40000000`; stagnation/budget may stop it sooner. This first version starts from pi_2 and the declared historical seed panels in each new campaign; it does not silently inherit a separate smoke campaign's pi_4. A future campaign-rebase protocol must lock any such inheritance explicitly.

Outputs: root `summary.json`, `progress.json`, `figures/coverage_cost.csv`, `figures/campaign_progress.png/pdf/svg`, per-round common-panel policy envelope figures, training logs/configs/checkpoints, support and attempt ledgers. Projection dots are observed arrivals, not a filled feasible region. Compact text review publishes automatically after each completed round and at exit to `agent/jit-run-reports`; raw checkpoints/snapshots stay local. Failure still emits a summary and a compact ZIP. If publication fails, local outputs remain and `publish_status.json` reports the reason.

## What remains for the paper

This implements a runnable bounded iteration mechanism; CPU mocks do not establish learning benefit. Required production evidence still includes: successful new-adapter training, at least one measured learning→discovery closure, matched-budget frozen-bank/no-new-PPO control, independent repetitions, resolution sensitivity, total-cost accounting including failures and historical bootstrap, and final held-out evaluation after protocol lock. Include bootstrap ablation if making an efficiency claim about the phased bootstrap. Stop wording must remain empirical stagnation under this model/search/budget, not certified physical-boundary convergence.

CPU verification: 58 unique targeted tests passed across the combined run and focused reruns after fixes. Includes two-round mocked orchestration, actual local-git report publication, JAX terminal/reset helpers, growing-bank figure export and canonical training preflight. No production GPU training or learning-benefit claim. See `JIT/review_evidence/autonomous_campaign_cpu_20260909.json`.

## Recover the completed callback-fixed smoke (no GPU work)

```bash
cd /home/qy/DVGC
git switch agent/two-phase-soft-tube
git pull --ff-only origin agent/two-phase-soft-tube
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/analyze_envelope_campaign.py \
  --source JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1
```

This writes to `JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1_analysis_v1` and auto-publishes compact evidence. It does not change locked source hashes, old status, checkpoints or labels. Missing/invalid labels still fail. Recovery only permits a completed discovery or a final-figures engineering failure; it does not promote arbitrary incomplete runs. New report status is `analysis_completed`; the original supervisor status remains recorded as `engineering_error`. The entry cannot launch PPO or rollout workers.

Outputs include per-policy/common-panel projections, old-versus-new shared TRAIN panel coverage, and campaign-seed-versus-new cumulative coverage. These two baseline scopes are explicitly different and neither is the full historical 3,463-cell union. pi_4 old-panel success/novelty/cumulative values are null because pi_4 was never evaluated there; its current-panel support is shown, not mislabeled entirely novel. Set-union novelty can still be computed between the old bank's witnessed support and the enlarged bank's new support.

The next experimental decision follows this recovered analysis: compare pi_2 versus pi_4 forward exploration with fixed perturbation/seeds and a common evaluator bank, account the new training cost separately, then decide on larger PPO/multiple rounds. Do not infer absence of proposer value from pi_4's zero unique suffix contribution on its own reached panel.
