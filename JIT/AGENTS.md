# JIT agent and maintenance rules

## Scope and safety

- `JIT/` is the active implementation area. Treat the repository-root `dvgc/`,
  `cli/`, `scripts/`, and `tests/` as read-only unless the user explicitly
  changes scope.
- Work only on `agent/two-phase-soft-tube` unless explicitly told otherwise.
- Never reset, clean, stash, overwrite, or reformat unrelated user work.
- Use only `/home/qy/mujoco_playground/.venv/bin/python`; do not reinstall or
  reconfigure the environment.
- The task XML is `assets/orange_bike_4kg_horizontal.xml` with SHA-256
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`.
  Preserve task physics, reward semantics, action semantics, snapshot semantics,
  50 Hz control, 2 kg payload, hip/knee +/-50 Nm limits, and TEST isolation.

## Scientific contract

The project is an iterative single-policy capability-envelope method, not a
runtime switch between two experts.

`experts -> Tube_0 -> pi_0 -> C^0 -> Tube_1 -> pi_1 -> C^1 -> Tube_2 -> pi_2 -> ...`

- `V_up/V_down` are bootstrap expert-conditioned Tube_0 authorities only.
- `C_up^k/C_down^k` are policy-conditioned empirical continuation fields bound
  to the exact frozen `pi_k` identity.
- PPO critic/value is not a JIT continuation field.
- Every Tube is training guidance/curriculum support, not a certified safe set,
  viability kernel, or formal invariant set.
- `Tube_(k+1)` must retain every existing Tube_k entry and add only qualifying
  logical-TRAIN expansion states; expansion is not a replacement level set.
- A larger Tube does not prove capability gain. After training `pi_(k+1)`, both
  core-preservation and boundary-gain gates must pass before claiming empirical
  jumping-capability-envelope expansion for that round.
- Calibration/validation is independent evidence and its rows never enter TRAIN
  or a Tube. Acceptance rows also never enter a Tube.
- TEST/final JCE/JEL evidence stays untouched until a final frozen policy has
  been selected. It may not affect Tube construction, threshold selection,
  policy training, checkpoint selection, or iteration stopping.

## Current authority — 2026-09-03

The Iteration-1 A/B initialization and checkpoint study is **closed**.

Authoritative state:

- Tube_1 is fixed at 3,119 TRAIN entries = exact 222-entry Tube_0 core + 2,897
  expansion; manifest SHA-256:
  `817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`.
- `repair02` used 75% retained core / 25% expansion inside Tube sampling with the
  outer 90% Tube / 10% natural reset mixture unchanged.
- repair02 frozen policy:
  `JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`.
- repair02 actor SHA-256:
  `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`.
- repair02 payload SHA-256:
  `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`.
- repair02 engineering quickcheck: Tube_0 `222/222`, upstream `117/117`,
  downstream `105/105`, zero core regressions, boundary `26/260` across 4 parent
  groups.
- Historical repair02 boundary gate has 3 baseline-reproduction failures caused
  by the old continuation-label vs paired-gate PRNG hierarchy mismatch.
- Warm-start A is discarded.
- Warm-start B checkpoint sweep is complete. No B checkpoint achieved both
  `Tube_0 = 222/222` and boundary success `>26/260`.
- Best B tradeoff was 7.5008M: `217/222 + 42/260`; B final was
  `212/222 + 46/260`.
- Every B core regression was upstream; downstream remained `105/105` at every
  checkpoint. Treat this as upstream expansion/retention interference, not a
  general descent/recovery failure and not simple monotonic overtraining.
- final TEST/JCE/JEL remains untouched.

### Selected pi_1 authority

The user has explicitly selected **repair02 as the engineering pi_1 authority**
for continuation of the main JIT loop.

This is intentionally a two-level claim:

1. **Engineering authority:** repair02 may generate pi_1-conditioned frontier
   evidence, `C_up^1/C_down^1`, Tube_2, and pi_2.
2. **Publication/formal historical claim:** do **not** state that the old
   Iteration-1 strict paired gate formally PASSed. The historical 3-state
   baseline-reproduction mismatch remains quarantined technical debt.

Register repair02 using `JIT/cli/select_iteration_policy.py` with
`--allow-baseline-reproduction-mismatch`. The selected artifact must preserve:

```text
engineering_selection = true
formal_acceptance_claim = false
baseline_reproduction_mismatch_quarantined = true
```

Do not alter the historical gate artifact to make it PASS.

### Mainline now

The active mainline is exactly:

```text
selected repair02 pi_1
  -> outcome-blind newest-shell frontier split
  -> pi_1 continuation labels
  -> C_up^1 / C_down^1 fit + disjoint calibration
  -> core-retaining Tube_2
  -> Tube_2-RSI smoke + split-isolation audit
  -> lock pi_1 acceptance baseline before pi_2 training
  -> fresh pi_2 training with strong retained-core replay
  -> freeze pi_2
  -> strict locked-baseline pi_1 -> pi_2 gate
  -> select pi_2 only if strict gate PASSes
```

Do **not** reopen A/B, continue B checkpoint sweeping, or run new warm-start
experiments before this mainline produces new evidence requiring a decision.

## Future gate protocol authority

For k >= 1, use `JIT/src/jit_dvgc/iterative_acceptance_gate.py` and the automatic
workflow's pre-candidate baseline lock.

The future gate contract is:

- full Tube_k is the structural retained-core bank;
- pi_k core outcomes are evaluated and locked before pi_(k+1) training;
- acceptance-role pi_k negatives are locked before candidate training;
- exact baseline PRNG identity is retained;
- baseline boundary negatives are not re-rolled after candidate training;
- candidate core uses the same locked core seeds;
- candidate boundary uses the same
  `labeling_seed -> candidate_index -> tick` PRNG hierarchy;
- core PASS requires zero baseline-success -> candidate-failure regressions;
- boundary PASS requires candidate success in the predeclared minimum number of
  parent groups;
- no validation/calibration/TEST/expert switching/training occurs during the
  gate.

This protocol hardens future rounds against the historical baseline-reproduction
mismatch. Do not retroactively rewrite the Iteration-1 historical claim.

## Modify-first repository policy

This rule is mandatory for future agent work:

1. **Modify or consolidate an existing production file first.** Do not create a
   new file merely because a new experiment, iteration, retry, or checkpoint
   exists.
2. A new production Python file is justified only for a genuinely new stable
   capability with a durable API. `pi_2`, `Tube_2`, `retry02`, or a new seed are
   data/config/run identities, not new capabilities.
3. Before adding a file, check whether the logic belongs in an existing package
   API, implementation module, CLI, test, or documentation page.
4. When a stage-specific implementation is superseded and has no active import
   or artifact-loader requirement, remove it from the active tree. Git history
   is the code archive; do not duplicate obsolete Python files into an archive
   directory.
5. **Deletion is a gated change, not a naming judgment.** Before deleting any
   Python/CLI/test file, prove that no retained production import, package API,
   active CLI, current test, artifact loader, config path, or frozen
   reproducibility path depends on it. After each deletion batch, run
   `compileall` plus the affected import/targeted tests before deleting anything
   else. If the dependency closure is uncertain, keep the file until the
   dependency is explicitly removed.
6. Do not move configs, frozen manifests, handoff locks, or other path-bound
   provenance merely for aesthetics. Paths recorded by artifacts are part of
   reproducibility.
7. Keep `JIT/cli/` thin: argument parsing + dispatch only. Reusable runtime,
   scientific, fitting, gating, and provenance logic belongs under
   `JIT/src/jit_dvgc/`. Tests belong under `JIT/tests/`.
8. Prefer one stable capability API over multiple stage-named CLIs. New envelope
   iterations must reuse the same production code with iteration/config data.

## Active package boundaries

Stable package APIs are exposed from:

- `jit_dvgc.training` — unified PPO, formal preflight, policy freezing
- `jit_dvgc.tube` — Soft Tube, Tube iteration, Tube-RSI smoke/sampling
- `jit_dvgc.snapshots` — handoff/unified snapshots and pools
- `jit_dvgc.acquisition` — real-dynamics boundary/transition-band acquisition
- `jit_dvgc.continuation` — policy-conditioned labels/fields/refit
- `jit_dvgc.analysis` — bounded TRAIN diagnostics
- `jit_dvgc.workflow` — resumable stage orchestration

Iteration-generic durable implementations additionally include:

- `iterative_frontier_protocol.py` — predeclared newest-shell train/calibration/
  acceptance frontier roles;
- `iterative_continuation_fields.py` — fixed-architecture C^k fit/calibration;
- `iterative_tube.py` — full-source-Tube retention + logical-TRAIN expansion;
- `iterative_acceptance_gate.py` — pre-candidate locked baseline and strict
  candidate gate.

Historical flat modules may remain only while an active artifact/import path
still depends on them. Do not add new flat stage-named modules merely for a new
iteration number.

## Iteration automation

The intended operator experience is one explicit launch, not repeated manual
command copying:

`python JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute`

The generic `k -> k+1` automatic path is now implemented for `k >= 1` through
`JIT/cli/prepare_iterative_envelope_workflow.py` and `jit_dvgc.workflow`.

Automation rules:

- Without `--execute`, print/validate the plan only.
- The workflow config SHA is immutable after state creation.
- Every stage declares prerequisites and a machine-readable completion artifact;
  the runner verifies the artifact before advancing.
- A failed scientific/engineering gate stops the workflow. Never auto-change a
  threshold, reward, replay ratio, PPO hyperparameter, network, physics setting,
  or acceptance criterion to force progress.
- Resume reuses completed stage artifacts only after revalidation; it does not
  overwrite a completed run or silently warm-start a `fresh_only` PPO stage.
- The iteration workflow must not contain final TEST/JCE/JEL stages.
- Frontier role assignment is outcome-blind and parent-disjoint before outcomes
  are observed.
- Automatic frontier generation uses only the newest Tube_k expansion shell. If
  the shell lacks sufficient two-phase/parent-group support, stop for a new
  scientific parent-generation decision; do not silently fall back to full Tube.
- Tube_(k+1) must retain every Tube_k entry exactly and may add only logical-TRAIN
  positive states above the frozen C^k disjoint-calibration threshold.
- The next policy's retained-core replay refers to the **whole source Tube_k**,
  not only the original 222 Tube_0 entries.
- Future policy selection requires the new strict locked-baseline gate to PASS.

Regression coverage for the generic automation contracts lives in:

`JIT/tests/test_iterative_envelope_automation.py`

See `JIT/docs/CURRENT_STATUS.md`, `JIT/docs/CODE_ORGANIZATION.md`, and
`JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md` before changing scientific flow.
