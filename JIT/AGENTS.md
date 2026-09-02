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

`experts -> Tube_0 -> pi_0 -> C^0 -> Tube_1 -> pi_1 -> gates -> C^1 -> Tube_2 -> pi_2 -> ...`

- `V_up/V_down` are bootstrap expert-conditioned Tube_0 authorities only.
- `C_up^k/C_down^k` are policy-conditioned empirical continuation fields bound
  to the exact frozen `pi_k` identity.
- Every Tube is training guidance/curriculum support, not a certified safe set,
  viability kernel, or formal invariant set.
- `Tube_(k+1)` must retain the existing Tube core and add qualifying TRAIN
  expansion states; expansion is not a replacement level set.
- A larger Tube does not prove capability gain. After training `pi_(k+1)`, both
  core-preservation and boundary-gain gates must pass before claiming empirical
  jumping-capability-envelope expansion.
- Validation is independent calibration/gating evidence and its rows never enter
  TRAIN or a Tube. Consumed validation is never reused for tuning.
- TEST/final JCE/JEL evidence stays untouched until a final frozen policy has
  been selected. It may not affect Tube construction, threshold selection,
  policy training, checkpoint selection, or iteration stopping.

## Current authority — 2026-09-02

The project is at **repaired iteration-1 acceptance**.

Completed authoritative chain:

- Tube_1: 3,119 TRAIN entries = exact 222-entry Tube_0 core + 2,897 expansion;
  manifest SHA-256:
  `817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`.
- The first completed Tube_1 `pi_1` candidate was frozen and scientifically
  rejected: 21 core regressions, core FAIL, old 56-state boundary PASS.
- Zero-interaction diagnosis established retained-core replay dilution as a
  material mechanism.
- The repaired method uses 50% retained core / 50% expansion inside each phase,
  with the outer 90% Tube / 10% natural reset mixture unchanged.
- Two baseline-only single-axis acceptance-bank readiness probes were consumed
  as FAIL evidence; neither produced downstream negatives.
- A systematic two-axis real-dynamics acquisition generated 3,720 fresh TRAIN
  candidates with zero exact overlap against the two consumed readiness probes.
- Frozen-`pi_0` labeling was completed as four sequential 930-candidate GPU
  processes after long single-process labeling hit CUDA/Warp allocator OOM. The
  execution shards were merged once in original catalog order without changing
  candidates, seed, horizon, label semantics, physics, or acceptance criteria.
- The resulting fresh acceptance bank contains 260 frozen-`pi_0` negatives:
  246 upstream across 4 parent groups and 14 downstream across 5 parent groups;
  pre-training readiness PASS.
- Repaired `pi_1` completed exactly 10,009,600 PPO transitions and was frozen at:
  `JIT/runs/frozen_unified/pi_1_core_replay50_10009600_20260902/frozen_unified_policy.json`.
- Repaired final checkpoint payload SHA-256:
  `ea93a534c2c6bb3bf145684cbea82df94fefa2df8099dcdcdd9492bd8007e205`.
- Repaired frozen manifest file SHA-256:
  `d5a1658530d475a67264aa5c621283d71c823200dbee6068f93413b93d06b7a8`.
- final TEST/JCE/JEL remains untouched.

The next scientific stage is exactly one paired repaired `pi_0 -> pi_1` gate:

1. structural core bank = all 222 Tube_0 entries;
2. fresh boundary bank = locked 260-state two-axis frozen-`pi_0` negatives;
3. core PASS requires zero baseline-success -> candidate-failure regressions;
4. all fresh baseline negatives must reproduce;
5. boundary PASS requires candidate success in at least 2 distinct parent groups;
6. no validation, TEST, expert switching, or training during the gate.

**Do not start accepted `C^1`, Tube_2, or pi_2 until both gates pass.** If either
gate fails, preserve and diagnose; do not tune the bank, replay ratio, reward,
PPO settings, or acceptance threshold to force PASS.

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

Historical flat modules may remain only while an active artifact/import path
still depends on them. Do not add new flat stage-named modules.

## Iteration automation

The intended operator experience is one explicit launch, not repeated manual
command copying:

`python JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute`

Automation rules:

- Without `--execute`, print/validate the plan only.
- The workflow config SHA is immutable after state creation.
- Every stage must declare prerequisites and a machine-readable completion
  artifact. The runner verifies the artifact before advancing.
- A failed scientific/engineering gate stops the workflow. Never auto-change a
  threshold, reward, PPO hyperparameter, network, physics setting, or acceptance
  criterion to force progress.
- Resume reuses completed stage artifacts only after revalidation; it does not
  overwrite a completed run or silently warm-start a `fresh_only` PPO stage.
- The iteration workflow must not contain final TEST/JCE/JEL stages.
- Automatic multi-iteration execution is authorized only after the current
  iteration-specific Tube/continuation code has been generalized and tested for
  `k -> k+1`; until then the runner is infrastructure, not permission to claim
  unattended pi_2 readiness.

See `JIT/docs/CURRENT_STATUS.md`, `JIT/docs/CODE_ORGANIZATION.md`, and
`JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md` before changing scientific flow.
