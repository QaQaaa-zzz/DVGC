# DVGC Repository Layout and Cleanup Rules

Current for `agent/two-phase-soft-tube` as of 2026-09-01.

This document replaces the old pre-JIT cleanup ledger. Historical deletion decisions and branch-specific migration notes remain available in Git history; they are no longer current execution guidance.

## 1. Active project boundary

The current research implementation lives under `JIT/`.

Root `dvgc/`, root `cli/`, root `scripts/`, and root `tests/` contain older infrastructure and user work. They are not cleanup targets for the active JIT branch unless scope is explicitly expanded.

## 2. Active layout

```text
JIT/
├── AGENTS.md
├── README.md
├── cli/                  thin executable entry points
├── configs/              scientific/run declarations and immutable identities
├── docs/                 current method/status/verification documentation
├── handoff/              path-bound locked provenance
├── scripts/              local verification and maintenance entry points
├── src/jit_dvgc/
│   ├── training/         unified PPO and policy-freeze API
│   ├── tube/             Soft-Tube and Tube-RSI API
│   ├── snapshots/        handoff/unified snapshot and pool API
│   ├── acquisition/      real-dynamics boundary acquisition API
│   ├── continuation/     continuation labels/fields/refit API
│   ├── analysis/         bounded TRAIN diagnostics
│   └── workflow/         resumable stage orchestration
└── tests/                active regression/scientific-contract tests
```

Flat modules under `JIT/src/jit_dvgc/` are transitional. Keep them only while an active import, artifact loader, CLI, test, or path-bound reproducibility contract still needs them. New iteration-specific flat modules should not be added.

## 3. Stable capability rule

A production file should represent a reusable capability, not an experiment identity.

Good boundaries:

- train/freeze one unified policy;
- collect real-dynamics boundary evidence;
- label policy-conditioned continuation;
- fit/validate continuation fields;
- build one core-retaining next Tube;
- run Tube-RSI engineering smoke;
- evaluate core preservation and boundary gain;
- orchestrate declared stages.

Bad reasons for new production files:

- `pi_2` instead of `pi_1`;
- `Tube_2` instead of `Tube_1`;
- a different seed/checkpoint/retry;
- another upstream/downstream experiment variant;
- a new threshold value that belongs in config.

Prefer modifying/generalizing an existing stable capability.

## 4. CLI rule

`JIT/cli/` is for argument parsing and dispatch. Reusable implementation logic belongs under `JIT/src/jit_dvgc/`.

One stable CLI per capability is preferred. Experiment variants belong in JSON config and run metadata.

## 5. Test rule

`JIT/tests/` verifies current retained capabilities and scientific contracts.

A historical route test does not by itself justify keeping obsolete production code forever. Before deleting an old route, migrate any still-important generic invariant into the retained capability tests.

Do not create a new test file merely because the iteration number changed; extend the closest existing capability test when practical.

## 6. Deletion policy

Git history is the archive for obsolete code. Do not duplicate dead Python modules into an archive directory.

A candidate file may be removed from the active tree only after checking all of these dependency classes:

1. retained production Python imports;
2. package-root exports;
3. active CLI imports;
4. current tests;
5. dynamic/artifact loaders and pickle/module-path compatibility;
6. current configs and protocol path references;
7. frozen manifests/handoff/reproducibility references;
8. maintenance scripts and current docs.

A file that merely *looks old* is not a deletion candidate until these checks pass.

### Required verification after each deletion batch

```bash
PY=/home/qy/mujoco_playground/.venv/bin/python
export PYTHONPATH="$PWD/JIT/src"

"$PY" -m compileall -q JIT/src JIT/cli
"$PY" -m pytest -q <affected targeted tests>
```

If import/test collection fails, cleanup stops immediately. Repair or restore the dependency before deleting anything else.

The 2026-09-01 removal of `upstream_boundary_lock.py` violated this rule: `soft_tube -> upstream_value -> upstream_boundary_lock` was still an active import chain. The file was restored and the deletion gate is now explicit in both root and JIT agent instructions.

## 7. Path-bound provenance

Do not move files solely for aesthetics when an immutable artifact records their path.

In particular, be conservative with:

- `JIT/configs/` used by frozen runs;
- `JIT/handoff/` locks/manifests;
- run/checkpoint paths referenced by frozen manifests;
- module paths embedded by pickle/serialization contracts.

Cleanup should reduce active source complexity without invalidating reproducibility.

## 8. Current retained legacy/bootstrap dependencies

Some upstream/downstream-named modules remain because the completed bootstrap artifacts and their loaders still depend on them. Examples include the first-pass V_up/V_down training/inference authority and boundary-lock contracts.

They should be removed only after their reusable inference/loading contracts are migrated into generic retained modules and regression-tested against existing artifacts.

Do not delete them simply because later iterations use unified-policy continuation fields.

## 9. Iteration-generalization debt

Before the workflow can run unattended beyond pi_1, the following must become iteration-generic:

- core-retaining Tube construction (`Tube_k -> Tube_(k+1)` rather than Tube_1 constants);
- continuation refit and fresh validation (`C^k` rather than iteration-0 helper assumptions);
- policy freeze and capability gates as machine-readable workflow stages;
- stable workflow configs/exports for `k -> k+1`.

The transition-band acquisition machinery is already substantially iteration-parameterized; downstream stages should consume its generic artifacts rather than recreating new iteration-named source modules.

## 10. Current scientific stage

Completed:

`experts -> Tube_0 -> pi_0 -> C^0 -> Tube_1 -> pi_1`

Next:

`freeze pi_1 -> core-preservation + boundary-gain gates`

Only after both gates pass may the project proceed with an empirical expansion claim and `C^1 -> Tube_2 -> pi_2`.

Final TEST/JCE/JEL remains outside the automatic iteration loop.
