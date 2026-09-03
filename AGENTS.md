# DVGC Repository Instructions

## Current research truth — 2026-09-03

DVGC/JIT is an iterative **single-policy empirical jumping-capability-envelope** project. The active method is:

```text
experts
  -> Tube_0
  -> pi_0
  -> C^0
  -> Tube_1
  -> selected pi_1
  -> C^1
  -> Tube_2
  -> pi_2
  -> strict gate
  -> repeat
```

The phase experts are bootstrap/data-generation tools. The deployable controller is always one unified Actor. A learned Soft Tube is empirical training/curriculum guidance only; it is not a certified safe set, viability kernel, reachability proof, or invariant set.

### Current authority

The Iteration-1 initialization/replay study is closed.

- frozen `pi_up_star` and `pi_down_star` exist;
- bootstrap `V_up/V_down` and 222-entry Tube_0 exist;
- frozen `pi_0` exists;
- `C_up^0/C_down^0` passed fresh independent validation/calibration;
- core-retaining Tube_1 is complete with 3,119 TRAIN entries = 222 retained Tube_0 + 2,897 expansion;
- the first Tube_1 policy failed core preservation and was scientifically rejected;
- retained-core replay repair was investigated;
- warm-start A is discarded;
- warm-start B checkpoint sweep is complete and must not be continued;
- **repair02 is selected as the engineering `pi_1` authority**;
- repair02 preserves Tube_0 at `222/222`, upstream `117/117`, downstream `105/105`, with historical boundary quickcheck `26/260` across 4 parent groups;
- no B checkpoint achieved both `Tube_0 = 222/222` and boundary success `> 26/260`;
- best B tradeoff was 7.5008M at `217/222 + 42/260`; B final was `212/222 + 46/260`;
- all B core regressions were upstream while downstream remained `105/105`, supporting upstream expansion/retention interference rather than simple monotonic overtraining;
- final TEST/JCE/JEL remains untouched.

Selected repair02 frozen policy:

```text
JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json
```

Actor SHA-256:

```text
85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a
```

Payload SHA-256:

```text
3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc
```

### Historical Iteration-1 claim boundary

repair02 is selected for **engineering continuation** of the JIT loop, but do **not** claim that the historical Iteration-1 strict paired gate formally PASSed.

The historical quickcheck has 3 baseline-reproduction failures caused by the old continuation-label vs paired-gate PRNG hierarchy mismatch. These are preserved as historical protocol debt.

Use `JIT/cli/select_iteration_policy.py --allow-baseline-reproduction-mismatch` so the selected artifact records the distinction rather than rewriting history.

Required meaning:

```text
engineering_selection = true
formal_acceptance_claim = false
baseline_reproduction_mismatch_quarantined = true
```

### Active mainline

The only active scientific path now is:

```text
selected repair02 pi_1
  -> outcome-blind newest-shell TRAIN/CALIBRATION/ACCEPTANCE frontier roles
  -> pi_1-conditioned continuation evidence
  -> C_up^1 / C_down^1 fit + disjoint calibration
  -> core-retaining Tube_2
  -> Tube_2-RSI smoke + role-isolation audit
  -> lock pi_1 acceptance baseline before candidate training
  -> fresh pi_2 training
  -> freeze pi_2
  -> strict locked-baseline pi_1 -> pi_2 gate
  -> select pi_2 only if the gate PASSes
```

Do **not** reopen A/B, continue B checkpoint sweeping, or launch new warm-start experiments before this mainline produces new evidence that requires a scientific decision.

Read `JIT/docs/CURRENT_STATUS.md` for exact current artifact identities and `JIT/docs/CODEX_HANDOFF_20260903.md` for the full takeover procedure.

## Immutable physical/task contracts

- Work on `agent/two-phase-soft-tube`; do not modify `main` unless explicitly authorized.
- Authoritative XML: `assets/orange_bike_4kg_horizontal.xml`.
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`.
- Payload: 2 kg. The `4kg` token in the filename is historical only.
- Control rate: 50 Hz.
- Hip/knee torque limits: +/-50 Nm.
- Action order: `[steer, rear-wheel drive, hip, knee]`.
- Unified policies never switch between the phase experts at runtime.
- Do not change physics, reward meaning, action semantics, snapshot semantics, task geometry, collision geometry, or TEST isolation during an iteration.
- Natural cold-start failure remains an out-of-domain diagnostic for the current declared JCE scope; do not redesign reward/reset ratio because of it without a separate method decision.

## Data-role and claim isolation

For the generic iterative regime:

- `TRAIN`: may fit `C^k` and may contribute qualifying expansion to `Tube_(k+1)`;
- `CALIBRATION`: threshold calibration only; never enters TRAIN or a Tube;
- `ACCEPTANCE`: pre-candidate baseline/candidate gate only; never trains/calibrates C^k and never enters a Tube;
- final TEST/JCE/JEL: untouched until the final policy and stopping decision are frozen.

Parent-group disjointness is required across TRAIN/CALIBRATION/ACCEPTANCE. Seed disjointness alone is not enough.

A larger Tube or higher PPO reward does not prove envelope expansion. A new policy becomes the next authority only after the declared capability gate passes.

## Iteration automation and code-control logic

The generic automatic `k -> k+1` path is implemented for `k >= 1`.

Prepare the workflow with:

```text
JIT/cli/prepare_iterative_envelope_workflow.py
```

Run it with:

```text
JIT/cli/run_iteration_workflow.py
```

Operator form:

```bash
python JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute
```

Without `--execute`, the runner must only resolve/print the plan.

The generated DAG is:

```text
selected pi_k + Tube_k
  -> prepare newest-shell frontier plan
  -> TRAIN frontier
  -> CALIBRATION frontier
  -> ACCEPTANCE frontier
  -> fit/calibrate C^k
  -> build Tube_(k+1)
  -> Tube-RSI smoke
  -> role-isolation audit
  -> lock exact pi_k acceptance baseline
  -> prepare/train pi_(k+1)
  -> freeze exact final checkpoint
  -> strict locked-baseline gate
  -> select pi_(k+1) only on PASS
```

Automation rules:

- workflow config SHA is immutable after state creation;
- every stage declares prerequisites and a machine-readable completion artifact;
- existing completion artifacts are revalidated before reuse;
- completed stages are not silently rerun;
- failed scientific/engineering assertions stop the workflow;
- automation may not change thresholds, rewards, replay ratios, PPO settings, network architecture, physics, reset semantics, or acceptance criteria to force PASS;
- final TEST/JCE/JEL stages must never appear in the iteration workflow;
- automatic frontier generation uses only the newest Tube_k expansion shell and must not silently fall back to the full Tube;
- `Tube_(k+1)` retains **every Tube_k entry exactly** and may add only qualifying logical-TRAIN expansion;
- for pi_2, retained core means all 3,119 Tube_1 states, not only the original 222 Tube_0 states;
- the selected mainline replay contract is outer 90% Tube / 10% natural, with 75% retained source Tube_k / 25% newest expansion inside Tube sampling;
- future acceptance uses the pre-candidate locked-baseline protocol so historical PRNG reproduction debt is not repeated.

Regression coverage for these contracts is in:

```text
JIT/tests/test_iterative_envelope_automation.py
```

## Repository-maintenance policy

The repository must become smaller and more reusable as iterations advance.

1. **Modify or consolidate an existing production file first.** Do not create a new production file merely because the experiment number, retry, seed, checkpoint, `pi_k`, or `Tube_k` changed.
2. New production Python files are allowed only for a genuinely new stable capability with a durable API.
3. Iteration identity belongs in config/artifact/run metadata, not filenames such as `pi2_*`, `tube2_*`, `upstream_v7_*`, or `retry03_*` source modules.
4. Keep CLI files thin: argument parsing and dispatch only. Reusable logic belongs in `JIT/src/jit_dvgc/`; tests belong in `JIT/tests/`.
5. Prefer the stable capability packages `jit_dvgc.training`, `tube`, `snapshots`, `acquisition`, `continuation`, `analysis`, and `workflow`.
6. Stage-specific research scaffolding that is superseded and no longer referenced should leave the active tree. Git history is the code archive; do not duplicate obsolete Python files into an archive directory.
7. Do not move path-bound configs, frozen manifests, handoff locks, or run identities merely for aesthetics; recorded paths are reproducibility data.

### Mandatory deletion gate

Never delete a Python/CLI/test file because it only *looks old*.

Before deletion, verify that no retained production import, package API, active CLI, current test, artifact loader, current config, or frozen reproducibility path still depends on it. After every deletion batch, run at least:

- `python -m compileall -q JIT/src JIT/cli`
- targeted import/tests for the affected capability

If collection/import fails, stop cleanup and repair the dependency closure before deleting anything else.

## Git and local-work safety

- Preserve unrelated user changes. Never reset, clean, stash, rebase, force-push, or overwrite them.
- Known unrelated local work may exist under root `dvgc/`, root `tests/`, `.vscode/`, local patches, or draft docs; do not touch it unless explicitly requested.
- Use `/home/qy/mujoco_playground/.venv/bin/python`; do not reinstall or reconfigure the environment.
- Keep formal run outputs/checkpoints/logs out of Git.
- Use focused commits and audit diffs after structural changes.

## Context recovery for Codex/agents

Read these in order before changing scientific flow:

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/CODEX_HANDOFF_20260903.md`
5. `PROJECT.md`
6. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
7. `JIT/docs/CODE_ORGANIZATION.md`

Then inspect the actual runtime artifacts referenced by the current status before launching or resuming the workflow.

Do **not** reconstruct current state from old Phase-U reports, obsolete watchdog documents, or historical experiment narratives when the current authority documents supersede them.
