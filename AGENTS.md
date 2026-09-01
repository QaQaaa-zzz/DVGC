# DVGC Repository Instructions

## Current research truth

DVGC/JIT is an iterative **single-policy empirical jumping-capability-envelope** project. The active method is:

`experts -> Tube_0 -> pi_0 -> C^0 -> Tube_1 -> pi_1 -> gates -> C^1 -> Tube_2 -> pi_2 -> ...`

The phase experts are bootstrap/data-generation tools. The deployable controller is always one unified Actor. A learned Soft Tube is training guidance only; it is not a certified safe set, viability kernel, or invariant set.

Current validated state on `agent/two-phase-soft-tube`:

- frozen `pi_up_star` and `pi_down_star` exist;
- bootstrap `V_up/V_down` and 222-entry Tube_0 exist;
- frozen `pi_0` exists;
- `C_up^0/C_down^0` passed fresh independent validation;
- core-retaining Tube_1 is complete with 3,119 TRAIN entries;
- `pi_1_tube1_natural10_10009600_seed821101_20260901_retry01` completed exactly 10,009,600 PPO training transitions with no validation/TEST usage and no expert switching;
- the next scientific step is **freeze exact pi_1 -> core-preservation gate + boundary-gain gate**;
- do not claim empirical envelope expansion until both gates pass;
- TEST/JCE/JEL remains untouched until a final frozen policy is selected.

Read `JIT/docs/CURRENT_STATUS.md` for exact current artifact identities.

## Immutable physical/task contracts

- Work on `agent/two-phase-soft-tube`; do not modify `main` unless explicitly authorized.
- Authoritative XML: `assets/orange_bike_4kg_horizontal.xml`.
- Current XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`.
- Payload: 2 kg. The `4kg` token in the filename is historical only.
- Control rate: 50 Hz.
- Hip/knee torque limits: +/-50 Nm.
- Action order: `[steer, rear-wheel drive, hip, knee]`.
- Do not change physics, reward meaning, action semantics, snapshot semantics, task geometry, or TEST isolation during cleanup/iteration work.
- Natural cold-start failure is an out-of-domain diagnostic for the current JCE scope; do not redesign reward/reset ratio because of it without a separate method decision.

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

## Iteration automation

The intended operator experience is one explicit workflow launch rather than repeated shell copy/paste:

`python JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute`

The workflow runner may sequence and resume stages, but it may not make scientific decisions. It must stop on failed gates; it must not automatically change thresholds, PPO hyperparameters, rewards, networks, physics, reset semantics, or acceptance criteria. It must never include final TEST/JCE/JEL stages.

Automatic `k -> k+1` execution is considered ready only after Tube construction, continuation fitting/validation, policy freezing, and capability gates are iteration-generic and covered by tests.

## Git and local-work safety

- Preserve unrelated user changes. Never reset, clean, stash, rebase, force-push, or overwrite them.
- Known unrelated local work may exist under root `dvgc/`, root `tests/`, `.vscode/`, local patches, or draft docs; do not touch it unless explicitly requested.
- Use `/home/qy/mujoco_playground/.venv/bin/python`; do not reinstall or reconfigure the environment.
- Keep formal run outputs/checkpoints/logs out of Git.
- Use focused commits and audit diffs after structural changes.

## Context recovery

For current work, read these in order:

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `PROJECT.md`
5. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`

Do **not** reconstruct current state from old Phase-U reports, obsolete watchdog documents, or historical experiment narratives when the current-status files already supersede them.
