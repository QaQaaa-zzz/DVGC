# Two-Phase Phase Expert Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task with review checkpoints.

**Goal:** Implement one auditable phase-expert training entrypoint, with Phase U natural-reset PPO smoke capability first and Phase D seed-gated support prepared without starting any training automatically.

**Architecture:** Add an external `PhaseExpertEnvAdapter` around the immutable `OrangeBikeDVGC` transition function. The adapter owns phase reset selection, two-phase event state, reward, success, timeout, and metrics while reusing the existing dynamics, action/observation/history contracts, physical failures, pure-JAX runtime, PPO runtime, and total-transition accounting.

**Tech Stack:** Python dataclasses and JSON, JAX/MJX/Brax, Orbax checkpoints, pytest, existing `dvgc.two_phase_runtime`, `dvgc.two_phase_semantics`, `dvgc.training_budget`, and `dvgc.runtime`.

## Global Constraints

- Use `/home/qy/mujoco_playground/.venv/bin/python`; do not modify the environment.
- Do not modify XML, action mapping, matcher, payload, force limits, or virtual environment.
- Do not replay or tune `reference_jump.csv` actions.
- Keep `OrangeBikeDVGC.step` unchanged unless a red test proves that a required read-only physical signal is unavailable; report such a finding before editing it.
- Treat the reference only as a kinematic guideline and weak prior.
- Gate C1 implements smoke capability for Phase U first. It does not authorize running PPO, pilot/formal training, Phase D training, feasibility learning, Soft Tubes, or unified PPO.
- Use total environment transitions as the public budget unit; legacy aliases must remain equal.
- Preserve the watchdog interlock and do not operate systemd.

---

### Task 1: Freeze run specification, authorization, and provenance schemas

**Files:**

- Create: `dvgc/phase_expert_training.py`
- Create: `tests/test_phase_expert_training.py`
- Create: `configs/phase_expert_smoke.json`

- [ ] Write failing tests for the two phase names, `PhaseExpertRunSpec`, immutable threshold-manifest input, smoke-only authorization, unique output directory, and Phase D seed requirements.
- [ ] Write failing tests that reject independent `requested_timesteps`, misaligned rollout-block budgets, pilot/formal levels, missing source hashes, and Phase D natural reset fallback.
- [ ] Implement the smallest schema/validator using `PPOBudgetReport`; expose only `requested_total_transitions` and assert both legacy/effective aliases equal their total-transition fields.
- [ ] Define the smoke config with one through four aligned PPO rollout blocks, explicit maximum interaction cost, fixed stop conditions, reward bounds, checkpoint cadence, and disjoint train/evaluation seeds.
- [ ] Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_expert_training.py
```

- [ ] Commit the validated schema paths explicitly with message `feat: define phase expert run contracts`.

### Task 2: Implement the pure-JAX Phase U environment adapter by TDD

**Files:**

- Modify: `dvgc/phase_expert_training.py`
- Modify: `tests/test_phase_expert_training.py`

- [ ] Write failing tests that every Phase U reset is an audited natural start, uses `source_phase=propulsion_ascent`, has valid three-frame history/last action, and never reads a reference row or action.
- [ ] Write failing scalar, `jax.jit`, `jax.vmap`, and batched-state tests for adapter reset/step and its namespaced event state.
- [ ] Write failing tests that early airborne does not terminate or succeed, early airborne may later latch the legal window, the latch remains monotonic, and full Apex-band membership is required for success.
- [ ] Write failing tests that pre-window takeoff/ascent progress reward is zero even when airborne, rising, or already latched; test bounded reward components after window entry.
- [ ] Write failing tests that prohibited contact, invalid wheel contact, roll, pitch, backward motion, platform back-edge exit, and nonfinite states remain terminal.
- [ ] Implement the minimal adapter by composing existing physical transitions and pure-JAX two-phase signals. Do not add latches to `env.step`.
- [ ] Run the targeted test file and inspect every failure cause before changing semantics.
- [ ] Commit explicit paths with message `feat: add propulsion ascent training adapter`.

### Task 3: Add deterministic evaluation and artifact lifecycle

**Files:**

- Modify: `dvgc/phase_expert_training.py`
- Modify: `tests/test_phase_expert_training.py`

- [ ] Write failing tests for immutable `run_manifest.json`, atomic `status.json`, append-only finite `metrics.jsonl`, resolved config, budget report, source hashes, and fixed evaluation protocol.
- [ ] Write failing tests for mutually exclusive `success`, `physical_failure`, `timeout`, and `other_failure` counts plus fine-grained terminal reasons and event ticks.
- [ ] Write failing tests that evaluation seeds are disjoint from training, repeat deterministically, never tune thresholds, and never promote a checkpoint beyond smoke evidence.
- [ ] Implement artifact initialization before environment construction and transitions `initialized -> running -> completed|failed|gate_pause`.
- [ ] Implement fixed evaluation against audited natural Phase U resets with component gates, episode lengths, returns, and outcome accounting.
- [ ] Run targeted tests and commit explicit paths with message `feat: add phase expert audit artifacts`.

### Task 4: Add exact checkpoint and resume contracts

**Files:**

- Modify: `dvgc/phase_expert_training.py`
- Modify: `tests/test_phase_expert_training.py`

- [ ] Write failing tests that checkpoints bind cumulative transitions, optimizer/normalizer state, PRNG lineage, phase, reset/reward/evaluation hashes, XML/action/observation/history hashes, and parent checkpoint.
- [ ] Write failing tests that exact resume requires matching `--resume-run` and `--restore-checkpoint`, writes a new output directory, preserves the parent, and rejects any contract drift.
- [ ] Implement checkpoint sidecars and exact validation by composing the existing runtime checkpoint facilities.
- [ ] Add an in-process tiny mocked update/resume test; do not run MuJoCo PPO in this task.
- [ ] Run targeted tests and commit explicit paths with message `feat: validate phase expert resume`.

### Task 5: Implement the single CLI and repository contract

**Files:**

- Create: `cli/train_phase_expert.py`
- Create: `tests/test_train_phase_expert.py`
- Modify: `tests/test_repository_contract.py`
- Modify: `dvgc/phase_expert_training.py`

- [ ] Write failing CLI tests for both phase choices, smoke-only execution, total-transition input, threshold manifest, Phase D seed inputs, exact resume pairing, collision-safe run creation, and structured pre-run errors.
- [ ] Replace the repository assertion that the future CLI is absent with assertions for the one stable entrypoint and no version-suffixed variants.
- [ ] Implement argument parsing and a `--preflight-only` path that validates/hashes inputs and emits no environment transitions.
- [ ] Wire the normal path to `run_phase_expert` without any automatic promotion or follow-on gate.
- [ ] Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py \
  tests/test_train_phase_expert.py \
  tests/test_repository_contract.py
```

- [ ] Commit explicit paths with message `feat: add stable phase expert cli`.

### Task 6: Enforce the Phase D seed boundary without training Phase D

**Files:**

- Modify: `dvgc/phase_expert_training.py`
- Modify: `tests/test_phase_expert_training.py`

- [ ] Write failing tests that accepted preliminary records are labeled only `physically_validated_descent_seed` and include MuJoCo-forward, finite, penetration, legal-geometry, short-horizon, real FIFO, and timing-explicit validation evidence.
- [ ] Write failing tests rejecting `reachable`, `expert_snapshot`, `Tube`, `safe`, and certification claims for preliminary records.
- [ ] Write failing tests that formal Phase D inputs require frozen-`pi_up` online Apex pre/nearest/post and early-descent provenance, with those records exceeding half of both admitted count and sampling mass.
- [ ] Implement manifest admission/validation only. Do not construct candidate states, build a bank, collect snapshots, or run Phase D PPO.
- [ ] Run targeted tests and commit explicit paths with message `feat: enforce descent seed provenance`.

### Task 7: Validate Gate C1 implementation and stop before PPO

**Files:**

- Modify: `docs/EXPERIMENT_STATE.md`

- [ ] Run static compilation and all focused tests.
- [ ] Run full pytest and `bash scripts/local_preflight.sh`.
- [ ] Do not run the runtime PPO gate or phase-expert smoke; record that the implementation review does not authorize interactions.
- [ ] Confirm no files under `runs/`, checkpoints, logs, caches, or local service state are staged.
- [ ] Update the experiment ledger with exact hashes, tests, zero new training transitions, remaining authorization boundary, and next permitted action.
- [ ] Use `verification-before-completion`, review the complete diff, and create a focused validation commit.
- [ ] Stop for explicit smoke-run authorization; do not enter Gate D1.
