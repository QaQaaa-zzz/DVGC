# Seven-policy Phase U Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warm-start three frozen unified policies into literal historical Phase U training for 14,991,360 transitions each, then run a paired seven-policy, four-onset, 1,000-episode-per-condition comparison.

**Architecture:** Extend the existing Phase U formal runner with one identity-checked Actor-plus-normalizer initialization path while keeping the Critic and optimizer fresh. Add a schema-driven experiment supervisor that locks four original policies, generates three Phase U configs, runs the arms sequentially, freezes their final checkpoints, and invokes the existing batched pulse evaluator once per fixed onset. Keep training, freezing, evaluation, reporting, and notification state in one immutable experiment root.

**Tech Stack:** Python 3.11, JAX, Brax PPO, MuJoCo/MJX, NumPy, pytest, existing JIT provenance/checkpoint and batched pulse comparison modules.

## Global Constraints

- Work only in `/home/qy/DVGC/JIT`; do not use STTW_CONTROL implementation or results.
- Preserve all existing run directories and unrelated working-tree files.
- Historical Phase U source is `phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928`.
- Each descendant uses literal Phase U `propulsion_ascent` reward, reset, PPO, horizon, observation, and action contracts.
- Each descendant restores only its source Actor and observation normalizer; Critic and optimizer are fresh.
- Each descendant trains exactly 14,991,360 transitions, or 610 complete 24,576-transition blocks.
- Final policies are four originals plus three descendants.
- Evaluation pulse windows are exactly steps 0-2, 5-7, 10-12, and 15-17, with zero requested disturbance elsewhere.
- Use 1,000 paired episodes per policy and onset: 28,000 disturbed episodes, maximum 11,200,000 evaluation transitions.
- Do not open final TEST or claim certified envelope expansion, universal recovery, or independent training repeats.
- Start and verify the JIT desktop error/completion watcher for the active experiment.
- The current STTW GPU process remains running; JIT training may coexist as authorized by the user.

---

### Task 1: Identity-checked Phase U Actor warm start

**Files:**
- Create: `JIT/src/jit_dvgc/phase_u_warm_start.py`
- Modify: `JIT/src/jit_dvgc/formal_training.py`
- Modify: `JIT/cli/train_phase_expert.py`
- Test: `JIT/tests/test_phase_u_warm_start.py`
- Test: `JIT/tests/test_formal_training.py`

**Interfaces:**
- Consumes: a `jit_frozen_development_checkpoint_v1` manifest and its identity-bound checkpoint.
- Produces: `load_phase_u_actor_initialization(frozen_policy: Path) -> ActorOnlyInitialization` and `run_phase_u_formal(..., actor_init_frozen_policy: Path | None = None)`.

- [ ] **Step 1: Write failing loader tests**

Create fixtures with a saved `CheckpointPayload`, frozen-policy hashes, and altered variants. Assert that the loader returns the exact normalizer and Actor, excludes the source Critic, and rejects Actor hash, normalizer hash, XML, action-order, observation-field, payload, transition, or manifest-status drift.

- [ ] **Step 2: Run the loader tests and confirm failure**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_phase_u_warm_start.py -q`

Expected: FAIL because `phase_u_warm_start` does not exist.

- [ ] **Step 3: Implement the focused loader**

Implement `load_phase_u_actor_initialization` using `CheckpointIdentity`, `load_checkpoint`, `pytree_sha256`, and the frozen manifest's declared identities. Return an `ActorOnlyInitialization` whose provenance records Actor and normalizer restoration with fresh Critic and optimizer.

- [ ] **Step 4: Write failing formal-runner tests**

Add a fake-trainer test that calls `run_phase_u_formal` with `actor_init_frozen_policy`, then asserts:

```python
assert kwargs["restore_params"] == (source_normalizer, source_actor)
assert kwargs["restore_value_fn"] is False
assert manifest["starting_training_transition"] == 0
assert manifest["resume_semantics"] == "parameter_warm_start_optimizer_reset"
assert manifest["parent_checkpoint"] == source_checkpoint
```

Also assert mutual exclusion between `restore_checkpoint` and `actor_init_frozen_policy`.

- [ ] **Step 5: Route Actor-only initialization through Phase U training**

Add `--actor-init-frozen-policy` to `train_phase_expert.py`. In `run_phase_u_formal`, load the initializer before the run declaration, preserve absolute training transition zero, set `restore_params` to the two-item Actor-only tuple and `restore_value_fn=False`, and record all source hashes in formal provenance. Keep legacy full-checkpoint resume unchanged.

- [ ] **Step 6: Run focused and existing formal tests**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_phase_u_warm_start.py \
  JIT/tests/test_formal_training.py \
  JIT/tests/test_formal_provenance.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add JIT/src/jit_dvgc/phase_u_warm_start.py JIT/src/jit_dvgc/formal_training.py \
  JIT/cli/train_phase_expert.py JIT/tests/test_phase_u_warm_start.py \
  JIT/tests/test_formal_training.py
git commit -m "Add Phase U actor-only warm starts"
```

### Task 2: Exact fixed-onset pulse verification

**Files:**
- Modify: `JIT/src/jit_dvgc/batched_pulse_comparison.py`
- Modify: `JIT/tests/test_batched_pulse_comparison.py`

**Interfaces:**
- Consumes: a saved rollout tape, one declared onset tick, and pulse length.
- Produces: `verify_fixed_pulse_tape(tape: Mapping[str, np.ndarray], onset: int, pulse_steps: int) -> None`.

- [ ] **Step 1: Write failing mask tests**

Cover onsets 0, 5, 10, and 15. For each, assert acceptance only when `mask` equals the live prefix intersected with `[onset, onset + 3)`, and assert rejection for a nonzero `requested_delta` before or after that interval.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_batched_pulse_comparison.py -q`

Expected: FAIL because only the onset-zero validator exists.

- [ ] **Step 3: Implement and apply the general validator**

Replace the onset-zero-only logic with `verify_fixed_pulse_tape`. During batch acceptance, require every condition spec to contain exactly one onset and validate that onset. Preserve cross-policy equality checks for random draws and requested residuals before termination.

- [ ] **Step 4: Run pulse and report regressions**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_batched_pulse_comparison.py \
  JIT/tests/test_pulse_schedule.py \
  JIT/tests/test_full_episode_pulse.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add JIT/src/jit_dvgc/batched_pulse_comparison.py JIT/tests/test_batched_pulse_comparison.py
git commit -m "Validate fixed pulse onset windows"
```

### Task 3: Seven-policy experiment supervisor

**Files:**
- Create: `JIT/src/jit_dvgc/seven_policy_phase_u.py`
- Create: `JIT/cli/run_seven_policy_phase_u.py`
- Create: `JIT/tests/test_seven_policy_phase_u.py`
- Modify: `JIT/AGENTS.md`

**Interfaces:**
- Consumes: four source policy declarations, the historical Phase U resolved config, a prior completed pulse-comparison template, seeds, and an output root.
- Produces: `prepare_experiment(...) -> dict`, `run_experiment(spec_path: Path) -> dict`, and `report_experiment(spec_path: Path) -> dict`.

- [ ] **Step 1: Write failing preparation tests**

Assert that preparation locks every source file and policy identity, emits three configs with all Phase U fields unchanged except run ID, seed, 14,991,360-transition budget, block-rounded schedules, and Actor initializer, and predeclares exactly seven final methods and four 1,000-episode conditions.

- [ ] **Step 2: Run the supervisor tests and confirm failure**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_seven_policy_phase_u.py -q`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement preparation and source locking**

Implement schema-driven source declarations with no policy-name conditionals in runtime code. Generate `spec.json`, `source_lock.json`, three resolved Phase U configs, `ACTIVE_RUN.json`, `status.json`, and `INDEX.md`. Refuse an existing output root or any mismatched source hash.

- [ ] **Step 4: Implement sequential training and freezing**

Run one descendant subprocess at a time with `JAX_PLATFORMS=cuda,cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Update the top-level status before and after each arm, retain child logs, verify the final formal report and checkpoint, then freeze the descendant into a development checkpoint manifest. On failure, retain the failed attempt and stop before evaluation.

- [ ] **Step 5: Implement four paired evaluation conditions**

For onsets 0, 5, 10, and 15, generate one batched pulse spec with `pulse_start_schedule=[onset]`, seven locked methods, 1,000 episodes, batch size 256, and shared condition seeds. Run and verify each condition, then combine per-episode rows and summaries without changing official success labels.

- [ ] **Step 6: Implement reporting and durable index**

Generate per-onset seven-policy success/failure tables, confidence intervals, landing/recovery timing data, common-axis trajectory-density panels, source/config hashes, cost reconciliation, and links from the experiment `INDEX.md`. Preserve raw uncropped arrays.

- [ ] **Step 7: Document reusable command and watcher contract**

Add the stable `prepare`, `run`, and `report` commands to `JIT/AGENTS.md`, including exact seven-policy membership, Phase U warm-start meaning, four fixed windows, and notification watcher requirement.

- [ ] **Step 8: Run supervisor and affected regression tests**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_seven_policy_phase_u.py \
  JIT/tests/test_batched_pulse_comparison.py \
  JIT/tests/test_formal_training.py \
  JIT/tests/test_formal_provenance.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add JIT/src/jit_dvgc/seven_policy_phase_u.py JIT/cli/run_seven_policy_phase_u.py \
  JIT/tests/test_seven_policy_phase_u.py JIT/AGENTS.md
git commit -m "Add seven-policy Phase U experiment pipeline"
```

### Task 4: Prepare, smoke-check, and launch the production experiment

**Files:**
- Create: `JIT/runs/experiments/seven_policy_phase_u_15m_pulse_20260918/` runtime artifacts
- Modify: `JIT/docs/CURRENT_STATUS.md`

**Interfaces:**
- Consumes: committed pipeline code and immutable source checkpoints.
- Produces: a detached supervisor PID, live status, verified notification watcher heartbeat, and eventually the full seven-policy report.

- [ ] **Step 1: Run preparation**

Run the CLI `prepare` command with the approved source paths, historical Phase U configuration, output root, three fixed training seeds, four fixed evaluation seeds, 1,000 episodes, and batch size 256.

- [ ] **Step 2: Verify the frozen declaration**

Run the CLI `verify` command and inspect `source_lock.json`, generated configs, exact 44,974,080 maximum training transitions, and 11,200,000 maximum final-evaluation transitions.

- [ ] **Step 3: Run a bounded shared-GPU smoke**

Use the production environment with preallocation disabled to load all three initializers and compile one Phase U environment/policy inference. Record peak GPU memory and ensure the existing STTW PID remains alive. Do not charge the smoke as scientific training.

- [ ] **Step 4: Launch the detached supervisor immediately**

Start `run` under `nohup` with unbuffered output. Verify the PID, first descendant child PID, GPU process listing, and live status. Do not wait for the STTW process to finish.

- [ ] **Step 5: Start and verify notifications**

Launch `watch_run_errors.py` against the experiment `ACTIVE_RUN.json` with preserved desktop DBus environment. Verify watcher PID, heartbeat, and `notification_status.json` without claiming that an unseen desktop popup was displayed.

- [ ] **Step 6: Update status documentation and commit**

Record source identities, exact budgets, live PIDs, current stage, smoke scope, and the fact that results are pending in `JIT/docs/CURRENT_STATUS.md`; commit and push only the code, tests, plan, and compact status metadata.

- [ ] **Step 7: Monitor through completion and publish the report**

Follow the supervisor status without restarting healthy training. When all three arms and four evaluation conditions complete, run report verification, update `INDEX.md` and `CURRENT_STATUS.md`, commit compact results, push, and report actual success rates, trajectories, interaction counts, failures, and limitations.
