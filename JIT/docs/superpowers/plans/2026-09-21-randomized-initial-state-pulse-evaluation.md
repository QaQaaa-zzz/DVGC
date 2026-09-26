# Randomized Initial State Pulse Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configuration-driven initial velocity randomization to fixed jump-start pulse evaluation and launch a five-policy, five-onset, 10,000-episode-per-cell comparison.

**Architecture:** Extend the reset path with a declared randomization object that modifies only qvel before the normal reset forward pass. Split its RNG from the pulse RNG, persist initialization noise in the rollout tape, and keep the existing pulse mask and success accounting unchanged. Prepare a new immutable experiment directory and launch it with the existing watcher contract.

**Tech Stack:** Python, JAX, MuJoCo MJX, NumPy, existing batched pulse comparison runner and Matplotlib report tooling.

## Global Constraints

- Keep the world jump-start `x=2.5 m` fixed.
- Use forward/lateral initial velocity noise `±0.1 m/s`; root angular velocity noise `±0.05 rad/s`; steering/hip/knee velocity noise `±0.10 rad/s`.
- Keep pose, joint positions, history, and contact geometry deterministic.
- Keep pulse residuals `U(-0.25,+0.25)` for exactly three ticks at onset `0/5/10/15/25`, zero elsewhere.
- Preserve old runs and write all new data under a new experiment directory.
- Run tests before launch and verify the watcher heartbeat after launch.

### Task 1: Add reset randomization contract and tape fields

**Files:**
- Modify: `src/jit_dvgc/unified_env.py`
- Modify: `src/jit_dvgc/pulse_exploration_runtime.py`
- Modify: `src/jit_dvgc/pulse_schedule.py`
- Test: `tests/test_pulse_schedule.py`
- Test: `tests/test_pulse_exploration.py`

**Interfaces:**
- Add `initial_velocity_randomization(spec)` returning a validated immutable mapping of seven velocity ranges.
- Add `reset_with_initial_velocity_noise(env, rng, randomization)` returning a reset state and the sampled noise vector.
- Extend the tape with `initial_velocity_noise` and `initial_velocity_noise_applied`.

- [ ] Add failing unit tests for defaults, range validation, fixed root position, and zero noise outside the declared fields.
- [ ] Run the focused tests and confirm the new API is absent or fails.
- [ ] Implement the smallest configuration-driven reset helper; do not hard-code experiment names or seeds.
- [ ] Split reset RNG from pulse RNG with `jax.random.split`, apply noise to qvel only, and preserve the root qpos x assignment.
- [ ] Run focused CPU tests and a reset fixture asserting finite state and exact root x.
- [ ] Commit the source and tests as `feat: randomize initial velocities for pulse evaluation`.

### Task 2: Prepare the 25-cell evaluation

**Files:**
- Create: `runs/experiments/five_policy_random_initial_pulse_20260921/spec.json`
- Create: `runs/experiments/five_policy_random_initial_pulse_20260921/run.py`
- Create: `runs/experiments/five_policy_random_initial_pulse_20260921/INDEX.md`

**Interfaces:**
- Consume the four historical policy banks and the frozen fusion student bank.
- Produce five onset directories, each with 10,000 episodes per policy, batch receipts, initialization-noise summaries, and reports clipped visually at `x≤4.5 m` while retaining complete episodes.

- [ ] Copy only immutable policy identities, configs, and source locks into a new declared snapshot.
- [ ] Declare common onset seeds, `batch_size=256`, horizon400, five policies, and the initial randomization mapping.
- [ ] Add report tables for initialization-noise percentiles and pulse-request percentiles separately.
- [ ] Run preflight identity and contract checks without simulation.
- [ ] Commit only durable source/config changes; keep run artifacts ignored.

### Task 3: Launch and verify the 250,000-episode run

**Files:**
- Modify: `runs/experiments/five_policy_random_initial_pulse_20260921/ACTIVE_RUN.json`
- Create: `runs/experiments/five_policy_random_initial_pulse_20260921/notifications/`

- [ ] Start the supervisor and desktop error/completion watcher.
- [ ] Verify status, child PID, GPU process, and watcher heartbeat.
- [ ] Check the first completed batch: initialization noise is nonzero and within bounds; pulse requests occur only in its three-tick window.
- [ ] Continue all 25 cells and preserve any failed batch logs without relabeling them.
- [ ] Generate success-rate and envelope reports using the new x≤4.5m display boundary.
- [ ] Verify completion status and report links before claiming results.
