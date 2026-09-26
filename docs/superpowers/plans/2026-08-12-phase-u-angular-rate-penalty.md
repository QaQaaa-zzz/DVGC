# Phase U Angular-Rate Penalty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether increasing only the bounded Phase U angular-rate penalty from 0.25 to 1.0 prevents the 2 kg policy from learning high-rate pre-window pitch failure while preserving learnable ascent progress.

**Architecture:** Keep `phase_u_reward_components` and every physical/runtime/PPO contract unchanged; change only the two stable Phase U training configuration values. Bind the new reward contract through existing manifest/hash validation, qualify it with red-green tests and the full repository gates, then run one fresh 512-environment smoke before deciding whether to launch a fresh formal experiment.

**Tech Stack:** Python 3.12, JAX, MJX-Warp, Brax PPO, pytest, JSON run manifests, Orbax inference checkpoints.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- The authoritative XML remains the 2 kg model at SHA-256 `e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192`.
- Change only `phase_u_reward.angular_rate_penalty_weight: 0.25 -> 1.0` in the two stable Phase U configs.
- Do not change any other reward term, exploration prior `[0.05, 0.05, 0.25, 0.05]`, reset, observation, threshold/deadline, XML, action mapping, network, PPO hyperparameter, horizon, or evaluation seed.
- Do not resume a checkpoint whose reward-contract hash is 0.25; all new PPO runs use fresh policy initialization and fresh run-bound authorization.
- Smoke is engineering evidence only. Snapshot/continuation acquisition requires real Apex successes and at least eight independent successful parents.
- Run outputs remain ignored and must not be committed. Preserve `.vscode/` untouched.

---

### Task 1: Red-Green Reward Configuration Contract

**Files:**
- Modify: `tests/test_phase_expert_training.py`
- Modify: `configs/phase_expert_smoke.json`
- Modify: `configs/phase_expert_phase_u.json`

**Interfaces:**
- Consumes: `resolve_phase_u_reward_config(mapping) -> PhaseURewardConfig` and `phase_u_reward_components(...) -> dict[str, Any]`.
- Produces: stable configs with `angular_rate_penalty_weight == 1.0` and a distinct reward-contract hash.

- [ ] **Step 1: Write the failing stable-config test**

Extend the existing stable Phase U config test so each config must resolve both the unchanged exploration vector and the new angular-rate coefficient:

```python
def test_phase_u_configs_select_explicit_exploration_and_angular_rate_penalty():
    module = _module()
    for path in (
        "configs/phase_expert_smoke.json",
        "configs/phase_expert_phase_u.json",
    ):
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        assert module.resolve_policy_initial_action_std(config) == (
            0.05, 0.05, 0.25, 0.05
        )
        assert module.resolve_phase_u_reward_config(
            config
        ).angular_rate_penalty_weight == 1.0
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py::test_phase_u_configs_select_explicit_exploration_and_angular_rate_penalty
```

Expected: FAIL because both configs still resolve `0.25`, while the exploration assertion passes.

- [ ] **Step 3: Make the minimal configuration change**

In both files change exactly:

```json
"angular_rate_penalty_weight": 0.25
```

to:

```json
"angular_rate_penalty_weight": 1.0
```

- [ ] **Step 4: Verify GREEN and bounded reward behavior**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py::test_phase_u_configs_select_explicit_exploration_and_angular_rate_penalty \
  tests/test_phase_expert_training.py::test_phase_u_reward_components_are_bounded_gated_and_future_free \
  tests/test_phase_expert_training.py::test_phase_u_reward_config_is_explicit_and_finite
```

Expected: PASS. The existing direct component test proves linear dependence on the configured coefficient and preserved window gating.

- [ ] **Step 5: Commit the single behavior change**

```bash
git add configs/phase_expert_smoke.json configs/phase_expert_phase_u.json tests/test_phase_expert_training.py
git commit -m "reward: strengthen phase u angular-rate penalty"
```

---

### Task 2: Static and Runtime Requalification

**Files:**
- Modify through generator: `docs/RUNTIME_GATE.json`
- Modify: `docs/BUILD_VALIDATION.json`
- Modify: `docs/EXPERIMENT_STATE.md`
- Create ignored: `runs/two_phase/runtime_gate/phase_u_2kg_angular_rate_20260812/`

**Interfaces:**
- Consumes: the new config/reward hash and unchanged 2 kg runtime.
- Produces: current source fingerprint and evidence that reset/step/PPO/checkpoint-resume remain valid.

- [ ] **Step 1: Run compilation and targeted tests**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py \
  tests/test_prelaunch_continuation.py \
  tests/test_two_phase_semantics.py
```

- [ ] **Step 2: Run full tests and repository preflight**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
```

- [ ] **Step 3: Refresh runtime gate exactly once**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate \
  --config configs/default.json \
  --output docs/RUNTIME_GATE.json \
  --work-dir runs/two_phase/runtime_gate/phase_u_2kg_angular_rate_20260812
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate \
  --config configs/default.json \
  --output docs/RUNTIME_GATE.json \
  --check-only
```

Expected: PASS, current fingerprint, 64+32 engineering transitions, finite metrics, exact snapshot/policy round trips.

- [ ] **Step 4: Record exact validation evidence and commit**

Update `docs/BUILD_VALIDATION.json` and `docs/EXPERIMENT_STATE.md` with actual test counts, runtime elapsed time, new reward-contract hash, and the 96 non-formal runtime interactions.

```bash
git add docs/RUNTIME_GATE.json docs/BUILD_VALIDATION.json docs/EXPERIMENT_STATE.md
git commit -m "test: qualify angular-rate reward contract"
git push origin agent/two-phase-soft-tube
```

---

### Task 3: Fresh Run-Bound Smoke

**Files:**
- Create ignored: `runs/two_phase/configs/phase_u_2kg_angrate1_env512_smoke_20260812.json`
- Create ignored: `runs/two_phase/authorizations/gate_c1_phase_u_2kg_angrate1_env512_smoke_20260812_seed720101.json`
- Create ignored: `runs/two_phase/phase_experts/gate_c1_phase_u_2kg_angrate1_env512_smoke_20260812_seed720101/`
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: new reward-contract/source hash, unchanged threshold manifest, fixed seed/evaluation protocol.
- Produces: one 12,800-transition engineering qualification with closed accounting and failure videos.

- [ ] **Step 1: Create the run-specific config and preflight it**

Copy the validated 512-env smoke config after Task 1. Keep one 12,800-transition block, 1,600 Brax-evaluation ceiling, 1,600 fixed-evaluation ceiling, and all non-reward fields byte-for-byte equivalent.

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.train_phase_expert \
  --phase propulsion_ascent --experiment-level smoke \
  --requested-total-transitions 12800 --seed 720101 \
  --config configs/default.json \
  --training-config runs/two_phase/configs/phase_u_2kg_angrate1_env512_smoke_20260812.json \
  --threshold-manifest runs/two_phase/gate_b_2kg_20260812/threshold_manifest.json \
  --run runs/two_phase/phase_experts/gate_c1_phase_u_2kg_angrate1_env512_smoke_20260812_seed720101 \
  --preflight-only
```

Expected: requested/effective 12,800 and zero executed transitions.

- [ ] **Step 2: Issue a new authorization**

Bind run ID, seed `720101`, absolute output, producer HEAD, source-tree/XML/training-config/threshold hashes, exact ceilings, engineering purpose, stop condition, `promotion_authorized=false`, and exclusions for formal `V_up`, Tube, Phase D, and unified PPO.

- [ ] **Step 3: Execute smoke once**

Run the same command without `--preflight-only` and with the authorization path.

- [ ] **Step 4: Audit integrity and physical diagnostics**

Require status `completed`, 12,800 training, finite PPO metrics, valid recursive checkpoint identity, closed `outcome_counts`, MP4/NPZ for every failure, and no broadphase overflow, NaN/Inf, OOM, traceback, timing/history/hash mismatch. Report window/liftoff/Apex/pitch descriptively; no success requirement for one block.

- [ ] **Step 5: Record and commit smoke evidence**

```bash
git add docs/EXPERIMENT_STATE.md
git commit -m "docs: record angular-rate phase u smoke"
git push origin agent/two-phase-soft-tube
```

---

### Task 4: Formal Decision and Sparse Supervision

**Files:**
- Create ignored if authorized: `runs/two_phase/configs/phase_u_2kg_angrate1_env512_formal_20260812.json`
- Create ignored if authorized: `runs/two_phase/authorizations/phase_u_2kg_angrate1_env512_998400_20260812_seed720102.json`
- Create ignored if authorized: `runs/two_phase/phase_experts/phase_u_2kg_angrate1_env512_998400_20260812_seed720102/`
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: clean smoke and the user's delegated maximum of 1,000,000 training transitions for a new run-bound hypothesis.
- Produces: a fresh, detached Phase U experiment or a documented no-launch decision.

- [ ] **Step 1: Make the formal authorization decision**

Authorize only if smoke has engineering integrity and no immediate reward-hacking evidence. Use fresh initialization; never restore the 0.25-reward checkpoint because its reward hash differs.

- [ ] **Step 2: If authorized, bind an aligned budget and checkpoints**

Use 512 environments and 998,400 maximum transitions with effective checkpoints `0/102400/256000/512000/755200/998400`. Keep independent ceilings: fixed evaluation 9,600, candidate 76,800, continuation 76,800, total 1,161,600.

- [ ] **Step 3: Launch persistently and inspect startup once**

Use `nohup setsid`, write PID/log/control/resume paths, confirm `running`, checkpoint 0, exact identities, and a clean startup log. Then stop polling until checkpoint/terminal scale.

- [ ] **Step 4: Audit only closed checkpoints**

At each sparse inspection report window reach, liftoff, stable airborne, clearance, Apex, physical failure, roll/pitch/illegal-contact rates, action saturation, return decomposition, parent diversity, candidate count, and continuation count. Gate Pause on the existing numerical/contract/degradation/plateau/reward-hacking conditions.

- [ ] **Step 5: Decide Soft-Tube progress**

Start candidate acquisition only after held-out Apex success is nonzero with at least eight independent successful parents and no contract failure. Otherwise preserve videos, diagnose one new hypothesis, and do not claim `pi_up_star`, formal `V_up`, or Tube.

