# JIT Phase U Formal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add, verify, push, and persistently launch one auditable 998,400-transition Propulsion-Ascent PPO run without changing the verified environment, reward, observations, action mapping, networks, or PPO hyperparameters.

**Architecture:** Keep the existing one-block smoke runner unchanged and add a formal-only configuration plus a focused `formal_training.py` orchestration module. A callback controller converts Brax segment-relative steps to absolute transitions, persists identity-bound checkpoints, runs the frozen eight-seed deterministic panel only at declared milestones, and records exact ledgers. Normal execution is one uninterrupted process; an abnormal-exit recovery is explicitly a parameter warm start with optimizer and PPO RNG reset.

**Tech Stack:** Python 3.12, JAX 0.6.2/CUDA, MuJoCo 3.6.0, MJX Warp, MuJoCo Playground, Brax 0.14.2 PPO, pytest, JSON/JSONL, NumPy NPZ, Git.

## Global Constraints

- Work only in `/home/qy/DVGC`; every generated source, test, document, config, launcher record, and run output belongs under `JIT/`.
- Use `/home/qy/mujoco_playground/.venv/bin/python` directly and do not modify that environment.
- Load only `assets/orange_bike_4kg_horizontal.xml`; retain the 2 kg payload, hip/knee limits of +/-50 N m, and action order `[steer, rear-wheel drive, hip, knee]`.
- Retain Actor `81 -> 256 -> 256 -> 256 -> 8`, deterministic four-action output, and critic `114 -> 256 -> 256 -> 256 -> 1`.
- Retain 1,024 environments, horizon 200, unroll 25, batch 128, eight minibatches, one update per batch, and 25,600 transitions per block.
- Formal training is exactly 39 blocks or 998,400 transitions with training seed `820101`.
- Fixed evaluation uses only seeds `920001` through `920008` at absolute transitions `102400`, `256000`, `512000`, `742400`, and `998400`.
- Checkpoints exist at absolute transitions `0`, `102400`, `256000`, `512000`, `742400`, and `998400`; the final checkpoint must be restored before completion.
- Do not add or claim Phase D, continuation labels, `V_up`/`V_down`, learned soft Tubes, unified PPO, JCE, or JEL.
- Do not start formal environment interaction until all implementation is verified, committed with explicit JIT-only staging, and pushed to `origin/agent/two-phase-soft-tube`.
- Preserve the user's modified and untracked paths outside `JIT/`; never stage them.

---

## File Responsibility Map

- `JIT/configs/phase_u_formal.json`: immutable formal budget, seed, and milestone declaration.
- `JIT/src/jit_dvgc/config.py`: parse and reject malformed formal schedules before run creation.
- `JIT/src/jit_dvgc/provenance.py`: immutable parent/start/resume metadata, predeclared-to-running state change, and strict formal ledger verification.
- `JIT/src/jit_dvgc/evaluation.py`: serialize every fixed-evaluation state trace independently of video.
- `JIT/src/jit_dvgc/formal_training.py`: own formal callback controller, fixed evaluation, warm resume, report, and terminal closure.
- `JIT/cli/train_phase_expert.py`: exactly-one-mode CLI dispatch for `--smoke` and `--formal`.
- `JIT/tests/test_formal_config.py`: exact budget, schedules, and malformed-config rejection.
- `JIT/tests/test_formal_provenance.py`: running state, resume metadata, trace serialization, and formal verifier behavior.
- `JIT/tests/test_formal_training.py`: callback orchestration, absolute offsets, checkpoint/evaluation schedule, finite metrics, and final restore.
- `JIT/tests/test_cli.py`: mode separation and restore-argument refusal boundaries.
- `JIT/README.md`: formal command, claim boundary, persistence, and sparse inspection instructions.
- `JIT/planning/{task_plan,findings,progress}.md`: persistent execution record without generated evidence outside `JIT/`.

### Task 1: Strict formal configuration and schedule

**Files:**
- Create: `JIT/configs/phase_u_formal.json`
- Create: `JIT/tests/test_formal_config.py`
- Modify: `JIT/src/jit_dvgc/config.py`

**Interfaces:**
- Consumes: existing `PPOConfig.block_transitions` and `load_config(path: Path) -> ResolvedConfig`.
- Produces: `FormalTrainingConfig(checkpoint_transitions, fixed_evaluation_transitions, resume_semantics)`, `ResolvedConfig.formal: FormalTrainingConfig | None`, and `FormalTrainingConfig.formal_blocks: int`.

- [ ] **Step 1: Write failing exact-layout tests**

```python
def test_formal_config_is_exactly_39_aligned_blocks(jit_root):
    config = load_config(jit_root / "configs/phase_u_formal.json")
    assert config.formal is not None
    assert config.ppo.seed == 820101
    assert config.ppo.requested_transitions == 998_400
    assert config.formal.formal_blocks == 39
    assert config.ppo.num_evals == 40
    assert config.formal.checkpoint_transitions == (0, 102_400, 256_000, 512_000, 742_400, 998_400)
    assert config.formal.fixed_evaluation_transitions == (102_400, 256_000, 512_000, 742_400, 998_400)
    assert config.ppo.held_out_seeds == tuple(range(920001, 920009))
```

Add parameterized mutations that reject a non-998,400 budget, a non-block-aligned milestone, a missing zero/final checkpoint, an evaluation outside the checkpoint set, a seed panel other than exactly eight disjoint seeds, `num_evals != 40`, and any formal section in the smoke schema.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_formal_config.py -q`

Expected: collection failure for missing `phase_u_formal.json` or missing `ResolvedConfig.formal`.

- [ ] **Step 3: Add the formal dataclass and strict validation**

```python
@dataclass(frozen=True)
class FormalTrainingConfig:
    checkpoint_transitions: tuple[int, ...]
    fixed_evaluation_transitions: tuple[int, ...]
    resume_semantics: str

    @property
    def formal_blocks(self) -> int:
        return 998_400 // 25_600
```

Parse JSON arrays as integer tuples. Require schema `jit_phase_u_formal_v1`, exact target `998_400`, exact block `25_600`, exact `num_evals=40`, ordered unique checkpoint/evaluation schedules, zero and target checkpoints, evaluation as the nonzero checkpoint set, exactly the frozen eight seeds, and `resume_semantics="parameter_warm_start_optimizer_reset"`. For the smoke schema set `formal=None` and reject a `formal` key.

- [ ] **Step 4: Create the immutable formal JSON by copying the verified smoke physics/reward fields and changing only declared formal fields**

Use schema `jit_phase_u_formal_v1`, seed `820101`, `requested_transitions=998400`, `num_evals=40`, and this object:

```json
"formal": {
  "checkpoint_transitions": [0, 102400, 256000, 512000, 742400, 998400],
  "fixed_evaluation_transitions": [102400, 256000, 512000, 742400, 998400],
  "resume_semantics": "parameter_warm_start_optimizer_reset"
}
```

- [ ] **Step 5: Run GREEN and regression tests**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_formal_config.py JIT/tests/test_contracts.py -q`

Expected: all pass; the smoke still resolves to exactly one block and `formal is None`.

- [ ] **Step 6: Review the diff but defer commit until the complete formal runner is validated**

Run: `git diff --check -- JIT/configs/phase_u_formal.json JIT/src/jit_dvgc/config.py JIT/tests/test_formal_config.py`

Expected: no output.

### Task 2: Running provenance, resume truth, and trace persistence

**Files:**
- Create: `JIT/tests/test_formal_provenance.py`
- Modify: `JIT/src/jit_dvgc/provenance.py`
- Modify: `JIT/src/jit_dvgc/evaluation.py`

**Interfaces:**
- Consumes: `RunDeclaration`, `predeclare_run`, `close_run`, `EpisodeTrace`, and `EpisodeFrame`.
- Produces: `mark_run_running(run_dir: Path, *, process_id: int, metadata: Mapping[str, Any]) -> None`, optional immutable declaration fields `parent_checkpoint`, `starting_training_transition`, `resume_semantics`, and `segment_seed`, plus `TraceArtifact` and `save_episode_trace(trace: EpisodeTrace, path: Path) -> TraceArtifact`.

- [ ] **Step 1: Write failing provenance-state and resume-metadata tests**

```python
def test_mark_running_preserves_predeclared_identity(tmp_path):
    run_dir = _predeclared_formal_run(tmp_path)
    mark_run_running(run_dir, process_id=1234, metadata={"started_utc": "2026-08-24T00:00:00Z"})
    status = json.loads((run_dir / "status.json").read_text())
    assert status == {"status": "running", "process_id": 1234, "started_utc": "2026-08-24T00:00:00Z"}
```

Also require `process_id > 0`, reject any current status except `predeclared`, preserve a warm segment's parent checkpoint/start transition/segment seed in `run_manifest.json`, and allow `close_run` from `running`.

- [ ] **Step 2: Write failing exact trace-artifact tests**

Create a three-transition `EpisodeTrace`, call `save_episode_trace(trace, tmp_path / "seed_920001")`, and assert NPZ arrays have four states for `qpos/qvel/ctrl/action/reward/terminated/truncated/end_code/success`, all reward component arrays and all metric arrays are present, the JSON sidecar reports seed, three transitions, four captured states, terminal flags, and both files are returned as absolute paths.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_formal_provenance.py -q`

Expected: import failures for `mark_run_running`, `TraceArtifact`, and `save_episode_trace`.

- [ ] **Step 4: Implement the atomic running transition and immutable declaration fields**

```python
@dataclass(frozen=True)
class RunDeclaration:
    # existing required fields remain first
    parent_checkpoint: str | None = None
    starting_training_transition: int = 0
    resume_semantics: str = "fresh"
    segment_seed: int | None = None

def mark_run_running(run_dir: Path, *, process_id: int, metadata: Mapping[str, Any]) -> None:
    if process_id <= 0:
        raise ValueError("process_id must be positive")
    status_path = Path(run_dir) / "status.json"
    current = json.loads(status_path.read_text(encoding="utf-8"))
    if current != {"status": "predeclared"}:
        raise ValueError("run is not predeclared")
    _atomic_json(status_path, {"status": "running", "process_id": process_id, **dict(metadata)})
```

Validate nonnegative starting transition and require parent checkpoint plus the exact optimizer-reset resume semantics when the start is nonzero. Fresh runs require start zero, no parent checkpoint, and `resume_semantics="fresh"`.

- [ ] **Step 5: Implement independent trace serialization**

`save_episode_trace` writes `<path>.npz` and `<path>.json`, checks `len(frames) == transitions + 1`, serializes fixed state/action/terminal fields, all `REWARD_COMPONENT_KEYS`, and the sorted union of metric names. Missing metrics in an individual frame serialize as NaN only after the metric-name union is established; production traces are expected to have a stable metric pytree. JSON uses `allow_nan=False` and records terminal summary without duplicating large arrays.

- [ ] **Step 6: Run GREEN plus existing provenance/evaluation regressions**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_formal_provenance.py JIT/tests/test_provenance_verify.py JIT/tests/test_evaluation.py -q`

Expected: all pass.

### Task 3: Formal callback controller and exact interaction ledger

**Files:**
- Create: `JIT/src/jit_dvgc/formal_training.py`
- Create: `JIT/tests/test_formal_training.py`

**Interfaces:**
- Consumes: `ResolvedConfig.formal`, `CheckpointIdentity`, `CheckpointPayload`, `save_checkpoint`, `load_checkpoint`, `capture_episode`, `summarize_phase_u`, `save_episode_trace`, `render_trace`, and `InteractionAccounting`.
- Produces: `FormalReport`, `validate_formal_report(report: FormalReport) -> FormalReport`, `FormalRunController.on_policy_params(relative_step, make_policy, params)`, `FormalRunController.on_progress(relative_step, metrics)`, and `run_phase_u_formal(config_path, run_id, *, restore_checkpoint=None, run_root=None, trainer=ppo_train.train, env_factory=TwoPhaseBikeEnv, panel_evaluator=None, backend_name=jax.default_backend) -> dict[str, Any]`.

- [ ] **Step 1: Write failing report validation tests**

```python
def test_formal_report_requires_exact_target_and_panels():
    report = validate_formal_report(FormalReport(
        requested_training_transitions=998_400,
        starting_training_transition=0,
        completed_training_transitions=998_400,
        segment_training_transitions=998_400,
        brax_evaluation_transitions=0,
        fixed_evaluation_transitions=8 * 5 * 200,
        checkpoint_transitions=(0, 102_400, 256_000, 512_000, 742_400, 998_400),
        evaluated_transitions=(102_400, 256_000, 512_000, 742_400, 998_400),
        final_metrics={"training/sps": 1.0},
        checkpoint_restored=True,
        resume_semantics="fresh",
    ))
    assert report.completed_training_transitions == 998_400
```

Reject nonfinite metrics, a final transition other than target, missing schedules, nonzero Brax evaluation, a segment count not equal to final minus start, and absent final restore evidence. Fixed evaluation counts may be below `8 * panels * 200` because episodes stop at done, but must be positive and no greater than that ceiling.

- [ ] **Step 2: Write failing callback schedule tests with in-memory parameters**

Provide a temporary run directory and an `evaluate_panel(absolute_step, make_policy, params)` spy returning eight traces and their counted transitions. Call `on_policy_params` at relative steps `0`, each 25,600 block, and the final step. Assert checkpoint directories and evaluation calls occur only at configured absolute milestones, metrics JSONL has one finite row per nonzero block, and duplicate/out-of-order callbacks are rejected.

- [ ] **Step 3: Write failing warm-offset tests**

Initialize the controller at `starting_training_transition=256000`; pass relative callbacks through `742400` remaining transitions. Assert all stored transitions are absolute, only future checkpoint/evaluation milestones execute, and the expected final absolute transition is `998400`. Reject a restore checkpoint whose payload transition differs from the declared start.

- [ ] **Step 4: Run tests and confirm RED**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_formal_training.py -q`

Expected: collection failure because `jit_dvgc.formal_training` does not exist.

- [ ] **Step 5: Implement the report and controller before the real trainer wiring**

The controller receives injected `checkpoint_saver`, `checkpoint_loader`, and `evaluate_panel` callables so unit tests execute the actual scheduling logic without GPU interaction. `on_policy_params` computes `absolute = starting + relative`, requires block alignment and monotonicity, saves the real initial parameters at absolute zero, performs declared checkpoints/evaluations, and accumulates fixed evaluation transitions. `on_progress` flattens only scalar finite values, writes one JSONL row atomically/appended, and updates the completed segment count.

- [ ] **Step 6: Implement fixed eight-seed evaluation**

For each held-out seed, build `deterministic_policy = make_policy(params, deterministic=True)`, use a seed-specific policy PRNG key, call `capture_episode` with JIT-compiled reset/step and horizon 200, save `evaluations/transition_<N>/seed_<S>.npz` plus JSON, then write `summary.json` from `summarize_phase_u`. Return exact summed transitions and retain traces in memory only for the current panel. At the final panel, render the first successful trace or otherwise the first trace; rendering consumes zero interactions.

- [ ] **Step 7: Run controller GREEN tests**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_formal_training.py -q`

Expected: all controller/report tests pass without creating a real environment.

### Task 4: Real Brax runner, warm restore, and CLI separation

**Files:**
- Modify: `JIT/src/jit_dvgc/formal_training.py`
- Modify: `JIT/cli/train_phase_expert.py`
- Modify: `JIT/tests/test_formal_training.py`
- Modify: `JIT/tests/test_cli.py`

**Interfaces:**
- Consumes: installed `ppo_train.train`, `wrapper.wrap_for_brax_training`, `make_network_factory`, `TwoPhaseBikeEnv`, and Task 3 controller callbacks.
- Produces: a stable mutually exclusive `--smoke | --formal` CLI and optional `--restore-checkpoint PATH` accepted only with `--formal`.

- [ ] **Step 1: Write a fake-trainer end-to-end orchestration test**

The injected trainer asserts `num_timesteps=998400`, `num_evals=40`, `run_evals=False`, `num_envs=1024`, and unchanged PPO hyperparameters. It invokes `policy_params_fn(0, make_policy, params)`, then one callback/progress pair for each of 39 exact blocks, and returns `(make_policy, params, final_metrics)`. Inject a lightweight environment factory, panel evaluator, and `backend_name=lambda: "gpu"`; assert predeclared -> running -> completed, exact accounting, final checkpoint restore, and `formal_report.json`.

- [ ] **Step 2: Write CLI mode tests**

Assert no mode and both modes fail before run creation; `--restore-checkpoint` with `--smoke` fails; formal mode dispatches `run_phase_u_formal`; smoke continues to dispatch `run_phase_u_smoke`; Descent-Recovery remains rejected.

- [ ] **Step 3: Run focused tests and confirm RED**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_formal_training.py JIT/tests/test_cli.py -q`

Expected: fake-trainer and CLI formal-dispatch tests fail because runner wiring is absent.

- [ ] **Step 4: Implement formal preflight and warm restore**

`run_phase_u_formal` loads only a formal config, requires `jax.default_backend() == "gpu"`, constructs the real environment before interaction, validates XML/reference/config identity, predeclares the run, writes backend metadata, and marks it running with PID/UTC/seed/target/resume semantics. Fresh mode starts at zero. Restore mode loads the identity-bound payload first, derives `starting_training_transition` from it, requires a future aligned target, records the resolved absolute parent path, sets segment seed to `820101 + starting_transition // 25600`, and passes `(normalizer, actor, critic)` as `restore_params`.

- [ ] **Step 5: Call Brax with exact segment math**

For `remaining = 998400 - start`, pass `num_timesteps=remaining`, `num_evals=remaining // 25600 + 1`, unchanged PPO/network/wrapper arguments, `progress_fn=controller.on_progress`, `policy_params_fn=controller.on_policy_params`, `run_evals=False`, and the optional `restore_params`. After return, require the controller reached absolute `998400`, restore `checkpoints/transition_998400`, validate one finite deterministic inference action, write the report, and close completed. On any exception, close `engineering_error` with transitions observed so far and re-raise.

- [ ] **Step 6: Implement exactly-one-mode CLI dispatch**

Use `mode = parser.add_mutually_exclusive_group(required=True)`, add `--smoke` and `--formal`, and reject restore unless formal. Import the formal runner lazily after argument validation so invalid modes never initialize JAX or create a run.

- [ ] **Step 7: Run GREEN and all non-GPU tests**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests -q -m "not gpu"`

Expected: all non-GPU tests pass; no real formal transitions occur.

### Task 5: Strict formal verifier, documentation, and launch contract

**Files:**
- Modify: `JIT/src/jit_dvgc/provenance.py`
- Modify: `JIT/tests/test_formal_provenance.py`
- Modify: `JIT/scripts/local_preflight.sh`
- Modify: `JIT/tests/test_preflight_contract.py`
- Modify: `JIT/README.md`
- Modify: `JIT/planning/task_plan.md`
- Modify: `JIT/planning/findings.md`
- Modify: `JIT/planning/progress.md`

**Interfaces:**
- Consumes: closed formal run manifest/status, `formal_report.json`, checkpoint sidecars, fixed evaluation summaries, and trace sidecars.
- Produces: `verify_run` formal branch that proves hashes, schedules, checkpoint payload identities, panel seeds/counts, final restore, and ledger equality.

- [ ] **Step 1: Write failing closed-formal verifier tests**

Build a small synthetic completed formal directory with identity-bound checkpoint sidecars and evaluation metadata. Assert verification rejects: missing milestone checkpoint; incorrect final payload hash; missing held-out seed; evaluation summary/trace transition mismatch; report/status fixed-evaluation mismatch; `checkpoint_restored != true`; and any completed absolute transition other than `998400`.

- [ ] **Step 2: Run the verifier tests and confirm RED**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_formal_provenance.py -q`

Expected: malformed formal directories are incorrectly accepted or formal evidence is not inspected.

- [ ] **Step 3: Implement the formal verification branch**

Dispatch on `purpose == "formal_propulsion_ascent_ppo"`. Recompute authoritative input hashes, verify the config is formal, require the segment training count to equal `998400 - starting_training_transition` and zero Brax evaluation, validate every checkpoint/evaluation milestone at or after the segment start, sum eight trace sidecars per executed panel, require exact held-out seed sets, match the status ledger to `formal_report.json`, and report Apex/terminal summaries without asserting promotion.

- [ ] **Step 4: Update preflight without launching or requiring a formal result**

Keep compilation, legacy-import audit, all non-GPU tests, GPU environment tests, reference verification, and the retained successful smoke verification. Add static parsing of the formal config through a short Python command. Preserve tests that prohibit `train_phase_expert.py` and `998400` as executable training commands inside preflight.

- [ ] **Step 5: Update README with the exact post-push command and claim boundary**

Document:

```bash
mkdir -p JIT/runs/phase_u
nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent --config JIT/configs/phase_u_formal.json \
  --run-id phase_u_formal_998400_seed820101_20260824 --formal \
  > JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824.launch.log 2>&1 \
  < /dev/null &
```

State that completion is formal training evidence, not automatically a trained expert or safety/certification result.

- [ ] **Step 6: Run verifier/preflight contract GREEN tests**

Run: `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_formal_provenance.py JIT/tests/test_preflight_contract.py -q`

Expected: all pass.

### Task 6: Verification, JIT-only GitHub delivery, and persistent launch

**Files:**
- Modify: `JIT/planning/task_plan.md`
- Modify: `JIT/planning/progress.md`
- Runtime output only: `JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824/`
- Runtime launcher only: `JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824.{launch.log,pid}`

**Interfaces:**
- Consumes: all preceding source/contracts and the fixed GitHub branch.
- Produces: one validated JIT-only source commit on GitHub, then one persistent formal process with auditable startup evidence.

- [ ] **Step 1: Run static and complete JIT verification**

Run: `bash JIT/scripts/local_preflight.sh`

Expected: compilation succeeds, no `dvgc` import is found, all non-GPU and GPU tests pass, reference analysis succeeds, retained smoke verifies, and no formal training is launched.

- [ ] **Step 2: Run repository compatibility verification while preserving the known user-dirty diagnostic test**

Run the repository pytest suite with only `tests/test_phase_u_launch_diagnostic.py` deselected if its known user-owned modification still fails. Record exact pass/fail counts and never edit that file merely to obtain green.

- [ ] **Step 3: Audit generated and staged content**

Run `git status --short`, `git diff --check -- JIT`, scan JIT source for forbidden `dvgc` imports and placeholder claims, explicitly stage only the changed JIT source/config/tests/docs/planning paths, then run `git diff --cached --name-only`, `git diff --cached --check`, and a scan proving no `JIT/runs`, checkpoints, videos, logs, caches, or outside-JIT path is staged.

- [ ] **Step 4: Commit and push before any formal interaction**

```bash
git commit -m "feat(jit): add auditable phase U formal training"
git fetch origin agent/two-phase-soft-tube
git rev-list --left-right --count origin/agent/two-phase-soft-tube...HEAD
git push origin HEAD:agent/two-phase-soft-tube
git ls-remote origin refs/heads/agent/two-phase-soft-tube
```

Expected: no remote divergence, push succeeds without force, and remote SHA equals local `HEAD`.

- [ ] **Step 5: Launch once, persistently, after verifying the run id does not exist**

Resolve the exact run directory, log, and pid paths; require all three absent. Launch the documented `nohup setsid env ... --formal` command, capture `$!` in the ignored pid file, and do not reuse the run id after any partial creation.

- [ ] **Step 6: Perform one sparse startup inspection**

After the process has initialized, inspect only PID liveness, `nvidia-smi`, the tail of the launcher log, `status.json`, `backend.json`, and `checkpoints/transition_0/identity.json`. Stop and report immediately for NaN, OOM/CUDA error, identity failure, or abnormal exit. Otherwise report `status=running`, exact PID/GPU, transition-zero checkpoint identity, and next declared inspection at transition `102400`.

- [ ] **Step 7: Inspect only milestones and terminal state**

At `102400`, inspect PPO finite metrics, KL, terminal causes, checkpoint restore identity, and all eight evaluation traces without changing reward/hyperparameters. Repeat sparse evidence checks only at later declared milestones or on process exit. On terminal completion, run `python -m jit_dvgc.provenance verify-run <run_dir>` and classify the learned behavior from held-out Apex/physical-failure results without upgrading the method claim automatically.

## Self-Review Result

- Spec coverage: every design section maps to Tasks 1-6; source push is a hard predecessor of Task 6 launch.
- Placeholder scan: no deferred implementation markers or unspecified error-handling steps remain.
- Type consistency: absolute transitions are integers everywhere; the controller receives Brax's `(normalizer, actor, critic)` tuple; fixed evaluation is counted separately from training and rendering; warm recovery never claims optimizer-exact resume.
- Execution choice: inline execution with `executing-plans`, because the user asked to begin without additional confirmations and repository instructions require work in the current `/home/qy/DVGC/JIT` tree while preserving outside-JIT changes.
