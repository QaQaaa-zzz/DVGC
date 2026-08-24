# Progress Log

## Session: 2026-08-24 Formal Phase U

### Phase 1: Formal-training design

- **Status:** complete
- Actions taken:
  - Confirmed the previous JIT commit is present on the GitHub branch before formal work.
  - Inspected Brax PPO training callbacks, checkpoint content, and restore behavior.
  - Selected one uninterrupted 39-block process with parameter-level warm resume.
  - Defined formal checkpoints, five eight-seed evaluation panels, persistence, and stopping conditions.
  - Wrote `JIT/docs/specs/2026-08-24-jit-phase-u-formal-training-design.md`.
  - Self-reviewed the design for exact budget, run id, resume truthfulness,
    evaluation accounting, and the no-training-before-push gate.
- No environment interactions or training transitions were consumed in this phase.

## Session: 2026-08-24

### Phase 1: Requirements and design

- **Status:** complete
- Actions taken:
  - Read repository method and runtime constraints.
  - Read all 1,242 lines of the rebuild guide.
  - Inspected the reference CSV schema, row count, time span, and hash.
  - Confirmed user-owned dirty-tree paths and preserved them.
  - Confirmed the independent-package and first-delivery scope with the user.
  - Verified the GPU backend outside device isolation without changing the environment.
- Files created/modified:
  - `JIT/planning/task_plan.md`
  - `JIT/planning/findings.md`
  - `JIT/planning/progress.md`
  - `JIT/docs/specs/2026-08-24-jit-phase-u-engineering-delivery-design.md`

### Phase 2: Detailed implementation planning

- **Status:** complete
- Actions taken:
  - Initialized persistent planning files under `JIT/planning/`.
  - Mapped the exact XML keyframe, joint, collision geom, sensor, and actuator names.
  - Confirmed `MjxEnv` timing and `mjx_env.step` interfaces.
  - Confirmed analytic geometry support and fixed-shape MJX contact interfaces.
  - Confirmed installed Brax PPO asymmetric-network and restore interfaces.
  - Wrote the detailed nine-task TDD implementation plan.
  - Replaced ambiguous plan ellipses and defined the 114-value privileged observation.
  - Corrected the Actor frame dimension from 23 to 27 during design self-review.
- Files created/modified:
  - `JIT/planning/task_plan.md`
  - `JIT/planning/findings.md`
  - `JIT/planning/progress.md`

### Phase 3: TDD implementation

- **Status:** complete
- Actions taken:
  - Read the good-test rules and began Task 1 with contract behavior tests.
  - Task 1 RED: two expected collection errors because `jit_dvgc` was absent.
  - Task 1 GREEN: implemented constants, strict config, provenance, and offline reference analysis.
  - Task 2 RED: two expected collection errors for missing model and action modules.
  - Task 2 first GREEN attempt: 6 passed, 1 failed on a false MuJoCo per-geom mass assumption.
  - Traced payload identity to the XML element because compiled MuJoCo retains aggregate body mass only.
  - Task 2 GREEN: all model-audit and action-mapping tests passed.
  - Task 3 RED: expected collection failure because the observation module was absent.
  - Task 3 GREEN: implemented 27-value frames, real three-frame FIFO, 81-value Actor input, and 114-value critic input.
  - Task 4 RED: expected collection failures for missing geometry and end-code semantics.
  - Task 4 first GREEN attempt: 18 passed, one false-precision assertion failed.
  - Task 4 GREEN: full collision support, event, Apex, and terminal tests passed.
  - Task 5 RED: expected collection failure because the reward module was absent.
  - Task 5 first GREEN attempt exposed JAX dictionary key reordering.
  - Task 5 GREEN: fixed-field component reward tests passed.
  - Added a new RED/GREEN cycle so zero hip action holds the XML keyframe target.
  - Probed MJX Warp and replaced contact-array logic with deployable geometry/IMU estimates.
  - Task 6 Host and GPU integration passed, including 1,024 environments.
  - Task 7 RED: three expected collection errors for missing evaluation, checkpoint, and video modules.
  - Task 7 GREEN: evaluation stopping/counting, identity-bound checkpoint restore, and exact-frame MP4 rendering all passed.
  - Task 8 RED: expected collection error because the independent PPO module was absent.
  - Added a RED/GREEN evaluation contract for compiled reset and step functions.
  - Corrected two test-only Brax API assumptions after inspecting the installed feed-forward network interface.
  - Task 8 contract GREEN: asymmetric Actor/critic keys, exact accounting, checkpoint restore requirement, and CLI refusal boundaries passed.
  - First PPO attempt closed at zero transitions after detecting training-wrapper `info` fields were not preserved.
  - Added a failing wrapper-field regression test, preserved outer wrapper fields in environment steps, and passed the full GPU group.
  - Second PPO attempt closed at zero transitions after detecting Brax's explicit `time_out` bootstrap field was absent.
  - Added a failing timeout-adapter test plus a real wrapped-environment contract test; both passed.
  - Third launch completed the unchanged one-block budget: 25,600 training transitions, restored checkpoint, and 31 separately counted diagnostic transitions.
  - Verified 32 captured states equal 32 encoded video frames and validated every retained input/checkpoint identity hash.
  - Task 9 provenance/preflight RED exposed the intentionally absent run-verification entrypoint.
  - Task 9 contract GREEN: closed-ledger verification, drift rejection, CLI, and no-training preflight contracts passed.
  - Complete JIT preflight GREEN: 66 non-GPU and 5 GPU tests passed with closed run verification.
  - Root preflight first exposed missing JIT source discovery; added JIT-only test source-path setup.
  - Root preflight second exposed duplicate test basenames; added explicit JIT/test package boundaries and collected 1,031 tests successfully.
  - Root full suite then reported one fixed JIT subprocess-path issue and one pre-existing user dirty-path signature failure.
  - Fixed the JIT subprocess environment and verified it independently.
  - Fresh repository suite excluding only the exact user dirty-path failure passed all remaining 1,030 tests.
  - Explicitly staged `JIT/`; index audit found no run output, checkpoint, video, cache, log, or outside-JIT path.
  - Cleared file-ending whitespace reported by `git diff --cached --check`; the staged whitespace audit then passed.
  - Created the single focused JIT-only commit without staging generated run evidence or user-owned outside-JIT changes.
- Files created/modified:
  - Task 1 test files will be written before their production modules.

## Test Results

| Test | Input | Expected | Actual | Status |
|---|---|---|---|---|
| GPU visibility | JAX backend check outside device isolation | `gpu`, one CUDA device | `gpu`, `CudaDevice(id=0)` | pass |
| Task 1 RED | Two focused test files | Fail because production package is missing | 2 expected collection errors | pass |
| Task 1 GREEN | `pytest JIT/tests/test_contracts.py JIT/tests/test_reference_boundary.py -q` | All pass | 6 passed in 0.02 s | pass |
| Task 2 RED | Model and action tests | Fail because modules are missing | 2 expected collection errors | pass |
| Task 2 GREEN | `pytest JIT/tests/test_model.py JIT/tests/test_action_mapping.py -q` | All pass | 7 passed in 2.24 s | pass |
| Task 3 RED | Observation tests | Fail because module is missing | 1 expected collection error | pass |
| Task 3 GREEN | `pytest JIT/tests/test_observation.py -q` | All pass | 6 passed in 2.53 s | pass |
| Task 4 RED | Geometry and semantics tests | Fail because modules/contracts are missing | 2 expected collection errors | pass |
| Task 4 GREEN | `pytest JIT/tests/test_geometry.py JIT/tests/test_semantics.py -q` | All pass | 19 passed in 3.42 s | pass |
| Task 5 RED | Reward tests | Fail because module is missing | 1 expected collection error | pass |
| Task 5 GREEN | `pytest JIT/tests/test_rewards.py -q` | All pass | 5 passed in 1.40 s | pass |
| Hip mapping RED | Action mapping tests | Reject missing `hip_initial` contract | 5 expected fixture errors | pass |
| Hip mapping GREEN | `pytest JIT/tests/test_action_mapping.py JIT/tests/test_model.py -q` | All pass | 7 passed in 2.36 s | pass |
| Warp no-contact RED | Focused geometry test | Fail on forbidden `data.contact` access | 1 expected failure | pass |
| Warp no-contact GREEN | `pytest JIT/tests/test_geometry.py -q` | All pass | 5 passed in 3.41 s | pass |
| Task 6 Host | `pytest JIT/tests/test_env_host.py -q` | All pass | 2 passed in 2.81 s | pass |
| Task 6 GPU | `pytest JIT/tests/test_env_gpu.py -m gpu -q` | JIT/reset/50 ticks/1024 finite | 3 passed in 8.81 s | pass |
| Task 7 RED | Evaluation, checkpoint, and video tests | Fail because modules are missing | 3 expected collection errors | pass |
| Task 7 GREEN | `pytest JIT/tests/test_evaluation.py JIT/tests/test_checkpoint.py JIT/tests/test_video.py -q` | All pass with exact state/frame counts | 5 passed in 4.16 s | pass |
| Task 8 RED | PPO and CLI contract tests | Fail because PPO module is missing | 1 expected collection error | pass |
| Compiled capture RED | Focused evaluation test | Reject missing `reset_fn` support | 1 expected failure | pass |
| Task 8 contract GREEN | Evaluation, PPO, and CLI tests | All contract boundaries pass | 12 passed in 11.85 s | pass |
| Wrapper field RED | Focused GPU regression | Preserve training wrapper fields | 1 expected failure | pass |
| Wrapper field GREEN | Complete GPU environment group | Wrapper keys retained | 4 passed in 10.37 s | pass |
| Timeout adapter RED | Focused GPU integration | Expose `time_out` from truncation | 1 expected failure | pass |
| Real wrapper GREEN | Complete GPU environment group | All PPO extra fields available | 5 passed in 13.09 s | pass |
| PPO engineering smoke | 1,024 envs, one 25,600-transition block | Compile, update, save, restore | 25,600 training + 31 diagnostic; completed | pass |
| Run verification | Successful `_retry2` run | Hashes/counts/frames closed | 32 captured = 32 encoded; checkpoint restored | pass |
| Task 9 contract GREEN | Provenance and preflight tests | All pass | 5 passed in 0.04 s | pass |
| Final JIT preflight | Static, 66 non-GPU, 5 GPU, reference/run verification | Exit 0 | 66 + 5 passed | pass |
| Root collection | Complete repository | Unique module collection | 1,031 collected | pass |
| Root full suite | Complete repository | Identify all remaining failures | 1,029 passed; 1 JIT failure; 1 user dirty-path failure | diagnostic |
| JIT subprocess fix | Focused provenance CLI test | Pass without ambient PYTHONPATH | 1 passed in 0.04 s | pass |
| Root compatibility | All except exact user dirty-path failure | No JIT regressions | 1,030 passed, 1 deselected in 158.15 s | pass |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|---|---|---:|---|
| 2026-08-24 | `CUDA_ERROR_NO_DEVICE` in sandboxed JAX discovery | 1 | Read-only check outside device isolation confirmed the GPU backend. |
| 2026-08-24 | Unmatched shell backtick in a plan self-review search | 1 | Used a simpler quoted search without repeating the fragile command. |
| 2026-08-24 | Combined plan patch missed one exact context line | 1 | Reapplied as a smaller exact-context patch. |
| 2026-08-24 | `model.geom("load").mass` is unavailable | 1 | Traced compiled mass to aggregate `uparm` body mass 2.31 kg; parse the unique XML load geom's explicit 2.0 kg mass after hash verification. |
| 2026-08-24 | Relative-x hand calculation differed from zero by `2.38e-7` | 1 | Corrected the float32 test tolerance to `1e-6`; production math was unchanged. |
| 2026-08-24 | JAX transformed reward dict keys into lexical order | 1 | Introduced an ordered fixed-field reward-component pytree with name lookup and dict export. |
| 2026-08-24 | MJX Warp `Data` raised `AttributeError` for `contact` | 1 | Confirmed the backend contract and selected pure geometry/IMU support estimation for training. |
| 2026-08-24 | Zero hip action targeted 0 instead of the -1.2 keyframe | 1 | Added a failing mapping test, then centered the piecewise action map on XML `hip_initial`. |
| 2026-08-24 | PPO network contract test omitted normalizer parameters and expected a length-one critic output | 1 | Matched the installed Brax three-argument apply API and its scalar unbatched critic result; production code was unchanged. |
| 2026-08-24 | Brax scan input had eight wrapper `info` fields absent from environment step output | 1 | Preserve all incoming `state.info` keys and override only environment-owned values; failed run retained with zero transitions. |
| 2026-08-24 | Brax timeout bootstrapping requested missing `state.info['time_out']` | 1 | Added the explicit float adapter `time_out = truncated` and verified the real training wrapper; failed run retained with zero transitions. |
| 2026-08-24 | Root pytest could not import `jit_dvgc` and then found duplicate test basenames | 1 | Added JIT-only source discovery plus `JIT.tests` package boundaries; root collection reached all 1,031 tests. |
| 2026-08-24 | Provenance CLI subprocess depended on ambient `PYTHONPATH` | 1 | Made the test pass an explicit JIT source environment, matching the documented CLI contract. |
| 2026-08-24 | Full root suite retains `frozen_manifest_payload(..., mode=...)` failure | 1 | Classified it as the user's pre-existing modified paths, preserved those files, and passed all other 1,030 tests. |

## 5-Question Reboot Check

| Question | Answer |
|---|---|
| Where am I? | Phase 5: final staged-content audit before the JIT-only commit. |
| Where am I going? | TDD implementation, runtime/PPO verification, then one JIT-only commit. |
| What is the goal? | Deliver the independent Phase U engineering smoke stack under `JIT/`. |
| What have I learned? | See `findings.md`. |
| What have I done? | Independent implementation, TDD, GPU/PPO smoke, evidence closure, JIT preflight, and repository compatibility checks are complete. |
