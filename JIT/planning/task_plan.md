# Task Plan: JIT Phase U Formal Training

## Goal

Extend the verified independent JIT engineering stack with an auditable formal
998,400-transition Phase U runner, push the validated source to GitHub, and
launch one persistent run without changing physics, reward, or observations.

## Next Step

Commit and push the reviewed formal-training design, then write its TDD implementation plan.

## Current Phase

Phase 1: formal-training design

## Phases

### Phase 0: Engineering-smoke baseline

- [x] Read the rebuild guide, current method boundary, experiment state, and reference CSV.
- [x] Confirm an independent package that does not import `dvgc`.
- [x] Limit delivery to the guide's first engineering stopping point.
- [x] Define architecture, data flow, validation, and claim boundaries.
- **Status:** complete

### Phase 1: Formal-training design

- [x] Inspect the installed Brax callback/checkpoint/restore boundary.
- [x] Select one uninterrupted persistent run with parameter-level warm resume.
- [x] Define aligned blocks, checkpoints, held-out evaluation, accounting, and stopping.
- [x] Self-review the design document; commit and push this focused design round.
- **Status:** complete

### Phase 2: Formal-training implementation plan

- [ ] Write a complete TDD plan with exact files, APIs, tests, and commands.
- [ ] Review the plan for spec coverage, placeholders, and type consistency.
- **Status:** pending

### Phase 3: TDD implementation

- [ ] Implement formal config and schedule validation.
- [ ] Implement trace persistence and formal provenance contracts.
- [ ] Implement formal runner, CLI mode separation, and warm resume.
- **Status:** pending

### Phase 4: Verification and GitHub delivery

- [ ] Run focused, complete JIT, GPU, and repository compatibility checks.
- [ ] Inspect staged JIT-only content and create a focused commit.
- [ ] Push the formal-training source commit before launching training.
- **Status:** pending

### Phase 5: Formal launch and sparse monitoring

- [ ] Predeclare and persistently launch the 998,400-transition run.
- [ ] Verify startup state and transition-0 checkpoint once.
- [ ] Inspect only declared milestones, completion, or abnormal exit.
- **Status:** pending

## Completed Baseline Detail

### Phase 2: Detailed implementation planning

- [x] Inspect exact XML object names, actuator contracts, and MJX environment API.
- [x] Define file responsibilities and public interfaces.
- [x] Write the test-first implementation plan.
- [x] Self-review the plan for coverage, placeholders, and type consistency.
- **Status:** complete

### Phase 3: TDD implementation

- [x] Implement each behavior only after its focused test fails for the expected reason.
- [x] Keep imports independent of the existing `dvgc` package.
- [x] Record every red/green command and result.
- **Status:** complete

### Phase 4: Runtime and PPO verification

- [x] Run pure and Host MuJoCo tests.
- [x] Run GPU JIT/vmap environment smoke at 1,024 environments.
- [x] Predeclare and run one 25,600-transition PPO smoke.
- [x] Verify checkpoint restore, transition accounting, and output provenance.
- **Status:** complete

### Phase 5: Final audit and Git delivery

- [x] Run static compilation, the complete JIT test suite, and JIT-local preflight.
- [x] Verify only `JIT/` was created or changed by this task.
- [x] Inspect metrics and termination causes without claiming learnability.
- [x] Explicitly stage only `JIT/` and inspect the complete index.
- [x] Create one focused JIT-only commit.
- **Status:** complete

## Key Questions

1. Can the installed MJX Warp backend compile the authoritative 2 kg model at
   0.005 s simulation timestep with four substeps per control tick?
2. Does a 1,024-environment one-block PPO smoke fit the available RTX 4090 D
   without changing physics, reward, observation, or PPO layout?
3. Are every success, failure, timeout, transition, and video frame accounted
   for by saved machine-readable evidence?

## Decisions Made

| Decision | Rationale |
|---|---|
| Use `JIT/src/jit_dvgc` as a unique package | Prevent import collision and accidental dependence on legacy production modules. |
| Read the authoritative XML and reference CSV from their retained repository paths | The XML must not be copied; the CSV is a weak offline prior only. |
| Keep all generated source, tests, docs, configs, scripts, plans, and run outputs under `JIT/` | Direct user requirement and clean repository isolation. |
| Use one stable `train_phase_expert.py` entrypoint | Avoid version-suffixed production routes. |
| Implement only Propulsion-Ascent in this delivery | Phase D requires real online Apex snapshots and is outside the approved first delivery. |
| Treat GPU/PPO output as engineering integrity evidence only | A one-block smoke cannot establish learnability or a trained expert. |
| Commit once after final verification | The user requested Git delivery only after checking the complete JIT work. |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Sandboxed JAX reported `CUDA_ERROR_NO_DEVICE` and selected CPU | 1 | Re-ran only the read-only backend check outside device isolation; JAX reported `gpu` and `CudaDevice(id=0)`. |
| A plan self-review `rg` command had an unmatched shell backtick | 1 | Replaced the fragile mixed-pattern command with a simple quoted search; no project data was affected. |
| A combined plan patch used one inexact context line | 1 | Split the edit around exact current lines and applied it successfully. |
| MuJoCo geom named view has no per-geom `mass` after compilation | 1 | Confirmed MuJoCo retains only aggregate body mass; audit the identity-bound XML's unique `load` geom mass before compilation. |
| Hand-computed relative-x test used exact-zero tolerance against JAX float32 | 1 | Kept production geometry unchanged and set the hand-derived assertion tolerance to `1e-6`. |
| JAX reordered ordinary reward-component dict keys under `jit/vmap` | 1 | Replaced the dict pytree with a fixed-field `RewardComponents` struct that preserves the declared metric order. |
| MJX Warp `Data` has no `contact` field | 1 | Use JAX geometry/IMU support and penetration estimates derived from geom transforms; do not access contacts in the training path. |
| Initial hip mapping centered zero action at 0 instead of XML keyframe -1.2 | 1 | Added `hip_initial` to the mapping and made both piecewise branches meet at the retained keyframe target. |
| Brax wrapper fields were dropped by `env.step` | 1 | Preserve incoming `state.info` and verify the real wrapper contract under GPU JIT. |
| Brax timeout bootstrapping required `time_out` | 1 | Bind `time_out` exactly to `truncated` and cover it with a regression test. |
| Root pytest found duplicate test module basenames | 1 | Added JIT package boundaries so all 1,031 root tests collect with unique module names. |
| Root preflight has one user dirty-path signature failure | 1 | Preserve user changes; verify all other 1,030 tests with only that exact case deselected. |

## Notes

- Existing modified/untracked paths outside `JIT/` belong to the user and must remain untouched.
- Start the 998,400-transition formal Phase U run only after its source passes
  verification and the focused JIT commit is present on GitHub.
- Do not add Phase D, continuation, feasibility, Soft Tube, unified PPO, or JCE/JEL placeholders.
