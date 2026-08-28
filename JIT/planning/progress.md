# Progress Log

## Session: 2026-08-28

### Phase 18: learned Soft Tube and unified Tube-RSI

- **Status:** in_progress
- Actions taken:
  - Read the local branch, Git history, approved handoff, project method,
    nested JIT instructions, relevant source/tests, and artifact manifests.
  - Confirmed user approval for the implementation outline and larger
    milestone commits rather than atomic commits.
  - Selected inline TDD execution within the existing JIT subtree; no extra
    worktree and no subagent delegation.
  - Wrote the design and implementation plan before production changes.
- Files created/modified:
  - `JIT/planning/task_plan.md`
  - `JIT/planning/findings.md`
  - `JIT/planning/progress.md`
  - `JIT/docs/plans/2026-08-28-soft-tube-tube-rsi-design.md`
  - `JIT/docs/plans/2026-08-28-soft-tube-tube-rsi-implementation.md`
- TDD evidence:
  - Soft Tube RED: 4 expected failures because `jit_dvgc.soft_tube` was absent.
  - First GREEN attempt: the hash-drift test passed; three tests exposed a
    synthetic fixture parent-group mismatch before production behavior.
  - Soft Tube GREEN: 5 focused tests passed, including the real CLI path.
  - Regression group GREEN: 23 Soft Tube/value/snapshot tests passed.
  - Tube-RSI RED: 7 expected failures because `jit_dvgc.tube_rsi` was absent.
  - Tube-RSI GREEN: 7 focused deterministic/split/identity tests passed.
  - Unified environment RED: 4 expected failures because the unified env and
    smoke modules were absent.
  - Unified environment GREEN: 4 host/GPU integration tests passed; the real
    CLI restored/stepped both phases with one action path and zero expert use.
  - Combined new-feature group GREEN: 17 tests passed.
  - Predeclared formal Soft Tube construction:
    - purpose: build the first TRAIN-only phase-aware learned Soft Tube;
    - inputs: frozen expert manifest, exact V_up/V_down model artifacts,
      hash-bound nominal/boundary/downstream catalogs, labels, and protocols;
    - interaction cost: 0 environment interactions, 0 training transitions;
    - stopping condition: close manifest/entries/diagnostics or fail closed;
    - output: `JIT/runs/soft_tube/soft_tube_train_v1_20260828`.
  - Formal Soft Tube construction completed with 222 TRAIN entries: 117
    upstream and 105 downstream; zero interactions/transitions and no
    validation/TEST use.
  - Predeclared formal Tube-RSI smoke:
    - purpose: prove unified restore and one-step execution from both phases;
    - inputs: completed Soft Tube manifest plus existing Phase U/Phase D smoke
      configs and the authoritative XML;
    - interaction cost: exactly 16 environment interactions, 0 training;
    - stopping condition: 8 fixed upstream and 8 fixed downstream entries each
      restore and step once with finite state and no expert switching;
    - output: `JIT/runs/tube_rsi/tube_rsi_smoke_train_v1_20260828`.
  - Formal Tube-RSI smoke completed `GO`: 16/16 finite restore/step records,
    split 8 upstream and 8 downstream, with zero expert switching and zero
    training, validation, or TEST use.
  - Artifact audit re-read the closed manifests/reports and their files. Soft
    Tube identity is `c1c1161e...df28b`; the smoke binds that exact identity.
  - Pre-unified local preflight exited zero: 351 non-GPU and 14 GPU tests
    passed. This was before adding the unified PPO runner, so final preflight
    must be repeated before launch.
  - Unified PPO RED: four focused failures because
    `jit_dvgc.unified_training` was absent.
  - Unified PPO GREEN: four focused tests passed for the checked-in exact
    25,600 contract, GO gates, fresh single-policy trainer arguments, saved
    checkpoint, and strict restore.
  - Unified preflight RED/GREEN: the new static-contract test first failed on
    the missing preflight integration, then the focused unified/preflight
    group passed 9 tests after adding non-launching config checks.
  - Final updated local preflight exited zero: 356 non-GPU and 14 GPU tests
    passed after static compilation and the no-legacy-import/config gates.
  - Predeclared unified pilot launch contract:
    - purpose: compile/update/save/restore the first fresh single-Actor
      Tube-RSI PPO while making only an engineering-integrity claim;
    - inputs: `pi_unified_pilot.json`, exact upstream/downstream configs, Soft
      Tube manifest `c1c1161e...df28b`, Tube-RSI smoke report
      `34f35b49...a0a2796`, and authoritative XML;
    - interaction cost: exactly 25,600 training transitions; zero Brax/fixed
      evaluation and zero diagnostic interactions;
    - stopping conditions: one exact block, any nonfinite metric, or checkpoint
      restore failure;
    - output: `JIT/runs/pi_unified/pi_unified_pilot_25600_seed821001_20260828`.
  - The first pilot completed 25,600 training transitions and restored its
    checkpoint, but the launch log contained 1,385 CCD-capacity warnings with
    a maximum request of 572 against inherited `naccdmax=512`. It is retained
    as an engineering diagnosis and is not the capacity-clean GO artifact.
  - Systematic root-cause evidence showed the warning came from aggregate
    runtime capacity under the new mixed 1,024-environment batch. No frozen
    model, physics, geometry, reward, snapshot, or action contract changed.
  - CCD-capacity RED: the checked pilot had no runtime capacity field and the
    unified environment rejected the proposed keyword. GREEN: 15 focused
    unified/smoke/preflight tests passed with pilot-local `naccdmax=1024`.
  - Post-fix local preflight exited zero: 356 non-GPU and 14 GPU tests passed.
  - Predeclared replacement pilot:
    - purpose/interaction/stopping contracts are identical to the first pilot;
    - only runtime aggregate CCD capacity changes from inherited 512 to the
      unified pilot's explicit 1024;
    - config hash: `16b12ca1...cca47`;
    - output: `JIT/runs/pi_unified/pi_unified_pilot_25600_seed821001_ccd1024_20260828`.
  - Replacement pilot completed exactly 25,600 training transitions with zero
    evaluation/diagnostic interactions, one fresh Actor, no expert switching,
    successful final-checkpoint restore, and zero CCD overflow warnings.
  - Brax returned an empty `final_metrics` mapping because evaluations were
    disabled, while its one training callback contained the complete
    transition-25,600 metric set and every numeric value was finite. The runner
    now uses that final training callback when the return mapping is empty and
    fails closed if neither source contains metrics; focused RED/GREEN passed.
  - Final artifact audit independently restored the transition-25,600
    checkpoint: 25 array leaves, 313,134 parameters, all finite. The single
    training callback contains 104 finite metrics at exactly transition
    25,600; all evaluation/diagnostic counts remain zero.
  - Final local preflight after every source/documentation change exited zero:
    356 non-GPU and 14 GPU tests passed. `git diff --check -- JIT` also exited
    zero.
  - Final Git delivery uses one explicit JIT-only milestone commit. The
    user-owned `JIT/jit_continuation_labels_phase1.patch` and all dirty paths
    outside JIT remain unstaged.
- **Status:** complete

## Session: 2026-08-25 v4 CCD/10M/automatic analysis

### Final Important review corrections

- **Status:** implementation and focused verification complete; full
  revalidation, Git gate, and smoke remain pending.
- RED: six completed-v4 adversarial provenance cases failed as intended. Five
  mutated fields/artifacts were accepted outright; the nonzero start reached
  only a generic segment-accounting failure rather than the required v4
  fresh-start gate.
- Completed-v4 verification now requires start 0, null parent, `fresh` resume
  semantics, segment seed equal to the resolved PPO seed, no
  `--restore-checkpoint` in the manifest command, and an exactly matching
  `resume_command.txt`. Retained v1-v3 verification remains schema-separated.
- GREEN: the focused formal/provenance suite reported 39 passed. Independent
  strict verification also passed for the retained completed v1, v2, and v3
  formal run directories.
- The design, task plan, implementation plan, and this progress ledger now all
  require the validated commit and confirmed push before any PPO interaction,
  including the bounded smoke. The formal launch command retains the required
  JAX/MuJoCo/unbuffered environment and explicitly sets `PYTHONPATH=JIT/src`.
- No PPO interaction or GPU smoke was run during these corrections.

### Design and implementation planning

- **Status:** complete
- Confirmed existing user-owned dirty paths outside JIT and left them untouched.
- Wrote and user-approved
  `JIT/docs/specs/2026-08-25-jit-v4-10m-ccd-auto-analysis-design.md`.
- Committed the pre-implementation design baseline as `e383f54` before source
  changes, following the user's new two-gate Git rule.
- Selected direct v4 replacement, practical `naccdmax=256` handling, window
  `[2.5,3.4]`, height coefficient 40, RSI 8%, and 9,977,856 transitions.
- Wrote the complete RED/GREEN implementation and launch plan at
  `JIT/docs/plans/2026-08-25-jit-v4-10m-ccd-auto-analysis.md`.
- No PPO interaction or GPU smoke has been run in this session yet.

### Task 5: formal evidence and preflight integration

- **Status:** complete
- RED: the focused formal/provenance command reported 1 failed and 21 passed;
  the 10M config supplied 9,977,856 transitions while the stale v4 test double
  still required 4,988,928.
- GREEN: the same command reported 22 passed after the v4-only runner fixture
  adopted 406 blocks and the approved six-checkpoint schedule.
- Final focused verification reported 23 passed after separating the completed-
  evidence method-drift proof into its own behavioral test.
- Completed-v4 evidence now proves rejection of consistently rehashed raw
  configs carrying any previous `naccdmax`, RSI, window, or height value.
- Local preflight loads `phase_u_continuation_10m.json` and checks the exact
  target, block count, schedules, seed, and held-out namespace without training.
- Retained v3 absolute-5M fixtures and historical documentation remain intact.
- `bash JIT/scripts/local_preflight.sh` exited zero: 188 non-GPU tests and 12
  GPU tests passed, followed by retained reference and smoke provenance checks.
- No PPO training, evaluation, diagnostic, or GPU-smoke transitions were run.

### Tasks 1-4: active method and watcher implementation

- **Status:** complete
- RED/GREEN tests replaced only v4 with window `[2.5,3.4]`, height coefficient
  40, airborne RSI 8%, explicit Warp `naccdmax=256`, fresh-only formal
  semantics, and the exact 406-block training/checkpoint schedule.
- The real GPU runtime reported aggregate CCD capacity 48 before the binding
  fix and 256 afterward. v1-v3 keep their prior implicit runtime behavior.
- A deterministic 4,096-reset GPU batch validated the 8% mixture and proved
  every airborne-RSI sample starts with jump signal one.
- The run-local watcher waits without model use, invokes at most one ephemeral
  read-only `codex exec` after terminal status, and records launch, timeout, and
  completion failures without retrying a paid call.
- Review added a finite one-hour Codex timeout and closed a fresh-only gap that
  previously allowed an explicit v4 restore argument.

### Task 6: complete source verification

- **Status:** final source verification complete; Git gate and smoke pending
- Static compilation passed.
- Complete JIT non-GPU suite: 200 passed, 12 deselected.
- Complete JIT GPU suite: 12 passed, 200 deselected.
- `bash JIT/scripts/local_preflight.sh` exited zero and repeated 200 non-GPU +
  12 GPU passes plus retained reference/provenance checks.
- Active resolved hashes: smoke
  `711ac0709c819d6b2ca4172ec1d1421a21fd469351d8661f5b17b6d2594cc96f`;
  formal `a77c539c84318c48fa1c1f5ea0f87b468295fec633338e7a95a5b6eff5360e2f`.
- No PPO interaction has been started; the user-required commit/push gate still
  precedes the bounded smoke.
- Independent final review approved the tree with no remaining Critical or
  Important finding after fresh-start provenance and launch-path corrections.

## Session: 2026-08-25 Reward/RSI/Diagnostics Rebuild

### Phase 7: Design

- **Status:** complete
- Audited the external planar-jump reward without executing or importing it.
- Mapped target-free reference formulas to Phase U and removed platform/deceleration scope.
- Resolved one-shot jump signal reset/exit behavior, Actor/critic sharing, and no history duplication.
- Removed approximate wheel-support observations and replaced complex Apex with root height/ascent/descent.
- Added 5% bounded airborne RSI and natural-only formal evaluation.
- Wrote and self-reviewed `JIT/docs/specs/2026-08-25-jit-phase-u-reference-reward-rsi-diagnostics-design.md`.
- No environment interactions, PPO transitions, commits, or outside-JIT changes occurred.

### Phase 8: Implementation planning

- **Status:** complete
- Mapped v2 config, pure JAX events/reward, mixed/natural reset paths, 76/106 observations, checkpoint identity, diagnostic artifacts, and schema-aware provenance.
- Wrote `JIT/docs/plans/2026-08-25-jit-phase-u-reference-reward-rsi-diagnostics.md` with explicit RED/GREEN commands.
- Preserved the user's single-final-commit preference; no implementation commit has been made.

### Phase 9: TDD implementation

- **Status:** complete
- Task 1 first test-patch attempt was rejected atomically because one
  `apply_patch` payload targeted the same test paths with both Delete and Add
  operations. No test or production file changed. The retry uses separate
  delete and add operations.
- Task 1 RED confirmed: semantics collection failed on missing `PhaseUSignals`;
  all 24 reference-reward cases failed on the old `RewardState`/formula.
- Independent arithmetic review corrected one test fixture: success components
  sum to `86 -> clipped 50`, while the selected failure fixture sums to
  `-54 -> clipped -50`.
- Task 1 GREEN: v2 config, one-shot signal/Apex semantics, fixed reward
  components, reference piecewise values, energy/action costs, unclipped sum,
  and total clipping passed 49 focused tests in 1.75 seconds.
- Task 2 RED confirmed old observation signatures/shapes, missing natural reset,
  and missing checkpoint task identity. Host GREEN passed 12 tests; GPU GREEN
  passed 7 tests, including 1,024 reproducible mixed resets and bounded RSI.
- Formal integration initially had 20 passes and 2 expected failures because
  provenance accepted only formal v1. This was classified as the planned
  schema-compatibility task rather than an environment failure.
- Task 3 RED confirmed the diagnostics module/report fields were missing.
  GREEN passed 6 evaluation/PNG/NPZ/video tests with exact state/frame counts.
- Task 4 added v2 artifact path/hash enforcement while retaining v1 verification.
  The formal/provenance group passed 27 tests; the broader focused group passed
  33 tests after diagnostic integration.
- Removed residual active names `target_height`, `platform_back_margin`, and
  the Phase U platform-overrun end reason to eliminate target/platform ambiguity.

### Phase 10: Verification and delivery

- **Status:** in progress
- Final complete non-GPU suite after review fixes: 134 passed, 7 deselected in
  24.75 s.
- Final complete GPU suite: 7 passed in 16.66 s.
- JIT local preflight: exit 0; repeated 134 non-GPU and 7 GPU tests, static
  compilation, legacy-import scan, reference analysis, and retained v1 smoke
  verification.
- Retained formal v1 run verified independently at 998,400 training plus 904
  fixed evaluation transitions with all historical checkpoint/trace hashes.
- Wrote the complete v2 modification and engineering-analysis report under
  `JIT/docs/experiments/phase_u_reward_rsi_diagnostics_v2_20260825/REPORT.md`.
- No PPO training or evaluation transitions were launched in this session.
- Independent review found two formal-label gaps in sequence: approved
  reset/event/reward/PPO constants, then model runtime-capacity metadata, were
  not exact-validated. Added mutation tests and made active v2 loading plus
  completed-v2 provenance enforce the complete model/action/reset/event/
  limits/reward/PPO contract. Final independent re-review found no remaining
  Critical or Important issue.
- Complete repository compatibility after all review fixes: 1,100 passed, the
  exact pre-existing user dirty-path test deselected, one third-party warning,
  in 171.66 s.
- User authorized a fresh approximately one-million-step run after successful
  source verification. The declared exact budget is 998,400 with no v1 restore.

### Phase 11: Fresh v2 formal training

- **Status:** complete
- Created and pushed source commit
  `c55a5d0f7236ddc1217ac84743b149634f7629bf`; remote ref equality passed before
  interaction.
- Fresh run `phase_u_v2_formal_998400_seed820101_20260825` started as PID 255182
  on RTX 4090 D with transition-0 identity and no parent checkpoint.
- Completed 998,400 training and 1,160 fixed natural-evaluation transitions;
  total environment interactions 999,560.
- Strict provenance verification passed all six checkpoint hashes, five panels,
  40 trace hashes/counts, final restore, and MP4/PNG/NPZ hashes/counts.

### Phase 12: v2 formal analysis and handoff

- **Status:** complete
- Every natural panel had 0 Apex, 0 height, 0 ascent, and 8/8 roll-limit
  failures. Final policy failed before the jump zone after 22 ticks.
- 742,400 was least-bad by return/length but had no successful task event and
  is not an expert candidate. Final policy had 13.64% action saturation.
- Final MP4 decoded as `(23,360,1280,3)`; NPZ had 23 aligned samples, 77 numeric
  series, and all finite values. Diagnostic PNG is 2240x2240.
- Wrote complete optimization, reward, action, artifact, limitation, and next
  step analysis into the v2 experiment report. Decision is NO_PROMOTION.

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

### Phase 2: Formal-training implementation plan

- **Status:** complete
- Actions taken:
  - Wrote `JIT/docs/plans/2026-08-24-jit-phase-u-formal-training.md`.
  - Mapped formal config, provenance, trace artifacts, callback orchestration,
    CLI, verification, GitHub delivery, persistence, and sparse monitoring.
  - Checked spec coverage, placeholder patterns, type names, absolute/segment
    transition semantics, and the source-push-before-interaction dependency.
- No environment interactions or training transitions were consumed in this phase.

### Phase 3: TDD implementation

- **Status:** complete
- Actions taken:
  - Added strict formal schema, exact 39-block budget, six checkpoints, five
    fixed evaluation panels, frozen seeds, and smoke/formal separation.
  - Added one-way running provenance, truthful optimizer-reset warm-start
    metadata, and complete per-seed NPZ/JSON trace persistence.
  - Added the formal callback controller, real Brax runner, absolute transition
    offsets, fixed evaluation/video, final checkpoint restore/inference, and
    mutually exclusive CLI modes.
  - Added strict formal verification for checkpoint payload hashes, panel seed
    sets, trace hashes/counts, final video state/frame counts, and report ledger.
  - Used injected trainer/environment/panel dependencies to execute all 39
    orchestration callbacks without consuming real formal environment transitions.
  - Self-review found and fixed a rejection gap: aligned alternate milestone
    tuples are now rejected in favor of the exact approved schedule.
- No formal environment interactions or training transitions were consumed.

### Phase 4: Verification and GitHub delivery

- **Status:** complete
- Results:
  - Complete JIT preflight: 106 non-GPU tests and 5 GPU tests passed; retained
    reference analysis and closed smoke verification passed.
  - Fresh repository compatibility with only the exact user-dirty relative-x
    manifest case deselected: 1,070 passed, 1 deselected in 167.06 seconds.
  - The sole remaining failure is the user's uncommitted relative-x diagnostic
    test calling `frozen_manifest_payload(..., mode=...)` while its user-modified
    production function does not yet accept `mode`; this is outside JIT and was
    reproduced independently.
  - No JIT test or formal-runner compatibility failure remains.
  - Explicitly staged only 17 JIT source/config/test/document paths; the index
    contained no run, checkpoint, video, log, cache, or outside-JIT path.
  - Created implementation commit `30573f82a2ea20ce89943ccb0e3d5f952fefa9a5`.
  - Pushed that exact commit to `origin/agent/two-phase-soft-tube` and verified
    the remote ref matches. The first push attempt ended during TLS handshake;
    a read-only remote check proved no partial update, then an ordinary
    non-force retry succeeded.
- No formal environment interactions or training transitions were consumed.

### Phase 5: Formal launch and sparse monitoring

- **Status:** complete
- First launch:
  - GitHub source gate passed at remote commit `fd534b9a39fa7af3d19ccd428492fb7a02a7fdd0`.
  - Started fresh run `phase_u_formal_998400_seed820101_20260824` as PID 1491714.
  - Startup inspection found an `engineering_error` after the first compiled
    25,600-transition block: Brax's in-epoch `EpisodeMetricsLogger` called the
    ordered block `progress_fn` before `policy_params_fn`.
  - The original callback ledger recorded zero because the exception preceded
    the policy callback. Source inspection proves one exact block was consumed;
    `accounting_correction.json` records the correction to 25,600 transitions.
  - Only transition-0 parameters were checkpointed, so the failed run is not a
    valid warm-start parent and remains preserved under its original run id.
- Fix:
  - RED reproduced by requiring the injected formal trainer contract to set
    `log_training_metrics=False`; the current runner passed `True`.
  - GREEN after disabling only the in-epoch episode logger. Once-per-block PPO
    loss/KL/SPS progress and milestone fixed evaluation remain enabled.
  - The replacement run id is
    `phase_u_formal_998400_seed820101_20260824_retry1` and must start fresh only
    after the fix passes verification, commit, and GitHub push.
  - Fix commit `fd3372c1e1a7ffc6099ae366e8302e7552474773` passed JIT preflight,
    was pushed to GitHub, and was the remote HEAD before the replacement launch.
- Successful replacement:
  - PID 1509891 started as a new session with `status=running`, GPU backend,
    transition-0 identity, and no startup anomaly.
  - Completed exactly 998,400 training transitions plus 904 fixed evaluation
    transitions; Brax evaluation and diagnostic counts remained zero.
  - Strict provenance verification closed all six checkpoint hashes, five panels,
    40 traces, final checkpoint restore, and final video state/frame accounting.
  - Every panel had zero Apex success and eight roll-limit physical failures;
    the policy is not promoted or frozen as an expert.

### Phase 6: Experiment analysis and handoff

- **Status:** complete
- Actions taken:
  - Analyzed all 39 PPO block metrics and five fixed evaluation summaries.
  - Inspected per-seed returns, lengths, terminal codes, reward components,
    action saturation, and the final tick-by-tick action/attitude trace.
  - Identified deterministic-reset seed redundancy: `env.reset(rng)` currently
    does not consume `rng`, so eight labels do not create eight independent conditions.
  - Wrote the full Chinese experiment and modification report at
    `JIT/docs/experiments/phase_u_formal_998400_seed820101_20260824/REPORT.md`.
  - Recommended no further PPO until frozen action intervention, gradient,
    normalizer, reward-clamp, and reset-diversity audits are designed.

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
| 2026-08-28 | First unified GPU GREEN attempt rejected intentional Phase U/Phase D yaw, contact, and horizon differences | 1 | Validate only shared roll/pitch/backward limits and track a separate phase-local episode step that resets at Apex. |
| 2026-08-28 | Soft Tube fixture used different parent-group strings in catalog and label rows | 1 | Corrected only the synthetic catalog identities; production validation remains strict. |
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
# 2026-08-25 Absolute Joint Target and 5M Continuation

- User approved hip-style absolute mapping for knee and a fresh approximately
  five-million-step run with no old checkpoint initialization.
- Inspected current action mapping, v2 exact config validator, model keyframe
  construction, checkpoint identity, formal controller, provenance verifier,
  CLI, tests, and working-tree state.
- Confirmed unrelated dirty paths remain outside JIT and must be preserved.
- Wrote and self-reviewed the JIT-local design and TDD implementation plan.
- Selected aligned target 4,988,928, block 24,576, and exact translated PPO
  batch layout 384/64/16/24/8.
- Plan self-review removed the proposed read-only evaluation of old
  checkpoints. The new workflow does not open them at all; RSI diagnostics use
  only current-run milestone parameters.
- The first fresh v3 smoke completed 24,576 transitions and restored its new
  checkpoint, but fixed-rate KL was 341.1 and the natural diagnostic ended on
  roll limit after 43 steps.
- Source tracing identified Brax's pre-SGD normalizer update as a confounder in
  that first KL. Two adaptive-KL comparisons (8 and 1 data passes) instead
  delayed cold-start normalizer warm-up and exploded policy outputs/KL. Removed
  the rejected experiment and retained the bounded fixed-rate smoke profile.
- Independent review found and closed three audit gaps: preserve truthful v2
  incremental runtime semantics, validate checkpoint sidecar identity before
  pickle loading, and prove natural/RSI reset source directly from every v3
  NPZ trace. Fresh local preflight: 151 non-GPU + 8 GPU tests, exit 0.
- Final review then found two artifact-integrity bypasses and one representative
  lineage gap. The verifier now checks every episode/diagnostic array length,
  real MP4/PNG decoding, distinct typed paths, and exact seed/episode/reset
  binding. Final re-review found no remaining Critical or Important issue.
- Created source commit `96cfd81c5bc2f462c6718e0ccd4313e0e62b40bd`,
  pushed it, and verified the remote branch matched before training.
- Launched fresh run `phase_u_absolute_4988928_seed820201_20260825` with no
  restore argument, null parent checkpoint, transition-zero start, and seed
  820201.
- The run completed 4,988,928 training, 192 fixed natural-evaluation, and 88
  forced-RSI diagnostic transitions. Strict completed-run provenance passed.
- All five natural panels had 0 Apex, 0 jump-window reach, and 8/8 physical
  failures. Final episodes terminated on illegal wheel contact after two ticks.
- All five RSI panels had 8/8 Apex and no physical failure, but diagnostics show
  this is an easy reset-provided height/upward-velocity subproblem with highly
  energetic terminal control, not natural jump success.
- Inspected both final 2240x2240 diagnostics and synchronized NPZ traces. The
  natural final action jumps to approximately `[1,-1,1,-1]`, yields joint-energy
  reward -47.56, illegal/physical penalties -30 each, and clips -103.607 to -50.
- Wrote the complete modification, PPO, milestone, reward, artifact, limitation,
  and next-step report at
  `JIT/docs/experiments/phase_u_absolute_4988928_seed820201_20260825/REPORT.md`.
- Final decision: `NO_PROMOTION`; do not resume this checkpoint or enlarge the
  budget before natural-start action intervention and retention diagnostics.
# 2026-08-25 v4 Apex-continuation implementation

- Confirmed Git was synchronized and JIT clean at
  `27923d3eeb34273a5fe299b4a70df06bb11cb88d` before modification.
- Added wheel-support regression coverage using the exact previous two-frame
  failure state; geometry/semantics/reward/GPU tests are green.
- Made Apex nonterminal and changed panel success accounting to event-based
  Apex accounting; full rollouts now continue to failure or 200-tick horizon.
- Added full/pre/post-Apex NPZ generation, dashboard/video Apex markers, hashes,
  transition accounting, and v4 provenance binding.
- Added block-aligned episode/PPO JSONL, PNG/NPZ/JSON training curves, raw-series
  binding, exact v4 configs, fresh seed namespace, and mutation tests.
- Pre-launch verification marker: 165 non-GPU tests and 9 GPU tests passed; retained
  completed v1, v2, and v3 formal provenance all passed. Final preflight and
  staged review remain before commit/push/launch.
- First v4 launch audit observed `episode/length=1` after the initial horizon.
  Stopped PID 1475264, closed the ignored run as aborted at the last certified
  4,325,376-transition callback, and prohibited all of its checkpoints.
- Added a single full-reset training wrapper for smoke/formal PPO plus exact-v4
  config identity and GPU reset-cycle regression. A new source commit/push and
  a new fresh run ID are required before retrying.
- Corrected-tree verification: 166 non-GPU and 10 GPU tests passed; the exact
  v4 formal config hash is
  `6129456320c3acb8c36ba0fe2d96a93ec6c553dfce3ffe4fcabb46559dbb666c`.
- `_retry1` confirmed fresh JIT state but showed that terminal episode evidence
  was replaced by reset zeros, leaving no episode JSONL. Stopped it at the last
  certified 1,769,472-step callback and closed it as aborted.
- Added preserve/expose wrappers for only `episode_done/episode_metrics` and
  bound `preserve_episode_evidence=true`; the new formal config hash is
  `d6b9476fc3097b8a1e9f7c1ca889f3bf2b93c9527210dcedda9e889e95eb0f43`.
- Final corrected-tree verification: 167 non-GPU and 10 GPU tests passed.
