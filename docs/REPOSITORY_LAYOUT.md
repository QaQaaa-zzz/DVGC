# Repository Cleanup Design and Dependency Layout

Date: 2026-08-03

Approved baseline: `main@b7bb815`

Working branch: `agent/repo-cleanup-two-phase`

Archive authority: `archive/pre-clean-20260731` (immutable)

## 1. Scope and stop point

This document is the single dependency inventory and deletion-decision ledger
for the dependency-closure cleanup. The approved strategy is conservative:
start from stable, runnable roots; follow imports and non-Python references;
delete only closed-route files outside the retained closure; and use `defer`
whenever a shared boundary is unclear.

This design commit does not delete code, alter tests, implement the two-phase
method, modify PPO/environment/reward/snapshot semantics, or run training. The
next implementation phase requires a separate review of this document.

## 2. Research truth boundary

The approved research direction is:

```text
guideline
  -> Propulsion-Ascent expert
  -> Descent-Recovery expert
  -> expert snapshots and continuation labels
  -> V_up and V_down feasibility models
  -> learned soft feasibility tubes
  -> unified Tube-RSI PPO
  -> frozen-policy empirical Jump Capability Envelope
```

The repository currently implements reusable environment, observation/history,
snapshot, bank, PPO, policy-bundle, rollout, provenance, certification, and
legacy stage infrastructure. It does **not** yet implement the approved
two-phase expert semantics, `V_up`/`V_down` training, learned soft-tube
construction, two-phase unified Tube-RSI PPO, or a new two-phase pipeline CLI.
No placeholder entrypoints will be created during cleanup.

The five-stage controllers and branch-funnel launchers are a legacy
implementation retained temporarily for dependency-safe migration. They are
not the current research mainline.

## 3. Dependency-closure method

The audit used the following edge types:

1. Static Python `import` and `from ... import ...` edges across tracked files.
2. Python files named by `cli.runtime_gate.source_fingerprint`, even when they
   are not imported during module initialization.
3. Shell and systemd `python -m cli.*` calls and shell-to-shell calls.
4. Config, XML, reference-data, test, and documentation path references.
5. Targeted searches for dynamic imports and source-path contract tests.

The lower-bound AST closure was 35 Python files. Adding runtime-fingerprint
dependencies and their imports raised it to 40 Python files. The scan found
dynamic `dvgc.env` imports in two legacy CLIs; this is one reason closure-external
`dvgc/` modules default to `defer`, not `delete`.

Dependency direction matters: a closed legacy launcher calling a retained CLI
does not make the launcher a retained root. Conversely, a retained root importing
a module, hashing its source, or requiring its data makes that dependency part
of the retained closure.

## 4. Retained roots

| path | role | retained_root | reverse_dependencies | decision | reason | validation_required |
|---|---|---|---|---|---|---|
| `scripts/local_preflight.sh` | configured-runtime preflight | yes | README, verification protocol | keep | required stable environment entry | shell syntax, `prepare_project`, full pytest |
| `cli/runtime_gate.py` | model/reset/step/snapshot/PPO-resume gate | yes | preflight contracts and training guards | keep | required runtime validation | source fingerprint and bounded gate |
| `cli/prepare_project.py` | XML/reference/config audit | yes | local preflight and README | keep | required minimum preflight | targeted project-preparation tests |
| `cli/build_candidates.py` | generic candidate construction | yes | current README, generic train flow | keep | runnable reusable capability | candidate/source-contract tests |
| `cli/train.py` | generic PPO training/resume | yes | current README and runtime fingerprint | keep | PPO update and resume capability | runtime gate and training tests |
| `cli/certify.py` | generic frozen-policy certification | yes | current README | keep | reusable evaluation infrastructure | certification tests |
| `cli/audit.py` | generic independent audit | yes | current README | keep | reusable audit infrastructure | audit tests |
| `cli/evaluate.py` | generic/full evaluation | yes | current README | keep | final end-to-end evaluation capability | evaluation tests |
| `tests/` via `pytest -q` | repository contract entry | yes | local preflight | keep | formal test runner; individual route tests may later split | full pytest |
| `pyproject.toml` | package registration | yes | build backend | keep | registers `dvgc` and `cli`; no console scripts are registered | package/import checks |

The tracked systemd watchdog is **not** a retained two-phase root. It points to
legacy `cli.pipeline_watchdog` and is deferred because external user-service
state cannot be inferred from static imports alone.

### Test root rule

`pytest -q` is a verification entrypoint, but legacy route tests are not
retained dependency roots. Only stable shared-contract tests retain production
modules. A legacy test importing a closed-route controller does not by itself
promote that controller into the retained closure. Shared assertions must be
extracted before route-only tests are removed.

## 5. Retained Python closure

The following 40 files are retained by a root, a runtime fingerprint, or a
transitive import. Package markers `cli/__init__.py` and `dvgc/__init__.py` are
also retained.

```text
cli/audit.py
cli/build_candidates.py
cli/certify.py
cli/certify_descent_entries.py
cli/evaluate.py
cli/evaluate_composite.py
cli/prepare_project.py
cli/runtime_gate.py
cli/train.py
cli/train_descent_local_block.py
cli/train_expert.py
dvgc/action_mapping.py
dvgc/audit.py
dvgc/audit_manifest.py
dvgc/bank.py
dvgc/bounded.py
dvgc/candidate_geometry.py
dvgc/certification.py
dvgc/composite.py
dvgc/config.py
dvgc/curriculum.py
dvgc/descent_entry.py
dvgc/descent_local.py
dvgc/env.py
dvgc/expert_training.py
dvgc/experts.py
dvgc/model.py
dvgc/policy.py
dvgc/ppo_integrity.py
dvgc/reference.py
dvgc/reference_joints.py
dvgc/research_semantics.py
dvgc/rewards.py
dvgc/rollout.py
dvgc/runtime.py
dvgc/seed_registry.py
dvgc/signals.py
dvgc/snapshot_provenance.py
dvgc/snapshot_timing.py
dvgc/wrappers.py
```

The retained non-Python closure includes `configs/default.json`,
`assets/orange_bike_4kg_horizontal.xml`, every mesh referenced by that XML,
`data/reference_jump.csv`, `requirements.txt`, and `pyproject.toml`. The model,
mesh geometry, payload, actuator limits, action mapping, and matcher contracts
are immutable in this cleanup.

## 6. First-round deletion candidates

`delete` below is a reviewed design decision, not an instruction to remove the
file in this commit. Each row must satisfy its validation prerequisite during
the later deletion phase. Reverse dependencies shown here are limited to the
relevant closed-route cluster; the implementation pass must repeat the search
immediately before deletion.

Files with no references and no reusable helpers may enter deletion batch 1.
Shared orchestration bases such as `cli/descent_local_controller.py` and
`cli/descent_tube_controller.py` may be deleted only after shard, seed,
OOM-backoff, failure-fuse, and certification contracts have been moved into
stable modules and tests.

| path | role | retained_root | reverse_dependencies | decision | reason | validation_required |
|---|---|---|---|---|---|---|
| `cli/activate_descent_envelope_pipeline.py` | legacy activator | none | none | delete | closed Descent-envelope route; outside closure | repeat refs; compile; controller tests migrated |
| `cli/activate_fast_handoff_route.py` | legacy activator | none | none | delete | closed fast-handoff route; outside closure | repeat refs; targeted tests |
| `cli/activate_jump_envelope_pipeline.py` | legacy activator | none | jump-envelope route test | delete | closed jump-envelope controller activation | remove/split route test; compile |
| `cli/activate_trajectory_mining_pipeline.py` | legacy activator | none | none | delete | closed trajectory-mining route | repeat refs; targeted tests |
| `cli/decoupled_bootstrap_controller.py` | persistent legacy controller | none | paired start/run scripts | delete | superseded bootstrap controller | delete route cluster; targeted tests |
| `cli/descent_envelope_controller.py` | persistent legacy controller | none | deferred trajectory controller, paired scripts/tests | defer | still imported by deferred trajectory-mining route | resolve that route before deletion |
| `cli/descent_local_controller.py` | shared base for old controllers | none | active-watchdog Descent-Tube path and old controller cluster | defer | stable helpers extracted, but enabled watchdog transitively imports this base | migrate/disable external service first |
| `cli/descent_tube_controller.py` | persistent legacy controller | none | old controller cluster, tests, active-watchdog fallback | defer | active watchdog can reach its start/run route | migrate/disable external service, then re-audit contracts |
| `cli/jump_envelope_controller.py` | persistent legacy controller | none | stage controller and route test | delete | closed jump-envelope route | delete cluster; targeted tests |
| `cli/stage_next_bootstrap_controller.py` | legacy controller | none | paired start script/test | delete | superseded stage-next bootstrap | targeted route tests |
| `cli/stage_next_v3_controller.py` | versioned legacy controller | none | paired start script/test and dynamic helpers | defer | broad subprocess/test closure is not yet isolated | complete dependency migration first |
| `cli/stage_reachability_controller.py` | legacy controller | none | migration, paired start script/test | delete | superseded sequential reachability route | delete with migration cluster |
| `cli/trajectory_mining_controller.py` | persistent legacy controller | none | paired scripts/tests | defer | its tests still carry reusable resume/bank preparation contracts | migrate those contracts first |
| `cli/migrate_stage_reachability.py` | one-time migration | none | route-migration test | delete | completed migration; Git history preserves it | remove route-only test; compile |
| `cli/normalize_descent_tube_v6.py` | one-time schema normalization | none | normalization test | delete | completed non-overwriting migration | preserve schema invariant test elsewhere if needed |
| `cli/normalize_stage_entry_bank.py` | one-time schema normalization | none | normalization test | delete | completed entry-bank migration | retain shared entry-label contract test |
| `cli/resume_roll_targeted_cycle5.py` | one-time resume entry | none | trajectory-controller test | delete | closed roll-targeted cycle | split route-only assertion |
| `cli/audit_descent_student_relabel_v2.py` | versioned one-off audit | none | none | delete | no reverse reference; closed Descent diagnostic | compile and targeted shared-module tests |
| `cli/run_unified_descent_teacher_cv_v2.py` | versioned one-off run | none | none | delete | no reverse reference; completed diagnostic | compile and teacher-module tests |
| `scripts/run_backward_bootstrap.sh` | old formal launcher | none | old README/summary and source-contract test | delete | superseded sequential shared-Actor route | switch docs; migrate shared source contracts |
| `scripts/run_decoupled_bootstrap_pipeline.sh` | controller runner | none | paired start script | delete | closed decoupled bootstrap route | shell syntax and ref scan |
| `scripts/start_decoupled_bootstrap_controller.sh` | systemd launcher | none | old controller metadata | delete | closed decoupled bootstrap route | external-unit check before implementation |
| `scripts/run_descent_envelope_pipeline.sh` | controller runner | none | deferred trajectory-mining controller | defer | retained transitively by deferred controller route | resolve that route first |
| `scripts/start_descent_envelope_controller.sh` | systemd launcher | none | deferred controller/test closure | defer | kept with unresolved Descent-envelope cluster | resolve cluster first |
| `scripts/run_descent_local_pipeline.sh` | controller runner | none | paired start script | delete | closed local Descent route | shell syntax and ref scan |
| `scripts/start_descent_local_controller.sh` | systemd launcher | none | controller test | delete | closed local Descent route | external-unit check; split test |
| `scripts/run_descent_tube_pipeline.sh` | controller runner | none | paired start script reached by active watchdog fallback | defer | enabled watchdog timer can transitively invoke it | migrate/disable external service first |
| `scripts/start_descent_tube_controller.sh` | systemd launcher | none | active watchdog fallback | defer | enabled watchdog timer directly retains this fallback | migrate/disable external service first |
| `scripts/resume_descent_tube_after_current_audit.sh` | one-time resume script | none | controller test | delete | completed audit continuation | split route-only assertion |
| `scripts/run_jump_envelope_pipeline.sh` | controller runner | none | paired start script | delete | closed jump-envelope route | shell syntax and ref scan |
| `scripts/start_jump_envelope_controller.sh` | systemd launcher | none | activator and route test | delete | closed jump-envelope route | external-unit check; split test |
| `scripts/run_stage_reachability_pipeline.sh` | controller runner | none | paired start script | delete | superseded sequential reachability route | shell syntax and ref scan |
| `scripts/start_stage_reachability_controller.sh` | systemd launcher | none | migration/controller metadata | delete | superseded sequential reachability route | delete migration cluster |
| `scripts/run_trajectory_mining_pipeline.sh` | controller runner | none | paired start script | defer | controller contracts are not yet migrated | migrate contracts first |
| `scripts/start_trajectory_mining_controller.sh` | systemd launcher | none | controller route test | defer | kept with unresolved trajectory-mining cluster | migrate contracts first |
| `scripts/start_stage_next_bootstrap_controller.sh` | versioned controller launcher | none | old controller/test | delete | superseded stage-next route | external-unit check; targeted test |
| `scripts/start_stage_next_v3_controller.sh` | versioned controller launcher | none | old controller/test | defer | kept with unresolved Stage-Next-v3 controller closure | complete dependency migration first |
| `scripts/start_final_shared_v2_followons.sh` | old final-shared launcher | none | old final pipeline/test | delete | superseded v2 follow-on; draft branch independently removed it | split route-only test; shell syntax |
| `scripts/run_final_shared_policy_pipeline.sh` | five-stage consolidation runner | none | old launcher/tests | delete | superseded five-stage unified-RSI controller | retain reusable CLIs; remove route contract tests |
| `scripts/run_final_shared_jel_audit.sh` | five-stage 4/8/32 audit runner | none | active-pointer launcher and tests | defer | enabled watchdog can invoke the active-pointer launcher that calls it | migrate/disable external service first |
| `scripts/run_corrected_apex_unified_rsi_pipeline.sh` | corrected old pilot runner | none | active-pointer launcher/test | defer | enabled watchdog can invoke its launcher | migrate/disable external service first |
| `scripts/start_corrected_apex_unified_rsi_followons.sh` | old pilot/JEL launcher | none | `runs/ACTIVE_PIPELINE.json`, enabled watchdog | defer | live external pointer names this launcher | migrate/disable external service first |
| `scripts/run_apex_reachability_funnel.sh` | old 4/8/32 local funnel | none | route contract test | delete | hard funnel is not the universal soft-Tube requirement | retain reusable cost/selection code if independently used |
| `tests/test_jump_envelope_controller.py` | jump-envelope route test | none | none | delete | asserts only removed controller assets and activator text | run shared policy/provenance tests |
| `tests/test_stage_route_migration.py` | completed migration route test | none | none | delete | asserts only removed migration/controller route | run stable lifecycle and stage-reachability tests |

The old draft branch `agent/streamline-current-mainline@fd2bf3f` was inspected,
not merged. Its README/method text still describes the superseded five-stage
route and is not reusable as project truth. Its deletions of
`scripts/run_backward_bootstrap.sh` and
`scripts/start_final_shared_v2_followons.sh` agree with this ledger, but will
be manually reproduced only after review and fresh validation.

### Deletion batch 1 validation

On 2026-08-03, fresh module/path and shell/systemd/docs/test scans found no
reverse references outside this ledger for these five closed-route files:

```text
cli/activate_descent_envelope_pipeline.py
cli/activate_fast_handoff_route.py
cli/activate_trajectory_mining_pipeline.py
cli/audit_descent_student_relabel_v2.py
cli/run_unified_descent_teacher_cv_v2.py
```

Their helper functions were not imported elsewhere; the two larger diagnostics
were bound to immutable one-off run paths and protocols. They were deleted as
one dependency-free batch. `compileall dvgc cli` passed, and the 15 focused
Descent probe/supervised/teacher plus repository/project contract tests passed.
The only output was an existing third-party JAXopt deprecation warning.

### Closed-route cluster deletion record

On 2026-08-03, a fresh reference and installed-unit audit found no retained
consumer for the Jump-Envelope/Stage-Reachability, Decoupled-Bootstrap, or
Stage-Next-Bootstrap routes. The 13 controller, activator, migration, runner,
and launcher files in those clusters were deleted together with their two
route-only tests. One source-text assertion for the removed Stage-Next launcher
was removed from the otherwise retained watchdog test.

After deletion, the removed names occurred only in this decision ledger.
`compileall dvgc cli` passed, and 60 focused watchdog, reachability,
local-entry, bootstrap-preparation, lifecycle, seed, certification, repository,
and project-state tests passed.

## 7. Explicit defer set

All unlisted tracked files default to `defer`. This prevents the audit from
turning “not yet classified” into deletion authority. The following high-risk
files are called out individually because they look legacy by name but contain
shared functions, internal imports, dynamic edges, or unresolved orchestration.

| path | role | retained_root | reverse_dependencies | decision | reason | validation_required |
|---|---|---|---|---|---|---|
| `dvgc/backward_search.py` | proposal/search utilities | no | 29 legacy CLIs/tests | defer | broad utility surface | migrate callers and unit-test API first |
| `dvgc/backward_tube.py` | tube proposal utilities | no | 11 legacy CLIs/tests | defer | possible reusable tube logic | semantic/API review |
| `dvgc/centroidal.py` | centroidal diagnostics | no | Apex audits and test | defer | possible reusable physics diagnostic | targeted test and method review |
| `dvgc/certification_merge.py` | certification merge utility | no | merge CLI and pipeline test | defer | generic certification capability | decide stable evaluation boundary |
| `dvgc/certifier_calibration.py` | calibration utility | no | calibration CLI/test | defer | generic evaluation capability | targeted calibration tests |
| `dvgc/construction_lifecycle.py` | stable construction/orchestration contracts | shared-contract tests | controllers and lifecycle tests | keep | owns shard, OOM, failure-fuse, liveness, resume, and provenance contracts | lifecycle and affected-controller tests |
| `dvgc/continuous.py` | continuous search/rollout utility | no | searches and test | defer | dynamic runtime behavior | targeted runtime review |
| `dvgc/delay_probe.py` | snapshot-delay audit utility | no | timing audits | defer | snapshot semantics are protected | snapshot audit tests |
| `dvgc/descent_balanced.py` | balanced training records | no | old pilot and test | defer | possible distillation reuse | data-contract review |
| `dvgc/descent_feedback.py` | feedback dataset logic | no | old probe and test | defer | teacher reuse unclear | targeted test |
| `dvgc/descent_membership.py` | membership logic | no | discrete-tube test | defer | soft-tube migration relevance unknown | evaluation review |
| `dvgc/descent_pilot.py` | bounded PPO pilot helpers | no | probes and `descent_probe` | defer | internal `dvgc` reverse dependency | untangle module boundary |
| `dvgc/descent_predecessor.py` | predecessor utilities | no | teacher/search CLIs and test | defer | possible snapshot acquisition reuse | targeted test |
| `dvgc/descent_probe.py` | restore/evaluate helpers | no | 7 CLIs, 3 `dvgc` modules, test | defer | explicitly shared internal dependency | migrate helpers before any deletion |
| `dvgc/descent_supervised.py` | supervised/distillation helpers | no | 14 CLIs and test | defer | likely reusable for expert snapshots | API extraction review |
| `dvgc/descent_teacher.py` | teacher dataset logic | no | old CLIs and test | defer | possible two-phase reuse | targeted test |
| `dvgc/discrete_tube.py` | discrete Tube logic | no | analysis/freezing CLIs and test | defer | future soft-tube contrast/ablation value | method review |
| `dvgc/entry.py` | stage entry/matcher utility | no | 22 CLIs and test | defer | broad shared stage contract | matcher semantics must not change |
| `dvgc/flight_augmentation.py` | candidate augmentation | no | 5 CLIs and test | defer | possible guideline-bank reuse | targeted test |
| `dvgc/local_entry.py` | local-entry utility | no | pilot CLI and test | defer | next-stage label reuse likely | two-phase label design review |
| `dvgc/observation_audit.py` | observation/timing audit | no | timing CLIs and test | defer | protected observation contract | observation audit tests |
| `dvgc/pipeline.py` | marker/gate utility | no | two CLIs and pipeline test | defer | reusable resumability logic | stable pipeline API decision |
| `dvgc/provisional_descent.py` | provisional reset helpers | no | 6 CLIs, `descent_pilot`, test | defer | internal dependency and role semantics | artifact-role review |
| `dvgc/reset_geometry.py` | reset geometry | no | 9 CLIs, `descent_pilot`, test | defer | proposed merge is not proven safe | geometry tests; no matcher change |
| `dvgc/roll_controllability.py` | roll diagnostic | no | audit CLI/test | defer | possible failure-analysis value | targeted test |
| `dvgc/stable_construction.py` | stable-state construction | no | controller CLIs/test | defer | lifecycle reuse unclear | targeted test |
| `dvgc/stage_reachability.py` | labels/models/sampling | no | 21 CLIs and 2 tests | defer | explicitly protected; likely feasibility reuse | API and label-semantic migration |
| `dvgc/support_diagnostic.py` | support geometry audit | no | timing/feedback CLIs and test | defer | protected snapshot/geometry diagnostics | targeted test |
| `dvgc/trajectory_mining.py` | trajectory utilities | no | 12 CLIs and 2 tests | defer | reusable snapshot collection logic | API extraction review |
| `dvgc/viability.py` | existing viability model | no | 3 CLIs | defer | must not be mislabeled as new `V_up/V_down` | model-semantics review |
| `cli/pipeline_watchdog.py` | legacy external-state watchdog | no | systemd and status scripts/tests | defer | static audit cannot establish user-service state | explicit external-service check |
| `scripts/dvgc_status.sh` | watchdog status | no | docs/watch/test | defer | coupled to deferred watchdog | watchdog decision |
| `scripts/dvgc_watch.sh` | status loop | no | watchdog docs | defer | coupled to deferred watchdog | watchdog decision |
| `scripts/dvgc_notification_helper.sh` | watchdog notification helper | no | external service possible | defer | external invocation cannot be ruled out | external-service check |
| `scripts/install_pipeline_watchdog.sh` | watchdog installer | no | external service possible | defer | local service state unknown | external-service check |
| `systemd/user/dvgc-pipeline-watchdog.service` | user service | no | timer | defer | external user service may be installed | inspect systemd state before deletion |
| `systemd/user/dvgc-pipeline-watchdog.timer` | user timer | no | service | defer | external user service may be installed | inspect systemd state before deletion |
| `scripts/run_remaining_pipeline.sh` | large old sequential runner | no | mixed source-contract test | defer | calls retained generic CLIs and embeds resumability contracts | split shared tests before deletion |
| `scripts/run_stage_expert_pipeline.sh` | old multi-substage expert runner | no | mixed source-contract test | defer | expert CLI remains runtime-fingerprint dependency | replace formal expert semantics first |

Route tests are also deferred by default. Direct inspection found reusable
contracts embedded in controller-specific tests, including seed disjointness,
shard completeness, OOM backoff, and certification seed rules. Shared portions
must move into stable tests before route-only assertions are removed. No test is
approved for deletion solely because its filename mentions a legacy route.

### Shared-contract extraction record

On 2026-08-03, shard completeness, OOM backoff/detection, failure-fuse
normalization, and lock-liveness behavior moved from legacy controller
definitions into `dvgc.construction_lifecycle`. Stable lifecycle tests were
written first and failed on the missing API before the behavior-preserving
move. Legacy controllers now import those helpers from the stable module.

The remaining requested contracts already have stable API-level coverage:

- exact seed-set disjointness: `dvgc.seed_registry` and
  `tests/test_seed_registry.py`;
- certification seed separation: `dvgc.certification` and
  `tests/test_certification.py`;
- resume/provenance idempotence: `dvgc.construction_lifecycle` lifecycle tests;
- non-overwrite policy ownership: `dvgc.experts`/`dvgc.policy` tests;
- snapshot and bank provenance: `dvgc.snapshot_provenance`, `dvgc.bank`, and
  their stable tests.

Static compilation passed. The 66 affected lifecycle/controller, seed,
certification, audit-manifest, expert, bank, and snapshot-provenance tests
passed; the only warning was the existing third-party JAXopt deprecation.

### External systemd state rule

No start script, watchdog helper, service, or timer may be deleted until the
real Ubuntu host's user-service state has been inspected. An installed,
enabled, loaded, or active service reference moves the target from `delete` to
`defer` until the service is disabled or migrated. Inspection is read-only;
this cleanup never stops a running user service without explicit permission.

Read-only inspection on 2026-08-03 found
`dvgc-pipeline-watchdog.timer` installed, enabled, loaded, and active. Its
service is installed/loaded and currently inactive. The installed service runs
`python3 -m cli.pipeline_watchdog` from `/home/qy/DVGC`; the active pointer
names `scripts/start_corrected_apex_unified_rsi_followons.sh`, and the watchdog
source retains `scripts/start_descent_tube_controller.sh` as a fallback. The
two transitive runner clusters have therefore moved from `delete` to `defer`.
No unit was stopped, disabled, reloaded, or otherwise changed.

## 8. Archive summary decision

| path | role | retained_root | reverse_dependencies | decision | reason | validation_required |
|---|---|---|---|---|---|---|
| `PROJECT_SUMMARY.md` | old clean-project/five-stage narrative | none | current README only | archive_summary | historically useful but must not remain current truth | replace README reference; summarize under `docs/archive/legacy_five_stage/` |

No Python source, JSON artifact, log, checkpoint, policy, or raw report will be
copied into `docs/archive/`. Git history and `archive/pre-clean-20260731` remain
the source-level archive.

## 9. Initial decision counts

The approved baseline contains 571 tracked paths. The current conservative
ledger classifies them as follows:

| decision | count | interpretation |
|---|---:|---|
| keep | 56 | retained closure plus stable construction-lifecycle contracts |
| delete | 30 | remaining closed-route candidates after moving unresolved clusters to defer |
| archive_summary | 1 | historical `PROJECT_SUMMARY.md` narrative only |
| defer | 484 | every other baseline path, including active-watchdog and unresolved dependency clusters |

These are design counts, not deletion results. A later fresh reference scan may
move a `delete` row to `defer`; it may not move a deferred file to deletion
without updating this ledger and reviewing the evidence.

## 10. Validation and commit sequence

After design review, implementation proceeds in focused commits:

1. `docs: switch project truth to two-phase research direction`
2. `cleanup: remove dependency-free legacy controllers and launchers`
3. `test: extract reusable legacy-route contracts`
4. `cleanup: remove obsolete migrations and route-only tests`
5. `docs: archive legacy five-stage route summary`
6. `test: validate retained repository entrypoints`

Before each deletion batch: repeat import/path/test/shell references. After each
batch: run `compileall` and targeted tests with
`/home/qy/mujoco_playground/.venv/bin/python`. Final validation is full pytest,
`scripts/local_preflight.sh`, and the runtime gate. The runtime gate's existing
64+32 timestep PPO is only a compile/update/checkpoint-resume smoke test; it is
not formal training or a learnability pilot.

This design phase stops after committing this document. It does not execute any
deletion or dynamic PPO validation.
