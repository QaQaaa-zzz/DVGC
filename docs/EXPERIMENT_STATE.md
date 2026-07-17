# DVGC Experiment State

- Active route: sequential shared-Actor backward training is formally closed.
  Proceed with stage-expert discovery, composite Final-Recovery validation,
  joint shared-policy consolidation, then final shared-policy recertification.
- Stage-expert infrastructure milestone: registry and irreversible composite
  controller implemented and validated.  Registry binds policy/params, XML,
  observation/action/PolicyState schema, candidate, downstream entry set and
  downstream controller-stack hashes.  Current policies are stateless MLPs;
  deployable observation history and last action remain in the uninterrupted
  environment PolicyState across handoff.
- Runtime gate v4: PASS/current; source `7ad49c7...58ca0`, config
  `50e9d14...64607`, XML `d7e9f43...ce794c`.  Composite handoff continuity
  continuity errors remain within declared tolerances; frozen downstream
  policy hash is unchanged.  Full suite: 60 passed plus targeted bridge tests.
- Immutable expert baseline root:
  `runs/stage_experts/flight_seed0_20260715T2045`.  Owned π_F initialization
  params `35fcb61...7000c` is a non-overwriting clone of
  `flight-20260715-124908`; frozen π_L params remain
  `fa3a518...34bb7e`.  C_L remains `a98a246...2d964`; initial registry hash
  `f78e5e7...901b2b`; current runtime registry is
  `expert_registry_runtime_gate.json`, hash `f63ace6...d8c3a`.  Before/after
  hashes of π_L manifest/params, C_L and the Flight bank are identical.
- Flight expert training protocol implemented: π_F episodes terminate with a
  distinct `chain_entry` success at fixed C_L, while full-bank evaluation uses
  uninterrupted `π_F→π_L` composite Final-Recovery.  Curriculum is late
  descent → descent → apex → ascent → full; each level uses at most four
  25,600-step blocks.  Every block records full-bank Chain/composite Final,
  Chain-missed Final, subintervals, physical/timeout causes, PPO reset and
  transition shares, π_F action drift, controller stack and immutable π_L/C_L
  hashes.  Full suite: 60 passed; runtime gate includes the new protocol.
- Current validated source commits: immutable composite stack `4cb2d97`,
  Flight expert protocol `06ec637`, route/protocol state `a30cf98`, nested
  support audit `7859e1f`, entry recovery `397027d`, continuation provenance
  `ba75bd0`, current-registry selection `f6dfd51`, and bounded apex bridge
  `9a2516a`.
- Pretraining composite baseline on the fixed 160 Flight bank: Chain=0,
  composite Final=7.5%, Chain-missed Final=7.5%, physical failure=92.5%,
  timeout=0; pitch/roll/recovery=101/47/12.  This exactly reproduces the old
  pilot baseline under the new composite semantics.
- Frozen π_L independent 96-state checks at seeds 8200000 and 2300000 both
  give Final=Chain=91.667% (88 recovery/8 roll).  The historical seed-2300000
  report was 89.583%: 94/96 terminal outcomes match and two change from roll
  to recovery.  Policy, candidate, XML and action hashes are unchanged and
  the stage-expert diff adds no active Landing dynamics/reward path; the old
  report lacks a runtime fingerprint, so exact cross-runtime replay is not
  claimed.  The current check has no degradation and is preserved separately.
- Prior shared-Actor runtime gate (superseded by v4): source `c7b7b09...bce12e`, config
  `307f41a...d28e9`, XML `d7e9f43...ce794c`; authoritative action mapping is
  unchanged.
- Landing policy: `landing-20260714-190401`, params `fa3a518b...34bb7e`.
- Landing candidates: `9784649...099505`; 96/96 eligible, 0 duplicates,
  one-step failure 0, 25-step physical failure 2/96, timeout 0.
- Landing Tube: `landing-98ee7a10f7`, `6f6b2ac...61ce8`; labels
  safe/boundary/dead/unknown = 79/12/4/1.
- Landing audit: 96 states, 1536 branches; Final=Chain=87.17%, physical
  failure=12.83%, timeout=horizon=0; precision/recall/coverage=98.73/96.30/
  82.29%, Brier=0.00888, ECE=0.06843. PASS; one held-out false-safe retained.
- Confirmed fixes: `9caaf23` chunked audit, `3421c47` bounded Warp replay gate.
- Flight augmented bank: `2d5d7de...f62934`; 83 fixed reference anchors plus
  77 constrained local candidates. Automatic ascent/apex/descent quotas and
  counts are 63/20/77. Build used 178/3000 proposals, 58 unique parents,
  maximum two children/parent; normalized NN min/p50/p95/max =
  0.1546/0.2807/1.3375/3.5079 at threshold 0.15. Rejections: joint range 74,
  normalized duplicate 26, pitch 1.
- Candidate gate: PASS. Overall and anchor/augmented grouped audits have
  contact=deep penetration=25-step physical failure=timeout=nonfinite=0;
  all 160 are finite/eligible/Flight-semantic and provenance-current.
  Anchor correction min/p50/p95/max=0/0.01236/0.15800/0.19023 m; augmented=
  0/0/0.00256/0.03318 m.
- Flight pilot: `runs/flight/pipeline_seed0_v5/pilot`, policy
  `flight-20260715-124908`, seed 0, 102400 effective steps. Healthy runtime,
  no NaN/OOM/compile restart; throughput 7.5k--11.3k SPS after compile.
  Fixed evaluation: Final=7.5%, Chain=0, physical failure=92.5%, timeout=0;
  pitch/roll/recovery=101/47/12. Anchor Final=6.02%, augmented=9.09%.
  Ascent/apex Final=0 in both groups; descent=15.58% overall.
- Frozen Landing-policy baseline on the same bank: Final=1.875%, Chain=0,
  physical failure=98.125%, timeout=0. Pilot improves Final fourfold and mean
  steps 20.82->25.30, but evaluation oscillates at 3.1--11.7% and misses the
  50% pilot gate. Value loss converged to 0.137, KL to 0.0131, policy std
  remained 0.0502; reset mix is 90% candidates/10% downstream rehearsal.
- Canonical Landing-entry v2: construction seed 4100000, certification seed
  4200000, audit seed 5200000; 90 proposals, 79 safe / 8 boundary / 0 dead /
  3 unknown.  Three proposals are true first-valid-contact snapshots; one is
  Final-safe.  Independent entry audit: precision/recall/coverage =
  98.73/92.86/87.78%, physical failure 7.29%, timeout=horizon=0.  Matcher
  radius 1.10679; independent matcher precision/recall=98.77/95.24%; bank
  `a98a246...2d964`.
- Flight handoff v3, fixed 160 candidates: frozen Landing policy has
  Chain/Final=1/3, false progress=0, Chain reward=8 and all Final trajectories
  have valid landing.  Existing Flight pilot has Chain/Final=0/12, so its
  learned trajectory does not reach `C_L`; this is no longer a missing event
  or reward signal.
- Curriculum v6 late-descent from Landing completed 102400 effective steps:
  Chain=0, Final=4.375%, physical failure=95.625%, timeout=0, Landing retention
  53.125%.  Runtime was healthy but early KL spikes (613.95 and 117.94) and the
  10% rehearsal mix caused destructive forgetting; the gate stopped before
  descent unlock.
- Start-policy comparison now includes retention: frozen Landing has
  Flight Chain/Final=1/3 and Landing retention=89.58%; old Flight pilot has
  0/12 and retention=87.50%.  The pilot wins combined fixed Chain+Final while
  satisfying retention, so v7 selected it by evidence.  Its repair round used
  learning rate 1e-5 and 30% canonical Landing-entry rehearsal.
- Curriculum v7 late-descent completed 102400 effective steps from the old
  Flight pilot with LR=1e-5 and 30% canonical-entry rehearsal.  Runtime was
  healthy (NaN/OOM/compile restart=0), KL stayed 0.00063--0.00314 and value
  loss fell 4.10->0.258.  Fixed Flight: Chain=0, Final=13.125%, physical
  failure=86.875%, timeout=0; anchor/augmented Final=12.05/14.29%; termination
  pitch/roll/recovery=99/40/21.  Landing retention=61.458% versus 89.583%
  reference, so the same late-descent unlock gate failed again.
- Read-only Chain support audit captured all 3/12/21 prior Flight Final
  trajectories.  Frozen Landing recovery among them is 3/10/11; C_L matches
  are 1/0/0.  Twenty-two recoverable unmatched states are isolated in pending
  bank `6a441ae...f977898`; none is active in training or Chain matching.
- Fixed-C_L bounded retention preflight: PASS.  Bank-reset weights are exactly
  Flight/canonical-entry/full-Landing = 60/10/30; source record counts are
  39/79/91.  Landing snapshots preserve Landing phase and PolicyState, and
  standalone/rehearsal reward, termination and recovery gates match.  Active
  C_L remains v2 `a98a246...2d964`; Flight bank remains `2d5d7de...f62934`.
- Next automatic step: one continuous old-pilot continuation with callbacks at
  25,600/51,200/76,800/102,400 steps.  Each callback gates Landing retention,
  fixed Flight Chain/Final, full-safe/boundary local probes, reset episode and
  transition sources, timeout and nonfinite health.  No pending C_L entry is
  eligible until this bounded run completes.
- v8 block 1 is retained as an invalid sampler diagnostic, not formal evidence:
  25,600 steps gave Flight Chain/Final=0/8.125% and Landing retention=89.583%,
  but exposed that the default Brax wrapper reused cached initial resets and
  the callback read the wrong metric prefix.  v9 enables per-episode full bank
  reset only for multi-source training; a 128-step short PPO smoke and the full
  runtime gate pass.  The formal bounded run restarts from the unchanged old
  pilot in a new v9 path, preserving v6/v7/v8.
- v9 block 1 is also retained as instrumentation-only: Flight Chain/Final =
  0/6.25%, Landing retention=89.583%, timeout=0.  Brax invokes the policy
  callback before publishing same-epoch metrics, so the source gate again saw
  empty data.  v10 defers block evaluation/gating until progress receives the
  matching epoch metrics; no policy, bank, C_L, reset ratio or PPO parameter
  changed.  Runtime gate and v10 fixed-C_L preflight pass.
- v10 stopped before completing block 1 evaluation: Playground full reset
  replaced terminal EpisodeWrapper metrics with the next reset's zero metrics.
  v12 uses a repository-owned wrapper that resamples all task/provenance state
  while preserving only terminal episode metrics/done/steps/truncation.  A
  64-environment, one-step terminal test proves every old source is counted and
  bank sources are resampled.  Targeted tests (14) and full runtime gate pass;
  v12 preflight keeps C_L and 60/10/30 inputs unchanged.
- v12 formal bounded run stopped at block 1 (25,600 cumulative steps).  Actual
  episode reset shares Flight/entry/full-Landing/natural = 55/11/30/4%; PPO
  completed-episode transition shares = 60.654/10.133/28.638/0.575%.
  Aggregate Landing retention=88.542% (gate 84.583%), but boundary Final fell
  75.0->58.333% and triggered the fixed local-collapse gate; full-safe and C_L
  Final both remained 97.468%.  Flight Chain/Final=0/6.875% versus old-pilot
  0/7.5%; ascent/apex/descent Final=0/0/14.286%.  Terminations were Flight
  pitch/roll/recovery=97/52/11 and Landing roll/recovery=11/85; timeout=0.
  No NaN, OOM, provenance error or compile restart occurred.  The active C_L
  remains `a98a246...2d964`; pending entries remain isolated.
- Decision: the fixed-C_L shared-Actor forgetting repair did not satisfy the
  required full-safe+boundary local retention criterion.  Stop shared-Actor
  repair now: no ratio/LR/budget adjustment, no pending-entry activation and
  no Flight curriculum continuation.  Await stage-expert plus final shared-
  policy direction.
- Stage-expert Flight late-descent round 1 completed four 25,600-step blocks
  (102400 cumulative) from the owned old-pilot clone.  Composite Final rose
  6.875/8.750/11.875/15.000%, but fixed-C_L Chain remained zero throughout;
  the final 24 recoveries are all Chain-missed Final.  Descent Final is
  31.169%; ascent/apex remain zero.  Physical failure=85%, timeout=nonfinite=0;
  frozen pi_L params remain `fa3a518...34bb7e`, C_L `a98a246...2d964` and
  Flight bank `2d5d7de...f62934` were unchanged during the run.
- Post-round support isolation: all 24 Final trajectories had valid contact;
  frozen pi_L recovered 14, all unmatched by original C_L.  Independent seed
  6700000 branch certification labels the 14 new proposals as 7 safe, 6
  boundary, 1 unknown.  Together with the previously isolated independently
  certified 13 safe proposals, immutable extended C_L
  `landing-entry-0ef5a5913228` has 99 Final-safe / 110 records, hash
  `185164d...2aa41`; matcher radius remains exactly 1.1067926888.
- Zero-training evaluation of block-4 pi_F against the extended C_L gives
  Chain=11.875%, composite Final=12.5%, Chain-missed Final=5.0%, physical
  failure=87.5%, timeout=0.  This proves the learned descent support is
  connectable after independently certified entry-envelope extension.  The
  recovery marker passes the late-descent dual gate without further training;
  the controller next enters full-descent, without changing matcher radius or
  downstream pi_L.  Predecessor bridge resets remain conditional on a later
  evidenced support gap rather than being added pre-emptively.
- Full-descent passed at block 2 (51200 cumulative): full-bank Chain/Final =
  10.0/13.125%, descent Chain/Final = 20.779/27.273%, physical failure
  86.875%, timeout=nonfinite=0.  Frozen pi_L and extended C_L hashes remained
  unchanged.
- Apex round 1 exhausted 102400 steps with apex Chain=Final=0 in all four
  blocks and 100% apex physical failure.  The single authorized bounded repair
  used a deterministic 40-state bridge layer (20 apex + 20 nearest early-
  descent candidates), fixed LR/PPO budget/reward/matcher, and restarted from
  the last passed descent checkpoint.  It also exhausted 102400 steps: final
  full-bank Chain/Final=3.125/14.375%, while apex remained 0/0 and 100%
  physical failure.  No timeout, nonfinite or provenance error occurred.
- Paused at the second evidenced apex gate failure; ascent was not started.
  No-training trace seed 6900000 shows all 20 apex trajectories terminate
  before valid contact (pitch 18, roll 2) at 16--22 steps.  Their minimum
  distance to C_L radius 1.10679 is min/p50/p95/max =
  4.463/6.145/7.080/7.111, so Chain reward credit is exactly zero.  Bridge
  block-4 action drift L2 mean/p95/max = 0.01053/0.01260/0.01917 and KL =
  0.02325/0.03240/0.07601.  The evidence indicates a real apex-to-descent
  dynamical support gap under the fixed method, not matcher or timeout failure.
- Apex C_D investigation (2026-07-16): froze π_F,D params
  `917f77c...af041`, π_L `fa3a518...34bb7e`, extended C_L
  `185164d...2aa41`, Flight bank `2d5d7de...f62934` and XML
  `d7e9f43...ce794c` in immutable run
  `runs/stage_experts/apex_seed0_20260716T124806`.
- C_D proposals use complete pre-contact Flight snapshots from successful
  frozen π_F,D→C_L→π_L rollouts, authoritative MuJoCo contact/geom checks,
  and no root correction.  Fixed dynamics variants produced 70 proposals;
  confirm-safe-to-32 certification (seed 7100000) gave Final labels
  safe/boundary/dead/unknown = 3/22/33/12, below the existing four-safe
  activation gate.  No C_D matcher was activated.
- Independent C_D audit seed 7200000 was recovered after one Warp OOM by seven
  new-process global-index shards.  Merge verified indices 0--69 and 2240
  unique branch seeds: Final=604/2240, physical failure=1043/2240,
  timeout=horizon=0.  One of the three confirmed-safe states audits at only
  23/32, so a 0.95-precision matcher cannot be justified from this support.
- A final bounded support extension used the already-declared 0.03 action
  noise.  It retained 32 byte-unique full snapshots from 21 successful
  rollouts (12 exact duplicates and 38 contact/penetration states rejected),
  spanning proposal steps 0--13.  Fresh confirm-safe-to-32 certification seed
  7150000 yielded Final safe/boundary/dead/unknown = 0/13/12/7;
  timeout=horizon=0.  This repair also fails C_D activation.
- Decision: do not run the reward baseline against an invalid C_D, do not add
  apex shaping, and do not start PPO.  The same critical C_D support gate has
  failed two evidence-backed repairs; pause for research direction rather than
  lowering Final-safe, matcher precision or minimum-support gates.
- Verification after tooling changes: local preflight 72 passed, final targeted
  suite 10 passed; runtime gate PASS/current, source `364209d...c8e53`, config
  `0c323d0...8fe7c`, XML
  unchanged.  OOM was isolated to the abandoned monolithic audit process; all
  seven accepted audit shards completed without OOM.
- Descent-local route authorized at HEAD `6c99b18`.  Immutable working root is
  `runs/stage_experts/descent_local_seed0_20260716T163504`; π_F,D-local is a
  params-only clone of π_F,D (`917f77c...af041`) with no optimizer state.
  Frozen π_F,D, π_L, C_L, Flight bank and XML hashes were rechecked unchanged;
  audit seed 7200000 is permanently marked consumed diagnostic evidence.
- Parent-bounded local pool `47c5854...1c9d9` contains 70 retained diagnostic
  records, 12 successful descent anchors and 60 structured children.  Training
  eligible groups safe-neighborhood/boundary-neighborhood/success-anchor are
  15/58/24; late/middle/early layers are 52/30/15.  Children use 37 original
  parents from 12 source trajectories, max four children/parent.  MuJoCo audit:
  contact=deep penetration=5-step physical failure=nonfinite=0; child states
  are unique against all base states.  The 41 duplicate historical diagnostic
  snapshots remain evaluation-only and do not create training multiplicity.
- The superseding deduplicated pool `cc2f2c1...7a76ff` keeps all 139 records for
  diagnostics but exposes only 83 byte-unique training states: 12 provisional-
  safe-neighborhood, 51 boundary-neighborhood and 20 successful-anchor states
  across late/middle/early = 50/27/6.  It contains 57 structured children from
  26 unique parents (maximum four per parent); the aspirational 60-child quota
  was not reached in 1200 fixed proposals, but every one of the 26 unique base
  parents has bounded local support and all three source groups are represented.
  Final candidate audit PASS: robot-terrain contact=0, deep penetration=0,
  five-step physical failure=0, timeout=0, nonfinite=0, and all eligible/child
  identities unique.  XML `d7e9f43...ce794c`, C_L `185164d...2aa41`, Flight
  bank `2d5d7de...f62934`, and candidate seed 7400000 remain fixed.
- No-training reward decomposition seed 7500000 exposed the old unified-reward
  failure: neutral Chain=0 but positive return p95=103.13.  The bounded local
  profile now keeps C_L Chain bonus=8, uses robust descent/C_L posture and
  velocity scales, potential distance progress, <=0.005 survival, and clips
  per-step shaping to [-0.35,0.25].  Final preflight on the same 83 states:
  pi_F,D-local Chain=74.70%, physical failure=12.05%, timeout=0, mean return
  5.03; neutral Chain=0, physical failure=53.01%, timeout=0, mean return=-2.83,
  positive shaping p95=0.482.  A missed C_L entry cannot collect the downstream
  Landing Recovery bonus during this candidate-guided episode.
- Parent-balanced bootstrap bank `1eb7898...bbd9d` has expected internal reset
  masses provisional-safe/boundary/successful-anchor=35/45/20%, 26 parents,
  and late/middle/early expected episode mass=52.67/41.33/6.00%; natural reset
  remains a separate 5%.  Actor observations exclude reset group/layer/parent,
  while episode and completed-transition ratios are instrumented for all three.
- Local bootstrap/reward/current-policy recertification controller is ready for
  cumulative 25,600-step blocks, exact optimizer-checkpoint continuation,
  immutable pi_F,D/pi_L checks, fresh construction seeds, and chunked fresh-
  process independent audit.  Full local preflight: 77 passed.  Runtime gate
  PASS/current: source `92ba819...0d534`, config `bf93618...04294b`, XML
  `d7e9f43...ce794c`.  Next automatic step is block 1; PPO has not yet started.
- First block attempt under commit `bac6380` completed 25,600 steps and is
  retained at `descent_local_seed0_20260716T163504/blocks/block_1_25600`, but
  was rejected before certification: block-end eval had 98 NaN aggregate
  metrics (no OOM/timeout; policy params and initial actions were finite).
  Fixed-candidate reproduction found 0/166 nonfinite episodes; a separate
  128-way full-reset probe completed 7,448 episodes without raw-state NaN,
  isolating a rare unclassified solver transition rather than bad candidates.
- Bounded repair adds an explicit `nonfinite` physical termination, preserves
  the prior finite qpos/qvel/ctrl/warmstart for the terminal sample, sanitizes
  terminal observations/diagnostics, and never changes normal finite dynamics.
  The failed checkpoint/run is not overwritten; the retry root is
  `descent_local_nonfinite_repair_seed0_20260716T1825`.  Full preflight now
  passes 78 tests; runtime gate PASS/current source `f3389b1...977ed`, config
  `bf93618...04294b`, authoritative XML unchanged.  Next step is a fresh block
  1 retry from the immutable pi_F,D-local clone.
- Formal nonfinite-repair block 1 is complete and healthy at 25,600 effective
  steps: policy `3ed8f3d...9dddea`, seed 0, exact optimizer checkpoint
  `000000025600`.  Candidate pool `cc2f2c1...7a76ff`, bootstrap bank
  `1eb7898...bbd9d`, C_L `185164d...2aa41`, pi_L
  `fa3a518...34bb7e`, source pi_F,D `917f77c...af041`, Flight bank
  `2d5d7de...f62934`, XML `d7e9f43...ce794c`, and runtime source
  `f3389b1...977ed` remain fixed.
- The original seed-7610000 monolithic current-policy certification reached
  global index 71, then exited on a GPU allocation failure before writing an
  atomic bank/report.  It is retained as an invalid/incomplete diagnostic and
  none of its log-only rows are reused.  Recovery uses 12-state new-process
  construction shards with the unchanged global branch-seed map, atomic
  completion JSON, strict 0--138 merge, and no block-1 retraining.
- The persistent descent-local controller now owns inspect -> sharded
  certification -> strict merge -> evidence-based decision -> exact optimizer
  continuation or pre-audit matcher freeze -> chunked independent audit ->
  viability state transitions.  Matcher radius is fixed solely from
  construction evidence before audit.  Targeted tests pass 16/16; a separate
  one-state seed-7609000 dynamic shard smoke passed with complete provenance,
  explicit end reasons and atomic output.  Formal seed 7610000 remains unused
  by completed shards and is the next automatic stage.
- Seed-7610000 block-1 construction is now complete: 12 atomic shards cover
  global indices 0--138 exactly once and strict merge passed.  Current-policy
  labels over 139 diagnostic records are safe/boundary/dead/unknown =
  8/41/73/17; after byte-state deduplication there are four Final-safe states
  from two sources.  Across 2660 branches, Chain=45.79%, Final=36.17%,
  physical failure=39.32%, timeout=nonfinite=0.
- Unit v3 stopped with `ExecMainCode=1`, `ExecMainStatus=1`; kernel, journal,
  CUDA and Warp logs show no OOM or signal.  Exact cause: the four-safe support
  admits no construction-only C_D matcher radius at the fixed 0.95 precision
  gate (best leave-one-out construction precision is 0.75).  No matcher or
  independent audit was activated and PPO did not continue.  Structured
  evidence is retained in `termination_diagnosis.json`.
- Controller recovery v4 separates persistent parent and GPU shard worker
  systemd services, validates every atomic marker, and applies certification-
  worker-only OOM backoff 12->6->3->1 without changing branch seeds/budgets.
  Exit 40 is now a non-restarting research `gate_pause`; exit 41 is an
  authorized stop; other nonzero exits remain restartable engineering errors.
  Full preflight is 86 passed, a fresh isolated runtime gate is PASS/current,
  and an independent worker-unit single-state smoke passed.  Current terminal
  state remains the fixed matcher-precision gate pause; there are no missing
  certification indices to rerun.
- Authorized exact-Tube branch freezes block-1 policy
  `3ed8f3d...9dddea`, certification `613626b...3a889`, and 98 byte/state-
  unique snapshots as D_emp_safe/boundary/dead/unknown = 4/31/51/12.  Exact
  safe construction evidence is 114/128 Final (89.06%), 9/128 physical
  failures, timeout=nonfinite=0; support spans three middle and one late state
  from two parents.  Exact membership requires immutable id + full snapshot
  hash + policy hash + certification hash and performs no distance expansion.
- The failed global matcher remains inactive and immutable: robust-axis-scaled
  isotropic radius 0.5186367 has construction TP/FP/FN/TN=3/1/1/93,
  precision=recall=0.75.  Its sole false-safe is a dead state from the dominant
  safe parent; normalized proximity is dominated by `wy`, with smaller hip,
  pitch, vx and vz contributions.  This diagnostic is not overwritten.
- New `descent-tube` route uses fresh pointwise audit seed 9310000 in isolated
  worker services.  All 98 unique candidates are audited so exact safe,
  boundary, dead, unknown and Tube-outside mass share one independent seed
  namespace.  Pointwise failure alone triggers exact-optimizer block 2;
  pointwise success activates snapshot-only D_emp_safe while continuous C_D
  remains inactive and sends construction state-level labels—not audit data—
  to the parent-group-split acquisition-only Viability ensemble.
- Round-1 pointwise audit completed all 98 states/3136 branches at seed
  9310000: exact-member precision/recall=75/60%, aggregate Final=25.45%,
  physical failure=48.34%, timeout=0.  It failed the fixed pointwise gate and
  correctly triggered the single exact-optimizer block 2.
- Block 2 is an exact 51,200-step optimizer continuation from block 1, policy
  `4ab92ad...f848c`; no retraining or restart occurred.  Construction seed
  9630000 completed 139 states/2490 branches with Final=36.39%, physical
  failure=37.27%, timeout=nonfinite=0.  Its 98-state frozen exact labels are
  safe/boundary/dead/unknown=3/36/48/11.
- The first round-2 audit at seed 9330000 is retained only as an invalid
  diagnostic: 1,259 branch seeds overlap construction because the two
  global-index seed grids differ by exactly 30 state strides.  Strict seed
  isolation correctly rejected analysis.  The controller now uses explicit
  round-2 seed 200000000, verifies the complete planned seed grid is disjoint
  before launching any worker, and stops identical engineering restart loops
  after three occurrences.  Next automatic step is the replacement round-2
  independent audit; block-1 and block-2 training outputs remain immutable.
- Unified exact seed registry is active at
  `descent_tube_seed0_20260716T2330/seed_registry.json`.  It records candidate
  generation, PPO root seed, both construction sets, both completed audit
  sets, the invalid seed-9330000 set, the active audit, and reserved future
  acquisition/matcher/final-audit categories.  Historical evidence records
  two pre-registry intersections (including 1,195 block-2-construction seeds
  reused from earlier namespaces); no audit labels or thresholds were reused,
  and these namespaces will never be allocated again.
- Active round-2 pointwise seed 200000000 expands to exactly 3,136 unique
  branch seeds.  Its persisted full-set proof has intersection=0 against the
  7,093-seed union of every prior registered candidate/PPO/construction/audit
  namespace.  The audit manifest binds policy `4ab92ad...f848c`, 98-state
  candidate hash `644635c...3e17`, XML `d7e9f43...ce794c`, C_L and pi_L;
  exact membership is mandatory and the continuous matcher remains inactive.
  Future controller launches use exact-set allocation with up to three
  automatic namespace reallocations before the deterministic-error fuse.
- The single authorized post-round-2 support repair is implemented but has not
  started while the active audit runs.  It is capped at 64 byte/state-unique
  proposals and four children per real trajectory parent, prioritizes new
  parents then boundary and safe-continuity neighborhoods, preserves complete
  PolicyState, and enforces XML contact/penetration, finite Flight phase,
  short-rollout and normalized dedup gates.  It uses no policy success or
  independent-audit labels for proposal selection.  On a round-2 FAIL the
  resumable route is fresh proposal certification -> parent-balanced reset
  bank -> exactly one additional 25,600-step optimizer continuation -> fresh
  recertification -> seed-600000000 pointwise audit.  A further pointwise gate
  failure is the structured research pause; no threshold/radius is relaxed.
- Persistent execution at HEAD `1291a53`: seed-200000000 pointwise audit has
  atomically completed global indices 0--23 (24/98) and is running 24--35
  under parent unit `dvgc-descent-tube-controller.service`, MainPID 57224.
  A non-controller watcher unit only observes the structured terminal state;
  after the current, already-running controller exits at its old round-2 FAIL
  branch, it restarts that same unit once so controller v3 migrates directly
  into the authorized support repair.  It cannot start while the parent is
  active/activating and therefore cannot duplicate the current audit.
