# DVGC Experiment State

## Expanded Descent Tube v4 and continuous entry v2 (2026-07-28)

- Three non-overwriting natural-targeted candidate rounds were run with full
  cross-round proposal and parent exclusion. Rounds 2 and 3 each found one P1
  state, but lineage validation correctly showed both came from the same
  origin parent; the parent-disjoint round 4 found P0/P1=0/0. These states do
  not satisfy the two-parent Apex gate.
- The two distinct same-parent states were nevertheless eligible Tube
  extensions. Fresh namespace `descent-natural-targeted-independent-audit-v2`
  gave Chain=Final=64/64 with no physical failure, timeout, horizon or
  nonfinite. Descent Tube v4 contains 22 independently certified states:
  `runs/descent_natural_bridge_candidates_v1/independent_audit_round2_round3_2x32/descent_tube_v4.pkl`,
  SHA-256 `7e2e6c87...ab8560`.
- A new six-anchor local matcher was reconstructed under one Tube-v4 scale;
  construction P0/P1=48/48, precision=1.0, recall=0.980769 and FP=0. Fresh
  independent audit evaluated 72 states x 8 branches across three dynamics:
  Final=576/576, physical failure=timeout=horizon=nonfinite=0, matcher
  precision=1.0, recall=0.623264 and FP=0. Predicted early/middle/late branch
  counts are 61/61/237.
- Frozen continuous entry v2:
  `runs/descent_local_entry_v2/independent_audit_v1/canonical_descent_local_entry_v1.pkl`,
  SHA-256 `1f3e7ea1...590d5f`. The four-anchor v1 and all failed acquisition
  rounds are preserved. No PPO ran; the two-parent Apex authorization gate is
  still closed pending natural-lineage re-evaluation against entry v2.

## Independently audited local continuous Descent entry (2026-07-28)

- Commit `b61ed9e` added backward-compatible per-anchor stage-support radii;
  canonical `C_L`, XML, action mapping and frozen policies were unchanged.
  Full preflight passed 411 tests and the refreshed runtime gate is PASS at
  fingerprint `cd0e695f...e2eb76`.
- A non-overwriting four-anchor construction pilot covered early, middle and
  late Descent. All 32 local perturbation states passed P0 and P1. The four
  calibrated radii are approximately 0.05 in the frozen robust-normalized
  physical feature; construction precision=1.0, recall=0.972222, FP=0.
- Fresh independent namespace `descent-local-entry-independent-audit-v1`
  evaluated 48 states x 8 branches across three dynamics variants. All
  384/384 branches reached Final-Recovery; physical failure, timeout, horizon
  exhaustion and nonfinite were zero. The frozen matcher predicted 239
  branches with precision=1.0 and conservative recall=0.622396; predicted
  early/middle/late branch counts are 61/61/117.
- Canonical continuous entry artifact:
  `runs/descent_local_entry_v1/independent_audit_v1/canonical_descent_local_entry_v1.pkl`,
  SHA-256 `e4011cfd...50de80`. It is now an independently audited formal local
  Descent handoff region; it does not alter Tube v3 or constitute final shared
  JEL. Next action is a bounded replay of existing parent-disjoint Apex
  trajectories against this frozen region before any Apex PPO.

## Descent continuous-entry representation diagnostic (2026-07-28)

- The inactive continuous `C_D` matcher was not relaxed.  A CPU-only
  construction diagnostic tested the exact deployable 140-dimensional online
  Actor observation/history on the immutable 24-state construction set.  No
  fixed robust-normalized radius met the 0.95 precision gate: across scale
  floors 0.01--0.50, the best precision was 0.157895 at recall 1.0.
- Adding the separately constructed 12-state natural-targeted pilot (five
  construction positives in 36 total states) did not resolve the aliasing:
  best precision was 0.208333 at recall 1.0 for every tested floor.  These are
  construction labels, not independent-audit feedback.
- Therefore neither the task-relative 16D feature nor a single global ball in
  the deployable Actor-history feature can define formal continuous Descent
  entry.  `C_D` remains inactive and exact handoff plus Tube v3 remain frozen.
  The next bounded construction test is per-anchor local neighborhoods with
  construction-only calibration and a fresh, isolated audit; no Apex PPO is
  authorized before two parent-disjoint upstream trajectories enter a frozen
  formal Descent region.

## Expanded certified compact-expert Descent Tube v3 (2026-07-28)

- A CPU-only parent-held-out reachability probe on the 24 construction states
  did not justify a history-based proposal model: actor-history Brier=0.24001
  versus physical-16D Brier=0.23200 (relative change -3.45%). The failed model
  is advisory-only and was not used to define Tube membership or matching.
- An outcome-blind natural-targeted candidate pilot instead selected 12 new
  parent-distinct proposals, four per early/middle/late region. Frozen compact
  pi_D -> pi_L evaluation found P0=3 and P1=2 from two late-region parents;
  no PPO, matcher expansion or audit-label feedback was used.
- Fresh independent namespace
  `descent-natural-targeted-independent-audit-v1` evaluated the two P1 states
  with 32 branches each and three fixed dynamics variants. Both were 32/32
  Final-safe: Final=Chain=64/64, physical failure=timeout=horizon=nonfinite=0.
- Non-overwriting standard Tube v3 is
  `runs/descent_natural_bridge_candidates_v1/independent_audit_2x32/descent_tube_v3.pkl`,
  version `descent-compact-5ed48f633ec7`, SHA-256
  `f63d4ff7...ae83332b`. It unions the 18 v2 states and two newly audited
  states: 20/20 records are standard Final-safe under one frozen policy
  identity, with 640 unique branch records total. Exact handoff v2 is stored
  beside it. Tube v2 is preserved; continuous C_D matcher remains inactive.
- This completes and expands the independently certified Descent phase Tube.
  It remains a bootstrap-expert phase Tube, not final shared-policy JEL. The
  natural/Apex predecessor bridge remains separately blocked; the new safe
  states are late support and did not establish a natural-start handoff.

## Descent Tube upstream bridge bounded audit (2026-07-28)

- The independently certified compact-expert Descent Tube v2 remains frozen
  and unchanged. A zero-training replay of eight previously saved physical
  Apex-to-Descent bridge snapshots yielded P0/P1=0/0: all eight hit
  `roll_limit` eight ticks after handoff and were 40.276--54.660 normalized
  units from the new Tube. Their dominant incompatibilities were pitch, roll,
  knee and heading, so this old lineage is not current predecessor support.
- A fresh natural-reset, single-MJX-lineage probe used the fixed reference
  controller before handoff and the frozen compact pi_D -> pi_L stack after
  handoff. The nominal closest Tube distance was 7.304; seven fixed handoff
  ticks produced at most 26 stable Descent ticks, but Landing=Final=0 and
  termination was pitch-limit except one early Takeoff clearance failure.
  The top natural replay was exact.
- Four evidence-bounded local stages then ran without PPO or CEM: (1) 17
  finite-difference plus 64 LHS residual trials, (2) an earlier
  approach/Takeoff window, (3) a 50-point reference offset/pulse timing grid,
  and (4) a 5-switch pose-feedback screen plus 64-point gain trust region.
  The best timing result reached distance 7.295 with 12 stable ticks. The
  feedback result improved pose margin from -0.07835 to -0.00251 and distance
  from 8.214 to 7.471, but still terminated at pitch limit with no Landing.
- A final 45-point preregistered combination of switch timing, feedback scale
  and the prior late residual found no improvement beyond feedback alone.
  Every selected result reproduced exactly twice from natural reset; no
  snapshot handoff, XML/action/failure/matcher change, NaN, timeout, OOM or
  policy update occurred.
- Classification: `BOUNDED_NATURAL_PREDECESSOR_BRIDGE_SUPPORT_GAP`. The
  complete phase Tube is valid, but current reference/open-loop controls do
  not reach it from natural/Apex support. Repeating local action search is
  closed. The next scientifically distinct action would be a bounded Apex
  expert/RSI training pilot or a new deployable state representation; PPO
  remains unauthorized, so neither starts automatically.

## Compact Descent matcher neighborhood (2026-07-28)

- Commit `67568fe` added a geometry-only, outcome-blind 24-state expansion
  with globally distinct parents, eight states in each early/middle/late
  region, and a fixed 4 -> 8 -> 16/32 branch funnel. Full preflight passed
  394 tests and the runtime gate was refreshed before dynamic execution.
- The immutable construction run produced safe/boundary/dead/unknown =
  3/3/16/2 over 432 branches. Chain=193/432, Final=189/432, physical
  failure=243/432 (roll 100, pitch 73, other physical 70), and
  timeout=horizon=nonfinite=0. The three safe extensions are one middle and
  two late states; early contains one 22/32 boundary state and no safe state.
- Combining the 18 already certified Tube anchors with the three new
  construction-safe states still admits no isotropic 16D C_D radius at the
  fixed 0.95 precision gate. Best construction precision is 0.842105 at
  recall 0.761905 and radius 0.148784; full recall gives precision 0.724138.
  Three boundary/unknown construction states have exactly zero normalized
  16D distance to a safe anchor, demonstrating that the missing distinction
  lies outside the task-relative physical feature (not in a tunable radius).
- Therefore the continuous 16D matcher remains inactive and the immutable
  exact handoff plus independently certified Descent Tube v2 remain current.
  No audit label entered training, no PPO ran, and no matcher/failure/XML
  contract was relaxed. Next mainline action is predecessor proposal
  acquisition using phase-conditioned reachability only as a guide; every
  promoted state still requires frozen expert-stack Final-Recovery branch
  certification.

## Certified compact-expert Descent Tube (2026-07-28)

- Started from clean `5812143`; authoritative XML/action mapping, frozen
  pi_D weights, pi_L, canonical C_L and v4 timing contract remained unchanged.
  Localized head-only and last-block consolidation both failed at 15/18 P1:
  neither recovered early node `b30f8496...` and both forgot `028dae28...` and
  `b74dae91...`, despite anchor action RMS below 0.004. No PPO ran.
- The accepted bootstrap expert keeps pi_D weights bit-identical and adds one
  actor-visible compact command adapter around the verified five-tick CEM
  prefix. Its normalized full-command core/radius are 0.311758/0.377304;
  three of four preregistered micro trajectories lie in the core, while the
  nearest preservation trajectory is 0.838453 away. XML, matcher, failure
  definition, reward and frozen downstream policies were not changed.
- Construction under policy identity `d149961a...4b4ce99` passed P0/P1=18/18
  over seven candidates, four layers and early/middle/late support, with zero
  forgetting. The residual-only and all-four-branch radius attempts are
  preserved separately as failed diagnostics; no valid result was overwritten.
- Fresh independent audit namespace
  `descent-compact-expert-independent-audit-v1` used 18 states x 32 branches
  and three fixed dynamics variants. Final=568/576 (98.611%), Chain=546/576,
  physical failure=8/576 (all roll), timeout=horizon=nonfinite=0. Tube
  precision/recall/coverage=1/1/1, Brier=0.001736 and ECE=0.013889. PASS.
- Standard frozen phase Tube v2:
  `runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1/descent_tube_v2.pkl`,
  version `descent-compact-9d98810f9f20`, SHA-256
  `0e204f69...ca31ed`. The first v1 serialization is preserved but superseded
  because it retained source proposal labels in the per-record `final` field;
  v2 writes all 576 independent branches, standard Chain/Final posteriors and
  entry features without changing any outcome. It is an independently certified phase Tube under
  a frozen bootstrap expert, not the final shared-policy JEL. Audit labels are
  not training inputs. PPO authorization remains false.
- Exact-only canonical Descent handoff set is
  `canonical_descent_exact_entry_v1.pkl`, SHA-256 `c5cd06f9...74c413`.
  Continuous matcher remains inactive; no radius was inferred from the audit.
- Validation commits: localized probe `5fb3c65`/`dacf3b0`, compact expert
  `c08d455`/`975d62c`/`a6b4cb3`, audit `1f30148` and Warp batch repair
  `de46cc2`; full preflight 389 passed and runtime gate is PASS/current.
- Current automatic step: construct a separate construction-only neighborhood
  with safe and negative-under-current-expert records, then freeze a C_D
  matcher before any new independent audit. Run a 2--5% pilot first; do not use
  the completed independent-audit labels to tune the handoff radius.

## Unified Descent snapshot timing/delay audit v2 (2026-07-28)

- Started from `c540d2f`; frozen pi_D/normalizer, 24 snapshots, 12
  corrections, 244 compatible pairs and held-out states remained read-only.
  No training, CEM, relabel, Landing retention or policy update ran.
- The saved actor tensor is the exact online input: L0 reproduces frozen pi_D
  actions 24/24 and local authority 12/12.  Snapshot physical state and actor
  input share tick `t`, while saved history is already post-update
  `[t-2,t-1,t]`; independent restore incorrectly reuses it as pre-history and
  appends frame `t`.  Classification of the restore defect is
  `HYBRID_STATE_RECONSTRUCTION_ERROR`, not an online one-frame delay.
- Local authority L0/R0/D1/D2/J12 is 12/10/9/6/9.  D1 preserves the
  preregistered structure (75% authority, 23 pitch + 1 roll versus L0's 24
  pitch, two candidate-layer changes); D2 does not (50% authority and four
  layer changes).  Initial action RMS/max delta is 0.0568/0.2036 for D1 and
  0.0797/0.2818 for D2, dominated by hip action.
- Frozen transfer diagonal/same/cross successes are L0 `12/3/40`, D1
  `9/6/69`, and D2 `6/4/61` from eligible totals `12/18/214`.  Transfer
  remains present, but local authority/support geometry is structurally
  two-tick-delay-sensitive.  Final classification:
  `DELAY_SENSITIVE_FEEDBACK_SUPPORT`; one added tick is tolerated under the
  fixed rule, two are not.
- Online results remain `empirical_online_evidence`; logged replay remains
  `logged_observation_replay_evidence`.  Legacy independent reconstruction is
  unverified and formal Tube/JEL evidence remains pending explicit v2 timing
  schema.  Recapture of the 24 states is not required for logged evidence.
  PPO/bootstrap authorization remains false.  Compact report:
  `docs/experiments/unified_descent_snapshot_timing_and_delay_sensitivity_audit_v2/`.

- 2026-07-24 `mjx_continuous_pipeline_repair_v1`: MJX-Warp/Newton
  deterministic replay fails independently of Jacobian AUTO/DENSE/SPARSE.
  The first natural divergence is the first physics step (`qpos`, about
  5.96e-8) and grows through contact; policy/history/PRNG remain identical.
  Snapshot replay first diverges around tick 18--19.  The supported MJX/CG
  solver is bit-exact for 20/20 snapshot and natural JIT replays and 2/2
  non-JIT replays (`runs/mjx_continuous_pipeline_repair_v1/determinism/report_v3.json`);
  full runtime gate is PASS/current at
  `runs/runtime_gate_mjx_continuous_repair_v1`.  No CPU--MJX gate is active.
- The new single-lineage runner never restores physics at a handoff and
  preserves observation history while monotonically switching temporary
  event controllers.  Its 100-episode natural-start smoke
  (`runs/mjx_continuous_pipeline_repair_v1/smoke_100_v1/report.json`) reaches
  Approach/Takeoff/Ascent/Apex/Descent in 100/100, with zero nonfinite,
  timeout or action-bound violations and valid trajectory reloads.  Landing
  and Final-Recovery are 0/100; every branch ends by `pitch_limit` after the
  frozen Descent policy takes over from a physical-Apex event outside its
  certified support.  Therefore the smoke gate is FAIL, not a pipeline PASS.
- Natural-lineage bounded search reduced normalized distance to immutable
  Descent support from 31.20 to 13.28 without matcher/failure changes.
  Reference-aligned temporary joint control reaches x=3.48 and reduces the
  closest distance to 7.075, but produces a second upward excursion/pitch
  failure; it does not constitute valid Descent entry.  At the closest valid
  descending tick, the main squared-distance deficits are wheel speed 15.85,
  vx 14.10, roll 5.11, x 5.10 and z 2.73.  Frozen Descent remains executable
  from its own support
  (8/10 auxiliary continuations reached Landing), so the current blocker is
  `natural_takeoff_to_frozen_descent_support_bridge_missing`, not snapshot
  restore or frozen-policy load failure.  PPO authorization remains false;
  XML/action mapping, matcher and frozen Descent/Landing hashes remain fixed.

- 2026-07-24 Takeoff-tail runtime-comparability gate (`9e18c20`, `61e40a9`,
  `bdbaeeb`, `6cd9a9b`, continuous-lineage correction `74fb9f1`,
  cross-engine gate `7e597cb`).  The historical 3161/5b73 records cross an
  Ascent handoff restore and therefore are composite policy traces, not one
  continuous physical lineage for a 22-tick contact-authority window.  A new
  continuous MJX replay from each real Takeoff source exposed material
  contact-branch sensitivity; CPU MuJoCo replay is bit-repeatable while the
  repeated MJX 3161/5b73 terminal spreads reach qpos L-infinity
  0.1133/0.1666 and qvel L-infinity 0.2450/0.1635.
- The formal four-parent CPU-MuJoCo/MJX gate replays the identical source
  snapshot and recorded action prefix.  Its fixed acceptance rule is
  separation-event delta <=1 control tick and separation centroidal-momentum
  L-infinity delta <=0.03.  Results are: 2f34 PASS (1 tick, 0.00279);
  89ff FAIL (1 tick, 0.11733); 3161 FAIL (16 ticks, 0.75370); and 5b73 FAIL
  (16 ticks, 0.75236).  Overall status is
  `takeoff_tail_cross_engine_mismatch`; authority/discovery and PPO are false.
  The completed v1 authority trends are retained as diagnostic-only and are
  not valid for selecting a control window.  No shortest window, E0,
  neighborhood, second-parent success, Gate A/B/C, Final result, or new
  proposal/Tubes were produced.
- Controller state is the explicit terminal `gate_pause` (exit 40), with no
  worker/GPU task.  The watchdog reads this as a research gate and does not
  recover it.  Next action requires choosing and validating one canonical
  contact runtime or reconciling CPU MuJoCo/MJX contact semantics, then
  rebuilding continuous lineage provenance; this is a runtime-method decision,
  not another bounded controller iteration.  Runtime gate itself remains
  PASS/current at source fingerprint `ce02673...dc96b`; authoritative XML
  remains `d7e9f43...ce794c`, config `949398...143d`, Takeoff bank
  `ccbfe9...933ce`, frozen Descent policy `527216...35f2`, and frozen Landing
  policy `fa3a51...bb7e`.

- 2026-07-24 event-aligned horizon/centroidal audit and contact-supported
  bridge (`e3aab98`, `50ee3d6`, paired-response correction `6242b58`, exact
  upstream lineage `6fd1672`, segmented bridge `485fc3e`).  The old
  `apex_local_correctable` diagnostic is now
  `apex_local_response_detected`; no closed-loop correctability is claimed.
  Across 36 nominal configurations (parents reference:131, 89ff and 2f34;
  event/-4/-6-or--8 starts; prediction horizons 3/6/9/12; two-tick
  replanning), 30 terminate by roll and 6 by pitch.  Stable Descent >=16
  ticks, formal Descent support, frozen Descent continuation and Landing Final
  are all 0.  Three-tick prediction reaches at most four stable ticks; longer
  horizons do not pass the gate.  Action-response latency is measured against
  a paired zero-action counterfactual, not natural state evolution.
- CPU MuJoCo exact-state centroidal momentum agrees with subtree momentum to
  <=2.34e-15.  Exact Takeoff-to-Apex lineage replay has qpos/qvel Linf error
  0 for 89ff, 2f34, 3161 and 5b73; reference:131 has no recorded Takeoff
  lineage and is explicitly unavailable.  Last-support/separation/Apex ticks
  are 1/2/18 (89ff), 1/2/19 (2f34), and 21/22/29 (3161/5b73).
  Separation Hx is -0.00877/-0.00795/-0.72398/-0.72425, with airborne
  relative Hx span 1.14%/1.05%/3.26e-5/3.06e-5.  Descent-terminal Hx p05/p95
  is 0.01665/0.08496; all 36 authority-normalized candidates remain outside
  the measured local envelope.
- A bounded real-last-support scan followed by 12-tick receding-horizon
  feedback evaluated 5 actions each on 89ff and 2f34.  All 10 branches end by
  roll; maximum stable Descent is 7 and 10 ticks respectively, with no formal,
  downstream or Final success.  Gate A/B = FAIL/FAIL and Apex PPO remains
  unauthorized.  Current stage-local blocker is
  `takeoff_tail_centroidal_momentum_blocker`; the next admissible action is to
  make bridge-admissible separation momentum an upstream Takeoff-tail proposal
  target, not more Apex search or PPO.  Controller is active/quiescent, with
  no worker/GPU task.

- 2026-07-24 pre-Apex feedback bridge diagnostic (`3a21582`, `cd5e2c3`,
  `731be20`, strict Gate-A correction `7da4a37`).  Existing 1,092 dynamic
  branches were aligned to physical Apex without replay: 1,006 cross Apex,
  failure-minus-Apex is p50/p95/max = 1/8/13 ticks, and 1,020 already show
  obvious pose divergence before the first nonzero bridge action.  A transient
  four-tick negative-vz segment occurs in 113 branches.  State-only
  leave-one-parent-out survival ranking is non-predictive (AUC 0.502);
  observed action-family rate variance is 5.37x the initial-candidate grouping
  variance.  These are diagnostics, never Tube/viability labels.
- Five-parent, 20-snapshot paired non-saturating pulse audit measures horizons
  1/2/4/8.  `reference:131` and `89ff...fcf0` retain rank-two event-local
  pose response; `2f34...a87` requires correction by Apex-3 ticks;
  `3161...7f7` and `5b73...69e4` have strong pitch but insufficient roll
  authority throughout the sampled window and classify as upstream-entry
  shaping required.  Current-runtime terminal proposals contain all 20
  Final-safe plus 29 stable boundary states; dead/unknown and one unstable
  boundary are excluded.  The old 35.48% reset-shock statistic is superseded:
  it counted successful next-entry terminals as shocks; physical reset shock
  reconstructed from the existing 372 branch records is 0/372.
- Event-start receding-horizon bounded shooting on robust parents
  `reference:131` and `89ff...fcf0` forms a four-tick stable negative-vz
  segment in 2/2 nominal and 8/8 fresh-dynamics branches, but every branch
  subsequently terminates by roll.  Formal Descent entry, frozen Descent
  continuation success and Landing Final-Recovery are all 0/10.  Direct
  continuation from each saved transient state also fails by roll.  Therefore
  strict Gate A/B/C = FAIL/FAIL/FAIL; the original run-local Gate-A boolean is
  superseded by `gate_reclassification_v2.json`, which enforces absence of
  later physical failure.  No PPO or frozen-asset change occurred.  Controller
  is active/quiescent at `pre_apex_feedback_bridge_stage_local_blocker`; next
  evidence-based action is earlier pre-Apex/upstream roll-momentum shaping,
  not matcher expansion or generic-Apex state collection.

- 2026-07-24 Apex->Descent interface audit and fresh-parent acquisition
  (`86e46b7`, `7417b70`, `649acf0`, `4a7321d`, `307a7bf`, OOM-resume
  `b912b4d`/`485b118`).  Frozen XML, Takeoff proposal bank, Landing policy,
  canonical C_L, detector/failure gates and matcher radius 2.213986 were not
  changed.  Existing parents classify as robust `reference:131` (10/10,
  fresh dynamics 4/4), robust `89ff...fcf0` (10/10, 4/4), and
  deterministic-only `5b73...69e4` (6/10, fresh 0/4).
- Current-runtime Descent support replay: 93 states x 4 branches, exact restore
  and t0 Descent phase 100%, five-step reset shock 35.48%, local Landing-entry
  reach 72.85%, Landing Final-Recovery 37.90%, physical failure 26.88%.
  Replayed labels are Final-safe/boundary/dead/unknown = 20/30/20/23.
  Only 12 historical labels were comparable (25% agreement), so the bank
  remains executable proposal support, never a certified Tube; runtime-stale
  is false.  Formal 16-D matcher semantics are internally consistent:
  angle wrapping changes 0/12 nearest neighbours, removing absolute x changes
  3/12 diagnostically, and a common landing-relative translation changes
  distances by exactly 0.  No metric/radius fix is authorized.
- Fresh acquisition preserved its atomic 24-entry bank and resumed per parent
  after two engineering OOMs without repeating completed work.  The entries
  are 12 canonical + 12 reference-aligned, controller mix old/new/specialist =
  9/7/8.  Round A evaluated 1,714 bounded proposals and found generic Apex
  entry from 3 new independent parents; pitch/roll terminations are
  1,219/492.  Eight reset-valid dynamic Apex snapshots were retained and one
  reset-shock snapshot rejected.  Merged bank v5
  `bd9328d...020e0` contains 13 dynamic states from 5 parents plus 4 diagnostic
  anchors; three normalized duplicates were rejected.  It still fails the
  required 16--32 dynamic-state gate.
- Expanded multi-knot Apex->Descent search evaluated 612 Round-A + 624
  Round-B branches on 13 dynamic states and 4 diagnostic anchors.  Dynamic
  Descent entry unique/parents/branches = 0/0/0; downstream success and
  Final-Recovery = 0/0.  Failure typing is pose-instability-before-Descent
  1,006 and apex-not-crossed 86; terminal pitch/roll = 48/1,044, with no
  timeout/nonfinite.  This is negative under the bounded controller bank, not
  physical unreachability.  Apex PPO remains unauthorized.  Controller is
  active/quiescent at `apex_descent_interface_support_blocker`, no worker/GPU,
  no global terminal/research gate; the next research action requires a new
  bounded pose-corridor/bridge-controller decision rather than more PPO steps.

- 2026-07-23 independent Ascent-parent acquisition and bounded late-Ascent
  discovery (`622db8a`, `d198319`, `797f5c2`).  Frozen Takeoff bank/config/XML,
  canonical C_L and Landing policy remain unchanged.  Parent-131 reproduces at
  Apex tick 15 under both deterministic/fresh seed replays; 140/144 fail by
  roll and 160/162/172 by pitch/roll under the identical control.
- Fresh Takeoff continuation generated 12 real Flight-confirmed Ascent entries
  from 12 independent upstream parents, balanced 6 canonical + 6
  reference-aligned.  Round A evaluated 820 bounded local proposals and found
  valid generic Apex entries from two distinct reference-aligned parents;
  Round B was correctly skipped.  The successful controls were
  `(hip,knee,start,duration)=(1.0,.65,0,16)` at tick 27 and
  `(.7,.245,0,16)` at tick 7.  Five dynamic Apex snapshots passed validation;
  one additional snapshot was rejected by the five-step reset-shock gate.
- BC on 34 action/observation examples from those two parents reduced action
  MSE from 0.1905 to 2.57e-6.  Late-Ascent PPO then ran as one continuation for
  two 25,600-step blocks.  Curriculum evaluation showed 55% then 40% generic
  Apex entry, but the fixed parent-disjoint reference set remained 0/6 and
  0/6 (block-2 early/late failures: roll 2, pitch 4).  The required two-block
  parent/unique stagnation rule stopped training; block 1 is retained as the
  tie-broken best proposal checkpoint.  No NaN/OOM/timeout occurred; block
  KL/value loss were 0.214/0.804 and 3.406/0.123, indicating large second-block
  policy drift without held-out coverage gain.
- Dynamic Apex bank v4 has 4 reset-valid reference anchors plus 8 dynamically
  reached states from 3 independent trajectory parents; it fails the required
  16--32 dynamic / >=4-parent authorization gate.  Apex->Descent bounded
  search evaluated 12 states x 11 controllers = 132 branches and found
  0 unique / 0 parents; terminations are roll 103 and pitch 29, with no
  timeout/nonfinite.  Frozen matcher schema/order/16-D normalization agree;
  current direct-linear angular handling is explicitly reported.  Minimum
  Descent-support distance remains 4.642; its leading squared contributions
  are pitch 8.971, x 4.919, vz 4.275 and hip 1.660.  This is
  negative-under-current-controller-bank only, not physical unreachability.
- Controller is active and quiescent at
  `apex_dynamic_support_stage_local_blocker`, with no worker/GPU task and no
  global terminal.  Chain (generic Apex entry), downstream Descent-support
  entry and Final-Recovery remain separate; current artifacts are proposal
  diagnostics, never Tube/JEL evidence.  Apex PPO is unauthorized.  Next
  bounded action is acquire more genuinely independent dynamic Apex parents,
  without increasing PPO budget or changing matcher/joint/XML gates.

- 2026-07-23 corrected-reset stage-next acquisition milestone (`f56b9ec`,
  `bb1b2f8`, `e0b24f9`, `fab17d9`, `854cae3`, `92e5f5d`, `b430547`,
  `f036338`, `f4846c0`, `e00eb99`, `9f86e3d`).  Takeoff reset v3 remains
  frozen at `ccbfe93...9933ce`; authoritative XML, frozen Landing policy/C_L
  and matcher were not changed.  Fixed Takeoff evaluation bank v2
  `798f2e7...6d514` has 12 canonical + 12 reference-aligned states from four
  held-out parents/stratum; all specialist training banks have zero parent
  overlap with it.
- Fresh v2 baseline: old Takeoff policy covers canonical/reference =
  9/12 and 6/12; corrected-reset pi_T covers 0/12 and 9/12; successful
  bounded sequences cover 1/12 and 9/12.  Their unique union is 9/12 and
  9/12.  Success return is about 8.05--8.08 versus missed-liftoff
  -2.18-- -1.55; the stage-entry event is present and one-shot.
- Canonical specialist was initialized from the old Takeoff policy and trained
  as one continuation.  Fixed evaluations at 25,600/51,200/76,800 steps are
  canonical/reference unique = 11/5, 10/4 and 8/3.  Two consecutive
  canonical stagnations stop before block 4.  Block 1 is the selected best
  (`9f834a3...679713`); final fresh frozen-bank evaluation gives
  old/new/specialist policy union plus bounded-sequence union =
  canonical 10/12, reference 9/12.  All three blocks completed without
  NaN/OOM/timeout; first-block KL was high (0.988), then 0.0447/0.0461.
- Frozen Takeoff controller-bank labels use 120 states x 3 policies x 4
  branches = 1,440 rollouts.  Next-stage entries are 862/1,440 (59.86%):
  canonical 403/720 and reference-aligned 459/720.  Labels are boundary 119 /
  positive 1; both strata contain success and failure branches, but neither
  contains an all-controller-fail state.  The old 307/480 and corrected
  single-controller 196/480 labels remain superseded/conditional and are not
  mixed with this bank.
- Conditional Takeoff reachability model v4
  `8234472...0df3c`, 16 physical features only, excludes candidate kind,
  reference index and controller/policy identity.  Leave-one-parent-out:
  Brier/ECE = 0.007735/0.006919; high-reach precision/recall =
  0.9245/0.9333.  Canonical precision/recall = 0.8947/0.9444 and
  reference-aligned = 0.9592/0.9216.  This is a proposal-support conditional
  model, not a Tube/JEL and not evidence of physical negatives.
- Apex reset construction exhausted three bounded validation rounds without
  weakening gates.  R1/R2 retain four reset-valid reference anchors and no
  dynamic states.  The action-aligned reset-shock correction in R3 captures
  three dynamically reached Apex states from reference parent 131; final bank
  `68352a3...fe247` has 7 states, only one dynamic parent, and FAILs the
  required 16--32 states / >=4 parents.  Apex->Descent bounded search on all
  seven gives 0/7 unique, pitch/roll = 25/45, support distance
  min/p50/max = 4.642/43.897/64.977.  Apex PPO was never started.
- Ascent reverse diagnostic uses exact aligned indices early
  131/140/144 and late 160/162/172.  Ten bounded/policy controllers find one
  early-parent Ascent->Apex success (index 131, hip-full/knee-half, tick 15)
  and 0/3 late successes; remaining causes are roll/pitch = 31/28.  This is a
  stage-local controller-support blocker, not physical unreachability.
- Dynamic snapshots exposed `int(None)` in apex-latch restore; `92e5f5d`
  fixes it and full runtime gate PASS/current at source fingerprint
  `ce02673...dc96b` (86.0 s).  Reference-action probes now use
  `source_reference_index` for dynamically reached states.  Controller is
  active with no worker at `stage_local_blockers_recorded`; next automatic
  research action is acquire additional independent Ascent/Apex trajectory
  parents.  No global gate_pause and no Apex/Ascent PPO are active.

- 2026-07-23 Takeoff reset authenticity correction is current (`63562c9`,
  watchdog `fdecf4a`).  Runtime gate v4 PASS; XML
  `d7e9f43...ce794c`, key `initial_state`, named qpos hip/knee addresses
  10/11 and values -1.2/2.5; actuators map to hip/knee with +/-50 N m.
  Old Takeoff policy/results (6/6, 307/480 and reachability model v1) are
  `superseded_by_takeoff_reset_correction` and are not current proposal or
  label evidence.
- New Takeoff bank v3 `ccbfe93...933ce`: 120/120 legal, 60 XML-key canonical
  plus 60 adjacent-reference-aligned compressed resets.  Body contact,
  premature Ascent entry and five-step reset-shock failures are all 0;
  all 120 have valid wheel support.  Old/new hip ranges are
  [-1.063,-0.991] -> [-1.299,-1.200], knee
  [1.663,1.862] -> [2.326,2.500].  Policy-free scan reaches Ascent on 6/12
  sampled states (all reference-aligned); frozen old-policy current-protocol
  baseline reaches 3/6 and 12/24 branches.
- New independent Takeoff pi_T (`takeoff-20260723-140623`, seed 103,
  6,400 steps) reaches 2/6 fixed states and 8/24 branches, all in the
  reference-aligned subset; no NaN/OOM/timeout.  Fresh 120 x 4 label pilot:
  196/480 branches, 49/120 unique positives, canonical 0/60 and
  reference-aligned 49/60; failures are 284 missed-liftoff deadlines.
  Reachability model v3 `e40235c...c0c2a5` is current-protocol but its near
  perfect held-out metrics primarily separate the two reset classes and do
  not establish broad Takeoff capability.
- Stage reset audit v2: Ascent has 6/6 exact reference-aligned anchors
  (indices 131--172), correct Flight phase, no t0 Apex entry and no five-step
  shock.  Its new independent 6,400-step pi_U remains 0/6 and 0/24
  (pitch/roll 14/10; timeout/nonfinite 0).  Apex has only four exact legal,
  non-premature anchors (217--220); Apex PPO is not authorized until bounded
  correlated local support supplies the two missing authentic states.
- Current stage is `reset_authenticity_followup`: Takeoff proposal/labels are
  regenerated and isolated; next action is diagnose canonical missed-liftoff
  versus the better frozen-old-policy baseline, and construct two validated
  local Apex reset proposals before any Apex PPO.  No controller/worker is
  active; watchdog pointer remains explicitly QUIESCENT and will not restart
  the superseded controller.

- 2026-07-20 route correction: old `flight_apex_expert_chain_blocker` is
  preserved only as `superseded_apex_to_c_l_objective`; global status is
  `stage_reachability_acquisition`.  Active local responsibilities are
  Takeoff->Ascent, Ascent->Apex, Apex->Descent, Descent->Landing/C_L, and
  Landing->Stable.  C_L remains frozen and is not a gate for Takeoff/Ascent/Apex.
- `descent_proposal_support_v1` build PASS: 93 byte-unique, finite Flight
  Descent snapshots under current authoritative XML; robot-terrain contact,
  deep penetration and nonfinite = 0.  Progress coverage is early/middle/late
  = 31/31/31.  Bank `1c39f9e...744dde`; it is proposal support only, never a
  certified Tube or formal JEL.
- Frozen Apex->Descent entry protocol `b419084...09f85` PASS: 80/93 support
  records satisfy the bounded entry gate; negative controls for missing
  apex-passed latch, rolling fall, body contact, wheel contact and nonfinite
  all reject.  Detector/support thresholds were frozen before policy relabel.
- Five-policy fixed relabel (20 Apex states x 4 branches each; 400 total,
  disjoint seeds/dynamics variants) finds Apex->Descent reach = 0/80 for each
  initial/Block-1/2/3/4 policy.  All terminate by pitch/roll; per-policy
  minimum support distance is 3.25--3.68.  This is a real local controller
  support gap, not evidence that Apex cannot reach Descent.
- Next automatic action: runtime gate refresh, then independent local pi_X
  25,600-step blocks with potential-based Descent-support progress.  Apex local
  failure remains nonblocking; Ascent and Takeoff local pilots continue and
  all outputs remain controller proposals/labels until final shared-policy
  recertification.
- Stage-next bounded acquisition result (`23718e5`, `19eac43`, `c3ea8d0`,
  `b04fba7`): runtime gate PASS/current; no NaN/OOM/timeout/provenance error.
  Apex pi_X stopped after 2 x 25,600 steps at 0/20 unique and 0/80 branches
  per block (pitch/roll 73/7 then 76/4).  Ascent pi_U likewise stopped after
  2 x 25,600 at 0/6 and 0/24 per block (all pitch-limit).  Both are
  controller-support gaps/negative-under-current-bank evidence only.
- Takeoff pi_T passed after one 25,600-step block: fixed pilot 6/6 unique,
  18/24 branches; policy `fc051ae...ad6d6f`.  A separate physically validated
  proposal pool contains 17 reference anchors + 103 constrained local states,
  hash `e7275df...b82d7d2`.  The 120-state x 4-branch label pilot gives 307/480
  next-stage entries across 91/120 unique states (63.958%); labels are
  positive/boundary/unknown = 56/35/29.  Failures: missed liftoff 131, pitch
  36, roll 5, positive-pitch gate 1; timeout/nonfinite = 0.
- Active controller proposal bank v2 contains only the successful Takeoff
  expert.  Failed Apex/Ascent policies are isolated as attempted evidence and
  cannot become teachers, positive generators, provisional envelopes, or JEL.
  Pipeline is at a valid research `gate_pause`: bounded Apex and Ascent local
  controller support is exhausted after independent Takeoff acquisition.

- 2026-07-19 route supersession: the handoff-first/descent-Final route is
  atomically closed as `SUPERSEDED_BY_STAGE_REACHABILITY_PROTOCOL`.  Old
  controller, watchdog and H1 worker are inactive.  The final already-started
  H1 atomic output completed normally (exit 0) and is preserved but will not
  extend C_L or trigger the retired A/B route.
- H4 remains diagnostic-only `runtime_sensitive_boundary`: formal terminal
  agreement 171/192 (89.0625%), complete evidence agreement 110/192
  (57.2917%), exact H4 3/8.  It is not eligible for branch pairing.  Before new
  label acquisition, a separate same-snapshot/policy/seed first-step and
  eight-step qpos/qvel/action deterministic replay smoke is mandatory.
- Active RA-L route: `safe_pause_old_route -> h4_replay_smoke ->
  define_stage_entry_protocol -> relabel_existing_data -> coverage_inventory
  -> stage_candidate_pilot -> stage_label_acquisition ->
  train_reachability_model -> active_candidate_acquisition ->
  build_stage_tubes -> Tube_RSI_final_PPO -> final_policy_certification`.
- Stage-entry protocol v1 maps Ascent/Apex/Descent to canonical Flight and
  defines physical-quality-gated Takeoff->Ascent, Ascent->Apex,
  Apex->Descent, Descent->Landing and Landing->Stable events.  Intermediate
  failures are controller-bank-conditioned, not claims of physical
  unreachability.  `proposal_support_set` and `certified_tube` remain distinct.
- Read-only coverage inventory (no replay): Takeoff 0 full snapshots; Ascent
  63; Apex 20; Descent 218 byte-unique states (including the historical 153
  set plus mined/support additions); Landing 120.  Existing composite reports
  lack next-entry snapshot/time fields, so their 9k+ branches are reusable as
  controller outcome evidence only.  Directly reusable Landing->Stable labels
  are 79 positive / 12 boundary / 4 negative-under-controller / 25 unknown.
- Candidate/acquisition cost gate: planned five-stage pool 750 unique states,
  four branches/state, horizon 200 = 3,000 rollouts / 600,000 environment
  steps.  The required 4% pilot is 30 states / 120 rollouts / 24,000 steps.
  No new MJX rollout has started yet; deterministic replay smoke gates it.
- Deterministic replay smoke PASS: 3/3 fixed snapshots, eight control steps
  each; action/qpos/qvel first-step and full-window maximum absolute error are
  all exactly 0.  New label collection is authorized; H4 remains a later
  runtime-sensitive boundary rather than a snapshot replay defect.
- Stage-label pilot v1 is preserved as an instrumentation diagnostic: it
  exposed that successful `recovery` termination was incorrectly counted as
  physical failure.  Fix `f938003` separates successful terminal events from
  physical failures and keeps 0/4 outcomes unknown rather than boundary.
- Corrected 4% pilot v2: 30 unique states, 120 rollouts, 200-step horizon,
  37 complete entry snapshots.  Landing->Stable = 24/24 (all step 24);
  Descent->Landing = 13/24 across 4/6 states, time-to-entry 3--36 steps;
  Takeoff->Ascent, Ascent->Apex and Apex->Descent = 0/24 under the current
  controller.  No timeout, nonfinite or worker failure occurred.  All outputs
  remain proposal support, not certified Tubes.
- Four-controller proposal-bank probe (`da9bd486`, `35fcb613`, `cc0288fe`,
  `917f77c8`) on the same 30 states also gives Takeoff/Ascent/Apex = 0,
  Descent = 11/24 and Landing = 24/24.  The zero stages are therefore marked
  controller-support gaps/unknown, never physically unreachable.  Do not
  expand to 750-state acquisition with this weak bank.
- Current validated commits: stage protocol/schema `f8c6323`, inventory and
  replay/cost gates `a89e447`, resumable migration `cacc86d`, pilot worker
  `40a6d5b`, terminal-label fix `f938003`, controller-bank probe `d42755d`.
- Controller stage: `stage_label_acquisition`; next automatic action is a
  bounded next-stage-objective controller/reward pilot for Takeoff, Ascent and
  Apex, using the supplied reward calculator only as read-only design input.
  It must precede large acquisition; no old-route task is eligible for
  recovery.

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
- Unattended monitoring is implemented as a separate two-minute user-systemd
  timer.  It atomically maintains `runs/CURRENT_PIPELINE_STATUS.{json,md}`;
  reports controller/worker PID, heartbeat, lock/provenance, shard progress,
  errors and next action; and performs at most three recoveries only for an
  inactive non-terminal controller or stale controller with no live worker.
  Desktop events are persistent-ID deduplicated and limited to major stage
  transitions, pipeline completion, research gate, or exhausted engineering
  recovery.  Monitoring does not import the training stack or alter any
  policy, bank, seed, branch budget, checkpoint, gate, or completed marker.
- Round-3 support repair completed one authorized continuation to 76,800
  cumulative steps, policy `5272166...353f2`.  Fresh construction over 114
  states labels exact safe/boundary/dead/unknown=4/43/52/15; aggregate
  Final=36.31%, physical failure=36.36%, timeout=nonfinite=0.  The first
  round-3 pointwise worker was correctly rejected before any branch completed
  because the controller supplied the original descent policy instead of the
  candidate bank's exact round-2 source policy `4ab92ad...f848c`; 31 transient
  restarts produced no accepted shard or audit evidence.  The controller now
  resolves candidate-source policy strictly by the bank metadata hash and
  separately verifies the current audit-policy hash.  Volatile worker launch
  timestamps are normalized in the three-failure fuse.  No seed, candidate,
  policy, checkpoint, branch budget, certification result, or gate changed.
- Round-3 pointwise audit seed 600000000 is complete and formally FAIL:
  114 states/3648 unique branches, Final=29.167%, physical failure=42.599%,
  timeout=horizon=0, exact precision/recall=50/50% with 4 construction-safe
  snapshots from two parents.  Policy `5272166...353f2`, D_all
  `16d61f7...d43dde`, D_emp_safe `5913f11...f9209`, C_L, pi_L and XML are
  frozen.  This audit is `CONSUMED_DEVELOPMENT_EVIDENCE`: it remains the
  official Round-3 failure but is ineligible as a future independent audit.
  Global matcher remains inactive; Round-3 PPO must not be repeated.
- Authorized next route: frozen-policy two-batch stable-safe construction with
  preallocated disjoint Stage-A/Stage-B/adaptive seeds, followed by a fresh
  pointwise audit when stable safe support and parent diversity pass; otherwise
  at most two parent-balanced Viability-guided acquisition rounds.  No PPO or
  apex starts before a discrete empirical descent Tube passes independently.
- Provenance/lifecycle repair commit `64843a7` makes the frozen Round-3 policy
  authoritative in monitoring, persists terminal audit manifests and detailed
  physical end reasons, and marks the seed-600000000 evidence consumed.  The
  new stable-envelope controller implements independent 32-branch Stage A/B,
  preregistered adaptive resolution, exact global seed-set proofs, stable-safe
  dual-batch admission, fresh pointwise audit, two bounded acquisition rounds,
  and 70/30 parent-balanced discrete Tube-RSI blocks.  Snapshot-source policy
  provenance is kept separate from the policy being certified.  Targeted
  verification is 32 passed; full suite/runtime gate and persistent launch are
  the remaining preflight actions.
- Stable-envelope run `descent_envelope_seed0_20260718T004058` exhausted its
  preregistered bounded protocol without engineering failure. Cycle 0/1/2
  stable-safe support was 2/3/3 states and 1/2/2 parents; final Cycle-2 labels
  are safe/boundary/dead/unknown=3/59/57/27 over 146 states and 9,248 branches.
  Final=40.39%, Chain=50.15%, physical failure=33.29%, timeout=nonfinite=0;
  roll accounts for 3,045/3,079 physical failures. Two acquisition rounds
  added 16 candidates each but no early stable-safe support. No pointwise
  audit, discrete Tube, Tube-RSI, or PPO started because minimum safe=4 failed.
- Authorized next branch mines complete snapshots from successful Final
  trajectories under the same frozen policy. Each successful branch becomes a
  distinct trajectory parent while retaining the original candidate parent;
  mixed snapshot PolicyState sources are recorded per state and must match the
  exact supplied source-policy hash set. Viability remains acquisition-only.
  The mined pool receives the unchanged Stage A/B/adaptive protocol and a new
  independent audit only after stable safe>=4. If support remains insufficient,
  a preregistered finite-difference roll controllability audit decides between
  one 25,600-step roll-targeted block and a structured research pause.
- Draft difference to reconcile only after the seed-0 chain succeeds: the
  draft states Nmax=32 and one shared Actor, while current discovery uses
  separate 32-branch Stage A/B confirmation plus bounded adaptive branches and
  a stage expert. Formal membership remains frozen-policy Final-Recovery only.
- Trajectory-mining selector engineering repair: failed run
  `trajectory_mining_seed0_20260718T141645` is frozen as
  `INVALID_ENGINEERING_DUPLICATE_SELECTION`; its 28 appended records are four
  selections of seven unique states and all Cycle-3 construction evidence is
  ineligible for formal Tube use. Commit `a328ae9` enforces global candidate
  ID, declared snapshot hash and canonical full-state byte uniqueness across
  every quota/fallback round, keeps the analyzer rejection, and reports valid
  support exhaustion instead of filling with duplicates. Full suite 146
  passed; runtime gate PASS/current.
- The first resume attempt
  `trajectory_mining_resume_seed0_20260718T193738` is retained as an invalid
  provenance-wrapper diagnostic: no branch completed because its byte-identical
  base records omitted the mining-added per-record source-policy declaration.
  Commit `a83d796` preserves the invalid bank's verified base wrapper and adds
  this declaration to the pre-worker hard gate. No scientific evidence from
  that attempt is consumed.
- Corrected non-overwriting resume root is
  `runs/stage_experts/trajectory_mining_resume_seed0_20260718T194327`.
  Corrected bank has 153 states = 146 unchanged base + seven unique additions,
  hash `d031e96...20e55`; unique ID/snapshot/byte counts are 153/153/153.
  Configured target is 64, parent-cap quota capacity is 28, selected unique is
  seven and quota shortfall is 21; selected layers middle/late/early=5/2/0,
  seven parents and maximum one selected child/parent. Fresh physical preflight
  seed 1700000001 accepts 7/7 with finite Flight semantics, zero construction
  rejection, complete two-policy per-record source provenance and current
  policy/XML/C_L/pi_L/config hashes.
- Prior evidence is not reused: existing Stage-A/B/adaptive artifacts are
  whole-bank-hash bound and do not bind per-state snapshot hashes, while the
  duplicate Cycle-3 evidence is explicitly prohibited. Resume Cycle 4 will
  therefore run fresh full stable construction on the corrected 153-state bank
  with new registered seed namespaces; no trajectory mining or PPO is rerun.
- Post-PPO lifecycle repair commit `31e4b75` invalidated the stale Cycle-4
  comparison without overwriting it, bound every construction artifact to the
  policy/bank/XML/C_L/pi_L/config/runtime/protocol/seed-epoch identity, and
  resumed directly at fresh Cycle 5. Targeted tests were 32 passed, the full
  preflight was 154 passed, and the retry runtime gate is PASS/current. The
  authorized roll-targeted PPO was not repeated: the only post-Cycle-4 policy
  remains `da9bd48...24fdce` at cumulative 102,400 steps.
- Fresh Cycle-5 construction is complete over the fixed corrected 153-state
  bank `d031e96...20e55`: Stage A 153 states, Stage B 101 selected states, and
  adaptive 53 states used 9,824 branches under
  `stable-descent-cross-seed-v1`, construction epoch 5. Final labels are
  safe/boundary/dead/unknown=0/65/60/28; stable-safe support is 0 states from
  0 parents and all layers are empty. Both Cycle-4 stable-safe states became
  unknown (84/96 and 73/96 combined Final branches), all seven trajectory-mined
  additions remain non-safe (5 boundary, 2 unknown), Stage-A/B consistency is
  58.42%, and boundary/unknown-to-stable-safe is 0. No Tube or audit candidate
  was frozen.
- The fresh before/after gate is formally FAIL with all lifecycle checks PASS.
  Cycle 4 -> Cycle 5 Final is 40.7586% -> 40.6963% (-0.0624 percentage points),
  stable-safe is 2 -> 0, and parents are 1 -> 0. Roll failures improve
  3,129/9,728 -> 2,757/9,824 and total physical failure improves 32.4630% ->
  28.2471%; pitch/back-edge are 2/16 after training, timeout/nonfinite/horizon
  remain zero. Chain is 4,859 -> 5,010 branches, but false-progress rises
  9.1900% -> 10.3013% and handoff-missed-final rises 2,605 -> 3,051. Thus the
  roll objective improved its targeted failure mode without retaining or
  expanding exact stable recoverability.
- Terminal state is a valid preregistered research gate (controller exit 40),
  not an engineering failure: `gate_pause`, `research_gate_valid=true`, last
  action `analyze_stable_construction`, reason `roll-targeted PPO failed fresh
  stable retention/expansion gate`. This is branch C; no pointwise independent
  audit, exact C_D, Tube-RSI, apex, or further PPO may start from this result.
  Authoritative artifacts are `cycle_5/stable/report.json` and
  `cycle_5/roll_targeted_block_gate.json` under the corrected resume root.
- Route revision (2026-07-19): the Cycle-5 result remains valid, but it no
  longer blocks full-jump discovery on an intermediate four-safe quota.  More
  roll-targeted descent PPO is prohibited.  The active order is no-training
  Cycle-4/Cycle-5 handoff H1/H2/H3/H4 decomposition, Landing-only certifier
  calibration, non-certified proposal-support construction, full-jump teacher,
  reverse trajectory mining, final shared Tube-RSI, and final independent
  certification.  `certified_tube` is the only safe-claim artifact;
  `proposal_support_bank` is training/search support and is ineligible for Tube
  precision or coverage.
- New immutable seed-0 run root is
  `runs/jump_envelope_seed0_20260719`.  Canonical C_L remains
  `185164d...2aa41`, radius 1.1067926888; frozen Landing policy remains
  `fa3a518...34bb7e`; corrected 153-state bank remains
  `d031e96...20e55`.  Initial support build is PASS: 93 legal records = 65
  boundary + 28 unknown active-sampling records, 60 dead excluded; artifact
  hash `f540122...dd879`, with safe claims and Tube metrics explicitly disabled.
- The first calibration artifact that split one historical batch in half is
  retained only as an invalid diagnostic and is not a research conclusion.
  The controller will run fresh fixed-branch Landing Stage A/B batches with
  independent seeds 2300000000/2400000000, select a rule on parent-split
  development data, and require validation precision >=0.95.  Descent results
  are not used to choose the protocol.
- Handoff decomposition replays 2,605 Cycle-4 and 3,051 Cycle-5 formal
  handoff-missed branches; 1,099 Cycle-5 branches are paired
  physical-to-handoff conversions.  Every first-contact snapshot is evaluated
  with fresh globally indexed Landing seeds and classified H1/H2/H3/H4.  H1
  proposals remain inactive until a second independent certification yields at
  least four Final-safe states from at least two parents; matcher radius is
  immutable.  The next automatic stage is `handoff_cycle4`; no PPO is active.
- Evidence-funnel override (2026-07-19): exhaustive handoff replay stopped at
  the atomic Cycle-4 `[168,192)` boundary because 192 events already contain
  99 H1 branches from 15 states/12 parents and all descent layers.  Continuing
  roughly 5,500 correlated events would require about 150k Landing rollouts
  without changing the immediate decision.  Global byte/normalized-feature
  dedup retained 96 distinct H1 contacts; the bounded selector chose 24 from
  12 parents (cap two), layers early/middle/late=16/2/6 and dynamics counts
  high/low/nominal=7/9/8.  Selected bank hash is
  `312af9c...eb778`; it remains pending and inactive for C_L matching.
- The fast route is H4 exact formal-code replay -> H1 independent fixed
  32-branch certification -> optional immutable C_L extension at >=4 safe from
  >=2 parents -> paired zero-training canonical/new-C_L evaluation on the full
  fixed bank.  Eight observed H4 branches span six candidate indices and are
  isolated for exact replay.  No PPO, matcher-radius change, threshold change,
  or result overwrite is authorized by this acceleration.
- Paired A/B smoke initially exposed different outcomes even when both arms
  referenced the identical C_L, because two separately constructed MJX physics
  environments were not a valid paired control.  The evaluator now shares the
  exact physics env, inference, state and seed and varies only the immutable
  matcher.  The repeated identical-bank control is exactly equal and PASS;
  formal canonical/extended evaluation is forbidden unless this control holds.
- H4 formal-code replay result: among the isolated six candidates, only
  171/192 branch terminal outcomes (89.0625%) and 110/192 complete evidence
  records (57.2917%) reproduce the original formal construction despite fixed
  seeds and unchanged hashes.  The eight H4 events themselves reproduce
  exactly in 3/8 formal replays.  They are therefore classified as
  runtime-sensitive boundary evidence, not an event-order bug; individual
  branch physical-to-handoff pairing is ineligible for causal claims.  H4 is
  excluded from entry proposals.  H1 proceeds only as independently recertified
  contact-state support, so the new C_L decision does not depend on exact replay.

## Stage-local controller pilot gate (2026-07-19)

- HEAD implementation commits: `f198043` pure-JAX next-stage reward,
  `aa8b324` reward preflight/tests, and `9b2bae1` resumable bounded pilot
  controller.  Runtime gate is PASS/current; the configured venv and
  authoritative XML were not modified.
- CPU reward preflight is PASS for Takeoff->Ascent, Ascent->Apex,
  Apex->Descent, and Descent->Landing.  All terms are finite/bounded, the
  entry event dominates shaping, correct direction beats the adverse case,
  neutral/random actions do not create false success, terminal classes are
  mutually exclusive, and Landing recovery is not a physical failure.
  Artifact: `stage_objective_controller_pilot/reward_preflight_v3.json`.
- The fixed Takeoff pilot support is 6 unique states x 4 branches, bank hash
  `ae654d5...ddf8f`; both blocks use the same config hash
  `e225ca1...30ac`.  Block 1 (6,400 effective steps) reached Ascent on 1/24
  branches from 1/6 states at step 5, policy `24f5c72...b81873`, with no
  nonfinite/OOM/timeout.  Failures were missed liftoff 9, missed wheel
  clearance 6, pitch limit 6, roll limit 1, and positive-pitch failure 1.
- Per the one-success rule, block 2 was a cumulative continuation to 12,800
  total steps with unchanged reward semantics.  It reached Ascent on 0/24
  branches from 0/6 states, policy `b27148b...bbe28`; failures were missed
  liftoff 13, missed wheel clearance 6, positive-pitch failure 4, and pitch
  limit 1.  No nonfinite/OOM/timeout occurred.  Dense shaping increased
  (episode 0.6695 -> 0.8024) while entry event fell to zero, so reward growth
  did not substitute for the fixed next-stage gate.
- Formal bounded decision: `gate_pause`, controller exit 40,
  `takeoff_stage_controller_blocker`.  Takeoff did not meet the required >=2
  successful unique states after two 6,400-step blocks.  No policy entered the
  controller proposal bank; Ascent/Apex PPO and 100--200-state label pilots
  were not started.  Next work requires a research decision on Takeoff reset,
  entry detector, or physical stage definition; increasing PPO budget is not
  authorized by this result.

## Decoupled bootstrap-expert route (2026-07-20)

- The sequential shared-Actor Flight-retention route and the bounded
  stage-local Takeoff blocker remain preserved as engineering diagnostics but
  are no longer active training evidence.  The active method is independent
  bootstrap experts `pi_A -> pi_T -> pi_F -> pi_L`, irreversible canonical
  handoffs, expert-conditioned provisional envelopes, phase-balanced
  distillation plus joint RSI PPO, then fresh final shared-policy branch
  recertification.
- Frozen contract PASS at
  `runs/decoupled_bootstrap_seed0_20260720/frozen/frozen_contract.json`:
  `pi_L=fa3a518...34bb7e`, initial independent
  `pi_F=35fcb613...7000c`, fixed canonical
  `C_L=185164d...2aa41` with 99 Final-safe entries, matcher radius
  1.1067926888 and three-step window, and Flight bank
  `2d5d7de...f62934`.  XML, action mapping, Actor schema and PolicyState
  history schema agree.  Landing retention is explicitly not a pi_F gate.
- Runtime gate PASS/current after the expert gate change; source fingerprint
  `8eb0cec...2f9fd`.  Pure/targeted verification passed.  Composite handoff
  preserves uninterrupted physics and PolicyState/history and switches
  irreversibly from pi_F to frozen pi_L.
- Current-runtime initial composite evaluation on the fixed 160 Flight states:
  Chain=7.5%, composite Final=7.5%, Chain-missed Final=2.5%, physical
  failure=92.5%, timeout=0.  Descent is already supported (Chain/Final
  15.584%/15.584% over 77 states); Apex and Ascent remain 0/20 and 0/63.
  Therefore late-descent and descent PPO are skipped as already reachable.
  The next active gap is Apex; its pretraining fixed-bank evaluation is in
  progress under the persistent decoupled controller.
- Composite-policy branch evidence must use artifact role
  `expert_conditioned_provisional_envelope` and is never final JEL evidence.
  The frozen final shared Actor receives new independent certification under
  `final_shared_policy_jel`; expert labels do not survive consolidation.
- Decoupled Flight Apex gate result: late-descent and descent were skipped
  without PPO because the frozen initial pi_F already produced canonical C_L
  Chain support.  Apex then used four cumulative 25,600-step blocks under
  `chain_only`, with no Landing rehearsal or Landing-retention requirement.
  Apex remained Chain=Final=0/20 and physical failure=100% at every block.
  Full-bank Chain/Final changed 7.5%/7.5% baseline -> 6.25%/6.25% ->
  4.375%/5.0% -> 5.0%/4.375% -> 3.125%/3.75%; timeout and nonfinite remained
  zero and provenance remained current.  Final full-bank terminations are
  pitch/roll/recovery=100/54/6.
- The persistent controller stopped normally with exit 40 at
  `flight_apex_expert_chain_blocker`.  The four failed Apex checkpoints are
  diagnostics and are not frozen experts.  The last valid pi_F therefore
  remains the immutable initialization `35fcb613...7000c`; Ascent,
  provisional-envelope certification, pi_T/pi_A, distillation and joint RSI
  were not started.  This demonstrates that decoupling removes the Landing
  forgetting constraint but does not by itself make the current Apex reset
  support reach C_L; adding PPO budget is not authorized by this gate.

## MJX continuous Descent-support bridge (2026-07-24)

- Deterministic CG runtime remains authoritative for development; XML
  `d7e9f43...ce794c`, support bank `1c39f9e...44dde`, matcher
  `56570a4...1630`, frozen Descent policy `5272166...35f2`, and frozen
  Landing policy `fa3a518...bb7e` were not changed.  PPO authorization remains
  false.
- Targeted frozen-asset replay PASS: 8 representative support states, 6/8
  Descent-to-Landing Final-Recovery, 2/8 roll failures, zero reset shock, and
  maximum 37 stable Descent ticks.  The frozen assets are therefore compatible
  with the current CG runtime.
- The nominal natural trajectory's closest support distance was 7.074872 at
  local tick 15 (Apex tick 3, physical handoff tick 4); pitch_limit occurred at
  tick 28.  The shortest useful action window was 8 ticks.  Knee action changed
  vx by +0.5068, wheel speed by +3.6364, z by +0.0334 and distance by -1.6771
  per action unit in the local finite difference; drive was already saturated.
- Bounded 4-D two-segment residual search improved the minimum distance to
  6.586300 with 20/20 bit-exact natural-start replay.  Dominant remaining
  squared normalized contributions were wheel speed 14.907, vx 10.896, roll
  5.269, z 3.048, x 2.916 and yaw 2.393.  No formal support entry, frozen
  Descent handoff, Landing, or Final-Recovery occurred.
- Bounded extensions exhausted the local family without a continuing joint
  trend: earlier window 6.8925, pulse-inclusive window 6.8428, larger residual
  bound 6.5863, and three-segment pitch guard 7.0782/roll_limit.  Delaying the
  launch by 1--6 ticks always caused missed liftoff.  A final six-dimensional
  search added hip/knee control over the last four pre-event Approach ticks;
  both optimized pre-event residuals remained exactly zero and the result was
  identical to 6.5863, confirming the momentum plateau within the authorized
  action family.  Current state is
  `gate_pause: bounded_local_bridge_plateau`; matcher, failure definition,
  frozen policies, XML/action limits and PPO authorization remain unchanged.
  Compact artifact:
  `runs/mjx_continuous_pipeline_repair_v1/descent_support_entry_bridge_v1/bounded_summary.json`.

## Descent provisional candidate bank (2026-07-27)

- The completed `bounded_local_bridge_plateau` result is retained but is now
  `superseded_as_training_gate`.  Old-support distance 6.586300, formal entry
  0, frozen-pi_D stable ticks 0, Landing 0 and Final-Recovery 0 are reference
  diagnostics, not candidate-bank blockers.  Frozen pi_D/pi_L, XML, action
  mapping, matcher and physical-failure semantics remain unchanged; PPO
  authorization remains false.
- `smoke_100_v1` contains 100 byte-identical trajectories: 4,100 stored state
  rows reduce to 41 byte-exact and 41 tolerance-unique states (41 ticks per
  trajectory, including 11 physical-Descent ticks).  It is repeatability
  evidence, not 100-parent diversity evidence.
- Provisional bank v2 is a `proposal_support_bank`, never a certified Tube or
  JEL artifact.  Hash `8e6342b...45a1`; 58 raw states (46 natural-continuous,
  12 local-RSI perturbations) yielded 14 byte/tolerance-unique eligible
  candidates in 9 clusters: core/frontier=3/11 and early/middle/late=4/4/6.
  Selected provenance is natural/local=12/2 (85.714%/14.286%).
- Short-horizon survivability is 14/14 at 8 ticks, 3/14 at 16 ticks and 1/14
  at 24 ticks; exact reset replay passed 14/14.  Physical rejection reasons
  among raw proposals were angular-rate margin 14, body clearance 5, absent
  local action authority 5 and wheel-terrain contact 5.  Fifteen additional
  rows were byte duplicates; no tolerance-only duplicates remained.
- RSI interface smoke PASS: all 14 records were reloadable; 240 resumable
  stratified draws covered every available core/frontier-time stratum.  JIT
  batch reset/step, finite state, observation/history/delay shapes, phase,
  action response and controlled batch cross-talk checks passed.  No state
  terminated during the five-step interface smoke.  Training interface is
  ready, but unified-policy RSI PPO requires separate authorization.
- Artifacts:
  `runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl`,
  `construction_report_v2.json`, and `rsi_smoke_report_v6.json`.  The failed
  v1 bank and earlier RSI diagnostic reports are preserved and not overwritten.

## Unified Descent RSI learnability pilot v1 (2026-07-27)

- The single authorized seed-0 pilot started from HEAD `04e1a2c`; runtime code
  was validated and committed at `e1f48cc`.  The immutable candidate bank hash
  remained `8e6342b...45a1`, runtime gate PASS/current fingerprint
  `763ad29...9155`, and frozen pi_D parameter copy
  `5272166...353f2` loaded with 100% compatible parameters.  PPO ran exactly
  6,400 effective steps: four intervals of `32*25*2=1,600` steps.
- Pointwise 8/16/24-tick survival changed `14/1/1` at checkpoint 0 to
  `14/0/0` at 1,600 and remained `14/0/0` through 6,400.  Median survival
  increased 10 -> 11 ticks, but the sole 24-tick middle/frontier survivor was
  lost.  Core median improved 9 -> 12 while core 8/16/24 stayed `3/0/0`;
  frontier stayed 11/11 at 8 ticks but fell `1/1 -> 0/0` at 16/24 ticks.
- Held-out evaluation used 10 legal sidecar states from 4 clusters.  Its
  8/16/24 counts stayed `6/0/0`; median increased only 8 -> 9 ticks.  In-bank
  terminations changed from pitch/roll/horizon=`12/1/1` to pitch=`14`.
  Action saturation fell 3.53% -> 0%, with no NaN/OOM/timeout or value
  collapse, but evaluation reward ended below baseline.  KL spiked to
  8169.58 on the first update and stabilized to 0.0932 by the final update.
- The exact reset audit sampled every candidate and achieved core/frontier
  4530/1870 draws (70.8%/29.2%).  A protocol limitation is recorded: frontier
  candidates were uniform, so frontier early/middle/late draws were
  182/660/1028 rather than temporal-layer equal.  No candidate was omitted or
  dominated within its label, but this must be corrected before any future
  run.  Final-policy expansion yielded 0 legal proposals.
- Formal result: `no_learnability_signal`; no reward hacking, but no reliable
  survival/generalization gain.  Longer training, another seed, Tube/JEL
  certification and automatic expansion are not recommended or authorized.
  PPO authorization is restored to false.  Full report:
  `runs/unified_descent_rsi_learnability_pilot_v1_seed0_20260727/UNIFIED_DESCENT_RSI_LEARNABILITY_PILOT_V1_REPORT.json`.

## Unified Descent RSI update-integrity repair v1 (2026-07-27)

- Authoritative start `8512e42`; bank `8e6342b...45a1`, XML/action mapping,
  failure semantics and frozen pi_D remained immutable. Runtime gate is
  PASS/current. The previous outcome is scoped as
  `no_learnability_signal_under_current_update_protocol_and_budget`.
- Effective-step accounting PASS: four 50-env x 32-tick rollouts are 6,400
  unique environment transitions. Two optimizer passes produce 16 gradient
  steps and 12,800 transition uses but no extra environment transitions.
- Frozen pi_D normalizer loaded correctly (hash `8f2e36b...93a7e`, count
  1,024,000). Old Brax updated it before same-batch loss, producing KL
  7,756.77 and losing the sole 16/24-tick survivor. With one fixed snapshot,
  old-logprob recomputation gives sample KL 1.13e-6 and analytic KL 3.99e-5.
- Optimizer-only remains invalid: analytic KL 184.45 and maximum deterministic
  action change 1.088. Existing desired-KL 0.01 rollback rejects all four
  proposed first-batch steps (candidate KL 140.89--192.35), restores
  optimizer/normalizer exactly, and preserves baseline survival `14/1/1`.
- Hierarchical RSI reset is repaired: core/frontier reset draws 4530/1870;
  frontier early/middle/late 661/616/593. Transition occupancy is 996/604 and
  differs legitimately with episode length; resume sequence is byte-exact.
- Phase A `FAIL` only at `effective_optimizer_update_under_fixed_hyperparameters`.
  Phase B 6,400-step rerun was not started; no held-out/checkpoint rerun results
  exist. PPO authorization is false and 25,600 steps are not recommended.
  Next work requires separate authorization of an optimizer/trust-region
  protocol change; no LR/clip/network/reward sweep was performed. Compact
  report: `docs/experiments/unified_descent_rsi_update_integrity_repair_v1/`.

## Unified Descent RSI optimizer trust-region repair v1 (2026-07-27)

- Authoritative start `e61639f`; immutable bank `8e6342b...45a1`, XML/action,
  failure semantics, frozen pi_D `5272166...353f2` and normalizer
  `8f2e36b...93a7e` (count 1,024,000) remained unchanged.
- Saved-rollout five-point calibration reconstructed prior advantage/return
  exactly without new training transitions. Reference LR `1e-4`,
  `m0=0.005206431959`; all fixed candidates `2.6032e-7` through `7.8096e-7`
  passed KL/action/survivor/finite/repeat gates. Maximum passing LR
  `7.809647938e-7` was selected independently of held-out and reward.
- The single authorized seed-0 rerun completed exactly 6,400 transitions:
  17 attempted optimizer updates, 16 accepted, one rollback and one permitted
  halving to `3.904823969e-7`. Normalizer hash was unchanged at every
  checkpoint; no NaN/OOM/timeout/provenance error occurred.
- Checkpoints 0/1600/3200/4800/6400 all remained in-bank `14/1/1`, median 10,
  lower quartile 9, with pitch/roll/horizon `12/1/1`. Held-out remained
  `6/0/0`, median 8, lower quartile 7. Core/frontier and all temporal strata
  were unchanged. The unique long survivor no longer disappears, but no
  positive physical learning signal emerged.
- Classification: `update_integrity_pass_but_no_learning_signal`. Do not run
  25,600 steps. Next diagnosis is reward controllability, pi_D initialization
  compatibility, and candidate horizon/curriculum difficulty. PPO
  authorization is false; no Tube/JEL/second-seed work started. Report:
  `docs/experiments/unified_descent_rsi_optimizer_trust_region_repair_v1/`.

## Unified Descent controllability/reward/curriculum probe v1 (2026-07-27)

- Authoritative start `3aa5758`; immutable bank `8e6342b...45a1`, XML/action
  mapping, formal failure semantics, frozen pi_D `5272166...353f2` and
  normalizer `8f2e36b...93a7e` (count 1,024,000) remained unchanged.
- Checkpoint 6400 relative to checkpoint 0 has analytic/sample KL
  `0.00413/0.00634`, deterministic action max/RMS delta
  `0.00505/0.000971`, actor relative L2 `2.70e-5`, and value-prediction RMS
  delta `0.0192`. The accepted updates were real but did not alter discrete
  candidate survival.
- The local action Jacobian is full rank from tick 2 but strongly
  ill-conditioned. No isolated action channel added more than one tick; no
  mapping, sign, dead-knee, or control-delay anomaly was found. Effective
  recovery requires coordinated time-structured action changes.
- Reward-free bounded CEM gives 8/14 states reaching 16 ticks, 3/14 reaching
  24, median gain 4.5, and 9/14 gaining at least four ticks. Exact replay is
  14/14. Reward alignment passes on 51,200 sequences: candidate-stratified
  Spearman 0.7661 and pairwise accuracy 0.8333.
- Final classification is `C: initialization_or_exploration_gap`. Only 1/9
  major improvements has residual RMS within the policy's 2-sigma limit
  (`0.10`); most require bound `0.20` and coordinated corrections beginning
  at tick 0. Phase-B curriculum PPO was not run, PPO authorization remains
  false, and no Tube/JEL claim was made. Proposed next work is a separately
  authorized short-horizon supervised CEM-teacher bootstrap, not more ordinary
  PPO. Targeted tests are 2 passed and the final GPU local preflight is
  293 passed (one external JAXopt deprecation warning). Compact report:
  `docs/experiments/unified_descent_controllability_reward_curriculum_probe_v1/`.

## Unified Descent CEM-teacher bootstrap/local-PPO probe v1 (2026-07-27)

- Started from `ff3e3a1`; immutable bank `8e6342b...45a1`, XML/action,
  failure semantics, frozen pi_D `5272166...353f2`, and normalizer
  `8f2e36b...93a7e` (count 1,024,000) remained unchanged.
- Corrected replay validation found that the prior CEM `exact_replay` flag
  checked repeat1==repeat2 but not replay==selected summary. Repeats are 14/14
  deterministic, but only 12/14 selected summaries match. `da0679b1` changes
  from saved 9->14 to actual 9->10 ticks, so valid gain>=4 teachers fall from
  nine to eight.
- The integrity-valid subset has 64 first-8-tick teacher samples and 198
  anchors. Action equation alignment is exact and representability passes
  (17.19% close opposite conflicts under the declared 20% limit), but the
  fixed nine-teacher source-authority contract fails.
- No gradient update, three-fold CV, student relabel, hidden-block unfreeze,
  held-out Descent evaluation, or PPO transition ran. PPO authorization is
  false. Frozen pi_D Landing baseline is 81/96 Final-Recovery, 15 roll
  failures, zero timeout. Targeted tests are 7 passed and the final GPU local
  preflight is 298 passed (one external JAXopt deprecation warning). Compact
  report:
  `docs/experiments/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v1/`.

## Unified Descent eight-teacher bootstrap/local-PPO probe v2 (2026-07-27)

- Started from `55284df`; the candidate bank `8e6342b...45a1`, authoritative
  XML/action mapping, frozen pi_D, normalizer `8f2e36b...93a7e`, and ten-state
  held-out set remained immutable. Exact replay admits exactly eight teachers.
  The mismatches are `da0679b1` (selected 14, exact 10; gain 5 -> 1, removed)
  and `173ee307` (18 -> 17; gain 5 -> 4, retained).
- The 64 teacher / 198 anchor dataset passes the declared representation gate
  only marginally: close opposite-label conflict is 17.1875% versus the 20%
  limit. The frozen three candidate-level folds are 3/3/2 states.
- Head-only CV reduced imitation error but could not satisfy anchor and
  physical-transfer constraints together. The one authorized support-gated
  student relabel round audited 40 visited states and admitted 11; it used no
  excluded-fold or held-out samples.
- Final excluded-state gains were Fold A `[1,1,0]`, Fold B `[0,0,0]`, and
  Fold C `[0,0]`. Combined `gain>=2` is 0/8, median gain 0, and only one fold
  has positive median gain. Fold A's best diagnostic checkpoint also violates
  the anchor max-action-drift gate; no new failure type appeared.
- Classification: `teacher_memorization_or_support_gap`. The relabel allowance
  is consumed; last-hidden-block training was not triggered. Final bootstrap,
  held-out evaluation, Landing retention, and conditional PPO were not run.
  PPO authorization is false and the held-out set remains sealed.
- A monolithic CV process encountered an engineering OOM. Independent fold
  processes reused completed checkpoints and retried only the incomplete LR
  shard without overwriting results; all three folds then completed. Report:
  `docs/experiments/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v2_8teacher/`.

## Unified Descent feedback-teacher support probe v1 (2026-07-27)

- Started from `439aa8f`; frozen pi_D/normalizer/critic/log_std, the eight
  exact-replay teacher assets, candidate/XML/action provenance, and ten-state
  held-out set remained immutable. PPO authorization stayed false.
- Frozen diagnostic support contains 24 snapshots, exactly three per teacher
  candidate. The quota can retain at most 10/11 historical accepted relabels
  because one candidate owns four; deterministic selection retained those ten,
  five audited rejects, and nine states for the previously excluded candidates.
- A uniform batch-256 discovery versus batch-1 replay-summary mismatch was
  isolated before making a research conclusion. CEM was not rerun. Authority
  was recomputed with two bit-exact batch-1 replays from each same snapshot.
- Local authority passes 12/24 snapshots, but only 3/8 candidates have at
  least 2/3 passing snapshots, below the fixed 16/24 and 6/8 gates. Five
  retained rejected-relabel states were tested; three now have authoritative
  local corrections, showing limited support gaps but not balanced coverage.
- The 12 passing snapshots have 12/12 valid real medoids, 12/12 valid action
  means, 11/12 valid action medians, and zero detected opposite successful
  clusters. The blocker is not widespread action non-identifiability.
- Classification: `BRITTLE_OPEN_LOOP_TEACHER`. Receding-horizon oracle, H/L
  candidate CV, held-out evaluation, Landing retention, final bootstrap, and
  PPO were not executed. Report:
  `docs/experiments/unified_descent_feedback_teacher_support_and_representation_probe_v1/`.

## Unified Descent correction-transfer/support-geometry audit v1 (2026-07-28)

- Started from `eff25d5`; the 24 frozen snapshots, 12 authoritative medoid
  corrections, frozen pi_D and all asset hashes remained read-only. No CEM,
  relabel, network training, held-out, Landing retention, or PPO was run.
- Authority accounting is exact: historical accepted/rejected/newly frozen
  contribute 4/3/5 passes from totals 10/5/9. Six historical accepts do not
  reproduce. Candidate counts are `[3,0,1,2,1,1,1,3]`; rejected passes are
  `485551c8` ticks 1 and 4 and `d7faea27` tick 1.
- Fixed compatibility conditions admit 244 double-replay pairs. Physical
  transfers are diagonal 12/12, same-candidate off-diagonal 3/18, and
  cross-candidate 40/214. Cross transfer reaches six target candidates and
  forms a seven-candidate weak component; unsupported `173ee307` is isolated.
  Robust-core -> frontier/sparse is 14/71; the reverse direction is 6/30.
- Candidate-grouped actor-visible linear diagnostics have balanced accuracy /
  precision / recall `0.667/0.700/0.583` and fail the fixed separability gate.
  Privileged features reach `0.750/0.800/0.667` and exceed permutation p95.
- Classification: `ACTOR_OBSERVATION_INFORMATION_GAP`. Under the current
  observation contract the CEM action-regression/bootstrap route is closed;
  only a separate observation/history sufficiency audit is permitted. These
  are provisional feedback-support results, not a Tube/JEL claim. PPO
  authorization remains false. Report:
  `docs/experiments/unified_descent_feedback_correction_transfer_and_support_geometry_audit_v1/`.

## Unified Descent observation/history sufficiency audit v1 (2026-07-28)

- Started from `a781250`; frozen pi_D/normalizer/critic/log_std, 24 snapshots,
  12 corrections, 244 transfer pairs, and held-out states remained read-only.
- Saved actor observations and actions restore 24/24 exactly only through the
  explicit `actor_observation` sidecar override. Independent reconstruction
  from physical state plus saved PolicyState is 0/24 exact for observation and
  0/24 for action, despite each path being internally deterministic.
- Observation max error is `29.225--464.311`; deterministic action max error
  is `0.00474--0.18995`. In all 24 states, saved `obs_history` equals saved
  observation frames 1--3 (post-current), while reconstruction requires frames
  0--2 (pre-current). This is a consistent snapshot history off-by-one.
- The compatibility `delay_buffer` is exactly `phase_probs[None,:]`, not an
  actuator-delay FIFO. Phase/contact estimator fields and qacc warmstart are
  present; action order and frozen normalizer hash remain unchanged.
- Classification: `OBSERVATION_PIPELINE_AUTHORITY_FAILURE`. Per the protocol,
  V0/P0, V1--V5, alias pairs, privileged reconstructability, and pairwise
  transfer diagnostics were not run. This version provides no observation
  amendment evidence. A new audit version must first correct snapshot history/
  schema; old policy, Tube, Final-Recovery, and JEL evidence cannot be inherited
  automatically. PPO and bootstrap authorization are false. Report:
  `docs/experiments/unified_descent_observation_history_sufficiency_audit_v1/`.

## Timing-explicit snapshot schema v4 and delay reaudit v1 (2026-07-28)

- Schema/runtime implementation commits: `0d0d58d`, `864654f`, and
  `f923161`. Full local preflight passed 333 tests; dynamic runtime gate is
  PASS/current with source fingerprint `8fc490e...e6b5f`.
- Construction-lineage smoke recovered 8/8 frozen candidates with real
  three-packet FIFO and exact logged actor input. Formal recapture then passed
  15/24 states, below the fixed 24/24 inheritance gate.
- All 24 physical hashes, logged observations, frozen pi_D actions,
  candidate/tick identities, post histories and semantic hashes match. Nine
  states fail only independent actor/current-frame bit-exact reconstruction.
- Classification: `V4_RECAPTURE_IDENTITY_FAILURE`. The 12 corrections and 244
  pairs were not inherited; L0/D1/D2/J12, transfer and held-out were not run.
  No training, CEM, PPO, bootstrap, Landing retention or Tube/JEL promotion
  occurred. Controller/worker are stopped; watchdog timer remains waiting.
- Result: `runs/timing_explicit_snapshot_schema_v4_and_delay_reaudit_v1/`.

## V4 current-frame independent reconstruction localization v1 (2026-07-28)

- Starting at `0be383f`, fieldwise localization found all nine mismatches in
  the single field `task_distance_to_front` (frame index 18), each exactly one
  float32 ULP (`7.45058e-09`). The other 831/840 values were already exact.
- All causal inputs and `(step_front_x - qpos_x)` were bit-exact. The first
  divergence was context-dependent lowering of division by `3.0`: fused online
  step matched explicit float32 reciprocal multiplication, while independent
  restore matched strict materialized float32 division. Classification:
  `DTYPE_DEVICE_OPERATION_ORDER_NUMERICAL_LINEAGE`.
- Commit `10baf34` makes the shared canonical producer use explicit float32
  scaling and prevents independent restore from copying logged frame/actor
  sidecars. No causal state or schema extension was required; no estimator
  pre/post mix or duplicated producer existed.
- Targeted tests passed 57, full local preflight passed 347, and dynamic
  runtime gate is PASS/current (`ce28878...b914d`). Full natural-lineage
  recapture now passes 24/24 snapshots and 840/840 frame dimensions.
- Classification: `V4_RECAPTURE_IDENTITY_GATE_PASS`. A later independent run
  may inherit correction/pair identity, but this run did not load or execute
  them and did not run L0/D1/D2/J12. Training/PPO/bootstrap/CEM remain false;
  held-out was not read. Controller/worker/GPU are idle; watchdog is unchanged.
- Result:
  `runs/v4_current_frame_independent_reconstruction_localization_v1/`.

## Unified Descent timing-explicit packet-delay reaudit v1 (2026-07-28)

- Started from `948fb24`; preregistration hash is `b7f70fd...3cab16`.
  Frozen v4 snapshots, 12 corrections, 244 pair identities, pi_D,
  normalizer, XML and action mapping passed the inherited/runtime identity
  gates. L0 then reproduced 12/12 corrections and 24/24 packet identities;
  all action/ctrl/packet active prefixes repeat bit-exact.
- Local authority is L0/D1/D2/J12 = `12/8/7/9` of 12. D1 and D2 fail the
  fixed stability gate (authority retention `0.667/0.583`, three candidate
  layer changes each); J12 passes its fixed aggregate gate at `9/12` with two
  layer changes. A new delayed-mode `roll_limit` failure appears.
- Transfer successes are L0/D1/D2/J12 = `55/82/86/100` of 244. Relative to
  L0, D1 has intersection/lost/gained/Jaccard `48/7/34/0.539`, D2
  `39/16/47/0.382`, and J12 `48/7/52/0.449`. The increases do not imply
  recovery: all modes have zero Chain, recovery success and Final-Recovery.
  Landing-entry counts are `0/4/10/6`; horizon counts are `1/6/14/5`.
- Classification: `DELAY_SENSITIVE_FEEDBACK_SUPPORT`. Evidence remains
  provisional feedback support only; no Tube/JEL was formed. A separate
  `pi_D_delay_aware_v1` stage is recommended because true FIFO and L0 identity
  passed, D2 failed the preregistered gate, and Final-Recovery remains zero;
  no policy was created here. Training/PPO/bootstrap/new-CEM are false and
  held-out was not read.
- Implementation/report commits: `30ef244`, `4a9b9a3`, `d754bd3`. Full local
  preflight passed 347 tests and the dynamic runtime gate is PASS/current.
  Controller/worker/GPU are idle; watchdog timer remains waiting. Result:
  `runs/unified_descent_timing_explicit_packet_delay_reaudit_v1/`.

## Backward recovery Tube fast-track v1 — Descent blocker (2026-07-28)

- Start `974a71c`; current HEAD `7940098`. Frozen XML, action mapping, pi_L,
  canonical C_L (99 Final-safe entries), pi_D and matcher hashes remained
  unchanged. L0 only; held-out and delay modes were not read.
- Recomputed 115 nominal proposal results plus 14 bounded CEM searches. The
  current provisional construction has 30 P0 and 15 P1 nodes, 6 candidates,
  4 layers and early/middle/late coverage. Parent lineage is complete 30/30
  and pointwise replay precision is 100%.
- `DESCENT_TUBE_RSI_START_GATE=FAIL`: P1 count is 15/16 and one candidate owns
  7/15 nodes (46.7%, above the 35% cap). Thus the gap is at least five diverse
  P1 nodes, not merely one numerical count.
- First-tier CEM searched 11 states and produced 4 P0 / 1 P1. The fixed
  second-tier 128x6, 12-tick search on three closest first-tier failures
  produced 0 P0 / 0 P1. No proposal received more than its allowed tiers.
- The single authorized 6,400-step P0-seeded RSI pilot completed without NaN,
  OOM or timeout but regressed fresh composite certification from P0/P1
  26/14 to 23/12. Its checkpoint is rejected as a Tube/policy successor and
  PPO authorization remains false.
- No unused, time-aligned predecessor of the 15 P1 nodes exists in the fixed
  proposal index, including parent/neighbor-anchor matching. Creating a new
  predecessor shell would require a newly specified, physically validated
  construction source; mixed reference states are forbidden.
- Phase blocker: `DESCENT_DIVERSE_P1_SUPPORT_EXHAUSTED`. Apex, Ascent,
  Takeoff, Approach and nominal full-chain construction were not started.
  Controller/worker are inactive; watchdog timer remains active. Results:
  `runs/backward_recovery_tube_fast_track_v1/`.

## Descent diverse-P1 predecessor recovery v1 (2026-07-28)

- Started from clean `b6e20ce`; authoritative XML/action mapping, frozen pi_D,
  pi_L, canonical C_L (99 Final-safe entries), matcher and v4 contract stayed
  unchanged. Runtime gate remains PASS/current (`ce28878...b914d`); held-out
  and delay modes were not used.
- Deterministic launch selection retained the complete 15-node history and
  formed a 13-node initial subset by retaining all eight non-dominant nodes
  and five diversity-selected dominant nodes. The preregistered gap was three
  non-dominant P1 nodes.
- Source-A forward MJX harvesting used all 30 successful active prefixes. A
  1/30 pilot produced seven v4 predecessors; the full pass produced 86 unique
  predecessors with 100% v4 independent-reconstruction identity and no state
  splicing. Only three prioritized proposals required certification; all three
  passed P0 and 3/4-or-better P1 and came from previously P1-empty candidate
  `0ba06e...`. Source B/C and new CEM were not needed.
- `DESCENT_BALANCED_P1_LAUNCH_GATE=PASS`: full P1=18, frozen balanced subset=16,
  seven candidates, four layers, early/middle/late, maximum candidate share
  5/16=31.25%, parent lineage complete and controller-conditioned pointwise
  Final-Recovery precision 100%. These artifacts remain nominal provisional
  launch support, not a formal Tube/JEL.
- Candidate-balanced behavior data passed with 16 teacher states (maximum
  candidate share 31.25%), eight one-per-candidate P0 anchors and 16 separate
  frozen-pi_L transition contracts. The first imbalanced 15-anchor artifact is
  preserved but superseded for training.
- The constrained RSI pilot stopped before PPO. Frozen pi_D policy replay gave
  balanced P0/P1=15/15 of 16 and fixed-bank P0/P1=17/17 of 18. The preregistered
  100-step head-only behavior prefit passed action drift (anchor RMS `0.00567`,
  max `0.01972`) but reduced balanced P0/P1 to 13/13 and fixed-bank P0/P1 to
  15/15. The prefit checkpoint was rejected; effective PPO steps are zero,
  block 2 is unauthorized, and no nominal Descent Tube/policy was frozen.
- Final classification: `DESCENT_RSI_RETENTION_FAILURE`. Per the hard stop,
  no learning-rate/rehearsal repair, Apex run, Tube/JEL promotion or additional
  budget was attempted. Controller/worker/GPU are idle; watchdog timer remains
  active. Result: `runs/descent_diverse_p1_predecessor_recovery_v1/`.
