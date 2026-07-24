# DVGC Experiment State

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
