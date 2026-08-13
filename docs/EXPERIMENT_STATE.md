# DVGC Experiment State

## Current method

The approved research direction is the two-phase learned soft-feasibility-Tube
method defined in `PROJECT.md` and `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`:
`Propulsion-Ascent` -> Apex transition band -> `Descent-Recovery`, followed by
`V_up`/`V_down`, soft-Tube guidance, one unified Tube-RSI PPO, and independent
frozen-policy JCE/JEL evaluation.

Gate A static contracts and the Gate B external pure-JAX runtime adapter,
geometry manifest/audit, deterministic guideline threshold builder, natural
reset audit, and timing-explicit snapshot/round-trip contracts are implemented.
The revised Gate B no longer requires a complete guideline open-loop event
sequence or guideline-generated banks. `data/reference_jump.csv` is a
kinematic guideline and weak prior, not an expert or authoritative controller.
Existing five-stage code and results remain legacy migration sources only.

This method is not yet implemented end-to-end: there are no trained phase
experts, feasibility models, learned soft Tubes, or unified two-phase policy.

On 2026-08-12 the user explicitly changed the single authoritative payload
contract from 4.0 kg to 2.0 kg and authorized one fresh Phase U retry up to
1,000,000 training transitions after full validation and a bounded smoke. The
only physical edit is `geom name="load" mass="4.0" -> mass="2.0"`; geometry,
obstacle, +/-50 N m hip/knee limits, action mapping, reward, reset, observation,
thresholds, PPO layout, and safety termination remain fixed. The configured
path `assets/orange_bike_4kg_horizontal.xml` is retained as a historical
filename and now has SHA-256
`e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192`.
All old 4 kg checkpoints, banks, runtime reports, and authorizations remain
immutable provenance and are incompatible inputs for the new run.

The 2 kg static and runtime requalification is complete. Static compilation,
the full 897-test suite, and `scripts/local_preflight.sh` pass. The fresh
managed runtime gate at `runs/two_phase/runtime_gate/phase_u_2kg_20260812/`
passes its model load, timing-explicit v4 snapshot/one-step round trip,
determinism, 64-transition PPO update, and 32-transition resume contracts;
`cli.runtime_gate --check-only` confirms its fingerprint is current. This gate
consumed 96 engineering-integrity transitions and no Phase U training
transitions.

The fixed one-shot 2 kg Gate B audit is retained at
`runs/two_phase/gate_b_2kg_20260812/`. Natural ground support and all three
pure-JAX/host-MuJoCo geometry cross-audit states pass; the largest absolute
clearance difference is `9.831815167560265e-09 m` with matching signs. The
kinematic-guideline open-loop diagnostic reached the jump window, liftoff,
stable-airborne, and ascending events, then ended at `roll_limit` after 22
environment transitions without Apex or stable recovery. Its two MP4/state
trace diagnostics are preserved. Under the revised Gate B contract this is
reference-controller provenance, not a Phase U training blocker and not a
bank or reachability claim. No retry, action repair, threshold move, or safety
limit change was made.

The fresh run-bound 2 kg Phase U engineering smoke completed at producer HEAD
`4db9f98ac700df851363f33b0baa4fab82a52820`:

```text
run id: gate_c1_phase_u_2kg_env512_smoke_20260812_seed720001
parallel environments: 512
training transitions: 12,800
Brax evaluation transitions: 1,600
fixed evaluation transitions: 216
total environment transitions: 14,616
status: completed
```

The PPO update ran at 890.07 training transitions/s with finite loss, KL, and
policy-distribution statistics. The transition-12,800 checkpoint sidecar passed
recursive identity validation (`843a1f25...ca82432`). Fixed outcome accounting
closed at 8 `other_failure`, 0 physical failure, 0 timeout, and 0 success. All
eight deterministic rollouts reached the legal jump window; none lifted off or
reached Apex, and all ended at `takeoff_missed_liftoff_deadline`. Roll/pitch
violations, illegal contact, action saturation, broadphase overflow, NaN/Inf,
OOM, traceback, timing/history mismatch, and hash mismatch were absent. Eight
MP4/state-trace pairs are retained under the run's `failure_videos/`. This is
engineering qualification only and does not claim learnability or `pi_up`.

The smoke-qualified fresh formal retry has been launched once as a persistent,
resumable process:

```text
run id: phase_u_2kg_env512_998400_20260812_seed720002
producer HEAD: 7a61860ed26d430434ae6e168ac29b2b9b773065
source-tree hash: a06aa44947783f79172d546f23c84f0906ccaa5ec71632b312ce50c9128a3a4e
startup PID: 693476
authorized/effective training ceiling: 1,000,000 / 998,400
effective checkpoints: 0/102,400/256,000/512,000/755,200/998,400
status: runs/two_phase/phase_experts/phase_u_2kg_env512_998400_20260812_seed720002/status.json
metrics: runs/two_phase/phase_experts/phase_u_2kg_env512_998400_20260812_seed720002/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_2kg_env512_998400_20260812_seed720002.log
control/resume: runs/two_phase/process_logs/phase_u_2kg_env512_998400_20260812_seed720002.control.txt
```

The single startup inspection found the PID live, `status=running`, a complete
transition-0 checkpoint and sidecar, matching 2 kg/source/config/threshold
identities, and no broadphase, numerical, timing/history, or hash fault. Based
on the measured 890 transitions/s plus fixed-evaluation/video overhead, the
next inspection is scheduled for an approximately 8--15 minute checkpoint or
terminal window. No continuous log polling is authorized. Candidate snapshot
and bounded continuation hooks remain gated on real held-out Apex success and
independent parent diversity; nothing here declares `pi_up_star`, formal
`V_up`, or a Soft Tube.

The 2 kg formal retry subsequently stopped at its complete 755,200-transition
checkpoint with `status=gate_pause` and
`held_out_physical_performance_plateau`; it must not be resumed under the
consumed authorization. Its closed checkpoint audit is:

| training transitions | outcomes | window | liftoff | Apex | mean return | mean ticks |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 missed-liftoff | 8/8 | 0/8 | 0/8 | -3.353 | 27 |
| 102,400 | 8 missed-liftoff | 8/8 | 0/8 | 0/8 | -3.751 | 27 |
| 256,000 | 8 pitch-limit | 0/8 | 0/8 | 0/8 | -24.603 | 14 |
| 512,000 | 8 pitch-limit | 0/8 | 0/8 | 0/8 | -23.821 | 10 |
| 755,200 | 8 pitch-limit | 0/8 | 0/8 | 0/8 | -24.737 | 15 |

Every checkpoint and recursive sidecar validates. All 40 held-out failures
have MP4 and timing-aligned NPZ traces. The run consumed 755,200 PPO training
and 744 fixed-evaluation transitions; candidate acquisition and continuation
diagnostics remained zero. There was no broadphase overflow, NaN/Inf, OOM,
hash mismatch, timing/history violation, illegal contact, or action saturation.
The traceback in the log is the intentional checkpoint-gate control path, not
an unhandled numerical failure.

The state traces establish the failure mechanism. At 256k and 512k the
deterministic policy moved the hip target from -1.2 toward approximately -0.52
and -0.09 within two control ticks while the knee target remained near 2.5.
Pitch rate then reached approximately -6 to +10 rad/s and the robot exceeded
the unchanged pitch limit in 10--15 ticks, far before the legal window. The
window/ascent/clearance/Apex reward components were exactly zero on these
held-out failures, so there is no pre-window task-reward leakage. Training-time
stochastic trajectories nevertheless obtained large ascent shaping without a
single Apex completion; the bounded angular-rate cost is only 0.25 per tick
against up to 4.0 ascent plus 2.0 clearance progress per tick.

Two fixed, non-training physical diagnostics tested alternatives without
adapting their grids. A 36-scenario hip/knee coordination grid consumed 1,000
environment transitions: 33 scenarios lifted off and 14 reached stable
airborne, but 23 hit pitch limit, 13 hit retained takeoff safety failures, and
none reached Apex. Increasing knee exploration is therefore rejected. An
18-scenario one-tick hip-impulse grid consumed 507 transitions: all scenarios
lifted off, but only five reached stable airborne and those five later hit
pitch limit; 0.10--0.15 impulses kept pitch around 0.07--0.19 rad but lacked
height. This proves usable low-pitch impulse authority exists but the current
objective does not sufficiently prefer it over high-angular-rate ascent.

The next single hypothesis changes only Phase U
`angular_rate_penalty_weight` from 0.25 to 1.0. It leaves the other reward
terms, exploration prior, reset, threshold/deadline, XML, action mapping,
network, optimizer, horizon, and evaluation unchanged. This change requires a
fresh reward-contract hash, red-green tests, full validation, one new 512-env
smoke, and a new run-bound formal authorization. It does not authorize
resuming the paused run or declaring a Tube.

The single-hypothesis implementation is now source-complete at `c7198ec` with
reward-contract hash
`18ac22843abaa66f97ac0aac85f4cc6577097040f7d5f3c0326a831b6db7b007`.
Red-green contract tests, 94 targeted tests, the full 897-test suite, and local
preflight pass. A fresh managed runtime gate at
`runs/two_phase/runtime_gate/phase_u_2kg_angular_rate_20260812/` also passes in
93.30 seconds, including its 64-transition update and 32-transition resume.
This requalification consumed 96 runtime-smoke transitions and zero Phase U
training transitions. The next permitted action is one fresh run-bound
512-environment Phase U engineering smoke; no formal retry has yet been
authorized or started for this hypothesis.

That fresh smoke completed under run id
`gate_c1_phase_u_2kg_angrate1_env512_smoke_20260812_seed720101`. It consumed
12,800 PPO training, 1,600 Brax evaluation, and 208 fixed-evaluation
transitions (14,608 total). The finite update ran at 910.84 training
transitions/s; the 12,800 checkpoint passed recursive identity validation as
`60787db9...84ef`. Outcome accounting closed at 8 `other_failure`, all due to
`takeoff_missed_liftoff_deadline`: 8/8 reached the legal window, 0/8 lifted
off, and 0/8 reached Apex. There was no physical failure, timeout, roll/pitch
violation, illegal contact, action saturation, broadphase overflow, NaN/Inf,
OOM, timing/history mismatch, or hash mismatch. Eight MP4 and eight aligned
state traces are retained. This is engineering qualification only.

After that qualification, the delegated PPO decision authorized and launched
one new fresh-initialization formal run:

```text
run id: phase_u_2kg_angrate1_env512_998400_20260812_seed720102
producer HEAD: 98e6de4ee492bc4c5e463214eadc51747d9d9197
source-tree hash: e636e4f11b4de701158e9e90be10dd51642087287beebc6a01e707303774a6a8
worker PID: 803812
authorized/effective training ceiling: 1,000,000 / 998,400
effective checkpoints: 0/102,400/256,000/512,000/755,200/998,400
status: runs/two_phase/phase_experts/phase_u_2kg_angrate1_env512_998400_20260812_seed720102/status.json
metrics: runs/two_phase/phase_experts/phase_u_2kg_angrate1_env512_998400_20260812_seed720102/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_2kg_angrate1_env512_998400_20260812_seed720102.log
control/resume: runs/two_phase/process_logs/phase_u_2kg_angrate1_env512_998400_20260812_seed720102.control.txt
```

The single startup audit found the worker live with matching source, model,
threshold, and training-config identities, a complete transition-0 checkpoint,
and `status=running`. The initial fixed evaluation consumed 216 transitions.
The estimated terminal/Gate-Pause window is approximately 18--30 minutes after
startup. Monitoring is sparse: inspect only terminal state, an explicit
checkpoint milestone, or abnormal exit. Candidate acquisition and bounded
continuation probing remain automatically gated on real held-out Apex success
and independent parent diversity; this run does not authorize a formal
`V_up`, Soft Tube, Phase D training, or unified PPO.

That formal run subsequently entered `gate_pause` at its complete 755,200
checkpoint with `held_out_physical_performance_plateau`; it must not be resumed
under the consumed authorization. It used 755,200 PPO training and 656 fixed
evaluation transitions, with zero candidate-acquisition and continuation
transitions. All 60 checkpoint sidecars pass recursive validation, and all 40
fixed-evaluation failures have an MP4 and aligned NPZ trace. There was no
NaN/Inf, overflow, OOM, identity mismatch, timing/history fault, illegal
contact, or action saturation.

The 0 and 102,400 checkpoints reached the legal window in 8/8 rollouts but did
not lift off. The 256,000, 512,000, and 755,200 checkpoints each ended 8/8 at
`pitch_limit`, with peak angular speeds 11.91, 24.66, and 24.25 rad/s. The
755,200 trace moves the hip target from -1.2 to +0.43 rad in two ticks and then
back near -1.21 rad. The current angular-rate term clips at one Apex threshold
(1.2466 rad/s), so it assigns the same per-tick cost to modest and 19.45x
threshold exceedance. Offline fixed-trace scoring shows that changing only the
cap ratio from 1 to 8 leaves the stable 0/102.4k traces unchanged while adding
approximately 49.86/51.49/52.23 cost to the three pitch-failure traces.

The next single hypothesis therefore adds the bounded, hashed
`angular_rate_penalty_cap_ratio` and sets it to 8.0 while retaining weight 1.0.
No other reward, exploration, PPO, reset, threshold, model, or runtime contract
changes. It requires red-green implementation, complete requalification, a
fresh smoke, and a new run-bound authorization before another formal run.

The cap-ratio implementation is now source-complete with reward-contract hash
`3802c068a068ddfffb27e52fa212fc1813977b03ab7e9de2964a2ef522de3782`.
The default cap remains 1.0 for compatibility; the two stable Phase U configs
explicitly select 8.0. Six focused red-green tests, 99 Phase U regressions,
static compilation, the full 902-test suite, and local preflight pass. A fresh
managed runtime gate at
`runs/two_phase/runtime_gate/phase_u_2kg_angular_rate_cap8_20260812/` passes in
93.70 seconds, including the 64-transition update and 32-transition resume.
This evidence consumes 96 runtime-smoke transitions and no new Phase U PPO
training transitions. The next permitted action is one fresh run-bound
512-environment engineering smoke for the cap-8 hypothesis.

The cap-8 smoke completed as
`gate_c1_phase_u_2kg_angrate_cap8_env512_smoke_20260812_seed720201`: 12,800
training, 1,600 Brax evaluation, and 216 fixed-evaluation transitions (14,616
total). The finite update ran at 903.30 training transitions/s. Its checkpoint
sidecar passes recursive identity validation as `70b61950...adfba`; closed
outcomes are 8 `other_failure`, all `takeoff_missed_liftoff_deadline`. All
eight rollouts reached the legal window, none lifted off or reached Apex, and
there was no physical failure, timeout, saturation, numerical/runtime fault,
or identity mismatch. Eight MP4/NPZ pairs are retained. This is engineering
qualification only.

That qualification permitted one new fresh-initialization formal run:

```text
run id: phase_u_2kg_angrate_cap8_env512_998400_20260812_seed720202
producer HEAD: ac514ec351e1267ef523f29c5fd3dc5b8e308305
source-tree hash: ea4f83fef961e2e80cfafad3056aa02fb22939229a9e88a99897184cf91b9011
worker PID: 885981
authorized/effective training ceiling: 1,000,000 / 998,400
effective checkpoints: 0/102,400/256,000/512,000/755,200/998,400
status: runs/two_phase/phase_experts/phase_u_2kg_angrate_cap8_env512_998400_20260812_seed720202/status.json
metrics: runs/two_phase/phase_experts/phase_u_2kg_angrate_cap8_env512_998400_20260812_seed720202/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_2kg_angrate_cap8_env512_998400_20260812_seed720202.log
control/resume: runs/two_phase/process_logs/phase_u_2kg_angrate_cap8_env512_998400_20260812_seed720202.control.txt
```

The one startup audit found the worker live, `status=running`, and a complete
transition-0 checkpoint with matching source/model/config/threshold identity.
Monitoring remains sparse with an estimated 18--30 minute terminal window.
Snapshot acquisition and continuation probing remain evidence-gated; formal
`V_up`, Soft Tube, Phase D training, and unified PPO remain unauthorized.

The cap-8 formal run subsequently entered `gate_pause` at 256,000 training
transitions with `held_out_physical_performance_plateau`; its authorization is
consumed and must not be resumed. It used 640 fixed-evaluation transitions and
zero candidate/continuation transitions. All 21 checkpoint sidecars validate,
and 24 MP4/NPZ failure pairs are retained. The 0/102.4k/256k fixed evaluations
each reached the legal window in 8/8 rollouts and ended at
`takeoff_missed_liftoff_deadline`, with no physical failure, pitch violation,
illegal contact, saturation, numerical fault, or identity mismatch. Peak
angular speed at 256k was 0.45 rad/s. Thus cap 8 fixed the prior high-rate
failure but produced a safe no-liftoff policy.

Training stochastic episodes nevertheless retained 4--8 units of ascent
shaping per batch while physical failures fell from 100% to 20%; no Apex
success occurred. The deterministic 256k trace holds hip near -1.23 rad and
reaches only 0.062 m/s vertical speed. Existing fixed impulse evidence shows
that one-tick hip actions 0.10--0.15 can produce post-window liftoff with about
0.07--0.19 rad peak pitch and no physical failure. The next single hypothesis
therefore adds an 8.0 one-shot `legal_liftoff_bonus` only on the monotonic
post-window liftoff transition. It does not grant success or termination and
does not reward early airborne. Cap 8 and every other model, safety, reward,
reset, PPO, and evaluation contract remain fixed.

The legal-liftoff implementation is now source-complete with reward-contract
hash `4f1fd69237f1fa50c01416dbc65158a6590d09e71af26b04a59b0acf3680df75`.
Default weight 0 preserves compatibility; stable Phase U configs explicitly
select 8.0. Red-green tests prove zero pre-window/window-entry/repeated-tick
bonus, exactly one +8 post-window liftoff transition, and no success or done
implication. Eight focused tests, 103 Phase U regressions, compileall, the full
906-test suite, and local preflight pass. A fresh managed runtime gate at
`runs/two_phase/runtime_gate/phase_u_2kg_legal_liftoff_bonus8_20260813/`
passes in 93.44 seconds, including the 64-transition update and 32-transition
resume. This requalification consumes 96 runtime-smoke transitions and zero
new Phase U training transitions. The next permitted action is one fresh
run-bound 512-environment engineering smoke.

That smoke completed as
`gate_c1_phase_u_2kg_liftoff8_cap8_env512_smoke_20260813_seed720301`: 12,800
training, 1,600 Brax evaluation, and 208 fixed-evaluation transitions (14,608
total). The finite checkpoint validates recursively as `47dd97af...558dfe`;
closed outcomes are 8 `other_failure`, all
`takeoff_missed_liftoff_deadline`, with no physical/numerical/runtime fault.
All eight MP4/NPZ pairs are retained. This is engineering qualification only.

A new fresh-initialization formal run is now active:

```text
run id: phase_u_2kg_liftoff8_cap8_env512_998400_20260813_seed720302
producer HEAD: 6459978a7e92108f32729f8f72e4d2e3f888d13f
source-tree hash: 0334c6b8c4894fcce5eb4466c72ea73c692797edc284bcbbb284ed7fd9f424d6
worker PID: 956415
authorized/effective training ceiling: 1,000,000 / 998,400
effective checkpoints: 0/102,400/256,000/512,000/755,200/998,400
status: runs/two_phase/phase_experts/phase_u_2kg_liftoff8_cap8_env512_998400_20260813_seed720302/status.json
metrics: runs/two_phase/phase_experts/phase_u_2kg_liftoff8_cap8_env512_998400_20260813_seed720302/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_2kg_liftoff8_cap8_env512_998400_20260813_seed720302.log
control: runs/two_phase/process_logs/phase_u_2kg_liftoff8_cap8_env512_998400_20260813_seed720302.control.txt
```

The one startup audit found the worker live, `status=running`, and a complete
transition-0 checkpoint with matching identities. Monitoring remains sparse.
Candidate/continuation work remains gated on real independent Apex successes;
formal `V_up`, Soft Tube, Phase D, and unified PPO remain unauthorized.

The liftoff-bridge formal run subsequently entered `gate_pause` at 256,000
training transitions with `held_out_physical_performance_plateau`; it must not
be resumed. It consumed 648 fixed-evaluation and zero candidate/continuation
transitions. All 21 checkpoint sidecars validate and 24 MP4/NPZ pairs are
retained. Every deterministic rollout at 0/102.4k/256k reached the window but
had zero liftoff, zero Apex, and zero physical failure; the 256k peak angular
speed was 0.51 rad/s.

The training distribution did contain legal liftoff: after the first block,
mean one-shot liftoff reward was 4.48--6.72, so roughly 56--84% of stochastic
episodes triggered it. At 256k the mean +6 liftoff reward coexisted with -41.16
angular-rate cost, -12 illegal-contact cost, 26% physical failure, and zero
Apex. Increasing liftoff reward would therefore reinforce low-quality events.
The next single hypothesis instead adds a one-shot +16 bridge only on the
existing post-window `stable_airborne` transition. It retains liftoff +8 and
cap 8 and does not imply ascending, Apex, success, or termination. Every other
model, safety, PPO, reset, threshold, and reward contract remains fixed.

The stable-airborne bridge implementation is source-complete with reward hash
`0cc722a818239026a5094b1145637022fd348755257bba4805c9d2f98f00242c`.
Default weight 0 preserves compatibility and stable Phase U configs select
16.0. Red-green tests prove liftoff-only zero, exactly one post-window stable
airborne +16 transition, repeat zero, and no success/done implication. Eight
focused tests, 107 Phase U regressions, compileall, the full 910-test suite,
and preflight pass. A fresh managed runtime gate at
`runs/two_phase/runtime_gate/phase_u_2kg_stable_airborne_bonus16_20260813/`
passes in 93.27 seconds including 64-transition update and 32-transition
resume. The next permitted action is one fresh run-bound 512-env smoke.

The stable-airborne smoke
`gate_c1_phase_u_2kg_stable16_liftoff8_cap8_env512_smoke_20260813_seed720401`
completed with 12,800 training, 1,600 Brax evaluation, and 208 fixed-evaluation
transitions (14,608 total). Its checkpoint validates recursively as
`3c1284de...9e27c`; all eight outcomes were nonphysical missed-liftoff and all
eight MP4/NPZ pairs are retained. No numerical, runtime, identity, saturation,
or physical fault occurred. This is engineering qualification only.

A fresh formal run is active:

```text
run id: phase_u_2kg_stable16_liftoff8_cap8_env512_998400_20260813_seed720402
producer HEAD: 9313a79d1153f8a64882491156d5c43762e153ba
source-tree hash: fa8620d925f117e3846b424bd983e707186927e914ccdd32454f11b0f58cd1ec
worker PID: 1022132
authorized/effective training ceiling: 1,000,000 / 998,400
status: runs/two_phase/phase_experts/phase_u_2kg_stable16_liftoff8_cap8_env512_998400_20260813_seed720402/status.json
metrics: runs/two_phase/phase_experts/phase_u_2kg_stable16_liftoff8_cap8_env512_998400_20260813_seed720402/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_2kg_stable16_liftoff8_cap8_env512_998400_20260813_seed720402.log
```

The startup audit found the worker live, `status=running`, and a complete
transition-0 checkpoint with matching identities. Monitoring remains sparse;
candidate/continuation work still requires independent Apex successes.

The stable-airborne formal run subsequently entered `gate_pause` at 256,000
training transitions with `held_out_physical_performance_plateau`; it must not
be resumed. It consumed 648 fixed-evaluation transitions and zero candidate or
continuation transitions. All 21 checkpoint sidecars validate, and all 24
held-out failures have MP4 and timing-aligned NPZ evidence. Every fixed rollout
at 0/102.4k/256k reached the legal window but had zero liftoff, zero Apex, and
zero physical failure, ending at `takeoff_missed_liftoff_deadline`. The
intentional checkpoint-gate RuntimeError is the only traceback.

At 256k, stochastic training still produced mean +5.36 legal-liftoff reward,
+1.76 stable-airborne reward (about 11% of episodes), and +0.283 bounded Apex-
approach reward, but zero Apex success. It also incurred -33.52 angular-rate
cost, -7.4 illegal-contact cost, and 14% physical failure. Increasing either
bridge bonus would reward events that are still too weak. The next single
hypothesis changes only the stable Phase U `apex_approach_weight` from 2.0 to
8.0. That existing term activates only after legal-window entry, stable full-
structure airborne, and ascending motion, and scores proximity to all seven
deployable Apex physical conditions. Physics, safety, reset, PPO, exploration,
thresholds, angular cap, bridge bonuses, and every other reward term remain
fixed. This requires red-green validation, a fresh runtime gate, one new smoke,
and a new run-bound authorization; no run has started for this hypothesis.

The Apex-approach configuration iteration is now source-complete with reward-
contract hash
`ccedd915fd9532550ba84b3d26ee84c4afe3ce198c94e521aa19569eddcd36f2`.
The red test failed against the old 2.0 value and passed after changing only
the two stable configs to 8.0. Seventy-three phase-expert tests, compileall,
the full 910-test suite, and local preflight pass. A fresh managed runtime gate
at `runs/two_phase/runtime_gate/phase_u_2kg_apex_approach_weight8_20260813/`
passes in 93.18 seconds, including its 64-transition PPO update and 32-
transition resume. This consumes 96 engineering-integrity transitions and zero
Phase U training transitions. The next permitted action is one fresh run-bound
512-environment smoke; no formal retry is yet authorized or active.

That smoke completed as
`gate_c1_phase_u_2kg_apex8_stable16_liftoff8_cap8_env512_smoke_20260813_seed720501`.
It consumed 12,800 PPO training, 1,600 Brax evaluation, and 208 held-out fixed-
evaluation transitions (14,608 total). The update was finite at 907.11 training
transitions/s and its checkpoint sidecar validates recursively as
`cf4a26fb...bef2e73`. All eight deterministic outcomes were nonphysical
`takeoff_missed_liftoff_deadline`; all reached the legal window, none lifted
off or reached Apex, and all eight MP4/NPZ pairs validate. There was no
numerical, runtime, identity, timing/history, saturation, contact, roll, or
pitch fault. This is engineering qualification only. It permits one fresh
run-bound formal authorization under the unchanged Gate Pause protocol; that
formal run has not yet started.

That qualification authorized and launched one fresh-initialization formal
run:

```text
run id: phase_u_2kg_apex8_stable16_liftoff8_cap8_env512_998400_20260813_seed720502
producer HEAD: 0d9e0cd1ac62d351c52a8f8fb2dce402393af129
source-tree hash: 45759d98e593abf75ccb7a7968fac536eb9ab2df760c6a1aa7632c05aca9b0e4
worker PID: 1095341
authorized/effective training ceiling: 1,000,000 / 998,400
effective checkpoints: 0/102,400/256,000/512,000/755,200/998,400
status: runs/two_phase/phase_experts/phase_u_2kg_apex8_stable16_liftoff8_cap8_env512_998400_20260813_seed720502/status.json
metrics: runs/two_phase/phase_experts/phase_u_2kg_apex8_stable16_liftoff8_cap8_env512_998400_20260813_seed720502/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_2kg_apex8_stable16_liftoff8_cap8_env512_998400_20260813_seed720502.log
control: runs/two_phase/process_logs/phase_u_2kg_apex8_stable16_liftoff8_cap8_env512_998400_20260813_seed720502.control.txt
```

The single startup audit found the detached worker live and `status=running`;
it had reached 64,000 training transitions with six complete checkpoint
sidecars, 216 transition-0 fixed-evaluation transitions, matching identities,
and no numerical/runtime/hash/timing fault. Monitoring is sparse: the next
inspection is the estimated 5--10 minute terminal/Gate-Pause window, not a log
poll. Candidate acquisition and continuation probing remain gated on real
held-out Apex success and parent diversity; formal `V_up`, Soft Tube, Phase D,
and unified PPO remain unauthorized.

The Apex-weight-8 formal run subsequently entered `gate_pause` at 256,000
training transitions with `held_out_physical_performance_plateau`; it must not
be resumed. It consumed 640 fixed-evaluation transitions and zero candidate or
continuation transitions. All 21 checkpoint sidecars and 24 MP4/timing-aligned
NPZ pairs validate. At 0/102.4k/256k all 24 held-out rollouts reached the legal
window, none lifted off or reached Apex, and all ended at the unchanged missed-
liftoff deadline without physical failure. The only traceback is the intended
checkpoint-gate control path.

The weight change worked numerically but not behaviorally. At 256k stochastic
mean Apex-approach contribution rose to 1.258 from 0.283 in the prior weight-2
run, but success remained zero. The same batch had +4.24 liftoff, +1.28 stable-
airborne, -34.03 angular-rate, -6.6 illegal-contact, and 22% physical failure.
The deterministic hip control stayed between -1.234 and -1.200 rad and never
lifted off. The next single hypothesis therefore changes only
`angular_rate_penalty_cap_ratio` from 8.0 to 4.0. This brackets the previously
disproven cap-1 high-rate/pitch-failure behavior and the repeatedly conservative
cap-8 no-liftoff behavior while retaining weight 1.0 and every physical safety
termination. No run has started for the cap-4 hypothesis.

The cap-4 rebalance is now source-complete with reward-contract hash
`1712e01841de0d44ae7816908072ab1ba4f25b0c773ec7b0c19fb17607a1503e`.
The red stable-config test failed against cap 8 and passed after changing only
the two stable configs to cap 4; the hash-drift regression uses cap 8 as its
counterfactual. Seventy-three phase-expert tests, compileall, the full 910-test
suite, and local preflight pass. A fresh managed runtime gate at
`runs/two_phase/runtime_gate/phase_u_2kg_apex8_angrate_cap4_20260813/`
passes in 93.36 seconds, including 64-transition update and 32-transition
resume. This consumed 96 engineering-integrity transitions and zero Phase U
training transitions. The next permitted action is one fresh run-bound 512-
environment smoke; no formal cap-4 run is authorized or active yet.

That cap-4 smoke completed as
`gate_c1_phase_u_2kg_apex8_stable16_liftoff8_cap4_env512_smoke_20260813_seed720601`.
It consumed 12,800 PPO training, 1,600 Brax evaluation, and 216 fixed-
evaluation transitions (14,616 total), with finite update throughput of 913.04
training transitions/s. Its checkpoint validates recursively as
`244f847d...0c55d8a`. All eight held-out rollouts reached the legal window and
ended at the nonphysical missed-liftoff deadline; none lifted off or reached
Apex. All eight MP4/NPZ pairs validate, with no numerical/runtime/identity,
timing/history, saturation, illegal-contact, roll, or pitch fault. This is
engineering qualification only and permits one fresh run-bound formal cap-4
authorization.

That qualification authorized and launched one fresh cap-4 formal run:

```text
run id: phase_u_2kg_apex8_stable16_liftoff8_cap4_env512_998400_20260813_seed720602
producer HEAD: d0e0eb65493c5e21bcfa102f016d4d1cd91b1fab
source-tree hash: 9d590cfe2f2f1be0cfcac2a84673ff94ef5eccde0683e729c283761c104391bc
worker PID: 1155781
authorized/effective training ceiling: 1,000,000 / 998,400
status: runs/two_phase/phase_experts/phase_u_2kg_apex8_stable16_liftoff8_cap4_env512_998400_20260813_seed720602/status.json
metrics: runs/two_phase/phase_experts/phase_u_2kg_apex8_stable16_liftoff8_cap4_env512_998400_20260813_seed720602/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_2kg_apex8_stable16_liftoff8_cap4_env512_998400_20260813_seed720602.log
```

The single startup audit found the detached worker live with matching
identities, a complete transition-0 checkpoint/sidecar, 216 initial fixed-
evaluation transitions, and no numerical/runtime/hash/timing fault. Monitoring
remains sparse, with the next inspection at the estimated 5--10 minute terminal
or Gate-Pause window. Candidate/continuation work remains gated on real Apex
success and independent parent diversity.

The cap-4 formal run subsequently entered `gate_pause` at 256,000 training
transitions with `held_out_physical_performance_plateau`; it must not be
resumed. It consumed 632 fixed-evaluation transitions and zero candidate or
continuation transitions. All 21 sidecars and 24 MP4/NPZ pairs validate. Every
held-out rollout reached the legal window, but none lifted off or reached Apex;
all ended at the nonphysical missed-liftoff deadline.

The reward-cap bracket is now closed. Cap 4, like cap 8, produced a conservative
mean, while cap 1 previously produced high-rate pitch failures. At 256k the
cap-4 stochastic distribution still averaged +5.68 liftoff, +1.60 stable-
airborne, +1.467 Apex approach, -32.53 angular-rate, -9.6 illegal-contact, 20%
physical failure, and zero success. Maximum policy standard deviation remained
0.2387 near its initial hip value 0.25, while the deterministic hip control
stayed between -1.232 and -1.200 rad. The actor already observes deployable
distance-to-obstacle-front plus three-frame joint/IMU/action history, so window
observability is present.

The next single hypothesis changes only hip initial action standard deviation
from 0.25 to 0.10, retaining steer/drive/knee at 0.05. This still covers the
physically validated low-pitch +0.10--+0.15 hip impulses across 512 environments
but reduces destructive tails and requires useful behavior to move the policy
mean rather than survive only as stochastic tail samples. Reward, reset,
physics, safety, PPO, thresholds, and fixed evaluation remain unchanged. No run
has started for this hypothesis.

After two workstation crashes during the broader 512-environment program, the
user explicitly requested a smaller parallel layout. No additional 512-env PPO
run may be launched or resumed. The selected safety layout is 256 environments
and 16 minibatches, preserving 400 samples per minibatch while changing the
native rollout block from 12,800 to 6,400 transitions. This is an operational
baseline revision, so the pending hip-std-0.10 run is not claimed as a strict
single-variable comparison to the old 512-env runs. Reward, physics, reset,
observations, optimizer hyperparameters, network, horizon, fixed evaluation,
and safety gates remain fixed. A 256-env smoke and fresh run-bound authorization
are required before formal training.

The method contract was revised on 2026-08-10 to make the phase experts local
continuation controllers and state-distribution generators rather than final
deployment outputs. Expert training and feasibility-data acquisition now
overlap: successful checkpoints may start real online candidate acquisition
and policy-bound continuation diagnostics while expert training continues.
Formal feasibility data later re-labels all accumulated candidates under the
selected frozen phase expert. An expert trajectory is never itself a Tube.

Gate C1's stable Phase U smoke capability is now implemented at
`cli/train_phase_expert.py` and `dvgc/phase_expert_training.py`. The first
single-run smoke authorization was consumed on 2026-08-10 and entered
`gate_pause` during Brax's trace-only initial evaluation because reset and step
published different `state.done` dtypes. No PPO rollout, optimizer update,
checkpoint, fixed evaluation, or failure trajectory was produced. The dtype
contract is corrected at `87c3f4d`, but the failed authorization was not reused
and no automatic retry was performed.

After a real Brax-wrapper regression reproduced and closed that defect, one
new run-bound replacement smoke completed at source HEAD `88d074d`. It executed
exactly one 1,600-transition PPO rollout block, the fixed 1,600-transition Brax
evaluation protocol, and 216 transitions across eight external fixed
evaluations. All fixed evaluations ended as `other_failure` with
`takeoff_missed_liftoff_deadline`; there were no successes, physical failures,
or timeouts. This is engineering smoke evidence only and does not authorize a
learnability pilot or establish `pi_up`.

The current task authorizes a new Phase U run with a maximum of 1,000,000 total
training transitions, requested checkpoints at 0/100k/250k/500k/750k/1M, and
separate evaluation/acquisition/continuation accounting. This is an upper
bound, not permission to ignore gate-pause conditions or a requirement to wait
until 1M before acquiring candidate states. The revised bounded reward,
physical evaluation, aligned checkpoint, truthful warm-start resume, and
evidence-gated acquisition hooks passed red-green implementation, full static
validation, a fresh runtime gate, and a new bounded PPO smoke at `b4c7fb5`.

On 2026-08-12 the user authorized a fresh one-million-transition Phase U
program after approving the single exploration change that reduces only the
hip initial action standard deviation from 0.50 to 0.25. Early airborne remains
nonterminal diagnostic telemetry: it receives neither a penalty nor Phase U
success, and jump/ascent task progress remains gated by the legal jump window.
The fixed 512-environment PPO block is 12,800 transitions, so the largest
aligned formal budget not exceeding the authorization is 998,400 training
transitions. This authorization is a new run-bound program and is not an
extension or relabeling of the exhausted earlier 995,200-transition program.

The qualifying smoke completed at producer HEAD `b5d564c`:

```text
run id: gate_c1_phase_u_hip025_env512_smoke_20260812_seed710010
training transitions: 12,800
Brax evaluation transitions: 1,600
fixed evaluation transitions: 224
total environment transitions: 14,624
status: completed
```

It wrote a complete checkpoint, preserved the configured action standard
deviation range 0.04998--0.24994, and had no broadphase overflow, NaN/Inf, OOM,
traceback, timing/history mismatch, hash mismatch, roll/pitch violation,
illegal contact, or physical failure. All eight deterministic fixed rollouts
reached the legal window and ended in `takeoff_missed_liftoff_deadline` without
liftoff. This is one-block engineering qualification only, not evidence that
the controller has learned the task. Its eight MP4 diagnostics are retained
under its `failure_videos/` directory.

The collision-qualified persistent formal run was then launched once:

```text
run id: phase_u_formal_hip025_env512_998400_20260812_seed710011
producer HEAD: b5d564c25da2be3d6f39901cf51f41b948f68431
source-tree hash: 69239142c016d11e48e2556910ef5cd5c30c467db3ce6881f28a00374363447e
training PID at startup: 453072
status: runs/two_phase/phase_experts/phase_u_formal_hip025_env512_998400_20260812_seed710011/status.json
metrics: runs/two_phase/phase_experts/phase_u_formal_hip025_env512_998400_20260812_seed710011/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_formal_hip025_env512_998400_20260812_seed710011.log
control/resume: runs/two_phase/process_logs/phase_u_formal_hip025_env512_998400_20260812_seed710011.control.txt
authorization: runs/two_phase/authorizations/phase_u_formal_hip025_env512_998400_20260812_seed710011.json
```

Its effective checkpoints are
0/102,400/256,000/512,000/755,200/998,400. The run independently caps fixed
evaluation, candidate acquisition, and continuation diagnostics and records
those interactions separately. The one permitted startup inspection observed
`status=running`, the correct phase/run/source hashes, and a clean log. It has
not been continuously polled. Candidate harvesting may start only after the
existing held-out success and independent-parent gates pass; no formal
`pi_up_star`, `V_up`, Soft Tube, Phase-D expert, or unified PPO is claimed.

The process has now exited at the complete 755,200-transition checkpoint with
`status=gate_pause` and
`held_out_physical_performance_plateau`. This is the terminal result of the
run-bound authorization; the unconsumed 243,200-transition difference to the
998,400 ceiling must not be resumed automatically. The checkpoint audit is:

| effective training transitions | held-out outcomes | window reach | liftoff | Apex success | mean return | mean episode ticks |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 `takeoff_missed_liftoff_deadline` | 8/8 | 0/8 | 0/8 | -3.299 | 27 |
| 102,400 | 8 `takeoff_missed_liftoff_deadline` | 8/8 | 0/8 | 0/8 | -4.159 | 28 |
| 256,000 | 8 `pitch_limit` | 0/8 | 0/8 | 0/8 | -48.583 | 26 |
| 512,000 | 8 `pitch_limit` | 0/8 | 0/8 | 0/8 | -23.822 | 9 |
| 755,200 | 8 `pitch_limit` | 0/8 | 0/8 | 0/8 | -23.464 | 8 |

The apparent return recovery after 256,000 is not physical improvement. The
policy terminates progressively earlier, so it accumulates fewer attitude and
angular-rate penalties while window reach, liftoff, clearance, and Apex
success remain zero. Training-time stochastic episodes also had zero success;
mean physical failure rose from 74.5% over 0--102,400 to 97.1% over
256,000--512,000 and 99.95% over 512,000--755,200. This is precisely the
approved condition in which scalar reward and physical task performance have
diverged.

Timing-aligned state traces confirm the terminal failure occurs before the
legal window. At 755,200, the deterministic policy moves the hip actuator
control target from the natural -1.2 rad position to -0.067 and then +0.336 in
the first two control ticks. Maximum angular speed reaches 22.34 rad/s and
pitch reaches 1.333 rad; termination occurs after 8 control ticks with no event
latch. This is a real `pitch_limit` physical failure, not the nonterminal
early-airborne telemetry that the current method intentionally leaves
unpunished.

Interaction accounting is closed at a known lower bound of 755,984 environment
transitions: 755,200 PPO training + 784 fixed evaluation + 0 candidate
acquisition + 0 continuation labeling. All five checkpoint evaluations retain
eight MP4 videos and eight timing-aligned state traces, for 40 of each. The log
contains no broadphase overflow, NaN/Inf, OOM, timing/history mismatch, or hash
mismatch; its only traceback is the deliberate Gate Pause exception. Producer
HEAD, source-tree hash, XML hash, training-config hash, and run-bound
authorization match the launch manifest.

No candidate snapshot or continuation dataset was produced because the
held-out Apex-success and independent-parent gates never opened. Therefore no
checkpoint can be selected as `pi_up_star`, and there is still no provisional
or formal `V_up`, learned Tube, or real Phase-D Apex seed. A canonical technical
audit payload is retained at
`runs/two_phase/phase_experts/phase_u_formal_hip025_env512_998400_20260812_seed710011/audit/artifact.json`.
The portable HTML packaging step was not available because the host has no
Node/npm; no package was installed or runtime environment changed.

The next permitted activity under the current method is read-only diagnosis of
the 256,000-transition onset of the pitch instability and design of one new
scientific hypothesis. Any code/config change or further PPO invocation needs
new red-green validation, a collision-qualified smoke, and a separate
run-bound authorization. The current result does not authorize relaxing
roll/pitch limits, changing the XML or action mapping, declaring a Tube, Phase-D
training, or unified PPO.

## Current branch and commit

- Branch: `agent/two-phase-soft-tube`
- Gate B baseline: `5331896bee08a920321a9b39b496f66c7b9b0879`
- Gate B implementation head: `387ae59`
- Failure-video audit implementation head: `5b8fe73`
- Prelaunch-continuation design: `bac2a93`, `691ad8e`, `ec90d6d`
- Prelaunch-continuation implementation: `23a746e`, `6ed2cdc`, `0b86435`
- Gate A implementation: `5e5da3b`
- Gate C1 run contracts: `07a435c`, `24b6217`, `1221b9a`
- Gate C1 Phase U adapter and smoke runtime: `0e4f718`, `b36cfec`
- Gate C1 failure-video archive support: `74723a5`
- Post-pause dtype correction: `87c3f4d`
- Brax-wrapper dtype regression: `88d074d`
- Completed-interaction accounting closure: `55b47d1`
- Interleaved Phase U method contract: `5632962`
- Bounded Phase U reward: `140ad02`
- Checkpoint evaluation protocol: `567ebc0`
- Candidate acquisition gate: `086bae5`
- Checkpoint/acquisition provenance closure: `4b6449e`, `a25a31f`
- Phase U exploration diagnosis and implementation: `ddf774f`, `46c1084`
- Phase U channel-exploration diagnosis and implementation: `e2bb067`,
  `cb0e384`, `b1b4f25`, `c6f5443`
- Phase U bounded hip-exploration design and implementation: `2b24ed1`,
  `1b19830`, `b5d564c`
- Current runtime fingerprint refresh: `c6f5443`
- Cleanup baseline: `main@b7bb815`
- Dependency design: `27f0aa3`
- Clarified design policy: `3cbd6c1`
- Validated cleanup commits through `3d6a6b5`; resolve the containing final
  validation commit with `git rev-parse HEAD`.

## Last validated gate

- Current reusable runtime gate: PASS at source fingerprint
  `31e78390ee81c2c615b1cd77f4e95ea45cc427e7b60e2f477588b600cfe3ed29`.
- It used the fresh ignored work directory
  `runs/two_phase/runtime_gate/phase_u_channel_std_20260812/` and took 97.711
  seconds.
- The gate's 64+32 = 96 transitions are compile/update/resume smoke evidence
  only, not expert, pilot, or formal training.

## Current and most recent formal run

The collision-qualified 512-environment Phase U formal rerun is paused:

```text
run id: phase_u_formal_env512_998400_20260810_seed710004
producer HEAD: 79fd4f39148ec77a586c60382067b576edb5ecaf
training PID: 2626553
status: runs/two_phase/phase_experts/phase_u_formal_env512_998400_20260810_seed710004/status.json
metrics: runs/two_phase/phase_experts/phase_u_formal_env512_998400_20260810_seed710004/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_formal_env512_998400_20260810_seed710004.log
resume record: runs/two_phase/process_logs/phase_u_formal_env512_998400_20260810_seed710004.control.txt
```

Its run-bound authorization capped training at 998,400 aligned transitions,
fixed evaluation at 9,600, candidate acquisition at 76,800, continuation
diagnostics at 76,800, and all environment interaction at 1,161,600. The one
startup check observed `status=running`, an absolute transition-0 checkpoint,
and no broadphase, OOM, NaN, Inf, traceback, error, or Gate Pause pattern. No
further polling was performed by the launching interaction. The process later
entered `gate_pause` at 256,000 training transitions after fixed evaluations at
0, 102,400, and 256,000 all reached the jump window but produced zero liftoff,
clearance, or Apex success. All 24 outcomes were
`takeoff_missed_liftoff_deadline`; physical failure, roll/pitch violation,
illegal contact, action saturation, candidate acquisition, and continuation
diagnostics remained zero. The process exited after writing the 256,000
checkpoint and 656 fixed-evaluation transitions.

A bounded natural-start action diagnostic then consumed 710 non-training
environment transitions. Window-triggered hip action at or above 0.5 produced
liftoff in 10/10 cases, while nonpositive hip action produced liftoff in 0/15.
This proves current action authority can initiate liftoff but does not establish
an Apex success. The failed policy kept its action standard deviation near the
Landing-oriented 0.05 prior, making a useful hip action effectively
unobservable during PPO exploration. The next single scientific hypothesis is
a Phase-U-only initial exploration standard deviation of 0.25; reward, reset,
optimizer, physics, action mapping, and evaluation stay fixed.

That hypothesis is implemented and frozen at producer HEAD `46c1084`. Fresh
validation compiled `dvgc` and `cli`, passed 881 tests, passed the complete
64+32-transition runtime gate, and passed `scripts/local_preflight.sh` (which
repeated the same 881 tests). The default/Landing actor prior remains 0.05;
only explicit Phase U configs select 0.25.

The requested 1,024-environment one-block qualification was repeated with the
new prior. It completed 25,600 smoke training transitions at 1,800.88
transitions/s and preserved a mean policy standard deviation of 0.24994, but
again emitted MJX Warp broadphase overflow warnings, with the reported
`naconmax` requirement reaching at least 1,296. It is therefore invalid as a
collision-complete training layout. The run and its eight failure videos are
preserved at:

```text
runs/two_phase/phase_experts/gate_c1_phase_u_std025_env1024_smoke_20260811_seed710005/
```

The otherwise identical 512-environment qualification completed 12,800 smoke
training transitions plus 1,600 Brax-evaluation and 216 fixed-evaluation
transitions. It ran at 893.26 training transitions/s, preserved a mean policy
standard deviation of 0.24996, wrote its transition-12,800 checkpoint, and had
no broadphase overflow, NaN, Inf, OOM, traceback, accounting failure, or video
failure. All eight deterministic fixed evaluations still ended in
`takeoff_missed_liftoff_deadline`; one smoke block is engineering integrity
evidence, not a learnability conclusion. Its eight failure videos are at:

```text
runs/two_phase/phase_experts/gate_c1_phase_u_std025_env512_smoke_20260811_seed710006/failure_videos/
```

The remaining collision-qualified formal Phase U run was launched once and is
not being polled by the launching interaction:

```text
run id: phase_u_formal_std025_env512_448000_20260811_seed710007
producer HEAD: 46c108492eb183f7e2a3f251bed849838e82616b
training PID at startup: 557999
status: runs/two_phase/phase_experts/phase_u_formal_std025_env512_448000_20260811_seed710007/status.json
metrics: runs/two_phase/phase_experts/phase_u_formal_std025_env512_448000_20260811_seed710007/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_formal_std025_env512_448000_20260811_seed710007.log
authorization: runs/two_phase/authorizations/phase_u_formal_std025_env512_448000_20260811_seed710007.json
```

Its one startup check observed a live process, `status=running`, zero consumed
transitions, and the absolute transition-0 checkpoint. This invocation is
bounded to 448,000 expert-training transitions, 6,400 fixed-evaluation
transitions, 51,200 candidate-acquisition transitions, and 51,200 continuation
diagnostic transitions. The prior formal runs consumed 547,200 expert-training
transitions, so the program-level maximum after this invocation is 995,200,
below the authorized 1,000,000 ceiling. Requested fixed checkpoints are
0/100k/250k/448k and resolve to aligned 0/102,400/256,000/448,000 checkpoints.
If the process exits abnormally, a new run-bound authorization and output ID
must be created, then `--resume-run` must name this directory and
`--restore-checkpoint` must name its latest complete `orbax/` checkpoint; the
existing output directory and authorization must never be reused.

The process subsequently exited at the complete transition-256,000 checkpoint
with `status=gate_pause` and
`held_out_physical_performance_plateau`. Fixed evaluations at 0, 102,400, and
256,000 all reached the legal jump window but produced zero liftoff, clearance,
or Apex success. Their mean returns degraded from -3.299 to -6.212 to -12.000,
while maximum held-out roll/pitch grew and all 24 outcomes remained
`takeoff_missed_liftoff_deadline`. The final eight failure videos are preserved
under `evaluations/000000256000/failure_videos/`. Training-time stochastic
episodes had zero success and 81%--97% physical-failure rates, showing that the
scalar 0.25 prior widened exploration but made most extra samples destructive.
Candidate acquisition and continuation diagnostics remained at zero. This run
consumed 256,000 expert-training and 680 fixed-evaluation transitions.

Across formal Phase U invocations, consumed expert training is now 803,200
transitions, leaving at most 196,800 under the 1,000,000 authorization. The
largest 512-environment-aligned remainder is 192,000. The next single
hypothesis is channel-specific initial exploration
`[steer=0.05, drive=0.05, hip=0.50, knee=0.05]`, based on the prior controlled
diagnostic in which hip action at or above 0.5 caused liftoff in 10/10 cases.
No reward, reset, threshold, deadline, optimizer, network-layer, XML, actuator,
or action-mapping change is included in that hypothesis.

The channel-specific exploration implementation passed red-green tests, fresh
static compilation, 897 full tests, `scripts/local_preflight.sh` (which repeated
the same 897 tests), and the complete runtime gate at source HEAD `c6f5443`.
The 512-environment one-block smoke then completed 12,800 training, 1,600 Brax
evaluation, and 216 fixed-evaluation transitions with closed total accounting
of 14,616. Its measured policy standard deviations ranged from 0.04997 to
0.50012, matching the ordered manifest `[0.05, 0.05, 0.5, 0.05]`. The log had
no broadphase overflow, NaN/Inf, OOM, traceback, timing/history mismatch, or
hash mismatch. All eight deterministic fixed rollouts still ended in
`takeoff_missed_liftoff_deadline`, and all eight failure videos are preserved:

```text
runs/two_phase/phase_experts/gate_c1_phase_u_channel_std_env512_smoke_20260812_seed710008/failure_videos/
```

This smoke is engineering integrity evidence only. It authorized the remaining
formal invocation, not a learnability or Tube claim. The persistent run has now
completed:

```text
run id: phase_u_formal_channel_std_env512_192000_20260812_seed710009
producer HEAD: c6f5443d5107c106e17bea29bc4d4eaabd14bae4
training PID at startup: 120899
status: runs/two_phase/phase_experts/phase_u_formal_channel_std_env512_192000_20260812_seed710009/status.json
metrics: runs/two_phase/phase_experts/phase_u_formal_channel_std_env512_192000_20260812_seed710009/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_formal_channel_std_env512_192000_20260812_seed710009.log
authorization: runs/two_phase/authorizations/phase_u_formal_channel_std_env512_192000_20260812_seed710009.json
```

It consumed exactly 192,000 expert-training and 552 fixed-evaluation
transitions, wrote checkpoints/evaluations at 0/102,400/192,000, and closed with
`status=completed`. Candidate acquisition and continuation diagnostics remained
zero because no held-out Apex success existed. The log contains no broadphase
overflow, OOM, NaN/Inf, traceback, timing/history mismatch, or hash mismatch.
All 24 held-out failures have MP4 videos and timing-aligned state traces under
the three `evaluations/<transition>/failure_videos/` directories.

The held-out outcome changed from eight
`takeoff_missed_liftoff_deadline` failures at transitions 0 and 102,400 to eight
`pitch_limit` physical failures at transition 192,000. Apex success, legal
liftoff, stable airborne, and clearance success were zero at every checkpoint.
At the final checkpoint the deterministic controller applied a strong hip
command immediately from the natural start: root vertical velocity reached
1.03 m/s by tick 2 while the root remained about 2.04 m before the obstacle
front, and pitch reached 1.379 rad before termination at tick 12. This is early
airborne diagnostic behavior only; the jump-window latch never fired and no
Phase U success was credited.

The actor is not wholly missing a deployable timing signal. Each of its four
35-value FIFO frames contains `task_distance_to_front =
(step_front_x-root_x)/3` at dimension 18. It does not directly receive the
formal `obstacle_relative_x` based on the robot's frontmost collision support,
nor the internal `jump_signal_latched` telemetry. A zero-interaction frozen-
checkpoint sensitivity audit held every other actor and critic input fixed and
changed this distance value in all four FIFO frames. At 102,400 transitions the
hip action changed from +0.146 at the natural start to -0.062 near the window
end; at 192,000 it changed from +0.413 to -0.215. The distance-gradient L1 norm
grew from 0.349 to 1.051. The actor therefore did use the available signal, but
learned the wrong timing direction: stronger positive hip far from the obstacle
and negative hip near the legal window.

The normalizer and stochastic rollout distribution explain how this failure
formed. Across all 15 PPO blocks, the mean training physical-failure rate was
97.1% and mean episode length was 17.34 ticks. At the final checkpoint, the
four normalized window-start distance values were already 2.27--2.72 standard
deviations below their running means; window-end values were 6.41--7.17
standard deviations below. Thus the 0.50 hip exploration prior caused training
to be dominated by early failures, leaving the legal timing region strongly
out of distribution even though deterministic transition-0 evaluation could
reach it. This evidence does not justify adding an event latch or changing the
observation contract. Any reward, observation, or exploration change remains a
new scientific hypothesis requiring a new design and training authorization.

Across the three counted formal invocations, Phase U has now consumed 995,200
of the authorized 1,000,000 expert-training transitions. The remaining 4,800
is less than one collision-qualified 512-environment PPO block of 12,800, so no
further expert-training invocation is authorized. No `pi_up_star`, formal
`V_up`, learned Soft Tube, or real Phase-D Apex seed exists.

The repaired 64-environment Phase U formal-expert run is now paused:

```text
run id: phase_u_formal_1m_20260810_seed710001_absckpt
producer HEAD: 6dad8dbd3d917eacba6c2771e1751c184da013aa
startup PID: 2477733
status: runs/two_phase/phase_experts/phase_u_formal_1m_20260810_seed710001_absckpt/status.json
metrics: runs/two_phase/phase_experts/phase_u_formal_1m_20260810_seed710001_absckpt/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_formal_1m_20260810_seed710001_absckpt.log
resume record: runs/two_phase/process_logs/phase_u_formal_1m_20260810_seed710001_absckpt.control.txt
```

The process was stopped at the last complete checkpoint, transition 291,200,
after fixed held-out evaluations at 0, 100,800, and 251,200 all produced zero
liftoff, clearance, and Apex successes. All three reached the jump window and
had no physical failure, roll/pitch violation, illegal contact, or action
saturation. This is a three-window physical-performance plateau even though
mean held-out return improved. Candidate acquisition and continuation probing
therefore remained at 0. The operator pause record is stored beside
`status.json`; its known interaction lower bound is 291,200 training + 664
fixed evaluation = 291,864. The old process was terminated after the pause and
will not be resumed under its loaded gate implementation.

Parallel-layout qualification then tested the requested 1,024 environments
without changing reward, reset, network, optimizer, horizon, XML, or physical
limits. That one-block run completed 25,600 training transitions at 1,782.60
training transitions/s, but repeatedly reported MJX Warp broadphase overflow:
the immutable runtime config provides `naconmax=1024`, while the largest
reported requirement was 1,443. Because collision candidates may be truncated,
the benchmark audit marks the run invalid and does not authorize 1,024 for
formal training. The artifacts remain as negative qualification evidence.

The 512-environment fallback completed one 12,800-transition block plus 216
fixed-evaluation transitions. It measured 884.92 training transitions/s,
14.46 seconds of training time, 57.21 seconds end to end, and 4,938,288 KiB
maximum resident memory. Its persistent log contains no broadphase overflow,
OOM, NaN, Inf, traceback, or contract warning. This is parallel-layout
integrity evidence, not learnability evidence. The stable formal layout is
therefore 512 environments with a 12,800-transition PPO block and an aligned
998,400-transition maximum.

The persistent Phase U formal-expert run was launched after the bounded smoke:

```text
run id: phase_u_formal_1m_20260810_seed710001
producer HEAD: f20a433813b7a6a8827ef482fd29d803ac1ec86c
startup PID: 2367646
status: runs/two_phase/phase_experts/phase_u_formal_1m_20260810_seed710001/status.json
metrics: runs/two_phase/phase_experts/phase_u_formal_1m_20260810_seed710001/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_formal_1m_20260810_seed710001.log
resume record: runs/two_phase/process_logs/phase_u_formal_1m_20260810_seed710001.control.txt
```

The startup health check initially observed `status=running`, but the process
then correctly entered `gate_pause` before any PPO rollout when its transition-0
formal checkpoint reached Orbax with a repository-relative path. Current Orbax
requires an absolute checkpoint path. Actual training, evaluation, candidate,
continuation, and total environment-transition counts are all 0. No dynamic
failure frames existed, so failure video is not applicable. The consumed
authorization and failed output directory are retained and will not be reused.
The checkpoint boundary now resolves its root before invoking Orbax, with a
red-green regression that reproduces the original relative-path failure.

The separately authorized exact-formal-path regression smoke then completed:

```text
runs/two_phase/phase_experts/gate_c1_phase_u_formal_checkpoint_smoke_20260810_seed710001/
```

It wrote absolute transition-0 and transition-1,600 checkpoints with truthful
normalizer/policy/value sidecars, consumed 1,600 PPO training + 216 fixed
evaluation = 1,816 total interactions, and ended `completed`. Its transition-0
held-out evaluation again had zero Apex success and eight post-window
`takeoff_missed_liftoff_deadline` outcomes, with no physical failure, timeout,
roll/pitch violation, illegal contact, saturation, NaN, or contract failure.
This validates the repaired formal checkpoint path only; it is not expert or
learnability evidence and is excluded from formal expert-training totals.

The original Gate C1 smoke attempt is retained at
`runs/two_phase/phase_experts/gate_c1_phase_u_smoke_20260810_seed710001/`
with status `gate_pause`. Its actual training, Brax evaluation, fixed
evaluation, and combined environment-transition counts are all 0. The failure
occurred during JAX type tracing, so no dynamic frames existed and a failure
video was not applicable. The run was not retried.

The separately authorized replacement run is complete at:

```text
runs/two_phase/phase_experts/gate_c1_phase_u_replacement_smoke_20260810_seed710001/
```

Its actual interaction accounting is 1,600 training + 1,600 Brax evaluation
+ 216 fixed evaluation = 3,416 environment transitions, within the authorized
4,800 ceiling. The run wrote an Orbax normalizer/policy/value checkpoint at
transition 1,600 and a recursive-identity sidecar that was historically
described as full-state. Inspection of the installed Brax payload proved that
optimizer and environment-step state are not present. Current validation
therefore rejects that old full-state claim and describes new checkpoints only
as policy/normalizer/value warm starts. It did not authorize promotion.

The reward-contract smoke for the current 1M implementation is complete at:

```text
runs/two_phase/phase_experts/gate_c1_phase_u_reward_smoke_20260810_seed710001/
```

It consumed 1,600 PPO training transitions, 1,600 Brax evaluation transitions,
and 216 fixed-evaluation transitions, for 3,416 total interactions under its
4,800 ceiling. PPO update, finite reward components, checkpoint writing,
warm-start sidecar identity, fixed evaluation, metrics, accounting, and eight
failure-video captures all completed. The fixed evaluation produced zero Apex
successes and eight `takeoff_missed_liftoff_deadline` outcomes after legal
jump-window entry, with no physical failure, timeout, roll/pitch violation,
illegal contact, action saturation, NaN, or contract failure. That is an
expected non-learned smoke outcome, not evidence of expert ability and not a
Gate pause.

The historical Gate B guideline event probe used 17 environment
transitions; its two outcome-video diagnostics used 25 transitions in total.
Those runs are retained provenance, not active work or a revised Gate B pass
condition. Gate C1 used one adapter integration diagnostic transition, 96
runtime-integrity transitions, and 3,416 replacement-smoke interactions.
Phase U formal-expert, Phase D, feasibility, Soft Tube, and unified-policy
training transitions remain exactly 0 at this marker. The two completed Phase
U smoke runs are engineering integrity evidence and are excluded from formal
expert-training totals.

## Pipeline automation safety interlock

On 2026-08-03 the live user-systemd state was inspected before any two-phase
implementation work. `dvgc-pipeline-watchdog.timer` was still enabled and
active, and `runs/ACTIVE_PIPELINE.json` still had status `ACTIVE` for
`scripts/start_corrected_apex_unified_rsi_followons.sh`. The watchdog source
also retains its legacy Descent-Tube fallback.

The timer was stopped and disabled without changing or deleting its unit files,
watchdog source, or referenced legacy scripts. The active pointer was preserved
byte-for-byte as
`runs/ACTIVE_PIPELINE.legacy-disabled-20260803T134842+0800.json` (SHA-256
`6546dc01d7d2579d733fcbf39b3544942467232dfec2a25319e078a24b3bfae4`), and
the default `runs/ACTIVE_PIPELINE.json` path is absent. The complete ignored
operations record is under
`runs/operations/watchdog_deactivation_20260803T134842+0800/`.

Restoring the retired automation requires an explicit decision followed, from
the repository root, by:

```bash
mv -- runs/ACTIVE_PIPELINE.legacy-disabled-20260803T134842+0800.json runs/ACTIVE_PIPELINE.json
systemctl --user enable --now dvgc-pipeline-watchdog.timer
```

## Current inputs and hashes

- XML: `assets/orange_bike_4kg_horizontal.xml`
- Historical filename retained: yes
- XML SHA-256: `e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192`
- Payload: 2.0 kg
- Hip/knee limits: +/-50 N m
- Action order: `[steer, rear-wheel drive, hip, knee]`
- Runtime: `/home/qy/mujoco_playground/.venv/bin/python`

## Latest result

Gate B has been revised and accepted as a runtime/state-foundation gate. The
formal path is the external pure-JAX adapter over `state.data`/`state.info` plus
immutable XML geometry; no latch was added to `env.step`. All
collision-relevant robot geoms are covered by the geometry manifest,
representative host `mj_geomDistance` cross-audit passed, the natural start is
valid, and the fixed threshold manifest was produced reproducibly at:

```text
runs/two_phase/gate_b_20260803_prelaunch_continuation/threshold_manifest.json
file SHA-256: eb7b517e5fb0bd4d49f90fd93f9223d4056718f9329e07b5400c35aa49119387
canonical hash: 2886802483c77a6d13817cbce889e9fec24807be2b536ba1429cdb7c6aeff900
```

The historical real guideline event trace began from a MuJoCo-audited grounded state
(wheel support present, no body-terrain contact). Early airborne was retained
as telemetry and did not terminate the rollout. The environment's monotonic
jump latch became true on the terminal tick, but the rollout then triggered the
unchanged roll safety limit after 17 control ticks with `end_code=4`
(`roll_limit`). The external two-phase adapter does not admit an event on a
physical-failure tick, so no valid Apex or stable recovery occurred. This
result proves that full open-loop dynamic compatibility with that historical
4 kg, +/-50 N m model was not demonstrated; it does not block expert training. The
CLI stopped before bank admission. Consequently:

- no authoritative Phase U bank was written;
- no authoritative Phase D bank was written;
- no real Gate B bank round-trip report was run or written;
- the implemented builder and round-trip contracts remain covered by dynamic
  tests but are not claims that an expert bank exists.

## Dynamic failure video evidence

Gate B failures now preserve videos after the physical event report is closed
and before the original Gate-pause exception is raised. Rendering consumes
captured qpos/qvel/ctrl only and cannot advance or influence environment
dynamics. A renderer failure is recorded separately and cannot mask or weaken
the original physical failure.

The current ignored audit directory is:

```text
runs/two_phase/gate_b_20260803_prelaunch_continuation/failure_videos/
```

- `full_guideline_continuation.mp4`: 238,955 bytes, SHA-256
  `08df9e45f7c711ed63cade78dd8aa8d25e5da4e384a55cf9723b7951a45a0ce5`,
  17 environment transitions, terminal `roll_limit` (`end_code=4`).
- `full_guideline_continuation.states.npz`: SHA-256
  `2109bacfbecdaabdf63d833e61a574be03ea0bf2e2347fd5641b9548ee267823`.
- `launch_history_window_latch.mp4`: 157,826 bytes, SHA-256
  `1ea9ebb58a438d922cf7f2c2144f9f8385ea2e03b25532f0965e9e1c3bb62e02`,
  8 environment transitions, nonterminal outcome diagnostic.
- `launch_history_window_latch.states.npz`: SHA-256
  `77d1a0dd70b3d07202f87c73e86a6c84a1f7b8ba1a758c8ab1851dbaf2a66a96`.
- `failure_video_manifest.json`: SHA-256
  `e8f4fc62f7370b165da4c56a4e895d6f8cb5e617aaaeac67940584ecef742e84`.

The run was executed at producer HEAD
`0b8643521797eac2d38ff52b8f108cda5fb6d283`. Because the pause-path manifest
version used by that run recorded only input hashes, the ignored
`producer_provenance.json` is an explicitly labeled post-run execution record
that binds that HEAD's env, reward, failure-video, runtime, guideline, and
builder hashes to the event-report, threshold-manifest, video-manifest, and
video-status hashes. It is not presented as an in-run signed record. The
current builder closes these producer hashes directly in all future pause
manifests.

The launch-history audit uses the formal timing contract exactly: state origin
83, initial `ctrl`/`last_action` from reference index 73, then control actions
83, 93, and 103. The earlier manually rendered launch-history video used a
one-action-shifted schedule and is superseded; it is not evidence.

At the first root-position window sample (tick 6), the host contact audit still
sees one wheel contact, but the deployable IMU/support estimator is false. The
root-position window event enters the legal window and the environment jump
latch becomes true without requiring wheel support; it remains true after the
window.
Liftoff is observed at tick 7. Both contact signals are shown independently in
the overlay. This diagnostic is nonterminal and does not assert Phase U success.

Final source validation on 2026-08-03:

- `python -m compileall dvgc cli`: passed.
- Prelaunch semantics/reward/failure/guideline/round-trip targeted files:
  115 passed.
- Full pytest: 811 passed, with one existing JAXopt deprecation warning.
- `bash scripts/local_preflight.sh`: passed and repeated 811 tests.
- Fresh runtime gate: PASS, 96 smoke transitions.
- Three source-fingerprint refreshes were required during review closure: 288
  cumulative engineering smoke transitions this round, all excluded from the
  formal-training total.
- Guideline event audit: 17 environment transitions, `gate_pause` at retained
  `roll_limit`.
- Outcome-video diagnostics: 25 environment transitions, rendering PASS and
  original Gate status still `gate_pause`.
- Formal training transitions: 0.

Gate C1 validation on 2026-08-10:

- `python -m compileall dvgc cli`: passed.
- Gate C1/two-phase/repository focused tests: passed.
- Full pytest after the final runtime implementation: 845 passed, with the
  existing JAXopt deprecation warning only.
- `bash scripts/local_preflight.sh`: passed and repeated all 845 tests.
- Fresh managed runtime gate: PASS, 96 engineering-integrity transitions.
- Static threshold refresh used only kinematic reconstruction and immutable
  geometry, with 0 environment and 0 training transitions; canonical hash
  `603ce888e40dae0d15a9cc6c6bf0704af538a62183d343e639e73c430743a881`.
- Real adapter integration probe: 1 environment transition, nonterminal, no
  success, no physical failure.
- Authorized Phase U PPO smoke attempt: `gate_pause` during trace-only initial
  evaluation; actual environment transitions 0 and no checkpoint.
- Automatic retry: none. Phase U pilot/formal authorization: none.
- A real Brax training-wrapper regression was added after the first pause. It
  reproduced the original `lax.scan` dtype error when the cast was removed and
  passed with the corrected contract.
- Replacement Phase U smoke: completed one 1,600-transition rollout block;
  Brax evaluation used 1,600 transitions; external fixed evaluation used 216
  transitions.
- Fixed evaluation outcomes: success 0, physical failure 0, timeout 0,
  other failure 8. All eight terminal reasons were
  `takeoff_missed_liftoff_deadline` (`end_code=12`).
- Jump-window entry occurred at tick 19 in every fixed rollout; no legal
  liftoff, stable-airborne, ascending, or Apex event was observed.
- Eight MP4 videos and eight timing-aligned state traces were saved under the
  replacement run's `failure_videos/` directory.
- Orbax transition-1600 checkpoint sidecar validation passed; recursive
  checkpoint SHA-256:
  `43adb2f97740c7a6348588df9b15f89eb461fe0c602f484c78a79d35d2a4d6b4`.
- Promotion, pilot, and formal authorization remain false.
- The interleaved Phase U implementation then passed 141 directly affected
  tests, full pytest with 867 passing tests, and `scripts/local_preflight.sh`
  with the same 867 tests; the only warning was the existing JAXopt
  deprecation warning.
- Fresh managed runtime gate
  `runs/runtime_gate_phase_u_1m_20260810_v2/` passed with 96 engineering
  transitions in 97.094 seconds.
- The current reward-contract smoke completed with 1,600 training + 1,600
  Brax evaluation + 216 held-out fixed-evaluation transitions. All eight
  held-out rollouts entered the jump window and then missed the legal liftoff
  deadline; the run remained `completed`, not `gate_pause`.
- Its checkpoint sidecar truthfully records a normalizer/policy/value warm
  start, with no optimizer or environment-step state, and recursive checkpoint
  SHA-256
  `2f6e73e163b6ef09488614cc50524917b235a58af1854cf9f03df1bb0e16caa4`.
- The first 1M formal invocation paused at transition 0 because the custom
  formal checkpoint callback passed a relative root to Orbax. The regression
  test failed on the old behavior and passed after resolving the root at the
  checkpoint boundary. Fresh full pytest and local preflight each passed 868
  tests; `cli.runtime_gate --check-only` confirmed the 96-transition runtime
  report remains current because this checkpoint helper is outside that gate's
  source fingerprint.
- A new one-block formal-path smoke then exercised the real transition-0 and
  per-block callbacks. It completed 1,600 training and 216 fixed-evaluation
  transitions, wrote both absolute checkpoints, and retained eight diagnostic
  failure videos. Checkpoint recursive identities were
  `0856c96e20227473988999476b2a9c71432150a4905e37d7943350b2dc8f4dcf`
  at transition 0 and
  `0501c33fbf904009c418ee1bb09d7f245a3cf6310f6c278cd3bcd4dc29b95e56`
  at transition 1,600.
- The 64-environment formal run reached checkpoints 0/100,800/251,200 with
  fixed physical scores unchanged: jump-window reach 8/8, but liftoff,
  clearance, and Apex success all 0/8. It was reversibly stopped, recorded as
  `gate_pause`, and terminated at checkpoint 291,200. The gate implementation
  previously detected only degradation, not the separately required
  three-window plateau; a red-green regression now closes that omission.
- The requested 1,024-environment benchmark reached 1,782.60 training
  transitions/s but is rejected because broadphase capacity overflowed, with a
  reported requirement up to `naconmax=1443` against the immutable 1,024
  capacity. The 512-environment benchmark reached 884.92 training
  transitions/s without that warning and is the selected formal layout.

Legacy five-stage experimental outcomes are not evidence for the dynamic
two-phase method and must not be promoted retrospectively.

## Known blockers

The retained `END_ROLL_LIMIT=4` outcome is not a revised Gate B blocker. It
shows only that the reference open-loop actions were not proven dynamically
compatible with the current model. No further reference-action repair is
planned, and roll/pitch/contact/nonfinite safety limits remain unchanged.

There are still no authoritative Gate B Phase U/Phase D banks, trained/frozen
phase experts, continuation-labeled two-phase snapshots, `V_up`/`V_down`,
learned soft Tubes, unified two-phase PPO, or independent frozen-final-policy
evaluation. The watchdog is disabled/inactive, its service is inactive, and
`runs/ACTIVE_PIPELINE.json` is absent.

The stable Gate C1 CLI and Phase U adapter now complete an engineering PPO
smoke, but the smoke checkpoint has zero fixed-evaluation success and is not a
trained/frozen `pi_up`. Phase U must train from audited natural starts and earn
the complete Apex-band success contract. Phase D preliminary candidates require the physical seed
validation protocol and cannot be called reachable or safe. Its formal reset
distribution must be sourced primarily from real frozen-`pi_up` Apex and early
descent rollouts. Phase D execution remains blocked at Gate C1.

The completed channel-exploration run exhausted the usable aligned Phase U
budget without producing a legal liftoff or Apex trajectory. Its final policy
learned enough hip authority to become airborne before the legal window, then
failed the unchanged pitch safety limit. There are consequently no eligible
parents for snapshot acquisition and no authority to repurpose the unused
candidate/continuation interaction ceilings as PPO training.

## Next permitted action

On 2026-08-13, after two host crashes were reported, the formal Phase U PPO
layout was conservatively reduced from 512 to 256 parallel environments. The
batch/minibatch relationship remains fixed at 16/16, so each minibatch still
contains 400 environment samples (`unroll_length=25`); the aligned rollout
block is now 6,400 total environment transitions. Training seed count,
checkpoint cadence, and per-checkpoint acquisition ceilings were reduced to
the same 256-environment layout. The current exploration hypothesis changes
only the hip initial action standard deviation from 0.25 to 0.10; reward,
reset, optimizer, network, horizon, XML, action mapping, and safety limits are
unchanged. No 512-environment formal run may be launched or resumed.

The 256-environment static/runtime qualification passes: 73 targeted tests,
910 full tests, `scripts/local_preflight.sh`, and the managed GPU runtime gate.
The runtime gate selected `cuda:0` on the RTX 4090 D and passed its fixed
64-transition update plus 32-transition resume in 96.46 seconds. An earlier
sandbox-only attempt could see only the CPU backend and consumed no Phase U
training transitions; its temporary failure report was replaced by the
successful GPU evidence in `docs/RUNTIME_GATE.json`.

The next permitted dynamic action is one 256-environment, one-block (6,400
training-transition) Phase U engineering smoke. Only after its checkpoint,
resume, fixed-evaluation, accounting, numerical, and host-stability audits pass
may a fresh run-bound formal authorization be created, up to 998,400 aligned
training transitions. This safety-layout change establishes a new baseline;
it is not a strict single-variable performance comparison with prior 512-env
runs.

That qualification is now complete. A low-load 64-environment smoke first
consumed 6,400 training, 1,600 Brax-evaluation, and 224 fixed-evaluation
transitions and passed the finite-update/checkpoint/video contracts. Because
that stable smoke config did not exercise the formal 256-environment layout,
it was not represented as doing so. A second run used the exact formal path
with 256 parallel environments, 16 minibatches, and one 6,400-transition
rollout block. It completed with 216 fixed-evaluation transitions; transition-0
and transition-6,400 checkpoint sidecars passed recursive identity validation
(`114352fd...cf87ad` and `8981fd8b...9b5f3`). The GPU returned to 956 MiB at
38 C after completion. There was no OOM, broadphase overflow, NaN/Inf, timing
or history mismatch, hash drift, physical failure, illegal contact, action
saturation, or host crash. All held-out rollouts reached the legal jump window
and ended at the retained missed-liftoff deadline; liftoff and Apex success were
0/8, as expected for an engineering smoke. Failure MP4/NPZ pairs are retained
under both ignored run directories.

The 256-environment formal path is therefore engineering-qualified for one new
run-bound Phase U authorization. The hip-std-0.10 hypothesis remains unproven
scientifically until fixed checkpoint evaluations exist; snapshot acquisition
and continuation probing remain conditional on real Apex-success parent
coverage.

The fresh hip-std-0.10 formal run
`phase_u_2kg_env256_hipstd010_apex8_cap4_998400_20260813_seed720703`
stopped under its implemented checkpoint gate at 256,000 training transitions
with `held_out_physical_performance_plateau`. This was a controlled Gate Pause,
not an OOM, host crash, numerical failure, or uncontrolled exception. All 41
transition-aligned checkpoint sidecars through 256,000 pass recursive identity
validation. The run consumed 640 fixed-evaluation transitions and zero
candidate-acquisition or continuation-labeling transitions. Twenty-four MP4
failure videos and twenty-four timing-aligned NPZ traces are retained.

| training transitions | outcomes | window | liftoff | Apex | mean return |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 8 missed-liftoff | 8/8 | 0/8 | 0/8 | -5.781 |
| 102,400 | 8 missed-liftoff | 8/8 | 0/8 | 0/8 | -6.998 |
| 256,000 | 8 missed-liftoff | 8/8 | 0/8 | 0/8 | -5.998 |

All three evaluations had zero physical failure, roll/pitch violation, illegal
contact, action saturation, clearance success, and Apex success. The decisive
reward audit found that the entirely wheel-supported held-out rollouts still
accumulated approximately 6.0--6.7 clearance-progress reward after entering
the jump window. The existing reward code gated ascent and clearance only on
`jump_window_entered`; it therefore allowed a ground-driving policy to collect
nominally airborne shaping. This is a reproduced reward shortcut, not evidence
that the legal-liftoff or Apex contracts are too strict.

The next single hypothesis closes only that shortcut. Define
`airborne_progress_enabled = jump_window_entered AND liftoff_seen` in the
external pure-JAX Phase U adapter and require it for `ascent_progress`,
`clearance_progress`, and `apex_approach`. Forward propulsion, jump-window
entry, legal-liftoff/stable-airborne bonuses, all penalties, weights, reset,
thresholds, PPO, XML, action mapping, and safety termination remain unchanged.
Early airborne remains nonterminal and unpenalized but cannot unlock airborne
progress. The reward-contract hash now binds an explicit semantics version so
old and new checkpoints cannot be confused despite identical numeric weights.
This implementation is source-complete under red-green tests but must pass the
full static/runtime/preflight and bounded smoke gates before another formal
authorization.

The user then explicitly revised the hard-task stopping contract: three flat
held-out checkpoints are diagnostic evidence only and may not terminate PPO
before the full approximately one-million-transition budget. The gate continues
to record `held_out_physical_performance_plateau`, but `pause=false` when that
is the only reason. Nonfinite evaluation, severe action saturation, sustained
physical degradation, and return-up/physics-down reward hacking remain active
immediate Gate Pause conditions. The next formal run is aligned to 998,400
training transitions and will be judged after that budget unless one of those
retained true failure conditions occurs.

The combined airborne-progress and hard-task stopping-policy implementation
now passes its red-green regressions, all 73 Phase-U expert tests, static
compilation, the full 912-test suite, and `scripts/local_preflight.sh`. A fresh
managed GPU runtime gate at
`runs/two_phase/runtime_gate/phase_u_2kg_airborne_progress_gate_20260813/`
passes in 95.32 seconds, including the fixed 64-transition update and
32-transition resume. It consumed 96 runtime-integrity transitions and zero
Phase U expert-training transitions. The next permitted action is one new
run-bound 256-environment, 6,400-training-transition formal-path smoke.

That formal-path smoke has now completed as
`gate_c1_phase_u_2kg_env256_airborne_gate_smoke_20260813_seed720801`. It
consumed 6,400 PPO training and 424 fixed-evaluation transitions (6,824 total),
with zero candidate-acquisition and continuation-labeling transitions. The
transition-0 and transition-6,400 checkpoint sidecars pass recursive identity
validation as `41727c68...55b3d` and `a63e63f5...af5d`. Both held-out panels
closed as eight `takeoff_missed_liftoff_deadline` outcomes: 8/8 reached the
legal window, 0/8 legally lifted off, and 0/8 reached Apex. Crucially,
`ascent_progress`, `clearance_progress`, and `apex_approach` were exactly zero
at both checkpoints. There was no physical failure, roll/pitch violation,
illegal contact, action saturation, numerical/runtime fault, identity drift,
or host instability. Sixteen MP4/NPZ failure pairs are retained. This smoke
qualifies one fresh 256-environment formal run up to the aligned 998,400
training-transition ceiling; early flat checkpoints are diagnostic only and
do not consume that authorization prematurely.

That one-shot authorization has been created and the fresh formal run is now
active:

```text
run id: phase_u_2kg_env256_airborne_gate_998400_20260813_seed720802
producer HEAD: 6ea37be70926e4c2f371cb91d35b43c0503a999d
source-tree hash: a95d5458a11cd172c452c86eb8bfc9f3df556945d08734a0f3840c2c0ecb84a3
reward-contract hash: 1686e196eb27da2980b592a1ace354899d239b15d9487e04c3027daed450e114
worker PID: 287484
parallel environments: 256
authorized/effective training ceiling: 1,000,000 / 998,400
effective checkpoints: 0/102,400/256,000/499,200/748,800/998,400
status: runs/two_phase/phase_experts/phase_u_2kg_env256_airborne_gate_998400_20260813_seed720802/status.json
metrics: runs/two_phase/phase_experts/phase_u_2kg_env256_airborne_gate_998400_20260813_seed720802/metrics.jsonl
log: runs/two_phase/process_logs/phase_u_2kg_env256_airborne_gate_998400_20260813_seed720802.log
completion marker: runs/two_phase/process_logs/phase_u_2kg_env256_airborne_gate_998400_20260813_seed720802.finished
```

The single startup audit found `status=running`, the worker using the GPU, and
a complete transition-0 checkpoint whose recursive identity is
`ca5e6dd2...b2236`. Source, XML, threshold, reward, and reset identities all
match the run authorization; no numerical, broadphase, timing/history, or hash
fault was present. A detached local watcher checks only process existence once
per minute and writes the completion marker on exit; it does not poll metrics
or logs and consumes no model tokens. The approximately one-million-transition
run must continue through early zero-success or flat checkpoint panels.
Candidate acquisition and bounded continuation probing remain automatically
gated on real Apex-success parent coverage.

That run completed the full aligned budget rather than stopping on its
diagnostic plateau:

```text
training transitions: 998,400
fixed-evaluation transitions: 1,304
total environment transitions: 999,704
candidate-acquisition transitions: 0
continuation-labeling transitions: 0
status: completed
```

All 157 checkpoint sidecars pass recursive identity validation.  Six fixed
held-out panels at 0/102,400/256,000/505,600/755,200/998,400 close their
outcome accounting, and all 48 failures have retained MP4 and timing-aligned
NPZ traces.  No NaN/Inf, OOM, broadphase overflow, state/timing/history fault,
identity mismatch, illegal contact, roll/pitch violation, or action saturation
occurred.  Every panel reached the jump window in 8/8 rollouts.  The first
three had 0/8 `liftoff_seen`; the last three had 8/8 apparent liftoff but 0/8
stable-airborne, ascending, clearance, or Apex success.  All outcomes remained
`takeoff_missed_liftoff_deadline`, so no candidate or continuation work was
eligible.

Offline reconstruction of the saved 505,600/755,200/998,400 traces against
the authoritative XML identifies a second event/reward shortcut.  At the
998,400 checkpoint's tick 22, the front tire bottom remained approximately
`-0.0017 m` while the rear tire bottom was `+0.0127 m`; the physical dual-wheel
liftoff predicate was false on every tick.  The other two apparent-liftoff
checkpoints show the same one-wheel pattern.  The external runtime had treated
loss of simultaneous stable support as `liftoff_seen`, thereby paying the +8
legal-liftoff bonus and airborne shaping while the retained takeoff deadline
correctly stayed armed.

The next single hypothesis fixes only that admission error.  `liftoff_seen`
now requires a previously latched legal window plus the existing deployable
temporally confirmed `ApexBandSignals.stable_airborne` signal and no physical
failure.  One-wheel, momentary, or early airborne remains nonterminal and
unpenalized but cannot unlock liftoff or airborne progress.  XML, action
mapping, thresholds, reward weights, PPO, reset, observation, horizon, and all
safety failures remain unchanged.  The hashed semantic identity is now
`phase_u.confirmed_airborne_liftoff_required.v3`.

This change passed its two expected RED regressions, 116 affected runtime and
Phase-U tests, static compilation, the full 914-test suite, and
`scripts/local_preflight.sh`.  A fresh managed GPU runtime gate at
`runs/two_phase/runtime_gate/phase_u_2kg_confirmed_airborne_liftoff_20260813/`
passes in 95.60 seconds, including its fixed 64-transition update and
32-transition resume.  The next permitted action is one fresh exact
256-environment, 6,400-training-transition formal-path smoke.  No further
formal PPO is authorized until that smoke passes.

The required static threshold provenance refresh is retained under
`runs/two_phase/gate_c1_20260813_confirmed_airborne_threshold_refresh/`.
It consumed zero environment transitions, preserves the selected Apex and
Recovery threshold values byte-for-byte, and updates only the bound
`two_phase_runtime.py` source identity.  The refreshed canonical threshold
manifest hash is
`66591b997a8dd1d9ef7698c210266074d1b4de83101edcce15095a7b955115ad`;
the normal Gate C1 loader validates its authoritative XML, reference, config,
code, geometry, action mapping, and canonical identity.

The confirmed-airborne formal-path smoke then completed as
`gate_c1_phase_u_2kg_env256_confirmed_airborne_smoke_20260813_seed720901`.
It consumed 6,400 PPO training and 432 fixed-evaluation transitions (6,832
total), with zero candidate-acquisition and continuation-labeling transitions.
The transition-0 and transition-6,400 checkpoint sidecars validate recursively
as `bafa707a...e74fe` and `ba877c2d...9580b`.  Both held-out panels closed as
eight `takeoff_missed_liftoff_deadline` outcomes: 8/8 reached the legal window,
0/8 had confirmed liftoff, and 0/8 reached Apex.  Legal-liftoff,
stable-airborne, ascent, clearance, and Apex reward sums were exactly zero.
There was no physical failure, numerical/runtime fault, identity drift,
action saturation, or host instability.  Sixteen MP4/NPZ failure pairs are
retained.  This smoke qualifies one fresh 256-environment formal run up to the
aligned 998,400 training-transition ceiling; early flat checkpoints remain
diagnostic only.

The earlier 4 kg one-million Phase U authorization was effectively exhausted at 995,200
aligned expert-training transitions. No additional long PPO run, Phase U
snapshot acquisition, or continuation probing is currently permitted: the
remaining 4,800 transitions cannot form one valid 512-environment rollout
block, and the acquisition gates have no successful parent trajectories.

The next scientifically defensible hypothesis is to reduce only the Phase U
hip initial action standard deviation from 0.50 to 0.25. A 0.25 prior still
samples a +0.50 hip action with finite frequency across 512 environments, while
avoiding the 16% one-sided tail rate produced by a zero-mean 0.50 prior. Unlike
the previous scalar-0.25 run, steer, drive, and knee remain at 0.05. The purpose
is specifically to preserve stochastic trajectories long enough for the
existing distance feature and running normalizer to cover the legal window;
it is not a reward, reset, observation, network, optimizer, horizon, or physics
change. Adding a geometry feature or changing normalization remains a fallback
only if this distribution-preservation hypothesis is separately authorized,
bounded, and falsified.

The static configuration for this hypothesis is now implemented: both
`configs/phase_expert_smoke.json` and `configs/phase_expert_phase_u.json` use
the ordered vector `[0.05, 0.05, 0.25, 0.05]`. No reward code changed. In
particular, early airborne remains nonterminal, earns no window/ascent/
clearance/Apex progress before legal window entry, and receives no new
early-airborne penalty. Forward-propulsion reward remains active so the vehicle
can learn to reach the window. This is static implementation only: it consumed
zero new environment transitions and has not passed a dynamic PPO smoke.

That iteration required preserving the 4 kg payload and had no remaining
aligned training block. It is now superseded only by the explicit 2 kg
single-variable authorization recorded above. The action mapping, 50 N m
limits, natural reset, early-airborne nonterminal rule, and unchanged
roll/pitch/contact/nonfinite safety gates still remain mandatory.

This marker does not authorize formal `V_up`, a learned Soft Tube declaration,
Phase D expert training, unified PPO, or JCE/JEL. Provisional acquisition aids
must remain clearly labeled and cannot shape Phase U reward or reset sampling.

## Closed routes

- Landing -> Flight -> Takeoff -> Approach sequential shared-Actor bootstrap
- exhaustive H1/C_L A/B
- roll-targeted shared-Actor retention
- sequential Flight retention repair
- final-shared v1/v2 and corrected Apex follow-ons
- five-stage phase-balanced unified RSI controllers
- universal 4 -> 8 -> 16/32 requirement for training Tubes
