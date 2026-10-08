# Experimental closed-loop implementation plan

User authorization 2026-10-08 supersedes the requirement that zero forgetting gate every research continuation. This is a new experimental continuation, not retroactive formal adoption of A2 candidates.

Design: reuse C as experimental Actor, saved G and cumulative demos; initialize E symmetrically with latent std .6. Two rounds: fresh 128 TRAIN episodes (80% learned E / 20% uniform), same-source suffix labels, up to 32 teacher searches with H16 + current Actor tail, cumulative demonstrations and all valid pending training support, 128000 student PPO, real TRAIN student replay, explicit experimental adoption, E PPO from only current learned known-outcome samples, G incremental 2000, atomic bundle publication. Preserve original reward weights, physical/task protocol, Actor and G structure. Formal deployment adoption remains separate and false. Unknown source repeats never receive invented conversion bonuses.

Experimental adoption requires completed finite learner/checkpoint and nominal full-task successes; old lost roots are reported, not a veto. If nominal fails retain previous experimental source; continue E/G only with eligible feedback. No automatic restart on engineering error. Student feedback admission retains checkpoint-bound adopted receipts, annotated experimental scope.

Fixed development stress panel: amplitudes .25/.4/.6, onset 0/10/20, 32 seeds per condition, paired P0/start-C/every-round policy; no training or G/E admission of these traces. Normalized action pulses, not force disturbances. Original final TEST unopened; repeated panel is development generalization evidence only. No equal-total-budget G causal benefit claim from this continuation.

Budget: two rounds, <=3300000 new physical interactions including baseline/evaluation/validation; 256000 student PPO; G <=4000 updates; E four PPO epochs per fresh round, no historical replay; 24h absolute deadline. Preserve all old artifacts. Gate execution on available GPU resources; never stop other projects.

Implementation tasks:
- [x] New production coordinator built on CampaignRunner and existing E admission/runtime; durable per-stage receipts, aggregate budgets and immutable round identities.
- [x] Experimental adoption, cumulative demo lineage, source-scoped novelty ledgers, strict provenance and unknown masks; fixed G normalizer reference P0.
- [x] Fixed paired stress evaluation and atomic publication, separate formal adoption and budget ledger.
- [ ] CPU regression tests plus bounded real GPU startup; TensorBoard reward visibility and notification watcher.
- [ ] Launch immutable snapshot, update shared research ledger and report exact progress.

Initialization: Actor+normalizer warm start and fresh PPO critic/optimizer per student round (not exact optimizer resume); G optimizer checkpoint continues; E optimizer checkpoint continues. History and source hashes preserved. Current research facts live only in shared research-hub/PROJECT_STATE.md.

Validation before launch: CPU193 passed (30.15s), focused7 passed after review fixes; actual saved C/hash and cumulative1679-demo/62-pending provenance check passed without physics. Independent review found and resolved reservation overflow, startup-failure status, quarantine metadata, exact budget validation and evaluated-candidate identity. GPU closed-loop validation remains pending resource gate.

## User-authorized neighborhood input (2026-10-08)

The unexecuted series_0001 GPU queue is cancelled and replaced, with the same two-round budget. Actor C and saved G remain unchanged; E starts fresh with neighborhood encoder, no 106D checkpoint migration. Opt-in `train_history_v1` observation semantics: base106 +16 neighbors*(relative physical state12 +current Actor success/failure2 +historical Actor success1 +verified teacher success1 +valid mask1) +near/far statistics8 =386 raw inputs. Existing learned encoder compresses neighborhood to64; Actor/Critic heads receive178. Feature scope is checkpoint-bound; legacy configurations retain their original interpretation.

All historical map rows are TRAIN-only real post-pulse contexts. Initial history uses1024 original TRAIN roots,32 C TRAIN evaluations and25 verified teachers, retaining actual evaluated-policy identity. Same-policy conflicting labels are unknown. Arrivals survive Actor changes; current ability flags only match actual evaluated Actor. Teacher success is never converted to student success. E sees sparse nearest historical samples, not a complete or certified reachable set.

Each collection freezes map contents and SHA before sampling. The E behavior receipt pins config/map hash and update validates it. Current collection and future teacher/student outcomes only join the next-round map. No development or final TEST rows enter the map. E uses its recorded386D observation for PPO, never recomputes with a newer map. Reward weights and source-specific novelty ledger are unchanged.

Neighborhood validation:233 CPU tests passed (30.40s), including actual RSL PPO on386D neighborhood observations, uniform/unknown exclusion, map mutation rejection and checkpoint scope mismatch. Initial map1116 evidence rows/1024unique TRAIN contexts includes35source recheck records; known same-policy0/1 conflicts are unknown. Independent review complete; new GPU physical validation remains pending the resource gate.

## Shared GPU continuation (2026-10-08)

User requested concurrent execution instead of GPU-idle waiting. `gpu_shared`
requires explicit authorization and a 20,000 MiB free-memory margin; compute
processes do not veto launch. Other jobs remain running; throughput is not guaranteed.
A successor directory may reuse only completed, hash-pinned P0 stress collections
with identical scientific configuration (gate excluded). Reused physical costs
remain charged within the original two-round budget and original wall deadline.
Never reuse partial stages or replay optimizer updates through this mechanism.

## Bounded continuation after completed pilot (2026-10-08)

User requests 200 additional rounds after checking current results. The completed
2-round series is immutable. A new series continues global rounds3..202 from its
published bundle: accepted Actor C, selected G checkpoint, E optimizer state,
TRAIN history, pending support, demos, corpus, novelty and tail lineage. Students
retain the existing weights-only Actor/normalizer start and fresh PPO critic and
optimizer each round; E/G continue their selected saved optimizer states.

Budget: 200*128000 student transitions, G<=200*2000 updates, E<=4epochs/round,
physical<=200*1500000, 7days, >=20GiB free disk before each round. Keep shared GPU
startup memory gate. Error/nonfinite/deadline/budget/disk failure stops with saved
artifacts, no automatic retry or extension. Baseline P0/start-C panels are locked
and reused without physics charges; per-round student stress panels remain full.
Round identities/seeds continue monotonically; no duplicate historical collection.

Engineering tasks: validate bounded continuation contract; test retained E/G and
round offsets with a fake runner; verify real parent identities; launch a pinned
snapshot; verify current scalar visibility and update shared PROJECT_STATE.md.
Pilot audit: 85 completed execution children and160 finite metric rows; R1/R2
nominal3/4 each so C retained. E optimizer8+6; G4000 performed but incumbent kept.
Stress P0=186/288, C=228/288, R1=112/288, R2=204/288, unknown0. No claim that the
continuing Actor improved or that G has a causal advantage.

## User correction: genuinely continuous student learning (2026-10-08)

This supersedes nominal-gated continuation above. User explicitly cancels nominal
checks and wants the 200 rounds to train the previous student continuously.
Stop series0004 (interrupted during teacher search, before any new student block).
The replacement continues latest evaluated R2 student, NOT retained C. It does
not execute student_nominal or source seed_support nominal qualification. Existing
valid historical witnessed seed states are reused with their original Actor labels;
new current TRAIN rows and all valid pending retain their own provenance.

Each finite completed student is the next experimental training source regardless
of success/forgetting; formal adoption remains false. Only real successful TRAIN
student trajectories enter G corpus. Invalid numerical state remains an engineering
error; it is not a performance adoption threshold.

Legacy pilot inference checkpoints did not save PPO optimizer/RNG. Restore its R2
Actor+normalizer+critic, bootstrap optimizer once and disclose that boundary. Then
save and restore full Brax TrainingState (Adam state, Actor, critic, normalizer),
training RNG and cumulative counters every block, with matching inference hashes.
New round simulator episodes reset to new declared TRAIN support, so this is full
learner continuity, not restoration of simulator contact/episode state. A narrow,
source-hash-pinned adapter adds host init/checkpoint hooks to installed Brax; rollout,
PPO and auxiliary loss equations are unchanged. Missing later learner state fails.

Reuse the existing 200-round/7-day budget; charge interrupted work and <=12800
engineering validation physics within the same total cap. Old pilot, interruption,
engineering and production results remain separate. E continuation, G's existing
checkpoint-selection protocol, rewards, H16, mappings and physics remain unchanged.

## Remove auxiliary random branches (2026-10-08 user instruction)

Scope: newly prepared closed-loop series use 100% learned E (uniform episode
fraction 0) and teacher pools of 1 source control + 16 diffusion proposals, with
zero colored-noise proposals. Keep stochastic E/G sampling, source-control replay,
student learner continuity and fixed-dev G selection unchanged. Existing frozen
plans and artifacts retain the historical 80/20, 32-candidate protocol; this change
does not resume the cancelled series0005 or authorize a new training launch.

Implementation plan: add explicit sampling settings to newly prepared plans and
propagate them to round specs; allow zero uniform fraction in collection/admission;
make teacher pool/search/same-layout replay honor the declared candidate count;
run targeted CPU regressions for both the new protocol and legacy defaults;
record validation and the exact G selection semantics in the shared research ledger.

Validation: 52 targeted tests passed (9.45s), covering learned-only collection,
rejecting old mixture receipts under the new protocol, 17-proposal generation and
same-batch replay, new/legacy round-spec propagation, neighborhood provenance,
source-conflict and recovery behavior. No production training or physical rollout.
New plans explicitly store `uniform_episode_fraction=0.0` and
`teacher_colored_noise_candidates=0`; historical plans without those fields retain
their original semantics. This is a changed 17-world teacher batch, not a claim of
bitwise physical equivalence to the old 32-world batch.

G selection remains `train_incremental` in `diffusion.py`: compare the incumbent
with checkpoints every 500 updates (and final update), using EMA noise-prediction
MSE on the fixed generator-dev fixture. Restore the selected complete state,
including optimizer/RNG/EMA; exact ties retain incumbent. With no new corpus data,
skip G training. Actual series0003 round1 scores: incumbent 0.03509113565 versus
update2000 0.03627486527; round2: incumbent 0.03509169817 versus update2000
0.03657495230. Both selected incumbent; these are denoising errors, not recovery
success rates. No G-selection change was requested or implemented here.

## Core-method diagram audit (2026-10-08, read-only analysis)

Compared against user PDF `JIT-core-method1.pdf` (one-page E/G/student diagram).
The PDF is a method description, not authorization to run or change experiments.

| Diagram element | Current implementation and limits |
| --- | --- |
| Frozen pi_k plus bounded Explorer E | Current source frozen during collection; E PPO updated after student TRAIN feedback. New prepared plans remove uniform coverage; E still samples its stochastic policy. Pulse is only ticks0–2, four channels bounded0.25, not arbitrary-time exploration. |
| G conditioned on state + history | 76D real Actor observation including three FIFO frames; not E's accumulated neighborhood map. G samples16 H16 action sequences (0.32s), executes each prefix then hands back to current pi_k. |
| Return to controllable states | Empirical test: full bridge+pi_k continuation must succeed and pass same-batch replay. No separately learned controllable-set estimator or formal guarantee. |
| Teacher data, imitation + RL, retention | Verified successful teacher trajectories accumulate; student uses joint PPO + imitation (.2 to .05) + fixed P0 action-retention (.2). Continuous mode inherits complete preceding learner; simulator resets are separate. |
| Successful rollouts feed denoising | Implemented TRAIN-only historical, new teacher and new student success groups, requested sampling .5/.25/.25, renormalized if a group is empty. Failures do not become denoising targets. |
| Learning progress feeds E | Uses fresh TRAIN source/student labels and novelty; teacher success alone earns no student-conversion bonus. Fixed student development pressure tests never feed E or G. |
| Iterative G updates | Extra implementation rule absent from figure: select incumbent or intermediate/final G by fixed old generator-dev noise MSE. Training therefore need not advance the continuing G weights. Figure alone does not specify last-versus-best selection. |

Actual inherited G fixture is
`/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0002/campaign/generator_dev_fixture.npz`.
Its manifest hash d6df6e0fbf7044401e238fb2b6db41317760d949c158691fbecdd0f54bd8e848
was recomputed and matched. `source_collections/plan.json` uses seed9281201,
64 generator_dev episodes separated from TRAIN: onset0, three-tick uniform pulses,
amplitudes0.1/0.25, 32 each, source bridge_baseline Actor a06acbc8.
The saved suffix results have30/32 and31/32 successes. The61 successful trajectories
form the window pool; bootstrap_from_results selects256 H16 windows without
replacement and fixes diffusion time indices0..99 and Gaussian noise. Arrays are
observations256x76/actions256x16x4/timesteps256/noise256x16x4. These are256 correlated
windows, not256 independent episodes; the manifest's61 ancestors describe the
pre-sampling pool. Later closed-loop rounds inherit the exact fixture.

G uses the new data in reality: pilot R1 history987/teacher_new4/actor_new0, R2
history991/teacher_new8/actor_new0; actual sampling is2/3 history and1/3 teacher.
Stopped series0005 R3 has history999/teacher_new9/actor_new88 and.5/.25/.25 mix,
new_data=true, but G was interrupted and has no completed selection receipt.
This is admission/optimizer evidence, not proof of improved recovery.

Assessment: a fixed fixture is useful for reproducible old-distribution denoising
monitoring and costs no new physical rollouts. Its MSE is not a measurement of
H16+current-pi recovery; old successful Actor windows omit new failed-state rescues
and a changing tail policy. As the sole incumbent-versus-new veto, it can suppress
adaptation (hypothesis); existing receipts prove rejection of updates, not that
rejected G would recover better. It is a development/validation set, not final TEST.
Repeated selection also prevents calling its score independent generalization.

Recommendation only, not implemented: for the explicitly continuous-learning
objective, retain last finite complete G learner as the next training state and
keep old-dev MSE as telemetry; save a separate best artifact if useful. Evaluate
old and new recovery contexts physically under the same current Actor tail before
claiming teacher improvement. A task-oriented selected-G alternative would need
predeclared old/new development cases and a bounded physical evaluation budget.
Last-weight continuation itself does not guarantee learning or prevent forgetting.
The diffusion-policy paper's Push-T real-world evaluation also uses a fixed train
budget and last checkpoints (except IBC), so fixed-old-MSE selection is not a
required property of diffusion:
https://diffusion-policy.cs.columbia.edu/diffusion_policy_2023.pdf (Appendix H.2).
No algorithm or run was changed as part of this audit.

Pre-push verification (2026-10-08): all184 bridge/generative_bridge tests passed
in62.23s;72 learner-continuation/execution-gate/TensorBoard/neighborhood tests passed
in9.84s (256 total). Independent read-only code review found no blocking issue.
These are software regressions, not new physical-performance experiments.

## Scheduled single pulse and continuous G (2026-10-08 implementation task)

User-approved design: scheduled_single_pulse_v1 chooses one balanced onset among
0/5/10/15 per episode; L defaults3 and is validated/parameterized, with request
bounds0.25. Each lane freezes its full post-pulse state at its own onset+L.
New-profile G continuation is last_valid full state with EMA inference; old fixed
dev MSE is monitoring only. Legacy protocols/results are immutable. Student full
learner, reward/loss/physics/H16 and data-role isolation remain unchanged.

Work plan (current task):
- [x] E timing, early terminal classes, masks, receipts and L1/3/5/8 CPU tests.
- [x] G complete last-valid inheritance, provenance telemetry and adversarial tests.
- [x] Strict profile, stopped-parent completed-bundle validation, CLI dry-run and budgets.
- [x] Independent review and relevant regressions. Immutable plan prepared after code commit.
- [ ] Two-round integration pilot: student128000/round, G<=2000/round, E<=4epochs,
      <=1500000 physical/round, <=3000000 total,24h; stage failures stop, no retries.
- [ ] If and only if pilot engineering acceptance passes, launch user's explicitly
      requested200-round successor with its own frozen budget/resource gate.

Pilot acceptance: both atomic bundles fully published; exact parent-child
Actor/E/G state identities and learner RNG/optimizer continuity checked; collection
arrays prove all four single onsets and local snapshot endpoints; G receipt points
to actual last full state even if old-dev best is older; TRAIN/DEV separation and
cost ceilings hold. Physical success improvement is not an engineering gate or a
claim implied by acceptance. Finite optional paired G monitoring is reported
separately, with NA for no failing-tail denominator and unknowns kept explicit.

Stopped-parent source must resolve to the latest actually published complete
bundle. series0005 published no new round; use its declared ancestor series0003
round2 bundle, including its published Actor C, NOT its unadopted evaluated R2
student or interrupted round3 components. Legacy missing learner permits only
one explicitly recorded optimizer bootstrap; next pilot round must fully resume.

Implementation evidence: scheduled E runtime reports requested/effective residual,
clipping, each lane's snapshot_control_step and early-terminal class. The on-policy
mask excludes pre-pulse, terminal padding and unexecuted actions; a single episode
feedback is deposited once, with gamma1 returns on its genuine action decisions.
Teacher root selection round-robins onset strata (8 each if available), then fills
shortages without dropping any pending student's training support. G sample records
retain source/onset/window offset/segment/recency; unchanged .5/.25/.25 probabilities.

Profile fields map directly to normalized pulse contract/round spec/collection
receipt and generator worker policy; unknown fields are rejected. New-profile CLI
requires --execute for physical work, --dry-run is hash/config/budget inspection.
The original snapshot reconstruction preserves episode/phase/source counters,
FIFO, RNG and task events; its mjx.make_data/forward reconstruction is unchanged,
and is not claimed bitwise restoration of contact-solver internals.

Historical G billing reconciliation:20500 updates in the published retained state,
28000 conservatively billed proposals across its explicit lineage (pretrain20000,
old incremental2000, pilot4000, interrupted reservation2000). Durable proposed
charges27655 imply27654..27655 completed updates; distinguish all three counters.
Interrupted stage reservation is not erased or presented as completed training.

New scientific claims remain pending. Optional paired G physical recovery@K
monitoring is explicitly disabled in this first integration budget; existing fixed
student pressure panels run once per round and are DEV only. The pilot establishes
actual pulse/snapshot execution and full-state inheritance, not superiority of G.

Software verification before pilot:335 relevant CPU tests passed in37.67s;
physical integration and performance conclusions remain pending. Existing dirty
AGENTS/PROJECT/CURRENT_STATUS edits are preserved outside this code submission.

Diagram revision instructions: add Execute + current pi continuation + full-task
verification before Teacher data; replace Return to controllable states with
Bridge to successful policy continuation (empirical, not a controllability proof).
Connect verified teacher successes and independent student TRAIN successes to G's
denoising corpus. Learning progress to E comes from paired TRAIN before/after;
generalization stays a read-only DEV/TEST branch. Next round replaces both pi_k
positions with pi_(k+1); Fixed parameters means frozen within collection/teacher
stage. Show E_k/G_k collection and teaching before student training and subsequent
E_(k+1)/G_(k+1) updates. G's76D three-frame observation and E's TRAIN neighborhood
history are separate inputs.

Two-round read-only acceptance entry: call
`jit_dvgc.generative_bridge.integration_audit.audit_integration(run_root)` after
completion; only `ready_for_200=true` authorizes the separately declared successor
under the user's conditional instruction. It does not score recovery improvement.


## Finite teacher replay rejection and bounded recovery (2026-10-08)

The200-round successor completed3 rounds(R5--R7), then stopped inR8 teacher
candidate6: search success, same17-lane replay failure, source0 in both. Locked
prefix plans/seeds/context match. Saved selected-lane qvel first differs by
3.814697e-6 at the first physical step; tail actions diverge after the16-step
G prefix. This establishes nonrepeatability, not its exact numerical cause.
Old failure records remain immutable. User explicitly authorized fixing and continuing.

Explicit plan/round teacher_replay_failure_policy=reject_finite_candidate accepts
only complete finite, hash/Actor/normalizer/model/context/lane-bound search and
repeat records, unchanged source0/0, selected1/0. Terminal labels are reconstructed
with the existing rule. The candidate is replay_rejected, teacher_found unknown,
no demo and no G teacher sample; valid independent TRAIN feedback/PPO eligibility
remains. No alternate candidate is retrospectively promoted. Missing, unknown,
nonfinite, incomplete or identity-inconsistent execution still stops. Without the
explicit policy the historical abort behavior remains unchanged. Physics and
success criteria are not relaxed; numerical reproducibility is not claimed fixed.

Recovery is from the last complete R7 bundle, never a splice of failedR8. Target
remains200 total:3 complete plus197 remaining. Prior870612 charged physical steps
(including failedR8),6000 G updates and384000 student steps remain booked. New
stage ceiling295500000 physical/394000 G/25216000 student lies within the original
aggregate300000000/400000/25600000 ceilings. Carry the original200-series wall-start
so the7-day deadline is not silently reset. No automatic retries or extra rounds.

Verification:264 related CPU regressions passed in35.40s. Strict guard replayed
the saved failing17-lane records without new simulation; rejected as intended,
no demonstration created. Independent review found no blocker for this contract.
