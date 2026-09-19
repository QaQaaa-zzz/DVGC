# Neighborhood-conditioned short-pulse exploration

The optional `neighborhood` configuration augments the existing residual explorer;
legacy configurations/checkpoints retain their 106-dimensional observation.
A new explorer is initialized when enabling this architecture. The base policy
remains frozen during each collection batch.

## Observation and timing contract

The default raw input contains the existing106 features,16 records of17 features,
and8 local statistics:386 total. Each record contains12 signed relative physical
coordinates in configured halfwidth units, four separate current-success,
current-failure, repair-success, repair-failure evidence flags, and a valid mask.
Unknown outcomes set neither success nor failure. An unsuccessful bounded repair
is not evidence of physical impossibility. Context conflicts are quarantined.

Actor and Critic independently encode each neighbor with16→64→64 ELU, perform
masked mean pooling, and concatenate the64 summary and8 statistics with the106
base features. The existing128×3 heads remain. Padding never affects the summary.
Only the base features use the frozen base observation normalizer; the added
features have declared physical scales. Checkpoints bind dimensions, scales and
feature layout; changing these requires a new explorer identity.

A per-round JSON map is frozen and hashed before collection. It contains only
previously observed source-policy records and nominal seed evidence, grouped by
source Actor SHA and phase. Initial nominal seeds are generated from the new
run's declared base policy; previous run outcomes are not silently imported.
Current and repaired labels are distinct. Counts refer to unique complete
contexts, not visits or certified volume. Near widths are [.2,.05,.1]m,
[.4,.2,.6]m/s,[2,5,5]degrees,[10,20,20]degrees/s; the far box doubles them.
All coordinates must satisfy the box simultaneously. Return at most16 nearest
records, with full-radius counts/fractions and nearest distances computed before
truncation. Empty data remains explicitly empty.

The compiled sampler performs a batched CPU lookup callback on pre-action states
only when a lane has a valid pulse action. The original frozen map is used for
every query in that batch. The augmented observation is saved in `prefixes.npz`;
PPO updates replay this saved observation, never a newly queried map. Map SHA is
part of generator identity. The unchanged novelty reward uses the preceding
visited-cell ledger and shares the per-cell novelty bonus among same-batch
arrivals. Coverage context does not make the full hidden-history process Markov.

`pulse_batch_mode=mixed` balances the declared fixed onset schedule within a
batch and rotates any remainder across rounds. Each lane ends at its own third
pulse action (or earlier physical termination); waiting, active and padding
interactions are accounted separately. Event-triggered schedules cannot be
combined with mixed fixed onsets. The stored phase and onset belong to each lane.

## Verification and limits (2026-09-16)

84 targeted CPU tests passed. GPU24-lane collection covered all six onsets with
four lanes each, exactly72 pulse actions and386-dimensional saved observations.
The source-only diagnostic evaluated19/24 successes (not a policy efficacy test).
The actual RSL update used72 valid samples; Torch/JAX log probability difference
was9.54e-7, replay error0. A full one-round outer-loop smoke completed in459.82s with141099 charged
interactions, including128000 repair transitions, reevaluation, retention, RSL
update and successor nominal reseeding. It repaired2/5 pending cases; retention
was16/16 and fixed-start succeeded. These24-lane engineering results do not
establish improved performance. The production explorer does not inherit this
validation update. Total measured validation cost:148475 interactions
(4592 restore +2784 short pipeline +141099 full loop); two preflight failures
launched no physics and remain recorded.

A fused full-state restoration prototype is available only through the engineering
validation CLI; production continues using canonical restoration. Restored263
leaves matched except two clearance metrics at1.49e-8, but paired suffix endpoints
differed. Canonical repeats also diverged after approximately24 physics ticks.
Consequently no accelerated restoration backend is adopted and no end-to-end
speedup is claimed. GPU tests consumed4592 suffix interactions, recorded separately.
The persistent-worker/runtime-cache proposal has not been implemented here.

Evidence lives under `/home/qy/DVGC/JIT/runs/monitoring/neighborhood_validation_20260916/`.
The new experiment is `/home/qy/DVGC/JIT/runs/experiments/neighborhood_mixed_rsl_20260916/`.
It starts from the stopped run's accepted `lineage_repair_0102` base policy and a
fresh neighborhood explorer. It declares150 rounds,1024 candidates each,3 pulse
steps,delta.25,128000 repair transitions per applicable round,unchanged0.5s recovery
criterion/horizon400, and a maximum reservation of172881600 new interactions.
Validation budgets and inherited base-policy provenance remain separate. This is
TRAIN development evidence, not a matched-cost ablation or final TEST.
