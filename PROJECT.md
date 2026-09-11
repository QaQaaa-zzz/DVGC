# DVGC / JIT — empirical jumping envelope discovery

Updated2026-09-11. The project studies fixed-condition jumping locomotion of a single-track two-wheeled bicycle–pendulum robot. The research product is a growing empirical capability envelope supported by real arrivals and successful landing continuations, not a single policy required to master every collected state.

## Current outcome

The two-round all-policy production campaign completed. pi_5 and pi_6 each trained128,000 transitions and were frozen. Coverage:1,689 inherited →4,629 →9,296 root cells; new candidates8,470; new interactions1,275,465. Stop reason was the declared two-round cap. This is development evidence, not convergence or a global physical boundary.

Old policies still discover substantial new space. The earlier pi_2/pi_4 retrospective control showed pi_4 exploration novelty benefits but not yet a training-inclusive cost advantage at its smaller matched budget. Multi-policy gains cannot all be assigned to newest training.

[Full handoff, results, risks and next tasks](JIT/docs/CODEX_HANDOFF_20260911.md) is the primary successor brief; [current status](JIT/docs/CURRENT_STATUS.md) is the concise record; [roadmap](JIT/docs/JIT_TRAINING_ROADMAP.md) is the implementation sequence.

## Method

Bootstrap up/down support → successful unified pi_0 and captured centerline → frozen-policy bank plus bounded legal action perturbations → real arrivals → successful suffix witnesses → deduplicated empirical Tube → witnessed-support PPO → new frozen policy → repeat within budget.

All four action channels are allowed. Fixed x2.5 near-ground start and accepted replay limits remain. Every candidate keeps provenance and full context. Phase-separated5cm slices support visualization; the existing physical quantization remains unchanged. No artificial state lifting/lowering, no interpolated reachable hull, no whole-Tube Actor gate.

## Implemented versus proposed

| Capability | State |
| --- | --- |
| Growing frozen bank, all-proposer fixed perturbations, two-round PPO loop | Production completed |
| Per-proposer/evaluator figures, replot CSV, phase slices, receipts, cost and automatic text report publication | Implemented; production artifacts reported; not all images visually inspected remotely |
| First-success evaluator stopping | Proposed, not implemented; current full-bank serial labels dominate interaction work |
| Multi-checkpoint capability evaluation and adaptive training length | Proposed, not implemented; current checkpoints0/final128k |
| Gain-based proposer budget allocation | Proposed; current equal32-trajectory calls |
| Learned residual exploration network | User-requested research direction; design proposed, not implemented |
| Conditional diffusion perturbation sequences | Optional later comparison; neither implemented nor established as beneficial |

## Next development

Audit four conflicting forward landing/failure receipts, produce geometry/sensitivity views, measure stage wall time, implement versioned existence-only labeling alongside full comparison panels, then checkpoint evaluation and a lightweight residual exploration baseline. Preserve the completed all_proposers_v1; do not start pi_7 or rerun that command by default.

Residual exploration should generate bounded actions conditioned on the frozen policy/state/goal and be rewarded primarily for new witnessed cumulative support. A diffusion model could generate temporally structured perturbations, but cannot certify reachability or automatically seek boundaries. Its extra cost must be justified against the lightweight baseline.

## Paper scope

Submission target discussed: RAL. Still needed: fixed-bank versus training-iteration total-cost comparison, independent repetitions, phase geometry/resolution evidence, full lifecycle costs, locked final evaluation and bootstrap ablations if their benefit is claimed. Do not invent a completion percentage or promise acceptance after a fixed number of rounds.

TRAIN is adaptive support; CALIBRATION serves optional predictors; used ACCEPTANCE is development. Final TEST/JCE/JEL remains unopened. Tube0 was weighted support (222 rows,42 historical negatives); preserve actual pi_0 lineage and the invalid historical pi_3 mixed-endpoint gate.

## Operations

Working branch agent/two-phase-soft-tube; report branch agent/jit-run-reports in QaQaaa-zzz/DVGC. Full artifacts on /home/qy/DVGC, compact reports on GitHub. Routine project pushes and report publication already authorized. Raw experimental runs immutable; new protocols need new outputs. Follow [AGENTS](AGENTS.md).
