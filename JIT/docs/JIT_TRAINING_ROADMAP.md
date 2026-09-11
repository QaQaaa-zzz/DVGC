# JIT implementation and experiment roadmap — 2026-09-11

Current baseline: completed all_proposers_v1, pi0–pi6 bank. Read [full handoff](CODEX_HANDOFF_20260911.md). No further broad campaign is the immediate next command.

| Order | Deliverable | Acceptance criterion |
| --- | --- | --- |
| 1 | Forward event audit | Explain four landing/failure conflicts, check collection/training/suffix event priority; immutable source + scoped derived impact report |
| 2 | Replot and timing | Phase5cm geometry, full successful low trajectories, resolution sensitivity and named cumulative baseline; stage start/end/elapsed timing, no new broad simulation for existing plots |
| 3 | Existence-only labels | New versioned first-success path; all-bank no-witness only after complete failures; untested distinct; same existence decisions on a fixed full-label panel, retries/cost/identity preserved |
| 4 | Training-length evaluation | Save32k/64k/128k, fixed TRAIN development panel and equal small exploration budgets, explicit maximum steps/extension rule and optimizer continuation semantics; all evaluation charged |
| 5 | Lightweight residual explorer | Frozen-bank fixed-perturbation control vs policy/state/goal-conditioned bounded residual, real prefix + suffix evidence, equal total cost |
| 6 | Paper experiments | Fixed-bank/no-new-PPO versus iteration; independent training/search repetitions; complete cost, geometry/resolution, locked final test and claim-dependent bootstrap ablation |
| Optional | Diffusion perturbation generator | Only after residual baseline/data audit; demonstrate temporal/multimodal benefit at measured added cost, no generated-state reachability claims |

First-success labeling changes the label matrix contract: do not just insert break into the old complete-label protocol. Keep full evaluator matrices for small common-panel comparisons. Missing labels never become zero. Stop exploration after a valid bank witness only in the new declared mode.

Batch/process acceleration must be measured against actual serial outcomes and memory. Do not repeat failed vectorization benchmarks blindly. 413 process launches are a concrete overhead candidate, not a measured attribution of all wall time.

128k is a pilot budget, not a convergence finding. Do not retrain old policies solely to recover missing intermediate checkpoints. New training recipes must state mixture, support, initialization, budget, seed and stop rules.

Residual explorer rewards should favor NEW cumulative witnessed support, not merely a large perturbation or policy-specific unfamiliarity. Avoid overly strict upright penalties that remove legitimate jumping poses. Freeze explorer/base-policy versions per stage; new policy IDs need an explicit conditioning/generalization design.

Diffusion Policy is a potential action-sequence generator, not proof of physical reachability or a substitute for exploration objectives. It is not implemented, required, or automatically beneficial for the current small correlated trajectory dataset. Preserve fixed perturbation and simple residual baselines.

Retain user-approved near-ground initialization and accepted replay limitations. No new exact replay gate. Final TEST remains unopened during development. Equal per-proposer32-trajectory allocation is current behavior; gain-adaptive allocation remains future work.
