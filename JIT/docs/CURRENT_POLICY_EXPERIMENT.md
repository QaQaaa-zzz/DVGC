# Current-policy-only iterative experiment

Approved by user 2026-09-14. Scope JIT only. Extend the existing pulse loop.

Implementation plan (writing-plans; implementation already authorized):
1. Reject initial banks containing any policy besides the declared source.
2. Generate a new deterministic, zero-residual source rollout from the fixed
   start; retain full tape, extract nonterminal snapshots every two ticks,
   evaluate them with source only. Require witnessed upstream and downstream
   support. No old campaign snapshots/ledger/explorer weights are imported.
3. Current source generates prefixes, alone evaluates candidates, and initializes
   successor PPO. Maintain new-lineage witnessed replay separately from labels.
4. After each repair, reevaluate pending states and a 16-state phase-balanced old
   panel plus fixed initial state under source/successor. Adopt only with at least
   one newly successful pending state, fixed-start success, no unknown on previously
   successful panel states, and >=90% retention on the sampled panel. This is a
   development choice, not whole-Tube retention or a stability guarantee.
5. On adoption regenerate nominal support/ledger for the new source and reset the
   explorer/optimizer; never reuse old PPO batches under a changed base policy.
   Preserve previous per-source ledgers and all trained policies as evidence.
6. Initial run:12 rounds,128 candidates/round,3-tick pulses,amplitude0.15,
   times5/10/15/20/25/0,128k PPO maximum per triggered repair. Finite worst-case
   interaction cap includes every possible reseeding and paired evaluation.
   No action-retention penalty (prior comparison showed no advantage). Physics,
   success semantics and original pi0 checkpoint unchanged. TRAIN only.
7. Behavioral tests cover singleton initial bank, promotion/unknown semantics,
   budget, legacy loop regressions. Persist losses, rewards/components, KL,
   configs, full traces, plotting CSV/PNG/PDF/SVG and actual/reserved costs.

The frozen pi0's original training history and environment configuration remain
dependencies, not newly sampled training rows. This is conditional on an existing
pi0, not a from-scratch bootstrap comparison. No historical pi1..pi6 validation.
