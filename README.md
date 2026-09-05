# OrangeBike DVGC / JIT

JIT studies **bootstrapping and budget-controlled discovery of empirical jumping envelopes** for a fixed bicycle–pendulum robot. A growing bank of complementary frozen policies supplies forward proposals and successful landing continuations. A single Actor need not realize the entire cumulative Tube.

## Research scope

The task begins at the complete declared ground preparation state at x=2.5 m. Exact states enter empirical support only after real forward dynamics and a successful first-valid-landing continuation from the same state/context. Training resets do not create arrival evidence. Physical cells describe sampled support, not a continuous safe region or a complete physical limit.

Phase-specific up/down learning supplies initial value-weighted training support, followed by unified seed development. The frozen pi_0 real trajectory is a longitudinal exploration coordinate. The new direction permits multiple frozen proposers/evaluators and retains valid historical witnesses independently of later Actor regressions.

## Current implementation boundary

Reviewed remote baseline: `bfc22f2`. It still uses a fixed pi_0/pi_1/pi_2 family and single-successor workflow. The new bank/registry/training recipe requires implementation and validation. Documentation alignment does not fix the runtime defects recorded in the review.

Recorded evidence includes initial 222-row weighted Tube0 (42 historical negative labels), a later Round1 pi_0 identity, 1,230/1,258 TRAIN family witnesses in the wide scan, 713 reported new causal root cells, and completed pi_3 training. The pi_3 historical core comparison mixed success endpoints and cannot serve as a fair comparison. Larger locked catalogs/scores exist, but family labels are incomplete after GPU failures.

Start with integrity fixes, small runtime replay/shard checks, and a matched-budget pilot using existing probes before another large PPO run. Full-Tube Actor mastery and predictor quality are not prerequisites for empirical support discovery. Final TEST/JCE/JEL stays unopened for this work.

## Read order

1. [AGENTS.md](AGENTS.md) and [JIT/AGENTS.md](JIT/AGENTS.md).
2. [Project definition](PROJECT.md) and [current status](JIT/docs/CURRENT_STATUS.md).
3. [Paper outline](JIT/docs/JIT_PAPER_OUTLINE.md).
4. [Code and evidence review](JIT/docs/JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md).
5. [Iteration protocol](JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md) and [training roadmap](JIT/docs/JIT_TRAINING_ROADMAP.md).
6. [Handoff](JIT/docs/CODEX_HANDOFF_20260904.md) and [code organization](JIT/docs/CODE_ORGANIZATION.md).

## Runtime and repository

Fixed XML: `assets/orange_bike_4kg_horizontal.xml`; 2 kg payload; 0.005 s simulation step; 0.020 s control; actions steering/rear-wheel drive/hip/knee; hip/knee +/-30 N m. Do not change task/physics/reward/reset semantics silently.

Scientific logic is in `JIT/src/jit_dvgc/`, thin CLIs in `JIT/cli/`, tests in `JIT/tests/`, configs in `JIT/configs/`, guidance in `JIT/docs/`, and lightweight evidence in `JIT/runs/`. Large artifacts need a resolvable external index and preserved identities.
