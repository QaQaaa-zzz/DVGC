# JIT implementation guidance — 2026-09-11

Root [AGENTS](../AGENTS.md) governs. Read the [latest complete handoff](docs/CODEX_HANDOFF_20260911.md) and [roadmap](docs/JIT_TRAINING_ROADMAP.md) before editing.

Production all_proposers_v1 is COMPLETE: pi_5/pi_6 trained and frozen, 9,296 campaign root cells, 256,000 new PPO transitions. The next task is efficiency/interpretation and controlled experiments, not repeating the same command. All-policy fixed perturbations exist; first-success stopping and fixed-panel multi-checkpoint evaluation now have opt-in implementations. The bounded residual module is a supervised warm-start prerequisite only. Actual learned exploration, diffusion, adaptive proposer allocation and automatic training extension remain unimplemented. See CURRENT_STATUS for validation scope.

Implementation priorities:

1. Audit four forward receipts with simultaneous landing/failure flags. Compare causal_jump.py event ordering with iterative_probe_training.py and continuation endpoint logic. Preserve original data; don't conflate this with the accepted replay limitation.
2. Reuse phase-separated geometry and trajectory data; preserve high-dimensional physical grid independently of 5cm slices. Add complete stage wall-clock timing.
3. Version a first-success existence-label path. Full common-panel labels remain available for policy comparison. Untested is explicit; no missing-to-zero conversion. Complete bank failure requires every declared evaluator completed. Preserve hash/endpoint/catalog/payload/context/seed/horizon checks and retry accounting.
4. Add bounded32k/64k/128k checkpoints and fixed TRAIN small-panel/exploration evaluation; charge all work. Define continuation optimizer state and maximum steps before extending training.
5. Frozen-bank lightweight residual exploration vs fixed-perturbation baseline. Preserve causal prefix and new explorer identity; then decide whether diffusion sequence generation is worth a separate experiment.

Core files: envelope_campaign.py, campaign_bank.py, iterative_probe_training.py, dense_tube.py/dense_tube_runtime.py, acquisition/causal_jump.py, continuation shard modules, analysis/capability_tube.py, result_bundle.py/result_publishing.py. Extend these capabilities instead of making another iteration-specific duplicate pipeline.

Production reset recipe is20% complete fixed x2.5 start /80% witnessed snapshots, Actor+normalizer initialization and fresh critic/optimizer. No whole-Tube retention gate. Preserve phase/group balancing and complete control history. Legacy three-policy scans keep original identities and schema; new partial evaluation must not retrofit them.

Server GPU stages are serial with bounded fresh-process shards after historical OOM/device-map failures. Larger shards/persistent workers/vectorization require measured capacity and outcome checks; do not assume more concurrent workers are safe. A CPU fixture pass is not a GPU rollout result.

Store each round's replot CSV/manifest, individual PNG/PDF/SVG, receipts and cost. Preserve observed gaps, missing evaluator outcomes and historical bank versions. Full images stay server-side; GitHub compact report absence of images is not a plotting failure.

Do not open final TEST. TRAIN adaptation and development ACCEPTANCE are not final held-out performance. Data rows share ancestors and are not independent repetitions. Follow root authority for authorization, git safety and report retrieval.
