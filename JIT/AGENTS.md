# JIT implementation guidance — 2026-09-12

Root [AGENTS](../AGENTS.md) governs. Read the [latest complete handoff](docs/CODEX_HANDOFF_20260911.md) and [roadmap](docs/JIT_TRAINING_ROADMAP.md) before editing.

Production all_proposers_v1 is COMPLETE: pi_5/pi_6 trained and frozen, 9,296 historical campaign root cells, 256,000 new PPO transitions. First-success stopping, checkpoint comparisons, vectorized continuation engineering, frozen-base residual PPO, per-policy arrival rewards and pending/delayed learning now exist. The latest two-round pilot completed two128k policy trainings with zero delayed no-witness-to-witness gains. The old supervised residual and full-action experiments are superseded research stages, not current architecture. Diffusion, critic quality penalties, adaptive proposer allocation and automatic training extension remain unimplemented. See CURRENT_STATUS for current evidence.

Implementation priorities:

1. Explain current novelty saturation and candidate caps from saved data, including active/padding and effective residuals; no new simulation needed for this analysis.
2. Reuse phase geometry and audit root-grid sensitivity separately from5cm candidate/plot spacing. Preserve historical landing/failure conflicts and accepted replay limitations.
3. Preserve current first-success/unknown semantics and full-matrix comparison mode; never turn missing/error/conflict into negative or compare different context identities.
4. Use bounded128k–256k working budgets instead of another open-ended length sweep. A4-state panel is diagnostic, not proof that pending candidates were learned.
5. Predeclare fair same-mode residual/random and independent pending-enabled/witnessed-only successor comparisons before claiming an exploration or training advantage. Diffusion is optional later work.

Core files: envelope_campaign.py, campaign_bank.py, iterative_probe_training.py, dense_tube.py/dense_tube_runtime.py, acquisition/causal_jump.py, continuation shard modules, analysis/capability_tube.py, result_bundle.py/result_publishing.py. Extend these capabilities instead of making another iteration-specific duplicate pipeline.

Historical production reset recipe is20% complete fixed x2.5 start /80% witnessed snapshots. Current delayed recipe allocates25% of a phase's snapshot mass to pending when available, preserving phase/group balance. In the latest pilot pending training support is upstream only, giving about10% total reset probability. New pi uses Actor+normalizer initialization and fresh critic/optimizer. Keep support separate from verified envelope admission. No whole-Tube retention gate. Preserve complete history; legacy scans retain original identities/schema.

Respect the declared backend and budget for each stage.4096/8192/16384 vectorized capacity engineering is complete, but a38→39tick difference prevents assuming exact equivalence to locked serial experiments. Large capacity does not authorize competing jobs or replacing old scientific protocols. A CPU fixture pass is not a GPU rollout result.

Store each round's replot CSV/manifest, individual PNG/PDF/SVG, receipts and cost. Preserve observed gaps, missing evaluator outcomes and historical bank versions. Full images stay server-side; GitHub compact report absence of images is not a plotting failure.

Do not open final TEST. TRAIN adaptation and development ACCEPTANCE are not final held-out performance. Data rows share ancestors and are not independent repetitions. Follow root authority for authorization, git safety and report retrieval.
