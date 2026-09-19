# Neighborhood-conditioned exploration Implementation Plan

> **For agentic workers:** Use subagent-driven-development for independent model and restore tasks; root owns integration and launch.

**Goal:** Add a causal frozen-batch local map to the explorer, mix onset times, validate accelerated restoration, and launch the user-authorized 150-round run.
**Architecture:** Immutable per-round map filtered by source actor identity and phase; batch CPU KDTree queries, saved augmented observations, shared-per-neighbor MLP and masked pooling for each Actor/Critic; full historical state restoration remains authoritative.
**Tech Stack:** Python, NumPy/SciPy, Torch RSL PPO, JAX/MJX Warp.

## Global Constraints
- JIT only, preserve user files and old frozen runs; separate worktree and new run.
- 1024 candidates, three pulse steps, delta .25, six onsets [5,10,15,20,25,0], 128000 repair transitions, stable_forward_recovery .5s, horizon400.
- 150 production rounds; validation budget separately recorded, no final TEST.
- Frozen map at batch start; new observations stored on tape; actor and critic same map version; reward uses prior visited set and existing duplicate sharing.
- Neighborhood row: 12 normalized relative coordinates + 4 evidence flags (current success/current failure/repair success/repair failure) + valid mask =17. K16. Eight statistics: near/far log counts, near/far success/failure fractions, near/far nearest normalized distance (empty capped2). Base106 +16*17+8 =386 input. Empty records zero/mask0. Evidence labels distinct and unknown unpunished.
- Medium widths [.2,.05,.1,.4,.2,.6,2,5,5,10,20,20]; far=2x. Group by current source actor hash and phase. Preserve legacy observations without neighborhood config.
- Fresh explorer initialization for new observation identity; use current stopped source policy as warm-start base, record inherited lineage. No old optimizer incorrectly restored into changed architecture.

## Tasks
- [x] Model: tests for masked/permuted neighborhoods, nonzero context influence and Torch/JAX distribution parity; implement configurable encoder and safe checkpoint identity.
- [x] Retrieval: tests for scale/phase filtering, empty masks, immutable inputs, no future labels; immutable index and per-round map artifact.
- [x] Sampling: tests for balanced rotating mixed onsets, exact three applied steps and end-of-pulse snapshots; store causal augmented obs, per-lane onset and phase.
- [x] Restoration: validate optional batched/fused preparation against canonical serial reconstruction; retain legacy fallback and fail closed on mismatch.
- [x] Integration: regression suite; bounded actual GPU collect/evaluate/update plus restore parity/timing, separately counted.
- [x] Production: freeze code/config, calculate 150-round maximum reservation, launch gated loop and desktop watcher; verify live first-stage progress and heartbeat; update docs, commit and push.

## Progress
Initial live audit: previous JIT lineage stopped by user after104 rounds. Other project training remains unrelated. All work starts from4dc3789 in agent/neighborhood-explorer.

CPU regression:84 passed. GPU24-lane mixed sampler and real PPO update passed. Fused restore not adopted: suffix parity failed with canonical-repeat variance too;4592 measured validation steps. Original canonical restore retained. Full-loop smoke completed:141099 interactions,459.82s,including repair/reassessment/retention/update/promotion reseeding.150-round run launched from frozen commit1af91cb; supervisor869897, watcher869898 healthy; GPU child running nominal seeding. TensorBoard6008. Main agent branch fast-forward integrated; delivery commit follows. Runtime worker caching deferred, not claimed implemented.
