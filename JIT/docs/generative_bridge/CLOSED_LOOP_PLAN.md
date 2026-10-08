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
