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
