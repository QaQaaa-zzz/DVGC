# Per-policy coverage PPO explorer — first TRAIN pilot

Implemented user-requested privileged106→256→256→256 stochastic4-action policy, same PPO Swish/tanh-normal architecture;161,032 Actor and159,233 fresh Critic parameters. Frozenpi6 Actor is widened with zero extra input rows; original76 observable input statistics preserved, added privileged inputs use fixed unit scaling for this pilot. Full normalized actions[-1,1], no old residual±.15 gate. Critic input includes existing3-frame Actor history; no additional temporal stack.

Reward: each new physical root-grid cell on a clean first-landing trajectory earns total1 relative to THIS frozenpi6 baseline/session ledger; duplicates within a trajectory earn nothing extra and same-batch shared cells split credit. Failed/conflicting/incomplete trajectories earn0 and do not consume cells. No separate landing bonus or inherited jumping reward. Baseline813 cells comes from initial unperturbed policy diagnostic rollouts; it is not the entire pi6 reachable region and is not a global union. Reward paid at terminal, then masked full-episode GAE and clipped PPO. No final TEST, independent repetitions or whole-space convergence claim.

## Outcome

Pilot completed20 rollout batches,284 optimizer updates in56.775s. Initial deterministic32/32 clean landings; final0/32. First stochastic batch (before any training update)31/32,1,403 novel cells. Next batch0/32;17 of20 training batches had zero success. Total training-discovered1,576 cells includes that initial1,403, so it cannot be attributed entirely to learned improvement. No promotion or automatic continuation.

Offline first-update displacement on1,498 old sampled states: mean deterministic action absolute change0.1862,88.65% probability ratios outside the0.8–1.2 clip interval; sampled old-minus-new logprob mean57.68 (not exact KL). CPU recomputation versus stored GPU old logprob RMSE0.00392/max0.03071, retained as numerical limitation; no exact replay assertion. An initial stricter CPU/GPU assertion failed and its log is retained. These observations identify an excessive-update problem; they do not isolate reward sparsity, privileged-input scaling or optimizer settings as the sole cause.

## Costs and artifacts

Pilot281,600 scheduled simulator slots:256,000 training +25,600 initial/final diagnostic slots. Actual active18,186 (15,917 training);263,414 padded slots. Training must NOT be described as256k valid PPO samples. Completed preflight4,800 slots/540 active; first compilation failure keeps full4,800 reservation. Combined charge291,200. Failure fixed via canonical shared Warp buffer handling; physics capacity checked at each substep.

- [Pilot status](pilot/result/status.json) / [process log](pilot/process.log)
- [Analysis summary](analysis_summary.json) / [first-update diagnostic](first_update_diagnostic.json)
- [All704 episode metrics](all_episode_metrics.csv) / [training plot data](training_metrics.csv)
- [PNG](training_analysis.png) / [PDF](training_analysis.pdf) / [SVG](training_analysis.svg)
- [Pilot declaration](pilot/result/declaration.json) / [runtime versions](runtime_versions.json)
- `pilot/result/*_trajectories.npz`: complete pre-action106 inputs, raw/applied actions, log probabilities, values, post-action physical arrays, phase/terminal flags/masks and initial arrays.
- `pilot/result/*_episodes.json`, `*_reward.json`, `*_learning.npz`: complete cells, credit evidence, rewards and advantages.
- `pilot/result/checkpoints/`: initial +20 update states, optimizer, normalizer,RNG and per-pi ledger; all21 hashes verified.
- `preflight/`: preserved compilation failure; `preflight_shared_buffers/`: completed real GPU preflight.

Next work: stabilize update size and improve active-step collection efficiency before a separately budgeted retry. The requested RL architecture/reward are implemented and exercised; evidence does not support an improved final explorer.

Compact publication: pilot/preflight raw paths above are server-only; included tables, figures and summaries are self-contained review evidence.
