# Frozen pi + learned residual + suffix novelty: engineering validation

Corrected user scheme: freeze original pi6 actor/normalizer;106-dimensional privileged input (existing3-frame history),256×3 Swish stochastic tanh-normal residual actor with zero-initialized mean head and fresh value/optimizer. Output4 residual channels, configured±0.15 here, composed as clip(frozen_pi(observation)+delta,-1,1). Pi is evaluated on the currently reached state, not a pre-recorded nominal action sequence. No supervised random-perturbation imitation and no replacement of frozen pi.

New-state reward is relative to the declared own-pi baseline/session ledger (not a multi-policy union). At each candidate-generating action tick, credit only a newly reached physical root cell with a validated complete-context frozen-bank suffix witness. Parent perturbed-trajectory landing is not required. Unknown stops the update without becoming negative; repeated new-cell credit is shared. Snapshot identity binds the composite controller and residual checkpoint; candidate receipt binds exact prefix arrays and suffix state/context/bank/result hashes. Suffix runs frozen evaluators without residual actions, using existing fresh-continuation semantics and first-valid-landing endpoint. Numerical replay limitations remain accepted; no exact replay guarantee.

## Real short test

184 CPU tests passed in7.89s. GPU2-world full400-tick initial/training/final collections plus serial suffix evaluation completed31.067s; one PPO update. Initial deterministic residual0 exactly reproduces recorded frozen base actions; baseactor hash and normalizer unchanged. Both training candidates witnessed by pi6 (2 new reward cells); final candidate comparison yields1 additional unique cell. Initial/training/final2/2 direct landings are engineering diagnostics only; this tiny unreplicated test cannot establish learned exploration improvement. The plotted training rollout occurred before its sole update.

Cost2,564 total:2,400 forward scheduled slots (263 active,2,137 padding)+164 actual suffix steps; engineering declaration ceiling13,600. Training used83 active forward steps. No long training or final TEST launched. Prior full-action pilot is off-target evidence, not this residual scheme's result.

## Complete evidence

- [Runtime status](attempt/result/status.json), [log](attempt/process.log), [declaration](attempt/result/declaration.json)
- [Audit summary](analysis_summary.json):263 action rows and4 candidate receipts rechecked
- [All action plot data](all_action_plot_data.csv), [all candidates](all_candidates.csv)
- [Action PNG](action_chain.png), [PDF](action_chain.pdf), [SVG](action_chain.svg)
- `attempt/result/*_trajectories.npz`: all forward physical/controller/snapshot arrays, actions/base/requested/effective residuals, masks, logprob/value, full history and event context.
- `attempt/result/suffixes/`: canonical snapshots, every evaluated suffix trajectory, identity and charge receipts.
- `attempt/result/*_candidates.json`, `*_reward.json`, `*_learning.npz`: causal action-tick credit and PPO data.
- `attempt/result/checkpoints/`: immutable initial and updated residual/value/optimizer/normalizer/RNG and per-pi ledger.
- `analyze.py`: offline audit and standalone plot regeneration; no simulation.

Next: a separately predeclared residual exploration pilot with fixed-random-perturbation matched-cost control; direct landing remains a diagnostic, not the reward gate. No claim of full-space convergence or certified boundary.

Compact report: attempt/ raw trajectories and checkpoints referenced above remain server-side; all CSV and figures linked at report root are included.
