# Continuous RSL pulse explorer — approved 2026-09-15

User authorizes implementation and immediate detached launch. JIT only.

- [x] Add RSL-RL 3.2.0 official PPO backend with 106→128→128→128 ELU actor/critic. Actor predicts four means plus four log-standard-deviations. Latent Gaussian PPO; environment applies tanh and declared amplitude. Preserve explicit latent-entropy convention.
- [x] Keep GPU JAX collection through equivalent exported inference weights; test means/scales/values/log probabilities against Torch. Freeze explorer normalization at its own first initialization, independent of successor pi normalizers. Carry model, Adam, adaptive LR, Torch RNG and JAX RNG across rounds and promotions.
- [x] Convert only eligible pulse actions to Monte Carlo terminal-outcome targets (gamma=lambda=1). The delayed outcome return applies to each of the <=3 actions. No suffix actions, padding, unknown labels or old-policy replay in PPO updates. Use official PPO.update, clipped values, LR=.001 (user correction; adaptive ceiling=.001) adaptive desired KL=.01, epochs8, minibatch512, entropy .001, value coefficient .5, gradient norm1.
- [x] Preserve outer loop: current pi generates/reviews → train successor from old witnessed+pending support if needed → reevaluate pending → retention/nominal promotion → delayed explorer update → next current pi. Explorer NEVER resets at promotion. Per-pi novelty ledger still resets to the new pi baseline.
- [x] Add behavior tests for cross-framework likelihood equivalence, short/unknown episode masks, actual official updates and checkpoint continuation. Preserve prior backend for historical reproducibility.
- [x] Launch two independent reward arms (original_all_phases / phase_recovery), 200 rounds each,1024 envs,3 ticks, fixed .25 limit on all four channels (latest user correction), same seed. Only jumping-policy repair reward differs. Explicit full interaction caps from budget_contract, no TEST, all artifacts/costs and popup watcher. Existing historical evidence immutable.
- [x] Commit/push scoped source/tests/docs; inspect live process/status before delivery, do not wait for 400 rounds.

Validation:30 CPU tests; actual GPU/JAX logprob vsTorch maxerror4.8e-7.
Current attempt: runs/experiments/rsl_reward_comparison_20260915/amp25/.
Setup failures retained;400 actual initialization ticks charged separately.
