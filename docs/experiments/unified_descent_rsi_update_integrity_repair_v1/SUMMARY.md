# Unified Descent RSI update-integrity repair v1

- Started from authoritative HEAD `8512e42`; immutable bank/XML/action/failure provenance stayed unchanged.
- The 6,400-step budget was counted correctly: 4 rollout batches × 50 environments × 32 ticks. The two optimizer passes reuse data and are not counted as environment transitions.
- Frozen π_D's observation normalizer loaded correctly (`count=1,024,000`, hash `8f2e36b...93a7e`). The old Brax path changed it to count 1,025,600 before evaluating the same batch's PPO loss.
- With the rollout normalizer, recomputed old log-prob is consistent: ratio p05/median/p95 `0.999950/1.000000/1.000053`, sample KL `1.13e-6`, analytic KL `3.99e-5`.
- Normalizer-only mutation produces analytic KL `7756.77`, changes deterministic actions by as much as `0.891`, and loses the sole 16/24-tick survivor.
- Optimizer-only with a fixed normalizer still produces analytic KL `184.45` and action change `1.088`. It retains the survivor in this reproduction, but the update is not a valid trust-region step.
- The existing desired-KL guard (`0.01`) rejects all four candidate gradient steps; their post-update KL values are `140.89–192.35`. Optimizer and normalizer rollback is exact and preserves baseline `14/1/1` survival.
- Reset draws are correctly hierarchical: core/frontier `4530/1870`; frontier early/middle/late `661/616/593`. Transition occupancy differs (`996/604`) because episode lengths differ; gradient sampling reuses those 1,600 transitions twice.
- Phase A therefore fails only because no effective optimizer update is possible with the fixed authorized hyperparameters. Phase B was not run, PPO authorization remains false, and 25,600 steps are not recommended.
- A meaningful next run needs separate authorization for the optimizer/trust-region protocol. This task explicitly prohibited learning-rate, network, clip, or reward changes, so no such search was performed.
