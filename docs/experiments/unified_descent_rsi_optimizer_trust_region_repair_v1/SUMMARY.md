# Unified Descent RSI optimizer trust-region repair v1

- Started from `e61639f`; bank, XML, action mapping, failure semantics, frozen π_D and its normalizer remained immutable.
- Reference LR was `1e-4`; `m0=0.005206431959`. The fixed five-point calibration tested `2.6032e-7`, `3.9048e-7`, `5.2064e-7`, `6.5080e-7`, and `7.8096e-7` using the saved first rollout. Advantage/return reconstruction error was exactly zero and no new training rollout was generated.
- All five candidates passed. Analytic KL ranged `0.001303–0.002701`, absolute sample KL `0.002122–0.004568`, fixed-observation deterministic action delta `0.00251–0.00342`, and every candidate preserved `14/1/1` survival in two deterministic repeats.
- The largest passing LR, `7.809647938e-7`, was selected without held-out or reward-based selection.
- The authorized rerun completed exactly 6,400 transitions. It attempted 17 optimizer updates, accepted 16, rolled back one, and halved LR once to `3.904823969e-7`. The normalizer hash stayed `8f2e36b...93a7e` throughout.
- In-bank survival was `14/1/1`, median 10 and lower quartile 9 at checkpoints 0, 1600, 3200, 4800 and 6400. Failure reasons stayed pitch/roll/horizon `12/1/1`.
- Held-out stayed `6/0/0`, median 8 and lower quartile 7 at every checkpoint. Core/frontier and early/middle/late results were also unchanged.
- Update integrity is now valid and the sole 16/24-tick survivor was preserved, but no physical learning signal appeared. Slightly higher shaping reward did not improve survival or failure codes.
- Do not increase to 25,600 steps. Next diagnose reward controllability, π_D initialization compatibility, and candidate horizon/curriculum difficulty. PPO authorization is false.
