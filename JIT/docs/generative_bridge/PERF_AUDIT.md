# 4090D closed-loop performance audit

Actual R31 source: `2e296f282e28df670de0dd72155ccb3eb1c06474`,
`/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/code_2e296f2`.
Evidence: `series_0008_original_reset_174/round_0031/production.json`,
`started.json`, `status.json`, `execution/*/status.json`, rollout timing receipts.

Round wall: 3997.1723 s. Union of completed child wall intervals: 2715.4469 s.
Outside child intervals: 1281.7253 s (21.3621 min). 834.6162 s of those gaps
precede teacher search/replay launches; other gaps mostly 18–26 s each. These
are interval locations, not retrospective attribution to particular functions.
Historical timestamps cannot distinguish CPU DDIM, lock hashing and reporting
inside a gap. New structured events explicitly measure these operations.

Source lock set: 14981 files, 9684819782 bytes. A fresh read-only hash audit took
9.2210 s on this host. A later full first cache verification took 4.7363 s;
subsequent verification of exactly the same set took 0.0773 s, without re-reading
payloads. Cold filesystem state and concurrent load differ between measurements;
no claim of exact attribution of all historical 21.4 minutes follows.

Installed versions: JAX/jaxlib0.6.2, MuJoCo/MJX3.6.0, Warp1.11.0,
Brax0.14.2, Flax0.11.2, Optax0.2.6. Logs show MJX-Warp and existing Warp cached
kernels in `/home/qy/.cache/warp/1.11.0`. Supervisor environment explicitly uses
JAX_PLATFORMS=cpu and does not explicitly set JAX_COMPILATION_CACHE_DIR.
This does not prove all JAX caching is disabled. No packages/drivers changed.

Implemented optimization boundaries:
- Indexed source/ancestor/trajectory/window sampling retains RNG call order.
- Verified trace arrays and file integrity checks have bounded caches, mutation
  notifications, stat identity checks and conservative unsupported-platform fallback.
- Incremental G guards reduce on device; no donation or successful bad proposal.
- Compact provenance retains sample trajectory IDs, windows, seed and update count.
- Existing corpus blobs are referenced rather than compressed again each round.
- Persistent B1 teacher execution is opt-in; production default is unchanged.

CPU synthetic sampling speedups (~19x) concern sampling only. Lock cache speedup
concerns repeated validation only. Neither proves whole-round acceleration,
physical replay equivalence, improved G recovery, or better learning outcomes.
Bounded GPU results and remaining adoption gates are recorded with benchmark artifacts.

## Bounded results, 2026-10-09

Actual 4482-trace corpus, 4096 fixed-seed draws: legacy sampling 2.44738 s,
indexed 0.032908 s (74.37x sampling only), index build 0.37417 s. Observations,
actions, sample provenance and RNG match exactly.

The complete G worker benchmark includes loading, 100 updates, sample logging
and checkpoint publication: 62.93864 -> 43.53951 s (1.446x, 30.82% less wall).
All 25600 sample records match. RNG, normalizer and counter match exactly;
params/EMA/Adam pass the predeclared calibrated FP32 rtol=1e-5, atol=2e-6.
The initial tighter tolerance failed; repeating the original implementation also
failed that tolerance (Adam max difference 1.124e-6). The calibrated acceptance
was frozen before a separate new-seed GPU test. Earlier failures are retained.
Standalone warm updates improve about 4.13x, but their complete process time
is 20.7378 -> 20.6471 s: startup dominates. No whole-round speedup is established.

Persistent B1 A->B->A FAILED acceptance. Baseline repeated A success IDs were
[1,7,13]; persistent repeated A changed from [1,7,13] to [1,5,7,13]. Initial
observations/actions match, but physical trajectories differ. Even baseline
repetition is not bitwise identical. Scratch contamination versus intrinsic
backend nondeterminism remains UNKNOWN. 60.13162 -> 31.97847 s is NOT an
accepted speedup: charged physical work differs (3757 vs 7089). Production
requires a hash-bound passed acceptance receipt; B1 is blocked. B4/B8 were not run.

The original 200-round campaign was paused by user request at R33; last complete
R32 corresponds to 28/200 campaign rounds. Its frozen code and models are unchanged.
An independent one-round integration uses the original teacher execution path,
PPO128000 and G2000 from a complete R30 bundle. It is not a continuation of the
paused campaign and cannot automatically adopt or extend to 200 rounds.

Raw evidence root: `/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008`.
Subdirectories: `performance_generator_20261009`,
`performance_generator_baseline_repeat_20261009`,
`performance_generator_calibrated_20261009`,
`performance_generator_pipeline_20261009`, `performance_benchmark_20261009_v3`.
Original failed attempts remain beside these directories.

## Reproduction and rollback

From this repository, use the existing bounded plan without overwriting artifacts:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python JIT/cli/benchmark_bridge_performance.py --mode prepare --input /home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/performance_benchmark_20261009_v3/plan.json --output /tmp/jit_perf_reproduction
JAX_PLATFORMS=cpu PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python JIT/cli/run_gated_plan.py --plan /tmp/jit_perf_reproduction/plan.json --output-dir /tmp/jit_perf_reproduction/execution
```

The cloned plan retains original frozen source identities, bounds and resource
gates. Choose a fresh output path. This reproduces the diagnostic, not production
adoption. Rollback uses the original `code_2e296f2` snapshot and original profile;
no historical model/result or success criterion was overwritten.

Implementation and CPU correctness do not prove physical equivalence or improved
G recovery. Full-round timing, stable repeated end-to-end gain and learning
effectiveness remain unverified. Unmeasured performance event fields remain null.

## Follow-up repeatability and explicitly authorized trial

User questioned whether original physics is itself variable and explicitly favored
trying B1. Six requests A/B/A/A/B/A per arm were executed with frozen inputs:
all six cross-arm success-ID sets match and each arm's repeated labels are stable.
A succeeds at IDs1/7/13, B has no successes. Both arms still show first-step
physical differences despite exact initial observations/actions. Baseline first-step
qvel repeat differences reach0.006; persistent reaches0.00572. This weakens an
assertion that the previous mismatch alone proves worker contamination; it does
not disprove intermittent failures or prove state equivalence. Previous failed
ABA remains part of the evidence.

Measured wall112.92269 vs38.87275seconds, charged physics12954 vs7582, so this
is not an equal-work throughput comparison. Evidence: performance_repeatability_20261009.
A separate user-authorized experimental trial receipt is now supported, bound
to one exact output directory, round, implementation commit and <=1.5M physics.
It explicitly records formal_acceptance=false and unresolved limitations. It
cannot authorize a subsequent round or automatically restart the original200.
The production physical-acceptance path remains distinct; no passing receipt
is fabricated from this user authorization. Full search/replay, thresholds,
replay rejection, RNG mapping and full learner inheritance remain unchanged.
