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
