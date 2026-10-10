# Three frozen policies: multidimensional perturbation envelope evaluation

User approved 1000 episodes/model, medium reset noise and one frozen explorer shared across all three. Models: repair10 initial, learned100 final0098, random100 final0099. No training. Four onsets0/5/10/15,250 each,3ticks,delta .25. Add three nominal reference rollouts. Nominal references use 250 matching-capacity replicas per model (one unique condition each), excluded from scored denominator. Formal maximum1,500,000 control transitions; separate engineering preflight at most4800. Existing training protected.

Reset bank: shared deterministic NumPy seed10102026, world x±.1m,y±.05m; Euler roll/pitch/yaw±3deg composed with nominal quaternion; vx/vy±.2m/s, body angular components±.1rad/s. No extra vertical velocity/joint noise. Root z retains nominal unless geometric clearance requires upward correction; record every correction. Validate finite normalized quaternions and CPU forward geometry before simulation. Rebuild observations/history/events from initialized state, never patch qpos after observing/resetting.

Frozen explorer: old learned100 last update0099, shared normalizer and fixed round0099 reference neighborhood map. Reference labels remain associated with their original Actor, never relabeled as target-policy capabilities. Actions adaptive to each policy's actual trajectory: same reset/latent RNG, not identical closed-loop pulse values. Save actual requested/effective pulse actions.

Implementation tasks:
1. Add explicit validated frozen-evaluation permission for full-horizon learned collector; preserve old default rejection and training behavior.
2. Optional complete initial qpos/qvel overrides in unified reset and declared input bank in collection; save pre-step initial arrays and use separate explicit evaluation seed. Add CPU contract tests before changes.
3. Allow declared frozen reference atlas only in frozen evaluation; retain legacy source-identity checks otherwise.
4. Validate tiny GPU run, then sequential 250-world batches with free memory>=20GiB, no preallocation. Nominal references use same250 capacity with no pulses; retain lane0 and disclose auxiliary duplicates in actual budget if needed (must amend before launch).
5. Report all3000 episodes, paired successes/failures, physical XYZ/XY/XZ/YZ,12D traces and90%bands on all/success sets, nominal-relative windows only where both traces exist. Report temporal coverage explicitly; no extrapolation after termination.
6. Common-unit12D cell coverage at fixed bins (.05m positions,3deg attitudes,.2m/s velocities,.2rad/s angular velocities), phase/time-stratified sensitivity at half/double scale; empirical occupancy not reachable volume. Per-axis widths and normalized deviation radii accompany success rate; no multiplication of marginal widths or inflated hull.
7. Update shared ledger under lock, index rawNPZ/PNG/results. No changing current training, no expansion beyond declared evaluation.
