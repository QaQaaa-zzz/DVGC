# Pulse memory and reward restart plan

User authorized fixing stopped JIT training, reducing novelty, heavily penalizing
physical failure during an actual pulse, and further optimizing training.

- Keep physics, success endpoints,1024 arrivals,3 pulse ticks,delta.25,neighborhood
  identity,128k repair and150-round finite limit. Restart a new TRAIN experiment
  from last accepted base; fresh explorer, no old reward/optimizer mixing.
- Reward novelty.05,success1,ordinary bounded-repair failure1,pulse physical failure5.
  Heavy penalty requires explicit physical failure plus actual pulse actions;
  success/conflict/unknown/pre-pulse failure never gets that special penalty.
- Optional evaluation_batch_size256: sequential fresh subprocesses, canonical
  restoration, unchanged per-candidate original RNG lane mapping, complete context
  and trace provenance. Parent assembles in original order; fail closed on child
  errors/identity drift; report actual active/padded interactions and peakRSS.
- Regression tests first, targeted suite, then GPU historical candidate comparison
  (unsharded subset vs sharded same subset) and full1024 previously failing workload.
  Predeclare validation budget separately; no scientific speedup claim without data.
- Freeze code/config, launch new150-round experiment, labeled desktop watcher,
  verify live stage progress and heartbeat; commit/push and link artifacts.

Progress: diagnosed kernel global OOM PID1176471 (~12GiB anonRSS), traces saved
but final results missing; failed original artifacts remain immutable.

Validation complete:94 targeted tests; new-reward72-action CPU PPO update;
GPU1024 split evaluation completed with4.981GiB peak childRSS and335872 charged
steps.24-state split differs1/24; full historical comparison differs42/1024;
explicit new numerical layout, no exact-equivalence/throughput claim.
Final audit fix records incomplete shard reservation separately on failure.
New experiment prepared with150 fresh rounds, last accepted base, fresh explorer.
