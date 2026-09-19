# Pulse reward and bounded-memory restart (2026-09-16)

The user authorized repair/restart and further training optimization after the
kernel killed the1024-candidate evaluator (PID1176471) with global host OOM.
The old run completed11 rounds. Its original status, checkpoints and failed
round12 remain immutable. The new experiment starts a fresh150-round TRAIN run
from accepted base lineage_repair_0010 and freshly initialized explorer/optimizer.

## Reward contract

Novelty maximum changes.25→.05. Success remains+1; bounded repair still failing
remains−1. Explicit physical failure terminating the prefix after at least one
actual pulse action receives−5 quality (up to−4.95 including novelty). Successful
termination is positive. Conflicting/unknown labels remain excluded. A failure
before the explorer applied an action cannot receive the special−5 penalty.
New candidates record prefix_physical_failure separately from terminal status;
legacy missing telemetry does not imply physical failure. Configuration without
pulse_failure preserves historical reward behavior. This is a changed reward
experiment, not a continuation of the old optimizer under its old objective.

## Execution optimization

Keep1024 arrivals, mixed six onsets,3 pulse steps,delta.25 and the existing
neighborhood encoder/128×3 heads. Set evaluation_batch_size=256 to evaluate
large singleton-policy panels sequentially in fresh child processes. Each exits
before the next begins, releasing allocator/compiler memory. All candidate
snapshots, original order, policy identity and original nonterminal RNG lane
indices are preserved. The sampler remains1024-wide. Retention panels and
nominal support sets below256 use the original direct evaluator.

Each child writes trace/results, active and padding costs, timings and host
peakRSS. The parent checks candidate identity/order and cost consistency before
publishing the combined results; failure stops the stage. SIGTERM propagates to
the owned child through the existing runner. Individual restored states are
released once the stacked world owns its buffers, before rollout compilation.
The optional fused restore remains disabled. Physics and endpoint tests do not
change, but changing batch shape can alter numerical trajectories and outcomes.
The new batch setting is explicit in frozen configuration, not claimed to be an
exact replay or an interchangeable same-cost control.

## Validation and evidence

Targeted CPU tests cover reward scope, unknowns, partition/RNG mapping, merged
order and costs, drift/failure rejection and named notifications. A saved72-action
RSL update under the new reward completed6 optimizer updates with finite metrics;
post-updateKL.03387 triggered the existing epoch-stop rule. This CPU update adds
zero simulation interactions and is not a learned-policy efficacy result.

GPU24-state diagnostic: direct20/24, split8-per-process19/24; one candidate label
differs. A repeated direct24 run retains20/24 labels but has different active
step counts. This does not establish that the split-label difference is solely
run-to-run noise; retain it as a batch-layout sensitivity limitation. There is no
exact-label-equivalence claim. The full1024-state memory reproduction completed:4 shards,826 successful labels,
peak childRSS4.981GiB,339.29s,335872 charged interactions versus409600 from the
old saved trace (18% fewer allocated steps). The old trace has832 positives;
42/1024 binary labels differ, despite preserved context identities. Do not claim
exact numerical equivalence or attribute changes solely to batching: the original
simulator already has replay variation, and these are separate executions.
Wall-clock speedup is NOT demonstrated. The change bounds per-process memory
and is explicitly declared as a new numerical execution layout for this new run.
94 targeted CPU tests pass. Total new validation physics359600 steps
(9600 direct +4528 split small +9600 repeat +335872 full). The final source change
after GPU validation only improves failed-shard reservation telemetry; regression
tests cover that path, and successful evaluation arithmetic is unchanged. All validation is TRAIN engineering work, not final TEST.

Evidence root: runs/monitoring/pulse_memory_reward_20260916/ (the initial report
script error and subsequent24-state mismatch remain recorded; completed physics
was reused, not silently discarded). New experiment:
runs/experiments/neighborhood_reward005_safe256_20260916/.
Maximum new reservation172881600; actual costs are recorded separately from the
old run and validation. No automatic extension beyond150 rounds.

Watcher manifests now accept a name appended to popup titles, distinguishing
engineering validation from the formal150-round run. Live state is authoritative.

Failed-shard status distinguishes completed measured costs from the incomplete
shard reservation; the outer runner conservatively reserves the whole failed
stage. Neither figure is silently presented as fully measured simulation cost.
