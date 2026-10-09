# 4090D pipeline performance implementation plan

Goal: reduce closed-loop wall time without changing candidate count, DDIM20/H16,
independent full-layout replay, 128000 student transitions, 2000 G updates,
physics, rewards, data roles or last-valid full learner inheritance.
Baseline: 2e296f282e28df670de0dd72155ccb3eb1c06474. The current production run
uses an immutable separate snapshot and is not modified or stopped by this work.

- [x] Bind R31 plan/production/timing receipts to actual frozen implementation;
  calculate the child interval union and external gaps, keeping unknown causes explicit.
- [x] Add optional structured performance events and read-only audit CLI. Fields:
  round/root/stage/pid/backend/batch_size/start/end/elapsed/input/output/hash bytes,
  cache hit, GPU memory when measured, status. Missing measurements stay null.
- [x] Cache verified trace arrays with bounded immutable storage and mutation rejection;
  reuse verified historical corpus blobs without rewriting histories.
- [x] Build immutable corpus sampling index once. Compare original group->ancestor->
  trajectory->window distribution and exact fixed-seed RNG/output/provenance.
- [x] Reduce numerical guards on device, reject nonfinite proposals without donating
  old state. Compact provenance must losslessly reconstruct sample identity and window.
- [x] Opt-in B1 teacher worker: one gated GPU process, pinned Actor/G, compiled DDIM,
  reusable physics kernel, fresh full-context restore for every search and replay.
  Supervisor remains CPU; preserve 17-world layout and logical RNG keys.
- [x] CPU regressions and bounded isolated benchmarks: fixed TRAIN roots/candidates,
  A->B->A contamination check; fixed complete G state/corpus, 100 updates per variant.
  Diagnostic artifacts never feed E/G/student training. Do not replace checkpoints.
- [ ] Only after B1 physical correctness and stable measured end-to-end improvement,
  consider B4/B8 benchmark; reject adoption on unresolved layout/RNG differences.
- [ ] Independent review, correctness report, exact commands, commit/push, shared ledger.

Execution boundaries: no package upgrades, unrelated process stops, current-run hot
patches, blanket cache deletion, physics parameter changes, or new long campaign.
GPU short benchmarks require the existing shared resource gate and finite declared
costs; lack of resources is recorded instead of preempting another experiment.
Production adoption needs successful bounded 1–2-round integration; the original
200-round run continues on its frozen implementation until a valid boundary handoff.

Rollback: keep the baseline snapshot and disable opt-in persistent teacher execution.
New provenance readers accept historical reports; corpus manifests retain old inputs.
No old artifact or success standard is rewritten.

Validation status: cache alias invalidation also covers multiple hardlink paths.
B1 implementation is present but physical acceptance failed; it remains disabled.
B4/B8 are deferred. Full-round integration and production adoption remain pending.

Follow-up: six-request physical repeatability completed; labels match on this
repeat but old failure retained. User authorized an isolated experimental B1
trial; scoped receipt cannot extend to next round. Formal acceptance remains unproved.

- [x] One explicitly authorized B1 integration completed, complete bundle verified.
- [x] Independent review, CPU regressions and remote code/report delivery.
- [ ] Formal physical equivalence and learning benefit (not claimed).
Original200 remains paused; no second trial or long campaign auto-launch.
