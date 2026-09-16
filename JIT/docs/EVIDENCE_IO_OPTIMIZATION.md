# Evidence IO optimization — 2026-09-16

Scope: continue the approved two-arm RSL pulse campaign from 94 completed rounds;
1024 environments, residual amplitude .25, LR .001, 200 rounds per arm unchanged.
Saved round95 collection and explorer optimizer state are reused. Old run files and
code snapshots remain immutable; new source locks and relocation receipt bind the restart.

Changes:
- Hash completed prefix/behavior/suffix files once before exporting their lane records.
  Every lane still records the same SHA256; simulation, labels, rewards and costs are unchanged.
- During a single bank-lock operation, share file SHA256 results across overlapping
  historical dependencies. Each request checks device/inode/size/mtime/ctime; changed
  identities are rehashed. Files modified within one second are always rehashed to
  avoid timestamp-resolution collisions. Cache is cleared at operation exit and no
  persistent verification receipt bypass is introduced. This assumes ordinary local
  filesystem metadata semantics and no privileged concurrent metadata forgery.
- Export suffix timings for runtime initialization, snapshot restoration,
  compilation+rollout, and trace/result export. This does not yet make evaluator
  processes persistent or batch the host snapshot restoration.

Validation: 27 relevant CPU tests passed, including cache lifetime, same-size rewrite
with restored mtime, replacement, bank and pulse contracts. Production-bank identity
comparison and measured timing are retained in
`runs/experiments/rsl_reward_comparison_20260915/efficiency_resume_20260916/`.
No end-to-end speedup claim until completed production rounds are measured.
