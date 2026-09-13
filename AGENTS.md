# DVGC / JIT agent authority — 2026-09-12

## Read first and current action

Read [PROJECT](PROJECT.md), [latest Codex handoff](JIT/docs/CODEX_HANDOFF_20260911.md), [current status](JIT/docs/CURRENT_STATUS.md), then the remote evidence and relevant source. [JIT instructions](JIT/AGENTS.md) also apply.

`all_proposers_v1` completed two production rounds, pi_5/pi_6 each 128,000 PPO transitions. Status `round_limit_reached`; 1,689 inherited root cells → 4,629 → 9,296. New interactions 1,275,465, including 1,004,034 continuation-label steps. Do NOT rerun the completed campaign or directly launch pi_7 because an old document says “next run.”

Current continuation: first-success labeling, conflict quarantine, vectorized continuation engineering, bounded checkpoint comparisons, actor-scale residual PPO, per-policy arrival novelty, pending reset support and delayed reevaluation are implemented. The two-round delayed pilot completed both 128k policy trainings; prior no-witness/unknown to witness transitions are zero in both arms. Latest completion: `JIT/runs/experiments/delayed_completion_start_20260912/`; preserve the earlier reuse gate-timeout error. Read CURRENT_STATUS before interpreting old queue files.

Next work: analyze existing data for novelty saturation, candidate caps, active/padded cost, pending-phase distribution and resolution sensitivity, then predeclare a bounded fair comparison. Do not keep sweeping PPO length: 128k–256k is a provisional working budget, not an optimality claim. Frozen-critic quality penalties, diffusion exploration, automatic training extension and adaptive proposer allocation remain UNIMPLEMENTED. Documentation/paper work does not launch experiments.

## Research contract

- Objective: empirical jumping envelope discovery by complementary frozen policies at controlled cost, not one Actor mastering/replacing the whole Tube.
- Fixed complete near-ground start x=2.5m; ~3.1cm initial wheel clearance accepted. Physics: `assets/orange_bike_4kg_horizontal.xml`, payload2kg, simulation0.005s/control0.020s; do not silently change physics/reward/start/endpoint.
- Actions `[steer,rear-wheel drive,hip,knee]`; all four legal channels may be perturbed. No invented qpos/qvel/height injection or unproven reachability by lowering a state.
- Keep original real-frame pi_0 centerline. 5cm sampling/display slices are distinct from root physical quantization (0.1m,0.1m/s,0.5deg,2deg/s plus phase).
- Admission requires real forward prefix provenance plus successful first-valid-landing continuation from the SAME complete state/context. Position/velocity equality alone is insufficient; preserve history/FIFO/events/RNG/time semantics.
- Existing numerical replay differences were explicitly accepted; no extra replay gate without new user request. Do not claim exact replay passed.
- Proposer arrival, evaluator continuation, training reset support, and projected physical-cell coverage are different objects. Success under a prefix/suffix pair is not necessarily single-Actor execution.
- No bank success means no witness under that bank, not physical impossibility. Untested/error/incomplete is not a negative label. A physical cell is not a certified continuous feasible region.
- Explorer reward uses the current frozen pi's own arrival ledger, not the global witnessed union. Pending arrivals may enter separately declared training support without becoming envelope witnesses. Delayed feedback updates candidate status, reset support and the next source pi; it is not backpropagation across the outer loop or stale PPO replay.
- Current explorer is frozen 76D base Actor plus a 106D privileged-history residual Actor (256x3 Swish, four bounded actions) and its own Critic. Base critic is telemetry only. Current hip/knee mapping is keyframe-centered absolute position. Do not mix other projects' action interfaces, observations or results into JIT.
- Finite budget/stagnation is not a physical-boundary proof. Peak root height is not wheel clearance or per-slice lower envelope. Never fill gaps/hulls as reachability evidence.

## Evidence, data and paper

- Keep historical runs immutable; new derived audit/results go to new directories. Verify complete identities before cache reuse/merging. Do not remove locks to resume across changed code or protocols.
- Four historical all-proposer forward receipts have both valid_landing and physical_failure; their substep ambiguity remains. New existence-label views quarantine conflicts. The derived pi6 view has735 witnessed/3 unknown versus the old738-positive union; do not silently relabel the original data or claim historical substep adjudication.
- TRAIN may guide exploration/training. CALIBRATION is for optional predictors. Used ACCEPTANCE is development evidence. Final TEST/JCE/JEL remains unopened. Ancestor/trajectory correlation matters.
- Preserve Tube0 history: 222 weighted training rows include42 negatives. Current pi_0 is later Round1. pi_3 old mixed-endpoint gate is invalid, but its frozen policy has valid new-protocol uses.
- Count acquisition, suffixes, PPO, evaluations, retries and bootstrap separately and jointly. Current inherited195,551 + new1,275,465 =1,471,016 does NOT include all historical bootstrap costs.
- Report cumulative novelty relative to a named deduplicated baseline; do not sum incompatible old totals. New9,296 is campaign-scoped. Do not attribute all multi-policy gains to newest PPO.
- Paper needs fixed-bank matched-total-cost control, independent repeats, geometric/resolution evidence, complete costs, locked final evaluation and bootstrap ablation if claimed. Diffusion is not needed merely to make the method sound novel.
- Latest learned/random comparison shares a successor pi trained only from learned-arm pending support; it is a conditional exploration control, not independent end-to-end matched-cost evidence. Candidate counts differ from root-cell counts. Preserve zero/negative results and avoid claiming learned residual superiority or complete-space convergence.

## Environment and repository work

- Code branch `agent/two-phase-soft-tube`; reports `agent/jit-run-reports`, repo `QaQaaa-zzz/DVGC`.
- Server `/home/qy/DVGC`; Python `/home/qy/mujoco_playground/.venv/bin/python`, `PYTHONPATH=JIT/src`; RTX4090D24GiB, Warp1.11.0 production logs.
- GPU children use `JAX_PLATFORMS=cuda,cpu`; preserve CPU callbacks. CPU analysis does not actively hide CUDA devices. No blanket reinstall for nonfatal startup logs.
- Raw snapshots/checkpoints/full figures stay server-side. Compact reports auto-publish. Read report branch when user reports completion; request ZIP only if publication unavailable.
- User authorized routine project edits/commits/pushes and report publication. Do not repeatedly ask. Do not launch unrequested large experiments from this workspace.
- Preserve unrelated edits; no reset/clean/stash/rebase/force-push. Durable logic in JIT/src/jit_dvgc, thin CLIs, meaningful tests, docs in JIT/docs. Distinguish CPU tests from production GPU validation.
- Old instructions are archived under JIT/docs/history/handoff_before_20260911. They are historical, not current action authority.
