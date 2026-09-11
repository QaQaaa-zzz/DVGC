> Completed in production, reviewed 2026-09-11: both rounds and pi_5/pi_6 training finished. The command below is historical, not the current next run. See [current handoff](CODEX_HANDOFF_20260911.md) for results and next implementation priorities.

# All-proposer autonomous campaign

The pi_2/pi_4 retrospective control completed: exploration-only matched budget 69,078 yielded 513 versus 685 novel seed-relative root cells. With completed training charged, pi_4 has not repaid its cost at the smaller matched budget. Lowest successful control-step peak root height was 0.606813 m versus 0.516082 m. These are observations, not global limits or independent replications.

## Production command

From `/home/qy/DVGC`, on `agent/two-phase-soft-tube` after `git pull --ff-only`:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/run_envelope_campaign.py --all-proposers --gpu 0 \
  --output-dir JIT/runs/campaign/all_proposers_v1 \
  --max-rounds 2 --ppo-steps 128000 --budget 40000000 \
  --patience 2 --min-gain 20
```

This DOES run new PPO: at most two successful 128,000-step trainings, creating pi_5 and pi_6. Failed attempts consume their reserved cost; retries remain bounded by the total new-interaction budget. 40M is a conservative ceiling, not expected work. All GPU acquisition, labels and training run sequentially in fresh processes. The old campaign entry remains available without `--all-proposers`; do not run it to restart the already completed pi_4.

Required server evidence: completed `pi2_pi4_paired_v1`, original `empirical_envelope_smoke_callbackfix_v1`, original knee and lower-boundary seed runs, all frozen manifests and snapshots. Source identities, labels, training completion and rollout implementation are checked before new computation. The inherited pi_4 plotting failure is accepted only via the existing narrowly scoped recovery validator. The completed paired report is required.

## Round sequence

1. Load original seed support plus verified pi_2/pi_4 arrivals. Preserve inherited policies and cost separately; known inherited charge is 195,551, excluding earlier shared bootstrap/seed costs.
2. Round 0 reuses the completed pi_2/pi_4 schedule, adds pi_0/pi_1/pi_3 with the same numeric perturbation schedule and five frozen evaluators. It does not retroactively change historical bank hashes or evaluator panels.
3. Save `round_000/before_training/figures`: current per-proposer images and data. Construct TRAIN-only witnessed reset support from all accumulated sources, excluding no-witness states. Use existing bounded phase/trajectory-balanced support; multiple current proposer sources receive the recent-source weight.
4. Train pi_5 from pi_4: Actor/normalizer warm start, fresh critic/optimizer; existing 20% fixed-start/80% witnessed snapshot reset recipe; unchanged first-valid-landing endpoint.
5. Add frozen pi_5, explore it with the six-member evaluator bank, export completed-round evidence. Update deduplicated cumulative cells and cost.
6. Round 1 gives pi_0 through pi_5 another declared schedule, trains pi_6 from pi_5, and explores pi_6 with seven evaluators.

Each proposer receives 32 legal signed single-action perturbation trajectories per call. This first version guarantees equal opportunity; it does NOT implement adaptive gain-based budget allocation. Action channels remain steer, drive, hip, knee. No injected state height or velocities. Later round profiles vary declared windows/strengths/seeds using the existing profile function.

Round gain is the union of ALL newly acquired witnessed cells relative to the complete round-start union, not a sum of overlapping per-proposer counts. Stop on budget, round cap, or two consecutive gains below 20. This is empirical search stagnation, never physical-boundary proof. Do not automatically run forever.

## Evidence for paper figures

Every completed round saves `round_NNN/figures/` and a separate `before_training/figures/`:

- `pi_N_envelope.png/pdf/svg`: each proposer's cumulative reached-state x-z and x-vz projections, with common axes; gray means no bank witness, not physically impossible.
- `all_proposers.png/pdf/svg`: witnessed arrivals by proposer, no filled hull or interpolated reachable region.
- `pi_N_points.csv` and `all_points.csv`: exact-state/context hashes, source, trajectory, phase, physical cell, all projected coordinates and evaluator outcomes. Blank evaluator entries are UNTESTED, not zero. These tables allow later plots without simulation.
- `x_slices_005m.csv`: phase-separated observed minima/maxima for all coordinates in 5 cm x bins, with counts. Missing bins remain missing; extrema at different x need not form one trajectory.
- `proposer_metrics.csv`: observed and exclusive cells, new cells relative to round start.
- `trajectory_receipts.json`: complete forward rollout outcomes, peak height, perturbation and cost records.
- `figure_manifest.json`: source plans, bank identities, resolution and interpretation.

Each underlying discovery child additionally retains its existing per-evaluator common-panel plots and data. Distinguish those evaluator abilities from cumulative proposer-conditioned figures: different proposers and bank versions have different panels. Root physical resolution remains 0.1m/0.1m/s/0.5deg/2deg/s; 5cm is the display/acquisition slice spacing, not a changed physical-cell metric.

Root `figures/campaign_progress.*` and `coverage_cost.csv` summarize rounds. Compact CSV/log reports auto-publish to `agent/jit-run-reports`; all PNG/PDF/SVG and authoritative snapshots stay on the server. Raw source runs are never deleted or rewritten. Resume the identical command after a process error; changed source/config requires a new directory. Do not restart completed evidence under the old commands after updating code.

## Scientific status

This is a development iteration, not a controlled proof that training is cheaper than frozen-bank exploration. Before paper submission: a fixed-bank no-new-training matched-cost control; independent training/search repetitions; cumulative union against all included historical sources; resolution sensitivity and geometric slice width; complete cost including shared bootstrap; locked final evaluation; bootstrap ablations if the paper claims their benefit. No final TEST is opened by this command. Current accepted numerical replay limits remain documented; no new replay gate is introduced.

CPU tests exercise orchestration, reuse, policy growth, cost accounting, failure stops and replot exports. Actual multi-proposer training rounds still require the server run above.
