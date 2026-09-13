# pi_2 / pi_4 proposer control — 2026-09-09

## Why this is next

The callback-fixed smoke completed 25,600 PPO transitions, froze pi_4, acquired 753 real states and completed five evaluator panels. The independent recovery report finished successfully: 743 bank witnesses, 734 root cells, 442 pre-training seed cells and 1,176 combined cells. All 734 new-panel cells are novel relative to these two seed panels, **not necessarily to the complete historical Tube**.

pi_4 has 732 successful suffixes versus pi_2's 710 on pi_4's reached panel, with 22 pi_4-only relative to pi_2 and zero pi_2-only. But pi_4 adds zero unique suffix successes to the entire old bank on that panel. Neither result answers whether pi_4 reaches useful states that pi_2 would not discover. A same-schedule proposer control is the next experiment; do not start pi_5 or re-train pi_4 yet.

## Execution

```bash
cd /home/qy/DVGC
git switch agent/two-phase-soft-tube
git pull --ff-only origin agent/two-phase-soft-tube
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/compare_pi2_pi4_discovery.py --gpu 0
```

Default source: `JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1`.
Default output: `JIT/runs/discovery/pi2_pi4_paired_v1`.
Use `--source`/`--output-dir` only if paths differ. Keep the full original source/checkpoints/snapshots and pre-training seed directories on this server. Re-run the exact same command to resume cached completed work after process failure. Changing source code or locked inputs requires a new output directory; no source hash rewriting.

Only **pi_2 forward acquisition and its five-evaluator labels** are newly run. Verified pi_4 arrivals/labels are reused from the completed GPU work even though its original supervisor ended at a plotting error. Every label, exact arrival context, frozen checkpoint and artifact binding is still checked. No new PPO; no new pi_4 rollouts; no extra numerical-replay investigation. Both policies are evaluated by the same frozen pi_0 through pi_4 bank, so pi_2 also benefits from pi_4 as a suffix evaluator.

The profile is copied exactly from the pi_4 run: 32 trajectories; x-window endpoints 2.85/2.95 m; 0.15 m lookback; signed offsets of 0.1/0.2 on all four normalized action channels; clipping unchanged; original acquisition/label seeds; fixed complete x=2.5 near-ground start; 5 cm real-frame sampling; original physical resolution; first-valid-landing suffix endpoint. The source request's per-proposer budget is retained so bank identity stays identical. First-attempt conservative ceiling for newly running pi_2 is **8,208,000 interactions** (16,000 acquisition + 32×128×5×400 suffix). Actual early termination normally uses far less. Failed/unknown attempts remain charged; no free retries.

The preflight checks unchanged hashes of the acquisition, environment, training-reset adapter, snapshot, continuation and memory-stable rollout implementations before spending new GPU work. After pi_2 finishes it also compares exact bank identity, sampling profile, task/grid, seeds and all 32 scheduled trajectory descriptors. Report-only and CPU/GPU launcher repairs are recorded as code changes; no reset or dynamics parameter is changed.

## Reading the results

`figures/proposer_metrics.csv`: each proposer’s candidates, bank-supported root cells, novelty relative to the same 442 seed cells, exclusive observed cells relative to the other proposer, own-policy cells, complete forward landings, truncations and minimum successful control-step peak root z. Exclusive discovery is empirical evidence for complementarity, not proof that the other policy can never reach that region. Lower peak height is not a globally minimal trajectory.

`figures/matched_budget.csv`: same-cost comparisons under three accounting views:

1. Exploration alone (both sides' acquisition, all suffix labels and retries).
2. A reconstructed no-new-PPO control uses pi_2 with only pi_0–pi_3 suffix evaluators, versus pi_4 with all five evaluators plus its completed training/diagnostic cost (25,661 in this source run). This compares the introduction of pi_4 as both proposer and evaluator, not solely proposer quality.
3. The same four-versus-five evaluator reconstruction plus the earlier recorded failed smoke reservation, when its report is present (27,200 in the known history). If absent, that view is omitted, not silently charged as zero. This extra view is an explicit historical-cost sensitivity check.

For the four-evaluator reconstruction, remove pi_4's measured completed label-job interactions and its success witnesses from the pi_2 curve; retain all unallocated acquisition/retry overhead conservatively. No rollout is rerun. This prevents the no-new-PPO control from receiving a free pi_4 evaluator. The primary proposer-only comparison always uses the same five-evaluator bank.

The common budget is the smaller of the two completed cost endpoints in each view. Curves replay the original catalog order; all acquisition/engineering/retry overhead is charged upfront and a candidate only contributes after all five labels are paid. These are conservative offline budget reconstructions, not a newly run online budget-stopping experiment. The third view is not a complete lifecycle-cost estimate: earlier bootstrap and old-policy training remain excluded and must be reported separately in a paper.

`figures/coverage_cost.csv` and `paired_discovery.png/pdf/svg`: observed x-z and x-vz projections plus the exploration-cost curve. No filled convex hull or interpolation of feasible states. Per-policy suffix common-panel plots are also produced for the new pi_2 panel. `pi_2_trajectories.csv` / `pi_4_trajectories.csv` preserve individual landing/truncation/height observations.

This is one **retrospective TRAIN development control**, using a pre-existing pi_4 result. It is not an independent training repetition, a held-out success estimate, or proof of reaching a physical boundary. A matched action schedule has different measured costs when trajectories terminate differently; compare both complete-panel counts and same-cost reconstructions.

The top-level summary publishes automatically to `agent/jit-run-reports`. It separates new pi_2 interactions, reused pi_4 discovery cost, completed pi_4 training cost and known failed-smoke cost. Original source directories are unchanged.

## Decision after this control

- If pi_4 contributes useful exclusive novel discovery or improves coverage at comparable cost, proceed to a locked larger-budget iterative learning run, then independent repeats.
- If pi_2 finds comparable coverage more cheaply and pi_4 adds little complementary discovery, first revise the training-support emphasis or exploration allocation. More sequential pi names alone are not evidence of method benefit.
- If the two policies cover different regions with similar totals, retain both; do not require pi_4 to dominate or replace pi_2.

## CPU startup log repair

CPU entry points still set `JAX_PLATFORMS=cpu`, but no longer actively hide GPU devices with `CUDA_VISIBLE_DEVICES=''`. Dense CPU workers keep the selected device visible while selecting only the CPU backend. This removes the conflicting visibility setting that caused CUDA plugin discovery to print `CUDA_ERROR_NO_DEVICE` on this server. GPU workers use `cuda,cpu`, keeping GPU as default and allowing CPU callbacks. No driver/library reinstall or additional simulation is needed for already completed analysis.
