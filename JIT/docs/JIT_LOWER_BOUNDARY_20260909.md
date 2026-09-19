# Observed lower successful boundary

Production support validation completed: 195 witnessed states, 95 unwitnessed
development targets, initializer pi_2. The user requested additional downward
exploration before training. A lower coordinate does not by itself establish
reachability or a successful platform landing. This experiment uses real dynamics
from the fixed x=2.5 m start; no height/velocity injection or external lift.

```bash
cd /home/qy/DVGC &&
git switch agent/two-phase-soft-tube &&
git pull --ff-only origin agent/two-phase-soft-tube &&
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/explore_lower_boundary.py --gpu 0
```

Requires completed `JIT/runs/training_support/complementary_boundary_v1` and
the existing shared-panel baseline and frozen checkpoints. Default output is
`JIT/runs/discovery/lower_boundary_v1`. Return the printed compact ZIP, even on
failure. No PPO. Serial evaluation, fresh bounded processes, no device benchmark.

The locked first pass uses pi_2, two window endpoints (2.85/2.95 m), two offset
magnitudes (0.1/0.2), and hip/knee individually with both signs: 16 trajectories.
Windows begin 0.15 m earlier; offsets are added to normalized Actor outputs and
clipped to [-1,1]. Neither sign is assumed to lower the trajectory; rank actual
outcomes. This is a local search family, not a global hip/knee-only restriction.
Four frozen evaluators provide suffix labels, with first valid landing unchanged.

Keep 5 cm real-frame sampling, at most 128 states per trajectory, 400 ticks per
rollout and an x=8 m sampling guard. Conservative acquisition reservation is
8,000 interactions, matching 10 family anchors times two possible variants times
400. There are 16 actual variants across both endpoints. First-attempt ceiling
3,284,800 interactions; total budget 3,500,000 includes failed/retried work.
Physical cell resolutions are unchanged. Lower plots are projection summaries.

Outputs distinguish:

- `observed_lower_slices.csv`: minimum witnessed root z in each phase/5 cm slice,
  with actual x and exact candidate/trajectory identity. Unwitnessed lower points
  are excluded. No interpolated states or lines filling missing coverage.
- `trajectory_outcomes.csv`: forward landing/failure/truncation, action direction,
  and maximum root z observed at every control step, including reset. This is
  root height, not center-of-mass height, wheel clearance or a continuous-time peak.
- `lowest_peak_successful_trajectory.csv`: retained points on the one complete
  successful forward trajectory with the lowest observed peak in this batch.
  Empty success sets produce a null best trajectory, never a fabricated solution.
- `lower_boundary.png`: ascent/descent point clouds and observed slice minima.

Different slice minima can belong to different paths and different successful
suffix policies. They cannot be stitched into an executable minimal trajectory.
The lowest successful full forward path is a separate output. This first pass
does not prove the global optimum, the exact platform-clearance limit, or that
the lower boundary improved over old pi_2 trajectories. Use measured successes
and neighboring failures to choose further local refinement only if needed.

The source support self-hash and TRAIN role are verified; the pi_2 frozen member
must equal its initializer identity. The original successful support remains
unchanged. Earlier support preparation is not a runnable PPO recipe.

## Automatic result publication

The user authorized pushing necessary results. Standard top-level runs under
`JIT/runs` now publish compact JSON/CSV/Markdown/log evidence to
`agent/jit-run-reports`, in `reports/<run-name>-<path-hash>/`. Raw snapshots,
checkpoints and images stay on the server; plots can be regenerated from tables.
A temporary checkout keeps the current code branch/worktree untouched. Existing
Git credential-helper or SSH authentication is required. No access token is
embedded in the code or result files. Non-fast-forward or authentication failure
is reported in `publish_status.json` and preserves the local ZIP and experiment.

When the run finishes, tell the assistant it is complete so it can read the
results branch. This does not create an autonomous background monitoring service.
Retry a failed publication without rerunning simulation:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/publish_results.py \
  --output-dir JIT/runs/discovery/lower_boundary_v1
```

`JIT_AUTO_PUBLISH=0` disables automatic publishing. Custom output directories
outside `JIT/runs` require the explicit publishing command. Full-detail ZIP exports
do not automatically publish. The first successful server publication creates
the results branch; it need not exist before the run.
