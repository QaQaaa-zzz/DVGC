# Pulse campaign x-z projection

Every completed `run_pulse_exploration.py --mode loop` now creates `lineage/analysis/tube_xz/` before declaring completion. No simulation is run by plotting. Existing failed/paused campaigns are not restarted.

To compare multiple existing lineages with identical axis limits:

```bash
cd /home/qy/DVGC
JAX_PLATFORMS=cpu PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python JIT/cli/plot_pulse_tube_xz.py \
  --lineage /absolute/path/to/first/lineage --label 'First scheme' \
  --lineage /absolute/path/to/second/lineage --label 'Second scheme' \
  --output /absolute/path/to/new/output
```

The output directory must be new. Each scheme gets PNG/PDF/SVG, trajectory coordinates in NPZ with segment offsets, JSON trajectory provenance, candidate CSV and input hashes. Complete round ancestry is resolved via recovery/resume receipts. Only active trace frames are used; successful suffix length, terminal success, physical-failure flags, context identities and file hashes are checked. Prefixes and suffixes are separate line segments so replay discrepancies cannot create artificial connecting lines.

The top panel shows observed successful trajectory witnesses across the policy lineage, including post-landing recovery. The lower panel distinguishes successful, unresolved and unknown handoff positions. Root x/z is not wheel clearance; projections collapse velocity, attitude and history. Unresolved points do not establish physical dead zones. No convex hull or interpolation is filled, and subsequent bounces in recorded successful trajectories are not hidden. This is not a single final Actor's envelope. Missing evidence fails plotting visibly instead of silently producing a partial envelope.
