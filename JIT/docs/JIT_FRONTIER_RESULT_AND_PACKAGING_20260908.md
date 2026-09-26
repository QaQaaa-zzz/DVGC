# Landing frontier result and compact review packages

Production completed: 48 trajectories, 1,141 arrivals, 1,129 states with at least
one landing witness, 1,107 novel root cells and 3,463 cumulative root cells.
Cost: 92,058 environment interactions; zero PPO. Forty trajectories reached first
valid landing, eight terminated earlier, none were truncated by sampling limits.
Only two of the eight early endings have `physical_failure=true`; do not call
all eight crashes. Existing terminal flags need more detailed reason reporting
before attributing those other six endings to a particular physical cause.

Of the novelty, 840 cells are in the previous observation corridor and 267 are
exclusive to its extension. The 47.0% increase in cell count is not a 47% increase
in geometric width or continuous volume. Profiles have unequal budgets and
strengths, so this round is not a fair ranking of the four policies.

There are 1,035 all-policy successes, 94 disagreements and 12 no-witness states.
All 12 belong to one pi_1 positive-knee offset 0.2 trajectory, with sampled root
x about 3.982–4.548 m. They are correlated frames, not independent failed trials.
See [the recorded production summary](verification/landing_frontier_production_20260908.json).

Next work should refine this specific TRAIN boundary with predeclared intermediate
action strengths and additional proposal groups, keeping real forward dynamics.
Use nearby witnessed states and disagreement cases to define a complementary
training support; lock initializer, reset mixture, budget and stopping criterion
before PPO. Evaluate the new probe by added witnessed support and cost. Do not
automatically promote every failed state to training or claim infeasibility.
No repeated broad four-policy scan or numerical replay gate is required now.

## Repackage existing results only

From the repository root after updating the branch:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/package_results.py \
  --output-dir JIT/runs/discovery/landing_frontier_v1
```

Send the printed `results_to_send.zip`. This command requires no GPU or rerun
and does not invoke experiment source-lock validation. Do not rerun an old
experiment command just to change packaging after its source code has changed.

Comparison, dense-pilot and discovery supervisors now use compact packages by
default. They retain overview PNGs, CSV analysis tables, run identity/plans,
summaries, cost and process receipts, protocol/merge audits, and log tails up to
32 KiB each. Repeated child plots, PDF/SVG editions and raw snapshot/label
payloads stay on the server. `bundle_inventory.json` identifies omissions,
included-file hashes and any truncated log copies. Raw files are never deleted.
This is a review package, not a standalone replay dataset; request a specific
omitted artifact if a later question requires it.

Optional `--full` writes `results_full.zip` with the prior detailed review formats
(still excluding checkpoints/NPZ and individual files over 20 MB). Atomic archive
replacement preserves the previous package if packaging fails.
