> **2026-09-08历史结果与当时缺口，保留溯源。** 2026-09-12当前实验矩阵见[训练路线](JIT_TRAINING_ROADMAP.md)和[论文草稿](paper/JIT_PAPER_DRAFT.md)。旧文“pending零reset准入/实现尚缺/下一条命令”只适用于当时协议，不是当前执行指令。

# Current boundary result, training support, and paper experiment gaps

The corrected run completed: 290 arrivals, 93 all-policy successes, 102
disagreements, 95 no-witness states. All 95 are real arrivals without a successful
continuation under the current bank; they are not certified impossible states.
Nine trajectories completed without sampling truncation. At window endpoint
x=2.85 m, all three strengths landed. At x=2.90/2.95, all six had physical failure.
This supports sensitivity to the action window in this declared experiment;
it does not prove that timing is universally more important than strength.
Nine conditions from one development run are not nine independent repetitions.

Continuation successes: pi_0=106, pi_1=97, pi_2=172, pi_3=135. Their union is 195.
The bank adds 23 witnessed states beyond pi_2 on this panel. It is a proposer-
conditioned TRAIN result, not a global policy ranking or physical-cell increment.
Do not add 195 to the previous 3,463 cells; cumulative physical deduplication of
this round has not been recomputed. Recorded cost 27,954; retain the earlier
failed-attempt reservation 4,500 separately for cross-attempt cost reporting.

## Next executable step: support construction, no new simulation

```bash
cd /home/qy/DVGC &&
git switch agent/two-phase-soft-tube &&
git pull --ff-only origin agent/two-phase-soft-tube &&
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/prepare_complementary_support.py
```

Source: `JIT/runs/discovery/knee_boundary_v1_budgetfix`. Output:
`JIT/runs/training_support/complementary_boundary_v1`. Send the compact ZIP.
The command reads local raw snapshots, complete labels and manifests, validates
arrival/context/policy/endpoint identities, and pins input file hashes. It does
not require historical source code to equal the current code when reading a
completed run, but it verifies historical plan and artifact identities.

`support.json` contains witnessed states only, split into all-success and
disagreement strata. Within each stratum, trajectory/phase groups have equal mass,
then entries within each group have equal mass. These are normalized conditional
weights, not a fully specified reset mixture. The 95 no-witness entries remain
separate development targets with zero reset admission. Select the frozen
initializer by maximum TRAIN successes (ties by policy name); currently pi_2.
This selection uses development outcomes and is not a blinded comparison.

This artifact is **not a PPO configuration**. Remaining implementation work:
consume complete unified snapshots in the training reset path, implement the
explicit fixed x=2.5 start/RSI mixture, lock strata probabilities, reward,
normalizer handling, fresh critic/optimizer, budget and control experiment, then
run a small training smoke. Do not use the old automatic successor/core-replay
config as if it already implemented this contract. No further broad scan or
additional numerical replay gate is scheduled.

## What remains for the current paper claims

These are evidence requirements for this project's outline, not journal rules
or guarantees of acceptance. Do not assign a precise completion percentage.

| Experiment block | Current evidence | Still needed |
| --- | --- | --- |
| Complementary learning and iteration | Existing four frozen policies and localized persistent gap | Train at least one deliberately complementary probe; measure new arrivals and new suffix witnesses, cumulative cells and total cost |
| Controlled discovery/learning benefit | Descriptive curves with changing domains/budgets | Lock one domain, physical resolution and interaction budget; compare pi_0/fixed bank/growing bank. Compare frontier-informed training with a matched uniform-support control |
| Independent repetition and final evaluation | One adaptive development run per recent pilot | Start with at least three independent seeds per main condition; freeze recipe and holdout before opening final evaluation; use trajectory/ancestor-group uncertainty |
| Bootstrap and reporting | Up/down, Tube0, pi_0 historical completion; plots exist | If claiming bootstrap efficiency, compare with direct full-task learning under accounted total cost. Add resolution sensitivity, incremental width/cell plots, failure taxonomy and end-to-end cost ledger |

For an initial controlled learning test, two training conditions × three seeds
means six new training jobs, subject to a locked affordable budget. Fixed frozen
bank controls require evaluation/discovery work, not necessarily new PPO.
Bootstrap comparisons may require additional full pipelines if historical jobs
do not match the locked protocol. These counts overlap and are not a promise of
an exact number of GPU hours or sufficient statistical power. Determine budgets
from the new training smoke, then report the complete run matrix without double
counting reusable baselines. More seeds may be needed if variance is large.

Resolution sensitivity and much of plotting/statistics can reuse stored data on
CPU. They should not trigger another large label scan. If bootstrap is presented
only as setup, remove untested efficiency claims rather than imply an absent
ablation was passed. With the current full outline, the four blocks above remain
material work. This is beyond pipeline debugging but not the final writing-only
stage. Simulated results support the declared model/condition only.
