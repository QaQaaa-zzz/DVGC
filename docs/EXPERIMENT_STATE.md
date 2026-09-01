# DVGC Experiment State

Current as of 2026-09-01 for branch `agent/two-phase-soft-tube`.

This file is intentionally compact. Historical experiment narratives remain recoverable from Git history and locked run/handoff artifacts; they are not active context for new work.

## Current scientific state

The active method is the iterative single-policy envelope pipeline:

`experts -> Tube_0 -> pi_0 -> C^0 -> Tube_1 -> pi_1 -> gates -> C^1 -> Tube_2 -> pi_2 -> ...`

### Completed bootstrap

- `pi_up_star`: frozen Propulsion-Ascent expert.
- `pi_down_star`: frozen Descent-Recovery expert.
- `V_up/V_down`: frozen bootstrap continuation models.
- Tube_0: 222 TRAIN-only entries, 117 upstream + 105 downstream.
- `pi_0`: unified Tube-RSI policy, frozen as iteration-0 expansion authority.

### Completed iteration 0 -> 1

Fresh independent validation authorized Tube_1 using frozen thresholds:

- upstream threshold: `0.9333483934566058`
- downstream threshold: `0.8721734129976408`

Tube_1:

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

- manifest SHA-256: `817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`
- exact retained Tube_0 core: 222
- expansion: 2,897 = 310 upstream + 2,587 downstream
- total: 3,119 = 427 upstream + 2,692 downstream
- validation rows embedded: 0
- TEST rows embedded: 0
- construction interactions: 0
- construction training transitions: 0

Tube_1 mixed-snapshot Tube-RSI engineering gate passed.

### pi_1

Authoritative completed run:

`JIT/runs/pi_unified/pi_1_tube1_natural10_10009600_seed821101_20260901_retry01`

Config:

`JIT/configs/pi_unified_iter1_tube1_natural10_retry01.json`

Config SHA-256 before launch:

`987ef5d31661482fd0bc05cea566c177d83ecd00ae3028ff0e8bb2ed462b7901`

Result:

- requested PPO training transitions: 10,009,600
- completed PPO training transitions: 10,009,600
- checkpoints: 0 / 1,024,000 / 2,508,800 / 5,017,600 / 7,500,800 / 10,009,600
- completed TRAIN panels: 5
- TRAIN-panel interactions: 2,838
- Brax evaluation transitions: 0
- reset mixture: 0.1 natural / 0.9 Tube
- actor/critic/optimizer: fresh initialization
- seed: 821101
- expert switching: false
- validation data used: false
- TEST data used: false
- final checkpoint restore: verified

Optimization metrics are training diagnostics only and are not capability gates.

## Preserved engineering-error attempt

The first pi_1 attempt:

`JIT/runs/pi_unified/pi_1_tube1_natural10_10009600_seed821101_20260901`

is preserved unchanged as engineering-error provenance.

It reached 1,024,000 training transitions. Its first TRAIN-panel `report.json` proves 449 environment interactions, while the terminal status recorded diagnostic interactions as 0 because plotting failed after rollout but before the callback returned.

Do not rewrite that historical artifact. Production source now contains a failed-run reconciliation path for future runs, and formal training performs a zero-interaction full-Tube preflight before training.

## Current scientific next step

No new training should start before this sequence is closed:

1. freeze exact `pi_1` final checkpoint as iteration-1 authority;
2. predeclare comparable core-preservation and boundary-gain protocols;
3. run core-preservation gate;
4. run boundary-gain gate;
5. only if both pass, record empirical `pi_0 -> pi_1` capability-envelope expansion;
6. collect/freeze pi_1-conditioned TRAIN evidence;
7. fit and independently validate `C_up^1/C_down^1`;
8. construct core-retaining Tube_2;
9. run Tube_2 engineering gate;
10. train pi_2;
11. repeat through the same generic production capabilities.

TEST/JCE/JEL stays untouched throughout the iteration loop.

## Repository state

Repository cleanup is active but must preserve dependency closure.

Completed maintenance includes:

- stable package-root APIs for training/tube/snapshots/acquisition/continuation/analysis/workflow;
- removal of redundant facade modules;
- retirement of a first batch of completed iteration-0 research scaffolding;
- current JIT status/organization/verification docs;
- resumable manifest-driven workflow infrastructure;
- Tube-RSI acceptance of `C_up^k/C_down^k` rather than only `C^0`;
- restoration of `upstream_boundary_lock.py` after a cleanup regression proved it remains in the active bootstrap import closure;
- explicit deletion-gate rules requiring compile/import/test closure before further removals.

Remaining migration debt before unattended pi_2+ execution:

- core-retaining Tube construction still contains Tube_1/iteration-0 constants;
- continuation refit/fresh validation still use a small number of upstream-specific helpers;
- freeze/gate stages need stable machine-readable iteration-generic APIs wired into the workflow;
- cleanup must continue only from a green compile/test baseline.

## Immutable task identity

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`

For exact JIT artifact identities, use `JIT/docs/CURRENT_STATUS.md` as the primary current-state ledger.
