# DVGC Experiment State

Current as of 2026-09-01 for branch `agent/two-phase-soft-tube`.

This file is intentionally compact. Historical experiment narratives remain recoverable from Git history and locked run/handoff artifacts; they are not active context for new work.

## Current scientific state

The active method is the iterative single-policy envelope pipeline:

`experts -> Tube_0 -> pi_0 -> C^0 -> Tube_1 -> pi_1 -> paired gate -> repair/accept -> C^1 -> Tube_2 -> pi_2 -> ...`

The first `pi_0 -> pi_1` paired scientific gate has now completed and **rejected pi_1 as the next accepted iteration authority because core preservation failed**, even though boundary gain passed.

### Completed bootstrap

- `pi_up_star`: frozen Propulsion-Ascent expert.
- `pi_down_star`: frozen Descent-Recovery expert.
- `V_up/V_down`: frozen bootstrap continuation models.
- Tube_0: 222 TRAIN-only entries, 117 upstream + 105 downstream.
- `pi_0`: unified Tube-RSI policy, frozen as iteration-0 expansion authority.

### Completed Tube_0 -> Tube_1 construction

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

Config SHA-256 before launch:

`987ef5d31661482fd0bc05cea566c177d83ecd00ae3028ff0e8bb2ed462b7901`

Result:

- requested/completed PPO transitions: 10,009,600 / 10,009,600
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

The exact final checkpoint was frozen as `pi_1` iteration authority with zero environment interactions and zero training transitions.

## Completed paired pi_0 -> pi_1 gate

Completed retry artifact:

`JIT/runs/pi_unified_gate/pi_0_to_pi_1_paired_core_boundary_20260901_retry01`

Locked protocol SHA-256:

`24a126ee94472eebbcb59fff66618ae00dae41074a1d1cfee8bb816afaff410a`

Run result:

- status: completed
- environment interactions: 23,695
- training transitions: 0
- validation data used: false
- TEST data used: false
- expert switching: false

Core preservation:

- 222 Tube_0 core states
- pi_0 success: 222
- pi_1 success: 201
- core regressions: 21
- upstream regressions: 16 / 117
- downstream regressions: 5 / 105
- **CORE PASS = false**

Boundary gain:

- 56 locked pi_0-negative frontier states
- pi_0 reproduction failures: 0
- pi_1 successes: 12
- successful parent groups: 5
- upstream gains: 12 / 26
- downstream gains: 0 / 30
- **BOUNDARY PASS = true**

Iteration decision:

- `ITERATION ACCEPTED = false`
- `EMPIRICAL ENVELOPE EXPANSION ACCEPTED = false`

Artifact file SHAs:

- summary: `cf63f59f4862c51351ceca80afa8796316592be515c28b572584e97e39d9f7fc`
- bank: `97cb62727e12824abc2a5238e9187e47773c80b5f169f2f286c0f412a8e2a6bd`
- records: `614d020198a38235f0a2bfddc6b087fdd5e3729c5fcff49f40c4fcf71683cae2`

This is a scientific rejection. Do not alter the consumed bank, acceptance rule, thresholds, reward, or PPO configuration post hoc to convert it into a PASS.

## Preserved engineering-error provenance

Two engineering-error attempts remain immutable:

1. first formal pi_1 attempt: plotting failed after 1,024,000 PPO transitions; first TRAIN panel had 449 real interactions;
2. first paired-gate attempt: Warp/MJX CUDA OOM after 2,546 interactions because the runner created repeated `jax.jit(env.step)` wrappers.

The gate runner was fixed to share one compiled `env.step`; retry01 then completed successfully under the unchanged scientific protocol.

## Current scientific blocker

No new training should start yet.

The current question is why pi_1 gained new upstream frontier ability while losing 21 previously successful Tube_0 core states.

Working hypothesis: retained-core replay may have been diluted by Tube_1 expansion. Tube_1 structurally retained all 222 core states, but contains 2,897 expansion states and sampling within each 50/50 phase is categorical by entry `sampling_weight`. This makes actual retained-core probability mass, not entry presence alone, the quantity that must be audited.

The hypothesis is not accepted until the existing frozen artifacts show it.

## Immediate next step

Perform zero-interaction diagnosis from the completed gate `records.json` and Tube artifacts:

1. list all 21 core regressions by phase/state/parent/source;
2. summarize pi_1 terminal outcome classes for those regressions;
3. compute Tube_1 retained-core sampling probability mass separately for upstream and downstream;
4. compare regression-state weights with preserved-core weights;
5. inspect whether regressions concentrate in particular parent/source groups or low-probability support;
6. distinguish curriculum/replay dilution from a deeper phase/runtime issue.

Only after this diagnosis may a revised policy-improvement method be predeclared. The rejected pi_1 is not allowed to generate the accepted `C^1 -> Tube_2` scientific chain.

TEST/JCE/JEL remains untouched.

## Repository state

Completed maintenance includes:

- stable package-root APIs for training/tube/snapshots/acquisition/continuation/analysis/workflow;
- removal of redundant facade modules and a first batch of obsolete iteration-0 scaffolding;
- current JIT/root context documentation;
- resumable workflow infrastructure;
- Tube-RSI support for phase-consistent `C^k` fields;
- dependency-closure deletion rules;
- generic paired-policy gate through the existing diagnostic CLI;
- shared compiled `env.step` for long paired audits.

Remaining migration debt before unattended later iterations:

- core-retaining Tube construction still contains Tube_1/iteration-0 constants;
- continuation refit/fresh validation still use a small number of upstream-specific helpers;
- workflow must stop and surface scientific-gate failure rather than proceeding;
- generic k -> k+1 Tube/continuation contracts remain to be completed after the current scientific blocker is resolved.

## Immutable task identity

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`

For exact JIT artifact identities, use `JIT/docs/CURRENT_STATUS.md` as the primary current-state ledger.
