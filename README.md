# OrangeBike DVGC

DVGC studies policy-conditioned jumping capability for a single-track two-wheeled robot. The active JIT method builds and iteratively expands an empirical jumping-capability envelope while keeping deployment as **one unified Actor**.

## Current method

```text
Propulsion-Ascent expert
        +
Descent-Recovery expert
        ↓
expert-conditioned continuation labels
        ↓
V_up / V_down
        ↓
Tube_0
        ↓
pi_0
        ↓
freeze pi_k
        ↓
real-dynamics boundary evidence
        ↓
C_up^k / C_down^k
        ↓
core-retaining Tube_(k+1)
        ↓
pi_(k+1)
        ↓
core-preservation + boundary-gain gates
        ↓
repeat until a predeclared stopping condition
        ↓
independent final frozen-policy JCE/JEL evaluation
```

The Apex region is a physical transition band between the two bootstrap phases, not a third runtime expert. Phase experts are data-generation/bootstrap tools only. Final and iterative unified policies never switch experts at runtime.

A learned Soft Tube is **training guidance**, not a certified safe set or viability kernel. A larger Tube alone is not evidence of improved capability.

## Current implementation status — 2026-09-01

Completed:

- frozen `pi_up_star` and `pi_down_star`;
- bootstrap `V_up/V_down`;
- 222-entry TRAIN-only Tube_0;
- unified `pi_0`, frozen as expansion authority;
- pi_0-conditioned boundary evidence and continuation fields `C_up^0/C_down^0`;
- fresh independent continuation validation/calibration;
- core-retaining Tube_1 with 3,119 TRAIN entries;
- Tube_1 mixed-snapshot Tube-RSI engineering gate;
- fresh `pi_1` formal PPO at exactly 10,009,600 training transitions.

The authoritative completed pi_1 run is:

`JIT/runs/pi_unified/pi_1_tube1_natural10_10009600_seed821101_20260901_retry01`

The next scientific action is **freeze exact pi_1 -> core-preservation gate + boundary-gain gate**. Only if both pass may the project record empirical envelope expansion and proceed to `C^1 -> Tube_2 -> pi_2`.

See `JIT/docs/CURRENT_STATUS.md` for exact hashes and artifact identities.

## Authoritative physical contract

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg (`4kg` in the filename is historical)
- control: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`

Repository cleanup and envelope iteration must not silently change physics, reward semantics, action semantics, snapshot semantics, or TEST isolation.

## Runtime

Use the configured environment directly:

```bash
export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python
```

Current lightweight verification:

```bash
"$PY" -m compileall -q JIT/src JIT/cli
JIT/scripts/local_preflight.sh
```

Formal GPU work is launched only after artifact/config gates pass.

## Active JIT layout

```text
JIT/
├── cli/                  thin executable entry points
├── configs/              run/protocol declarations
├── docs/                 current method, status and verification contracts
├── handoff/              path-bound locked provenance
├── scripts/              local verification/maintenance entry points
├── src/jit_dvgc/
│   ├── training/         unified PPO + freezing API
│   ├── tube/             Tube construction / Tube-RSI API
│   ├── snapshots/        snapshot/pool API
│   ├── acquisition/      real-dynamics boundary acquisition API
│   ├── continuation/     continuation labels/fields API
│   ├── analysis/         bounded TRAIN diagnostics
│   └── workflow/         resumable stage orchestration
└── tests/                current regression and scientific-contract tests
```

Historical flat modules may remain temporarily when current artifacts/imports still require them. New iteration-specific production modules should not be added; iteration numbers belong in configs and artifacts.

## Automation direction

`JIT/cli/run_iteration_workflow.py` is the stable orchestration entry point. The intended end state is one explicit launch that can sequence and resume:

`freeze pi_k -> gates -> TRAIN evidence -> C^k -> fresh validation -> Tube_(k+1) -> Tube-RSI smoke -> pi_(k+1)`.

The runner may automate execution and artifact verification, but it may not retune the method or bypass a failed scientific gate. Final TEST/JCE/JEL is intentionally outside this loop.

## Where to read next

- `AGENTS.md` — repository-wide execution and cleanup rules
- `JIT/AGENTS.md` — active JIT implementation rules
- `PROJECT.md` — scientific method/claim boundary
- `JIT/docs/CURRENT_STATUS.md` — exact current state
- `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md` — iteration/leakage contract
