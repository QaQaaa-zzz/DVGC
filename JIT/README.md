# JIT — policy-conditioned Soft-Tube envelope iteration

`JIT/` is the active implementation of a single unified jumping policy trained
through iteratively expanded, TRAIN-only Soft-Tube reset support.

The method is:

```text
pi_up + pi_down
      ↓
bootstrap V_up / V_down
      ↓
Tube_0
      ↓
pi_0
      ↓
TRAIN boundary evidence → C_up^0 / C_down^0 → fresh validation
      ↓
core-retaining Tube_1
      ↓
pi_1 candidate
      ↓
core-preservation + boundary-gain gates
      ↓
accepted pi_1 → C^1 → Tube_2 → pi_2 → ...
```

A Soft Tube is training guidance, not a certified safe set. A Tube expansion is
not a capability claim; empirical envelope expansion requires the newly trained
policy to preserve the prior core and gain on a locked boundary challenge bank.

## Current state — 2026-09-02

The project is at **repaired iteration-1 acceptance**.

Completed:

- frozen `pi_up_star`: 9,977,856 transitions
- frozen `pi_down_star`: 25,600 transitions
- Tube_0: 222 TRAIN entries
- frozen `pi_0`: 10,009,600-transition unified iteration-0 authority
- `C^0` continuation fields with independent fresh validation
- Tube_1: 3,119 TRAIN entries = exact 222-entry Tube_0 core + 2,897 expansion
- first Tube_1 `pi_1` candidate: completed, frozen, then scientifically rejected
  because the paired gate found 21 core regressions while boundary gain passed
- zero-interaction diagnosis: retained-core replay dilution was a material
  mechanism
- repaired Tube-RSI contract: 50% retained core / 50% expansion inside each
  phase, giving 45% core / 45% expansion / 10% natural reset mass overall
- two consumed single-axis fresh-bank readiness probes: both FAIL, especially
  because downstream produced zero baseline negatives
- acquisition generalized to iteration-generic sparse action directions
- fresh two-axis acquisition: 3,720 unique TRAIN candidates
- long frozen-policy labeling recovered from CUDA/Warp single-process OOM by
  four sequential 930-candidate GPU processes with one merged logical label set
- fresh locked acceptance bank: 260 frozen-`pi_0` negatives = 246 upstream
  across 4 parent groups + 14 downstream across 5 parent groups; readiness PASS
- repaired `pi_1`: completed exactly 10,009,600 fresh PPO transitions and frozen
- final TEST/JCE/JEL: untouched

Repaired formal run:

`JIT/runs/pi_unified/pi_1_tube1_core_replay50_natural10_10009600_seed821101_20260902`

Frozen repaired candidate:

`JIT/runs/frozen_unified/pi_1_core_replay50_10009600_20260902/frozen_unified_policy.json`

Known local identities:

- final checkpoint payload SHA-256: `ea93a534c2c6bb3bf145684cbea82df94fefa2df8099dcdcdd9492bd8007e205`
- frozen manifest file SHA-256: `d5a1658530d475a67264aa5c621283d71c823200dbee6068f93413b93d06b7a8`

**Training/freeze completion does not accept iteration 1.**

The next stage is exactly one repaired `pi_0 -> pi_1` paired gate using:

- all 222 Tube_0 core states;
- the fresh locked 260-state two-axis frozen-`pi_0` negative bank;
- zero baseline-success -> candidate-failure core regressions;
- complete baseline-negative reproduction;
- candidate gains in at least 2 distinct parent groups;
- no validation, TEST, expert switching or training.

Only if both core preservation and boundary gain pass may the project accept
iteration 1 and advance to `C^1 -> Tube_2 -> pi_2`.

For the authoritative ledger see `JIT/docs/CURRENT_STATUS.md`. For the compact
resume marker see `docs/EXPERIMENT_STATE.md`. For today's execution history see
`JIT/docs/experiments/2026-09-02-iteration1-repair-handoff.md`.

## Repository layout

```text
JIT/
├── cli/                  thin executable entry points
├── configs/              immutable run/protocol declarations
├── docs/                 method, status, organization, verification, handoffs
├── handoff/              path-bound locked provenance; do not casually move
├── runs/                 ignored runtime evidence/artifacts
├── scripts/              repository verification/preflight helpers
├── src/jit_dvgc/
│   ├── acquisition/      stable real-dynamics acquisition API
│   ├── analysis/         bounded TRAIN diagnostics API
│   ├── continuation/     continuation-label/field API
│   ├── snapshots/        snapshot API
│   ├── training/         unified training/freeze API
│   ├── tube/             Tube/Tube-RSI API
│   └── workflow/         resumable iteration orchestration
└── tests/                active regression/contract tests
```

Package directories expose stable APIs from `__init__.py`. Do not create
iteration-named production modules such as `pi2.py`, `tube2.py`, retry-specific
source trees, or one-off wrappers when the capability belongs in an existing API.

## Maintenance policy

- Modify/consolidate existing code before adding production files.
- Experiment/iteration/retry identities belong in configs, data and run names.
- Keep CLI files thin; reusable runtime/scientific logic belongs under
  `JIT/src/jit_dvgc/`.
- Do not move path-bound configs, frozen manifests or handoff provenance for
  aesthetics.
- Before deleting code, prove dependency closure and run compile/targeted tests.
- Git history is the archive for superseded source code.

See `JIT/AGENTS.md` for the enforced rules.

## Verification

Use the repository environment only:

```bash
cd ~/DVGC
export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python

$PY -m compileall -q JIT/src JIT/cli
$PY -m pytest JIT/tests -q -m "not gpu"
```

Curated repository preflight:

```bash
JIT/scripts/local_preflight.sh
```

Explicit GPU regression:

```bash
JIT_RUN_GPU_TESTS=1 JIT/scripts/local_preflight.sh
```

## Iteration workflow

`jit_dvgc.workflow` is an execution/resume orchestrator. It must not invent
scientific thresholds or auto-convert a failed gate into a passing method.

Plan only:

```bash
$PY JIT/cli/run_iteration_workflow.py --config <workflow.json>
```

Explicit execution/resume:

```bash
$PY JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute
```

The workflow config SHA is immutable after state creation. Completed artifacts
may be reused only after identity revalidation. Scientific or engineering failure
must stop the workflow.

Unattended later iterations are not authorized until the remaining
iteration-0-specific continuation/Tube construction paths are generalized for
`k -> k+1` and covered by tests.
