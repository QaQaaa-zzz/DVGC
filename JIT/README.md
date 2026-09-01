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
pi_1
      ↓
core-preservation + boundary-gain gates
      ↓
C^1 → Tube_2 → pi_2 → ...
```

A Soft Tube is training guidance, not a certified safe set. A Tube expansion is
not a capability claim; capability expansion requires a newly trained policy to
preserve the prior core and gain on the boundary.

## Current state

As of 2026-09-01:

- frozen `pi_up_star`: 9,977,856 transitions
- frozen `pi_down_star`: 25,600 transitions
- Tube_0: 222 TRAIN entries
- frozen `pi_0`: completed 10,009,600-transition unified policy
- frozen shared continuation architecture: `76 -> 8 tanh -> 1`, phase-specific
  weights and phase-specific calibration
- Tube_1: 3,119 TRAIN entries, including the exact 222-entry Tube_0 core and
  2,897 expansion states
- `pi_1` retry01: completed exactly 10,009,600 fresh PPO transitions with the
  same 0.1 natural / 0.9 Tube reset mixture used by pi_0
- final TEST/JCE/JEL: untouched

The first pi_1 attempt is intentionally retained as an engineering-error run.
The retry is the completed pi_1 training result. Training completion alone does
not establish envelope expansion.

The next stage is **freeze pi_1 -> core-preservation gate -> boundary-gain
gate**. Only after both pass may the project claim empirical envelope gain and
advance to the next continuation/Tube iteration.

## Repository layout

```text
JIT/
├── cli/                  thin executable entry points
├── configs/              immutable run/protocol declarations
├── docs/                 current method, status, organization, verification
├── handoff/              path-bound locked provenance; do not casually move
├── runs/                 ignored runtime evidence/artifacts
├── scripts/              repository verification/preflight helpers
├── src/jit_dvgc/
│   ├── acquisition/      stable boundary/transition-band API
│   ├── analysis/         bounded TRAIN diagnostics API
│   ├── continuation/     continuation-label/field API
│   ├── snapshots/        snapshot API
│   ├── training/         unified training/freeze API
│   ├── tube/             Tube/Tube-RSI API
│   └── workflow/         resumable iteration orchestration
└── tests/                active regression/contract tests
```

The package directories expose stable APIs from their `__init__.py`; we do not
create a three-line facade file for every historical flat module. A small number
of flat legacy modules remain while active loaders/builders still depend on
them. They are migration debt, not a pattern for new code.

## Maintenance policy

- Modify/consolidate existing code before adding files.
- Do not create `pi2_*.py`, `tube2_*.py`, retry-specific production modules, or
  version-suffixed source trees.
- Completed stage-specific research scaffolding is removed from the active tree
  once superseded and unreferenced; Git history preserves it.
- Configs/frozen manifests/handoff paths that are part of artifact identity stay
  in place even when their producing research script has been retired.
- CLI files stay thin. Scientific/runtime logic belongs in `src`.

See `JIT/AGENTS.md` for the enforced maintenance rules.

## Verification

Use the repository environment only:

```bash
cd ~/DVGC
export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python

$PY -m compileall -q JIT/src JIT/cli
$PY -m pytest JIT/tests -q -m "not gpu"
```

For the curated repository preflight:

```bash
JIT/scripts/local_preflight.sh
```

GPU regression is explicit:

```bash
JIT_RUN_GPU_TESTS=1 JIT/scripts/local_preflight.sh
```

## Iteration workflow

`jit_dvgc.workflow` sequences production CLIs and verifies machine-readable
artifacts. It is deliberately scientifically ignorant: it does not invent
thresholds or decide that a failed gate should pass.

Plan only:

```bash
$PY JIT/cli/run_iteration_workflow.py --config <workflow.json>
```

Explicit execution/resume:

```bash
$PY JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute
```

The same workflow state is resumable after an engineering failure, provided the
workflow config SHA is unchanged and completed artifacts still validate.

The orchestration engine exists now, but the iteration-0-specific continuation
and Tube construction contracts are still being generalized before unattended
`pi_1 -> Tube_2 -> pi_2` execution is enabled. Do not mistake orchestration
infrastructure for completed scientific generalization.

## Historical material

Old Phase-U plans, one-off CV/refinement scripts, and superseded research
scaffolding are intentionally not kept in the active tree. They remain available
through Git history. Path-bound frozen artifacts/configs/handoff records are not
removed merely to make the repository look smaller.
