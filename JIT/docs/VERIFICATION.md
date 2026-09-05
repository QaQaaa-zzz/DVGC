# JIT verification — current production checks

This page contains current verification only. Historical Phase-U v2/v3/v4
instructions were removed from the active tree and remain available in Git
history.

## 1. Repository/static preflight

```bash
cd ~/DVGC
export PYTHONPATH="$PWD/JIT/src"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
PY=/home/qy/mujoco_playground/.venv/bin/python

$PY -m compileall -q JIT/src JIT/cli
$PY -m pytest JIT/tests -q -m "not gpu"
```

Or use:

```bash
JIT/scripts/local_preflight.sh
```

GPU tests are explicit:

```bash
JIT_RUN_GPU_TESTS=1 JIT/scripts/local_preflight.sh
```

Do not interpret an XLA autotuning warning or the existing
`ccd_iterations=35` warning as the cause of a run failure unless it is actually
in the exception path. Do not change physics/solver settings during a pi_k vs
pi_(k+1) single-variable comparison merely to silence a warning.

## 2. Current pi_1 completion evidence

Completed run:

`JIT/runs/pi_unified/pi_1_tube1_natural10_10009600_seed821101_20260901_retry01`

Required local checks before freezing:

- `formal_report.json` status is `completed`
- completed transitions = 10,009,600
- checkpoint list ends at `transition_10009600`
- all five TRAIN panels are present
- train-panel interactions = 2,838
- Brax evaluation transitions = 0
- validation/TEST flags are false
- expert switching is false
- checkpoint restoration is true

The first failed pi_1 run is not a scientific checkpoint source and must not be
used for warm-start. It remains an engineering-error provenance record.

## 3. Formal-training plotting hardening

`jit_dvgc.training.run_unified_formal` performs a full configured-Tube static
snapshot/plot-point preflight before constructing the training environment. It
must report zero environment interactions and zero training transitions.

The mixed-snapshot regression must cover both `handoff_snapshot_v1` and
`jit_unified_envelope_snapshot_v1`.

## 4. Freeze then capability gates

The next legal scientific sequence is:

```text
completed pi_1 final checkpoint
        ↓
freeze exact pi_1 identity
        ↓
core-preservation gate
        ↓
boundary-gain gate
        ↓
PASS + PASS ? allow empirical envelope expansion : stop / diagnose
```

Neither training reward nor a larger Tube substitutes for these gates.

## 5. Iteration automation verification

Plan a workflow without executing it:

```bash
$PY JIT/cli/run_iteration_workflow.py --config <workflow.json>
```

Execute/resume only with an explicit flag:

```bash
$PY JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute
```

For every stage, verify that:

- the declared completion artifact exists
- JSON assertions pass before the next stage starts
- exported SHA/path values come from that artifact, not from handwritten shell
  variables
- an engineering/scientific failure stops the workflow at that stage
- restarting with the unchanged workflow config revalidates completed artifacts
  and resumes rather than overwriting them
- the workflow contains no final TEST/JCE/JEL stage

## 6. Claims that remain prohibited

Until core-preservation and boundary-gain both pass for frozen pi_1, do not
claim:

- empirical capability-envelope expansion from pi_0 to pi_1
- pi_1 as final unified policy
- Tube_1 as a certified safe/viable set
- JCE/JEL final performance

TEST remains untouched until a final frozen policy is selected.
