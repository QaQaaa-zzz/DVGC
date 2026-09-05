# JIT code organization and lifecycle

## Rule

Scientific behavior lives in reusable modules under `JIT/src/jit_dvgc/`.
`JIT/cli/` parses arguments and calls those modules. Tests belong in
`JIT/tests/`; configs in `JIT/configs/`; durable status and method descriptions
in `JIT/docs/`; run artifacts in `JIT/runs/`.

Modify an existing module when changing an existing capability. Do not create
an iteration-numbered source file for each policy, retry or experiment. New
modules are justified only by a genuinely new durable capability.

## Stable areas

### Environment and policy runtime

- `config.py`, `unified_formal.py`, `unified_env.py`
- `checkpoint.py`, `ppo.py`, `unified_training.py`
- `unified_policy_freeze.py`

These own task/runtime identity, Actor observations, checkpoint compatibility,
training and frozen-policy loading. Actor-only warm start imports Actor and
normalizer through the shared loader; critic/optimizer reset is explicit.

### Exact state capture and restoration

- `unified_envelope_snapshot.py`
- `unified_continuation_labels.py`
- `unified_continuation_shards.py`

Snapshots preserve physics, Actor FIFO, action/control and event context.
Sharding is execution-only: catalog order, global candidate index, PRNG scheme,
horizon, endpoint and identities remain invariant.

### Fixed-jump-start acquisition

- `analysis/nominal_jump_centerline.py`
- `acquisition/resolution_frontier.py`
- `acquisition/causal_jump.py`
- `causal_frontier_protocol.py`

These own the locked π0 centerline, every-slice role plan, π0 proposal prefix and
real `env.step` arrival provenance. They do not establish natural-reset
connectivity or formal reachability.

### Family landing labels

- `policy_family_landing.py`
- `cli/label_policy_family_first_landing.py`

This capability evaluates π0/π1/π2 to first valid landing, validates each
evaluator identity, ORs aligned rows and supports independent evaluator shards.
Incomplete evaluator attempts are archived, never overwritten.

### Capability and Tube analysis

- `analysis/causal_jump_capability.py`
- `analysis/capability_tube.py`
- `analysis/jump_tube_view.py`
- `iterative_tube.py`
- `soft_tube.py`

Causal positive cells, all-state control occupancy, semantic jump corridor and
raw Tube rows are distinct outputs. Do not merge their counts or claims.

### Role isolation

- `iterative_frontier_protocol.py`
- `cli/audit_iterative_role_isolation.py`

Derived holdout views remove cross-role and target-Tube exact states using
outcome-blind identities and preserve excluded counts/reasons. The raw role
artifacts remain immutable.

### Predictor

- `family_landing_predictor.py`
- `cli/fit_family_landing_predictor.py`

The module owns upstream fit/calibration, pre-outcome score locking and the
post-label exact join. It is advisory and has no path to arrival proof or Tube
admission. Extend this module for PR-AUC, group-aware intervals and calibration;
do not create iteration-specific predictor files.

### Baseline, gate and selection

- `iterative_acceptance_gate.py`
- `analysis/capability_progression.py`
- selection CLIs under `JIT/cli/`

Baseline and candidate must use the same endpoint, state panel, horizon and
remaining-time semantics. Current code rejects/records endpoint identity, but
historical mixed-endpoint artifacts remain historical and are not repaired by
new code.

### Workflow

- `cli/prepare_iterative_envelope_workflow.py`
- `cli/run_causal_jump_frontier_role.py`

Workflow preparation records a DAG and commands; readiness is not execution
authorization. The workflow must stop at failed acquisition, isolation,
endpoint, support, training or selection gates.

## CLI policy

CLIs should:

- use `argparse` and explicit paths;
- print machine-readable JSON summaries;
- avoid implementing scientific logic;
- preserve existing invocation compatibility when adding a mode;
- fail closed on identity or endpoint drift;
- never silently choose a checkpoint, seed or retry.

## Test policy

Targeted tests should cover:

- candidate/snapshot and policy identity binding;
- first-valid-landing versus stable-recovery separation;
- family OR row alignment and evaluator identities;
- shard coverage/order/seed equivalence;
- role isolation and outcome-blind exclusions;
- Actor-only warm-start parameter routing;
- predictor score locking before label joins;
- historical artifact refusal when required identity fields are missing.

CPU contract tests do not replace a real GPU compile/one-step test or a real
shard equivalence smoke.

## Artifact lifecycle

1. predeclare protocol/config;
2. write raw acquisition without outcome labels;
3. lock optional scores;
4. write immutable evaluator outputs or preserved failure attempts;
5. strictly merge logical labels;
6. derive holdout views without mutating raw data;
7. construct Tube/analysis artifacts;
8. freeze policy and evaluation contracts;
9. index lightweight evidence in Git; keep large checkpoints externally or
   ignored with auditable manifests.

Never edit a historical result to change its endpoint, pass/fail meaning or
creation time.

## Current structural gap

The family CLI can run and merge evaluator shards, but the expanded round still
needs real GPU shard execution and a small-bank serial-equivalence check. A
higher-level family supervisor may be added to the same CLI only after this path
is validated; it must launch fresh child processes and avoid initializing JAX in
the parent.
