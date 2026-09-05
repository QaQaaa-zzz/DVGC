# Two-Phase Gate C Expert-Training Foundation Design

Implementation marker: Gate C1 source capability is implemented through
`b36cfec`; dynamic smoke evidence is recorded separately and never upgrades
this design into a learnability or formal-training authorization.

## Status and scope

This document defines the future stable two-phase expert-training interface:

```text
cli/train_phase_expert.py
dvgc/phase_expert_training.py
```

It covers both `propulsion_ascent` and `descent_recovery` without creating
version-suffixed scripts. The current implementation target is Gate C1 smoke
capability, with Phase U first. After implementation review, the user's
delegated decision permits Codex to authorize at most one run-bound Phase U
smoke. It does not authorize a learnability pilot, formal expert training,
Phase D execution, snapshot collection, feasibility learning, Soft Tube
construction, or unified PPO.

## Architectural choice

The new training path uses an external `PhaseExpertEnvAdapter` around the
authoritative `OrangeBikeDVGC` dynamics. It reuses the immutable XML, action
mapping, actor/privileged observations, history machinery, physical failures,
and pure-JAX two-phase runtime. It owns only the new phase reset selection,
two-phase event state, success predicate, reward, timeout, and metrics.

This is preferred over adding `propulsion_ascent` and `descent_recovery` to the
legacy `training_stage` state machine or copying the environment. Legacy phase
ids, matcher terminals, chain success, and old expert policies cannot satisfy a
new expert success condition.

Adapter-owned event/counter leaves are namespaced and carried by the wrapper;
`OrangeBikeDVGC.step` remains the physical transition function and is not
modified merely to store two-phase latches.

## Stable public interfaces

`dvgc.phase_expert_training` will define these stable contracts:

```python
PHASE_PROPULSION_ASCENT = "propulsion_ascent"
PHASE_DESCENT_RECOVERY = "descent_recovery"

@dataclass(frozen=True)
class PhaseExpertRunSpec:
    phase: str
    experiment_level: str
    requested_total_transitions: int
    seed: int
    config_path: str
    training_config_path: str
    threshold_manifest_path: str
    authorization_manifest_path: str | None
    output_dir: str
    descent_seed_bank: str | None
    descent_seed_manifest: str | None
    resume_run: str | None
    restore_checkpoint: str | None

@dataclass(frozen=True)
class PhaseExpertResetProtocol:
    phase: str
    mode: str
    seed_tier: str | None
    source_hash: str | None
```

The module exposes these operations with typed inputs and returns:

- `validate_phase_expert_run_spec(spec)` returns validated immutable manifest
  inputs;
- `build_phase_expert_budget(spec, layout)` returns `PPOBudgetReport`;
- `build_phase_expert_environment(spec)` returns `PhaseExpertEnvAdapter`;
- `evaluate_phase_expert_fixed(policy, protocol, checkpoint_step)` returns the
  evaluation report;
- `run_phase_expert(spec)` returns the final run report.

Implementation variants belong in the resolved project/training configs; they
do not create alternate public phase names, budget fields, reset modes,
artifact roles, or output filenames.

## CLI contract

The single entrypoint is:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.train_phase_expert \
  --phase propulsion_ascent|descent_recovery \
  --experiment-level smoke \
  --requested-total-transitions N \
  --config configs/default.json \
  --training-config configs/phase_expert_smoke.json \
  --threshold-manifest runs/two_phase/<gate_b_run>/threshold_manifest.json \
  --run runs/two_phase/phase_experts/<run_id> \
  --seed S
```

Optional resume inputs are:

```text
--resume-run <prior run directory>
--restore-checkpoint <exact Orbax checkpoint>
```

Phase D additionally requires:

```text
--descent-seed-bank <timing-explicit snapshot bank>
--descent-seed-manifest <physical-validation manifest>
```

The CLI has no independent `--timesteps` input. The only public budget input is
`requested_total_transitions`; Brax's `requested_timesteps` is a report alias
that must be identical. A new run directory must not already exist. Resume
continues into a new output directory and never overwrites its parent.

At the initial Gate C1 implementation, `--experiment-level smoke` is the only
executable level. Pilot and formal values are rejected with a clear
authorization error even though their budgets are documented below.

Normal execution also requires `--authorization-manifest`. `--preflight-only`
may omit it because that path constructs no environment and emits zero
transitions. The authorization is a single-run JSON contract binding phase,
experiment level, run id, source HEAD, XML hash, threshold-manifest canonical
hash, training-config hash, requested/effective training-transition ceilings,
fixed-evaluation and combined-interaction ceilings, issuer, issue time, and the
literal decision `authorize`. Reuse for another run or any hash/budget drift is
rejected. Authorization is evidence of permission, not evidence of safety or
learnability.

At Gate C1, `descent_recovery` is accepted only by `--preflight-only` for seed
manifest validation. Any attempt to construct a Phase D environment or execute
PPO fails closed until Gate C2 is separately released.

## Phase-specific reset contracts

### Propulsion-Ascent

Phase U uses:

```text
mode = natural_start
source_phase = propulsion_ascent
use_bank_resets = false
```

Every reset originates from the current model's natural reset and passes the
Gate B natural-start checks. Neither a reference row nor a reference action may
select or modify the initial state. The training distribution may use only the
declared config randomization and training seed namespace.

The target sequence is:

```text
natural stable start -> forward propulsion -> jump -> valid Apex band
```

### Descent-Recovery

Phase D never uses the natural ground reset. The reset protocol accepts only a
bank whose manifest declares one of:

```text
physically_validated_descent_seed
pi_up_online_apex_snapshot
```

Preliminary reference-proposed seeds are permitted only for smoke or a
separately authorized early pilot. Formal expert training requires a declared
mixture dominated by `pi_up_online_apex_snapshot` records from frozen Phase U
rollouts, including Apex pre/nearest/post and early descent. Dominated means
that this source is more than half of both admitted seed records and declared
reset sampling mass. Missing or invalid seed provenance is a pre-run error, not
a reason to fall back to natural reset.

## Success, failure, and timeout

`propulsion_ascent_success` is the instantaneous physical membership gate over
deployable `ApexBandSignals` and the frozen Gate B thresholds. Formal Phase U
terminal success is the first `TwoPhaseEventState.apex_band_entered` tick,
which additionally closes the monotonic legal-window, liftoff, airborne, and
ascending event order. It requires:

```text
obstacle-relative x window
full-structure clearance
vertical-velocity band
roll/pitch/angular-rate bounds
minimum forward velocity
stable airborne state
no illegal contact
no physical failure
```

Early airborne, jump-window entry, liftoff, height, or reference proximity
alone cannot succeed.

Phase D success is exactly `descent_recovery_success`: sustained legal wheel
support in the landing region, no chassis/payload contact, pose/rate/speed
bounds, no physical failure, and the full consecutive recovery hold.

Existing prohibited contact, invalid wheel contact, roll, pitch, backward,
platform back-edge, and nonfinite failures remain hard. Expert-specific timeout
is truncation. Fixed evaluation reports the mutually exclusive outcome
categories `success`, `physical_failure`, `timeout`, and `other_failure`, plus
fine-grained terminal reasons.

For Gate C1 the base environment is resolved to `training_stage=full`,
`use_bank_resets=false`, `expert_chain_termination=false`, an empty legacy
reachability objective, and no domain randomization. Its raw step supplies only
the immutable physical transition, observations/history, contacts, and failure
telemetry. The adapter clears an incoming legacy `done`, discards legacy reward,
success, recovery, chain-entry, stage-entry, and timeout decisions, and owns the
published reward/done/metrics. Physical end codes 2-7 and 15 remain terminal;
takeoff task codes 10-13 may terminate only after the legal jump latch and are
reported as `other_failure`. Codes 1, 8, 14, and 16 cannot terminate a Phase U
expert episode. The adapter horizon and Brax `episode_length` are identical, so
there is one authoritative timeout.

Natural-reset validation has two levels. A host preflight validates the bounded
configured reset domain and fixed seed suite without changing root height or
searching for a passing state. Every JAX reset also evaluates finite state,
legal-support/contact, pose, nonterminal, neutral-action, and history predicates;
an invalid reset fails closed. Gate C1 smoke disables domain randomization;
enabling it later requires a separately reviewed reset-domain audit.

## Reward contracts

Rewards are phase-specific, bounded, physical, and independent of reference
action replay, feasibility models, learned Tubes, matchers, teacher identity,
or outcome metadata.

Phase U reward contains:

- bounded forward propulsion before the jump window;
- zero airborne/takeoff progress reward before legal window activation;
- bounded legal-window jump and ascent progress;
- bounded approach toward the full Apex-band components;
- pose/rate, control smoothness, and energy regularization;
- a dominant one-time reward only for full Apex-band success;
- unchanged physical-failure penalty.

It must be impossible to earn expert success by hovering, jumping early,
following reference hip/knee values, or reaching a global height away from the
obstacle.

Phase D reward contains:

- bounded descent/landing progress derived from deployable physical signals;
- legal support and landing-region shaping;
- pose/rate/forward-speed and impact-absorption terms;
- control smoothness and energy regularization;
- a dominant one-time reward only after stable-recovery success;
- physical-failure penalty and no survival-only substitute for recovery.

All reward components and their bounds are emitted in the run manifest and
metrics. Changing reward meaning after a smoke run requires a new reviewed
config/hash; resume cannot silently change it.

## PPO budgets and authorization

All public accounting uses total environment transitions and
`dvgc.training_budget.PPOBudgetReport`.

```text
Smoke:
  1-4 aligned PPO rollout blocks
  typically 1,600-6,400 total environment transitions

Learnability pilot:
  exactly 102,400 total environment transitions

Formal first authorization:
  500,000 requested total environment transitions

Maximum current authorization:
  effective total environment transitions <= 2,000,000
```

The report must satisfy:

```text
requested_timesteps == requested_total_transitions
effective_timesteps == effective_total_transitions
```

Gate C wraps this report in a phase interaction budget. A valid smoke request
has zero alignment overhead and exactly one through four rollout blocks. The
wrapper also records Brax-internal and fixed-evaluation environments, horizon,
cadence/count, and their transition ceilings. Its combined ceiling is training
plus both evaluation categories; Brax internal evaluation is never an
unreported interaction channel. Run status reports training, evaluation, and
combined actual totals separately.

Smoke validates compilation, update, checkpoint, resume, metrics, and fixed
evaluation only. It cannot establish learnability. The CLI never auto-promotes
from smoke to pilot or formal training, and Phase U authorization does not
authorize Phase D.

## Fixed evaluation

`configs/phase_expert_smoke.json` will declare the PPO layout, reward weights,
episode horizon, checkpoint cadence, and fixed evaluation seed namespace. The
root seed deterministically derives a training namespace; fixed evaluation uses
an explicit seed list under a different namespace. Validation rejects any
collision. The evaluation protocol is hashed into every run.

Phase U evaluation always uses a fixed set of natural resets disjoint from the
training seed namespace. Phase D evaluation uses fixed validated seed IDs and
PRNG seeds disjoint from training sampling. Evaluation is deterministic and
reports component gates, event ticks, outcome counts, terminal reasons,
episode lengths, returns, and success rate. Evaluation results select no
threshold and cannot relabel a preliminary seed as reachable or safe.

## Checkpoint and exact resume

Every run writes periodic Orbax checkpoints and immutable policy bundles. The
checkpoint sidecar records cumulative effective transitions, optimizer state,
normalizer state, PRNG lineage, phase, reset protocol hash, reward/config hash,
XML/action/observation hashes, and parent checkpoint.

Exact resume requires both `--resume-run` and its matching
`--restore-checkpoint`. Validation rejects phase, XML, action mapping,
observation/history, thresholds, reward, seed protocol, PPO layout, or source
hash drift. A policy bundle alone may initialize a separately authorized new
experiment but cannot be called an exact resume.

Each Orbax directory has an immutable Gate C sidecar containing its recursive
file manifest/hash, saved step, cumulative transitions, contract hash, phase,
reset/reward/evaluation hashes, XML/action/observation/history hashes, PRNG
lineage, parent run/checkpoint, and the assertion that the checkpoint was
written by Brax's full-training-state checkpoint path. Exact resume validates
the sidecar and recursive checkpoint identity before passing the path to Brax;
it does not claim to reconstruct optimizer state from a policy bundle.

The cumulative transition total is parent completed transitions plus the new
effective budget. Parent artifacts remain immutable.

## Run artifacts

Before environment construction, the CLI creates the run directory and writes:

```text
run_manifest.json
status.json
metrics.jsonl
```

The complete run layout is:

```text
run_manifest.json
status.json
metrics.jsonl
resolved_config.json
budget_report.json
source_hashes.json
fixed_evaluation_protocol.json
fixed_evaluations/<checkpoint>.json
orbax/<step>/
policies/<step>/
final_report.json
```

`run_manifest.json` records purpose, phase, experiment level, authorization,
requested/effective total transitions, PPO layout, inputs, source hashes, seed
namespaces, reset/reward/evaluation protocols, maximum interaction cost,
stopping conditions, parent run/checkpoint, output path, and claim boundary.

`status.json` is atomically replaced and uses:

```text
initialized -> running -> completed
                       -> failed
                       -> gate_pause
```

It records completed transitions, last valid checkpoint, error type/message,
terminal decision, and next permitted action. `metrics.jsonl` is append-only;
each row includes timestamp, phase, checkpoint/effective transitions, PPO
metrics, phase reward components, reset counts, outcome counts, terminal reason
counts, and fixed-evaluation summary. Nonfinite values are rejected before
serialization.

## Source and artifact provenance

Source hashes include at least:

```text
authoritative XML
resolved project config
phase-expert training config
action and observation schemas
dvgc/env.py
dvgc/rewards.py
dvgc/two_phase_semantics.py
dvgc/two_phase_runtime.py
dvgc/training_budget.py
dvgc/runtime.py
dvgc/phase_expert_training.py
cli/train_phase_expert.py
seed bank and seed manifest when Phase D
parent run/checkpoint when resuming
```

The manifest must state `formal_tube_or_jel=false`. An expert checkpoint is a
phase-local policy proposal; it is not a Tube, feasibility model, unified
policy, safe controller, or JCE/JEL result.

Threshold loading is a host-only pre-run operation. It recomputes the canonical
manifest hash, validates all authoritative source hashes and paths, checks the
current XML/action/geometry identities, parses the selected threshold
dataclasses, and returns an immutable resolved object consumed downstream.
New threshold manifests use the repository `ACTION_MAPPING_VERSION` and record
the reference role as `kinematic_guideline_envelope`. Historical Gate B
manifests with obsolete action/controller terminology remain unchanged as
provenance and are not valid inputs to a new expert run.

## Gate sequence

The authorized sequence is:

```text
Gate C1: implement Phase U CLI and PPO smoke capability
Gate D1: separately authorize Phase U learnability pilot
Gate D2: separately authorize Phase U formal training
Gate E1: collect real frozen-pi_up Apex snapshots
Gate C2: enable Phase D CLI seed smoke
Gate D3: separately authorize Phase D pilot and formal training
```

The two experts need not begin formal training together. Phase D formal data
depends on a valid Phase U checkpoint and real `pi_up` snapshots.

## Gate C1 implementation boundary

Gate C1 may create the stable module, CLI, smoke config, and focused tests. It
may run only a single explicitly authorized Phase U smoke after implementation
review and authorization-manifest validation. It must not create `V_up`,
`V_down`, a Soft Tube, unified PPO, or any pilot/formal training process.

Before a future Gate C1 smoke can pass, tests and the bounded run must show:

- the CLI accepts only the two public phase names and only an authorized smoke;
- the Phase U adapter resets from audited natural starts and remains compatible
  with `jax.jit`, `jax.vmap`, and batched MJX state;
- early airborne alone cannot succeed and pre-window airborne/ascent progress
  reward remains zero;
- success requires the complete frozen Apex-band contract, while unchanged
  contact, roll, pitch, backward, platform-edge, and nonfinite failures remain
  terminal;
- the requested/effective transition report is aligned to one through four PPO
  rollout blocks and uses total environment transitions as its public unit;
- fixed evaluation, checkpoint, exact resume, source hashes, run manifest,
  `status.json`, and append-only `metrics.jsonl` satisfy their schemas;
- the run stops after smoke and performs no pilot, formal training, snapshot
  collection, feasibility learning, Soft Tube construction, or unified PPO.
