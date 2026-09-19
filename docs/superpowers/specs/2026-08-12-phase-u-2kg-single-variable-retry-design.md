# Phase U 2 kg Single-Variable Retry Design

## Decision

The authoritative DVGC payload changes from 4.0 kg to 2.0 kg. The existing
path `assets/orange_bike_4kg_horizontal.xml` remains the single authoritative
XML; its filename is historical after this change. Creating a second XML or
renaming the current path would introduce model selection and widespread path
churn that are outside this experiment.

The sole physical variable is the named `load` geom mass. Geometry, body and
joint layout, obstacle, meshes, initial state, hip/knee +/-50 N m force limits,
action mapping, control timing, reward, reset, observation, PPO network and
optimizer, episode horizon, two-phase thresholds, and safety termination
criteria remain fixed.

## Scientific question

The prior 4 kg Phase U run with hip initial action standard deviation 0.25
reached a Gate Pause at 755,200 training transitions. It changed from reaching
the legal jump window without liftoff to failing the pitch limit before the
window. This experiment tests one hypothesis only:

> Reducing payload mass to 2 kg while holding the controller and task contract
> fixed gives Phase U sufficient usable actuation authority to learn legal
> propulsion-ascent and reach the physical Apex transition band.

The experiment does not assume the hypothesis is true. The unchanged fixed
evaluation and Gate Pause conditions decide whether the retry continues.

## Model and provenance migration

The XML edit is exactly:

```text
geom name="load" mass="4.0" -> mass="2.0"
```

The new byte hash becomes the authoritative XML identity. All live static
contracts that claim the authoritative payload or hash must be updated. Current
model/build/runtime reports must be regenerated from the changed XML rather
than edited by hand where a stable generator exists.

Historical reports, run directories, checkpoint manifests, snapshot banks,
videos, and experiment JSON retain their 4 kg hash and wording as immutable
provenance. They are incompatible inputs under the 2 kg authority and must not
be rewritten, resumed, or silently promoted.

The configured XML path remains unchanged. Documentation must explicitly call
the `4kg` filename a retained historical path and identify the parsed load mass
and current SHA-256 as authoritative.

## TDD and validation flow

Before editing the XML, tests must be changed to demand:

- parsed `named_masses_kg["load"] == 2.0`;
- the new authoritative hash binds configuration and runtime validation;
- hip/knee force ranges remain exactly `-50 50`;
- geometry, obstacle dimensions, action order, and model path remain unchanged;
- runtime-gate model validation rejects a non-2 kg payload;
- current build/model reports identify 2 kg and the new hash;
- old checkpoint/bank hashes cannot pass current compatibility checks.

The test must first fail for the expected 4 kg value. The minimal implementation
then changes the XML and live contracts. Verification order is targeted model
tests, affected two-phase/runtime/provenance suites, compileall, full pytest,
`scripts/local_preflight.sh`, refreshed Gate B static artifacts, and a complete
64+32-transition runtime gate because the runtime fingerprint changed.

No PPO smoke or formal training starts until all preceding checks pass.

## Gate B refresh

The pure-JAX runtime geometry formulas are geometry-based and should remain
numerically unchanged, but all generated manifests are model-bound. A fresh
Gate B static refresh must regenerate or verify:

- XML/model identity and collision-geom manifest provenance;
- guideline threshold manifest under the new XML hash;
- natural-start audit;
- Phase U reset legality;
- timing-explicit snapshot/restore and three-frame history contracts;
- JAX/host MuJoCo clearance cross-audit.

The reference remains a kinematic guideline and weak prior. No reference action
replay is reintroduced, and no synthetic bank is called reachable or safe.

## Smoke and formal retry

The collision-qualified layout remains 512 parallel environments. The prior
1024 layout is not reused because it exceeded the immutable MJX Warp collision
candidate capacity.

One fresh smoke uses a single PPO rollout block:

```text
training transitions: 12,800
purpose: compile/reset/reward/update/checkpoint/fixed-eval/accounting only
```

It must have finite metrics and updates, a complete checkpoint, closed
interaction accounting, clean hash/timing/history contracts, no broadphase
overflow, and saved videos/state traces for every fixed-evaluation failure.
Smoke success does not imply learnability.

After a clean smoke, one fresh formal Phase U run is authorized up to the
largest aligned value below one million:

```text
requested user ceiling: 1,000,000 training transitions
effective aligned ceiling: 998,400 training transitions
effective checkpoints: 0, 102,400, 256,000, 512,000, 755,200, 998,400
```

The run starts from a fresh natural reset and fresh policy initialization. It
must not warm-start from any 4 kg checkpoint. Training, fixed evaluation,
candidate acquisition, and continuation probing retain separate ceilings and
accounting.

## Reward and success boundary

The current bounded Phase U reward is unchanged. Forward propulsion may be
earned before the legal window. Jump, ascent, clearance, and Apex task progress
remain zero until legal window activation. Early airborne is nonterminal
diagnostic telemetry: it receives no new penalty, does not count as legal
liftoff, and cannot grant Phase U or Apex success.

Roll, pitch, nonfinite, illegal contact, and other physical safety failures
remain active and unchanged. The experiment may not move the jump window,
lower Apex/recovery thresholds, or relax a physical failure limit to obtain a
pass.

## Sparse supervision

The training process runs detached, persistently, and resumably. Codex performs
one startup health check and then inspects only fixed checkpoint milestones,
terminal completion, or abnormal exit. It does not repeatedly read full logs.

Historical 512-env throughput suggests PPO computation is a few minutes, while
compilation, fixed evaluation, and failure-video rendering can extend total
wall time. The first terminal/milestone check should be scheduled on an
8--15-minute scale rather than minute polling. `status.json`, `metrics.jsonl`,
the process log, checkpoints, and a control/resume record are the wake-up
evidence surfaces. The persistent Codex goal carries the continuation across
interactions; the legacy watchdog remains disabled and is never reused.

## Autonomous follow-up discipline

If the 2 kg run completes or pauses without sufficient Apex coverage, Codex may
continue under the user's delegated supervision authority, but each new
experiment must:

1. audit fixed physical metrics, failure causes, action/state traces, and videos;
2. state one falsifiable hypothesis explaining the observed failure;
3. change only one experimental factor within the two-phase method;
4. add or adjust a failing contract test first when source/config changes;
5. pass targeted/full/static/runtime checks proportional to its fingerprint;
6. run a new bounded smoke;
7. issue a fresh run ID, seed, output directory, hashes, ceilings, and run-bound
   authorization before any new long run.

This delegated loop does not authorize XML mass changes beyond the fixed 2 kg,
geometry/obstacle/action-map changes, safety-limit relaxation, reference
tracking, five-stage restoration, formal Tube claims without labels, Phase D
training without real Apex sources, or unified PPO.

## Promotion and stopping

Candidate snapshot acquisition opens only after nonzero held-out Apex success,
at least eight independent successful online parent trajectories, and clean
contracts. Candidate states and checkpoint-dependent continuation diagnostics
remain provisional. A formal `pi_up_star`, `V_up`, or Soft Tube requires the
separate coverage, selection, relabeling, feature, calibration, and support
contracts already defined by the method.

The run pauses immediately for numerical failure, state/history corruption,
hash mismatch, collision truncation, accounting failure, reward hacking,
severe action saturation, repeated held-out degradation, or the implemented
three-window physical plateau. Low success at the first checkpoint alone is not
a pause condition.

## Deliverables

Each validated iteration records and commits only source/config/documentation:

- branch and producer HEAD;
- authoritative 2 kg XML hash and parsed model contract;
- changed files and tests;
- Gate B/runtime/smoke evidence;
- run ID, PID, status/metrics/log/control paths, budget, and checkpoints;
- checkpoint physical metrics and failure videos;
- training/evaluation/acquisition/continuation interaction counts;
- whether `pi_up_star`, candidate snapshots, continuation data, provisional or
  formal `V_up`, Tube, and real Phase-D seeds exist;
- next single permitted hypothesis or method gate.

Run outputs remain ignored and are not committed. `.vscode/` remains untouched.
