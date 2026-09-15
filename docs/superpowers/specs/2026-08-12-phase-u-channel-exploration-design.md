# Phase U Channel-Specific Exploration Design

## Decision

The next Phase U iteration changes one scientific hypothesis only: replace the
scalar Phase U initial action standard deviation `0.25` with the channel-wise
vector:

```text
[steer=0.05, rear-wheel drive=0.05, hip=0.50, knee=0.05]
```

The default runtime and all non-Phase-U callers retain the scalar `0.05`
prior. Reward, reset, optimizer, network layers, observation, episode horizon,
XML, actuator limits, action mapping, thresholds, and fixed evaluation seeds
remain unchanged.

## Evidence and root cause

The scalar-0.25 formal run stopped at 256,000 transitions with three held-out
evaluations at 0, 102,400, and 256,000. Every held-out rollout reached the
jump window, but none achieved liftoff, clearance, or Apex success. Meanwhile,
stochastic training episodes had an 81%--97% physical-failure rate and zero
success. Their illegal-contact penalty remained material even as scalar return
improved. This shows that exploration was widened, but most additional samples
were destructive rather than useful.

The earlier natural-start action diagnostic isolated the useful direction:
after legal window entry, a constant normalized hip action of at least `0.5`
produced liftoff in 10/10 tested cases within one or two control ticks. The
256,000 held-out deterministic policy reached only about `0.183` normalized
hip action. Applying `0.25` to steering, drive, hip, and knee together therefore
spends most exploration mass on channels that were not implicated by the
liftoff diagnostic.

## Rejected alternatives

1. **Increase the scalar standard deviation again.** This increases hip
   coverage but also increases steering, drive, and knee noise after the
   scalar-0.25 run already showed predominantly physical failures.
2. **Change reward, reset, deadline, or event thresholds.** That would combine
   a task-contract hypothesis with the exploration hypothesis. Current
   evidence establishes a narrower action-channel problem first.
3. **Add a hip-action bonus or reference-action tracking.** This would reward a
   controller command rather than physical progress and would violate the
   approved reference-as-weak-prior contract.

## Runtime contract

`dvgc.runtime` accepts either:

- one finite scalar standard deviation, broadcast to every action channel; or
- one finite vector whose length exactly equals `action_size`.

Every value must satisfy `0.001 < std < 1.0`. Boolean values, empty vectors,
wrong-length vectors, nested vectors, NaN, and infinity are rejected. The
initialized distribution location remains exactly zero. Checkpoint network
metadata and PPO training must receive the identical resolved vector.

`dvgc.phase_expert_training` resolves the explicit Phase U configuration value
to an immutable tuple. Run manifests record the ordered vector using the
authoritative action order `[steer, rear-wheel drive, hip, knee]`.

## Budget and gates

Previous formal Phase U invocations consumed 803,200 expert-training
transitions in total. The remaining aligned budget is therefore 192,000, which
would bring the program total to 995,200 under the authorized 1,000,000 ceiling.

The stable 512-environment PPO block remains 12,800 transitions. The new formal
schedule is requested at `0/100k/192k` and resolves to
`0/102,400/192,000`. Fixed evaluation is capped at 4,800 transitions;
candidate acquisition and continuation diagnostics are each independently
capped at 38,400 transitions. A one-block smoke precedes formal execution and
does not count as expert training.

Formal execution pauses on all existing numerical, contract, accounting,
physical-degradation, saturation, and three-window plateau gates. Snapshot
acquisition still requires real held-out Apex success and independent parent
coverage. No result from this iteration may be called `pi_up_star`, formal
`V_up`, or a Soft Tube without the later selection and re-labeling gates.

## Verification

Red-green tests cover scalar backward compatibility, exact vector scale,
invalid vector rejection, configuration resolution, PPO forwarding,
checkpoint forwarding, manifest serialization, and remaining-budget
accounting. Final verification requires compileall, targeted tests, full
pytest, `scripts/local_preflight.sh`, and a fresh full runtime gate because the
network factory fingerprint changes.

The 512-environment smoke must complete one finite PPO update, write a
checkpoint, close interaction accounting, render every unsuccessful fixed
evaluation, and show no broadphase overflow, NaN/Inf, OOM, traceback, history
violation, or hash mismatch. Formal training is launched persistently only if
that smoke passes engineering integrity.
