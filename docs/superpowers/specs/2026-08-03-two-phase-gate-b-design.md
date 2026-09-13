# Revised Two-Phase Gate B Runtime Foundation Design

## Status and revision boundary

This document is the current Gate B contract. It supersedes the earlier
requirement that `data/reference_jump.csv` replay a complete open-loop jump on
the authoritative 4 kg, +/-50 N m model. The failed prelaunch and roll-limit
replays remain immutable provenance, but they are not blockers for two-phase
expert training and must not be repaired by changing actions, thresholds,
windows, XML, payload, force limits, or posture limits.

Gate B validates the runtime and state-foundation needed by the two experts. It
does not train an expert, create an authoritative Phase U or Phase D bank,
label continuation, train `V_up`/`V_down`, construct a learned soft Tube, train
unified PPO, or support a JCE/JEL claim.

## Reference classification

`data/reference_jump.csv` is formally classified as:

```text
kinematic guideline and weak prior
```

It may provide:

- broad jump-space intervals;
- Apex, descent, and recovery kinematic envelopes;
- hip/knee motion trends;
- initial threshold suggestions;
- physical state seed proposals;
- weak reward and evaluation priors.

It is not an expert policy, `pi_up`, `pi_down`, a trained policy, an
authoritative dynamic controller, a reachability proof, or a safety proof.
Its states are not presumed reachable merely because they occur in the CSV.
Complete open-loop action replay on the current model is neither required nor
used as Gate B evidence.

Reference time/index and action columns are provenance-only inputs. Online
two-phase events, expert rewards, expert success, and expert actions must not
depend on following a reference time, index, pose, or action sequence.

## Gate B acceptance contract

Gate B passes when all of the following are validated:

1. the external pure-JAX two-phase runtime adapter is correct under scalar,
   `jax.jit`, `jax.vmap`, and batched MJX use;
2. every collision-relevant robot geom has a supported analytic formula and
   representative host MuJoCo cross-audits agree within the declared tolerance;
3. the Apex/Recovery threshold manifest is deterministic and reproducible from
   the immutable model, declared kinematic envelope, fixed margins, and source
   hashes;
4. the natural physical start is finite, nonpenetrating, legally supported,
   nonterminal, and compatible with the actor/history timing contract;
5. the Phase U reset protocol uses those real natural starts and does not
   require a guideline action or guideline-derived bank;
6. a physically validated Phase D seed protocol is specified with explicit
   rejection gates and claim restrictions;
7. timing-explicit three-frame snapshot construction, validation, formal
   restore, and deterministic round-trip are implemented and tested for any
   snapshot admitted by a later seed/rollout collector;
8. all documents and manifests classify the reference only as a kinematic
   guideline and weak prior.

Gate B does not require the guideline replay to emit the historical ten-event
sequence, enter the Apex band, reach stable recovery, or create nonempty Phase
U/Phase D banks. A missing expert bank is expected at this point.

## Runtime architecture

The formal signal path remains:

```text
state.data/state.info + immutable XML geometry
  -> pure JAX dvgc.two_phase_runtime adapter
  -> ApexBandSignals / RecoverySignals / TwoPhaseEventState
```

The adapter remains external to `OrangeBikeDVGC`. It does not use reward,
legacy oracle phase, matcher distance, reference time/index, reference action,
or outcome labels. Host `mj_geomDistance` remains representative cross-audit
only and cannot enter training, online events, or bulk state construction.

`full_structure_clearance` and `robot_frontmost_x` continue to cover every
collision-relevant robot geom. The fixed relative-position convention remains:

```text
obstacle_relative_x = obstacle_front_x - robot_frontmost_x
> 0: robot front has not reached the obstacle front
= 0: robot front aligns with the obstacle front
< 0: robot front has passed the obstacle front
```

## Threshold manifest

The guideline may propose the initial kinematic envelope used to select Apex
and Recovery thresholds. The manifest must still record immutable XML/config/
reference hashes, geometry coverage, definitions and units, fixed extraction
slices, raw extrema, fixed engineering margins, selected thresholds, source
category, and a canonical hash.

Thresholds may later be revised only through a separately reviewed physical
contract change. They may not be tuned until reference actions replay
successfully, and the historical roll-limit result cannot justify weakening
roll, pitch, clearance, Apex, contact, or stable-recovery requirements.

## Natural-start and Phase U reset contract

The Phase U reset source is the environment's real natural reset under the
authoritative XML. Before Gate C1 smoke, a fixed reset audit must show:

```text
finite qpos/qvel/ctrl and observations
legal wheel/ground support
no chassis, payload, or other prohibited terrain contact
no material penetration beyond the declared tolerance
roll/pitch inside existing hard limits
nonterminal and nontruncated initial state
neutral last action and valid three-frame actor history initialization
source phase = propulsion_ascent
```

Natural-start randomization, if enabled for a declared training level, comes
from the fixed config and seed namespace. It must never be derived by replaying
reference actions or selecting a convenient reference row.

## Physically validated Phase D seed protocol

Phase D does not reset from the natural ground start. Two seed tiers are
defined.

### Preliminary seeds

A reference Apex/descending state may be used only as a kinematic proposal.
Every candidate must pass, in order:

1. deterministic mapping into the immutable model with recorded source row and
   coordinate convention;
2. MuJoCo forward computation;
3. finite qpos, qvel, ctrl, sensor, actor, and privileged observations;
4. no material penetration and complete legal-geometry checks;
5. no prohibited chassis/payload contact and no active physical failure;
6. a fixed short-horizon dynamic validation using declared seed, actions,
   control ticks, and stopping conditions, without offset/action search;
7. real consecutive control ticks that construct `t-2 -> t-1 -> t` history,
   last action, ctrl timing, actor observation, and snapshot timing fields;
8. `validate_snapshot_v4`, `validate_phase_snapshot`, and formal timing-explicit
   restore/round-trip validation.

Accepted candidates are labeled only:

```text
physically_validated_descent_seed
```

They must not be labeled `reachable`, `expert snapshot`, `Tube`, `safe`, or
certified. Preliminary seeds are limited to Phase D engineering smoke and a
separately authorized early pilot; they are not the formal Phase D training
distribution.

### Formal seeds

After a valid frozen Phase U checkpoint exists, the primary formal Phase D
source is:

```text
frozen pi_up rollout
  -> Apex pre/nearest/post
  -> early descent
  -> real online timing-explicit snapshots
```

The collector must preserve policy/XML/config hashes, seed namespace,
trajectory lineage, event ticks, actions, history, terminal state, and outcome.
Reference proximity cannot substitute for a real `pi_up` rollout.

## Timing-explicit snapshot contract

The top-level v4 legacy phase remains only for old restore compatibility. The
formal phase identity is exclusively:

```text
two_phase_context.source_phase = propulsion_ascent | descent_recovery
```

History must come from consecutive real control ticks. Copying the current
frame, independently reconstructing CSV observations, or mixing post-update
history with the current frame is forbidden. Restore uses only
`timing_explicit_independent_reconstruction`; compatibility fallback cannot
support Gate B or expert reset admission.

Round-trip compares qpos, qvel, ctrl, last action, actor and privileged
observations, three-frame history, contact/event state, termination/truncation,
and two-phase signals under the same snapshot, PRNG seed, actions, and number
of control ticks.

## Historical replay evidence

The existing prelaunch and roll-limit reports, MP4 files, NPZ traces, manifests,
and producer provenance remain retained evidence that full reference-action
compatibility with the current model was not demonstrated. They must not be
deleted, rewritten as success, or used to tune the physical contract.

`cli.build_two_phase_guideline_banks` and the historical guideline-bank output
are migration/diagnostic artifacts. They are no longer the mandatory Gate B
entrypoint and must not be invoked again merely to make reference replay pass.
No existing output is relabeled as an authoritative expert bank.

## Gate B outcome and next boundary

Under this revised scope, the recorded reference `roll_limit` is not a Gate B
failure. The runtime, geometry, threshold, natural-start, seed-protocol, and
timing contracts release Gate C1 design and implementation work. This is not a
claim that `pi_up`, `pi_down`, Phase D seed banks, or expert training already
exist.

The next executable gate is Gate C1: implement the unified phase-expert CLI and
Phase U PPO smoke capability. Starting smoke, pilot, or formal training still
requires the authorization stated in the current experiment ledger; this
design revision itself runs zero PPO and zero MuJoCo rollouts.
