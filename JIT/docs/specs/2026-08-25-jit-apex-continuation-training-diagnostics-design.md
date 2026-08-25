# JIT Apex Continuation and Training Diagnostics Design

## Status and authorization

The user approved this design direction on 2026-08-25 and authorized one new
fresh approximately five-million-transition run after implementation,
verification, a JIT-only Git commit, and push. The initial launch must not load
the previous `phase_u_absolute_4988928_seed820201_20260825` checkpoint or any
other checkpoint.

## Problem evidence

The previous v3 natural panel ended after two transitions because
`geometric_penetration_signals` classified wheel penetration deeper than
`0.01 m` as `illegal_wheel_contact`. At the terminal state, the analytic rear
wheel clearance was `-0.014175 m`, and Host MuJoCo reported the explicitly
allowed XML pair `floor`/`rearwheel_collision` at the same distance. Wheel-floor
contact is required for support and propulsion; this heuristic therefore
misclassified normal compliant contact as a terminal failure and reward
penalty.

The forced-airborne RSI panel also ended after two transitions for a separate
reason: first Apex set `terminated=true`. This prevented inspection of the
post-Apex trajectory and made an event-chain success look like a complete
flight result.

Formal PPO disabled Brax episode logging to preserve ordered policy callbacks,
so `metrics.jsonl` contains KL/loss/distribution values but no mean completed-
episode reward or length. A scientifically useful training plot cannot be
reconstructed from that file alone.

## Active v4 behavior

### Wheel/structure contact

Active v4 does not treat wheel contact or wheel penetration depth as illegal,
terminal, or penalized. The XML physics continues to resolve wheel-floor and
wheel-step contacts normally. Raw per-wheel terrain clearances and maximum
wheel penetration depth are diagnostic metrics only.

Nonfinite state, roll limit, pitch limit, prohibited non-wheel structure
contact, and backward exit remain physical failures. This change does not edit
the XML, contact pairs, solver, collision geometry, or friction.

Historical v1-v3 run evidence remains verifier-readable. No historical
checkpoint is a v4 input.

### Apex continuation

The first valid Apex remains:

```text
jump window visited
and prior ascent >= +0.05 m/s
and root height >= 0.5 m
and current vertical velocity <= -0.05 m/s
and no physical failure on that transition
```

First Apex sets the monotonic `event/apex_seen` flag and pays the existing
one-time `+50` bonus. It does not terminate or truncate the episode. The
environment continues until a retained physical failure or the 200-control-
tick horizon. `terminal/success` is not used to disguise an Apex event as an
episode terminal; evaluation counts Apex from `event/apex_seen`.

An episode that reaches Apex and later physically fails is reported as
`apex_seen=true, post_apex_physical_failure=true`. An episode that reaches Apex
and reaches the horizon without physical failure is reported separately as
post-Apex survival. This preserves both local Phase U evidence and what happens
after the handoff.

### Before/after Apex data

Every fixed evaluation episode retains its complete NPZ/JSON trace. In
addition, the final natural and forced-RSI representative artifacts contain:

- `apex_frame_index` and `apex_time_seconds` (`-1` when Apex is absent);
- a boolean `segment_pre_apex` mask, including the reset state through the
  first Apex frame;
- a boolean `segment_post_apex` mask, starting at the shared Apex boundary
  frame and continuing to the final frame;
- `representative_pre_apex.npz` and `representative_post_apex.npz` slices;
- segment metadata with state and transition counts;
- a full MP4 and dashboard with a visible Apex boundary.

The Apex boundary state is intentionally present in both segment files so the
handoff state is independently inspectable. Transition accounting counts that
state zero times; `pre_transitions + post_transitions = full_transitions`.
When Apex is absent, pre-Apex contains the full trace and post-Apex contains an
empty, shape-valid payload.

Panel summaries add Apex-frame, pre-Apex transitions, post-Apex transitions,
post-Apex physical-failure rate, and post-Apex horizon-survival rate. Natural
and forced-RSI results remain separate; only natural results can support Phase
U promotion.

## Training metrics and plot

Brax training episode logging is enabled for v4, but its asynchronous in-epoch
callback is routed to a dedicated method that does not depend on the ordered
policy-parameter callback. The existing block-level `metrics.jsonl` path
continues to receive checkpoint-aligned PPO metrics only.

At each 24,576-transition block, `episode_metrics.jsonl` records the rolling
mean over Brax's last 100 completed episodes, including:

- `episode/sum_reward` as mean episode reward;
- `episode/length` as mean episode frames/transitions;
- reset-source and terminal/event episode metrics already exported by the
  environment;
- the callback's exact absolute training transition.

After training completes, JIT writes:

- `training_curves.png` with step-aligned mean reward, mean episode length, KL,
  policy/value/total loss, policy mean standard deviation, and SPS;
- `training_curves.npz` with every plotted numeric series;
- `training_curves.json` with sample counts, semantic labels, paths, and
  SHA-256 hashes.

The chart labels episode values as rolling completed-episode means; it does not
claim they are an average over incomplete environments or a natural-only
metric. The 5% RSI mixture remains present in training, so reset-source episode
statistics must appear alongside the reward/length plot.

## Configuration and identity

New exact v4 smoke/formal configurations isolate this method from v3:

- schema: `jit_phase_u_*_v4`;
- action mapping: unchanged v3 keyframe-centered absolute mapping;
- observations: unchanged `(76,)/(106,)` Actor/critic dimensions;
- reward coefficients: unchanged except wheel contact no longer feeds
  `illegal_contact`;
- RSI: unchanged 5% bounded training mixture;
- episode horizon: 200 control ticks;
- formal target: 4,988,928 training transitions;
- block: 384 environments x 64 ticks = 24,576 transitions;
- minibatch/update layout: 16 x 64 = 1,024 samples, 24 minibatches, 8 passes;
- learning rate/entropy/gamma/GAE/clip/gradient norm:
  `1e-4/0.01/0.99/0.95/0.2/0.5`;
- seed: 820301;
- held-out seeds: 940001 through 940008.

The new config hash, schema, event/terminal contract, and fresh manifest make
v1-v3 checkpoints incompatible. The launch command contains no restore option,
and the manifest must state a null parent checkpoint, transition-zero start,
and fresh resume semantics.

## Formal evidence and provenance

The v4 completed-run verifier requires:

- exact training/checkpoint/evaluation/diagnostic interaction ledgers;
- all full episode NPZ arrays and reset-source semantics;
- final natural and forced-RSI MP4/PNG/full NPZ hashes and decoded frame counts;
- pre/post-Apex segment NPZ paths, distinct names, hashes, shapes, boundary
  equality, and transition-count conservation;
- training metrics JSONL monotonic steps;
- training-curve PNG/NPZ/JSON types, hashes, finite arrays, aligned step axes,
  and required reward/length/KL/loss/std series;
- final checkpoint restore and finite inference.

The verifier continues to accept retained completed v1-v3 evidence without
retroactively requiring v4 artifacts.

## Validation and launch gate

Implementation is test-driven. Focused semantics/contact, evaluation/segment,
training-metrics, formal-runner, and provenance tests must pass, followed by
the complete non-GPU suite, complete GPU suite, and `JIT/scripts/local_preflight.sh`.
Independent review must have no Critical or Important findings.

Only then may one explicit JIT-only source/config/test/doc commit be created and
pushed. The fresh v4 run is launched persistently after remote equality is
confirmed. Startup is inspected once for a transition-zero checkpoint and
fresh manifest; no continuous supervision is required. Based on the previous
47k transitions/s and longer fixed panels/artifacts, the expected completion
window is approximately 3-6 minutes after successful compilation/startup.

## Claim boundary

Completing the budget, generating long videos, or observing forced-RSI Apex
does not create a trained expert. Promotion remains dependent on frozen natural
panels with legal window access, ascent, Apex, and acceptable post-Apex physical
behavior. This work does not implement Phase D, feasibility models, learned
soft Tubes, unified PPO, or JCE/JEL certification.
