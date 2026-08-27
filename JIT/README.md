# JIT Phase U training stack

## Current Plan B decision — 2026-08-27

The latest verified run is:

```text
JIT/runs/phase_u/phase_u_v4_pitch15penalty_9977856_seed820901_20260826
```

Strict `verify-run` exits 0. Training completed exactly `9,977,856`
transitions, and the checkpoint was restored successfully. All eight held-out-
seed natural-reset rollouts reach Apex; they later end at `pitch_limit`. Mean
post-Apex continuation is `182.125` environment transitions. These are not
eight independent initial conditions: the natural reset states are very
similar.

`transition_9977856` is the main `pi_up_candidate`, not final `pi_up_star`.
Phase U training stops. The next work is freezing candidates and designing the
handoff snapshot bank. Phase D, `pi_down`, continuation labels, `V_up`/`V_down`,
learned soft Tubes, unified PPO, and final JCE/JEL remain unimplemented.

The execution guide is
[`JIT/docs/plans/2026-08-27-jit-plan-b-execution-guide.md`](docs/plans/2026-08-27-jit-plan-b-execution-guide.md).
Everything below is retained v4/v3/v2 history; it does not override this
current decision.

## Historical v4 Apex-continuation run contract

`phase_u_continuation_10m.json` was the retained fresh-training contract. Ordinary
front/rear wheel contact with the terrain is support telemetry. Prohibited
chassis/body contact is also telemetry only: it neither terminates nor adds an
illegal-contact reward penalty. Roll, pitch, backward motion, and numerical
failure remain physical terminals. Two task
failures prevent the policy from exploiting the approach: while the jump
signal is active and the 0.5 m height event has not occurred, forward speed at
or below 0.3 m/s is immediately `stuck`; the active jump window ends at 3.9 m;
world-frame root yaw beyond 45 degrees is
`yaw_limit`. Each has its own `-100` reward component and end code and is not
double-counted as a generic physical failure. On either terminal, the final
step reward offsets the reward already accumulated so the complete episode
return is exactly `-100`. The yaw comes from the root free-joint quaternion,
not the steering joint. The first Apex event still pays its one-time configured
bonus, but it no longer ends the episode; simulation continues until a retained
terminal or the exact 400-control-tick horizon (8 seconds).

The v4 airborne RSI is a low-height bridge rather than an already-completed
jump: root height is sampled from `[0.38, 0.45] m` with upward velocity
`[3.0, 3.6] m/s`. Steering adds `-2.0 * (steer_t-steer_(t-1))^2` and
`-0.5 * |steer_t|^1.5`; rear-wheel drive adds
`-2.0 * (drive_t-drive_(t-1))^2` on top of the retained all-action costs. A
full-scale steering or rear-wheel reversal therefore costs about `-8`, while
small corrections remain available.

While the current one-shot jump signal is active below 0.35 m, v4 adds a dense
low-height cost. It is `-8` at and below 0.15 m, decreases linearly to zero at
0.35 m, and is zero whenever the jump signal is off. This makes remaining low
in the 3.9 m jump window worse than the retained survival/posture reward.

Training uses full environment resets after every completed episode. This is a
required runtime contract, not an optimization option: cached data/observation-
only resets would leak `episode_step`, event, timeout, and RSI state into the
next episode. The v4 config hash explicitly binds `full_reset=true`, making the
aborted pre-fix checkpoint incompatible with the corrected run. It also binds
`preserve_episode_evidence=true`: terminal `episode_done` and accumulated
episode metrics cross the reset boundary solely for Brax logging, while JIT
physics/event state comes from the fresh reset.

Final natural and forced-RSI representatives encode the complete episode in
MP4/PNG/NPZ. They additionally save independently hashed
`representative_pre_apex.npz` and `representative_post_apex.npz` files with the
first Apex state shared as the segment boundary. A no-Apex rollout has a full
pre-Apex file and an empty, shape-valid post-Apex file.

Formal v4 training writes raw block PPO metrics and rolling last-100 completed-
episode means, then produces synchronized `training_curves.png`,
`training_curves.npz`, and `training_curves.json`. The curves include mean
episode reward/length, airborne-RSI fraction, KL, policy/value/total loss,
policy distribution standard deviation, and steps per second.

## Historical v4 one-shot post-run analysis watcher

After launching one declared training run, start its independent local watcher
with the same exact run directory, PID file, and launch log:

```bash
setsid nohup /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/watch_training_and_analyze.py \
  --run-dir JIT/runs/phase_u/<run-id> \
  --pid-file JIT/runs/phase_u/<run-id>.pid \
  --launch-log JIT/runs/phase_u/<run-id>.launch.log \
  --poll-seconds 30 \
  > JIT/runs/phase_u/<run-id>.watcher.log 2>&1 < /dev/null &
```

The watcher only reads local process/status files while the run is active. A
missing or dead training PID does not authorize analysis: the watcher waits
until `status.json` says `completed`, `engineering_error`, or `aborted`. It then
atomically creates `codex_analysis.started.json` and makes at most one
ephemeral `codex exec` call using `gpt-5.6-luna` with an explicit read-only
sandbox. Restarts never
repeat that call, including after a failed attempt.

When Codex produces a final message, the CLI writes it to `AUTO_ANALYSIS.md` in
the ignored run directory. Whether the attempt succeeds, fails, or times out,
captured stdout/stderr and the return code remain there as `codex_exec.log` and
`codex_analysis.completed.json`. For a completed training run, the analysis
prompt also requires strict `verify-run` provenance checking before
interpreting the natural-start and airborne-RSI evidence panels.

The retained v4 target was 15,015,936 transitions (611 blocks), under the fresh seed
namespace `820801` with frozen held-out seeds `990001..990008`.
The first run must have a null parent checkpoint, start at transition zero, and
omit `--restore-checkpoint`.

## Historical v3 absolute-joint 5M experiment

The completed experiment used `phase_u_absolute_5m.json` and started from newly
initialized PPO parameters. It never restored a v1/v2 checkpoint. Hip and knee
share one keyframe-centered absolute-target rule: action zero commands the
XML keyframe angle, while negative/positive actions interpolate to that joint's
lower/upper limit. Consequently hip maps `[-1, 0, 1]` to
`[-1.3, -1.2, 0.5]` radians and knee maps it to `[-1.5, 2.5, 2.5]` radians.
Steering and rear-wheel mappings are unchanged.

The exact target is 4,988,928 transitions: 203 aligned PPO blocks with 384
parallel environments, 64-step unrolls, 16 chunks per minibatch, 24
minibatches, and 8 optimizer passes. The fixed learning rate is `1e-4`;
entropy is `0.01`; gamma/GAE/clip/max-gradient are `0.99/0.95/0.2/0.5`.
The first-block KL includes observation-normalizer warm-up and is not treated
as a pure policy-shift metric; later blocks use an established normalizer.

At each declared milestone, natural-reset held-out evaluation and forced
airborne-RSI diagnostics are stored separately. Only natural-reset panels can
support promotion. Both routes save complete NPZ traces; their final
representatives also save MP4 and aligned reward/state PNG diagnostics. RSI
interactions are recorded only in the diagnostic ledger.

The retained launch command had deliberately no `--restore-checkpoint`:

```bash
nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_absolute_5m.json \
  --run-id phase_u_absolute_4988928_seed820201_20260825 \
  --formal \
  > JIT/runs/phase_u/phase_u_absolute_4988928_seed820201_20260825.launch.log 2>&1 \
  < /dev/null &
```

The run completed 4,988,928 training transitions and passed strict provenance,
but it is `NO_PROMOTION`. Every natural-start panel had 0/8 Apex and 8/8
physical failures; the final policy caused illegal wheel contact after two
control steps. Every forced-airborne RSI panel had 8/8 Apex, but those resets
already supplied height and upward velocity and do not count as natural jump
success. Do not resume or promote the final checkpoint. See the complete
analysis in
`docs/experiments/phase_u_absolute_4988928_seed820201_20260825/REPORT.md`.

Everything below documents retained v1/v2 behavior and historical evidence;
those checkpoints are audit artifacts, not inputs to the active v3 run.

`JIT` is an independent implementation of the first Propulsion-Ascent
engineering delivery described in the repository rebuild guide. It does not
import the existing `dvgc` package and does not copy the authoritative XML.

The retained v2 scope introduced the target-free reference reward, a one-shot
root-x jump signal, 5% bounded airborne RSI for training resets, natural-only
held-out evaluation, height/descent Apex termination, and synchronized
numeric/PNG/video diagnostics. It also retains environment/runtime integrity,
the aligned 25,600-transition PPO engineering-smoke entrypoint, and an
auditable formal-only Phase U runner. It does not implement Phase D, continuation labels,
`V_up`/`V_down`, learned soft Tubes, unified PPO, or JCE/JEL certification.

The v2 Actor input is `3 x 25 + 1 = 76`: three real sensor-history frames plus
one current `jump_signal`. The critic receives that complete 76-value input
plus 30 privileged values, for 106 total. The signal is therefore available to
both networks and is not repeated in the history. v1 checkpoints (`81/114`)
are deliberately incompatible and must not be resumed into v2.

One fresh v2 formal run completed on 2026-08-25 with 998,400 training
transitions. All five frozen natural-reset panels had zero Apex/height/ascent
events and 100% roll-limit failures, so the result is `NO_PROMOTION` and no
checkpoint is a trained expert. The retained 2026-08-24 smoke/formal artifacts
remain historical v1 evidence only.

The full v2 experiment analysis is in
`docs/experiments/phase_u_reward_rsi_diagnostics_v2_20260825/REPORT.md`. The
ignored final evidence is under
`JIT/runs/phase_u/phase_u_v2_formal_998400_seed820101_20260825/`; do not reuse
that run ID or extend its final checkpoint.

Use the retained interpreter directly:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests -q
```

Generated run evidence belongs under `JIT/runs/` and is ignored by Git.
Every new representative v2 video also produces a full-trajectory diagnostic
PNG, an aligned compressed NPZ, and SHA-256 fields in its JSON report.

Run the complete local verification without launching training:

```bash
bash JIT/scripts/local_preflight.sh
```

The explicitly bounded smoke command remains available:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_smoke.json \
  --run-id <unique-run-id> \
  --smoke
```

## Historical formal Phase U training

Formal mode is exactly 39 aligned blocks, or 998,400 training transitions,
with seed `820101`. Identity-bound checkpoints are written at transitions 0,
102,400, 256,000, 512,000, 742,400, and 998,400. The five nonzero milestones
each run deterministic evaluation on held-out seeds 920001 through 920008.
Brax evaluation is disabled and fixed evaluation is accounted separately.

Source verification, a focused JIT-only commit, and its GitHub push are hard
predecessors of this persistent launch:

```bash
mkdir -p JIT/runs/phase_u
nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_formal.json \
  --run-id phase_u_formal_998400_seed820101_20260824_retry1 \
  --formal \
  > JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824_retry1.launch.log 2>&1 \
  < /dev/null &
JIT_FORMAL_PID=$!
printf '%s\n' "${JIT_FORMAL_PID}" \
  > JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824_retry1.pid
```

Inspect startup once, then only the declared milestones, completion, or an
abnormal exit. A high but finite KL is evidence to inspect, not permission to
change rewards or PPO hyperparameters during the run.

If an abnormal exit requires recovery, `--restore-checkpoint PATH` starts a
new run segment from the saved observation normalizer, Actor, and critic.
Brax resets optimizer state and PPO RNG, so this is explicitly a parameter
warm start and never a bit-exact continuation.

Finishing the transition budget does not by itself establish a trained expert.
Promotion requires multiple legal, low-rotation Apex successes across the
frozen held-out seeds. It never establishes a safe Tube, JCE, or JEL.
