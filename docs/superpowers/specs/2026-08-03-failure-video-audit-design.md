# Dynamic Failure Video Audit Design

## Purpose

Every bounded dynamic DVGC task that terminates in a physical failure or a
Gate pause must preserve a human-viewable video alongside its machine-readable
evidence. The first application covers the two Gate B guideline failures:

1. the full guideline trace ending in `prelaunch_airborne`;
2. the fixed launch-history trace entering the jump window after wheel support
   has already been lost.

Video is audit output only. It must not feed training, event extraction,
threshold selection, bank admission, reward, termination, or observation.

## Stable Interface

Add one stable CLI, `cli/render_two_phase_failures.py`, backed by focused helper
functions in `dvgc/failure_video.py`. It consumes a fully specified failure
scenario, replays the exact seed and reference actions through the unchanged
environment, captures every MJX state, and renders those captured qpos/qvel/ctrl
values through host MuJoCo.

Supported Gate B scenarios are named, not inferred:

```text
full_guideline_prelaunch_airborne
launch_history_airborne_before_window
```

The CLI writes beneath an ignored run directory:

```text
full_guideline_prelaunch_airborne.mp4
full_guideline_prelaunch_airborne.states.npz
launch_history_airborne_before_window.mp4
launch_history_airborne_before_window.states.npz
failure_video_manifest.json
```

## Exact Reproduction Contract

Each scenario records and validates:

```text
XML/config/reference SHA-256
scenario name
PRNG seed
initial reference index
reference rows per control tick
action reference index per tick
qpos/qvel/ctrl per rendered tick
phase and deployable event latches
wheel/body support audit
termination/truncation and end_code
first event ticks
environment transitions
formal training transitions = 0
```

The initial vertical mapping and support placement are identical to the Gate B
event code. The full trace starts at reference index 0 and applies actions
`0 -> 10 -> 20 ...`. The launch-history trace starts at the already-fixed
first Phase U history origin, three control ticks before launch-front index
113, which is reference index 83: its initial `ctrl` and `last_action` are from
index 73, followed by step actions `83 -> 93 -> 103`. Neither scenario searches
for a favorable start, action, threshold, or seed.

## Rendering and Overlay

Host MuJoCo renders only captured states; it never advances the authoritative
rollout. MP4 encoding uses the repository's existing `mediapy` H.264 path.
Frames show a side view with a camera tracking the robot and overlays containing:

```text
scenario and tick
reference/action index
x, z, vx, vz
jump-window bounds and inside/outside state
host wheel/body contact and deployable wheel-support estimate
jump signal and two-phase event state
phase, end_code, and termination reason
```

The initial and terminal frames are held briefly and playback is slowed enough
to make the premature liftoff visible. Every video and compressed state trace
receives a file SHA-256 entry in the manifest. The manifest reloads each NPZ,
recomputes the ordered qpos/qvel/ctrl trace digest, derives first-event ticks
from telemetry, and validates frame, transition, action-schedule, and named
failure-end-state accounting before reporting `status=pass`.

## Automatic Failure Archive

`cli.build_two_phase_guideline_banks` will invoke the same renderer after it has
written a failing `guideline_event_report.json`. Video creation happens before
the CLI raises its existing Gate-pause exception. The original Gate-pause
status and exception remain authoritative. If rendering itself fails, the CLI
records `video_status=render_failed` and the rendering error, then still exits
for the original physical failure; it must never convert failure into success.

Host contact and deployable support are separate audit fields. In particular,
the launch-history failure may retain one MuJoCo wheel contact while the
deployment estimator is false; the overlay must never describe these as the
same signal.

Future dynamic CLIs may call the same explicit scenario interface. This design
does not add a generic environment hook or write rendering logic into
`env.step`.

## Tests and Verification

Red-green tests require:

- exact scenario seeds, start indices, and action-index schedule;
- captured traces reproduce the expected Gate B end states;
- both MP4 files exist, are nonempty, and have manifest hashes;
- telemetry contains the window/support/event/termination fields;
- renderer consumes captured states without calling `env.step`;
- Gate B failure invokes video archiving before raising;
- video failure cannot mask or weaken the original Gate pause;
- formal training transitions remain 0.

Verification uses the configured Python, targeted tests, full pytest, local
preflight, and explicit inspection of both generated MP4 artifacts. No PPO or
training run is authorized by this feature.
