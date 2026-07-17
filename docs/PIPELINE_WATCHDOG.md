# DVGC unattended pipeline watchdog

The watchdog is an out-of-process, low-overhead monitor.  It does not import
JAX or MuJoCo, acquire the controller flock, modify experiment artifacts, or
launch workers while the controller is healthy.

User commands:

```bash
bash /home/qy/DVGC/scripts/dvgc_status.sh
bash /home/qy/DVGC/scripts/dvgc_watch.sh
```

The first command is strictly read-only.  The second only refreshes that same
view and is not required for pipeline execution.  Stable machine- and
human-readable snapshots are atomically written by the timer to
`runs/CURRENT_PIPELINE_STATUS.json` and `.md`.

`dvgc-pipeline-watchdog.timer` invokes a oneshot service every two minutes.
Healthy controllers and active long-running GPU workers are never interrupted.
An inactive non-terminal controller or a controller with stale heartbeat,
stale progress, and no live worker may be resumed from its existing state.
Recovery is capped at three consecutive attempts.  Completed shard markers,
checkpoints, seed allocation, OOM backoff, and provenance validation remain
owned by the existing controller.

Desktop notifications are deduplicated by run, stage, event type, and policy
hash.  They are emitted only for major stage transitions, pipeline completion,
research gates, or exhausted engineering recovery.  If desktop D-Bus delivery
fails, the request is retained in `runs/PENDING_NOTIFICATION.json`; critical
events additionally use a one-time `wall` fallback.  No runtime files under
`runs/` are tracked by Git.
