# JIT TensorBoard

The current local dashboard is http://127.0.0.1:6007. For a remote VS Code session, forward port 6007 in the Ports panel. No external account or upload is required. TensorBoard is installed separately in `/home/qy/.local/share/jit-tensorboard-venv`; production simulator dependencies are unchanged.

Current manifest and receipt: `runs/monitoring/current_policy_tensorboard_20260914/manifest.json` and `events/receipt.json`. Server PID and logs are in `server.json` and `server.log` in that directory. This is an export of saved data, not a new training run.

- `jump_policy/*`: optimization loss, KL and learning rate versus training transitions.
- `jump_episode/*`: recorded episode reward and components versus training transitions. These are the original logged aggregates, not reconstructed per-control-step rewards.
- `explorer_outer_rounds`: explorer reward components, losses, KL, novelty and costs versus outer round. The source policy changes during this series; it is not one continuously trained network.
- `fixed_panel_evaluation`: success/failure/unknown and gains/losses versus policy ordinal. Order is pi_0, repair_0000, repair_0005, repair_0006, repair_0009, repair_0011. These are reused development states, not held-out performance.
- Text: source paths and full hyperparameter/configuration JSON where declared.

For another run, declare sources in JSON with `name`, `path`, `step`, optional `rows_key` and `configs`. CSV or list-of-objects JSON is supported. Export to a new directory:

```bash
/home/qy/.local/share/jit-tensorboard-venv/bin/python cli/export_tensorboard.py --manifest MANIFEST.json --output NEW_EVENTS_DIRECTORY
```

Use `--watch` to poll the declared files every 15 seconds. Writers append new metric/step pairs; source files must be append-only in step semantics. Restart into a fresh output directory. New sources require a new manifest/export session. This bridge follows saved metric files; it does not add missing telemetry to a trainer. Missing/nonfinite values are omitted, never replaced with zero. Original files remain authoritative.

Validation: current export was read back with TensorBoard EventAccumulator; 34,134 scalar points matched the export receipt. HTTP page and scalar-plugin responses were checked. Official usage: https://www.tensorflow.org/tensorboard/get_started
