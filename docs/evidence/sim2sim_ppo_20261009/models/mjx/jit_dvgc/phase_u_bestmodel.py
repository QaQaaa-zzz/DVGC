"""Select the highest fixed-panel episode return among saved Phase U checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel_returns(run_dir: Path, panel: str, step: int) -> tuple[list[int], list[float]] | None:
    directory = run_dir / panel / f"transition_{step}"
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seeds = summary["held_out_seeds"]
    artifacts = summary["trace_artifacts"]
    if summary["absolute_transition"] != step or len(seeds) != len(artifacts):
        raise ValueError(f"invalid {panel} panel at {step}")
    if sorted(seeds) != sorted(item["seed"] for item in artifacts):
        raise ValueError(f"mismatched {panel} panel seeds at {step}")
    returns = []
    for artifact in artifacts:
        path = directory / f"seed_{artifact['seed']}.npz"
        if _sha256(path) != artifact["npz_sha256"]:
            raise ValueError(f"trace hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as trace:
            reward = np.asarray(trace["reward"], dtype=np.float64)
        if reward.ndim != 1 or len(reward) < 2 or not np.isfinite(reward).all():
            raise ValueError(f"invalid trace reward: {path}")
        returns.append(float(reward[1:].sum()))
    return seeds, returns


def update_best_model(run_dir: Path) -> dict | None:
    """Write a bestmodel link and score only from complete paired fixed panels."""
    run_dir = Path(run_dir).resolve()
    candidates = []
    for checkpoint in sorted(
        (run_dir / "checkpoints").glob("transition_*"),
        key=lambda path: int(path.name.removeprefix("transition_")),
    ):
        step = int(checkpoint.name.removeprefix("transition_"))
        if step == 0:
            continue
        natural_panel = _panel_returns(run_dir, "evaluations", step)
        airborne_panel = _panel_returns(run_dir, "diagnostics/airborne_rsi", step)
        if natural_panel is None or airborne_panel is None:
            continue
        natural_seeds, natural = natural_panel
        airborne_seeds, airborne = airborne_panel
        if natural_seeds != airborne_seeds:
            raise ValueError(f"paired panel seeds differ at {step}")
        identity = json.loads((checkpoint / "identity.json").read_text(encoding="utf-8"))
        if identity["training_transitions"] != step or _sha256(checkpoint / "payload.pkl") != identity["payload_sha256"]:
            raise ValueError(f"checkpoint identity mismatch at {step}")
        score = float(np.mean(natural + airborne))
        if not math.isfinite(score):
            raise ValueError(f"nonfinite score at {step}")
        candidates.append({
            "training_transitions": step,
            "mean_episode_return": score,
            "natural_mean_episode_return": float(np.mean(natural)),
            "airborne_rsi_mean_episode_return": float(np.mean(airborne)),
            "episode_count": len(natural) + len(airborne),
            "checkpoint_payload_sha256": identity["payload_sha256"],
        })
    if not candidates:
        return None
    best = max(candidates, key=lambda row: (row["mean_episode_return"], -row["training_transitions"]))
    report = {
        "criterion": "highest mean actual clipped episode return over paired fixed natural and airborne RSI panels",
        "scope": "saved checkpoints with complete paired panels; development selection, not task success or independent test",
        **best,
        "candidates": candidates,
    }
    output = run_dir / "bestmodel"
    output.mkdir(exist_ok=True)
    temporary_link = output / ".model.tmp"
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(Path("..") / "checkpoints" / f"transition_{best['training_transitions']}", target_is_directory=True)
    os.replace(temporary_link, output / "model")
    with tempfile.NamedTemporaryFile("w", dir=output, suffix=".json", delete=False) as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary_report = Path(stream.name)
    os.replace(temporary_report, output / "best_model.json")
    return report
