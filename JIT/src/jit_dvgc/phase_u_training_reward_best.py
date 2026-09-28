"""Select a Phase U checkpoint from the recorded training reward peak."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def select_training_reward_best(run_dir: Path, *, target: int, block: int) -> dict:
    """Attribute each sampled block's rolling episode return to its source Actor."""
    root = Path(run_dir).resolve()
    if target <= 0 or block <= 0 or target % block:
        raise ValueError("target and block must define whole positive PPO blocks")
    status_path = root / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text())["status"] != "completed":
        raise ValueError("training has not completed")
    path = root / "episode_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = list(range(block, target + block, block))
    steps = [row["training_transitions"] for row in rows]
    if steps != expected:
        raise ValueError("complete consecutive episode reward blocks are required")
    candidates = []
    for row in rows:
        reward = float(row["metrics"]["episode/sum_reward"])
        if not math.isfinite(reward):
            raise ValueError("nonfinite training reward")
        sampled_until = int(row["training_transitions"])
        candidate = sampled_until - block
        checkpoint = root / "checkpoints" / f"transition_{candidate}"
        if not checkpoint.is_dir():
            raise ValueError(f"missing source checkpoint at {candidate}")
        candidates.append({
            "reward_peak_logged_transition": sampled_until,
            "selected_checkpoint_transition": candidate,
            "rolling_100_completed_episode_return": reward,
        })
    best = max(candidates, key=lambda item: (
        item["rolling_100_completed_episode_return"],
        -item["selected_checkpoint_transition"],
    ))
    checkpoint = root / "checkpoints" / f"transition_{best['selected_checkpoint_transition']}"
    identity_path = checkpoint / "identity.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text())
        payload = checkpoint / "payload.pkl"
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        if identity["training_transitions"] != best["selected_checkpoint_transition"] or identity["payload_sha256"] != digest:
            raise ValueError("selected checkpoint identity mismatch")
    report = {
        "schema": "jit_phase_u_training_reward_best_v1",
        "criterion": "maximum rolling mean cumulative episode return over last 100 completed training episodes, with earliest peak breaking ties",
        "attribution": "PPO samples a block with the pre-update Actor, then saves the post-update checkpoint at the block-end transition; the sampled block is attributed to its preceding checkpoint. Rolling 100 episodes may include older Actors, so this is a training proxy only.",
        "scope": "training samples; no fixed-seed evaluation or valid-landing qualification",
        **best,
        "unscored_final_checkpoint_transition": target,
        "scored_candidate_count": len(candidates),
        "candidates": candidates,
    }
    output = root / "bestmodel"
    output.mkdir(exist_ok=True)
    temporary_link = output / ".model.tmp"
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(Path("..") / "checkpoints" / checkpoint.name, target_is_directory=True)
    os.replace(temporary_link, output / "model")
    with tempfile.NamedTemporaryFile("w", dir=output, suffix=".json", delete=False) as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary_report = Path(stream.name)
    os.replace(temporary_report, output / "best_model.json")
    return report
