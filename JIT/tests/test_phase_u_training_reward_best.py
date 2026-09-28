import json
from pathlib import Path

import pytest

from jit_dvgc.phase_u_training_reward_best import select_training_reward_best


def test_peak_block_scores_preupdate_checkpoint_and_final_is_unscored(tmp_path):
    root = tmp_path
    block = 24_576
    for step in (0, block, 2 * block, 3 * block):
        (root / "checkpoints" / f"transition_{step}").mkdir(parents=True)
    rows = [
        {"training_transitions": block, "metrics": {"episode/sum_reward": 5.0}},
        {"training_transitions": 2 * block, "metrics": {"episode/sum_reward": 9.0}},
        {"training_transitions": 3 * block, "metrics": {"episode/sum_reward": 6.0}},
    ]
    (root / "episode_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = select_training_reward_best(root, target=3 * block, block=block)
    assert report["reward_peak_logged_transition"] == 2 * block
    assert report["selected_checkpoint_transition"] == block
    assert report["unscored_final_checkpoint_transition"] == 3 * block
    assert (root / "bestmodel/model").resolve() == (root / "checkpoints" / f"transition_{block}")


def test_missing_reward_block_is_rejected(tmp_path):
    (tmp_path / "episode_metrics.jsonl").write_text(json.dumps({"training_transitions": 24_576, "metrics": {"episode/sum_reward": 1.0}}) + "\n")
    with pytest.raises(ValueError, match="complete"):
        select_training_reward_best(tmp_path, target=49_152, block=24_576)
