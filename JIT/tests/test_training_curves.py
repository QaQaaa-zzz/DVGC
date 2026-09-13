from __future__ import annotations

import json

import mediapy as media
import numpy as np
import pytest

from jit_dvgc.training_curves import save_training_curves


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_training_curves_persist_plot_data_and_report(tmp_path):
    _write_jsonl(
        tmp_path / "episode_metrics.jsonl",
        [
            {
                "training_transitions": 24_576,
                "metrics": {
                    "episode/sum_reward": 10.0,
                    "episode/length": 20.0,
                    "episode/reset/source_airborne_rsi": 0.0,
                },
            },
            {
                "training_transitions": 49_152,
                "metrics": {
                    "episode/sum_reward": 12.0,
                    "episode/length": 24.0,
                    "episode/reset/source_airborne_rsi": 24.0,
                },
            },
        ],
    )
    _write_jsonl(
        tmp_path / "metrics.jsonl",
        [
            {
                "training_transitions": 24_576,
                "metrics": {
                    "training/kl_mean": 0.01,
                    "training/policy_loss": -0.2,
                    "training/v_loss": 0.4,
                    "training/total_loss": 0.0,
                    "training/policy_dist_mean_std": 0.8,
                    "training/sps": 1000.0,
                },
            },
            {
                "training_transitions": 49_152,
                "metrics": {
                    "training/kl_mean": 0.02,
                    "training/policy_loss": -0.1,
                    "training/v_loss": 0.3,
                    "training/total_loss": 0.05,
                    "training/policy_dist_mean_std": 0.7,
                    "training/sps": 1200.0,
                },
            },
        ],
    )

    report = save_training_curves(tmp_path)

    assert media.read_image(report.plot_path).ndim == 3
    assert report.data_path.is_file()
    assert report.report_path.is_file()
    assert len(report.plot_sha256) == 64
    assert len(report.data_sha256) == 64
    assert report.episode_sample_count == 2
    assert report.ppo_sample_count == 2
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["episode_metric_semantics"] == "rolling_last_100_completed_episodes"
    with np.load(report.data_path) as arrays:
        required = {
            "episode_training_transitions",
            "episode_mean_reward",
            "episode_mean_length",
            "episode_airborne_rsi_fraction",
            "ppo_training_transitions",
            "ppo_kl_mean",
            "ppo_policy_loss",
            "ppo_value_loss",
            "ppo_total_loss",
            "ppo_policy_mean_std",
            "ppo_sps",
        }
        assert required <= set(arrays.files)
        np.testing.assert_array_equal(
            arrays["episode_training_transitions"], [24_576, 49_152]
        )
        np.testing.assert_allclose(arrays["episode_mean_reward"], [10.0, 12.0])
        np.testing.assert_allclose(
            arrays["episode_airborne_rsi_fraction"], [0.0, 1.0]
        )


def test_training_curves_reject_missing_or_nonmonotonic_evidence(tmp_path):
    _write_jsonl(
        tmp_path / "episode_metrics.jsonl",
        [
            {
                "training_transitions": 24_576,
                "metrics": {"episode/length": 20.0},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "metrics.jsonl",
        [
            {"training_transitions": 49_152, "metrics": {}},
            {"training_transitions": 24_576, "metrics": {}},
        ],
    )

    with pytest.raises(ValueError, match="episode/sum_reward"):
        save_training_curves(tmp_path)

    _write_jsonl(
        tmp_path / "episode_metrics.jsonl",
        [
            {
                "training_transitions": 24_576,
                "metrics": {
                    "episode/sum_reward": 1.0,
                    "episode/length": 20.0,
                    "episode/reset/source_airborne_rsi": 0.0,
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        save_training_curves(tmp_path)
