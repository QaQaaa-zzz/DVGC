from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import jax.numpy as jp
import numpy as np
import pytest

from dvgc.phase_u_launch_diagnostic import (
    close_diagnostic_outcomes,
    feedback_launch_action,
    feedback_launch_specs,
    relative_x_launch_action,
    relative_x_launch_specs,
    rank_diagnostic_row,
    select_representative_rows,
)
from cli.diagnose_phase_u_feedback_launch import (
    frozen_manifest_payload,
    write_diagnostic_run,
)


def test_feedback_launch_grid_is_exact_frozen_cartesian_product():
    specs = feedback_launch_specs()
    assert len(specs) == 384
    payloads = [tuple(sorted(asdict(spec).items())) for spec in specs]
    assert len(set(payloads)) == 384
    assert {spec.launch_bias for spec in specs} == {0.2, 0.3, 0.4, 0.5}
    assert {spec.knee_ratio for spec in specs} == {0.0, 0.5, 1.0, 1.5}
    assert {spec.pitch_gain for spec in specs} == {0.0, 0.5, 1.0}
    assert {spec.pitch_rate_gain for spec in specs} == {0.0, 0.03, 0.06, 0.1}
    assert {spec.active_ticks for spec in specs} == {4, 7}
    assert {spec.action_limit for spec in specs} == {0.8}


def test_relative_x_launch_grid_is_exact_unfiltered_product():
    base = feedback_launch_specs()
    specs = relative_x_launch_specs()
    assert len(specs) == 1152
    assert {spec.obstacle_relative_x_onset for spec in specs} == {1.17, 1.12, 1.07}
    pairs = {
        (spec.obstacle_relative_x_onset, tuple(sorted(asdict(spec.feedback).items())))
        for spec in specs
    }
    assert len(pairs) == 1152
    for onset in (1.17, 1.12, 1.07):
        assert [spec.feedback for spec in specs if spec.obstacle_relative_x_onset == onset] == list(base)


def test_feedback_launch_action_is_neutral_before_window_and_after_duration():
    spec = feedback_launch_specs()[0]
    before = feedback_launch_action(
        spec, pitch=0.0, pitch_rate=0.0, window_latched=False, active_age=0
    )
    after = feedback_launch_action(
        spec,
        pitch=0.0,
        pitch_rate=0.0,
        window_latched=True,
        active_age=spec.active_ticks,
    )
    np.testing.assert_array_equal(before, jp.zeros((4,), jp.float32))
    np.testing.assert_array_equal(after, jp.zeros((4,), jp.float32))


def test_feedback_launch_action_brakes_positive_pitch_and_rate_and_clips():
    spec = next(
        item
        for item in feedback_launch_specs()
        if item.launch_bias == 0.5
        and item.knee_ratio == 1.5
        and item.pitch_gain == 1.0
        and item.pitch_rate_gain == 0.1
        and item.active_ticks == 4
    )
    nominal = feedback_launch_action(
        spec, pitch=0.0, pitch_rate=0.0, window_latched=True, active_age=0
    )
    braked = feedback_launch_action(
        spec, pitch=0.2, pitch_rate=1.0, window_latched=True, active_age=0
    )
    reversed_action = feedback_launch_action(
        spec, pitch=2.0, pitch_rate=10.0, window_latched=True, active_age=0
    )
    assert nominal.tolist() == pytest.approx([0.0, 0.0, 0.5, 0.75])
    assert braked.tolist() == pytest.approx([0.0, 0.0, 0.2, 0.3])
    assert reversed_action.tolist() == pytest.approx([0.0, 0.0, -0.8, 0.0])


def test_relative_x_action_is_neutral_before_onset_and_active_at_equality():
    spec = relative_x_launch_specs()[0]
    before = relative_x_launch_action(
        spec, obstacle_relative_x=1.170001, pitch=0.0, pitch_rate=0.0, active_age=0
    )
    equal = relative_x_launch_action(
        spec, obstacle_relative_x=1.17, pitch=0.0, pitch_rate=0.0, active_age=0
    )
    after_duration = relative_x_launch_action(
        spec,
        obstacle_relative_x=1.0,
        pitch=0.0,
        pitch_rate=0.0,
        active_age=spec.feedback.active_ticks,
    )
    np.testing.assert_array_equal(before, jp.zeros((4,), jp.float32))
    assert equal.tolist() == pytest.approx([0.0, 0.0, 0.2, 0.0])
    np.testing.assert_array_equal(after_duration, jp.zeros((4,), jp.float32))


def test_close_diagnostic_outcomes_is_standard_and_closed():
    rows = [
        {"success": True, "physical_failure": False, "timeout": False},
        {"success": False, "physical_failure": True, "timeout": False},
        {"success": False, "physical_failure": False, "timeout": True},
        {"success": False, "physical_failure": False, "timeout": False},
    ]
    report = close_diagnostic_outcomes(rows)
    assert report == {
        "num_branches": 4,
        "outcome_counts": {
            "success": 1,
            "physical_failure": 1,
            "timeout": 1,
            "other_failure": 1,
        },
        "empirical_success_rate": 0.25,
        "physical_failure_rate": 0.25,
        "timeout_rate": 0.25,
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        close_diagnostic_outcomes(
            [{"success": True, "physical_failure": True, "timeout": False}]
        )


def _row(identifier, *, success=False, stable=False, ascending=False,
         physical=False, residual=2.0, rate=1.0, energy=1.0, reason="deadline"):
    return {
        "branch_id": identifier,
        "success": success,
        "stable_airborne_reached": stable,
        "ascending_reached": ascending,
        "physical_failure": physical,
        "minimum_apex_contract_residual": residual,
        "maximum_angular_speed": rate,
        "action_energy": energy,
        "terminal_reason": reason,
    }


def test_diagnostic_ranking_prefers_apex_then_safe_progress_then_quality():
    rows = [
        _row("unsafe", stable=True, ascending=True, physical=True, residual=0.1),
        _row("partial", stable=True, ascending=True, residual=0.5, rate=0.7),
        _row("better_partial", stable=True, ascending=True, residual=0.3, rate=0.8),
        _row("apex", success=True, physical=False, residual=0.8, rate=1.2),
    ]
    assert [row["branch_id"] for row in sorted(rows, key=rank_diagnostic_row)] == [
        "apex",
        "better_partial",
        "partial",
        "unsafe",
    ]


def test_media_selection_depends_on_outcome_not_renderer_state():
    rows = [
        _row("a", reason="deadline", residual=2.0),
        _row("b", reason="deadline", residual=1.0),
        _row("c", reason="pitch_limit", physical=True, residual=3.0),
        _row("d", reason="timeout", residual=0.8),
        _row("e", success=True, reason="apex", residual=0.0),
    ]
    rows[0]["render_status"] = "rendered"
    rows[1]["render_status"] = "render_failed"
    selected = select_representative_rows(rows, maximum=8)
    assert [row["branch_id"] for row in selected] == ["e", "b", "c", "d"]


def test_frozen_manifest_binds_budget_hashes_and_claim_boundaries():
    payload = frozen_manifest_payload(
        run_id="diagnostic",
        seed=731000,
        horizon=80,
        source_head="head",
        source_tree_sha256="source",
        xml_sha256="xml",
        config_sha256="config",
        training_config_sha256="training",
        threshold_manifest_canonical_hash="threshold",
        reward_contract_hash="reward",
    )
    assert payload["status"] == "frozen_before_outcomes"
    assert payload["branch_count"] == 384
    assert payload["maximum_diagnostic_environment_transitions"] == 30_720
    assert payload["ppo_training_transitions"] == 0
    assert payload["controller_grid"] == [
        asdict(spec) for spec in feedback_launch_specs()
    ]
    assert payload["claims"] == {
        "expert": False,
        "reachable": False,
        "safe": False,
        "tube": False,
        "training_reset": False,
    }
    assert payload["source_hashes"] == {
        "source_tree": "source",
        "xml": "xml",
        "config": "config",
        "training_config": "training",
        "threshold_manifest": "threshold",
        "reward_contract": "reward",
    }


def test_relative_x_manifest_binds_exact_onsets_and_ceiling():
    specs = relative_x_launch_specs()
    payload = frozen_manifest_payload(
        run_id="relative_x",
        seed=731100,
        horizon=80,
        source_head="head",
        source_tree_sha256="source",
        xml_sha256="xml",
        config_sha256="config",
        training_config_sha256="training",
        threshold_manifest_canonical_hash="threshold",
        reward_contract_hash="reward",
        mode="relative_x_onset",
        specs=specs,
    )
    assert payload["mode"] == "relative_x_onset"
    assert payload["branch_count"] == 1152
    assert payload["maximum_diagnostic_environment_transitions"] == 92_160
    assert payload["ppo_training_transitions"] == 0
    assert payload["obstacle_relative_x_onsets"] == [1.17, 1.12, 1.07]
    assert payload["controller_grid"] == [asdict(spec) for spec in specs]


def test_write_diagnostic_run_freezes_before_outcomes_and_closes_accounting(tmp_path):
    run = tmp_path / "run"
    manifest = frozen_manifest_payload(
        run_id=run.name,
        seed=731000,
        horizon=80,
        source_head="head",
        source_tree_sha256="source",
        xml_sha256="xml",
        config_sha256="config",
        training_config_sha256="training",
        threshold_manifest_canonical_hash="threshold",
        reward_contract_hash="reward",
    )
    calls = []

    def branch_runner(index, spec):
        assert (run / "frozen_manifest.json").is_file()
        calls.append(index)
        return _row(
            f"branch_{index:03d}",
            success=index == 3,
            physical=index == 2,
            stable=index == 1,
            ascending=index == 1,
            residual=float(index),
            reason="apex" if index == 3 else "pitch_limit" if index == 2 else "deadline",
        ) | {
            "timeout": False,
            "diagnostic_environment_transitions": 2,
            "spec": asdict(spec),
        }

    rendered = []

    def renderer(row):
        rendered.append(row["branch_id"])
        return {"branch_id": row["branch_id"], "status": "rendered"}

    report = write_diagnostic_run(
        run,
        manifest=manifest,
        specs=feedback_launch_specs(),
        branch_runner=branch_runner,
        media_renderer=renderer,
    )
    assert calls == list(range(384))
    assert report["num_branches"] == 384
    assert report["outcome_counts"] == {
        "success": 1,
        "physical_failure": 1,
        "timeout": 0,
        "other_failure": 382,
    }
    assert report["diagnostic_environment_transitions"] == 768
    assert report["ppo_training_transitions"] == 0
    assert len((run / "outcomes.jsonl").read_text().splitlines()) == 384
    assert json.loads((run / "frozen_manifest.json").read_text()) == manifest
    selected = json.loads((run / "representative_media.json").read_text())
    assert rendered == [row["branch_id"] for row in selected]
    with pytest.raises(ValueError, match="refusing overwrite"):
        write_diagnostic_run(
            run,
            manifest=manifest,
            specs=feedback_launch_specs(),
            branch_runner=branch_runner,
        )
