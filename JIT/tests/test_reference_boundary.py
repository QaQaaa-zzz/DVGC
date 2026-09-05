from __future__ import annotations

import json

import pytest

from jit_dvgc.constants import EXPECTED_REFERENCE_SHA256
from jit_dvgc.provenance import (
    InteractionAccounting,
    RunDeclaration,
    close_run,
    predeclare_run,
)
from jit_dvgc.reference_analysis import analyze_reference


def test_reference_analysis_is_identity_bound_and_reports_hand_checked_extent(
    repository_root,
):
    report = analyze_reference(repository_root / "data" / "reference_jump.csv")

    assert report["reference_sha256"] == EXPECTED_REFERENCE_SHA256
    assert report["row_count"] == 821
    assert report["time"] == pytest.approx({"min": 0.002, "max": 1.642})
    assert set(report["ranges"]) == {
        "roll_angle",
        "pitch_angle",
        "pos_x",
        "pos_z",
        "vel_x",
        "vel_z",
        "hip_position",
        "knee_position",
    }
    assert report["training_runtime_dependency"] is False


def test_run_is_frozen_before_interaction_and_closes_exact_accounting(tmp_path):
    declaration = RunDeclaration(
        run_id="unit_contract",
        purpose="compile_update_checkpoint_restore_engineering_smoke",
        output_dir=tmp_path / "unit_contract",
        config_sha256="1" * 64,
        xml_sha256="2" * 64,
        reference_sha256="3" * 64,
        training_transition_ceiling=25_600,
        stopping_conditions=("stop_after_one_aligned_ppo_block", "stop_on_runtime_failure"),
        resume_command="python JIT/cli/train_phase_expert.py --smoke",
    )

    run_dir = predeclare_run(declaration, resolved_config={"phase": "propulsion_ascent"})
    assert json.loads((run_dir / "status.json").read_text())["status"] == "predeclared"
    assert json.loads((run_dir / "run_manifest.json").read_text())["training_transition_ceiling"] == 25_600
    assert (run_dir / "resolved_config.json").is_file()
    assert (run_dir / "resume_command.txt").read_text().strip().endswith("--smoke")

    close_run(
        run_dir,
        status="completed",
        accounting=InteractionAccounting(training=25_600, brax_evaluation=0, fixed_evaluation=0, diagnostic=0),
        reason="one_block_complete",
    )
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["environment_transitions"] == 25_600
    assert status["interaction_accounting"] == {
        "training": 25_600,
        "brax_evaluation": 0,
        "fixed_evaluation": 0,
        "diagnostic": 0,
    }


def test_predeclared_run_cannot_silently_overwrite_existing_evidence(tmp_path):
    declaration = RunDeclaration(
        run_id="immutable",
        purpose="engineering_smoke",
        output_dir=tmp_path / "immutable",
        config_sha256="1" * 64,
        xml_sha256="2" * 64,
        reference_sha256="3" * 64,
        training_transition_ceiling=1,
        stopping_conditions=("stop_after_one_transition",),
        resume_command="python smoke.py",
    )
    predeclare_run(declaration, resolved_config={"phase": "propulsion_ascent"})

    with pytest.raises(FileExistsError):
        predeclare_run(declaration, resolved_config={"phase": "propulsion_ascent"})
