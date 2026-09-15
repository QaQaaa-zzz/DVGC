from __future__ import annotations

import importlib

import pytest


def test_parser_exposes_one_smoke_only_total_transition_interface():
    cli = importlib.import_module("cli.train_phase_expert")
    parser = cli.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "phase" in destinations
    assert "requested_total_transitions" in destinations
    assert "timesteps" not in destinations
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "legacy_flight"])


def test_resume_arguments_must_be_paired():
    cli = importlib.import_module("cli.train_phase_expert")
    args = cli.build_parser().parse_args(
        [
            "--phase", "propulsion_ascent",
            "--requested-total-transitions", "1600",
            "--threshold-manifest", "threshold.json",
            "--run", "runs/two_phase/phase_experts/test",
            "--resume-run", "old",
            "--preflight-only",
        ]
    )
    with pytest.raises(ValueError, match="paired"):
        cli.spec_from_args(args)
