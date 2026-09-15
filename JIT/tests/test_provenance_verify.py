from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from jit_dvgc.config import canonical_sha256
from jit_dvgc.provenance import (
    InteractionAccounting,
    RunDeclaration,
    close_run,
    predeclare_run,
    verify_run,
)


def _closed_run(tmp_path):
    resolved = {"phase": "propulsion_ascent", "model": {}}
    run_dir = tmp_path / "closed"
    predeclare_run(
        RunDeclaration(
            run_id="closed",
            purpose="unit_test",
            output_dir=run_dir,
            config_sha256=canonical_sha256(resolved),
            xml_sha256="a" * 64,
            reference_sha256="b" * 64,
            training_transition_ceiling=10,
            stopping_conditions=("stop",),
            resume_command="false",
        ),
        resolved_config=resolved,
    )
    close_run(
        run_dir,
        status="completed",
        accounting=InteractionAccounting(
            training=10,
            brax_evaluation=0,
            fixed_evaluation=0,
            diagnostic=2,
        ),
        reason="unit test",
    )
    return run_dir


def test_verify_run_checks_closed_status_config_hash_and_accounting(tmp_path):
    report = verify_run(_closed_run(tmp_path))
    assert report["status"] == "completed"
    assert report["training_transitions"] == 10
    assert report["total_environment_transitions"] == 12


def test_verify_run_rejects_config_identity_drift(tmp_path):
    run_dir = _closed_run(tmp_path)
    resolved_path = run_dir / "resolved_config.json"
    payload = json.loads(resolved_path.read_text())
    payload["phase"] = "changed"
    resolved_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256"):
        verify_run(run_dir)


def test_provenance_module_exposes_verify_run_cli(jit_root, tmp_path):
    run_dir = _closed_run(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(jit_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jit_dvgc.provenance",
            "verify-run",
            str(run_dir),
        ],
        cwd=jit_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"status": "completed"' in result.stdout
