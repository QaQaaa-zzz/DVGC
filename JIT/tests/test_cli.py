from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_cli_rejects_unimplemented_phase_before_creating_run(jit_root, tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(jit_root / "src")
    environment["JIT_RUN_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(jit_root / "cli/train_phase_expert.py"),
            "--phase",
            "descent_recovery",
            "--config",
            str(jit_root / "configs/phase_u_smoke.json"),
            "--run-id",
            "forbidden",
            "--smoke",
        ],
        cwd=jit_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "only propulsion_ascent is implemented" in result.stderr
    assert not (tmp_path / "forbidden").exists()


def test_cli_requires_explicit_smoke_flag(jit_root, tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(jit_root / "src")
    environment["JIT_RUN_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(jit_root / "cli/train_phase_expert.py"),
            "--phase",
            "propulsion_ascent",
            "--config",
            str(jit_root / "configs/phase_u_smoke.json"),
            "--run-id",
            "missing_smoke",
        ],
        cwd=jit_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--smoke" in result.stderr
    assert not (tmp_path / "missing_smoke").exists()


def _run_cli(jit_root, tmp_path, *extra):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(jit_root / "src")
    environment["JIT_RUN_ROOT"] = str(tmp_path)
    return subprocess.run(
        [
            sys.executable,
            str(jit_root / "cli/train_phase_expert.py"),
            "--phase",
            "propulsion_ascent",
            "--config",
            str(jit_root / "configs/phase_u_formal.json"),
            "--run-id",
            "mode_test",
            *extra,
        ],
        cwd=jit_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_rejects_both_training_modes(jit_root, tmp_path):
    result = _run_cli(jit_root, tmp_path, "--smoke", "--formal")
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr
    assert not (tmp_path / "mode_test").exists()


def test_cli_rejects_restore_checkpoint_in_smoke_mode(jit_root, tmp_path):
    result = _run_cli(
        jit_root,
        tmp_path,
        "--smoke",
        "--restore-checkpoint",
        str(tmp_path / "checkpoint"),
    )
    assert result.returncode != 0
    assert "only valid with --formal" in result.stderr
    assert not (tmp_path / "mode_test").exists()
