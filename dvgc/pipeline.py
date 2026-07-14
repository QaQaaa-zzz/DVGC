"""Small pure helpers for resumable experiment steps and evidence gates."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def path_digest(path: str | Path) -> str:
    """Hash one file or a directory tree, including relative names."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    if root.is_file():
        digest.update(root.read_bytes())
        return digest.hexdigest()
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def input_digest(tokens: Sequence[str], inputs: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(str(token).encode()); digest.update(b"\0")
    for value in inputs:
        path = Path(value)
        digest.update(str(path.resolve()).encode()); digest.update(b"\0")
        digest.update(path_digest(path).encode()); digest.update(b"\0")
    return digest.hexdigest()


def marker_is_current(
    marker_path: str | Path,
    *,
    tokens: Sequence[str],
    inputs: Sequence[str | Path],
    outputs: Sequence[str | Path],
) -> tuple[bool, str]:
    path = Path(marker_path)
    if not path.is_file():
        return False, "missing marker"
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("status") != "COMPLETE" or int(marker.get("exit_status", -1)) != 0:
            return False, "last attempt did not complete"
        if marker.get("input_digest") != input_digest(tokens, inputs):
            return False, "input digest changed"
        expected = marker.get("outputs", {})
        for value in outputs:
            output = Path(value)
            key = str(output.resolve())
            if not output.exists() or expected.get(key) != path_digest(output):
                return False, f"output missing or changed: {output}"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"invalid marker: {exc}"
    return True, "current"


def write_marker(
    marker_path: str | Path,
    *,
    step: str,
    tokens: Sequence[str],
    inputs: Sequence[str | Path],
    outputs: Sequence[str | Path],
    exit_status: int,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    marker = Path(marker_path)
    payload = {
        "step": str(step),
        "status": "COMPLETE" if int(exit_status) == 0 else "FAILED",
        "exit_status": int(exit_status),
        "recorded_at": time.time(),
        "input_digest": input_digest(tokens, inputs),
        "inputs": {str(Path(p).resolve()): path_digest(p) for p in inputs},
        "outputs": {
            str(Path(p).resolve()): path_digest(p)
            for p in outputs if Path(p).exists()
        },
        "log": str(Path(log_path).resolve()) if log_path else None,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{marker.name}.", dir=marker.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, marker)
    except Exception:
        try: os.unlink(temporary)
        except OSError: pass
        raise
    return payload


def training_decision(
    analysis: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    minimum_final: float,
    maximum_timeout: float,
    reference_evaluation: Mapping[str, Any] | None = None,
    maximum_final_drop: float = 0.05,
) -> dict[str, Any]:
    health = analysis.get("health", {})
    final = float(evaluation.get("final_recovery_rate", float("nan")))
    chain = float(evaluation.get("chain_rate", float("nan")))
    failure = float(evaluation.get("physical_failure_rate", float("nan")))
    timeout = float(evaluation.get("timeout_rate", float("nan")))
    reasons: list[str] = []
    if analysis.get("analysis_status") != "COMPLETED_HEALTHY": reasons.append("training runtime is not healthy")
    for key in ("oom", "broadphase_overflow", "narrowphase_overflow", "ccd_overflow", "compile_restart"):
        if bool(health.get(key, False)): reasons.append(key)
    if int(health.get("nonfinite_count", 0)) != 0: reasons.append("nonfinite metrics")
    if not all(math.isfinite(value) for value in (final, chain, failure, timeout)): reasons.append("nonfinite outcomes")
    if final < float(minimum_final): reasons.append("Final-Recovery below stage gate")
    if timeout > float(maximum_timeout): reasons.append("timeout above stage gate")
    reference_final = None
    if reference_evaluation is not None:
        reference_final = float(reference_evaluation["final_recovery_rate"])
        if final < reference_final - float(maximum_final_drop): reasons.append("Final-Recovery regressed from prior accepted policy")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "final_recovery_rate": final,
        "chain_success_rate": chain,
        "physical_failure_rate": failure,
        "timeout_rate": timeout,
        "minimum_final": float(minimum_final),
        "maximum_timeout": float(maximum_timeout),
        "reference_final_recovery_rate": reference_final,
        "maximum_final_drop": float(maximum_final_drop),
        "health": dict(health),
        "reasons": reasons,
    }


def certification_decision(report: Mapping[str, Any], phase: str, minimum_safe: int) -> dict[str, Any]:
    summary = report["summary"][phase]
    terminal = report["terminal_summary"]
    labels = summary["final"]
    reasons = []
    if int(labels.get("safe", 0)) < int(minimum_safe): reasons.append("insufficient Final-safe states")
    if int(terminal.get("horizon_exhaustions", 0)) != 0: reasons.append("horizon exhaustion present")
    if int(terminal.get("branches", 0)) <= 0: reasons.append("missing branch evidence")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "phase": phase,
        "labels": dict(labels),
        "chain_labels": dict(summary["chain"]),
        "terminal_summary": dict(terminal),
        "minimum_safe": int(minimum_safe),
        "reasons": reasons,
    }


def audit_decision(
    report: Mapping[str, Any],
    *,
    minimum_precision: float,
    maximum_timeout: float,
) -> dict[str, Any]:
    rows = report.get("rows", [])
    states = int(report.get("states", 0)); per_state = int(report.get("branches_per_state", 0))
    indices = [int(row["state_index"]) for row in rows]
    seeds = [int(branch["branch_seed"]) for row in rows for branch in row.get("branches", [])]
    terminal = report.get("terminal_summary", {})
    precision = float(report.get("tube_precision", float("nan")))
    timeout = float(report.get("timeout_rate", float("nan")))
    reasons = []
    if indices != list(range(states)): reasons.append("global state coverage is incomplete")
    if len(seeds) != states * per_state or len(seeds) != len(set(seeds)): reasons.append("branch seeds are incomplete or reused")
    if not math.isfinite(precision) or precision < float(minimum_precision): reasons.append("Tube precision below audit gate")
    if not math.isfinite(timeout) or timeout > float(maximum_timeout): reasons.append("timeout above audit gate")
    if int(terminal.get("horizon_exhaustions", 0)) != 0: reasons.append("horizon exhaustion present")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "policy_version": report.get("policy_version"),
        "states": states,
        "branches": len(seeds),
        "tube_precision": precision,
        "recoverable_recall": report.get("recoverable_recall"),
        "candidate_mass_coverage": report.get("candidate_mass_coverage"),
        "brier": report.get("brier"),
        "ece_5bin": report.get("ece_5bin"),
        "terminal_summary": dict(terminal),
        "minimum_precision": float(minimum_precision),
        "maximum_timeout": float(maximum_timeout),
        "reasons": reasons,
    }
