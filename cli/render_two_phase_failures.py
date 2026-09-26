"""Render the two fixed Gate B dynamic-outcome audit videos."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.failure_video import FAILURE_SCENARIOS, render_failure_archive
from dvgc.reference import ReferenceTrajectory
from dvgc.two_phase_runtime import TwoPhaseThresholds, build_two_phase_geometry
from dvgc.two_phase_semantics import ApexBandThresholds, RecoveryThresholds


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_output(output: Path, manifest: dict) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Failure-video output directory is not absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "run_manifest.json", manifest)


def build(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    run_manifest = {
        "contract_version": 1,
        "purpose": "Render exact Gate B dynamic outcome audit videos",
        "inputs": {
            "config": str(args.config),
            "reference": str(args.reference),
            "threshold_manifest": str(args.threshold_manifest),
            "seed": int(args.seed),
            "scenarios": [asdict(value) for value in FAILURE_SCENARIOS.values()],
        },
        "interaction_cost": {
            "maximum_environment_transitions": 108,
            "expected_current_environment_transitions": 25,
            "formal_training_transitions": 0,
        },
        "stopping_condition": "stop after both named failure videos or first capture/render error",
        "output_directory": str(output),
    }
    _prepare_output(output, run_manifest)
    cfg = load_config(
        args.config,
        {
            "domain_randomization": False,
            "obs_noise_enable": False,
            "use_bank_resets": False,
        },
    )
    reference = ReferenceTrajectory.load(args.reference)
    threshold_manifest = json.loads(Path(args.threshold_manifest).read_text())
    expected_hashes = {
        "xml": file_sha256(cfg.xml_path),
        "config": file_sha256(args.config),
        "reference": file_sha256(args.reference),
    }
    for name, expected in expected_hashes.items():
        if threshold_manifest.get("source_hashes", {}).get(name) != expected:
            raise ValueError(f"Threshold manifest has stale {name} hash")
    selected = threshold_manifest["selected_thresholds"]
    thresholds = TwoPhaseThresholds(
        apex=ApexBandThresholds(**selected["apex"]),
        recovery=RecoveryThresholds(**selected["recovery"]),
    )
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    geometry = build_two_phase_geometry(env.mj_model, cfg)
    manifest = render_failure_archive(
        env,
        reference,
        geometry,
        thresholds,
        output_dir=output,
        seed=int(args.seed),
        source_hashes=expected_hashes,
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
    )
    report = {
        "status": "pass",
        "videos": [
            {key: value for key, value in row.items() if key != "telemetry"}
            for row in manifest["videos"]
        ],
        "environment_transitions": sum(
            int(row["environment_transitions"]) for row in manifest["videos"]
        ),
        "formal_training_transitions": 0,
    }
    _save_json(output / "failure_video_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", default="data/reference_jump.csv")
    parser.add_argument("--threshold-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=44_000)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
