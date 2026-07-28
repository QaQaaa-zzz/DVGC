"""Atomically freeze the current assets for the diverse P1 predecessor run."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from cli.runtime_gate import source_fingerprint
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.runtime import save_json


PATHS = {
    "landing_policy": Path("runs/decoupled_bootstrap_seed0_20260720/frozen/pi_l_frozen/params.pkl"),
    "canonical_C_L": Path("runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl"),
    "descent_policy": Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy/params.pkl"),
    "descent_normalizer_bundle": Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy/params.pkl"),
    "p0_p1_manifest": Path("runs/backward_recovery_tube_fast_track_v1/descent_cem3_tier2/descent_cem_pilot_report.json"),
    "parent_lineage_and_tube_graph": Path("runs/backward_recovery_tube_fast_track_v1/descent_cem3_tier2/descent_cem_pilot_report.json"),
    "proposal_index": Path("runs/backward_recovery_tube_fast_track_v1/proposal_state_index.json"),
    "original_student_trajectory_states_actions": Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v1/dataset_v3/teacher_dataset.pkl"),
    "v4_snapshot_schema_source": Path("dvgc/snapshot_timing.py"),
    "runtime_gate": Path("docs/RUNTIME_GATE.json"),
}


def current_assets(root: Path) -> dict:
    cfg = load_config("configs/default.json")
    paths = {"xml": Path(cfg.xml_path), **PATHS, "preregistration": root / "preregistration.json"}
    assets = {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()}
    assets["action_mapping"] = {"version": ACTION_MAPPING_VERSION,
                                "sha256": hashlib.sha256(ACTION_MAPPING_VERSION.encode()).hexdigest()}
    assets["runtime_source"] = {"sha256": source_fingerprint(Path.cwd())}
    return assets


def verify_frozen_assets(root: Path) -> tuple[bool, list[str]]:
    import json
    path = root / "frozen_asset_manifest.json"
    if not path.exists(): return False, ["manifest_missing"]
    expected = json.loads(path.read_text())["assets"]
    actual = current_assets(root)
    failed = sorted(name for name in actual if expected.get(name) != actual[name])
    return not failed, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(); root = Path(args.run)
    if not (root / "preregistration.json").exists(): raise SystemExit("preregistration missing")
    forbidden = [root / name for name in ("trajectory_harvested_snapshots.pkl", "source_a_certification_results.json")]
    if any(path.exists() for path in forbidden): raise SystemExit("cannot refreeze after dynamic acquisition")
    save_json(root / "frozen_asset_manifest.json", {"status": "FROZEN", "assets": current_assets(root)})
    valid, failed = verify_frozen_assets(root)
    if not valid: raise SystemExit(f"asset freeze verification failed: {failed}")
    print(f"frozen_asset_identity=PASS assets={len(current_assets(root))}")


if __name__ == "__main__": main()
