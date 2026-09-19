from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jit_dvgc.handoff_snapshot import HandoffSnapshot, save_snapshot
from jit_dvgc.upstream_boundary import canonical_sha256, file_sha256, physical_state_sha256
from jit_dvgc.upstream_value import ContinuationValueMLP


UP_ACTOR = "a" * 64
DOWN_ACTOR = "b" * 64
XML_SHA = "c" * 64
CONFIG_SHA = "d" * 64
COMPATIBILITY = {"xml_sha256": XML_SHA, "contract": "test"}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _snapshot(root: Path, relative: str, marker: float) -> tuple[Path, str]:
    snapshot = HandoffSnapshot(
        qpos=np.full(12, marker, dtype=np.float32),
        qvel=np.full(11, marker + 0.25, dtype=np.float32),
        observation_fifo=np.full((3, 25), marker, dtype=np.float32),
        history_valid_count=3,
        observation=np.full(76, marker, dtype=np.float32),
        last_action=np.zeros(4, dtype=np.float32),
        ctrl=np.zeros(4, dtype=np.float32),
        rng=np.asarray([1, 2], dtype=np.uint32),
        events={"jump_signal": np.asarray(False)},
        tick=10,
        parent_trajectory=f"parent-{marker}",
        parent_tick=10,
        config_sha256=CONFIG_SHA,
        xml_sha256=XML_SHA,
        policy_sha256="e" * 64,
        compatibility_identity=COMPATIBILITY,
    )
    path = root / relative
    save_snapshot(path, snapshot)
    return path, physical_state_sha256(snapshot)


def _model(root: Path, *, target: str, actor: str) -> Path:
    root.mkdir(parents=True)
    model = ContinuationValueMLP(hidden_sizes=(4,))
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 2), dtype=jnp.float32)
    )["params"]
    (root / "params.msgpack").write_bytes(serialization.to_bytes(params))
    np.savez(root / "normalization.npz", mean=np.zeros(2), std=np.ones(2))
    schema = {
        "V_up": "jit_upstream_value_model_v1",
        "V_down": "jit_downstream_value_model_v1",
    }[target]
    manifest = {
        "schema": schema,
        "status": "completed",
        "target": target,
        "expert_actor_sha256": actor,
        "observation_size": 2,
        "hidden_sizes": [4],
        "params_sha256": file_sha256(root / "params.msgpack"),
        "normalization_sha256": file_sha256(root / "normalization.npz"),
        "test_data_used": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _write(root / "manifest.json", manifest)
    return root


def _label(
    *,
    candidate: str,
    state_sha: str,
    split: str,
    phase: str,
    source_bank: str,
    snapshot: str,
    marker: float,
    protocol: str,
) -> dict:
    return {
        "candidate_id": candidate,
        "state_sha256": state_sha,
        "parent_group_id": f"parent-{split}-{candidate}",
        "seed": {"train": 1000001, "validation": 1000006, "test": 1000007}[split],
        "role": "ascending_entry" if phase == "upstream" else "nearest_apex",
        "split": split,
        "source_bank": source_bank,
        "snapshot": snapshot,
        "tick": 10,
        "actor_observation": [marker, marker + 1.0],
        "branch_count": 1,
        "success_count": int(marker > 0.5),
        "expert_actor_sha256": UP_ACTOR if phase == "upstream" else DOWN_ACTOR,
        "protocol_sha256": protocol,
    }


def _fixture(tmp_path: Path):
    from jit_dvgc.soft_tube import SoftTubeInputs

    frozen = {
        "schema": "jit_frozen_phase_experts_v1",
        "status": "frozen",
        "experts": {
            "pi_up_star": {
                "actor_sha256": UP_ACTOR,
                "payload_sha256": "1" * 64,
                "normalizer_sha256": "2" * 64,
                "config_sha256": CONFIG_SHA,
                "xml_sha256": XML_SHA,
            },
            "pi_down_star": {
                "actor_sha256": DOWN_ACTOR,
                "payload_sha256": "3" * 64,
                "normalizer_sha256": "4" * 64,
                "config_sha256": CONFIG_SHA,
                "xml_sha256": XML_SHA,
            },
        },
    }
    frozen_path = tmp_path / "frozen.json"
    _write(frozen_path, frozen)
    frozen_sha = file_sha256(frozen_path)
    up_model = _model(tmp_path / "up_model", target="V_up", actor=UP_ACTOR)
    down_model = _model(tmp_path / "down_model", target="V_down", actor=DOWN_ACTOR)

    up_nominal_root = tmp_path / "up_nominal"
    _, up_nominal_sha = _snapshot(
        up_nominal_root, "source_a/snapshots/train", 0.25
    )
    up_nominal_catalog = up_nominal_root / "catalog.json"
    _write(
        up_nominal_catalog,
        {
            "schema": "jit_upstream_candidate_catalog_v1",
            "frozen_pi_up_actor_sha256": UP_ACTOR,
            "xml_sha256": XML_SHA,
            "entries": [
                {
                    "candidate_id": "up-nominal",
                    "state_sha256": up_nominal_sha,
                    "source_bank": "source_a",
                    "snapshot": "snapshots/train",
                    "seed": 1000001,
                    "parent_group_id": "parent-train-up-nominal",
                    "role": "ascending_entry",
                    "tick": 10,
                }
            ],
        },
    )
    up_protocol_sha = "5" * 64
    up_nominal_protocol = up_nominal_root / "protocol.json"
    _write(
        up_nominal_protocol,
        {
            "schema": "jit_continuation_labels_v1",
            "target": "V_up",
            "protocol_sha256": up_protocol_sha,
            "catalog_sha256": file_sha256(up_nominal_catalog),
            "expert_actor_sha256": UP_ACTOR,
            "frozen_manifest_sha256": frozen_sha,
            "xml_sha256": XML_SHA,
            "train_seeds": [1000001],
            "validation_seeds": [1000006],
            "test_seeds": [1000007],
        },
    )
    up_nominal_labels = up_nominal_root / "labels.json"
    _write(
        up_nominal_labels,
        [
            _label(
                candidate="up-nominal",
                state_sha=up_nominal_sha,
                split="train",
                phase="upstream",
                source_bank="source_a",
                snapshot="snapshots/train",
                marker=0.25,
                protocol=up_protocol_sha,
            ),
            _label(
                candidate="up-validation",
                state_sha="f" * 64,
                split="validation",
                phase="upstream",
                source_bank="missing",
                snapshot="must-not-be-read",
                marker=91.0,
                protocol="wrong-validation-protocol",
            ),
            _label(
                candidate="up-test",
                state_sha="0" * 64,
                split="test",
                phase="upstream",
                source_bank="missing",
                snapshot="must-not-be-read",
                marker=92.0,
                protocol="wrong-test-protocol",
            ),
        ],
    )

    up_boundary_root = tmp_path / "up_boundary"
    _, up_boundary_sha = _snapshot(
        up_boundary_root, "boundary_bank/snapshots/train", 0.75
    )
    boundary_protocol_sha = "6" * 64
    up_boundary_catalog = up_boundary_root / "catalog.json"
    _write(
        up_boundary_catalog,
        {
            "schema": "jit_upstream_boundary_catalog_v1",
            "protocol_sha256": boundary_protocol_sha,
            "split": "train",
            "entries": [
                {
                    "state_sha256": up_boundary_sha,
                    "source_bank": "boundary_bank",
                    "snapshot": "snapshots/train",
                    "seed": 1000001,
                    "parent_group_id": "parent-train-up-boundary",
                    "role": "ascending_entry",
                    "tick": 10,
                    "anchor_source_checkpoint": "transition_4988928",
                    "anchor_source_training_transitions": 4988928,
                }
            ],
        },
    )
    up_boundary_protocol = up_boundary_root / "protocol.json"
    _write(
        up_boundary_protocol,
        {
            "schema": "jit_upstream_boundary_protocol_v1",
            "protocol_sha256": boundary_protocol_sha,
            "frozen_pi_up_actor_sha256": UP_ACTOR,
            "frozen_manifest_sha256": frozen_sha,
            "nominal_catalog_sha256": file_sha256(up_nominal_catalog),
            "nominal_labels_sha256": file_sha256(up_nominal_labels),
            "xml_sha256": XML_SHA,
            "split": "train",
            "train_seeds": [1000001],
        },
    )
    up_boundary_labels = up_boundary_root / "labels.json"
    boundary_label = _label(
        candidate="up-boundary",
        state_sha=up_boundary_sha,
        split="train",
        phase="upstream",
        source_bank="boundary_bank",
        snapshot="snapshots/train",
        marker=0.75,
        protocol="7" * 64,
    )
    boundary_label["boundary_protocol_sha256"] = boundary_protocol_sha
    _write(up_boundary_labels, [boundary_label])

    down_root = tmp_path / "down"
    _, down_sha = _snapshot(down_root, "bank_d/snapshots/train", 1.25)
    down_catalog = down_root / "catalog.json"
    _write(
        down_catalog,
        {
            "entries": [
                {
                    "state_sha256": down_sha,
                    "source_bank": "bank_d",
                    "snapshot": "snapshots/train",
                    "seed": 1000001,
                    "parent_group_id": "parent-train-down-train",
                    "role": "nearest_apex",
                    "tick": 10,
                }
            ]
        },
    )
    down_protocol_sha = "8" * 64
    down_protocol = down_root / "protocol.json"
    _write(
        down_protocol,
        {
            "schema": "jit_continuation_labels_v1",
            "target": "V_down",
            "protocol_sha256": down_protocol_sha,
            "catalog_sha256": file_sha256(down_catalog),
            "expert_actor_sha256": DOWN_ACTOR,
            "frozen_manifest_sha256": frozen_sha,
            "xml_sha256": XML_SHA,
            "train_seeds": [1000001],
            "validation_seeds": [1000006],
            "test_seeds": [1000007],
        },
    )
    down_labels = down_root / "labels.json"
    _write(
        down_labels,
        [
            _label(
                candidate="down-train",
                state_sha=down_sha,
                split="train",
                phase="downstream",
                source_bank="bank_d",
                snapshot="snapshots/train",
                marker=1.25,
                protocol=down_protocol_sha,
            ),
            _label(
                candidate="down-test",
                state_sha="9" * 64,
                split="test",
                phase="downstream",
                source_bank="missing",
                snapshot="must-not-be-read",
                marker=93.0,
                protocol="wrong-test-protocol",
            ),
        ],
    )

    up_manifest = json.loads((up_model / "manifest.json").read_text())
    up_manifest.update(
        nominal_labels_sha256=file_sha256(up_nominal_labels),
        boundary_train_labels_sha256=file_sha256(up_boundary_labels),
    )
    up_manifest.pop("manifest_sha256")
    up_manifest["manifest_sha256"] = canonical_sha256(up_manifest)
    _write(up_model / "manifest.json", up_manifest)
    down_manifest = json.loads((down_model / "manifest.json").read_text())
    down_manifest.update(
        labels_sha256=file_sha256(down_labels),
        continuation_protocol_sha256=down_protocol_sha,
        xml_sha256=XML_SHA,
    )
    down_manifest.pop("manifest_sha256")
    down_manifest["manifest_sha256"] = canonical_sha256(down_manifest)
    _write(down_model / "manifest.json", down_manifest)

    return SoftTubeInputs(
        frozen_experts=frozen_path,
        up_model_dir=up_model,
        down_model_dir=down_model,
        up_nominal_labels=up_nominal_labels,
        up_nominal_catalog=up_nominal_catalog,
        up_nominal_protocol=up_nominal_protocol,
        up_boundary_labels=up_boundary_labels,
        up_boundary_catalog=up_boundary_catalog,
        up_boundary_protocol=up_boundary_protocol,
        down_labels=down_labels,
        down_catalog=down_catalog,
        down_protocol=down_protocol,
    )


def test_builder_scores_only_train_rows_with_the_phase_local_model(tmp_path):
    from jit_dvgc.soft_tube import build_soft_tube

    inputs = _fixture(tmp_path)
    calls = []

    def score_up(_model, observations):
        calls.append(("upstream", np.asarray(observations).copy()))
        return np.asarray([0.2, 0.8])

    def score_down(_model, observations):
        calls.append(("downstream", np.asarray(observations).copy()))
        return np.asarray([0.4])

    output = tmp_path / "soft_tube"
    result = build_soft_tube(
        inputs, output, score_up=score_up, score_down=score_down
    )

    entries = result.entries
    assert [entry["phase"] for entry in entries] == [
        "upstream",
        "upstream",
        "downstream",
    ]
    assert [entry["value_score"] for entry in entries] == pytest.approx([0.2, 0.8, 0.4])
    assert [entry["sampling_weight"] for entry in entries] == pytest.approx(
        [0.24, 0.81, 0.43]
    )
    assert [name for name, _ in calls] == ["upstream", "downstream"]
    assert calls[0][1].shape == (2, 2)
    assert calls[1][1].shape == (1, 2)
    assert not any(float(value) >= 90.0 for _, array in calls for value in array.ravel())
    assert {entry["split"] for entry in entries} == {"train"}
    assert result.manifest["test_data_used"] is False
    assert result.manifest["validation_data_used"] is False
    assert result.manifest["training_transitions"] == 0
    assert result.manifest["environment_interactions"] == 0
    assert result.manifest["certified_safe"] is False
    assert result.manifest["training_guidance_only"] is True
    assert json.loads((output / "entries.json").read_text()) == list(entries)


def test_builder_fails_closed_when_a_value_payload_hash_drifts(tmp_path):
    from jit_dvgc.soft_tube import build_soft_tube

    inputs = _fixture(tmp_path)
    (inputs.up_model_dir / "params.msgpack").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="V_up params SHA-256 mismatch"):
        build_soft_tube(inputs, tmp_path / "soft_tube")


def test_builder_rejects_one_physical_state_assigned_to_both_phases(tmp_path):
    from jit_dvgc.soft_tube import build_soft_tube

    inputs = _fixture(tmp_path)
    up_catalog = json.loads(inputs.up_nominal_catalog.read_text())
    up_row = up_catalog["entries"][0]
    down_catalog = json.loads(inputs.down_catalog.read_text())
    down_row = down_catalog["entries"][0]
    source = inputs.up_nominal_catalog.parent / up_row["source_bank"] / up_row["snapshot"]
    target = inputs.down_catalog.parent / down_row["source_bank"] / down_row["snapshot"]
    (target / "snapshot.pkl").write_bytes((source / "snapshot.pkl").read_bytes())
    (target / "identity.json").write_bytes((source / "identity.json").read_bytes())
    down_row["state_sha256"] = up_row["state_sha256"]
    _write(inputs.down_catalog, down_catalog)
    protocol = json.loads(inputs.down_protocol.read_text())
    protocol["catalog_sha256"] = file_sha256(inputs.down_catalog)
    _write(inputs.down_protocol, protocol)
    labels = json.loads(inputs.down_labels.read_text())
    labels[0]["state_sha256"] = up_row["state_sha256"]
    _write(inputs.down_labels, labels)
    manifest = json.loads((inputs.down_model_dir / "manifest.json").read_text())
    manifest["labels_sha256"] = file_sha256(inputs.down_labels)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _write(inputs.down_model_dir / "manifest.json", manifest)

    with pytest.raises(ValueError, match="conflicting phase"):
        build_soft_tube(
            inputs,
            tmp_path / "soft_tube",
            score_up=lambda _m, x: np.full(len(x), 0.5),
            score_down=lambda _m, x: np.full(len(x), 0.5),
        )


def test_builder_requires_every_selected_train_snapshot(tmp_path):
    from jit_dvgc.soft_tube import build_soft_tube

    inputs = _fixture(tmp_path)
    catalog = json.loads(inputs.up_boundary_catalog.read_text())
    row = catalog["entries"][0]
    snapshot = inputs.up_boundary_catalog.parent / row["source_bank"] / row["snapshot"]
    (snapshot / "snapshot.pkl").unlink()
    (snapshot / "identity.json").unlink()
    snapshot.rmdir()
    with pytest.raises(FileNotFoundError, match="selected TRAIN snapshot"):
        build_soft_tube(
            inputs,
            tmp_path / "soft_tube",
            score_up=lambda _m, x: np.full(len(x), 0.5),
            score_down=lambda _m, x: np.full(len(x), 0.5),
        )


def test_cli_builds_a_loadable_soft_tube_artifact(tmp_path):
    from jit_dvgc.soft_tube import load_soft_tube

    inputs = _fixture(tmp_path)
    output = tmp_path / "soft_tube_cli"
    command = [
        sys.executable,
        "JIT/cli/build_soft_tube.py",
        "--frozen-experts",
        str(inputs.frozen_experts),
        "--up-model-dir",
        str(inputs.up_model_dir),
        "--down-model-dir",
        str(inputs.down_model_dir),
        "--up-nominal-labels",
        str(inputs.up_nominal_labels),
        "--up-nominal-catalog",
        str(inputs.up_nominal_catalog),
        "--up-nominal-protocol",
        str(inputs.up_nominal_protocol),
        "--up-boundary-labels",
        str(inputs.up_boundary_labels),
        "--up-boundary-catalog",
        str(inputs.up_boundary_catalog),
        "--up-boundary-protocol",
        str(inputs.up_boundary_protocol),
        "--down-labels",
        str(inputs.down_labels),
        "--down-catalog",
        str(inputs.down_catalog),
        "--down-protocol",
        str(inputs.down_protocol),
        "--output-dir",
        str(output),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "JIT/src"
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    artifact = load_soft_tube(output)
    assert artifact.manifest["entry_count"] == 3
