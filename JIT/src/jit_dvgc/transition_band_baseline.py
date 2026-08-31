"""Build an audited accumulated TRAIN baseline for transition-band search.

This capability merges already completed frozen-policy continuation-label banks
without spending interactions or training transitions.  Exact physical states
are deduplicated by SHA-256; conflicting labels are rejected.  The resulting
artifact is only a TRAIN search baseline and carries no Tube/JCE/JEL claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .config import file_sha256


BASELINE_BUILD_CONFIG_SCHEMA = "jit_unified_transition_band_baseline_build_config_v1"
BASELINE_PROTOCOL_SCHEMA = "jit_unified_transition_band_baseline_protocol_v1"
BASELINE_SUMMARY_SCHEMA = "jit_unified_transition_band_baseline_v1"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def load_transition_band_baseline_config(path: Path) -> dict[str, Any]:
    raw = _read_json(Path(path))
    if raw.get("schema") != BASELINE_BUILD_CONFIG_SCHEMA:
        raise ValueError("unsupported transition-band baseline config schema")
    protocol = raw.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("transition-band baseline config requires protocol")
    if protocol.get("schema") != BASELINE_PROTOCOL_SCHEMA:
        raise ValueError("transition-band baseline protocol schema drift")
    if protocol.get("status") != "predeclared":
        raise ValueError("transition-band baseline protocol must be predeclared")
    if int(protocol.get("iteration", -1)) < 0:
        raise ValueError("transition-band baseline iteration must be nonnegative")
    sources = protocol.get("sources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("transition-band baseline requires at least two completed sources")
    names: set[str] = set()
    previous_duration = 0
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("transition-band baseline source must be an object")
        for key in (
            "name",
            "labels_path",
            "expected_label_protocol_sha256",
            "expected_acquisition_protocol_sha256",
            "expected_candidate_count",
            "expected_positive_count",
            "expected_negative_count",
            "maximum_duration",
        ):
            if key not in source:
                raise ValueError(f"transition-band baseline source missing {key}")
        name = str(source["name"])
        if not name or name in names:
            raise ValueError("transition-band baseline source names must be unique")
        names.add(name)
        count = int(source["expected_candidate_count"])
        positive = int(source["expected_positive_count"])
        negative = int(source["expected_negative_count"])
        if count <= 0 or positive < 0 or negative < 0 or positive + negative != count:
            raise ValueError("transition-band baseline source count declaration invalid")
        maximum_duration = int(source["maximum_duration"])
        if maximum_duration <= previous_duration:
            raise ValueError("transition-band baseline source durations must increase")
        previous_duration = maximum_duration
    expected_sha = str(raw.get("expected_protocol_sha256", ""))
    actual_sha = _canonical_sha256(protocol)
    if expected_sha != actual_sha:
        raise ValueError("transition-band baseline protocol SHA-256 drift")
    expected = raw.get("expected_output")
    if not isinstance(expected, Mapping):
        raise ValueError("transition-band baseline config requires expected_output")
    if int(expected.get("maximum_duration", 0)) != previous_duration:
        raise ValueError("transition-band baseline expected maximum duration drift")
    if not str(raw.get("output_dir", "")):
        raise ValueError("transition-band baseline config requires output_dir")
    return raw


def _validate_source(
    source: Mapping[str, Any],
    *,
    policy_actor_sha256: str,
    policy_payload_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_path = Path(str(source["labels_path"]))
    summary_path = labels_path.parent / "summary.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    summary = _read_json(summary_path)
    if not isinstance(labels, list):
        raise ValueError(f"completed baseline source labels must be an array: {labels_path}")
    if summary.get("status") != "completed":
        raise ValueError(f"completed baseline source is not completed: {labels_path}")
    if summary.get("schema") != "jit_unified_continuation_labels_v1":
        raise ValueError("completed baseline source label schema drift")
    if summary.get("artifact_role") != "pi_k_conditioned_expansion_train_labels":
        raise ValueError("completed baseline source artifact role drift")
    if summary.get("split") != "train":
        raise ValueError("completed baseline source is not TRAIN")
    if summary.get("training_transitions") != 0:
        raise ValueError("completed baseline source unexpectedly trained")
    if summary.get("expert_switching_used") is not False:
        raise ValueError("completed baseline source used expert switching")
    if summary.get("validation_data_used") is not False:
        raise ValueError("completed baseline source used validation data")
    if summary.get("test_data_used") is not False:
        raise ValueError("completed baseline source used TEST data")
    if summary.get("final_evaluation_data_used") is not False:
        raise ValueError("completed baseline source used final evaluation data")
    if summary.get("policy_actor_sha256") != policy_actor_sha256:
        raise ValueError("completed baseline source actor identity drift")
    if summary.get("policy_payload_sha256") != policy_payload_sha256:
        raise ValueError("completed baseline source payload identity drift")
    if summary.get("protocol_sha256") != source["expected_label_protocol_sha256"]:
        raise ValueError("completed baseline source label protocol drift")
    if summary.get("candidate_catalog_protocol_sha256") != source[
        "expected_acquisition_protocol_sha256"
    ]:
        raise ValueError("completed baseline source acquisition protocol drift")
    for key, summary_key in (
        ("expected_candidate_count", "candidate_count"),
        ("expected_positive_count", "positive_count"),
        ("expected_negative_count", "negative_count"),
    ):
        if int(summary.get(summary_key, -1)) != int(source[key]):
            raise ValueError(f"completed baseline source {summary_key} drift")
    if len(labels) != int(source["expected_candidate_count"]):
        raise ValueError("completed baseline source label count drift")

    validated: list[dict[str, Any]] = []
    for row in labels:
        if row.get("split") != "train":
            raise ValueError("completed baseline source contains non-TRAIN label")
        if int(row.get("label", -1)) not in (0, 1):
            raise ValueError("completed baseline source contains invalid binary label")
        if row.get("policy_actor_sha256") != policy_actor_sha256:
            raise ValueError("completed baseline label actor identity drift")
        if row.get("policy_payload_sha256") != policy_payload_sha256:
            raise ValueError("completed baseline label payload identity drift")
        if row.get("label_protocol_sha256") != source["expected_label_protocol_sha256"]:
            raise ValueError("completed baseline row label protocol drift")
        if row.get("acquisition_protocol_sha256") != source[
            "expected_acquisition_protocol_sha256"
        ]:
            raise ValueError("completed baseline row acquisition protocol drift")
        if not str(row.get("state_sha256", "")):
            raise ValueError("completed baseline row missing physical-state SHA-256")
        copied = dict(row)
        copied["baseline_source"] = str(source["name"])
        copied["baseline_source_maximum_duration"] = int(source["maximum_duration"])
        validated.append(copied)

    identity = {
        "name": str(source["name"]),
        "labels_path": str(labels_path),
        "labels_file_sha256": file_sha256(labels_path),
        "summary_file_sha256": file_sha256(summary_path),
        "label_protocol_sha256": str(summary["protocol_sha256"]),
        "acquisition_protocol_sha256": str(summary["candidate_catalog_protocol_sha256"]),
        "candidate_count": len(validated),
        "positive_count": int(summary["positive_count"]),
        "negative_count": int(summary["negative_count"]),
        "environment_interactions": int(summary["environment_interactions"]),
        "maximum_duration": int(source["maximum_duration"]),
    }
    return validated, identity


def build_transition_band_baseline(config_path: Path) -> dict[str, Any]:
    config = load_transition_band_baseline_config(config_path)
    protocol = dict(config["protocol"])
    protocol_sha = _canonical_sha256(protocol)
    actor_sha = str(protocol["policy_actor_sha256"])
    payload_sha = str(protocol["policy_payload_sha256"])

    source_sets: list[list[dict[str, Any]]] = []
    source_identities: list[dict[str, Any]] = []
    for source in protocol["sources"]:
        rows, identity = _validate_source(
            source,
            policy_actor_sha256=actor_sha,
            policy_payload_sha256=payload_sha,
        )
        source_sets.append(rows)
        source_identities.append(identity)

    by_state: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for rows in source_sets:
        for row in rows:
            state_sha = str(row["state_sha256"])
            previous = by_state.get(state_sha)
            if previous is not None:
                if int(previous["label"]) != int(row["label"]) or previous["phase"] != row["phase"]:
                    raise ValueError("duplicate physical state has conflicting baseline label")
                duplicate_count += 1
                continue
            by_state[state_sha] = row
    merged = list(by_state.values())
    merged.sort(key=lambda row: (str(row["phase"]), str(row["state_sha256"])))
    positive = sum(int(row["label"]) for row in merged)
    negative = len(merged) - positive
    expected = config["expected_output"]
    if len(merged) != int(expected["candidate_count"]):
        raise ValueError("transition-band baseline merged candidate count drift")
    if positive != int(expected["positive_count"]):
        raise ValueError("transition-band baseline merged positive count drift")
    if negative != int(expected["negative_count"]):
        raise ValueError("transition-band baseline merged negative count drift")

    acquisition_sources = {
        "schema": "jit_unified_transition_band_baseline_acquisition_sources_v1",
        "source_acquisition_protocol_sha256": [
            str(source["expected_acquisition_protocol_sha256"])
            for source in protocol["sources"]
        ],
    }
    acquisition_composite_sha = _canonical_sha256(acquisition_sources)
    if acquisition_composite_sha != config["expected_composite_acquisition_protocol_sha256"]:
        raise ValueError("transition-band baseline composite acquisition SHA-256 drift")

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline_protocol.json").write_text(
        json.dumps({**protocol, "protocol_sha256": protocol_sha}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "labels.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    phase_counts: dict[str, dict[str, int]] = {}
    for phase in ("upstream", "downstream"):
        rows = [row for row in merged if row["phase"] == phase]
        pos = sum(int(row["label"]) for row in rows)
        phase_counts[phase] = {
            "candidate_count": len(rows),
            "positive_count": pos,
            "negative_count": len(rows) - pos,
        }
    summary = {
        "schema": BASELINE_SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "accumulated_transition_band_train_baseline",
        "split": "train",
        "iteration": int(protocol["iteration"]),
        "policy_actor_sha256": actor_sha,
        "policy_payload_sha256": payload_sha,
        "protocol_sha256": protocol_sha,
        "candidate_catalog_protocol_sha256": acquisition_composite_sha,
        "candidate_count": len(merged),
        "label_count": len(merged),
        "positive_count": positive,
        "negative_count": negative,
        "phase_counts": phase_counts,
        "maximum_duration": int(expected["maximum_duration"]),
        "duplicate_state_count": duplicate_count,
        "sources": source_identities,
        "environment_interactions": 0,
        "source_environment_interactions": sum(
            int(source["environment_interactions"]) for source in source_identities
        ),
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
