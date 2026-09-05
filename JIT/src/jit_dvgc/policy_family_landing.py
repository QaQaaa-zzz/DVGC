"""Any-controller landing labels for a frozen unified-policy family."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax

from .evidence_integrity import read_verified_protocol, validate_label_row
from .checkpoint import load_checkpoint
from .config import file_sha256
from .ppo import make_checkpoint_policy
from .unified_continuation_labels import label_unified_continuations
from .unified_continuation_shards import (
    contiguous_shard_bounds,
    label_unified_continuation_shard,
    merge_unified_continuation_shards,
)
from .unified_formal import build_unified_formal_environment
from .unified_policy_freeze import load_frozen_unified_manifest
from .unified_training import checkpoint_identity


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def required_policy_family_label_shards(
    candidate_count: int, max_candidates_per_process: int = 600
) -> int:
    candidate_count = int(candidate_count)
    maximum = int(max_candidates_per_process)
    if candidate_count <= 0 or maximum <= 0:
        raise ValueError("policy-family shard sizes must be positive")
    return int(math.ceil(candidate_count / maximum))


def merge_any_policy_landing_labels(
    labels_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """OR aligned first-landing labels while retaining every policy outcome."""
    if not labels_by_policy:
        raise ValueError("policy-family landing merge requires at least one policy")
    policy_names = sorted(str(name) for name in labels_by_policy)
    row_sets = {name: list(labels_by_policy[name]) for name in policy_names}
    count = len(row_sets[policy_names[0]])
    if count <= 0 or any(len(rows) != count for rows in row_sets.values()):
        raise ValueError("policy-family landing label counts differ")

    members = []
    for name in policy_names:
        first = row_sets[name][0]
        members.append(
            {
                "name": name,
                "actor_sha256": str(first["evaluator_actor_sha256"]),
                "payload_sha256": str(first["evaluator_payload_sha256"]),
            }
        )
    identity = {
        "policy_names": policy_names,
        "members": members,
        "aggregation": "positive_if_any_policy_reaches_first_valid_landing",
    }
    identity["actor_family_sha256"] = _canonical_sha256(
        {"member_actor_sha256": [row["actor_sha256"] for row in members]}
    )
    identity["payload_family_sha256"] = _canonical_sha256(
        {"member_payload_sha256": [row["payload_sha256"] for row in members]}
    )
    identity["policy_family_sha256"] = _canonical_sha256(identity)

    merged: list[dict[str, Any]] = []
    identity_fields = (
        "candidate_id",
        "state_sha256",
        "phase",
        "phase_index",
        "parent_group_id",
        "parent_state_sha256",
        "snapshot",
        "source_bank",
        "snapshot_context_sha256",
    )
    seen_candidates = set()
    for index in range(count):
        rows = {name: row_sets[name][index] for name in policy_names}
        base = rows[policy_names[0]]
        key = (base.get("candidate_id"), base.get("state_sha256"))
        if key in seen_candidates:
            raise ValueError("duplicate policy-family candidate")
        seen_candidates.add(key)
        for name, row in rows.items():
            if any(row.get(field) != base.get(field) for field in identity_fields):
                raise ValueError(
                    f"policy-family candidate identity drift at index {index}: {name}"
                )
            first = row_sets[name][0]
            validate_label_row(row, name=name, actor=first["evaluator_actor_sha256"],
                               payload=first["evaluator_payload_sha256"],
                               criterion="first_valid_landing")
        successful = [name for name, row in rows.items() if int(row["label"]) == 1]
        per_policy = {
            name: {
                "label": int(row["label"]),
                "outcome_class": str(row["outcome_class"]),
                "environment_interactions": int(row["environment_interactions"]),
                "evaluator_actor_sha256": str(row["evaluator_actor_sha256"]),
                "evaluator_payload_sha256": str(row["evaluator_payload_sha256"]),
            }
            for name, row in rows.items()
        }
        merged.append(
            {
                **dict(base),
                "label": int(bool(successful)),
                "continuation_success": bool(successful),
                "outcome_class": (
                    "first_valid_landing_by_any_policy"
                    if successful
                    else "no_policy_reached_first_valid_landing"
                ),
                "successful_policy_names": successful,
                "per_policy_outcomes": per_policy,
                "continuation_policy_family_sha256": identity[
                    "policy_family_sha256"
                ],
                "policy_actor_sha256": identity["actor_family_sha256"],
                "policy_payload_sha256": identity["payload_family_sha256"],
                "policy_identity_kind": "frozen_policy_family",
                "continuation_policy_family_aggregation": identity["aggregation"],
            }
        )
    return merged, identity


def _write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _archive_incomplete_evaluator(
    output_dir: Path,
    *,
    evaluator_name: str,
) -> Path | None:
    """Preserve a partial evaluator directory and free its canonical retry path."""
    output = Path(output_dir)
    if not output.exists():
        return None
    if not output.is_dir():
        raise FileExistsError(f"landing evaluator output is not a directory: {output}")
    summary_path = output / "summary.json"
    if summary_path.is_file() and json.loads(summary_path.read_text()).get("status") in {"completed", "completed_shard"}:
        raise ValueError("refusing to archive a completed evaluator; verify or use a new output")
    attempt = 1
    while True:
        archive = output.with_name(
            f"{output.name}_incomplete_attempt_{attempt:03d}"
        )
        if not archive.exists():
            break
        attempt += 1
    preserved_files = sorted(
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    )
    output.rename(archive)
    _write_json(
        archive / "incomplete_attempt.json",
        {
            "schema": "jit_policy_family_incomplete_evaluator_attempt_v1",
            "status": "preserved_before_evaluator_retry",
            "evaluator_policy_name": str(evaluator_name),
            "original_output_dir": str(output),
            "archived_output_dir": str(archive),
            "preserved_files": preserved_files,
            "completed_result_reused": False,
            "retry_may_write_canonical_output_dir": True,
            "training_transitions": 0,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        },
    )
    return archive


def _load_frozen(path: Path) -> tuple[dict[str, Any], str]:
    frozen_path = Path(path)
    frozen = load_frozen_unified_manifest(frozen_path)
    return dict(frozen["policy"]), file_sha256(frozen_path)


def run_policy_family_evaluator_shard(
    *,
    catalog_path: Path,
    acquisition_frozen_policy: Path,
    evaluator_frozen_policy: Path,
    output_dir: Path,
    shard_index: int,
    shard_count: int,
    max_ticks: int = 400,
    protocol_seed: int = 9_521_201,
) -> dict[str, Any]:
    """Run one bounded first-landing evaluator shard in one GPU process."""
    output = Path(output_dir)
    acquisition_record, acquisition_frozen_sha = _load_frozen(
        Path(acquisition_frozen_policy)
    )
    evaluator_record, evaluator_frozen_sha = _load_frozen(
        Path(evaluator_frozen_policy)
    )
    if acquisition_record["xml_sha256"] != evaluator_record["xml_sha256"]:
        raise ValueError("policy-family shard acquisition/evaluator XML mismatch")
    contract = _requested_contract(catalog_path, acquisition_record, evaluator_record,
                                   evaluator_frozen_sha, max_ticks, protocol_seed)
    catalog = json.loads(Path(catalog_path).read_text())
    start, stop = contiguous_shard_bounds(int(catalog["candidate_count"]), shard_index, shard_count)
    if (output / "summary.json").is_file():
        report = json.loads((output / "summary.json").read_text())
        if report.get("status") != "completed_shard":
            raise RuntimeError(f"existing evaluator shard is incomplete: {output}")
        protocol = _verify_cached_contract(output, contract)
        execution = json.loads((output / "execution.json").read_text())
        for obj in (report, execution):
            for key, value in {"shard_index": shard_index, "shard_count": shard_count,
                               "candidate_start_index": start, "candidate_stop_index_exclusive": stop,
                               "logical_protocol_sha256": protocol["protocol_sha256"]}.items():
                if obj.get(key) != value:
                    raise ValueError(f"cached shard {key} drift")
        if execution.get("status") != "completed":
            raise ValueError("cached shard execution incomplete")
        if report.get("labels_file_sha256") is not None and report["labels_file_sha256"] != file_sha256(output / "labels.json"):
            raise ValueError("cached shard label file hash drift")
        labels = json.loads((output / "labels.json").read_text())
        _verify_cached_rows(labels, catalog["entries"][start:stop], protocol)
        if report.get("candidate_count") != len(labels):
            raise ValueError("cached shard count drift")
        for index, row in enumerate(labels, start):
            if row.get("candidate_index") != index or row.get("policy_key_candidate_index") != index:
                raise ValueError("cached shard global candidate index drift")
        return report
    config, _artifact, env = build_unified_formal_environment(
        Path(str(evaluator_record["formal_config"]))
    )
    payload = load_checkpoint(
        Path(str(evaluator_record["checkpoint"])),
        expected=checkpoint_identity(config, env),
    )
    policy = make_checkpoint_policy(env, payload, deterministic=True)
    from .frontier_label_shard_runner import _build_memory_stable_step

    step_fn = _build_memory_stable_step(env)
    return label_unified_continuation_shard(
        Path(catalog_path),
        output,
        env=env,
        policy=policy,
        policy_record=evaluator_record,
        frozen_manifest_sha256=evaluator_frozen_sha,
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        max_ticks=int(max_ticks),
        protocol_seed=int(protocol_seed),
        compiled_step_fn=step_fn,
        acquisition_policy_record=acquisition_record,
        acquisition_frozen_manifest_sha256=acquisition_frozen_sha,
        success_criterion="first_valid_landing",
    )


def _requested_contract(catalog_path, acquisition, evaluator, frozen_sha, max_ticks, seed):
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog.get("status") != "completed" or catalog.get("candidate_count") != len(catalog.get("entries", [])):
        raise ValueError("requested catalog is not completed/aligned")
    return {
        "candidate_catalog_file_sha256": file_sha256(Path(catalog_path)),
        "candidate_catalog_protocol_sha256": catalog["protocol_sha256"],
        "candidate_count": catalog["candidate_count"],
        "policy_name": evaluator["name"],
        "policy_actor_sha256": evaluator["actor_sha256"],
        "policy_payload_sha256": evaluator["payload_sha256"],
        "policy_formal_config_sha256": evaluator["formal_config_sha256"],
        "frozen_unified_manifest_sha256": frozen_sha,
        "acquisition_policy_name": acquisition["name"],
        "acquisition_policy_actor_sha256": acquisition["actor_sha256"],
        "acquisition_policy_payload_sha256": acquisition["payload_sha256"],
        "success_criterion": "first_valid_landing",
        "protocol_seed": int(seed), "max_ticks_per_candidate": int(max_ticks),
    }


def _verify_cached_contract(output, expected):
    protocol = read_verified_protocol(Path(output) / "protocol.json")
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"cached evaluator requested {key} drift")
    return protocol


def _verify_cached_rows(labels, candidates, protocol):
    if not isinstance(labels, list) or len(labels) != len(candidates):
        raise ValueError("cached evaluator label coverage drift")
    for row, candidate in zip(labels, candidates, strict=True):
        for field in ("candidate_id", "state_sha256", "phase", "phase_index", "parent_group_id",
                      "parent_state_sha256", "source_bank", "snapshot", "snapshot_context_sha256"):
            if row.get(field) != candidate.get(field):
                raise ValueError(f"cached evaluator candidate {field} drift")
        validate_label_row(row, name=protocol["policy_name"], actor=protocol["policy_actor_sha256"],
                           payload=protocol["policy_payload_sha256"], criterion=protocol["success_criterion"])
        if row.get("label_protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("cached evaluator row protocol drift")
        if row.get("acquisition_protocol_sha256") != protocol["candidate_catalog_protocol_sha256"]:
            raise ValueError("cached evaluator acquisition protocol drift")
        if row["environment_interactions"] > protocol["max_ticks_per_candidate"]:
            raise ValueError("cached evaluator horizon exceeded")


def merge_policy_family_evaluator_shards(
    *, catalog_path: Path, shard_dirs: Sequence[Path], output_dir: Path,
    evaluator_name: str, acquisition_frozen_policy: Path,
    evaluator_frozen_policy: Path, max_ticks: int = 400, protocol_seed: int = 9_521_201,
) -> dict[str, Any]:
    """Validate in a temporary directory; publish only a complete requested result."""
    acquisition, _ = _load_frozen(acquisition_frozen_policy)
    evaluator, frozen_sha = _load_frozen(evaluator_frozen_policy)
    if evaluator["name"] != evaluator_name:
        raise ValueError("requested evaluator name drift")
    contract = _requested_contract(catalog_path, acquisition, evaluator, frozen_sha, max_ticks, protocol_seed)
    # Validate even when a canonical cache exists: supplied shards must match the request.
    for directory in shard_dirs:
        _verify_cached_contract(directory, contract)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".landing-merge-", dir=output.parent) as temporary:
        staged = Path(temporary) / "result"
        report = merge_unified_continuation_shards(Path(catalog_path), shard_dirs, staged)
        _load_completed_evaluator(staged, evaluator=evaluator, catalog_path=catalog_path,
                                  expected_contract=contract)
        existing = _load_completed_evaluator(output, evaluator=evaluator, catalog_path=catalog_path,
                                              expected_contract=contract)
        if existing is not None:
            if json.loads((staged / "labels.json").read_text()) != existing[1]:
                raise ValueError("completed evaluator differs from supplied shards")
            return existing[0]
        _archive_incomplete_evaluator(output, evaluator_name=evaluator_name)
        staged.rename(output)
    return report


def _load_completed_evaluator(
    output_dir: Path,
    *,
    evaluator: Mapping[str, Any],
    catalog_path: Path,
    expected_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Load one closed evaluator result so an interrupted family run can resume."""
    summary_path = Path(output_dir) / "summary.json"
    labels_path = Path(output_dir) / "labels.json"
    if not summary_path.is_file():
        return None
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        return None
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if report.get("labels_file_sha256") is not None and report["labels_file_sha256"] != file_sha256(labels_path):
        raise ValueError("completed label file hash drift")
    protocol = _verify_cached_contract(output_dir, expected_contract)
    if report.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("completed evaluator summary/protocol drift")
    catalog = json.loads(Path(catalog_path).read_text())
    _verify_cached_rows(labels, catalog["entries"], protocol)
    if report.get("evaluator_policy_name") != evaluator.get("name"):
        raise ValueError("completed landing evaluator name drift")
    if report.get("policy_actor_sha256") != evaluator.get("actor_sha256"):
        raise ValueError("completed landing evaluator actor drift")
    if report.get("policy_payload_sha256") != evaluator.get("payload_sha256"):
        raise ValueError("completed landing evaluator payload drift")
    if report.get("success_criterion") != "first_valid_landing":
        raise ValueError("completed landing evaluator criterion drift")
    if len(labels) != int(report.get("label_count", -1)):
        raise ValueError("completed landing evaluator label count drift")
    return report, [dict(row) for row in labels]


def label_policy_family_first_landing(
    *,
    catalog_path: Path,
    acquisition_frozen_policy: Path,
    evaluator_frozen_policies: Sequence[Path],
    output_dir: Path,
    max_ticks: int = 400,
    protocol_seed: int = 9_521_201,
) -> dict[str, Any]:
    """Evaluate one acquisition catalog under every frozen family member."""
    acquisition_record, acquisition_frozen_sha = _load_frozen(
        Path(acquisition_frozen_policy)
    )
    evaluators = []
    seen_names = set()
    for frozen_path in evaluator_frozen_policies:
        record, frozen_sha = _load_frozen(Path(frozen_path))
        name = str(record["name"])
        if name in seen_names:
            raise ValueError(f"duplicate continuation evaluator: {name}")
        seen_names.add(name)
        evaluators.append((name, Path(frozen_path), record, frozen_sha))
    if not evaluators:
        raise ValueError("policy-family landing labeling requires evaluators")
    if {name for name, *_ in evaluators} != {"pi_0", "pi_1", "pi_2"}:
        raise ValueError("active landing policy family must be exactly pi_0/pi_1/pi_2")
    xmls = {str(record["xml_sha256"]) for _, _, record, _ in evaluators}
    xmls.add(str(acquisition_record["xml_sha256"]))
    if len(xmls) != 1:
        raise ValueError("policy-family landing evaluators do not share one XML")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_policy_root = output / "per_policy"
    per_policy_root.mkdir(exist_ok=True)
    labels_by_policy: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    try:
        for name, frozen_path, record, frozen_sha in evaluators:
            policy_output = per_policy_root / name
            completed = _load_completed_evaluator(
                policy_output,
                evaluator=record,
                catalog_path=catalog_path,
                expected_contract=_requested_contract(catalog_path, acquisition_record, record, frozen_sha, max_ticks, protocol_seed),
            )
            if completed is not None:
                reports[name], labels_by_policy[name] = completed
                continue
            _archive_incomplete_evaluator(
                policy_output,
                evaluator_name=name,
            )
            config, _artifact, env = build_unified_formal_environment(
                Path(str(record["formal_config"]))
            )
            payload = load_checkpoint(
                Path(str(record["checkpoint"])),
                expected=checkpoint_identity(config, env),
            )
            policy = make_checkpoint_policy(env, payload, deterministic=True)
            report = label_unified_continuations(
                Path(catalog_path),
                policy_output,
                env=env,
                policy=policy,
                policy_record=record,
                frozen_manifest_sha256=frozen_sha,
                max_ticks=int(max_ticks),
                protocol_seed=int(protocol_seed),
                acquisition_policy_record=acquisition_record,
                acquisition_frozen_manifest_sha256=acquisition_frozen_sha,
                success_criterion="first_valid_landing",
            )
            reports[name] = report
            labels = json.loads(
                (policy_output / "labels.json").read_text(encoding="utf-8")
            )
            if not isinstance(labels, list):
                raise ValueError(f"{name} landing labels are not a JSON array")
            labels_by_policy[name] = labels
            del policy, payload, env, _artifact, config
            jax.clear_caches()
            gc.collect()

        merged, family = merge_any_policy_landing_labels(labels_by_policy)
        phase_counts = {}
        for phase in ("upstream", "downstream"):
            rows = [row for row in merged if row["phase"] == phase]
            positives = sum(int(row["label"]) for row in rows)
            phase_counts[phase] = {
                "candidate_count": len(rows),
                "positive_count": positives,
                "negative_count": len(rows) - positives,
                "parent_group_count": len(
                    {str(row["parent_group_id"]) for row in rows}
                ),
            }
        report = {
            "schema": "jit_policy_family_first_landing_labels_v1",
            "status": "completed",
            "candidate_catalog": str(Path(catalog_path)),
            "candidate_catalog_file_sha256": file_sha256(Path(catalog_path)),
            "acquisition_policy": {
                "name": str(acquisition_record["name"]),
                "actor_sha256": str(acquisition_record["actor_sha256"]),
                "payload_sha256": str(acquisition_record["payload_sha256"]),
                "frozen_manifest": str(Path(acquisition_frozen_policy)),
                "frozen_manifest_sha256": acquisition_frozen_sha,
            },
            "continuation_policy_family": family,
            "success_criterion": "first_valid_landing_before_physical_failure",
            "post_landing_recovery_required": False,
            "candidate_count": len(merged),
            "positive_count": sum(int(row["label"]) for row in merged),
            "negative_count": sum(1 - int(row["label"]) for row in merged),
            "phase_counts": phase_counts,
            "per_policy_reports": {
                name: {
                    "path": str((per_policy_root / name / "summary.json").resolve()),
                    "positive_count": int(reports[name]["positive_count"]),
                    "negative_count": int(reports[name]["negative_count"]),
                    "environment_interactions": int(
                        reports[name]["environment_interactions"]
                    ),
                }
                for name in sorted(reports)
            },
            "environment_interactions": sum(
                int(item["environment_interactions"]) for item in reports.values()
            ),
            "execution_mode": "one_gpu_serial_per_policy_with_first_landing_early_stop",
            "training_transitions": 0,
            "expert_switching_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        }
        _write_json(output / "labels.json", merged)
        _write_json(output / "summary.json", report)
        return report
    except BaseException as exc:
        _write_json(
            output / "failure.json",
            {
                "schema": "jit_policy_family_first_landing_labels_v1",
                "status": "engineering_error",
                "completed_policy_names": sorted(reports),
                "error": f"{type(exc).__name__}: {exc}",
                "training_transitions": 0,
            },
        )
        raise
