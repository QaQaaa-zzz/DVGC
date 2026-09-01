"""Resumable manifest-driven orchestration for JIT envelope iterations.

The runner is deliberately scientifically ignorant: every scientific decision
must already be encoded by a production CLI and a machine-readable artifact.
This module only sequences stages, verifies immutable prerequisites/completion
artifacts, exports declared JSON values, persists state, and stops on failure.
It never deletes artifacts, changes thresholds, consumes TEST data, or silently
reruns an already-completed stage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "jit_iteration_workflow_v1"
STATE_SCHEMA = "jit_iteration_workflow_state_v1"


class WorkflowError(RuntimeError):
    """Raised when orchestration state or a declared artifact gate is invalid."""


@dataclass(frozen=True)
class CompletionGate:
    path: str
    kind: str
    assertions: tuple[Mapping[str, Any], ...]
    exports: Mapping[str, str]


@dataclass(frozen=True)
class WorkflowStage:
    name: str
    command: tuple[str, ...]
    cwd: str
    requires: tuple[Mapping[str, Any], ...]
    completion: CompletionGate


@dataclass(frozen=True)
class WorkflowConfig:
    raw: Mapping[str, Any]
    config_sha256: str
    workflow_name: str
    state_dir: str
    variables: Mapping[str, str]
    environment: Mapping[str, str]
    stages: tuple[WorkflowStage, ...]


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_text(value: Any, *, field: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _parse_completion(raw: Any, *, stage_name: str) -> CompletionGate:
    if not isinstance(raw, Mapping):
        raise ValueError(f"stage {stage_name} completion must be an object")
    if set(raw).difference({"path", "kind", "assertions", "exports"}):
        raise ValueError(f"stage {stage_name} completion contains unknown fields")
    kind = str(raw.get("kind", "json"))
    if kind not in {"file", "directory", "json"}:
        raise ValueError(f"stage {stage_name} completion kind is unsupported")
    assertions_raw = raw.get("assertions", [])
    if not isinstance(assertions_raw, list):
        raise ValueError(f"stage {stage_name} assertions must be a list")
    assertions: list[Mapping[str, Any]] = []
    for index, assertion in enumerate(assertions_raw):
        if not isinstance(assertion, Mapping):
            raise ValueError(f"stage {stage_name} assertion {index} must be an object")
        if set(assertion) != {"pointer", "op", "value"}:
            raise ValueError(f"stage {stage_name} assertion {index} contract drift")
        if assertion["op"] not in {"eq", "ne", "gt", "ge", "lt", "le", "in"}:
            raise ValueError(f"stage {stage_name} assertion {index} op unsupported")
        assertions.append(dict(assertion))
    exports_raw = raw.get("exports", {})
    if not isinstance(exports_raw, Mapping):
        raise ValueError(f"stage {stage_name} exports must be an object")
    exports = {
        _require_text(key, field=f"stage {stage_name} export name"): _require_text(
            value, field=f"stage {stage_name} export pointer"
        )
        for key, value in exports_raw.items()
    }
    return CompletionGate(
        path=_require_text(raw.get("path", ""), field=f"stage {stage_name} completion path"),
        kind=kind,
        assertions=tuple(assertions),
        exports=exports,
    )


def load_workflow_config(path: Path) -> WorkflowConfig:
    raw = _read_json(Path(path))
    if not isinstance(raw, Mapping) or raw.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported iteration workflow schema")
    if set(raw).difference(
        {"schema", "workflow_name", "state_dir", "variables", "environment", "stages"}
    ):
        raise ValueError("iteration workflow contains unknown top-level fields")
    variables_raw = raw.get("variables", {})
    environment_raw = raw.get("environment", {})
    if not isinstance(variables_raw, Mapping) or not isinstance(environment_raw, Mapping):
        raise ValueError("workflow variables/environment must be objects")
    variables = {str(key): str(value) for key, value in variables_raw.items()}
    environment = {str(key): str(value) for key, value in environment_raw.items()}
    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("iteration workflow requires at least one stage")
    stages: list[WorkflowStage] = []
    names: set[str] = set()
    export_names = set(variables)
    for index, item in enumerate(stages_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"stage {index} must be an object")
        if set(item) != {"name", "command", "cwd", "requires", "completion"}:
            raise ValueError(f"stage {index} contract drift")
        name = _require_text(item["name"], field=f"stage {index} name")
        if name in names:
            raise ValueError(f"duplicate workflow stage name: {name}")
        names.add(name)
        command_raw = item["command"]
        if not isinstance(command_raw, list) or not command_raw:
            raise ValueError(f"stage {name} command must be a non-empty argv list")
        command = tuple(_require_text(value, field=f"stage {name} command arg") for value in command_raw)
        requires_raw = item["requires"]
        if not isinstance(requires_raw, list):
            raise ValueError(f"stage {name} requires must be a list")
        requires: list[Mapping[str, Any]] = []
        for req_index, requirement in enumerate(requires_raw):
            if not isinstance(requirement, Mapping) or set(requirement) != {"path", "kind"}:
                raise ValueError(f"stage {name} requirement {req_index} contract drift")
            if requirement["kind"] not in {"file", "directory", "json"}:
                raise ValueError(f"stage {name} requirement {req_index} kind unsupported")
            requires.append(dict(requirement))
        completion = _parse_completion(item["completion"], stage_name=name)
        duplicate_exports = export_names.intersection(completion.exports)
        if duplicate_exports:
            raise ValueError(f"workflow export names must be unique: {sorted(duplicate_exports)}")
        export_names.update(completion.exports)
        stages.append(
            WorkflowStage(
                name=name,
                command=command,
                cwd=_require_text(item["cwd"], field=f"stage {name} cwd"),
                requires=tuple(requires),
                completion=completion,
            )
        )
    return WorkflowConfig(
        raw=dict(raw),
        config_sha256=canonical_sha256(raw),
        workflow_name=_require_text(raw.get("workflow_name", ""), field="workflow_name"),
        state_dir=_require_text(raw.get("state_dir", ""), field="state_dir"),
        variables=variables,
        environment=environment,
        stages=tuple(stages),
    )


def _json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise WorkflowError(f"JSON pointer must start with '/': {pointer}")
    current = value
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as exc:
                raise WorkflowError(f"JSON pointer not found: {pointer}") from exc
        elif isinstance(current, Mapping) and token in current:
            current = current[token]
        else:
            raise WorkflowError(f"JSON pointer not found: {pointer}")
    return current


def _assert_value(actual: Any, assertion: Mapping[str, Any]) -> None:
    expected = assertion["value"]
    op = assertion["op"]
    if op == "eq":
        passed = actual == expected
    elif op == "ne":
        passed = actual != expected
    elif op == "gt":
        passed = actual > expected
    elif op == "ge":
        passed = actual >= expected
    elif op == "lt":
        passed = actual < expected
    elif op == "le":
        passed = actual <= expected
    elif op == "in":
        passed = actual in expected
    else:  # validated on load
        raise WorkflowError(f"unsupported assertion op: {op}")
    if not passed:
        raise WorkflowError(
            f"artifact assertion failed at {assertion['pointer']}: "
            f"actual={actual!r} op={op} expected={expected!r}"
        )


def _format(text: str, values: Mapping[str, str], *, field: str) -> str:
    try:
        return text.format_map(values)
    except KeyError as exc:
        raise WorkflowError(f"{field} references unknown variable {exc.args[0]!r}") from exc


def _verify_path(path: Path, kind: str) -> Any:
    if kind == "file":
        if not path.is_file():
            raise WorkflowError(f"required file missing: {path}")
        return None
    if kind == "directory":
        if not path.is_dir():
            raise WorkflowError(f"required directory missing: {path}")
        return None
    if kind == "json":
        if not path.is_file():
            raise WorkflowError(f"required JSON missing: {path}")
        try:
            return _read_json(path)
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"invalid JSON artifact: {path}") from exc
    raise WorkflowError(f"unsupported path kind: {kind}")


def _verify_completion(stage: WorkflowStage, values: Mapping[str, str]) -> dict[str, str]:
    path = Path(_format(stage.completion.path, values, field=f"stage {stage.name} completion"))
    payload = _verify_path(path, stage.completion.kind)
    if stage.completion.kind != "json" and (stage.completion.assertions or stage.completion.exports):
        raise WorkflowError(f"stage {stage.name} non-JSON completion cannot assert/export JSON values")
    if stage.completion.kind == "json":
        for assertion in stage.completion.assertions:
            actual = _json_pointer(payload, str(assertion["pointer"]))
            _assert_value(actual, assertion)
        return {
            name: str(_json_pointer(payload, pointer))
            for name, pointer in stage.completion.exports.items()
        }
    return {}


def _initial_state(config: WorkflowConfig) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "workflow_name": config.workflow_name,
        "config_sha256": config.config_sha256,
        "status": "pending",
        "completed_stages": [],
        "exports": dict(config.variables),
        "failed_stage": None,
        "last_error": None,
    }


def _load_state(config: WorkflowConfig, state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return _initial_state(config)
    state = _read_json(state_path)
    if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA:
        raise WorkflowError("invalid workflow state schema")
    if state.get("workflow_name") != config.workflow_name:
        raise WorkflowError("workflow state name drift")
    if state.get("config_sha256") != config.config_sha256:
        raise WorkflowError("workflow config changed after state creation")
    completed = state.get("completed_stages")
    exports = state.get("exports")
    if not isinstance(completed, list) or not isinstance(exports, Mapping):
        raise WorkflowError("workflow state structure drift")
    return dict(state)


def plan_workflow(config: WorkflowConfig, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = {str(key): str(value) for key, value in state["exports"].items()}
    completed = set(str(name) for name in state["completed_stages"])
    plan: list[dict[str, Any]] = []
    for stage in config.stages:
        command = [_format(arg, values, field=f"stage {stage.name} command") for arg in stage.command]
        plan.append(
            {
                "name": stage.name,
                "status": "completed" if stage.name in completed else "pending",
                "cwd": _format(stage.cwd, values, field=f"stage {stage.name} cwd"),
                "command": command,
                "completion": _format(
                    stage.completion.path, values, field=f"stage {stage.name} completion"
                ),
            }
        )
        if stage.name in completed:
            values.update(_verify_completion(stage, values))
    return plan


def run_workflow(config_path: Path, *, execute: bool = False) -> dict[str, Any]:
    config = load_workflow_config(Path(config_path))
    state_dir = Path(config.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = _load_state(config, state_path)
    if not execute:
        return {
            "schema": "jit_iteration_workflow_plan_v1",
            "workflow_name": config.workflow_name,
            "config_sha256": config.config_sha256,
            "state_path": str(state_path),
            "plan": plan_workflow(config, state),
        }

    values = {str(key): str(value) for key, value in state["exports"].items()}
    completed = list(str(name) for name in state["completed_stages"])
    completed_set = set(completed)
    state.update({"status": "running", "failed_stage": None, "last_error": None})
    _write_json_atomic(state_path, state)

    for stage in config.stages:
        if stage.name in completed_set:
            exports = _verify_completion(stage, values)
            values.update(exports)
            continue
        try:
            for requirement in stage.requires:
                req_path = Path(
                    _format(str(requirement["path"]), values, field=f"stage {stage.name} requirement")
                )
                _verify_path(req_path, str(requirement["kind"]))

            completion_path = Path(
                _format(stage.completion.path, values, field=f"stage {stage.name} completion")
            )
            if completion_path.exists():
                exports = _verify_completion(stage, values)
                completed.append(stage.name)
                completed_set.add(stage.name)
                values.update(exports)
                state.update(
                    {
                        "completed_stages": completed,
                        "exports": values,
                        "status": "running",
                    }
                )
                _write_json_atomic(state_path, state)
                continue

            command = [
                _format(arg, values, field=f"stage {stage.name} command")
                for arg in stage.command
            ]
            cwd = Path(_format(stage.cwd, values, field=f"stage {stage.name} cwd"))
            if not cwd.is_dir():
                raise WorkflowError(f"stage {stage.name} cwd missing: {cwd}")
            env = os.environ.copy()
            env.update(
                {
                    key: _format(value, values, field=f"workflow environment {key}")
                    for key, value in config.environment.items()
                }
            )
            logs = state_dir / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (logs / f"{stage.name}.stdout.log").write_text(result.stdout, encoding="utf-8")
            (logs / f"{stage.name}.stderr.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise WorkflowError(
                    f"stage {stage.name} command exited {result.returncode}; see {logs}"
                )
            exports = _verify_completion(stage, values)
            completed.append(stage.name)
            completed_set.add(stage.name)
            values.update(exports)
            state.update(
                {
                    "completed_stages": completed,
                    "exports": values,
                    "status": "running",
                }
            )
            _write_json_atomic(state_path, state)
        except Exception as exc:
            state.update(
                {
                    "status": "failed",
                    "failed_stage": stage.name,
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "completed_stages": completed,
                    "exports": values,
                }
            )
            _write_json_atomic(state_path, state)
            raise

    state.update(
        {
            "status": "completed",
            "failed_stage": None,
            "last_error": None,
            "completed_stages": completed,
            "exports": values,
        }
    )
    _write_json_atomic(state_path, state)
    return state
