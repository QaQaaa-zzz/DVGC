"""Host-only Gate C1 contracts for auditable phase-expert smoke runs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Mapping

from .config import (
    ACTION_MAPPING_VERSION,
    AUTHORITATIVE_XML_PATH,
    AUTHORITATIVE_XML_SHA256,
)
from .training_budget import PPOBudgetReport, build_ppo_budget_report
from .two_phase_guideline import canonical_manifest_hash
from .two_phase_semantics import ApexBandThresholds, RecoveryThresholds


PHASE_PROPULSION_ASCENT = "propulsion_ascent"
PHASE_DESCENT_RECOVERY = "descent_recovery"
PHASE_EXPERT_PHASES = (PHASE_PROPULSION_ASCENT, PHASE_DESCENT_RECOVERY)
_THRESHOLD_SOURCE_HASHES = frozenset(
    {"xml", "reference", "config", "code", "geometry_manifest"}
)
_THRESHOLD_SOURCE_PATHS = frozenset({"xml", "reference", "config", "code"})
_DESCENT_SEED_TIERS = frozenset(
    {"physically_validated_descent_seed", "pi_up_online_apex_snapshot"}
)
_BASE_MODE = MappingProxyType(
    {
        "training_stage": "full",
        "use_bank_resets": False,
        "expert_chain_termination": False,
        "stage_reachability_objective": "",
        "domain_randomization": False,
    }
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PHASE_EXPERT_SOURCE_PATHS = (
    "configs/default.json",
    "configs/phase_expert_smoke.json",
    "dvgc/config.py",
    "dvgc/env.py",
    "dvgc/phase_expert_training.py",
    "dvgc/rewards.py",
    "dvgc/runtime.py",
    "dvgc/training_budget.py",
    "dvgc/two_phase_guideline.py",
    "dvgc/two_phase_runtime.py",
    "dvgc/two_phase_semantics.py",
    "cli/train_phase_expert.py",
)


@dataclass(frozen=True)
class PhaseExpertRunSpec:
    phase: str
    experiment_level: str
    requested_total_transitions: int
    seed: int
    config_path: str
    training_config_path: str
    threshold_manifest_path: str
    authorization_manifest_path: str | None
    output_dir: str
    descent_seed_bank: str | None
    descent_seed_manifest: str | None
    resume_run: str | None
    restore_checkpoint: str | None


@dataclass(frozen=True)
class PhaseExpertResetProtocol:
    phase: str
    mode: str
    seed_tier: str | None
    source_hash: str | None


@dataclass(frozen=True)
class ResolvedThresholdManifest:
    manifest: Mapping[str, Any]
    canonical_manifest_hash: str
    action_mapping_version: str
    reference_rollout_source: str
    apex_thresholds: ApexBandThresholds
    recovery_thresholds: RecoveryThresholds


@dataclass(frozen=True)
class PhaseExpertSeedNamespaces:
    training_namespace: str
    training_seeds: tuple[int, ...]
    evaluation_namespace: str
    evaluation_seeds: tuple[int, ...]


@dataclass(frozen=True)
class PhaseExpertInteractionBudget:
    training: PPOBudgetReport
    brax_evaluation_transition_ceiling: int
    fixed_evaluation_transition_ceiling: int
    combined_transition_ceiling: int


@dataclass(frozen=True)
class ValidatedPhaseExpertRunSpec:
    spec: PhaseExpertRunSpec
    thresholds: ResolvedThresholdManifest
    seeds: PhaseExpertSeedNamespaces
    interaction_budget: PhaseExpertInteractionBudget
    authorization: Mapping[str, Any] | None


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist: {candidate}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def phase_expert_source_tree_sha256() -> str:
    """Hash every Gate C1 managed source, including uncommitted file contents."""
    rows = []
    for relative in _PHASE_EXPERT_SOURCE_PATHS:
        path = _REPOSITORY_ROOT / relative
        digest = _sha256_file(path) if path.is_file() else "missing"
        rows.append(f"{relative}:{digest}\n")
    return hashlib.sha256("".join(rows).encode("ascii")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_phase_expert_threshold_manifest(path: str | Path) -> ResolvedThresholdManifest:
    """Load a current, provenance-complete threshold contract without mutation."""
    manifest_path = Path(path)
    manifest = _read_json(manifest_path, "threshold manifest")
    recorded_hash = manifest.get("canonical_manifest_hash")
    if not _is_sha256(recorded_hash) or recorded_hash != canonical_manifest_hash(manifest):
        raise ValueError("threshold manifest canonical hash is invalid")
    source_hashes = manifest.get("source_hashes")
    source_paths = manifest.get("source_paths")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != _THRESHOLD_SOURCE_HASHES
        or not all(_is_sha256(value) for value in source_hashes.values())
    ):
        raise ValueError("threshold manifest source hashes are incomplete")
    if (
        not isinstance(source_paths, Mapping)
        or set(source_paths) != _THRESHOLD_SOURCE_PATHS
        or not all(isinstance(value, str) and value for value in source_paths.values())
    ):
        raise ValueError("threshold manifest source paths are incomplete")
    if _resolve_repository_path(source_paths["xml"]) != _resolve_repository_path(
        AUTHORITATIVE_XML_PATH
    ):
        raise ValueError("threshold manifest must use the authoritative XML path")
    for source in sorted(_THRESHOLD_SOURCE_PATHS):
        source_path = _resolve_repository_path(source_paths[source])
        if not source_path.is_file() or _sha256_file(source_path) != source_hashes[source]:
            raise ValueError(f"threshold manifest source hash mismatch: {source}")
    if source_hashes["xml"] != AUTHORITATIVE_XML_SHA256:
        raise ValueError("threshold manifest XML is not the authoritative model")
    geometry_path = manifest_path.with_name("geometry_manifest.json")
    if not geometry_path.is_file():
        raise ValueError("threshold manifest geometry identity is unavailable")
    geometry = _read_json(geometry_path, "geometry manifest")
    if _canonical_payload_hash(geometry) != source_hashes["geometry_manifest"]:
        raise ValueError("threshold manifest geometry identity mismatch")
    if manifest.get("action_mapping_version") != ACTION_MAPPING_VERSION:
        raise ValueError("threshold manifest action mapping does not match current configuration")
    if manifest.get("reference_rollout_source") != "kinematic_guideline_envelope":
        raise ValueError("threshold manifest reference rollout provenance is invalid")
    if manifest.get("controller_provenance") != "kinematic guideline envelope":
        raise ValueError("threshold manifest controller provenance is invalid")
    if manifest.get("source_category") != "guideline_physical_envelope":
        raise ValueError("threshold manifest source category is invalid")
    selected = manifest.get("selected_thresholds")
    if not isinstance(selected, Mapping):
        raise ValueError("threshold manifest selected thresholds are missing")
    try:
        apex = ApexBandThresholds(**dict(selected["apex"]))
        recovery = RecoveryThresholds(**dict(selected["recovery"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("threshold manifest selected thresholds are invalid") from exc
    return ResolvedThresholdManifest(
        manifest=_freeze(manifest),
        canonical_manifest_hash=recorded_hash,
        action_mapping_version=ACTION_MAPPING_VERSION,
        reference_rollout_source="kinematic_guideline_envelope",
        apex_thresholds=apex,
        recovery_thresholds=recovery,
    )


def resolve_gate_c1_base_mode(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable physical base-mode overrides for the Gate C1 adapter."""
    if not isinstance(config, Mapping) or dict(config.get("base_mode", {})) != dict(
        _BASE_MODE
    ):
        raise ValueError("Gate C1 base mode does not match the frozen contract")
    if config.get("adapter_ownership") != {
        "reward": True,
        "done": True,
        "timeout": True,
    }:
        raise ValueError("Gate C1 adapter ownership contract is invalid")
    return dict(_BASE_MODE)


def _layout_value(layout: Mapping[str, Any], name: str) -> Any:
    if name not in layout:
        raise ValueError(f"PPO layout is missing {name}")
    return layout[name]


def build_phase_expert_budget(
    spec: PhaseExpertRunSpec, layout: Mapping[str, Any]
) -> PPOBudgetReport:
    """Build a smoke budget without allowing Brax alignment to enlarge it."""
    if not isinstance(layout, Mapping):
        raise ValueError("PPO layout must be a mapping")
    report = build_ppo_budget_report(
        requested_total_transitions=spec.requested_total_transitions,
        num_parallel_envs=_layout_value(layout, "num_parallel_envs"),
        episode_horizon=_layout_value(layout, "episode_horizon"),
        unroll_length=_layout_value(layout, "unroll_length"),
        batch_size=_layout_value(layout, "batch_size"),
        num_minibatches=_layout_value(layout, "num_minibatches"),
        num_updates_per_batch=_layout_value(layout, "num_updates_per_batch"),
        num_evals=_layout_value(layout, "num_evals"),
        experiment_level=spec.experiment_level,
    )
    assert report.requested_timesteps == report.requested_total_transitions
    assert report.effective_timesteps == report.effective_total_transitions
    if report.alignment_overhead != 0:
        raise ValueError("requested_total_transitions must be aligned to a PPO rollout block")
    if not 1 <= report.ppo_rollout_blocks <= 4:
        raise ValueError("smoke budget must use one through four PPO rollout blocks")
    return report


def _derive_training_seeds(root_seed: int, namespace: str, count: int) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"{root_seed}:{namespace}:{index}".encode()).digest()[:8],
            "big",
        )
        for index in range(count)
    )


def validate_phase_expert_seed_namespaces(
    spec: PhaseExpertRunSpec, training_config: Mapping[str, Any]
) -> PhaseExpertSeedNamespaces:
    """Derive reproducible training seeds and reject fixed-evaluation overlap."""
    namespace = training_config.get("training_seed_namespace")
    train_count = training_config.get("training_seed_count")
    evaluation = training_config.get("evaluation")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("training seed namespace is required")
    train_count = _positive_int("training_seed_count", train_count)
    if not isinstance(evaluation, Mapping):
        raise ValueError("fixed evaluation configuration is required")
    evaluation_namespace = evaluation.get("seed_namespace")
    evaluation_seeds = evaluation.get("seeds")
    if not isinstance(evaluation_namespace, str) or not evaluation_namespace:
        raise ValueError("evaluation seed namespace is required")
    if evaluation_namespace == namespace:
        raise ValueError("training and evaluation seed namespaces must be disjoint")
    if (
        not isinstance(evaluation_seeds, list)
        or not evaluation_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in evaluation_seeds)
        or len(set(evaluation_seeds)) != len(evaluation_seeds)
    ):
        raise ValueError("evaluation seeds must be unique integers")
    training_seeds = _derive_training_seeds(spec.seed, namespace, train_count)
    if set(training_seeds) & set(evaluation_seeds):
        raise ValueError("training and evaluation seeds must be disjoint")
    return PhaseExpertSeedNamespaces(
        training_namespace=namespace,
        training_seeds=training_seeds,
        evaluation_namespace=evaluation_namespace,
        evaluation_seeds=tuple(evaluation_seeds),
    )


def build_phase_expert_interaction_budget(
    spec: PhaseExpertRunSpec, training_config: Mapping[str, Any]
) -> PhaseExpertInteractionBudget:
    """Bind declared training and fixed-evaluation ceilings into one total cost."""
    layout = training_config.get("ppo_layout")
    maximum = training_config.get("maximum_interaction_cost")
    evaluation = training_config.get("evaluation")
    if (
        not isinstance(layout, Mapping)
        or not isinstance(maximum, Mapping)
        or not isinstance(evaluation, Mapping)
    ):
        raise ValueError(
            "smoke config must declare PPO layout, fixed evaluation, and interaction ceilings"
        )
    report = build_phase_expert_budget(spec, layout)
    training_ceiling = _positive_int(
        "training transition ceiling", maximum.get("training_transitions")
    )
    evaluation_ceiling = _positive_int(
        "fixed evaluation transition ceiling", maximum.get("fixed_evaluation_transitions")
    )
    brax_evaluation_ceiling = _positive_int(
        "Brax evaluation transition ceiling",
        maximum.get("brax_evaluation_transitions"),
    )
    maximum_combined_ceiling = _positive_int(
        "combined interaction transition ceiling", maximum.get("combined_transitions")
    )
    brax_evaluation_environments = _positive_int(
        "Brax evaluation environment count", layout.get("num_eval_envs")
    )
    declared_brax_evaluation_cost = (
        brax_evaluation_environments
        * report.episode_horizon
        * report.num_evals
    )
    if brax_evaluation_ceiling != declared_brax_evaluation_cost:
        raise ValueError(
            "Brax evaluation transition ceiling must equal evaluation "
            "environments times horizon times evaluations"
        )
    evaluation_environments = _positive_int(
        "fixed evaluation environment count", evaluation.get("environment_count")
    )
    evaluation_horizon = _positive_int(
        "fixed evaluation episode horizon", evaluation.get("episode_horizon")
    )
    evaluation_episodes = _positive_int(
        "fixed evaluation episode count", evaluation.get("episodes")
    )
    declared_evaluation_cost = (
        evaluation_environments * evaluation_horizon * evaluation_episodes
    )
    if evaluation_ceiling != declared_evaluation_cost:
        raise ValueError(
            "fixed evaluation transition ceiling must equal environments times horizon times episodes"
        )
    if (
        training_ceiling % report.ppo_rollout_block_size != 0
        or not 1
        <= training_ceiling // report.ppo_rollout_block_size
        <= 4
    ):
        raise ValueError(
            "training transition ceiling must be aligned to one through four PPO rollout blocks"
        )
    if maximum_combined_ceiling != (
        training_ceiling + brax_evaluation_ceiling + evaluation_ceiling
    ):
        raise ValueError(
            "combined interaction ceiling must equal training plus Brax and fixed "
            "evaluation ceilings"
        )
    if report.effective_total_transitions > training_ceiling:
        raise ValueError("training budget exceeds its interaction ceiling")
    return PhaseExpertInteractionBudget(
        training=report,
        brax_evaluation_transition_ceiling=brax_evaluation_ceiling,
        fixed_evaluation_transition_ceiling=evaluation_ceiling,
        combined_transition_ceiling=(
            report.effective_total_transitions
            + brax_evaluation_ceiling
            + evaluation_ceiling
        ),
    )


def _validate_descent_seed_inputs(spec: PhaseExpertRunSpec) -> Mapping[str, Any] | None:
    if spec.phase != PHASE_DESCENT_RECOVERY:
        return None
    if not spec.descent_seed_bank or not spec.descent_seed_manifest:
        raise ValueError("descent seed bank and manifest are required")
    bank = Path(spec.descent_seed_bank)
    if not bank.is_file():
        raise ValueError("descent seed bank does not exist")
    manifest = _read_json(spec.descent_seed_manifest, "descent seed manifest")
    if manifest.get("reset_mode") == "natural_start":
        raise ValueError("Phase D natural reset fallback is forbidden")
    if manifest.get("reset_mode") != "bank":
        raise ValueError("Phase D descent seed manifest must use bank reset")
    if manifest.get("seed_tier") not in _DESCENT_SEED_TIERS:
        raise ValueError("Phase D descent seed manifest has an invalid seed tier")
    if manifest.get("source_hash") != _sha256_file(bank):
        raise ValueError("Phase D descent seed source hash mismatch")
    return _freeze(manifest)


def _current_source_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _validate_authorization(
    spec: PhaseExpertRunSpec,
    thresholds: ResolvedThresholdManifest,
    interaction: PhaseExpertInteractionBudget,
) -> Mapping[str, Any]:
    if not spec.authorization_manifest_path:
        raise ValueError("normal execution requires an authorization manifest")
    authorization = _read_json(spec.authorization_manifest_path, "authorization manifest")
    expected = {
        "decision": "authorize",
        "run_id": Path(spec.output_dir).name,
        "phase": spec.phase,
        "experiment_level": spec.experiment_level,
        "source_head": _current_source_head(),
        "source_tree_sha256": phase_expert_source_tree_sha256(),
        "seed": spec.seed,
        "output_directory": str(Path(spec.output_dir).resolve()),
        "xml_sha256": AUTHORITATIVE_XML_SHA256,
        "threshold_manifest_canonical_hash": thresholds.canonical_manifest_hash,
        "training_config_sha256": _sha256_file(spec.training_config_path),
        "requested_training_transition_ceiling": interaction.training.requested_total_transitions,
        "effective_training_transition_ceiling": interaction.training.effective_total_transitions,
        "brax_evaluation_transition_ceiling": interaction.brax_evaluation_transition_ceiling,
        "fixed_evaluation_transition_ceiling": interaction.fixed_evaluation_transition_ceiling,
        "combined_interaction_transition_ceiling": interaction.combined_transition_ceiling,
    }
    for field, value in expected.items():
        if authorization.get(field) != value:
            label = "run id" if field == "run_id" else field.replace("_", " ")
            raise ValueError(f"authorization manifest {label} does not match this run")
    if not isinstance(authorization.get("issuer"), str) or not authorization["issuer"]:
        raise ValueError("authorization manifest issuer is required")
    if not isinstance(authorization.get("issued_at"), str) or not authorization["issued_at"]:
        raise ValueError("authorization manifest issue time is required")
    return _freeze(authorization)


def validate_phase_expert_run_spec(
    spec: PhaseExpertRunSpec, *, preflight_only: bool
) -> ValidatedPhaseExpertRunSpec:
    """Validate a no-overwrite, smoke-only run contract before any environment work."""
    if not isinstance(spec, PhaseExpertRunSpec):
        raise TypeError("spec must be a PhaseExpertRunSpec")
    if spec.phase not in PHASE_EXPERT_PHASES:
        raise ValueError(f"phase must be one of {PHASE_EXPERT_PHASES}")
    if spec.experiment_level != "smoke":
        raise ValueError("only smoke is authorized at Gate C1")
    _positive_int("requested_total_transitions", spec.requested_total_transitions)
    if isinstance(spec.seed, bool) or not isinstance(spec.seed, int):
        raise ValueError("seed must be an integer")
    output_path = _resolve_repository_path(spec.output_dir)
    required_output_parent = (
        _REPOSITORY_ROOT / "runs" / "two_phase" / "phase_experts"
    ).resolve()
    if output_path.parent != required_output_parent or not output_path.name:
        raise ValueError(
            "output directory must be runs/two_phase/phase_experts/<run_id>"
        )
    if output_path.exists():
        raise ValueError("output directory must not already exist")
    if spec.resume_run is not None or spec.restore_checkpoint is not None:
        raise ValueError("exact resume is unavailable until the Gate C1 resume contract is implemented")
    thresholds = load_phase_expert_threshold_manifest(spec.threshold_manifest_path)
    project_config_path = _resolve_repository_path(spec.config_path)
    threshold_config_path = _resolve_repository_path(
        thresholds.manifest["source_paths"]["config"]
    )
    if not project_config_path.is_file() or project_config_path != threshold_config_path:
        raise ValueError(
            "project config path must match the threshold manifest source config"
        )
    training_config = _read_json(spec.training_config_path, "phase expert training config")
    resolve_gate_c1_base_mode(training_config)
    seeds = validate_phase_expert_seed_namespaces(spec, training_config)
    interaction = build_phase_expert_interaction_budget(spec, training_config)
    _validate_descent_seed_inputs(spec)
    if spec.phase == PHASE_DESCENT_RECOVERY and not preflight_only:
        raise ValueError("Phase D is preflight-only at Gate C1")
    authorization = (
        None
        if preflight_only
        else _validate_authorization(spec, thresholds, interaction)
    )
    return ValidatedPhaseExpertRunSpec(
        spec=spec,
        thresholds=thresholds,
        seeds=seeds,
        interaction_budget=interaction,
        authorization=authorization,
    )
