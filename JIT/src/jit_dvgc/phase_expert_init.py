"""Actor-only initialization contract for the Phase D expert."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import ResolvedConfig
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from .env import TwoPhaseBikeEnv
from .handoff_bank import pytree_sha256
from .handoff_snapshot import compatibility_identity
from .ppo import make_network_factory


@dataclass(frozen=True)
class ActorOnlyInitialization:
    """The only parent state permitted to cross into a Phase D trainer."""

    observation_normalizer: Any
    actor_params: Any
    parent_transition: int
    payload_sha256: str
    actor_sha256: str
    provenance: Mapping[str, bool]

    @property
    def restore_params(self) -> tuple[Any, Any]:
        return self.observation_normalizer, self.actor_params


def build_actor_only_initialization(
    checkpoint: Path,
    *,
    source_config: ResolvedConfig,
    target_env: TwoPhaseBikeEnv,
) -> ActorOnlyInitialization:
    """Load a Phase U checkpoint and validate the executable Phase D contract."""
    if source_config.phase != "propulsion_ascent":
        raise ValueError("actor-only source config must be Phase U")
    if target_env.resolved_config.phase != "descent_recovery":
        raise ValueError("actor-only target environment must be Phase D")

    source_env = TwoPhaseBikeEnv(source_config, convert_model=False)
    expected = CheckpointIdentity(
        config_sha256=source_config.config_sha256,
        xml_sha256=source_env._bundle.xml_sha256,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )
    payload = load_checkpoint(Path(checkpoint), expected=expected)

    if source_env.actor_observation_size != target_env.actor_observation_size:
        raise ValueError("actor observation size compatibility mismatch")
    if source_env.privileged_observation_size != target_env.privileged_observation_size:
        raise ValueError("privileged observation size compatibility mismatch")
    if compatibility_identity(source_env) != compatibility_identity(target_env):
        raise ValueError("runtime compatibility mismatch")

    sidecar = json.loads((Path(checkpoint) / "identity.json").read_text(encoding="utf-8"))
    payload_hash = sidecar.get("payload_sha256")
    if not isinstance(payload_hash, str) or not payload_hash:
        raise ValueError("checkpoint sidecar lacks payload hash")
    return ActorOnlyInitialization(
        observation_normalizer=payload.observation_normalizer,
        actor_params=payload.actor_params,
        parent_transition=int(payload.training_transitions),
        payload_sha256=payload_hash,
        actor_sha256=pytree_sha256(payload.actor_params),
        provenance={
            "actor_initialized": True,
            "critic_fresh": True,
            "optimizer_fresh": True,
        },
    )


def make_actor_only_policy(
    env: Any, initialization: ActorOnlyInitialization, *, deterministic: bool = True
):
    """Rebuild only the parent Actor; the value network is deliberately absent."""
    networks = make_network_factory()(
        {"state": env.actor_observation_size, "privileged_state": env.privileged_observation_size},
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    return ppo_networks.make_inference_fn(networks)(
        initialization.restore_params + (None,), deterministic=deterministic
    )


def trainer_kwargs_for_actor_only(initialization: ActorOnlyInitialization) -> dict[str, Any]:
    """Arguments documenting a fresh Phase D value/optimizer initialization."""
    return {"restore_params": initialization.restore_params, "restore_value_fn": False}
