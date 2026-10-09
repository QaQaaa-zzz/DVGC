"""Evaluate one saved Phase U checkpoint panel in a fresh GPU process."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax

from .checkpoint import load_checkpoint
from .config import load_config
from .env import TwoPhaseBikeEnv
from .formal_training import (
    _checkpoint_identity,
    _evaluate_airborne_rsi_panel,
    _evaluate_fixed_panel,
)
from .ppo import make_checkpoint_policy


def evaluate_saved_phase_u_panel(
    config_path: Path,
    run_dir: Path,
    step: int,
    panel: str,
    *,
    backend_name: Callable[[], str] = jax.default_backend,
) -> int:
    """Restore the declared checkpoint and save one eight-seed fixed panel."""
    config = load_config(config_path)
    if config.formal is None or step not in config.formal.checkpoint_transitions[1:]:
        raise ValueError("step is absent from the checkpoint schedule")
    if panel not in {"natural", "airborne_rsi"}:
        raise ValueError("panel must be natural or airborne_rsi")
    if backend_name() != "gpu":
        raise RuntimeError("fixed Phase U panel requires GPU backend")
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints" / f"transition_{step}"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    env = TwoPhaseBikeEnv(config)
    identity = _checkpoint_identity(config, env._bundle.xml_sha256)
    payload = load_checkpoint(checkpoint_dir, expected=identity)

    def make_policy(_params, *, deterministic: bool):
        return make_checkpoint_policy(env, payload, deterministic=deterministic)

    evaluator = (
        _evaluate_fixed_panel if panel == "natural" else _evaluate_airborne_rsi_panel
    )
    result = evaluator(env, run_dir, config, step, make_policy, None)
    return int(result.environment_transitions)
