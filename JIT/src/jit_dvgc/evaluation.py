"""Deterministic host-side evaluation from saved runtime state and metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import jax
import numpy as np

from .constants import END_REASONS, REWARD_COMPONENT_KEYS


@dataclass(frozen=True)
class EpisodeFrame:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    action: np.ndarray
    reward: float
    reward_components: dict[str, float]
    metrics: dict[str, float]
    terminated: bool
    truncated: bool
    end_code: int
    success: bool
    physical_failure: bool
    timeout: bool


@dataclass(frozen=True)
class EpisodeTrace:
    seed: int
    frames: tuple[EpisodeFrame, ...]
    environment_transitions: int


def _array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value)).copy()


def _scalar(value: Any, cast):
    return cast(np.asarray(jax.device_get(value)))


def _capture_frame(state: Any) -> EpisodeFrame:
    metrics = {
        name: _scalar(value, float) for name, value in dict(state.metrics).items()
    }
    components = {
        key: metrics.get(f"reward/{key}", 0.0) for key in REWARD_COMPONENT_KEYS
    }
    info = state.info
    return EpisodeFrame(
        qpos=_array(state.data.qpos),
        qvel=_array(state.data.qvel),
        ctrl=_array(state.data.ctrl),
        action=_array(info.get("last_action", np.zeros(4))),
        reward=_scalar(state.reward, float),
        reward_components=components,
        metrics=metrics,
        terminated=_scalar(info.get("terminated", False), bool),
        truncated=_scalar(info.get("truncated", False), bool),
        end_code=_scalar(info.get("end_code", 0), int),
        success=_scalar(info.get("success", False), bool),
        physical_failure=_scalar(info.get("physical_failure", False), bool),
        timeout=_scalar(info.get("timeout", False), bool),
    )


def capture_episode(
    env: Any,
    policy: Callable[[Any], Any],
    *,
    seed: int,
    horizon: int,
    reset_fn: Callable[[Any], Any] | None = None,
    step_fn: Callable[[Any, Any], Any] | None = None,
) -> EpisodeTrace:
    if horizon <= 0:
        raise ValueError("evaluation horizon must be positive")
    reset = env.reset if reset_fn is None else reset_fn
    state = reset(jax.random.PRNGKey(int(seed)))
    step = env.step if step_fn is None else step_fn
    frames = [_capture_frame(state)]
    transitions = 0
    for _ in range(int(horizon)):
        result = policy(state.obs)
        action = result[0] if isinstance(result, tuple) else result
        state = step(state, action)
        transitions += 1
        frames.append(_capture_frame(state))
        if frames[-1].terminated or frames[-1].truncated:
            break
    return EpisodeTrace(
        seed=int(seed),
        frames=tuple(frames),
        environment_transitions=transitions,
    )


def _ever(trace: EpisodeTrace, metric: str) -> bool:
    return any(frame.metrics.get(metric, 0.0) > 0.5 for frame in trace.frames)


def _rate(values: Iterable[bool]) -> float:
    items = tuple(values)
    return float(sum(items) / len(items)) if items else 0.0


def summarize_phase_u(traces: tuple[EpisodeTrace, ...]) -> dict[str, Any]:
    if not traces:
        raise ValueError("fixed evaluation requires at least one trace")
    terminals = [trace.frames[-1] for trace in traces]
    clearances = [
        frame.metrics.get("signal/structure_clearance", float("-inf"))
        for trace in traces
        for frame in trace.frames
    ]
    rolls = [
        abs(frame.metrics.get("signal/roll", 0.0))
        for trace in traces
        for frame in trace.frames
    ]
    pitches = [
        abs(frame.metrics.get("signal/pitch", 0.0))
        for trace in traces
        for frame in trace.frames
    ]
    rates = [
        frame.metrics.get("signal/angular_speed", 0.0)
        for trace in traces
        for frame in trace.frames
    ]
    component_sums = {
        key: float(
            sum(
                frame.reward_components[key]
                for trace in traces
                for frame in trace.frames[1:]
            )
        )
        for key in REWARD_COMPONENT_KEYS
    }
    saturation_values = [
        np.mean(np.abs(frame.action) >= 0.999)
        for trace in traces
        for frame in trace.frames[1:]
    ]
    end_counts = Counter(
        END_REASONS.get(frame.end_code, f"unknown_{frame.end_code}")
        for frame in terminals
    )
    return {
        "rollouts": len(traces),
        "environment_transitions": sum(t.environment_transitions for t in traces),
        "window_reach_rate": _rate(_ever(t, "event/window_latched") for t in traces),
        "liftoff_rate": _rate(_ever(t, "event/liftoff_seen") for t in traces),
        "stable_airborne_rate": _rate(
            _ever(t, "event/stable_airborne_seen") for t in traces
        ),
        "ascending_rate": _rate(_ever(t, "event/ascending_seen") for t in traces),
        "apex_success_rate": _rate(frame.success for frame in terminals),
        "maximum_clearance": float(max(clearances)),
        "maximum_abs_roll": float(max(rolls)),
        "maximum_abs_pitch": float(max(pitches)),
        "maximum_angular_speed": float(max(rates)),
        "physical_failure_rate": _rate(frame.physical_failure for frame in terminals),
        "timeout_rate": _rate(frame.timeout for frame in terminals),
        "end_reason_counts": dict(sorted(end_counts.items())),
        "reward_component_sums": component_sums,
        "action_saturation_fraction": float(
            np.mean(saturation_values) if saturation_values else 0.0
        ),
    }
