from __future__ import annotations

from typing import NamedTuple

from jax import numpy as jp
import pytest

from jit_dvgc.evaluation import (
    capture_episode,
    select_representative_trace,
    split_trace_at_apex,
    summarize_phase_u,
)


class FakeData(NamedTuple):
    qpos: object
    qvel: object
    ctrl: object


class FakeState(NamedTuple):
    data: FakeData
    obs: dict
    reward: object
    done: object
    metrics: dict
    info: dict


class DoneAfterThree:
    def __init__(self):
        self.step_calls = 0

    def _state(self, tick: int) -> FakeState:
        terminal = tick == 3
        return FakeState(
            data=FakeData(
                qpos=jp.full((12,), float(tick)),
                qvel=jp.zeros(11),
                ctrl=jp.zeros(4),
            ),
            obs={"state": jp.zeros(76), "privileged_state": jp.zeros(106)},
            reward=jp.asarray(float(tick)),
            done=jp.asarray(float(terminal)),
            metrics={
                "reward": jp.asarray(float(tick)),
                "event/jump_signal": jp.asarray(float(tick == 1)),
                "event/jump_zone_seen": jp.asarray(float(tick >= 1)),
                "event/jump_zone_consumed": jp.asarray(float(tick >= 2)),
                "event/ascending_seen": jp.asarray(float(tick >= 2)),
                "event/height_seen": jp.asarray(float(tick >= 2)),
                "event/apex_seen": jp.asarray(0.0),
                "signal/root_x": jp.asarray(1.5 + 0.5 * tick),
                "signal/root_y": jp.asarray(0.0),
                "signal/root_z": jp.asarray(0.15 + 0.2 * tick),
                "signal/roll": jp.asarray(0.0),
                "signal/pitch": jp.asarray(0.0),
                "signal/angular_speed": jp.asarray(0.0),
            },
            info={
                "last_action": jp.zeros(4),
                "terminated": jp.asarray(terminal),
                "truncated": jp.asarray(False),
                "end_code": jp.asarray(4 if terminal else 0),
                "success": jp.asarray(False),
                "physical_failure": jp.asarray(terminal),
                "timeout": jp.asarray(False),
            },
        )

    def reset(self, _key):
        return self._state(0)

    def step(self, _state, _action):
        self.step_calls += 1
        return self._state(self.step_calls)


def test_capture_stops_at_done_and_counts_initial_state_once():
    env = DoneAfterThree()
    trace = capture_episode(
        env,
        lambda _obs: jp.zeros(4),
        seed=1,
        horizon=200,
    )

    assert env.step_calls == 3
    assert trace.environment_transitions == 3
    assert len(trace.frames) == 4
    assert trace.frames[-1].terminated
    assert not trace.frames[-1].truncated
    assert trace.frames[-1].end_code == 4


def test_capture_can_use_compiled_reset_and_step_functions():
    env = DoneAfterThree()
    calls = {"reset": 0, "step": 0}

    def reset_fn(key):
        calls["reset"] += 1
        return env.reset(key)

    def step_fn(state, action):
        calls["step"] += 1
        return env.step(state, action)

    trace = capture_episode(
        env,
        lambda _obs: jp.zeros(4),
        seed=3,
        horizon=20,
        reset_fn=reset_fn,
        step_fn=step_fn,
    )
    assert trace.environment_transitions == 3
    assert calls == {"reset": 1, "step": 3}


def test_phase_u_summary_uses_saved_state_metrics_not_video():
    trace = capture_episode(
        DoneAfterThree(), lambda _obs: jp.zeros(4), seed=1, horizon=200
    )
    summary = summarize_phase_u((trace,))

    assert summary["rollouts"] == 1
    assert summary["jump_zone_reach_rate"] == 1.0
    assert summary["height_reach_rate"] == 1.0
    assert summary["ascending_rate"] == 1.0
    assert summary["apex_success_rate"] == 0.0
    assert summary["physical_failure_rate"] == 1.0
    assert summary["end_reason_counts"] == {"pitch_limit": 1}
    assert summary["maximum_root_height"] == pytest.approx(0.75, abs=1e-6)
    assert summary["apex_rollouts"] == 0
    assert summary["post_apex_environment_transitions"] == 0


def test_apex_split_shares_boundary_and_conserves_transitions():
    base = capture_episode(
        DoneAfterThree(), lambda _obs: jp.zeros(4), seed=1, horizon=200
    )
    frames = list(base.frames)
    for index, frame in enumerate(frames):
        frames[index] = frame.__class__(
            **{
                **frame.__dict__,
                "metrics": {
                    **frame.metrics,
                    "event/apex_seen": float(index >= 2),
                },
            }
        )
    trace = base.__class__(
        seed=base.seed,
        frames=tuple(frames),
        environment_transitions=base.environment_transitions,
    )

    split = split_trace_at_apex(trace)

    assert split.apex_frame_index == 2
    assert split.pre_frame_indices == (0, 1, 2)
    assert split.post_frame_indices == (2, 3)
    assert split.pre_environment_transitions == 2
    assert split.post_environment_transitions == 1
    assert split.pre_environment_transitions + split.post_environment_transitions == 3

    summary = summarize_phase_u((trace,))
    assert summary["apex_success_rate"] == 1.0
    assert summary["apex_rollouts"] == 1
    assert summary["post_apex_environment_transitions"] == 1
    assert summary["post_apex_physical_failure_rate"] == 1.0


def test_no_apex_split_keeps_full_pre_and_empty_post():
    trace = capture_episode(
        DoneAfterThree(), lambda _obs: jp.zeros(4), seed=1, horizon=200
    )
    split = split_trace_at_apex(trace)

    assert split.apex_frame_index == -1
    assert split.pre_frame_indices == (0, 1, 2, 3)
    assert split.post_frame_indices == ()
    assert split.pre_environment_transitions == 3
    assert split.post_environment_transitions == 0


def test_representative_prefers_apex_then_highest_reward():
    base = capture_episode(
        DoneAfterThree(), lambda _obs: jp.zeros(4), seed=1, horizon=200
    )
    high_no_apex = base.__class__(
        seed=2,
        frames=tuple(
            frame.__class__(
                **{
                    **frame.__dict__,
                    "metrics": {**frame.metrics, "reward/unclipped": 100.0},
                }
            )
            for frame in base.frames
        ),
        environment_transitions=base.environment_transitions,
    )
    apex = base.__class__(
        seed=3,
        frames=tuple(
            frame.__class__(
                **{
                    **frame.__dict__,
                    "metrics": {
                        **frame.metrics,
                        "event/apex_seen": float(index >= 2),
                        "reward/unclipped": 1.0,
                    },
                }
            )
            for index, frame in enumerate(base.frames)
        ),
        environment_transitions=base.environment_transitions,
    )

    assert select_representative_trace((high_no_apex, apex)).seed == 3
    assert select_representative_trace((base, high_no_apex)).seed == 2
