"""Locate the first same-engine MJX replay divergence.

This diagnostic deliberately avoids CPU MuJoCo comparison.  It runs one
MJX-Warp world at a time, freezes the action sequence, and compares repeated
snapshot and natural-reset lineages tick by tick.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


TRACE_FIELDS = (
    "qpos", "qvel", "act", "ctrl", "time", "observation",
    "observation_history", "last_action", "delay_buffer",
    "controller_state", "prng_key", "phase", "estimated_phase",
    "physical_failure", "end_code", "contact_count", "contact_geom",
    "contact_position", "contact_normal", "contact_distance", "contact_force",
    "subtree_angmom",
)


def _array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _trace_point(state, action: np.ndarray, *, max_contacts: int = 32,
                 max_constraints: int = 64) -> dict[str, np.ndarray]:
    data = jax.device_get(state.data)
    info = jax.device_get(state.info)
    impl = data._impl
    nacon = int(np.asarray(impl.nacon).reshape(-1)[0])
    ncontact = min(max(nacon, 0), max_contacts)
    contact_geom = np.full((max_contacts, 2), -1, np.int32)
    contact_position = np.full((max_contacts, 3), np.nan, np.float32)
    contact_normal = np.full((max_contacts, 3), np.nan, np.float32)
    contact_distance = np.full((max_contacts,), np.nan, np.float32)
    if ncontact:
        contact_geom[:ncontact] = np.asarray(impl.contact__geom)[:ncontact]
        contact_position[:ncontact] = np.asarray(impl.contact__pos)[:ncontact]
        contact_normal[:ncontact] = np.asarray(
            impl.contact__frame
        )[:ncontact, 0, :]
        contact_distance[:ncontact] = np.asarray(
            impl.contact__dist
        )[:ncontact]
    force = np.zeros((max_constraints,), np.float32)
    raw_force = np.asarray(impl.efc__force).reshape(-1)
    force[:min(len(raw_force), max_constraints)] = raw_force[:max_constraints]
    phase_probs = np.asarray(info["phase_probs"], np.float32)
    return {
        "qpos": np.asarray(data.qpos),
        "qvel": np.asarray(data.qvel),
        "act": np.asarray(data.act),
        "ctrl": np.asarray(data.ctrl),
        "time": np.asarray(data.time),
        "observation": np.asarray(state.obs["state"]),
        "observation_history": np.asarray(info["obs_history"]),
        "last_action": np.asarray(info["last_action"]),
        # The deployed event filter has no delayed command actuator.  Its
        # declared delay buffer is the current phase posterior.
        "delay_buffer": phase_probs[None, :],
        "controller_state": np.zeros((0,), np.float32),
        "prng_key": np.asarray(info["rng"]),
        "phase": np.asarray(info["phase"]),
        "estimated_phase": np.asarray(info["estimated_phase"]),
        "physical_failure": np.asarray(info["terminated"]),
        "end_code": np.asarray(info["end_code"]),
        "contact_count": np.asarray(nacon, np.int32),
        "contact_geom": contact_geom,
        "contact_position": contact_position,
        "contact_normal": contact_normal,
        "contact_distance": contact_distance,
        "contact_force": force,
        "subtree_angmom": np.asarray(impl.subtree_angmom),
        "fixed_action": np.asarray(action, np.float32),
    }


def _stack(traces: list[list[dict[str, np.ndarray]]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([
            np.stack([point[key] for point in trace])
            for trace in traces
        ])
        for key in (*TRACE_FIELDS, "fixed_action")
    }


def _first_divergence(stacked: dict[str, np.ndarray]) -> dict[str, Any] | None:
    for tick in range(stacked["qpos"].shape[1]):
        for field in TRACE_FIELDS:
            values = stacked[field][:, tick]
            reference = values[0]
            if values.size == 0:
                continue
            if np.issubdtype(values.dtype, np.floating):
                finite_equal = np.array_equal(
                    np.isfinite(values), np.broadcast_to(
                        np.isfinite(reference), values.shape
                    )
                )
                delta = np.abs(
                    values.astype(np.float64) - reference.astype(np.float64)
                )
                finite_delta = delta[np.isfinite(delta)]
                error = (
                    float(np.max(finite_delta)) if finite_equal and finite_delta.size
                    else (0.0 if finite_equal else float("inf"))
                )
            else:
                error = float(np.max(np.abs(
                    values.astype(np.int64) - reference.astype(np.int64)
                ))) if values.size else 0.0
            if error != 0.0:
                if field in ("last_action", "controller_state", "prng_key"):
                    location = "policy_or_controller_state"
                elif field in ("observation", "observation_history", "delay_buffer"):
                    location = "history_update"
                elif field.startswith("contact_"):
                    location = "contact_extraction"
                else:
                    location = "physics_step"
                return {
                    "tick": tick,
                    "field": field,
                    "max_abs_error": error,
                    "location": location,
                }
    return None


def _spread(stacked: dict[str, np.ndarray], field: str) -> float:
    values = stacked[field]
    if values.size == 0:
        return 0.0
    reference = values[0]
    return float(np.nanmax(np.abs(
        values.astype(np.float64) - reference.astype(np.float64)
    )))


def _event_summary(stacked: dict[str, np.ndarray], root_body: int) -> dict[str, Any]:
    separation_ticks = []
    momenta = []
    for repeat in range(stacked["contact_count"].shape[0]):
        counts = stacked["contact_count"][repeat]
        contacts = np.flatnonzero(counts > 0)
        last = int(contacts[-1]) if contacts.size else None
        separation = next((
            tick for tick in range((last + 1) if last is not None else 0, len(counts))
            if counts[tick] == 0
        ), None) if last is not None else None
        separation_ticks.append(separation)
        momenta.append(
            None if separation is None else
            stacked["subtree_angmom"][repeat, separation, root_body].tolist()
        )
    finite_momenta = np.asarray(
        [row for row in momenta if row is not None], np.float64
    )
    tick_values = [tick for tick in separation_ticks if tick is not None]
    return {
        "separation_ticks_zero_based": separation_ticks,
        "separation_tick_spread": (
            None if not tick_values else max(tick_values) - min(tick_values)
        ),
        "separation_momentum": momenta,
        "separation_momentum_linf_spread": (
            None if not len(finite_momenta) else float(np.max(
                np.ptp(finite_momenta, axis=0)
            ))
        ),
    }


def _run_repeats(
    reset: Callable[[], Any], step_fn: Callable, actions: list[np.ndarray],
    repeats: int,
) -> dict[str, np.ndarray]:
    traces = []
    for _ in range(repeats):
        state = reset()
        trace = [_trace_point(state, np.zeros_like(actions[0]))]
        for action in actions:
            state = step_fn(state, action)
            trace.append(_trace_point(state, action))
        traces.append(trace)
    return _stack(traces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takeoff-bank", required=True)
    parser.add_argument("--lineage-report", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compatibility-repeats", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config, {
        "training_stage": "flight",
        "use_bank_resets": False,
        "domain_randomization": False,
        "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(config, snapshot_bank=SnapshotBank())
    lineage = json.loads(Path(args.lineage_report).read_text())["parents"][
        args.parent
    ]
    source = next(
        row for row in SnapshotBank.load(args.takeoff_bank).records
        if row["id"] == lineage["source_takeoff_state_id"]
    )
    entry = next(
        row for row in SnapshotBank.load(
            Path(lineage["source_acquisition"]) / "fresh_ascent_entries.pkl"
        ).records
        if row["id"] == lineage["source_entry_id"]
    )
    seed = int(entry["dynamics_seed"])
    actions = [
        np.asarray(row["action"], np.float32)
        for row in lineage["trace"] if int(row["tick"]) > 0
    ]
    if not actions:
        raise RuntimeError("lineage contains no fixed actions")

    reset_snapshot = lambda: restore_snapshot(
        env, source, jax.random.PRNGKey(seed)
    )
    reset_natural = lambda: env.reset(jax.random.PRNGKey(seed))
    modes = {
        "snapshot_jit_single": (reset_snapshot, jax.jit(env.step), args.repeats),
        "natural_jit_single": (reset_natural, jax.jit(env.step), args.repeats),
        "snapshot_nonjit_single": (
            reset_snapshot, env.step, args.compatibility_repeats
        ),
        "natural_nonjit_single": (
            reset_natural, env.step, args.compatibility_repeats
        ),
    }
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, (reset, step_fn, repeats) in modes.items():
        stacked = _run_repeats(reset, step_fn, actions, repeats)
        trace_path = trace_dir / f"{name}.npz"
        np.savez_compressed(trace_path, **stacked)
        divergence = _first_divergence(stacked)
        event = _event_summary(stacked, env._root_body_id)
        results[name] = {
            "repeats": repeats,
            "ticks_including_initial": int(stacked["qpos"].shape[1]),
            "first_divergence": divergence,
            "strict_repeatable": divergence is None,
            "max_spread": {
                field: _spread(stacked, field)
                for field in (
                    "qpos", "qvel", "act", "ctrl", "time", "observation",
                    "observation_history", "last_action", "prng_key",
                )
            },
            "event": event,
            "event_repeatable": (
                event["separation_tick_spread"] in (None, 0, 1)
                and event["separation_momentum_linf_spread"] in (None,)
                or (
                    event["separation_tick_spread"] is not None
                    and event["separation_tick_spread"] <= 1
                    and event["separation_momentum_linf_spread"] is not None
                    and event["separation_momentum_linf_spread"] <= .03
                )
            ),
            "trace_path": str(trace_path),
            "trace_sha256": file_sha256(trace_path),
        }
    strict = all(row["strict_repeatable"] for row in results.values())
    payload = {
        "status": "PASS" if strict else "FAIL",
        "artifact_role": "mjx_same_engine_determinism_gate",
        "blocker": None if strict else "mjx_rollout_nondeterminism",
        "development_runtime": "MJX",
        "cpu_mujoco_comparison": False,
        "impl": str(config.impl),
        "effective_solver": env._effective_mjx_solver,
        "xml_declared_solver": int(env._xml_solver),
        "device": str(jax.devices()[0]),
        "world_batch_size": int(reset_natural().data._impl.nworld),
        "vmap_batch_size_one": (
            "represented by the native MJX-Warp nworld=1 axis; nested vmap "
            "is unsupported by DataWarp"
        ),
        "fixed_seed": seed,
        "fixed_actions": len(actions),
        "source_id": source["id"],
        "dynamics_parameters": {
            "mass_scale": float(config.mass_scale),
            "friction_scale": float(config.friction_scale),
            "actuator_force_scale": float(config.actuator_force_scale),
            "gravity_scale": float(config.gravity_scale),
        },
        "results": results,
        "snapshot_replay_repeatable": results[
            "snapshot_jit_single"
        ]["strict_repeatable"],
        "natural_continuous_repeatable": results[
            "natural_jit_single"
        ]["strict_repeatable"],
        "continuous_pipeline_allowed": strict,
        "ppo_authorization": False,
        "xml_sha256": file_sha256(config.xml_path),
        "config_sha256": file_sha256(args.config),
        "takeoff_bank_sha256": file_sha256(args.takeoff_bank),
        "lineage_report_sha256": file_sha256(args.lineage_report),
    }
    save_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "blocker": payload["blocker"],
        "effective_solver": payload["effective_solver"],
        "results": {
            name: {
                "repeatable": row["strict_repeatable"],
                "first_divergence": row["first_divergence"],
                "event": row["event"],
            }
            for name, row in results.items()
        },
    }, indent=2))
    if not strict:
        raise SystemExit(40)


if __name__ == "__main__":
    main()
