#!/usr/bin/env python3
"""Train the unified Tube1 policy from frozen pi0 actor and critic parameters.

Only initialization differs from the matched repair02 config: observation
normalizer, actor, and critic come from frozen pi0. Optimizer remains fresh.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile

from brax.training.agents.ppo import train as ppo_train

import jit_dvgc.unified_formal as flat_formal
import jit_dvgc.training.formal as canonical_formal
from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
from jit_dvgc.config import canonical_sha256
from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from jit_dvgc.handoff_bank import pytree_sha256


_ORIGINAL_LOAD_FORMAL_CONFIG = flat_formal.load_unified_formal_config


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_warm_target_config(path: Path):
    """Reuse canonical validation while allowing one explicit full warm-start shape."""
    path = Path(path)
    raw = _read_json(path)
    initialization = raw.get("initialization")
    expected = {
        "actor": "warm_start_pi_0",
        "critic": "warm_start_pi_0",
        "optimizer": "fresh",
        "source_frozen_policy": (
            "JIT/runs/frozen_unified/"
            "pi_0_round1_10009600_20260831/frozen_unified_policy.json"
        ),
    }
    if initialization != expected:
        raise ValueError("pi0 full warm-start initialization contract drift")

    sanitized = dict(raw)
    sanitized["initialization"] = {
        "actor": "fresh",
        "critic": "fresh",
        "optimizer": "fresh",
        "restore_checkpoint": None,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(sanitized, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    try:
        parsed = _ORIGINAL_LOAD_FORMAL_CONFIG(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    return replace(parsed, raw=raw, config_sha256=canonical_sha256(raw))


def _load_pi0_restore_params(config_path: Path):
    raw = _read_json(config_path)
    frozen_path = Path(raw["initialization"]["source_frozen_policy"])
    frozen = _read_json(frozen_path)
    if frozen.get("schema") != "jit_frozen_unified_policy_v1" or frozen.get("status") != "frozen":
        raise ValueError("pi0 source is not a frozen unified policy")
    policy = frozen.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("frozen pi0 policy record missing")
    if policy.get("name") != "pi_0" or int(policy.get("iteration", -1)) != 0:
        raise ValueError("warm-start source must be pi_0")
    if policy.get("actor_sha256") != "43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029":
        raise ValueError("pi0 actor identity drift")
    if policy.get("payload_sha256") != "fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719":
        raise ValueError("pi0 payload identity drift")

    source_config = _ORIGINAL_LOAD_FORMAL_CONFIG(Path(policy["formal_config"]))
    expected = CheckpointIdentity(
        config_sha256=source_config.config_sha256,
        xml_sha256=str(policy["xml_sha256"]),
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )
    payload = load_checkpoint(Path(policy["checkpoint"]), expected=expected)
    if int(payload.training_transitions) != int(policy["source_training_transitions"]):
        raise ValueError("pi0 checkpoint transition drift")
    if pytree_sha256(payload.observation_normalizer) != str(policy["normalizer_sha256"]):
        raise ValueError("pi0 normalizer identity drift")
    if pytree_sha256(payload.actor_params) != str(policy["actor_sha256"]):
        raise ValueError("pi0 actor payload drift")
    if pytree_sha256(payload.critic_params) != str(policy["critic_sha256"]):
        raise ValueError("pi0 critic payload drift")
    return (
        payload.observation_normalizer,
        payload.actor_params,
        payload.critic_params,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    raw = _read_json(args.config)
    declared_run = raw.get("run_declaration", {}).get("run_id")
    if declared_run != args.run_id:
        raise ValueError("run-id must match the warm-start config declaration")

    restore_params = _load_pi0_restore_params(args.config)

    def warm_trainer(**kwargs):
        if kwargs.get("restore_params") is not None:
            raise ValueError("unexpected pre-existing PPO restore_params")
        if kwargs.get("restore_value_fn") is not False:
            raise ValueError("unexpected base restore_value_fn setting")
        call = dict(kwargs)
        call["restore_params"] = restore_params
        call["restore_value_fn"] = True
        return ppo_train.train(**call)

    previous_flat = flat_formal.load_unified_formal_config
    previous_canonical = canonical_formal.load_unified_formal_config
    flat_formal.load_unified_formal_config = _load_warm_target_config
    canonical_formal.load_unified_formal_config = _load_warm_target_config
    try:
        result = canonical_formal.run_unified_formal(
            args.config,
            args.run_id,
            trainer=warm_trainer,
        )
    finally:
        flat_formal.load_unified_formal_config = previous_flat
        canonical_formal.load_unified_formal_config = previous_canonical

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
