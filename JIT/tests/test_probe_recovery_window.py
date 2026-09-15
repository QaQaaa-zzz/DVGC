"""Probe training inherits the frozen continuous recovery window and its identity."""
from copy import deepcopy
from pathlib import Path

import pytest

from jit_dvgc.config import load_config as load_phase_config
from jit_dvgc.evidence_integrity import canonical_sha256
from jit_dvgc.iterative_probe_training import load_config, make_config
from jit_dvgc.jump_evidence_validation import file_sha, read, write


@pytest.fixture
def recovery_source(tmp_path, jit_root):
    def create(filename, *, descent_updates=None, hash_kind="canonical"):
        phase = read(jit_root / "configs" / filename)
        phase["descent"].update(descent_updates or {})
        down_path = tmp_path / "down.json"
        write(down_path, phase)
        bootstrap_path = jit_root / "configs/pi_unified_formal.json"
        bootstrap = read(bootstrap_path)
        source = deepcopy(bootstrap)
        source["inputs"]["up_config_path"] = str(
            jit_root / "configs/phase_u_continuation_smoke.json"
        )
        source["inputs"]["down_config_path"] = str(down_path)
        source["inputs"]["down_config_sha256"] = (
            canonical_sha256(phase) if hash_kind == "canonical" else file_sha(down_path)
        )
        source["success_criterion"] = "stable_forward_recovery"
        source["reward_contract"] = "Frozen phase rewards and configured stable recovery endpoint"
        source["ppo"].update(num_parallel_envs=128, batch_size=16, num_minibatches=8)
        source_path = tmp_path / "source_config.json"
        write(source_path, source)
        initializer = tmp_path / "frozen.json"
        write(initializer, {"policy": {"formal_config": str(source_path)}})
        support = dict(
            schema="jit_iterative_witnessed_support_v1", role="train",
            final_test_used=False, inputs={}, entries=[
                dict(phase=phase_name, key=phase_name, snapshot=phase_name, witnessed=True)
                for phase_name in ("upstream", "downstream")
            ],
        )
        support["support_sha256"] = canonical_sha256(support)
        support_path = tmp_path / "support.json"
        write(support_path, support)
        return support_path, initializer, bootstrap_path, tmp_path / "training.json", source
    return create


def generate(source):
    support, initializer, bootstrap, output, _ = source
    return make_config(support, initializer, bootstrap, output, "recovery_fixture", 1,
                       3200, 123, checkpoints=[3200], panel_samples_per_phase=1)


@pytest.mark.parametrize("filename,ticks", [
    ("descent_stable_forward_0p5s.json", 25),
    ("descent_stable_forward_2s.json", 100),
])
def test_supplementation_accepts_frozen_window_and_preserves_source_contract(
    recovery_source, filename, ticks,
):
    source = recovery_source(filename)
    original = source[-1]
    raw = generate(source)
    resolved = load_config(source[3])
    phase = load_phase_config(Path(resolved.down_config_path))
    assert phase.descent.recovery_ticks == ticks
    assert resolved.down_config_sha256 == canonical_sha256(read(resolved.down_config_path))
    assert raw["inputs"] == original["inputs"]
    assert raw["reward_contract"] == original["reward_contract"]
    assert raw["reward_mode"] == "phase_recovery"
    assert raw["jump_start_probability"] == .2
    assert raw["ppo"] == {**original["ppo"], "requested_transitions": 3200, "seed": 123}
    assert raw["fixed_train_panel"]["success_criterion"] == "stable_forward_recovery"
    assert resolved.formal.checkpoint_transitions == (0, 3200)


def test_recovery_source_file_hash_cannot_replace_canonical_config_hash(recovery_source):
    source = recovery_source("descent_stable_forward_0p5s.json", hash_kind="file")
    with pytest.raises(ValueError, match="canonical configuration hash"):
        generate(source)


@pytest.mark.parametrize("refresh_file_lock", [False, True])
def test_changed_recovery_window_requires_new_file_and_canonical_locks(
    recovery_source, refresh_file_lock,
):
    source = recovery_source("descent_stable_forward_0p5s.json")
    raw = generate(source)
    down_path = Path(raw["inputs"]["down_config_path"])
    phase = read(down_path)
    phase["descent"]["recovery_ticks"] = 100
    write(down_path, phase)
    if refresh_file_lock:
        raw["input_files"][str(down_path)] = file_sha(down_path)
        write(source[3], raw)
    with pytest.raises(ValueError, match="recovery phase config drift" if refresh_file_lock
                       else "probe training input changed"):
        load_config(source[3])


@pytest.mark.parametrize("updates", [
    {"continuous_stability": False},
    {"recovery_ticks": 0},
    {"recovery_ticks": -1},
    {"recovery_ticks": 1.5},
    {"recovery_ticks": True},
])
def test_stable_endpoint_rejects_noncontinuous_or_invalid_tick_window(recovery_source, updates):
    source = recovery_source("descent_stable_forward_0p5s.json", descent_updates=updates)
    with pytest.raises(ValueError, match="continuous|recovery_ticks"):
        generate(source)
