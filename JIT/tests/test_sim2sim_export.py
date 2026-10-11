"""Offline export contracts; no environment or physics is constructed."""
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks

from jit_dvgc.checkpoint import CheckpointIdentity, CheckpointPayload, save_checkpoint
from jit_dvgc.config import canonical_sha256
from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from jit_dvgc.handoff_bank import pytree_sha256


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def export_case(tmp_path):
    model = tmp_path / 'model.xml'
    model.write_text('<mujoco><option timestep="0.001"/></mujoco>')
    config = dict(model=dict(xml_path=str(model), xml_sha256=digest(model)),
                  action=dict(joint_target_semantics='keyframe_centered_absolute'),
                  success_criterion='stable_forward_recovery',
                  descent=dict(recovery_ticks=25, continuous_stability=True,
                               stable_min_forward_velocity=.5,
                               stable_max_abs_roll=.17453292519943295,
                               stable_max_abs_pitch=.2617993877991494,
                               stable_max_wheel_clearance=.02,
                               max_wheel_penetration=.01, terminate_on_body_contact=True))
    task = tmp_path / 'task.json'
    task.write_text(json.dumps(config))
    net = networks.make_ppo_networks({'state': 76, 'privileged_state': 106}, 4,
        policy_hidden_layer_sizes=(256, 256, 256), value_hidden_layer_sizes=(256, 256, 256),
        policy_obs_key='state', value_obs_key='privileged_state', distribution_type='tanh_normal',
        preprocess_observations_fn=running_statistics.normalize)
    norm = running_statistics.init_state({'state': jnp.zeros(76), 'privileged_state': jnp.zeros(106)})
    norm = norm.replace(mean={'state': jnp.arange(76) / 10, 'privileged_state': jnp.zeros(106)},
                        std={'state': jnp.full(76, 2.), 'privileged_state': jnp.ones(106)})
    payload = CheckpointPayload(CheckpointIdentity(canonical_sha256(config), digest(model),
        ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS, ACTION_ORDER), 32000, norm,
        net.policy_network.init(jax.random.PRNGKey(71)), net.value_network.init(jax.random.PRNGKey(72)))
    checkpoint = tmp_path / 'checkpoint'
    save_checkpoint(checkpoint, payload)
    observations = np.random.default_rng(3).normal(size=(9, 76)).astype('f4')
    obsfile = tmp_path / 'observations.npz'
    np.savez(obsfile, observations=observations)
    manifest = tmp_path / 'observations.json'
    manifest.write_text(json.dumps(dict(schema='jit_sim2sim_observations_v1',
        actor_frame_fields=list(ACTOR_FRAME_FIELDS), actor_task_fields=list(ACTOR_TASK_FIELDS),
        xml_sha256=digest(model), entries=[dict(path=str(obsfile), sha256=digest(obsfile),
        key='observations', role='TRAIN')], final_test_used=False)))
    args = dict(checkpoint=checkpoint, task_config=task, observations_manifest=manifest,
        output=tmp_path / 'export', repository=tmp_path,
        expected_payload_sha256=digest(checkpoint / 'payload.pkl'),
        expected_actor_sha256=pytree_sha256(payload.actor_params),
        expected_normalizer_sha256=pytree_sha256(norm))
    return args, payload, net, observations


def exporter():
    from jit_dvgc.sim2sim_export import export_checkpoint
    return export_checkpoint


def test_exported_npz_matches_source_inference_and_binds_normalizer(export_case):
    args, payload, net, observations = export_case
    report = exporter()(**args)
    fn = networks.make_inference_fn(net)((payload.observation_normalizer,
        payload.actor_params, payload.critic_params), deterministic=True)
    expected = np.asarray(jax.vmap(lambda obs: fn({'state': obs}, jax.random.PRNGKey(0))[0])(observations))
    from jit_dvgc.sim2sim_export import numpy_action
    with np.load(args['output'] / 'actor.npz') as bundle:
        np.testing.assert_allclose(numpy_action(dict(bundle), observations), expected, atol=3e-5, rtol=0)
        np.testing.assert_array_equal(bundle['mean'], np.arange(76, dtype='f4') / 10)
    assert report['parity']['observation_count'] == 9
    assert report['new_physics_control_steps'] == 0
    assert report['engine_evaluation_status'] == 'not_run'
    assert report['source']['normalizer_sha256'] == args['expected_normalizer_sha256']
    assert report['task_contract']['success']['recovery_ticks'] == 25
    assert report['task_contract']['success']['stable_min_forward_velocity'] == .5
    assert report['target_adapter_ready'] is False


def test_runtime_timing_override_is_distinct_from_raw_xml(export_case):
    args, *_ = export_case
    report = exporter()(**args)
    contract = report['task_contract']
    assert contract['sim_dt_s'] == .005
    assert contract['control_dt_s'] == .02
    assert contract['timing']['raw_xml_timestep_s'] == [.001]
    assert contract['timing']['physics_substeps_per_control'] == 4
    assert contract['timing']['json_declares_timing'] is False
    assert contract['timing']['validation'] == 'source_constants_and_override_audited; no environment instantiated'


@pytest.mark.parametrize('field', ['expected_payload_sha256', 'expected_actor_sha256', 'expected_normalizer_sha256'])
def test_mismatched_identity_cannot_produce_export(export_case, field):
    args, *_ = export_case
    args[field] = '0' * 64
    with pytest.raises(ValueError, match='identity|sha256'):
        exporter()(**args)
    assert not args['output'].exists()


def test_test_role_rejected_before_opening_observation_archive(export_case):
    args, *_ = export_case
    manifest = json.loads(args['observations_manifest'].read_text())
    manifest['entries'][0].update(role='TEST', path='/does/not/exist/test.npz')
    args['observations_manifest'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='role|TEST'):
        exporter()(**args)
    assert not args['output'].exists()


def test_changed_task_configuration_is_not_silently_rebound(export_case):
    args, *_ = export_case
    task = json.loads(args['task_config'].read_text())
    task['descent']['stable_min_forward_velocity'] = 0.
    args['task_config'].write_text(json.dumps(task))
    with pytest.raises(ValueError, match='config'):
        exporter()(**args)
    assert not args['output'].exists()


def test_existing_output_is_preserved(export_case):
    args, *_ = export_case
    args['output'].mkdir()
    marker = args['output'] / 'historical.txt'
    marker.write_text('unchanged')
    with pytest.raises(FileExistsError):
        exporter()(**args)
    assert marker.read_text() == 'unchanged'


@pytest.mark.parametrize('observations', [np.zeros((3, 106)), np.full((3, 76), np.nan)])
def test_invalid_saved_observations_cannot_produce_export(export_case, observations):
    args, *_ = export_case
    manifest = json.loads(args['observations_manifest'].read_text())
    entry = manifest['entries'][0]
    from pathlib import Path
    np.savez(entry['path'], observations=observations)
    entry['sha256'] = digest(Path(entry['path']))
    args['observations_manifest'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='76|finite'):
        exporter()(**args)
    assert not args['output'].exists()
