import jax
import jax.numpy as jp
import numpy as np
import pytest
from jit_dvgc.generative_bridge import rollout


def test_closed_loop_prefix_handoff_uses_current_observation_and_continuous_clock():
    def prefix(obs, key):
        return jp.full(4, obs['clock'] + 1.), {}
    def tail(obs, key):
        return jp.full(4, -obs['clock'] - 1.), {}
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    def frame(state, tick):
        action = rollout.closed_loop_action(tick, state, keys, tail, prefix)
        return {'clock':state['clock']+1, 'recovery':state['recovery']+1}, action
    final, actions = jax.jit(lambda: jax.lax.scan(frame, {'clock':jp.zeros(2),
                'recovery':jp.zeros(2)}, jp.arange(18)))()
    np.testing.assert_array_equal(actions[:,0,0], list(range(1,17))+[-17,-18])
    np.testing.assert_array_equal(final['clock'], [18,18])
    np.testing.assert_array_equal(final['recovery'], [18,18])


def test_closed_loop_options_fail_closed_and_label_composites():
    members = {'old':{'policy':{'actor_sha256':'oldhash'}},
               'student':{'policy':{'actor_sha256':'newhash'}}}
    spec = dict(order=['old'], horizon=400, success_criterion='stable_forward_recovery',
                closed_loop_prefix_policy='student')
    assert rollout.validate_evaluation_options(spec, members) == 'student'
    for change in [dict(bridge_action_plan={}),dict(reuse_results='old'),dict(horizon=16),
                   dict(closed_loop_prefix_policy='missing')]:
        with pytest.raises(ValueError):
            rollout.validate_evaluation_options({**spec, **change}, members)
    provenance = rollout.evaluation_controller_provenance(spec, members['old']['policy'], members)
    assert provenance['controller_kind'] == 'composite_closed_loop'
    assert provenance['prefix_actor_sha256'] == 'newhash'
    assert provenance['tail_actor_sha256'] == 'oldhash'
    assert provenance['actor_witness_eligible'] is False
    overridden = rollout.evaluation_controller_provenance(dict(warmup_initializer={'path':'x'}),
                        dict(actor_sha256='candidate',inference_override={}), members)
    assert overridden['actor_sha256'] == 'candidate'
    assert overridden['actor_witness_eligible'] is False


def test_warmup_evaluation_uses_candidate_hash_and_leaves_source_frozen(tmp_path, monkeypatch):
    import pickle
    from types import SimpleNamespace
    from jit_dvgc.checkpoint import CheckpointPayload, CheckpointIdentity
    from jit_dvgc import checkpoint, unified_formal, unified_training, ppo
    from jit_dvgc.handoff_bank import pytree_sha256
    from jit_dvgc.generative_bridge.contracts import file_sha
    identity = CheckpointIdentity('config','xml',(),(),())
    norm = jp.zeros(2); actor = {'weights':jp.zeros(4)}; candidate_actor = {'weights':jp.ones(4)*.3}
    source = CheckpointPayload(identity,123,norm,actor,{'value':jp.ones(1)})
    policy_record = dict(actor_sha256=pytree_sha256(actor),normalizer_sha256=pytree_sha256(norm),
                         formal_config='source.json',checkpoint='checkpoint',xml_sha256='xml')
    path = tmp_path/'warm.pkl'
    with path.open('wb') as f: pickle.dump((norm,candidate_actor,None), f)
    reference = dict(path=str(path),sha256=file_sha(path),source_actor_sha256=policy_record['actor_sha256'],
                     normalizer_sha256=policy_record['normalizer_sha256'])
    monkeypatch.setattr(unified_formal,'load_unified_policy_formal_config',lambda _: SimpleNamespace())
    monkeypatch.setattr(unified_training,'checkpoint_identity',lambda cfg,env:identity)
    monkeypatch.setattr(checkpoint,'load_checkpoint',lambda path,expected:source)
    monkeypatch.setattr(ppo,'make_checkpoint_policy',lambda env,payload,deterministic:
                        lambda obs,key:(payload.actor_params['weights'],{}))
    policy, actual = rollout.warmup_evaluation_policy(object(),policy_record,reference)
    np.testing.assert_allclose(policy({},jax.random.PRNGKey(0))[0],.3)
    assert actual['actor_sha256'] == pytree_sha256(candidate_actor)
    assert actual['actor_sha256'] != policy_record['actor_sha256']
    np.testing.assert_array_equal(source.actor_params['weights'],0.)
    assert rollout.evaluation_controller_provenance({'warmup_initializer':reference},actual,{})['actor_witness_eligible'] is False
    with pytest.raises(ValueError,match='hash drift'):
        rollout.warmup_evaluation_policy(object(),policy_record,{**reference,'sha256':'wrong'})
