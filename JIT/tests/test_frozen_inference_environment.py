"""Frozen policies must never inherit TRAIN reset adapters or augmentation."""
from copy import deepcopy
from types import SimpleNamespace as NS

import jax
import jax.numpy as jp
import numpy as np
import pytest


def constructor_fixture(monkeypatch):
    from jit_dvgc import iterative_probe_training as probe
    import jit_dvgc.config as phase
    import jit_dvgc.unified_env as unified_env
    import jit_dvgc.snapshot_pool as snapshots
    import jit_dvgc.tube_rsi as tube
    import jit_dvgc.unified_diagnostic as diagnostic
    import jit_dvgc.unified_formal as formal
    import jit_dvgc.handoff_snapshot as handoff
    import jit_dvgc.retention_reset as resets
    installs=[]
    support=dict(schema='support',support_sha256='support-hash',entries=[
        dict(phase='upstream',key='up',snapshot='up-snapshot',sampling_weight=1.),
        dict(phase='downstream',key='down',snapshot='down-snapshot',sampling_weight=1.)])
    monkeypatch.setattr(probe,'read',lambda path:support)
    monkeypatch.setattr(probe,'resolve_bootstrap_config',lambda path:path)
    monkeypatch.setattr(formal,'load_unified_policy_formal_config',lambda path:None)
    monkeypatch.setattr(diagnostic,'_load_runtime',lambda config:(None,None,'physical-bootstrap',None))
    monkeypatch.setattr(phase,'load_config',lambda path,**kwargs:NS(
        config_sha256=str(path)+'-hash',model={'xml_sha256':'xml'}))
    monkeypatch.setattr(snapshots.SnapshotPool,'from_paths',lambda *args,**kwargs:'snapshot-pool')
    monkeypatch.setattr(handoff,'compatibility_identity',lambda env:'physical-identity')
    monkeypatch.setattr(tube,'TubeRSIPool',lambda *args:NS(sample=lambda key:pytest.fail('sampled TRAIN RSI')))
    monkeypatch.setattr(resets,'install_declared_student_reset',lambda env,raw:installs.append(raw['student_reset_inventory']))
    class Physics:
        def __init__(self,*args,**kwargs):self._bundle=NS(xml_sha256='xml')
        def _reset_jump_start_unified(self,key):return ('complete-start',key)
        def step(self,state,action):return {'state':state,'physical_action':action}
    monkeypatch.setattr(unified_env,'UnifiedTubeRSIEnv',Physics)
    raw=dict(bootstrap_formal_config='bootstrap',support='training-support',panel_support='panel-support',
        student_reset_inventory='teacher-inventory',success_criterion='stable_forward_recovery',
        reward_mode='original_all_phases',input_files={},training_action_pulse=dict(
            pulse_scope='full_task_start_only',onset_selection='uniform_complete_start',
            onsets=[0,5,10,15],steps=3,probability=1.,delta_limit=[.25]*4))
    config=NS(raw=raw,ppo=NS(episode_horizon=400),up_config_path='up',down_config_path='down',
        up_config_sha256='up-hash',down_config_sha256='down-hash',config_sha256='original-config-sha')
    return probe,config,installs


def test_inference_constructor_uses_complete_reset_and_steps_metadata_free_snapshot(monkeypatch):
    probe,config,installs=constructor_fixture(monkeypatch)
    before=deepcopy(config.raw)
    artifact,env=probe.build_environment(config,inference=True)
    assert not installs and env._training_action_pulse is None
    assert env.reset(jax.random.PRNGKey(7))[0]=='complete-start'
    state=NS(info={})
    action=jp.array([.1,.2,.3,.4])
    result=env.step(state,action)
    assert result['state'] is state
    np.testing.assert_array_equal(result['physical_action'],action)
    assert env._reward_mode=='original_all_phases'
    assert config.raw==before and config.config_sha256=='original-config-sha'


def test_default_training_constructor_keeps_declared_reset_inventory_and_pulses(monkeypatch):
    probe,config,installs=constructor_fixture(monkeypatch)
    _,env=probe.build_environment(config)
    assert installs==['teacher-inventory']
    assert env._training_action_pulse==config.raw['training_action_pulse']


def test_unified_constructor_forwards_inference_without_rewriting_config(monkeypatch):
    from jit_dvgc import unified_formal as formal
    from jit_dvgc import iterative_probe_training as probe
    config=NS(schema=probe.SCHEMA,raw={'student_reset_inventory':'TRAIN'},config_sha256='original')
    monkeypatch.setattr(formal,'load_unified_policy_formal_config',lambda path:config)
    def build(actual,**kwargs):
        assert actual is config and kwargs=={'inference':True}
        return 'artifact','inference-env'
    monkeypatch.setattr(probe,'build_environment',build)
    returned,artifact,env=formal.build_unified_formal_environment('frozen-config',inference=True)
    assert returned is config and artifact=='artifact' and env=='inference-env'


@pytest.mark.parametrize('continuous',[False,True])
def test_explorer_and_full_dev_networks_request_inference_constructor(monkeypatch,continuous):
    import jit_dvgc.pulse_exploration_runtime as runtime
    import jit_dvgc.unified_formal as formal
    import jit_dvgc.iterative_probe_training as probe
    class Captured(Exception):pass
    member={'name':'policy','policy':{'formal_config':'config'}}
    monkeypatch.setattr(runtime,'load_probe_bank',lambda path:{'members':[member]})
    monkeypatch.setattr(runtime,'read',lambda path:{'continuous_learner':{}} if continuous else {})
    monkeypatch.setattr(probe,'load_config',lambda path,**kwargs:'configuration')
    def build(*args,**kwargs):
        assert kwargs.get('inference') is True
        raise Captured()
    monkeypatch.setattr(formal,'build_unified_formal_environment',build)
    monkeypatch.setattr(probe,'build_environment',build)
    with pytest.raises(Captured):
        runtime.networks({'bank':'bank','proposer':'policy','frozen_explorer_evaluation':True})


def test_frozen_suffix_runtime_requests_inference_constructor(monkeypatch,tmp_path):
    from jit_dvgc.exploration_continuation import FrozenSuffixEvaluator
    import jit_dvgc.unified_formal as formal
    import jit_dvgc.probe_bank as banks
    import jit_dvgc.unified_policy_freeze as frozen
    class Captured(Exception):pass
    policy={'formal_config':'original','checkpoint':'saved-student32k'}
    runner=FrozenSuffixEvaluator.__new__(FrozenSuffixEvaluator)
    runner._runtimes={};runner.bank_path=tmp_path/'bank';runner.bank={}
    runner.members={'student':{'frozen_policy':'frozen','policy':policy}}
    monkeypatch.setattr(jax,'default_backend',lambda:'gpu')
    monkeypatch.setattr(banks,'load_probe_bank',lambda path:runner.bank)
    monkeypatch.setattr(frozen,'load_frozen_unified_manifest',lambda path:{'policy':policy})
    def build(path,**kwargs):
        assert kwargs.get('inference') is True
        raise Captured()
    monkeypatch.setattr(formal,'build_unified_formal_environment',build)
    with pytest.raises(Captured):runner._runtime('student')


@pytest.mark.parametrize('inference',[False,True])
def test_legacy_formal_runtime_complete_reset_is_explicit_and_training_default_unchanged(monkeypatch,inference):
    import jit_dvgc.unified_formal as formal
    config=NS(schema='legacy',runtime_naccdmax=1024,reset_mixture=NS(natural_reset_probability=.2))
    monkeypatch.setattr(formal,'load_unified_policy_formal_config',lambda path:config)
    monkeypatch.setattr(formal,'_load_runtime',lambda c:(NS(model={'xml_sha256':'xml'}),None,'artifact',None))
    class Runtime:
        def __init__(self,*args,**kwargs):
            self._bundle=NS(xml_sha256='xml');self._training_action_pulse='TRAIN'
        def reset(self,key):return 'training-reset'
        def _reset_jump_start_unified(self,key):return 'complete-start'
    returned,_,env=formal.build_unified_formal_environment('original',env_factory=Runtime,inference=inference)
    assert returned is config
    assert env.reset(None)==('complete-start' if inference else 'training-reset')
    assert env._training_action_pulse==(None if inference else 'TRAIN')


def test_real_training_reset_step_remains_strict_about_training_metadata():
    from jit_dvgc.retention_reset import FourPoolReset
    from .test_retention_t1_reset import saved_state
    adapter=FourPoolReset.__new__(FourPoolReset)
    adapter.original_step=lambda state,action:state
    state=saved_state(0)
    state=state.replace(info={k:v for k,v in state.info.items() if not k.startswith('retention_reset_')})
    with pytest.raises(KeyError,match='retention_reset_group'):
        adapter.step(state,jp.zeros(4))
