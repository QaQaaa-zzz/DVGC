"""CPU-only campaign -> student config contract fixture; no production receipt."""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest


@pytest.mark.parametrize('arm',['A','B','C'])
def test_campaign_student_contract_loads_and_writes_probe(monkeypatch,tmp_path,arm):
    import jax.numpy as jp
    from brax.training.acme import running_statistics as rs
    from jit_dvgc.generative_bridge.campaign import CampaignRunner
    from jit_dvgc.generative_bridge import student
    from jit_dvgc.generative_bridge.protocol import atomic_json
    from jit_dvgc.generative_bridge.contracts import file_sha
    from jit_dvgc.handoff_bank import pytree_sha256
    from jit_dvgc import iterative_probe_training,unified_policy_freeze,probe_bank
    norm=rs.init_state(jp.zeros(76));actor=jp.array(0.);critic=jp.array(9.)
    source={'name':'P0','actor_sha256':pytree_sha256(actor),'normalizer_sha256':pytree_sha256(norm)}
    frozen=tmp_path/'source.json';atomic_json(frozen,{'policy':source})
    retention=tmp_path/'retention.npz';np.savez(retention,actor_observation_before=np.ones((5,76)))
    ref={'path':str(retention),'sha256':file_sha(retention),'role':'train','full_success':True}
    support=tmp_path/'support.json';atomic_json(support,{'schema':'jit_iterative_candidate_support_v1'})
    demo=tmp_path/'demo.json';atomic_json(demo,{'schema':'jit_bridge_demo_v1_2','count':0,'entries':[]})
    bank=tmp_path/'bank.json';atomic_json(bank,dict(task='fixture',max_ticks=400,label_interaction_budget=1,max_candidates_per_process=32))
    phase=dict(support_path=str(support),panel_support_path='fixture_panel',retention_ref=ref,
        demo_manifest={'path':str(demo),'sha256':file_sha(demo)})
    runner=CampaignRunner.__new__(CampaignRunner);runner.root=tmp_path;runner.costs=[];runner.source=source
    runner.spec={'source_frozen_policy':str(frozen),'seed':123}
    runner.runtime={'bootstrap_config':'fixture_runtime','bank':str(bank)}
    def make_config(*args,**kwargs):return {'run_declaration':{'run_id':'fixture_'+arm},
        'initialization':{'actor':'warm_start_frozen_unified','source_frozen_policy':str(frozen)},'ppo':{'requested_transitions':128000}}
    monkeypatch.setattr(iterative_probe_training,'make_config',make_config)
    monkeypatch.setattr(student,'load_retention_reference',lambda r:((norm,actor),source))
    warm_path=tmp_path/'warm.pkl'
    with warm_path.open('wb') as stream:pickle.dump((norm,jp.array(.25),jp.array(-99.)),stream)
    warm={'path':str(warm_path),'sha256':file_sha(warm_path),'source_actor_sha256':source['actor_sha256'],'normalizer_sha256':source['normalizer_sha256']}
    def child(name,args,maximum,**kwargs):
        raw=json.loads(Path(args[args.index('--config')+1]).read_text())
        run=Path(kwargs['extra_env']['JIT_RUN_ROOT'])/raw['run_declaration']['run_id'];run.mkdir(parents=True)
        def fake_trainer(**options):
            restored=options['restore_params']
            assert float(restored[1])==(.25 if arm=='C' else 0.)
            assert float(restored[2])==9.  # Saved warmup critic must be ignored.
        student.trainer_from_config(fake_trainer,raw,run)(restore_params=(norm,actor,critic))
        assert (run/'learning_probe.json').exists()
        atomic_json(run/'formal_report.json',dict(completed_training_transitions=128000,train_panel_interactions=0))
        return {}
    runner.child=child
    def freeze(directory,**kwargs):
        Path(directory).mkdir();atomic_json(Path(directory)/'frozen_unified_policy.json',{'policy':{**source,'name':'fixture_'+arm}})
    monkeypatch.setattr(unified_policy_freeze,'freeze_development_checkpoint',freeze)
    monkeypatch.setattr(probe_bank,'lock_probe_bank',lambda raw,path:atomic_json(path,raw))
    result=runner.train_arm(arm,phase,warm if arm=='C' else None)
    assert Path(result['training'],'learning_probe.json').exists()
