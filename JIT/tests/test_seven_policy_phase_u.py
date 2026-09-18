"""CPU tests of preparation, fail-closed execution and unchanged outcome reporting."""
import copy
from pathlib import Path

import numpy as np
import pytest

from jit_dvgc.jump_evidence_validation import read, write, file_sha
from jit_dvgc.evidence_integrity import canonical_sha256
from jit_dvgc import seven_policy_phase_u as pipeline


@pytest.fixture
def declarations(tmp_path):
    from jit_dvgc.checkpoint import CheckpointIdentity, CheckpointPayload, save_checkpoint
    from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
    from jit_dvgc.handoff_bank import pytree_sha256
    config = read(Path(__file__).parents[1] / 'configs/phase_u_continuation_10m.json')
    historical = tmp_path / 'historical.json'; write(historical, config)
    sources=[]
    for i in range(4):
        checkpoint=tmp_path/f'source{i}/checkpoints/transition_128000'
        identity=CheckpointIdentity(canonical_sha256(config),config['model']['xml_sha256'],
                                    ACTOR_FRAME_FIELDS,ACTOR_TASK_FIELDS,ACTION_ORDER)
        payload=CheckpointPayload(identity,128000,{'mean':np.array([i+1.])},
                                  {'w':np.array([i+2.])},{'w':np.array([i+3.])})
        save_checkpoint(checkpoint,payload)
        report=tmp_path/f'report{i}.json';write(report,{'status':'completed'})
        formal=tmp_path/f'formal{i}.json';write(formal,config)
        policy=dict(name=f'p{i}',checkpoint=str(checkpoint),formal_config=str(formal),
            formal_config_sha256=identity.config_sha256,formal_config_file_sha256=file_sha(formal),
            source_formal_report=str(report),source_formal_report_sha256=file_sha(report),
            xml_sha256=identity.xml_sha256,actor_frame_fields=list(ACTOR_FRAME_FIELDS),
            actor_task_fields=list(ACTOR_TASK_FIELDS),action_order=list(ACTION_ORDER),
            checkpoint_identity_sha256=file_sha(checkpoint/'identity.json'),payload_sha256=file_sha(checkpoint/'payload.pkl'),
            source_training_transitions=128000,normalizer_sha256=pytree_sha256(payload.observation_normalizer),
            actor_sha256=pytree_sha256(payload.actor_params),critic_sha256=pytree_sha256(payload.critic_params),
            policy_role='development_checkpoint',data_role='train')
        frozen=tmp_path/f'frozen{i}.json'
        manifest=dict(schema='jit_frozen_development_checkpoint_v1',status='frozen',immutable_parameters=True,
            copied_checkpoint=False,training_transitions=0,environment_interactions=0,expert_switching_used=False,
            policy=policy,claim_boundary={key:False for key in ('envelope_expansion_authority','pi_unified_star_claim','jce_jel_claim','certified_safe_tube_claim')})
        manifest['freeze_protocol_sha256']=canonical_sha256(manifest);write(frozen,manifest)
        bank=tmp_path/f'bank{i}.json';write(bank,{'members':[{'name':f'p{i}','policy':policy,'frozen_policy':str(frozen)}]})
        template=tmp_path/f'template{i}.json'
        ev=dict(bank=str(bank),proposer=f'p{i}',horizon=400,pulse_steps=3,pulse_start_schedule=[0],
                pulse_batch_mode='per_environment',full_episode_rollout=True,controller_mode='fixed_random',
                delta_limit=[.25]*4,success_criterion='stable_forward_recovery',reward_mode='original_all_phases')
        if i==3:
            ev['phase_policy']=dict(name='historical',source_phase_config=str(historical),source_checkpoint=str(checkpoint),
                input_files={str(historical):file_sha(historical),str(checkpoint/'identity.json'):file_sha(checkpoint/'identity.json'),str(checkpoint/'payload.pkl'):file_sha(checkpoint/'payload.pkl')})
        write(template,ev)
        sources.append(dict(key=f'm{i}',label=f'Method {i}',template_path=str(template),frozen_policy=str(frozen),
                            warm_start=i<3,**({'descendant_key':f'd{i}'} if i<3 else {})))
    declaration=tmp_path/'sources.json';write(declaration,dict(schema='jit_seven_policy_sources_v1',sources=sources))
    previous=tmp_path/'previous';previous.mkdir()
    write(previous/'spec.json',dict(repo=str(Path(__file__).parents[2]),python='/usr/bin/python3',root_qpos_address=0))
    write(previous/'status.json',dict(phase='completed'))
    return declaration,historical,previous,tmp_path/'experiment'


def prepare(declarations):
    sources,historical,previous,output=declarations
    return pipeline.prepare_experiment(sources,historical,previous,output,training_seeds=[11,12,13],condition_seeds=[21,22,23,24],execution_repository=Path(__file__).parents[2])


def test_prepare_locks_sources_and_exact_budget_without_runtime_changes(declarations):
    spec=prepare(declarations);original=read(declarations[1])
    assert len(spec['methods'])==7 and len(spec['arms'])==3
    assert [(c['onset'],c['episodes'],c['batch_size']) for c in spec['conditions']]==[(i,1000,256) for i in (0,5,10,15)]
    for arm in spec['arms']:
        config=read(arm['config'])
        for key in ('schema','phase','model','action','reset','events','physical_limits','reward','training_wrapper'):
            assert config[key]==original[key]
        assert config['ppo']['requested_transitions']==14991360
        assert config['ppo']['seed'] in (11,12,13)
        assert config['formal']['checkpoint_transitions'][0]==0
        assert config['formal']['checkpoint_transitions'][-1]==14991360
        assert all(n%24576==0 for n in config['formal']['checkpoint_transitions'])
    locks=read(declarations[3]/'source_lock.json')['input_files']
    for source in read(declarations[0])['sources']:
        assert source['frozen_policy'] in locks and source['template_path'] in locks
        policy=read(source['frozen_policy'])['policy']
        assert str(Path(policy['checkpoint'])/'identity.json') in locks
        assert str(Path(policy['checkpoint'])/'payload.pkl') in locks
    assert read(declarations[3]/'ACTIVE_RUN.json')['execution']==str(declarations[3]/'status.json')
    with pytest.raises(FileExistsError):prepare(declarations)


def test_preparation_rejects_changed_source_identity_before_output(declarations):
    frozen=Path(read(declarations[0])['sources'][0]['frozen_policy'])
    manifest=read(frozen);manifest['policy']['actor_sha256']='wrong';write(frozen,manifest)
    with pytest.raises(ValueError):prepare(declarations)
    assert not declarations[3].exists()


def test_run_rejects_source_drift_and_never_launches(declarations,monkeypatch):
    prepare(declarations);Path(declarations[1]).write_text('{}')
    monkeypatch.setattr(pipeline,'_run_child',lambda *a,**k:pytest.fail('launched changed source'))
    with pytest.raises(ValueError,match='drift'):pipeline.run_experiment(declarations[3]/'spec.json')
    assert read(declarations[3]/'status.json')['phase']=='error'


def test_failed_training_stops_before_next_arm_or_evaluation(declarations,monkeypatch):
    prepare(declarations);calls=[]
    def fail(command,log,**kwargs):
        calls.append(command);Path(log).parent.mkdir(parents=True,exist_ok=True);Path(log).write_text('child failure\n');raise RuntimeError('child failed')
    monkeypatch.setattr(pipeline,'_run_child',fail)
    with pytest.raises(RuntimeError,match='child failed'):pipeline.run_experiment(declarations[3]/'spec.json')
    assert len(calls)==1
    assert read(declarations[3]/'status.json')['phase']=='error'
    assert not (declarations[3]/'conditions').exists()


def _fake_tape(path, ev):
    from jit_dvgc.constants import END_TIMEOUT, END_RECOVERY_SUCCESS
    n=ev['num_envs'];h=ev['horizon'];onset=ev['pulse_start_schedule'][0]
    alive=np.ones((h,n),bool);alive[2:,1]=False
    tape={key:np.zeros((h,n),bool) for key in ('success','physical_failure','terminal','first_valid_contact')}
    tape.update(prefix_mask=alive,mask=alive & ((np.arange(h)[:,None]>=onset)&(np.arange(h)[:,None]<onset+3)),
        qpos=np.zeros((h,n,3)),time=np.broadcast_to((np.arange(h)+1)[:,None]*.02,(h,n)),
        front_wheel_clearance=np.zeros((h,n)),rear_wheel_clearance=np.zeros((h,n)),
        recovery_ticks=np.zeros((h,n),int),end_code=np.zeros((h,n),int))
    tape['qpos'][:,:,0]=2.5+np.arange(h)[:,None]*.01;tape['qpos'][:,:,2]=.2
    tape['success'][-1,0]=True;tape['physical_failure'][1,1]=True
    tape['terminal'][-1,:]=True;tape['terminal'][1,1]=True;tape['end_code'][-1,:]=END_TIMEOUT
    tape['end_code'][-1,0]=END_RECOVERY_SUCCESS;tape['first_valid_contact'][h-5,:]=True
    tape['delta']=np.ones((h,n,4))*.1
    window=(np.arange(h)>=onset)&(np.arange(h)<onset+3)
    tape['requested_delta']=tape['delta']*window[:,None,None];tape['effective_delta']=tape['requested_delta']*alive[:,:,None]
    path.mkdir(parents=True);np.savez_compressed(path/'prefixes.npz',**tape)
    write(path/'status.json',dict(phase='completed',charged_interactions=h*n))


def test_end_to_end_sequence_fixed_windows_and_real_verification_report(declarations,monkeypatch):
    sources=read(declarations[0])
    for source in sources['sources']:
        template=read(source['template_path']);template['horizon']=20;write(source['template_path'],template)
    spec=prepare(declarations);events=[]
    def child(command,log,**kwargs):
        Path(log).parent.mkdir(parents=True,exist_ok=True);Path(log).write_text('fixture child\n')
        if '_train-arm' in command:
            events.append('train:'+command[-1]);return
        ev=read(command[command.index('--spec')+1]);events.append('eval')
        _fake_tape(Path(command[command.index('--output')+1]),ev)
    def freeze(spec,arm):
        events.append('freeze:'+arm['key'])
        phase=dict(name=arm['key'],source_phase_config=arm['config'],source_checkpoint='fixture',input_files={})
        manifest=dict(schema='jit_frozen_phase_u_descendant_v1',status='frozen',phase_policy=phase,
                      verification=dict(total_environment_transitions=pipeline.TARGET+100,training_transitions=pipeline.TARGET))
        manifest['freeze_protocol_sha256']=canonical_sha256(manifest)
        path=Path(arm['output_manifest']);path.parent.mkdir(exist_ok=True);write(path,manifest)
        return manifest
    monkeypatch.setattr(pipeline,'_run_child',child);monkeypatch.setattr(pipeline,'freeze_descendant',freeze)
    summary=pipeline.run_experiment(declarations[3]/'spec.json')
    assert events[:6]==['train:d0','freeze:d0','train:d1','freeze:d1','train:d2','freeze:d2']
    assert len(events[6:])==112
    assert summary['episodes']==28000
    assert summary['training_transitions']==3*pipeline.TARGET
    assert summary['charged_interactions']==3*(pipeline.TARGET+100)+560000
    for condition in spec['conditions']:
        cspec=read(Path(condition['output'])/'spec.json')
        assert cspec['contract']['pulse_start_schedule']==[condition['onset']]
        assert len(cspec['methods'])==7
        assert [batch['count'] for batch in cspec['batches']]==[256,256,256,232]
        assert all(m['template']['pulse_start_schedule']==[condition['onset']] for m in cspec['methods'])
        assert summary['conditions'][str(condition['onset'])]['m0']['episodes']==1000
        assert summary['conditions'][str(condition['onset'])]['m0']['successes']==4
    assert read(declarations[3]/'status.json')['phase']=='completed'
    assert Path(summary['episodes_csv']).is_file()
    assert all(Path(p).is_file() for p in summary['density_panels'])
    with pytest.raises(ValueError,match='already attempted'):pipeline.run_experiment(declarations[3]/'spec.json')


def test_cli_advertises_prepare_run_report():
    import os
    import subprocess
    import sys
    cli=Path(__file__).parents[1]/'cli/run_seven_policy_phase_u.py'
    result=subprocess.run([sys.executable,str(cli),'--help'],capture_output=True,text=True,env=os.environ)
    assert result.returncode==0,result.stderr
    assert all(word in result.stdout for word in ('prepare','run','report'))


def test_identity_equivalent_warm_start_wrapper_can_differ_from_bank_manifest(declarations):
    declared=read(declarations[0]);source=declared['sources'][0]
    wrapper=Path(source['frozen_policy']).with_name('warm_start_wrapper.json')
    write(wrapper,read(source['frozen_policy']));source['frozen_policy']=str(wrapper)
    write(declarations[0],declared)
    spec=prepare(declarations)
    assert spec['arms'][0]['frozen_policy']==str(wrapper)
    assert str(wrapper) in read(declarations[3]/'source_lock.json')['input_files']


def test_freeze_retains_phase_u_checkpoint_identity_and_rejects_unverified_final(declarations,monkeypatch):
    from types import SimpleNamespace
    from jit_dvgc import provenance, config as config_module
    from jit_dvgc.checkpoint import CheckpointIdentity, CheckpointPayload, save_checkpoint
    from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
    spec=prepare(declarations);arm=spec['arms'][0];run=Path(arm['run_dir']);run.mkdir(parents=True)
    raw=read(arm['config']);write(run/'resolved_config.json',raw)
    report=dict(requested_training_transitions=pipeline.TARGET,starting_training_transition=0,
        completed_training_transitions=pipeline.TARGET,segment_training_transitions=pipeline.TARGET,
        brax_evaluation_transitions=0,fixed_evaluation_transitions=40,diagnostic_transitions=40,
        checkpoint_transitions=list(pipeline.SCHEDULE),evaluated_transitions=list(pipeline.SCHEDULE[1:]),
        final_metrics={'loss':1.},checkpoint_restored=True,resume_semantics='parameter_warm_start_optimizer_reset')
    write(run/'formal_report.json',report)
    write(run/'actor_initialization.json',dict(source_frozen_policy=arm['frozen_policy'],critic_fresh=True,optimizer_fresh=True))
    expected=CheckpointIdentity(canonical_sha256(raw),raw['model']['xml_sha256'],ACTOR_FRAME_FIELDS,ACTOR_TASK_FIELDS,ACTION_ORDER)
    checkpoint=run/'checkpoints'/f'transition_{pipeline.TARGET}'
    save_checkpoint(checkpoint,CheckpointPayload(expected,pipeline.TARGET,{'m':np.ones(2)},{'a':np.ones(2)},{'c':np.ones(2)}))
    old_identity=file_sha(checkpoint/'identity.json')
    monkeypatch.setattr(provenance,'verify_run',lambda _:dict(status='completed',training_transitions=pipeline.TARGET,total_environment_transitions=pipeline.TARGET+80))
    formal=SimpleNamespace(checkpoint_transitions=pipeline.SCHEDULE,fixed_evaluation_transitions=pipeline.SCHEDULE[1:])
    ppo=SimpleNamespace(requested_transitions=pipeline.TARGET,held_out_seeds=range(8),episode_horizon=400)
    monkeypatch.setattr(config_module,'load_config',lambda _:SimpleNamespace(schema='jit_phase_u_formal_v4',formal=formal,ppo=ppo))
    frozen=pipeline.freeze_descendant(spec,arm)
    assert frozen['schema']=='jit_frozen_phase_u_descendant_v1'
    assert frozen['policy']['config_sha256']==canonical_sha256(raw)
    assert file_sha(checkpoint/'identity.json')==old_identity
    assert frozen['phase_policy']['runtime_compatibility']['historical_config']==str(declarations[1])
    report['checkpoint_restored']=False;write(run/'formal_report.json',report)
    with pytest.raises(ValueError,match='restore'):pipeline.freeze_descendant(spec,arm)


def test_prepare_rejects_evaluation_runtime_differences_outside_small_contract(declarations):
    declaration=read(declarations[0]);path=declaration['sources'][1]['template_path']
    template=read(path);template['pulse_delay_steps']=99;write(path,template)
    with pytest.raises(ValueError,match='runtime'):prepare(declarations)
    assert not declarations[3].exists()


def test_preparation_locks_physical_assets_and_configs_pass_real_loader(declarations):
    from jit_dvgc.config import load_config
    spec=prepare(declarations)
    locks=read(spec['source_lock'])['input_files']
    historical=read(declarations[1]);repo=Path(spec['repo'])
    for field in ('xml_path','reference_path'):
        assert str((repo/historical['model'][field]).resolve()) in locks
    for arm in spec['arms']:
        config=load_config(Path(arm['config']))
        assert config.ppo.requested_transitions==pipeline.TARGET
        assert config.formal.checkpoint_transitions==pipeline.SCHEDULE


def test_explicit_execution_repository_replaces_old_evidence_snapshot(declarations,tmp_path):
    previous=read(declarations[2]/'spec.json')
    old=tmp_path/'old_snapshot';old.mkdir();previous['repo']=str(old)
    write(declarations[2]/'spec.json',previous)
    current=Path(__file__).parents[2]
    spec=pipeline.prepare_experiment(*declarations,training_seeds=[11,12,13],condition_seeds=[21,22,23,24],
                                     execution_repository=current)
    assert spec['repo']==str(current)
    assert spec['source_comparison_repository']==str(old)
    assert spec['execution_identity']['repository']==str(current)
    assert str(current/'JIT/cli/run_seven_policy_phase_u.py') in spec['execution_identity']['input_files']
    assert str(current/'JIT/src/jit_dvgc/formal_training.py') in spec['execution_identity']['input_files']


def test_missing_execution_entrypoint_rejected_before_output(declarations,tmp_path):
    incomplete=tmp_path/'incomplete';incomplete.mkdir()
    with pytest.raises(ValueError,match='entrypoint'):
        pipeline.prepare_experiment(*declarations,training_seeds=[11,12,13],condition_seeds=[21,22,23,24],
                                     execution_repository=incomplete)
    assert not declarations[3].exists()


def test_execution_entrypoint_drift_stops_before_any_child(declarations,tmp_path,monkeypatch):
    import shutil
    repo=Path(__file__).parents[2];snapshot=tmp_path/'execution_snapshot'
    for source in ('JIT/src/jit_dvgc','JIT/cli','assets','data'):
        shutil.copytree(repo/source,snapshot/source,ignore=shutil.ignore_patterns('__pycache__'))
    pipeline.prepare_experiment(*declarations,training_seeds=[11,12,13],condition_seeds=[21,22,23,24],
                                execution_repository=snapshot)
    entry=snapshot/'JIT/cli/run_seven_policy_phase_u.py';entry.write_text(entry.read_text()+'\n# drift\n')
    monkeypatch.setattr(pipeline,'_run_child',lambda *a,**k:pytest.fail('launched drifted entrypoint'))
    with pytest.raises(ValueError,match='drift'):
        pipeline.run_experiment(declarations[3]/'spec.json')
