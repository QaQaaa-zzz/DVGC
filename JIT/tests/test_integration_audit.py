import json
import numpy as np


def test_partial_pilot_cannot_be_ready(tmp_path):
    from jit_dvgc.generative_bridge.integration_audit import audit_integration
    (tmp_path/'plan.json').write_text(json.dumps({'rounds':2,'output':str(tmp_path)}))
    (tmp_path/'status.json').write_text(json.dumps({'phase':'running','completed_rounds':1}))
    result=audit_integration(tmp_path)
    assert not result['ready_for_200'] and not result['simulation_executed']
    assert 'two completed' in result['errors'][0]


def test_completed_flag_with_old_g_rule_cannot_be_ready(tmp_path):
    from jit_dvgc.generative_bridge.integration_audit import audit_integration
    from jit_dvgc.generative_bridge.continuation_profile import FIXED
    profile={**FIXED,'pulse_steps':3,'generator_update_policy':'fixed_dev_best'}
    (tmp_path/'plan.json').write_text(json.dumps({'rounds':2,'output':str(tmp_path),'profile':profile}))
    (tmp_path/'status.json').write_text(json.dumps({'phase':'completed','completed_rounds':2}))
    result=audit_integration(tmp_path)
    assert not result['ready_for_200'] and 'generator_update_policy' in result['errors'][0]


def timing_fixture():
    onsets=np.asarray([0,5,10,15])[(np.arange(128)+1)%4];ticks=np.arange(18)[:,None]
    prefix=ticks<(onsets+3)[None,:];pulse=prefix&(ticks>=onsets[None,:])
    terminal=np.zeros((18,128),bool)
    # One pre-pulse and one during-pulse terminal; complete denominator remains128.
    prefix[3:,0]=False;terminal[2,0]=True;pulse[:,0]=False
    prefix[12:,1]=False;terminal[11,1]=True;pulse[12:,1]=False
    tape={'prefix_mask':prefix,'mask':pulse,'terminal':terminal}
    rows=[]
    for i,s in enumerate(onsets):
        end=int(np.flatnonzero(prefix[:,i])[-1]);n=int(pulse[:,i].sum())
        outcome='pre_pulse_terminal' if n==0 else 'during_pulse_terminal' if terminal[end,i] else 'valid_post_pulse'
        rows.append(dict(index=i,pulse_scheduled_start_step=int(s),pulse_applied_steps=n,
            snapshot_control_step=end+1,pulse_outcome=outcome,prefix_terminal=bool(terminal[end,i]),
            endpoint_kind='terminal_prefix' if terminal[end,i] else 'continuation_snapshot',
            episode_performance_denominator=True))
    return dict(num_envs=128,pulse_steps=3,pulse_start_schedule=[0,5,10,15],pulse_batch_mode='mixed',round_index=1,horizon=400),tape,rows


def test_timing_audit_keeps_early_terminal_denominator_and_lane_endpoints():
    from jit_dvgc.generative_bridge.integration_audit import audit_timing_rows
    spec,tape,rows=timing_fixture();report=audit_timing_rows(spec,tape,rows)
    assert report['episodes']==128
    assert report['outcomes']=={'pre_pulse_terminal':1,'during_pulse_terminal':1,'valid_post_pulse':126}
    assert report['onset_counts']=={'0':32,'5':32,'10':32,'15':32}
    rows[3]['snapshot_control_step']=18
    import pytest
    with pytest.raises(ValueError,match='endpoint'):
        audit_timing_rows(spec,tape,rows)


def test_audit_g_latest_full_state_and_rejects_hidden_old_dev_rule(tmp_path):
    import jax.numpy as jp
    import pytest
    from .test_generator_last_valid import fixture
    from jit_dvgc.generative_bridge import artifacts,diffusion
    from jit_dvgc.generative_bridge.contracts import file_sha
    from jit_dvgc.generative_bridge.integration_audit import audit_generator,Evidence
    state,corpus,dev=fixture();identity={'fixture':True}
    diffusion.save_state(tmp_path/'initial',state,identity)
    manifest=tmp_path/'initial/manifest.json'
    old={'checkpoint_manifest':str(manifest),'checkpoint_manifest_sha256':file_sha(manifest)}
    receipt=artifacts.save_corpus(corpus,tmp_path/'corpus')
    root=tmp_path/'round';root.mkdir()
    result=artifacts.GeneratorUpdateStage(root/'generator_incremental',state=state,
        predict=lambda p,x,o,k:jp.ones_like(x)*p['w'],dev_fixture=dev,identity=identity,
        updates=2,max_wall_seconds=60,batch_size=2,generator_update_policy='last_valid',charged_updates_before=10)(receipt)
    devpath=tmp_path/'dev.npz';np.savez(devpath,observations=np.zeros((1,76)))
    previous={'generator':old,'generator_lifetime_charged_updates':10,
        'actor':{'frozen_policy':'fixed-current-tail'},'dev_fixture':str(devpath),'dev_fixture_sha256':file_sha(devpath)}
    config={'incumbent':old,'generator_update_policy':'last_valid','source_frozen_policy':'fixed-current-tail',
            'dev_fixture':str(devpath),'dev_fixture_sha256':file_sha(devpath)}
    (root/'generator_incremental_0000_config.json').write_text(json.dumps(config))
    current={'generator':result,'corpus':receipt}
    report=audit_generator(root,previous,current,Evidence())
    assert report['state_updates']==2 and report['monitoring_best']['checkpoint']=='incumbent'
    # A true history-only corpus preserves the published G and its billed count.
    from jit_dvgc.generative_bridge.feedback_data import build_corpus
    history=build_corpus(corpus['groups']['actor_new'],[],[],adoption={},expected={},splits={'parent':'generator_train'})
    history_receipt=artifacts.save_corpus(history,tmp_path/'history')
    skipped={**result,'status':'skipped_no_new_data','updates':0,'charged_updates':0}
    no_data=audit_generator(root,{**previous,'generator':result,'generator_lifetime_charged_updates':12},
        {'generator':skipped,'corpus':history_receipt},Evidence())
    assert no_data['updates']==0 and no_data['state_updates']==2
    skipped['total_charged_updates']=13
    with pytest.raises(ValueError,match='fabricated'):
        audit_generator(root,{**previous,'generator':result,'generator_lifetime_charged_updates':12},
            {'generator':skipped,'corpus':history_receipt},Evidence())
    selected=root/'generator_incremental/attempt_0000/generator_selection.json'
    raw=json.loads(selected.read_text());raw['old_dev_metric']='selection';selected.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='hidden old-dev'):
        audit_generator(root,previous,current,Evidence())


def test_explorer_audit_normalizes_existing_msgpack_list_convention_only():
    from jit_dvgc.generative_bridge.integration_audit import explorer_state_equal
    widths=[.2,.05,.1,.4,.2,.6,2.,5.,5.,10.,20.,20.]
    a={'neighborhood':{'medium_halfwidths':widths,'far_scale':2.},'params':{'weight':np.array([1.])}}
    b={**a,'neighborhood':{'medium_halfwidths':{str(i):v for i,v in enumerate(widths)},'far_scale':2.}}
    assert explorer_state_equal(a,b)
    assert not explorer_state_equal(a,{**b,'params':{'weight':np.array([2.])}})
