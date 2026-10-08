"""CPU contract tests for continuous G state and sampling provenance."""
import json
import numpy as np
import pytest


def fixture():
    import jax
    import jax.numpy as jp
    from jit_dvgc.generative_bridge.diffusion import create_train_state
    from jit_dvgc.generative_bridge.feedback_data import build_corpus
    from .test_generative_bridge_contracts import trace
    t=trace();t['metadata']['onset']=15;t['metadata']['trace_start_step']=18
    corpus=build_corpus([],[],[t],adoption={'adopted':True,'actor_sha256':'actor','normalizer_sha256':'norm'},
                        expected={},splits={'parent':'generator_train'})
    state=create_train_state({'w':jp.array(0.)},jax.random.PRNGKey(42),{'mean':jp.zeros(76),'std':jp.ones(76)})
    dev=(jp.zeros((1,76)),jp.zeros((1,16,4)),jp.zeros((1,),int),jp.zeros((1,16,4)))
    return state,corpus,dev


@pytest.mark.parametrize('policy,expected',[('last_valid',2),('fixed_dev_best',0)])
def test_old_dev_prefers_incumbent_but_continuation_is_explicit(tmp_path,policy,expected):
    import jax.numpy as jp
    from jit_dvgc.generative_bridge import diffusion as d
    from jit_dvgc.handoff_bank import pytree_sha256
    state,corpus,dev=fixture()
    selected,report=d.train_incremental(state,corpus,predict=lambda p,x,o,k:jp.ones_like(x)*p['w'],
        dev_fixture=dev,output=tmp_path/'train',identity={'test':1},updates=2,batch_size=2,
        max_wall_seconds=60,generator_update_policy=policy,charged_updates_before=7)
    assert int(selected['updates'])==expected
    assert report['monitoring_best']['checkpoint']=='incumbent'
    assert report['monitoring_best']['state_updates']==0
    assert report['total_charged_updates']==9
    assert report['state_updates']==expected
    if policy=='last_valid':
        saved=d.restore_state(tmp_path/'train/update_0002',state,{'test':1})
        assert pytree_sha256(saved)==pytree_sha256(selected)
        assert report['monitoring_best']['age_updates']==2
        assert not np.array_equal(state['rng'],selected['rng'])
        assert pytree_sha256(state['optimizer'])!=pytree_sha256(selected['optimizer'])
        assert pytree_sha256(state['normalizer'])==pytree_sha256(selected['normalizer'])
    samples=report['metrics'][0]['sample_provenance']
    assert all(r['onset']==15 and r['source_group']=='actor_new' for r in samples)
    assert all(r['window_start_step']==18+r['window_start'] for r in samples)


def test_no_data_preserves_state_rng_and_billed_count(tmp_path):
    from jit_dvgc.generative_bridge import diffusion as d
    state,corpus,dev=fixture();corpus['new_data']=False
    selected,report=d.train_incremental(state,corpus,predict=None,dev_fixture=dev,output=tmp_path/'absent',
        identity={},updates=2,max_wall_seconds=30,generator_update_policy='last_valid',charged_updates_before=9)
    assert selected is state and report['updates']==0 and not report['rng_advanced']
    assert report['total_charged_updates']==9 and report['state_updates']==0
    assert not (tmp_path/'absent').exists()


def test_sampling_metadata_does_not_change_rng_or_draws():
    from jit_dvgc.generative_bridge.feedback_data import sample_corpus
    _,corpus,_=fixture()
    one=np.random.default_rng(123);two=np.random.default_rng(123)
    obs,act,sources=sample_corpus(corpus,one,8)
    obs2,act2,sources2,rows=sample_corpus(corpus,two,8,return_metadata=True)
    np.testing.assert_array_equal(obs,obs2);np.testing.assert_array_equal(act,act2)
    assert sources==sources2 and one.bit_generator.state==two.bit_generator.state
    assert all(r['recency']=='new' and r['segment']=='actor_only' for r in rows)


def test_nonfinite_stops_and_retains_charged_evidence(tmp_path,monkeypatch):
    import jax.numpy as jp
    from jit_dvgc.generative_bridge import diffusion as d
    state,corpus,dev=fixture()
    def fail(*args):raise FloatingPointError('bad update')
    monkeypatch.setattr(d,'make_train_step',lambda *a,**kw:fail)
    with pytest.raises(FloatingPointError,match='bad update'):
        d.train_incremental(state,corpus,predict=lambda p,x,o,k:jp.zeros_like(x),dev_fixture=dev,
            output=tmp_path/'train',identity={},updates=2,max_wall_seconds=30,
            generator_update_policy='last_valid',charged_updates_before=4,batch_size=1)
    failure=json.loads((tmp_path/'train/failure.json').read_text())
    assert failure['total_charged_updates']==5 and failure['completed_updates']==0
    assert not (tmp_path/'train/generator_selection.json').exists()


def test_worker_receipt_passes_latest_complete_state_to_next_round(tmp_path,monkeypatch):
    import jax
    import jax.numpy as jp
    from jit_dvgc.generative_bridge import worker,artifacts,diffusion as d
    from jit_dvgc.generative_bridge.contracts import file_sha
    from jit_dvgc.handoff_bank import pytree_sha256
    state,corpus,dev=fixture();identity={'test':1}
    class Net:
        def apply(self,p,x,o,k):return jp.ones_like(x)*p['w']
    monkeypatch.setattr(jax,'default_backend',lambda:'gpu') # CPU math, mock only production guard.
    monkeypatch.setattr(worker,'generator_template',lambda *args:(Net(),state,identity))
    d.save_state(tmp_path/'initial',state,identity)
    manifest=tmp_path/'initial/manifest.json'
    incumbent={'checkpoint_manifest':str(manifest),'checkpoint_manifest_sha256':file_sha(manifest),
               'total_charged_updates':10,'charged_updates_scope':'lifetime'}
    receipt=artifacts.save_corpus(corpus,tmp_path/'corpus')
    devpath=tmp_path/'dev.npz'
    np.savez(devpath,**dict(zip(('observations','actions','timesteps','noise'),dev)))
    config={'source_frozen_policy':'unused','seed':1,'corpus':receipt,'dev_fixture':str(devpath),
        'dev_fixture_sha256':file_sha(devpath),'mode':'incremental','incumbent':incumbent,
        'output':str(tmp_path/'round1'),'result':str(tmp_path/'result1.json'),
        'updates':2,'max_wall_seconds':60,'generator_update_policy':'last_valid'}
    worker.run_generator(config)
    result=json.loads((tmp_path/'result1.json').read_text())
    assert result['state_updates']==2 and result['total_charged_updates']==12
    assert result['monitoring_best']['checkpoint']=='incumbent'
    assert result['checkpoint_manifest'].endswith('update_0002/manifest.json')
    selected=d.restore_state(tmp_path/'round1/attempt_0000/update_0002',state,identity)
    assert result['selected_state_sha256']==pytree_sha256(selected)
    second={**config,'incumbent':result,'output':str(tmp_path/'round2'),'result':str(tmp_path/'result2.json')}
    worker.run_generator(second)
    next_result=json.loads((tmp_path/'result2.json').read_text())
    inherited=d.restore_state(tmp_path/'round2/attempt_0000/incumbent',state,identity)
    assert pytree_sha256(inherited)==pytree_sha256(selected)
    assert next_result['state_updates']==4 and next_result['total_charged_updates']==14
    assert next_result['initial_state_updates']==2
    corpus['new_data']=False
    empty=artifacts.save_corpus(corpus,tmp_path/'no_new_corpus')
    worker.run_generator({**second,'incumbent':next_result,'corpus':empty,
        'output':str(tmp_path/'no_update'),'result':str(tmp_path/'skipped.json')})
    skipped=json.loads((tmp_path/'skipped.json').read_text())
    assert skipped['checkpoint_manifest']==next_result['checkpoint_manifest']
    assert skipped['state_updates']==4 and skipped['total_charged_updates']==14
    assert skipped['status']=='skipped_no_new_data' and not (tmp_path/'no_update').exists()


def test_checkpoint_write_error_is_not_a_completed_rollback(tmp_path,monkeypatch):
    import jax.numpy as jp
    from jit_dvgc.generative_bridge import diffusion as d
    state,corpus,dev=fixture();save=d.save_state
    def disk_error(path,*args):
        if path.name.startswith('update_'):raise OSError('disk full')
        return save(path,*args)
    monkeypatch.setattr(d,'save_state',disk_error)
    with pytest.raises(OSError,match='disk full'):
        d.train_incremental(state,corpus,predict=lambda p,x,o,k:jp.ones_like(x)*p['w'],dev_fixture=dev,
            output=tmp_path/'train',identity={},updates=1,max_wall_seconds=30,
            generator_update_policy='last_valid',charged_updates_before=4,batch_size=1)
    failure=json.loads((tmp_path/'train/failure.json').read_text())
    assert failure['total_charged_updates']==5 and failure['completed_updates']==1
    assert not (tmp_path/'train/generator_selection.json').exists()
    assert (tmp_path/'train/incumbent/manifest.json').exists()
