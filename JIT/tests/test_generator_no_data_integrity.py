import json
import numpy as np
import pytest


def inputs(tmp_path):
    import jax.numpy as jp
    from .test_generator_last_valid import fixture
    from jit_dvgc.generative_bridge import artifacts, diffusion, production
    from jit_dvgc.generative_bridge.contracts import file_sha
    state, corpus, dev = fixture()
    state = {**state, 'updates': jp.asarray(7)}
    corpus['new_data'] = False
    receipt = artifacts.save_corpus(corpus, tmp_path/'corpus')
    diffusion.save_state(tmp_path/'state', state, {})
    manifest = tmp_path/'state/manifest.json'
    incumbent = dict(checkpoint_manifest=str(manifest), checkpoint_manifest_sha256=file_sha(manifest),
        state_updates=7, initial_state_updates=2, old_dev_metric='selection',
        generator_update_policy='fixed_dev_best', monitoring_best={'checkpoint':'old'})
    runner = object.__new__(production.ProductionRunner)
    runner.spec = dict(generator_update_policy='last_valid', generator_charged_updates_before=19,
                       generator_charged_updates_scope='lifetime_conservative_billed')
    class Journal:
        def stage(self, name, payload, execute): return execute()
    runner.journal = Journal()
    return state, receipt, dev, incumbent, runner


def test_no_data_stage_rejects_nonfinite_in_memory_state(tmp_path):
    from jit_dvgc.generative_bridge.artifacts import GeneratorUpdateStage
    state, receipt, dev, _, _ = inputs(tmp_path)
    state['params'] = {'w': np.array(np.nan)}
    stage = GeneratorUpdateStage(tmp_path/'update',state=state,predict=None,dev_fixture=dev,
        identity={},updates=2,max_wall_seconds=30,generator_update_policy='last_valid')
    with pytest.raises((ValueError,FloatingPointError),match='nonfinite'):
        stage(receipt)
    assert not (tmp_path/'update').exists()


def test_no_data_production_rejects_missing_corpus_manifest(tmp_path):
    _, receipt, dev, incumbent, runner = inputs(tmp_path)
    receipt['path'] = str(tmp_path/'missing.json')
    with pytest.raises((ValueError,FileNotFoundError)):
        runner.generator('incremental',receipt,dev,incumbent=incumbent)


@pytest.mark.parametrize('corruption', ['nan', 'missing_optimizer', 'wrong_updates'])
def test_no_data_production_checks_deserialized_full_state_even_with_matching_hashes(tmp_path,corruption):
    from pathlib import Path
    from flax import serialization
    from jit_dvgc.generative_bridge.contracts import file_sha
    _, receipt, dev, incumbent, runner = inputs(tmp_path)
    manifest = Path(incumbent['checkpoint_manifest']); payload=manifest.parent/'state.msgpack'
    raw=serialization.msgpack_restore(payload.read_bytes())
    if corruption=='nan': raw['ema']['w'] = np.array(np.nan)
    elif corruption=='missing_optimizer': del raw['optimizer']
    else: raw['updates'] = np.array(6)
    payload.write_bytes(serialization.msgpack_serialize(raw))
    metadata=json.loads(manifest.read_text());metadata['state_sha256']=file_sha(payload)
    manifest.write_text(json.dumps(metadata));incumbent['checkpoint_manifest_sha256']=file_sha(manifest)
    with pytest.raises((ValueError,FloatingPointError)):
        runner.generator('incremental',receipt,dev,incumbent=incumbent)


def test_no_data_production_reports_current_stage_and_preserves_charge_scope(tmp_path):
    _, receipt, dev, incumbent, runner = inputs(tmp_path)
    result=runner.generator('incremental',receipt,dev,incumbent=incumbent)
    assert result['initial_state_updates']==result['state_updates']==7
    assert result['updates']==result['charged_updates']==0
    assert result['total_charged_updates']==19
    assert result['historical_charge_accounting']=='lifetime_conservative_billed'
    assert result['old_dev_metric']=='monitor_only'
    assert result['inference_parameters']=='ema'
    assert result['monitoring_evaluation_performed'] is False
    assert result['checkpoint_manifest']==incumbent['checkpoint_manifest']
    assert result['rng_advanced'] is False
