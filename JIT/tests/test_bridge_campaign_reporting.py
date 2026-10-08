import pytest
from jit_dvgc.generative_bridge.protocol import atomic_json
from jit_dvgc.generative_bridge.contracts import digest, file_sha


def test_inherited_warmup_selection(tmp_path):
    from jit_dvgc.generative_bridge.campaign import warmup_selection
    old = tmp_path / 'old/warmup'
    (old / 'training').mkdir(parents=True)
    checkpoint = old / 'training/update_0000.pkl'
    checkpoint.write_bytes(b'weights')
    selection = {'selected_update': 0, 'scores': {'0': 8}}
    atomic_json(old / 'selection.json', selection)
    result = {'path': str(checkpoint), 'sha256': file_sha(checkpoint), 'update': 0}
    current = tmp_path / 'recovery'
    atomic_json(current / 'stages/warmup.json', {'result': result, 'output_sha256': digest(result)})
    assert warmup_selection(current) == selection
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        warmup_selection(current)


def test_selection_update_must_match_receipt(tmp_path):
    from jit_dvgc.generative_bridge.campaign import warmup_selection
    p = tmp_path / 'warmup/training/update_0000.pkl'
    p.parent.mkdir(parents=True); p.write_bytes(b'weights')
    result = {'path': str(p), 'sha256': file_sha(p), 'update': 0}
    atomic_json(tmp_path / 'stages/warmup.json', {'result': result, 'output_sha256': digest(result)})
    atomic_json(tmp_path / 'warmup/selection.json', {'selected_update': 500})
    with pytest.raises(ValueError, match='selection changed'):
        warmup_selection(tmp_path)


def test_finalizer_refuses_existing_output(tmp_path):
    from jit_dvgc.generative_bridge.campaign import finalize_saved_campaign
    with pytest.raises(FileExistsError):
        finalize_saved_campaign(tmp_path / 'missing', tmp_path)


def test_finalizer_refuses_active_training(tmp_path):
    from jit_dvgc.generative_bridge.campaign import finalize_saved_campaign
    atomic_json(tmp_path / 'status.json', {'phase': 'running'})
    with pytest.raises(ValueError, match='failed reporting'):
        finalize_saved_campaign(tmp_path, tmp_path / 'reports')
    assert not (tmp_path / 'reports').exists()
