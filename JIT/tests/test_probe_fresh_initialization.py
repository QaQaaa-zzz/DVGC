import json
import pytest
from jit_dvgc.evidence_integrity import canonical_sha256
from jit_dvgc.iterative_probe_training import make_config, load_config, restore_params


def test_fresh_probe_is_valid_but_refuses_checkpoint_restore(tmp_path, jit_root):
    support = dict(schema='jit_iterative_witnessed_support_v1', role='train', final_test_used=False,
                   inputs={}, entries=[dict(phase=p, key=p, snapshot=p, witnessed=True)
                                       for p in ('upstream', 'downstream')])
    support['support_sha256'] = canonical_sha256(support)
    sp = tmp_path/'support.json'; sp.write_text(json.dumps(support))
    base = jit_root/'configs/pi_unified_formal.json'
    init = tmp_path/'source.json'; init.write_text(json.dumps({'policy': {'formal_config': str(base)}}))
    path = tmp_path/'config.json'
    make_config(sp, init, base, path, 'fresh_fixture', 0, 3200, 123, initialization_mode='fresh')
    config = load_config(path)
    assert config.raw['initialization']['actor'] == 'fresh'
    assert config.raw['initialization']['normalizer'] == 'fresh'
    with pytest.raises(ValueError, match='fresh'):
        restore_params(path)
    raw = json.loads(path.read_text()); raw['initialization']['normalizer'] = 'restored'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='fresh'):
        load_config(path)
