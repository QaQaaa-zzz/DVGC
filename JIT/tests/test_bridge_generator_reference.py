import json
import pytest
from jit_dvgc.generative_bridge import worker
from jit_dvgc.generative_bridge.contracts import file_sha


def test_default_generator_reference_keeps_legacy_source():
    assert worker.generator_reference({'source_frozen_policy':'tail.json'})=='tail.json'


def test_explicit_generator_reference_is_separate_and_verified(tmp_path, monkeypatch):
    baseline=tmp_path/'baseline.json'; baseline.write_text('{}')
    identities={'tail.json':{'xml_sha256':'xml','actor_sha256':'new'},str(baseline):{'xml_sha256':'xml','actor_sha256':'old'}}
    monkeypatch.setattr(worker,'source_payload',lambda path:(identities[str(path)],None))
    spec={'source_frozen_policy':'tail.json','generator_reference_frozen_policy':{'path':str(baseline),'sha256':file_sha(baseline)}}
    assert worker.generator_reference(spec)==str(baseline)
    identities[str(baseline)]['xml_sha256']='other'
    with pytest.raises(ValueError,match='physics'):worker.generator_reference(spec)
    baseline.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash'):worker.generator_reference(spec)


def test_worker_and_teacher_use_generator_reference(monkeypatch):
    import jax
    from jit_dvgc.generative_bridge.production import ProductionRunner
    calls=[]
    class Captured(Exception):pass
    monkeypatch.setattr(jax,'default_backend',lambda:'gpu')
    monkeypatch.setattr(worker,'generator_reference',lambda spec:'baseline.json')
    def capture(path,seed):
        calls.append((path,seed));raise Captured()
    monkeypatch.setattr(worker,'generator_template',capture)
    with pytest.raises(Captured):worker.run_generator({'source_frozen_policy':'new_tail.json','seed':2})
    runner=object.__new__(ProductionRunner)
    runner.panels={'new_roots':['root']};runner.spec={'source_frozen_policy':'new_tail.json','seed':2}
    with pytest.raises(Captured):runner.teacher_search({})
    assert calls==[('baseline.json',2),('baseline.json',2)]
