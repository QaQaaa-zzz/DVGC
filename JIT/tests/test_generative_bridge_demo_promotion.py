"""Actual two-round bank/serialized-checkpoint fixtures; no physics or adoption claim."""
import json
from copy import deepcopy
import numpy as np
import pytest
from .test_generative_bridge_student_v12 import teacher
from jit_dvgc.generative_bridge.student_demo_bank import build_student_demo_bank
from jit_dvgc.generative_bridge.student import load_optional_demo
from jit_dvgc.generative_bridge.contracts import file_sha
from jit_dvgc.generative_bridge.protocol import atomic_json


def lock(path):return {'path':str(path),'sha256':file_sha(path)}


@pytest.fixture
def lineage_fixture(tmp_path,monkeypatch):
    from jit_dvgc import unified_policy_freeze as freeze
    from jit_dvgc.checkpoint import CheckpointIdentity,CheckpointPayload,save_checkpoint
    from jit_dvgc.handoff_bank import pytree_sha256
    identity=CheckpointIdentity('config','model',(),(),())
    actor={'w':np.array([.2,.3])};norm={'mean':np.array([.1,.2])}
    checkpoint=tmp_path/'accepted_checkpoint'
    save_checkpoint(checkpoint,CheckpointPayload(identity,128000,norm,actor,{'v':np.array([.4])}))
    # Stub the formal experiment-manifest contract only; retain the real existing
    # load_retention_reference -> hash-checked load_checkpoint/parameter hashes.
    monkeypatch.setattr(freeze,'load_frozen_unified_manifest',lambda path:json.loads(path.read_text()))
    monkeypatch.setattr(freeze,'_load_policy_formal_config',lambda path:identity)
    monkeypatch.setattr(freeze,'_checkpoint_identity',lambda config:config)
    policy=dict(name='accepted_P1',actor_sha256=pytree_sha256(actor),normalizer_sha256=pytree_sha256(norm),
        xml_sha256='model',formal_config='fixture_formal',checkpoint=str(checkpoint))
    frozen=tmp_path/'frozen.json';atomic_json(frozen,{'policy':policy})
    baseline=dict(actor_sha256='old',normalizer_sha256='norm',model_sha256='model',protocol_sha256='protocol')
    candidate={**baseline,'actor_sha256':policy['actor_sha256'],'normalizer_sha256':policy['normalizer_sha256']}
    receipt=dict(schema='jit_bridge_teacher_tail_adoption_v1_2',adopted=True,baseline_identity=baseline,
        previous_tail={k:baseline[k] for k in ('actor_sha256','normalizer_sha256')},
        candidate_tail={k:candidate[k] for k in ('actor_sha256','normalizer_sha256')},frozen_policy=lock(frozen))
    receipt_path=tmp_path/'adoption.json';atomic_json(receipt_path,receipt)
    return baseline,candidate,receipt_path,receipt,checkpoint


def test_accepted_tail_retains_actual_history_and_original_tail_identities(tmp_path,lineage_fixture):
    baseline,candidate,receipt_path,receipt,_=lineage_fixture
    first=build_student_demo_bank([teacher('first')],tmp_path/'round0',source_identity=baseline,round_id=0)
    previous=lock(tmp_path/'round0/manifest.json')
    trace=teacher('second',n=21)
    trace['metadata'].update(source_actor_sha256=candidate['actor_sha256'],
        actor_sha256=candidate['actor_sha256'],normalizer_sha256=candidate['normalizer_sha256'])
    second=build_student_demo_bank([trace],tmp_path/'round1',source_identity=candidate,round_id=1,
        previous=previous,baseline_identity=baseline,teacher_tail_lineage=[lock(receipt_path)])
    obs,actions,weights=load_optional_demo(second)
    assert second['count']==39 and set(second['sample_roots'])=={'first','second'}
    assert second['entries'][0]==first['entries'][0]
    assert second['entries'][0]['source_actor_sha256']=='old'
    assert second['entries'][1]['source_actor_sha256']==candidate['actor_sha256']
    assert second['entries'][1]['normalizer_sha256']==candidate['normalizer_sha256']
    assert second['baseline_identity']==baseline and second['source_identity']==candidate
    assert weights[np.array(second['sample_roots'])=='first'].sum()==pytest.approx(.5)
    # A rejected next candidate changes neither head nor adoption receipts, and
    # does not erase already verified demonstrations.
    third=build_student_demo_bank([],tmp_path/'round2',source_identity=candidate,round_id=2,
        previous=lock(tmp_path/'round1/manifest.json'),baseline_identity=baseline,
        teacher_tail_lineage=[lock(receipt_path)])
    assert third['entries']==second['entries'] and third['count']==second['count']
    assert np.array_equal(load_optional_demo(third)[0],obs)


def test_unreceipted_changed_source_and_rejected_candidate_disallowed(tmp_path,lineage_fixture):
    baseline,candidate,path,receipt,_=lineage_fixture
    with pytest.raises(ValueError,match='verified accepted-tail'):
        build_student_demo_bank([],tmp_path/'missing',source_identity=candidate,round_id=1,
            baseline_identity=baseline,teacher_tail_lineage=[])
    rejected=tmp_path/'rejected.json';atomic_json(rejected,{**receipt,'adopted':False})
    with pytest.raises(ValueError,match='adopted'):
        build_student_demo_bank([],tmp_path/'rejectedbank',source_identity=candidate,round_id=1,
            baseline_identity=baseline,teacher_tail_lineage=[lock(rejected)])
    build_student_demo_bank([teacher()],tmp_path/'first',source_identity=baseline,round_id=0)
    with pytest.raises(ValueError,match='identity'):
        build_student_demo_bank([],tmp_path/'legacydefault',source_identity=candidate,round_id=1,
            previous=lock(tmp_path/'first/manifest.json'))


def test_tail_receipts_cannot_change_task_or_forge_checkpoint(tmp_path,lineage_fixture):
    baseline,candidate,path,receipt,checkpoint=lineage_fixture
    with pytest.raises(ValueError,match='physical/task'):
        build_student_demo_bank([],tmp_path/'physics',source_identity={**candidate,'model_sha256':'different'},
            round_id=1,baseline_identity=baseline,teacher_tail_lineage=[lock(path)])
    bad=deepcopy(receipt);bad['candidate_tail']['actor_sha256']='forged'
    forged=tmp_path/'forged.json';atomic_json(forged,bad)
    with pytest.raises(ValueError,match='checkpoint identity'):
        build_student_demo_bank([],tmp_path/'forgedbank',source_identity=candidate,round_id=1,
            baseline_identity=baseline,teacher_tail_lineage=[lock(forged)])
    with (checkpoint/'payload.pkl').open('ab') as stream:stream.write(b'corruption')
    with pytest.raises(ValueError,match='payload_sha256'):
        build_student_demo_bank([],tmp_path/'corrupt',source_identity=candidate,round_id=1,
            baseline_identity=baseline,teacher_tail_lineage=[lock(path)])


def test_no_automatic_v11_bank_migration(tmp_path,lineage_fixture):
    baseline,candidate,path,receipt,_=lineage_fixture
    old=tmp_path/'old.json';atomic_json(old,{'schema':'jit_bridge_demo_v1_1','count':0})
    with pytest.raises(ValueError,match='old-protocol'):
        build_student_demo_bank([],tmp_path/'migrate',source_identity=candidate,round_id=1,
            previous=lock(old),baseline_identity=baseline,teacher_tail_lineage=[lock(path)])


def test_rejected_baseline_continues_without_receipt_and_cannot_drop_adopted_lineage(tmp_path,lineage_fixture):
    baseline,candidate,path,receipt,_=lineage_fixture
    first=build_student_demo_bank([teacher('a')],tmp_path/'first',source_identity=baseline,
        round_id=0,baseline_identity=baseline,teacher_tail_lineage=[])
    second=build_student_demo_bank([teacher('b')],tmp_path/'rejected_round',source_identity=baseline,
        round_id=1,previous=lock(tmp_path/'first/manifest.json'),baseline_identity=baseline,teacher_tail_lineage=[])
    assert second['count']==36 and second['source_identity']==baseline
    promoted=build_student_demo_bank([],tmp_path/'promoted',source_identity=candidate,round_id=2,
        previous=lock(tmp_path/'rejected_round/manifest.json'),baseline_identity=baseline,teacher_tail_lineage=[lock(path)])
    assert promoted['entries']==second['entries']
    with pytest.raises(ValueError,match='retained unchanged'):
        build_student_demo_bank([],tmp_path/'drop',source_identity=baseline,round_id=3,
            previous=lock(tmp_path/'promoted/manifest.json'),baseline_identity=baseline,teacher_tail_lineage=[])
    altered=deepcopy(receipt);altered['previous_tail']['actor_sha256']='unrelated'
    unrelated=tmp_path/'unrelated.json';atomic_json(unrelated,altered)
    with pytest.raises(ValueError,match='not contiguous'):
        build_student_demo_bank([],tmp_path/'badchain',source_identity=candidate,round_id=1,
            baseline_identity=baseline,teacher_tail_lineage=[lock(unrelated)])
