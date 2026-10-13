"""Explicit OOM recovery may reuse only fully bound immutable teacher roots."""
from pathlib import Path
from copy import deepcopy
import pytest

from jit_dvgc.generative_bridge.contracts import file_sha
from jit_dvgc.generative_bridge import teacher_worker
from .test_teacher_recovery_worker import fixture_runner


def prepared(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir()
    first,_,generator=fixture_runner(old,monkeypatch)
    row=first.teacher_search(generator)['r']
    path=first.root/'teachers/0000_result.json'
    proposal=first.root/'teachers/0000_proposals.npz'
    ref=dict(path=str(path),sha256=file_sha(path),proposal_path=str(proposal),
        proposal_sha256=file_sha(proposal),root_context_sha256='context',source_actor_sha256='actor',
        generator_checkpoint_manifest_sha256='G-sha',protocol_sha256='protocol')
    return first,row,generator,ref


def test_completed_root_reuse_skips_generated_search_and_both_reviews(tmp_path,monkeypatch):
    first,original,generator,ref=prepared(tmp_path,monkeypatch)
    second=tmp_path/'new';second.mkdir()
    runner,calls,_=fixture_runner(second,monkeypatch)
    runner.spec['recovery_completed_teacher_roots']={'r':ref}
    result=runner.teacher_search(generator)['r']
    assert len(calls)==1 and calls[0]['name']=='teacher_source'
    assert result['teacher_status']=='verified_solution'
    assert result['verification']==original['verification']
    assert result['proposal_path']==ref['proposal_path']
    assert file_sha(ref['path'])==ref['sha256']
    assert (runner.root/'teacher_root_reuse.json').is_file()


@pytest.mark.parametrize('change',[('sha256','bad'),('proposal_sha256','bad'),
    ('root_context_sha256','other'),('source_actor_sha256','other'),
    ('generator_checkpoint_manifest_sha256','other'),('protocol_sha256','other')])
def test_changed_reuse_binding_stops_before_physics(tmp_path,monkeypatch,change):
    _,_,generator,ref=prepared(tmp_path,monkeypatch)
    second=tmp_path/'new';second.mkdir()
    runner,calls,_=fixture_runner(second,monkeypatch)
    ref[change[0]]=change[1]
    runner.spec['recovery_completed_teacher_roots']={'r':ref}
    with pytest.raises(ValueError,match='reuse'):
        runner.teacher_search(generator)
    assert calls==[]


def test_changed_review_trace_cannot_hide_behind_final_result_hash(tmp_path,monkeypatch):
    _,original,generator,ref=prepared(tmp_path,monkeypatch)
    Path(original['verification']['reviews'][1]['selected_attempt']['trace']).write_bytes(b'corrupt')
    second=tmp_path/'new';second.mkdir()
    runner,calls,_=fixture_runner(second,monkeypatch)
    runner.spec['recovery_completed_teacher_roots']={'r':ref}
    with pytest.raises(ValueError):runner.teacher_search(generator)
    assert calls==[]


def test_partial_root_uses_exact_saved_proposal_without_regenerating_noise_actions(tmp_path,monkeypatch):
    _,original,generator,ref=prepared(tmp_path,monkeypatch)
    second=tmp_path/'new';second.mkdir()
    runner,calls,_=fixture_runner(second,monkeypatch)
    runner.spec['recovery_teacher_worker']={'partial_root':dict(ref,root_id='r',ordinal=0,
        snapshot_context_sha256='context')}
    from jit_dvgc.generative_bridge import diffusion
    monkeypatch.setattr(diffusion,'ddim_sample',lambda *args:pytest.fail('regenerated immutable partial proposal'))
    result=runner.teacher_search(generator)['r']
    assert len(calls)==4 and result['selected_candidate_id']==original['selected_candidate_id']
    assert file_sha(runner.root/'teachers/0000_proposals.npz')==ref['proposal_sha256']


def test_reused_root_cannot_ignore_changed_source_control_label(tmp_path,monkeypatch):
    _,_,generator,ref=prepared(tmp_path,monkeypatch)
    second=tmp_path/'new';second.mkdir()
    runner,calls,_=fixture_runner(second,monkeypatch,source=1)
    runner.spec['recovery_completed_teacher_roots']={'r':ref}
    with pytest.raises(ValueError,match='source'):
        runner.teacher_search(generator)
    assert len(calls)==1
