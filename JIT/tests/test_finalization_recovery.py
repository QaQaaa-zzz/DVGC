import pytest
from jit_dvgc.generative_bridge.finalization_recovery import validate_exit


def test_only_exact_finalization_abort_can_be_recovered():
    status={'phase':'failed','stages':[{'phase':'failed','returncode':-6}]}
    log='Fatal Python error: PyInterpreterState_Delete: remaining subinterpreters\nPython runtime state: finalizing'
    validate_exit(status,log)
    for code in (0,1,-9):
        with pytest.raises(ValueError):validate_exit({'phase':'failed','stages':[{'returncode':code}]},log)
    with pytest.raises(ValueError):validate_exit(status,'segmentation fault')
    with pytest.raises(ValueError):validate_exit(status,log.replace('finalizing','initialized'))


def test_publication_plan_cannot_launch_training():
    from jit_dvgc.generative_bridge.closed_loop import _run_closed_loop
    with pytest.raises(ValueError,match='publication-only'):
        _run_closed_loop({'publication_only':True})


def test_failed_source_must_be_next_unpublished_round(tmp_path,monkeypatch):
    import json
    from jit_dvgc.generative_bridge import finalization_recovery as m
    root=tmp_path/'series';round=root/'round_0028';round.mkdir(parents=True)
    def write(path,obj):path.write_text(json.dumps(obj))
    write(round/'production.json',{})
    write(root/'plan.json',{'round_offset':7})
    write(root/'status.json',{'phase':'failed','completed_rounds':19})
    write(root/'completed_rounds.json',list(range(19)))
    monkeypatch.setattr(m,'_assert_not_live',lambda *args:None)
    with pytest.raises(ValueError,match='next unpublished round'):m._inspect(round)
    assert not (round/'current_source.json').exists()


def test_dry_run_removes_temporary_publication_and_keeps_requested_destination_absent(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge import finalization_recovery as m
    source=tmp_path/'source';source.mkdir();sentinel=source/'status.json';sentinel.write_text('failed')
    monkeypatch.setattr(m,'_inspect',lambda root:(source,{}, {'original_started_unix':123}, {}, None, {}, {}))
    built=[]
    def build(*args):
        output=args[-2];output.mkdir();built.append(output)
        return {'bundle':{},'validated':True}
    monkeypatch.setattr(m,'_build',build)
    output=tmp_path/'actual-publication'
    assert m.recover_finalization(source,output,original_started_unix=123)=={'validated':True,'dry_run':True}
    assert not output.exists() and not built[0].exists() and sentinel.read_text()=='failed'
    with pytest.raises(ValueError,match='deadline identity'):m.recover_finalization(source,original_started_unix=999)


def test_publication_execution_rejection_preserves_status(tmp_path):
    from jit_dvgc.generative_bridge.closed_loop import run_closed_loop
    status=tmp_path/'status.json';status.write_text('{"phase":"completed"}')
    with pytest.raises(ValueError,match='publication-only'):
        run_closed_loop({'publication_only':True,'output':str(tmp_path)})
    assert status.read_text()=='{"phase":"completed"}'
