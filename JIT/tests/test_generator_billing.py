import json
from pathlib import Path
import pytest


def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return path


def history(tmp_path):
    old=tmp_path/'original';recovered=tmp_path/'recovered';series=tmp_path/'series'
    stage='generator_pretrain_0000'
    costs=[dict(stage=stage,charged_updates=20,phase='completed')]
    put(old/'production.json',{});put(old/'costs.json',costs)
    put(old/(stage+'_config.json'),{'output':str(old/'generator_pretrain')})
    put(old/'generator_pretrain/cost_progress.json',{'charged_updates':20})
    put(old/'generator_pretrain/generator_selection.json',{'updates':20,'initial_updates':0,'charged_updates_before':0})
    put(recovered/'production.json',{'recovery':{'previous':str(old)}})
    put(recovered/'costs.json',costs)
    put(series/'plan.json',{'parent_campaign':str(recovered)})
    r=series/'round_0001';stage='generator_incremental_0000'
    put(r/'costs.json',[dict(stage=stage,charged_updates=4)])
    put(r/(stage+'_config.json'),{'output':str(r/'generator_incremental')})
    put(r/'generator_incremental/attempt_0000/cost_progress.json',{'charged_updates':3})
    manifest=put(old/'generator_pretrain/update_20/manifest.json',{'updates':20})
    return dict(bundle_series=str(series),parent_series=str(series),
                bundle={'generator':{'checkpoint_manifest':str(manifest)}})


def test_billing_deduplicates_inherited_stages_and_keeps_interrupted_reservation(tmp_path):
    from jit_dvgc.generative_bridge.generator_costs import reconcile_generator_billing
    result=reconcile_generator_billing(history(tmp_path))
    assert result['total_charged_updates']==24
    assert result['proposed_update_charges']==23
    assert result['state_updates']==20
    assert result['completed_updates_lower_bound']==22
    assert result['completed_updates_upper_bound']==23
    assert len(result['entries'])==2 and result['locks']
    assert result['charged_updates_scope']=='lifetime_conservative_billed'


def test_billing_fails_closed_for_missing_origin_config(tmp_path):
    from jit_dvgc.generative_bridge.generator_costs import reconcile_generator_billing
    boundary=history(tmp_path)
    (tmp_path/'original/generator_pretrain_0000_config.json').unlink()
    with pytest.raises(ValueError,match='origin config'):
        reconcile_generator_billing(boundary)


def test_billing_rejects_unresolved_pretraining_origin(tmp_path):
    from jit_dvgc.generative_bridge.generator_costs import reconcile_generator_billing
    boundary=history(tmp_path)
    put(tmp_path/'original/generator_pretrain/generator_selection.json',{'updates':20,'initial_updates':5})
    with pytest.raises(ValueError,match='cannot reconcile'):
        reconcile_generator_billing(boundary)
