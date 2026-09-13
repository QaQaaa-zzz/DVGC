import pytest
from jit_dvgc import lower_boundary as lower
from jit_dvgc.jump_evidence_validation import write
from jit_dvgc.evidence_integrity import canonical_sha256


def source(tmp_path):
    support={'role':'train','final_test_used':False,'initializer':{'policy':{'name':'pi_2'}}}
    support['support_sha256']=canonical_sha256(support)
    write(tmp_path/'support.json',support)
    write(tmp_path/'summary.json',{'status':'completed','support_sha256':support['support_sha256']})
    return tmp_path


def test_lower_minima_exclude_failed_states_and_keep_phase_and_provenance():
    rows=[dict(phase=phase,x_m=x,z_m=z,bank_success=valid,candidate_id=str(i),
               trajectory_id=str(i),state_sha256=str(i))
          for i,(phase,x,z,valid) in enumerate([
              ('upstream',3.01,.5,True),('upstream',3.01,.3,False),
              ('upstream',3.01,.4,True),('downstream',3.01,.2,True)])]
    result=lower.lower_slices(rows)
    assert len(result)==2
    assert next(r for r in result if r['phase']=='upstream')['observed_min_root_z_m']==.4
    assert all(r['continuous_path_claim'] is False for r in result)


def test_profile_budget_matches_real_partition_bound(tmp_path):
    profile=lower.select_profile(source(tmp_path/'source'));lower.validate_profile(profile)
    from jit_dvgc.acquisition.causal_jump import _variant_specs,VARIANT_PARTITION_MODULUS
    from jit_dvgc.unified_boundary import action_sparse_directions
    directions=action_sparse_directions(action_names=profile['action_names'],signs=profile['signs'])
    variants=_variant_specs(lookbacks_m=[.15],strengths=profile['strengths'],directions=directions)
    import math
    bound=len(profile['targets'])*5*math.ceil(len(variants)/VARIANT_PARTITION_MODULUS)*400
    assert bound==profile['acquisition_ceiling']==8000
    assert len(profile['targets'])*len(variants)==16
    assert lower.CEILING==3_284_800


def test_no_gpu_launch_when_budget_insufficient(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('GPU should not start')
    monkeypatch.setattr(lower,'run_dense',forbidden)
    result=lower.run(tmp_path,tmp_path/'out',source=source(tmp_path/'source'),budget=1)
    assert result['status']=='engineering_error'
    assert (tmp_path/'out/results_to_send.zip').exists()


def test_source_change_refuses(tmp_path):
    profile=lower.select_profile(source(tmp_path))
    write(tmp_path/'summary.json',{})
    with pytest.raises(ValueError,match='changed'):lower.validate_profile(profile)


def test_plot_does_not_need_interpolation(tmp_path):
    rows=[dict(phase='downstream',x_m=3.5,z_m=.6,bank_success=True,candidate_id='c',
               trajectory_id='t',state_sha256='s')]
    lower.render(rows,lower.lower_slices(rows),tmp_path)
    assert (tmp_path/'lower_boundary.png').exists()
