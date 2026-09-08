"""TRAIN-informed bounded exploration profiles; no online outcome adaptation."""
from pathlib import Path
import math
from .jump_evidence_validation import read, verify_hash
from .evidence_integrity import canonical_sha256

DEFAULT_OUTPUT = 'JIT/runs/discovery/landing_frontier_v1'
DEFAULT_PREVIOUS = 'JIT/runs/discovery/four_proposers_5cm_v1'


def allocate(previous_report):
    verify_hash(previous_report, 'report_sha256')
    if previous_report.get('status') != 'completed' or not previous_report.get('scope','').startswith('TRAIN'):
        raise ValueError('completed TRAIN discovery report required')
    if previous_report.get('final_test_used') is not False:
        raise ValueError('final TEST must not guide allocation')
    rows = previous_report['metrics']
    if {r['proposer'] for r in rows} != {'pi_0','pi_1','pi_2','pi_3'} or len(rows) != 4:
        raise ValueError('four distinct proposers required')
    scores={}
    for row in rows:
        cost, gain=row['charged_interactions'],row['novel_vs_previous_train_union']
        if type(cost) is not int or cost <= 0 or type(gain) is not int or gain < 0:
            raise ValueError('invalid TRAIN gain/cost')
        scores[row['proposer']]=gain/cost
    selected=sorted(scores,key=lambda n:(-scores[n],n))[:2]
    profiles={}
    for name in sorted(scores):
        strengths=[.10,.20] if name in selected else [.15]
        count=8*len(strengths)
        # Same spatial window and all four signed action axes for every proposer.
        profiles[name]={'version':'landing_frontier_v1','targets':[2.9],'strengths':strengths,
            'max_trajectories':count,'max_candidates':128,'sampling_max_x_m':8.,
            'acquisition_ceiling':4000 if count==8 else 8000,
            'acquisition_seed':9842101,'label_seed':9842201,'serial_only':True}
    result={'source_report_sha256':previous_report['report_sha256'],'scores':scores,'priority_proposers':selected,
            'profiles':profiles,'scope':'TRAIN-informed allocation; unequal profiles are not a fair policy ranking',
            'total_trajectories':sum(p['max_trajectories'] for p in profiles.values()),
            'no_ppo':True,'same_physics_and_original_centerline':True}
    result['allocation_sha256']=canonical_sha256(result)
    return result


def validate_profile(profile):
    if profile is None:return
    if profile.get('version') == 'knee_boundary_v1':
        from .boundary_refinement import validate_refinement
        validate_refinement(profile)
        return
    if profile.get('version')!='landing_frontier_v1' or profile.get('targets')!=[2.9]:
        raise ValueError('unknown frontier profile')
    if profile.get('strengths') not in ([.10,.20],[.15]):
        raise ValueError('undeclared perturbation strengths')
    n=8*len(profile['strengths'])
    expected={'max_trajectories':n,'max_candidates':128,'sampling_max_x_m':8.,
              'acquisition_ceiling':4000 if n==8 else 8000,
              'acquisition_seed':9842101,'label_seed':9842201,'serial_only':True}
    if any(profile.get(k)!=v for k,v in expected.items()):
        raise ValueError('frontier profile budget/seed/scope drift')
