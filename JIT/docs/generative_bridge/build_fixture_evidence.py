"""Constructed CPU example, NOT robot trajectory or research performance."""
from pathlib import Path
import json
import numpy as np
from jit_dvgc.generative_bridge.feedback_data import build_corpus
from jit_dvgc.generative_bridge.outcomes import root_outcome,result_matrix
from jit_dvgc.generative_bridge.protocol import atomic_json


def main():
    root=Path(__file__).parent/'fixture_evidence';root.mkdir(exist_ok=True)
    obs=np.arange(19*76,dtype=np.float32).reshape(19,76)/1000
    metadata=dict(root_id='CPU_FIXTURE_ROOT',root_episode_id='CPU_FIXTURE_ANCESTOR',root_context_sha256='fixture_context',
        actor_sha256='fixture_student',normalizer_sha256='fixture_normalizer',model_sha256='fixture_model',
        protocol_sha256='fixture_protocol',origin_type='adopted_actor',role='train',inherited_split='generator_train',
        full_success=True,complete=True,prior_teacher_status='searched_no_solution',source_actor_sha256='fixture_old_actor',
        success_criterion='stable_forward_recovery',evidence_role='constructed_cpu_fixture_not_physics')
    t={'metadata':metadata,'arrays':dict(actor_observation_before=obs[:-1].copy(),actor_observation_after=obs[1:].copy(),
        normalized_action_executed=np.zeros((18,4),np.float32),valid_mask=np.ones(18,bool),done=np.arange(18)==17,
        success=np.arange(18)==17,failure=np.zeros(18,bool),timeout=np.zeros(18,bool),phase_before=np.zeros(18,int),
        action_origin=np.array(['actor_only']*18))}
    corpus=build_corpus([],[],[t],adoption={'adopted':True,'actor_sha256':'fixture_student','normalizer_sha256':'fixture_normalizer'},
        expected={'model_sha256':'fixture_model','protocol_sha256':'fixture_protocol'},splits={'CPU_FIXTURE_ANCESTOR':'generator_train'})
    atomic_json(root/'generator_corpus_update.json',{k:v for k,v in corpus.items() if k!='groups'})
    rows=[]
    for status in ('verified_solution','searched_no_solution','not_scheduled','incomplete','invalid'):
        for label in (0,1,None):
            row=root_outcome({'root_id':f'fixture_{status}_{label}','teacher_status':status,
                'student_label':label,'student_adopted':label==1,
                'evidence_role':'constructed_cpu_fixture_not_physics'})
            rows.append(row)
    (root/'bidirectional_root_outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    atomic_json(root/'result_matrix.json',result_matrix(rows))
    atomic_json(root/'example_boundary.json',{'physical_interactions':0,'real_teacher_success_examples':0,
        'real_teacher_unsolved_student_success_examples':0,'production_fake_samples_created':False,
        'fixture_only':True,'note':'The synthetic trace exists solely to exercise admission logic; never a production dataset.'})

if __name__=='__main__':main()
