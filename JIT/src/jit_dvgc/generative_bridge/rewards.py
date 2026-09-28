"""User reward supplement, opt-in only. No teacher labels enter the quality term."""
from ..discovery_reward import conversion_feedback

WEIGHTS=dict(conversion=0.,adoption_bonus=2.,novelty=.02,repeat=.02,failure=.10,pulse_failure=2.)


def feedback(rows, seen):
    checked=[]
    for raw in rows:
        r=dict(raw)
        direct=(r.get('prefix_terminal',False) and r.get('prefix_physical_failure',False)
                and r.get('pulse_applied_steps',0)>0)
        # Missing final adoption is unresolved, not a rejection or a failed attempt.
        if not direct and r.get('initial_label')==0 and type(r.get('successor_adopted')) is not bool:
            r['learning_attempted']=False
        checked.append(r)
    reward,mask,ledger,parts=conversion_feedback(checked,seen,WEIGHTS)
    for i,r in enumerate(checked):
        # A historical pending root already solved by the current source has no new 0-to-1 gain.
        if (mask[i] and r.get('source_recheck_label')==1 and r.get('initial_label')==0
                and r.get('learning_attempted') and r.get('label')==1 and r.get('successor_adopted') is True):
            reward[i]-=WEIGHTS['adoption_bonus'];parts['quality'][i]-=WEIGHTS['adoption_bonus']
        # A physical failure can occur before the first quantizable pulse cell.
        if (r.get('stage_reached',True) and r.get('prefix_terminal',False)
                and r.get('prefix_physical_failure',False) and r.get('pulse_applied_steps',0)>0):
            reward[i]=-2.;mask[i]=True;parts['novelty'][i]=0.;parts['quality'][i]=-2.
    return reward,mask,ledger,parts
