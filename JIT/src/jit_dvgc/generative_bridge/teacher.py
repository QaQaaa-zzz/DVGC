"""Full-success-first selection; never infer physical success from a score."""
import numpy as np


def action_delta_cost(actions,last_action):
    a=np.asarray(actions);last=np.asarray(last_action)
    if a.ndim!=2 or a.shape[1]!=4 or last.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(last).all():
        raise ValueError('finite real actions and root last_action required')
    return float(np.square(np.diff(np.concatenate((last[None],a)),axis=0)).sum())


def select_teacher(candidates):
    good=[]
    for c in candidates:
        if c.get('invalid') or c.get('incomplete'):raise ValueError('teacher engineering failure, not unsolved')
        if not np.isfinite(c['action_delta_cost']) or not np.isfinite(c['task_return']):
            raise ValueError('nonfinite teacher scores')
        if c.get('kind')=='source_only' and c.get('full_success') is True:
            raise ValueError('source_succeeded_on_recheck: quarantine conflicting root, not teacher rescue')
        if c.get('full_success') is True:good.append(c)
    return min(good,key=lambda c:(c['action_delta_cost'],-c['task_return'],c['candidate_id'])) if good else None


def search_status(candidates, *, expected_count, verified=None):
    if len(candidates)!=expected_count:return 'incomplete'
    if any(c.get('invalid') for c in candidates):return 'invalid'
    if any(c.get('incomplete') for c in candidates):return 'incomplete'
    selected=select_teacher(candidates)
    if selected is None:return 'searched_no_solution'
    if verified is None:return 'incomplete'
    if verified.get('candidate_id')!=selected['candidate_id'] or verified.get('full_success') is not True:
        return 'invalid'
    return 'verified_solution'
