"""Opt-in verified learning-gain reward on the frozen source arrival ledger."""
from collections import Counter
import numpy as np


def conversion_feedback(rows, seen, weights):
    required=('novelty','conversion','repeat','failure','pulse_failure')
    if any(k not in weights or not np.isfinite(weights[k]) or weights[k]<0 for k in required):
        raise ValueError('finite nonnegative discovery reward weights required')
    old=set(seen); windows=[]
    for row in rows:
        if 'pulse_cells' not in row:
            raise ValueError('discovery reward requires pulse_cells provenance')
        cells=row['pulse_cells']
        if not isinstance(cells,list) or any(not isinstance(c,str) for c in cells):
            raise ValueError('pulse_cells must be a list of physical cell identities')
        windows.append(set(cells) if row.get('stage_reached',True) else set())
    counts=Counter(c for cells in windows for c in cells-old)
    novelty=np.zeros(len(rows));quality=np.zeros(len(rows));mask=np.zeros(len(rows),bool)
    for i,(r,cells) in enumerate(zip(rows,windows)):
        if not cells or not r.get('stage_reached',True):
            continue
        direct=(r.get('prefix_terminal',False) and r.get('prefix_physical_failure',False)
                and r.get('pulse_applied_steps',0)>0)
        initial=r.get('initial_label');final=r.get('label');trained=r.get('learning_attempted',False)
        known=(initial==1 and final==1) or (initial==0 and trained and final in (0,1))
        if not direct and not known:
            continue  # Unknown/unattempted outcomes never receive pseudo-negatives.
        mask[i]=True
        if direct:
            quality[i]=-weights['pulse_failure'];continue
        fresh=cells-old
        novelty[i]=(weights['novelty']*sum(1/counts[c] for c in fresh)/len(cells)
                    if fresh else -weights['repeat'])
        if initial==0 and trained:
            quality[i]=weights['conversion'] if final==1 else -weights['failure']
    updated=old.union(*windows)
    return novelty+quality,mask,sorted(updated),dict(novelty=novelty.tolist(),quality=quality.tolist())
