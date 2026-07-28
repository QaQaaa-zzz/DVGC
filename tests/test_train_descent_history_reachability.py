import pickle
import numpy as np
from cli.train_descent_history_reachability_v1 import fit_ridge,folds,load_proposals,predict

def test_ridge_and_parent_fold_assignment_are_deterministic():
 x=np.arange(24,dtype=float).reshape(6,4);y=np.linspace(.1,.9,6);model=fit_ridge(x,y);p=predict(model,x)
 assert p.shape==(6,) and np.isfinite(p).all()
 rows=[{'entry_source_id':str(i),'final':{'label':('safe' if i%2 else 'dead')}} for i in range(8)]
 assert np.array_equal(folds(rows),folds(rows)) and set(folds(rows))=={0,1,2,3}

def test_plain_v4_snapshot_sources_are_supported(tmp_path):
 path=tmp_path/'v4.pkl';path.write_bytes(pickle.dumps([{'snapshot_v4':{'id':'x'}}]))
 assert load_proposals([{'source_artifact':str(path),'source_index':0}])==[{'id':'x'}]
