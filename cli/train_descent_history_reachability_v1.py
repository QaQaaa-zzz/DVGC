"""CPU-only actor-history reachability probe from construction labels."""
from __future__ import annotations
import argparse,json,os,pickle,subprocess,tempfile
from pathlib import Path
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256

CONSTRUCTION=Path('runs/descent_compact_matcher_neighborhood_v1/construction_24_adaptive/construction_bank.pkl')
INDEX=Path('runs/backward_recovery_tube_fast_track_v1/proposal_state_index.json')
DEFAULT_RUN=Path('runs/descent_history_reachability_v1/construction_24_parent_cv')

def save_json(path,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as stream:json.dump(payload,stream,indent=2,sort_keys=True);stream.write('\n')
  os.replace(name,path)
 except BaseException:
  try:os.unlink(name)
  except FileNotFoundError:pass
  raise

def sigmoid(x):
 x=np.clip(x,-30,30);return 1/(1+np.exp(-x))

def fit_ridge(x,y,weight=1.0):
 mean=x.mean(0);scale=np.maximum(x.std(0),1e-3);z=(x-mean)/scale;design=np.c_[np.ones(len(z)),z]
 target=np.log(np.clip(y,.02,.98)/(1-np.clip(y,.02,.98)));penalty=np.eye(design.shape[1])*weight;penalty[0,0]=0
 coef=np.linalg.solve(design.T@design+penalty,design.T@target);return {'mean':mean,'scale':scale,'coef':coef}

def predict(model,x):return sigmoid(np.c_[np.ones(len(x)),(x-model['mean'])/model['scale']]@model['coef'])

def folds(rows,count=4):
 assignment={}
 by={}
 for i,row in enumerate(rows):by.setdefault(row['final']['label'],[]).append((row['entry_source_id'],i))
 for values in by.values():
  for rank,(_,i) in enumerate(sorted(values)):assignment[i]=rank%count
 return np.asarray([assignment[i] for i in range(len(rows))])

def metrics(y,p):
 brier=float(np.mean((p-y)**2));binary=y>=.7;order=np.argsort(-p);tp=0;ap=0.
 for rank,i in enumerate(order,1):
  if binary[i]:tp+=1;ap+=tp/rank
 ap=ap/max(int(binary.sum()),1);ece=0.
 for lo in np.linspace(0,1,6)[:-1]:
  mask=(p>=lo)&(p<(lo+.2) if lo<.8 else p<=1)
  if mask.any():ece+=float(mask.mean())*abs(float(p[mask].mean()-y[mask].mean()))
 return {'brier':brier,'ece':ece,'average_precision_safe':float(ap),'mean_prediction':float(p.mean()),'mean_target':float(y.mean())}

def panel_cv(x,y,assignment):
 pred=np.zeros(len(y))
 for fold in sorted(set(assignment)):
  train=assignment!=fold;pred[~train]=predict(fit_ridge(x[train],y[train]),x[~train])
 return pred,metrics(y,pred)

def load_proposals(index_rows):
 banks={};records=[]
 for row in index_rows:
  path=row['source_artifact']
  if path not in banks:
   try:banks[path]=SnapshotBank.load(path).records
   except ValueError:banks[path]=[item.get('snapshot_v4',item) for item in pickle.loads(Path(path).read_bytes())]
  records.append(banks[path][int(row['source_index'])])
 return records

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default=str(DEFAULT_RUN));a=p.parse_args();root=Path(a.run)
 if root.exists():raise SystemExit(f'refusing overwrite {root}')
 bank=SnapshotBank.load(CONSTRUCTION);rows=bank.records;y=np.asarray([r['final']['successes']/r['final']['branches'] for r in rows],float)
 physical=np.asarray([r['entry_feature'] for r in rows],float);history=np.asarray([r['policy_state']['actor_observation'] for r in rows],float);assignment=folds(rows)
 pp,pm=panel_cv(physical,y,assignment);hp,hm=panel_cv(history,y,assignment)
 aliases=[]
 for i in range(len(rows)):
  for j in range(i+1,len(rows)):
   if np.linalg.norm(physical[i]-physical[j])<=1e-7 and abs(y[i]-y[j])>=.1:
    aliases.append({'a':rows[i]['id'],'b':rows[j]['id'],'target_a':y[i],'target_b':y[j],
                    'actor_observation_l2':float(np.linalg.norm(history[i]-history[j]))})
 full=fit_ridge(history,y);index_rows=json.loads(INDEX.read_text())['rows'];proposal_records=load_proposals(index_rows);scores=predict(full,np.asarray([r['policy_state']['actor_observation'] for r in proposal_records],float))
 used={r['id'] for r in rows};ranked=[]
 for source,record,score_value in zip(index_rows,proposal_records,scores,strict=True):
  if source['proposal_id'] in used:continue
  ranked.append({k:source[k] for k in ('proposal_id','candidate_id','region','shell_layer','physical_state_sha256')}|{'reachability_score':float(score_value)})
 selected=[];caps={'early':6,'middle':5,'late':5};parents=set()
 for region,count in caps.items():
  for row in sorted((x for x in ranked if x['region']==region),key=lambda x:(-x['reachability_score'],x['candidate_id'],x['proposal_id'])):
   if row['candidate_id'] in parents:continue
   selected.append(row);parents.add(row['candidate_id'])
   if sum(x['region']==region for x in selected)==count:break
 root.mkdir(parents=True);np.savez(root/'model.npz',mean=full['mean'],scale=full['scale'],coef=full['coef'])
 save_json(root/'ranked_proposals.json',{'status':'ADVISORY_ONLY','selected':selected,'all_ranked':sorted(ranked,key=lambda x:-x['reachability_score'])})
 improvement=(pm['brier']-hm['brier'])/max(pm['brier'],1e-9);status='PASS' if improvement>0 and len(selected)==16 else 'FAIL'
 report={'status':status,'artifact_role':'phase_conditioned_reachability_proposal_guide','formal_tube_or_matcher':False,'cpu_only':True,
  'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'states':len(rows),'unique_parents':len({r['entry_source_id'] for r in rows}),
  'targets':{'min':float(y.min()),'median':float(np.median(y)),'max':float(y.max())},'folds':assignment.tolist(),
  'physical_16D':pm,'actor_history_140D':hm,'relative_brier_improvement':improvement,'physical_alias_pairs':aliases,
  'selected_proposals':len(selected),'selected_regions':{r:sum(x['region']==r for x in selected) for r in caps},
  'construction_bank_sha256':file_sha256(CONSTRUCTION),'proposal_index_sha256':file_sha256(INDEX),'PPO_authorization':False,
  'next':'certify_history_ranked_construction_pilot' if status=='PASS' else 'history_estimator_insufficient'}
 save_json(root/'DESCENT_HISTORY_REACHABILITY_V1_REPORT.json',report);save_json(root/'completed.json',{'status':status,'next':report['next']});print(json.dumps(report,indent=2))
if __name__=='__main__':main()
