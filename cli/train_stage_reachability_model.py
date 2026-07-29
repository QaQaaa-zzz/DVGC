"""CPU-only soft-label reachability-model pilot with parent-held-out QA."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def auprc(y,p):
 order=np.argsort(-p);y=np.asarray(y)[order];tp=np.cumsum(y);fp=np.cumsum(1-y);precision=tp/np.maximum(tp+fp,1);recall=tp/max(float(y.sum()),1.);return float(np.sum((recall-np.r_[0.,recall[:-1]])*precision))
def metrics(target,pred):
 target=np.asarray(target,float);pred=np.asarray(pred,float);binary=(target>0).astype(int);bins=np.minimum((pred*10).astype(int),9);ece=sum((bins==i).mean()*abs(pred[bins==i].mean()-target[bins==i].mean()) for i in range(10) if np.any(bins==i))
 return {'states':len(target),'soft_brier':float(np.mean((pred-target)**2)),'ece_10':float(ece),'auprc_any_success':auprc(binary,pred),'mean_target':float(target.mean()),'mean_prediction':float(pred.mean())}
def classification(target,pred,threshold=.5):
 truth=np.asarray(target)>=threshold;positive=np.asarray(pred)>=threshold
 tp=int(np.sum(truth&positive));fp=int(np.sum(~truth&positive));fn=int(np.sum(truth&~positive))
 return {'threshold':threshold,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'tp':tp,'fp':fp,'fn':fn}
def parent_key(row):
 return str(row.get('independent_trajectory_parent_id') or row.get('source_parent_id') or row.get('origin_parent') or row.get('trajectory_parent_id') or row.get('parent_candidate_id') or row.get('parent_anchor_pair') or f"reference:{row.get('reference_index')}")
def fit(group):
 X=np.asarray([x[0]['physical_feature'] for x in group],float);y=np.asarray([x[1]['p_next'] for x in group],float);center=np.median(X,axis=0);scale=np.maximum(np.median(np.abs(X-center),axis=0)*1.4826,np.asarray([.03,.01,.01,.01,.01,.01,.08,.03,.08,.08,.08,.08,.01,.02,.02,.5]));Z=(X-center)/scale;w=np.zeros(Z.shape[1]);b=0.
 for _ in range(4000):
  pred=sigmoid(Z@w+b);err=pred-y;w-=.03*((Z.T@err)/len(Z)+1e-3*w);b-=.03*err.mean()
 return center,scale,w,b
def predict(group,model):
 center,scale,w,b=model;return sigmoid((np.asarray([x[0]['physical_feature'] for x in group])-center)/scale@w+b)
def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--labels',required=True);p.add_argument('--output-model',required=True);p.add_argument('--output-report',required=True);p.add_argument('--output-proposals',required=True);p.add_argument('--seed',type=int,default=9840000);p.add_argument('--stage',choices=('takeoff','ascent','apex','descent','landing'),default='takeoff');a=p.parse_args()
 bank=SnapshotBank.load(a.bank);report=json.load(open(a.labels));by_id={r['id']:r for r in bank.records};rows=[]
 for label in report['labels']:
  if label.get('stage') not in (None,a.stage):raise SystemExit(f"label stage mismatch: {label.get('stage')} != {a.stage}")
  row=by_id[label['candidate_id']];rows.append((row,label,parent_key(row)))
 if not rows:raise SystemExit(f'no {a.stage} labels')
 parents=sorted({x[2] for x in rows});rng=np.random.default_rng(a.seed);rng.shuffle(parents);hold=set(parents[:max(1,int(np.ceil(.25*len(parents))))]);train=[x for x in rows if x[2] not in hold];test=[x for x in rows if x[2] in hold]
 model=fit(train);center,scale,w,b=model;ptr=predict(train,model);pte=predict(test,model);all_pred=predict(rows,model)
 lopo=np.zeros(len(rows))
 for parent in parents:
  tr=[x for x in rows if x[2]!=parent];te=[(i,x) for i,x in enumerate(rows) if x[2]==parent];pred=predict([x for _,x in te],fit(tr))
  for (index,_),value in zip(te,pred):lopo[index]=value
 def grouped(kind):
  indices=[i for i,x in enumerate(rows) if x[0].get('candidate_kind')==kind];target=[rows[i][1]['p_next'] for i in indices];pred=lopo[indices]
  return {**metrics(target,pred),'high_reach':classification(target,pred)} if indices else None
 kinds=sorted({x[0].get('candidate_kind') for x in rows if x[0].get('candidate_kind')})
 payload={'status':'PASS','artifact_role':'stage_reachability_model_pilot','stage':a.stage,'target':'next_stage_reach_probability','cpu_only':True,'unique_states':len(rows),'unique_parents':len(parents),'train_parents':len(set(x[2] for x in train)),'heldout_parents':len(hold),'split':'parent-held-out 75/25','seed':a.seed,'features':'16D physical feature only','forbidden_inputs_excluded':['reward','success_label_as_input','controller_id','oracle_phase','teacher_id','candidate_kind','reference_index','policy_hash'],'train':metrics([x[1]['p_next'] for x in train],ptr),'heldout':metrics([x[1]['p_next'] for x in test],pte),'leave_one_parent_out':{'overall':{**metrics([x[1]['p_next'] for x in rows],lopo),'high_reach':classification([x[1]['p_next'] for x in rows],lopo)},'by_candidate_kind':{kind:grouped(kind) for kind in kinds}},'scope_limitation':'negative means negative under the frozen controller bank, never physical unreachability','data':{'bank_sha256':file_sha256(a.bank),'labels_sha256':file_sha256(a.labels)}}
 Path(a.output_model).parent.mkdir(parents=True,exist_ok=True);np.savez(a.output_model,center=center,scale=scale,weights=w,bias=np.asarray(b));payload['model_sha256']=file_sha256(a.output_model);save_json(a.output_report,payload)
 proposals=[]
 for (row,label,parent),prediction in zip(rows,all_pred):
  proposals.append({'candidate_id':row['id'],'parent':parent,'observed_s':label['s'],'observed_n':label['n'],'observed_p_next':label['p_next'],'observed_label':label['label'],'predicted_p_next':float(prediction),'ranking_only':True})
 save_json(a.output_proposals,{'status':'PASS','artifact_role':'stage_reachability_ranked_proposals','stage':a.stage,'not_certified_tube':True,'not_safe_labels':True,'model_sha256':payload['model_sha256'],'records':sorted(proposals,key=lambda x:-x['predicted_p_next'])});print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
