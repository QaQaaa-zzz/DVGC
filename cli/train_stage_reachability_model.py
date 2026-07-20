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
def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--labels',required=True);p.add_argument('--output-model',required=True);p.add_argument('--output-report',required=True);p.add_argument('--output-proposals',required=True);p.add_argument('--seed',type=int,default=9840000);a=p.parse_args()
 bank=SnapshotBank.load(a.bank);report=json.load(open(a.labels));by_id={r['id']:r for r in bank.records};rows=[]
 for label in report['labels']:
  row=by_id[label['candidate_id']];parent=row.get('parent_anchor_pair') or f"reference:{row.get('reference_index')}";rows.append((row,label,parent))
 parents=sorted({x[2] for x in rows});rng=np.random.default_rng(a.seed);rng.shuffle(parents);hold=set(parents[:max(1,int(np.ceil(.25*len(parents))))]);train=[x for x in rows if x[2] not in hold];test=[x for x in rows if x[2] in hold]
 X=np.asarray([x[0]['physical_feature'] for x in train],float);y=np.asarray([x[1]['p_next'] for x in train],float);center=np.median(X,axis=0);scale=np.maximum(np.median(np.abs(X-center),axis=0)*1.4826,np.asarray([.03,.01,.01,.01,.01,.01,.08,.03,.08,.08,.08,.08,.01,.02,.02,.5]));Z=(X-center)/scale;w=np.zeros(Z.shape[1]);b=0.
 for _ in range(4000):
  pred=sigmoid(Z@w+b);err=pred-y;w-=.03*((Z.T@err)/len(Z)+1e-3*w);b-=.03*err.mean()
 def predict(group):return sigmoid((np.asarray([x[0]['physical_feature'] for x in group])-center)/scale@w+b)
 ptr=predict(train);pte=predict(test);all_pred=predict(rows);payload={'status':'PASS','artifact_role':'stage_reachability_model_pilot','stage':'takeoff','target':'next_stage_reach_probability','cpu_only':True,'unique_states':len(rows),'unique_parents':len(parents),'train_parents':len(set(x[2] for x in train)),'heldout_parents':len(hold),'split':'parent-held-out 75/25','seed':a.seed,'features':'16D physical feature only','forbidden_inputs_excluded':['reward','success_label_as_input','controller_id','oracle_phase','teacher_id'],'train':metrics(y,ptr),'heldout':metrics([x[1]['p_next'] for x in test],pte),'data':{'bank_sha256':file_sha256(a.bank),'labels_sha256':file_sha256(a.labels)}}
 Path(a.output_model).parent.mkdir(parents=True,exist_ok=True);np.savez(a.output_model,center=center,scale=scale,weights=w,bias=np.asarray(b));payload['model_sha256']=file_sha256(a.output_model);save_json(a.output_report,payload)
 proposals=[]
 for (row,label,parent),prediction in zip(rows,all_pred):
  proposals.append({'candidate_id':row['id'],'parent':parent,'observed_s':label['s'],'observed_n':label['n'],'observed_p_next':label['p_next'],'observed_label':label['label'],'predicted_p_next':float(prediction),'ranking_only':True})
 save_json(a.output_proposals,{'status':'PASS','artifact_role':'takeoff_reachability_ranked_proposals','not_certified_tube':True,'not_safe_labels':True,'model_sha256':payload['model_sha256'],'records':sorted(proposals,key=lambda x:-x['predicted_p_next'])});print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
