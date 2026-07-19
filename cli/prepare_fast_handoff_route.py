"""Select diverse H1 contacts and exact H4 candidates from completed shards."""
from __future__ import annotations
import argparse, copy, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from cli.build_descent_entries import snapshot_identity
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shard-root',required=True);p.add_argument('--entry-bank',required=True);p.add_argument('--output-bank',required=True);p.add_argument('--output-report',required=True);p.add_argument('--h4-indices',required=True);p.add_argument('--target',type=int,default=24);p.add_argument('--parent-cap',type=int,default=2);p.add_argument('--dedup-distance',type=float,default=.15);a=p.parse_args()
    outputs=[Path(a.output_bank),Path(a.output_report),Path(a.h4_indices)]
    if any(x.exists() for x in outputs):raise SystemExit('Fast handoff preparation output exists')
    root=Path(a.shard_root);reports=sorted(root.glob('shard_*.json'));banks=sorted(root.glob('shard_*.pkl'))
    if not reports or len(reports)!=len(banks):raise SystemExit('Completed report/proposal shard mismatch')
    rows=[r for f in reports for r in json.loads(f.read_text())['rows']]
    h1={r['formal_branch_seed']:r for r in rows if r['handoff_class']=='H1'}
    h4=sorted({int(r['candidate_index']) for r in rows if r['handoff_class']=='H4'})
    proposals=[]
    for path in banks:
        for row in SnapshotBank.load(path).records:
            evidence=h1.get(int(row['entry_construction_seed']))
            if evidence:proposals.append((copy.deepcopy(row),evidence))
    entry=SnapshotBank.load(a.entry_bank);scale=np.asarray(entry.metadata['entry_matcher']['scale'],np.float64)
    exact=set();dedup=[];rejected=Counter()
    for row,evidence in sorted(proposals,key=lambda item:(item[1]['minimum_distance'],item[1]['formal_branch_seed'])):
        identity=snapshot_identity(row)
        if identity in exact:rejected['exact_duplicate']+=1;continue
        feature=np.asarray(row['entry_feature'],np.float64)
        if any(np.linalg.norm((feature-np.asarray(old[0]['entry_feature'],np.float64))/scale)<a.dedup_distance for old in dedup):rejected['normalized_duplicate']+=1;continue
        exact.add(identity);row['state_byte_hash']=identity;dedup.append((row,evidence))
    selected=[];selected_seeds=set();parent_counts=Counter();layer_counts=Counter();dynamics_counts=Counter()
    layers=('early','middle','late')
    while len(selected)<min(a.target,len(dedup)):
        progress=False
        for layer in layers:
            options=[item for item in dedup if item[1]['formal_branch_seed'] not in selected_seeds and item[1].get('layer')==layer and parent_counts[item[1].get('parent_id') or item[1]['candidate_id']]<a.parent_cap]
            if not options:continue
            options.sort(key=lambda item:(layer_counts[layer],dynamics_counts[item[1]['dynamics_variant']],parent_counts[item[1].get('parent_id') or item[1]['candidate_id']],item[1]['minimum_distance']))
            item=options[0];selected.append(item);ev=item[1];selected_seeds.add(ev['formal_branch_seed']);parent_counts[ev.get('parent_id') or ev['candidate_id']]+=1;layer_counts[layer]+=1;dynamics_counts[ev['dynamics_variant']]+=1;progress=True
            if len(selected)>=min(a.target,len(dedup)):break
        if not progress:break
    records=[]
    for row,ev in selected:
        row.update({'artifact_role':'pending_entry_proposals','active_for_matching':False,'safe_claim_allowed':False,'selection_layer':ev.get('layer'),'selection_dynamics_variant':ev['dynamics_variant'],'selection_parent':ev.get('parent_id') or ev['candidate_id'],'source_minimum_c_l_distance':ev['minimum_distance']})
        records.append(row)
    metadata={'artifact_role':'pending_entry_proposals','active_for_matching':False,'safe_claim_allowed':False,'selection_protocol':'h1_global_byte_feature_parent_layer_v1','entry_bank_sha256':file_sha256(a.entry_bank),'entry_matcher_radius':entry.metadata['entry_matcher']['radius'],'entry_matcher_unchanged':True,'xml_sha256':entry.metadata['xml_sha256'],'action_mapping_version':entry.metadata['action_mapping_version'],'actor_history_steps':entry.metadata.get('actor_history_steps'),'dedup_distance':a.dedup_distance,'parent_cap':a.parent_cap,'source_shards':[str(x.resolve()) for x in reports]}
    SnapshotBank(records,metadata).save(a.output_bank)
    Path(a.h4_indices).parent.mkdir(parents=True,exist_ok=True);Path(a.h4_indices).write_text(json.dumps(h4,indent=2)+'\n')
    report={'status':'PASS','artifact_role':'fast_handoff_selection','completed_events':len(rows),'h1_branches':len(h1),'h1_states':len({r['candidate_id'] for r in h1.values()}),'h1_parents':len({r.get('parent_id') or r['candidate_id'] for r in h1.values()}),'proposals_before_dedup':len(proposals),'proposals_after_dedup':len(dedup),'selected':len(records),'selected_parents':len(parent_counts),'maximum_children_per_parent':max(parent_counts.values(),default=0),'selected_layers':dict(layer_counts),'selected_dynamics':dict(dynamics_counts),'rejected':dict(rejected),'h4_branches':sum(r['handoff_class']=='H4' for r in rows),'h4_candidate_indices':h4,'source_entry_bank_sha256':file_sha256(a.entry_bank),'output_bank_sha256':file_sha256(a.output_bank)}
    save_json(a.output_report,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
