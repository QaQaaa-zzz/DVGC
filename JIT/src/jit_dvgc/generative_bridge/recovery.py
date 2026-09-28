"""Explicit receipt migration for the pre-PPO source-recheck guard failure."""
from copy import deepcopy
from pathlib import Path
import shutil
import time
from .contracts import digest,file_sha
from .protocol import StageJournal,atomic_json


def read(path):
    import json
    return json.loads(Path(path).read_text())


class RecoveryJournal(StageJournal):
    def __init__(self,path,contract):
        super().__init__(path,contract)
        self.recovery=contract.get('recovery',{})

    def stage(self,name,inputs,callback):
        inherited=self.recovery.get('stages',{}).get(name)
        if inherited and not (self.path/(name+'.json')).exists():
            if file_sha(inherited['path'])!=inherited['sha256']:
                raise ValueError('inherited receipt changed: '+name)
            record=read(inherited['path'])
            expected=digest({'round':self.recovery['previous_contract_identity'],'inputs':inputs})
            if record['input_sha256']!=expected:raise ValueError('inherited stage inputs changed: '+name)
            if record['output_sha256']!=digest(record['result']):raise ValueError('inherited output changed: '+name)
            return super().stage(name,inputs,lambda:record['result'])
        return super().stage(name,inputs,callback)


def prepare_recovery(previous,output,repository):
    from .production import implementation_identity,implementation_files,read_teacher_traces
    from .artifacts import validate_generator_receipt,load_corpus
    previous=Path(previous).resolve();output=Path(output).resolve()
    # The original CLI parsed its JSON declaration with YAML (1e-05 became a string).
    # Preserve exactly the executed contract, rather than silently changing runtime types.
    import yaml
    old=yaml.safe_load((previous/'production.json').read_text());status=read(previous/'status.json')
    if status.get('phase')!='failed' or 'source_succeeded_or_unknown_on_recheck' not in status.get('error',''):
        raise ValueError('this migration only handles the diagnosed source recheck guard failure')
    if old.get('recovery') or (previous/'student').exists():
        raise ValueError('migration requires original pilot with no student execution')
    commit=implementation_identity(repository)
    started=read(previous/'started.json');costs=read(previous/'costs.json')
    if time.time()-started['started_unix']>=old['budgets']['max_wall_seconds']:
        raise TimeoutError('original wall budget exhausted')
    if any(c.get('phase')!='completed' for c in costs):raise ValueError('unreconciled child cost')
    contract=read(previous/'stages/round_contract.json')
    if contract['contract']!=old or contract['sha256']!=digest(old):raise ValueError('old contract changed')
    spec=deepcopy(old);locks=spec['locks']
    for path,sha in locks.items():
        if file_sha(path)!=sha:raise ValueError('source lock changed: '+path)
    visited=set()
    def pin(value):
        if isinstance(value,dict):
            for v in value.values():pin(v)
        elif isinstance(value,list):
            for v in value:pin(v)
        elif isinstance(value,str) and value.startswith(str(previous)+'/'):
            p=Path(value)
            if not p.is_file() or value in visited:return
            visited.add(value);locks[value]=file_sha(p)
            if p.suffix=='.json':pin(read(p))
    stages={}
    allowed={'semantic_smoke','bootstrap','generator_pretrain','nominal'}
    for path in sorted((previous/'stages').glob('*.json')):
        name=path.stem
        if name not in allowed and not (name.startswith('eval_') and not name.endswith('.running')):continue
        record=read(path)
        if record['output_sha256']!=digest(record['result']):raise ValueError('old stage output changed')
        stages[name]={'path':str(path),'sha256':file_sha(path)};pin(str(path))
    if not allowed<=stages.keys() or 'eval_teacher_source' not in stages:
        raise ValueError('required completed stages missing')
    rechecks=read(read(stages['eval_teacher_source']['path'])['result']['path'])
    if any(r['label'] not in (0,1) for r in rechecks):raise ValueError('unknown source recheck needs separate diagnosis')
    bootstrap=read(stages['bootstrap']['path'])['result']
    load_corpus(bootstrap['corpus'])
    if file_sha(bootstrap['dev_fixture'])!=bootstrap['dev_fixture_sha256']:raise ValueError('dev fixture changed')
    generator=read(stages['generator_pretrain']['path'])['result']
    validate_generator_receipt(generator)
    payload=Path(generator['checkpoint_manifest']).parent/'state.msgpack';pin(str(payload))
    teachers={}
    for path in sorted((previous/'teachers').glob('*_result.json')):
        row=read(path)
        if (row['source_actor_sha256']!=old['source']['actor_sha256'] or row['generator']!=generator
            or row['teacher_status'] not in ('searched_no_solution','verified_solution')):
            raise ValueError('incompatible completed teacher receipt')
        if not any(r['root_id']==row['root_id'] and r['label']==0 for r in rechecks):
            raise ValueError('teacher receipt lacks current source negative')
        read_teacher_traces({row['root_id']:row})
        teachers[row['root_id']]={'path':str(path),'sha256':file_sha(path)};pin(str(path))
        proposal=path.with_name(path.name.replace('_result.json','_proposals.npz'))
        if file_sha(proposal)!=row['proposal_sha256']:raise ValueError('teacher proposals changed')
        pin(str(proposal))
    for name in ('production.json','status.json','costs.json','started.json','stages/round_contract.json'):
        pin(str(previous/name))
    spec.update(output=str(output),repository=str(Path(repository).resolve()),implementation_commit=commit,
                implementation_files=implementation_files(repository))
    spec['recovery']={'previous':str(previous),'previous_contract_identity':contract['sha256'],
        'stages':stages,'teachers':teachers,'reason':'source_recheck_success_is_legal',
        'budget_policy':'inherit all charged costs and original wall deadline; no budget reset'}
    output.mkdir(parents=True,exist_ok=False)
    for name in ('panels.json','training_support.json','retention_support.json'):
        shutil.copyfile(previous/name,output/name);locks[str(output/name)]=file_sha(output/name)
    atomic_json(output/'started.json',started);atomic_json(output/'costs.json',costs)
    atomic_json(output/'production.json',spec)
    atomic_json(output/'status.json',{'phase':'prepared','inherited_physics':sum(c['charged_interactions'] for c in costs),
        'inherited_supervised_updates':sum(c.get('charged_updates',0) for c in costs)})
    return spec
