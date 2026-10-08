"""JSON receipt adapters for corpus and bounded offline generator stages."""
import json
from pathlib import Path
import time
import numpy as np
from .contracts import file_sha,digest
from .protocol import atomic_json
from .data import validate_trace,trajectory_identity


def save_corpus(corpus,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    groups={k:[] for k in corpus['groups']}
    for group,traces in corpus['groups'].items():
        for i,trace in enumerate(traces):
            target=path/f'{group}_{i:06d}.npz'
            np.savez_compressed(target,**trace['arrays'])
            groups[group].append({'metadata':trace['metadata'],'adoption':trace.get('adoption'),
                'path':str(target.resolve()),'sha256':file_sha(target),
                'trajectory_sha256':trajectory_identity(trace)})
    manifest={**{k:v for k,v in corpus.items() if k!='groups'},'groups':groups,
              'schema':'jit_generator_corpus_update_v1_1'}
    target=path/'manifest.json';atomic_json(target,manifest)
    return {'path':str(target.resolve()),'sha256':file_sha(target),'new_data':corpus['new_data']}


def load_corpus(receipt):
    if file_sha(receipt['path'])!=receipt['sha256']:raise ValueError('corpus receipt hash changed')
    raw=json.loads(Path(receipt['path']).read_text());groups={}
    if raw['new_data']!=receipt['new_data']:raise ValueError('corpus new-data status drift')
    admitted={r['trajectory_sha256'] for r in raw['admission'] if r['eligible']}
    for group,records in raw['groups'].items():
        groups[group]=[]
        for record in records:
            if file_sha(record['path'])!=record['sha256']:raise ValueError('corpus trace changed')
            with np.load(record['path'],allow_pickle=False) as a:
                trace={'metadata':record['metadata'],'arrays':{k:a[k] for k in a.files}}
            if record.get('adoption') is not None:trace['adoption']=record['adoption']
            validate_trace(trace)
            identity=trajectory_identity(trace)
            if identity!=record['trajectory_sha256'] or identity not in admitted:
                raise ValueError('trace lacks matching committed admission')
            groups[group].append(trace)
    return {**raw,'groups':groups}


def validate_generator_receipt(receipt):
    manifest=Path(receipt['checkpoint_manifest'])
    if file_sha(manifest)!=receipt['checkpoint_manifest_sha256']:
        raise ValueError('selected G manifest changed')
    metadata=json.loads(manifest.read_text())
    if file_sha(manifest.parent/'state.msgpack')!=metadata['state_sha256']:
        raise ValueError('selected G payload changed')
    if 'state_updates' in receipt and receipt['state_updates']!=metadata.get('updates'):
        raise ValueError('selected G state update count changed')
    if ('inference_parameters' in receipt or 'inference_parameters' in metadata) and metadata.get('inference_parameters')!='ema':
        raise ValueError('generator inference must use checkpoint EMA')
    if receipt.get('charged_updates_scope')=='lifetime' and receipt['total_charged_updates']<metadata['updates']:
        raise ValueError('generator charged count cannot be younger than its state')


def validate_generator_full_state(receipt):
    """Validate a skipped incumbent without building the network or using a GPU."""
    from flax import serialization
    from .diffusion import _finite
    validate_generator_receipt(receipt)
    manifest=Path(receipt['checkpoint_manifest'])
    metadata=json.loads(manifest.read_text())
    state=serialization.msgpack_restore((manifest.parent/'state.msgpack').read_bytes())
    required={'params','ema','optimizer','rng','normalizer','updates'}
    if not isinstance(state,dict) or not required.issubset(state):
        raise ValueError('incomplete generator training state')
    if not _finite(state):raise FloatingPointError('nonfinite incumbent generator state')
    count=np.asarray(state['updates'])
    norm=state['normalizer']
    if (count.shape!=() or count.dtype.kind not in 'iu' or int(count)!=metadata['updates']
        or int(count)<0 or not isinstance(norm,dict)
        or np.shape(norm.get('mean'))!=(76,) or np.shape(norm.get('std'))!=(76,)
        or np.any(np.asarray(norm['std'])<=0)):
        raise ValueError('invalid complete generator state count/normalizer')
    return state



class GeneratorUpdateStage:
    """Explicit calls create separate attempts under one cumulative compute cap.

    A failed attempt is retained and charged. There is no automatic retry.
    Retrying starts from the declared incumbent and spends only the remaining
    budget; discarded optimization work stays in the ledger, never counted free.
    """
    def __init__(self,path,*,state,predict,dev_fixture,identity,updates,max_wall_seconds,batch_size=256,
                 generator_update_policy='fixed_dev_best',charged_updates_before=None):
        self.path=Path(path);self.state=state;self.predict=predict;self.dev_fixture=dev_fixture
        self.identity=identity;self.updates=updates;self.max_wall=max_wall_seconds;self.batch_size=batch_size
        if generator_update_policy not in ('fixed_dev_best','last_valid'):
            raise ValueError('unknown generator update policy')
        self.policy=generator_update_policy
        self.charged_before=int(state['updates']) if charged_updates_before is None else charged_updates_before
        if type(self.charged_before) is not int or self.charged_before<int(state['updates']):
            raise ValueError('cumulative charged updates must cover state updates')

    def __call__(self,receipt):
        from .diffusion import train_incremental, _finite
        from ..handoff_bank import pytree_sha256
        corpus=load_corpus(receipt)
        if not _finite(self.state):raise FloatingPointError('nonfinite incumbent generator state')
        incumbent_sha=pytree_sha256(self.state)
        if not corpus['new_data']:
            return {'status':'skipped_no_new_data','selected_state_sha256':incumbent_sha,'updates':0,
                'charged_updates':0,'total_charged_updates':self.charged_before,'charged_updates_scope':'lifetime',
                'state_updates':int(self.state['updates']),'initial_state_updates':int(self.state['updates']),
                'generator_update_policy':self.policy,'rng_advanced':False,'inference_parameters':'ema',
                'old_dev_metric':'monitor_only' if self.policy=='last_valid' else 'selection',
                'monitoring_evaluation_performed':False}
        if type(self.updates) is not int or not 0<self.updates<=2000 or not 0<self.max_wall<float('inf'):
            raise ValueError('finite cumulative incremental budgets required')
        self.path.mkdir(parents=True,exist_ok=True)
        contract={'corpus_sha256':receipt['sha256'],'identity':self.identity,'incumbent':incumbent_sha,
            'dev_fixture':pytree_sha256(self.dev_fixture),'max_updates':self.updates,
            'max_wall_seconds':self.max_wall,'batch_size':self.batch_size,
            'generator_update_policy':self.policy,'charged_updates_before':self.charged_before}
        cp=self.path/'contract.json'
        if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('G retry contract drift')
        if not cp.exists():atomic_json(cp,contract)
        completed=self.path/'completed.json'
        if completed.exists():
            result=json.loads(completed.read_text())
            validate_generator_receipt(result)
            return result
        attempts=sorted(self.path.glob('attempt_*'))
        charged=0;wall=0.
        for previous in attempts:
            if not (previous/'cost_receipt.json').exists():
                raise RuntimeError('unreconciled generator attempt; no automatic replay')
            cost=json.loads((previous/'cost_receipt.json').read_text())
            charged+=cost['charged_updates'];wall+=cost['wall_seconds']
        remaining=self.updates-charged;seconds=self.max_wall-wall
        if remaining<=0 or seconds<=0:raise RuntimeError('cumulative generator budget exhausted')
        attempt=self.path/f'attempt_{len(attempts):04d}';start=time.monotonic()
        try:
            selected,report=train_incremental(self.state,corpus,predict=self.predict,
                dev_fixture=self.dev_fixture,output=attempt,identity=self.identity,
                updates=remaining,batch_size=self.batch_size,max_wall_seconds=seconds,
                generator_update_policy=self.policy,charged_updates_before=self.charged_before+charged)
        except BaseException as error:
            attempt.mkdir(exist_ok=True)
            progress=attempt/'cost_progress.json'
            count=json.loads(progress.read_text())['charged_updates'] if progress.exists() else 0
            atomic_json(attempt/'cost_receipt.json',{'status':'failed','error':repr(error),
                'charged_updates':count,'wall_seconds':time.monotonic()-start})
            raise
        cost={'status':'completed','charged_updates':report['updates'],'wall_seconds':time.monotonic()-start}
        atomic_json(attempt/'cost_receipt.json',cost)
        manifest=attempt/report['selected']/'manifest.json'
        result={'status':'completed','attempt':len(attempts),'checkpoint_manifest':str(manifest.resolve()),
            'checkpoint_manifest_sha256':file_sha(manifest),'selected_state_sha256':pytree_sha256(selected),
            'updates':report['updates'],'charged_updates':charged+report['updates'],
            'total_charged_updates':self.charged_before+charged+report['updates'],
            'charged_updates_before':self.charged_before,'charged_updates_scope':'lifetime',
            'initial_state_updates':int(self.state['updates']),'state_updates':int(selected['updates']),
            'generator_update_policy':self.policy,'inference_parameters':'ema',
            'monitoring_best':{**report['monitoring_best'],
                'checkpoint_manifest':str((attempt/report['monitoring_best']['checkpoint']/'manifest.json').resolve()),
                'checkpoint_manifest_sha256':file_sha(attempt/report['monitoring_best']['checkpoint']/'manifest.json')},
            'old_dev_metric':report['old_dev_metric'],
            'total_wall_seconds':wall+cost['wall_seconds'],'corpus_sha256':receipt['sha256']}
        atomic_json(completed,result)
        return result
