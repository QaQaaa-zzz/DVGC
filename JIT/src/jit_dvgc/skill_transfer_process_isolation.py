"""Resume immutable query blocks with fresh process memory, then train once."""
import json,os,subprocess,sys,time
from pathlib import Path
from .generative_bridge.protocol import atomic_json
from .generative_bridge.contracts import file_sha


def run_isolated(config,output):
    c=json.loads(Path(config).read_text());root=Path(output);banks=[];charged=0
    def status(**kw):atomic_json(root/'dense_status.json',dict(phase='running',updated_unix=time.time(),**kw))
    try:
        for batch in range(4):
            for position in (0,1,2,4,8,12,16):
                name=f'batch{batch:02d}_step{position:02d}';target=root/name
                recovered=root/(name+'_process_recovery001')
                if (recovered/'status.json').exists() and json.loads((recovered/'status.json').read_text()).get('state')=='completed':target=recovered
                st=target/'status.json'
                reusable=st.exists() and json.loads(st.read_text()).get('state')=='completed' and (target/'labels.json').exists()
                if not reusable:
                    if target.exists():target=root/(name+'_process_recovery001')
                    child=dict(c);child.update(student_initializer=c['student_initializer']['path'],student_old_offset=batch*7,
                        query_namespace=name+'_',strict_query_outcomes=True,allow_empty_queries=True,query_only=True)
                    child.pop('previous_online_banks',None)
                    cfg=root/(target.name+'_isolated_config.json');atomic_json(cfg,child)
                    status(stage='queries_process_isolated',block=target.name,completed_blocks=len(banks),declared_blocks=28)
                    with (root/(target.name+'_isolated.log')).open('x') as log:
                        subprocess.run([sys.executable,str(Path(c['repository'])/'JIT/cli/run_skill_transfer.py'),'consolidate',
                            '--config',str(cfg),'--output',str(target),'--recovery-checkpoint',c['recovery_inference']['path'],
                            '--updates','0','--visit-steps',str(position)],env=os.environ.copy(),cwd=c['repository'],stdout=log,stderr=subprocess.STDOUT,check=True)
                bank=target/'labels.json';a=json.loads(bank.read_text())
                if a['schema']!='frontierbridge_online_labels_v1':raise ValueError('bad completed query bank')
                q=target/'query_receipts.json'
                if q.exists():charged+=json.loads(q.read_text())['charged_interactions']
                if charged>940800:raise ValueError('declared query transition ceiling exceeded')
                banks.append(str(bank))
                atomic_json(root/'completed_query_receipts.json',dict(banks=[dict(path=p,sha256=file_sha(p)) for p in banks],charged_interactions=charged))
        c['completed_query_banks']=banks;cfg=root/'finish_config.json';atomic_json(cfg,c)
        from .skill_transfer_dense import run_dense
        run_dense(cfg,root)
    except BaseException as exc:
        atomic_json(root/'dense_status.json',dict(phase='error',error=repr(exc),completed_blocks=len(banks),updated_unix=time.time()))
        raise
