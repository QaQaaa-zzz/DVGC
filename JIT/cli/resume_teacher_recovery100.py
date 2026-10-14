"""Prepare an audited sibling campaign and detach its continuation coordinator."""
import argparse,json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True,help='Latest stopped campaign, never the original initial run')
    p.add_argument('--output',type=Path,help='Fresh sibling output; defaults to a timestamped sibling')
    p.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--port',type=int,default=6039)
    p.add_argument('--active-link',type=Path,help='Atomically point this run-specific symlink to the new continuation')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if not (a.previous/'status.json').is_file():
        p.error('Previous status is unavailable. Check that its data disk is mounted before retrying.')
    if a.active_link and a.active_link.exists() and not a.active_link.is_symlink():
        p.error('Active link is a regular file/directory; refusing to replace it.')
    previous=a.previous.resolve();repository=a.repository.resolve()
    output=a.output or previous.parent/(previous.name+'_resume_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    from jit_dvgc.generative_bridge.recovery_campaign import prepare_stopped_resume
    try:
        d=prepare_stopped_resume(previous,output,repository,port=a.port)
    except (ValueError,OSError) as error:
        p.error(str(error))
    root=Path(d['campaign_output']);declaration=root/'declaration.json'
    if a.prepare_only:
        print(json.dumps(dict(phase='prepared',output=str(root),declaration=str(declaration)),ensure_ascii=False));return
    env=dict(os.environ,PYTHONPATH=str(repository/'JIT/src'),JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    args=[sys.executable,'-u',str(repository/'JIT/cli/run_teacher_recovery_campaign.py'),'run','--declaration',str(declaration)]
    with (root/'coordinator.log').open('ab') as log:
        process=subprocess.Popen(args,cwd=repository,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=process.pid,argv=args,cwd=str(repository),started_unix=time.time(),declaration=str(declaration),
        previous_campaign=str(previous),output=str(root),status=str(root/'status.json'),tensorboard_url=f'http://localhost:{a.port}/')
    if a.active_link:
        link=a.active_link.absolute();link.parent.mkdir(parents=True,exist_ok=True)
        receipt['previous_active_link']=os.readlink(link) if link.is_symlink() else None
        temp=link.with_name(link.name+f'.tmp_{process.pid}_{time.time_ns()}')
        temp.symlink_to(root,target_is_directory=True);os.replace(temp,link)
        receipt['active_link']=str(link)
    (root/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
