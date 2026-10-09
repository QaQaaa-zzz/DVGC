"""Persistent local desktop error/completion watcher for a finite PhysX run."""
import argparse,json,os,subprocess,time
from pathlib import Path


def terminal_event(status,service_state):
    if status.get('completed') and status.get('training_stage_complete'):
        return 'training_complete'
    if status.get('stage')=='failed' or service_state in ('failed','inactive'):
        return 'error'
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--service',required=True);p.add_argument('--state',type=Path,required=True);a=p.parse_args()
    previous=json.loads(a.state.read_text()) if a.state.exists() else {};started=time.time()
    while True:
        status_path=a.run/'status.json'
        try:status=json.loads(status_path.read_text()) if status_path.exists() else {}
        except json.JSONDecodeError:time.sleep(2);continue
        service=subprocess.run(['systemctl','--user','is-active',a.service],capture_output=True,text=True).stdout.strip()
        event=terminal_event(status,service) if time.time()-started>10 else None
        previous.update(pid=os.getpid(),heartbeat_unix=time.time(),service=service,training_transitions=status.get('training_transitions'),stage=status.get('stage'))
        if event and not previous.get(event+'_notified'):
            title='Isaac jump_ori：训练阶段结束' if event=='training_complete' else 'Isaac jump_ori：运行错误'
            body=f"{status.get('training_transitions',0):,} / {status.get('budget',0):,} 环境转移；"+('评估尚未执行。' if event=='training_complete' else f"请检查 {status_path}")
            result=subprocess.run(['notify-send','--app-name=Isaac PPO','--urgency=normal' if event=='training_complete' else '--urgency=critical',title,body],capture_output=True,text=True)
            previous[event+'_notified']=result.returncode==0;previous['notification_returncode']=result.returncode
            previous['notification_error']=result.stderr
        a.state.parent.mkdir(parents=True,exist_ok=True);tmp=a.state.with_suffix('.tmp');tmp.write_text(json.dumps(previous,indent=2)+'\n');tmp.replace(a.state)
        if event and previous.get(event+'_notified'):return
        time.sleep(15)


if __name__=='__main__':main()
