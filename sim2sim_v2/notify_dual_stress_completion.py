"""Wait for process-exit event; finalize evidence and notify desktop once."""
import argparse,os,select,subprocess,json,hashlib,fcntl,re,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dual_stress_protocol import write_json
parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);a=parser.parse_args()
root=Path(__file__).resolve().parent;out=root/'runs/dual_stress1000_20261013'
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def notify(title,text,kind):
 env=os.environ.copy();commands=[]
 for cmd in [['notify-send','-u','normal' if kind=='complete' else 'critical',title,text],['zenity','--info' if kind=='complete' else '--error','--title='+title,'--text='+text,'--width=680','--no-wrap']]:
  try:
   proc=subprocess.Popen(cmd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);commands.append(dict(command=cmd[0],pid=proc.pid))
  except Exception as e:commands.append(dict(command=cmd[0],error=repr(e)))
 write_json(out/(kind+'_notification.json'),dict(requested_at=stamp(),kind=kind,title=title,text=text,commands=commands))
try:
 try:
  cmd=Path(f'/proc/{a.pid}/cmdline').read_bytes().decode().replace('\0',' ')
  if 'dual_stress_run.py --engine physx' not in cmd:raise ValueError('PID is not owned PhysX evaluation')
  fd=os.pidfd_open(a.pid)
 except FileNotFoundError:fd=None
 write_json(out/'completion_watch.json',dict(state='waiting_for_process_exit_event',target_pid=a.pid,watcher_pid=os.getpid(),mechanism='Linux pidfd select event; no recurring status-file reads',started_at=stamp()))
 if fd is not None:
  select.select([fd],[],[]);os.close(fd)
 # Read final status only once after the process-exit event.
 states={e:json.loads((out/e/'status.json').read_text()) for e in ['mjx','physx']}
 if any(s['state']!='complete' or s['completed_cases']!=1000 for s in states.values()):raise RuntimeError('Incomplete engine results: '+json.dumps(states))
 declaration=json.loads((out/'declaration.json').read_text())
 for path,expected in declaration['identities'].items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen identity drift: '+path)
 subprocess.run(['/home/qy/ISAAC——SIM/.venv/bin/python',str(root/'report_dual_stress.py')],check=True,cwd=root)
 summary=json.loads((out/'summary.json').read_text());m=summary['totals']['mjx'];p=summary['totals']['physx']
 # Preserve both projects under the shared ledger lock.
 hub=Path('/home/qy/STTW_CONTROL/research-hub')
 with (hub/'.write.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);ledger=hub/'PROJECT_STATE.md';text=ledger.read_text();tag='DUAL-STRESS1000-20261013 completion event'
  if tag not in text:
   revision=int(re.search(r'state_revision: (\d+)',text).group(1))+1;text=re.sub(r'state_revision: \d+',f'state_revision: {revision}',text,count=1)
   text+=f"""
\n### {tag} — {stamp()}
repo_artifact/reproduced：同Actor01966080/冻结round0099探索器/现成hard1000初态与noise tape，四起扰0/5/10/15各250、.25x3tick，两端各1000计分回合均完成。MJX稳定平台落地1s {m['successes']}/1000 ({m['rate']:.1%})；PhysX {p['successes']}/1000 ({p['rate']:.1%})。落地且至8s未失败终止分别{m['landing_and_timeout_count']}/1000、{p['landing_and_timeout_count']}/1000；后续失败单列，timeout不单独算成功。无最低前进速度，不混用旧.5s/forward成功标准。
实际charged：MJX {states['mjx']['charged_controls']}控制/{states['mjx']['physics_steps']}物理；PhysX {states['physx']['charged_controls']}控制/{states['physx']['physics_steps']}物理（含500校准控制）；工程另296控制/1184物理。全部冻结身份核验、1000初态逐值、有限性、3步/.25脉冲、终止mask核对与CSV/原始batch/图保存。报告 {out/'INDEX.md'}；配对计数与回合区间见summary.json。共同自适应网络并非相同实际pulse，限定开发压力分布，不称多训练seed泛化。
用户要求不主动轮询：agent停止主动查询；完成通知用Linux pidfd进程退出事件，结束自动报告/台账/桌面弹窗，错误独立通知。训练更新0，未停止其它任务/改正式环境，未push。自动图已生成，人工视觉复核待用户叫回。execution=completed；evaluation=complete（声明开发面板）；adoption=not_applicable。
"""
   ledger.write_text(text)
 write_json(out/'completion_watch.json',dict(state='complete',target_pid=a.pid,watcher_pid=os.getpid(),mechanism='Linux pidfd process exit event',finished_at=stamp(),summary=str(out/'summary.json')))
 notify('双引擎1000回合测评完成',f"MJX：{m['successes']}/1000（{m['rate']:.1%}）\nIsaac：{p['successes']}/1000（{p['rate']:.1%}）\n落地后至8秒未失败终止：{m['landing_and_timeout_count']}/{p['landing_and_timeout_count']}（各分母1000）\n\n统计、轨迹和对比图已保存：\n{out/'INDEX.md'}",'complete')
except Exception:
 error=traceback.format_exc();(out/'completion_error.log').write_text(error);write_json(out/'completion_watch.json',dict(state='error',error=error,finished_at=stamp()))
 notify('双引擎测评出错',f'失败日志已保留。\n{out/"completion_error.log"}','error');raise
