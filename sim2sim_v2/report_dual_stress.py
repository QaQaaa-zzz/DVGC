"""Verify all cases and report paired native-contact landing success."""
from pathlib import Path
import json,csv,math,hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent/'runs/dual_stress1000_20261013'
d=json.loads((ROOT/'declaration.json').read_text())
def wilson(k,n):
 z=1.95996398454;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return [c-h,c+h]
records={};coords={};budgets={}
for engine in ['mjx','physx']:
 p=ROOT/engine;st=json.loads((p/'status.json').read_text());assert st['state']=='complete' and st['completed_cases']==1000
 rr=json.loads((p/'episodes.json').read_text());assert sorted(r['case'] for r in rr)==list(range(1000));budgets[engine]=st
 traces={};strength=[]
 for file in sorted(p.glob('batch_*.npz')):
  begin=int(file.stem.split('_')[1]);t=np.load(file);N=t['qpos'].shape[1]
  for lane in range(N):
   case=begin+lane;onset=d['onsets'][case//250];bank=np.load(ROOT/f'initial_onset_{onset:02d}.npz')
   np.testing.assert_array_equal(t['initial_qpos'][lane],bank['qpos'][case%250]);np.testing.assert_array_equal(t['initial_qvel'][lane],bank['qvel'][case%250]);mask=t['mask'][:,lane]
   assert np.all(np.isfinite(t['qpos'][:,lane][mask])) and np.all(np.isfinite(t['qvel'][:,lane][mask]))
   pulse=np.any(t['requested'][:,lane]!=0,axis=1);assert pulse.sum()<=3 and not pulse[:onset].any() and not pulse[onset+3:].any()
   assert abs(t['requested'][:,lane]).max()<=.250001
   x=t['qpos'][:,lane][mask];v=t['qvel'][:,lane][mask];traces[case]=(x,v)
   request=t['requested'][:,lane][pulse];effect=t['effective'][:,lane][pulse];r=rr[case]
   r['pulse_requested_rms']=float(np.sqrt(np.mean(request**2))) if len(request) else 0.;r['pulse_effective_rms']=float(np.sqrt(np.mean(effect**2))) if len(effect) else 0.;r['pulse_applied_steps']=int(pulse.sum())
 records[engine]=rr;coords[engine]=traces
allrows=[dict(engine=e,**r) for e,rr in records.items() for r in rr]
with (ROOT/'episodes.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
summary=dict(protocol_source=d['protocol_source'],actor=d['actor'],actor_sha256=d['actor_sha256'],criterion=d['landing'],training_updates=0,totals={},onsets={},budgets=budgets)
for e,rr in records.items():
 k=sum(r['success'] for r in rr);summary['totals'][e]=dict(successes=k,n=1000,rate=k/1000,ci95=wilson(k,1000),end_code_counts=dict(Counter(r['end_code'] for r in rr)),takeoff_count=sum(r['takeoff_s'] is not None for r in rr),pulse_requested_rms=float(np.mean([r['pulse_requested_rms'] for r in rr])),pulse_effective_rms=float(np.mean([r['pulse_effective_rms'] for r in rr])))
 summary['onsets'][e]={str(onset):dict(successes=sum(r['success'] for r in rr if r['onset']==onset),n=250) for onset in d['onsets']}
m=np.array([r['success'] for r in records['mjx']]);p=np.array([r['success'] for r in records['physx']]);summary['paired']=dict(both_success=int((m&p).sum()),mjx_only=int((m&~p).sum()),physx_only=int((~m&p).sum()),both_failure=int((~m&~p).sum()))
rng=np.random.default_rng(10132026);delta=p.astype(float)-m.astype(float);boot=np.concatenate([rng.choice(delta[gi*250:(gi+1)*250],(10000,250),replace=True).mean(1)[:,None] for gi in range(4)],axis=1).mean(1)
summary['paired']['rate_difference_physx_minus_mjx']=float(delta.mean());summary['paired']['bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
for engine,color,style in [('mjx','#0072B2','-'),('physx','#D55E00','--')]:
 for case,(q,v) in coords[engine].items():
  axs[0,0].plot(q[:,0],q[:,1],color=color,ls=style,lw=.5,alpha=.045)
  axs[0,1].plot(np.arange(1,len(v)+1)*.02,v[:,0],color=color,ls=style,lw=.5,alpha=.04)
 axs[0,0].plot([],[],color=color,ls=style,label=engine.upper())
 axs[0,1].plot([],[],color=color,ls=style,label=engine.upper())
 rates=np.array([summary['onsets'][engine][str(k)]['successes']/250 for k in d['onsets']]);x=np.arange(4)+(0 if engine=='mjx' else .36)
 axs[1,0].bar(x,rates,width=.36,color=color,label=engine.upper())
 strength=[r['pulse_requested_rms'] for r in records[engine]]
 axs[1,1].hist(strength,bins=np.linspace(0,.25,26),histtype='step',color=color,ls=style,label=engine.upper())
axs[0,0].axhline(0,color='.5',lw=.7,label='nominal straight axis');axs[0,0].set_aspect('equal',adjustable='datalim');axs[0,0].set(title='Actual XY trajectories, all 1000 cases per engine',xlabel='world x [m]',ylabel='world y [m]')
axs[0,1].set(title='Actual forward velocity (no success threshold)',xlabel='time [s]',ylabel='vx [m/s]')
axs[1,0].set(title='Stable platform landing >= 1 s',xticks=np.arange(4)+.18,xticklabels=d['onsets'],xlabel='pulse onset [control tick]',ylabel='success fraction',ylim=(0,1))
axs[1,1].set(title='Actual requested pulse strength (3 active ticks)',xlabel='normalized action RMS',ylabel='episodes')
for ax in axs.ravel():ax.grid(alpha=.15);ax.legend()
fig.suptitle('Frozen transition_01966080 | same hard1000 initial bank / frozen explorer / noise tape\nAdaptive pulses depend on each engine state; actual failure endpoints retained',fontsize=12)
fig.savefig(ROOT/'paired_stress.png',dpi=170);plt.close(fig)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
a,b=summary['totals']['mjx'],summary['totals']['physx']
text=f"""# 双引擎现成扰动/探索协议1000回合

先看 [XY实际轨迹、速度、起扰分组成功率与实际探索幅度](paired_stress.png)。

同一冻结模型transition_01966080：MJX **{a['successes']}/1000 ({a['rate']:.1%})**，Isaac/PhysX **{b['successes']}/1000 ({b['rate']:.1%})**。成功定义为真实自然起跳后平台轮支撑、姿态/角速度/非法接触满足要求并连续保持1秒；不要求最低前进速度。timeout、apex和reward均不单独视为成功。

模型及现成方案身份见[声明](declaration.json)，Actor SHA {d['actor_sha256']}。直接复用2026-10-10 hard1000初态与冻结round0099探索网络及atlas，四组起扰0/5/10/15各250，±.25 normalized动作、3控制步。原现成方案的stable_forward_recovery/.5秒分数不是本次1秒无最低前进速度分数，不能直接拿历史成功率当基线。
初态bank位置x约2.5m，有x/y/姿态/速度/角速度预设随机扰动；两端逐案例qpos/qvel完全匹配。没有另加未声明外力或RSI。

|起扰步|MJX成功/250|PhysX成功/250|
|---|---:|---:|
"""
for onset in d['onsets']:text+=f"|{onset}|{summary['onsets']['mjx'][str(onset)]['successes']}|{summary['onsets']['physx'][str(onset)]['successes']}|\n"
text+=f"""
同编号配对：双方成功{summary['paired']['both_success']}、仅MJX成功{summary['paired']['mjx_only']}、仅PhysX成功{summary['paired']['physx_only']}、双方失败{summary['paired']['both_failure']}。两端共享采样噪声带，但探索器依赖各自实际状态，所以共同网络不等于相同实际扰动。requested RMS均值 MJX {a['pulse_requested_rms']:.5f}/PhysX {b['pulse_requested_rms']:.5f}，effective各为{a['pulse_effective_rms']:.5f}/{b['pulse_effective_rms']:.5f}；不据此声称同外力因果比较。

[逐回合CSV](episodes.csv)、[完整统计与区间JSON](summary.json)、[兼容性核验](engineering_verification.json)、[MJX原始batch轨迹](mjx)、[PhysX原始batch轨迹](physx)。每batch保存完整qpos/qvel、原生contact/solver聚合支撑力、Actor observation、合成动作、请求/有效扰动及真实终止mask。图在真实终点截断，没有用padding轨迹补齐失败。

原MJX-Warp sim5ms/ctrl20ms，关闭训练DR/低速失败包装；PhysX既有kd5/5ms/20ms。同Actor/normalizer冻结、没有策略/探索/生成训练更新，未改正式环境/物理/动作映射。采用源端native vmap/shared Warp容量、共享JAX cache与同形状编译复用；目标原生vector task。缓存和批处理不代表跨引擎轨迹逐位相等。

正式charged预算MJX {budgets['mjx']['charged_controls']}控制/{budgets['mjx']['physics_steps']}物理、PhysX {budgets['physx']['charged_controls']}控制/{budgets['physx']['physics_steps']}物理；其中inactive padding物理仍计费，计分有效控制各{budgets['mjx']['scored_controls']}/{budgets['physx']['scored_controls']}。PhysX正式charged含500个接触标定控制。工程兼容性另296控制/1184物理，首导入错误0物理保留。每端实际1000个计分回合，没有重跑历史基线。

这是声明初态与自适应探索施扰分布下、一个冻结训练checkpoint的开发压力结果；Wilson与配对bootstrap区间仅反映回合抽样，不代表多训练种子或未见工况泛化。失败全部保留；不自动训练或更换checkpoint。
"""
(ROOT/'INDEX.md').write_text(text);(ROOT/'status.json').write_text(json.dumps(dict(state='complete',completed_mjx=1000,completed_physx=1000,summary=str(ROOT/'summary.json'),training_updates=0),indent=2))
print(json.dumps(summary['totals'],indent=2))
