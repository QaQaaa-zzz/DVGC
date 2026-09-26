#!/usr/bin/env python3
"""Replot audited JIT paper data and editable control diagrams; no simulation."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'; OUT=HERE/'figures'; OUT.mkdir(exist_ok=True)
BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#00856A'; PURPLE='#77569C'; GREY='#59636E'; LIGHT='#EEF2F5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'savefig.facecolor':'white',
 'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'svg.hashsalt':'jit-paper-20260912'})
manifest={'schema':'jit_paper_figures_v1','scientific_scope':'TRAIN/development; no simulator calls','figures':[]}
def rows(name):
 with (DATA/name).open() as f:return list(csv.DictReader(f))
def save(fig,name,sources,caption):
 paths=[]
 for ext in ['svg','pdf','png']:
  p=OUT/f'{name}.{ext}'; fig.savefig(p,dpi=260,bbox_inches='tight',pad_inches=.14,metadata={'Creator':'JIT reproducible paper figures'} if ext=='pdf' else None);paths.append(p)
  if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
 manifest['figures'].append({'name':name,'caption':caption,'source_files':sources,
  'source_sha256':{s:hashlib.sha256((HERE/s).read_bytes()).hexdigest() for s in sources},
  'files':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
 plt.close(fig)
def canvas(w=16,h=9):
 fig,ax=plt.subplots(figsize=(w,h));ax.set_xlim(0,w);ax.set_ylim(0,h);ax.axis('off');return fig,ax
def box(ax,x,y,w,h,title,detail='',color=BLUE,size=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.09',fc=color+'0D',ec=color,lw=1.3))
 if detail:
  ax.text(x+w/2,y+h*.73,title,ha='center',va='center',fontsize=size+1,fontweight='bold',color=color)
  ax.text(x+w/2,y+h*.36,detail,ha='center',va='center',fontsize=size,linespacing=1.45,color='#26323A')
 else:ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=size+1,color=color,fontweight='bold')
def arrow(ax,points,color=GREY,dashed=False,label=None,at=None,size=10):
 for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.2,ls='--' if dashed else '-')
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,lw=1.2,color=color,linestyle='--' if dashed else '-'))
 if label:ax.text(*at,label,fontsize=size,color=color,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','pad':1.3})

fig,ax=canvas(16,9)
ax.text(.25,8.73,'JIT  |  Frozen-policy residual exploration',fontsize=17,fontweight='bold',color='#233544')
ax.text(.25,8.32,'Real control loop at 50 Hz  •  Four 5 ms physics substeps per control tick',fontsize=11,color=GREY)
box(ax,.35,4.65,2.15,2.45,'Observation','3-frame FIFO\n'+r'$o_t\in\mathbb{R}^{76}$'+'\n'+r'$z_t\in\mathbb{R}^{106}$',GREY)
box(ax,3.25,6.1,2.85,1.1,r'Frozen base $\pi_i$','76 → 256 × 3 → 8',BLUE)
box(ax,3.25,3.7,2.85,1.7,r'Residual Actor $\theta$','106 → 256 × 3 → 8\nSwish; Gaussian parameters',ORANGE)
box(ax,6.65,3.7,2.25,1.7,'Bounded residual',r'$u\sim\mathcal{N}(\mu,\sigma^2)$'+'\n'+r'$\delta^{req}=d\odot\tanh u$',ORANGE)
ax.add_patch(Circle((9.55,6.65),.29,fc='white',ec=GREY,lw=1.3));ax.text(9.55,6.65,'+',fontsize=22,ha='center',va='center',color=GREY)
box(ax,10.25,6.07,1.25,1.16,'Clip','[−1, 1]',GREY)
box(ax,12.1,5.65,3.5,1.95,'Actuator map + MuJoCo','Steer / rear drive / hip / knee\nComplete state and events',BLUE)
arrow(ax,[(2.5,6.65),(3.25,6.65)],label=r'$o_t$',at=(2.85,6.95))
arrow(ax,[(2.5,5.15),(2.9,5.15),(2.9,4.55),(3.25,4.55)],label=r'$z_t$',at=(2.85,5.55))
arrow(ax,[(6.1,6.65),(9.26,6.65)],BLUE,label=r'$a_t^{base}$ (stop gradient)',at=(7.7,6.96))
arrow(ax,[(6.1,4.55),(6.65,4.55)],ORANGE)
arrow(ax,[(8.9,4.55),(9.55,4.55),(9.55,6.36)],ORANGE,label=r'$d_j=0.15$',at=(9.6,5.45),size=9)
arrow(ax,[(9.84,6.65),(10.25,6.65)])
arrow(ax,[(11.5,6.65),(12.1,6.65)],label=r'$a_t$',at=(11.8,6.95))
arrow(ax,[(13.85,7.6),(13.85,7.9),(1.43,7.9),(1.43,7.1)],BLUE,label='State, sensor features, action and event history',at=(7.2,7.9))
box(ax,12.1,2.45,3.5,1.65,'Causal trajectory record','Actions, log probabilities, masks\nSnapshots + prefix / context IDs',GREY)
arrow(ax,[(13.85,5.65),(13.85,4.1)],label='Actual transitions',at=(13.85,4.86))
box(ax,7.1,1.08,3.1,1.65,'Per-policy arrival credit','New selected root cell → total 1\nCurrent π ledger; suffix not required',GREEN,size=9)
box(ax,3.25,1.08,2.85,1.65,'Residual PPO','Actor θ + own Critic ψ\n106D → 256 × 3 → 1',PURPLE,size=9)
arrow(ax,[(12.1,3.25),(11.05,3.25),(11.05,1.9),(10.2,1.9)],GREY)
arrow(ax,[(7.1,1.9),(6.1,1.9)],GREEN,label=r'$r^{exp}$ + active samples',at=(6.55,1.4),size=9)
arrow(ax,[(4.68,2.73),(4.68,3.7)],PURPLE,True,label='PPO update',at=(4.68,3.23))
arrow(ax,[(1.43,4.65),(1.43,1.9),(3.25,1.9)],GREY,label=r'$z_t$',at=(1.42,3.3))
ax.text(.35,.44,'Solid: control / data flow    Dashed: parameter update    Base Actor + normalizer remain frozen within this exploration block.',fontsize=10,color=GREY)
ax.text(.35,.05,'Suffix evaluation and successor-policy training are separate stages (Fig. 2). Simulation observations; no stability or deployment guarantee.',fontsize=9.5,color=GREY)
save(fig,'fig01_control_architecture',['data/evidence.json'], 'Implemented online control and residual PPO; no suffix reward gate or outer-loop gradient.')

fig,ax=canvas(16,8.25)
ax.text(.25,7.95,'JIT  |  Finite exploration–learning loop',fontsize=17,fontweight='bold',color='#233544')
ax.text(.25,7.54,'Arrival credit, training support and landing witnesses remain distinct',fontsize=11,color=GREY)
box(ax,.4,5.35,2.35,1.65,'Frozen source π','Actor + normalizer\nPolicy bank retained',BLUE)
box(ax,3.35,5.35,3.1,1.65,'Real forward exploration','Learned / random residual\nSame source π, bounded actions',ORANGE,size=9.5)
box(ax,7.05,5.35,3.4,1.65,'Same-context bank suffix','Residual OFF; first valid landing\n1 / 0 / unknown kept separately',BLUE,size=9.5)
box(ax,11.15,5.35,4.2,1.65,'Candidate archive','Full-state / context identities\nEvery attempt and past label retained',GREY,size=9.5)
for a,b in [((2.75,6.18),(3.35,6.18)),((6.45,6.18),(7.05,6.18)),((10.45,6.18),(11.15,6.18))]:arrow(ax,[a,b])
box(ax,11.15,2.9,4.2,1.45,'Witnessed W → empirical cells','Real prefix + successful suffix\nHistorical valid evidence retained',GREEN,size=9.5)
box(ax,7.05,2.9,3.4,1.45,'Pending P','No witness yet ≠ impossible\nKeep 0 and unknown histories',GREY,size=9.5)
arrow(ax,[(13.25,5.35),(13.25,4.35)],GREEN,label='Any valid witness',at=(13.25,4.82))
arrow(ax,[(11.15,5.6),(10.75,5.6),(10.75,4.7),(8.75,4.7),(8.75,4.35)],GREY,label='No valid witness',at=(8.75,4.82),size=9)
box(ax,3.35,2.9,3.1,1.45,'Declared reset support S','20% fixed start + 80% snapshots\nPhase / trajectory balance',PURPLE,size=9)
arrow(ax,[(7.05,3.63),(6.45,3.63)],GREY)
arrow(ax,[(13.25,2.9),(13.25,2.4),(4.9,2.4),(4.9,2.9)],GREEN,label='Witnessed support preserved',at=(9.15,2.4),size=9)
box(ax,.4,.75,2.7,1.5,'Task PPO → new π','Warm Actor + normalizer\nFresh Critic + optimizer',PURPLE,size=9)
arrow(ax,[(3.35,3.63),(1.75,3.63),(1.75,2.25)],PURPLE)
arrow(ax,[(.65,2.25),(.65,4.68),(1.57,4.68),(1.57,5.35)],PURPLE,True,label='Next round',at=(1.57,4.68),size=9)
box(ax,5.25,.75,4.8,1.1,'Delayed reevaluation of old pending','New frozen π only; append outcomes',PURPLE,size=9.5)
arrow(ax,[(3.1,1.5),(5.25,1.5)],PURPLE)
arrow(ax,[(8.75,2.9),(8.75,1.85)],GREY)
arrow(ax,[(10.05,1.3),(14.1,1.3),(14.1,2.9)],PURPLE,True,label='New witness only if suffix succeeds',at=(12.1,1.8),size=9)
ax.text(.4,.15,'Current pilot: only learned-arm pending trains the shared successor π. Conditional exploration control; no end-to-end outer-loop backpropagation.',fontsize=9.6,color=GREY)
save(fig,'fig02_delayed_loop',['data/evidence.json'], 'Actual candidate, training and reevaluation loop; shared-source comparison limitation.')

hist=rows('historical_campaign.csv'); ck=rows('checkpoint_comparison.csv')
fig,axs=plt.subplots(1,3,figsize=(13.8,4.5),gridspec_kw={'width_ratios':[1.08,1,1]})
x=np.array([int(r['cumulative_new_interactions']) for r in hist])/1e6;y=[int(r['cumulative_root_cells']) for r in hist]
axs[0].plot(x,y,'o-',color=BLUE,lw=1.8,ms=6)
for a,b in zip(x,y):axs[0].annotate(f'{b:,}',(a,b),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
axs[0].set(xlabel='New campaign interactions (millions)',ylabel='Cumulative witnessed root cells',title='(a) Historical all-proposer campaign',ylim=(0,10800),xlim=(-.14,1.42))
axs[0].set_xticks([0,.48332,1.275465],['0','0.483','1.275'])
for ax,experiment,title in zip(axs[1:],['early','late'],['(b) 128k pilot; budget 4,058','(c) Fresh 256k pilot; budget 4,135']):
 rr=[r for r in ck if r['experiment']==experiment];val=[int(r['common_budget_novel_root_cells']) for r in rr]
 labels=['π6' if r['arm']=='pi_6' else str(int(r['arm'].split('_')[-1])//1000)+'k' for r in rr]
 ax.bar(range(len(rr)),val,color=BLUE,width=.58)
 for i,v in enumerate(val):ax.text(i,v+5,str(v),ha='center')
 ax.set(xticks=range(len(rr)),xticklabels=labels,ylabel='Novel witnessed root cells',xlabel='Frozen proposer checkpoint',title=title,ylim=(0,225))
for ax in axs:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.text(.055,.015,'Historical protocol labels in (a). Panels (b–c): offline declared-order cost truncation; training costs excluded, no independent statistical repeats.',fontsize=9,color=GREY)
fig.tight_layout(rect=[0,.07,1,1],w_pad=2.2)
save(fig,'fig03_campaign_checkpoints',['data/historical_campaign.csv','data/checkpoint_comparison.csv'],'Historical expansion and both complete checkpoint comparisons, separate fresh-run scopes; same training seed.')

delayed=rows('delayed_rounds.csv');cost=rows('cost_breakdown.csv')
fig,axs=plt.subplots(2,2,figsize=(12,8.4))
for arm,color,label in [('learned_residual',ORANGE,'Learned residual'),('fixed_random',BLUE,'Fixed random')]:
 rr=sorted([r for r in delayed if r['arm']==arm],key=lambda r:int(r['round_index']))
 xx=[int(r['round_index'])+1 for r in rr]
 for ax,key in [(axs[0,0],'verified_root_cells'),(axs[0,1],'pending_contexts')]:
  yy=[int(r[key]) for r in rr];ax.plot(xx,yy,'o-',color=color,label=label,lw=1.8)
  for a,b in zip(xx,yy):ax.annotate(str(b),(a,b),xytext=(10,2 if arm=='fixed_random' else -13),textcoords='offset points',color=color)
axs[0,0].set(title='(a) Cumulative witnessed cells',ylabel='Root cells',ylim=(0,350));axs[0,0].legend(frameon=False,loc='upper left')
axs[0,1].set(title='(b) Pending candidate contexts',ylabel='Candidate contexts',ylim=(0,15));axs[0,1].legend(frameon=False,loc='upper left')
for ax in axs[0]:ax.set(xlabel='Completed outer round',xticks=[1,2],xlim=(.9,2.2));ax.grid(axis='y',alpha=.18)
ax=axs[1,0];ax.axis('off');ax.text(.02,.94,'(c) Delayed capability gain',fontweight='bold',fontsize=12,transform=ax.transAxes)
ax.text(.02,.63,'0 → 1:  0 in both arms, both rounds',fontsize=13,color='#29333B',transform=ax.transAxes)
ax.text(.02,.41,'Unknown → 1:  0 in both arms, both rounds',fontsize=11,color=GREY,transform=ax.transAxes)
ax.text(.02,.1,'One adaptive lineage; only learned pending trains\nthe shared next π. Counts are not independent trials.',fontsize=10,color=GREY,linespacing=1.6,transform=ax.transAxes)
cc={}
for r in cost:
 if r['scope']=='delayed_pilot':cc[r['stage']]=cc.get(r['stage'],0)+int(r['charged_interactions'])
# The cost source keeps all detailed labels; classify only for this additive display.
forward=sum(int(r['forward_scheduled_interactions']) for r in delayed);active=sum(int(r['forward_active_interactions']) for r in delayed)
ppo=sum(int(r['charged_interactions']) for r in cost if r['scope']=='delayed_pilot' and r['stage']=='policy_ppo')
panel=sum(int(r['charged_interactions']) for r in cost if r['scope']=='delayed_pilot' and r['stage']=='fixed_train_panel')
total=sum(cc.values());suffix=total-forward-ppo-panel
assert (total,forward,active,ppo,panel,suffix)==(348371,76800,8719,256000,149,15422)
ax=axs[1,1];vv=[ppo,forward,suffix,panel];labs=['Policy PPO','Forward slots','Suffix labels','TRAIN panel']
ax.barh(range(4),np.array(vv)/1000,color=[PURPLE,ORANGE,BLUE,GREY],height=.6)
for i,v in enumerate(vv):ax.text(v/1000+3,i,f'{v:,}',va='center',fontsize=10)
ax.set(yticks=range(4),yticklabels=labs,xlabel='Charged interactions (thousands)',title='(d) Two-round associated cost: 348,371',xlim=(0,313));ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.text(.045,.012,'Forward slots: 8,719 active + 68,081 padding. One inherited 19,200-slot acquisition counted once. Older bootstrap / separate attempts excluded.',fontsize=9,color=GREY)
fig.tight_layout(rect=[0,.055,1,1],h_pad=3,w_pad=2.3)
save(fig,'fig04_delayed_results',['data/delayed_rounds.csv','data/cost_breakdown.csv'],'All two-round outcomes and separately audited additive cost; zero delayed gains retained.')

geom=rows('geometry_points.csv');assert len(geom)==10464
fig,axs=plt.subplots(2,2,figsize=(12,8.1),sharex=True)
for col,phase in enumerate(['upstream','downstream']):
 pp=[r for r in geom if r['phase']==phase]
 for row,key,ylab in [(0,'root_z_m','Root height z (m)'),(1,'root_vz_mps','Root vertical velocity (m/s)')]:
  ax=axs[row,col]
  for witnessed,color,marker,label in [('True',BLUE,'.','Historical witnessed'),('False',ORANGE,'x','No historical witness')]:
   rr=[r for r in pp if r['witnessed']==witnessed]
   ax.scatter([float(r['root_x_m']) for r in rr],[float(r[key]) for r in rr],s=7 if marker=='.' else 13,c=color,marker=marker,alpha=.32 if marker=='.' else .7,lw=.65,label=f'{label} ({len(rr):,})',rasterized=False)
  ax.set_ylabel(ylab);ax.grid(alpha=.14);ax.set_axisbelow(True)
  if row==0:ax.set_title(f'{phase.capitalize()} — all {len(pp):,} records');ax.legend(frameon=False,fontsize=8.5,loc='best')
  else:ax.set_xlabel('Longitudinal root position x (m)')
fig.text(.045,.012,'All 10,464 records retained; historical label semantics. Point projections are not continuous feasible regions. Root height is not wheel clearance.',fontsize=9,color=GREY)
fig.tight_layout(rect=[0,.05,1,1],w_pad=2,h_pad=1.4)
save(fig,'fig05_geometry',['data/geometry_points.csv'],'Complete historical phase geometry, including unwitnessed points; no hull or interpolated feasibility.')
(HERE/'figures_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built',len(manifest['figures']),'figures in SVG/PDF/PNG')
