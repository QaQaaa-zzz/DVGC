"""Assemble stored-state paper plates and source-backed captions; no physics."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/publication/frontierbridge_sim2sim_20261014'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
for variant in ['color','grayscale']:
 fig,axes=plt.subplots(2,1,figsize=(7.2,6.05))
 for ax,engine,label in zip(axes,['mjx','physx'],['MuJoCo','Isaac']):
  im=Image.open(OUT/'renders'/f'{engine}_overlay_{variant}.png')
  if variant=='grayscale': im=ImageOps.grayscale(im)
  ax.imshow(im,cmap='gray' if variant=='grayscale' else None,vmin=0,vmax=255)
  ax.axis('off');ax.text(0,1.01,label,transform=ax.transAxes,ha='left',va='bottom')
 fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.96,hspace=.14)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'01_jump_sequence_{variant}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.02)
 plt.close(fig)
 fig,axes=plt.subplots(2,5,figsize=(10,3.05))
 for row,(engine,label) in enumerate([('mjx','MuJoCo'),('physx','Isaac')]):
  for col in range(5):
   im=Image.open(OUT/'renders'/f'{engine}_frame_{col}.png')
   if variant=='grayscale':im=ImageOps.grayscale(im)
   axes[row,col].imshow(im,cmap='gray' if variant=='grayscale' else None,vmin=0,vmax=255);axes[row,col].axis('off')
  axes[row,0].text(0,1.01,label,transform=axes[row,0].transAxes,va='bottom')
 fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.95,wspace=.01,hspace=.18)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'01b_jump_frames_{variant}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.02)
 plt.close(fig)
# Normalize generated CSV line endings for downstream LaTeX and Git tooling.
for p in (OUT/'data').glob('*.csv'):p.write_text(p.read_text())
identity_path=OUT/'source_identity.json'
identity=json.loads(identity_path.read_text());identity.update(visual_revision=2,display_labels=['MuJoCo','Isaac'],renderers={'mjx':'MuJoCo native OpenGL EGL','physx':'Isaac native RTX'},knee_visual='historical stretch, renderer-local visual geometry only',previous_package='frontierbridge_sim2sim_20261014_revision1.zip')
identity_path.write_text(json.dumps(identity,indent=2)+'\n')
text=OUT/'text'
section=r'''\subsection{Cross-simulator validation}
We evaluated the same frozen 76-dimensional Actor and observation normalizer in MuJoCo and Isaac, with a 5\,ms physics step and a 20\,ms control period. The evaluated checkpoint, transition\_01966080, was obtained after 1,966,080 Isaac PPO adaptation transitions from the MuJoCo checkpoint transition\_9904128. The experiment therefore establishes execution of a single fixed policy in both simulators. It does not establish source-only, zero-target-training transfer or evaluate a FrontierBridge-trained student. From the paired nominal ground initialization, both engines produced takeoff, platform touchdown, and a continuous one-second stable landing interval. No minimum forward speed was imposed. The figures display measured states only until the first control sample with root $x\geq5$\,m (1.76\,s in MuJoCo and 2.68\,s in Isaac). The full-window landing labels are retained independently of this display cutoff.

A separate existing stress evaluation used 1,000 paired perturbed initial states per engine and the same frozen exploration network and noise tape. Normalized action disturbances were bounded by 0.25 over three control ticks, with onset indices 0, 5, 10, and 15, each covering 250 cases. Because the explorer is state-dependent, the realized action disturbances can differ between engines. Stable landing for at least one continuous second occurred in 170/1,000 MuJoCo cases (17.0\%, Wilson 95\% CI: 14.8--19.5\%) and 279/1,000 Isaac cases (27.9\%, 25.2--30.8\%). Seventeen of the Isaac landing-positive cases failed later in the eight-second evaluation window; 262/1,000 cases (26.2\%, 23.6--29.0\%) both landed and avoided a subsequent failure. All 170 MuJoCo landing-positive cases avoided a subsequent failure in that window. These rates describe this frozen candidate under the declared stress protocol and are not a baseline-versus-FrontierBridge gain measurement.
'''
(text/'section_cross_simulator.tex').write_text(section)
(text/'section_cross_simulator.md').write_text(section.replace('\\subsection{Cross-simulator validation}','## Cross-simulator validation').replace('\\,',' ').replace('\\_','_').replace('\\%','%').replace('$x\\geq5$','x ≥ 5').replace('--','–'))
captions={
 '01_jump_sequence':'Five measured poses of the same frozen policy in MuJoCo (top) and Isaac (bottom): approach, takeoff, apex, touchdown, and the first root-x sample at or beyond 5 m. Colors indicate chronological order. Poses and trajectories are unedited; each row uses its simulator’s own native renderer (MuJoCo OpenGL and Isaac RTX) and a level camera. The historical visual-only knee stretch is applied in both engines. Stable-landing labels use the original full-window contact evaluation.',
 '01b_jump_frames':'Chronological state-replay frames for the same five measured poses in each engine. Tracking cameras retain a level orientation; no robot pose correction is applied.',
 '02_trajectory':'World XY and XZ root trajectories from a paired nominal ground initialization, ending at the first measured root-x sample at or beyond 5 m. Ground and platform profiles are shown in XZ. The same frozen Actor is used in both engines.',
 '03_states':'Nominal root height, roll, pitch, forward velocity, and body-frame angular velocities under the same frozen policy. Each engine retains its own physical time axis and stops at its first root-x sample at or beyond 5 m.',
 '04_contact':'Native measured wheel contact support on the ground and platform during the nominal jump. Contact forces are in newtons; the display cutoff does not replace the full-window stable-landing evaluation.',
 '05_actions':'Four normalized Actor outputs during nominal cross-simulator execution. Action timestamps denote the start of their 20 ms control interval. The Actor and normalizer are frozen and shared.',
 '06_robustness':'Stable-landing rates under the existing paired 1,000-case stress protocol. Overall rates are shown with Wilson 95% confidence intervals; onset-conditioned rates use 250 cases per onset. Stable landing requires a continuous one-second interval and no minimum forward speed. Subsequent failures are reported separately in the accompanying table.'}
(text/'captions.md').write_text('\n\n'.join(f'**{k}**\n\n{v}' for k,v in captions.items())+'\n')
(text/'captions.tex').write_text('\n\n'.join('% '+k+'\n\\caption{'+v.replace('%','\\%')+'}' for k,v in captions.items())+'\n')
(text/'results_table.tex').write_text(r'''\begin{tabular}{lrrr}
\hline
Simulator & Cases & Stable landing & Landing without later failure \\
\hline
MuJoCo & 1,000 & 170 (17.0\%) & 170 (17.0\%) \\
Isaac & 1,000 & 279 (27.9\%) & 262 (26.2\%) \\
\hline
\end{tabular}
''' )
(text/'draft_gap_inventory.md').write_text('''# Draft asset map

The supplied PDF remains unchanged. This package supplies measured cross-simulator assets, not synthetic substitutes for missing FrontierBridge experiments.

| Draft item | Deliverable / missing measurement | Status |
|---|---|---|
| Fig. 9 cross-simulator results | Figures 01–06, same-policy identity, raw nominal/stress traces, real contact labels | Verified for Isaac-adapted01966080; source-only zero-shot/FrontierBridge student unverified |
| Fig. 3 headline72.4→92.4 | Exact baseline/student checkpoint, sample denominator and matched evaluation protocol | Unverified; current17.0/27.9 are different-engine stress rates, not these headline rates |
| Table III baselines | Common states, reward/objective, interaction budget, success rule and repeated training runs | Matched production comparison not run |
| Fig. 4 exploration efficiency | Verified distinct recoveries versus all charged interactions | Existing historical explorer evidence is available; not automatically evidence for the new draft method |
| Fig. 5 teacher search ablations | Horizon×candidate budget, actual query cost, independent student recovery | Complete matched ablation not run |
| Fig. 6 source/teacher/student example | Same augmented initial state and native state/action/contact traces for each role | Current package compares simulators, not source/teacher/student |
| Fig. 7 / Table IV acquisition and retention | Matched frozen-parent retention bank and independent new-skill evaluation per ablation | Existing engineering BC/guarded-BC evidence cannot be renamed PPO objective ablation |
| Fig. 8 amplitude×onset improvement | Paired baseline/student case outcomes across amplitudes | Current stress bank has one amplitude(.25); gain heatmap unavailable |

Other paper assets produced by the separate JIT task are indexed at `/home/qy/DVGC/JIT/runs/monitoring/frontierbridge_paper_figures_20261014/INDEX.md`. Their own claim map and historical-method boundaries must be retained. No additional experiment was launched to fill these gaps.
''')
(text/'literature_visual_conventions.md').write_text('''# Primary-paper visual conventions

- [DexNDM](https://arxiv.org/html/2510.08556v1), Fig.28 and Table6: paired simulator qualitative views accompanied by quantitative cross-simulator comparison. Used here as a presentation convention, not as borrowed evidence or imagery.
- [ResMimic](https://arxiv.org/html/2510.05070v1), Section IV-B, Fig.5 and TableI: cross-simulator motion/error curves together with task-success metrics. Used here to motivate paired trajectories, state curves, and explicit success denominators.

This package separates nominal examples from stress-population rates and keeps minimal engine legends, axis labels and units in the figures. Provenance, event timing and limitations belong in captions and data manifests.
''')
(OUT/'INDEX.md').write_text('''# FrontierBridge draft: cross-simulator assets

原PDF未修改。当前为修订版2：MuJoCo原生渲染、Isaac原生RTX，两端knee视觉拉长，显示名称简化。本包包含同一个冻结Actor的双引擎真实轨迹、五时刻图、统计数据与英文插入文本。没有新增训练或物理仿真。

## 主图

- [XY/XZ轨迹：彩色](figures/02_trajectory_color.png) / [PDF](figures/02_trajectory_color.pdf) / [黑白](figures/02_trajectory_grayscale.png)
- [五时刻跳跃叠图：彩色](figures/01_jump_sequence_color.png) / [PDF](figures/01_jump_sequence_color.pdf) / [黑白](figures/01_jump_sequence_grayscale.png) / [黑白PDF](figures/01_jump_sequence_grayscale.pdf)
- [双引擎五帧序列](figures/01b_jump_frames_color.png) / [黑白](figures/01b_jump_frames_grayscale.png)
- [速度、姿态、高度、角速度](figures/03_states_color.png) / [PDF](figures/03_states_color.pdf)
- [接触力](figures/04_contact_color.png) · [四路动作](figures/05_actions_color.png) · [1000次扰动成功率](figures/06_robustness_color.png)

所有图有彩色/黑白PNG、PDF、SVG。推荐正文用01+02+06，状态、接触、动作放补充材料。画面只保留引擎标签，曲线只保留坐标单位和图例。

## 数据与可直接使用内容

- [MuJoCo逐步CSV](data/nominal_mjx_x5.csv) / [NPZ含76维观测](data/nominal_mjx_x5.npz)
- [Isaac逐步CSV](data/nominal_physx_x5.csv) / [NPZ含76维观测](data/nominal_physx_x5.npz)
- [1000次×双引擎原完整窗口结果](data/stress_episodes_full_window.csv)
- [MuJoCo1000条截短轨迹](data/stress_mjx_x5.npz) / [Isaac1000条截短轨迹](data/stress_physx_x5.npz)：offsets为每条轨迹边界，保留真实失败终点，无末帧补齐。
- [MuJoCo成功率/置信区间](data/success_mjx.csv) / [Isaac](data/success_physx.csv)
- [英文段落LaTeX](text/section_cross_simulator.tex) · [图注](text/captions.md) · [LaTeX图注](text/captions.tex) · [表格](text/results_table.tex)
- [草稿缺失数据逐项表](text/draft_gap_inventory.md) · [其他论文的展示方式](text/literature_visual_conventions.md)

## 身份与统计口径

Actor01966080 SHA256 `d2648c4f33b3e191293d2a4281d9a95f475f2a2be09c14a0ad8f605235c2d1a0`。它来自9904128之后的1,966,080步Isaac PPO适配，不能称原MuJoCo策略零目标域训练迁移，也不能称FrontierBridge学生的实验结果。两端Actor/normalizer冻结；5ms物理、20ms控制。

名义自然初态两端都完成一秒稳定落地，不要求保持前向速度。图截止到首次root x≥5m：MuJoCo1.76s、Isaac2.68s；这早于原完整窗口的一秒保持确认(2.68s/2.80s)，故图仅显示运动过程，统计标签仍来自原8s记录。1000次扰动一秒稳定落地：MuJoCo170/1000，Isaac279/1000；Isaac随后17例失败，整个窗口落地且无后续失败262/1000。

扰动采用已有hard1000初态bank、round0099冻结探索器/atlas及相同噪声，动作幅度.25、3ticks、onset0/5/10/15各250例。探索器按实时状态输出，所以实际脉冲不保证逐值相同。当前数据不是草稿72.4→92.4的同协议比较。

序列图分别使用MuJoCo原生OpenGL和Isaac原生RTX重放各自保存的qpos。两边均复用历史knee视觉拉长：固定局部末端，上连杆伸至knee锚点，约1.345794倍；只修改渲染副本的外观，不改碰撞、质量、惯量、轨迹或最后一帧姿态。所有显示名称统一MuJoCo和Isaac。原动力学/接触数据不受影响。

原始记录：[nominal](../../no_speed_dual_landing/REPORT.md)、[stress1000](../../dual_stress1000_20261013/INDEX.md)、[模型与视频存档](../../deliveries/dual_landing_01966080_20261009/INDEX.md)。[显示契约](data/plot_contract.json)、[输入身份](source_identity.json)、[渲染核验](renders/verification.json)、[数据与交付核验](verification.json)。重绘入口位于sim2sim_v2/build_publication_assets.py、render_publication_sequence.py和package_publication_assets.py、render_publication_mujoco.py和publication_knee_visual.py。旧版ZIP保留为frontierbridge_sim2sim_20261014_revision1.zip。
''')
print('Packaged plates, captions, tables and evidence map')
