# JIT 论文、科研图件与维护索引

2026-09-12。本交付仅针对DVGC/JIT。依据原始工件核验后形成详细工作稿，明确保留当前两轮延迟0→1为0的负结果。本次没有启动仿真、采集、续接或PPO。

## 直接阅读

- **[详细中文论文 PDF](JIT_PAPER_DRAFT.pdf)**：问题、相关工作、30组编号公式、网络/奖励/动作表、算法、全量对照结果、局限与下一步实验。
- [可编辑Markdown正文](JIT_PAPER_DRAFT.md) · [离线HTML](JIT_PAPER_DRAFT.html) · [论文大纲](../JIT_PAPER_OUTLINE.md)
- **控制框图：[SVG](figures/fig01_control_architecture.svg) · [PDF](figures/fig01_control_architecture.pdf) · [PNG](figures/fig01_control_architecture.png)**
- 外循环图：[SVG](figures/fig02_delayed_loop.svg) · [PDF](figures/fig02_delayed_loop.pdf) · [PNG](figures/fig02_delayed_loop.png)

论文目前是方法与真实开发证据草稿。历史all-proposer有经验支持增长，当前学习残差尚未超过随机条件探索，未建立延迟学习收益。没有把方案、工程成功或理论期望写成已验证结论。

## 图表目录

| 图 | 内容与完整范围 | 可编辑图 /可打印图 /预览 |
| --- | --- | --- |
| 1 | 冻结76D基础Actor、106D残差Actor与独立Critic、四通道合成、物理反馈及探索信用 | [SVG](figures/fig01_control_architecture.svg) /[PDF](figures/fig01_control_architecture.pdf) /[PNG](figures/fig01_control_architecture.png) |
| 2 | pending/witnessed分流、声明reset支持、新π任务训练及延迟重评；共享后继π限制 | [SVG](figures/fig02_delayed_loop.svg) /[PDF](figures/fig02_delayed_loop.pdf) /[PNG](figures/fig02_delayed_loop.png) |
| 3 | 历史三个累计时点与两次checkpoint日程全部8个臂 | [SVG](figures/fig03_campaign_checkpoints.svg) /[PDF](figures/fig03_campaign_checkpoints.pdf) /[PNG](figures/fig03_campaign_checkpoints.png) |
| 4 | 两轮×两臂结果、零延迟增益、完整关联成本348,371 | [SVG](figures/fig04_delayed_results.svg) /[PDF](figures/fig04_delayed_results.pdf) /[PNG](figures/fig04_delayed_results.png) |
| 5 | 历史第二轮全部10,464行up/down几何点；含未见证记录，无填充可行凸包 | [SVG](figures/fig05_geometry.svg) /[PDF](figures/fig05_geometry.pdf) /[PNG](figures/fig05_geometry.png) |

统一采用白底、蓝/橙/绿/紫和灰色科研配色，使用位置、标记和文字共同编码，SVG保留文字、PDF嵌入TrueType字体。框图由可复核绘图程序生成，没有AI生成的装饰性结构。英文图标注适合后续英文论文；中文正文提供逐图解释。

## 数据、证据与成本

- [data/README.md](data/README.md)：所有字段、分母、来源和不能混用的口径。
- [evidence.json](data/evidence.json)：原始路径/SHA、精简源字段快照、修订前HEAD与局限。
- [delayed_rounds.csv](data/delayed_rounds.csv)、[cost_breakdown.csv](data/cost_breakdown.csv)、[historical_campaign.csv](data/historical_campaign.csv)、[historical_proposers.csv](data/historical_proposers.csv)、[checkpoint_comparison.csv](data/checkpoint_comparison.csv)、[geometry_points.csv](data/geometry_points.csv)。
- [figures_manifest.json](figures_manifest.json)：图件及绘图输入哈希；[build_manifest.json](build_manifest.json)：正文/PDF构建；[validation.json](validation.json)：本次核验范围与实际结果。
- [references.bib](references.bib)：8篇查验原文的参考文献；本地项目结论来自工件，不来自外部文献。

紧凑sources可离线复核本文数字和绘图。完整状态快照、动作NPZ、所有历史图、模型checkpoint与视频仍位于服务器`JIT/runs/`，论文目录不冒充独立可重放仿真的完整数据集。各原始INDEX入口见[当前状态](../CURRENT_STATUS.md)。旧失败记录没有改写。

## 重建（CPU作者侧工作，不调用仿真）

从仓库根目录：

```bash
python3 JIT/docs/paper/data/rebuild.py --check
/home/qy/mujoco_playground/.venv/bin/python JIT/docs/paper/build_figures.py
```

Markdown与MathML构建依赖可安装到专用临时目录，不改生产训练环境：

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pip install \
  --target /tmp/jit_paper_authoring_deps markdown==3.8.2 latex2mathml==3.78.1
PYTHONPATH=/tmp/jit_paper_authoring_deps \
  /home/qy/mujoco_playground/.venv/bin/python JIT/docs/paper/build_manuscript.py
python3 JIT/docs/paper/verify_artifacts.py
```

绘图依赖生产环境现有NumPy/Matplotlib；PDF使用本机Chrome/Chromium，独立临时profile、headless且`--disable-gpu`。页面使用原生MathML和本地矢量图，无远程脚本/字体依赖。没有Chrome时可用`--html-only`。中文字体需Noto Sans CJK（本机已安装）；PDF读取验证使用poppler的pdfinfo/pdftotext。生成的PDF含时间元数据，因此重建后文件SHA可能变化；源数据、内容与图件来源仍可核对。

## 本次维护文档覆盖清单

“所有维护文档”按现行权威入口处理，不把不可变的历史实验结论改造成今天的状态。

| 文档类别 | 文件 | 本次处理 |
| --- | --- | --- |
| 根入口/项目 | [README](../../../README.md)、[PROJECT](../../../PROJECT.md) | 统一JIT目标、最新完成状态与下一步 |
| 工作规则 | [根AGENTS](../../../AGENTS.md)、[JIT AGENTS](../../AGENTS.md) | 保留科学与Git规则，纠正实现状态/当前优先级 |
| JIT入口/状态 | [README](../../README.md)、[CURRENT_STATUS](../CURRENT_STATUS.md) | 最新两轮证据、成本、口径与完整目录 |
| 接手文档 | [CODEX_HANDOFF_20260911](../CODEX_HANDOFF_20260911.md) | 顶部更新当前入口，9/11正文明确保留为历史 |
| 方法/路线 | [协议](../ENVELOPE_ITERATION_PROTOCOL.md)、[路线](../JIT_TRAINING_ROADMAP.md) | 真实到达奖励、pending支持、同上下文见证、有限外循环 |
| 结构/验证 | [CODE_ORGANIZATION](../CODE_ORGANIZATION.md)、[VERIFICATION](../VERIFICATION.md)、[根布局](../../../docs/REPOSITORY_LAYOUT.md) | 映射现有模块，区分CPU/GPU/开发/最终验证 |
| 论文 | [大纲](../JIT_PAPER_OUTLINE.md)、本目录草稿/图件 | 按历史讨论与真实结果重写并可重绘 |
| 根实验恢复入口 | [EXPERIMENT_STATE](../../../docs/EXPERIMENT_STATE.md) | 指向现行状态，撤下旧π1 next命令 |
| 旧根论文/修改报告 | [DVGC_PAPER_SPEC](../../../docs/research/DVGC_PAPER_SPEC.md)、[RA_L_METHOD_SPEC](../../../docs/RA_L_METHOD_SPEC.md)、[METHOD_TWO_PHASE](../../../docs/METHOD_TWO_PHASE_SOFT_TUBE.md)、[REVISION_REPORT](../../../REVISION_REPORT.md) | 加明确历史标识，保留旧正文以便溯源 |
| 旧日期缺口报告 | [JIT_PAPER_EXPERIMENT_GAPS_20260908](../JIT_PAPER_EXPERIMENT_GAPS_20260908.md) | 标注历史协议，指向当前矩阵 |
| 其他带日期结果、history、run INDEX | 既有实验记录 | 保持原内容与角色，当前入口不把旧next作为执行授权 |

代码分支`agent/two-phase-soft-tube`；报告分支截至此次fetch为`6121da9`，早于本地最新pilot。文档提交不会把此前未发布的训练工件变成远端原始证据。
