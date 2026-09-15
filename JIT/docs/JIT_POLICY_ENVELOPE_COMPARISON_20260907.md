# π0—π3 经验包线比较：运行与回传

## 当前任务与用户决定

2026-09-07 用户确认：接受约 3.1 cm 初始轮胎间隙的近地初始化；接受已观察到的数值轨迹偏差，不继续追加重放验收。上一批 6 个状态的四种接续全部首次有效落地，对应落地步数一致，串行/分片标签一致；逐状态重放并未全部通过。保留这个事实，不改写为精确重放通过。见[运行记录](verification/jump_evidence_user_run_20260907.json)。

现在执行已授权的既有数据补标和全策略比较，不训练新策略。

本次实现已通过 46 项针对性 CPU 回归、Python 语法及文档链接检查，并检查了合成样例的 PNG/PDF/SVG 排版。见[验证记录](verification/policy_comparison_cpu_20260907.json)。真实 checkpoint、GPU 补标和最终论文数据图等待本次服务器运行；合成图仅用于软件验证。

## 一条命令

在保留真实 checkpoint、candidate catalog 和 snapshot 的服务器执行：

```bash
cd /home/qy/DVGC &&
git switch agent/two-phase-soft-tube &&
git pull --ff-only origin agent/two-phase-soft-tube &&
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/compare_policy_envelopes.py --gpu 0
```

`--gpu 0` 选择 GPU 0；CPU 预检、合并和画图不占用 GPU。GPU 标签任务逐个独立进程执行，每进程默认最多 200 个候选，不并发运行 evaluator。每 30 秒打印当前日志路径。

默认输出目录固定为：

```text
JIT/runs/policy_comparison/expanded_pi0_pi3_20260907/
```

运行失败或中断后，原样再次执行命令即可续跑。已完成分片经过合同检查后复用；未完成尝试保留，重试写入新目录，不覆盖旧失败。可以在续跑时更换 `--gpu`；若要修改分片大小、预算、策略路径或源扫描，必须指定新的 `--output-dir`。不要同时在两个终端运行相同输出目录，程序会拒绝第二个进程。

结束时打印 `Return this file: .../results_to_send.zip`。**回传这个 ZIP 即可，包括失败时的包。** 包含图、CSV、计划、标签来源、日志和成本，不包含大型 checkpoint 或 pickle snapshot。

CPU 预检需要现有生产依赖及 Matplotlib；缺文件/依赖会在开始新标签前失败并打包。不要改名 checkpoint 来绕过身份检查。默认从原计划读取 π0/π1/π2 的精确冻结记录，并搜索唯一 `pi_3`；如果 π3 存在多个版本，报告会列出路径。此时通过四个 `--frozen-policy` 参数明确指定 π0—π3，前三个必须仍匹配原锁定实验。

## 这次实际做什么

1. 读取既有 expanded scan 的计划、真实到达候选、来源、快照和冻结策略身份。
2. 逐候选提取物理坐标；保留准确快照上下文身份，另行计算物理单元。
3. 原 π0/π1/π2 使用原 role seed、400 步 horizon 和首次有效落地终点，验证后复用完整输出，只补缺失 evaluator。
4. π3 在新比较计划锁定后，对同一批状态按同一 seed/horizon/endpoint 评价。它不加入旧 family，也不使用旧 mixed-endpoint selected gate。
5. 按 TRAIN、CALIBRATION、ACCEPTANCE 分开生成四策略与并集的比较结果。原三成员 family 另有派生闭合视图，原始历史目录不修改。

按当前已提交记录，三个角色分别有 1,754 / 583 / 574 个候选，共 2,911 个。CAL/ACCEPT 的 π0/π1 完整标签可在本机实际校验后复用。如果状态未变化，待补 9,330 个策略—候选评价，最多 3,732,000 次环境交互，约 48 个 GPU 分片进程。实际首次落地/失败会提前结束；以运行时 `plan.json` 为准。

默认新标签预算 5,000,000 次，包含失败/重试的保守预留；不是 PPO 步数。中断或失败分片按完整分片上限计入预算，完整分片使用记录成本。历史 acquisition 和 evaluator 的已知记录另存，缺失历史重试及 bootstrap/PPO 成本不凭空补齐。并集的 suffix cost 是所用各 evaluator 标签成本的和，不代表一个 Actor 的执行成本，也不是已闭合的论文端到端成本曲线。

## 可以画出哪些“包线”

可以画出 π0、π1、π2、π3 的独立图及并集，但这批图的准确含义是：

> **同一批 π0 前向采集到的状态上，各策略能成功接续落地的经验支持。**

这样四个策略的候选范围完全相同，可直接比较接续能力与互补贡献。这还不是“每个策略自己从起点探索得到的全部包线”；后者需要单独开展多 proposer 同预算采集。原扫描还排除了已有训练支持中的重复状态，所以本批属于 frontier panel，不能称为完整物理能力范围。up/down 专家不纳入统一策略排名，其任务和终点不同。

灰色点是共同到达候选，彩色点是该策略成功接续的状态；圆点/三角区分 upstream/downstream。黑色虚线只在有原始坐标的投影中显示固定 π0 centerline，作为参考，不给任何策略额外添加成功样本。

所有策略共用坐标范围与物理分辨率。只绘制观测点、计数及逐切片观测最小—最大跨度，不用凸包填充未知区域，不把跨度内部宣称为全部可行。

## 输出图和表

每个角色目录 `figures/train/`、`figures/calibration/`、`figures/acceptance/` 包含：

| 文件 | 用途 |
| --- | --- |
| `all_policy_projections` | 四策略 + 并集，统一坐标的 x–z、x–vx、x–vz、roll–pitch 投影 |
| `pi_0_projections` 至 `pi_3_projections` | 每个策略独立图，可用于论文排版 |
| `union_projections` | 当前四策略并集支持 |
| `cross_section_occupancy` | 每个 x 切片、各阶段的成功物理单元数 |
| `cross_section_spans` | 各策略的 z、vx、vz、pitch 观测跨度，不填充可行域 |
| `policy_contributions` | 各策略总覆盖与“移除它后损失”的独有物理覆盖 |
| `policy_overlap` | root/full 物理单元的 Jaccard 重叠矩阵；双方均空时为 N/A |
| `policy_metrics.csv` | 成功/失败候选、准确上下文数、root/full 单元、独有贡献、suffix 成本 |
| `candidate_outcomes.csv` | 每个候选的物理坐标、身份和各策略结果，便于重新排图 |
| `x_slices.csv` | 切片覆盖及观测跨度的原始数值 |
| `resolution_sensitivity.csv` | 所有物理维度分辨率同时取 0.5×、1×、2× 的覆盖统计 |
| `legacy_family_labels.json` | 原 π0/π1/π2 family 的派生闭合标签，保持原成员范围 |
| `summary.json` | 比较语义、策略身份、标签来源和统计结果 |

所有图同时输出 **300 dpi PNG、PDF 和 SVG**。论文优先使用 PDF/SVG。图中单元数表示“该物理单元里至少一个采样上下文成功”，不是该单元所有状态都能成功。独有准确上下文与独有物理单元分别统计。

## 看完这次结果后做什么

先看四策略是否存在独有物理贡献，以及哪些 x/phase 切片覆盖稀薄。再以已有策略开展单 proposer 与多 proposer 的同预算探索对照，建立新增到达、旧到达的新接续及成本的累计账本。根据这些缺口锁定互补训练分布；不要求新 Actor 覆盖全部 Tube。预测器不阻断这条主线。
