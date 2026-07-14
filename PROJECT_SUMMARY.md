# OrangeBike DVGC 项目总结（Clean Verifiable Edition）

## 1. 项目目标

本项目训练单轨双轮机器人从自然起点完成：

`Approach → Takeoff → Flight → Landing → Recovery`

项目解决的不是单纯“让车离地”，而是自然起点 PPO 因早期失稳而无法访问后续阶段的 **Survival Bottleneck**。正式方法采用：

1. Event-Anchored Landing-first Backward Bootstrap；
2. 冻结共享 Actor 后的递归 Chain 与端到端 Final Recovery 双分支认证；
3. Beta 后验划分 Safe、Boundary、Dead、Unknown；
4. Final-safe/Boundary Tube-guided RSI；
5. 下游已认证状态 rehearsal，抑制共享 Actor 灾难性遗忘；
6. 独立 seed namespace 的 Tube audit；
7. 最终 natural-start 完整跳跃验证。

参考轨迹只用于候选状态范围和物理诊断，不进入逐点轨迹跟踪奖励，也不替代 Final Recovery 标签。

## 2. 本次输入审计

### 2.1 参考轨迹 `data/reference_jump.csv`

- 821 行；
- 中位采样间隔 0.002 s；
- 总时长 1.64 s；
- 姿态角单位为度；
- 最大 `pos_z = 0.55155 m`；
- 最大 `vel_z = 2.53645 m/s`；
- hip 约为 `[-1.3163, -0.9906] rad`；
- knee 约为 `[1.6625, 2.5771] rad`；
- 自动事件锚点：Approach 结束 0.228 s、Takeoff 结束 0.260 s、apex 0.442 s、Landing 开始 0.660 s、Recovery 开始 0.760 s；
- 后段 roll 失稳已从“成功恢复 envelope”中排除。

需要注意：轨迹 knee 峰值略高于 XML 的 2.5 rad，属于约 0.077 rad 的仿真/记录超调。本版本不会扩大关节限位，而是在候选构造时裁剪至 XML 合法范围。

### 2.2 模型 `assets/orange_bike_4kg_horizontal.xml`

- 障碍物前沿 `x=3.6 m`，后沿 `x=7.6 m`，顶面 `z=0.16 m`；
- 动作固定为 `[steer, rear-wheel drive, hip, knee]`；
- hip 范围 `[-1.3, 0.5] rad`；
- knee 范围 `[-1.5, 2.5] rad`；
- 上臂末端负载为 `4.0 kg`；
- hip/knee 位置执行器 `kp=200, kv=5, force ±50 N·m`；
- 工程只使用 `assets/orange_bike_4kg_horizontal.xml`；不生成 runtime XML，不删除视觉 mesh，不转换轮胎碰撞体，也不改写任何 XML 节点；
- MuJoCo 通过 `MjModel.from_xml_path` 按原 XML 的 `meshdir="meshes"` 读取你已有的 STL；
- 原 ellipsoid 轮胎碰撞体保持不变，默认后端恢复为 `MJX-Warp + IMU/event contact estimation`。

## 3. 旧工程的主要问题与本次修复

| 问题 | Clean 版本处理 |
|---|---|
| Actor 直接读取 oracle phase | Actor 只接收独立事件滤波得到的 soft phase probabilities；oracle phase 仅用于标签和切片。 |
| knee legacy 映射正半轴无效 | 统一为 `incremental_positive_flexion.v2`：`q_target=clip(q_current-action_3×0.20,qmin,qmax)`；正动作屈膝、负动作伸展、零动作保持。 |
| 训练/评估/认证动作映射可能不同 | 所有入口只读取统一配置；policy manifest 保存 action mapping、XML/config/bank 哈希。 |
| Takeoff 进入 Flight 后奖励消失 | Takeoff local reward 只负责地面 launch；进入 Flight 后切换到统一 continuation、Chain 和 Final Recovery reward。 |
| Chain 只读最后一步瞬时值 | 新增 `chain_ever` 锁存；认证只读取锁存值。 |
| certification 未显式加载下游 bank | Flight/Takeoff/Approach 认证强制要求 `--downstream-bank`。 |
| 16 维混合单位 raw Euclidean match | 使用下游 Final-safe bank 的 median/MAD 标准化特征距离。 |
| Chain 和 Final 共用一套统计 | bank 为 Chain、Final 分别保存 successes/failures/Beta posterior/label。 |
| Boundary 与证据不足混淆 | posterior 宽度未收敛时标记 Unknown；只有宽度满足阈值才是 Boundary。 |
| timeout 等同于 Failure | 环境分别保存 `terminated` 与 `truncated`；认证可将有限时域未恢复记为 Final 失败，但报告中保持物理失败/超时分离。 |
| 共享 Actor 逐阶段训练遗忘下游能力 | 训练时将下游 safe/boundary 状态作为 training-only rehearsal reset 混入；永不进入当前阶段认证。 |
| branch 仅有 action noise | certification 建立质量、摩擦、执行器力限的实际模型变体，再叠加未来 action noise。 |
| policy/bank/version 混用 | Policy bundle 和每条 Tube 记录均保存 policy、estimator、tube 版本；不同版本的 Bernoulli 统计不合并。 |
| 固定周期重标注 | `relabel_plan.py` 根据 KL、固定评估下降、标签年龄、Beta 宽度、模型不确定性和使用率生成固定预算刷新集合。 |

## 4. Clean 工程范围

这是一个 **完整可验证的 DVGC-Physical 主版本**，同时保留 Physical-Belief snapshot 接口：

- 保存 observation history、last action、phase probabilities、contact probabilities、phase progress、confidence 和 delay buffer；
- Viability 使用 physical + belief 特征；
- 当前 phase estimator 是可部署事件滤波器，不是已经训练好的 GRU。

因此，本版本可以验证 DVGC 的核心因果链：Backward Bootstrap、双目标认证、Tube-guided RSI、重标注和 natural-start 成功率；但在论文中不能把当前事件滤波器描述成已经完成的 Streaming GRU Estimator。要完成 v23 的“学习式 Estimator”扩展，还需增加序列数据集、GRU 训练和校准实验。

## 5. 唯一正式流水线

### Stage A：Landing → Recovery

- 构建 Landing candidates；
- 训练共享 Actor；
- 冻结策略；
- Chain=Recovery，Final=Recovery；
- 建立 Landing Tube 并独立审计。

### Stage B：Flight → certified Landing entry → Recovery

- Flight Chain 要求进入 Landing Final-safe entry；
- rollout 不在 Chain 时终止，继续运行到 Recovery/Failure/horizon；
- 分别保存 Chain 和 Final。

### Stage C：Takeoff ground → certified Flight entry → Recovery

- 正式 candidate 必须在地面，`had_airborne=0`、`airborne_count=0`；
- velocity-seeded 状态只能是 `training_only=True`；
- 地面阶段使用 launch-quality reward；进入 Flight 后使用统一 continuation reward；
- 认证只使用非 training-only main candidates。

### Stage D：Approach → certified Takeoff entry → Recovery

- 逐步提高自然起点比例；
- 通过后才能做 natural-start 完整跳跃评估。

## 6. Gate 标准

每个阶段必须同时报告：

- Chain posterior；
- Final Recovery posterior；
- Safe/Boundary/Dead/Unknown 数量；
- 独立 Tube precision；
- recoverable recall；
- candidate-mass coverage；
- Brier/ECE calibration；
- Chain 成功但 Final 失败的 false-progress rate；
- Final 成功但未经过认证入口的 missed-success rate；
- physical failure rate 与 timeout rate。

进入下一阶段的最低条件不是“训练 return 变好”或“有 checkpoint”，而是：

1. 当前阶段存在非零 Chain-safe；
2. 当前阶段存在非零 Final-safe；
3. 独立 audit 没有显示 Tube precision 崩溃；
4. 固定下游评估没有显著退化。

## 7. 目录职责

```text
DVGC_clean_project/
├── PROJECT_SUMMARY.md
├── README.md
├── configs/default.json
├── assets/
│   └── orange_bike_4kg_horizontal.xml
├── data/reference_jump.csv
├── dvgc/
│   ├── config.py          # 唯一配置源
│   ├── model.py           # 原始 XML 只读审计
│   ├── action_mapping.py  # 唯一 knee/动作映射
│   ├── reference.py       # 参考轨迹锚点/envelope
│   ├── bank.py            # Physical-Belief snapshot + 双 Beta
│   ├── env.py             # 统一 MJX 环境
│   ├── signals.py         # Takeoff 物理信号
│   ├── rewards.py         # 正式 reward
│   ├── runtime.py         # PPO/推理适配
│   ├── policy.py          # 带 provenance 的 policy bundle
│   ├── rollout.py         # 冻结策略 rollout
│   └── viability.py       # Physical-Belief ensemble
├── cli/
│   ├── prepare_project.py
│   ├── build_candidates.py
│   ├── train.py
│   ├── certify.py
│   ├── audit.py
│   ├── fit_viability.py
│   ├── relabel_plan.py
│   └── evaluate.py
├── scripts/run_backward_bootstrap.sh
├── tests/
└── docs/
```

## 8. 当前验证状态

本交付已完成：

- 全部 Python 文件静态编译；
- 参考轨迹解析和事件锚点审计；
- 原始 XML 结构、哈希、关节、执行器、mesh 引用和障碍尺寸只读审计；
- knee 增量动作映射正/负方向与限位单元测试；
- bank 双 posterior 和 Unknown/Boundary 逻辑单元测试；
- 关键修复的源码合同测试；
- clean project 中不再包含历次 V2/V3/V4/V5 实验脚本。

当前容器没有安装 MuJoCo Playground/Brax/Flax/Optax，因此无法在这里执行 GPU PPO、MJX 动态 rollout 和最终 branch certification。`test_optional_runtime.py` 会在目标训练环境安装依赖后自动运行动态 smoke test。代码的动态有效性仍必须在你的原训练服务器上通过 `pytest`、短 rollout 和各阶段 audit 证明；本总结不会把“静态编译通过”冒充“策略已经成功完成跳跃”。
