# JIT 代码实施与生产验收说明 — 2026-09-05

本次在 `bfc22f2` 代码、`b7af5a9` 论文方向文档的基础上实施。目标是“多个冻结探针发现经验跳跃支持”，不要求新 Actor 覆盖累计 Tube。以下区分代码验收与真实实验验收。

## 已修改及验收边界

| 项目 | 本次实现 | 当前验收 |
| --- | --- | --- |
| 混合终点选策 | 分析器与 selector 都检查 core baseline/candidate 和 boundary 终点，缺失或混用直接拒绝 | 真实历史 π3 报告拒绝用例通过；不修改原报告 |
| evaluator 缓存 | 核对 catalog 文件/协议、Actor/payload/config/freeze、proposer、seed、horizon；逐行核对身份与上下文；分片复用核对范围与索引 | CPU 正反例通过 |
| 标签合并 | 检查每行 Actor、payload、终点、二元结果、成本；临时目录完整验证后发布；不得把 completed 目录当 incomplete 归档 | 错误合并保留旧结果用例通过 |
| 标签文件身份 | 新 serial/shard/merge 输出记录 labels 文件哈希；读取有哈希的文件必须匹配 | 旧文件不补造历史哈希；旧文件仍需协议和逐行核对 |
| 预测器 | 锁分 self-hash、catalog/protocol 绑定；新锁显式绑定目标家族；AP 按相同分数的组计算；单类别返回不可用指标；拟合资格不使用 ACCEPTANCE 类别数量 | CPU 通过；旧三份锁分不修改，缺少事前目标身份会明确标记 |
| 公共 warm-start | 公共 formal 入口识别 fresh/warm 配置并注入 Actor/normalizer 恢复参数；`restore_value_fn=False`；历史 CLI 不再 monkey-patch loader | 公共调用链的模拟 trainer 验证通过；未运行真实 PPO |
| 前向证据采集 | 新模式不按已有训练支持排除；保存未归一化的真实 reached snapshot；用包含 FIFO/事件/时钟的快照身份去重 | 表示身份单测通过；完整采集 GPU 尚未验收 |
| 版本化探针 | 新 bank 锁定 proposer/evaluator 角色；每个 proposer 独立采集，多个 catalog × 多个 evaluator 构成计划 | 2 proposer × 4 evaluator 的 CPU 计划用例通过 |
| 独立进程标签与见证索引 | 每个 shard 一个子进程；锁定计划；断点恢复；逐 evaluator 严格合并；保留每条策略结果与独有上下文贡献 | 用模拟 GPU 子进程输出、真实合并器的流程测试通过 |
| 成本 | 每次 suffix 尝试先预留上限；失败/未知尝试保留；无可信计数按预留上限收费；单输出禁止并发 supervisor | 重试预算耗尽拒绝用例通过；不是端到端总账 |

## 新入口怎样使用

代码：`JIT/src/jit_dvgc/probe_bank.py`；CLI：`JIT/cli/probe_bank.py`。

从仓库根目录执行，生产解释器为 `/home/qy/mujoco_playground/.venv/bin/python`，设置 `PYTHONPATH=JIT/src`。先准备真实工件和下面的配置，不能把示例占位符当身份。

1. `lock --spec BANK_SPEC.json --output BANK.json`：冻结 bank 内容。
2. 对每个 proposer/role 使用 `acquire --spec ACQUISITION_SPEC.json --output NEW_ACQUISITION_DIR`：从固定起点进行真实前向采集。
3. `prepare-labels --bank BANK.json --catalog CATALOG_A.json --catalog CATALOG_B.json --role train --seed SEED --output LABEL_PLAN.json`：锁定全部 catalog/evaluator/shard 组合及最大成本。
4. `run-labels --plan LABEL_PLAN.json --output NEW_LABEL_RUN_DIR`：逐个启动新进程；成功后产生 `witness_index.json`。

`BANK_SPEC.json` 必填：

```json
{
  "version": "pilot-001",
  "task": {
    "xml_sha256": "<真实 64 位 SHA256>",
    "start_contract_sha256": "<完整起点声明文件的 SHA256>",
    "centerline_sha256": "<固定 pi0 centerline 的内部身份>",
    "resolution_sha256": "<现有 resolution_contract() 的身份>",
    "success_criterion": "first_valid_landing",
    "continuation_start_semantics": "fresh_continuation_v1"
  },
  "max_ticks": 400,
  "label_interaction_budget": 16000,
  "max_candidates_per_process": 16,
  "members": [
    {"frozen_policy": "/absolute/path/to/pi0/frozen_unified_policy.json", "roles": ["proposer", "evaluator"]},
    {"frozen_policy": "/absolute/path/to/pi1/frozen_unified_policy.json", "roles": ["proposer", "evaluator"]}
  ]
}
```

上面的 16 和 16000 仅展示配置格式，不是已经锁定的论文实验预算；horizon 必须与冻结策略配置一致。正式配置要用生产 host 的真实文件与事前预算。

`ACQUISITION_SPEC.json` 必填字段：`bank`、`proposer`、`role`、`start_contract`、`nominal_centerline`、`anchors`、`seed`、`strengths`、`action_names`、`signs`、`lookbacks_m`、`max_forward_ticks`、`interaction_ceiling`。其中 `anchors` 是 JSON 数组，复用现有 role plan 的声明 anchor 行，包含 phase、x_target_m、proposal_family_index、parent_group_id、state_sha256 等字段；不得随意生成跨角色共享祖先。采集上界在 rollout 前核对。起点声明文件需写清完整状态及随机键/时钟规则；当前检查其身份，尚未证明所有字段与实际 reset 等价。

标签计划只接受该 bank 新采集模式的 catalog，核对 role、centerline、起点声明、到达 provenance、实际快照和 proposer。它不会将历史 wide/expanded catalog 重命名为新实验。新 bank 使用独立入口，旧 family CLI 的 π0/π1/π2 限制继续保留。

恢复 `run-labels` 时使用同一个 plan 和 output。完成分片重新验证后复用；失败尝试保留在独立 attempt 目录。预算不够时停止；不能删除失败目录或更改计划来让重试免费。不同计划的标签预算是分别计费的，跨计划、采集和训练总账仍需补齐。

## 为什么现在还不能宣称包线已经跑通

`witness_index.json` 是完成的 suffix 观察索引，包含 catalog 来源 ID、完整存储上下文身份、每条 evaluator 结果和独有上下文贡献。它刻意记录：

- `snapshot_replay_equivalence_verified=false`；
- `formal_envelope_claim_authorized=false`；
- `physical_cell_count=null`；
- `end_to_end_interaction_total=null`。

原因是当前 `fresh_unified_continuation_start` 会重置 episode/phase 计数。新前向采集保留真实 reached snapshot，但 suffix 仍走已声明的 fresh continuation 语义。必须实测该变化是否影响失败/超时/事件判定，再决定采用剩余全任务时间还是明确的局部续接任务。**快照哈希相同不是动力学重放等价证明。** 接触求解器状态、观测 FIFO 和相位切换时刻也要检验。

索引也不是训练 Tube，不直接授权 PPO。新版本保留旧 bank/索引文件，但尚未实现跨版本累计物理单元账本、全局角色隔离、自动互补训练和探针准入。不要手改上述 false/null 字段冒充验收完成。

## 生产环境的下一步顺序

1. **物化资产**：真实 Round1 π0、π1/π2/π3 checkpoint、config、Tube、centerline 和少量快照；核对固定 XML 与 x=2.5 完整起点。保留旧 manifest；路径映射另建。
2. **小批物理验收**：从 upstream、apex 邻域、downstream 选实际 reached states，比较不间断接续与 snapshot 恢复接续的首步观测/动作、事件、终点和时间语义；相同 catalog/seed 做 serial 与 shard 逐行等价检查。记录数值容差与选择理由。未通过先修复，不扩大训练。
3. **补齐旧 expanded 标签**：保留原 π0 proposer、π0/π1/π2 evaluator 和全部锁定配置；按小批新进程补齐 OOM 中断项。旧 completed 输出不得覆盖，旧分数不重新锁定。预测器审计只作旁路。
4. **已有探针 A/B pilot**：先比较 π0 与固定现有策略集合，在同一起点、范围、物理单元分辨率和总交互预算下评估。π3 可作为技术兼容的候选探针；其独有贡献必须从逐状态结果得到。补上跨 proposer/role 隔离、物理覆盖与采集/重试总账。
5. **再训练互补探针**：冻结 TRAIN 支持选取、起点/RSI 混合、initializer、奖励、步数、种子、预算及停止规则。新 reset 语义尚未实现；`natural10` 不是自动等于 x=2.5 起点。不要直接启动旧 π4 selected-policy 工作流。

论文此时最缺的是 **可重放的前后缀见证 + 公平预算下的发现效率曲线 + 互补探针训练机制与消融**，不是更高的单 Actor 全 Tube 成功率。

## 本次验证

针对 bank、family/cache、endpoint、predictor、warm-start、shards、selection 和 causal contract 的 CPU 回归。子进程流程测试模拟 GPU 产生标签，使用真实规划/预算/合并逻辑；公共训练入口测试在 trainer 开始前停止。运行记录见 [CURRENT_STATUS](CURRENT_STATUS.md)。没有运行生产 GPU、完整 PPO、旧缺失标签或最终 TEST/JCE/JEL。

后续同一项目、既定工作分支上的常规代码提交与推送已获用户授权；无需重复询问。
