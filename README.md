# DVGC / JIT：经验跳跃包线探索

当前研究与实现位于 **JIT/**，维护日期 **2026-09-12**，工作分支 `agent/two-phase-soft-tube`。

我们研究固定自行车摆机器人、固定近地起点和首次有效落地终点下，如何利用互补冻结策略与有界残差探索，发现有真实到达和成功续接见证的经验跳跃支持。待学习候选与已见证包线分开保存；新策略训练后重新评价旧候选，逐轮积累证据。

- [项目状态与下一步](PROJECT.md)
- [当前结果与完整证据入口](JIT/docs/CURRENT_STATUS.md)
- [论文大纲](JIT/docs/JIT_PAPER_OUTLINE.md) · [详细中文草稿](JIT/docs/paper/JIT_PAPER_DRAFT.md) · [论文 PDF](JIT/docs/paper/JIT_PAPER_DRAFT.pdf)
- [控制框图与全部图件](JIT/docs/paper/README.md) · [方法契约](JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md)
- [代码组织](JIT/docs/CODE_ORGANIZATION.md) · [验证范围](JIT/docs/VERIFICATION.md)

历史 `all_proposers_v1` 已完成 π5、π6 两轮各 128k 训练，历史口径累计 root cells 为 1,689 → 4,629 → 9,296。最新残差探索与延迟评价 pilot 的两轮 128k 训练也已完成；两臂旧无见证候选新增成功见证均为 0。执行链已跑通，残差探索的效率优势与长期收益尚未成立。

所有结果目前属于 TRAIN/开发证据，不代表完整物理可达集或单策略全包线控制能力。最终 TEST/JCE/JEL 未开启。不要重复运行完成的 all-proposer campaign。

实际代码、训练、证据均属于 DVGC/JIT；其他项目的日志、模型和结果不得混入。根目录旧 `dvgc/` 等基础设施保留，当前任务入口遵循 [AGENTS.md](AGENTS.md)。
