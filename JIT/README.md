# JIT：从成功跳跃到经验跳跃包线

当前维护日期：2026-09-12。完整项目状态见 [PROJECT](../PROJECT.md)，证据见 [CURRENT_STATUS](docs/CURRENT_STATUS.md)。

当前链条：冻结跳跃 π → 学习有界残差探索 → 按当前 π 的真实到达新颖性奖励 → 保存 witnessed/pending/unknown 候选 → 从完整快照训练新 π → 延迟续接重评 → 累计已见证经验包线。

历史 π5、π6 两轮 all-proposer 训练已完成。最新两轮残差探索外循环也完成了两次 128k PPO；旧无见证候选新增成功见证为 0。随机条件探索臂最终 308 个已见证 root cells，学习残差臂 289；这不是独立全流程基线比较，不能宣称学习残差已经优于随机扰动。

## 阅读入口

- [当前交接](docs/CODEX_HANDOFF_20260911.md) · [方法协议](docs/ENVELOPE_ITERATION_PROTOCOL.md)
- [训练路线](docs/JIT_TRAINING_ROADMAP.md) · [代码组织](docs/CODE_ORGANIZATION.md) · [验证](docs/VERIFICATION.md)
- [论文大纲](docs/JIT_PAPER_OUTLINE.md) · [详细草稿](docs/paper/JIT_PAPER_DRAFT.md) · [PDF](docs/paper/JIT_PAPER_DRAFT.pdf)
- [科研控制框图、结果图和重绘数据](docs/paper/README.md)

核心代码在 `src/jit_dvgc/`，薄入口在 `cli/`，声明配置在 `configs/`。使用 `/home/qy/mujoco_playground/.venv/bin/python`，从仓库根目录设置 `PYTHONPATH=JIT/src`。原始工件在 `runs/`，禁止重写历史失败和旧协议结果。

106D privileged 历史输入只属于当前仿真探索器；基础跳跃 Actor 是 76D。当前输出是四通道残差，基础策略冻结，未实现 critic 质量惩罚或扩散探索。固定近地起点、首次有效落地、完整上下文和数据角色见 [AGENTS](AGENTS.md)。
