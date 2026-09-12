# DVGC / JIT 当前实验恢复入口

更新：2026-09-12。当前状态的唯一维护页为 [JIT CURRENT_STATUS](../JIT/docs/CURRENT_STATUS.md)，项目计划为 [PROJECT](../PROJECT.md)。本页不再复制旧π1 gate阶段的过时执行命令。

历史π5/π6 all-proposer两轮已完成。最新per-policy残差探索及pending延迟学习pilot也完成了两次128k PPO；旧无见证候选新增成功见证为0。最新完成文件：`JIT/runs/experiments/delayed_completion_start_20260912/pipeline_status.json`。此前reuse的error记录保留，不能靠修改历史状态伪造连续完成。

当前论文与图件：[论文入口](../JIT/docs/paper/README.md)。后续先分析现有数据的新颖性饱和、phase分布和成本，再设计有界公平对照；不重跑all_proposers_v1，不启动最终TEST。

旧Tube0、π0、π1 repaired gate和lineage记录在Git历史、旧实验工件及JIT历史交接中保留，不能因当前目标变化而宣称当年的gate已经通过。
