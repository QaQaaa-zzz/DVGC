# JIT 关键结果包 · 2026-10-10

![关键模型 XY / XZ 轨迹](all_models_hard_initial1000_20261010/key_models_xy_xz.png)

## 先看这四项

- [连续训练总览](continuous_key_metrics_20261010/INDEX.md)：R1–R73 连续曲线；原 200 轮任务完成 69 轮，R74 因 ptxas 编译退出 139 停止。本次未重启训练。
- [15 个模型测试](all_models_hard_initial1000_20261010/INDEX.md)：每模型 1000 回合。旧 Phase U 成功率 65.8%，桥接 R71 为 30.9%；测试模型是 R71，并非训练最后完整的 R73。
- [π0 对照](bridge_pi0_hard_initial1000_20261010/INDEX.md) 与 [成功范围量化](success_envelope_quantification_20261010/INDEX.md)：独立补充对照 π0 为 65.1%；R71 的成功轨迹典型宽度指数 109.6（π0=100），不能解释为可控体积增大或整体能力增强。
- [研究台账快照](ledger/PROJECT_STATE.md)：保留历史、负结果及研究边界；这是归档副本，日后不会自动更新。

训练回报近期上升，固定 DEV 恢复率却下降或波动；G 去噪 MSE 下降不证明物理恢复改善。[训练总览图](continuous_key_metrics_20261010/overview.png) 保留这一区别。

## 内容与范围

只包含关键 PNG、汇总指标、模型身份和协议、台账与 R74 错误证据。metrics.csv 是 73 个训练轮次的聚合指标，并非逐回合数据。没有逐回合 CSV、轨迹 NPZ、逐步日志、模型权重或视频。源运行目录完全保留；本次只整理已有结果，没有新仿真或训练。

子报告中的外部路径和历史 localhost TensorBoard 地址需要原机器；包内图表和汇总可离线阅读。原始 status 中 analysis_pending 是原任务记录，后续分析见 result_audit 和报告，不改写原始凭证。

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) 记录原始来源；[SHA256SUMS](SHA256SUMS) 校验包内文件。原始完整研究台账中的历史外链未打包，不代表附带全部历史实验。
