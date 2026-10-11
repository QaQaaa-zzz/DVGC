# T1_002：GPU工程通过，两臂训练完成，固定评价进行中

用户明确授权“修复并且继续”后新建运行，原001失败记录保留。源码29174b3；同pilot1010257101与同π₀/fresh Adam，无BC初始化。四池真实terminal reset检查通过；工程两臂各3200转移／8优化更新丢弃，正式两臂各32000／80更新，全部learner验收通过。

[训练及HTTP标量收据](training_receipt.json)；[实时运行状态](/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/T1_paired_ppo_002/status.json)；[TensorBoard](http://localhost:6033)。两臂分别加载实际reward与total_loss。训练阶段通知已送达，流水线完成／错误另行通知。

当前评价未完成，t1_passed=PENDING，无新策略采用，不宣称恢复能力或保持性改善。原001失败200计入prior1123052，本阶段≤700200，最坏D2累计1823252/2000000。TRAIN/DEV固定三次、solver固定首评；完整结果由流水线输出mechanism_decision.json及report，不自动启动正式矩阵或TEST。

当前T1评价正在生成，完整XY与逐根结果尚待齐全；不能称论文证据交付完整。[既有T0重点根XY](../T0_saved_diagnostics_002/xy_focus.png)。
