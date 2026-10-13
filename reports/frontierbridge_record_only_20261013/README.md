# FrontierBridge P1/P2完整结果

[实际XY/XZ对照](comparison.png)

| 最终attempt1000 | P1普通BC+keep | P2逐次保护 |
|---|---:|---:|
| 接受/尝试 | 1000/1000 | 20/1000 |
| 稳定旧成功保留 | 23/28 | 24/28 |
| 稳定旧成功丢失 | 4 | 2 |
| 稳定旧正的学生混合结果 | 1 | 2 |
| 保持面板相对新增成功 | 7 | 0 |
| 教师验证DEV恢复吸收 | 1/3 | 0/3 |

P1在root4独立恢复成功三次，π0失败三次、教师成功三次；root21/18两策略均未吸收。此为用过的开发快照恢复证据，不是独立最终TEST，也不是全任务泛化成功率。名义π0六重复混合，仍未验证。所有门槛实际计算、只记录不阻断；未自动采用。

P2最后接受在attempt36，随后964次连续拒绝。其100与1000 checkpoint Actor及Adam完全相同，但B_14和D_14重复标签从[1,1,1]变为[1,1,0]/[1,0,1]，证实相同模型的数值重复仍有波动。不能把P2更少稳定丢失直接解释为已解决保持。

P2拒绝检查主要触及old_recovery/p99/channel0（转向）的固定行为预算。有限保护库动作约束有效执行，但未保证物理保持，且当前预算/更新方向阻断吸收。仅接受20次更新，因此还不能排除“少训练”解释。下一步优先接受更新数匹配对照与约束下更新方向，而不是追加相同训练或直接全面放宽预算；此为建议，尚未运行。

完整流水线completed：全部129物理child完成；含历史预检总charge849189/1000000控制步，包含失败及padding。P1/P2各实际1000attempt、同一数据且全部1000次mini-batch indices逐项一致，normalizer/critic冻结、Actor/Adam文件SHA与计数核验。BC/loss与BCreward实际HTTP读到step1000；reward为负监督目标。完整最终参数留服务器。

数据：checkpoint_metrics.csv分更新双计数/保持/吸收；group_metrics.csv分组；condition_labels.csv覆盖56工况与全部声明checkpoint；updates.csv全部2000尝试；trajectories.npz保存987真实轨迹，xyz世界米/time实际秒，offsets切片，model/case_id/repeat/label对应每条。P0保持49条件为历史六次重复、七solver为三次；学生均三次。数值重复不是独立训练种子。所有轨迹真实终点截止，无额外仿真。
