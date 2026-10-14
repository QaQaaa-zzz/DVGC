# JIT100显式继续训练入口

本入口保留100大轮目标、已完成轮次、教师搜索和完整学生learner；不重置Actor/critic/Adam/归一化统计/RNG，不启用未通过GPU验收的快照去重加速。物理环境从已声明支持重置，不是逐帧恢复模拟器。

## 后续继续训练

```bash
cd /home/qy/DVGC/runs/worktrees/jit-teacher-recovery-resume
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/resume_teacher_recovery100.py \
  --previous /home/qy/DVGC/JIT/runs/experiments/jit_teacher_recovery100_CURRENT \
  --active-link /home/qy/DVGC/JIT/runs/experiments/jit_teacher_recovery100_CURRENT \
  --port 6039
```

成功后打印新运行目录、PID、status路径和TensorBoard地址，在独立目录创建声明/恢复校验/费用/日志，再后台启动协调器。CURRENT链接自动更新；下次仍用这一命令，不要反复指向旧历史目录。

只有确认当前运行已经停止时才调用。正在运行或没有可认证的恢复边界时，检查会拒绝启动；不要删除started文件、running journal或检查锁来强行恢复。当前支持已完成32k学生、尚未开始E/G更新的停止边界；其他中断阶段需要相应的状态审计，不允许自动回退重训或丢弃已有更新。

## 重启后数据盘未挂载

若CURRENT或原运行入口无法读取，先挂载已授权数据盘：

```bash
udisksctl mount --block-device /dev/disk/by-uuid/6A7E3DB07E3D75C3 --no-user-interaction
```

不使用格式化、强制修复或删除数据。若盘已挂载无需重复执行。

## 只检查，不启动

在同一命令后追加`--prepare-only`，可生成新的独立声明和审计而不启动GPU或学习。准备后的底层声明用于审计；日常启动仍使用完整resume入口，它会创建新的独立目录并更新CURRENT。不要删除started记录强行重用旧声明。

## 实时查看

TensorBoard：<http://localhost:6039/>。状态文件：`/home/qy/DVGC/JIT/runs/experiments/jit_teacher_recovery100_CURRENT/status.json`。完整启动参数见新目录`launch_receipt.json`；独立奖励/loss manifest、HTTP验证与桌面错误/完成watcher同目录保留。
