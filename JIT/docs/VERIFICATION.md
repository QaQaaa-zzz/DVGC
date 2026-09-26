# JIT 当前验证范围

更新：2026-09-12。本页替代旧π1 paired core gate作为所有后继工作的唯一入口。历史gate及其失败记录仍保留原有科学含义；当前经验见证不要求单Actor覆盖全部Tube。

## 证据分层

| 层次 | 当前已核验 | 不等价于 |
| --- | --- | --- |
| 源码/CPU行为 | 动作冻结与限幅、信用按cell分配、unknown不变negative、候选池追加、pending支持、预算与复用协议有测试 | GPU数值与实际探索收益 |
| GPU工程 | corrected残差PPO动作→真实prefix/context→suffix→更新链跑通；批量接续容量测量 | 最终科学比较或实车能力 |
| TRAIN开发 | 历史all-proposer、checkpoint pilot和两轮延迟pilot完成 | 独立重复/最终holdout |
| 最新机制结果 | 两次128k后旧无见证→成功仍为0 | 延迟学习已有效、包线已收敛 |
| 稿件与维护 | 以本次paper数据哈希、图件和文档链接验证记录为准 | 重新运行历史仿真 |

## 修改相关验证

从仓库根目录使用生产Python；仅对相关模块运行必要CPU测试，不以文档编辑触发全量GPU测试。

```bash
PYTHONPATH=JIT/src JAX_PLATFORMS=cpu   /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/<相关测试文件> -q
```

全仓库静态/CPU preflight入口为 `JIT/scripts/local_preflight.sh`；GPU测试显式配置`JIT_RUN_GPU_TESTS=1`，须有相应工程预算。CPU通过不能写成GPU通过。

对动作/观测变更检查76D/106D顺序、history/FIFO和checkpoint绑定；对奖励变更检查candidate tick、per-π baseline、重复cell共享、unknown资格及旧reward版本；对pool检查完整identity与追加标签；对reset检查完整状态、phase/group和pending质量，而不是只检查配置数字；对运行控制检查actual/reserved、旧artifact复用和失败保留。

## 当前科学核对清单

- 两轮完成由reuse首轮记录与completion次轮记录共同提供；原始error不改。
- 最新128,407补完成本与两轮348,371关联成本分别核对，不重复计继承19,200。
- 标签冲突为unknown；旧738-positive及新735-witness视图分开；历史4条forward冲突没有新增substep裁决。
- 相同物理单元不等于相同context；fresh_continuation的计时变换必须公开，已接受数值差异不得写成精确replay通过。
- 16384容量测量及38→39tick差异必须随性能解释，不能无条件替换锁定serial结果。
- finalTEST/JCE/JEL未打开；没有从并行环境数虚构独立seed数量。

## 文档/论文交付验证

运行 `JIT/docs/paper/build_figures.py` 从已保存CSV重绘；`build_manuscript.py`生成离线HTML/PDF；`verify_artifacts.py`核验数据哈希、源文件、图件和关键数值。构建依赖与命令见[论文索引](paper/README.md)。图件需实际查看，PDF需确认中文/公式与页面不裁切。构建程序不导入仿真环境，不执行env.step或PPO。

本轮具体检查结果写入 `JIT/docs/paper/validation.json`。没有执行的检查不写passed；过去的测试数不冒充本轮测试。
