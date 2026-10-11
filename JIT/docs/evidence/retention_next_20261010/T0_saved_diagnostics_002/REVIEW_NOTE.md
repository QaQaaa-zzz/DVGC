# T0 review 与首版更正

本目录是T0最终可审版本。`../T0_saved_diagnostics/`保留首版派生结果；其状态分歧使用了动作链标定容差，混合不同物理单位，不用于状态首次分歧结论。首版不覆盖、不冒充最终版本。

本版在执行前声明：qpos、qvel、原始观测及归一化动作的描述性差异均按各分量1e-6报告，并注明分量单位。它与动作链工程门槛独立。root55首次新学生成功/失败复核差异为qvel tick24、原始预动作观测tick25、动作tick26；这描述差异出现顺序，不证明具体数值内核因果。

工程门槛公式始终为`max(1e-6,10*pi0_reconstruction_maxabs)`，没有事后放宽：π0最大误差0.0005594640970230103，门槛0.0055946409702301025。96条保存执行链上独立CPU→原生Brax CPU最大误差2.682209014892578e-7，原生CPU→保存执行动作最大误差0.000741809606552124，独立CPU→保存执行动作最大误差0.000741865485906601。差异主要存在于共同CPU重建与保存执行之间；未精确确定GPU内核或精度机制。

当前安装Brax的normalize直接执行`(data-mean)/std`，未额外添加epsilon；T0独立计算按保存std重建，Swish隐藏层、前四个输出的tanh mode及[-1,1]动作限幅单独实现。没有导入项目环境、调用仿真、执行clip梯度或Adam更新。

10项T0行为测试通过；完整真实CPU运行完成，源文件143项在结束时重新核对hash不变。共同观测总数7291=525+228+6289+249，分别保留数据角色与来源。三个重点分歧PNG及XY总览已经视觉核验。最终gate为`T0_gate.json`，SHA-256：`63e5df4da9f5d1cb2df9b7055310521ce6c404e8bb618ab8f5b4b4c3bbd0079b`。

可重复命令（输出须用新的空目录，不能覆盖本次结果）：

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python JIT/cli/run_retention_t0.py --control /home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_keep_coverage_003 --intervention /home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_self_visit_demo_001 --output <new-empty-report-directory>
```

输出CSV/JSON只有预测误差、梯度统计和来源索引；未复制模型权重、全观测或物理轨迹。模型与原始数据仍在锁定运行目录，保持未修改。
