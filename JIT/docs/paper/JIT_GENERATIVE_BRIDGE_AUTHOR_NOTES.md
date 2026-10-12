# JIT 英文期刊稿：作者修改入口

本文件是英文稿的配套说明，不是论文正文，也不替代共享研究台账。2026-10-12，根据用户对旧稿“像报告”的反馈，从研究问题和方法重写，未启动训练、仿真或最终 TEST。

## 阅读与修改

- [英文 PDF](JIT_GENERATIVE_BRIDGE_MANUSCRIPT_EN.pdf)：固定阅读版。
- [英文 LaTeX](JIT_GENERATIVE_BRIDGE_MANUSCRIPT_EN.tex)：可编辑数学源稿；图片使用仓库相对路径。
- [英文 Word](JIT_GENERATIVE_BRIDGE_MANUSCRIPT_EN.docx)：正文、表格、Office Math 公式可编辑。
- [Markdown 作者源](JIT_GENERATIVE_BRIDGE_MANUSCRIPT_EN.md)：本文生成入口。
- 下载目录中的 JIT_English_LaTeX_20261012.zip 是独立工程：上传 Overleaf 后选 XeLaTeX，主文件 main.tex；或在工程目录执行 xelatex main.tex 两次。也支持 tectonic main.tex。
- 导出的 LaTeX 有真实引用键和公式标签，参考文献内嵌，因此直接编译不需要 BibTeX。另附已核验 .bib，选定期刊后可转为其要求的文献样式。
- 若直接编辑 LaTeX，请将其作为该修改分支的作者源保存；重新从 Markdown 构建会重新生成同名 .tex/.pdf/.docx。

## 这次怎样改写

采用三个写作 skill：academic-paper 的 revision 流程、research-paper-writing 的方法/段落结构，以及 anti-defensive-writing 的主线审读。将源代码取证和引用审读分开交给协作者，再独立核查结果口径。旧中文版本保留，英文版成为本次阅读修改入口。

RAS附件为 Huo等的 Hierarchical residual reinforcement learning based path tracking control method for unmanned bicycle（RAS 190, 104996, 2025）；用户提到的 ridelines 经确认是 LineRides。借鉴两篇“控制难点→形式化→模块设计→对应科学问题实验”的结构，不沿用其文字、图或结果。相关工作准确区分路径跟踪、空间引导特技与JIT恢复学习。

英文正文采用：引言→相关工作→问题定义→E/G/学生/反馈方法→实验设计→已有结果及待填主比较→讨论→结论。运行编号、工程失败过程、TensorBoard和费用台账不占据方法叙事。数学附录保留动作映射、完整分段奖励、执行后状态索引、PPO实现约定及扩散/探索细节。原框图完整保留；图中外力泛化明确作为待检验目标。

这是一份可继续补实验的期刊结构草稿，不是已经具备完整投稿证据的定稿。当前数据不支持完整JIT优于同预算PPO、扩散优于Noise或当前最终学生已完成零目标训练迁移，正文没有写成已证实结论。

## 图表怎样补齐

| 稿件位置 | 已有内容 | 后续应放的实测结果 |
|---|---|---|
| 图1 | 用户原E→G→π及反馈框图 | 保留；若后续更改方法，再明确方法版本 |
| 表2 / §6.1 | D1全部16 TRAIN、7 solver根资格矩阵；自访12/24见证 | 形式实验按seed/round报告query分母、合格率、失败与歧义 |
| 图2 / 表3 / 图3 | T1全部8根实际XY、三次标签、两臂32k数据；历史π0根筛选脚注 | 主方法通过机制门后，保留相同的教师→独立学生对照 |
| 表4 / 图4 | PENDING主比较和保持性空图 | M0—M4×3训练seed：宏平均稳定成功、逐题gain/loss、B/D差值及区间、实际费用 |
| 表5 | PENDING转化矩阵 | 每轮查询数、合格教师数、独立稳定吸收数、未吸收和歧义；各列分母不同不能共用 |
| 图5 | 物理交互/GPU时间双横轴空图 | 同5个预声明节点、全部seed；不补画理想化提升曲线 |
| 表6 / 图6 | PENDING同身份双引擎图表 | π0、3个M0、3个M1；各33条件×3重复，共同成功端点；真实XY、起跳/接触/稳定区间、失败类型 |
| 摘要、引言末段、结论 | 真实pilot机制结论，无优势宣称 | 正式主结果出来后改为主比较数值，不把pilot数值移作正式结果 |

现有空图均明确写有 RESULTS PENDING，表格均为 PENDING。没有生成合成成功率、趋势曲线或置信区间。

## 数学与证据检查入口

- [公式逐项源码审计](generative_bridge_assets/method_equation_audit.md)：E网络/掩码、完整损失、G日程与DDIM、资格协议、PPO优势/价值/熵和时钟。
- [参照论文、引用与英文审读](generative_bridge_assets/reference_style_audit.md)：13篇原始来源核验、正文映射、具体修订意见和复核。
- [已核验BibTeX](generative_bridge_assets/references_verified.bib)。
- [动作/奖励数值核查](generative_bridge_assets/formula_numerical_check.json)：256个奖励输入与81个动作组合对原代码；最大绝对误差分别2.85e-5、4.77e-8。仅CPU公式检查，0物理交互、0学习更新。
- [原始T1完整结果](../evidence/retention_next_20261010/T1_paired_ppo_002/INDEX.md)。
- [自访教案与学生结果](../evidence/retention_next_20261010/self_visit_demo/INDEX.md)。
- [D1原矩阵](../evidence/retention_next_20261010/D1_summary.json)。
- [导出推理预检](../evidence/retention_next_20261010/sim2sim_preflight/INDEX.md)。

已修正的易错点包括：DDIM裁剪后重算噪声；E编码器不吃valid位；E的latent熵与学生tanh变换熵不同；PPO实际value系数0.25；每minibatch用当前Critic重算目标；32k的demo系数止于0.1625；D1与self-visit源策略复核协议不同；reward物理信号来自执行后状态；M2/M3只能保留各自适用的反馈路径。

## 稿件边界与作者待确认项

最终TEST未开放。T1未通过机制筛查，正式15次主矩阵未执行，没有从本文写作操作启动新实验。原π0保留，未发布或采用新学生。

作者、单位、CRediT、基金、利益冲突和数据归档地址必须由研究团队补充。文稿包含真实AI辅助写作声明草案。未选择期刊，不代填投稿系统或发送稿件。

后续优先事项：先修复并验证教师到学生的转化缺口；达门后完成冻结的三seed主比较和消融；最后冻结模型并完成一次TEST及同身份迁移。详细实验授权与停止条件继续以共享台账和用户既定计划为准。
