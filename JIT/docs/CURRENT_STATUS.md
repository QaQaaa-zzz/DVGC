# JIT 当前状态与证据

核验日期：2026-09-12；文档修订前代码 HEAD `830a10f`。本页仅描述 DVGC/JIT。原始记录保持不变。

## 最新完成结果

`runs/experiments/delayed_completion_start_20260912/pipeline_status.json` 为 `completed`。该任务补完两轮 pilot 的第二次跳跃策略训练及延迟重评：PPO 128,000，固定 TRAIN panel 77，random suffix 111，learned suffix 219，总计 **128,407**，墙钟 **221.031 s（含等待）**。

首轮执行及两轮采集来自 `runs/experiments/delayed_exploration_reuse_20260912/queue/stage_0_result/`。其中首轮learned前向原件实际在`runs/experiments/delayed_exploration_20260912/queue/stage_0_result/round_000/learned_residual/arrivals/`，reuse保存锁定引用。该历史任务在第二轮等待阶段停止，旧 `error` 保留。两目录合起来提供已完成两轮的证据；不能从任一旧队列状态独立推断全流程。

| 轮次 | 条件探索臂 | 已见证候选条数 | 累计已见证 root cells | pending 条数 | no-witness→witness | unknown→witness |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | learned_residual | 158 | 150 | 2 | 0 | 0 |
| 0 | fixed_random | 151 | 151 | 6 | 0 | 0 |
| 1 | learned_residual | 308 | 289 | 12 | 0 | 0 |
| 1 | fixed_random | 308 | 308 | 9 | 0 | 0 |

`previously_witnessed` 是候选记录计数；`verified_root_cells` 是物理网格去重计数。每格有准确状态见证不意味着整格连续可行。上表来自两次 `round_metrics.csv`，不是把候选计数相加后的包线体积。

## 关联成本，避免重复计费

| 组成 | 实际 interactions | 说明 |
| --- | ---: | --- |
| 两轮两臂前向 | 76,800 | 每臂每轮19,200，含计费 padding；其中19,200继承自首个停止任务，只计一次 |
| 两次新跳跃策略 PPO | 256,000 | 每次128,000；不含历史 π0–π6 获得成本 |
| 固定 TRAIN panel | 149 | 首轮72 + 第二轮77 |
| 全部相关 suffix | 15,422 | 获取标签及延迟重评 |
| 本 pilot 关联实际合计 | **348,371** | 200,764（reuse 新增）+128,407（补完）+19,200（继承前向） |

预算上限、失败预留和实际执行必须分开。早期失败链和 3,302 步恢复工程测试在各自账本中保留，不假装属于上述无重复主 pilot。更不能把348,371称为从零获得整个系统的总成本。最新补完墙钟不是两轮端到端墙钟；中间等待时间也不是 GPU 训练时间。

## 当前能说明什么

执行已经覆盖：当前 π 内新颖性 ledger、冻结 Actor 的残差 PPO、pending 训练支持、两次新 π warm start、成功见证保留和旧候选延迟重评。

尚未观察到：旧 no-witness/unknown 候选被新 π 转化为成功见证。不能把最终已有308条 witnessed 解释为308条延迟学习收益。学习残差没有在本次条件探索结果中超过随机臂。

两臂前向预算相同并共享基线，但只有 learned pending 驱动共享后继 π；末次 learned 采用确定性动作，random 为随机采样。整个池包含训练采集与末次诊断，因此不是冻结同模式测试。每批最多32个候选，学习臂各批新到达均32，随机臂除首批30外也32；细网格新颖性几乎饱和，奖励尚不足以显示动作质量区别。一个连续训练谱系不支持独立重复统计。

## 历史证据仍有效，但口径分开

| 证据 | 结果 | 不能声称 |
| --- | --- | --- |
| all_proposers_v1 | 1,689→4,629→9,296 root cells；新增1,275,465；π5/π6各128k | 这些增长来自学习残差；所有增长都是新 π 的功劳 |
| checkpoint 32/64/128k | 同4,058探索预算 novelty132/186/132/165（π6/32k/64k/128k） | 32k是通用最优训练长度 |
| 重新初始化256k pilot | 同4,135探索预算 novelty134/122/103/161（π6/128k/192k/256k） | 更多训练必然更优；两张表属于同一连续训练 |
| first-success 离线调度 | 98,093→15,388有用步，假设节省84.31% | 实测墙钟加速84.31% |
| 批量接续工程 | 16,384环境容量测量，峰值14,522MiB、157,227 useful steps/s | 当前完整管线同倍提速或端点时间严格相同 |
| 小型残差正确性 | 基础 Actor不变、零残差回归基础、真实动作/快照链检查 | 科学性能已显著提高 |

历史4条前向 landing/failure冲突已进入事件审查；新existence视图隔离冲突。π6旧完整矩阵738-positive union在新派生审查中为735 witnessed、3 unknown，不能回写为原始实验标签。数值 replay差异按用户此前决定保留，不新增精确 replay门槛，也不称已精确通过。

两次长度pilot均从同一冻结π6、使用同seed9871101分别初始化fresh critic/optimizer；不是独立统计重复，也不是恢复旧optimizer继续训练。

## 完整原始证据与派生稿件

相对本仓库根目录：

- `JIT/runs/campaign/all_proposers_v1/`：历史生产（准确源路径另见论文 evidence.json）。
- `JIT/runs/experiments/checkpoint_discovery_20260911/INDEX.md`
- `JIT/runs/experiments/multicheckpoint_pi6_256k_20260911/exploration/`。
- `JIT/runs/engineering/continuation_parallel_20260911/INDEX.md`
- `JIT/runs/engineering/existence_checkpoint_residual_20260911/INDEX.md`
- `JIT/runs/experiments/delayed_exploration_reuse_20260912/queue/stage_0_result/INDEX.md`
- `JIT/runs/experiments/delayed_completion_start_20260912/INDEX.md`
- [稿件、控制框图和来源哈希](paper/README.md)；[下一步](JIT_TRAINING_ROADMAP.md)。

已核验报告分支 `origin/agent/jit-run-reports` 的最新提交为 `6121da9`（review-bf486f2945），其时间与内容早于本地最新 pilot。本地完整工件是本页最新结果的直接依据。最终 TEST/JCE/JEL 未开启；本次文档制作没有新仿真或 PPO。

## Quality-aware exploration training — 2026-09-12

User approved direct reward-ablation training and required inspectable learning histories/hyperparameters for every network. Implemented opt-in trajectory_quality_v1: full valid nonterminal trajectory novelty independent of candidate caps, once-per-episode physical-failure penalty, and normalized residual-energy cost. Pure weights1/0/0 versus quality1/50/0.01; matched uniform residual control. Base Actor and critic are explicitly frozen; their task reward and before/after-action critic values are telemetry, not exploration reward.

142related CPU tests pass. Real2world/2batch GPU smoke completed3200interactions;2optimizerupdates, complete reward-array summation checked, named losses/hyperparameters/inventory/CSV/PNG/PDF/SVG verified present. Background supervisor1149051 then started the three32batch8world arms,329600combined forward maximum. Current live status:`../runs/experiments/quality_exploration_20260912/execution/status.json`. Each arm preserves per-update optimizer_updates.jsonl, per-batch training_metrics.json/training_process.csv, hyperparameters.json/network_inventory.json, per-transition reward_components NPZ and task reward components, full trajectories/candidates/checkpoints and figures. No suffix labels/new jumping-policy training or claimed exploration gain yet. This stage keeps fixed starts to isolate reward effects; existing-state curriculum is not yet run. Compact evidence:`../review_evidence/quality_exploration_20260912.json`.

## Long quality exploration continuation — 2026-09-12

The three32batch quality-ablation arms completed: novelty/quality/random training new cells10897/10910/10759; successful forward episodes229/230/229 of256; physical failures17/15/15. This is not a demonstrated learned-exploration advantage or witnessed envelope expansion (zero suffix interactions).

User explicitly authorized JIT-only fixes and substantially longer residual training. Resume now restores actor/critic/Adam/normalizer/JAX RNG and per-policy novelty ledger; inherited global batch preserves stateless shuffle identity. Contract drift or corrupt checkpoint is rejected. Fresh full-batch post-update KL, clip fraction and value explained variance are recorded separately from minibatch pre-update metrics. Curves refresh every32 batches.

67 related CPU tests passed. Plan: two-batch real GPU resume smoke, byte-identical initial-state/ledger and process-artifact verification, then1024 additional batches from the original quality batch32 checkpoint. 8worlds, minibatch256, epoch1, Adam3e-5 and quality reward1/50/0.01 remain frozen to test duration. Not a hyperparameter-optimality claim. Maximum combined3,296,000 forward slots; expected main3,280,000 plus smoke9,600, effective samples reported separately. Bounded GPU-process wait and no automatic retries. No external project files inspected; GPU gate uses nvidia-smi only. Exact live state and all artifacts: `../runs/experiments/quality_exploration_long_20260912/INDEX.md`. GPU validation and main completion must be verified from that live execution, not inferred from this launch declaration.

## Long-run restart — 2026-09-13

The original GPU resume smoke completed9600forward slots, but the byte-identical serialization check blocked the long stage. Decoded59leaves (network, optimizer, normalizer, RNG) and ledger match exactly; only dictionary serialization order differed. Canonical verify_restored_checkpoint now checks file hashes and decoded structure/dtype/shape/exact values. Reused existing GPU smoke; corrected verification passed with no new simulation. Original failed queue and script remain immutable. User explicitly requested immediate long training: restarted only the1024batch stage from quality batch32, maximum3,283,200new forward slots; no extra smoke. Live status: `../runs/experiments/quality_exploration_long_20260912/restart_20260913/execution/status.json`; actual training artifacts remain `../runs/experiments/quality_exploration_long_20260912/long/`. No external repository files were read or changed.

## Short-pulse delayed quality loop — 2026-09-13 implementation

User authorized replacing continuous pi6 disturbances with pi0 + three-control-step bounded residuals at the fixed jump start. Residual is removed at handoff; full snapshot/history/event/RNG context is preserved. Real first-valid-landing outcomes define quality, not critic predictions. Same-context bank order pi0..pi6, then any newly trained bank members; stop each candidate at its first witness. Unresolved candidates are pooled into one128k jump-policy training per round using pi0-positive inherited support plus accumulated positives and pending snapshots, then reevaluated by the new policy. One bounded learning attempt per new batch; no claims of physical impossibility.

Implementation plan: (1) canonical pulse runtime for three-step capture and batched, early-stopped suffixes with full traces; (2) CPU orchestrator reusing candidate_support_view/make_config/freeze/lock_probe_bank; (3) resolved delayed-feedback PPO update on unchanged behavior-policy samples, no stale replay; (4) bounded direct launch after focused semantic checks. New modules pulse_exploration.py/pulse_exploration_runtime.py and thin run_pulse_exploration.py CLI. Historical continuous-residual results remain immutable.

Exploration reward:0.25shared credit per new handoff root cell +1successful bank/new-policy continuation or -1after attempted learning still fails. Prefix terminal failures are penalized; unknown/error is excluded or stops the stage. Visited ledger retains failed cells. The projected grid is not a certified continuous region. PPO uses privileged106D/256x3 networks,0.15four-action limits, minibatch128, up to4epochs, Adam3e-5, clip0.2, entropy0.001, value0.5, gradient clip1; full-batch KL threshold0.01 stops remaining epochs. Gamma/lambda1 for the three-step terminal quality return. New jump-policy PPO keeps its separate existing128world configuration and unchanged task reward. Every trainable network retains loss/reward/hyperparameters; per-round PNG/PDF/SVG and replot CSV accompany full prefixes, suffix traces, snapshots and costs.


Short-pulse live evidence: `../runs/experiments/pi0_short_pulse_20260913/INDEX.md`, current `training/status.json`. Real GPU same-start baseline: pi0..pi6 all successful,41/44/45/53/48/43/49steps (323total). First128pulses:76witnessed by pi0,52by pi1, no new-policy training needed;128×3prefix steps and13,476allocated suffix steps. Second128pulses also fully witnessed by pi0/pi1. Both completed explorer updates,6minibatchupdates each; KL0.01436/0.01155 stopped remaining epochs. These are TRAIN results, not independent validation. Continued from both completed rounds after restricting bank order to pi0..pi6 plus only newly learned members; unused historical late-checkpoint controls were excluded. Original failed launch/source-lock records retained; no repeated baseline or completed-round simulation. Focused15semantic/support/routing tests pass plus actual GPU prefix/suffix/update chain. The repair-policy branch is implemented but not yet triggered by these all-success batches.

## Short-pulse conflict recovery — 2026-09-13

The fourth-round pi0 suffix batch had two simultaneous landing/failure flags (lanes3/25), so the previous fail-fast implementation stopped after writing its128lane trace. Three rounds/384candidates had already completed explorer updates. Classifier now records each conflict as unknown, attempts other policies, and assigns bank failure only when every required evaluator explicitly fails. Unknown outcomes receive neither failure penalties nor PPO samples; an all-unknown batch preserves optimizer/network state and advances collection RNG without inventing an update.

Existing fourth-round128prefixes and all pi0 suffix traces were imported with context/actor/trace hashes; no completed prefix/suffix simulation repeated. Remaining candidates continue at pi1. Resume supports completed-round ancestry and retains all prior outcomes. Derived cost audit:48,451completed-round steps +384fourth-round prefix +13,440allocated pi0 suffix =62,275actual inherited steps, versus407,235historically reserved. Old failure files untouched.

Five focused reward/classifier tests pass. Live resumed job and acceptance evidence: `../runs/experiments/pi0_short_pulse_20260913/conflict_recovery/training/status.json`; derived receipts `conflict_recovery/reused_results.json` and `cost_audit.json`. Finite32round budget unchanged; child errors remain visible, no automatic retries.

## Multiple pulse locations — 2026-09-13

User authorized perturbations at multiple points along the real pi0 trajectory. Added configured pulse_start_schedule, with unperturbed pi0 approach and exactly3residual steps, amplitude0.15unchanged. Collector retains prefix_mask for all physical approach steps, separate mask for actual PPO pulse steps, per-candidate timing and complete handoff snapshots. Budget includes approach and pulse physics; pre-pulse terminal candidates cannot create invented pulse samples. Per-round metrics/region CSV include pulse_start_step. Existing zero-delay mode remains default.

New bounded run: pi0_multilocation_pulse_20260913,36rounds×128candidates, schedule5/10/15/20/25/0 repeated6times. Same initial start and current7-policy bank; fresh explorer with existing delayed-quality reward and PPO. One cyclic training lineage, not independent timing ablations. Maximum51,741,424 interactions includes worst-case expanded-bank suffixes and36bounded128krepair-policy trainings; actual separately recorded. Six focused tests pass. Live state and complete artifacts: `../runs/experiments/pi0_multilocation_pulse_20260913/INDEX.md`. GPU launch status is to be read live; no effectiveness claim yet.

## Long amplitude/seed campaign — 2026-09-13

User authorized substantially longer unattended JIT training and multiple attempts. Frozen plan `configs/pi0_pulse_campaign_20260913_plan.json` queues six fresh 72-round explorer lineages after the existing multi-location run: limits0.10/0.15/0.25 × explorer seeds9961302/9961303, each128candidates/round and3pulse steps at cyclic5/10/15/20/25/0. Total55,296candidates. Identical optimizer and initial pi0 support; bank policies are shared frozen inputs, not independent trained-policy replicates. Pending candidates retain the bounded128k successor-training branch.

Maximum1,019,011,488interactions is the conservative expanded-bank/repair reservation across six groups, not expected or actual consumption. Each group execution ceiling8hours; six finite stages plus120s comparison; gates have24h finite waits. Existing runner stops on any child failure or source/input drift, no retries or automatic extension. Every round retains loss/KL/reward components, hyperparameters, plots/replot data, full traces/snapshots and actual cost receipts. Automatic cross-group comparison runs only after all groups complete. Schema, all declared hashes, exact budget formula and comparison-script syntax verified; no extra GPU test run. Active multi-location first collection confirms5zero-residual approach steps and3training steps per lane, maximum effective residual0.14778815.

Live queue and declaration: `../runs/experiments/pi0_pulse_campaign_20260913/INDEX.md`; read execution/status.json for current status, not this dated snapshot. No final TEST or scientific superiority claim.

## Terminal prefix repair and resumed campaign — 2026-09-13

The original six-arm queue stopped at amplitude0.25 seed9961302 round0001: two terminal lanes were passed to the nonterminal snapshot API. Completed72round0.10/0.15 arms stay immutable. Collector now emits terminal trace records without restart snapshots; actual terminal landing/failure/conflict labels are distinct, and terminal positives cannot enter restart support. New tapes retain physical_failure/end_code. Legacy landing terminals without conflict telemetry remain unknown. No clearing done or inventing continuation states.

Seven focused tests pass. Real GPU recovery collection reused the saved13×128prefix plus locked behavior/post-collection optimizer/RNG without new physics:128candidate records,126valid snapshots and2terminal records (lanes21/82), collection receipt0new interactions. Prior11893interactions include10229completed receipts plus1664saved physics. Suffix evaluation has started for the126nonterminal candidates. Recovery queue continues this arm to72rounds then runs three unstarted seed9961303 arms; same amplitude/reward/PPO budgets,8hper-stage ceiling, no automatic retry. Frozen configs are pi0_pulse_campaign_recovery_20260913_*.json. Live index: `../runs/experiments/pi0_pulse_campaign_recovery_20260913/INDEX.md`.
