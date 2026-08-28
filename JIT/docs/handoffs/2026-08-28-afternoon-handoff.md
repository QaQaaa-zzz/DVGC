# DVGC/JIT 2026-08-28 下午交接报告

## 一句话状态

两阶段路线已经从“冻结专家”推进到“一次完整的 10M+ 单 Actor Tube-RSI
训练完成”：`pi_up`、`pi_down`、`V_up`、`V_down` 已冻结/锁定，TRAIN-only
learned Soft Tube 已构建，统一策略已完成 `10,009,600` 个 PPO 训练交互。
现在缺的不是继续堆训练步数，而是先完成独立、预声明、冻结最终策略的
Final-Recovery 评估；在该评估通过以前，不能声称最终部署策略、JCE/JEL 或安全
Tube 已经成立。

## 当前方法阶段

```text
pi_up*       FROZEN  (9,977,856 transitions)
pi_down*     FROZEN  (25,600 transitions)
    |
V_up         LOCKED first pass (117 TRAIN / 41 validation)
V_down       LOCKED first pass (105 TRAIN / 21 validation)
    |
learned Soft Tube  BUILT (222 TRAIN entries; guidance only)
    |
Tube-RSI smoke     GO (16 engineering interactions)
    |
pi_unified pilot   GO (25,600 transitions)
    |
pi_unified formal  COMPLETED (10,009,600 transitions)
    |
independent frozen-policy Final-Recovery evaluation  NOT RUN
    |
JCE/JEL claim      NOT AVAILABLE
```

这里的 Apex 是两个阶段之间的 transition band，不是第三个专家。统一训练没有
做 expert switching；`pi_unified` 是一个新的单 Actor。

## 今天下午完成的工作

### 1. Learned Soft Tube 与 Tube-RSI 接口

- 将冻结专家、延续标签和两个 value model 绑定成一个 TRAIN-only learned Soft
  Tube。
- Tube 共 `222` 个状态，其中 upstream `117`、downstream `105`。
- 采样权重固定为 `0.05 + 0.95 * value_score`，保证低 value 样本仍有非零支持。
- Soft Tube manifest SHA-256：
  `c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b`。
- 生成了真实的 x-z 投影图：
  `JIT/runs/soft_tube/soft_tube_train_v1_20260828/xz_projection.png`。
- Tube-RSI 工程 smoke 使用 8 个 upstream 和 8 个 downstream 状态，共 16 次
  交互，有限值检查通过，结论为 `GO`。
- 全过程没有使用 validation 或 TEST 数据。这个 Tube 仅是训练指导，不是认证
  安全集。

### 2. 统一策略 pilot 与固定 TRAIN panel

- 先完成 `25,600` transitions 的 capacity-clean pilot，使用 1,024 个并行环境，
  `naccdmax=1024`，checkpoint 可恢复，无 CCD capacity warning。
- 随后增加固定的 8-up/8-down TRAIN panel 和 x-z visitation 图，验证统一策略能从
  Tube 两侧状态直接 rollout，不依赖专家切换。
- pilot panel 共 454 次诊断交互：2/16 recovery success、12/16 见到 Apex、4 次
  upstream-to-downstream phase transition。它只用于工程诊断，不是最终评估。

### 3. 10M+ formal unified Tube-RSI PPO

- 按用户要求把正式预算设置为不低于一千万步，并对齐 PPO block：
  `10,009,600 = 391 x 25,600`。
- 固定 seed `821101`、1,024 并行环境、`naccdmax=1024`。
- checkpoint 位点：`0 / 1,024,000 / 2,508,800 / 5,017,600 /
  7,500,800 / 10,009,600`。
- 每个非零 checkpoint 运行同一组 8-up/8-down TRAIN panel，并生成 x-z 图；没有
  Brax eval、固定 validation panel 或 TEST panel。
- 完成的 retry 记录了 391 行 PPO metrics、391 行 episode metrics，所有 block
  与 checkpoint 对齐且数值有限。
- 训练交互 `10,009,600`，TRAIN panel 诊断交互 `3,251`，合计
  `10,012,851`；独立评估交互为 `0`。
- 最终 checkpoint payload SHA-256：
  `a9914d5f774c83f923ea1d42712b14731ff6e70a37cfc4e79b5e0a1c8a64d2ed`。
- 最终 checkpoint 已成功 restore，并产生有限动作。参数总数 `313,134`：Actor
  `153,352`，Critic `159,233`，normalizer `550`。

### 4. Formal 首次启动故障与修复

第一次正式 run
`pi_unified_formal_10009600_seed821101_20260828` 在首个 block 的 Brax episode
callback 内触发错误：episode logger 比 policy callback 更早调用共享 progress
callback，违反了已有的 checkpoint/metrics 顺序约束。

处理方式：

- 将该 run 明确关闭为 `engineering_error`，训练账本保持 0，不重用它的
  transition-0 checkpoint。
- 增加回归测试。
- formal mode 不再把 Brax in-epoch episode logger 接到有顺序要求的 progress
  callback；episode evidence 走独立持久化通道，PPO loss/KL/SPS 仍按 block
  记录。
- 使用全新 run id `_retry1` 从头训练。修复后本地 preflight 为 362 个 non-GPU
  和 14 个 GPU tests 全通过，随后 retry 完整结束。

## 10M 训练结果

固定 TRAIN panel 的趋势如下。它们是同一组训练分布诊断，不能作为独立最终
泛化证据。

| transitions | success | Apex | 主要失败 |
| ---: | ---: | ---: | --- |
| 1,024,000 | 3/16 | 11/16 | roll 8, pitch 1, stuck 4 |
| 2,508,800 | 10/16 | 16/16 | prohibited contact 3, roll 3 |
| 5,017,600 | 12/16 | 16/16 | prohibited contact 4 |
| 7,500,800 | 14/16 | 15/16 | pitch 1, stuck 1 |
| 10,009,600 | 13/16 | 15/16 | prohibited contact 1, pitch 1, stuck 1 |

最终 panel 中 upstream 为 5/8 success，downstream 为 8/8 success。7.5M
checkpoint 在 TRAIN panel 上比最终 checkpoint 高 1 个成功样本，但不能因为看到
这个结果就事后改选 7.5M；当前预声明的最终候选仍是 10,009,600 checkpoint，
必须先对它做独立 held-out 评估。

最终 x-z 图在：
`JIT/runs/pi_unified/pi_unified_formal_10009600_seed821101_20260828_retry1/train_panels/transition_10009600/xz_visitation.png`。

## 锁定策略与结果

### 冻结专家

- `pi_up_star`：
  `JIT/runs/phase_u/phase_u_v4_pitch15penalty_9977856_seed820901_20260826/checkpoints/transition_9977856`
  - actor SHA-256：`f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`
  - payload SHA-256：`965b2f141d5cff17f7cef943004fe4e4f62f0fecfe88b47353dc78b14ced723a`
- `pi_down_star`：
  `JIT/runs/phase_d/phase_d_new_reward_smoke_25600_seed820001_20260827/checkpoints/transition_25600`
  - actor SHA-256：`7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`
  - payload SHA-256：`67b0a7406e7fc2a3b73bac3024aaab2215a50f627746a0bc4b0551e3554113ea`
- 权威冻结清单：
  `JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`。

### Value models

- `V_up` manifest：`fd8afee42bcad15ad2f25903fbd4c0aab8900a2a9cb023b32da533e6d6eab1a7`
  - params：`4e1647d2ed5d353e78802c234d48e4fc6061f4237f88201b8479b6cc9040fcc7`
  - normalization：`9f2b5494a94a4d7d6eedbc4adf10f668e74fe3c23076198809b543df917baf28`
- `V_down` manifest：`20025c88409030c77e8cc71f691ec6adcb22377e04471e649b81e273a5f3facd`
  - params：`acedbac3df76e7d6480c25a268cfff530bb6e7ecb3619883a8fbe4edad1460cc`
  - normalization：`fbfabea8bc587808fda85904a197a35dbfb861bfe8eef070c20712288424c4ea`
- 两者都是固定 first-pass schedule；validation 只报告，TEST 完全未使用。

### 统一策略

- 当前 frozen-evaluation candidate：
  `JIT/runs/pi_unified/pi_unified_formal_10009600_seed821101_20260828_retry1/checkpoints/transition_10009600`。
- 这是训练完成的 candidate，不是已通过独立评估的 deployment policy。
- 归档同时保留 0、1.024M、2.5088M、5.0176M、7.5008M 和 10.0096M
  checkpoint，便于审计训练轨迹；禁止根据 TRAIN panel 事后挑最好点来冒充预声明
  选择。

## 当前仍然存在的问题

### 1. 缺少独立 Final-Recovery 评估

这是当前唯一的科学晋级阻塞项。现在只能说“统一 Tube-RSI PPO 已实现并在 TRAIN
panel 上改善”，不能说自然起跳泛化、完整跳跃部署、JCE/JEL 或安全保证成立。

### 2. 早期 PPO 更新不够稳定

第一行 KL 为 `118.6839`，391 个 block 中有 9 个 `KL > 1`、64 个
`KL > 0.1`；最终 KL 降到 `0.01937`。policy min std 最低 `0.00377`，最终
`0.00713`。训练没有 NaN/Inf 且最终策略可恢复，但这说明冷启动阶段仍有大幅策略
漂移。先评估现有 final candidate；只有独立评估失败且用户批准新训练时，才应一次
只测试一个稳定性假设，例如 normalizer warm-up、较低初始 learning rate、或明确的
KL guard，不能同时改奖励/物理/Tube/优化器。

### 3. 训练元数据有一处语义冲突

resolved formal config 中 `formal_method_stage_training=true`，而通用 run manifest
仍有 `formal_training_evidence=false`。实际 interaction ledger、formal report 和
checkpoint 均完整，但对外汇报前应统一这两个字段的语义，避免审稿/审计时把“方法
阶段正式训练”和“独立科学证据”混为一谈。修复只能改元数据生成和测试，不能改写
已经发生的交互账本。

### 4. Soft Tube entry 使用了本机绝对路径

222 个 entry 的 snapshot 字段含 `/home/qy/DVGC/...`。在本机继续接手没有问题；
如果从远端克隆到其他路径，先做一个经过测试的 repo-root rebinding 工具或保持同一
clone 路径，不能手工批量改 JSON 后丢失 hash/provenance 对应关系。

## 建议的下一步（顺序不能倒）

1. 从远端检出本分支，运行 archive verifier 和 JIT preflight，确认策略/结果没有在
   传输中漂移。
2. 仅修复上述 formal metadata 字段冲突并加回归测试；不触碰策略参数、环境、奖励、
   Soft Tube 或交互账本。
3. 为 `transition_10009600` 预声明一次独立 frozen-final-policy Final-Recovery
   evaluation：固定新 seed namespace、初始条件、样本数、成功/失败定义、交互预算、
   stopping condition 和输出目录；不得使用 TRAIN/validation/TEST 来调参或选点。
4. 评估结束后一次性分析 terminal cause、Apex、phase transition、recovery success、
   x-z 投影、姿态、位移和动作统计。不要训练过程中轮询。
5. 只有独立评估通过，才冻结 `pi_unified_star` 并进入 empirical JCE/JEL protocol。
   若失败，先按失败原因提出一个单变量假设，再申请下一次 10M+ 训练；不要直接续训
   当前 checkpoint，也不要把 7.5M TRAIN panel 的优势当作 selection evidence。

## 接手与验证命令

```bash
git switch agent/two-phase-soft-tube
/home/qy/mujoco_playground/.venv/bin/python JIT/scripts/verify_handoff_archive.py
bash JIT/scripts/local_preflight.sh
```

归档根列表在 `JIT/handoff/2026-08-28/ARTIFACT_ROOTS.txt`，逐文件大小与
SHA-256 在 `JIT/handoff/2026-08-28/LOCKED_ARTIFACTS.json`。归档没有包含整个
`JIT/runs/phase_u` 历史目录（约 774 MiB），只包含方法当前依赖闭包；单个最大文件
远低于 GitHub 100 MiB 限制。当前清单为 1,149 个文件、39,937,705 bytes，且
额外验证了 222 个 Soft Tube snapshot 引用和 2 个冻结专家 checkpoint 引用均在
归档闭包内。

## Git 边界

本次交接提交只显式包含 JIT 方法代码、报告、归档清单以及上述锁定 artifacts。
以下是用户原有/无关修改，保持未暂存且不会被提交：

- `dvgc/phase_u_launch_diagnostic.py`
- `tests/test_phase_u_launch_diagnostic.py`
- `.vscode/`
- `JIT/jit_continuation_labels_phase1.patch`
- `docs/TWO_PHASE_REBUILD_GUIDE.md`

此前两个本地逻辑阶段提交也应一并推送：

- `2808827 feat(jit): add soft tube and unified Tube-RSI pilot`
- `48cb620 feat(jit): launch formal unified Tube-RSI training`

本报告对应的归档提交在完成验证后统一创建并推送到
`origin/agent/two-phase-soft-tube`，不 merge `main`。
