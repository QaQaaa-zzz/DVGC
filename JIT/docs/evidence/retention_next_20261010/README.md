# Retention-first V/D1 and conditional D2 engineering evidence

![V actual XY](/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/V_002/report_001/xy_four_cells.png)

![D1 actual XY all23roots](/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/D1_003/report_003/xy_teacher_roots.png)

Based on b83b586 D0/D0b; full D0 was not rerun. R73/R74 continuation was not started, original R74 failure retained. Reward/network/action/physical/success contracts unchanged. V/D1 student/E/G updates0. Complete figures, traces and datasets remain server-side; this Git report is a compact evidence index.

|Policy|A nominal no pulse|B random no pulse|C nominal fixed request|D random same request|
|---|---|---|---|---|
|pi0|27/32|189/256|187/256|167/256|
|R5|32/32|231/256|245/256|210/256|

V charged800000,177.754seconds. A is one physical start with numerical repeats. B/D starts and C/D requested residuals are paired. B gains55/losses13; C68/10; D69/26; BCD gains192/losses49/net143. R5 retains its own Actor+normalizer, remains frozen best_dev_candidate, published_policy=null. [V intervals/onsets](V_summary.json), [actual input identities](input_identity_audit.json).

|D1 role|G only|Noise only|Both|Neither|Ambiguous|
|---|---|---|---|---|---|
|TRAIN|3|5|5|2|1|
|SOLVER_DEV|1|1|2|3|0|

Fresh128TRAIN+32solver-dev complete ancestors; all160post-pulse roots legal. Initial independent pi0 suffixes: TRAIN91/128 successes, solver-dev24/32. Before teachers,16TRAIN chosen roots and7solver-dev roots each failed three independent pi0 continuations; one mixed baseline solver-dev root was excluded. Selected TRAIN onsets0/5/10/15 have2/1/7/6roots; teacher selection never drives root selection. R5 same-snapshot success8/16TRAIN and2/7solver-dev is distinct from independent full-task R5 performance.

Eight qualified G TRAIN lessons,525actual observation/action examples, cover all4onsets. Every lesson has independent pi0[0,0,0], source-batch[0,0,0], frozen original winner[1,1,1] and complete same17-world search+2replay receipts; one source ambiguity is quarantined. No winner re-selection. Five lessons are already rescued by frozen R5 at the same pi0+E snapshot, three are not. G and Noise both provide unique solutions; diffusion necessity is unproved. D0 new87 is not teacher conversion.

Raw context-aware diagnosis: all23selected independent pi0 first physical failures are after valid contact;19have roll-limit events,0prohibited contacts. The raw control-step log cannot locate substep contact time. [Per-root repeats and costs](D1_root_labels_and_costs.json), [failure stages](D1_failure_stages.json), [D1 summary](D1_summary.json). Full root source/winner/loser traces and first-event timing: `/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/D1_003/report_003`.

D1 charged180909 including preserved failed6800reservation. D1_002 failed during CUDA allocation at root012 G search after72teacher dispatches; that attempt remains engineering UNKNOWN, not no-solution. Explicit D1_003 retains completed roots and failed-root proposals, uses one bounded GPU worker per root with the same original17-world layout, and completes remaining11roots. It does not prove all GPU allocator failures eliminated.

D2 preparation, not formal training: successful fresh TRAIN keep26/32(14/16 nominal numerical repeats,12/16 random),1951pre-action observations; no DEV input. Keep sampling balances nominal/random then ancestors/trajectories; demos balance8ancestors then H16prefix/pi0tail. No near-label conflict under the declared descriptive normalized76D-RMSE<=.02/action-RMSE>.05 check; this is not a general consistency proof.

Real GPU micro100/100transitions,2actual SGD updates passed all invariants in29.274seconds: entire Actor+privileged-critic normalizer hash unchanged, Actor/critic/Adam changed, env_steps and restored clock100, full learner/RNG restore exact, serialization action difference0. lambda_demo=.1999414→.1998828 on completed_transitions. Shared Actor+critic clipping scales.002748/.0006504; actual weighted Actor norms/cosines and Adam deltas recorded. Fixed TRAIN before/after probes use actual saved weights and are explicitly separate from actual minibatch gradients. [Micro receipt](GPU_micro_verification.json), [actual update audit](actual_update_analysis.json), [verified TensorBoard](http://localhost:6025), [visibility receipt](visibility.json). Micro metrics do not establish ability improvement.

[Conditional D2 plan](D2_preparation.json), [refined budget](D2_budget_dry_run.json):1846700/2000000charged physics, BC<=2000supervised updates,12hours. B first; A/C each<=128000PPO. Source PPO LR3e-5→7.5e-6 applies to whole learner. BC inherits1e-5/demo1/keep1. A/C may bootstrap PPO optimizer once at the explicit BC/PPO boundary; subsequent32kchunks restore full learner and preserve the one128kstage clock. Full BC checkpoints save Adam/RNG/absolute supervised clock at100/500/1000/2000. Keep is not blindly multiplied600.

Four shared TRAIN reset pools and new64episode DEV four-cell inputs are locked in the server-side preparation supplement. GPU execution of the four-pool reset adapter and new DEV physical baselines remain required before a separate finite B launch. B selection: independent student success count on seven fixed solver-dev roots, confirmed B/D retention guard, earliest checkpoint on ties; no absorption means no automatic PPO. Formal A/B/C, student absorption, student old-capability loss and final independent student success are NOT_RUN. R5 remains frozen candidate; nothing published as a new policy.

Actual charged total993809 = V800000+D1180909+D2engineering12900. No TEST opened.57related CPU tests passed; GPU micro is separate evidence. [Machine-readable summary](summary.json).

## D2 B-only finite execution

[BC+keep实际结果、三组XY、逐根转化与旧能力损失](B_only/INDEX.md)。2000监督更新完成；BC1000为阶段候选，learner_last2000，R5冻结全局候选，未发布。TRAIN稳定4/8、未见SOLVER_DEV3/7，配对旧成功仍有损失；暂不启动PPO，先闭环与保持覆盖诊断。


- [B_keep_coverage独立入口与零仿真准备](keep_coverage/INDEX.md)：当前prepared，缺新阶段执行授权，真实采集/BC均未启动。
