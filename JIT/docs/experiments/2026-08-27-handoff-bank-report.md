# Handoff bank smoke report

这次只做了冻结 Phase U checkpoint 的在线状态采集，没有做扰动、Phase D、labels、V、Tube 或 PPO。

输入是同一个 `phase_u_continuation_10m.json` 和三个 checkpoint。每条轨迹最多 1250 ticks；每个 bank 的 interaction ceiling 是 6666。采集使用 held-out seeds 1000001--1000008，并在 Apex 前后固定位置保存完整 snapshot。旧的 7987200 bank 因显存问题中断，后来只补采 1000003--1000008。

结果：4988928 有 8 条轨迹、56 snapshots、697 transitions；7987200 合并 2+6 条轨迹、56 snapshots、1694 transitions；9977856 有 8 条轨迹、56 snapshots、1876 transitions。总 catalog 有 168 条 snapshot 引用；按 qpos/qvel hash 检查没有重复。每条成功轨迹通常有 7 个角色：2 pre_apex、1 nearest_pre_apex、1 nearest_apex、2 post_apex、1 early_descent。

新增采集使用 bounded streaming deque 和一次编译的 JIT reset/policy step，避免保留整条 MJX rollout。旧路径同时存在完整 state retention 和未 JIT step 两个差异；只有两者都修正后观察到显存稳定，无法把 OOM 贡献单独归因给其中一项。

总 catalog 是 `JIT/runs/handoff_bank/catalog_20260827.json`，只引用各 bank 的 snapshot 路径，不复制文件。以后 train/eval 只能按 `parent_group_id` 切分；同一 seed 在不同 checkpoint 的数据必须放在同一侧，避免泄漏。

最终有效 run 路径为 `handoff_bank_4988928`、`handoff_bank_7987200_seeds1000003_8`、`handoff_bank_7987200_jit2` 和 `handoff_bank_9977856_jit8`；catalog 为 `JIT/runs/handoff_bank/catalog_20260827.json`。当前局限是 manifest 尚未记录逐轨迹 termination cause，且这些 snapshot 只是 Phase D 的输入候选，不是安全 Tube，也不支持 JCE/JEL。下一步才是独立设计 Phase D 的 reset、训练和 continuation labels。
