# Teacher recovery reward and retention-focused 100 rounds

> For agentic workers: execute independently owned tasks, then integrate and review. Required workflow: executing-plans; dispatching-parallel-agents for disjoint modules.

**Goal:** Launch the user-approved fresh 100 full JIT loops and supervise five complete rounds, preserving the paused old campaign.
**Architecture:** Extend opt-in generative_bridge contracts, existing PPO trainer and original-layout teacher worker. Freeze pi0 and physical/endpoint identity. Teacher-only explorer reward, retention-heavy student, fixed teacher development monitor.
**Tech Stack:** Python/JAX/Brax/MJX, RSL Torch, JSON contracts, pytest, TensorBoard.

## Global constraints
- Baseline b6f951f8; isolated branch. Do not edit the old worktree/run, unpause old processes, open final TEST, alter physics/endpoints or touch unrelated jobs.
- User confirmed E128, teacher128, 16G candidates/root, H16, .25 three-tick pulse at0/5/10/15; E fresh Adam/weights, LR <=1e-4, entropy .001, value .5, minibatch256, <=4 epochs, clip .2, targetKL .01, gradientclip1.
- Teacher reward: preserve window novelty .02, repeat-.02 and direct failure-2; +2 only fresh state and verified generated full recovery, independent of student. Searched-unsolved -.1; unscheduled/error/ambiguous UNKNOWN excluded, source-only cannot masquerade as G.
- Student pi0 Actor/critic, freshAdam, frozen all normalizers, G73 initialization,32k interactions/round; .1 PPO+1demo+1keep, demo256 fresh/history50/50,keep512. Reset40% full start+30% current verified teacher snapshot+30% historical success. Full-start random four-channel uniform[-.25,.25] three ticks at0/5/10/15, independent TRAIN seeds; no history/clock relabeling.
- Fixed teacher diagnostics use frozen pi0, fixed DEV roots and paired generation streams, C1/2/4/8/16, H16 and same DDIM; retain fixed winners and two reviews. Save G0/G5/G10/G20 and later scheduled checkpoints, raw candidate successes, costs including suffix/review/padding, novelty gains/losses, stages, SAR, noise MSE, action diversity/delta/clipping/failure counts. Never train on DEV panel; distinguish development from final TEST. Cross G/pi saved students configurable, not 100*100 exhaustive sweep.
- Every round actual cost, model identities, real metrics, last/published and reason. No automatic retry/extension. Supervise first five rounds, then stop active polling, leave authorized100 running and desktop notifications.

## Tasks
- [x] Task1: teacher-only E reward/telemetry, compatibility defaults and tests. Files rewards.py + focused new telemetry module/tests, root wires closed_loop.py.
- [x] Task2: student coefficients/frozen stats/reset mixture and checked four-onset pulses. Files student.py and trainer/reset modules/tests; expose explicit config, root wires configs.
- [x] Task3: fixed teacher development monitor module/CLI, candidate-budget metrics, leakage and provenance tests; root hooks checkpoints/launcher.
- [x] Task4: integrate closed_loop/prepare/schema/launcher, pi0/G73 identity, fresh learner initialization, preflight budget/declaration, baseline and CPU tests, code review, commit.
- [ ] Task5: necessary bounded GPU check then launch100 with independent TB and notifications, verify loadedreward/loss, supervise five whole loops; diagnose errors without changing scientific settings silently; synchronize shared ledger before handoff.


## Engineering recovery, 2026-10-13

The first attempt failed before any student/E/G update: the kernel killed the persistent teacher worker for host OOM. Preserve the failed attempt and all completed/partial outcomes. Eager snapshot restoration creates fresh vendored Warp FFI callbacks whose native graph caches outlive evaluations. Release graphs through the actual MuJoCo backend official API after host traces are written; retain explicitly compiled rollout/G kernels and stop at an 8 GiB worker RSS boundary without automatic retry.

An independent audit certified 57 completed stages (71,065 controls), plus a 6,800-control upper bound for the unfinished review; 19 completed roots comprise 15 verified, three unsolved and one quarantined. Explicit recovery uses a sibling attempt, pinned collection/source/search/review receipts and original proposal bytes, unchanged seeds/configuration, and carries all previous charges. No failed label is rewritten or winner reselected.

The first strict historical-trajectory check failed despite identical 17 labels (1,207 controls); retain that negative receipt. A causal four-evaluation control passed all 17 labels, exact initial observations/generated prefixes and retained compiled-kernel identity; actual vendored graph captures fell to zero and RSS was about 2.79 GB (4,828 controls). Full trajectory differences remain descriptive. Both costs count against the original round ceiling. Focused integration: 77 tests passed; subsequent recovery locks received focused tests and the actual collection configuration passed a zero-physics dry check. Full 128-root memory behavior and five complete rounds still require live supervision.
