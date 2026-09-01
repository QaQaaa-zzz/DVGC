# Current JIT status — 2026-09-01

## Completed scientific artifacts

### Tube_1

Authoritative path:

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

- manifest SHA-256: `817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`
- entries SHA-256: `61c6796aaf4c4b1e43624c5cf06bce0d39736a6d1743c5142c6c250d23155ec9`
- 222 retained Tube_0 core entries
- 2,897 expansion entries: 310 upstream + 2,587 downstream
- total 3,119 entries: 427 upstream + 2,692 downstream
- no validation/TEST rows embedded
- zero environment interactions and zero training transitions during construction
- training guidance only; not a certified safe set

### pi_1 formal Tube_1 PPO

Completed run:

`JIT/runs/pi_unified/pi_1_tube1_natural10_10009600_seed821101_20260901_retry01`

Config:

`JIT/configs/pi_unified_iter1_tube1_natural10_retry01.json`

Pre-run canonical config SHA-256:

`987ef5d31661482fd0bc05cea566c177d83ecd00ae3028ff0e8bb2ed462b7901`

Exact final artifact identities:

- final checkpoint: `checkpoints/transition_10009600`
- checkpoint payload SHA-256: `fb5c364057933d62c4e1b6ed49f3181cd36584c5b270f305eef18dff150e68e5`
- checkpoint identity JSON SHA-256: `7053f5cd7acd00f75849d8eceea38f81daf78d6965875aa44baf52a68be953d0`
- formal report SHA-256: `7ec696605244182357cfe2831eaa207e45968c9e382807ecbc8d2d710ce8714a`
- terminal status SHA-256: `0d82971a868b6e48d8c62af815b6a19b4a7372cd67b9825817863c86fac63e45`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`

Formal result:

- requested/completed training transitions: 10,009,600 / 10,009,600
- checkpoints: 0, 1,024,000, 2,508,800, 5,017,600, 7,500,800, 10,009,600
- all five nonzero TRAIN panels completed
- TRAIN-panel interactions: 2,838
- Brax evaluation transitions: 0
- reset mixture: 0.1 natural / 0.9 Soft Tube
- fresh actor/critic/optimizer initialization, seed 821101
- expert switching: false
- validation data: false
- TEST data: false
- final checkpoint restore: verified

Final optimization metrics are diagnostics only; they are not capability gates.

### frozen pi_1 iteration authority

Authoritative path:

`JIT/runs/frozen_unified/pi_1_iter1_10009600_20260901/frozen_unified_policy.json`

The existing production freeze path verified the exact final checkpoint and generated the immutable iteration-1 comparison authority with zero environment interactions and zero training transitions.

- policy name: `pi_1`
- policy role: `envelope_expansion_authority`
- checkpoint payload SHA-256: `fb5c364057933d62c4e1b6ed49f3181cd36584c5b270f305eef18dff150e68e5`
- actor SHA-256: `cc8f202075479aa90c0451732773016dfa25ef8a0c849278b5917319070284f0`
- critic SHA-256: `1519803788302a936d3129eb7010de5fe153ea65410a49c35c1328e22fdf8d8f`
- normalizer SHA-256: `94ed2587601dbe187774f664a741f427d943148a0c5391cdf57a45939c50a191`
- freeze protocol SHA-256: `242301b29f3510895a0f13934deac44c1b38c976bb9a8c8385a85f621613a1ed`
- freeze manifest file SHA-256: `9401240b8b9600669cb9ef5f9edca79f38bee4b0c138966e7288f148a8993942`
- `pi_unified_star` claim: false
- JCE/JEL claim: false
- certified-safe-Tube claim: false

Freezing alone does not establish capability-envelope expansion.

## Preserved engineering-error attempt

`pi_1_tube1_natural10_10009600_seed821101_20260901` remains preserved.
It reached 1,024,000 training transitions and wrote that checkpoint. Its first
TRAIN panel actually used 449 environment interactions, while terminal status
recorded zero diagnostic interactions because plotting failed after rollout but
before the callback returned. Do not rewrite the historical status file.

The mixed-snapshot plotting defect was fixed and the formal training API now
performs a static all-Tube plotting/snapshot preflight before environment
construction, so the same class of error fails at zero interactions.

For future newly-created failed unified formal runs, the canonical training
wrapper also reconciles terminal diagnostic accounting from already-persisted
`train_panels/*/report.json` files. This closes the exact failure mode that
caused the historical 449-interaction undercount while leaving old artifacts
immutable.

## Active scientific blocker: paired pi_0 -> pi_1 gate

The generic machine-readable gate implementation and its pi_0 -> pi_1
predeclaration now exist:

- implementation: `JIT/src/jit_dvgc/analysis/paired_policy_gate.py`
- stable CLI surface: `JIT/cli/diagnose_unified.py --gate-config ...`
- config: `JIT/configs/envelope_iter1_paired_policy_gate.json`
- predeclared protocol SHA-256: `24a126ee94472eebbcb59fff66618ae00dae41074a1d1cfee8bb816afaff410a`

The locked method is:

1. core bank = all 222 Tube_0 source-core states;
2. boundary bank = frozen pi_0 TRAIN continuation-negative frontier states only;
3. reject boundary states already present in Tube_1;
4. write/self-hash the bank before any policy rollout;
5. evaluate frozen pi_0 and frozen pi_1 on the exact same state, horizon,
   deterministic policy mode, continuation semantics, XML, and runtime;
6. core preservation requires zero baseline-success -> candidate-failure
   regressions and at least one baseline core success in each phase;
7. boundary gain requires baseline-negative reproduction plus candidate success
   in at least two distinct parent groups;
8. both gates must pass before empirical pi_0 -> pi_1 envelope expansion can be
   accepted.

The implementation/predeclaration is complete, but it has not yet been locally
compiled/regression-tested at the current HEAD or executed on the real frozen
artifacts. Therefore no gate result or envelope-expansion claim exists yet.

## Scientific next step

1. Sync and regression-test the current paired-gate code/config.
2. Execute the paired audit on frozen pi_0 and frozen pi_1.
3. If either gate fails, preserve the result and stop; do not automatically tune
   thresholds, reward, PPO, or the audit bank.
4. If both gates pass, record empirical pi_0 -> pi_1 capability-envelope
   expansion.
5. Only after acceptance, collect pi_1-conditioned TRAIN evidence, fit/validate
   `C_up^1/C_down^1`, construct core-retaining Tube_2, and train pi_2.
6. Keep final TEST/JCE/JEL untouched throughout the iteration loop.

## Repository-maintenance state

Completed maintenance:

- package-root APIs for training, Tube, snapshots, acquisition, continuation,
  analysis, and workflow
- removed the redundant three-line facade layer
- retired a first batch of completed iteration-0 upstream/downstream research
  scaffolding and tests
- removed obsolete `JIT/planning/` working notes from the active tree
- added explicit resumable manifest-driven workflow orchestration
- added `run_iteration_workflow.py` as the single workflow entry point
- generalized Tube-RSI continuation-field acceptance from only `C^0` to phase-
  consistent `C^k`
- restored `upstream_boundary_lock.py` after dependency-closure verification
  proved it is still required by the retained bootstrap loader path
- added a mandatory compile/import/test deletion gate to agent instructions
- refreshed root AGENTS/README/PROJECT/experiment-state/repository-layout docs
  so context recovery starts from Tube_1/pi_1 rather than old Phase-U state
- implemented a generic paired-policy iteration-selection gate without adding a
  pi1-specific CLI.

Remaining migration debt before unattended pi_2+ iteration:

- `core_retaining_tube_iteration.py` still encodes Tube_1/iteration-0 constants;
- shared continuation refit/fresh validation still depend on a few upstream-
  specific evidence/CV helpers;
- those contracts must be made iteration-generic without changing the already
  completed Tube_1/pi_1 artifact identities;
- the paired gate must be validated/executed, then wired into the workflow;
- the gate runner should reuse one compiled `env.step` across all paired
  rollouts to avoid unnecessary JIT setup overhead. This is an engineering
  optimization, not a scientific acceptance rule.

Until those items are closed, workflow automation may sequence existing stages
but must not be advertised as a complete unattended `k -> k+1` scientific loop.
