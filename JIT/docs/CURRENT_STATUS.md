# Current JIT status — 2026-09-02

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

### rejected pi_1 formal Tube_1 PPO candidate

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

### frozen rejected pi_1 comparison authority

Authoritative path:

`JIT/runs/frozen_unified/pi_1_iter1_10009600_20260901/frozen_unified_policy.json`

- policy name: `pi_1`
- policy role: `envelope_expansion_authority`
- checkpoint payload SHA-256: `fb5c364057933d62c4e1b6ed49f3181cd36584c5b270f305eef18dff150e68e5`
- actor SHA-256: `cc8f202075479aa90c0451732773016dfa25ef8a0c849278b5917319070284f0`
- critic SHA-256: `1519803788302a936d3129eb7010de5fe153ea65410a49c35c1328e22fdf8d8f`
- normalizer SHA-256: `94ed2587601dbe187774f664a741f427d943148a0c5391cdf57a45939c50a191`
- freeze protocol SHA-256: `242301b29f3510895a0f13934deac44c1b38c976bb9a8c8385a85f621613a1ed`
- freeze manifest file SHA-256: `9401240b8b9600669cb9ef5f9edca79f38bee4b0c138966e7288f148a8993942`
- zero environment interactions and zero training transitions during freeze
- `pi_unified_star` claim: false
- JCE/JEL claim: false
- certified-safe-Tube claim: false

This frozen policy remains immutable comparison provenance, but the completed paired gate rejected it as the next accepted iteration authority.

## Preserved engineering-error attempts

### formal pi_1 attempt

`JIT/runs/pi_unified/pi_1_tube1_natural10_10009600_seed821101_20260901`

This run remains immutable engineering-error provenance. It reached 1,024,000 PPO transitions and its first TRAIN-panel report proves 449 environment interactions, while the terminal status recorded zero diagnostic interactions because plotting failed after rollout but before callback accounting returned.

The plotting defect was fixed. Future formal runs perform zero-interaction all-Tube plotting/snapshot preflight and reconcile failed-run diagnostic accounting from persisted TRAIN-panel reports.

### first paired pi_0 -> pi_1 gate attempt

Preserved path:

`JIT/runs/pi_unified_gate/pi_0_to_pi_1_paired_core_boundary_20260901`

Preflight was valid, but the real audit ended with a Warp/MJX CUDA OOM after 2,546 environment interactions. This attempt has no scientific gate result. The runner was changed to reuse one compiled `env.step` across all paired rollouts; the scientific protocol remained unchanged.

## Completed paired pi_0 -> rejected-pi_1 scientific gate

Authoritative completed retry path:

`JIT/runs/pi_unified_gate/pi_0_to_pi_1_paired_core_boundary_20260901_retry01`

Config:

`JIT/configs/envelope_iter1_paired_policy_gate_retry01.json`

Scientific protocol SHA-256:

`24a126ee94472eebbcb59fff66618ae00dae41074a1d1cfee8bb816afaff410a`

Artifact identities:

- summary file SHA-256: `cf63f59f4862c51351ceca80afa8796316592be515c28b572584e97e39d9f7fc`
- bank file SHA-256: `97cb62727e12824abc2a5238e9187e47773c80b5f169f2f286c0f412a8e2a6bd`
- records file SHA-256: `614d020198a38235f0a2bfddc6b087fdd5e3729c5fcff49f40c4fcf71683cae2`
- status: completed
- environment interactions: 23,695
- training transitions: 0
- expert switching: false
- validation data used: false
- TEST data used: false
- final evaluation data used: false

### Core-preservation result — FAIL

Locked bank: all 222 Tube_0 core states.

- pi_0 successes: 222 / 222
- rejected pi_1 successes: 201 / 222
- baseline-success -> candidate-failure regressions: 21
- improvements on core: 0
- upstream regressions: 16 / 117
- downstream regressions: 5 / 105
- core gate: **FAIL**

### Boundary-gain result — PASS

Locked challenge bank: 56 frozen pi_0 TRAIN continuation-negative frontier states, excluded from Tube_1.

- pi_0 reproduced all 56 failures; reproduction failures: 0
- rejected pi_1 successes: 12 / 56
- successful pi_1 parent groups: 5
- required parent groups: 2
- upstream gains: 12 / 26
- downstream gains: 0 / 30
- boundary gate: **PASS**

### Iteration decision

- iteration accepted: **false**
- empirical pi_0 -> pi_1 envelope expansion accepted: **false**

This is a scientific rejection under the predeclared protocol, not an engineering error. Do not change the completed gate threshold, bank, reward, PPO settings, or acceptance rule to convert it to a PASS.

## Zero-interaction core-regression diagnosis — completed

Diagnosis artifact:

`JIT/runs/pi_unified_gate_analysis/pi_0_to_pi_1_core_regression_20260901/diagnosis.json`

Diagnosis SHA-256:

`61e12385ef0e77180b773a2e0de04b36e2a27f649c24d509b4f18345c00a7689`

The diagnosis used zero environment interactions, zero training transitions, no validation, and no TEST.

Observed Tube_1 sampling mass under the rejected candidate's original sampler:

- upstream: 427 entries = 117 retained core + 310 expansion;
- upstream retained-core count share: 27.40%; retained-core sampling-weight share: 24.87%;
- downstream: 2,692 entries = 105 retained core + 2,587 expansion;
- downstream retained-core count share: 3.90%; retained-core sampling-weight share: 3.10%;
- Tube-conditional probability of selecting any old core state: 13.98%;
- all-episode old-core reset probability after the 0.9 Tube reset mixture: 12.59%;
- all-episode expansion reset probability: 77.41%;
- natural reset probability: 10.00%.

All 21 core regressions ended in physical failure. Regression provenance:

- upstream: 16 regressions;
- downstream: 5 regressions;
- source names: 9 `up_nominal`, 7 `up_boundary`, 5 `down_nominal`;
- all five downstream regressions had high per-entry weights near 1.0 but only about 0.000377 conditional probability inside the downstream phase because of expansion cardinality;
- several upstream regressions had the minimum weight near 0.05 and about 0.000125 conditional probability inside the upstream phase.

This establishes **retained-core replay dilution as a material mechanism**. Because pi_1 used fresh actor/critic/optimizer initialization, the precise description is core-support under-replay / capability regression under the expanded training distribution, not classical parameter forgetting from pi_0.

The diagnosis does not prove that replay dilution is the only mechanism. Some high-weight upstream states also regressed, so policy interference may remain after the replay repair. The first repair therefore changes only the training sampling contract and must itself pass a new gate.

## Predeclared iteration-1 replacement method

A replacement candidate remains at **iteration 1**. The rejected pi_1 does not authorize progression to C^1, Tube_2, or pi_2.

Predeclared config:

`JIT/configs/pi_unified_iter1_tube1_core_replay50_natural10.json`

The repair keeps fixed:

- exact Tube_1 support and manifest;
- 0.1 natural / 0.9 Soft-Tube reset probabilities;
- 50/50 upstream/downstream phase mixture;
- fresh actor/critic/optimizer initialization;
- seed 821101;
- PPO hyperparameters and exact 10,009,600-transition budget;
- reward, physics, XML, action semantics, horizon, and task definition.

The only method change versus the rejected candidate is the within-Tube sampling contract:

1. choose phase with the existing 50/50 phase mixture;
2. inside the selected phase, choose retained source core with probability 0.5 and current expansion with probability 0.5;
3. sample retained core uniformly so low-V core states cannot be starved;
4. sample expansion with the existing value/continuation-weighted categorical rule.

Thus Tube-conditioned replay gives 50% mass to retained core and 50% to expansion in every phase. With the unchanged 0.9 Soft-Tube reset probability, the declared all-episode reset mass becomes 45% retained core, 45% expansion, and 10% natural reset.

The implementation is generic and optional. Legacy configs that omit `tube_sampling` retain the historical value-weighted behavior, preserving old pi_0/pi_1 reproducibility.

## Acceptance-data boundary for the replacement candidate

The completed 56-state boundary bank has been consumed by the rejected-candidate decision and diagnosis. It may be reused only as descriptive regression evidence; it must not become the sole independent acceptance evidence for the repaired candidate.

Before the replacement candidate can be accepted, a new non-final paired audit must be predeclared with:

- the same complete 222-state Tube_0 core preservation bank as the structural core contract;
- a fresh pi_0-negative boundary challenge bank generated/locked without inspecting replacement-candidate outcomes;
- physical-state and parent/near-state exclusion against the consumed 56-state boundary bank, Tube_1 admitted states, validation, and TEST;
- the same frozen pi_0 baseline, deterministic continuation semantics, horizon, XML, and no expert switching;
- no final TEST/JCE/JEL use.

The old gate may still be rerun on the repaired candidate as a diagnostic, but not as the only fresh boundary-gain acceptance evidence.

## Active scientific blocker

The immediate blocker is now engineering validation of the new replay contract followed by one predeclared replacement training run. Do not start C^1 / Tube_2 / pi_2.

Required order:

1. compile/regression-test the replay sampler and canonical formal wrapper;
2. run a zero-interaction preflight on real Tube_1 and verify exact 50/50 core/expansion mass in both phases;
3. lock a fresh replacement-candidate boundary audit protocol before inspecting replacement outcomes;
4. run exactly one fresh replacement pi_1 candidate under the declared core replay repair;
5. freeze the replacement final checkpoint;
6. run core preservation plus a fresh boundary-gain audit;
7. if either gate fails, preserve and diagnose again rather than retuning the same gate;
8. only if both pass does iteration 1 become accepted and authorize pi_1-conditioned C^1 / Tube_2 work.

Final TEST/JCE/JEL remains untouched.

## Repository-maintenance state

Completed maintenance:

- stable package-root APIs for training, Tube, snapshots, acquisition, continuation, analysis, and workflow;
- redundant three-line facade layer removed;
- first batch of completed iteration-0 research scaffolding retired;
- obsolete `JIT/planning/` working notes removed from the active tree;
- resumable manifest-driven workflow infrastructure added;
- `run_iteration_workflow.py` is the single workflow entry point;
- Tube-RSI generalized from only `C^0` to phase-consistent `C^k`;
- dependency-closure deletion rules added after cleanup regression;
- root/current project documentation refreshed;
- generic paired-policy iteration gate added through existing `diagnose_unified.py` CLI;
- paired gate runner reuses one compiled `env.step`;
- Tube-RSI now supports an optional config-bound retained-core replay contract without changing legacy behavior;
- the canonical formal wrapper injects and records the declared replay contract rather than relying on environment variables or one-off scripts.

Remaining migration debt before unattended later iterations:

- fresh replacement boundary-audit acquisition/locking must be made workflow-addressable;
- `core_retaining_tube_iteration.py` still contains Tube_1 / iteration-0 constants;
- shared continuation refit/fresh validation still depend on some upstream-specific evidence/CV helpers;
- generic k -> k+1 Tube/continuation contracts must be completed only after an iteration-1 policy is actually accepted;
- workflow automation must stop on gate failure and surface diagnosis rather than automatically advancing.
