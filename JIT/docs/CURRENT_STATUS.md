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

## Preserved engineering-error attempts

### formal pi_1 attempt

`JIT/runs/pi_unified/pi_1_tube1_natural10_10009600_seed821101_20260901`

This run remains immutable engineering-error provenance. It reached 1,024,000 PPO transitions and its first TRAIN-panel report proves 449 environment interactions, while the terminal status recorded zero diagnostic interactions because plotting failed after rollout but before callback accounting returned.

The plotting defect was fixed. Future formal runs perform zero-interaction all-Tube plotting/snapshot preflight and reconcile failed-run diagnostic accounting from persisted TRAIN-panel reports.

### first paired pi_0 -> pi_1 gate attempt

Preserved path:

`JIT/runs/pi_unified_gate/pi_0_to_pi_1_paired_core_boundary_20260901`

Preflight was valid, but the real audit ended with a Warp/MJX CUDA OOM after 2,546 environment interactions. This attempt has no scientific gate result. The runner was changed to reuse one compiled `env.step` across all paired rollouts; the scientific protocol remained unchanged.

## Completed paired pi_0 -> pi_1 scientific gate

Authoritative completed retry path:

`JIT/runs/pi_unified_gate/pi_0_to_pi_1_paired_core_boundary_20260901_retry01`

Config:

`JIT/configs/envelope_iter1_paired_policy_gate_retry01.json`

Scientific protocol SHA-256:

`24a126ee94472eebbcb59fff66618ae00dae41074a1d1cfee8bb816afaff410a`

Artifact identities from the completed local audit:

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
- pi_1 successes: 201 / 222
- baseline-success -> candidate-failure regressions: 21
- improvements on core: 0
- upstream regressions: 16 / 117
- downstream regressions: 5 / 105
- non-vacuous baseline coverage: true
- core gate: **FAIL**

### Boundary-gain result — PASS

Locked challenge bank: 56 frozen pi_0 TRAIN continuation-negative frontier states, excluded from Tube_1.

- pi_0 reproduced all 56 failures; reproduction failures: 0
- pi_1 successes: 12 / 56
- successful pi_1 parent groups: 5
- required parent groups: 2
- upstream gains: 12 / 26
- downstream gains: 0 / 30
- boundary gate: **PASS**

### Iteration decision

- iteration accepted: **false**
- empirical pi_0 -> pi_1 envelope expansion accepted: **false**

This is a scientific rejection under the predeclared protocol, not an engineering error. Do not change the gate threshold, audit bank, reward, PPO settings, or acceptance rule after seeing this result in order to convert it to a PASS.

The result demonstrates a real tradeoff: pi_1 acquired measurable new upstream frontier capability while losing previously established core capability. The current working hypothesis is catastrophic forgetting / insufficient retained-core replay under Tube_1 sampling, but that mechanism is not yet established and must be diagnosed from the frozen gate and Tube artifacts before changing the method.

## Active scientific blocker: explain and repair core regression

No C^1 / Tube_2 / pi_2 stage is authorized yet.

Next steps are:

1. preserve the completed paired-gate artifact unchanged;
2. perform zero-interaction diagnosis on the 21 core regressions, including phase, candidate terminal outcome, parent/source provenance, and Tube_1 sampling mass;
3. test whether retained Tube_0 core replay was materially diluted inside Tube_1, especially per phase;
4. distinguish a sampling/curriculum failure from a deeper policy/runtime/phase failure;
5. predeclare the method repair only after diagnosis;
6. train a new candidate only under that new declared method; do not reinterpret the rejected pi_1 as accepted;
7. repeat a newly predeclared comparable core/boundary gate for the repaired candidate;
8. only after both gates pass may the next accepted policy generate C^1 and Tube_2 evidence.

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
- paired gate runner reuses one compiled `env.step` and completed the retry without OOM.

Remaining migration debt before unattended later iterations:

- `core_retaining_tube_iteration.py` still contains Tube_1 / iteration-0 constants;
- shared continuation refit/fresh validation still depend on some upstream-specific evidence/CV helpers;
- the workflow must be extended to stop on scientific gate failure and surface diagnosis rather than automatically continuing;
- generic k -> k+1 Tube/continuation contracts must be completed before a future accepted policy can proceed to Tube_2+.

Do not spend effort on those later-iteration migrations in a way that bypasses the current core-regression blocker. Scientific diagnosis comes first.
