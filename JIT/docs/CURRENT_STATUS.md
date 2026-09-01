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

## Active scientific next step

1. Freeze the exact completed pi_1 final checkpoint as `pi_1` iteration authority.
2. Run core-preservation gate against retained prior-core evidence.
3. Run boundary-gain gate using new/disjoint TRAIN iteration evidence.
4. Only if both pass, record empirical envelope expansion.
5. Then collect/freeze pi_1-conditioned continuation evidence, fit/validate
   `C_up^1/C_down^1`, construct core-retaining Tube_2, and train pi_2.
6. Keep final TEST/JCE/JEL untouched throughout the iteration loop.

## Repository-maintenance state

The active tree is being converted from experiment-stage scripts into reusable
iteration capabilities.

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

Remaining migration debt before unattended pi_2+ iteration:

- `core_retaining_tube_iteration.py` still encodes Tube_1/iteration-0 constants
- shared continuation refit/fresh validation still depend on a few upstream-
  specific evidence/CV helpers
- those contracts must be made iteration-generic without changing the already
  completed Tube_1/pi_1 artifact identities
- core-preservation and boundary-gain need stable machine-readable production
  gates wired into the workflow

Until those items are closed, workflow automation may sequence existing stages
but must not be advertised as a complete unattended `k -> k+1` scientific loop.
