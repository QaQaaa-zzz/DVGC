# DVGC / JIT project — bootstrap and empirical envelope discovery

Latest production result (2026-09-08): landing_frontier_v1 completed 48 trajectories, 1,141 arrivals, 1,129 bank witnesses, +1,107 root cells (cumulative 3,463), with 92,058 interactions and no PPO. Forty trajectories landed; eight terminated before landing; none hit sampling limits. Twelve no-witness states come from one pi_1 positive-knee 0.2 trajectory, not twelve independent failures. Next: targeted TRAIN boundary refinement and a locked complementary training recipe, not another complete broad scan or automatic PPO. Review bundles are now compact by default; use JIT/cli/package_results.py --output-dir <existing-run> to repackage without simulation. See JIT/docs/JIT_FRONTIER_RESULT_AND_PACKAGING_20260908.md (relative to repository root). Earlier next-run notes below are historical.


Current action after four-proposer production results (2026-09-08): 1,116 arrivals, 1,115 bank witnesses and +904 root cells; cumulative baseline 2,356. All formal labels used serial execution. Run [landing/frontier exploration](JIT/docs/JIT_LANDING_FRONTIER_20260908.md) via JIT/cli/explore_frontier.py --gpu 0. Use a locked TRAIN-informed 48-trajectory allocation, stronger legal action offsets, and an explicitly versioned sampling guard to x=8 m while retaining the original pi_0 reference. Report landing/failure/truncation and separate old-corridor novelty from newly observed extended-domain support. Skip repeated device benchmarks; preserve serial identity checks and cost accounting. No PPO, no extra snapshot replay checks, no automatic training admission. Older “next” actions below remain historical.


Latest action (2026-09-08): the 5 cm pi_0 pilot completed with 242 arrivals, 240 bank witnesses and +227 root cells (cumulative 1,452 versus the prior TRAIN panel). All four Warp device benchmarks failed at contact__dim; serial fallback completed labels. The user authorized continuing with the single-world device-map fix and four-proposer discovery comparison. Follow [the current run guide](JIT/docs/JIT_MULTI_PROPOSER_DISCOVERY_20260908.md) and run JIT/cli/compare_discovery.py --gpu 0. Preserve prior labels, 5 cm real-frame sampling, the old physical grid and accepted replay limits. New discovery counts use the old shared TRAIN union plus the completed dense pilot. No PPO, no extra snapshot replay investigation; one exploration seed remains development evidence. Earlier “next run” instructions below are historical.


Latest action (2026-09-07): the four-policy comparison completed in production (48 shards, 197,604 new interactions). The user authorized 5 cm real-frame multi-state acquisition and measured label acceleration, with all four legal action channels allowed. Run [the bounded dense Tube pilot](JIT/docs/JIT_DENSE_TUBE_PILOT_20260907.md) next; it automatically compares serial/device execution and falls back to serial if needed. Preserve the original centerline and physical-grid resolution, all old evidence, and the accepted replay limitation. No new PPO or additional snapshot replay investigation. This is a 16-trajectory TRAIN pilot, not a completed matched-budget multi-proposer experiment. Earlier run instructions below are historical.


## Objective

**Discover broader empirically witnessed jumping support at controlled total interaction cost for a fixed bicycle–pendulum robot and operating condition.** A single Actor is a seed/probe and optional application object; it need not realize the entire cumulative Tube.

Working title: **JIT: Bootstrapping and Exploring Empirical Jumping Envelopes for a Bicycle–Pendulum Robot**.

This direction follows the user's confirmed paper outline. It supersedes the former main objective of selecting one successor Actor that retains the whole replay support. It does not retroactively validate historical results or implement a new workflow.

## Why this task and why two stages

Dynamic balance, wheel–ground interaction, pendulum actuation and takeoff/landing transitions make successful jump experience difficult to obtain. The proposed motivation is that early failures reduce visits to useful airborne/landing stages. Measure this using phase-entry counts and matched learning curves; short task duration alone is not evidence of poor learning efficiency.

The method has two connected stages:

1. **Bootstrap:** phase-specific up/down policies and their observed state/continuation data produce value-weighted initial training support; unified-policy training and subsequent development produce a successful frozen seed pi_0.
2. **Discovery:** pi_0's real trajectory supplies fixed longitudinal coordinates; complementary frozen probes generate real forward arrivals and landing continuations; exact witnesses accumulate into an empirical Tube and support training of new probes.

The historical bootstrap is not an idealized all-positive filter. The committed initial Soft Tube has **222 rows: 117 upstream (99 positive, 18 negative) and 105 downstream (81 positive, 24 negative)**, with nonzero value-based sampling weights. These historical phase labels are not automatically current first-landing labels. Later pi_0 references point to the **Round1** checkpoint, not merely the first completed 2026-08-28 unified run. The bootstrap benefit still needs controlled evidence.

## Scope and definitions

Task starts at the declared full near-ground preparation state at `x=2.5 m`. Earlier natural approach is out of scope. Fixed runtime: `assets/orange_bike_4kg_horizontal.xml`, 2 kg payload, 0.005 s simulation step, 0.020 s control, actions `[steer, rear-wheel drive, hip, knee]`, hip/knee +/-30 N m.

Let `R_hat_k` be exact states reached from the declared start, `Pi_k` the versioned frozen probe bank, and `S_k` training/reset support. Define empirical support by:

$$
\widehat T_k = \{s\in\widehat R_k : \exists\pi\in\Pi_k,\;\operatorname{LandingWitness}(s,\pi)=1\}.
$$

The witness must use the same accurate state, controller/event history, endpoint and declared remaining-time semantics. First valid landing is sufficient; post-landing recovery is outside the active endpoint. One success is an observed witness, not a success probability estimate.

Only after matching exact evidence do we project to physical cells. Keep original state/context records: physical deduplication is a coverage statistic, not permission to erase additional witnesses. `S_k` can contain historical reset rows without valid current arrival evidence; those rows do not enter `T_hat_k` automatically.

The centerline stays fixed: real pi_0 frames, 0.1 m x slices from 2.5 m to first landing or the existing 4.2 m cap. It is an exploration coordinate, not an Actor goal or reward. Changing the corridor requires a new comparable protocol.

## Probe-bank design

Multiple frozen policies may both propose forward paths and evaluate continuations. A new probe may add useful witnesses despite reduced realization of old support. Retain previous frozen policies and evidence; no universal-Actor assumption is required.

Keep four separate decisions:

1. Is a checkpoint technically compatible and reproducibly frozen?
2. Is an individual arrival/landing witness valid?
3. Does this probe add unique states/cells or improve discovery per unit cost?
4. How well does this Actor realize a common panel? (diagnostic/application)

A versioned bank and witness registry must track roles, exact Actor/payload identities, predecessor version, tasks, outcomes and exclusions. Prior `selected_policy.json` files do not define this bank. Phase experts need explicit runtime/context compatibility before they can serve as unified probes; their bootstrap role is already established.

If all declared evaluators fail, record no witness under that bank and budget. Keep untested, partially evaluated and engineering-error candidates distinct. A newly successful suffix does not rewrite the old failed experiment.

## Current implementation and evidence

Original audit: `bfc22f2e32cb78cb269b0e522c3bdd7c6e7a8d42`. Follow-up code changes implement identity fixes and a first complementary-probe workflow; see [implementation and production gates](JIT/docs/JIT_PROBE_BANK_IMPLEMENTATION_20260905.md).

| Item | Current evidence | Practical limit |
| --- | --- | --- |
| up/down and handoff bootstrap | Frozen expert manifest, 56-snapshot handoff bank, continuation data, value-weighted Tube0 | Does not establish bootstrap superiority or all-success Tube0 |
| Unified seed | Completed early unified run and later Round1 pi_0 identity referenced by causal scans | Exact Round1 freeze/trace/checkpoint chain must be materialized on the runtime host |
| Fixed-start causal scanning | pi_0 prefix plus real dynamics and family landing reports | New per-proposer catalog/bank supervisor passes CPU fixtures; real GPU acquisition and replay still pending |
| Wide family round | 1,230/1,258 TRAIN witnesses; 713 reported new causal root cells | Summary-level evidence; raw catalogs/snapshots not all in Git |
| Training Tube3 | 4,803 rows, +1,159 over Tube2 | Training rows are not all capability evidence |
| pi_3 | 10,009,600 training transitions; 1,130/1,258 source-state landings | Old core comparison mixed endpoints; complementary contribution not yet established |
| Expanded audit round | 1,754 TRAIN, 583 CALIBRATION, 574 ACCEPTANCE arrivals, saved scores | Missing family labels after GPU failures |
| New paper loop | First versioned bank, suffix supervisor, observation index and retry ledger implemented | GPU equivalence, cumulative physical/cost registry, complementarity recipe and matched-budget results remain missing |

Full evidence paths and issue IDs are in [the review](JIT/docs/JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md). Runtime state is in [CURRENT_STATUS](JIT/docs/CURRENT_STATUS.md).

## Primary outcomes and paper experiments

Primary: valid novel physical support versus total interactions, per-slice/per-phase support, and marginal probe contributions. Auxiliary: bootstrap learning, single-Actor realization, forward-task application performance, failure modes and optional predictor quality.

Minimum comparisons:

- End-to-end complete-task learning versus phase-support bootstrap, counting bootstrap cost.
- Frozen pi_0 exploration versus a fixed multi-probe bank with uniform exploration versus the proposed iterative discovery method.
- Fixed versus growing probe bank; attribute new-arrival and new-continuation gains separately.
- Same acquired TRAIN support pooled once versus iteratively supplied, if claiming benefit from the curriculum schedule.

Use declared resolution, budget and stopping rules, independent seeds and parent-group-aware uncertainty. No claim of a complete physical envelope, a continuous safe region or superiority based on raw Tube size.

## Data and cost

TRAIN may guide/train probes. CALIBRATION and decision-used ACCEPTANCE are development data and cannot enter TRAIN. Re-run global isolation when the bank/ancestor structure changes. Final TEST/JCE/JEL stays unopened; historical bootstrap splits named `test` are not the untouched final distribution.

Charge expert/seed acquisition, prefixes, all suffix evaluations, excluded/failed attempts, retries, PPO and development evaluations. Count reused evidence once in a cumulative ledger while reporting each attempt and any shared bootstrap separately. Wall time does not replace environment interaction accounting.

## Next implementation and training sequence

The next server action is [complete labels and compare pi_0 through pi_3](JIT/docs/JIT_POLICY_ENVELOPE_COMPARISON_20260907.md). The user accepted the near-ground reset (about 0.031 m initial wheel clearance) and observed numerical replay differences on 2026-09-07, and declined additional replay validation. The six-state GPU run had consistent first-landing outcomes and serial/shard labels; exact trajectory equivalence remains unverified. Use the shared pi_0-arrival panel for per-evaluator continuation support, physical occupancy and exclusive contributions. Keep each data role separate; these plots do not represent each policy's own forward exploration envelope.

Follow [JIT_TRAINING_ROADMAP](JIT/docs/JIT_TRAINING_ROADMAP.md): repair correctness first; close the legacy locked audit without changing its family; materialize and verify existing probes; introduce a new versioned multi-probe experiment; compare existing probes before spending another large PPO budget; then lock a complementarity training recipe.

Predictor completion and full-Tube Actor retention are not universal prerequisites for discovery. Identity, exact witness semantics, data roles and declared costs remain mandatory.

The detailed paper structure is in [JIT_PAPER_OUTLINE](JIT/docs/JIT_PAPER_OUTLINE.md).
