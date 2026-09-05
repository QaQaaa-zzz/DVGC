# JIT empirical jumping-envelope iteration protocol

Version: user-confirmed envelope direction, 2026-09-05. Read [root authority](../../AGENTS.md), [status](CURRENT_STATUS.md), and [review](JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md). The first bank/suffix scheduler and integrity fixes now exist; [implementation status](JIT_PROBE_BANK_IMPLEMENTATION_20260905.md) identifies unverified GPU replay and the still-pending cumulative physical/training loop.

## 1. Objective and legacy boundary

Optimize discovery of exact forward-arrived states with successful first-landing witnesses, and report novel physical support versus total interactions. A single successor Actor and full-Tube retention are not acceptance requirements for empirical witnesses.

Already locked 2026-09-04/05 experiments retain their original pi_0 proposer, pi_0/pi_1/pi_2 evaluators, catalogs, endpoints, seeds, horizons and roles. Finish them as legacy experiments. New multi-probe work uses a new protocol/run identity and must not overwrite old outputs or reinterpret historical negative labels.

## 2. Lock task, bank and budget

Before a new experiment, declare:

- XML, physics, actuator and observation-schema identities;
- complete x=2.5 m ground state/context and task/phase time rules;
- fixed real-frame pi_0 centerline, 0.1 m slices, phase assignment and corridor cap;
- frozen policy records and normalizer/Actor/payload identities;
- explicit proposer and evaluator membership, bank version and predecessor;
- per-role allocation, seeds, perturbation windows/actions/strengths and horizons;
- first-valid-landing and physical/task failure semantics;
- exact snapshot/context identity and separate physical cell resolution;
- per-attempt/per-role/total interaction budgets and stopping rules;
- data role/ancestor grouping and holdout exclusion rules;
- optional predictor identity and target bank, if used;
- training recipe and admission decision, if a new probe is trained.

Proposed bank lifecycle must distinguish technically compatible, admitted-for-experiment, resource-inactive and retired probes. Retiring a probe from execution does not delete its prior witnesses. Names such as pi_3 are display labels, not complete identity.

## 3. Bootstrap provenance

Preserve up/down training, handoff capture, phase labels, value models, initial weighted support and unified development history. The committed Tube0 includes negative phase labels with nonzero sampling weights; it is `S_0`, not automatically `T_hat_0`.

Materialize the exact later Round1 pi_0 freeze/checkpoint/normalizer/config and fixed-start centerline trace. Do not substitute the first completed unified checkpoint because its run has the same transition count. A bootstrap comparison must include actual phase-expert and seed development cost.

## 4. Forward discovery

Run the declared frozen proposer from the complete fixed start, with bounded perturbations at declared ancestors/windows. Capture only states reached by actual env.step. Multiple proposers may generate separate catalogs; a supervisor must preserve stable attempt IDs and a global role/ancestor namespace.

General ancestor restoration is permitted only after a verified prefix and full-context replay contract exist. Current code does not implement that general path; starting each attempt from the fixed start is a valid initial implementation.

Retain successful and unsuccessful attempts and their interactions. A state already present in training support must not be excluded from witness acquisition solely for that reason. Preserve distinct controller/time contexts; deduplicate physical coverage only in the metric view. Repeated physical states under different probes can have useful attribution.

## 5. Exact continuation witnesses

Restore the candidate's physical and controller/event context for each declared evaluator. Stop at first valid landing, declared failure or horizon. Define explicitly whether horizon is suffix-local or remaining full-task time. If administrative counters are reset, show that the restoration still corresponds to the stated task contract; do not claim unconditional concatenation equivalence without a smoke test.

Every completed outcome binds:

- attempt/candidate and exact snapshot/context identity;
- arrival provenance, proposer, role and bank version;
- evaluator Actor/payload/config identity;
- endpoint, horizon, remaining-time rule and seed;
- label, outcome class, interaction count and execution status.

The OR label means at least one declared evaluator succeeded. Completed failure, not-yet-evaluated and engineering error remain distinct. Downstream all-positive is a valid observed sample distribution; it does not require synthetic negatives or block envelope construction.

## 6. Process isolation and merge

Use fresh evaluator processes. The historical 600-candidate suggestion is an operational starting limit, not a validated guarantee. Validate small banks first and adjust downward if required. Do not run multiple long-lived GPU evaluators concurrently without a measured resource budget.

Cache reuse must compare the complete requested contract before returning. Merge must validate all files, exact requested catalog, non-overlapping complete global indices, consistent per-row identities/endpoint, seed scheme and labels before atomically publishing. Never archive a completed valid result as an incomplete attempt. Preserve failed/partial directories and link retries in the cost ledger.

Exit check: same small catalog serially and in two or more processes yields the same ordered state/evaluator/seed/endpoint/label outcomes under the declared numeric tolerance. CPU fixture tests are insufficient for this exit check.

## 7. Witness registry and metric views

Store exact witnesses independently of training support. A new registry version adds verified observations with source bank/attempt/evaluator IDs. Historical valid witnesses remain intact; invalidated evidence needs an explicit derived exclusion reason rather than silent deletion.

Build separate role-aware views for:

1. exact forward arrivals;
2. exact states with a landing witness;
3. root/full physical cells and longitudinal/phase sections;
4. TRAIN reset support;
5. each policy's realization and unique contribution;
6. attempted-without-witness and untested coverage.

Report both cumulative union and per-round gain. Attribute gains to new arrivals, new suffix successes on old arrivals, and overlapping contributions. Do not credit each probe with the entire union or sum overlapping cell counts. Fixed resolution/corridor and a stable comparison population are required for curves.

Existing physical metrics can be reused, but existing `continuation_viability_proven` fields are historical names for observed outcomes, not certificates. New schemas should use witness terminology and retain old readers without rewriting raw results.

## 8. Data-role isolation

Only TRAIN can drive adaptive discovery and train new probes. CALIBRATION is for optional threshold calibration; decision-used ACCEPTANCE remains development data. Final TEST/JCE/JEL remains sealed for this work.

Audit exact/context and declared-near overlap across all bank members, existing training supports, ancestor families and versions. Separate parent IDs alone do not prove independence. Preserve excluded counts and immutable raw roles. Historical bootstrap `test` labels have already been evaluated; reserve and audit the final distribution independently.

## 9. Probe training and admission

Before training, freeze support/weighting, initializer and normalizer rules, reset mixture, reward, phase balance, PPO transitions, seeds, exploration/labeling budget, comparison arms and stop rule.

Legacy `natural_reset_probability` implements `existing_phase_u_natural_reset`. It is not a synonym for the selected x=2.5 start. A new fixed-start training mixture requires explicit configuration/runtime support. Actor-only warm start must import Actor/normalizer and reset critic/optimizer as declared through the public entry point; historical CLI monkey-patching is not a completed public API migration.

Admission has separate stages:

- technical compatibility/freeze/smoke check;
- predeclared bounded evaluation as a new probe;
- valid witness retention independent of aggregate Actor coverage;
- optional active-bank scheduling based on incremental contribution and cost.

A candidate may add useful support despite lower old-panel coverage. Report that diagnostic; do not use it as a universal witness veto. Conversely, training completion alone does not prove discovery gain. Use existing pi_0/pi_1/pi_2 and a verified pi_3 pilot before spending another large PPO run.

No default sampling ratio, diversity loss or numerical admission threshold has been accepted as the method yet. The roadmap lists a controlled recipe to specify; do not invent a predeclared result after seeing outcomes.

## 10. Optional predictor

Predictors rank/diagnose only unless a separate controlled allocation study is declared. They never create arrivals, positive labels or Tube admission. A predictor trained for one bank must not silently change its target when the bank grows.

Before fresh labels, lock score order, exact candidate/context identity, catalog/protocol, model/normalizer, threshold and target bank. At audit time verify self-hash and every link before joining outcomes. Record the pre-outcome history; hashes alone do not establish chronology.

Compute tied-score-correct AP, explicitly distinguish AP from trapezoidal PR-AUC, report recall/FPR/class/group counts, and flag undefined metrics for single-class samples. Do not require ACCEPTANCE class balance to authorize TRAIN fitting. Predictor quality is not a prerequisite for predictor-free discovery.

## 11. Cost, comparisons and stop

A cumulative ledger links physical attempts and retries to consumed interactions, with no double counting of cached evidence. Charge bootstrap, acquisition prefixes/exclusions, all evaluators, PPO and development decisions; report wall time/hardware separately. Summaries of completed evaluators alone omit failed-run cost.

Predeclare matched-budget pi_0-only, fixed-bank uniform and iterative/growing-bank arms. Attribute benefit of probe growth separately from changed perturbation grids. If claiming a curriculum schedule effect, compare the same TRAIN support pooled once versus incrementally supplied.

Stop on the declared budget or predeclared marginal-gain rule, not when one Actor realizes the whole Tube. Coverage plateau under a finite bank/budget is not a physical boundary proof.

## 12. Current execution boundary

Correctness blockers and the missing new lifecycle prevent starting a formal new-probe run from the old selected-policy workflow. Follow [training roadmap](JIT_TRAINING_ROADMAP.md). Finish historical labels under their locked contract, then run an existing-probe discovery pilot under a new versioned protocol. Documentation editing itself executes no GPU work and does not change old manifests.
