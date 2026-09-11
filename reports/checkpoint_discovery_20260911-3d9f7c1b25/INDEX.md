# Frozen checkpoint discovery comparison — 2026-09-11

Authorized comparison of original pi6 versus32k/64k/128k checkpoints from the completed new training run. This is a bounded TRAIN development pilot, not final holdout or policy-bank promotion.

## Locked method

- Same fixed full x2.5 jump start; unchanged model, rewards and physics.
- Eight trajectories per proposer: all four action channels, signs±1,strength0.15,anchor x2.9,lookback0.15m. Shared acquisition seed9881101,shared continuation seed9881601.
- Real5cm trajectory frames, max64 candidates per trajectory; horizon400.
- One common frozen pi0–pi6 evaluator bank, first clean success stops evaluation; conflict/error/untested remains explicit. Diagnostic checkpoints are proposer-only and do not enlarge the evaluator bank.
- Same per-arm maximum1,500,000 interactions;6,000,000 global ceiling. One attempt, bounded subprocess groups, acquisition/evaluator timeout600s. All attempts/padding included. Reserved costs in the live ledger are ceilings until validated completion.
- Full fixed-schedule metrics are separate from conservative offline common-budget curves. The latter charge all acquisition/failure/padding overhead upfront and then atomic candidate labels in catalog order; they do not reconstruct adaptive evaluator wall-clock timing.
- Incremental training-inclusive surcharges: originalpi6=0;32k=32,076;64k=64,152;128k=128,225. Historical pi6/bootstrap cost is shared/excluded; this is not total lifecycle cost.
- Novelty excludes all9,296 historical witnessed root cells from completed all_proposers_v1. This preserves the historical label scope as a conservative exclusion set; it does not retroactively relabel it under new conflict quarantine.

## Evidence map

`plan.json` locks all inputs and code; tracked specification:`JIT/configs/checkpoint_discovery_pi6_20260911.json`; implementation commit5a8cc03.

`../checkpoint_discovery_20260911_inputs/` holds hash-bound diagnostic manifests and common bank/anchors; original checkpoints remain under the prior training directory. No model files are committed to Git.

Each proposer directory contains acquisition specification, exact command, reservation/exit receipts, complete logs, catalog, real prefixes and snapshots, projected points, the first-success plan, all evaluator attempt records and witness index, and analysis inputs. `cost_ledger.json` preserves stage charges; `process.log` is supervisor output.

On completion, `figures/` contains all-point CSV, metrics, equal-budget tables, complete cost curves, phase-separated observed geometry and PNG/PDF/SVG exports. No hull, filled gap or interpolation is treated as reachability evidence. `summary.json` records result/status and full comparisons.

Before launch:100 related CPU tests passed in4.59s; py_compile and diff checks passed. These test results are separate from actual GPU evidence.

## Completed result

Four complete matched8-trajectory schedules,809 candidate states. No engineering retries. Actual comparison interactions20,733,wall1127.341s (18.79min); zero PPO/final TEST. All stages and label receipts were verified before cost reduction. Implementation/config commit5a8cc03. Review copies of frozen metadata are in `review_inputs/`, with source/hash joins in `review_inputs_manifest.json`; original files remain in place.

| Proposer | Candidates | Witnessed / no-bank-witness / unknown | Full-schedule novel cells | Exploration interactions | Clean forward landings |
| --- | ---: | --- | ---: | ---: | ---: |
| pi6 | 202 | 202 / 0 / 0 | 146 | 4,225 | 8/8 |
| 32k | 187 | 187 / 0 / 0 | 186 | 4,058 | 7/8 |
| 64k | 204 | 183 / 20 / 1 | 182 | 6,502 | 6/8 |
| 128k | 216 | 206 / 5 / 5 | 205 | 5,948 | 7/8 |

At common exploration budget4,058,conservative atomic catalog-order replay yields132/186/132/165 novel witnessed cells (pi6/32k/64k/128k).32k is strongest on this narrow incremental-cell metric, not universally best control. At training-inclusive common budget4,225,additional32k/64k/128k training alone exceeds the budget; no training-cost advantage was demonstrated. This does not prove training can never amortize over larger exploration budgets.

Geometry was visually inspected: most projected structure overlaps. Maximum observed witnessed root z is0.628682/0.705164/0.664064/0.707406m, respectively, showing some higher observed states. Minimum clean forward trajectory peak root z is0.520797/0.559106/0.526544/0.537335m: this pilot did not improve the lowest complete successful peak. Low failed/unknown points do not establish a lower envelope. Discrete high-dimensional novelty must not be equated with geometric volume, certified cells, or a continuous feasible region.

`figures/phase_slices_005m.csv` groups only positive witness rows by proposer,phase,and floor(root_x/0.05), retaining observed z/vz minima/maxima and counts. Empty bins are absent; extrema may belong to different trajectories and must not be joined into executable paths. All778 positive rows are accounted for. Physical cell quantization is unchanged.

Decision supported by this pilot: pause additional PPO; retain pi6 and prioritize32k as a complementary proposer for follow-up verification. No automatic promotion was performed. Single seed/eight correlated trajectories are insufficient for a robust checkpoint ranking; independent repetitions and larger training-inclusive matched-cost controls remain needed.
