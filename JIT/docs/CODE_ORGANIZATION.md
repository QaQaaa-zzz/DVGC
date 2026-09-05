# JIT code organization and migration map

Active objective: [empirical-envelope project](../../PROJECT.md). Current implementation gaps: [review](JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md). New schemas below are design requirements, not shipped capabilities.

## Placement

Durable scientific behavior belongs in `JIT/src/jit_dvgc/`; CLIs in `JIT/cli/` parse arguments and call modules; tests in `JIT/tests/`; configs in `JIT/configs/`; guidance in `JIT/docs/`. Extend existing capabilities instead of iteration-specific source copies.

## Existing components and next responsibility

| Area | Existing owner | Current role / required change |
| --- | --- | --- |
| Bootstrap/reset support | `soft_tube.py`, phase environments and value modules | Preserve value-weighted historical S0; do not label all rows as witnessed capability |
| Unified training | `unified_formal.py`, `training/formal.py`, `cli/train_unified_from_pi0.py` | Consolidate warm-start in public implementation; explicit fixed-start versus legacy natural reset |
| Frozen policy identity | `unified_policy_freeze.py`, `unified_training.py`, `checkpoint.py` | Reuse for technical probe eligibility; do not equate freeze with scientific selection |
| Exact snapshot | `unified_envelope_snapshot.py`, `unified_continuation_labels.py` | Separate physical/context identities; verify restoration/time semantics |
| Centerline | `analysis/nominal_jump_centerline.py` | Fixed real pi_0 trajectory as coordinates |
| Causal arrival | `acquisition/causal_jump.py`, `causal_frontier_protocol.py` | Reuse per-proposer primitive; remove training-support membership as witness veto in new mode; add supervisor/namespace |
| Family outcomes | `policy_family_landing.py`, `unified_continuation_shards.py` | Versioned members, exact requested identity, row checks and safe publishing |
| Empirical geometry | `analysis/causal_jump_capability.py`, `analysis/capability_tube.py`, `analysis/jump_tube_view.py` | Separate exact witnesses, cumulative cells, roles, physical projections and marginal attribution |
| Training Tube updates | `iterative_tube.py`, `tube_rsi.py` | Keep training sampling independent of evidence admission and coverage deduplication |
| Role isolation | `iterative_frontier_protocol.py`, isolation CLI | Extend cross-proposer/ancestor/bank-version isolation |
| Optional predictor | `family_landing_predictor.py` | Verify score lock and target bank; tied AP; no admission labels |
| Legacy selection | `analysis/capability_progression.py`, `iterative_acceptance_gate.py`, selection CLI | Close invalid historical evidence route; preserve legacy Actor comparisons as diagnostics |
| Workflow | `workflow/iteration_loop.py`, `cli/prepare_iterative_envelope_workflow.py` | Migrate new protocol to bank/registry/declared-budget decisions rather than single selected successor |

`probe_bank.py` now owns the new bank, multi-catalog label plan, fresh-process supervisor, attempt reservations and observation index. `evidence_integrity.py` supplies shared CPU-only endpoint/row/protocol checks. `cli/probe_bank.py` is the new entry point. Cumulative physical cells and complementary PPO remain future work; see [implementation status](JIT_PROBE_BANK_IMPLEMENTATION_20260905.md).

## New lifecycle requirements

1. Freeze task/start/centerline/roles/budget and bank membership.
2. Capture per-proposer arrivals with complete provenance.
3. Evaluate exact suffix outcomes with immutable attempted/completed/error status.
4. Publish validated witness registry and separate role/physical/Actor views.
5. Account attempts and retries once in the cumulative ledger.
6. Train/admit new probes under a declared recipe; technical eligibility, evidence validity and marginal utility are separate decisions.

Old selected-policy artifacts remain readable; they do not automatically authorize new discovery. Legacy scans retain their fixed family and signatures.

## Required verification

- Wrong catalog/Actor/payload/seed/horizon/endpoint/cache request must refuse.
- Mixed identity rows, duplicate/missing shard indices and incomplete files must refuse before publication.
- Existing S states can gain missing witnesses without inflating physical counts.
- Equal physical coordinates with distinct required contexts remain distinguishable.
- Prefix/capture/restore/suffix and serial/sharded paths agree on a small real runtime bank.
- Public warm-start actually routes Actor/normalizer and resets critic/optimizer as declared.
- Role isolation spans all sources and versions.
- Predictor score drift and target-bank drift are rejected; tied AP is order invariant.
- Old mixed-endpoint gates cannot become fresh scientific eligibility evidence.

Fixture/CPU/source checks do not replace real checkpoint/GPU/rollout checks. Document validation scope accurately.

## Evidence storage

Keep lightweight run summaries and reproducible source/config identities in Git. Large checkpoints and catalogs may remain external with a resolvable artifact index. Preserve old absolute paths as historical records and add a materialization map rather than rewriting provenance. Raw historical JSON must not be edited to retrofit the new protocol.
