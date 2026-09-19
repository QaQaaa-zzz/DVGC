# Legacy Five-Stage Route

This note preserves the useful method history from the former root
`PROJECT_SUMMARY.md`. It is not current project truth and does not describe a
currently supported formal pipeline.

## Route overview

The legacy route decomposed the jump into Approach, Takeoff, Flight, Landing,
and Recovery. It used event-aligned downstream-entry labels, backward
construction, frozen-policy certification, proposal banks, and Tube-RSI-style
reset support. Chain/entry evidence and end-to-end Final-Recovery evidence were
tracked separately, and audit seeds were intended to remain separate from
training and proposal seeds.

## Why it was replaced

The sequential shared-Actor and five-stage controller stack accumulated many
route-specific launchers, migrations, and orchestration layers. Local entry
success did not by itself establish final unified-policy Final-Recovery, and
the route no longer matched the approved concise research direction. The
current direction is the two-phase Propulsion-Ascent / Descent-Recovery design
defined in `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`; that method is a design target,
not an implemented result.

## Reusable baseline and ablation assets

The authoritative XML, action mapping, environment, snapshot and bank formats,
policy bundles, rollout machinery, provenance/seed registry, certification,
and independent-audit infrastructure remain useful. Historical five-stage
policies and reports may be used only as provenance-matched baselines or
ablations. They must not be mixed across XML hashes, action mappings, policy
versions, bank roles, or seed namespaces.

## Recovery authority

The designated immutable source archive is `archive/pre-clean-20260731`.
The cleanup baseline is `main@b7bb815`; nearby legacy route evolution includes
`7497870` (certified local Apex entry alignment) and `b7bb815` (the corrected
Apex unified-RSI smoke/pilot commit). Git history is the authority for removed
source. No source, checkpoint, log, large JSON, or raw report is duplicated in
this documentation archive.

## Unsupported claims

This archive does not support claims that the five-stage route is current,
that a learned GRU estimator or learned soft tubes exist, that the two-phase
experts or `V_up`/`V_down` are implemented, or that local Chain/entry support is
a formal Tube or Jump Capability Envelope. Only a fresh independent audit of a
frozen final unified policy may support formal Final-Recovery/JEL claims.
