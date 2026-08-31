# Downstream Refinement Recovery Design

## Status and objective

This design repairs the zero-interaction startup failure and closes bounded
prelaunch audit/resume gaps in the Iteration-0 downstream transition-band local
refinement. It does not change the policy, Tube, physics, rewards, actions,
strengths, duration grid, labeling horizon, readiness criteria, or scientific
claim boundary.

The reproduced startup failure is:

```text
ValueError: prior search upstream readiness drift
```

The completed coarse-search readiness payload includes exact
`candidate_count` fields (`571` upstream and `565` downstream), while the new
config omitted them. Every other declared value and the 1,136-label total
matches. The output directory is absent and no environment interaction was
spent.

## Selected approach

Use a bounded orchestration repair rather than weakening validation or adding
per-candidate checkpointing.

1. Strengthen the config and prelaunch declaration with the exact completed
   candidate counts. Keep exact readiness equality so completed evidence cannot
   drift silently.
2. Add one shared zero-interaction preparation/audit path. Both `--audit-only`
   and the real search use it to validate the frozen `pi_0`, completed coarse
   search, accumulated TRAIN labels, formal config, Tube identity, checkpoint
   payload identity, and deterministic downstream anchor selection.
3. Harden duration-level resume. A completed labeled duration is accepted only
   after its acquisition protocol/catalog, snapshot identities, label protocol,
   label rows, counts, policy identities, and hashes are revalidated. A
   completed zero-candidate duration is a valid checkpoint and is reconstructed
   without requiring nonexistent labels.
4. Keep an interrupted partial duration fail-closed. Re-running an incomplete
   candidate/label branch would double-spend and obscure interaction accounting;
   per-candidate resume would require a separate shared-label protocol change.
   The current run has no partial directory, so that larger change is neither
   necessary nor authorized here.

## Interfaces and data flow

`audit_downstream_transition_refinement(config_path)` returns a JSON-safe report
containing exact input identities, prior readiness/counts, Tube identity,
checkpoint payload identity, and the selected downstream anchor identities.
It constructs no MJX environment and consumes zero interactions.

`search_downstream_transition_refinement()` calls the same preparation helper
before creating the output directory. It then performs the existing GPU/runtime
checks and dynamic search. This prevents audit/run validation drift.

`_load_completed_duration(...)` returns one of:

- completed label rows plus a reconstructed duration report;
- no label rows plus a validated `no_candidates` report;
- a fail-closed exception for missing, non-completed, inconsistent, or tampered
  artifacts.

The main loop reconstructs readiness and cumulative interaction accounting from
these validated duration checkpoints, then starts only at the first absent
duration.

## Error and provenance handling

- Do not modify the completed coarse-search artifact to fit the new config.
- Do not relax dictionary equality or drop `candidate_count` during comparison.
- `--audit-only` must return nonzero on the same pre-runtime artifact drift that
  would stop a real search.
- Resume must never accept a label count without checking its protocol and
  policy identities.
- A zero-candidate duration contributes acquisition interactions and progress,
  contributes no labels, and allows later durations to continue.
- Existing output directories remain immutable; no automatic deletion or
  overwrite is introduced.

## Tests and gates

TDD regressions cover:

1. the real readiness schema, including exact candidate counts and count sums;
2. zero-interaction audit rejection of readiness drift and successful audit of
   the current local artifacts;
3. completed zero-candidate duration resume;
4. rejection of tampered completed duration protocol/labels;
5. unchanged terminal-clipping and TRAIN-only contracts.

After focused RED/GREEN cycles, run static compilation, the focused CPU suite,
the declared GPU tests, full `JIT/scripts/local_preflight.sh`, and the enhanced
`--audit-only`. Only a fully validated, committed JIT repair may precede new
refinement interactions.

## Claim boundary

This repair provides engineering/provenance integrity only. It does not train a
continuation field, construct `Tube_1`, train `pi_1`, establish JCE/JEL, or
certify a safe set.

## Post-launch repair extension

The committed run at repository HEAD `ad416e836e0548494f8a6d96ce14a66508f936c5`
completed durations 17 through 22, then failed while labeling duration 23 with
`Failed to allocate 32768 bytes on device 'cuda:0'`. Duration 23 acquisition is
complete (120 candidates, 2,760 interactions), but its label process persisted
only an engineering-error summary (113 completed in memory, 998 interactions)
and no label rows. Those 113 results cannot enter readiness.

The bounded recovery therefore adds a one-time, explicitly source-bound repair
resume. It preserves the original search protocol, records the old and repair
repository heads plus the failed-summary hash, reuses the completed acquisition,
and writes the replacement labels to `labels_retry_01/`. The failed 998
interactions remain separately visible and are included in total labeling cost.
No partial label is reconstructed or silently accepted.

The search also constructs reset/step JIT callables once and reuses them across
all acquisition and labeling durations. Previously each duration wrapped the
same bound methods anew, retaining multiple compiled executables until GPU
allocation failed. Shared callables preserve numerical semantics while bounding
compilation/cache growth.
