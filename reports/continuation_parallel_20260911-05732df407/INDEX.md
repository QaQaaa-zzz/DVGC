# Parallel continuation engineering evidence — 2026-09-11

No PPO, new acquisition or final TEST. Capacity worlds replicate16 existing TRAIN snapshots; they are not new discoveries or independent trials. Full-cost scope is this engineering task only.

- [Summary and complete cost ledger](summary.json):3,384,811 charged interactions, including inactive lanes and interrupted/memory-stopped reservation ceilings.
- [All capacity measurements](capacity_measurements.csv), [plotting data](timings.csv), [PNG](timings.png), [PDF](timings.pdf), [SVG](timings.svg), [figure identities](figure_manifest.json).
- [4096→8192→16384 capacity sweep](capacity4096/summary.json):initial implementation; substep capacity checks were added afterward.
- [16384 with every-physics-substep capacity checks](capacity16384_checked/summary.json):completed;14,522MiB observed peak;1.908s execution;157,227 useful steps/s;no observed substep capacity saturation.
- [24576 protected stop](capacity24576_checked/summary.json):stopped at22,542MiB;no valid throughput result;1,572,864 reservation charge. This is a conservative memory-limit stop, not observed CUDA OOM.
- [Same-run16-candidate execution comparison](paired16/summary.json):serial24.41s full process;vectorized16 18.11s. Endpoints agree, termination ticks differ;strict execution-equivalence gate remains unpassed.
- [Final checked label-path diagnostic](checked_labels16_comparison.json):16 endpoints agree;one candidate38→39 ticks. The catalog path was relative versus absolute;derived comparison first verified all other protocol fields and resolved-path identity. Raw protocols and labels unchanged.
- [123-candidate comparison interruption](paired123/interruption.json):serial123 completed;the user redirected work to4096-world capacity before the vectorized attempt was finalized. Current charge includes its full49,200 ceiling.
- [Four existing landing/failure conflicts](event_conflicts.json):preserved receipts. Contact detection does not exclude roll/pitch/backward failure;terminal classification does. Acquisition/continuation prioritizes contact, while training masks physical failure. This source audit does not identify substep order for historical receipts and does not reclassify them.

Existing production campaigns retain their serial contracts. New explicit evaluator calls support`--execution-backend vectorized --batch-size N` with chosen shard count. Large capacity success does not prove full scientific label equivalence or full-campaign speedup. Snapshot loading/process startup remains a bottleneck;do not duplicate candidates just to occupy VRAM.
