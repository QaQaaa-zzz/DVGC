# Descent feedback correction transfer and support geometry audit v1

Result: **ACTOR_OBSERVATION_INFORMATION_GAP**.

- Authority accounting reproduces all declared totals: 4/10 historical accepted, 3/5 historical rejected, and 5/9 newly frozen states pass, totaling 12. Six historical accepts do not reproduce. Candidate pass counts are exactly `[3,0,1,2,1,1,1,3]`.
- The 12 immutable medoid corrections produced 244 eligible double-replay pairs: 12/12 diagonal, 3/18 same-candidate off-diagonal, and 40/214 cross-candidate transfers reached gain >=2 without a new failure type.
- Cross-candidate transfers reach six target candidates. The transfer graph has one seven-candidate weak component; only unsupported `173ee307` is isolated. Robust-core corrections transfer to frontier/sparse states in 14/71 eligible pairs; the reverse direction succeeds in 6/30.
- Candidate-grouped actor-visible linear diagnostics yield balanced accuracy 0.667, precision 0.700, recall 0.583 and fail the fixed separability gate. Privileged features yield 0.750/0.800/0.667 and exceed their permutation p95. Fixed kNN does not contradict the interpretation but is weaker.
- The corrections are not merely pointwise, but the current actor observation/history cannot reliably identify their support. Under the current observation contract, the CEM action-regression/bootstrap route is closed. Only a separate observation/history sufficiency audit is permitted; no teacher expansion, network training, held-out evaluation, Tube claim, or PPO is authorized.

