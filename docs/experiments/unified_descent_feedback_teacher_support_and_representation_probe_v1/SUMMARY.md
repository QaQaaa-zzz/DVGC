# Unified Descent feedback-teacher support probe v1

Result: **BRITTLE_OPEN_LOOP_TEACHER**.

- The fixed diagnostic set contains 24 snapshots, exactly three per authority candidate. The historical 11 accepted relabels cannot all fit this quota because one candidate owns four; the deterministic rule retained the maximum feasible 10/11, plus five audited rejects and nine snapshots from the previously excluded candidates.
- Batch-256 CEM discovery and batch-1 authority were separated after a uniform summary mismatch was detected. Authority uses two bit-exact batch-1 replays; CEM was not rerun.
- Local authority passed 12/24 snapshots. Only 3/8 candidates had at least two passing snapshots, below the required 16/24 and 6/8 gates.
- Among five retained previously rejected relabel states, local CEM found three authoritative corrections. Thus some old rejection was a teacher-support gap, but it was not broad enough to pass the candidate-balanced gate.
- The 12 authoritative snapshots showed no opposite successful action clusters. Real medoids remained valid at 12/12; action means at 12/12 and medians at 11/12 remained physically valid. The failure is therefore coverage, not widespread set-valued ambiguity.
- Receding-horizon oracle, H/L representation branches, held-out evaluation, final bootstrap and PPO were not authorized. PPO authorization remains false.

