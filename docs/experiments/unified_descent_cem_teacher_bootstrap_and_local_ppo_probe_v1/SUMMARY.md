# Unified Descent CEM-teacher bootstrap/local-PPO probe v1

- The run started from `ff3e3a1` with a clean tree. Bank, XML, action mapping, failure semantics, π_D and its normalizer remained immutable.
- Before gradient updates, corrected replay validation found a defect in the previous CEM authority: all 14 stored knot sequences are repeatable, but only 12/14 reproduce the saved selected summary. The prior `exact_replay` flag compared two repeats only.
- Candidate `da0679b1` was previously reported as 9→14 ticks (gain 5), but the saved residual knots now deterministically replay as 9→10 (gain 1). Thus only eight—not the required nine—candidates have gain at least four.
- The valid dataset contains 64 teacher samples (8 candidates × first 8 ticks) and 198 anchors: 86 teacher-tail, 48 non-significant in-bank, 32 canonical Landing and 32 natural-start. Command/action-space alignment is exact for this subset.
- Residual RMS for ticks 0–3 is `[0.1262, 0.1072, 0.1174, 0.0681]`; for ticks 4–7 it is `[0.0957, 0.1068, 0.1357, 0.1000]` in `[steer, drive, hip, knee]` order.
- The 64-sample representability check passes: closest-quartile opposite-label conflict is 17.19% under the declared 20% bound. This is not a state-unobservability stop.
- The source-authority contract fails first. No supervised optimizer step, candidate-fold validation, relabel, hidden-block unfreeze, held-out access or PPO transition was executed. PPO authorization remains false.
- Frozen π_D canonical Landing baseline is 81/96 Final-Recovery (84.375%), 15/96 roll failures and zero timeout. It is baseline evidence only because no student checkpoint exists.
- The code now rejects replay evidence unless repeat1, repeat2 and the selected CEM summary all match, including actions/features. The next experiment requires separate authorization to repair the missing ninth teacher or amend the fixed nine-candidate protocol.
- Verification passed: seven targeted tests and the final GPU local preflight (`298 passed`; one external JAXopt deprecation warning).
