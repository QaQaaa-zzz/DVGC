# Unified Descent observation/history sufficiency audit v1

Result: **OBSERVATION_PIPELINE_AUTHORITY_FAILURE**.

- All 24 snapshots restore the saved actor observation and frozen action exactly only because `restore_snapshot` passes the saved `actor_observation` sidecar back into the environment, which directly overrides reconstruction.
- Independent reconstruction from the frozen physical state and saved PolicyState fails for 24/24 observations and 24/24 actions. Observation max error ranges 29.225--464.311; deterministic action max error ranges 0.00474--0.18995. Both reconstruction paths are internally repeatable, so this is a schema/alignment defect rather than nondeterminism.
- The saved `obs_history` equals frames 1--3 of the four-frame saved actor observation for 24/24 states. Reconstruction needs frames 0--2. The snapshot therefore stores post-current history even though `state.obs` was constructed from pre-update history plus the current frame.
- Phase/contact fields are present and the compatibility `delay_buffer` exactly aliases `phase_probs[None,:]`; it is not an actuator-delay FIFO. Normalizer and action order remain unchanged.
- Per protocol, V0/P0 reproduction, V1--V5 panels, alias pairs, privileged reconstruction, and the 244-pair action-conditioned diagnostic were not run. This version cannot justify an observation amendment. A new audit version must first correct the snapshot history/schema and must not inherit old policy/Tube/JEL evidence automatically. PPO and bootstrap authorization remain false.

