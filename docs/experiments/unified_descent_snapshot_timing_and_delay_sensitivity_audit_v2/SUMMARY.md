# Unified Descent snapshot timing and delay sensitivity audit v2

The saved actor tensor is a valid `authoritative_logged_actor_input`: it is
the exact `state.obs` consumed by the online frozen policy at physical state
`t`.  Logged replay reproduces all 24 deterministic actions exactly and
reproduces the original 12/12 local-authority diagonal.  The legacy snapshot
also stores the already advanced history `[t-2,t-1,t]`; treating that value as
pre-update history and appending frame `t` again creates a hybrid input that
the online policy never saw.  The timing defect is therefore
`HYBRID_STATE_RECONSTRUCTION_ERROR`, not proof of a policy information gap and
not a fixed one-frame delay in the online controller.

The frozen complete-packet delay experiment used no training, CEM, relabel or
held-out state.  Local authority counts are L0/R0/D1/D2/J12 =
12/10/9/6/9.  Baseline failures are L0 24 pitch; R0 23 pitch + 1 roll; D1 23
pitch + 1 roll; D2 and J12 24 pitch.  Relative to L0, initial action RMS/max
difference is 0.04475/0.18995 for R0, 0.05681/0.20364 for D1, and
0.07975/0.28180 for D2.  The hip channel is the largest contributor.

The exact 244 compatible correction pairs remain comparable.  L0
diagonal/same-candidate/cross-candidate successes are 12/3/40; D1 gives
9/6/69; D2 gives 6/4/61.  Thus transfer does not disappear with delay, but
pointwise authority and candidate support strata are delay-sensitive.  D1
meets the preregistered tolerance rule; D2 does not (50% authority retention
and four candidate-layer changes).  The final causal classification is
`DELAY_SENSITIVE_FEEDBACK_SUPPORT`.

Old online rollouts remain `empirical_online_evidence`; logged-input replays
remain `logged_observation_replay_evidence`.  Legacy independent restoration
is `independent_reconstruction_unverified`, and Tube/JEL certification remains
`certification_pending_timing_audit`.  The 24 states do not need recapture for
logged evidence, but future self-contained certification snapshots must use
the explicit v2 pre/current/post history schema.

PPO and bootstrap authorization remain false.
