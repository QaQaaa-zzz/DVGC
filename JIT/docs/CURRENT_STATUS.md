# Current JIT status — 2026-09-01

## pi_1 formal Tube_1 PPO completed

The authoritative local completed run is:

`JIT/runs/pi_unified/pi_1_tube1_natural10_10009600_seed821101_20260901_retry01`

It used `JIT/configs/pi_unified_iter1_tube1_natural10_retry01.json`, whose
pre-run canonical config SHA-256 was
`987ef5d31661482fd0bc05cea566c177d83ecd00ae3028ff0e8bb2ed462b7901`.
The training support was Tube_1 manifest
`817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`
with 3,119 entries (222 retained Tube_0 core + 2,897 policy-conditioned
expansion states).

The operator-reported formal report completed exactly 10,009,600 requested
training transitions and restored the final checkpoint successfully. Scheduled
checkpoints were written at 0, 1,024,000, 2,508,800, 5,017,600, 7,500,800, and
10,009,600 transitions. All five nonzero TRAIN-only milestone panels completed,
using 2,838 diagnostic environment interactions in total. Brax evaluation was
disabled. TEST data, validation data, and expert-policy switching were not used.
The reset mixture remained 0.1 natural / 0.9 Soft Tube, matching pi_0.

Final reported optimization metrics include KL mean 0.1033896953, policy loss
0.02139844, value loss 4.9664841, total loss 4.99119, approximately 54,058
training steps/s, and 220.824 s reported training walltime. These metrics are
training diagnostics only; they do not establish capability-envelope gain.

## Preserved failed attempt

The first pi_1 attempt
`pi_1_tube1_natural10_10009600_seed821101_20260901` remains a preserved
`engineering_error` artifact. It reached 1,024,000 training transitions and
wrote that checkpoint. Its first TRAIN panel actually consumed 449 environment
interactions, but the terminal `status.json` recorded diagnostic interactions
as zero because the failure happened after rollout, while loading mixed Tube
snapshots for plotting, before the panel callback returned. Do not rewrite or
delete that historical artifact.

The plotting path now supports both `handoff_snapshot_v1` and
`jit_unified_envelope_snapshot_v1`, and the production formal-training namespace
adds a zero-interaction static Tube preflight.

## Scientific claim boundary

pi_1 training completion does **not** by itself prove an expanded jumping
capability envelope. pi_1 is not `pi_unified_star`, Tube_1 is not a certified
safe set, and final JCE/JEL is not authorized yet.

The next authorized sequence is:

1. freeze the exact completed pi_1 checkpoint as the iteration-1 envelope
   authority;
2. run the core-preservation gate;
3. run the boundary-gain gate on disjoint TRAIN/iteration evidence;
4. only if both gates pass, claim empirical envelope expansion and decide
   whether another Tube/policy iteration is justified;
5. keep untouched TEST/final JCE/JEL evidence isolated until a final frozen
   policy is selected.

This file records the operator-reported terminal result. Exact actor, critic,
normalizer, checkpoint-payload, and freeze-manifest SHA-256 identities must be
derived from the local completed checkpoint by `freeze_unified_policy.py`; they
are not guessed here.
