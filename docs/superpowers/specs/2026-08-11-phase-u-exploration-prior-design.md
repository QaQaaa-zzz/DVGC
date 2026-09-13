# Phase U Exploration-Prior Design

## Evidence and root cause

The collision-valid 512-environment Phase U run stopped at 256,000 training
transitions on the required three-window held-out physical plateau. Fixed
evaluations at 0, 102,400, and 256,000 reached the legal jump window in 24/24
rollouts but achieved 0/24 liftoff, clearance, or Apex success. All terminated
with `takeoff_missed_liftoff_deadline`; none had roll, pitch, illegal-contact,
nonfinite, or action-saturation failure.

The policy distribution remained near its initial scale: approximately 0.05
at every checkpoint. Deterministic hip and knee actions remained near zero, and
the knee target stayed at its compressed upper limit of 2.5 radians. This actor
head was originally designed as a neutral, low-variance prior for Landing
recovery and is currently reused by Phase U.

A bounded natural-start action diagnostic separated exploration from physical
infeasibility. It used zero action before the legal window and scanned 25
constant hip/knee pairs only after the deployed jump latch. Hip action at or
above 0.5 produced liftoff in 10/10 cases within one or two control ticks;
nonpositive hip action produced liftoff in 0/15 cases. The diagnostic consumed
710 environment transitions and no training transitions. Strong constant
actions later violated pitch constraints, so they are evidence of action
authority, not successful controllers.

The root cause is therefore that a useful launch action is effectively outside
the Phase U exploration distribution. At initial standard deviation 0.05, a
hip action of 0.5 is approximately a ten-standard-deviation event before the
tanh transform. PPO receives no liftoff examples from which to learn the
bounded ascent reward.

## Alternatives

1. Phase-specific initial exploration scale. Keep the global Landing/default
   prior at 0.05 and set the Phase U actor's initial standard deviation to
   0.25. This is the selected approach because it changes the identified cause
   while preserving task semantics.
2. Raise the global actor standard deviation. This is rejected because it
   would silently change Landing, legacy PPO integrity, and future unified
   policy behavior.
3. Add a hip-action bonus or reference-action tracking. This is rejected
   because it would shape a particular control command instead of downstream
   physical progress and would violate the approved no-imitation/no-reference-
   tracking boundary.

## Public contract

The stable training configuration gains an explicit
`policy_initial_action_std`. The runtime network factory accepts the value as a
keyword-only parameter, validates it as finite and strictly between 0.001 and
1.0, and otherwise retains the existing 0.05 default. Phase U smoke and formal
training pass 0.25 to both PPO construction and Orbax network metadata.

Only initial exploration changes. The deterministic initial mode remains
exactly zero. Reward, reset, optimizer, network layer sizes, observation,
episode horizon, XML, mass, force limits, action mapping, event contracts, and
Gate Pause thresholds remain unchanged.

## Budget and evaluation

Formal Phase U training already consumed 291,200 transitions in the
64-environment run and 256,000 in the 512-environment run. The cumulative
expert-training total is 547,200. Under the authorized one-million-transition
ceiling, the new run may use at most 448,000 transitions: 35 complete
12,800-transition blocks, bringing the cumulative total to 995,200.

The new fixed checkpoints are 0, 102,400, 256,000, and 448,000. Each checkpoint
uses the unchanged eight held-out seeds and reports physical outcomes. The run
still pauses at 256,000 if the first three evaluations form another physical
plateau. Candidate acquisition remains gated by independent Apex-success
parents; no snapshot, continuation label, `V_up`, or Tube is declared merely
because liftoff occurs.

Before the formal run, a single-block PPO smoke must validate finite sampling,
update, checkpoint metadata, fixed evaluation, accounting, and failure-video
capture. Smoke and diagnostic transitions remain separately accounted from
formal expert-training transitions.

