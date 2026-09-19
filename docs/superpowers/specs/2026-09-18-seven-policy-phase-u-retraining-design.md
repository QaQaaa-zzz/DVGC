# Seven-policy Phase U retraining and pulse evaluation design

## Objective

Preserve the four current policy identities, train three new descendants with the
historical Phase U `propulsion_ascent` recipe, and compare all seven policies under
four fixed three-step pulse timings. This is development evidence. It does not open
final TEST data or promote any policy to a certified envelope policy.

## Frozen source policies

Create an immutable source lock containing absolute paths and SHA-256 identities for:

1. `lineage_repair_0010` from the completed neighborhood lineage used by the current
   four-policy comparison;
2. `lineage_repair_0070` from nonfinite recovery attempt 2;
3. `fresh_rsi` at transition 8,000,000;
4. historical Phase U at transition 4,988,928.

Freezing means recording and verifying the existing configuration, checkpoint,
Actor, normalizer, payload, XML, observation, and action identities. Existing run
directories remain immutable; checkpoints are referenced rather than overwritten.

## Three Phase U descendants

Create one descendant from each of repair10, repair70, and fresh RSI. At transition
zero, each descendant restores its source Actor and matching observation normalizer.
Its Critic and optimizer are newly initialized. This is parameter warm start with an
optimizer reset, not continuation of the source trainer's Adam state.

Apart from initialization, training follows the resolved configuration belonging to
the historical Phase U run containing `transition_4988928`:

- phase: `propulsion_ascent`;
- 384 parallel environments and unroll length 64;
- batch size 16, 24 minibatches, and 8 optimizer passes per batch;
- learning rate 1e-4, clipping 0.2, discount 0.99, GAE 0.95;
- entropy cost 0.01, reward scaling 0.1, and gradient norm limit 0.5;
- episode horizon 400;
- full `initial_state` reset with initial forward velocity 2.0 m/s;
- airborne RSI probability 0.08 and the original Phase U RSI bounds;
- the complete historical Phase U reward coefficients and termination semantics.

No full-task unified RSI reward or 20/80 snapshot reset mixture is used in these three
training arms. Consequently, these descendants are specifically Phase U adaptation
results and may forget landing or recovery behavior. The later full-episode evaluation
measures that effect rather than assuming it away.

One Phase U rollout block is 384 x 64 = 24,576 transitions. Each arm therefore trains
for 610 blocks, or 14,991,360 transitions, the closest valid amount that does not
exceed 15,000,000. Seeds are distinct and declared before launch. Checkpoint and fixed
evaluation locations use the original Phase U fractional schedule, rounded to complete
rollout blocks, with a mandatory final checkpoint. Automatic extension is disabled.

The maximum training interaction charge is 44,974,080 transitions across the three
arms. Evaluation interactions produced during training are accounted separately in
each run report.

## Seven-policy evaluation

The final comparison contains:

1. frozen repair10;
2. frozen repair70;
3. frozen fresh RSI;
4. frozen historical Phase U transition 4,988,928;
5. Phase U descendant of repair10;
6. Phase U descendant of repair70;
7. Phase U descendant of fresh RSI.

For each policy, run 1,000 episodes at each fixed pulse onset. The four pulse windows
are control steps 0-2, 5-7, 10-12, and 15-17. A pulse contains three independently
sampled residual action vectors using the existing comparison distribution and legal
four-action ordering. Requested disturbance is exactly zero outside the selected
three-step window. The policy controls the vehicle from step zero through the full
episode.

For every onset and global episode index, all seven policies share the same random
draw and reset seed. This gives 7 x 4 x 1,000 = 28,000 disturbed episodes and a maximum
of 11,200,000 evaluation transitions at horizon 400. Batched execution may change
floating-point reduction details, so the report claims paired requests and common
protocol, not bitwise equality across different batch capacities.

## Outputs and acceptance checks

The experiment root contains the source lock, resolved training configurations,
status files, accounting, checkpoints, frozen descendant manifests, evaluation batch
receipts, per-episode rows, trajectory arrays, plot data, and a unified index.

Before a training launch, validate source hashes, XML, Actor observation fields,
action order, restored transition-zero Actor output, reset distribution, reward
coefficients, and the exact transition budget. A short engineering smoke may compile
and validate one update but is not counted as scientific training evidence.

Before accepting an evaluation batch, validate the requested pulse mask at array
level, including exact zeros outside the selected window; verify shared episode seeds
and pulse draws across all seven policies; retain every termination and nonfinite
outcome in the denominator; and reconcile requested, charged, active, and padded
interaction counts.

The main report gives, for every policy and onset, the official stable-forward-recovery
success rate with binomial confidence interval, failure composition, landing and
recovery timing, and trajectory distributions. Plots use common axes and separate
panels or density layers so seven policies remain distinguishable. Raw trajectories
are not discarded by display cropping. Any alternative success relabel is placed in a
separate counterfactual directory and never overwrites the official statistics.

## Execution and failure handling

The three training arms run sequentially on the available GPU to avoid memory
contention. The currently running STTW job is not stopped. Launch begins only after
the GPU is available. Every launch starts and verifies the JIT desktop error/completion
watcher required by `JIT/AGENTS.md`.

A failed arm or evaluation batch retains its logs and error state. Resumption may only
use an exact matching run declaration and checkpoint identity; changed configuration,
source, reward, seed, or protocol creates a new attempt directory. Completed source
runs and completed batches are never overwritten.
