# v1 to v1.1 local migration audit

Baseline: c98701a1a84363435977736728d6d94229d427d9. No generative_bridge package,
CLI or tests found in main checkout or registered worktrees. This is an additive
implementation around existing components, not a rewrite of a local v1 package.

| Audit | Existing evidence | Required delta |
|---|---|---|
| teacher gate | pulse_exploration.run uses pending and not fixed_policy | preserve original support independently |
| pending support | candidate_support_view balances phases and ancestors before training | lock membership/weights before search |
| empty demo | policy_distillation.load_dataset rejects empty arrays | separate optional loader; never call legacy loader for empty |
| student evaluation | after_learning evaluates all pending, suffix frames omit raw preobs | opt-in real pre/post recording |
| G feedback | no G data path | exact adoption/role/identity admission, actor_only |
| labels | initial_label, label and final successor_adopted already separate | teacher status and result matrix kept separate |
| reward | discovery_reward has main requested weights; adoption checked for positive bonus | new strict v1.1 adapter, historical path unchanged |
| safety | unified_formal.guard_ppo_updates wraps trainer | single combined loss within same guard |
| initialization | load_frozen_actor_restore_params, fresh critic/optimizer | reuse; never call fresh distill |

Source and live run paths are inspected read-only. current_source.json in the IDE is
an audit candidate, not authorization to use that checkpoint for a new experiment.

User subsequently confirmed there is no prior v1.0 implementation. The migration
name follows the requested deliverable, but the actual work is a new opt-in bridge
package around the existing JIT components. No missing old implementation is inferred.

Live audit snapshot: declared IDE pointer resolves to lineage_repair_0093; this is
an inspection source, not automatically a new experiment initializer. Actor 153352,
Critic 159233, exact frozen tree hashes checked. Source pending_fraction is 0.5,
not the older 0.25 example. Preserve actual source settings. Audited round0095 has
162 original pending, 80 student-success/82 student-failure evaluation labels. Its
promotion result and trace hashes are in source_audit.json. Missing raw preobs means
zero currently admitted G windows from this audited subset; no recapture launched.

Incremental legacy edits:
- pulse_exploration: new reward contract routing only when explicitly selected;
  opt-in recording flag for the existing all-pending after_learning evaluation.
- pulse_exploration_runtime: default-disabled raw pre/post recording and identity-bound
  H16 prefix substitution in canonical single-Actor continuation; no reset at handoff.
- training/formal: optional single joint loss adapter under the existing numeric guard;
  incompatible nested action-retention wrappers are rejected.

Unchanged: original reward module, Actor architecture, XML/action mapping, pending
support construction/caps/weights, reset implementation, old configurations and runs.
