# Phase U numerical recovery and adopted-conversion reward

User authorized implementation, verification and bounded continuation on 2026-09-26.

- [ ] Replay failed round 0089 with first-invalid-gradient capture. Preserve original science inputs and failures; diagnostic ceiling 128000 PPO + 1600 panel interactions, one attempt.
- [ ] Reproduce root cause on captured inputs, write failing regression, implement minimal project-local numerical fix; do not edit installed shared Brax or hide NaNs.
- [ ] Verify the failed training workload with the fix, preserve actual and reserved retry costs.
- [ ] Add opt-in adopted-conversion bonus: local conversion 0.2, final adopted successor +1.8, novelty <=0.05, repeat -0.1, trained failure -0.5, physical pulse failure -5. Unknown remains excluded. Final nominal support check must precede bonus and explorer update. Existing reward defaults remain identical.
- [ ] Repair recovery preparation for initial failed lineages lacking recovery.json. Resume 89->100 in new directory with immutable completed artifacts. Keep original reward for completing the original matched comparison; new reward is a separately declared protocol, never relabel old 89 rounds.
- [ ] Rebind waiting random queue to the actual recovery completion, preserve original failed queue and verify GPU gate/watcher.
- [ ] Run matched development evaluation of available final models after completion; finite queued budget and no final TEST. Report actual completion versus waiting/running, commit and push only relevant changes.

No physics, success criterion, initial pose, perturbation semantics or historical artifacts change. A GPU desktop process is not a competing training task; never kill external processes. Numerical retries remain bounded; do not promise arbitrary simulations cannot fail.

## Execution receipt

Code2c58dae;100 CPU tests pass. Independent code review completed; explicit optimizer gating and repeated recovery were corrected. Diagnostic replay completed128000+184 interactions without reproducing the NaN, so root-cause task remains unresolved. Recovery CLI preparation and hash locks validated on actual89-round artifacts. Formal GPU validation and continuation are queued under idle gates, not yet completed. Original random queue supervisor was safely replaced while waiting with zero reservations. Optional adopted reward remains separately opt-in and unlaunched. Final-model matched evaluation remains pending model completion.
