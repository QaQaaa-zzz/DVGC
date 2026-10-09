# E1 prepared, NOT launched

Formal hold: P0/P1 completion and root launch authorization are required. Remaining layout divergence is not explained. Preparation does not change this gate or authorize training.

- Config: `configs/e1_stable_learner.json`, `launch_ready=false`.
- Source: transition9904128 actor+critic+source normalizer; fresh Adam. This prepared source candidate does not automatically follow a later development-selected checkpoint.
-20blocks x384environments x64control steps=491520learner/control transitions;1966080physics steps. No automatic extension to40.
- Frozen actor normalizer, lr3e-5, two SGDepochs,clip.1,entropy.002, gradcap.5, full-rollout KL(new||behavior) soft.02/hard.05 with complete block rollback.
- Task reward/reset/physics/action/input untouched. `learner_overrides` are explicit; `effective_learner_ppo` and its hash are checked against the merge. Runtime writes paired task+learner configuration/hashes in `combined_resolved_contract.json`. Conflicting effective declarations reject.
- Checkpoints at0/245760/491520. Contact-grounded evaluation atblocks10and20, three deterministic ground-natural repeats per checkpoint. Current ground reset ignores envseed for state selection, so these are repeatability episodes, **not independent initial states**. Each includes10contact calibration controls then at most400scored controls: total extra evaluation budget2460controls/9840physics steps.
- Best selection lexicographically uses strict1s stable landing fraction, stable hold, takeoff fraction; reward never selects. Ties retain earlier. `qualified=false` because repeatability panel is not robustness qualification.
- TensorBoard separate runlogdir, launcher checks current training reward tag through HTTP and records verification. No server or run started during preparation.

Root must first write a gate JSON with `config_sha256` matching the exact config bytes, `P0_complete=true`, `P1_complete=true`, `root_authorized_for_E1=true`. No such launch gate is created by this preparation.

Exact future command (only after gate):

```bash
cd /home/qy/ISAAC——SIM/worktrees/jump-landing-v2/sim2sim_v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/qy/ISAAC——SIM/.venv/bin/python -u e1_training_control.py --config configs/e1_stable_learner.json --gate runs/e1_authorization_gate.json --output runs/e1_source9904128_trial_0001 --tensorboard-port 6028
```

Status: `runs/e1_source9904128_trial_0001/status.json`; intended TensorBoard `http://127.0.0.1:6028` is **not currently launched/verified**. No video rendering.

Validation: four CPU integration tests verify unchangedtaskaxes, exactbudget/checkpoints/hash, conflicting learner payload rejection, explicit gate requirement, and physical ranking independent of return. Six learner tests and real-data disposable candidate gate tests pass. Native training-plus-child-evaluation integration remains unrun until authorization; competing Isaac instances/RTX initialization are not claimed tested by CPUtests.
