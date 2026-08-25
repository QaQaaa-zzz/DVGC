# JIT Phase U Verification

## Active v4 pre-training verification — 2026-08-25

The v4 contract fixes the v3 false failure diagnosis without changing XML
physics: the observed `floor/rearwheel_collision` distance of approximately
`-0.014175 m` is normal compliant wheel support and is now raw telemetry only.
Prohibited body contact remains illegal. Apex is a monotonic nonterminal event;
the rollout continues to retained physical failure or the 200-tick horizon.

Final representative evidence contains the complete video plus separately
hashed pre/post-Apex NPZ segments. Formal training also persists raw JSONL
callbacks and synchronized PNG/NPZ/JSON learning curves for episode reward,
episode length, KL, PPO losses, policy standard deviation, and throughput.
Completed-v4 provenance binds both Apex segments to the full trace and binds
the plotted series to the raw callbacks.

Startup inspection of the first v4 attempt exposed a wrapper defect before it
could be accepted as a result: after the first 200-tick horizon, the default
cached reset restored only data/observations and leaked JIT episode/event info,
so subsequent logged episode lengths collapsed to one. That process was
stopped, its run was closed as `aborted`, and its checkpoints are prohibited.
The corrected training wrapper now requires a real reset and has a GPU
regression proving a two-tick terminal is followed by a fresh nonterminal
episode with event step one. `training_wrapper.full_reset=true` is part of the
exact v4 config identity.

The first full-reset retry then exposed a separate third-party reporting
behavior: Playground replaced terminal `episode_done/episode_metrics` with the
new reset's zero values. PPO state reset was correct, but episode curves could
not be produced, so `_retry1` was also stopped and closed as `aborted` without
checkpoint reuse. The final wrapper preserves those two logging fields across
the reset boundary and re-exposes them to Brax while keeping all JIT state
fresh. `preserve_episode_evidence=true` is independently config-bound and the
GPU regression requires terminal length two followed by fresh length one.

The new exact config uses seed `820301`, held-out seeds `940001..940008`, and
4,988,928 transitions. It is identity-incompatible with the v3 checkpoint and
must start fresh. Final command results and the launch commit are recorded only
after the complete verification round succeeds; the validated command results
are recorded below, while the launch commit is added after Git delivery.

Validated pre-launch evidence:

| Evidence | Result |
|---|---|
| v4 smoke config SHA-256 | `b178512e5f5555994c83dc4ecf8301d62f6c646afbc99b3cd1c9a08c97545a75` |
| v4 formal config SHA-256 | `d6b9476fc3097b8a1e9f7c1ca889f3bf2b93c9527210dcedda9e889e95eb0f43` |
| Complete non-GPU suite | 167 passed, 10 GPU tests deselected |
| Complete GPU suite | 10 passed, 167 non-GPU tests deselected |
| `JIT/scripts/local_preflight.sh` | exit 0 |
| Retained v1/v2/v3 formal provenance | all exit 0 |
| Repository compatibility | 1,136 passed; one pre-existing dirty-path failure outside JIT |

The sole repository-level failure is unchanged user work in
`tests/test_phase_u_launch_diagnostic.py`: it passes `mode=` to the separately
modified `dvgc/phase_u_launch_diagnostic.py`, whose current
`frozen_manifest_payload` does not accept that argument. Neither file is part
of this JIT change or commit.

## Completed v3 formal contract — 2026-08-25

The active source/config pair is `jit_phase_u_*_v3`. It changes only the joint
target semantics and the approved PPO/training schedule: hip and knee both use
keyframe-centered absolute targets; the formal run is a fresh 4,988,928-step
run with seed 820201. Its manifest must have a null parent checkpoint,
transition zero start, `fresh` resume semantics, and no restore argument.

Natural held-out evaluation and forced-airborne RSI diagnostics are separate
milestone panels. The provenance verifier independently checks both ledgers,
all trace hashes, and both final MP4/PNG/NPZ groups. RSI results cannot be
counted as natural-start success or promotion evidence.

Fresh verification results and the v3 smoke identity are recorded below only
after those commands finish successfully. Historical v1/v2 sections remain
for auditability and do not authorize loading their checkpoints.

The v3 formal run completed 4,988,928 training, 192 natural-evaluation, and 88
forced-RSI diagnostic transitions. All five natural panels failed 8/8 with
zero Apex; the final panel terminated on illegal wheel contact after two
transitions per rollout. All five forced-RSI panels reached Apex 8/8, but RSI
is diagnostic-only and cannot support promotion. Decision: `NO_PROMOTION`.

Verified v3 identities and engineering smoke:

| Evidence | Result |
|---|---|
| v3 smoke config SHA-256 | `5bc666658f335b0d394816d0d0b2ba117166c221682b9299dfa400e49302dd7e` |
| v3 formal config SHA-256 | `58e0302c82de0e267f28679dbe680fb5ef4a1538ffbfcd6ac63904cf6c2bc210` |
| Active fixed-rate smoke | 24,576 training + 43 diagnostic transitions |
| Final checkpoint | restored; payload SHA-256 `b2af29d94bb53ee32f402dc17457bb3cfca1b1790ab04578af7f36c9d87ce1b4` |
| Natural diagnostic | 44 states/frames; roll limit after 43 transitions |
| Complete non-GPU suite | 151 passed, 8 GPU tests deselected |
| Complete GPU suite | 8 passed |
| `JIT/scripts/local_preflight.sh` | exit 0, including retained v1 verification |
| v3 formal strict provenance | exit 0; 4,989,208 total interactions |
| v3 final checkpoint | restored; payload SHA-256 `1125d9edbec3cd31ec08bbe5cf88777e84974044ba980e3531fdb938f34596fd` |

The first fixed-rate block reported KL 341.1 because Brax updates the cold
observation normalizer before its fixed-rate SGD/KL calculation; policy means
and standard deviations remained bounded. Two additional isolated smokes
tested adaptive-KL with eight and one data pass. Both postponed cold-start
normalizer warm-up and exploded policy outputs/KL, so that experiment was
rejected and removed from the active code/config. These three ignored smoke
directories are engineering evidence only and are never checkpoint inputs.
The final audit additionally proves v2 incremental semantics remain truthful,
checkpoint sidecar identity is checked before pickle deserialization, natural
v3 traces contain reset-source zero, and forced-RSI traces contain reset-source
one. Final representative reports bind a legal seed and exact episode-NPZ
path/hash; diagnostic NPZ arrays are compared to that episode trace, MP4 frame
counts are decoded, PNG files are decoded, and MP4/PNG/NPZ types and distinct
paths are enforced. Independent review found no remaining Critical or
Important issue.

The complete v3 result, PPO comparison, milestone tables, per-tick reward and
action diagnosis, artifact hashes, limitations, and next-step decision are in
`docs/experiments/phase_u_absolute_4988928_seed820201_20260825/REPORT.md`.

## Historical v2 reward/RSI/diagnostics rebuild — 2026-08-25

The historical v2 configuration is `jit_phase_u_*_v2`. That round changed reward,
event, reset, observation, checkpoint, evaluation, video, and provenance
contracts without starting PPO training or consuming training/evaluation
transitions.

Verified active identities:

| Input | SHA-256 |
|---|---|
| v2 smoke resolved config | `6b0519344d8403d38556fb0a5fc4be8a6bd0cc70e0c077e4bca7f8a5c33fdc27` |
| v2 formal resolved config | `df565a03c0c8f40531a5ac57bd6c2c2674d9249ca52c31df143484f4ad484112` |
| authoritative XML | `e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192` |
| reference CSV | `612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f` |

Focused and complete evidence currently recorded:

- reward/config and formal-method drift suites: 47 passed;
- diagnostic/evaluation/video/formal/provenance focused suite: 33 passed;
- complete non-GPU suite: 134 passed, 7 deselected;
- complete GPU suite: 7 passed, 134 deselected;
- `bash JIT/scripts/local_preflight.sh`: exit 0, including static compilation,
  legacy-import scan, reference analysis, and retained v1 smoke verification;
- retained formal v1 run verification: exit 0, 998,400 training plus 904 fixed
  evaluation transitions, with all historical checkpoint/trace hashes intact.
- complete repository compatibility: 1,100 passed, one exact pre-existing user
  dirty-path test deselected, and one third-party deprecation warning.

The GPU group verifies reproducible 5% mixture selection over 1,024 resets,
exact RSI bounds, immediate reset-time jump signal, forced-natural evaluation,
fixed `(76,)/(106,)` observation shapes, JIT pytree stability, and finite
batched reset/steps. The diagnostic group verifies that one saved state becomes
exactly one composite video frame and one numeric sample, while the renderer
never calls `env.step`.

v2 representative evidence is stricter than v1: the verifier requires the
MP4, diagnostic PNG, aligned NPZ, matching paths, and SHA-256 hashes. v1
verification remains supported only so old evidence is auditable. It does not
make an old checkpoint compatible with the active v2 network/config identity.

No result in this section is learnability or promotion evidence. The user has
authorized one fresh 998,400-transition v2 formal run after source delivery;
its natural-start panels and RSI training metrics must remain separate.

## Historical v2 formal result

Run `phase_u_v2_formal_998400_seed820101_20260825` completed and passed strict
provenance verification:

- 998,400 training + 1,160 fixed natural-evaluation transitions;
- six checkpoint hashes and five eight-rollout panel ledgers closed;
- final checkpoint restored; final MP4/PNG/NPZ hashes matched;
- all five panels: 0 Apex, 0 height, 0 ascent, and 8/8 roll-limit failures;
- final panel: 0/8 reached the jump zone and all failed after 22 ticks;
- decision: `NO_PROMOTION`; no checkpoint is a trained Phase U expert.

The final video has 23 frames at 50 fps and the aligned NPZ has 23 samples,
77 numeric series, and no nonfinite value. See the complete analysis in
`docs/experiments/phase_u_reward_rsi_diagnostics_v2_20260825/REPORT.md`.

## Verified delivery

The historical v1 independent `JIT/` Propulsion-Ascent engineering stack passed its declared
first-delivery gate on 2026-08-24. This establishes environment, PPO update,
checkpoint, accounting, and saved-video integrity only. It does not establish
Phase U learnability or a trained expert.

## Immutable inputs

| Input | Verified SHA-256 |
|---|---|
| `assets/orange_bike_4kg_horizontal.xml` | `e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192` |
| `data/reference_jump.csv` | `612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f` |
| Resolved smoke config | `e96f4f4e35df041ffdf1525ee13fc356598917f5a600db15f9284c3e1d9ebed3` |

Host model tests additionally verified the 2 kg payload, hip/knee force limits
of `[-50, 50]`, action order `[steer, rear-wheel drive, hip, knee]`, and the
in-memory `0.005 s` simulation timestep.

## Fresh JIT preflight evidence

Command:

```bash
bash JIT/scripts/local_preflight.sh
```

Result: exit code 0.

- Static compilation and AST legacy-import scan passed.
- Non-GPU suite: `66 passed, 5 deselected in 15.87s`.
- GPU suite: `5 passed in 13.04s`.
- GPU coverage includes JIT reset/step, 50 ticks = 1 second, 1,024-environment
  finite rollout, outer wrapper-field preservation, and the real Brax wrapper
  extra-field contract.
- Offline reference analysis revalidated the retained CSV boundary.
- Successful run provenance verification passed.

No `jit_dvgc` runtime or CLI source imports the existing `dvgc` package.

## Repository compatibility

The repository-level preflight collected all 1,031 tests after JIT added an
isolated test-package boundary. Its complete run reported `1029 passed, 2
failed`; one JIT subprocess-path test was then fixed and passed independently.
The remaining failure is in the user's pre-existing modified
`tests/test_phase_u_launch_diagnostic.py`, where `frozen_manifest_payload()` is
called with an unsupported `mode=` argument. That JIT-external user work was
not changed.

A fresh repository suite with only that specific known user failure deselected
then reported `1030 passed, 1 deselected, 1 warning in 158.15s`. Thus JIT has
no remaining observed repository regression, while the unmodified full root
preflight is accurately recorded as not green because of the pre-existing
dirty-path mismatch.

## PPO engineering smoke

Successful run ID:
`phase_u_1024_one_block_20260824_seed820001_retry2`.

| Evidence | Result |
|---|---|
| JAX backend | GPU, NVIDIA GeForce RTX 4090 D |
| Parallel environments | 1,024 |
| Training transitions | exactly 25,600 |
| Brax evaluation transitions | 0 |
| Fixed evaluation transitions | 0 |
| Restored-policy diagnostic transitions | 31 |
| Total environment transitions | 25,631 |
| Final checkpoint restored | yes |
| Final checkpoint payload SHA-256 | `b4edf62e9b47b311df3893d6327f606320d755ae08a7e72c68995e9e6e10cb0d` |
| Captured states | 32 |
| Encoded video frames | 32 |
| Diagnostic terminal cause | `roll_limit` |

The restored-policy diagnostic reached the legal jump window but did not
liftoff, become stably airborne, ascend, or satisfy Apex. It ended on the roll
limit after 31 transitions; Apex success rate was 0 and physical failure rate
was 1. The final training KL was approximately 436.49. These values are
recorded as engineering diagnostics and are not promotion or learnability
evidence.

## Abnormal-attempt evidence

Two earlier run directories are retained and closed rather than overwritten:

1. The original run closed as `engineering_error` with 0 transitions after a
   Brax scan detected that wrapper-added `info` fields were not preserved.
2. `_retry1` closed as `engineering_error` with 0 transitions because Brax
   timeout bootstrapping required an explicit `time_out` adapter.

Both causes received focused regression tests before `_retry2`. No physics,
reward, observation, parallelism, or PPO-budget parameter was weakened for the
successful retry.

## Artifact and claim boundary

Run manifests, statuses, metrics, checkpoints, diagnostic states, and video
are under ignored `JIT/runs/` paths and are not Git delivery content. Rendering
replayed the 32 saved states without calling `env.step`, so it consumed zero
additional environment transitions and encoded no duplicated tail frames.

This delivery does not implement or claim a frozen Phase U expert, Phase D,
continuation labels, `V_up`/`V_down`, learned soft Tubes, unified Tube-RSI PPO,
or independent JCE/JEL certification.

## Formal Phase U run

Formal run `phase_u_formal_998400_seed820101_20260824_retry1` completed after
the engineering delivery above. Its strict verifier closed:

- 998,400 exact training transitions;
- 904 fixed-evaluation transitions and zero Brax-evaluation/diagnostic transitions;
- six identity-bound checkpoint payload hashes;
- five eight-label fixed panels and every trace hash;
- final checkpoint restore and deterministic finite inference;
- 22 saved/encoded frames for the final 21-transition representative trace.

All five panels returned 0 Apex successes and 8 `roll_limit` physical failures.
The final policy also regressed from reaching the window at 102,400/256,000 to
failing before the window at later milestones. It is not a trained expert and
must not be promoted. The complete modification rationale, PPO trends,
milestone tables, limitations, action diagnosis, interaction cost, and next-step
decision are in
`experiments/phase_u_formal_998400_seed820101_20260824/REPORT.md`.

The latest JIT preflight reported 106 non-GPU tests plus 5 GPU tests passing.
The fresh repository compatibility run reported 1,070 passed and only the
specific user-dirty relative-x manifest case deselected.
