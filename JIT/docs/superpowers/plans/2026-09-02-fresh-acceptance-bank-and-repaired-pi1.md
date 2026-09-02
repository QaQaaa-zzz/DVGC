# Fresh Acceptance Bank and Repaired pi1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the two pre-training acceptance-bank engineering defects, lock a fresh frozen-pi0 negative bank from the existing immutable acquisition, and launch exactly one repaired iteration-1 pi1 run if every gate passes.

**Architecture:** Keep provenance validation in the existing labeling CLI and bank readiness/locking in the existing `jit_dvgc.continuation` API. Treat semantic predeclaration identity, engineering failure, and scientific readiness as separate contracts; reuse the completed 659-candidate acquisition and never regenerate it.

**Tech Stack:** Python 3, pathlib/json/hashlib, pytest, JAX/MJX, existing `jit_dvgc` APIs, Git.

## Global Constraints

- Work only on `agent/two-phase-soft-tube`; do not modify `main`.
- Preserve the five known unrelated dirty paths and never reset, clean, stash, rebase, or overwrite them.
- Use only `/home/qy/mujoco_playground/.venv/bin/python` with `PYTHONPATH=$PWD/JIT/src`.
- Do not add a production module; modify the existing labeling CLI, continuation API, and existing tests.
- Do not repeat or overwrite `pi_0_repair_acceptance_supportwide_acquisition_20260902`.
- Preserve Tube1, seed 821101, fresh initialization, PPO budget 10,009,600, reward, physics, XML, task, action semantics, 0.1/0.9 reset mixture, 0.5/0.5 phase mixture, 0.5/0.5 Tube core/expansion replay, validation isolation, and TEST isolation.
- Stop before repaired-policy training on any integrity, compile, test, snapshot, GPU/runtime, interaction-accounting, readiness, or existing-output failure.
- Do not begin `C^1`, Tube2, or pi2 during this plan.
- Repository preflight must discover unified formal configs by schema; it must not hardcode pi1, pi2, Tube1, Tube2, retry, seed, or checkpoint names.

---

### Task 1: Semantic Predeclaration-Copy Validation

**Files:**
- Modify: `JIT/cli/label_unified_continuations.py`
- Modify: `JIT/tests/test_continuation_labels.py`

**Interfaces:**
- Consumes: source predeclaration payload/SHA, acquisition `predeclaration.json`, and `anchor_audit.json`.
- Produces: `_validate_repair_predeclaration_binding(predeclared: dict, predeclared_sha: str, copied_path: Path, audit_path: Path) -> dict` returning the validated audit.

- [ ] **Step 1: Add a CLI-loader helper to the existing test file**

Use `importlib.util.spec_from_file_location` to load
`JIT/cli/label_unified_continuations.py` without constructing a GPU runtime.

- [ ] **Step 2: Write the failing canonical-copy regression tests**

Create source and copied JSON files with different whitespace/key ordering but
equal parsed payloads. Assert the helper accepts them when the audit records the
source file SHA. Add independent rejection assertions for a semantic field
change, an incorrect audit source SHA, and a copied protocol whose canonical
SHA does not match `expected_protocol_sha256`.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
export PYTHONPATH="$PWD/JIT/src"
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  JIT/tests/test_continuation_labels.py -k predeclaration
```

Expected: FAIL because `_validate_repair_predeclaration_binding` does not exist.

- [ ] **Step 4: Implement the minimum validation helper**

The helper must perform, in order:

```python
if audit.get("predeclaration_file_sha256") != predeclared_sha:
    raise ValueError("repair acceptance anchor audit/source predeclaration SHA-256 drift")
copied = _read_json(copied_path)
if copied != predeclared:
    raise ValueError("repair acceptance acquisition/source predeclaration semantic drift")
expected = str(predeclared["expected_protocol_sha256"])
if str(copied.get("expected_protocol_sha256")) != expected:
    raise ValueError("repair acceptance copied expected protocol SHA-256 drift")
for name, payload in (("source", predeclared), ("copied", copied)):
    if _canonical_sha256(payload["protocol"]) != expected:
        raise ValueError(f"repair acceptance {name} canonical protocol SHA-256 drift")
```

Call it before snapshot validation or GPU runtime construction. Remove the raw
byte-hash equality check between the copied and source JSON files.

- [ ] **Step 5: Run the focused tests and confirm GREEN**

Run the command from Step 3. Expected: all selected tests PASS.

- [ ] **Step 6: Commit the code and tests together**

```bash
git add JIT/cli/label_unified_continuations.py JIT/tests/test_continuation_labels.py
git commit -m "Fix repair predeclaration copy validation"
```

---

### Task 2: Persist Phasewise Readiness Failure

**Files:**
- Modify: `JIT/src/jit_dvgc/continuation/__init__.py`
- Modify: `JIT/tests/test_continuation_labels.py`

**Interfaces:**
- Produces: `AcceptanceBankReadinessError(ValueError)` carrying an `audit` mapping.
- Produces on readiness failure: `acceptance_readiness.json` beside the requested `acceptance_bank.json`.
- Preserves on success: `lock_negative_acceptance_bank(...) -> dict[str, Any]` and a normal locked bank.

- [ ] **Step 1: Update the insufficient-readiness test to exercise the lock API**

Use temporary `labels.json` and `catalog.json`, monkeypatch the existing
`load_soft_tube` symbol to return a TRAIN-only target Tube, and assert:

```python
with pytest.raises(AcceptanceBankReadinessError):
    lock_negative_acceptance_bank(...)
assert labels_path.read_bytes() == original_labels
assert (tmp_path / "acceptance_readiness.json").is_file()
assert not (tmp_path / "acceptance_bank.json").exists()
```

Verify the status is `not_ready_before_repair_training`, the selection rule is
`all_baseline_continuation_negative_candidates`, every required phase field is
present and correct, and all training/validation/TEST/final-evaluation flags
are zero/false.

- [ ] **Step 2: Add a passing lock regression test**

Provide at least the declared minimum unique negative states and parent groups
in both phases. Assert `acceptance_bank.json` is written with status
`locked_before_repair_training`, `acceptance_readiness.json` is absent, and its
entry count/readiness audit match all selected negatives.

- [ ] **Step 3: Run both lock tests and confirm RED**

```bash
export PYTHONPATH="$PWD/JIT/src"
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  JIT/tests/test_continuation_labels.py -k acceptance
```

Expected: readiness-persistence test FAIL because no status artifact or
dedicated exception exists; the passing behavior remains a regression guard.

- [ ] **Step 4: Add the stable readiness exception and audit payload**

Construct the full selection audit before raising. On insufficient minima,
raise `AcceptanceBankReadinessError(audit)` rather than a plain `ValueError`.
Keep the human-readable phase counts in the exception message.

- [ ] **Step 5: Persist readiness without creating a fake bank**

In `lock_negative_acceptance_bank`, catch only
`AcceptanceBankReadinessError`, write
`output_path.with_name("acceptance_readiness.json")` with:

```python
{
    "schema": "jit_repair_acceptance_readiness_v1",
    "status": "not_ready_before_repair_training",
    "selection": "all_baseline_continuation_negative_candidates",
    **error.audit,
    "training_transitions": 0,
    "validation_data_used": False,
    "test_data_used": False,
    "final_evaluation_data_used": False,
}
```

Then re-raise the dedicated exception. Do not create `acceptance_bank.json`.

- [ ] **Step 6: Run the focused tests and confirm GREEN**

Run the command from Step 3. Expected: all acceptance tests PASS.

- [ ] **Step 7: Commit the code and tests together**

```bash
git add JIT/src/jit_dvgc/continuation/__init__.py JIT/tests/test_continuation_labels.py
git commit -m "Persist acceptance bank readiness failures"
```

---

### Task 3: Static and Regression Verification

**Files:**
- Verify only; no production changes unless a directly caused regression is found.

**Interfaces:**
- Consumes the Task 1 and Task 2 commits.
- Produces fresh compiler/test evidence required before GPU labeling.

- [ ] **Step 1: Compile all active JIT source and CLI files**

```bash
export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python
"$PY" -m compileall -q JIT/src JIT/cli
```

Expected: exit 0 with no output.

- [ ] **Step 2: Run the required focused suite**

```bash
"$PY" -m pytest -q \
  JIT/tests/test_unified_boundary.py \
  JIT/tests/test_continuation_labels.py \
  JIT/tests/test_tube_rsi.py \
  JIT/tests/test_tube_rsi_mixed_snapshot.py \
  JIT/tests/test_tube_rsi_prng_key_contract.py \
  JIT/tests/test_unified_formal.py \
  JIT/tests/test_unified_continuation_labels.py
```

Expected: exit 0 with zero failures.

- [ ] **Step 3: Run the curated non-GPU preflight**

```bash
JIT_PYTHON="$PY" JIT/scripts/local_preflight.sh
```

Expected: package/API, formal contract, Tube config, and all non-GPU tests pass.

- [ ] **Step 4: Audit the Git diff and unrelated dirty paths**

```bash
git diff --check
git status --short
git diff -- JIT/cli/label_unified_continuations.py \
  JIT/src/jit_dvgc/continuation/__init__.py \
  JIT/tests/test_continuation_labels.py
```

Expected: no accidental changes outside the declared JIT scope.

---

### Task 4: Revalidate Acquisition, Label Once, and Lock the Bank

**Files:**
- Read: existing acquisition and anchor-audit artifacts.
- Create runtime artifacts only: `JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_supportwide_labels_20260902/`.

**Interfaces:**
- Consumes: the immutable 659-candidate catalog and frozen `pi_0`.
- Produces: completed labels plus either a locked `acceptance_bank.json` or a non-locked `acceptance_readiness.json`.

- [ ] **Step 1: Revalidate immutable acquisition identities and accounting**

Check catalog/summary/protocol/anchor audit SHAs, 659 unique TRAIN states,
1,151/1,152 interactions, zero training, false expert switching, false
validation/TEST/final evaluation, and byte-identical anchor audit SHA
`56a7049bd8d7172987eeafb2a8dd8915e84aaba027bbb659feb00659f8a531f4`.

- [ ] **Step 2: Verify every candidate snapshot before GPU construction**

Use the existing CLI preflight path and confirm `snapshot_preflight=GO
candidates=659`. Any mismatch is an immediate engineering stop.

- [ ] **Step 3: Confirm the labeling output directory does not exist**

```bash
test ! -e JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_supportwide_labels_20260902
```

- [ ] **Step 4: Run deterministic frozen-pi0 labeling exactly once**

```bash
export PYTHONPATH="$PWD/JIT/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
/home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/label_unified_continuations.py \
  --frozen-policy JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json \
  --catalog JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_supportwide_acquisition_20260902/catalog.json \
  --predeclaration JIT/configs/envelope_iter1_repair_acceptance_boundary_acquisition_supportwide.json \
  --output-dir JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_supportwide_labels_20260902
```

- [ ] **Step 5: Apply the stop rule**

If `acceptance_readiness.json` exists or the command reports readiness failure,
verify its counts, report `PRE-TRAINING FRESH ACCEPTANCE BANK = FAIL`, and stop
with zero repaired-policy training transitions.

- [ ] **Step 6: Verify a passing locked bank**

If `acceptance_bank.json` exists, recompute label and bank file SHA-256, canonical
bank SHA, phasewise negative states/groups, Tube1 state overlap, outcome closure,
interaction ceiling, and all isolation flags. Only exact closure authorizes
Task 5 and Task 6.

---

### Task 5: Record Observed Bank Provenance

**Files:**
- Modify: `JIT/docs/CURRENT_STATUS.md`
- Modify: `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: verified observed paths, counts, and hashes from Task 4.
- Produces: current ledgers identifying the fresh bank as launch authority, not repaired-policy acceptance evidence.

- [ ] **Step 1: Update current status with only observed values**

Record acquisition, labeling, and bank paths; protocol/file/canonical hashes;
phasewise readiness; interaction accounting; zero training; and false
validation/TEST flags. State that repaired `pi_1` remains untrained at the
moment represented by the bank artifact.

- [ ] **Step 2: Update protocol and compact experiment ledger**

Replace stale immediate-next-step text with the locked-bank result and the
single authorized repaired iteration-1 launch. Do not claim core preservation,
boundary gain, iteration acceptance, or envelope expansion.

- [ ] **Step 3: Verify and commit documentation**

```bash
git diff --check -- JIT/docs/CURRENT_STATUS.md \
  JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md docs/EXPERIMENT_STATE.md
git add JIT/docs/CURRENT_STATUS.md JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md \
  docs/EXPERIMENT_STATE.md
git commit -m "Record fresh repair acceptance bank provenance"
```

---

### Task 6: Preflight and Launch Exactly One Repaired pi1

**Files:**
- Read: `JIT/configs/pi_unified_iter1_tube1_core_replay50_natural10.json` and bound artifacts.
- Create runtime artifact only: `JIT/runs/pi_unified/pi_1_tube1_core_replay50_natural10_10009600_seed821101_20260902/`.

**Interfaces:**
- Consumes: the PASS bank from Task 4 and the immutable repaired config.
- Produces: one formal repaired iteration-1 training run, or one preserved engineering-error run followed by STOP.

- [ ] **Step 1: Verify the launch contract independently**

Check config canonical identity, Tube1 manifest, fresh actor/critic/optimizer,
seed 821101, 10,009,600 transitions, natural/Tube 0.1/0.9, Tube core/expansion
0.5/0.5, upstream/downstream 0.5/0.5, no validation, no TEST, no expert
switching, and `resume_semantics=fresh_only`.

- [ ] **Step 2: Run zero-interaction Tube and runtime preflight**

Call `jit_dvgc.training.preflight_unified_formal_tube` and verify 3,119 Tube1
entries, exact manifest identity, 117/310 upstream core/expansion and 105/2,587
downstream core/expansion, with 0.5/0.5 source probability in both phases and
zero interactions/training transitions. Run the required GPU reset/one-step
checks without creating the formal output directory.

- [ ] **Step 3: Confirm the formal output directory does not exist**

```bash
test ! -e JIT/runs/pi_unified/pi_1_tube1_core_replay50_natural10_10009600_seed821101_20260902
```

- [ ] **Step 4: Launch the single formal run**

```bash
export PYTHONPATH="$PWD/JIT/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
/home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_unified.py \
  --config JIT/configs/pi_unified_iter1_tube1_core_replay50_natural10.json \
  --run-id pi_1_tube1_core_replay50_natural10_10009600_seed821101_20260902
```

- [ ] **Step 5: Monitor without altering the method**

Inspect persisted `status.json`, checkpoint directories, TRAIN-panel reports,
GPU process state, and transition accounting. On engineering failure, preserve
the directory, do not create a retry, report the exact failure, and stop.

- [ ] **Step 6: Verify completion and freeze the exact final checkpoint**

Only after fresh verification of 10,009,600 completed transitions and the final
checkpoint identity, invoke the existing `JIT/cli/freeze_unified_policy.py`
into a new immutable frozen-policy directory. Verify payload, identity, report,
status, actor, critic, and normalizer hashes. Freezing does not accept the
candidate scientifically.

- [ ] **Step 7: Report and stop before scientific gates**

Report the required 18-point handoff. The mandatory next blocker is the full
222-state core-preservation plus fresh-bank paired boundary-gain gate. Do not
start `C^1`, Tube2, or pi2.

---

### Task 7: Repair the Iteration-Generic Preflight Contract

**Files:**
- Modify: `JIT/scripts/local_preflight.sh`
- Modify: `JIT/src/jit_dvgc/tube/__init__.py`
- Modify: `JIT/tests/test_preflight_contract.py`

**Interfaces:**
- Consumes: `training.FORMAL_SCHEMA`, `training.load_unified_formal_config`, and the existing `tube_rsi.normalize_core_replay_contract` implementation.
- Produces: schema-driven static validation of every unified formal config and a stable `tube.normalize_core_replay_contract` package API.

- [ ] **Step 1: Replace obsolete text assertions with iteration-generic RED tests**

Assert the preflight discovers `JIT/configs/*.json`, selects configs through
`training.FORMAL_SCHEMA`, validates them with
`training.load_unified_formal_config`, and validates optional replay contracts
through `tube.normalize_core_replay_contract`. Assert it does not hardcode the
repaired pi1 config filename. Replace old README assertions with the stable
iteration-workflow command and final TEST/JCE/JEL claim boundary already
documented in `JIT/README.md`.

- [ ] **Step 2: Run the contract tests and confirm RED**

```bash
export PYTHONPATH="$PWD/JIT/src"
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  JIT/tests/test_preflight_contract.py
```

Expected: FAIL because current preflight hardcodes one rejected pi1 config and
does not expose/use the stable replay normalizer.

- [ ] **Step 3: Export the existing replay normalizer from the stable Tube API**

Import and add `normalize_core_replay_contract` to `jit_dvgc.tube.__all__`.
Do not duplicate its implementation or add a module.

- [ ] **Step 4: Make the preflight schema-driven**

In the existing embedded Python preflight, iterate over sorted
`Path("JIT/configs").glob("*.json")`, read each JSON object, select only
`training.FORMAL_SCHEMA`, and call `training.load_unified_formal_config(path)`.
For each optional `tube_sampling`, call
`tube.normalize_core_replay_contract(raw["tube_sampling"])`. Require at least
one unified formal config and at least one replay contract, but never name an
iteration, Tube, retry, seed, or checkpoint in source.

- [ ] **Step 5: Run GREEN and the affected package-facade test**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  JIT/tests/test_preflight_contract.py JIT/tests/test_package_facades.py
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the generic preflight repair**

```bash
git add JIT/scripts/local_preflight.sh JIT/src/jit_dvgc/tube/__init__.py \
  JIT/tests/test_preflight_contract.py
git commit -m "Generalize unified formal preflight discovery"
```
