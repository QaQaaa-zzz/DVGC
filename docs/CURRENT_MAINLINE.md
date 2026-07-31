# DVGC Current Mainline

## 1. Active research claim

The active implementation validates three claims only:

1. Event-aligned next-stage labels expose local phase reachability without requiring each intermediate expert to finish the entire jump.
2. Phase-conditioned reachability models rank proposal coverage but never assign safety.
3. Phase-balanced snapshot RSI can consolidate independently owned experts into one shared Actor, whose final empirical envelope must be established by fresh frozen-policy Final-Recovery certification.

The formal Tube/JEL distinction is strict:

- Takeoff, Ascent and Apex records are independently audited local-entry proposal support, not formal Tubes.
- Descent and Landing contribute independently audited Final-safe Tube states.
- The phase-balanced RSI bank is training-only and cannot carry a safe claim.
- Only the final frozen shared Actor's independent end-to-end recertification may define the final JEL.

## 2. Current execution chain

```text
stage experts and audited snapshot assets
        ↓
build_phase_balanced_tube_rsi_bank
        ↓
build_phase_balanced_teacher_dataset
        ↓
phase-balanced distillation
        ↓
preflight_phase_balanced_unified_rsi
        ↓
train_phase_balanced_unified_rsi_pilot (bounded joint RSI PPO)
        ↓
fixed Final/retention/action-drift promotion decision
        ↓
final shared-policy JEL audit: 4 → 8 → 32 branches
```

The current controller entry is:

```bash
bash scripts/start_corrected_apex_unified_rsi_followons.sh
```

It starts the corrected bounded Apex-contract unified RSI pilot and, only after `PASS_PROMOTE`, starts the final shared-policy JEL audit.

## 3. How current RSI works

### 3.1 Reset bank construction

`cli/build_phase_balanced_tube_rsi_bank.py` loads exactly five source banks:

- Takeoff
- Ascent
- Apex
- Descent
- Landing

Each phase receives total reset probability mass `0.2`. Within each phase, probability is first divided equally across distinct parent trajectories and then equally across states belonging to each parent. This prevents a phase or a prolific parent trajectory from dominating PPO sampling.

Every output record is copied into a training-only role:

```text
artifact_role = proposal_support_bank
training_only = true
phase_rsi_stage = <takeoff|ascent|apex|descent|landing>
reset_weight = phase mass / parent count / states in parent
```

Certification fields and embedded safe claims are removed from the copied reset rows. The source hashes and source roles remain in metadata for provenance.

### 3.2 Reset sampling during PPO

The unified environment runs with:

```text
training_stage = flight
stage_reachability_objective = phase_balanced_rsi
use_bank_resets = true
```

A reset samples one snapshot from the phase-balanced bank according to `reset_weight`. The snapshot restores physical state plus the online controller context required by the snapshot schema, including observation history, last action/control state, phase/event state and delay-related fields when present.

The sampled row carries `phase_rsi_stage`; the environment maps that stage to the corresponding local objective:

- Takeoff → reach Ascent
- Ascent → reach Apex
- Apex → reach stable physical Descent entry
- Descent → reach Landing/C_L
- Landing → stable recovery

Thus one shared Actor is optimized from all five phase starts, but each reset receives the objective appropriate to its originating phase.

### 3.3 Initialization and constraints

The PPO initializer is a phase-balanced distilled policy produced from frozen expert actions. The bounded joint PPO then applies the following safeguards:

- frozen observation normalizer;
- fixed pre/post evaluation states;
- no loss of Descent and Landing retained Final states;
- action drift RMS ≤ 0.02 and max absolute drift ≤ 0.05 for every phase;
- finite training and no new nonfinite termination;
- promotion only when upstream or total fixed Final evidence improves.

The current corrected pilot budget is 4,096 effective PPO steps. A failed pilot does not trigger an open-ended continuation.

## 4. Where each phase snapshot comes from

### Landing

Landing snapshots come from frozen Landing-policy rollouts and the independently audited Landing Tube. They represent physical Landing/Recovery states that satisfy the locked Final-Recovery certification protocol. Only Final-safe Landing rows are admitted to the phase-balanced RSI bank. The Landing completion analysis must match the exact Tube hash and independent-audit status.

### Descent

Descent snapshots come from the Descent reachability/construction pipeline. Candidate states were obtained from real flight/descent trajectories and predecessor searches, screened with frozen controllers, and independently audited under the certified Descent controller identity. The current formal source is Descent Tube v6, which preserves the v5 safe states while normalizing schema/provenance fields. Only rows with `final.label == safe` and `certified_safe == true` enter the phase-balanced RSI bank.

A separate `descent_proposal_support_v1` bank is also supplied to the environment as the Descent→Landing matcher/support contract. It is proposal-only and is not a Tube.

### Apex

Apex snapshots are local-entry support states produced from upstream flight/ascent trajectories and Apex reachability acquisition. Frozen feedback sequences/controllers are evaluated with disjoint branch seeds. Surviving states must independently reach the locked stable physical Descent-entry event. The corrected current contract uses four consecutive stable physical-Descent ticks; it does not require the older formal C_D matcher to fire. Apex rows remain local proposal support, not Final-safe Tube states.

### Ascent

Ascent snapshots originate from real Takeoff→Ascent entries collected under frozen upstream controllers. A phase-conditioned reachability model ranks candidates; selected parent-diverse states are screened and independently audited for valid Apex entry. Exact local-entry survivors become `stage_entry_certified_proposal_support`. They are not formal Tube states.

### Takeoff

Takeoff snapshots originate from physically valid ground/reset states aligned with the authoritative XML and Takeoff reference envelope. Candidate states are ranked by the Takeoff reachability model, assigned a frozen controller using construction evidence, then evaluated with fresh isolated branches. Exact local-entry survivors that reach Ascent become `stage_entry_certified_proposal_support`; they do not receive a formal Tube claim.

## 5. Snapshot semantics

A snapshot is not only `qpos/qvel`. The active snapshot contract is intended to recreate an online control instant and may include:

- `qpos`, `qvel`, `ctrl`, warm-start data;
- actor and privileged observation state;
- observation history and previous action;
- phase probabilities/progress and event latches;
- contact/IMU-derived belief state;
- termination and timeout semantics;
- delay buffer or timing fields for newer schemas;
- source policy, XML, action mapping and bank provenance.

Legacy post-update histories must not be reconstructed as pre-update history plus a duplicated current frame. Runtime snapshot gates check restore fidelity and next-step behavior with field-specific tolerances plus exact discrete event equality.

## 6. Files that define the active path

Core runtime:

- `dvgc/env.py`
- `dvgc/bank.py`
- `dvgc/config.py`
- `dvgc/policy.py`
- `dvgc/rollout.py`
- `dvgc/runtime.py`
- `dvgc/stage_reachability.py`
- `dvgc/descent_supervised.py`

Active unified-policy tools:

- `cli/build_phase_balanced_tube_rsi_bank.py`
- `cli/build_phase_balanced_teacher_dataset.py`
- `cli/train_phase_balanced_distillation.py`
- `cli/preflight_phase_balanced_unified_rsi.py`
- `cli/train_phase_balanced_unified_rsi_pilot.py`
- `cli/verify_final_shared_policy_jel.py`

Active scripts:

- `scripts/start_corrected_apex_unified_rsi_followons.sh`
- `scripts/run_corrected_apex_unified_rsi_pipeline.sh`
- `scripts/run_final_shared_jel_audit.sh`
- `scripts/local_preflight.sh`
- `scripts/dvgc_status.sh`

## 7. Archive boundary

The complete pre-clean repository is preserved at branch:

```text
archive/pre-clean-20260731
```

Superseded controllers and one-off experiment routes may be removed from the active branch after their evidence and source remain reachable through this immutable branch and Git history.
