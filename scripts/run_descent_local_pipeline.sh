#!/usr/bin/env bash
set -euo pipefail

# Resumable bounded local descent bootstrap.  Each expensive step writes to a
# new block directory and is reused only when its input hashes still match.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/qy/mujoco_playground/.venv/bin/python}"
ASSET_RUN="${DESCENT_LOCAL_ASSET_RUN:-runs/stage_experts/descent_local_seed0_20260716T163504}"
RUN="${DESCENT_LOCAL_RUN:-runs/stage_experts/descent_local_nonfinite_repair_seed0_20260716T1825}"
POOL="$ASSET_RUN/candidate_pool_final.pkl"
RESET_BANK="$ASSET_RUN/bootstrap_reset_bank.pkl"
INITIAL="$ASSET_RUN/pi_f_descent_local"
LANDING="runs/landing/refinement_seed0/policy"
ENTRY="runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl"
RUNTIME_GATE="docs/RUNTIME_GATE.json"

cd "$ROOT"
[[ -x "$PYTHON" && -f "$POOL" && -f "$RESET_BANK" && -d "$INITIAL" && -d "$LANDING" && -f "$ENTRY" ]] || { echo "descent-local inputs missing" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "tracked worktree must be clean" >&2; exit 2; }
"$PYTHON" -m cli.runtime_gate --config configs/default.json --output "$RUNTIME_GATE" --check-only >/dev/null

initial_hash="$(sha256sum "$INITIAL/params.pkl" | cut -d' ' -f1)"
landing_hash="$(sha256sum "$LANDING/params.pkl" | cut -d' ' -f1)"
previous_safe=-1
previous_failure=1
stagnant=0

for block in 1 2 3 4; do
  steps=$((block*25600)); root="$RUN/blocks/block_${block}_${steps}"; policy="$root/train/policy"
  cert="$root/current_policy_certified.pkl"; analysis="$root/current_policy_analysis.json"
  mkdir -p "$root"
  if [[ -f "$root/train/report.json" ]]; then
    "$PYTHON" - "$root/train/report.json" "$POOL" "$RESET_BANK" "$ENTRY" "$steps" <<'PY' || { echo "stale train block output; refusing overwrite" >&2; exit 3; }
import hashlib,json,sys
h=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest(); r=json.load(open(sys.argv[1]))
ok=r['candidate_bank_sha256']==h(sys.argv[2]) and r['bootstrap_bank_sha256']==h(sys.argv[3]) and r['entry_set_sha256']==h(sys.argv[4]) and r['cumulative_effective_steps']==int(sys.argv[5]) and r['status']=='PASS'
raise SystemExit(0 if ok else 1)
PY
  fi
  if [[ ! -f "$root/train/report.json" ]]; then
    args=("$PYTHON" -u -m cli.train_descent_local_block --resume-policy "$INITIAL" --bootstrap-bank "$RESET_BANK" --candidate-bank "$POOL" --entry-set "$ENTRY" --run "$root/train" --cumulative-steps "$steps" --seed 0)
    if (( block>1 )); then
      prior_steps=$(((block-1)*25600)); prior="$RUN/blocks/block_$((block-1))_${prior_steps}"
      args=("$PYTHON" -u -m cli.train_descent_local_block --resume-policy "$prior/train/policy" --bootstrap-bank "$RESET_BANK" --candidate-bank "$POOL" --entry-set "$ENTRY" --run "$root/train" --cumulative-steps "$steps" --restore-checkpoint "$prior/train/orbax/$(printf '%012d' "$prior_steps")" --seed 0)
    fi
    "${args[@]}" >"$root/train.log" 2>&1
  fi
  [[ "$(sha256sum "$INITIAL/params.pkl" | cut -d' ' -f1)" == "$initial_hash" ]] || { echo "immutable pi_F,D changed" >&2; exit 3; }
  [[ "$(sha256sum "$LANDING/params.pkl" | cut -d' ' -f1)" == "$landing_hash" ]] || { echo "immutable pi_L changed" >&2; exit 3; }
  if [[ -f "${cert%.pkl}.cert.json" ]]; then
    "$PYTHON" - "${cert%.pkl}.cert.json" "$policy" "$POOL" "$ENTRY" <<'PY' || { echo "stale certification output; refusing overwrite" >&2; exit 3; }
import hashlib,json,sys
h=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest(); r=json.load(open(sys.argv[1]))
ok=r['descent_policy_hash']==h(sys.argv[2]+'/params.pkl') and r['candidate_bank_sha256']==h(sys.argv[3]) and r['landing_entry_set_sha256']==h(sys.argv[4])
raise SystemExit(0 if ok else 1)
PY
  fi
  if [[ ! -f "$cert" ]]; then
    "$PYTHON" -u -m cli.certify_descent_entries --descent-policy "$policy" --candidate-source-policy "$INITIAL" --landing-policy "$LANDING" --candidate-bank "$POOL" --landing-entry-set "$ENTRY" --output "$cert" --seed $((7600000+block*10000)) --namespace "descent_local_block_${block}" --confirm-safe-to-max >"$root/certification.log" 2>&1
  fi
  if [[ ! -f "$analysis" ]]; then
    "$PYTHON" -m cli.analyze_descent_local_certification --bank "$cert" --cert-report "${cert%.pkl}.cert.json" --output "$analysis" >"$root/analysis.log" 2>&1
  fi
  read -r safe failure ready < <("$PYTHON" - "$analysis" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print(r["unique_final_safe_states"],r["original_70"]["physical_failure_rate"],int(r["minimum_tube_support_ready"]))
PY
)
  echo "[descent-local] block=$block steps=$steps safe=$safe original_failure=$failure tube_support_ready=$ready"
  if (( ready==1 )); then
    audit="$root/independent_audit.json"; tube="$root/canonical_descent_entry.pkl"; parts=()
    if [[ ! -f "$audit" ]]; then
      total=$("$PYTHON" -c "from dvgc.bank import SnapshotBank; print(len(SnapshotBank.load('$cert').records_for_phase('flight',include_training_only=False)))")
      shard=0
      for ((start=0; start<total; start+=35)); do
        stop=$((start+35)); (( stop>total )) && stop=$total; part="$root/audit_part_${shard}.json"; parts+=(--shard "$part")
        if [[ ! -f "$part" ]]; then
          "$PYTHON" -u -m cli.certify_descent_entries --audit-only --descent-policy "$policy" --candidate-source-policy "$INITIAL" --landing-policy "$LANDING" --candidate-bank "$cert" --landing-entry-set "$ENTRY" --output "$part" --seed $((8700000+block*100000)) --namespace "audit_descent_local_block_${block}" --start-index "$start" --end-index "$stop" >"$root/audit_part_${shard}.log" 2>&1
        fi
        shard=$((shard+1))
      done
      "$PYTHON" -m cli.merge_descent_entry_audits "${parts[@]}" --output "$audit" >"$root/audit_merge.log" 2>&1
    fi
    if [[ ! -f "$tube" ]]; then
      "$PYTHON" -m cli.calibrate_descent_entries --certified-bank "$cert" --audit-report "$audit" --output-bank "$tube" >"$root/tube_calibration.log" 2>&1
    fi
    echo "[descent-local] minimum empirical C_D certified and independently audited: $tube"
    exit 0
  fi
  if (( previous_safe>=0 )); then
    improved=$("$PYTHON" -c "print(int(int('$safe')>int('$previous_safe') or float('$failure')<float('$previous_failure')))" )
    if (( improved==0 )); then stagnant=$((stagnant+1)); else stagnant=0; fi
    if (( stagnant>=2 )); then echo "[descent-local] two blocks without support/failure improvement; candidate-support repair required" >&2; exit 2; fi
  fi
  previous_safe="$safe"; previous_failure="$failure"
done

echo "[descent-local] bounded four-block budget exhausted without minimum C_D support" >&2
exit 2
