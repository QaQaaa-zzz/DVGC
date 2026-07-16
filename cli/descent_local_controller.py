"""Persistent, resumable controller for the bounded local C_D pipeline."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cli.build_descent_entries import snapshot_identity
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


PYTHON = "/home/qy/mujoco_playground/.venv/bin/python"
ASSET_RUN = Path("runs/stage_experts/descent_local_seed0_20260716T163504")
POOL = ASSET_RUN / "candidate_pool_final.pkl"
RESET_BANK = ASSET_RUN / "bootstrap_reset_bank.pkl"
INITIAL = ASSET_RUN / "pi_f_descent_local"
LANDING = Path("runs/landing/refinement_seed0/policy")
ENTRY = Path("runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl")
RUNTIME_GATE = Path("docs/RUNTIME_GATE.json")
SHARD_SIZE = 12


class Controller:
    def __init__(self, run: Path):
        self.run = run
        self.run.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run / "controller_state.json"
        self.log_path = self.run / "persistent_controller.log"
        self.lock_path = self.run / "controller.lock"
        self.lock = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another descent-local controller owns the lock") from exc
        self.state = self._load_state()
        self._write_lock_info()

    def _load_state(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {
            "controller_version": 2,
            "run_id": self.run.name,
            "current_stage": "inspect",
            "last_completed_action": None,
            "in_progress_action": None,
            "expected_outputs": [],
            "next_decision": "certify_block",
            "retry_count": 0,
            "heartbeat": time.time(),
            "stop_reason": None,
            "history": [],
            "provenance": {},
        }

    def _write_lock_info(self):
        policy_hash = self.state.get("provenance", {}).get("current_policy_hash")
        if not policy_hash:
            block = int(self.state.get("active_block", 1))
            candidate = self.run / f"blocks/block_{block}_{block * 25600}/train/policy/params.pkl"
            if candidate.exists():
                policy_hash = file_sha256(candidate)
        self.lock.seek(0)
        self.lock.truncate()
        self.lock.write(json.dumps({"pid": os.getpid(), "run_id": self.run.name,
                                    "policy_hash": policy_hash,
                                    "started_at": time.time()}))
        self.lock.flush()
        os.fsync(self.lock.fileno())

    def save(self, **updates):
        self.state.update(updates)
        self.state["heartbeat"] = time.time()
        save_json(self.state_path, self.state)
        self._write_lock_info()

    def log(self, message):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n"
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line)
        print(message, flush=True)

    def run_command(self, action, command, log_path, outputs):
        self.save(in_progress_action=action, expected_outputs=[str(x) for x in outputs])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log(f"START {action}: {' '.join(map(str, command))}")
        with log_path.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                self.save()
                time.sleep(30)
        if process.returncode:
            self.save(retry_count=int(self.state.get("retry_count", 0)) + 1,
                      stop_reason=f"{action} exited {process.returncode}")
            raise RuntimeError(f"{action} exited {process.returncode}; see {log_path}")
        missing = [str(path) for path in outputs if not Path(path).exists()]
        if missing:
            raise RuntimeError(f"{action} completed without outputs: {missing}")
        self.state.setdefault("history", []).append({"action": action, "completed_at": time.time(),
                                                       "outputs": [str(x) for x in outputs]})
        self.save(last_completed_action=action, in_progress_action=None, expected_outputs=[], retry_count=0,
                  stop_reason=None)
        self.log(f"DONE {action}")

    def inspect(self):
        if shutil.which(PYTHON) is None and not Path(PYTHON).is_file():
            raise RuntimeError(f"Configured runtime missing: {PYTHON}")
        if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip():
            raise RuntimeError("Tracked worktree must be clean")
        required = [POOL, RESET_BANK, INITIAL / "params.pkl", LANDING / "params.pkl", ENTRY, RUNTIME_GATE]
        if any(not path.exists() for path in required):
            raise RuntimeError("One or more immutable pipeline inputs are missing")
        subprocess.run([PYTHON, "-m", "cli.runtime_gate", "--config", "configs/default.json",
                        "--output", str(RUNTIME_GATE), "--check-only"], check=True,
                       stdout=subprocess.DEVNULL)
        provenance = {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "xml_sha256": file_sha256("assets/orange_bike_4kg_horizontal.xml"),
            "candidate_bank_sha256": file_sha256(POOL),
            "reset_bank_sha256": file_sha256(RESET_BANK),
            "landing_entry_set_sha256": file_sha256(ENTRY),
            "landing_policy_hash": file_sha256(LANDING / "params.pkl"),
            "initial_policy_hash": file_sha256(INITIAL / "params.pkl"),
            "flight_bank_sha256": file_sha256("artifacts/flight_candidates_augmented_v1.pkl"),
            "current_policy_hash": file_sha256(self.run / "blocks/block_1_25600/train/policy/params.pkl"),
        }
        gate = json.loads(RUNTIME_GATE.read_text(encoding="utf-8"))
        provenance["runtime_source_fingerprint"] = gate["source_fingerprint"]
        old_log = self.run / "blocks/block_1_25600/certification.log"
        classification = self.run / "recovery_classification.json"
        if old_log.exists() and not classification.exists():
            text = old_log.read_text(encoding="utf-8", errors="replace")
            save_json(classification, {
                "status": "corrupt/incomplete diagnostic",
                "reason": "monolithic construction exited before atomic bank/report" if "Failed to allocate" in text else "monolithic construction has no atomic bank/report",
                "retained_log": str(old_log.resolve()),
                "retained_log_sha256": file_sha256(old_log),
                "completed_atomic_shards": 0,
                "partial_log_rows_not_reused": True,
            })
        self.save(provenance=provenance, current_stage="certify_block", active_block=1,
                  last_completed_action="inspect", in_progress_action=None, expected_outputs=[],
                  next_decision="merge_certification")

    def block_paths(self, block):
        steps = block * 25600
        root = self.run / f"blocks/block_{block}_{steps}"
        return root, root / "train/policy", root / "train/report.json"

    def validate_training_block(self, block):
        root, policy, report_path = self.block_paths(block)
        if not report_path.exists() or not (policy / "params.pkl").exists():
            return False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return bool(report.get("status") == "PASS" and report.get("cumulative_effective_steps") == block * 25600
                    and report.get("candidate_bank_sha256") == file_sha256(POOL)
                    and report.get("bootstrap_bank_sha256") == file_sha256(RESET_BANK)
                    and report.get("entry_set_sha256") == file_sha256(ENTRY)
                    and report.get("policy_hash") == file_sha256(policy / "params.pkl"))

    def certify_block(self, block):
        if not self.validate_training_block(block):
            raise RuntimeError(f"Block {block} training output is incomplete or stale")
        root, policy, _ = self.block_paths(block)
        seed = 7600000 + block * 10000
        shard_root = root / f"construction_seed{seed}_sharded_v1"
        shard_root.mkdir(parents=True, exist_ok=True)
        total = len(SnapshotBank.load(POOL).records_for_phase("flight", include_training_only=False))
        shards = []
        for start in range(0, total, SHARD_SIZE):
            end = min(start + SHARD_SIZE, total)
            output = shard_root / f"shard_{start:03d}_{end:03d}.completed.json"
            shards.append(output)
            if output.exists():
                payload = json.loads(output.read_text(encoding="utf-8"))
                valid = (payload.get("complete") is True and payload.get("start_index") == start
                         and payload.get("end_index") == end and payload.get("seed") == seed
                         and payload.get("descent_policy_hash") == file_sha256(policy / "params.pkl")
                         and payload.get("candidate_bank_sha256") == file_sha256(POOL))
                if not valid:
                    invalid = shard_root / "invalid_diagnostic"
                    invalid.mkdir(exist_ok=True)
                    output.rename(invalid / f"{output.name}.{int(time.time())}")
                else:
                    continue
            command = [PYTHON, "-u", "-m", "cli.certify_descent_construction_shard",
                       "--descent-policy", policy, "--candidate-source-policy", INITIAL,
                       "--landing-policy", LANDING, "--candidate-bank", POOL,
                       "--landing-entry-set", ENTRY, "--output", output, "--seed", seed,
                       "--namespace", f"descent_local_block_{block}", "--start-index", start,
                       "--end-index", end, "--confirm-safe-to-max"]
            self.run_command(f"certify_block_{block}_shard_{start}_{end}", command,
                             shard_root / f"shard_{start:03d}_{end:03d}.log", [output])
        self.save(current_stage="merge_certification", next_decision="decide_after_certification",
                  expected_outputs=[str(path) for path in shards])

    def merge_certification(self, block):
        root, _, _ = self.block_paths(block)
        seed = 7600000 + block * 10000
        shard_root = root / f"construction_seed{seed}_sharded_v1"
        shards = sorted(shard_root.glob("shard_*.completed.json"))
        bank = root / "current_policy_certified_sharded.pkl"
        report = root / "current_policy_certified_sharded.cert.json"
        analysis = root / "current_policy_analysis_sharded.json"
        if not bank.exists() or not report.exists():
            command = [PYTHON, "-m", "cli.merge_descent_construction_shards"]
            for shard in shards:
                command.extend(["--shard", shard])
            command.extend(["--candidate-bank", POOL, "--output-bank", bank, "--output-report", report])
            self.run_command(f"merge_certification_block_{block}", command, root / "merge_sharded.log", [bank, report])
        if not analysis.exists():
            self.run_command(f"analyze_certification_block_{block}",
                             [PYTHON, "-m", "cli.analyze_descent_local_certification", "--bank", bank,
                              "--cert-report", report, "--output", analysis],
                             root / "analysis_sharded.log", [analysis])
        self.save(current_stage="decide_after_certification", next_decision="build_matcher_or_run_next_block")

    def _improvement(self, certified_bank, analysis_path):
        current = SnapshotBank.load(certified_bank)
        source = SnapshotBank.load(POOL)
        current_by = {row["id"]: row for row in current.records_for_phase("flight", include_training_only=False)}
        old_rows = [row for row in source.records_for_phase("flight", include_training_only=False)
                    if row.get("old_policy_label") in {"safe", "boundary", "dead", "unknown"}]
        old_evidence = [ev for row in old_rows for ev in row.get("certification_branches", [])]
        new_evidence = [ev for row in old_rows for ev in current_by[row["id"]].get("certification_branches", [])]
        def rate(evidence, key, value=True):
            return sum(ev.get(key) == value for ev in evidence) / len(evidence) if evidence else 0.0
        old_boundary = [row for row in old_rows if row.get("old_policy_label") == "boundary"]
        improved_boundary = sum(current_by[row["id"]]["final"]["posterior"]["mean"]
                                >= row["final"]["posterior"]["mean"] + 0.05 for row in old_boundary)
        old_safe_parents = {str(row.get("entry_source_id", row["id"])) for row in old_rows if row.get("old_policy_label") == "safe"}
        new_safe = [row for row in current_by.values() if row["final"]["label"] == "safe"]
        new_safe_parents = {str(row.get("entry_source_id", row["id"])) for row in new_safe}
        evidence = {
            "old_chain_rate": rate(old_evidence, "chain_success"),
            "current_chain_rate": rate(new_evidence, "chain_success"),
            "old_final_rate": rate(old_evidence, "final_recovery"),
            "current_final_rate": rate(new_evidence, "final_recovery"),
            "old_physical_failure_rate": rate(old_evidence, "terminal_cause", "physical_failure"),
            "current_physical_failure_rate": rate(new_evidence, "terminal_cause", "physical_failure"),
            "boundary_posterior_improved_by_0p05": improved_boundary,
            "old_safe_parent_count": len(old_safe_parents),
            "current_safe_parent_count": len(new_safe_parents),
        }
        improved = (improved_boundary >= 2
                    or evidence["current_physical_failure_rate"] <= evidence["old_physical_failure_rate"] - 0.02
                    or (evidence["current_chain_rate"] >= evidence["old_chain_rate"] + 0.01
                        and evidence["current_final_rate"] >= evidence["old_final_rate"] + 0.01)
                    or len(new_safe_parents) > len(old_safe_parents))
        return improved, evidence

    def decide_after_certification(self, block):
        root, _, _ = self.block_paths(block)
        bank = root / "current_policy_certified_sharded.pkl"
        analysis_path = root / "current_policy_analysis_sharded.json"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        decision_path = root / "decision_after_certification.json"
        if bool(analysis["minimum_tube_support_ready"]):
            decision = "build_matcher"
            evidence = {"unique_final_safe": analysis["unique_final_safe_states"],
                        "safe_parent_count": analysis["safe_source_count"]}
            next_stage = "build_matcher"
        else:
            improved, evidence = self._improvement(bank, analysis_path)
            decision = "run_next_block" if improved and block < 4 else "pause_with_reason"
            next_stage = decision
            evidence["unique_final_safe"] = analysis["unique_final_safe_states"]
        save_json(decision_path, {"status":"PASS", "block":block, "decision":decision, "evidence":evidence})
        self.save(current_stage=next_stage, next_decision=decision, last_completed_action="decide_after_certification")
        if decision == "pause_with_reason":
            self.save(stop_reason="No evidenced block-1 improvement; one bounded reward/candidate repair must be selected from diagnostics")

    def run_next_block(self, block):
        next_block = block + 1
        root, policy, report = self.block_paths(next_block)
        if self.validate_training_block(next_block):
            self.save(current_stage="certify_block", active_block=next_block)
            return
        prior_root, prior_policy, _ = self.block_paths(block)
        prior_steps = block * 25600
        checkpoint = prior_root / f"train/orbax/{prior_steps:012d}"
        command = [PYTHON, "-u", "-m", "cli.train_descent_local_block",
                   "--resume-policy", prior_policy, "--bootstrap-bank", RESET_BANK,
                   "--candidate-bank", POOL, "--entry-set", ENTRY, "--run", root / "train",
                   "--cumulative-steps", next_block * 25600, "--restore-checkpoint", checkpoint, "--seed", 0]
        self.run_command(f"run_next_block_{next_block}", command, root / "train.log", [report, policy / "params.pkl"])
        self.state["provenance"]["current_policy_hash"] = file_sha256(policy / "params.pkl")
        self.save(current_stage="certify_block", active_block=next_block, next_decision="merge_certification")

    def build_matcher(self, block):
        root, _, _ = self.block_paths(block)
        certified = root / "current_policy_certified_sharded.pkl"
        matcher = root / "canonical_descent_entry_pre_audit.pkl"
        manifest = root / "canonical_descent_entry_pre_audit.manifest.json"
        if not matcher.exists() or not manifest.exists():
            self.run_command("build_matcher",
                             [PYTHON, "-m", "cli.build_descent_matcher", "--certified-bank", certified,
                              "--output-bank", matcher, "--manifest", manifest],
                             root / "build_matcher.log", [matcher, manifest])
        self.save(current_stage="run_independent_audit", next_decision="audit_analysis")

    def run_independent_audit(self, block):
        root, policy, _ = self.block_paths(block)
        matcher = root / "canonical_descent_entry_pre_audit.pkl"
        manifest = root / "canonical_descent_entry_pre_audit.manifest.json"
        audit_root = root / "independent_audit_sharded_v1"
        audit_root.mkdir(exist_ok=True)
        total = len(SnapshotBank.load(matcher).records_for_phase("flight", include_training_only=False))
        seed = 8700000 + block * 100000
        shards = []
        for start in range(0, total, SHARD_SIZE):
            end = min(start + SHARD_SIZE, total)
            output = audit_root / f"shard_{start:03d}_{end:03d}.completed.json"
            shards.append(output)
            if not output.exists():
                self.run_command(f"independent_audit_shard_{start}_{end}",
                                 [PYTHON, "-u", "-m", "cli.certify_descent_entries", "--audit-only",
                                  "--descent-policy", policy, "--candidate-source-policy", INITIAL,
                                  "--landing-policy", LANDING, "--candidate-bank", matcher,
                                  "--landing-entry-set", ENTRY, "--output", output, "--seed", seed,
                                  "--namespace", f"audit_descent_local_block_{block}",
                                  "--start-index", start, "--end-index", end],
                                 audit_root / f"shard_{start:03d}_{end:03d}.log", [output])
        merged = root / "independent_audit_sharded.json"
        if not merged.exists():
            command = [PYTHON, "-m", "cli.merge_descent_entry_audits"]
            for shard in shards:
                command.extend(["--shard", shard])
            command.extend(["--output", merged])
            self.run_command("merge_independent_audit", command, root / "audit_merge.log", [merged])
        result = root / "descent_matcher_independent_audit.json"
        if not result.exists():
            self.run_command("audit_analysis",
                             [PYTHON, "-m", "cli.audit_descent_matcher", "--matcher-bank", matcher,
                              "--matcher-manifest", manifest, "--audit-report", merged, "--output", result],
                             root / "audit_analysis.log", [result])
        report = json.loads(result.read_text(encoding="utf-8"))
        self.save(current_stage="train_viability" if report["status"] == "PASS" else "pause_with_reason",
                  next_decision="train_viability" if report["status"] == "PASS" else "pause_with_reason",
                  stop_reason=None if report["status"] == "PASS" else "; ".join(report["reasons"]))

    def train_viability(self, block):
        root, _, _ = self.block_paths(block)
        bank = root / "current_policy_certified_sharded.pkl"
        model = root / "viability/ensemble.pkl"
        report = root / "viability/report.json"
        if not report.exists():
            self.run_command("train_viability",
                             [PYTHON, "-m", "cli.fit_viability", "--bank", bank, "--output", model,
                              "--report", report, "--seed", 9100000 + block],
                             root / "viability/train.log", [model, report])
        self.save(current_stage="run_acquisition", next_decision="run_acquisition",
                  stop_reason="Acquisition/Tube-RSI controller continuation pending validated implementation")

    def loop(self):
        self.log(f"controller pid={os.getpid()} run={self.run}")
        while True:
            stage = self.state.get("current_stage", "inspect")
            block = int(self.state.get("active_block", 1))
            self.save()
            if stage == "inspect": self.inspect()
            elif stage == "certify_block": self.certify_block(block)
            elif stage == "merge_certification": self.merge_certification(block)
            elif stage == "decide_after_certification": self.decide_after_certification(block)
            elif stage == "run_next_block": self.run_next_block(block)
            elif stage == "build_matcher": self.build_matcher(block)
            elif stage == "run_independent_audit": self.run_independent_audit(block)
            elif stage == "train_viability": self.train_viability(block)
            elif stage in {"run_acquisition", "run_tube_rsi", "pause_with_reason", "complete"}:
                self.log(f"controller stopped at {stage}: {self.state.get('stop_reason')}")
                return 0 if stage == "complete" else 2
            else: raise RuntimeError(f"Unknown controller stage: {stage}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/stage_experts/descent_local_nonfinite_repair_seed0_20260716T1825")
    a = p.parse_args()
    controller = Controller(Path(a.run))
    try:
        raise SystemExit(controller.loop())
    except Exception as exc:
        controller.log(f"ERROR {type(exc).__name__}: {exc}")
        controller.save(stop_reason=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
