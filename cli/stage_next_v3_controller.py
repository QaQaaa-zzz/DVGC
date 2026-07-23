"""Resumable controller for corrected-reset stage-to-next-stage acquisition."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cli.descent_local_controller import Controller, PYTHON
from dvgc.config import file_sha256
from dvgc.runtime import save_json


RUN = Path(os.environ.get("STAGE_NEXT_V3_RUN", "runs/stage_next_reset_v3_seed0_20260723"))
TAKEOFF_BANK = Path("runs/stage_next_takeoff_keyposture_seed0_20260723/takeoff_reset_bank_v3_120.pkl")
TAKEOFF_EVAL = RUN / "takeoff/fixed_balanced_eval_v1_24.pkl"
BASELINE = RUN / "takeoff/controller_bank_baseline_24x4_r3.json"
OLD_TAKEOFF = Path("runs/stage_next_bootstrap_seed0_20260720/takeoff/blocks/block_1_025600/train/policy")
NEW_TAKEOFF = Path("runs/stage_next_takeoff_keyposture_seed0_20260723/takeoff/local_expert_006400/train/policy")
TAKEOFF_CONFIG = Path("runs/stage_next_takeoff_keyposture_seed0_20260723/controller_inputs_v4_balanced/takeoff/config.json")
EXACT_STAGE_BANK = Path("runs/stage_next_takeoff_keyposture_seed0_20260723/preflight/stage_reference_resets_v2_10.pkl")
FLIGHT_INITIAL = Path("runs/decoupled_bootstrap_seed0_20260720/frozen/pi_f_init")
ASCENT_ATTEMPT = Path("runs/stage_next_takeoff_keyposture_seed0_20260723/ascent/local_expert_006400/train/policy")
DESCENT_SUPPORT = Path("runs/stage_next_bootstrap_seed0_20260720/support_v2/descent_proposal_support_v1.pkl")
LANDING_BANK = Path("artifacts/landing_tube.pkl")
LANDING_POLICY = Path("runs/landing/refinement_seed0/policy")


def _controller_counts(report: Path, name: str) -> tuple[int, int, int]:
    row = json.loads(report.read_text())["controllers"][name]["strata"]
    return (int(row["canonical_compressed"]["successful_unique_states"]),
            int(row["reference_aligned_compressed"]["successful_unique_states"]),
            int(row["all"]["successful_branches"]))


class StageNextV3Controller(Controller):
    def __init__(self, run: Path):
        super().__init__(run)
        if self.state.get("controller_type") != "stage_next_reset_v3":
            self.state = {
                "controller_type": "stage_next_reset_v3", "controller_version": 1,
                "controller_module": "cli.stage_next_v3_controller",
                "controller_unit": "dvgc-stage-next-v3-controller.service",
                "run_id": run.name, "current_stage": "freeze_takeoff_reset_v3",
                "last_completed_action": None, "in_progress_action": None,
                "expected_outputs": [], "next_decision": "revalidate_takeoff_controller_bank",
                "retry_count": 0, "heartbeat": time.time(), "stop_reason": None,
                "terminal_state": None, "research_gate_valid": False,
                "global_status": "stage_reachability_acquisition",
                "history": [], "stage_status": {},
            }
            self.save()

    def _worker(self, action, command, log, outputs):
        result = self.run_worker_command(
            action, command, log, outputs,
            unit_suffix=f"stage-next-v3-{action}-{int(time.time())}", preallocate=False,
        )
        if not result["ok"]:
            raise RuntimeError(f"{action} failed: {result}")

    def freeze(self):
        for path in (TAKEOFF_BANK, TAKEOFF_EVAL, OLD_TAKEOFF / "params.pkl",
                     NEW_TAKEOFF / "params.pkl", Path("docs/RUNTIME_GATE.json")):
            if not path.exists():
                raise RuntimeError(f"missing immutable input {path}")
        gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
        if gate.get("status") != "PASS":
            raise RuntimeError("runtime gate is not PASS")
        save_json(Path("runs/ACTIVE_PIPELINE.json"), {
            "status": "ACTIVE", "activated_at": time.time(), "run_path": str(self.run),
            "controller_unit": "dvgc-stage-next-v3-controller.service",
            "start_script": "scripts/start_stage_next_v3_controller.sh",
            "supersedes": "runs/stage_next_takeoff_keyposture_seed0_20260723",
            "supersession_reason": "FROZEN_TAKEOFF_RESET_V3_STAGE_NEXT_CONTINUATION",
        })
        self.save(
            current_stage="revalidate_takeoff_controller_bank",
            last_completed_action="freeze_takeoff_reset_v3",
            next_decision="revalidate_takeoff_controller_bank",
            provenance={
                "xml_sha256": file_sha256("assets/orange_bike_4kg_horizontal.xml"),
                "takeoff_bank_sha256": file_sha256(TAKEOFF_BANK),
                "takeoff_eval_bank_sha256": file_sha256(TAKEOFF_EVAL),
                "landing_policy_sha256": file_sha256(LANDING_POLICY / "params.pkl"),
                "landing_bank_sha256": file_sha256(LANDING_BANK),
            },
            takeoff_reset_protocol_status="accepted_and_frozen",
        )

    def revalidate(self):
        if not BASELINE.exists():
            self._worker("takeoff_controller_bank_baseline", [
                PYTHON, "-u", "-m", "cli.evaluate_takeoff_controllers",
                "--bank", TAKEOFF_EVAL,
                "--policy", f"old_takeoff={OLD_TAKEOFF}",
                "--policy", f"new_takeoff={NEW_TAKEOFF}",
                "--include-bounded-sequences", "--output", BASELINE,
                "--branches", "4", "--horizon", "200", "--seed", "10100000",
            ], RUN / "takeoff/controller_bank_baseline_24x4_controller.log", [BASELINE])
        reward = RUN / "takeoff/reward_pretraining_diagnostic.json"
        curriculum = RUN / "takeoff/specialist_curriculum/report.json"
        if not reward.exists():
            self.run_command("takeoff_reward_pretraining_diagnostic", [
                PYTHON, "-m", "cli.analyze_takeoff_reward_diagnostic",
                "--evaluation", BASELINE, "--output", reward,
            ], RUN / "takeoff/reward_pretraining_diagnostic.log", [reward])
        if json.loads(reward.read_text())["status"] != "PASS":
            self.state["stage_status"]["takeoff"] = {
                "status": "reward_diagnostic_blocker", "report": str(reward),
            }
            self.save(current_stage="diagnostic_milestone", last_completed_action="revalidate_takeoff_controller_bank",
                      next_decision="bounded_takeoff_reward_repair", stop_reason="takeoff reward diagnostic failed")
            return
        if not curriculum.exists():
            self.run_command("build_takeoff_specialist_curriculum", [
                PYTHON, "-m", "cli.build_takeoff_specialist_curriculum",
                "--source-bank", TAKEOFF_BANK, "--eval-bank", TAKEOFF_EVAL,
                "--baseline-evaluation", BASELINE,
                "--output-root", RUN / "takeoff/specialist_curriculum",
                "--output-report", curriculum,
            ], RUN / "takeoff/specialist_curriculum/build.log", [curriculum])
        old = _controller_counts(BASELINE, "old_takeoff")
        new = _controller_counts(BASELINE, "new_takeoff")
        self.state["stage_status"]["takeoff"] = {
            "status": "partial_controller_support", "old_policy_counts": old,
            "new_policy_counts": new, "baseline_report": str(BASELINE),
            "reward_diagnostic": str(reward),
        }
        self.save(current_stage="start_takeoff_canonical_specialist",
                  last_completed_action="revalidate_takeoff_controller_bank",
                  next_decision="train_takeoff_specialist_block_1",
                  takeoff_specialist_block=1, takeoff_specialist_resume=str(OLD_TAKEOFF),
                  takeoff_specialist_best=None, takeoff_specialist_best_score=None,
                  takeoff_canonical_stagnant_blocks=0, takeoff_previous_canonical=old[0])

    def specialist(self):
        block = int(self.state["takeoff_specialist_block"])
        resume = Path(self.state["takeoff_specialist_resume"])
        root = RUN / f"takeoff/canonical_specialist/block_{block}_{block*25600:06d}"
        train = root / "train"; evaluation = root / "evaluation_24x4.json"
        if not (train / "policy/params.pkl").exists():
            bank = RUN / f"takeoff/specialist_curriculum/block_{block}_reset_bank.pkl"
            self._worker(f"takeoff_specialist_train_b{block}", [
                PYTHON, "-u", "-m", "cli.train", "--stage", "takeoff",
                "--bank", bank, "--config", TAKEOFF_CONFIG, "--run", train,
                "--resume", resume, "--timesteps", "25600", "--num-envs", "80",
                "--num-eval-envs", "40", "--num-evals", "2", "--batch-size", "40",
                "--num-minibatches", "4", "--seed", "104", "--segment-index", str(block - 1),
            ], root / "train.log", [train / "policy/params.pkl", train / "training_metrics.json"])
        if not evaluation.exists():
            self._worker(f"takeoff_specialist_eval_b{block}", [
                PYTHON, "-u", "-m", "cli.evaluate_takeoff_controllers",
                "--bank", TAKEOFF_EVAL, "--policy", f"specialist={train/'policy'}",
                "--output", evaluation, "--branches", "4", "--horizon", "200",
                "--seed", str(10_200_000 + block * 10_000),
            ], root / "evaluation.log", [evaluation])
        canonical, aligned, branches = _controller_counts(evaluation, "specialist")
        score = [min(canonical, aligned), canonical + aligned, canonical, branches]
        best_score = self.state.get("takeoff_specialist_best_score")
        if best_score is None or score > best_score:
            best_score = score
            self.state["takeoff_specialist_best_score"] = score
            self.state["takeoff_specialist_best"] = str(train / "policy")
            self.state["takeoff_specialist_best_report"] = str(evaluation)
        previous = int(self.state.get("takeoff_previous_canonical", 0))
        stagnant = 0 if canonical > previous else int(self.state.get("takeoff_canonical_stagnant_blocks", 0)) + 1
        self.state.setdefault("takeoff_specialist_blocks", []).append({
            "block": block, "policy": str(train / "policy"), "report": str(evaluation),
            "canonical_unique": canonical, "reference_unique": aligned,
            "successful_branches": branches, "score": score,
        })
        stop = block >= 4 or stagnant >= 2
        if stop:
            self.save(current_stage="freeze_takeoff_controller_bank",
                      last_completed_action=f"takeoff_specialist_block_{block}",
                      next_decision="freeze_takeoff_controller_bank",
                      takeoff_canonical_stagnant_blocks=stagnant,
                      takeoff_previous_canonical=max(previous, canonical))
        else:
            self.save(takeoff_specialist_block=block + 1,
                      takeoff_specialist_resume=str(train / "policy"),
                      takeoff_canonical_stagnant_blocks=stagnant,
                      takeoff_previous_canonical=max(previous, canonical),
                      last_completed_action=f"takeoff_specialist_block_{block}",
                      next_decision=f"train_takeoff_specialist_block_{block+1}")

    def freeze_bank(self):
        best = Path(self.state["takeoff_specialist_best"])
        report = RUN / "takeoff/frozen_controller_bank_evaluation.json"
        if not report.exists():
            self._worker("freeze_takeoff_controller_bank", [
                PYTHON, "-u", "-m", "cli.evaluate_takeoff_controllers",
                "--bank", TAKEOFF_EVAL, "--policy", f"old_takeoff={OLD_TAKEOFF}",
                "--policy", f"new_takeoff={NEW_TAKEOFF}",
                "--policy", f"canonical_specialist={best}",
                "--include-bounded-sequences", "--output", report,
                "--branches", "4", "--horizon", "200", "--seed", "10250000",
            ], RUN / "takeoff/frozen_controller_bank_evaluation.log", [report])
        payload = json.loads(report.read_text())
        save_json(RUN / "takeoff/frozen_controller_bank.json", {
            "status": "PASS", "artifact_role": "controller_proposal_bank",
            "policies": [
                {"id": "old_takeoff", "path": str(OLD_TAKEOFF), "params_sha256": file_sha256(OLD_TAKEOFF / "params.pkl")},
                {"id": "new_takeoff", "path": str(NEW_TAKEOFF), "params_sha256": file_sha256(NEW_TAKEOFF / "params.pkl")},
                {"id": "canonical_specialist", "path": str(best), "params_sha256": file_sha256(best / "params.pkl")},
            ],
            "bounded_sequence_controller": True,
            "fixed_evaluation": str(report), "union": payload["union_of_controllers"],
            "not_certified_tube": True,
        })
        self.state["stage_status"]["takeoff"].update({
            "status": "controller_bank_frozen", "best_policy": str(best),
            "best_score": self.state["takeoff_specialist_best_score"],
            "frozen_bank_evaluation": str(report),
        })
        self.save(current_stage="build_apex_reset_bank_v3",
                  last_completed_action="freeze_takeoff_controller_bank",
                  next_decision="build_apex_reset_bank_v3")

    def apex_bank(self):
        bank = RUN / "apex/apex_reset_bank_v3.pkl"
        report = RUN / "apex/apex_reset_bank_v3.json"
        if not report.exists():
            self._worker("build_apex_reset_bank_v3", [
                PYTHON, "-u", "-m", "cli.build_apex_reset_bank_v3",
                "--flight-bank", EXACT_STAGE_BANK, "--output-bank", bank,
                "--output-report", report, "--target", "24",
                "--policy", f"flight_initial={FLIGHT_INITIAL}",
                "--policy", f"ascent_attempt={ASCENT_ATTEMPT}",
            ], RUN / "apex/apex_reset_bank_v3.log", [bank, report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"] = {
            "status": "reset_bank_ready" if payload["status"] == "PASS" else "reset_bank_local_blocker",
            "reset_bank_report": str(report),
        }
        self.save(current_stage="apex_bounded_support_search",
                  last_completed_action="build_apex_reset_bank_v3",
                  next_decision="apex_bounded_support_search", apex_reset_bank=str(bank))

    def apex_search(self):
        report = RUN / "apex/bounded_support_search.json"
        if not report.exists():
            self._worker("apex_bounded_support_search", [
                PYTHON, "-u", "-m", "cli.search_stage_support", "--stage", "apex",
                "--bank", self.state["apex_reset_bank"], "--support-bank", DESCENT_SUPPORT,
                "--policy", f"flight_initial={FLIGHT_INITIAL}",
                "--policy", f"ascent_attempt={ASCENT_ATTEMPT}",
                "--output", report, "--horizon", "80",
            ], RUN / "apex/bounded_support_search.log", [report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"].update({
            "bounded_search": str(report),
            "successful_unique_states": payload["successful_unique_states"],
            "successful_parent_count": payload["successful_parent_count"],
            "status": "bounded_support_found" if payload["successful_parent_count"] >= 2
            else "bounded_controller_support_gap",
        })
        self.save(current_stage="ascent_reverse_controllability",
                  last_completed_action="apex_bounded_support_search",
                  next_decision="ascent_reverse_controllability")

    def ascent_search(self):
        bank = RUN / "ascent/reverse_diagnostic_v4_6.pkl"
        report = RUN / "ascent/bounded_support_search.json"
        if not report.exists():
            self._worker("ascent_reverse_controllability", [
                PYTHON, "-u", "-m", "cli.search_stage_support", "--stage", "ascent",
                "--bank", bank, "--policy", f"flight_initial={FLIGHT_INITIAL}",
                "--policy", f"ascent_attempt={ASCENT_ATTEMPT}",
                "--output", report, "--horizon", "100",
            ], RUN / "ascent/bounded_support_search.log", [report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["ascent"] = {
            "status": "bounded_support_found" if payload["successful_parent_count"] >= 2
            else "bounded_controller_support_gap",
            "bounded_search": str(report),
            "successful_unique_states": payload["successful_unique_states"],
            "successful_parent_count": payload["successful_parent_count"],
            "strata": payload["strata"],
        }
        next_action = ("apex_local_training_if_supported"
                       if self.state["stage_status"]["apex"]["status"] == "bounded_support_found"
                       else "ascent_late_training_if_supported"
                       if self.state["stage_status"]["ascent"]["status"] == "bounded_support_found"
                       else "record_stage_local_blockers_and_continue_candidate_acquisition")
        self.save(current_stage="diagnostic_milestone", global_status="stage_reachability_acquisition",
                  last_completed_action="ascent_reverse_controllability",
                  next_decision=next_action, report_milestone_ready=True,
                  terminal_state=None, research_gate_valid=False)

    def loop(self):
        while True:
            self.save()
            stage = self.state["current_stage"]
            if stage == "freeze_takeoff_reset_v3": self.freeze()
            elif stage == "revalidate_takeoff_controller_bank": self.revalidate()
            elif stage == "start_takeoff_canonical_specialist": self.specialist()
            elif stage == "freeze_takeoff_controller_bank": self.freeze_bank()
            elif stage == "build_apex_reset_bank_v3": self.apex_bank()
            elif stage == "apex_bounded_support_search": self.apex_search()
            elif stage == "ascent_reverse_controllability": self.ascent_search()
            elif stage == "diagnostic_milestone":
                self.save()
                time.sleep(30)
            else:
                raise RuntimeError(f"unknown v3 stage {stage}")


def main():
    controller = StageNextV3Controller(RUN)
    try:
        raise SystemExit(controller.loop())
    except SystemExit:
        raise
    except Exception as exc:
        controller.save(
            retry_count=int(controller.state.get("retry_count", 0)) + 1,
            stop_reason=f"{type(exc).__name__}: {exc}",
            terminal_state=("engineering_failure_after_retries"
                            if int(controller.state.get("retry_count", 0)) + 1 >= 3 else None),
        )
        raise


if __name__ == "__main__":
    main()
