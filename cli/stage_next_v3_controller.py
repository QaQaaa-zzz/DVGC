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
TAKEOFF_EVAL = RUN / "takeoff/fixed_balanced_eval_v2_24.pkl"
BASELINE = RUN / "takeoff/controller_bank_baseline_v2_24x4.json"
OLD_TAKEOFF = Path("runs/stage_next_bootstrap_seed0_20260720/takeoff/blocks/block_1_025600/train/policy")
NEW_TAKEOFF = Path("runs/stage_next_takeoff_keyposture_seed0_20260723/takeoff/local_expert_006400/train/policy")
TAKEOFF_CONFIG = Path("runs/stage_next_takeoff_keyposture_seed0_20260723/controller_inputs_v4_balanced/takeoff/config.json")
EXACT_STAGE_BANK = Path("runs/stage_next_takeoff_keyposture_seed0_20260723/preflight/stage_reference_resets_v2_10.pkl")
FLIGHT_INITIAL = Path("runs/decoupled_bootstrap_seed0_20260720/frozen/pi_f_init")
ASCENT_ATTEMPT = Path("runs/stage_next_takeoff_keyposture_seed0_20260723/ascent/local_expert_006400/train/policy")
DESCENT_SUPPORT = Path("runs/stage_next_bootstrap_seed0_20260720/support_v2/descent_proposal_support_v1.pkl")
LANDING_BANK = Path("artifacts/landing_tube.pkl")
LANDING_POLICY = Path("runs/landing/refinement_seed0/policy")
DESCENT_POLICY = Path(
    "runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy"
)
FEEDBACK_ROOT = RUN / "apex/feedback_bridge_v1"


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
        elif (self.state.get("terminal_state") == "engineering_failure_after_retries"
              and self.state.get("current_stage") == "apex_bounded_support_search_r3"):
            self.state.setdefault("recovery_history", []).append({
                "recovered_at": time.time(),
                "failed_action": self.state.get("in_progress_action"),
                "prior_retry_count": self.state.get("retry_count"),
                "prior_error": self.state.get("stop_reason"),
                "fix": "None-safe dynamic snapshot apex latch restore",
                "runtime_gate_source_fingerprint": json.loads(
                    Path("docs/RUNTIME_GATE.json").read_text()
                ).get("source_fingerprint"),
            })
            self.save(retry_count=0, terminal_state=None, stop_reason=None,
                      active_worker_unit=None, in_progress_action=None,
                      expected_outputs=[],
                      next_decision="resume_missing_apex_bounded_support_search_r3")
        elif (self.state.get("current_stage") in (
                "audit_descent_support_runtime_compatibility",
                "audit_descent_feature_semantics",
                "apex_descent_multiknot_bounded_search",
                "mine_fourth_independent_apex_parent",
              )
              and not (run / "apex/interface_v5/parent_robustness_v2.json").exists()):
            self.state.setdefault("recovery_history", []).append({
                "recovered_at": time.time(),
                "fix": "complete fourth fresh dynamics seed without changing frozen variant grid",
                "superseded_diagnostic": str(
                    run / "apex/interface_v5/parent_robustness.json"
                ),
            })
            self.save(
                current_stage="reproduce_three_dynamic_parents",
                in_progress_action=None, active_worker_unit=None,
                expected_outputs=[],
                next_decision="reproduce_three_dynamic_parents_v2",
            )
        elif (
            self.state.get("current_stage") == "descent_support_interface_coverage_gap"
            and not (
                run / "apex/feedback_bridge_v1/fresh_validation/"
                "gate_reclassification_v2.json"
            ).exists()
        ):
            self.state.setdefault("recovery_history", []).append({
                "recovered_at": time.time(),
                "fix": (
                    "require no later physical failure for feedback bridge Gate A"
                ),
                "source_report": str(
                    run / "apex/feedback_bridge_v1/fresh_validation/report.json"
                ),
            })
            self.save(
                current_stage="formal_descent_support_and_final_recovery_test",
                next_decision="reclassify_transient_stable_segments",
                in_progress_action=None, active_worker_unit=None,
                expected_outputs=[],
            )

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

    def apex_bank_r2(self):
        bank = RUN / "apex/apex_reset_bank_v3_r2.pkl"
        report = RUN / "apex/apex_reset_bank_v3_r2.json"
        if not report.exists():
            self._worker("build_apex_reset_bank_v3_r2", [
                PYTHON, "-u", "-m", "cli.build_apex_reset_bank_v3",
                "--flight-bank", EXACT_STAGE_BANK, "--output-bank", bank,
                "--output-report", report, "--target", "24", "--branches", "3",
                "--policy", f"flight_initial={FLIGHT_INITIAL}",
                "--policy", f"ascent_attempt={ASCENT_ATTEMPT}",
            ], RUN / "apex/apex_reset_bank_v3_r2.log", [bank, report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["refinement_round_2"] = {
            "report": str(report), "status": payload["status"],
            "records": payload["records"], "dynamically_reached": payload["dynamically_reached"],
            "dynamic_parent_count": payload["dynamic_parent_count"],
        }
        self.save(current_stage="apex_bounded_support_search_r2",
                  last_completed_action="build_apex_reset_bank_v3_r2",
                  next_decision="apex_bounded_support_search_r2",
                  apex_refinement_round=2, apex_reset_bank_r2=str(bank))

    def apex_search_r2(self):
        report = RUN / "apex/bounded_support_search_r2.json"
        if not report.exists():
            self._worker("apex_bounded_support_search_r2", [
                PYTHON, "-u", "-m", "cli.search_stage_support", "--stage", "apex",
                "--bank", self.state["apex_reset_bank_r2"], "--support-bank", DESCENT_SUPPORT,
                "--policy", f"flight_initial={FLIGHT_INITIAL}",
                "--policy", f"ascent_attempt={ASCENT_ATTEMPT}",
                "--output", report, "--horizon", "80",
            ], RUN / "apex/bounded_support_search_r2.log", [report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["refinement_round_2"].update({
            "bounded_search": str(report),
            "successful_unique_states": payload["successful_unique_states"],
            "successful_parent_count": payload["successful_parent_count"],
        })
        self.save(current_stage="diagnostic_milestone_final",
                  last_completed_action="apex_bounded_support_search_r2",
                  next_decision="freeze_stage_controller_banks_then_label_pilots",
                  report_milestone_ready=True, terminal_state=None,
                  research_gate_valid=False)

    def apex_bank_r3(self):
        bank = RUN / "apex/apex_reset_bank_v3_r3.pkl"
        report = RUN / "apex/apex_reset_bank_v3_r3.json"
        if not report.exists():
            self._worker("build_apex_reset_bank_v3_r3", [
                PYTHON, "-u", "-m", "cli.build_apex_reset_bank_v3",
                "--flight-bank", EXACT_STAGE_BANK, "--output-bank", bank,
                "--output-report", report, "--target", "24", "--branches", "3",
                "--policy", f"flight_initial={FLIGHT_INITIAL}",
                "--policy", f"ascent_attempt={ASCENT_ATTEMPT}",
            ], RUN / "apex/apex_reset_bank_v3_r3.log", [bank, report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["refinement_round_3"] = {
            "report": str(report), "status": payload["status"],
            "records": payload["records"], "dynamically_reached": payload["dynamically_reached"],
            "dynamic_parent_count": payload["dynamic_parent_count"],
        }
        self.save(current_stage="apex_bounded_support_search_r3",
                  last_completed_action="build_apex_reset_bank_v3_r3",
                  next_decision="apex_bounded_support_search_r3",
                  apex_refinement_round=3, apex_reset_bank_r3=str(bank))

    def apex_search_r3(self):
        report = RUN / "apex/bounded_support_search_r3.json"
        if not report.exists():
            self._worker("apex_bounded_support_search_r3", [
                PYTHON, "-u", "-m", "cli.search_stage_support", "--stage", "apex",
                "--bank", self.state["apex_reset_bank_r3"], "--support-bank", DESCENT_SUPPORT,
                "--policy", f"flight_initial={FLIGHT_INITIAL}",
                "--policy", f"ascent_attempt={ASCENT_ATTEMPT}",
                "--output", report, "--horizon", "80",
            ], RUN / "apex/bounded_support_search_r3.log", [report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["refinement_round_3"].update({
            "bounded_search": str(report),
            "successful_unique_states": payload["successful_unique_states"],
            "successful_parent_count": payload["successful_parent_count"],
        })
        self.save(current_stage="diagnostic_milestone_final",
                  last_completed_action="apex_bounded_support_search_r3",
                  next_decision="freeze_stage_controller_banks_then_label_pilots",
                  report_milestone_ready=True, terminal_state=None,
                  research_gate_valid=False)

    def takeoff_labels(self):
        root = RUN / "takeoff/frozen_label_pilot_120x4x3"
        report = root / "labels.json"; entries = root / "entries.pkl"
        analysis = root / "analysis_v2.json"
        best = Path(self.state["takeoff_specialist_best"])
        if not report.exists():
            self._worker("takeoff_frozen_label_pilot_120x4x3", [
                PYTHON, "-u", "-m", "cli.stage_label_pilot",
                "--takeoff-bank", TAKEOFF_BANK, "--flight-bank", EXACT_STAGE_BANK,
                "--landing-bank", LANDING_BANK,
                "--flight-policy", OLD_TAKEOFF, "--flight-policy", NEW_TAKEOFF,
                "--flight-policy", best, "--landing-policy", LANDING_POLICY,
                "--output", report, "--entry-bank", entries,
                "--states-per-stage", "120", "--branches", "4", "--horizon", "200",
                "--action-noise", ".03", "--only-stage", "takeoff",
            ], root / "run.log", [report, entries])
        if not analysis.exists():
            self.run_command("analyze_takeoff_frozen_labels", [
                PYTHON, "-m", "cli.analyze_takeoff_frozen_labels",
                "--labels", report,
                "--fixed-controller-evaluation", RUN / "takeoff/frozen_controller_bank_evaluation.json",
                "--output", analysis,
            ], root / "analysis.log", [analysis])
        payload = json.loads(analysis.read_text())
        self.state["stage_status"]["takeoff"]["label_pilot"] = {
            "report": str(report), "analysis": str(analysis),
            "strata": payload["strata"],
            "source_confounding_resolved": payload[
                "source_confounding_resolved_for_model_training"
            ],
        }
        self.save(current_stage="reachability_model_pending",
                  last_completed_action="takeoff_frozen_label_pilot_120x4x3",
                  next_decision=("train_source_stratified_takeoff_reachability"
                                 if payload["model_training_authorized"]
                                 else "expand_frozen_takeoff_controller_support"),
                  terminal_state=None, research_gate_valid=False)

    def reachability_model(self):
        root = RUN / "takeoff/reachability_model_frozen_v4"
        model = root / "model.npz"; report = root / "report.json"
        proposals = root / "ranked_proposals.json"
        analysis = RUN / "takeoff/frozen_label_pilot_120x4x3/analysis_v2.json"
        labels = RUN / "takeoff/frozen_label_pilot_120x4x3/labels.json"
        if not analysis.exists():
            self.run_command("reanalyze_takeoff_frozen_labels_v2", [
                PYTHON, "-m", "cli.analyze_takeoff_frozen_labels",
                "--labels", labels,
                "--fixed-controller-evaluation", RUN / "takeoff/frozen_controller_bank_evaluation.json",
                "--output", analysis,
            ], RUN / "takeoff/frozen_label_pilot_120x4x3/analysis_v2.log", [analysis])
        if not json.loads(analysis.read_text())["model_training_authorized"]:
            self.save(current_stage="stage_local_blockers_recorded",
                      next_decision="expand_takeoff_candidate_support_before_model")
            return
        if not report.exists():
            self.run_command("train_source_stratified_takeoff_reachability", [
                PYTHON, "-m", "cli.train_stage_reachability_model",
                "--bank", TAKEOFF_BANK, "--labels", labels,
                "--output-model", model, "--output-report", report,
                "--output-proposals", proposals, "--seed", "9841000",
            ], root / "train.log", [model, report, proposals])
        self.state["stage_status"]["takeoff"]["reachability_model"] = {
            "report": str(report), "model": str(model),
            "artifact_role": "conditional_proposal_support_model",
        }
        self.save(current_stage="stage_local_blockers_recorded",
                  last_completed_action="train_source_stratified_takeoff_reachability",
                  next_decision="acquire_additional_ascent_apex_trajectory_parents",
                  terminal_state=None, research_gate_valid=False)

    def acquire_ascent_apex_parents(self):
        root = RUN / "ascent/independent_parent_acquisition_v1"
        report = root / "report.json"
        if not report.exists():
            self._worker("acquire_ascent_apex_trajectory_parents", [
                PYTHON, "-u", "-m", "cli.acquire_ascent_apex_parents",
                "--takeoff-bank", TAKEOFF_BANK,
                "--reference-ascent-bank", RUN / "ascent/reverse_diagnostic_v4_6.pkl",
                "--descent-support-bank", DESCENT_SUPPORT,
                "--policy", f"old_takeoff={OLD_TAKEOFF}",
                "--policy", f"new_takeoff={NEW_TAKEOFF}",
                "--policy", f"canonical_specialist={self.state['takeoff_specialist_best']}",
                "--output-root", root, "--target-parents", "12",
                "--round-b-proposals", "96", "--seed", "10610000",
            ], root / "controller.log", [
                report, root / "fresh_ascent_entries.pkl",
                root / "dynamic_apex_proposals.pkl",
            ])
        payload = json.loads(report.read_text())
        parent_count = int(payload["successful_parent_count"])
        self.state["stage_status"]["ascent"]["independent_parent_acquisition"] = {
            "report": str(report),
            "fresh_ascent_entries": payload["fresh_ascent_entries"],
            "successful_parent_count": parent_count,
            "dynamic_apex_snapshots": payload["dynamic_apex_snapshots"],
            "stage_local_blocker": payload["stage_local_blocker"],
        }
        self.save(
            current_stage=("prepare_late_ascent_discovery"
                           if parent_count >= 2 else "ascent_multi_parent_controller_gap"),
            last_completed_action="acquire_ascent_apex_trajectory_parents",
            next_decision=("build_dynamic_apex_bank"
                           if parent_count >= 2 else "stage_local_gate_no_unbounded_ppo"),
            report_milestone_ready=True, terminal_state=None,
            research_gate_valid=False,
        )

    def prepare_late_ascent(self):
        acquisition = RUN / "ascent/independent_parent_acquisition_v1"
        bc = RUN / "ascent/late_discovery/bc_policy"
        bc_report = RUN / "ascent/late_discovery/bc_report.json"
        curriculum = RUN / "ascent/late_discovery/curriculum_v2/report.json"
        config = Path(
            "runs/stage_next_takeoff_keyposture_seed0_20260723/"
            "controller_inputs_v4_ascent/ascent/config.json"
        )
        if not bc_report.exists():
            self._worker("behavior_clone_late_ascent_sequences", [
                PYTHON, "-u", "-m", "cli.behavior_clone_ascent_sequences",
                "--entry-bank", acquisition / "fresh_ascent_entries.pkl",
                "--acquisition-report", acquisition / "report.json",
                "--initial-policy", FLIGHT_INITIAL, "--output-policy", bc,
                "--output-report", bc_report, "--config", config,
            ], RUN / "ascent/late_discovery/bc.log", [
                bc / "params.pkl", bc / "manifest.json", bc_report,
            ])
        if not curriculum.exists():
            self.run_command("build_late_ascent_curriculum", [
                PYTHON, "-m", "cli.build_late_ascent_curriculum",
                "--ascent-entry-bank", acquisition / "fresh_ascent_entries.pkl",
                "--dynamic-apex-bank", acquisition / "dynamic_apex_proposals.pkl",
                "--acquisition-report", acquisition / "report.json",
                "--output-root", RUN / "ascent/late_discovery/curriculum_v2",
            ], RUN / "ascent/late_discovery/curriculum_v2.log", [curriculum])
        self.save(
            current_stage="late_ascent_discovery_training",
            last_completed_action="prepare_late_ascent_discovery",
            next_decision="train_late_ascent_block_1",
            late_ascent_block=1, late_ascent_resume=str(bc),
            late_ascent_best=None, late_ascent_best_score=None,
            late_ascent_stagnant_blocks=0, late_ascent_best_parent_count=0,
            late_ascent_best_unique_count=0,
        )

    def train_late_ascent(self):
        block = int(self.state["late_ascent_block"])
        root = RUN / f"ascent/late_discovery/block_{block}_{block*25600:06d}"
        train = root / "train"
        evaluation = root / "evaluation_reference_6.json"
        config = Path(
            "runs/stage_next_takeoff_keyposture_seed0_20260723/"
            "controller_inputs_v4_ascent/ascent/config.json"
        )
        if not (train / "policy/params.pkl").exists():
            self._worker(f"late_ascent_train_b{block}", [
                PYTHON, "-u", "-m", "cli.train", "--stage", "flight",
                "--bank", RUN / f"ascent/late_discovery/curriculum_v2/block_{block}_reset_bank.pkl",
                "--config", config, "--run", train,
                "--resume", self.state["late_ascent_resume"],
                "--timesteps", "25600", "--num-envs", "80",
                "--num-eval-envs", "40", "--num-evals", "2",
                "--batch-size", "40", "--num-minibatches", "4",
                "--seed", "106", "--segment-index", str(block - 1),
            ], root / "train.log", [
                train / "policy/params.pkl", train / "training_metrics.json",
            ])
        if not evaluation.exists():
            self._worker(f"late_ascent_eval_b{block}", [
                PYTHON, "-u", "-m", "cli.search_stage_support",
                "--stage", "ascent",
                "--bank", RUN / "ascent/reverse_diagnostic_v4_6.pkl",
                "--policy", f"late_ascent={train/'policy'}",
                "--output", evaluation, "--horizon", "100",
                "--seed", str(10_700_000 + block * 10_000),
            ], root / "evaluation.log", [evaluation])
        payload = json.loads(evaluation.read_text())
        policy_rows = [
            row for row in payload["outcomes"]
            if row["controller"] == "policy:late_ascent"
        ]
        parents = len({
            row["trajectory_parent"] for row in policy_rows if row["success"]
        })
        unique = len({
            row["candidate_id"] for row in policy_rows if row["success"]
        })
        branches = sum(bool(row["success"]) for row in policy_rows)
        score = [parents, unique, branches]
        best_score = self.state.get("late_ascent_best_score")
        if best_score is None or score > best_score:
            self.state["late_ascent_best_score"] = score
            self.state["late_ascent_best"] = str(train / "policy")
            self.state["late_ascent_best_report"] = str(evaluation)
        improved = (
            parents > int(self.state.get("late_ascent_best_parent_count", 0))
            or unique > int(self.state.get("late_ascent_best_unique_count", 0))
        )
        stagnant = 0 if improved else int(
            self.state.get("late_ascent_stagnant_blocks", 0)
        ) + 1
        self.state.setdefault("late_ascent_blocks", []).append({
            "block": block, "policy": str(train / "policy"),
            "evaluation": str(evaluation), "successful_parents": parents,
            "successful_unique_states": unique,
            "successful_branches": branches, "score": score,
        })
        stop = block >= 4 or stagnant >= 2
        if stop:
            self.save(
                current_stage="build_dynamic_apex_bank",
                last_completed_action=f"late_ascent_discovery_block_{block}",
                next_decision="build_dynamic_apex_bank",
                late_ascent_stagnant_blocks=stagnant,
                late_ascent_best_parent_count=max(
                    parents, int(self.state.get("late_ascent_best_parent_count", 0))
                ),
                late_ascent_best_unique_count=max(
                    unique, int(self.state.get("late_ascent_best_unique_count", 0))
                ),
            )
        else:
            self.save(
                late_ascent_block=block + 1,
                late_ascent_resume=str(train / "policy"),
                late_ascent_stagnant_blocks=stagnant,
                late_ascent_best_parent_count=max(
                    parents, int(self.state.get("late_ascent_best_parent_count", 0))
                ),
                late_ascent_best_unique_count=max(
                    unique, int(self.state.get("late_ascent_best_unique_count", 0))
                ),
                last_completed_action=f"late_ascent_discovery_block_{block}",
                next_decision=f"train_late_ascent_block_{block+1}",
            )

    def build_dynamic_apex(self):
        root = RUN / "apex/dynamic_bank_v4"
        bank = root / "bank.pkl"; report = root / "report.json"
        if not report.exists():
            self.run_command("assemble_dynamic_apex_bank", [
                PYTHON, "-m", "cli.assemble_dynamic_apex_bank",
                "--base-bank", RUN / "apex/apex_reset_bank_v3_r3.pkl",
                "--new-bank", RUN / "ascent/independent_parent_acquisition_v1/dynamic_apex_proposals.pkl",
                "--output-bank", bank, "--output-report", report,
            ], root / "build.log", [bank, report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["dynamic_bank_v4"] = {
            "report": str(report), "bank": str(bank),
            "status": payload["status"],
            "reference_reset_valid": payload["reference_reset_valid"],
            "dynamically_reached": payload["dynamically_reached"],
            "dynamic_parent_count": payload["dynamic_parent_count"],
        }
        self.save(
            current_stage="apex_descent_bounded_search_v4",
            last_completed_action="build_dynamic_apex_bank",
            next_decision="apex_descent_bounded_search",
        )

    def apex_search_v4(self):
        root = RUN / "apex/dynamic_bank_v4"
        report = root / "bounded_descent_search.json"
        if not report.exists():
            policies = [
                "--policy", f"flight_initial={FLIGHT_INITIAL}",
                "--policy", f"ascent_attempt={ASCENT_ATTEMPT}",
            ]
            if self.state.get("late_ascent_best"):
                policies += [
                    "--policy", f"late_ascent={self.state['late_ascent_best']}"
                ]
            self._worker("apex_descent_bounded_search_v4", [
                PYTHON, "-u", "-m", "cli.search_stage_support",
                "--stage", "apex", "--bank", root / "bank.pkl",
                "--support-bank", DESCENT_SUPPORT, *policies,
                "--output", report, "--horizon", "80", "--seed", "10810000",
            ], root / "bounded_descent_search.log", [report])
        payload = json.loads(report.read_text())
        bank_report = json.loads((root / "report.json").read_text())
        authorized = (
            bank_report["status"] == "PASS"
            and int(payload["successful_parent_count"]) >= 2
        )
        self.state["stage_status"]["apex"]["dynamic_bank_v4"].update({
            "bounded_search": str(report),
            "descent_positive_unique": payload["successful_unique_states"],
            "descent_positive_parents": payload["successful_parent_count"],
            "apex_training_authorized": authorized,
        })
        self.save(
            current_stage=("apex_training_authorized" if authorized
                           else "apex_dynamic_support_stage_local_blocker"),
            last_completed_action="apex_descent_bounded_search_v4",
            next_decision=("train_apex_block_1" if authorized
                           else "acquire_more_dynamic_apex_parents_without_unbounded_ppo"),
            report_milestone_ready=True, terminal_state=None,
            research_gate_valid=False,
        )

    def freeze_interface_evidence(self):
        root = RUN / "apex/interface_v5"
        manifest = root / "frozen_inputs.json"
        if not manifest.exists():
            save_json(manifest, {
                "status": "PASS", "artifact_role": "frozen_apex_descent_interface_inputs",
                "apex_bank": str((RUN / "apex/dynamic_bank_v4/bank.pkl").resolve()),
                "apex_bank_sha256": file_sha256(RUN / "apex/dynamic_bank_v4/bank.pkl"),
                "support_bank": str(DESCENT_SUPPORT.resolve()),
                "support_bank_sha256": file_sha256(DESCENT_SUPPORT),
                "descent_policy": str(DESCENT_POLICY.resolve()),
                "descent_policy_sha256": file_sha256(DESCENT_POLICY / "params.pkl"),
                "landing_policy": str(LANDING_POLICY.resolve()),
                "landing_policy_sha256": file_sha256(LANDING_POLICY / "params.pkl"),
                "xml_sha256": file_sha256("assets/orange_bike_4kg_horizontal.xml"),
                "late_ascent_policy_role": "proposal_checkpoint_only",
                "late_ascent_policy_sha256": file_sha256(
                    RUN / "ascent/late_discovery/block_1_025600/train/policy/params.pkl"
                ),
                "matcher_radius_frozen": 2.213986224699026,
                "no_ppo_authorized": True,
            })
        self.state["stage_status"]["apex"]["evidence_classification"] = {
            "ascent_generic_apex": "multi_parent_dynamic_reachability_confirmed",
            "late_ascent_ppo": "training_neighborhood_memorization_without_parent_disjoint_generalization",
            "dynamic_apex_bank": "dynamic_apex_proposal_bank_incomplete",
            "local_blocker": "apex_descent_interface_support_blocker",
            "manifest": str(manifest),
        }
        self.save(
            current_stage="reproduce_three_dynamic_parents",
            last_completed_action="freeze_current_proposal_evidence",
            next_decision="reproduce_three_dynamic_parents",
            terminal_state=None, research_gate_valid=False,
        )

    def reproduce_dynamic_parents(self):
        root = RUN / "apex/interface_v5"
        report = root / "parent_robustness_v2.json"
        if not report.exists():
            self._worker("reproduce_three_dynamic_apex_parents", [
                PYTHON, "-u", "-m", "cli.reproduce_dynamic_apex_parents",
                "--reference-bank", RUN / "ascent/reverse_diagnostic_v4_6.pkl",
                "--entry-bank", RUN / "ascent/independent_parent_acquisition_v1/fresh_ascent_entries.pkl",
                "--acquisition-report", RUN / "ascent/independent_parent_acquisition_v1/report.json",
                "--output", report, "--seed", "10900000",
            ], root / "parent_robustness_v2.log", [report])
        self.state["stage_status"]["apex"]["parent_robustness"] = {
            "report": str(report),
            "parents": json.loads(report.read_text())["parents"],
        }
        self.save(
            current_stage="audit_descent_support_runtime_compatibility",
            last_completed_action="reproduce_three_dynamic_parents",
            next_decision="audit_descent_support_runtime_compatibility",
        )

    def audit_descent_runtime(self):
        root = RUN / "apex/interface_v5"
        report = root / "descent_support_runtime_audit.json"
        if not report.exists():
            self._worker("audit_descent_support_runtime_compatibility", [
                PYTHON, "-u", "-m", "cli.audit_descent_support_compatibility",
                "--support-bank", DESCENT_SUPPORT,
                "--descent-policy", DESCENT_POLICY,
                "--landing-policy", LANDING_POLICY,
                "--output", report, "--branches", "4", "--horizon", "200",
                "--seed", "11000000",
            ], root / "descent_support_runtime_audit.log", [report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["descent_runtime_compatibility"] = {
            "report": str(report),
            "runtime_stale": payload["descent_support_runtime_stale"],
            "reset_valid_rate": payload["reset_valid_rate"],
            "descent_controller_success_rate": payload["descent_controller_success_rate"],
            "landing_final_recovery_rate": payload["landing_final_recovery_rate"],
        }
        self.save(
            current_stage="audit_descent_feature_semantics",
            last_completed_action="audit_descent_support_runtime_compatibility",
            next_decision="audit_descent_feature_semantics",
        )

    def audit_descent_features(self):
        root = RUN / "apex/interface_v5"
        report = root / "descent_feature_semantics.json"
        if not report.exists():
            self.run_command("audit_descent_feature_semantics", [
                PYTHON, "-m", "cli.audit_descent_feature_semantics",
                "--apex-bank", RUN / "apex/dynamic_bank_v4/bank.pkl",
                "--support-bank", DESCENT_SUPPORT, "--output", report,
            ], root / "descent_feature_semantics.log", [report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["feature_semantics"] = {
            "report": str(report),
            "deterministic_error": payload["deterministic_semantics_error_found"],
            "angle_rank_changes": payload["angle_wrapping_changed_nearest_count"],
            "drop_x_rank_changes": payload["drop_absolute_x_changed_nearest_count"],
        }
        self.save(
            current_stage="apex_descent_multiknot_bounded_search",
            last_completed_action="audit_descent_feature_semantics",
            next_decision="apex_descent_multiknot_bounded_search",
        )

    def multiknot_apex_search(self):
        root = RUN / "apex/interface_v5"
        report = root / "multiknot_search.json"
        if not report.exists():
            self._worker("apex_descent_multiknot_bounded_search", [
                PYTHON, "-u", "-m", "cli.search_apex_descent_multiknot",
                "--apex-bank", RUN / "apex/dynamic_bank_v4/bank.pkl",
                "--support-bank", DESCENT_SUPPORT,
                "--descent-policy", DESCENT_POLICY,
                "--landing-policy", LANDING_POLICY,
                "--output", report, "--horizon", "100",
                "--downstream-horizon", "200",
                "--round-b-proposals", "48", "--seed", "11100000",
            ], root / "multiknot_search.log", [report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["multiknot_search"] = {
            "report": str(report),
            "descent_positive_unique": payload["dynamic_descent_positive_unique"],
            "descent_positive_parents": payload["dynamic_descent_positive_parents"],
            "final_recovery_branches": payload["final_recovery_branches"],
            "failure_modes": payload["failure_modes"],
        }
        self.save(
            current_stage="mine_fourth_independent_apex_parent",
            last_completed_action="classify_interface_failure_modes",
            next_decision="mine_fourth_independent_apex_parent",
            terminal_state=None, research_gate_valid=False,
        )

    def mine_fourth_apex_parent(self):
        root = RUN / "ascent/independent_parent_acquisition_v2_24"
        report = root / "report.json"
        if not report.exists():
            self._worker("mine_fourth_independent_apex_parent", [
                PYTHON, "-u", "-m", "cli.acquire_ascent_apex_parents",
                "--takeoff-bank", TAKEOFF_BANK,
                "--reference-ascent-bank", RUN / "ascent/reverse_diagnostic_v4_6.pkl",
                "--descent-support-bank", DESCENT_SUPPORT,
                "--policy", f"old_takeoff={OLD_TAKEOFF}",
                "--policy", f"new_takeoff={NEW_TAKEOFF}",
                "--policy", f"canonical_specialist={self.state['takeoff_specialist_best']}",
                "--output-root", root, "--target-parents", "24",
                "--entry-pool-multiplier", "2", "--takeoff-policy-only",
                "--required-successful-parents", "1",
                "--round-b-proposals", "96", "--seed", "11200000",
            ], root / "controller.log", [
                report, root / "fresh_ascent_entries.pkl",
                root / "dynamic_apex_proposals.pkl",
            ])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["fourth_parent_mining"] = {
            "report": str(report),
            "fresh_entries": payload["fresh_ascent_entries"],
            "source_mix": payload["fresh_ascent_entry_source_mix"],
            "controller_mix": payload["fresh_ascent_entry_controller_mix"],
            "new_successful_parents": payload["successful_parent_count"],
            "new_dynamic_snapshots": payload["dynamic_apex_snapshots"],
        }
        self.save(
            current_stage="expand_dynamic_apex_bank_v5",
            last_completed_action="mine_fourth_independent_apex_parent",
            next_decision="expand_dynamic_apex_bank",
        )

    def expand_dynamic_apex_v5(self):
        root = RUN / "apex/dynamic_bank_v5"
        bank = root / "bank.pkl"; report = root / "report.json"
        if not report.exists():
            self.run_command("expand_dynamic_apex_bank_v5", [
                PYTHON, "-m", "cli.assemble_dynamic_apex_bank",
                "--base-bank", RUN / "apex/dynamic_bank_v4/bank.pkl",
                "--new-bank", RUN / "ascent/independent_parent_acquisition_v2_24/dynamic_apex_proposals.pkl",
                "--output-bank", bank, "--output-report", report,
            ], root / "build.log", [bank, report])
        payload = json.loads(report.read_text())
        self.state["stage_status"]["apex"]["dynamic_bank_v5"] = {
            "report": str(report), "bank": str(bank),
            "status": payload["status"],
            "reference_reset_valid": payload["reference_reset_valid"],
            "dynamically_reached": payload["dynamically_reached"],
            "dynamic_parent_count": payload["dynamic_parent_count"],
        }
        self.save(
            current_stage="apex_descent_multiknot_expanded_bank",
            last_completed_action="expand_dynamic_apex_bank",
            next_decision="rerun_multiknot_on_new_dynamic_states",
        )

    def multiknot_expanded_apex_search(self):
        root = RUN / "apex/dynamic_bank_v5"
        report = root / "multiknot_search.json"
        if not report.exists():
            self._worker("apex_descent_multiknot_expanded_bank", [
                PYTHON, "-u", "-m", "cli.search_apex_descent_multiknot",
                "--apex-bank", root / "bank.pkl",
                "--support-bank", DESCENT_SUPPORT,
                "--descent-policy", DESCENT_POLICY,
                "--landing-policy", LANDING_POLICY,
                "--output", report, "--horizon", "100",
                "--downstream-horizon", "200",
                "--round-b-proposals", "48", "--seed", "11300000",
            ], root / "multiknot_search.log", [report])
        payload = json.loads(report.read_text())
        bank_report = json.loads((root / "report.json").read_text())
        runtime = json.loads(
            (RUN / "apex/interface_v5/descent_support_runtime_audit.json").read_text()
        )
        authorization_checks = {
            "dynamic_bank_16_to_32_and_four_parents": bank_report["status"] == "PASS",
            "two_descent_positive_parents": payload["dynamic_descent_positive_parents"] >= 2,
            "successful_sequences_fresh_seed_reproduced": False,
            "descent_support_runtime_compatible": not runtime["descent_support_runtime_stale"],
            "reward_diagnostic_passed": False,
            "reset_and_full_runtime_gate_passed_after_success": False,
        }
        authorized = all(authorization_checks.values())
        self.state["stage_status"]["apex"]["expanded_multiknot_search"] = {
            "report": str(report),
            "descent_positive_unique": payload["dynamic_descent_positive_unique"],
            "descent_positive_parents": payload["dynamic_descent_positive_parents"],
            "final_recovery_branches": payload["final_recovery_branches"],
            "failure_modes": payload["failure_modes"],
            "authorization_checks": authorization_checks,
            "apex_training_authorized": authorized,
        }
        self.save(
            current_stage=("apex_training_authorized" if authorized
                           else "apex_descent_interface_support_blocker"),
            last_completed_action="apex_descent_multiknot_expanded_bank",
            next_decision=("apex_training_if_and_only_if_authorized" if authorized
                           else "stage_local_interface_diagnosis_complete"),
            report_milestone_ready=True, terminal_state=None,
            research_gate_valid=False,
        )

    def freeze_v5_bridge_evidence(self):
        inputs = {
            "xml_sha256": file_sha256("assets/orange_bike_4kg_horizontal.xml"),
            "apex_bank_sha256": file_sha256(RUN / "apex/dynamic_bank_v5/bank.pkl"),
            "multiknot_report_sha256": file_sha256(
                RUN / "apex/dynamic_bank_v5/multiknot_search.json"
            ),
            "descent_support_sha256": file_sha256(DESCENT_SUPPORT),
            "descent_runtime_audit_sha256": file_sha256(
                RUN / "apex/interface_v5/descent_support_runtime_audit.json"
            ),
            "landing_policy_sha256": file_sha256(LANDING_POLICY / "params.pkl"),
        }
        save_json(FEEDBACK_ROOT / "frozen_inputs.json", {
            "status": "PASS", "artifact_role": "frozen_apex_bridge_inputs",
            "formal_matcher_unchanged": True, "apex_ppo_authorized": False,
            **inputs,
        })
        self.save(
            current_stage="analyze_existing_branch_failure_timing",
            last_completed_action="freeze_v5_evidence",
            next_decision="analyze_existing_branch_failure_timing",
        )

    def analyze_bridge_failure_timing(self):
        report = FEEDBACK_ROOT / "failure_timing.json"
        if not report.exists():
            self.run_command("analyze_existing_branch_failure_timing", [
                PYTHON, "-m", "cli.analyze_apex_bridge_timing",
                "--search-report", RUN / "apex/dynamic_bank_v5/multiknot_search.json",
                "--apex-bank", RUN / "apex/dynamic_bank_v5/bank.pkl",
                "--output", report,
            ], FEEDBACK_ROOT / "failure_timing.log", [report])
        self.save(
            current_stage="audit_pre_post_apex_control_authority",
            last_completed_action="analyze_existing_branch_failure_timing",
            next_decision="audit_pre_post_apex_control_authority",
        )

    def audit_apex_control_authority(self):
        report = FEEDBACK_ROOT / "control_authority.json"
        bank = FEEDBACK_ROOT / "control_authority_snapshots.pkl"
        if not report.exists():
            self._worker("audit_pre_post_apex_control_authority", [
                PYTHON, "-u", "-m", "cli.audit_apex_control_authority",
                "--apex-bank", RUN / "apex/dynamic_bank_v5/bank.pkl",
                "--run-root", RUN, "--output-bank", bank,
                "--output-report", report, "--seed", "11400000",
            ], FEEDBACK_ROOT / "control_authority.log", [report, bank])
        self.save(
            current_stage="build_current_runtime_descent_terminal_clusters",
            last_completed_action="audit_pre_post_apex_control_authority",
            next_decision="build_current_runtime_descent_terminal_clusters",
        )

    def build_descent_terminal_clusters(self):
        report = FEEDBACK_ROOT / "descent_terminal_clusters_current.json"
        bank = FEEDBACK_ROOT / "descent_terminal_proposals_current.pkl"
        if not report.exists():
            self.run_command("build_current_runtime_descent_terminal_clusters", [
                PYTHON, "-m", "cli.build_descent_terminal_clusters",
                "--support-bank", DESCENT_SUPPORT,
                "--runtime-audit",
                RUN / "apex/interface_v5/descent_support_runtime_audit.json",
                "--apex-bank", RUN / "apex/dynamic_bank_v5/bank.pkl",
                "--output-bank", bank, "--output-report", report,
            ], FEEDBACK_ROOT / "descent_terminal_clusters_current.log",
                [report, bank])
        self.save(
            current_stage="select_bridge_start_before_apex",
            last_completed_action="build_current_runtime_descent_terminal_clusters",
            next_decision="select_bridge_start_before_apex",
        )

    def select_apex_bridge_starts(self):
        report = FEEDBACK_ROOT / "bridge_start_selection.json"
        bank = FEEDBACK_ROOT / "bridge_start_selection.pkl"
        if not report.exists():
            self.run_command("select_bridge_start_before_apex", [
                PYTHON, "-m", "cli.select_apex_bridge_starts",
                "--authority-bank", FEEDBACK_ROOT / "control_authority_snapshots.pkl",
                "--authority-report", FEEDBACK_ROOT / "control_authority.json",
                "--parent", "reference:131",
                "--parent",
                "89ff1a0e3cb74319b16742932c97decf38be3a39a100d49e855613162e23fcf0",
                "--output-bank", bank, "--output-report", report,
            ], FEEDBACK_ROOT / "bridge_start_selection.log", [report, bank])
        self.save(
            current_stage="discover_nominal_stable_descent_trajectory",
            last_completed_action="select_bridge_start_before_apex",
            next_decision="discover_nominal_stable_descent_trajectory",
        )

    def discover_nominal_feedback_bridge(self):
        root = FEEDBACK_ROOT / "nominal_bridge"
        report = root / "report.json"
        if not report.exists():
            self._worker("discover_nominal_stable_descent_trajectory", [
                PYTHON, "-u", "-m", "cli.discover_apex_feedback_bridge",
                "--authority-bank", FEEDBACK_ROOT / "control_authority_snapshots.pkl",
                "--authority-report", FEEDBACK_ROOT / "control_authority.json",
                "--start-bank", FEEDBACK_ROOT / "bridge_start_selection.pkl",
                "--support-bank", DESCENT_SUPPORT,
                "--terminal-bank",
                FEEDBACK_ROOT / "descent_terminal_proposals_current.pkl",
                "--descent-policy", DESCENT_POLICY,
                "--landing-policy", LANDING_POLICY,
                "--output-root", root, "--parent", "reference:131",
                "--parent",
                "89ff1a0e3cb74319b16742932c97decf38be3a39a100d49e855613162e23fcf0",
                "--branch-mode", "deterministic", "--horizon", "40",
                "--lookahead", "3", "--downstream-horizon", "200",
                "--seed", "11500000",
            ], root / "controller.log", [
                report, root / "stable_physical_descent.pkl",
                root / "formal_descent_entries.pkl",
            ])
        self.save(
            current_stage="synthesize_local_feedback_bridge",
            last_completed_action="discover_nominal_stable_descent_trajectory",
            next_decision="synthesize_receding_horizon_feedback_bridge",
        )

    def synthesize_feedback_bridge(self):
        save_json(FEEDBACK_ROOT / "feedback_controller_manifest.json", {
            "status": "PASS",
            "controller": "receding_horizon_bounded_shooting",
            "replanning_interval_ticks": 1, "lookahead_ticks": 3,
            "event_relative_start_selection": True,
            "formal_success_is_not_soft_cost": True,
            "apex_ppo_authorized": False,
        })
        self.save(
            current_stage="fresh_seed_bridge_validation",
            last_completed_action="synthesize_local_feedback_bridge",
            next_decision="fresh_seed_bridge_validation",
        )

    def validate_fresh_feedback_bridge(self):
        root = FEEDBACK_ROOT / "fresh_validation"
        report = root / "report.json"
        if not report.exists():
            self._worker("fresh_seed_bridge_validation", [
                PYTHON, "-u", "-m", "cli.discover_apex_feedback_bridge",
                "--authority-bank", FEEDBACK_ROOT / "control_authority_snapshots.pkl",
                "--authority-report", FEEDBACK_ROOT / "control_authority.json",
                "--start-bank", FEEDBACK_ROOT / "bridge_start_selection.pkl",
                "--support-bank", DESCENT_SUPPORT,
                "--terminal-bank",
                FEEDBACK_ROOT / "descent_terminal_proposals_current.pkl",
                "--descent-policy", DESCENT_POLICY,
                "--landing-policy", LANDING_POLICY,
                "--output-root", root, "--parent", "reference:131",
                "--parent",
                "89ff1a0e3cb74319b16742932c97decf38be3a39a100d49e855613162e23fcf0",
                "--branch-mode", "fresh",
                "--nominal-report", FEEDBACK_ROOT / "nominal_bridge/report.json",
                "--horizon", "40", "--lookahead", "3",
                "--downstream-horizon", "200", "--seed", "11600000",
            ], root / "controller.log", [
                report, root / "stable_physical_descent.pkl",
                root / "formal_descent_entries.pkl",
            ])
        self.save(
            current_stage="formal_descent_support_and_final_recovery_test",
            last_completed_action="fresh_seed_bridge_validation",
            next_decision="formal_descent_support_and_final_recovery_test",
        )

    def classify_feedback_bridge(self):
        nominal = json.loads(
            (FEEDBACK_ROOT / "nominal_bridge/report.json").read_text()
        )
        fresh = json.loads(
            (FEEDBACK_ROOT / "fresh_validation/report.json").read_text()
        )
        physical_reasons = {
            "roll_limit", "pitch_limit", "prohibited_contact",
            "invalid_wheel_step_contact", "backward", "platform_back_edge_exit",
            "nonfinite",
        }
        clean_fresh_by_parent = {}
        for row in fresh["outcomes"]:
            if (row["stable_physical_descent"]
                    and row["termination_reason"] not in physical_reasons):
                clean_fresh_by_parent[row["parent"]] = (
                    clean_fresh_by_parent.get(row["parent"], 0) + 1
                )
        gate_a = any(value >= 2 for value in clean_fresh_by_parent.values())
        gate_b = bool(fresh["gate_b_formal_interface_feasibility"])
        reclassification = {
            "status": "PASS",
            "artifact_role": "feedback_bridge_gate_reclassification_v2",
            "source_report": str(
                FEEDBACK_ROOT / "fresh_validation/report.json"
            ),
            "source_report_sha256": file_sha256(
                FEEDBACK_ROOT / "fresh_validation/report.json"
            ),
            "short_stable_segments": fresh["hierarchy"][
                "stable_physical_descent"
            ],
            "stable_without_later_physical_failure_by_parent":
                clean_fresh_by_parent,
            "gate_a_local_physical_feasibility": gate_a,
            "gate_b_formal_interface_feasibility": gate_b,
            "supersedes_source_gate_a": True,
            "reason": (
                "Gate A requires the feedback bridge branch to avoid physical "
                "failure; a transient four-tick negative-vz segment followed "
                "by roll_limit is diagnostic evidence, not Gate A."
            ),
        }
        save_json(FEEDBACK_ROOT / "fresh_validation/gate_reclassification_v2.json",
                  reclassification)
        if gate_b:
            stage = "expand_bridge_admissible_apex_bank"
            decision = "expand_only_from_bridge_corridor"
            blocker = None
        elif gate_a:
            stage = "descent_support_interface_coverage_gap"
            decision = "audit_stable_descent_downstream_and_version_support"
            blocker = "stable_physical_descent_without_two_parent_formal_entry"
        else:
            stage = "pre_apex_feedback_bridge_stage_local_blocker"
            decision = "upstream_entry_shaping_required"
            blocker = "feedback_bridge_failed_gate_a"
        self.state["stage_status"]["apex"]["feedback_bridge_v1"] = {
            "failure_timing": str(FEEDBACK_ROOT / "failure_timing.json"),
            "control_authority": str(FEEDBACK_ROOT / "control_authority.json"),
            "terminal_clusters": str(
                FEEDBACK_ROOT / "descent_terminal_clusters_current.json"
            ),
            "nominal_report": str(FEEDBACK_ROOT / "nominal_bridge/report.json"),
            "fresh_report": str(FEEDBACK_ROOT / "fresh_validation/report.json"),
            "gate_reclassification": str(
                FEEDBACK_ROOT / "fresh_validation/gate_reclassification_v2.json"
            ),
            "nominal_hierarchy": nominal["hierarchy"],
            "fresh_hierarchy": fresh["hierarchy"],
            "gate_a": gate_a, "gate_b": gate_b,
            "gate_c_apex_ppo_authorized": False,
            "stage_local_blocker": blocker,
        }
        self.save(
            current_stage=stage,
            last_completed_action="formal_descent_support_and_final_recovery_test",
            next_decision=decision, report_milestone_ready=True,
            terminal_state=None, research_gate_valid=False,
        )

    def prepare_pre_apex_horizon_audit(self):
        root = FEEDBACK_ROOT / "horizon_momentum_v2"
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / "cost_estimate.json", {
            "status": "PASS",
            "task": "minimal_event_aligned_apex_horizon_audit",
            "prediction_horizons": [3, 6, 9, 12],
            "parents": 3, "maximum_event_aligned_starts_per_parent": 4,
            "action_candidates": 13, "controller_horizon_ticks": 40,
            "estimated_mjx_steps_upper_bound": 249600,
            "estimated_wall_time": "under_2_hours",
            "ppo_steps": 0, "generic_apex_search": False,
        })
        self.save(
            current_stage="minimal_pre_apex_horizon_audit",
            last_completed_action="prepare_pre_apex_horizon_audit",
            next_decision="minimal_pre_apex_horizon_audit",
        )

    def run_pre_apex_horizon_audit(self):
        root = FEEDBACK_ROOT / "horizon_momentum_v2"
        report = root / "horizon_audit.json"
        if not report.exists():
            self._worker("minimal_pre_apex_horizon_audit", [
                PYTHON, "-u", "-m", "cli.audit_apex_mpc_horizons",
                "--authority-bank", FEEDBACK_ROOT / "control_authority_snapshots.pkl",
                "--authority-report", FEEDBACK_ROOT / "control_authority.json",
                "--support-bank", DESCENT_SUPPORT,
                "--terminal-bank",
                FEEDBACK_ROOT / "descent_terminal_proposals_current.pkl",
                "--descent-policy", DESCENT_POLICY,
                "--landing-policy", LANDING_POLICY,
                "--output", report,
                "--parent", "reference:131",
                "--parent",
                "89ff1a0e3cb74319b16742932c97decf38be3a39a100d49e855613162e23fcf0",
                "--parent",
                "2f3411630c83455a034be8083587a76e2901503c626cde1f2c0b578d68b4ca87",
                "--prediction-horizon", "3", "--prediction-horizon", "6",
                "--prediction-horizon", "9", "--prediction-horizon", "12",
                "--control-horizon", "2", "--controller-horizon", "40",
                "--seed", "11700000",
            ], root / "horizon_audit.log", [report])
        self.save(
            current_stage="apex_centroidal_contact_audit",
            last_completed_action="minimal_pre_apex_horizon_audit",
            next_decision="apex_centroidal_contact_audit",
        )

    def run_apex_centroidal_contact_audit(self):
        root = FEEDBACK_ROOT / "horizon_momentum_v2"
        report = root / "centroidal_contact_audit.json"
        if not report.exists():
            self.run_command("apex_centroidal_contact_audit", [
                PYTHON, "-m", "cli.audit_apex_centroidal_contact",
                "--authority-bank", FEEDBACK_ROOT / "control_authority_snapshots.pkl",
                "--authority-report", FEEDBACK_ROOT / "control_authority.json",
                "--terminal-bank",
                FEEDBACK_ROOT / "descent_terminal_proposals_current.pkl",
                "--terminal-report",
                FEEDBACK_ROOT / "descent_terminal_clusters_current.json",
                "--horizon-report", root / "horizon_audit.json",
                "--output", report,
            ], root / "centroidal_contact_audit.log", [report])
        result = json.loads(report.read_text())
        blocker = result["blocker_classification"]
        self.state["stage_status"]["apex"]["horizon_momentum_v2"] = {
            "horizon_audit": str(root / "horizon_audit.json"),
            "centroidal_contact_audit": str(report),
            "blocker_classification": blocker,
            "selected_event_aligned_entries": len(
                result["selected_event_aligned_entry_proposals"]
            ),
            "apex_ppo_authorized": False,
        }
        self.save(
            current_stage=blocker,
            last_completed_action="apex_centroidal_contact_audit",
            next_decision=(
                "contact_supported_momentum_shaping_required"
                if blocker == "takeoff_tail_centroidal_momentum_blocker"
                else "segmented_pre_apex_bridge_required"
            ),
            report_milestone_ready=True, terminal_state=None,
            research_gate_valid=False,
        )

    def audit_apex_upstream_lineage(self):
        root = FEEDBACK_ROOT / "contact_supported_v1"
        report = root / "upstream_lineage.json"
        bank = root / "event_aligned_entries.pkl"
        root.mkdir(parents=True, exist_ok=True)
        if not report.exists():
            self._worker("audit_apex_upstream_lineage", [
                PYTHON, "-u", "-m", "cli.audit_apex_upstream_lineage",
                "--run-root", RUN,
                "--takeoff-bank",
                "runs/stage_next_takeoff_keyposture_seed0_20260723/"
                "takeoff_reset_bank_v3_120.pkl",
                "--output-bank", bank, "--output-report", report,
                "--parent", "reference:131",
                "--parent",
                "89ff1a0e3cb74319b16742932c97decf38be3a39a100d49e855613162e23fcf0",
                "--parent",
                "2f3411630c83455a034be8083587a76e2901503c626cde1f2c0b578d68b4ca87",
                "--parent",
                "3161a8a4ac16bc7aeebf807ab9a8f821f87dc4776f86760a2cb9701f885697f7",
                "--parent",
                "5b73a426d2dfd65d1ad426b5d9d577f4b397eef4e514845e95a2e84af78569e4",
            ], root / "upstream_lineage.log", [report, bank])
        result = json.loads(report.read_text())
        if result["status"] != "PASS":
            raise RuntimeError("exact upstream lineage replay failed")
        self.save(
            current_stage="contact_supported_bridge_discovery",
            last_completed_action="audit_apex_upstream_lineage",
            next_decision="contact_supported_bridge_discovery",
        )

    def discover_contact_supported_bridge(self):
        root = FEEDBACK_ROOT / "contact_supported_v1"
        report = root / "segmented_bridge_nominal.json"
        if not report.exists():
            self._worker("contact_supported_bridge_discovery", [
                PYTHON, "-u", "-m",
                "cli.discover_contact_supported_apex_bridge",
                "--lineage-bank", root / "event_aligned_entries.pkl",
                "--lineage-report", root / "upstream_lineage.json",
                "--support-bank", DESCENT_SUPPORT,
                "--terminal-bank",
                FEEDBACK_ROOT / "descent_terminal_proposals_current.pkl",
                "--descent-policy", DESCENT_POLICY,
                "--landing-policy", LANDING_POLICY,
                "--output", report,
                "--parent",
                "89ff1a0e3cb74319b16742932c97decf38be3a39a100d49e855613162e23fcf0",
                "--parent",
                "2f3411630c83455a034be8083587a76e2901503c626cde1f2c0b578d68b4ca87",
                "--contact-candidates", "5", "--prediction-horizon", "12",
                "--controller-horizon", "40", "--seed", "11800000",
            ], root / "segmented_bridge_nominal.log", [report])
        result = json.loads(report.read_text())
        gate_a = result["summary"]["gate_a_nominal"]
        gate_b = result["summary"]["gate_b_nominal_two_formal_parents"]
        self.state["stage_status"]["apex"]["contact_supported_v1"] = {
            "upstream_lineage": str(root / "upstream_lineage.json"),
            "segmented_bridge_nominal": str(report),
            "gate_a_nominal": gate_a, "gate_b_nominal": gate_b,
            "apex_ppo_authorized": False,
        }
        if gate_a:
            stage = "contact_supported_bridge_fresh_validation"
            decision = "fresh_dynamics_gate_a_validation"
        else:
            stage = "takeoff_tail_centroidal_momentum_blocker"
            decision = "bridge_admissible_takeoff_tail_target_required"
        self.save(
            current_stage=stage,
            last_completed_action="contact_supported_bridge_discovery",
            next_decision=decision, report_milestone_ready=True,
            research_gate_valid=False, terminal_state=None,
        )

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
                if not self.state.get("apex_refinement_round"):
                    self.save(current_stage="build_apex_reset_bank_v3_r2",
                              next_decision="build_apex_reset_bank_v3_r2")
                    continue
                self.save()
                time.sleep(30)
            elif stage == "build_apex_reset_bank_v3_r2": self.apex_bank_r2()
            elif stage == "apex_bounded_support_search_r2": self.apex_search_r2()
            elif stage == "diagnostic_milestone_final":
                if int(self.state.get("apex_refinement_round", 0)) < 3:
                    self.save(current_stage="build_apex_reset_bank_v3_r3",
                              next_decision="build_apex_reset_bank_v3_r3")
                    continue
                self.save(current_stage="takeoff_frozen_label_pilot",
                          next_decision="takeoff_frozen_label_pilot_120x4x3")
            elif stage == "takeoff_frozen_label_pilot": self.takeoff_labels()
            elif stage == "reachability_model_pending": self.reachability_model()
            elif stage == "stage_local_blockers_recorded":
                if self.state.get("next_decision") == "acquire_additional_ascent_apex_trajectory_parents":
                    self.save(
                        current_stage="mine_independent_ascent_apex_parents",
                        next_decision="reproduce_parent_131_then_round_a_b",
                    )
                    continue
                self.save(); time.sleep(30)
            elif stage == "mine_independent_ascent_apex_parents":
                self.acquire_ascent_apex_parents()
            elif stage == "prepare_late_ascent_discovery":
                self.prepare_late_ascent()
            elif stage == "late_ascent_discovery_authorized":
                self.save(current_stage="prepare_late_ascent_discovery",
                          next_decision="behavior_clone_successful_sequences")
            elif stage == "late_ascent_discovery_training":
                self.train_late_ascent()
            elif stage == "build_dynamic_apex_bank":
                self.build_dynamic_apex()
            elif stage == "apex_descent_bounded_search_v4":
                self.apex_search_v4()
            elif stage in ("apex_training_authorized",
                           "ascent_multi_parent_controller_gap"):
                self.save(); time.sleep(30)
            elif stage == "apex_dynamic_support_stage_local_blocker":
                self.save(current_stage="freeze_current_proposal_evidence",
                          next_decision="freeze_current_proposal_evidence")
            elif stage == "freeze_current_proposal_evidence":
                self.freeze_interface_evidence()
            elif stage == "reproduce_three_dynamic_parents":
                self.reproduce_dynamic_parents()
            elif stage == "audit_descent_support_runtime_compatibility":
                self.audit_descent_runtime()
            elif stage == "audit_descent_feature_semantics":
                self.audit_descent_features()
            elif stage == "apex_descent_multiknot_bounded_search":
                self.multiknot_apex_search()
            elif stage == "mine_fourth_independent_apex_parent":
                self.mine_fourth_apex_parent()
            elif stage == "expand_dynamic_apex_bank_v5":
                self.expand_dynamic_apex_v5()
            elif stage == "apex_descent_multiknot_expanded_bank":
                self.multiknot_expanded_apex_search()
            elif stage == "apex_descent_interface_support_blocker":
                self.save(current_stage="freeze_v5_evidence",
                          next_decision="freeze_v5_evidence")
            elif stage == "freeze_v5_evidence":
                self.freeze_v5_bridge_evidence()
            elif stage == "analyze_existing_branch_failure_timing":
                self.analyze_bridge_failure_timing()
            elif stage == "audit_pre_post_apex_control_authority":
                self.audit_apex_control_authority()
            elif stage == "build_current_runtime_descent_terminal_clusters":
                self.build_descent_terminal_clusters()
            elif stage == "select_bridge_start_before_apex":
                self.select_apex_bridge_starts()
            elif stage == "discover_nominal_stable_descent_trajectory":
                self.discover_nominal_feedback_bridge()
            elif stage == "synthesize_local_feedback_bridge":
                self.synthesize_feedback_bridge()
            elif stage == "fresh_seed_bridge_validation":
                self.validate_fresh_feedback_bridge()
            elif stage == "formal_descent_support_and_final_recovery_test":
                self.classify_feedback_bridge()
            elif stage in (
                "expand_bridge_admissible_apex_bank",
                "descent_support_interface_coverage_gap",
                "pre_apex_feedback_bridge_stage_local_blocker",
            ):
                if stage == "pre_apex_feedback_bridge_stage_local_blocker":
                    self.prepare_pre_apex_horizon_audit()
                else:
                    self.save(); time.sleep(30)
            elif stage == "minimal_pre_apex_horizon_audit":
                self.run_pre_apex_horizon_audit()
            elif stage == "apex_centroidal_contact_audit":
                self.run_apex_centroidal_contact_audit()
            elif stage in (
                "takeoff_tail_centroidal_momentum_blocker",
                "ballistic_morphology_feedback_blocker",
                "downstream_controller_gap",
            ):
                corrected = (
                    FEEDBACK_ROOT / "horizon_momentum_v2"
                    / "centroidal_contact_audit.json"
                )
                lineage = (
                    FEEDBACK_ROOT / "contact_supported_v1"
                    / "upstream_lineage.json"
                )
                if not corrected.exists():
                    self.prepare_pre_apex_horizon_audit()
                elif (stage == "takeoff_tail_centroidal_momentum_blocker"
                      and not lineage.exists()):
                    self.save(
                        current_stage="audit_apex_upstream_lineage",
                        next_decision="audit_apex_upstream_lineage",
                    )
                else:
                    self.save(); time.sleep(30)
            elif stage == "audit_apex_upstream_lineage":
                self.audit_apex_upstream_lineage()
            elif stage == "contact_supported_bridge_discovery":
                self.discover_contact_supported_bridge()
            elif stage == "contact_supported_bridge_fresh_validation":
                self.save(); time.sleep(30)
            elif stage == "build_apex_reset_bank_v3_r3": self.apex_bank_r3()
            elif stage == "apex_bounded_support_search_r3": self.apex_search_r3()
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
