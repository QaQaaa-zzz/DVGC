"""Persistent controller for decoupled bootstrap experts and consolidation."""
from __future__ import annotations

import json
import time
from pathlib import Path

from cli.descent_local_controller import Controller, PYTHON
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.curriculum import select_flight_reset_records
from dvgc.runtime import save_json


LANDING_SOURCE = Path("runs/landing/refinement_seed0/policy")
FLIGHT_SOURCE = Path("runs/flight/pipeline_seed0_v5/pilot/policy")
CANONICAL_C_L = Path("runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl")
FLIGHT_BANK = Path("artifacts/flight_candidates_augmented_v1.pkl")
RETIRED_STAGE_RUN = Path("runs/stage_reachability_seed0_20260719")
CURRICULA = ("late_descent", "descent", "apex", "ascent")


class DecoupledBootstrapController(Controller):
    def __init__(self, run: Path):
        super().__init__(run)
        if self.state.get("controller_type") != "decoupled_bootstrap_consolidation":
            self.state = {
                "controller_type": "decoupled_bootstrap_consolidation",
                "controller_version": 1,
                "controller_module": "cli.decoupled_bootstrap_controller",
                "controller_unit": "dvgc-decoupled-bootstrap-controller.service",
                "run_id": run.name,
                "current_stage": "freeze_contract",
                "last_completed_action": None,
                "in_progress_action": None,
                "expected_outputs": [],
                "next_decision": "freeze_contract",
                "retry_count": 0,
                "heartbeat": time.time(),
                "stop_reason": None,
                "terminal_state": None,
                "research_gate_valid": False,
                "history": [],
                "provenance": {},
                "sequential_shared_actor_route": "diagnostic_only",
            }
            self.save()

    def freeze_contract(self) -> None:
        root = self.run / "frozen"
        contract = root / "frozen_contract.json"
        registry = root / "expert_registry.json"
        if not contract.exists():
            self.run_command(
                "freeze_decoupled_contract",
                [
                    PYTHON, "-m", "cli.prepare_decoupled_bootstrap",
                    "--landing-policy", LANDING_SOURCE,
                    "--flight-policy", FLIGHT_SOURCE,
                    "--landing-entry-set", CANONICAL_C_L,
                    "--flight-bank", FLIGHT_BANK,
                    "--output-root", root,
                ],
                self.run / "logs/freeze_contract.log",
                [contract, registry, root / "pi_l_frozen/params.pkl", root / "pi_f_init/params.pkl"],
            )
        payload = json.loads(contract.read_text())
        cost = self.run / "cost_estimate.json"
        if not cost.exists():
            save_json(cost, {
                "status": "PASS", "artifact_role": "cost_estimate",
                "route": "decoupled_bootstrap_experts_then_shared_consolidation_v1",
                "estimated_wall_hours": 8.0,
                "bounded_next_action": "one fixed-bank composite preflight",
                "expensive_stages": {
                    "flight_curricula": "4 levels, <=4 x 25,600 effective steps each, early Chain stop",
                    "provisional_certification": "4-branch pilot before adaptive expansion",
                    "takeoff_approach_and_consolidation": "not launched before Flight expert gate",
                },
            })
        supersession = self.run / "supersession.json"
        if not supersession.exists():
            retired = json.loads((RETIRED_STAGE_RUN / "controller_state.json").read_text())
            if retired.get("terminal_state") != "gate_pause" or retired.get("active_worker_unit"):
                raise RuntimeError("Retired stage-reachability route is not at an inactive gate boundary")
            save_json(supersession, {
                "status": "PASS", "supersedes": str(RETIRED_STAGE_RUN),
                "preserved_terminal_state": retired.get("terminal_state"),
                "preserved_stop_reason": retired.get("stop_reason"),
                "research_use": "engineering_diagnostic_only",
                "no_artifact_overwritten": True,
            })
        save_json(Path("runs/ACTIVE_PIPELINE.json"), {
            "status": "ACTIVE", "activated_at": time.time(), "run_path": str(self.run),
            "controller_unit": "dvgc-decoupled-bootstrap-controller.service",
            "start_script": "/home/qy/DVGC/scripts/start_decoupled_bootstrap_controller.sh",
            "supersedes": str(RETIRED_STAGE_RUN),
            "supersession_reason": "DECOUPLED_BOOTSTRAP_EXPERT_CONSOLIDATION_AUTHORIZED",
        })
        self.save(
            current_stage="composite_preflight",
            last_completed_action="freeze_contract",
            next_decision="composite_preflight",
            frozen_contract=str(contract),
            current_registry=str(registry),
            current_policy=payload["flight_initial"]["policy"],
            canonical_c_l=payload["canonical_c_l"]["path"],
            provenance={
                "landing_policy_hash": payload["landing"]["policy_hash"],
                "flight_initial_policy_hash": payload["flight_initial"]["policy_hash"],
                "canonical_c_l_sha256": payload["canonical_c_l"]["sha256"],
                "candidate_bank_sha256": payload["flight_candidate_bank"]["sha256"],
                "runtime_source_fingerprint": payload["runtime"]["source_fingerprint"],
            },
        )

    def composite_preflight(self) -> None:
        output = self.run / "flight/composite_preflight.json"
        if not output.exists():
            result = self.run_worker_command(
                "flight_composite_preflight",
                [PYTHON, "-u", "-m", "cli.evaluate_composite",
                 "--registry", self.state["current_registry"], "--bank", FLIGHT_BANK,
                 "--entry-set", self.state["canonical_c_l"], "--output", output,
                 "--seed", "2810000000"],
                self.run / "logs/flight_composite_preflight.log", [output],
                unit_suffix=f"decoupled-composite-preflight-{int(time.time())}", preallocate=False,
            )
            if not result["ok"]:
                raise RuntimeError(f"Composite preflight failed: {result}")
        report = json.loads(output.read_text())
        if report.get("timeout_rate") != 0 or report.get("runtime_source_fingerprint") != self.state["provenance"]["runtime_source_fingerprint"]:
            self.save(current_stage="gate_pause", terminal_state="gate_pause", research_gate_valid=True,
                      stop_reason="composite_preflight_provenance_or_timeout_failure", next_decision=None)
            raise SystemExit(40)
        self.save(current_stage="flight_late_descent", last_completed_action="composite_preflight",
                  next_decision="flight_late_descent", flight_composite_preflight=str(output))

    @staticmethod
    def _target_support(report: dict, curriculum: str) -> tuple[int, int]:
        rows = report["rows"]
        bank_rows = SnapshotBank.load(FLIGHT_BANK).records_for_phase("flight", include_training_only=False)
        target_ids = {row["id"] for row in select_flight_reset_records(bank_rows, curriculum)}
        target = [row for row in rows if row["candidate_id"] in target_ids]
        return sum(bool(row["chain"]) for row in target), len(target)

    def _advance_curriculum(self, curriculum: str, *, policy: str, registry: str, action: str) -> None:
        index = CURRICULA.index(curriculum)
        next_stage = f"flight_{CURRICULA[index + 1]}" if index + 1 < len(CURRICULA) else "freeze_flight_expert"
        self.save(current_stage=next_stage, last_completed_action=action, next_decision=next_stage,
                  current_policy=policy, current_registry=registry)

    def flight_curriculum(self, curriculum: str) -> None:
        root = self.run / "flight/curriculum" / curriculum
        baseline = root / "pretraining_composite.json"
        if not baseline.exists():
            result = self.run_worker_command(
                f"flight_{curriculum}_pretraining_composite",
                [PYTHON, "-u", "-m", "cli.evaluate_composite",
                 "--registry", self.state["current_registry"], "--bank", FLIGHT_BANK,
                 "--entry-set", self.state["canonical_c_l"], "--output", baseline,
                 "--seed", str(2820000000 + CURRICULA.index(curriculum) * 100000)],
                self.run / f"logs/flight_{curriculum}_preflight.log", [baseline],
                unit_suffix=f"decoupled-{curriculum}-preflight-{int(time.time())}", preallocate=False,
            )
            if not result["ok"]:
                raise RuntimeError(f"{curriculum} pretraining evaluation failed: {result}")
        support, total = self._target_support(json.loads(baseline.read_text()), curriculum)
        if support > 0:
            marker = root / "already_reachable.json"
            if not marker.exists():
                save_json(marker, {"status": "PASS", "decision": "skip_PPO_existing_Chain_support",
                                   "curriculum": curriculum, "chain_successes": support,
                                   "target_states": total, "policy": self.state["current_policy"],
                                   "policy_hash": file_sha256(Path(self.state["current_policy"]) / "params.pkl"),
                                   "canonical_c_l_sha256": file_sha256(self.state["canonical_c_l"])})
            self._advance_curriculum(curriculum, policy=self.state["current_policy"],
                                     registry=self.state["current_registry"], action=f"{curriculum}_already_reachable")
            return
        train = root / "train"
        metrics = train / "training_metrics.json"
        if not metrics.exists():
            command = [
                PYTHON, "-u", "-m", "cli.train_expert", "--stage", "flight",
                "--curriculum", curriculum, "--bank", FLIGHT_BANK,
                "--entry-set", self.state["canonical_c_l"], "--registry", self.state["current_registry"],
                "--resume", self.state["current_policy"], "--run", train,
                "--seed", "0", "--learning-rate", "0.0001", "--gate-mode", "chain_only",
                "--initial-composite-evaluation", baseline,
                "--landing-baseline", "runs/stage_experts/flight_seed0_20260715T2045/frozen_landing_baseline_fixed.json",
            ]
            result = self.run_worker_command(
                f"flight_{curriculum}_expert_train", command,
                self.run / f"logs/flight_{curriculum}_train.log", [metrics],
                unit_suffix=f"decoupled-{curriculum}-train-{int(time.time())}", preallocate=False,
            )
            if not result["ok"]:
                if result.get("returncode") == 2 and metrics.exists():
                    self.save(current_stage="gate_pause", terminal_state="gate_pause", research_gate_valid=True,
                              blocked_stage=f"flight_{curriculum}", stop_reason=f"flight_{curriculum}_expert_chain_blocker",
                              next_decision=None, stage_metrics=str(metrics))
                    raise SystemExit(40)
                raise RuntimeError(f"{curriculum} expert worker failed: {result}")
        payload = json.loads(metrics.read_text())
        if payload.get("status") != "gate_pass":
            self.save(current_stage="gate_pause", terminal_state="gate_pause", research_gate_valid=True,
                      blocked_stage=f"flight_{curriculum}", stop_reason=f"flight_{curriculum}_expert_chain_blocker",
                      next_decision=None, stage_metrics=str(metrics))
            raise SystemExit(40)
        reports = [path for path in (train / "blocks").glob("block_*/report.json") if json.loads(path.read_text()).get("status") == "PASS"]
        if len(reports) != 1:
            raise RuntimeError(f"Expected one passing {curriculum} block, found {len(reports)}")
        block = reports[0].parent
        self._advance_curriculum(curriculum, policy=str((block / "policy").resolve()),
                                 registry=str((block / "expert_registry.json").resolve()),
                                 action=f"flight_{curriculum}_expert_pass")

    def freeze_flight_expert(self) -> None:
        output = self.run / "flight/frozen_flight_expert.json"
        if not output.exists():
            save_json(output, {
                "status": "PASS", "artifact_role": "bootstrap_expert",
                "stage": "flight", "objective": "Flight_to_fixed_canonical_C_L",
                "policy": self.state["current_policy"],
                "policy_hash": file_sha256(Path(self.state["current_policy"]) / "params.pkl"),
                "registry": self.state["current_registry"],
                "canonical_c_l": self.state["canonical_c_l"],
                "canonical_c_l_sha256": file_sha256(self.state["canonical_c_l"]),
                "landing_retention_required": False,
                "curriculum": list(CURRICULA),
                "next_artifact_role": "expert_conditioned_provisional_envelope",
            })
        self.save(current_stage="flight_provisional_certification", last_completed_action="freeze_flight_expert",
                  next_decision="flight_provisional_certification_pilot", frozen_flight_expert=str(output))

    def provisional_prepare(self) -> None:
        marker = self.run / "flight/provisional_certification/cost_estimate.json"
        if not marker.exists():
            save_json(marker, {
                "status": "PASS", "artifact_role": "cost_estimate",
                "pilot": {"states": 160, "branches_per_state": 4, "rollouts": 640},
                "adaptive_formal_followup": "4 -> 8 -> 16/32 only after pilot evidence",
                "output_role": "expert_conditioned_provisional_envelope",
                "formal_jel_eligible": False,
            })
        self.save(current_stage="flight_provisional_certification_ready",
                  last_completed_action="flight_provisional_certification_prepare",
                  next_decision="implement_chunked_composite_branch_pilot",
                  provisional_certification_cost=str(marker))

    def loop(self) -> int:
        while True:
            self.save()
            stage = self.state["current_stage"]
            if stage == "freeze_contract": self.freeze_contract()
            elif stage == "composite_preflight": self.composite_preflight()
            elif stage.startswith("flight_") and stage.removeprefix("flight_") in CURRICULA:
                self.flight_curriculum(stage.removeprefix("flight_"))
            elif stage == "freeze_flight_expert": self.freeze_flight_expert()
            elif stage == "flight_provisional_certification": self.provisional_prepare()
            elif stage == "flight_provisional_certification_ready":
                time.sleep(60)
            elif stage == "gate_pause": raise SystemExit(40)
            elif stage == "pipeline_complete": return 0
            else: raise RuntimeError(f"Unknown decoupled pipeline stage: {stage}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    controller = DecoupledBootstrapController(Path(args.run))
    try:
        raise SystemExit(controller.loop())
    except SystemExit:
        raise
    except Exception as exc:
        count = int(controller.state.get("retry_count", 0)) + 1
        controller.save(retry_count=count, stop_reason=f"{type(exc).__name__}: {exc}")
        if count >= 3:
            controller.save(terminal_state="engineering_failure_after_retries")
            raise SystemExit(41)
        raise


if __name__ == "__main__":
    main()
