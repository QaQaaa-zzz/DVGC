"""Persistent controller for trajectory mining and bounded roll recovery."""
from __future__ import annotations

import json,subprocess,time
from pathlib import Path

from cli.descent_envelope_controller import DescentEnvelopeController
from cli.descent_local_controller import Controller,ENTRY,LANDING,PYTHON,RUNTIME_GATE
from cli.descent_tube_controller import failure_fuse_update
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256,load_config
from dvgc.construction_lifecycle import validate_policy_update_gate
from dvgc.runtime import save_json
from dvgc.seed_registry import save_registry
from dvgc.snapshot_provenance import validate_snapshot_source_records
from cli.analyze_stable_descent_construction import validate_unique_candidates


SOURCE_RUN=Path("runs/stage_experts/descent_envelope_seed0_20260718T004058")
SOURCE_BANK=SOURCE_RUN/"cycle_2/stable/current_policy.pkl"
SOURCE_REPORT=SOURCE_RUN/"cycle_2/stable/report.json"
SOURCE_POLICY=Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy")
SOURCE_CHECKPOINT=Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/orbax/000000076800")
VIABILITY=SOURCE_RUN/"cycle_1/viability/ensemble.pkl"
SOURCE_POLICY_ORIGIN=Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_2/train/policy")
XML=Path("assets/orange_bike_4kg_horizontal.xml")


class TrajectoryMiningController(DescentEnvelopeController):
    def __init__(self,run:Path):
        Controller.__init__(self,run)
        if self.state.get("controller_type")!="trajectory_mining":
            self.state={"controller_type":"trajectory_mining","controller_version":1,
                "controller_unit":"dvgc-trajectory-mining-controller.service","controller_module":"cli.trajectory_mining_controller",
                "run_id":run.name,"current_stage":"inspect","current_cycle":3,"acquisition_round":2,
                "last_completed_action":None,"in_progress_action":None,"expected_outputs":[],"next_decision":"trajectory_mining",
                "retry_count":0,"heartbeat":time.time(),"stop_reason":None,"active_worker_unit":None,"history":[],
                "provenance":{},"failure_signature":None,"consecutive_failure_count":0}
            self.save()

    def inspect(self):
        if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip():raise RuntimeError("Tracked worktree must be clean")
        subprocess.run([PYTHON,"-m","cli.runtime_gate","--config","configs/default.json","--output",RUNTIME_GATE,"--check-only"],check=True,stdout=subprocess.DEVNULL)
        required=[SOURCE_BANK,SOURCE_REPORT,SOURCE_POLICY/"params.pkl",SOURCE_CHECKPOINT,VIABILITY,LANDING/"params.pkl",ENTRY,XML,SOURCE_RUN/"seed_registry.json"]
        missing=[str(path) for path in required if not path.exists()]
        if missing:raise RuntimeError(f"Trajectory-mining input missing: {missing}")
        old=json.loads((SOURCE_RUN/"controller_state.json").read_text());report=json.loads(SOURCE_REPORT.read_text())
        if old.get("current_stage")!="gate_pause" or report.get("stable_safe_states")!=3 or report.get("states")!=146:raise RuntimeError("Source stable-support gate changed")
        policy_hash=file_sha256(SOURCE_POLICY/"params.pkl");bank=SnapshotBank.load(SOURCE_BANK)
        if bank.metadata.get("policy_hash")!=policy_hash:raise RuntimeError("Frozen source policy/bank mismatch")
        registry=json.loads((SOURCE_RUN/"seed_registry.json").read_text());save_registry(self.run/"seed_registry.json",registry["claims"],status="ACTIVE",
            source_registry=str(SOURCE_RUN/"seed_registry.json"),source_registry_sha256=file_sha256(SOURCE_RUN/"seed_registry.json"))
        gate=json.loads(RUNTIME_GATE.read_text());self.save(provenance={"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "current_policy_hash":policy_hash,"policy_hash":policy_hash,"candidate_bank_sha256":file_sha256(SOURCE_BANK),
            "xml_sha256":file_sha256(XML),"c_l_hash":file_sha256(ENTRY),"pi_l_hash":file_sha256(LANDING/"params.pkl"),
            "runtime_source_fingerprint":gate["source_fingerprint"],"source_stable_report_sha256":file_sha256(SOURCE_REPORT)},
            current_candidate=str(SOURCE_BANK),current_policy=str(SOURCE_POLICY),current_checkpoint=str(SOURCE_CHECKPOINT),
            current_cumulative_steps=76800,policy_history=[str(SOURCE_POLICY_ORIGIN),str(SOURCE_POLICY)],route_phase="trajectory_mining",
            source_stable_report=str(SOURCE_REPORT),current_stage="trajectory_mining",last_completed_action="inspect",next_decision="stable_stage_a",stop_reason=None)

    def trajectory_mining(self):
        root=self.run/"trajectory_mining";bank=root/"candidate_pool.pkl";report=root/"report.json";cfg=load_config("configs/default.json")
        total=len(SnapshotBank.load(self.state["current_candidate"]).records_for_phase("flight",include_training_only=False));branches=int(cfg.trajectory_mining_branches_per_state)
        seed,proof,attempts=self._allocate_indices("success_trajectory_mining","development_trajectory_mining",1_700_000_000,list(range(total)),branches)
        save_json(root/"seed_intersection_proof.json",{**proof,"allocation_attempts":attempts})
        if not report.exists():
            result=self.run_worker_command("success_trajectory_mining",[PYTHON,"-u","-m","cli.mine_success_trajectories",
                "--base-bank",self.state["current_candidate"],"--descent-policy",self.state["current_policy"],"--landing-policy",LANDING,
                "--landing-entry-set",ENTRY,"--viability-model",VIABILITY,"--output-bank",bank,"--output-report",report,
                "--seed",seed,"--branches-per-state",branches,"--target",int(cfg.trajectory_mining_target_snapshots),
                "--namespace",self.run.name],root/"worker.log",[bank,report],unit_suffix=f"trajectory-mining-{int(time.time())}",preallocate=False)
            if not result["ok"]:raise RuntimeError(f"Trajectory mining worker failed: {result}")
        payload=json.loads(report.read_text())
        if payload["status"]!="PASS" or payload["maximum_children_per_trajectory_parent"]>4 or payload["prediction_can_promote_empirical_safe"]:
            self.save(current_stage="gate_pause",stop_reason="Trajectory-mining candidate quality gate failed");return
        self.state["provenance"]["candidate_bank_sha256"]=file_sha256(bank)
        self.save(current_candidate=str(bank),current_cycle=3,current_stage="stable_stage_a",next_decision="stable_stage_b")

    def stable_decision(self):
        report=self._validate_stable_report(Path(self.state["current_stable_report"]))
        if self.state.get("route_phase")=="trajectory_mining":
            if report["activation_support_pass"] and report["parent_diversity_pass"]:
                self.save(current_stage="stable_freeze",next_decision="fresh_pointwise_audit");return
            self.save(current_stage="roll_controllability",next_decision="roll_targeted_or_gate");return
        if self.state.get("route_phase")=="roll_targeted":
            before_path=Path(self.state["pre_roll_stable_report"]);after_path=Path(self.state["current_stable_report"])
            baseline,report,lifecycle=validate_policy_update_gate(before_path,after_path,
                before_policy_hash=self.state["pre_roll_policy_hash"],after_policy_hash=file_sha256(Path(self.state["current_policy"])/"params.pkl"),
                before_cycle=int(self.state["pre_roll_cycle"]),after_cycle=int(self.state["current_cycle"]))
            before={row["id"] for row in baseline["rows"] if row["stable_safe"]};after={row["id"] for row in report["rows"] if row["stable_safe"]}
            bterm=baseline["terminal_summary"];aterm=report["terminal_summary"]
            gate={"stable_safe_retained":before<=after,"roll_failure_decreased":aterm["physical_end_reasons"]["roll"] / aterm["branches"] < bterm["physical_end_reasons"]["roll"] / bterm["branches"],
                "final_not_worse":aterm["final_recovery_rate"]>=bterm["final_recovery_rate"],
                "stable_safe_not_decreased":int(report["stable_safe_states"])>=int(baseline["stable_safe_states"]),
                "stable_safe_at_least_four":report["stable_safe_states"]>=4,
                "parent_diversity":bool(report["parent_diversity_pass"])}
            passed=all(gate[key] for key in ("stable_safe_retained","roll_failure_decreased","final_not_worse","stable_safe_at_least_four","parent_diversity"))
            clear_improvement=(gate["roll_failure_decreased"] and gate["final_not_worse"] and gate["stable_safe_not_decreased"]
                and int(report["stable_safe_states"]) in (2,3))
            mined_ids={row["id"] for row in SnapshotBank.load(self.state["current_candidate"]).records_for_phase("flight",include_training_only=False)
                if row.get("candidate_kind")=="successful_trajectory_snapshot"}
            def compact(row):return {key:row.get(key) for key in ("id","parent","layer","label","stable_safe","stage_a","stage_b","adaptive","combined")}
            comparison={"status":"PASS" if passed else "BOUNDED_ACQUISITION" if clear_improvement else "FAIL",
                "artifact_role":"fresh_roll_targeted_cycle_comparison","lifecycle_checks":lifecycle,"checks":gate,
                "before":{"cycle":self.state["pre_roll_cycle"],"policy_hash":baseline["policy_hash"],"report_sha256":file_sha256(before_path),
                    "stable_safe":baseline["stable_safe_states"],"parents":baseline["stable_safe_parents"],"layers":baseline["stable_safe_layers"],"terminal":bterm},
                "after":{"cycle":self.state["current_cycle"],"policy_hash":report["policy_hash"],"report_sha256":file_sha256(after_path),
                    "stable_safe":report["stable_safe_states"],"parents":report["stable_safe_parents"],"layers":report["stable_safe_layers"],"terminal":aterm},
                "prior_safe_state_results":[compact(row) for row in report["rows"] if row["id"] in before],
                "trajectory_mined_state_results":[compact(row) for row in report["rows"] if row["id"] in mined_ids]}
            comparison_path=self._cycle_root()/"roll_targeted_block_gate.json";save_json(comparison_path,comparison)
            if passed:self.save(current_stage="stable_freeze",next_decision="fresh_pointwise_audit",research_gate_valid=False)
            elif clear_improvement and not self.state.get("roll_policy_acquisition_used"):
                self.save(roll_cycle5_comparison=str(comparison_path),current_stage="roll_acquisition_prepare",
                    next_decision="single_policy_conditioned_acquisition",research_gate_valid=False)
            else:
                blocker=self._cycle_root()/"research_blocker.json";save_json(blocker,{"status":"VALID_RESEARCH_GATE",
                    "reason":"Single roll-targeted PPO did not establish sufficient stable-safe descent support","comparison":comparison})
                self.save(current_stage="gate_pause",stop_reason="Valid research gate: roll-targeted PPO failed fresh stable retention/expansion gate",
                    research_gate_valid=True,last_successful_artifact=str(comparison_path),next_decision="research_direction")
            return
        if self.state.get("route_phase")=="roll_targeted_acquisition":
            if report["activation_support_pass"] and report["parent_diversity_pass"]:
                self.save(current_stage="stable_freeze",next_decision="fresh_pointwise_audit",research_gate_valid=False)
            else:
                blocker=self._cycle_root()/"research_blocker.json";save_json(blocker,{"status":"VALID_RESEARCH_GATE",
                    "reason":"Single policy-conditioned acquisition exhausted below stable support gate",
                    "stable_report":str(self.state["current_stable_report"]),"stable_report_sha256":file_sha256(self.state["current_stable_report"])})
                self.save(current_stage="gate_pause",stop_reason="Valid research gate: bounded post-PPO acquisition exhausted",
                    research_gate_valid=True,last_successful_artifact=str(self.state["current_stable_report"]),next_decision="research_direction")
            return
        raise RuntimeError("Unexpected stable decision route")

    def stage_a(self):
        candidate=Path(self.state["current_candidate"])
        audit_path=Path(self.state.get("roll_acquisition_report") if self.state.get("route_phase")=="roll_targeted_acquisition"
            else self.state.get("candidate_physical_audit",""))
        preflight=self._cycle_root()/"candidate_preflight.json"
        try:
            tracked=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip()
            subprocess.run([PYTHON,"-m","cli.runtime_gate","--config","configs/default.json","--output",RUNTIME_GATE,"--check-only"],
                check=True,stdout=subprocess.DEVNULL)
            gate=json.loads(RUNTIME_GATE.read_text());policy_hash=file_sha256(Path(self.state["current_policy"])/"params.pkl")
            candidate_bank=SnapshotBank.load(candidate);candidate_records=candidate_bank.records_for_phase("flight",include_training_only=False)
            uniqueness=validate_unique_candidates(candidate_records)
            source_hashes=validate_snapshot_source_records(candidate_records,candidate_bank.metadata)
            audit=json.loads(audit_path.read_text()) if audit_path.is_file() else {}
            checks={"tracked_worktree_clean":not tracked,"runtime_gate_pass_current":gate.get("status")=="PASS",
                "three_layer_uniqueness":uniqueness["status"]=="PASS","physical_audit_pass":audit.get("status")=="PASS",
                "candidate_hash_matches":audit.get("candidate_bank_sha256",audit.get("bank_sha256"))==file_sha256(candidate),
                "source_policy_records_complete":bool(source_hashes),
                "policy_hash_matches":self.state["provenance"].get("current_policy_hash")==policy_hash,
                "xml_hash_matches":self.state["provenance"].get("xml_sha256")==file_sha256(XML),
                "c_l_hash_matches":self.state["provenance"].get("c_l_hash")==file_sha256(ENTRY),
                "pi_l_hash_matches":self.state["provenance"].get("pi_l_hash")==file_sha256(LANDING/"params.pkl")}
            save_json(preflight,{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
                "uniqueness":uniqueness,"candidate_bank_sha256":file_sha256(candidate),"physical_audit":str(audit_path)})
            if not all(checks.values()):
                self.save(current_stage="authorized_stop",stop_reason="Corrected candidate preflight failed",
                    recommended_resume_action="repair_corrected_candidate_preflight");return
            self.state["provenance"].update({"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                "runtime_source_fingerprint":gate["source_fingerprint"],"current_policy_hash":policy_hash,
                "candidate_bank_sha256":file_sha256(candidate)})
            self.save()
        except (SystemExit,Exception) as exc:
            save_json(preflight,{"status":"FAIL","error":str(exc),"candidate_bank":str(candidate)})
            self.save(current_stage="authorized_stop",stop_reason=f"Corrected candidate uniqueness preflight failed: {exc}",
                recommended_resume_action="repair_corrected_candidate_preflight");return
        super().stage_a()

    def roll_controllability(self):
        root=self.run/"roll_controllability";report=root/"report.json";bank=Path(self.state["current_stable_bank"])
        from cli.audit_roll_controllability import selected_records
        records=SnapshotBank.load(bank).records_for_phase("flight",include_training_only=False);total=len(selected_records(records))
        seed,proof,attempts=self._allocate_indices("roll_controllability","development_roll_controllability",1_800_000_000,list(range(total)),1)
        save_json(root/"seed_intersection_proof.json",{**proof,"allocation_attempts":attempts})
        if not report.exists():
            result=self.run_worker_command("roll_controllability_audit",[PYTHON,"-u","-m","cli.audit_roll_controllability",
                "--candidate-bank",bank,"--policy",self.state["current_policy"],"--output",report,"--seed",seed],root/"worker.log",[report],unit_suffix=f"roll-control-{int(time.time())}",preallocate=False)
            if not result["ok"]:raise RuntimeError(f"Roll controllability worker failed: {result}")
        payload=json.loads(report.read_text())
        if payload["roll_controllable"]:self.save(current_stage="roll_targeted_prepare",next_decision="single_roll_targeted_ppo")
        else:self.save(current_stage="gate_pause",stop_reason="Roll authority insufficient on current candidate support")

    def roll_targeted_prepare(self):
        root=self.run/"roll_targeted";bank=root/"reset_bank.pkl";report=root/"reset_bank.report.json"
        if not report.exists():self.run_command("build_roll_targeted_reset_bank",[PYTHON,"-m","cli.build_roll_targeted_reset_bank",
            "--stable-bank",self.state["current_stable_bank"],"--output-bank",bank,"--output-report",report],root/"reset_bank.log",[bank,report])
        self.save(pre_roll_cycle=int(self.state["current_cycle"]),pre_roll_policy_hash=file_sha256(Path(self.state["current_policy"])/"params.pkl"),
            pre_roll_stable_report=self.state["current_stable_report"],pending_roll_reset_bank=str(bank),current_stage="roll_targeted_train",next_decision="stable_stage_a")

    def roll_targeted_train(self):
        root=self.run/"roll_targeted/train";report=root/"report.json";cumulative=int(self.state["current_cumulative_steps"])+25600
        if not report.exists():
            result=self.run_worker_command("single_roll_targeted_ppo",[PYTHON,"-u","-m","cli.train_descent_local_block",
                "--resume-policy",self.state["current_policy"],"--bootstrap-bank",self.state["pending_roll_reset_bank"],
                "--candidate-bank",self.state["current_stable_bank"],"--entry-set",ENTRY,"--run",root,
                "--cumulative-steps",cumulative,"--restore-checkpoint",self.state["current_checkpoint"],"--seed",0],
                self.run/"roll_targeted/train.log",[report,root/"policy/params.pkl"],unit_suffix=f"roll-targeted-ppo-{int(time.time())}")
            if not result["ok"]:raise RuntimeError(f"Roll-targeted PPO failed: {result}")
        payload=json.loads(report.read_text())
        if payload.get("health",{}).get("nonfinite_metric_keys") or payload.get("health",{}).get("oom") or payload.get("health",{}).get("timeout"):
            self.save(current_stage="gate_pause",stop_reason="Roll-targeted PPO health gate failed");return
        policy=root/"policy";checkpoint=root/"orbax"/f"{cumulative:012d}";history=list(self.state.get("policy_history",[]));history.append(str(policy))
        self.state["provenance"]["current_policy_hash"]=file_sha256(policy/"params.pkl")
        next_cycle=int(self.state["pre_roll_cycle"])+1
        self.save(route_phase="roll_targeted",current_policy=str(policy),current_checkpoint=str(checkpoint),current_cumulative_steps=cumulative,
            policy_history=history,current_cycle=next_cycle,current_stage="stable_stage_a",next_decision="stable_stage_b")

    def roll_acquisition_prepare(self):
        root=self.run/"roll_targeted/policy_conditioned_acquisition";model=root/"viability.pkl";report=root/"viability.report.json"
        seed,proof,attempts=self._allocate_indices("roll_policy_conditioned_viability","viability_training",1_900_000_000,[0],1)
        save_json(root/"viability.seed_proof.json",{**proof,"allocation_attempts":attempts})
        if not report.exists():self.run_command("fit_roll_policy_conditioned_viability",[PYTHON,"-m","cli.fit_viability",
            "--bank",self.state["current_stable_bank"],"--output",model,"--report",report,"--seed",seed],root/"viability.log",[model,report])
        self.save(current_viability_model=str(model),current_stage="roll_acquisition",next_decision="fresh_extended_stable_construction")

    def roll_acquisition(self):
        root=self.run/"roll_targeted/policy_conditioned_acquisition";bank=root/"candidate_pool.pkl";report=root/"candidate_pool.report.json"
        indices=list(range(3001));seed,proof,attempts=self._allocate_indices("roll_policy_conditioned_candidates","candidate_generation",2_000_000_000,indices,1)
        save_json(root/"candidate.seed_proof.json",{**proof,"allocation_attempts":attempts})
        if not report.exists():
            result=self.run_worker_command("single_roll_policy_conditioned_acquisition",[PYTHON,"-u","-m","cli.build_descent_support_repair",
                "--base-bank",self.state["current_stable_bank"],"--policy",self.state["current_policy"],"--landing-entry-set",ENTRY,
                "--output-bank",bank,"--output-report",report,"--seed",seed,"--target",32,"--proposal-budget",3000,
                "--parent-cap",4,"--viability-model",self.state["current_viability_model"],"--acquisition-round",1],
                root/"candidate.log",[bank,report],unit_suffix=f"roll-policy-acquisition-{int(time.time())}",preallocate=False)
            if not result["ok"]:raise RuntimeError(f"Policy-conditioned acquisition worker failed: {result}")
        payload=json.loads(report.read_text())
        if payload.get("status")!="PASS" or payload.get("maximum_children_per_parent",99)>4 or not payload.get("all_state_unique"):
            self.save(current_stage="gate_pause",stop_reason="Valid research gate: policy-conditioned candidate acquisition quality exhausted",
                research_gate_valid=True,next_decision="research_direction");return
        self.save(roll_policy_acquisition_used=True,roll_acquisition_report=str(report),route_phase="roll_targeted_acquisition",current_candidate=str(bank),
            current_source_policy=str(self.state["current_policy"]),current_cycle=int(self.state["current_cycle"])+1,
            current_stage="stable_stage_a",next_decision="stable_stage_b")

    def pointwise_decision(self):
        super().pointwise_decision()
        if self.state.get("current_stage")=="gate_pause":self.save(research_gate_valid=True,next_decision="research_direction")
        if self.state.get("current_stage")=="tube_rsi_prepare":
            self.save(tube_rsi_base_cumulative=int(self.state.get("current_cumulative_steps",76800)))

    def loop(self):
        self.log(f"trajectory-mining controller run={self.run}")
        while True:
            stage=self.state["current_stage"];self.save()
            if stage=="inspect":self.inspect()
            elif stage=="trajectory_mining":self.trajectory_mining()
            elif stage=="roll_controllability":self.roll_controllability()
            elif stage=="roll_targeted_prepare":self.roll_targeted_prepare()
            elif stage=="roll_targeted_train":self.roll_targeted_train()
            elif stage=="roll_acquisition_prepare":self.roll_acquisition_prepare()
            elif stage=="roll_acquisition":self.roll_acquisition()
            elif stage=="stable_stage_a":self.stage_a()
            elif stage=="stable_stage_b":self.stage_b()
            elif stage=="stable_adaptive":self.adaptive()
            elif stage=="stable_analyze":self.analyze_stable()
            elif stage=="stable_decision":self.stable_decision()
            elif stage=="stable_freeze":self.freeze_stable()
            elif stage=="pointwise_audit":self.pointwise_audit()
            elif stage=="pointwise_decision":self.pointwise_decision()
            elif stage=="tube_rsi_prepare":self.tube_rsi_prepare()
            elif stage=="tube_rsi_train":self.tube_rsi_train()
            elif stage=="continuous_cd":self.save(current_stage="gate_pause",stop_reason="Continuous C_D requires independently audited early support")
            elif stage=="gate_pause":return 40
            elif stage=="authorized_stop":return 41
            elif stage=="pipeline_complete":return 0
            else:raise RuntimeError(f"Unknown trajectory-mining stage {stage}")
            self.save(failure_signature=None,consecutive_failure_count=0)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True);a=p.parse_args()
    controller=TrajectoryMiningController(Path(a.run))
    try:raise SystemExit(controller.loop())
    except Exception as exc:
        stage=str(controller.state.get("current_stage","unknown"))
        retry_key=f"{stage}:{controller.state.get('provenance',{}).get('current_policy_hash')}:{controller.state.get('provenance',{}).get('candidate_bank_sha256')}"
        signature,count=failure_fuse_update(controller.state,retry_key,exc)
        controller.log(f"ERROR {type(exc).__name__}: {exc} (identical failure {count}/3)")
        controller.save(stop_reason=f"{type(exc).__name__}: {exc}",failure_signature=signature,consecutive_failure_count=count)
        if count>=3:raise SystemExit(41) from exc
        raise


if __name__=="__main__":main()
