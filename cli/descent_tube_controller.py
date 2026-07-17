"""Persistent controller for exact descent Tube audit and expansion."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from cli.descent_local_controller import (
    Controller,
    ENTRY,
    INITIAL,
    LANDING,
    POOL,
    PYTHON,
    RESET_BANK,
    RUNTIME_GATE,
    SHARD_SIZE,
    completed_coverage,
    split_range_after_oom,
)
from dvgc.bank import SnapshotBank
from dvgc.certification import assert_disjoint_branch_seeds, branch_seed
from dvgc.config import file_sha256
from dvgc.runtime import save_json
from dvgc.seed_registry import (
    allocate_disjoint_grid,
    make_claim,
    save_registry,
)


SOURCE_RUN = Path("runs/stage_experts/descent_local_nonfinite_repair_seed0_20260716T1825")
BLOCK1_ROOT = SOURCE_RUN / "blocks/block_1_25600"
BLOCK1_POLICY = BLOCK1_ROOT / "train/policy"
BLOCK1_CERT = BLOCK1_ROOT / "current_policy_certified_sharded.pkl"
BLOCK1_CERT_REPORT = BLOCK1_ROOT / "current_policy_certified_sharded.cert.json"
FROZEN = Path("runs/stage_experts/descent_tube_seed0_20260716T2330/frozen_block1_v2")
POINTWISE_SEEDS = {1: 9_310_000, 2: 200_000_000}
AUDIT_BRANCHES = 32
LEGACY_COLLIDING_ROUND2_SEED = 9_330_000


def pointwise_seed(round_id: int) -> int:
    """Return an explicitly reviewed audit namespace for each bounded round."""
    try:
        return POINTWISE_SEEDS[int(round_id)]
    except KeyError as exc:
        raise ValueError(f"No independent pointwise seed declared for round {round_id}") from exc


def planned_branch_seeds(base_seed: int, state_count: int, branches: int = AUDIT_BRANCHES):
    """Enumerate the exact global-index seed map before launching an audit."""
    return [
        branch_seed(base_seed, state_index, branch_index)
        for state_index in range(int(state_count))
        for branch_index in range(int(branches))
    ]


def failure_fuse_update(state, stage: str, exc: Exception):
    """Count identical deterministic controller failures across service restarts."""
    message = f"{stage}|{type(exc).__name__}|{exc}"
    signature = hashlib.sha256(message.encode("utf-8")).hexdigest()
    previous = state.get("failure_signature")
    count = int(state.get("consecutive_failure_count", 0)) + 1 if previous == signature else 1
    return signature, count


class DescentTubeController(Controller):
    def __init__(self, run: Path):
        super().__init__(run)
        if self.state.get("controller_type") != "descent_tube":
            self.state = {
                "controller_type": "descent_tube", "controller_version": 2,
                "run_id": run.name, "current_stage": "inspect", "active_round": 1,
                "last_completed_action": None, "in_progress_action": None,
                "expected_outputs": [], "next_decision": "pointwise_audit",
                "retry_count": 0, "heartbeat": time.time(), "stop_reason": None,
                "active_worker_unit": None, "history": [], "provenance": {},
                "failure_signature": None, "consecutive_failure_count": 0,
            }
            self.save()
        elif int(self.state.get("controller_version", 1)) < 2:
            self.save(controller_version=2)

    def inspect_tube(self):
        if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip():
            raise RuntimeError("Tracked worktree must be clean")
        subprocess.run([PYTHON, "-m", "cli.runtime_gate", "--config", "configs/default.json",
                        "--output", str(RUNTIME_GATE), "--check-only"], check=True, stdout=subprocess.DEVNULL)
        required = [BLOCK1_POLICY/"params.pkl", BLOCK1_CERT, BLOCK1_CERT_REPORT,
                    FROZEN/"D_all_unique.pkl", FROZEN/"D_emp_safe.pkl",
                    FROZEN/"discrete_tube_manifest.json", FROZEN/"failed_global_matcher_calibration.json"]
        if any(not path.exists() for path in required): raise RuntimeError("Frozen exact Tube input missing")
        manifest = json.loads((FROZEN/"discrete_tube_manifest.json").read_text())
        if manifest["continuous_matcher_active"] or manifest["network_predictions_are_members"]:
            raise RuntimeError("Exact Tube semantic manifest is unsafe")
        provenance = {"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                      "policy_hash":file_sha256(BLOCK1_POLICY/"params.pkl"),
                      "current_policy_hash":file_sha256(BLOCK1_POLICY/"params.pkl"),
                      "construction_certification_hash":file_sha256(BLOCK1_CERT_REPORT),
                      "all_unique_hash":file_sha256(FROZEN/"D_all_unique.pkl"),
                      "exact_safe_hash":file_sha256(FROZEN/"D_emp_safe.pkl"),
                      "candidate_hash":file_sha256(POOL), "c_l_hash":file_sha256(ENTRY),
                      "pi_l_hash":file_sha256(LANDING/"params.pkl")}
        self.save(provenance=provenance,current_stage="pointwise_audit",last_completed_action="inspect",
                  in_progress_action=None,expected_outputs=[],next_decision="pointwise_decision")

    def _audit_paths(self, round_id):
        overrides=self.state.get("pointwise_seed_overrides",{})
        seed=int(overrides.get(str(round_id),pointwise_seed(round_id)))
        root=self.run/f"round_{round_id}/pointwise_audit_seed{seed}"
        candidate=FROZEN/"D_all_unique.pkl" if round_id==1 else self.run/f"round_{round_id}/frozen/D_all_unique.pkl"
        policy=BLOCK1_POLICY if round_id==1 else self.run/f"round_{round_id}/train/policy"
        construction=BLOCK1_CERT_REPORT if round_id==1 else self.run/f"round_{round_id}/construction/current_policy.cert.json"
        manifest=FROZEN/"discrete_tube_manifest.json" if round_id==1 else self.run/f"round_{round_id}/frozen/discrete_tube_manifest.json"
        return root,candidate,policy,construction,manifest

    def _registered_audit_seed(self, round_id, total):
        """Allocate and persist a grid disjoint from every registered seed set."""
        registry_path=self.run/"seed_registry.json"
        if not registry_path.exists():
            subprocess.run([PYTHON,"-m","cli.build_descent_seed_registry","--run",self.run],check=True)
        registry=json.loads(registry_path.read_text())
        claim_name=("round1_pointwise_audit" if int(round_id)==1
                    else f"round{round_id}_pointwise_seed{pointwise_seed(round_id)}")
        existing=[claim for claim in registry["claims"] if claim["name"]!=claim_name]
        preferred=int(self.state.get("pointwise_seed_overrides",{}).get(str(round_id),pointwise_seed(round_id)))
        allocated,proof,attempts=allocate_disjoint_grid(preferred,total,AUDIT_BRANCHES,existing)
        overrides=dict(self.state.get("pointwise_seed_overrides",{}));overrides[str(round_id)]=allocated
        claim=make_claim(
            claim_name,"pointwise_audit",planned_branch_seeds(allocated,total),status="active",
            base_seed=allocated,state_count=total,branches_per_state=AUDIT_BRANCHES,
            branch_variation_indices=list(range(AUDIT_BRANCHES)),
        )
        claims=[row for row in registry["claims"] if row["name"]!=claim_name]+[claim]
        metadata={key:value for key,value in registry.items() if key not in {"schema_version","claims","historical_intersections"}}
        save_registry(registry_path,claims,**metadata)
        self.save(pointwise_seed_overrides=overrides,seed_registry=str(registry_path))
        root=self.run/f"round_{round_id}/pointwise_audit_seed{allocated}";root.mkdir(parents=True,exist_ok=True)
        proof_path=root/"seed_intersection_proof.json"
        save_json(proof_path,{**proof,"allocation_attempts":attempts,"registry":str(registry_path),
                              "registry_sha256":file_sha256(registry_path)})
        self.save(seed_intersection_proof=str(proof_path))
        return allocated

    def _mark_legacy_round2_audit_invalid(self):
        """Retain, but permanently exclude, the collided round-2 audit."""
        legacy = self.run/f"round_2/pointwise_audit_seed{LEGACY_COLLIDING_ROUND2_SEED}"
        merged = legacy/"merged.json"
        construction = self.run/"round_2/construction/current_policy.cert.json"
        marker = legacy/"INVALID_DIAGNOSTIC.json"
        if marker.exists() or not (merged.exists() and construction.exists()):
            return
        construction_payload=json.loads(construction.read_text())
        audit_payload=json.loads(merged.read_text())
        construction_seeds={
            int(ev["branch_seed"])
            for row in construction_payload["rows"] for ev in row["branch_evidence"]
        }
        audit_seeds={
            int(ev["branch_seed"])
            for row in audit_payload["rows"] for ev in row["branch_evidence"]
        }
        overlap=sorted(construction_seeds & audit_seeds)
        if not overlap:
            raise RuntimeError("Legacy round-2 audit expected a documented seed collision")
        save_json(marker,{
            "status":"INVALID_DIAGNOSTIC",
            "reason":"audit branch seeds overlap round-2 Tube construction branch seeds",
            "excluded_from_analysis_and_gates":True,
            "construction_report":str(construction),
            "construction_report_sha256":file_sha256(construction),
            "audit_report":str(merged),
            "audit_report_sha256":file_sha256(merged),
            "construction_seed":int(construction_payload["seed"]),
            "audit_seed":int(audit_payload["seed"]),
            "overlap_count":len(overlap),
            "overlap_preview":overlap[:10],
            "replacement_audit_seed":pointwise_seed(2),
        })

    def _run_sharded_audit(self, round_id):
        if round_id==2:
            self._mark_legacy_round2_audit_invalid()
        root,candidate,policy,construction,manifest=self._audit_paths(round_id)
        total=len(SnapshotBank.load(candidate).records_for_phase("flight",include_training_only=False))
        seed=self._registered_audit_seed(round_id,total)
        root,candidate,policy,construction,manifest=self._audit_paths(round_id);root.mkdir(parents=True,exist_ok=True)
        construction_payload=json.loads(construction.read_text())
        construction_evidence=[ev for row in construction_payload["rows"] for ev in row["branch_evidence"]]
        assert_disjoint_branch_seeds(construction_evidence,planned_branch_seeds(seed,total))
        pending=[(s,min(s+SHARD_SIZE,total)) for s in range(0,total,SHARD_SIZE)];single={}
        while pending:
            start,end=pending.pop(0);out=root/f"shard_{start:03d}_{end:03d}.completed.json"
            if out.exists():
                p=json.loads(out.read_text())
                if p.get("status")=="PASS" and p.get("start_index")==start and p.get("end_index")==end:continue
                invalid=root/"invalid_diagnostic";invalid.mkdir(exist_ok=True);out.rename(invalid/f"{out.name}.{int(time.time())}")
            cmd=[PYTHON,"-u","-m","cli.certify_descent_entries","--audit-only","--descent-policy",policy,
                 "--candidate-source-policy",INITIAL,"--landing-policy",LANDING,"--candidate-bank",candidate,
                 "--landing-entry-set",ENTRY,"--output",out,"--seed",seed,"--namespace",f"descent_pointwise_round_{round_id}",
                 "--start-index",start,"--end-index",end]
            result=self.run_worker_command(f"pointwise_audit_r{round_id}_{start}_{end}",cmd,root/f"shard_{start:03d}_{end:03d}.log",[out],
                                           unit_suffix=f"tube-r{round_id}-audit-{start}-{end}-{int(time.time())}",preallocate=(end-start==SHARD_SIZE))
            if result["ok"]:continue
            if not result["oom"]:raise RuntimeError(f"Pointwise audit worker failed: {result}")
            if end-start==1:
                single[start]=single.get(start,0)+1
                if single[start]>=2:
                    self.save(current_stage="gate_pause",stop_reason=f"Pointwise state {start} OOM twice");return
                pending.insert(0,(start,end))
            else:pending=split_range_after_oom(start,end)+pending
        shards=sorted(root.glob("shard_*.completed.json"));payloads=[json.loads(p.read_text()) for p in shards];completed_coverage(payloads,total)
        merged=root/"merged.json"
        if not merged.exists():
            cmd=[PYTHON,"-m","cli.merge_descent_entry_audits"]
            for shard in shards:cmd.extend(["--shard",shard])
            cmd.extend(["--output",merged]);self.run_command(f"merge_pointwise_round_{round_id}",cmd,root/"merge.log",[merged])
        analysis=root/"analysis.json"
        if not analysis.exists():
            self.run_command(f"analyze_pointwise_round_{round_id}",[PYTHON,"-m","cli.analyze_pointwise_descent_audit",
                             "--candidate-bank",candidate,"--manifest",manifest,"--construction-report",construction,
                             "--audit-report",merged,"--output",analysis],root/"analysis.log",[analysis])
        self.save(current_stage="pointwise_decision",next_decision="viability_train_or_exact_optimizer_block2")

    def pointwise_decision(self,round_id):
        analysis=json.loads((self._audit_paths(round_id)[0]/"analysis.json").read_text())
        if analysis["status"]=="PASS":
            marker=self.run/f"round_{round_id}/discrete_tube_active.json"
            save_json(marker,{"status":"PASS","role":"discrete_empirical_C_D","continuous_matcher_active":False,
                              "pointwise_audit_sha256":file_sha256(self._audit_paths(round_id)[0]/"analysis.json"),
                              "policy_hash":self.state["provenance"]["policy_hash"]})
            self.save(current_stage="viability_train",next_decision="acquisition")
        elif round_id==1:
            self.save(current_stage="exact_optimizer_block2",next_decision="block2_construction")
        else:
            self.save(current_stage="gate_pause",stop_reason="Round-2 exact pointwise Tube precision remains below 0.95")

    def train_block2(self):
        root=self.run/"round_2";report=root/"train/report.json";policy=root/"train/policy"
        if not report.exists():
            cmd=[PYTHON,"-u","-m","cli.train_descent_local_block","--resume-policy",BLOCK1_POLICY,
                 "--bootstrap-bank",RESET_BANK,"--candidate-bank",POOL,"--entry-set",ENTRY,"--run",root/"train",
                 "--cumulative-steps",51200,"--restore-checkpoint",BLOCK1_ROOT/"train/orbax/000000025600","--seed",0]
            result=self.run_worker_command("exact_optimizer_block2",cmd,root/"train.log",[report,policy/"params.pkl"],
                                           unit_suffix=f"tube-block2-{int(time.time())}")
            if not result["ok"]:raise RuntimeError(f"Block-2 worker failed: {result}")
        self.state["provenance"]["policy_hash"]=file_sha256(policy/"params.pkl")
        self.state["provenance"]["current_policy_hash"]=file_sha256(policy/"params.pkl")
        self.save(active_round=2,current_stage="block2_construction",next_decision="round2_pointwise_audit")

    def block2_construction(self):
        root=self.run/"round_2/construction";root.mkdir(parents=True,exist_ok=True);policy=self.run/"round_2/train/policy";seed=9630000
        total=len(SnapshotBank.load(POOL).records_for_phase("flight",include_training_only=False));pending=[(s,min(s+SHARD_SIZE,total)) for s in range(0,total,SHARD_SIZE)]
        while pending:
            start,end=pending.pop(0);out=root/f"shard_{start:03d}_{end:03d}.completed.json"
            if out.exists():continue
            cmd=[PYTHON,"-u","-m","cli.certify_descent_construction_shard","--descent-policy",policy,
                 "--candidate-source-policy",INITIAL,"--landing-policy",LANDING,"--candidate-bank",POOL,
                 "--landing-entry-set",ENTRY,"--output",out,"--seed",seed,"--namespace","descent_tube_block2",
                 "--start-index",start,"--end-index",end,"--confirm-safe-to-max"]
            result=self.run_worker_command(f"block2_construction_{start}_{end}",cmd,root/f"shard_{start:03d}_{end:03d}.log",[out],
                                           unit_suffix=f"tube-b2-cert-{start}-{end}-{int(time.time())}")
            if not result["ok"]:raise RuntimeError(f"Block-2 construction worker failed: {result}")
        shards=sorted(root.glob("shard_*.completed.json"));completed_coverage([json.loads(p.read_text()) for p in shards],total)
        bank=root/"current_policy.pkl";report=root/"current_policy.cert.json"
        if not report.exists():
            cmd=[PYTHON,"-m","cli.merge_descent_construction_shards"]
            for shard in shards:cmd.extend(["--shard",shard])
            cmd.extend(["--candidate-bank",POOL,"--output-bank",bank,"--output-report",report]);self.run_command("merge_block2_construction",cmd,root/"merge.log",[bank,report])
        frozen=self.run/"round_2/frozen"
        if not frozen.exists():self.run_command("freeze_block2_exact_sets",[PYTHON,"-m","cli.freeze_discrete_descent_tube","--certified-bank",bank,"--cert-report",report,"--policy",self.run/"round_2/train/policy","--output-dir",frozen],root/"freeze.log",[frozen/"discrete_tube_manifest.json"])
        self.save(current_stage="pointwise_audit",active_round=2,next_decision="pointwise_decision")

    def viability_train(self,round_id):
        root=self.run/f"round_{round_id}/viability";model=root/"ensemble.pkl";report=root/"report.json"
        candidate=self._audit_paths(round_id)[1]
        if not report.exists():self.run_command("viability_train",[PYTHON,"-m","cli.fit_viability","--bank",candidate,"--output",model,"--report",report,"--seed",9710000+round_id],root/"train.log",[model,report])
        self.save(current_stage="acquisition",next_decision="candidate_certification",stop_reason="Acquisition implementation gate")

    def loop(self):
        self.log(f"descent-tube controller pid={__import__('os').getpid()} run={self.run}")
        while True:
            stage=self.state["current_stage"];round_id=int(self.state.get("active_round",1));self.save()
            if stage=="inspect":self.inspect_tube()
            elif stage=="pointwise_audit":self._run_sharded_audit(round_id)
            elif stage=="pointwise_decision":self.pointwise_decision(round_id)
            elif stage=="exact_optimizer_block2":self.train_block2()
            elif stage=="block2_construction":self.block2_construction()
            elif stage=="viability_train":self.viability_train(round_id)
            elif stage=="acquisition":self.save(current_stage="gate_pause",stop_reason="Acquisition implementation not yet validated")
            elif stage=="gate_pause":return 40
            elif stage=="pipeline_complete":return 0
            else:raise RuntimeError(f"Unknown descent-tube stage {stage}")
            self.save(failure_signature=None,consecutive_failure_count=0)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",default="runs/stage_experts/descent_tube_seed0_20260716T2330");args=parser.parse_args()
    controller=DescentTubeController(Path(args.run))
    try:raise SystemExit(controller.loop())
    except Exception as exc:
        stage=str(controller.state.get("current_stage","unknown"))
        signature,count=failure_fuse_update(controller.state,stage,exc)
        controller.log(f"ERROR {type(exc).__name__}: {exc} (identical failure {count}/3)")
        controller.save(stop_reason=f"{type(exc).__name__}: {exc}",failure_signature=signature,
                        consecutive_failure_count=count)
        if count>=3:
            controller.log("ENGINEERING PAUSE: identical deterministic failure reached restart fuse")
            raise SystemExit(41) from exc
        raise


if __name__=="__main__":main()
