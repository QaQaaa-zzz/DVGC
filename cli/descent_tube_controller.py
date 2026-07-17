"""Persistent controller for exact descent Tube audit and expansion."""
from __future__ import annotations

import hashlib
import json
import re
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
    seed_set_sha256,
)


SOURCE_RUN = Path("runs/stage_experts/descent_local_nonfinite_repair_seed0_20260716T1825")
BLOCK1_ROOT = SOURCE_RUN / "blocks/block_1_25600"
BLOCK1_POLICY = BLOCK1_ROOT / "train/policy"
BLOCK1_CERT = BLOCK1_ROOT / "current_policy_certified_sharded.pkl"
BLOCK1_CERT_REPORT = BLOCK1_ROOT / "current_policy_certified_sharded.cert.json"
FROZEN = Path("runs/stage_experts/descent_tube_seed0_20260716T2330/frozen_block1_v2")
POINTWISE_SEEDS = {1: 9_310_000, 2: 200_000_000, 3: 600_000_000}
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
    # Worker unit names contain launch timestamps.  They must not make the
    # same deterministic failure look unique on every systemd restart.
    normalized = re.sub(r"-\d{10}\.service", "-<launch>.service", str(exc))
    message = f"{stage}|{type(exc).__name__}|{normalized}"
    signature = hashlib.sha256(message.encode("utf-8")).hexdigest()
    previous = state.get("failure_signature")
    count = int(state.get("consecutive_failure_count", 0)) + 1 if previous == signature else 1
    return signature, count


def select_policy_for_hash(source_hash: str, candidates, hash_fn=file_sha256) -> Path:
    """Resolve candidate provenance to an existing immutable policy bundle."""
    matches = [Path(path) for path in candidates
               if (Path(path)/"params.pkl").exists() and hash_fn(Path(path)/"params.pkl") == str(source_hash)]
    if not matches:
        raise RuntimeError(f"No immutable policy matches candidate source hash {source_hash}")
    return sorted(matches, key=lambda path: str(path))[0]


class DescentTubeController(Controller):
    def __init__(self, run: Path):
        super().__init__(run)
        if self.state.get("controller_type") != "descent_tube":
            self.state = {
                "controller_type": "descent_tube", "controller_version": 3,
                "run_id": run.name, "current_stage": "inspect", "active_round": 1,
                "last_completed_action": None, "in_progress_action": None,
                "expected_outputs": [], "next_decision": "pointwise_audit",
                "retry_count": 0, "heartbeat": time.time(), "stop_reason": None,
                "active_worker_unit": None, "history": [], "provenance": {},
                "failure_signature": None, "consecutive_failure_count": 0,
            }
            self.save()
        elif int(self.state.get("controller_version", 1)) < 3:
            self.save(controller_version=3)
        if (self.state.get("current_stage")=="gate_pause"
                and "Round-2 exact pointwise Tube precision" in str(self.state.get("stop_reason"))):
            self.save(controller_version=3,current_stage="support_repair_build",active_round=3,
                      stop_reason=None,next_decision="support_repair_construction")

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

    def _audit_candidate_source_policy(self, candidate: Path) -> Path:
        bank=SnapshotBank.load(candidate);source_hash=bank.metadata.get("descent_policy_hash")
        if not source_hash:raise RuntimeError("Pointwise candidate bank lacks descent source-policy hash")
        candidates=[INITIAL,BLOCK1_POLICY]
        candidates.extend(sorted(self.run.glob("round_*/train/policy")))
        return select_policy_for_hash(str(source_hash),candidates)

    def _registered_audit_seed(self, round_id, total):
        """Allocate and persist a grid disjoint from every registered seed set."""
        claim_name=("round1_pointwise_audit" if int(round_id)==1
                    else f"round{round_id}_pointwise_seed{pointwise_seed(round_id)}")
        allocated,proof,attempts=self._registered_grid(
            claim_name,"pointwise_audit",pointwise_seed(round_id),total,AUDIT_BRANCHES)
        overrides=dict(self.state.get("pointwise_seed_overrides",{}));overrides[str(round_id)]=allocated
        self.save(pointwise_seed_overrides=overrides)
        root=self.run/f"round_{round_id}/pointwise_audit_seed{allocated}";root.mkdir(parents=True,exist_ok=True)
        proof_path=root/"seed_intersection_proof.json"
        save_json(proof_path,{**proof,"allocation_attempts":attempts,"registry":str(self.run/"seed_registry.json"),
                              "registry_sha256":file_sha256(self.run/"seed_registry.json")})
        self.save(seed_intersection_proof=str(proof_path))
        return allocated

    def _registered_grid(self, claim_name, category, preferred, total, branches):
        registry_path=self.run/"seed_registry.json"
        if not registry_path.exists():
            subprocess.run([PYTHON,"-m","cli.build_descent_seed_registry","--run",self.run],check=True)
        registry=json.loads(registry_path.read_text())
        existing=[claim for claim in registry["claims"] if claim["name"]!=claim_name]
        allocated,proof,attempts=allocate_disjoint_grid(preferred,total,branches,existing)
        claim=make_claim(
            claim_name,category,planned_branch_seeds(allocated,total,branches),status="active",
            base_seed=allocated,state_count=total,branches_per_state=branches,
            branch_variation_indices=list(range(branches)),
        )
        claims=[row for row in registry["claims"] if row["name"]!=claim_name]+[claim]
        metadata={key:value for key,value in registry.items() if key not in {"schema_version","claims","historical_intersections"}}
        save_registry(registry_path,claims,**metadata)
        self.save(seed_registry=str(registry_path))
        return allocated,proof,attempts

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
        candidate_bank=SnapshotBank.load(candidate)
        total=len(candidate_bank.records_for_phase("flight",include_training_only=False))
        if candidate_bank.metadata.get("policy_hash")!=file_sha256(policy/"params.pkl"):
            raise RuntimeError("Pointwise candidate/current-policy provenance mismatch")
        source_policy=self._audit_candidate_source_policy(candidate)
        seed=self._registered_audit_seed(round_id,total)
        root,candidate,policy,construction,manifest=self._audit_paths(round_id);root.mkdir(parents=True,exist_ok=True)
        audit_manifest=root/"pointwise_audit_manifest.json"
        if not audit_manifest.exists():
            save_json(audit_manifest,{"status":"ACTIVE","seed":seed,
                "seed_namespace":f"descent_pointwise_round_{round_id}:descent_entry","global_indices":[0,total],
                "states":total,"branches_per_state":AUDIT_BRANCHES,"branch_variation_indices":list(range(AUDIT_BRANCHES)),
                "seed_set_sha256":seed_set_sha256(planned_branch_seeds(seed,total)),
                "policy_hash":file_sha256(policy/"params.pkl"),"candidate_bank_sha256":file_sha256(candidate),
                "xml_sha256":file_sha256("assets/orange_bike_4kg_horizontal.xml"),
                "landing_entry_set_sha256":file_sha256(ENTRY),"landing_policy_hash":file_sha256(LANDING/"params.pkl"),
                "exact_membership_only":True,"continuous_matcher_active":False,"invalid_seed9330000_excluded":True})
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
                 "--candidate-source-policy",source_policy,"--landing-policy",LANDING,"--candidate-bank",candidate,
                 "--landing-entry-set",ENTRY,"--output",out,"--seed",seed,"--namespace",f"descent_pointwise_round_{round_id}",
                 "--start-index",start,"--end-index",end]
            result=self.run_worker_command(f"pointwise_audit_r{round_id}_{start}_{end}",cmd,root/f"shard_{start:03d}_{end:03d}.log",[out],
                                           unit_suffix=f"tube-r{round_id}-audit-{start}-{end}-{int(time.time())}",preallocate=(end-start==SHARD_SIZE))
            if result["ok"]:continue
            if not result["oom"]:raise RuntimeError(f"Pointwise audit worker failed: {result}")
            if end-start==1:
                single[start]=single.get(start,0)+1
                if single[start]>=2:
                    self.save(current_stage="authorized_stop",stop_reason=f"Pointwise state {start} OOM twice");return
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
        elif round_id==2:
            self.save(current_stage="support_repair_build",active_round=3,
                      next_decision="support_repair_construction",stop_reason=None)
        else:
            self.save(current_stage="gate_pause",stop_reason="Post-repair exact pointwise Tube gate failed")

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

    def support_repair_build(self):
        root=self.run/"round_3/support_repair";bank=root/"candidate_pool.pkl";report=root/"candidate_pool.report.json"
        seed,proof,attempts=self._registered_grid("support_repair_candidate_generation","candidate_generation",300_000_000,3001,1)
        root.mkdir(parents=True,exist_ok=True)
        save_json(root/"candidate_seed_proof.json",{**proof,"allocation_attempts":attempts})
        if not report.exists():
            result=self.run_worker_command("support_repair_build",[PYTHON,"-u","-m","cli.build_descent_support_repair",
                "--base-bank",self.run/"round_2/frozen/D_all_unique.pkl","--policy",self.run/"round_2/train/policy",
                "--landing-entry-set",ENTRY,"--output-bank",bank,"--output-report",report,"--seed",seed],
                root/"build.log",[bank,report],unit_suffix=f"tube-support-build-{int(time.time())}",preallocate=False)
            if not result["ok"]:
                if report.exists() and json.loads(report.read_text()).get("status")=="FAIL":
                    self.save(current_stage="gate_pause",stop_reason="Support-repair candidate quality gate failed");return
                raise RuntimeError(f"Support-repair builder failed: {result}")
        payload=json.loads(report.read_text())
        if payload["status"]!="PASS" or not payload["all_state_unique"] or payload["maximum_children_per_parent"]>4:
            self.save(current_stage="gate_pause",stop_reason="Support-repair candidate quality gate failed");return
        self.save(current_stage="support_repair_construction",next_decision="support_repair_train")

    def _certify_candidate_bank(self,root,candidate,policy,source_policy,claim_name,preferred,namespace):
        root.mkdir(parents=True,exist_ok=True);total=len(SnapshotBank.load(candidate).records_for_phase("flight",include_training_only=False))
        seed,proof,attempts=self._registered_grid(claim_name,"construction_certification",preferred,total,AUDIT_BRANCHES)
        save_json(root/"seed_intersection_proof.json",{**proof,"allocation_attempts":attempts})
        pending=[(s,min(s+SHARD_SIZE,total)) for s in range(0,total,SHARD_SIZE)];single={}
        while pending:
            start,end=pending.pop(0);out=root/f"shard_{start:03d}_{end:03d}.completed.json"
            if out.exists():continue
            cmd=[PYTHON,"-u","-m","cli.certify_descent_construction_shard","--descent-policy",policy,
                 "--candidate-source-policy",source_policy,"--landing-policy",LANDING,"--candidate-bank",candidate,
                 "--landing-entry-set",ENTRY,"--output",out,"--seed",seed,"--namespace",namespace,
                 "--start-index",start,"--end-index",end,"--confirm-safe-to-max"]
            result=self.run_worker_command(f"{claim_name}_{start}_{end}",cmd,root/f"shard_{start:03d}_{end:03d}.log",[out],
                                           unit_suffix=f"tube-{claim_name}-{start}-{end}-{int(time.time())}",preallocate=(end-start==SHARD_SIZE))
            if result["ok"]:continue
            if not result["oom"]:raise RuntimeError(f"Support certification worker failed: {result}")
            if end-start==1:
                single[start]=single.get(start,0)+1
                if single[start]>=2:self.save(current_stage="authorized_stop",stop_reason=f"Support state {start} OOM twice");return None,None
                pending.insert(0,(start,end))
            else:pending=split_range_after_oom(start,end)+pending
        shards=sorted(root.glob("shard_*.completed.json"));completed_coverage([json.loads(path.read_text()) for path in shards],total)
        bank=root/"current_policy.pkl";report=root/"current_policy.cert.json"
        if not report.exists():
            cmd=[PYTHON,"-m","cli.merge_descent_construction_shards"]
            for shard in shards:cmd.extend(["--shard",shard])
            cmd.extend(["--candidate-bank",candidate,"--output-bank",bank,"--output-report",report])
            self.run_command(f"merge_{claim_name}",cmd,root/"merge.log",[bank,report])
        return bank,report

    def support_repair_construction(self):
        support=self.run/"round_3/support_repair";candidate=support/"candidate_pool.pkl"
        bank,report=self._certify_candidate_bank(support/"construction",candidate,self.run/"round_2/train/policy",
            self.run/"round_2/train/policy","support_repair_pretrain_construction",400_000_000,"descent_support_repair_pretrain")
        if bank is None:return
        reset=support/"bootstrap_reset_bank.pkl";reset_report=support/"bootstrap_reset_bank.report.json"
        if not reset_report.exists():self.run_command("build_support_repair_bootstrap",[PYTHON,"-m","cli.build_descent_bootstrap_bank",
            "--bank",bank,"--output-bank",reset,"--output-report",reset_report],support/"bootstrap.log",[reset,reset_report])
        self.save(current_stage="support_repair_train",next_decision="post_repair_construction")

    def support_repair_train(self):
        root=self.run/"round_3";train=root/"train";report=train/"report.json";support=root/"support_repair"
        if not report.exists():
            result=self.run_worker_command("support_repair_train",[PYTHON,"-u","-m","cli.train_descent_local_block",
                "--resume-policy",self.run/"round_2/train/policy","--bootstrap-bank",support/"bootstrap_reset_bank.pkl",
                "--candidate-bank",support/"construction/current_policy.pkl","--entry-set",ENTRY,"--run",train,
                "--cumulative-steps",76800,"--restore-checkpoint",self.run/"round_2/train/orbax/000000051200","--seed",0],
                root/"train.log",[report,train/"policy/params.pkl"],unit_suffix=f"tube-support-train-{int(time.time())}")
            if not result["ok"]:raise RuntimeError(f"Support-repair training failed: {result}")
        self.state["provenance"]["policy_hash"]=file_sha256(train/"policy/params.pkl")
        self.save(current_stage="post_repair_construction",next_decision="post_repair_pointwise_audit")

    def post_repair_construction(self):
        root=self.run/"round_3";candidate=root/"support_repair/construction/current_policy.pkl"
        bank,report=self._certify_candidate_bank(root/"construction",candidate,root/"train/policy",self.run/"round_2/train/policy",
            "support_repair_posttrain_construction",500_000_000,"descent_support_repair_posttrain")
        if bank is None:return
        frozen=root/"frozen"
        if not frozen.exists():self.run_command("freeze_post_repair_exact_sets",[PYTHON,"-m","cli.freeze_discrete_descent_tube",
            "--certified-bank",bank,"--cert-report",report,"--policy",root/"train/policy","--output-dir",frozen],root/"freeze.log",[frozen/"discrete_tube_manifest.json"])
        self.save(current_stage="pointwise_audit",active_round=3,next_decision="pointwise_decision")

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
            elif stage=="support_repair_build":self.support_repair_build()
            elif stage=="support_repair_construction":self.support_repair_construction()
            elif stage=="support_repair_train":self.support_repair_train()
            elif stage=="post_repair_construction":self.post_repair_construction()
            elif stage=="viability_train":self.viability_train(round_id)
            elif stage=="acquisition":self.save(current_stage="gate_pause",stop_reason="Acquisition implementation not yet validated")
            elif stage=="gate_pause":return 40
            elif stage=="authorized_stop":return 41
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
