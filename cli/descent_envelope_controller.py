"""Persistent controller for stable descent Tube acquisition and activation."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from cli.descent_local_controller import (
    Controller, ENTRY, LANDING, PYTHON, RUNTIME_GATE, SHARD_SIZE,
)
from cli.descent_tube_controller import select_policy_for_hash
from dvgc.construction_lifecycle import failure_fuse_update, split_range_after_oom
from dvgc.audit_manifest import completed_manifest
from dvgc.bank import SnapshotBank
from dvgc.certification import branch_seed
from dvgc.config import config_hash,file_sha256, load_config
from dvgc.construction_lifecycle import PROTOCOL_VERSION,validate_artifact
from dvgc.policy import load_bundle
from dvgc.runtime import save_json
from dvgc.seed_registry import exact_intersection_proof, make_claim, save_registry
from dvgc.snapshot_provenance import declared_snapshot_source_hashes
from dvgc.stable_construction import adaptive_indices,protocol_from_config,stage_b_indices


SOURCE_RUN = Path("runs/stage_experts/descent_tube_seed0_20260716T2330")
SOURCE_CANDIDATE = SOURCE_RUN / "round_3/frozen/D_all_unique.pkl"
SOURCE_POLICY = SOURCE_RUN / "round_3/train/policy"
SOURCE_POLICY_ORIGIN = SOURCE_RUN / "round_2/train/policy"
SOURCE_CONSTRUCTION = SOURCE_RUN / "round_3/construction/current_policy.cert.json"
SOURCE_DEVELOPMENT_AUDIT = SOURCE_RUN / "round_3/pointwise_audit_seed600000000/merged.json"
SOURCE_DEVELOPMENT_MANIFEST = SOURCE_RUN / "round_3/pointwise_audit_seed600000000/pointwise_audit_manifest.json"
XML = Path("assets/orange_bike_4kg_horizontal.xml")


def planned_index_seeds(base_seed, indices, branches):
    return [branch_seed(base_seed, index, branch) for index in indices for branch in range(int(branches))]


class DescentEnvelopeController(Controller):
    def __init__(self, run: Path):
        super().__init__(run)
        if self.state.get("controller_type") != "descent_envelope":
            self.state = {
                "controller_type": "descent_envelope", "controller_version": 1,
                "controller_unit": "dvgc-descent-envelope-controller.service",
                "controller_module": "cli.descent_envelope_controller", "run_id": run.name,
                "current_stage": "inspect", "current_cycle": 0, "acquisition_round": 0,
                "last_completed_action": None, "in_progress_action": None,
                "expected_outputs": [], "next_decision": "stable_stage_a",
                "retry_count": 0, "heartbeat": time.time(), "stop_reason": None,
                "active_worker_unit": None, "history": [], "provenance": {},
                "failure_signature": None, "consecutive_failure_count": 0,
            }
            self.save()

    def inspect(self):
        tracked = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if tracked:
            raise RuntimeError("Tracked worktree must be clean")
        subprocess.run([PYTHON, "-m", "cli.runtime_gate", "--config", "configs/default.json",
                        "--output", RUNTIME_GATE, "--check-only"], check=True, stdout=subprocess.DEVNULL)
        required = [SOURCE_CANDIDATE, SOURCE_POLICY/"params.pkl", SOURCE_POLICY_ORIGIN/"params.pkl",
                    SOURCE_CONSTRUCTION, SOURCE_DEVELOPMENT_AUDIT, SOURCE_DEVELOPMENT_MANIFEST,
                    LANDING/"params.pkl", ENTRY, XML, SOURCE_RUN/"seed_registry.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Stable descent input missing: {missing}")
        development = json.loads(SOURCE_DEVELOPMENT_MANIFEST.read_text())
        if development.get("evidence_role") != "CONSUMED_DEVELOPMENT_EVIDENCE" or development.get("eligible_as_future_independent_audit") is not False:
            raise RuntimeError("Round-3 audit is not isolated as consumed development evidence")
        policy_hash = file_sha256(SOURCE_POLICY/"params.pkl")
        source = SnapshotBank.load(SOURCE_CANDIDATE)
        if source.metadata.get("policy_hash") != policy_hash:
            raise RuntimeError("Frozen Round-3 candidate/current-policy mismatch")
        registry = json.loads((SOURCE_RUN/"seed_registry.json").read_text())
        claims=[]
        for claim in registry["claims"]:
            item=dict(claim)
            if item.get("name")=="round3_pointwise_seed600000000":
                item["status"]="consumed_development_evidence"
            claims.append(item)
        save_registry(self.run/"seed_registry.json", claims, status="ACTIVE",
                      source_registry=str(SOURCE_RUN/"seed_registry.json"),
                      source_registry_sha256=file_sha256(SOURCE_RUN/"seed_registry.json"))
        gate=json.loads(RUNTIME_GATE.read_text())
        provenance={
            "head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "current_policy_hash":policy_hash,"policy_hash":policy_hash,
            "candidate_bank_sha256":file_sha256(SOURCE_CANDIDATE),
            "xml_sha256":file_sha256(XML),"c_l_hash":file_sha256(ENTRY),
            "pi_l_hash":file_sha256(LANDING/"params.pkl"),
            "runtime_source_fingerprint":gate["source_fingerprint"],
            "development_audit_sha256":file_sha256(SOURCE_DEVELOPMENT_AUDIT),
        }
        self.save(provenance=provenance,current_candidate=str(SOURCE_CANDIDATE),
                  current_policy=str(SOURCE_POLICY),current_source_policy=str(SOURCE_POLICY_ORIGIN),
                  current_checkpoint=str(SOURCE_RUN/"round_3/train/orbax/000000076800"),
                  policy_history=[str(SOURCE_POLICY_ORIGIN),str(SOURCE_POLICY)],route_phase="stable_discovery",
                  current_stage="stable_stage_a",last_completed_action="inspect",
                  next_decision="stable_stage_b",stop_reason=None)

    def _cycle_root(self):
        return self.run/f"cycle_{int(self.state.get('current_cycle',0))}"

    def _construction_identity(self):
        policy=Path(self.state["current_policy"]);_,saved_cfg,_=load_bundle(policy,verify_files=True)
        cfg=load_config("configs/default.json",{**saved_cfg,"training_stage":"flight","expert_chain_termination":False,
            "domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
        gate=json.loads(RUNTIME_GATE.read_text())
        return {"policy_hash":file_sha256(policy/"params.pkl"),"candidate_bank_sha256":file_sha256(self.state["current_candidate"]),
            "xml_sha256":file_sha256(XML),"landing_entry_set_sha256":file_sha256(ENTRY),
            "landing_policy_hash":file_sha256(LANDING/"params.pkl"),"config_hash":config_hash(cfg),
            "protocol":protocol_from_config(cfg),"certification_protocol_version":PROTOCOL_VERSION,
            "construction_seed_epoch":int(self.state["current_cycle"]),"runtime_source_fingerprint":gate["source_fingerprint"]}

    def _ensure_cycle_manifest(self):
        root=self._cycle_root();root.mkdir(parents=True,exist_ok=True);path=root/"construction_cycle_manifest.json"
        expected={"status":"ACTIVE","artifact_role":"immutable_stable_construction_cycle","cycle":int(self.state["current_cycle"]),
            "identity":self._construction_identity(),"checkpoint":str(self.state["current_checkpoint"]),
            "policy":str(self.state["current_policy"]),"candidate_bank":str(self.state["current_candidate"])}
        if path.exists():
            actual=json.loads(path.read_text())
            if actual!=expected:raise RuntimeError("Construction cycle manifest does not match current policy/bank provenance")
        else:save_json(path,expected)
        return expected

    def _validate_stage_artifact(self,path,stage,seed,*,complete=False):
        identity=self._ensure_cycle_manifest()["identity"]
        return validate_artifact(path,identity,checkpoint_mtime=(Path(self.state["current_policy"])/"params.pkl").stat().st_mtime,
            stage=stage,seed=seed,require_complete=complete)

    def _validate_stable_report(self,path):
        identity=self._ensure_cycle_manifest()["identity"]
        return validate_artifact(path,identity,checkpoint_mtime=(Path(self.state["current_policy"])/"params.pkl").stat().st_mtime)

    def _allocate_indices(self, name, category, preferred, indices, branches):
        registry_path=self.run/"seed_registry.json";registry=json.loads(registry_path.read_text())
        existing=[row for row in registry["claims"] if row["name"]!=name]
        attempts=[]
        for offset in range(4):
            seed=int(preferred)+offset*100_000_000
            claim=make_claim(name,category,planned_index_seeds(seed,indices,branches),status="active",
                             base_seed=seed,candidate_indices=list(indices),branches_per_state=int(branches))
            proof=exact_intersection_proof(claim,existing);attempts.append({"base_seed":seed,"proof":proof})
            if proof["status"]=="PASS":break
        else:raise RuntimeError(f"Seed allocator exhausted for {name}")
        save_registry(registry_path,existing+[claim],status="ACTIVE")
        return seed,proof,attempts

    def _source_policies(self, candidate):
        source_hashes=set(declared_snapshot_source_hashes(SnapshotBank.load(candidate).metadata))
        candidates=[SOURCE_POLICY_ORIGIN,SOURCE_POLICY]
        candidates.extend(Path(path) for path in self.state.get("policy_history",[]))
        resolved=[]
        for source_hash in sorted(source_hashes):resolved.append(select_policy_for_hash(source_hash,candidates))
        return resolved

    def _write_indices(self, path, indices):
        if not path.exists():save_json(path,[int(value) for value in indices])
        stored=json.loads(path.read_text())
        if stored!=[int(value) for value in indices]:raise RuntimeError("Persisted stable index selection changed")

    def _empty_stage(self, stage, seed, indices, output, reference):
        payload={key:reference[key] for key in (
            "candidate_bank_sha256","candidate_source_policy_hash","candidate_source_policy_hashes","descent_policy_hash","descent_policy_version",
            "landing_policy_hash","landing_policy_version","landing_entry_set_sha256","xml_sha256","config_hash",
            "runtime_source_fingerprint","protocol","certification_protocol_version","construction_seed_epoch",
            "branch_horizon","total_states")}
        payload.update({"status":"PASS","artifact_role":"merged_stable_construction_stage","stage":stage,
                        "seed":seed,"seed_namespace":f"stable_cycle_{self.state['current_cycle']}:{stage}:descent_entry",
                        "branches_per_state":reference["protocol"]["adaptive_max_branches" if stage=="adaptive" else "stage_branches"],
                        "selected_states":0,"selected_indices_sha256":__import__('hashlib').sha256(b"[]").hexdigest(),
                        "states":0,"candidate_indices":[],"shards":[],"terminal_summary":{},"rows":[]})
        save_json(output,payload)

    def _run_stable_stage(self, stage, indices, preferred):
        self._ensure_cycle_manifest()
        cfg=load_config("configs/default.json")
        branches=int(cfg.stable_construction_adaptive_max_branches if stage=="adaptive" else cfg.stable_construction_stage_branches)
        cycle=int(self.state["current_cycle"]);root=self._cycle_root()/stage;root.mkdir(parents=True,exist_ok=True)
        indices_path=root/"candidate_indices.json";self._write_indices(indices_path,indices)
        claim=f"stable_cycle_{cycle}_{stage}"
        seed,proof,attempts=self._allocate_indices(claim,f"stable_construction_{stage}",preferred,indices,branches)
        save_json(root/"seed_intersection_proof.json",{**proof,"allocation_attempts":attempts,
                  "registry":str(self.run/"seed_registry.json"),"registry_sha256":file_sha256(self.run/"seed_registry.json")})
        merged=root/"merged.json"
        if not indices:
            if not merged.exists():
                reference=json.loads((self._cycle_root()/"stage_a/merged.json").read_text())
                self._empty_stage(stage,seed,indices,merged,reference)
            return merged
        candidate=Path(self.state["current_candidate"]);policy=Path(self.state["current_policy"]);sources=self._source_policies(candidate)
        pending=[(start,min(start+SHARD_SIZE,len(indices))) for start in range(0,len(indices),SHARD_SIZE)];single={}
        while pending:
            start,end=pending.pop(0);out=root/f"shard_{start:03d}_{end:03d}.completed.json"
            if out.exists():self._validate_stage_artifact(out,stage,seed,complete=True);continue
            command=[PYTHON,"-u","-m","cli.certify_stable_descent_shard","--stage",stage,
                     "--descent-policy",policy]
            for source in sources:command.extend(["--candidate-source-policy",source])
            command.extend(["--landing-policy",LANDING,
                     "--candidate-bank",candidate,"--landing-entry-set",ENTRY,"--indices-file",indices_path,
                     "--start-index",start,"--end-index",end,"--seed",seed,
                     "--namespace",f"stable_cycle_{cycle}","--output",out])
            result=self.run_worker_command(f"{claim}_{start}_{end}",command,root/f"shard_{start:03d}_{end:03d}.log",[out],
                                           unit_suffix=f"envelope-{claim}-{start}-{end}-{int(time.time())}",preallocate=end-start==SHARD_SIZE)
            if result["ok"]:continue
            if not result["oom"]:raise RuntimeError(f"Stable construction worker failed: {result}")
            if end-start==1:
                single[start]=single.get(start,0)+1
                if single[start]>=2:self.save(current_stage="gate_pause",stop_reason=f"Stable state offset {start} OOM twice");return None
                pending.insert(0,(start,end))
            else:pending=split_range_after_oom(start,end)+pending
        shards=sorted(root.glob("shard_*.completed.json"))
        if not merged.exists():
            command=[PYTHON,"-m","cli.merge_stable_descent_stage"]
            for shard in shards:command.extend(["--shard",shard])
            command.extend(["--indices-file",indices_path,"--output",merged])
            self.run_command(f"merge_{claim}",command,root/"merge.log",[merged])
        self._validate_stage_artifact(merged,stage,seed)
        return merged

    def stage_a(self):
        candidate=SnapshotBank.load(self.state["current_candidate"])
        indices=list(range(len(candidate.records_for_phase("flight",include_training_only=False))))
        cycle=int(self.state["current_cycle"])
        if self._run_stable_stage("stage_a",indices,800_000_000+cycle*10_000_000) is None:return
        self.save(current_stage="stable_stage_b",next_decision="stable_adaptive")

    def stage_b(self):
        cfg=load_config("configs/default.json");a=json.loads((self._cycle_root()/"stage_a/merged.json").read_text())
        self._validate_stage_artifact(self._cycle_root()/"stage_a/merged.json","stage_a",a["seed"])
        indices=stage_b_indices(a["rows"],cfg);cycle=int(self.state["current_cycle"])
        if self._run_stable_stage("stage_b",indices,900_000_000+cycle*10_000_000) is None:return
        self.save(current_stage="stable_adaptive",next_decision="stable_analyze")

    def adaptive(self):
        cfg=load_config("configs/default.json")
        a=json.loads((self._cycle_root()/"stage_a/merged.json").read_text());b=json.loads((self._cycle_root()/"stage_b/merged.json").read_text())
        self._validate_stage_artifact(self._cycle_root()/"stage_a/merged.json","stage_a",a["seed"])
        self._validate_stage_artifact(self._cycle_root()/"stage_b/merged.json","stage_b",b["seed"])
        indices=adaptive_indices(a["rows"],b["rows"],cfg);cycle=int(self.state["current_cycle"])
        if self._run_stable_stage("adaptive",indices,1_000_000_000+cycle*10_000_000) is None:return
        self.save(current_stage="stable_analyze",next_decision="stable_decision")

    def analyze_stable(self):
        root=self._cycle_root();bank=root/"stable/current_policy.pkl";report=root/"stable/report.json"
        if not report.exists():
            self.run_command("analyze_stable_construction",[PYTHON,"-m","cli.analyze_stable_descent_construction",
                "--candidate-bank",self.state["current_candidate"],"--stage-a",root/"stage_a/merged.json",
                "--stage-b",root/"stage_b/merged.json","--adaptive",root/"adaptive/merged.json",
                "--policy",self.state["current_policy"],"--output-bank",bank,"--output-report",report],
                root/"stable/analyze.log",[bank,report])
        self._validate_stable_report(report)
        stable_payload=json.loads(report.read_text());self.state.setdefault("provenance",{}).update({
            "certified_policy_hash":stable_payload["policy_hash"],"candidate_bank_sha256":stable_payload["candidate_bank_sha256"]})
        self.save(current_stable_bank=str(bank),current_stable_report=str(report),
                  current_stage="stable_decision",next_decision="freeze_or_viability",last_successful_artifact=str(report))

    def stable_decision(self):
        report=json.loads(Path(self.state["current_stable_report"]).read_text())
        previous=int(self.state.get("last_passed_stable_safe",0) or 0)
        if self.state.get("route_phase")=="tube_rsi" and int(report["stable_safe_states"])<previous:
            self.save(current_policy=self.state["last_passed_policy"],current_stable_bank=self.state["last_passed_stable_bank"],
                      current_stable_report=self.state["last_passed_stable_report"],current_checkpoint=self.state["last_passed_checkpoint"],
                      current_stage="viability_train" if int(self.state.get("acquisition_round",0))<2 else "gate_pause",
                      stop_reason=None if int(self.state.get("acquisition_round",0))<2 else "Tube-RSI stable-safe retention declined after rollback",
                      next_decision="candidate_acquisition")
            return
        if report["activation_support_pass"] and report["parent_diversity_pass"]:
            self.save(current_stage="stable_freeze",next_decision="fresh_pointwise_audit")
        elif int(self.state.get("acquisition_round",0))<int(load_config("configs/default.json").descent_acquisition_max_rounds):
            self.save(current_stage="viability_train",next_decision="candidate_acquisition")
        else:self.save(current_stage="gate_pause",stop_reason="Stable-safe support remains below activation gate after two acquisition rounds")

    def freeze_stable(self):
        root=self._cycle_root();frozen=root/"frozen"
        if not (frozen/"discrete_tube_manifest.json").exists():
            self.run_command("freeze_stable_exact_tube",[PYTHON,"-m","cli.freeze_discrete_descent_tube",
                "--certified-bank",self.state["current_stable_bank"],"--cert-report",self.state["current_stable_report"],
                "--policy",self.state["current_policy"],"--output-dir",frozen],root/"stable/freeze.log",
                [frozen/"discrete_tube_manifest.json"])
        self.state["provenance"].update({"frozen_manifest":str(frozen/"discrete_tube_manifest.json"),
            "current_policy_hash":file_sha256(Path(self.state["current_policy"])/"params.pkl")})
        self.save(current_stage="pointwise_audit",next_decision="pointwise_decision")

    def viability_train(self):
        root=self._cycle_root()/"viability";model=root/"ensemble.pkl";report=root/"report.json"
        seed,proof,attempts=self._allocate_indices(f"stable_cycle_{self.state['current_cycle']}_viability","viability_training",
                                                   1_200_000_000+int(self.state["current_cycle"])*10_000_000,[0],1)
        if not report.exists():
            self.run_command("train_development_viability",[PYTHON,"-m","cli.fit_viability",
                "--bank",self.state["current_stable_bank"],"--output",model,"--report",report,"--seed",seed,
                "--development-report",SOURCE_CONSTRUCTION,"--development-report",SOURCE_DEVELOPMENT_AUDIT],
                root/"train.log",[model,report])
        self.save(current_viability_model=str(model),current_stage="acquisition",next_decision="stable_stage_a")

    def acquisition(self):
        acquisition_round=int(self.state.get("acquisition_round",0))+1;root=self.run/f"acquisition_{acquisition_round}"
        bank=root/"candidate_pool.pkl";report=root/"candidate_pool.report.json"
        preferred=1_300_000_000+(acquisition_round-1)*10_000_000
        indices=list(range(3001));seed,proof,attempts=self._allocate_indices(f"acquisition_round_{acquisition_round}","candidate_generation",preferred,indices,1)
        save_json(root/"seed_intersection_proof.json",{**proof,"allocation_attempts":attempts})
        if not report.exists():
            result=self.run_worker_command(f"acquisition_round_{acquisition_round}",[PYTHON,"-u","-m","cli.build_descent_support_repair",
                "--base-bank",self.state["current_stable_bank"],"--policy",self.state["current_policy"],
                "--landing-entry-set",ENTRY,"--output-bank",bank,"--output-report",report,"--seed",seed,
                "--target",64,"--proposal-budget",3000,"--parent-cap",4,
                "--viability-model",self.state["current_viability_model"],"--acquisition-round",acquisition_round],
                root/"build.log",[bank,report],unit_suffix=f"envelope-acquisition-{acquisition_round}-{int(time.time())}",preallocate=False)
            if not result["ok"]:
                if report.exists() and json.loads(report.read_text()).get("status")=="FAIL":
                    self.save(current_stage="gate_pause",stop_reason=f"Acquisition round {acquisition_round} candidate quality FAIL");return
                raise RuntimeError(f"Acquisition worker failed: {result}")
        payload=json.loads(report.read_text())
        if payload["status"]!="PASS" or payload["maximum_children_per_parent"]>4 or payload["prediction_can_promote_empirical_safe"]:
            self.save(current_stage="gate_pause",stop_reason=f"Acquisition round {acquisition_round} provenance/quality FAIL");return
        self.save(acquisition_round=acquisition_round,current_cycle=acquisition_round,current_candidate=str(bank),
                  current_source_policy=str(self.state["current_policy"]),current_stage="stable_stage_a",next_decision="stable_stage_b")

    def _audit_paths(self):
        root=self._cycle_root()/"pointwise_audit";frozen=self._cycle_root()/"frozen"
        return root,Path(self.state["current_stable_bank"]),Path(self.state["current_policy"]),Path(self.state["current_stable_report"]),frozen/"discrete_tube_manifest.json"

    def pointwise_audit(self):
        root,candidate,policy,construction,manifest=self._audit_paths();root.mkdir(parents=True,exist_ok=True)
        total=len(SnapshotBank.load(candidate).records_for_phase("flight",include_training_only=False));indices=list(range(total))
        seed,proof,attempts=self._allocate_indices(f"stable_cycle_{self.state['current_cycle']}_independent_audit","pointwise_audit",1_500_000_000+int(self.state["current_cycle"])*10_000_000,indices,32)
        save_json(root/"seed_intersection_proof.json",{**proof,"allocation_attempts":attempts,"registry":str(self.run/"seed_registry.json")})
        audit_manifest=root/"pointwise_audit_manifest.json"
        if not audit_manifest.exists():save_json(audit_manifest,{"status":"ACTIVE","seed":seed,
            "seed_namespace":f"stable_cycle_{self.state['current_cycle']}_audit:descent_entry","global_indices":[0,total],
            "states":total,"branches_per_state":32,"branch_variation_indices":list(range(32)),
            "policy_hash":file_sha256(policy/"params.pkl"),"candidate_bank_sha256":file_sha256(candidate),
            "xml_sha256":file_sha256(XML),"landing_entry_set_sha256":file_sha256(ENTRY),
            "landing_policy_hash":file_sha256(LANDING/"params.pkl"),"exact_membership_only":True,
            "continuous_matcher_active":False,"development_evidence_excluded":True})
        sources=self._source_policies(candidate);pending=[(start,min(start+SHARD_SIZE,total)) for start in range(0,total,SHARD_SIZE)];single={}
        while pending:
            start,end=pending.pop(0);out=root/f"shard_{start:03d}_{end:03d}.completed.json"
            if out.exists():continue
            command=[PYTHON,"-u","-m","cli.certify_descent_entries","--audit-only","--descent-policy",policy]
            for source in sources:command.extend(["--candidate-source-policy",source])
            command.extend(["--landing-policy",LANDING,"--candidate-bank",candidate,"--landing-entry-set",ENTRY,
                "--output",out,"--seed",seed,"--namespace",f"stable_cycle_{self.state['current_cycle']}_audit",
                "--start-index",start,"--end-index",end])
            result=self.run_worker_command(f"independent_audit_{start}_{end}",command,
                root/f"shard_{start:03d}_{end:03d}.log",[out],unit_suffix=f"envelope-audit-{start}-{end}-{int(time.time())}",preallocate=end-start==SHARD_SIZE)
            if result["ok"]:continue
            if not result["oom"]:raise RuntimeError(f"Independent audit worker failed: {result}")
            if end-start==1:
                single[start]=single.get(start,0)+1
                if single[start]>=2:self.save(current_stage="gate_pause",stop_reason=f"Independent audit state {start} OOM twice");return
                pending.insert(0,(start,end))
            else:pending=split_range_after_oom(start,end)+pending
        shards=sorted(root.glob("shard_*.completed.json"));merged=root/"merged.json"
        if not merged.exists():
            command=[PYTHON,"-m","cli.merge_descent_entry_audits"]
            for shard in shards:command.extend(["--shard",shard])
            command.extend(["--output",merged]);self.run_command("merge_stable_pointwise_audit",command,root/"merge.log",[merged])
        analysis=root/"analysis.json"
        if not analysis.exists():self.run_command("analyze_stable_pointwise_audit",[PYTHON,"-m","cli.analyze_pointwise_descent_audit",
            "--candidate-bank",candidate,"--manifest",manifest,"--construction-report",construction,
            "--audit-report",merged,"--output",analysis],root/"analysis.log",[analysis])
        launch=json.loads(audit_manifest.read_text());terminal=completed_manifest(launch,json.loads(analysis.read_text()),json.loads(merged.read_text()))
        if launch!=terminal:save_json(audit_manifest,terminal)
        self.save(current_stage="pointwise_decision",next_decision="tube_rsi_or_acquisition")

    def pointwise_decision(self):
        root,*_=self._audit_paths();analysis=json.loads((root/"analysis.json").read_text());cfg=load_config("configs/default.json")
        passed=(analysis["status"]=="PASS" and analysis["pointwise"]["precision"]>=.95
                and len(analysis["member_results"])>=int(cfg.stable_construction_min_safe_states)
                and analysis["member_parent_count"]>=int(cfg.stable_construction_min_safe_parents)
                and analysis["timeout_rate"]==0 and analysis["provenance"]["consistent"])
        if passed:
            marker=self._cycle_root()/"discrete_tube_active.json";save_json(marker,{"status":"PASS","role":"stable_discrete_empirical_descent_tube",
                "policy_hash":analysis["provenance"]["policy_hash"],"pointwise_audit_sha256":file_sha256(root/"analysis.json"),
                "continuous_matcher_active":False})
            construction=json.loads(Path(self.state["current_stable_report"]).read_text())
            if self.state.get("route_phase")=="tube_rsi":
                early=int(construction.get("stable_safe_layers",{}).get("early",0));previous=int(self.state.get("last_passed_stable_safe",0) or 0)
                growth=int(construction["stable_safe_states"])>previous or int(construction["stable_safe_parents"])>int(self.state.get("last_passed_stable_parents",0) or 0)
                self.save(last_passed_policy=self.state["current_policy"],last_passed_stable_bank=self.state["current_stable_bank"],
                          last_passed_stable_report=self.state["current_stable_report"],last_passed_checkpoint=self.state["current_checkpoint"],
                          last_passed_stable_safe=int(construction["stable_safe_states"]),last_passed_stable_parents=int(construction["stable_safe_parents"]))
                if early>0:self.save(current_stage="continuous_cd",next_decision="continuous_cd_audit");return
                if growth and int(self.state.get("tube_rsi_block",0))<4:self.save(current_stage="tube_rsi_prepare",next_decision="tube_rsi_train");return
                if int(self.state.get("acquisition_round",0))<int(cfg.descent_acquisition_max_rounds):self.save(current_stage="viability_train",next_decision="candidate_acquisition");return
                self.save(current_stage="gate_pause",stop_reason="Tube-RSI support did not grow toward early descent");return
            self.save(route_phase="tube_rsi",tube_rsi_block=0,last_passed_policy=self.state["current_policy"],
                      last_passed_stable_bank=self.state["current_stable_bank"],last_passed_stable_report=self.state["current_stable_report"],
                      last_passed_checkpoint=self.state["current_checkpoint"],last_passed_stable_safe=int(construction["stable_safe_states"]),
                      last_passed_stable_parents=int(construction["stable_safe_parents"]),current_stage="tube_rsi_prepare",next_decision="tube_rsi_train")
        elif int(self.state.get("acquisition_round",0))<int(cfg.descent_acquisition_max_rounds):
            self.save(current_stage="viability_train",next_decision="candidate_acquisition")
        else:self.save(current_stage="gate_pause",stop_reason="Stable pointwise Tube failed after bounded acquisition")

    def tube_rsi_prepare(self):
        block=int(self.state.get("tube_rsi_block",0))+1;root=self.run/f"tube_rsi/block_{block}"
        bank=root/"reset_bank.pkl";report=root/"reset_bank.report.json"
        if not report.exists():self.run_command(f"build_tube_rsi_bank_{block}",[PYTHON,"-m","cli.build_discrete_tube_rsi_bank",
            "--stable-bank",self.state["current_stable_bank"],"--output-bank",bank,"--output-report",report],
            root/"reset_bank.log",[bank,report])
        self.save(pending_tube_rsi_block=block,pending_tube_rsi_bank=str(bank),current_stage="tube_rsi_train",next_decision="stable_stage_a")

    def tube_rsi_train(self):
        block=int(self.state["pending_tube_rsi_block"]);root=self.run/f"tube_rsi/block_{block}";train=root/"train";report=train/"report.json"
        cumulative=int(self.state.get("tube_rsi_base_cumulative",76800))+block*25600
        if not report.exists():
            result=self.run_worker_command(f"tube_rsi_train_{block}",[PYTHON,"-u","-m","cli.train_descent_local_block",
                "--resume-policy",self.state["current_policy"],"--bootstrap-bank",self.state["pending_tube_rsi_bank"],
                "--candidate-bank",self.state["current_stable_bank"],"--entry-set",ENTRY,"--run",train,
                "--cumulative-steps",cumulative,"--restore-checkpoint",self.state["current_checkpoint"],"--seed",0],
                root/"train.log",[report,train/"policy/params.pkl"],unit_suffix=f"envelope-tube-rsi-{block}-{int(time.time())}")
            if not result["ok"]:raise RuntimeError(f"Tube-RSI block {block} failed: {result}")
        payload=json.loads(report.read_text())
        if payload.get("health",{}).get("oom") or payload.get("health",{}).get("timeout") or payload.get("health",{}).get("nonfinite_metric_keys"):
            self.save(current_stage="gate_pause",stop_reason=f"Tube-RSI block {block} health gate failed");return
        policy=train/"policy";checkpoint=train/"orbax"/f"{cumulative:012d}"
        history=list(self.state.get("policy_history",[]));history.append(str(policy))
        self.state["provenance"]["current_policy_hash"]=file_sha256(policy/"params.pkl")
        self.save(tube_rsi_block=block,current_policy=str(policy),current_checkpoint=str(checkpoint),policy_history=history,
                  current_cycle=int(self.state["current_cycle"])+1,current_stage="stable_stage_a",next_decision="stable_stage_b")

    def loop(self):
        self.log(f"descent-envelope controller run={self.run}")
        while True:
            stage=self.state["current_stage"];self.save()
            if stage=="inspect":self.inspect()
            elif stage=="stable_stage_a":self.stage_a()
            elif stage=="stable_stage_b":self.stage_b()
            elif stage=="stable_adaptive":self.adaptive()
            elif stage=="stable_analyze":self.analyze_stable()
            elif stage=="stable_decision":self.stable_decision()
            elif stage=="stable_freeze":self.freeze_stable()
            elif stage=="viability_train":self.viability_train()
            elif stage=="acquisition":self.acquisition()
            elif stage=="pointwise_audit":self.pointwise_audit()
            elif stage=="pointwise_decision":self.pointwise_decision()
            elif stage=="tube_rsi_prepare":self.tube_rsi_prepare()
            elif stage=="tube_rsi_train":self.tube_rsi_train()
            elif stage=="continuous_cd":self.save(current_stage="gate_pause",stop_reason="Continuous C_D calibration requires completed Tube-RSI evidence")
            elif stage=="gate_pause":return 40
            elif stage=="authorized_stop":return 41
            elif stage=="pipeline_complete":return 0
            else:raise RuntimeError(f"Unknown descent-envelope stage {stage}")
            self.save(failure_signature=None,consecutive_failure_count=0)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);args=parser.parse_args()
    controller=DescentEnvelopeController(Path(args.run))
    try:raise SystemExit(controller.loop())
    except Exception as exc:
        stage=str(controller.state.get("current_stage","unknown"));signature,count=failure_fuse_update(controller.state,stage,exc)
        controller.log(f"ERROR {type(exc).__name__}: {exc} (identical failure {count}/3)")
        controller.save(stop_reason=f"{type(exc).__name__}: {exc}",failure_signature=signature,consecutive_failure_count=count)
        if count>=3:raise SystemExit(41) from exc
        raise


if __name__=="__main__":main()
