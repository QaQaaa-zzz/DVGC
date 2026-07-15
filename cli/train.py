"""Train one backward-bootstrap stage without modifying Tube labels."""
from __future__ import annotations
import argparse, copy, datetime as dt, json, math, time
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.bounded import drift_probe, evaluate_records
from dvgc.config import STAGE_ID, file_sha256, load_config, save_config
from dvgc.curriculum import FLIGHT_RESET_STAGES, select_flight_reset_records
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import make_ppo_train_fn, ppo_effective_timesteps, save_json, validate_ppo_batch_layout


LANDING_ENTROPY_COST = 1e-4
DEFAULT_ENTROPY_COST = 1e-3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["landing","flight","takeoff","approach","full"])
    p.add_argument("--bank", required=True)
    p.add_argument("--downstream-bank", default="")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--run", required=True)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    # The authoritative Warp model exceeds its aggregate contact capacity at
    # 1024 parallel Landing environments even when each individual state is
    # valid.  This 320 x (80 * 4) layout is the validated formal default.
    p.add_argument("--num-envs", type=int, default=320)
    p.add_argument("--num-eval-envs", type=int, default=128)
    p.add_argument("--num-evals", type=int, default=11)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--segment-index", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--require-final-safe-rsi", action="store_true")
    p.add_argument("--batch-size", type=int, default=80)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--entropy-cost", type=float, default=None)
    p.add_argument("--flight-reset-stage", choices=FLIGHT_RESET_STAGES, default="full")
    p.add_argument("--downstream-rehearsal-mass", type=float, default=None)
    p.add_argument("--entry-rehearsal-bank", default="")
    p.add_argument("--landing-rehearsal-bank", default="")
    p.add_argument("--flight-reset-mass", type=float, default=.60)
    p.add_argument("--entry-reset-mass", type=float, default=.10)
    p.add_argument("--landing-reset-mass", type=float, default=.30)
    p.add_argument("--bounded-block-dir", default="")
    p.add_argument("--reference-policy", default="")
    p.add_argument("--fixed-flight-bank", default="")
    p.add_argument("--landing-retention-bank", default="")
    p.add_argument("--baseline-flight-evaluation", default="")
    p.add_argument("--landing-reference-evaluation", default="")
    p.add_argument("--retention-minimum", type=float, default=.80)
    p.add_argument("--preflight-output", default="")
    a=p.parse_args(); ppo_seed=int(a.seed)+int(a.segment_index)*1_000_003
    run=Path(a.run)
    if run.exists(): raise SystemExit(f"Run directory already exists: {run}")
    validate_ppo_batch_layout(num_envs=a.num_envs,batch_size=a.batch_size,num_minibatches=a.num_minibatches)
    effective_timesteps=ppo_effective_timesteps(
        a.timesteps,unroll_length=32,batch_size=a.batch_size,
        num_minibatches=a.num_minibatches,num_evals=a.num_evals,
    )
    overrides={"training_stage":a.stage}
    if a.downstream_rehearsal_mass is not None:
        if not 0.0<=a.downstream_rehearsal_mass<1.0: raise SystemExit("--downstream-rehearsal-mass must be in [0,1)")
        overrides["downstream_rehearsal_mass"]=a.downstream_rehearsal_mass
    cfg=load_config(a.config, overrides)
    bank=SnapshotBank.load(a.bank); downstream=SnapshotBank.load(a.downstream_bank) if a.downstream_bank else SnapshotBank()
    entry_rehearsal=SnapshotBank.load(a.entry_rehearsal_bank) if a.entry_rehearsal_bank else None
    landing_rehearsal=SnapshotBank.load(a.landing_rehearsal_bank) if a.landing_rehearsal_bank else None
    restore_params=None; resume_manifest={}
    if a.resume:
        restore_params,resume_cfg,resume_manifest=load_bundle(a.resume,verify_files=True)
        if resume_manifest.get("xml_sha256") != file_sha256(cfg.xml_path):
            raise SystemExit("Resume policy was trained with a different XML model")
        if resume_manifest.get("action_mapping_version") != cfg.action_mapping_version:
            raise SystemExit("Resume policy uses a different action mapping")
        if int(resume_cfg.get("actor_history_steps",-1)) != int(cfg.actor_history_steps):
            raise SystemExit("Resume policy uses an incompatible Actor observation history")

    certified_rows=[r for r in bank.records_for_phase(a.stage,include_training_only=False) if r["final"]["branches"]>0] if a.stage!="full" else []
    safe_count=sum(r["final"]["label"]=="safe" for r in certified_rows)
    boundary_count=sum(r["final"]["label"]=="boundary" for r in certified_rows)
    if certified_rows:
        if not a.resume: raise SystemExit("Policy-conditioned Tube resets require --resume with the certified policy")
        try: bank.validate_certification_provenance(a.stage,policy_version=resume_manifest["policy_version"],estimator_version=resume_manifest.get("estimator_version","event_filter_v1"))
        except ValueError as exc: raise SystemExit(str(exc)) from exc
    if a.require_final_safe_rsi:
        if a.stage=="full": raise SystemExit("--require-final-safe-rsi is only valid for backward-bootstrap phases")
        if safe_count<int(cfg.tube_activation_min_safe):
            raise SystemExit(f"Final-safe RSI requires at least {int(cfg.tube_activation_min_safe)} safe {a.stage} records, found {safe_count}")
        reset_mode="final_safe_boundary_tube_rsi"
    elif certified_rows:
        reset_mode="certified_bank_fallback" if safe_count<int(cfg.tube_activation_min_safe) else "final_safe_boundary_tube_rsi"
    else:
        reset_mode="geometric_bootstrap"
    reset_protocol={"mode":reset_mode,"bank":str(Path(a.bank).resolve()),"certified_records":len(certified_rows),"final_safe_records":safe_count,"boundary_records":boundary_count,"initial_policy_version":resume_manifest.get("policy_version")}
    # Rehearse the already certified downstream phase while extending the same
    # shared Actor backward.  These copies are training-only and can never be
    # certified as current-stage states.
    current_records=bank.records
    if a.stage=="flight":
        current_records=select_flight_reset_records(bank.records_for_phase("flight",include_training_only=False),a.flight_reset_stage)
    elif a.flight_reset_stage!="full":
        raise SystemExit("--flight-reset-stage is only valid for Flight")
    train_records=[]
    multi_source = a.stage=="flight" and entry_rehearsal is not None and landing_rehearsal is not None
    if multi_source:
        masses={"flight_curriculum":float(a.flight_reset_mass),"canonical_entry_rehearsal":float(a.entry_reset_mass),"landing_tube_rehearsal":float(a.landing_reset_mass)}
        if any(value<=0 for value in masses.values()) or abs(sum(masses.values())-1.0)>1e-6:
            raise SystemExit("Multi-source reset masses must be positive and sum to one")
        sources={
            "flight_curriculum":(current_records,a.bank),
            "canonical_entry_rehearsal":([r for r in entry_rehearsal.records if r["final"]["label"]=="safe" and not r.get("training_only",False)],a.entry_rehearsal_bank),
            "landing_tube_rehearsal":([r for r in landing_rehearsal.records if r["final"]["label"] in ("safe","boundary") and not r.get("training_only",False)],a.landing_rehearsal_bank),
        }
        for source,(source_rows,source_path) in sources.items():
            if not source_rows: raise SystemExit(f"Reset source {source} is empty")
            safe=[r for r in source_rows if r["final"]["label"]=="safe"]
            boundary=[r for r in source_rows if r["final"]["label"]=="boundary"]
            groups=[(source_rows,1.0)]
            if source=="landing_tube_rehearsal" and boundary:
                groups=[(safe,.85),(boundary,.15)]
            for group,within_mass in groups:
                for row in group:
                    item=copy.deepcopy(row)
                    item.setdefault("origin_phase",item.get("source_phase"))
                    item["reset_source"]=source; item["reset_weight"]=masses[source]*within_mass/len(group)
                    item["original_bank_path"]=str(Path(source_path).resolve()); item["original_bank_sha256"]=file_sha256(source_path)
                    train_records.append(item)
        reset_protocol.update({"reset_source_masses":masses,"entry_rehearsal_bank":str(Path(a.entry_rehearsal_bank).resolve()),"entry_rehearsal_bank_sha256":file_sha256(a.entry_rehearsal_bank),"landing_rehearsal_bank":str(Path(a.landing_rehearsal_bank).resolve()),"landing_rehearsal_bank_sha256":file_sha256(a.landing_rehearsal_bank),"landing_safe_within_source_mass":.85,"landing_boundary_within_source_mass":.15})
    else:
        train_records=[copy.deepcopy(r) for r in current_records]
    if a.downstream_bank and not multi_source:
        for row in downstream.records:
            if row["final"]["label"] in ("safe","boundary") and not row.get("training_only",False):
                rehearsal=copy.deepcopy(row); rehearsal["source_phase"]=a.stage; rehearsal["training_only"]=True; rehearsal["candidate_kind"]="downstream_rehearsal"; train_records.append(rehearsal)
    reset_protocol.update({"flight_reset_stage":a.flight_reset_stage if a.stage=="flight" else None,"current_stage_reset_records":len(current_records),"full_candidate_records":len(bank.records_for_phase(a.stage,include_training_only=False)) if a.stage!="full" else 0,"downstream_rehearsal_mass":float(cfg.downstream_rehearsal_mass)})
    training_metadata=copy.deepcopy(bank.metadata)
    if multi_source: training_metadata["reset_source_protocol"]=copy.deepcopy(reset_protocol)
    training_bank=SnapshotBank(train_records,training_metadata)
    env=OrangeBikeDVGC(cfg,snapshot_bank=training_bank,cert_bank=downstream)
    eval_cfg=load_config(a.config,{"training_stage":a.stage,"domain_randomization":False,"obs_noise_enable":False})
    eval_env=OrangeBikeDVGC(eval_cfg,snapshot_bank=training_bank if multi_source else bank,cert_bank=downstream)
    if a.preflight_output:
        if not multi_source: raise SystemExit("Multi-source preflight requires all three reset sources")
        source_weight={name:0.0 for name in ("flight_curriculum","canonical_entry_rehearsal","landing_tube_rehearsal")}; source_count={name:0 for name in source_weight}
        for row in train_records: source_weight[row["reset_source"]]+=float(row["reset_weight"]); source_count[row["reset_source"]]+=1
        sample=next(r for r in train_records if r["reset_source"]=="landing_tube_rehearsal")
        lcfg=load_config(a.config,{"training_stage":"landing","domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False}); fcfg=load_config(a.config,{"training_stage":"flight","domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
        lenv=OrangeBikeDVGC(lcfg,snapshot_bank=SnapshotBank()); fenv=OrangeBikeDVGC(fcfg,snapshot_bank=SnapshotBank(),cert_bank=downstream); key=jax.random.PRNGKey(99001); action=jp.zeros(lenv.action_size,jp.float32)
        ls=lenv.step(restore_snapshot(lenv,sample,key),action); fs=fenv.step(restore_snapshot(fenv,sample,key),action); terms=("reward/total","reward/recovery_shaping","reward/recovery_streak","reward/failure_penalty","reward/instability_penalty")
        consistency={term:{"landing":float(np.asarray(ls.metrics[term])),"flight_rehearsal":float(np.asarray(fs.metrics[term])),"equal":bool(np.allclose(np.asarray(ls.metrics[term]),np.asarray(fs.metrics[term]),rtol=1e-6,atol=1e-6))} for term in terms}
        from cli.runtime_gate import source_fingerprint
        gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text()); root=Path(__file__).resolve().parents[1]
        report={"status":"PASS","reset_source_weights":source_weight,"reset_source_counts":source_count,"distinct_sources":len(source_count),"natural_probability":float(cfg.natural_prob_flight),"landing_records_preserve_origin":all(r.get("origin_phase")=="landing" and int(r.get("oracle_phase",-1))==STAGE_ID["landing"] and bool(r.get("policy_state")) for r in train_records if r["reset_source"]!="flight_curriculum"),"original_bank_hashes":sorted({r["original_bank_sha256"] for r in train_records}),"reward_gate_consistency":consistency,"phase_equal":int(ls.info["phase"])==int(fs.info["phase"]),"termination_equal":int(ls.info["end_code"])==int(fs.info["end_code"]),"recovery_equal":int(ls.info["recovery_success"])==int(fs.info["recovery_success"]),"active_c_l_sha256":file_sha256(a.downstream_bank),"runtime_gate_status":gate.get("status"),"runtime_gate_current":gate.get("source_fingerprint")==source_fingerprint(root)}
        report["status"]="PASS" if all(item["equal"] for item in consistency.values()) and report["landing_records_preserve_origin"] and report["phase_equal"] and report["termination_equal"] and report["recovery_equal"] and report["runtime_gate_status"]=="PASS" and report["runtime_gate_current"] else "FAIL"
        save_json(a.preflight_output,report); print(json.dumps(report,indent=2)); raise SystemExit(0 if report["status"]=="PASS" else 2)
    run.mkdir(parents=True,exist_ok=False); save_config(cfg,run/"config.json")
    metrics_path=run/"training_metrics.json"
    learning_rate=float(a.learning_rate)
    entropy_cost=(LANDING_ENTROPY_COST if a.stage=="landing" else DEFAULT_ENTROPY_COST) if a.entropy_cost is None else float(a.entropy_cost)
    if learning_rate<=0: raise SystemExit("--learning-rate must be positive")
    if entropy_cost<0: raise SystemExit("--entropy-cost must be non-negative")
    rows=[]
    started_at=time.time()
    metric_log={"stage":a.stage,"seed":a.seed,"segment_index":a.segment_index,"ppo_seed":ppo_seed,"requested_timesteps":a.timesteps,"effective_timesteps":effective_timesteps,"ppo_layout":{"num_envs":a.num_envs,"num_eval_envs":a.num_eval_envs,"unroll_length":32,"batch_size":a.batch_size,"num_minibatches":a.num_minibatches,"num_evals":a.num_evals},"ppo_hyperparameters":{"learning_rate":learning_rate,"entropy_cost":entropy_cost,"reward_scaling":0.1,"num_updates_per_batch":2,"discounting":.995,"gae_lambda":.97,"clipping_epsilon":.10,"max_grad_norm":.75},"reset_protocol":reset_protocol,"status":"initialized","started_at":started_at,"progress":rows}
    save_json(metrics_path,metric_log)
    def progress(step,metrics):
        row={"step":int(step),"recorded_at":time.time(),**{k:float(v) for k,v in metrics.items() if hasattr(v,"__float__")}}; rows.append(row)
        metric_log["status"]="running"; save_json(metrics_path,metric_log)
        print(f"[train] stage={a.stage} step={step:,}")
    class BoundedStop(RuntimeError): pass
    block_history=[]
    if a.bounded_block_dir:
        required=(a.reference_policy,a.fixed_flight_bank,a.landing_retention_bank,a.baseline_flight_evaluation,a.landing_reference_evaluation)
        if not multi_source or not all(required): raise SystemExit("Bounded protocol requires all fixed banks, reference policy, and multi-source sampler")
        if a.timesteps!=102400 or a.num_evals!=5 or effective_timesteps!=102400: raise SystemExit("Bounded protocol must be one continuous 4 x 25,600-step run")
        reference_params,_,reference_manifest=load_bundle(a.reference_policy,verify_files=True)
        fixed_flight=SnapshotBank.load(a.fixed_flight_bank); retention_bank=SnapshotBank.load(a.landing_retention_bank)
        baseline_flight=json.loads(Path(a.baseline_flight_evaluation).read_text()); landing_reference=json.loads(Path(a.landing_reference_evaluation).read_text())
        retention_gate=max(float(a.retention_minimum),float(landing_reference["final_recovery_rate"])-.05)
        probe_groups={
            "canonical_entry":("landing",[r for r in entry_rehearsal.records if r["final"]["label"]=="safe" and not r.get("training_only",False)]),
            "landing_full_safe":("landing",[r for r in landing_rehearsal.records if r["final"]["label"]=="safe" and not r.get("training_only",False)]),
            "landing_boundary":("landing",[r for r in landing_rehearsal.records if r["final"]["label"]=="boundary" and not r.get("training_only",False)]),
            "flight_late_descent":("flight",current_records),
        }
        block_root=Path(a.bounded_block_dir); block_root.mkdir(parents=True,exist_ok=False)
        previous_chain=float(baseline_flight["chain_rate"]); previous_final=float(baseline_flight["final_recovery_rate"]); stagnant=0
        def source_ratios(prefix):
            candidates=[row for row in rows if any(key.startswith(prefix) for key in row)]
            row=candidates[-1] if candidates else {}; values={name:float(row.get(prefix+name,0.0)) for name in ("flight_curriculum","canonical_entry_rehearsal","landing_tube_rehearsal","natural")}; total=sum(values.values())
            return {key:(value/total if total else 0.0) for key,value in values.items()}
        def block_callback(step,_,params):
            nonlocal previous_chain,previous_final,stagnant
            if step==0: return
            if step not in (25600,51200,76800,102400): raise BoundedStop(f"Unexpected cumulative block step {step}")
            index=step//25600; block=block_root/f"block_{index}_{step:06d}"; block.mkdir()
            version=f"flight-bounded-{step:06d}"; save_bundle(block/"policy",params=params,config=cfg,xml_path=cfg.xml_path,candidate_bank=a.bank,downstream_bank=a.downstream_bank,policy_version=version,extra={"stage":"flight","seed":a.seed,"cumulative_timesteps":step,"reset_protocol":reset_protocol})
            flight_report=evaluate_records(params,cfg.to_dict(),"flight",fixed_flight.records_for_phase("flight",include_training_only=False),downstream,7100000+index*1000)
            landing_report=evaluate_records(params,cfg.to_dict(),"landing",retention_bank.records_for_phase("landing",include_training_only=False),SnapshotBank(),7200000+index*1000)
            probes=drift_probe(params,reference_params,cfg.to_dict(),probe_groups,downstream,7300000+index*10000)
            improved=flight_report["chain_rate"]>previous_chain or flight_report["final_recovery_rate"]>previous_final
            stagnant=0 if improved else stagnant+1; previous_chain=flight_report["chain_rate"]; previous_final=flight_report["final_recovery_rate"]
            episode_ratios=source_ratios("episode/reset/episode/")
            transition_ratios=source_ratios("episode/reset/transition/")
            reasons=[]
            for source in ("flight_curriculum","canonical_entry_rehearsal","landing_tube_rehearsal"):
                if episode_ratios[source]<=0: reasons.append(f"missing episode reset source: {source}")
                if transition_ratios[source]<=0: reasons.append(f"missing transition reset source: {source}")
            if landing_report["final_recovery_rate"]<retention_gate: reasons.append("Landing retention below fixed gate")
            for group in ("landing_full_safe","landing_boundary"):
                if probes[group]["policy_final_rate"]<probes[group]["reference_final_rate"]-.10: reasons.append(f"{group} local Final collapse")
            if flight_report["timeout_rate"]>0 or landing_report["timeout_rate"]>0: reasons.append("timeout present")
            if stagnant>=2: reasons.append("Chain and Final unchanged for two consecutive blocks")
            if step==102400 and flight_report["final_recovery_rate"]<float(baseline_flight["final_recovery_rate"]): reasons.append("final Flight rate below old pilot")
            finite=all(math.isfinite(float(value)) for group in probes.values() for metric in ("kl","action_l2") for value in group[metric].values())
            if not finite: reasons.append("nonfinite phase-conditioned probe")
            decision="STOP" if reasons else ("PASS_COMPLETE" if step==102400 else "PASS_CONTINUE")
            report={"status":decision,"block":index,"cumulative_steps":step,"flight":flight_report,"landing_retention":landing_report,"retention_gate":retention_gate,"probes":probes,"reset_episode_ratio":episode_ratios,"reset_transition_ratio":transition_ratios,"improved":improved,"consecutive_stagnant_blocks":stagnant,"reference_policy_version":reference_manifest["policy_version"],"active_c_l_sha256":file_sha256(a.downstream_bank),"reasons":reasons}
            save_json(block/"report.json",report); block_history.append({"block":index,"step":step,"status":decision,"report":str((block/"report.json").resolve())}); metric_log["bounded_blocks"]=block_history; save_json(metrics_path,metric_log)
            print(f"[bounded] block={index} step={step} status={decision} chain={flight_report['chain_rate']:.4f} final={flight_report['final_recovery_rate']:.4f} retention={landing_report['final_recovery_rate']:.4f}")
            if reasons: raise BoundedStop("; ".join(reasons))
    else:
        block_callback=lambda *_:None
    train_fn=make_ppo_train_fn(timesteps=a.timesteps,episode_length=int(cfg.episode_length),num_envs=a.num_envs,num_eval_envs=a.num_eval_envs,num_evals=a.num_evals,seed=ppo_seed,learning_rate=learning_rate,entropy_cost=entropy_cost,reward_scaling=0.1,checkpoint_dir=run/"orbax",unroll_length=32,batch_size=a.batch_size,num_minibatches=a.num_minibatches,num_updates_per_batch=2,discounting=.995,gae_lambda=.97,clipping_epsilon=.10,max_grad_norm=.75,restore_params=restore_params,policy_params_fn=block_callback,full_reset=multi_source)
    try:
        _,params,final_metrics=train_fn(environment=env,progress_fn=progress,eval_env=eval_env)
    except BoundedStop as exc:
        metric_log.update({"status":"early_stopped","error_type":type(exc).__name__,"error":str(exc),"finished_at":time.time(),"elapsed_seconds":time.time()-started_at}); save_json(metrics_path,metric_log); print(f"[bounded] early stop: {exc}"); raise SystemExit(2)
    except BaseException as exc:
        metric_log.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc),"finished_at":time.time(),"elapsed_seconds":time.time()-started_at})
        save_json(metrics_path,metric_log)
        raise
    metric_log.update({"status":"completed","final_metrics":final_metrics,"finished_at":time.time(),"elapsed_seconds":time.time()-started_at})
    save_json(metrics_path,metric_log)
    version=f"{a.stage}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_bundle(run/"policy",params=params,config=cfg,xml_path=cfg.xml_path,candidate_bank=a.bank,downstream_bank=(a.downstream_bank or None),policy_version=version,extra={"stage":a.stage,"seed":a.seed,"timesteps":a.timesteps,"reset_protocol":reset_protocol,"ppo_hyperparameters":metric_log["ppo_hyperparameters"]})
    print(run/"policy")
if __name__=="__main__": main()
