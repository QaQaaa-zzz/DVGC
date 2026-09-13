"""CPU-only supervisor for resumable, shared-panel policy envelope comparisons."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import traceback

from .jump_evidence_validation import read, write, file_sha, verify_hash

DEFAULT_SCAN = "JIT/runs/iteration_auto/pi_3_to_pi_4_pi0_centerline_family_landing_predictor_audit_20260905"
DEFAULT_OUTPUT = "JIT/runs/policy_comparison/expanded_pi0_pi3_20260907"


def verify_plan(path):
    plan = read(path)
    verify_hash(plan, "plan_sha256")
    for source, sha in plan["input_files"].items():
        if file_sha(source) != sha:
            raise ValueError(f"locked comparison input changed: {source}")
    for source, sha in plan["sources"].items():
        if file_sha(Path(plan["repo"]) / source) != sha:
            raise ValueError(f"comparison source changed after lock: {source}")
    return plan


def attempt_cost(report, reservation):
    cost = report.get("environment_interactions")
    if report.get("status") == "completed_shard" and type(cost) is int and 0 <= cost <= reservation:
        return cost, True
    return reservation, False


def ledger(output, plan):
    records, charged, known = [], 0, 0
    for path in sorted(Path(output).glob("jobs/*/shard_*/attempt_*/reservation.json")):
        receipt = read(path)
        if receipt["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("attempt plan identity drift")
        job = plan["jobs"][receipt["job_index"]]
        index = receipt["shard_index"]
        if receipt["maximum_interactions"] != job["shard_maximum_interactions"][index]:
            raise ValueError("attempt budget reservation drift")
        summary = path.parent / "result/summary.json"
        report = read(summary) if summary.exists() else {}
        cost, measured = attempt_cost(report, receipt["maximum_interactions"])
        charged += cost
        known += cost if measured else 0
        records.append({"attempt": str(path.parent), "charged_interactions": cost,
                        "measured": measured, "status": report.get("status", "unknown")})
    return {"attempts": records, "charged_new_label_interactions": charged,
            "known_new_label_interactions": known, "all_attempt_costs_measured": all(r["measured"] for r in records),
            "budget": plan["request"]["interaction_budget"], "end_to_end_training_and_acquisition_total": None}


from .result_bundle import bundle


def run_comparison(repo, output, *, scan_root=None, gpu="0", shard_size=200,
                   interaction_budget=5_000_000, frozen_policies=()):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Process lifetime lock: two terminals cannot reserve/run the same GPU jobs.
    with (output / "execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("comparison already running in this output directory") from None
        return _run(repo, output, scan_root=scan_root, gpu=gpu, shard_size=shard_size,
                    interaction_budget=interaction_budget, frozen_policies=frozen_policies)


def _run(repo, output, *, scan_root, gpu, shard_size, interaction_budget, frozen_policies):
    request = {"repo": str(repo), "output": str(output), "scan_root": str(Path(scan_root or repo / DEFAULT_SCAN).resolve()),
               "shard_size": int(shard_size), "interaction_budget": int(interaction_budget),
               "frozen_policies": [str(Path(p).resolve()) for p in frozen_policies]}
    request_path = output / "request.json"
    if request_path.exists() and read(request_path) != request:
        raise ValueError("existing output belongs to a different request; use a new --output-dir")
    if not request_path.exists():
        write(request_path, request)
    summary = {"status": "running", "training_transitions": 0, "final_test_used": False,
               "additional_replay_verification_requested": False, "completed_roles": []}
    plan = None
    cli = repo / "JIT/cli/compare_policy_envelopes.py"
    def child(kind, destination, *, job=0, shard=0, role="train"):
        destination.mkdir(parents=True, exist_ok=True)
        count = len(list(destination.glob("process_*.log")))
        log_path = destination / f"process_{count:03d}.log"
        command = [sys.executable, str(cli), "--worker", kind, "--output-dir", str(output),
                   "--worker-output", str(destination), "--job-index", str(job), "--shard-index", str(shard), "--role", role]
        env = dict(os.environ, PYTHONPATH=str(repo / "JIT/src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1")
        if kind == "label":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["JAX_PLATFORMS"] = "cuda"
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JAX_PLATFORMS"] = "cpu"
        print(f"[comparison] {kind}: {destination}", flush=True)
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        print(f"[comparison] running; log: {log_path}", flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        write(destination / f"exit_{count:03d}.json", {"returncode": code, "gpu_selector": str(gpu) if kind == "label" else None})
        if code:
            raise RuntimeError(f"{kind} failed; see {log_path}")
    try:
        child("prepare", output / "preflight")
        plan = verify_plan(output / "plan.json")
        summary["plan_sha256"] = plan["plan_sha256"]
        print(f"[comparison] policies={[m['policy']['name'] for m in plan['members']]}; "
              f"new first-attempt ceiling={plan['first_attempt_maximum_interactions']}; no PPO", flush=True)
        for role in ("train", "calibration", "acceptance"):
            for j, job in enumerate(plan["jobs"]):
                if job["role"] != role:
                    continue
                if job["reuse_directory"]:
                    print(f"[comparison] reuse verified {role}/{job['evaluator']}", flush=True)
                    continue
                root = output / "jobs" / job["job_id"]
                for i, maximum in enumerate(job["shard_maximum_interactions"]):
                    shard_root = root / f"shard_{i:03d}"
                    attempts = sorted(shard_root.glob("attempt_*"))
                    completed = [p for p in attempts if (p / "result/summary.json").exists() and
                                 read(p / "result/summary.json").get("status") == "completed_shard"]
                    if completed:
                        # CPU preflight validates existing shards against the exact job contract.
                        continue
                    account = ledger(output, plan)
                    if account["charged_new_label_interactions"] + maximum > interaction_budget:
                        raise RuntimeError("label budget exhausted including failed/unknown attempts")
                    attempt = shard_root / f"attempt_{len(attempts):04d}"
                    write(attempt / "reservation.json", {"plan_sha256": plan["plan_sha256"], "job_index": j,
                          "shard_index": i, "maximum_interactions": maximum})
                    child("label", attempt, job=j, shard=i)
                    write(output / "cost_ledger.json", ledger(output, plan))
                child("merge", root, job=j)
            child("analyze", output / "figures" / role, role=role)
            summary["completed_roles"].append(role)
            write(output / "summary.json", summary)
        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="engineering_error", error=f"{type(exc).__name__}: {exc}")
        failure_dir = output / "failures"
        write(failure_dir / f"failure_{len(list(failure_dir.glob('*.json'))):04d}.json",
              {"error": summary["error"], "traceback": traceback.format_exc()})
    finally:
        if plan is not None:
            try:
                account = ledger(output, plan)
                write(output / "cost_ledger.json", account)
                summary["charged_new_label_interactions"] = account["charged_new_label_interactions"]
            except Exception as exc:
                summary.update(status="engineering_error", cost_ledger_error=str(exc))
        write(output / "summary.json", summary)
        print(f"[comparison] {summary['status']}\nReturn this file: {bundle(output)}", flush=True)
    return summary
