"""Desktop error alerts for explicitly selected active JIT runs (CPU only)."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


FAILURES = {"failed", "error", "engineering_error", "timeout", "gate_timeout"}


def read_json(path):
    return json.loads(Path(path).read_text())


def find_failure(manifest_path):
    """Prefer detailed lineage errors; never follow previous failed attempts."""
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    for key in ("lineage", "execution"):
        if key not in manifest:
            continue
        path = Path(manifest[key])
        if not path.is_absolute():
            path = manifest_path.parent / path
        try:
            status = read_json(path)
        except (OSError, ValueError):
            continue  # A status writer may be between writes.
        if status.get("phase", status.get("status")) not in FAILURES:
            continue
        stage = status.get("stage", "")
        error = str(status.get("error", ""))
        for row in reversed(status.get("stages", [])):
            if row.get("phase") in FAILURES:
                stage = row.get("name", stage)
                error = error or str(row.get("error", f"returncode={row.get('returncode')}"))
                break
        return {"path": str(path.resolve()), "stage": stage,
                "error": error, "phase": status.get("phase", status.get("status"))}
    return None


def notify(title, body):
    subprocess.run(["notify-send", "--app-name=JIT", "--urgency=critical",
                    title, body], check=True, capture_output=True, text=True, timeout=10)


def check_once(manifests, state, send=notify):
    seen = state.setdefault("delivered", {})
    state["delivery_errors"] = []
    for manifest in manifests:
        try:
            failure = find_failure(manifest)
            if failure is None:
                continue
            fingerprint = hashlib.sha256(json.dumps(failure, sort_keys=True).encode()).hexdigest()
            if fingerprint in seen:
                continue
            send("JIT 程序报错", f"阶段：{failure['stage']}\n{failure['error'][:1200]}\n状态文件：{failure['path']}")
            seen[fingerprint] = {**failure, "notified_unix": time.time()}
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            state["delivery_errors"].append({"manifest": str(manifest), "error": str(exc)})
    state.update(pid=os.getpid(), checked_unix=time.time(), manifests=list(map(str, manifests)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-run", type=Path, action="append", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "watcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.state_dir / "notification_status.json"
    state = read_json(state_path) if state_path.exists() else {}
    while True:
        check_once(args.active_run, state)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(state_path)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
