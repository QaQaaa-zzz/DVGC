"""Small review packages; authoritative raw experiment files stay on the server."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import zipfile


def bundle(output, *, full=False):
    root = Path(output).resolve()
    if not root.is_dir():
        raise ValueError(f"result directory does not exist: {root}")
    destination = root / ("results_full.zip" if full else "results_to_send.zip")
    records = []
    fd, temporary = tempfile.mkstemp(prefix="bundle_", suffix=".tmp", dir=root)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                    continue
                if path.name in {"bundle_inventory.json", "publish_status.json"} or path.suffix not in {
                    ".json", ".csv", ".md", ".log", ".png", ".pdf", ".svg"
                }:
                    continue
                # Keep all compact analysis tables and JSON metadata. Raw catalogs,
                # snapshot/label payloads and repeated vector graphics are server-only.
                selected = full or path.suffix in {".csv", ".md", ".log"}
                if path.suffix == ".png":
                    selected |= len(relative.parts) <= 2
                if path.suffix == ".json":
                    selected |= len(relative.parts) <= 2 or path.name in {
                        "summary.json", "reservation.json", "receipt.json",
                        "completion.json", "process.json", "backend_decision.json",
                        "exit.json", "execution.json", "worker_failure.json",
                        "failure.json", "error.json", "protocol.json", "merge_audit.json",
                    } or path.name.startswith("process_")
                size = path.stat().st_size
                record = {"path": relative.as_posix(), "source_bytes": size}
                if not selected or size > 20_000_000:
                    record["disposition"] = "server_only_size" if size > 20_000_000 else "server_only_detail"
                    records.append(record)
                    continue
                data = path.read_bytes()
                record["source_sha256"] = hashlib.sha256(data).hexdigest()
                if not full and path.suffix == ".log" and len(data) > 32768:
                    data = b"[Review copy: final 32 KiB; complete log remains on server]\n" + data[-32768:]
                    record["disposition"] = "log_tail"
                else:
                    record["disposition"] = "included"
                archive.writestr(relative.as_posix(), data)
                records.append(record)
            archive.writestr("bundle_inventory.json", json.dumps({
                "schema": "jit_review_bundle_v2", "mode": "full" if full else "compact",
                "raw_files_deleted": False, "standalone_replay_archive": False,
                "files": records,
            }, indent=2))
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if not full:
        from .result_publishing import auto_publish
        auto_publish(root)
    return destination
