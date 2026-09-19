import json
import zipfile

from jit_dvgc.result_bundle import bundle


def test_failure_diagnostics_and_tables_survive_without_raw_payloads(tmp_path):
    files = {
        "summary.json": '{"status":"engineering_error"}',
        "pi_1/tasks/label/attempt_0000/process_000.json": '{"exit_code":1}',
        "pi_1/tasks/label/attempt_0000/process_000.log": "x" * 40000 + "Traceback: OOM",
        "pi_1/tasks/label/attempt_0000/result/labels.json": "raw labels",
        "pi_1/figures/detail.png": "duplicate plot",
        "figures/boundary_candidates.csv": "state,status\n1,no_success_witness\n",
        "figures/boundary_states.png": "overview",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with zipfile.ZipFile(bundle(tmp_path)) as archive:
        assert "figures/boundary_states.png" in archive.namelist()
        assert "pi_1/figures/detail.png" not in archive.namelist()
        assert not any(name.endswith("labels.json") for name in archive.namelist())
        assert archive.read("pi_1/tasks/label/attempt_0000/process_000.log").endswith(b"Traceback: OOM")
        assert archive.read("pi_1/tasks/label/attempt_0000/process_000.json") == b'{"exit_code":1}'
        inventory = json.loads(archive.read("bundle_inventory.json"))
        assert any(row["disposition"] == "log_tail" for row in inventory["files"])
    for name, content in files.items():
        assert (tmp_path / name).read_text() == content
