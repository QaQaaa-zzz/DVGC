"""Exercise CLI result publication with a completed evaluator and no GPU."""
import importlib.util,json,sys
from pathlib import Path
import pytest


def test_baseline_cli_retains_child_timing_without_duplicate_keyword(tmp_path,monkeypatch):
    cli=Path(__file__).resolve().parents[1]/'cli/benchmark_bridge_performance.py'
    spec=importlib.util.spec_from_file_location('perf_cli',cli);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows=tmp_path/'rows.json';rows.write_text(json.dumps([{}]*17))
    config=tmp_path/'spec.json';config.write_text(json.dumps(dict(candidates=str(rows),role='TRAIN',budget=6800)))
    request=tmp_path/'request.json';request.write_text(json.dumps(dict(specs=[str(config)],max_interactions=6800)))
    out=tmp_path/'out'
    def run(argv,check):
        target=Path(argv[-1]);target.mkdir();(target/'status.json').write_text(json.dumps(dict(phase='completed',charged_interactions=17,wall_seconds=2.)))
    import subprocess
    monkeypatch.setattr(subprocess,'run',run)
    monkeypatch.setattr(sys,'argv',[str(cli),'--mode','evaluate','--input',str(request),'--output',str(out)])
    m.main();report=json.loads((out/'benchmark.json').read_text())
    assert report['records'][0]['wall_seconds']==2. and report['charged_interactions']==17
