"""A native G-worker crash must leave a Python stack, not just XLA warnings."""
import os
from pathlib import Path
import signal
import subprocess
import sys


def test_worker_cli_enables_native_fault_stack_before_running_worker(tmp_path):
    spec=tmp_path/'worker.json';spec.write_text('{"mode":"incremental"}')
    cli=Path(__file__).resolve().parents[1]/'cli/run_generative_bridge.py'
    code='''
import os,resource,runpy,signal,sys
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
from jit_dvgc.generative_bridge import worker
worker.run_generator=lambda spec:os.kill(os.getpid(),signal.SIGSEGV)
sys.argv=[sys.argv[1],'worker','--spec',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    env={**os.environ,'JAX_PLATFORMS':'cpu'};env.pop('PYTHONFAULTHANDLER',None)
    result=subprocess.run([sys.executable,'-c',code,str(cli),str(spec)],env=env,capture_output=True,text=True)
    assert result.returncode==-signal.SIGSEGV
    assert 'Fatal Python error: Segmentation fault' in result.stderr
    assert 'run_generative_bridge.py' in result.stderr
