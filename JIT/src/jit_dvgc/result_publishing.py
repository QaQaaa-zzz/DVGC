"""Publish compact text evidence on an isolated results branch, never experiment code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

BRANCH='agent/jit-run-reports'


def text_evidence(archive):
    result={}
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            path=Path(item.filename)
            if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe archive path')
            if path.suffix not in {'.json','.csv','.log','.md'} or item.file_size>20_000_000:continue
            result[path.as_posix()]=z.read(item).decode('utf-8',errors='replace')
    return result


def publish(repo, output):
    repo,output=Path(repo).resolve(),Path(output).resolve()
    archive=output/'results_to_send.zip'
    content=text_evidence(archive)
    if not content:raise ValueError('no compact evidence to publish')
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0')
    def git(directory,*args):
        proc=subprocess.run(['git',*args],cwd=directory,env=env,text=True,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
        if proc.returncode:raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()}")
        return proc.stdout.strip()
    remote=git(repo,'remote','get-url','origin')
    # Reuse the server's existing SSH/credential-helper authentication.
    with tempfile.TemporaryDirectory(prefix='jit-report-publish-') as temporary:
        temp=Path(temporary)
        git(temp,'init','--quiet');git(temp,'remote','add','origin',remote)
        exists=git(temp,'ls-remote','--heads','origin',f'refs/heads/{BRANCH}')
        if exists:
            git(temp,'fetch','--quiet','--depth=1','origin',BRANCH)
            git(temp,'checkout','--quiet','-B',BRANCH,'FETCH_HEAD')
        else:git(temp,'checkout','--quiet','--orphan',BRANCH)
        git(temp,'config','user.name','JIT Result Publisher')
        git(temp,'config','user.email','jit-results@users.noreply.github.com')
        run_id=output.name+'-'+hashlib.sha256(str(output).encode()).hexdigest()[:10]
        directory=temp/'reports'/run_id
        # A repeated publication replaces this run's view without deleting history.
        if directory.exists():
            import shutil
            shutil.rmtree(directory)
        for name,value in content.items():
            path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
        manifest={'run_id':run_id,'local_run':str(output),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'files':{name:hashlib.sha256(value.encode()).hexdigest() for name,value in content.items()},
                  'scope':'compact text review; raw snapshots, checkpoints and images stay on server'}
        (directory/'publication.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (temp/'README.md').write_text('# JIT experiment reports\n\nCompact text evidence lives in reports/. '
            'Each directory preserves a run summary, analysis tables, identities and diagnostics. '
            'Code lives on agent/two-phase-soft-tube; raw simulation files remain on the server.\n')
        git(temp,'add','README.md',f'reports/{run_id}')
        if git(temp,'status','--porcelain'):
            git(temp,'commit','--quiet','-m',f'Report {run_id}')
            git(temp,'push','--quiet','origin',f'HEAD:refs/heads/{BRANCH}')
        sha=git(temp,'rev-parse','HEAD')
    return {'status':'published','branch':BRANCH,'commit':sha,'run_id':run_id,
            'url':f'https://github.com/QaQaaa-zzz/DVGC/tree/{BRANCH}/reports/{run_id}'}


def publish_safely(repo,output):
    output=Path(output)
    try:result=publish(repo,output)
    except Exception as exc:result={'status':'publication_failed','error':str(exc),'local_results_preserved':True}
    (output/'publish_status.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[publish] '+(result.get('url') or result['error']),flush=True)
    return result


def auto_publish(output):
    repo=Path(__file__).resolve().parents[3];output=Path(output).resolve()
    # Only final top-level production run bundles; avoid recursive child uploads
    # and network access from tests or arbitrary temporary review directories.
    if not output.is_relative_to(repo/'JIT/runs'):return
    if output.name in {'pi_0','pi_1','pi_2','pi_3'} and (output.parent/'execution.lock').exists():return
    if os.environ.get('JIT_AUTO_PUBLISH','1')=='0':return
    publish_safely(repo,output)
