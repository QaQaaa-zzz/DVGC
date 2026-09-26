import json
import subprocess
import zipfile
import pytest
from jit_dvgc.result_publishing import publish,text_evidence,BRANCH,publish_safely


def test_local_git_publication_isolated_idempotent_and_text_only(tmp_path):
    remote=tmp_path/'remote.git';repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
    subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'remote','add','origin',str(remote)],check=True)
    output=tmp_path/'run';output.mkdir()
    with zipfile.ZipFile(output/'results_to_send.zip','w') as z:
        z.writestr('summary.json','{"status":"completed"}')
        z.writestr('figures/table.csv','x,z\n1,2\n')
        z.writestr('figures/plot.png',b'not an image')
    before=subprocess.check_output(['git','-C',str(repo),'symbolic-ref','HEAD'])
    first=publish(repo,output);second=publish(repo,output)
    assert first['commit']==second['commit']
    assert before==subprocess.check_output(['git','-C',str(repo),'symbolic-ref','HEAD'])
    paths=subprocess.check_output(['git','--git-dir',str(remote),'ls-tree','-r','--name-only',BRANCH],text=True)
    assert 'summary.json' in paths and 'plot.png' not in paths


def test_publish_failure_preserves_bundle(tmp_path):
    archive=tmp_path/'results_to_send.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('summary.json','{}')
    data=archive.read_bytes()
    result=publish_safely(tmp_path,tmp_path)
    assert result['status']=='publication_failed' and archive.read_bytes()==data


def test_path_traversal_rejected(tmp_path):
    archive=tmp_path/'x.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.json','{}')
    with pytest.raises(ValueError):text_evidence(archive)
