"""Physical storage aliases must not weaken frozen checkpoint byte identity."""
from pathlib import Path

import pytest

from jit_dvgc import unified_policy_freeze as frozen
from .test_unified_policy_freeze import _development_source


def relocated(jit_root,tmp_path):
    original=tmp_path/'original';original.mkdir()
    config,_,run,checkpoint=_development_source(jit_root,original)
    manifest=frozen.freeze_development_checkpoint(tmp_path/'frozen',config_path=config,
        checkpoint=checkpoint,name='saved_student')
    path=tmp_path/'frozen/frozen_unified_policy.json'
    before=path.read_bytes()
    target=tmp_path/'relocated'
    original.rename(target)
    original.symlink_to(target,target_is_directory=True)
    return manifest,path,before


def test_symlink_storage_migration_accepts_same_files_without_rewriting_manifest(jit_root,tmp_path):
    manifest,path,before=relocated(jit_root,tmp_path)
    loaded=frozen.load_frozen_unified_manifest(path)
    assert loaded==manifest and path.read_bytes()==before
    inspected=frozen.verify_frozen_unified_record(loaded['policy'])
    for key in ('checkpoint','formal_config','source_formal_report'):
        assert Path(loaded['policy'][key]).resolve()==Path(getattr(inspected,key)).resolve()


@pytest.mark.parametrize('field',['formal_config','source_formal_report','checkpoint_identity','payload'])
def test_same_symlink_path_still_rejects_changed_bytes(jit_root,tmp_path,field):
    manifest,path,before=relocated(jit_root,tmp_path)
    policy=manifest['policy']
    target=(Path(policy['checkpoint'])/('identity.json' if field=='checkpoint_identity' else 'payload.pkl')
        if field in ('checkpoint_identity','payload') else Path(policy[field]))
    target.write_bytes(target.read_bytes()+b'\n')
    with pytest.raises(ValueError):frozen.load_frozen_unified_manifest(path)
    assert path.read_bytes()==before


@pytest.mark.parametrize('field',['actor_sha256','formal_config_file_sha256','source_formal_report_sha256',
                                 'source_training_transitions','unexpected_field'])
def test_path_equivalence_does_not_relax_nonpath_values_or_extra_fields(jit_root,tmp_path,field):
    manifest,_,_=relocated(jit_root,tmp_path)
    record=dict(manifest['policy'])
    record[field]=None if field=='unexpected_field' else 'changed'
    with pytest.raises(ValueError,match='record drift'):
        frozen.verify_frozen_unified_record(record)
