import json
import os
from pathlib import Path
import numpy as np
import pytest
from .test_generative_bridge_contracts import trace
from jit_dvgc.generative_bridge import artifacts
from jit_dvgc.generative_bridge.feedback_data import build_corpus


def corpus():
    return build_corpus([], [], [trace()],
        adoption={'adopted': True, 'actor_sha256': 'actor', 'normalizer_sha256': 'norm'},
        expected={}, splits={'parent': 'generator_train'})


def test_history_references_verified_original_blob_without_recompression(tmp_path):
    first = artifacts.save_corpus(corpus(), tmp_path / 'first')
    raw_first = json.loads(Path(first['path']).read_text())
    original = raw_first['groups']['actor_new'][0]
    original_bytes = Path(original['path']).read_bytes()
    loaded = artifacts.load_corpus(first)
    history = build_corpus(loaded['groups']['actor_new'], [], [], adoption={'adopted': False},
                           expected={}, splits={'parent': 'generator_train'})
    second = artifacts.save_corpus(history, tmp_path / 'second')
    raw_second = json.loads(Path(second['path']).read_text())
    reused = raw_second['groups']['history'][0]
    assert reused['path'] == original['path']
    assert reused['sha256'] == original['sha256']
    assert not list((tmp_path / 'second').rglob('*.npz'))
    assert Path(original['path']).read_bytes() == original_bytes
    reread = artifacts.load_corpus(second)['groups']['history'][0]
    assert reread['adoption'] == loaded['groups']['actor_new'][0]['adoption']
    for name, values in loaded['groups']['actor_new'][0]['arrays'].items():
        np.testing.assert_array_equal(reread['arrays'][name], values)
        assert not reread['arrays'][name].flags.writeable


def test_reuse_rejects_tampering_even_after_original_mtime_restored(tmp_path):
    receipt = artifacts.save_corpus(corpus(), tmp_path / 'first')
    loaded = artifacts.load_corpus(receipt)
    raw = json.loads(Path(receipt['path']).read_text())
    path = Path(raw['groups']['actor_new'][0]['path'])
    before = path.stat()
    path.chmod(0o600)
    with path.open('r+b') as stream:
        stream.seek(40); b = stream.read(1); stream.seek(40); stream.write(bytes([b[0] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='hash'):
        artifacts.save_corpus(loaded, tmp_path / 'second')


def test_new_blob_is_content_named_and_readonly(tmp_path):
    receipt = artifacts.save_corpus(corpus(), tmp_path / 'first')
    record = json.loads(Path(receipt['path']).read_text())['groups']['actor_new'][0]
    assert Path(record['path']).stem == record['trajectory_sha256']
    assert Path(record['path']).stat().st_mode & 0o222 == 0
