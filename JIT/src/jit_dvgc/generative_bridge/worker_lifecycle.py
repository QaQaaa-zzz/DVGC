"""Commit isolated last-valid G workers before terminating their process.

A production worker aborted in CPython finalization after all training and
checkpoint writes had returned. This contains that teardown failure; it does
not identify or fix the native library which left interpreter state behind.
Only a normally returned, independently validated and fsynced result can take
this path. Exceptions and failed subprocess exit codes are never converted.
"""
import json
import os
from pathlib import Path
import sys
from .contracts import digest, file_sha
from .protocol import atomic_json


def _read(path):
    return json.loads(Path(path).read_text())


def _sync(paths):
    directories = set()
    for path in paths:
        path = Path(path)
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        directories.add(path.parent)
    for directory in directories:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def commit_generator_result(config):
    from .artifacts import validate_generator_full_state
    import jax
    if config.get('mode') != 'incremental' or config.get('generator_update_policy') != 'last_valid':
        raise ValueError('explicit last-valid incremental process required')
    # Surface outstanding asynchronous errors before committing success.
    jax.effects_barrier()
    root = Path(config['output'])
    result_path = Path(config['result'])
    result = _read(result_path)
    raw = validate_generator_full_state(result)
    incumbent = config['incumbent']
    old = validate_generator_full_state(incumbent)
    from .diffusion import create_train_state, restore_state
    from ..handoff_bank import pytree_sha256
    template = create_train_state(raw['params'], raw['rng'], raw['normalizer'])
    state = restore_state(Path(result['checkpoint_manifest']).parent, template,
                          _read(incumbent['checkpoint_manifest'])['identity'])
    if result.get('selected_state_sha256') != pytree_sha256(state):
        raise ValueError('selected full-state tree identity mismatch')
    before = config['charged_updates_before']
    if type(before) is not int or before < int(old['updates']):
        raise ValueError('invalid lifetime charge origin')
    manifest = Path(result['checkpoint_manifest'])
    if result.get('status') == 'skipped_no_new_data':
        incumbent = config['incumbent']
        if (config['corpus'].get('new_data') is not False or result.get('updates') != 0
                or result.get('checkpoint_manifest_sha256') != incumbent['checkpoint_manifest_sha256']
                or manifest != Path(incumbent['checkpoint_manifest'])
                or result.get('total_charged_updates') != before):
            raise ValueError('invalid unchanged no-data incumbent')
    elif result.get('status') == 'completed':
        if _read(root/'completed.json') != result:
            raise ValueError('worker result differs from completed G receipt')
        if (result.get('charged_updates') != config['updates']
                or result.get('corpus_sha256') != config['corpus']['sha256']
                or result.get('generator_update_policy') != 'last_valid'):
            raise ValueError('worker result differs from declared training')
        attempt = manifest.parent.parent
        if attempt.parent.resolve() != root.resolve():
            raise ValueError('selected checkpoint outside worker output')
        updates = result['updates']
        if (type(updates) is not int or not 0 < updates <= config['updates']
                or result.get('initial_state_updates') != int(old['updates'])
                or result['state_updates'] != int(old['updates']) + updates
                or result.get('charged_updates_before') != before
                or result.get('total_charged_updates') != before + result['charged_updates']):
            raise ValueError('G update or billing lineage mismatch')
        attempts = sorted(root.glob('attempt_*'))
        if not attempts or attempts[-1] != attempt:
            raise ValueError('selected G is not the latest attempt')
        billed = [_read(p/'cost_receipt.json')['charged_updates'] for p in attempts]
        if any(type(n) is not int or n < 0 for n in billed) or sum(billed) != result['charged_updates']:
            raise ValueError('G attempt charge reconciliation failed')
        cost = _read(attempt/'cost_receipt.json')
        selection = _read(attempt/'generator_selection.json')
        if (cost.get('status') != 'completed' or cost.get('charged_updates') != updates
                or selection.get('status') != 'completed' or selection.get('updates') != updates
                or selection.get('selected') != manifest.parent.name
                or manifest.parent.name != f"update_{updates:04d}"):
            raise ValueError('incomplete or mismatched final G selection/cost')
    else:
        raise ValueError('worker has no successful terminal result')
    root.mkdir(parents=True, exist_ok=True)
    paths = [p for p in root.rglob('*') if p.is_file()]
    paths.extend([result_path, manifest, manifest.parent/'state.msgpack'])
    _sync(set(paths))
    try:
        import _xxsubinterpreters
        interpreters = [int(i) for i in _xxsubinterpreters.list_all()]
    except ImportError:
        interpreters = None
    receipt = dict(schema='jit_generator_process_commit_v1',status='committed',
        config_sha256=digest(config),result=str(result_path),result_sha256=file_sha(result_path),
        checkpoint_manifest=str(manifest),checkpoint_manifest_sha256=file_sha(manifest),
        state_updates=result['state_updates'],interpreter_ids=interpreters,
        exit_policy='validated_durable_result_then_os_exit',
        limitation='native finalization contained; underlying library cause not established')
    target = root/'process_commit.json'
    atomic_json(target,receipt)
    _sync([target])
    return receipt


def run_generator_process(config):
    """CLI-only terminal call; never use this in an in-process supervisor."""
    from .worker import run_generator
    run_generator(config)
    commit_generator_result(config)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
