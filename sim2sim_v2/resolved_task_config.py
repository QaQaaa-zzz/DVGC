"""Single effective task configuration; archived dependencies are read-only."""
from __future__ import annotations
import copy
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
FROZEN_SOURCE = ROOT.parent / 'docs/evidence/sim2sim_ppo_20261009/models/mjx'
SECTIONS = ('reward', 'reset', 'events', 'action', 'physical_limits', 'ppo')

def frozen_imports():
    expected = FROZEN_SOURCE / 'jit_dvgc'
    if not expected.is_dir():
        raise RuntimeError(f'Frozen jit_dvgc dependency missing: {expected}')
    loaded = sys.modules.get('jit_dvgc')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(expected):
        raise RuntimeError(f'Unfrozen jit_dvgc already imported: {loaded.__file__}')
    sys.path.insert(0, str(FROZEN_SOURCE)) if str(FROZEN_SOURCE) not in sys.path else None
    return importlib.import_module('jit_dvgc.config')

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

class ResolvedTaskConfig:
    """legacy rejects conflicting declarations; v2 requires a full explicit payload."""
    def __init__(self, spec, *, mode=None, source_config=None):
        config_module = frozen_imports()
        self.mode = mode or spec.get('task_config_mode', 'legacy_replay')
        if self.mode not in ('legacy_replay', 'jump_landing_v2'):
            raise ValueError(f'Unknown task config mode: {self.mode}')
        self.source_path = Path(source_config or ROOT / 'policy/resolved_config.json').resolve()
        if self.mode == 'legacy_replay':
            raw = json.loads(self.source_path.read_text())
            if 'resolved_task_config' in spec and spec['resolved_task_config'] != raw:
                raise ValueError('legacy_replay explicit payload differs from frozen source')
        else:
            if 'resolved_task_config' not in spec:
                raise ValueError('jump_landing_v2 requires explicit resolved_task_config payload')
            raw = copy.deepcopy(spec['resolved_task_config'])
        for key in SECTIONS:
            if key in spec and spec[key] != raw.get(key):
                raise ValueError(f'Conflicting duplicate task configuration: {key}')
        if raw['action'].get('joint_target_semantics') != 'keyframe_centered_absolute':
            raise ValueError('Unsupported action semantics; requires independent action ablation')
        self.raw = copy.deepcopy(raw)
        self.cfg = config_module.resolve_config_payload(self.raw, runtime_only=True)
        physics_dt = float(spec['physics_dt']); control_dt = float(spec['control_dt'])
        if physics_dt <= 0 or abs(control_dt / physics_dt - round(control_dt / physics_dt)) > 1e-9:
            raise ValueError('control_dt must be an integer multiple of positive physics_dt')
        # Frozen reward energy integration and event deadlines use this exact tick.
        if control_dt != .02:
            raise ValueError('Frozen reward/events require control_dt=0.02')
        observation = copy.deepcopy(spec.get('observation_protocol', {'schema':'legacy_76_v1', 'history_frames':3, 'frame_size':25, 'gravity_z':-9.81, 'obstacle_front_x':3.6}))
        if observation['history_frames'] != 3 or observation['frame_size'] != 25 or observation['schema'] != 'legacy_76_v1':
            raise ValueError('Observation input expansion needs an independent ablation')
        self.observation = observation
        self.effective = {'schema':'resolved_task_config_v1','mode':self.mode,'source_task':self.raw,
            'physics_dt':physics_dt,'control_dt':control_dt,'substeps':round(control_dt/physics_dt),
            'target_drive_gains':copy.deepcopy(spec['target_drive_gains']),
            'contact_required':copy.deepcopy(spec['contact_required']), 'observation_protocol':observation,
            'source_ppo_port':bool(spec.get('source_ppo_port',False))}
        self.sha256 = canonical_hash(self.effective)
    def write(self, output):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        versions={}
        for package in ('mujoco','jax','jaxlib','brax','numpy','scipy','isaacsim'):
            try: versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package]=None
        deps={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.startswith('jit_dvgc') and getattr(module,'__file__',None)}
        identity={'effective_sha256':self.sha256,'source_path':str(self.source_path),'source_file_sha256':hashlib.sha256(self.source_path.read_bytes()).hexdigest() if self.source_path.exists() else None,'versions':versions,'imports':deps,'frozen_source':str(FROZEN_SOURCE)}
        identity['runtime_files_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'resolved_task_config.py',ROOT/'policy_runtime.py',ROOT/'physx_task.py',ROOT/'model/source.xml']}
        (output/'resolved_config.json').write_text(json.dumps(self.effective,indent=2)+'\n')
        (output/'config_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        return identity
