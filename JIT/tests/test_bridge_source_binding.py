"""The Phase bridge must reject semantic drift before creating any binding."""
from copy import deepcopy

import pytest

from jit_dvgc.config import canonical_sha256
from jit_dvgc.generative_bridge.source_binding import validate_runtime_contract


@pytest.fixture
def contracts():
    phase = dict(action={'joint_target_semantics':'keyframe_centered_absolute'},
                 model={'xml_sha256':'physical-identity'},physical_limits={'max_abs_roll':.6})
    downstream = deepcopy(phase)
    downstream['descent'] = dict(continuous_stability=True,recovery_ticks=25)
    runtime = dict(inputs={'up_config_sha256':canonical_sha256(phase)},
                   reward_mode='original_all_phases',success_criterion='stable_forward_recovery')
    return phase,runtime,downstream


def test_retains_distinct_phase_termination_semantics(contracts):
    source,runtime,down = contracts
    down['physical_limits']['min_post_contact_forward_progress'] = .3
    validate_runtime_contract(source,runtime,down)


@pytest.mark.parametrize('target,key,value,match',[
    (1,'reward_mode','phase_recovery','reward'),
    (1,'success_criterion','first_valid_landing','endpoint'),
    (2,'action',{'joint_target_semantics':'incremental_knee'},'action'),
    (2,'model',{'xml_sha256':'changed-physical-identity'},'XML'),
    (2,'descent',{'continuous_stability':False,'recovery_ticks':25},'continuous'),
    (2,'descent',{'continuous_stability':True,'recovery_ticks':1},'25-tick'),
])
def test_rejects_task_drift(contracts,target,key,value,match):
    contracts[target][key] = value
    with pytest.raises(ValueError,match=match):
        validate_runtime_contract(*contracts)


def test_rejects_upstream_configuration_substitution(contracts):
    contracts[0]['physical_limits']['max_abs_roll'] = 1.2
    with pytest.raises(ValueError,match='upstream configuration'):
        validate_runtime_contract(*contracts)
