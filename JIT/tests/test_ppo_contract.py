from __future__ import annotations

import jax
from jax import numpy as jp
import pytest
from brax.training.acme import running_statistics

from jit_dvgc.constants import ACTOR_OBSERVATION_SIZE, PRIVILEGED_OBSERVATION_SIZE
from jit_dvgc.ppo import SmokeReport, make_network_factory, validate_smoke_report


def test_network_factory_separates_actor_and_critic_observation_keys():
    factory = make_network_factory()
    ppo_networks = factory(
        {
            "state": (ACTOR_OBSERVATION_SIZE,),
            "privileged_state": (PRIVILEGED_OBSERVATION_SIZE,),
        },
        4,
        preprocess_observations_fn=lambda observation, _: observation,
    )
    params = ppo_networks.policy_network.init(jax.random.PRNGKey(0))
    normalizer = running_statistics.init_state(
        {
            "state": jp.zeros((ACTOR_OBSERVATION_SIZE,)),
            "privileged_state": jp.zeros((PRIVILEGED_OBSERVATION_SIZE,)),
        }
    )
    action_distribution = ppo_networks.policy_network.apply(
        normalizer,
        params,
        {"state": jp.zeros((ACTOR_OBSERVATION_SIZE,))},
    )
    assert action_distribution.shape == (8,)

    value_params = ppo_networks.value_network.init(jax.random.PRNGKey(1))
    value = ppo_networks.value_network.apply(
        normalizer,
        value_params,
        {"privileged_state": jp.zeros((PRIVILEGED_OBSERVATION_SIZE,))},
    )
    assert value.shape == ()


def test_network_factory_does_not_accept_privileged_state_as_actor_input():
    factory = make_network_factory()
    ppo_networks = factory(
        {
            "state": (ACTOR_OBSERVATION_SIZE,),
            "privileged_state": (PRIVILEGED_OBSERVATION_SIZE,),
        },
        4,
        preprocess_observations_fn=lambda observation, _: observation,
    )
    params = ppo_networks.policy_network.init(jax.random.PRNGKey(0))
    normalizer = running_statistics.init_state(
        {
            "state": jp.zeros((ACTOR_OBSERVATION_SIZE,)),
            "privileged_state": jp.zeros((PRIVILEGED_OBSERVATION_SIZE,)),
        }
    )
    with pytest.raises(KeyError):
        ppo_networks.policy_network.apply(
            normalizer,
            params,
            {"privileged_state": jp.zeros((PRIVILEGED_OBSERVATION_SIZE,))},
        )


def test_interaction_accounting_closes_one_exact_block():
    report = validate_smoke_report(
        SmokeReport(
            requested_training_transitions=25_600,
            completed_training_transitions=25_600,
            brax_evaluation_transitions=0,
            fixed_evaluation_transitions=0,
            diagnostic_transitions=0,
            final_metrics={"training/sps": 1.0},
            checkpoint_restored=True,
        )
    )
    assert report.total_environment_transitions == 25_600


@pytest.mark.parametrize("completed", [0, 25_599, 25_601])
def test_smoke_report_rejects_inexact_training_count(completed):
    with pytest.raises(ValueError, match="exactly"):
        validate_smoke_report(
            SmokeReport(
                requested_training_transitions=25_600,
                completed_training_transitions=completed,
                brax_evaluation_transitions=0,
                fixed_evaluation_transitions=0,
                diagnostic_transitions=0,
                final_metrics={},
                checkpoint_restored=True,
            )
        )


def test_smoke_report_requires_checkpoint_restore():
    with pytest.raises(ValueError, match="restore"):
        validate_smoke_report(
            SmokeReport(
                requested_training_transitions=25_600,
                completed_training_transitions=25_600,
                brax_evaluation_transitions=0,
                fixed_evaluation_transitions=0,
                diagnostic_transitions=0,
                final_metrics={},
                checkpoint_restored=False,
            )
        )
