import numpy as np
import pytest


def spec(length=3, count=128):
    return dict(pulse_contract_version='scheduled_single_pulse_v1',
                pulse_steps=length, pulse_start_schedule=[0, 5, 10, 15],
                pulse_batch_mode='mixed', single_pulse_per_episode=True,
                delta_limit=[.25]*4, horizon=400, num_envs=count, round_index=0)


@pytest.mark.parametrize('length', [1, 3, 5, 8])
def test_scheduled_windows_draws_and_budget(length):
    import jax
    from jit_dvgc.generative_bridge.pulse_protocol import normalize_pulse_contract, requested_draws, pulse_delta_for_step
    from jit_dvgc.pulse_schedule import lane_onsets, collection_budget, pulse_activity
    from jit_dvgc.pulse_exploration import budget_contract
    s = spec(length)
    contract = normalize_pulse_contract(s)
    assert contract['pulse_steps'] == length
    delays = lane_onsets(s, 0)
    assert [(delays == onset).sum() for onset in s['pulse_start_schedule']] == [32]*4
    budget = collection_budget(s, 0)
    assert budget['charged_maximum'] == 128*(15+length)
    assert budget['active_prefix_maximum'] == 32*sum(t+length for t in [0, 5, 10, 15])
    assert budget_contract(dict(s, rounds=1, policy_steps=128000), 1)['prefixes'] == budget['charged_maximum']
    draws = requested_draws(jax.random.PRNGKey(0), amplitude=.25, pulse_steps=length)
    assert draws.shape == (length, 4) and np.abs(draws).max() <= .25
    selected = pulse_delta_for_step(np.broadcast_to(draws, (4, length, 4)),
                                   np.array([0, length-1, length, length+1]), np.ones(4, bool))
    np.testing.assert_array_equal(selected[2:], 0)
    alive = np.ones(128, bool); counts = np.zeros(128, int); triggers = np.full(128, -1)
    endpoint = np.zeros(128, int)
    for t in range(15+length):
        triggers, mask = pulse_activity(triggers, counts, t >= delays, alive, t, length)
        np.testing.assert_array_equal(mask, alive & (t >= delays) & (t < delays+length))
        endpoint[alive] = t+1
        counts += np.asarray(mask, int)
        alive &= counts < length
    np.testing.assert_array_equal(endpoint, delays+length)
    np.testing.assert_array_equal(counts, length)


@pytest.mark.parametrize('changes', [dict(pulse_steps=0), dict(pulse_steps=True),
    dict(pulse_steps=1.5), dict(pulse_steps=385), dict(pulse_start_schedule=[-1]),
    dict(pulse_start_schedule=[0, 0]), dict(single_pulse_per_episode=False),
    dict(delta_limit=[.3]*4), dict(maximum_collection_steps=17), dict(pulse_contract_version='typo')])
def test_new_contract_rejects_invalid_duration_schedule_bounds_and_horizon(changes):
    from jit_dvgc.generative_bridge.pulse_protocol import normalize_pulse_contract
    with pytest.raises(ValueError):
        normalize_pulse_contract({**spec(), **changes})


def test_legacy_rng_is_unchanged_and_remainder_rotates():
    import jax
    from jit_dvgc.generative_bridge.pulse_protocol import requested_draws, normalize_pulse_contract
    from jit_dvgc.pulse_schedule import lane_onsets
    key = jax.random.PRNGKey(42)
    np.testing.assert_array_equal(requested_draws(key), jax.random.uniform(key, (3, 4), minval=-1., maxval=1.))
    assert normalize_pulse_contract({}) is None
    counts = np.zeros(4, int)
    for batch in range(4):
        delays = lane_onsets(spec(count=5), batch)
        counts += [(delays == onset).sum() for onset in [0, 5, 10, 15]]
    np.testing.assert_array_equal(counts, [5]*4)


def test_nonexecuted_learned_samples_have_no_raw_action_or_log_probability():
    import jax.numpy as jp
    from jit_dvgc.generative_bridge.explorer_admission import mix_sample
    raw, delta, lp, value, valid = mix_sample(jp.ones((2, 4)), jp.ones((2, 4)),
        jp.ones(2), jp.ones(2), jp.ones(2, bool), jp.zeros((2, 4)), jp.array([True, False]))
    assert np.isfinite(raw[0]).all() and np.isnan(raw[1]).all()
    assert np.isfinite(lp[0]) and np.isnan(lp[1])
    assert not valid[1]


def test_prefix_outcomes_distinguish_pre_pulse_failure_and_incomplete_pulse():
    from jit_dvgc.pulse_schedule import pulse_outcome
    assert pulse_outcome(0, 3, True) == 'pre_pulse_terminal'
    assert pulse_outcome(2, 3, True) == 'during_pulse_terminal'
    assert pulse_outcome(3, 3, True) == 'during_pulse_terminal'
    assert pulse_outcome(3, 3, False) == 'valid_post_pulse'
    with pytest.raises(ValueError): pulse_outcome(2, 3, False)


def scheduled_admission_fixture(length=8):
    from jit_dvgc.generative_bridge import explorer_admission as a
    from jit_dvgc.pulse_schedule import lane_onsets
    s = spec(length, 4)
    s.update(controller_mode='learned_residual', explorer_backend='rsl_rl',
        explorer_initialization=dict(mode='symmetric', latent_std=.6),
        explorer_admission_v1_2=dict(run_id='new', round=0, collection_id='c',
            master_seed=3, episode_ids=list(range(4)), uniform_episode_fraction=0.))
    state = dict(params={'actor': {'weight': np.ones(1)}}, normalizer_mean=np.zeros(1), normalizer_std=np.ones(1))
    identity = a.behavior_identity(state)
    s['explorer_admission_v1_2'].update(identity)
    receipt = a.collection_receipt(s, state, np.ones(4, bool))
    t = np.arange(15+length)[:, None]; onsets = lane_onsets(s, 0)
    # Lane 1 terminates before onset; lane 2 after its first pulse action.
    last = np.array([length-1, 2, 10, 15+length-1])
    prefix = t <= last
    mask = prefix & (t >= onsets) & (t < onsets+length)
    terminal = (t == last) & np.array([False, True, True, False])
    requested = np.broadcast_to(np.where(mask[..., None], .1, 0.), mask.shape+(4,)).copy()
    tape = dict(mask=mask, prefix_mask=prefix, terminal=terminal,
        explorer_learned=np.ones_like(mask), on_policy_mask=mask.copy(), log_prob_valid=mask.copy(),
        log_prob=np.where(mask, -.2, np.nan), raw_action=np.where(mask[..., None], np.ones(mask.shape+(4,)), np.nan),
        requested_delta=requested, base_action=np.zeros_like(requested), action=requested.copy(),
        effective_delta=requested.copy(), action_clipped=np.zeros_like(requested, bool))
    return s, receipt, tape, identity


def test_scheduled_admission_binds_timing_and_rejects_phantom_actions():
    from copy import deepcopy
    from jit_dvgc.generative_bridge import explorer_admission as a
    s, receipt, tape, identity = scheduled_admission_fixture()
    assert receipt['pulse_contract']['pulse_steps'] == 8
    assert receipt['lane_onsets'] == [0, 5, 10, 15]
    np.testing.assert_array_equal(a.validate_admission(s, receipt, tape, identity), tape['mask'])
    bad = deepcopy(receipt); bad['pulse_contract']['pulse_steps'] = 3
    with pytest.raises(ValueError, match='pulse'):
        a.validate_admission(s, bad, tape, identity)
    for tick, lane in [(0, 1), (3, 1), (22, 0)]:
        bad = deepcopy(tape)
        for name in ['mask', 'on_policy_mask', 'log_prob_valid']: bad[name][tick, lane] = True
        bad['log_prob'][tick, lane] = -.2; bad['raw_action'][tick, lane] = .1
        with pytest.raises(ValueError): a.validate_admission(s, receipt, bad, identity)
    bad = deepcopy(tape); bad['requested_delta'][0, 0, 0] = .251
    with pytest.raises(ValueError): a.validate_admission(s, receipt, bad, identity)


def test_frozen_padding_keeps_each_lanes_own_full_state():
    import jax.numpy as jp
    from jit_dvgc.pulse_exploration_runtime import freeze_inactive_worlds
    previous = {'qpos': jp.array([[3.], [8.], [13.], [17.]]),
                'history': jp.arange(24).reshape(4, 2, 3), 'clock': jp.array([3, 8, 13, 17])}
    following = {key: value+1 for key, value in previous.items()}
    actual = freeze_inactive_worlds(following, previous, jp.array([False, False, False, True]))
    for key in previous:
        np.testing.assert_array_equal(actual[key][:3], previous[key][:3])
        np.testing.assert_array_equal(actual[key][3], following[key][3])


@pytest.mark.parametrize('length', [1, 3, 5, 8])
def test_scheduled_real_ppo_uses_only_executed_actions_and_one_episode_reward(length):
    import jax.numpy as jp
    from jit_dvgc.generative_bridge import explorer_admission as a
    from jit_dvgc.rsl_pulse import initialize, infer, update_batch
    s, _, tape, _ = scheduled_admission_fixture(length)
    s.update(seed=2, learning_rate=.001, epochs=1, minibatch_size=16, clip=.2,
        target_kl=.01, entropy_coefficient=.001, value_coefficient=.5, max_grad_norm=1.)
    state = initialize(s, np.zeros(106), np.ones(106))
    identity = a.behavior_identity(state)
    s['explorer_admission_v1_2'].update(identity)
    receipt = a.collection_receipt(s, state, np.ones(4, bool))
    s.update(_validated_explorer_admission=receipt, _validated_behavior_identity=identity)
    shape = tape['mask'].shape
    obs = np.random.default_rng(42).normal(size=shape+(106,)).astype(np.float32)
    mu, sd, value = map(np.asarray, infer(state, jp.asarray(obs)))
    raw = mu+sd*.2
    lp = (-.5*((raw-mu)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)).sum(-1)
    raw[~tape['mask']] = np.nan; lp[~tape['mask']] = np.nan
    tape.update(observation=obs, raw_action=raw, log_prob=lp, value=value)
    feedback = dict(eligible=[True, False, True, True], rewards=[1., 0., -.5, .75], component_sums={})
    _, metrics, learning, _ = update_batch(s, state, tape, feedback)
    assert metrics['effective_training_samples'] == 2*length+1
    assert metrics['eligible_episodes'] == 3
    np.testing.assert_allclose(learning['reward'].sum(axis=0), feedback['rewards'])
    assert metrics['optimizer_updates'] > 0


def test_scheduled_random_receipt_binds_length_and_preserves_legacy_rejection():
    from jit_dvgc.generative_bridge.pulse_protocol import collection_draws
    s=spec(8, 4)
    s.update(controller_mode='fixed_random', pulse_protocol_v1_2=dict(
        role='train', master_seed=7, round=0, episode_ids=[0, 1, 2, 3]))
    draws, receipt=collection_draws(s)
    assert draws.shape==(4, 8, 4) and receipt['draw_shape']==[8, 4]
    assert receipt['pulse_contract']['pulse_contract_version']=='scheduled_single_pulse_v1'
    del s['pulse_contract_version']
    with pytest.raises(ValueError, match='three-step'):collection_draws(s)


def test_scheduled_learned_collection_requires_fresh_learned_only_admission():
    from jit_dvgc.generative_bridge.explorer_admission import validate_collection_options
    s, _, _, _ = scheduled_admission_fixture()
    s['explorer_admission_v1_2']['uniform_episode_fraction'] = .2
    with pytest.raises(ValueError, match='learned-only'):validate_collection_options(s)
    del s['explorer_admission_v1_2']
    with pytest.raises(ValueError, match='admission'):validate_collection_options(s)


def test_scheduled_admission_checks_actual_clipping_receipt():
    from jit_dvgc.generative_bridge.explorer_admission import validate_admission
    s, receipt, tape, identity = scheduled_admission_fixture()
    tape['base_action'][0, 0, 0] = .98
    tape['action'][0, 0, 0] = 1.
    tape['effective_delta'][0, 0, 0] = .02
    tape['action_clipped'][0, 0, 0] = True
    validate_admission(s, receipt, tape, identity)
    tape['action_clipped'][0, 0, 0] = False
    with pytest.raises(ValueError, match='clipping'):validate_admission(s, receipt, tape, identity)


def test_scheduled_summary_keeps_all_episode_outcomes_and_actual_amplitudes():
    from jit_dvgc.pulse_exploration_runtime import scheduled_pulse_summary
    s, _, tape, _ = scheduled_admission_fixture(3)
    rows=[dict(pulse_outcome=outcome, snapshot_control_step=t) for outcome,t in zip(
        ['valid_post_pulse','pre_pulse_terminal','during_pulse_terminal','valid_post_pulse'], [3,3,11,18])]
    summary=scheduled_pulse_summary(s,tape,rows)
    assert summary['episode_performance_denominator']==4
    assert summary['pulse_outcome_counts']==dict(valid_post_pulse=2,pre_pulse_terminal=1,during_pulse_terminal=1)
    assert summary['pulse_onset_summary']['5']['applied_actions']==0
    assert summary['pulse_onset_summary']['0']['snapshot_control_steps']==[3]
    assert summary['pulse_onset_summary']['15']['requested_absolute_max']==pytest.approx(.1)
    assert scheduled_pulse_summary({}, {}, [])=={}
