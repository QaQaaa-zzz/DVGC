from __future__ import annotations


def test_fresh_shared_validation_config_is_locked(jit_root):
    from jit_dvgc.fresh_shared_continuation_validation import (
        load_fresh_shared_validation_config,
    )

    config = load_fresh_shared_validation_config(
        jit_root / "configs/envelope_iter0_fresh_shared_continuation_validation.json"
    )
    protocol = config["protocol"]
    assert config["expected_protocol_sha256"] == (
        "6a30e6334672c6a404fbaac245d974c358cef9f4f0ea8767d7424b37bb9222ac"
    )
    assert protocol["fresh_seed_families"] == [1000007, 1000008]
    assert 1000006 not in protocol["fresh_seed_families"]
    assert protocol["source_parent_generation"]["upstream_source_transitions"] == [
        4988928, 7987200, 9977856
    ]
    assert protocol["source_parent_generation"]["downstream_source_transitions"] == [4988928]
    assert protocol["interaction_budget"] == {
        "source_parent_rollout_maximum_environment_interactions": 768,
        "attempt_count": 304,
        "maximum_acquisition_environment_interactions": 3168,
        "maximum_labeling_environment_interactions": 121600,
        "training_transitions": 0,
    }
    assert protocol["calibration"]["validation_hyperparameter_search"] is False
    assert protocol["calibration"]["model_parameters_refit_on_validation"] is False
    assert protocol["data_policy"]["consumed_validation_outcomes_read"] is False
    assert protocol["data_policy"]["fresh_validation_rows_may_enter_train_or_tube"] is False
    assert protocol["claim_boundary"]["tube_1_constructed"] is False


def test_fresh_shared_validation_attempt_schedule_is_304(jit_root):
    from jit_dvgc.expansion_validation_runtime import enumerate_validation_attempts
    from jit_dvgc.fresh_shared_continuation_validation import (
        _scientific_protocol,
        load_fresh_shared_validation_config,
    )

    protocol = load_fresh_shared_validation_config(
        jit_root / "configs/envelope_iter0_fresh_shared_continuation_validation.json"
    )["protocol"]
    entries = []
    for transitions in (4988928, 7987200, 9977856):
        for seed in (1000007, 1000008):
            entries.append(
                {
                    "phase": "upstream",
                    "source_bank": f"source_parents/source_{transitions}",
                    "snapshot": f"snapshots/seed_{seed}_ascending_entry",
                    "parent_group_id": f"transition_{transitions}__{seed}",
                    "state_sha256": f"{len(entries)+1:064x}",
                    "role": "ascending_entry",
                    "tick": 32,
                }
            )
    for seed in (1000007, 1000008):
        entries.append(
            {
                "phase": "downstream",
                "source_bank": "source_parents/source_4988928",
                "snapshot": f"snapshots/seed_{seed}_post_apex",
                "parent_group_id": f"transition_4988928__{seed}",
                "state_sha256": f"{len(entries)+1:064x}",
                "role": "post_apex",
                "tick": 57,
            }
        )
    scientific = _scientific_protocol(protocol, source_report={"entries": entries})
    attempts = enumerate_validation_attempts(scientific)
    assert len(attempts) == 304
    assert sum(row["phase"] == "upstream" for row in attempts) == 288
    assert sum(row["phase"] == "downstream" for row in attempts) == 16


def test_fresh_shared_validation_preserves_shared_field_identity(jit_root):
    from jit_dvgc.fresh_shared_continuation_validation import (
        load_fresh_shared_validation_config,
    )

    protocol = load_fresh_shared_validation_config(
        jit_root / "configs/envelope_iter0_fresh_shared_continuation_validation.json"
    )["protocol"]
    assert protocol["shared_refit_summary_sha256"] == (
        "8abbbfbc205e42bb28ee422691eeaef9ab30bdc4c56ad9067a4d55f366a2d9ef"
    )
    assert protocol["architecture_manifest_sha256"] == (
        "9ba92bd51b124d992a82001d260c40dc435e6ed60f68f4dea16485149e2e097e"
    )
    assert protocol["upstream_field_file_sha256"] == (
        "37ea10e9afa6048098af2ef1fd9e7a196feaeeef71e474a8ad7e345c94512f61"
    )
    assert protocol["downstream_field_file_sha256"] == (
        "423f0584f7060c6b56480bd329d787f768a4877ef387f433091d7960e8644cc9"
    )
