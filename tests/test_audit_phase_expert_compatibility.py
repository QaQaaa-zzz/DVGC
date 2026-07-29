import pytest

from cli.audit_phase_expert_compatibility import compatible, parse_assignment


def _row(shape=(140, 256)):
    return {"policy_network_version": "v", "action_mapping_version": "a",
            "xml_sha256": "x", "actor_history_steps": 4,
            "parameter_signature": [{"shape": list(shape), "dtype": "float32"}]}


def test_compatibility_requires_identical_actor_contracts():
    assert compatible([_row(), _row()])
    assert not compatible([_row(), _row((141, 256))])
    assert not compatible([])


def test_assignment_parser_is_strict():
    assert parse_assignment("apex=feedback-v1") == ("apex", "feedback-v1")
    with pytest.raises(ValueError):
        parse_assignment("missing-separator")
