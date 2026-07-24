from cli.audit_takeoff_tail_cross_engine import _event, _spread


def test_contact_event_requires_observed_contact_then_airborne():
    trace = [
        {"tick": 1, "contact_count": 2},
        {"tick": 2, "contact_count": 1},
        {"tick": 3, "contact_count": 0},
    ]
    assert _event(trace) == (2, 3)
    assert _event([{"tick": 1, "contact_count": 0}]) == (None, None)


def test_cross_engine_gate_is_strict_and_blocks_discovery():
    text = open("cli/audit_takeoff_tail_cross_engine.py").read()
    assert '"authority_or_discovery_allowed": bool(gate_pass)' in text
    assert '"ppo_authorization": False' in text
    assert "takeoff_tail_cross_engine_mismatch" in text


def test_controller_turns_cross_engine_failure_into_research_pause():
    text = open("cli/stage_next_v3_controller.py").read()
    assert 'current_stage="takeoff_tail_cross_engine_gate"' in text
    assert 'terminal_state="gate_pause"' in text
    assert "research_gate_valid=True" in text
    assert "raise SystemExit(40)" in text
    assert "resolve_runtime_cross_engine_contact_semantics_before_discovery" in text
