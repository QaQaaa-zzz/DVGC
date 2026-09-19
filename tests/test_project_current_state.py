from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_current_documents_define_two_phase_direction_without_implementation_claim():
    project = _read("PROJECT.md")
    method = _read("docs/METHOD_TWO_PHASE_SOFT_TUBE.md")
    state = _read("docs/EXPERIMENT_STATE.md")
    combined = "\n".join((project, method, state))

    for term in (
        "Propulsion-Ascent",
        "Descent-Recovery",
        "Apex transition band",
        "V_up",
        "V_down",
        "soft feasibility",
        "Tube-RSI",
        "Jump Capability Envelope",
    ):
        assert term in combined

    assert "## Not implemented" in project
    assert "No current file may claim these capabilities exist" in project
    assert "This method is not yet implemented" in state
    assert "not an implementation claim" in method


def test_legacy_five_stage_route_is_not_advertised_as_current():
    current = "\n".join(
        _read(path)
        for path in ("PROJECT.md", "README.md", "docs/EXPERIMENT_STATE.md")
    )
    assert "legacy migration source" in current
    assert "bash scripts/run_backward_bootstrap.sh" not in current
    assert "There is currently no formal two-phase pipeline command" in current
