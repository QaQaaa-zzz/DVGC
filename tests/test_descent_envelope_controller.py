import json

from cli.descent_envelope_controller import DescentEnvelopeController, planned_index_seeds
from dvgc.seed_registry import exact_intersection_proof, make_claim


def test_stable_stage_and_audit_seed_sets_are_exact_and_disjoint():
    stage_a=make_claim("a","stage_a",planned_index_seeds(800_000_000,[0,2,7],32))
    stage_b=make_claim("b","stage_b",planned_index_seeds(900_000_000,[0,7],32))
    audit=make_claim("audit","audit",planned_index_seeds(1_500_000_000,list(range(10)),32))
    assert exact_intersection_proof(stage_b,[stage_a])["intersection_count"]==0
    assert exact_intersection_proof(audit,[stage_a,stage_b])["intersection_count"]==0
    assert len(stage_a["seeds"])==96


def test_new_controller_resumes_persisted_stage_without_reset(tmp_path):
    first=DescentEnvelopeController(tmp_path)
    first.save(current_stage="stable_stage_b",current_cycle=1,acquisition_round=1)
    first.lock.close()
    second=DescentEnvelopeController(tmp_path)
    assert second.state["current_stage"]=="stable_stage_b"
    assert second.state["current_cycle"]==second.state["acquisition_round"]==1
    payload=json.loads((tmp_path/"controller_state.json").read_text())
    assert payload["controller_type"]=="descent_envelope"
    second.lock.close()


def test_controller_has_no_round3_training_transition():
    source=__import__("pathlib").Path("cli/descent_envelope_controller.py").read_text()
    assert "round_3/train/orbax/000000076800" in source
    assert "run_next_block" not in source
