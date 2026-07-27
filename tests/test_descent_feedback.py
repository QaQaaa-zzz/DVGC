import numpy as np

from dvgc.descent_feedback import distinct_top_sequences, local_support_gate, select_three_ticks


def test_three_tick_selection_is_fixed_and_caps_overrepresented_accepts():
    audits=[{"tick": tick, "accepted": tick in {1,2,3,4}} for tick in range(8)]
    assert select_three_ticks(audits) == [1,2,3]
    assert select_three_ticks([]) == [1,4,7]


def test_top_sequences_are_physical_ranked_and_action_distinct():
    def row(survival, margin, action):
        return {"survival":survival,"minimum_margin":margin,"terminal_margin":margin,
                "residual_rms":.1,"actions":np.full((8,4),action),"generation":0,"sample":0}
    rows=[row(6,.2,0),row(8,.1,1),row(8,.1,1),row(8,.05,2)]
    chosen=distinct_top_sequences(rows,3)
    assert [x["survival"] for x in chosen] == [8,8,6]
    assert len(chosen) == 3


def test_local_support_gate_requires_both_snapshot_and_candidate_coverage():
    rows=[]
    for candidate in range(8):
        for tick in range(3):
            rows.append({"candidate_id":str(candidate),"authoritative_correction":candidate<6 or tick==0})
    assert local_support_gate(rows)["gate"]
    rows[0]["authoritative_correction"]=False
    rows[1]["authoritative_correction"]=False
    assert not local_support_gate(rows)["gate"]
