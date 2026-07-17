import numpy as np

from cli.build_descent_support_repair import choose_parent, parent_key, proposal_group


def _row(identifier, source, label):
    return {"id":identifier,"entry_source_id":source,"final":{"label":label}}


def test_support_groups_prioritize_new_real_parents():
    assert proposal_group(_row("a","new","boundary"),{"safe"})=="new_parent"
    assert proposal_group(_row("b","safe","boundary"),{"safe"})=="boundary"
    assert proposal_group(_row("c","safe","safe"),{"safe"})=="safe_continuity"


def test_parent_cap_is_applied_to_real_source_parent():
    rows=[_row("a","p0","boundary"),_row("b","p0","boundary"),_row("c","p1","boundary")]
    source_children={"p0":4,"p1":0};row_children={"a":0,"b":0,"c":0}
    parent=choose_parent(rows,source_children,row_children,4,np.random.default_rng(1))
    assert parent_key(parent)=="p1"
    source_children["p1"]=4
    assert choose_parent(rows,source_children,row_children,4,np.random.default_rng(1)) is None
