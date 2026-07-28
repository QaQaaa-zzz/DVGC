import pytest

from dvgc.backward_tube import (
    BackwardTubeNode,
    balanced_p1_launch_gate,
    balanced_p1_launch_subset,
)


def _node(index, candidate, layer=1, region="late", parent="root"):
    return BackwardTubeNode(
        node_id=f"n{index:02d}", phase="descent", layer=layer, region=region,
        candidate_id=candidate, source_state_hash=f"s{index}", physical_state={},
        actor_observation=[], parent_node_id=parent, parent_tube="canonical_C_L",
        controller_type="test", controller_artifact_sha256="controller",
        entry_tick=1, downstream_entry_state={}, final_recovery=True, p0=True, p1=True,
    )


def test_balanced_subset_keeps_non_dominant_and_is_deterministic():
    nodes = [_node(i, "main", 3 if i == 6 else 1, "middle" if i == 6 else "late")
             for i in range(7)]
    nodes += [_node(10 + i, f"other{i // 2}", 4 if i == 0 else 2,
                    "early" if i == 0 else "middle", f"p{i}") for i in range(8)]
    features = {node.node_id: [float(i), float(i % 3)] for i, node in enumerate(nodes)}
    first, rejected = balanced_p1_launch_subset(nodes, features, [1, 1])
    second, rejected2 = balanced_p1_launch_subset(list(reversed(nodes)), features, [1, 1])
    assert [node.node_id for node in first] == [node.node_id for node in second]
    assert rejected == rejected2
    assert len(first) == 13 and len(rejected) == 2
    assert sum(node.candidate_id == "main" for node in first) == 5
    assert all(node in first for node in nodes if node.candidate_id != "main")


def test_balanced_gate_requires_four_layers_and_candidate_cap():
    nodes = [_node(i, f"c{i % 6}", i % 4 + 1,
                   ("early", "middle", "late")[i % 3], f"p{i}") for i in range(16)]
    assert balanced_p1_launch_gate(nodes)["status"] == "PASS"
    dominated = [_node(i, "main" if i < 6 else f"c{i % 6}", i % 4 + 1,
                        ("early", "middle", "late")[i % 3], f"p{i}") for i in range(16)]
    gate = balanced_p1_launch_gate(dominated)
    assert gate["status"] == "FAIL"
    assert not gate["checks"]["candidate_cap"]


def test_balanced_subset_refuses_to_hide_required_clusters():
    nodes = [_node(i, "main", i + 1, ("early", "middle", "late")[i % 3], f"p{i}")
             for i in range(6)]
    features = {node.node_id: [float(i)] for i, node in enumerate(nodes)}
    with pytest.raises(ValueError, match="cannot preserve"):
        balanced_p1_launch_subset(nodes, features, [1], per_candidate_cap=5)
