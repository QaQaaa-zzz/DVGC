from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "apex_seen": True,
                "terminated": False,
                "truncated": False,
                "physical_failure": False,
                "timeout": False,
                "end_code": 0,
            },
            (True, False, False, False, "apex_success"),
        ),
        (
            {
                "apex_seen": False,
                "terminated": True,
                "truncated": False,
                "physical_failure": True,
                "timeout": False,
                "end_code": 3,
            },
            (False, True, False, False, "roll_limit"),
        ),
        (
            {
                "apex_seen": False,
                "terminated": True,
                "truncated": False,
                "physical_failure": False,
                "timeout": False,
                "end_code": 8,
            },
            (False, False, True, False, "stuck"),
        ),
        (
            {
                "apex_seen": False,
                "terminated": False,
                "truncated": False,
                "physical_failure": False,
                "timeout": False,
                "end_code": 0,
                "label_horizon": True,
            },
            (False, False, False, True, "label_horizon"),
        ),
    ],
)
def test_upstream_outcome_partition(kwargs, expected):
    from jit_dvgc.upstream_labels import classify_upstream_outcome

    result = classify_upstream_outcome(**kwargs)
    assert (
        result["success"],
        result["physical_failure"],
        result["task_failure"],
        result["timeout"],
        result["reason"],
    ) == expected
    assert (
        int(result["success"])
        + int(result["physical_failure"])
        + int(result["task_failure"])
        + int(result["timeout"])
        == 1
    )


def test_upstream_outcome_rejects_open_state():
    from jit_dvgc.upstream_labels import classify_upstream_outcome

    with pytest.raises(ValueError, match="not closed"):
        classify_upstream_outcome(
            apex_seen=False,
            terminated=False,
            truncated=False,
            physical_failure=False,
            timeout=False,
            end_code=0,
        )


def test_apex_does_not_override_same_step_terminal():
    from jit_dvgc.upstream_labels import classify_upstream_outcome

    result = classify_upstream_outcome(
        apex_seen=True,
        terminated=True,
        truncated=False,
        physical_failure=True,
        timeout=False,
        end_code=3,
    )
    assert not result["success"]
    assert result["physical_failure"]
