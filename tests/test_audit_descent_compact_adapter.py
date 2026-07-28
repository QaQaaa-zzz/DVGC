from cli.audit_descent_compact_adapter_v1 import audit_gate


def test_audit_gate_requires_precision_and_clean_terminal_contract():
    report={"tube_precision":1.0,"terminal_summary":{"timeouts":0,"horizon_exhaustions":0,
        "physical_end_reasons":{"nonfinite":0}}}
    assert all(audit_gate(report).values())
    report["tube_precision"]=.94
    assert not audit_gate(report)["precision"]
