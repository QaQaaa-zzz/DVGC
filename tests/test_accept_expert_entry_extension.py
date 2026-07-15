from types import SimpleNamespace

from cli.accept_expert_entry_extension import validate


class Registry:
    registry_hash = "registry"
    specs = {"flight": SimpleNamespace(downstream_entry_set_sha256="entry")}


def test_validate_accepts_positive_chain_and_final(monkeypatch):
    monkeypatch.setattr("cli.accept_expert_entry_extension.file_sha256", lambda path: {"bank": "bank", "entry": "entry"}[path])
    report = {"candidate_bank_sha256": "bank", "entry_set_sha256": "entry", "registry_hash": "registry", "chain_rate": .1, "composite_final_rate": .12, "timeout_rate": 0}
    assert validate(Registry(), report, "bank", "entry") == []


def test_validate_rejects_chain_zero(monkeypatch):
    monkeypatch.setattr("cli.accept_expert_entry_extension.file_sha256", lambda path: {"bank": "bank", "entry": "entry"}[path])
    report = {"candidate_bank_sha256": "bank", "entry_set_sha256": "entry", "registry_hash": "registry", "chain_rate": 0, "composite_final_rate": .12, "timeout_rate": 0}
    assert "Chain is zero" in validate(Registry(), report, "bank", "entry")
