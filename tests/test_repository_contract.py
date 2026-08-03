from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_documented_repository_entrypoints_exist():
    for relative in (
        "scripts/local_preflight.sh",
        "cli/prepare_project.py",
        "cli/runtime_gate.py",
        "cli/build_candidates.py",
        "cli/train.py",
        "cli/certify.py",
        "cli/audit.py",
        "cli/evaluate.py",
    ):
        assert (ROOT / relative).is_file(), relative


def test_authoritative_assets_and_current_documents_exist():
    for relative in (
        "assets/orange_bike_4kg_horizontal.xml",
        "configs/default.json",
        "data/reference_jump.csv",
        "docs/METHOD_TWO_PHASE_SOFT_TUBE.md",
        "docs/REPOSITORY_LAYOUT.md",
        "docs/EXPERIMENT_STATE.md",
    ):
        assert (ROOT / relative).is_file(), relative


def test_future_two_phase_cli_placeholders_do_not_exist():
    for name in (
        "build_guideline_banks",
        "train_phase_expert",
        "collect_phase_snapshots",
        "label_phase_snapshots",
        "train_feasibility_model",
        "build_soft_tube_bank",
        "train_unified_tube_rsi",
        "evaluate_jump_envelope",
    ):
        assert not (ROOT / "cli" / f"{name}.py").exists(), name
