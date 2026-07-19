from pathlib import Path
from cli.jump_envelope_controller import CYCLE4_POLICY, CYCLE5_POLICY, CANDIDATES, ENTRY, LANDING

def test_jump_envelope_controller_binds_immutable_inputs():
    for path in (CYCLE4_POLICY/'params.pkl', CYCLE5_POLICY/'params.pkl', CANDIDATES, ENTRY, LANDING/'params.pkl'):
        assert Path(path).exists()


def test_activation_module_exposes_jump_envelope_unit():
    source = Path("cli/activate_jump_envelope_pipeline.py").read_text()
    assert "dvgc-jump-envelope-controller.service" in source
    assert "start_jump_envelope_controller.sh" in source
