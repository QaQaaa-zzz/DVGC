from pathlib import Path

from cli.freeze_descent_predecessor_assets import verify_frozen_assets


def test_missing_asset_manifest_fails_closed(tmp_path: Path):
    valid, failed = verify_frozen_assets(tmp_path)
    assert not valid
    assert failed == ["manifest_missing"]
