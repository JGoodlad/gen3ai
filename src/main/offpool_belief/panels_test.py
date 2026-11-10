"""Which panels the meter reads, and where it refuses to write."""
from __future__ import annotations

from pathlib import Path

import pytest

from main.offpool_belief import HOME_PANEL, OFFPOOL_PANELS, OffPoolBeliefError
from main.offpool_belief import panels as PN


def test_the_three_read_panels_are_held_out_and_home_is_the_in_pool_reference() -> None:
    for name in OFFPOOL_PANELS:
        ref = PN.panel_ref(name)
        assert ref.held_out and ref.kind == "OFF-POOL" and len(ref.teams) == 72, name
    home = PN.panel_ref(HOME_PANEL)
    assert not home.held_out and home.kind == "IN-POOL reference" and len(home.teams) == 72


def test_a_sealed_panel_is_refused_before_a_team_is_read() -> None:
    from main import heldout_panels as HP

    sealed = [n for n in HP.panel_names() if _is_sealed(n)]
    assert sealed, "the registry holds no sealed panel: this test would pass vacuously"
    for name in sealed:
        with pytest.raises(OffPoolBeliefError, match="SEALED"):
            PN.panel_ref(name)


def _is_sealed(name: str) -> bool:
    from main import heldout_panels as HP

    try:
        HP.check_unsealed(name)
    except HP.PanelSealedError:
        return True
    return False


def test_a_retired_or_unregistered_panel_is_refused() -> None:
    with pytest.raises(OffPoolBeliefError, match="RETIRED"):
        PN.panel_ref("heldout72")
    with pytest.raises(OffPoolBeliefError, match="not a registered panel"):
        PN.panel_ref("no_such_panel")


def test_nothing_is_written_into_a_checkout_or_under_models(tmp_path: Path) -> None:
    from utils.paths import main_models_dir, repo_root

    with pytest.raises(OffPoolBeliefError, match="inside the checkout"):
        PN.refuse_inside_repo(repo_root() / "bank_x", "the bank directory")
    (tmp_path / "co" / ".git").mkdir(parents=True)                  # any checkout, not only this one
    with pytest.raises(OffPoolBeliefError, match="inside the checkout"):
        PN.refuse_inside_repo(tmp_path / "co" / "deep" / "bank", "the bank directory")
    PN.refuse_inside_repo(tmp_path / "elsewhere" / "bank", "the bank directory")
    md = main_models_dir()
    if md is not None:
        with pytest.raises(OffPoolBeliefError, match="read-only"):
            PN.refuse_under_models(Path(md) / "some_run" / "bank", "the bank directory")
    PN.refuse_under_models(tmp_path / "bank", "the bank directory")
