"""``main.h2h play --panel home72`` on the real engine (CPU, tiny: ONE batch of 2 mirrored pairs): the row validates,
carries the registered panel's regime (its content digest in the label, its team set as identity), and EVERY team
either side piloted is one of the panel's -- the proof both sides drew from the panel, not the pool. The numbers are
not the point (perturbed-fresh checkpoints)."""
from __future__ import annotations

import pytest

from agents.training import eval_ledger as L
from main import heldout_panels as HP
from main.h2h import panel as PN
from main.h2h import play as PL

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]  # ship gate (2026-10-19): 87 s of routine-gate time beside a live run (designs/ops/testing.md 'The SHIP GATE')

#: one intra-op torch thread (designs/ops/testing.md "One torch thread for the CPU plays")
COMPUTE = PL.Compute(device="cpu", backend="eager", n_envs=4, threads=2, torch_threads=1, front="ffi",
                     profile="selfcheck")
ROWS = L.ReaderDecl(name="h2h_panel_test", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                    selection="include", flags_ok=frozenset(), inference="conditional")


@pytest.fixture(scope="module")
def played(built, checkpoints, tmp_path_factory):
    a, b = checkpoints
    out = tmp_path_factory.mktemp("h2h_panel") / "ledger"
    summ = PL.play_edge(str(out), PL.resolve_player(a), PL.resolve_player(b), pairs=2, batch_pairs=2,
                        schedule_seed=3, run_label="panel-test", compute=COMPUTE, panel="home72",
                        emit=lambda _m: None)
    return summ, list(L.read(ROWS, root=out).rows)


def test_the_row_carries_the_registered_panels_regime(played):
    _summ, rows = played
    assert len(rows) == 1
    reg = rows[0]["regime"]
    assert reg["team_set"] == PN.panel_team_set_id("home72")
    assert HP.panel_entry("home72")["content_digest"] in reg["team_source"]
    assert reg["regime_id"] == PN.panel_regime(reg["turn_limit"], "home72")["regime_id"]
    assert rows[0]["pairs"]["n_pairs"] == 2 and sum(rows[0]["counts"][k] for k in ("w", "l", "d")) == 4


def test_every_piloted_team_on_both_sides_is_a_panel_team(played):
    _summ, rows = played
    tb = PN.PanelTeams("home72")
    panel_ids = {L.team_id(p) for p in tb.packed_teams}
    teams = rows[0]["teams"]
    assert teams, "the row recorded no team"
    assert set(teams) <= panel_ids, f"teams off the panel: {sorted(set(teams) - panel_ids)}"
    # both seats' draws are counted: 2 pairs x 2 games, one player-side and one opponent-side count per game
    assert sum(t["p"][0] for t in teams.values()) == 4 and sum(t["o"][0] for t in teams.values()) == 4
