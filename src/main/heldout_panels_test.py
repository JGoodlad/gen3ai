"""The held-out panel registry (B-D3): the committed panels match it, and the near-neighbour rule bites at 4."""
import json
import re

import pytest

from main import heldout_panels as H
from utils.paths import repo_path


def test_registry_verifies_against_the_committed_panels():
    got = H.verify()
    assert got == {"home72": 72, "away": 20, "heldout72": 72, "heldout72_hpfix": 72, "forum72": 72, "p72": 72}


def test_held_out_panels_are_exactly_the_unfamiliar_ones():
    assert set(H.panel_names(held_out_only=True)) == {"heldout72", "heldout72_hpfix", "forum72", "p72"}
    # The in-pool panels exist, but nothing is refused for holding them (the pool does).
    assert set(H.panel_names()) - set(H.panel_names(held_out_only=True)) == {"home72", "away"}


def test_registry_names_the_digests_the_x37a_registrations_froze():
    """The registry restates what X37a / TSR registered; it must not silently re-freeze a different set."""
    reg = H.load_registry()["panels"]
    m = "designs/research_state/measurements/x37a_eval_matrix_2026-10-18"
    s1 = json.loads(repo_path(m, "x37a_stage1_registration.json").read_text())
    s2 = json.loads(repo_path(m, "x37a_stage2_registration.json").read_text())
    assert reg["home72"]["content_digest"] == s1["home72"]["content_digest"]
    assert reg["heldout72_hpfix"]["content_digest"] == s1["heldout72_hpfix"]["content_digest"]
    assert reg["forum72"]["content_digest"] == s2["panels"]["forum72"]["content_digest"]
    assert reg["p72"]["content_digest"] == s2["panels"]["p72"]["content_digest"]


def _blocks(text):
    return [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]


def _with_k_from(panel_team, other, k):
    """A team holding the first ``k`` mon blocks of ``panel_team`` and the rest from ``other``."""
    a, b = _blocks(panel_team), _blocks(other)
    return "\n\n".join(a[:k] + b[k:6])


def test_near_neighbour_rule_bites_at_four_shared_mon_sets():
    forum = dict(H.panel_team_texts("forum72"))
    p72 = H.panel_team_texts("p72")
    label, team = sorted(forum.items())[0]
    # a P72 team that shares nothing with that forum team, to fill the other slots
    filler = next(t for _, t in p72 if H.nearest(t, ["forum72"])[0][1] == 0)
    exact = H.contamination([("x", team)])
    assert exact and exact[0].panel == "forum72" and exact[0].shared == 6
    four = _with_k_from(team, filler, 4)
    three = _with_k_from(team, filler, 3)
    got4 = [h for h in H.contamination([("four", four)]) if h.panel == "forum72"]
    got3 = [h for h in H.contamination([("three", three)]) if h.panel == "forum72"]
    assert got4 and got4[0].shared >= 4 and got4[0].nearest == label
    assert not got3, "3 shared mon sets is NOT a near neighbour (the radius is >= 4)"


def test_in_pool_panels_are_not_held_out_but_still_match():
    home = H.panel_team_texts("home72")[:3]
    assert H.contamination(list(home)) == []
    assert {h.panel for h in H.hits(list(home))} >= {"home72"}


def test_the_training_pool_is_clean_against_every_held_out_panel():
    from utils.team_loader import TeamLoader

    pool = TeamLoader(str(repo_path("data", "teams"))).get_all_teams()
    assert len(pool) >= 700
    assert H.contamination([(f"pool{i}", t) for i, t in enumerate(pool)]) == []


def test_a_changed_panel_is_refused(monkeypatch):
    reg = json.loads(H.registry_path().read_text())
    reg["panels"]["p72"]["content_digest"] = "0" * 64
    monkeypatch.setattr(H, "load_registry", lambda: reg)
    H.panel_team_texts.cache_clear()
    try:
        with pytest.raises(H.PanelRegistryError, match="content digest"):
            H.panel_team_texts("p72")
    finally:
        H.panel_team_texts.cache_clear()
