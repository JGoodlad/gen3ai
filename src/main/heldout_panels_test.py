"""The held-out panel registry (B-D3): the committed panels match it, and the near-neighbour rule bites at 4."""
import json
import re

import pytest

from main import heldout_panels as H
from utils.paths import repo_path


def test_registry_verifies_against_the_committed_panels():
    got = H.verify()
    assert got == {"home72": 72, "away": 20, "heldout72": 72, "heldout72_hpfix": 72, "forum72": 72, "p72": 72,
                   "forum216b": 216}


def test_held_out_panels_are_exactly_the_unfamiliar_ones():
    assert set(H.panel_names(held_out_only=True)) == {"heldout72", "heldout72_hpfix", "forum72", "p72", "forum216b"}
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


# ----------------------------------------------- the duplicate-team guard (X37b F-DV-25)

def _team(species: str) -> str:
    return f"{species} @ Leftovers\nAbility: Pressure\nEVs: 252 HP\n- Tackle\n- Protect"


def test_the_frozen_panels_duplicates_are_exactly_the_named_ones():
    """home72's pair is X37b's F-DV-25; `away`'s byte-identical pair was found by this guard. Both are named, no other panel
    has any, so `verify()` (above) passing means the allow list is neither missing nor stale."""
    assert H.duplicate_groups(H.panel_team_texts("home72")) == [("63eda9d8d491d6a4.txt", "dbbfac7bd5.txt")]
    assert H.duplicate_groups(H.panel_team_texts("away")) == [("team_13.gen3ou_team", "team_9.gen3ou_team")]
    for name in ("heldout72", "heldout72_hpfix", "forum72", "p72", "forum216b"):
        assert H.duplicate_groups(H.panel_team_texts(name)) == []
    assert set(H.KNOWN_DUPLICATES) == {"home72", "away"}
    assert all("X37b" in e["reason"] or "F-DV-25" in e["reason"] for e in H.KNOWN_DUPLICATES["home72"])


def test_a_panel_loader_refuses_an_unnamed_duplicate(monkeypatch):
    """The wiring: with home72's allow entry removed the REAL loader raises (fails on revert of the check)."""
    monkeypatch.setattr(H, "KNOWN_DUPLICATES", {})
    H.panel_team_texts.cache_clear()
    try:
        with pytest.raises(H.PanelDuplicateError, match="63eda9d8d491d6a4.txt = dbbfac7bd5.txt"):
            H.panel_team_texts("home72")
    finally:
        H.panel_team_texts.cache_clear()
    assert isinstance(H.PanelDuplicateError("x"), H.PanelRegistryError)     # h2h.panel turns this into PanelRefused


def test_duplicate_groups_see_byte_identical_and_repacked_copies_but_not_distinct_teams():
    a, b = _team("Snorlax"), _team("Skarmory")
    padded = "\n" + a + "\n\n\n"                       # the same paste with Metamon-style padding: same packed team
    items = [("a1", a), ("b", b), ("a2", padded), ("a3", a)]
    assert H.duplicate_groups(items) == [("a1", "a2", "a3")]
    assert H.duplicate_groups([("a", a), ("b", b)]) == []


def test_check_distinct_allows_only_the_exact_named_group_with_a_reason():
    a, b = _team("Snorlax"), _team("Skarmory")
    items = [("a1", a), ("a2", a), ("b", b)]
    ok = {"p": ({"labels": ("a2", "a1"), "reason": "X37b: frozen"},)}
    H.check_distinct("p", items, ok)                                              # label order does not matter
    with pytest.raises(H.PanelDuplicateError, match="a1 = a2"):
        H.check_distinct("p", items, {})
    with pytest.raises(H.PanelDuplicateError, match="a1 = a2 = a3"):             # a third copy is a NEW duplicate
        H.check_distinct("p", items + [("a3", a)], ok)
    with pytest.raises(H.PanelDuplicateError, match="no reason"):
        H.check_distinct("p", items, {"p": ({"labels": ("a1", "a2"), "reason": " "},)})
    with pytest.raises(H.PanelDuplicateError, match="no longer a duplicate"):    # a stale entry cannot outlive its defect
        H.check_distinct("p", [("a1", a), ("b", b)], ok)


# ------------------------------------------------------------- forum216b: X54's sealed confirmation panel (2026-10-20)
F216 = "designs/research_state/measurements/forum216b_x54_2026-10-20"


def test_forum216b_restates_its_draw_manifest():
    """The registry entry is the drawn panel: its digest, count and manifest, and forum72's strata x 3."""
    ent = H.panel_entry("forum216b")
    man = json.loads(repo_path(F216, "forum216b_manifest.json").read_text())
    assert ent["content_digest"] == man["content_digest"] and ent["n"] == man["n"] == 216
    assert ent["held_out_from"] == ["training"] and ent["seed"] == man["seed"] == 5400
    f72 = json.loads(repo_path("designs/research_state/measurements/x37a_eval_matrix_2026-10-18",
                               "forum72_manifest.json").read_text())["strata"]
    strata = {s: sum(1 for t in man["teams"].values() if t["stratum"] == s) for s in ("CORE", "NEAR", "FAR")}
    assert strata == {s: 3 * len(v) for s, v in f72.items()} == {"CORE": 12, "NEAR": 162, "FAR": 42}


def test_forum216b_copies_and_near_copies_no_pool_or_panel_team():
    """No team of the panel shares >= 4 full mon sets with a pool team, another registered panel's team, or another
    team of its own (the registry's near-neighbour radius; the draw rejected each)."""
    from main.anchors.team_overlap import Reference, parse_items, parse_team
    from utils.team_loader import TeamLoader

    items = list(H.panel_team_texts("forum216b"))
    others = [p for p in H.panel_names() if p != "forum216b"]
    assert set(others) >= {"home72", "away", "heldout72", "heldout72_hpfix", "forum72", "p72"}
    assert H.hits(items, others) == []
    pool = TeamLoader(str(repo_path("data", "teams"))).get_all_teams()
    ref = Reference(parse_items([(f"pool{i}", t) for i, t in enumerate(pool)]))
    assert max(ref.max_shared(parse_team(t))[0] for _, t in items) < H.NEAR_NEIGHBOUR_MIN_SHARED
    teams = parse_items(items)
    for j, t in enumerate(teams):
        assert Reference(teams[:j] + teams[j + 1:]).max_shared(t)[0] < H.NEAR_NEIGHBOUR_MIN_SHARED, t


def test_forum216b_is_sealed_and_only_it():
    assert set(H.SEALED) == {"forum216b"} and "X54" in H.SEALED["forum216b"]
    with pytest.raises(H.PanelSealedError, match="SEALED until X54"):
        H.check_unsealed("forum216b")
    for name in H.panel_names():
        if name != "forum216b":
            H.check_unsealed(name)
    # sealed is not unregistered: the exclusion rule still reads its content
    assert len(H.panel_team_texts("forum216b")) == 216 and "forum216b" in H.panel_names(held_out_only=True)
