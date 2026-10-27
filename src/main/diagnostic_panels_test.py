"""The DIAGNOSTIC panel registry (``designs/research_state/diagnostic_panels.json``) and its live panel, rc72b (the
H-CTX read's recombined pool sets, ``measurements/hctx_rc72_2026-10-26``): outside the exclusion rule, verified like
any panel, and built as its PREREG §2 and Amendment 2 say. ``rc72`` is the retired first draw."""
import collections
import json

import pytest

from main import heldout_panels as H
from main.anchors import team_overlap as TO


def test_a_diagnostic_panel_is_verified_but_never_part_of_the_exclusion_rule():
    assert H.diagnostic_panel_names() == ["rc72", "rc72b"]
    assert not {"rc72", "rc72b"} & set(H.panel_names()) and H.panel_entry("rc72b")["held_out_from"] == []
    assert len(H.panel_team_texts("rc72b")) == 72 and H.verify()["rc72b"] == 72
    assert H.panel_entry("rc72")["role"].startswith("RETIRED")
    with pytest.raises(KeyError, match="diagnostic_panels.json"):
        H.panel_entry("no_such_panel")


def test_a_name_in_both_registries_is_refused(monkeypatch, tmp_path):
    reg = json.loads(H.registry_path().read_text())
    diag = {"schema": H.DIAGNOSTIC_SCHEMA, "panels": {"home72": reg["panels"]["home72"]}}
    p = tmp_path / "diag.json"
    p.write_text(json.dumps(diag))
    monkeypatch.setattr(H, "DIAGNOSTIC_REL", str(p))
    H.load_diagnostic.cache_clear()
    try:
        with pytest.raises(H.PanelRegistryError, match="one registry"):
            H.load_diagnostic()
    finally:
        monkeypatch.undo()
        H.load_diagnostic.cache_clear()
    assert H.diagnostic_panel_names() == ["rc72", "rc72b"]


@pytest.fixture(scope="module")
def rc72():
    return TO.parse_items(H.panel_team_texts("rc72b"))


def _hp_as_written_mismatches(text):
    """Hidden Power sets whose named type is not the type their IVs AS WRITTEN give (a missing IV reads 31): the
    class the live server rejects (rc72__00, "Aerodactyl has Hidden Power Bug, but its IVs are for Hidden Power
    Water")."""
    import re

    from agents.gen3_data import hidden_power

    stats = {"hp": "hp", "atk": "atk", "def": "def", "spa": "spa", "spd": "spd", "spe": "spe"}
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        m = re.search(r"^-\s*Hidden Power\s*\[?\s*([A-Za-z]+)", block, re.M)
        if not m:
            continue
        ivs = {}
        iv = re.search(r"^IVs:\s*(.+)$", block, re.M)
        for part in (iv.group(1).split("/") if iv else []):
            n, name = part.split()[:2]
            ivs[stats[name.strip().lower()]] = int(n)
        if hidden_power.type_of(ivs).lower() != m.group(1).lower():
            out.append(block.split("\n", 1)[0])
    return out


def test_every_rc72b_hidden_power_plays_its_named_type_as_written_and_rc72s_did_not():
    bad = {lab: b for lab, t in H.panel_team_texts("rc72b") if (b := _hp_as_written_mismatches(t))}
    assert bad == {}
    assert sum(len(_hp_as_written_mismatches(t)) for _, t in H.panel_team_texts("rc72")) == 71, \
        "the retired draw's defect, which this check must see"


def test_rc72_is_made_only_of_pool_sets_and_shares_at_most_two_with_any_training_team(rc72):
    from main.anchors.session import pool_team_texts
    from utils import team_corpora as TC

    pool = TO.parse_items((str(i), t) for i, t in enumerate(pool_team_texts()))
    pool_strict = {m.strict_key for t in pool for m in t.mons}
    assert all(m.strict_key in pool_strict for t in rc72 for m in t.mons), "every rc72 set is a pool set, verbatim"
    refs = {"pool": TO.Reference(pool)}
    for name in TC.CORPORA:
        c = TC.load_corpus(name, "ALL" if name != "w_forum_v1" else "ABC")
        refs[name] = TO.Reference(TO.parse_items(zip(c.ids, c.texts)))
    for name, ref in refs.items():
        assert max(ref.max_shared(t)[0] for t in rc72) <= 2, name


def test_rc72_has_six_species_per_team_and_is_far_from_the_held_out_panels(rc72):
    from main.anchors.session import load_team_texts
    from utils.paths import repo_path

    assert all(len(t.mons) == 6 and len({m.species for m in t.mons}) == 6 for t in rc72)
    held = {p: TO.Reference(TO.parse_items(H.panel_team_texts(p))) for p in H.panel_names(held_out_only=True)}
    held["n1p72"] = TO.Reference(TO.parse_items(load_team_texts(repo_path(
        "designs/research_state/measurements/x61_n1_breadth_2026-10-25/teams_n1p72"))))
    for name, ref in held.items():
        assert max(ref.max_shared(t)[0] for t in rc72) <= 2, name
    for j, t in enumerate(rc72):
        assert TO.Reference(rc72[:j] + rc72[j + 1:]).max_shared(t)[0] <= 3


def test_rc72_tracks_home72s_species_profile(rc72):
    home = collections.Counter(m.species for t in TO.parse_items(H.panel_team_texts("home72")) for m in t.mons)
    got = collections.Counter(m.species for t in rc72 for m in t.mons)
    assert set(got) <= set(home)
    tv = sum(abs(got[s] - home[s]) for s in set(got) | set(home)) / (2 * 432)
    assert tv < 0.2, tv
