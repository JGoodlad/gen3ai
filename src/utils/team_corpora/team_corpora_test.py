"""``w_forum_v1`` is what its manifest says, it holds no panel team or near neighbour, and a tampered file is refused."""
import collections
import json
import shutil

import pytest

from main import heldout_panels as H
from utils import team_corpora as TC
from utils.paths import repo_path
from utils.team_corpora import build_w_forum as B


def _manifest():
    return json.loads((TC.CORPORA["w_forum_v1"] / TC.MANIFEST_NAME).read_text())


def test_every_tier_verifies_and_the_tiers_are_prefixes():
    m = _manifest()
    full = TC.load_corpus("w_forum_v1", "ABC")
    assert TC.load_corpus("w_forum_v1") == full, "the default tier is ABC"
    for name, members in B.TIERS.items():
        c = TC.load_corpus("w_forum_v1", name)
        assert len(c) == m["tiers"][name]["n"] > 0
        assert c.texts == full.texts[:len(c)]
    _, rows, _, _ = TC.read_dir(TC.CORPORA["w_forum_v1"])
    order = [B.TIER_ORDER.index(r["tier"]) for r in rows]
    assert order == sorted(order), "rows are in tier order A, B, C"
    assert [r["i"] for r in rows] == list(range(len(rows)))


def test_the_corpus_holds_no_registered_panel_team_or_near_neighbour():
    c = TC.load_corpus("w_forum_v1", "ABC")
    assert H.hits(list(zip(c.ids, c.texts))) == []
    assert _manifest()["panel_registry"]["sha256"] == H.registry_sha256(), \
        "the registry changed since the corpus was built: rebuild and re-check"


def test_the_corpus_holds_no_exact_pool_team_and_no_open_crash_class():
    from main.anchors.team_overlap import Reference, parse_items, parse_team
    from utils.team_loader import TeamLoader

    pool = TeamLoader(str(repo_path("data", "teams"))).get_all_teams()
    ref = Reference(parse_items([(str(i), t) for i, t in enumerate(pool)]))
    c = TC.load_corpus("w_forum_v1", "ABC")
    assert not [i for i, t in zip(c.ids, c.texts) if ref.keys[parse_team(t).key]]
    assert not [i for i, t in zip(c.ids, c.texts) if set(B.moves_of(t)) & set(B.CRASH_MOVES)]
    assert len({parse_team(t).key for t in c.texts}) == len(c), "no two corpus teams share all six mon sets"


def test_the_curator_cap_holds_and_every_row_is_a_verbatim_source_team():
    m = _manifest()
    _, rows, _, _ = TC.read_dir(TC.CORPORA["w_forum_v1"])
    per = collections.Counter(r["curator"] for r in rows)
    assert max(per.values()) == m["curator_cap"]["cap"] <= B.CURATOR_SHARE_CAP * len(rows)
    src = {r["id"]: r for r in map(json.loads, repo_path(B.SRC_REL, "manifest_teams.jsonl").read_text().splitlines())
           if r.get("status") == "kept"}
    for r in rows:
        s = src[r["id"]]
        assert s["status"] == "kept" and s["fidelity"] == "verbatim" and s["tier"] == r["tier"], r["id"]


def test_the_build_accounts_for_every_verbatim_source_team():
    m = _manifest()
    dropped = sum(v["n"] for v in m["drops"].values()) + sum(m["curator_cap"]["dropped_by_tier"].values())
    assert m["tiers"]["ABC"]["n"] + dropped == m["source"]["kept_verbatim"]


def test_curator_cap_rule():
    # c = 3: 3 <= 0.25 * (3 + 3 + 3 + 3 + 1) = 3.25; c = 4: 4 > 0.25 * (4 + 4 + 3 + 3 + 1) = 3.75
    assert B.curator_cap({"a": 10, "b": 9, "c": 3, "d": 3, "e": 1}, 0.25) == 3


@pytest.mark.parametrize("what", ["data", "tier"])
def test_a_tampered_corpus_is_refused(tmp_path, what):
    src = TC.CORPORA["w_forum_v1"]
    d = tmp_path / "w"
    shutil.copytree(src, d)
    if what == "data":
        b = bytearray((d / TC.DATA_NAME).read_bytes())
        b[-9] ^= 1
        (d / TC.DATA_NAME).write_bytes(bytes(b))
        with pytest.raises(TC.CorpusError, match="sha256"):
            TC.load_dir(d, "A")
    else:
        m = json.loads((d / TC.MANIFEST_NAME).read_text())
        m["tiers"]["A"]["n"] -= 1
        (d / TC.MANIFEST_NAME).write_text(json.dumps(m))
        with pytest.raises(TC.CorpusError, match="tier A sha256"):
            TC.load_dir(d, "A")


def test_an_unknown_corpus_or_tier_is_refused():
    with pytest.raises(TC.CorpusError, match="unknown team corpus"):
        TC.load_corpus("w_forum_v0")
    with pytest.raises(TC.CorpusError, match="tier"):
        TC.load_corpus("w_forum_v1", "ABCD")


@pytest.mark.integration
def test_the_committed_corpus_reproduces_from_its_source():
    """Node (the pinned validator) + the engine's coverage oracle: the build is deterministic."""
    assert B.main(["--check"]) == 0
