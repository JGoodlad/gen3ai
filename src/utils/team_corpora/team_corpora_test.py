"""``w_forum_v1`` is what its manifest says, it holds no panel team or near neighbour, and a tampered file is refused."""
import collections
import hashlib
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
    # no OPEN crash class remains (CRASH_MOVES is empty since the Transform/HP and Imprison classes closed, 2026-10-19);
    # the filter stays: were a class opened again, a move re-added to CRASH_MOVES must leave the corpus free of it
    assert not [i for i, t in zip(c.ids, c.texts) if set(B.moves_of(t)) & set(B.CRASH_MOVES)]
    assert B.CRASH_MOVES == (), "an open crash class is registered: the corpus must be rebuilt without its teams"
    assert _manifest()["crash_class_moves"] == list(B.CRASH_MOVES) and _manifest()["drops"]["crash_class"]["n"] == 0
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


@pytest.mark.slow  # ship gate (2026-10-19): 20 s beside a live run
@pytest.mark.integration
def test_the_committed_corpus_reproduces_from_its_source():
    """Node (the pinned validator) + the engine's coverage oracle: the build is deterministic."""
    assert B.main(["--check"]) == 0


# ---------------------------------------------------------------- THE LABEL-PATH GUARD (launch sheet AM-D1-FIX)
_SFV2_5051 = repo_path(B.SRC_REL, "teams", "sfv2_5051.txt")


def test_label_path_defects_names_every_non_100_level_and_nothing_else():
    assert TC.label_path_defects("Krabby @ Salac Berry\nLevel: 5\n- Endure") == ["Level: 5"]
    assert TC.label_path_defects("Mew\nlevel: 50 \n- Psychic\n\nAbra\nLevel: 100\n- Psychic") == ["Level: 50"]
    assert TC.label_path_defects("Mew\nLevel: 100\n- Psychic") == []
    assert TC.label_path_defects("Mew\nAbility: Synchronize\n- Psychic") == [], "no Level line = Showdown's 100"


def test_the_corpus_holds_no_team_the_label_path_cannot_encode_and_sfv2_5051_is_out():
    c = TC.load_corpus("w_forum_v1", "ABC")
    assert not [i for i, t in zip(c.ids, c.texts) if TC.label_path_defects(t)]
    assert "sfv2_5051" not in c.ids, "the level-5 team that crash-looped D1 s3501 (AM-D1-FIX)"
    g = _manifest()["label_guard"]
    assert g["dropped"] == {"sfv2_5051": ["Level: 5"]} and _manifest()["drops"]["level"]["by_tier"] == {"C": 1}
    assert g["source_scan"]["sfv2_5051"]["dropped_by"] == "level"
    assert all(v["dropped_by"] in B.RULES for v in g["source_scan"].values())


def test_a_corpus_holding_a_non_100_level_team_is_refused_by_every_reader(tmp_path):
    """The throwing guard in ``load_dir``: a consistent corpus (shas re-stamped) with one level-5 team never loads."""
    man, rows, _, _ = TC.read_dir(TC.CORPORA["w_forum_v1"])
    rows = [dict(r) for r in rows]
    text = _SFV2_5051.read_text().strip()
    rows[0].update(id="sfv2_5051", text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
    data = TC.encode_rows(rows)
    man = dict(man, data_sha256=hashlib.sha256(data).hexdigest())
    texts = [r["text"] for r in rows]
    man["tiers"] = {k: dict(v, sha256=TC.tier_sha256(texts[:v["n"]])) for k, v in man["tiers"].items()}
    d = tmp_path / "w"
    d.mkdir()
    (d / TC.DATA_NAME).write_bytes(data)
    (d / TC.MANIFEST_NAME).write_text(json.dumps(man))
    with pytest.raises(TC.CorpusError, match=r"label path cannot encode.*sfv2_5051 \['Level: 5'\]"):
        TC.load_dir(d, "A")


def test_the_level_rule_reads_showdowns_packed_level_and_refuses_a_disagreeing_text():
    one = "Krabby||salacberry|hypercutter|swordsdance,return|Adamant|196,,,,116,196||,30,30,,30,|"
    assert B.packed_levels("]".join([one + "|5|", one + "||", one + "|100|"])) == ["5", "100", "100"]
    assert B.level_defects("t", "Krabby\nLevel: 5", one + "|5|") == ["Level: 5"]
    assert B.level_defects("t", "Krabby", one + "||") == []
    with pytest.raises(RuntimeError, match="disagree"):
        B.level_defects("t", "Krabby", one + "|5|")


@pytest.mark.sim
@pytest.mark.integration
def test_the_soak_runs_the_label_path_and_fails_on_the_level_5_team_and_passes_it_at_level_100():
    """Schema v1 declared no label family and PASSED sfv2_5051; the v2 soak drives the label writers (every core
    family), so the D1 s3501 fault reproduces here, and the same team with its ``Level:`` lines removed passes."""
    from utils.team_corpora import soak as S

    good = TC.load_corpus("w_forum_v1", "A")
    bad = _SFV2_5051.read_text().strip()
    kw = dict(rounds=1, n_envs=2, threads=1)
    r = S.soak_texts([good.ids[0], "sfv2_5051"], [good.texts[0], bad], {"corpus": "test"}, **kw)
    assert r["schema"] == S.SOAK_SCHEMA and r["labels"] == S.all_label_families() and r["labels"]
    assert r["verdict"] == "FAIL", r
    assert "at L100 but the server's stats" in r["core_errors"][0]["error"]
    l100 = "\n".join(ln for ln in bad.splitlines() if not ln.strip().startswith("Level:"))
    ok = S.soak_texts([good.ids[0], "sfv2_5051_l100"], [good.texts[0], l100], {"corpus": "test"}, **kw)
    assert ok["verdict"] == "PASS" and ok["battles_finished"] == 2, ok
