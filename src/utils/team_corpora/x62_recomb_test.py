"""``x62_recomb_v1`` (X62's recombination corpus) is what ``design_x62_team_recombination.md`` §2 and its manifest say:
12,000 new sixes of the pool's own STRICT-class sets (verbatim, nickname stripped), <= 2 sets shared with any one pool
team, < 3 with any held-out panel / n1p72 / rc72b team, <= 4 within the corpus, no pool species six, prefix tiers, and
a deterministic draw. Its crash / label-path soak is the union's (``x63_union_v1`` holds all 12,000 teams)."""
import collections
import json

import pytest

from main.anchors import team_overlap as TO
from utils import team_corpora as TC
from utils.team_corpora import build_x62_recomb as B

NAME = B.NAME


@pytest.fixture(scope="module")
def corpus():
    return TC.load_corpus(NAME)


@pytest.fixture(scope="module")
def rows():
    return TC.read_dir(TC.CORPORA[NAME])[1]


@pytest.fixture(scope="module")
def manifest():
    return json.loads((TC.CORPORA[NAME] / TC.MANIFEST_NAME).read_text())


@pytest.fixture(scope="module")
def inputs():
    return B.Inputs()


@pytest.fixture(scope="module")
def teams(corpus):
    return [TO.parse_team(t, i) for i, t in zip(corpus.ids, corpus.texts)]


def test_the_corpus_has_twelve_thousand_teams_in_prefix_tiers_and_candidate_order(corpus, rows, manifest):
    assert corpus.tier == B.DEFAULT_TIER and len(corpus) == B.N_CORPUS == 12_000
    assert [t for t, _ in B.TIERS] == ["K1", "K3", "ALL"]
    for name, n in B.TIERS:
        c = TC.load_corpus(NAME, name)
        assert len(c) == n == manifest["tiers"][name]["n"] and c.texts == corpus.texts[:n]
    assert [r["i"] for r in rows] == list(range(len(rows)))
    assert [r["candidate"] for r in rows] == sorted(r["candidate"] for r in rows)
    assert manifest["candidates_drawn"] == rows[-1]["candidate"] + 1
    refused = sum(manifest["candidate_refusals"].values())
    assert refused + len(rows) == manifest["candidates_drawn"], "every drawn candidate is accepted or counted"


def test_every_mon_is_a_strict_pool_occurrence_verbatim_but_for_the_nickname(rows, inputs):
    """Each block is its source's STRICT occurrence with the nickname stripped, byte for byte (the LAST block up to the
    trailing whitespace the team text's ``.strip()`` removes)."""
    verbatim = {(sid, B.strip_nickname(b)) for v in inputs.occ.values() for b, _m, sid in v}
    tail = {(sid, b.rstrip()) for sid, b in verbatim}
    pool_sids = {B.pool_source_id(t) for t in inputs.pool_texts}
    for r in rows:
        assert r["text"] == r["text"].strip(), r["id"]
        blocks = B.split_blocks(r["text"])
        assert len(blocks) == 6 and len(r["sources"]) == 6 and set(r["sources"]) <= pool_sids, r["id"]
        for k, (b, sid) in enumerate(zip(blocks, r["sources"])):
            assert (sid, b) in (verbatim if k < 5 else tail), (r["id"], b)
            assert not B.strict_text_bad(b), (r["id"], b)
        assert not TC.label_path_defects(r["text"]) and "Level:" not in r["text"]


def test_the_distance_rules_hold(teams, inputs):
    pool = inputs.refs["train"]["pool"]
    assert max(pool.max_shared(t)[0] for t in teams) <= B.MAX_SHARED_POOL
    assert set(inputs.held) == {"heldout72", "heldout72_hpfix", "forum72", "p72", "forum216b", "n1p72", "rc72b"}
    for name, ref in inputs.held.items():
        assert max(ref.max_shared(t)[0] for t in teams) < B.HELDOUT_REFUSE, name
    assert not [t.label for t in teams if frozenset(t.species6) in inputs.pool_six]
    assert all(len(t.species6) == 6 for t in teams), "Species Clause"
    idx = collections.defaultdict(list)
    worst = 0
    for i, t in enumerate(teams):
        h = collections.Counter(j for k in t.key for j in idx[k])
        worst = max(worst, max(h.values(), default=0))
        for k in t.key:
            idx[k].append(i)
    assert worst <= B.WITHIN_MAX and len({t.key for t in teams}) == len(teams)


def test_the_manifest_records_section_2_4_and_matches_the_prototype(manifest):
    d = manifest["describe"]
    s = d["strict_filter"]
    # strict_draw_sim.log: 739 of 4314 pool mons removed, 572 of 658 sets drawable, 6 species lost, 63 left
    assert (s["pool_mons"], s["removed_mons"], s["drawable_mons"]) == (4314, 739, 3575)
    assert (s["pool_distinct_sets"], s["drawable_distinct_sets"], s["drawable_species"]) == (658, 572, 63)
    assert s["species_lost"] == ["arcanine", "blaziken", "dragonite", "glalie", "linoone", "weezing"]
    assert d["withdrawn_sets"]["n"] == 86 == len(d["withdrawn_sets"]["sets"])
    assert d["distinct_sets_drawn"] == 572 and d["species_drawn"] == 63 and d["exact_pool_species_six"] == 0
    assert max(int(k) for k in d["max_shared_hist"]["rc72b"]) <= 2
    assert all(v["at_ge_5"] == 0 for v in d["near_duplicates_vs_breadth_corpora"].values())
    assert set(manifest["candidate_refusals"]) == set(B.CANDIDATE_RULES)
    assert set(manifest["slot_refusals"]) == set(B.SLOT_RULES)
    assert manifest["engine"]["output_unplayable"] == {k: [] for k in ("ability", "item", "move", "species")}


def test_the_union_soak_covers_this_corpus(corpus):
    """X62 has no soak of its own: all 12,000 teams are rows of ``x63_union_v1``, whose committed soak must pass."""
    from utils.team_corpora import build_x63_union as U

    man = json.loads((TC.CORPORA[U.NAME] / TC.MANIFEST_NAME).read_text())
    comp = {c["name"]: c for c in man["components"]}[NAME]
    assert (comp["tier"], comp["n"], comp["data_sha256"], comp["tier_sha256"]) == \
        ("ALL", len(corpus), corpus.data_sha256, corpus.tier_sha256)
    soak = json.loads((TC.CORPORA[U.NAME] / "soak_selfcheck.json").read_text())
    assert soak["verdict"] == "PASS" and soak["core_errors"] == [] and soak["corpus"]["data_sha256"] == man["data_sha256"]


# ----------------------------------------------------------------------------------------- the pure core
@pytest.mark.parametrize("head, want", [
    ("Kaorine (Claydol) @ Leftovers  ", "Claydol @ Leftovers  "),                  # nickname, no gender
    ("e (Swampert) (F) @ Leftovers  ", "Swampert (F) @ Leftovers  "),              # nickname + gender
    ("Airmure (Skarmory)  ", "Skarmory"),                                            # nickname, no item
    ("Tyranitar (M) @ Leftovers  ", "Tyranitar (M) @ Leftovers  "),                # gender only: unchanged
    ("Blissey @ Leftovers  ", "Blissey @ Leftovers  "),                            # plain: unchanged
    ("Skarmory  ", "Skarmory  "),                                                    # plain, no item: unchanged
])
def test_the_nickname_strip_pins_both_head_line_forms(head, want):
    block = head + "\nAbility: Keen Eye\n- Spikes"
    out = B.strip_nickname(block)
    assert out == want + "\nAbility: Keen Eye\n- Spikes"
    assert TO.parse_mon(out).key == TO.parse_mon(block).key, "the strip never moves the mon set"


def test_the_strict_text_filter_and_the_strict_hidden_power_verdict():
    assert B.strict_text_bad("Zapdos\n- Hidden Power [Ice]")
    assert not B.strict_text_bad("Zapdos\nIVs: 30 SpA\n- Hidden Power [Grass]")
    assert not B.strict_text_bad("Gengar\n- Hidden Power [Dark]"), "all-31 IVs are Dark at 70: plays as written"
    assert not B.strict_text_bad("Gengar\n- Thunderbolt")
    ok = {"hp_move": "Hidden Power Ice", "hp_type": "Ice", "power": 70, "ivs_in": {"hp": 31}, "ivs_after_validate": {"hp": 31}}
    assert B.hp_bad_strict({"sets": [ok, {"hp_move": None}]}) == []
    for bad in ({"power": 68}, {"hp_type": "Water"}, {"ivs_after_validate": {"hp": 30}}):
        assert B.hp_bad_strict({"sets": [dict(ok, **bad)]}) == [dict(ok, **bad)]


def test_the_draw_is_deterministic_and_streams_differ(inputs):
    import collections as C

    a = B.draw_team(17, inputs.profile, inputs.occ, inputs.lift, inputs.refs, C.Counter())
    b = B.draw_team(17, inputs.profile, inputs.occ, inputs.lift, inputs.refs, C.Counter())
    assert a == b and a is not None
    other = B.draw_team(17, inputs.profile, inputs.occ, inputs.lift, inputs.refs, C.Counter(), stream="other")
    assert [x[0] for x in other] != [x[0] for x in a]


def test_the_profile_is_the_prototypes(inputs):
    """c_pool(s) counts ALL the pool's mons of a species with a STRICT occurrence (strict_draw_sim.py; the spec's
    'STRICT-class' count moved the species TVD from 0.048 to 0.109 in a dry draw, so it is NOT what §2.2 measured)."""
    allc = collections.Counter(m.species for t in inputs.pool for m in t.mons)
    assert inputs.profile == {s: allc[s] for s in inputs.occ}


@pytest.mark.slow  # Node (the pinned validator + hp_set_check) over ~12.4k teams + the engine's coverage oracle
@pytest.mark.integration
def test_the_committed_corpus_reproduces():
    assert B.main(["--check"]) == 0
