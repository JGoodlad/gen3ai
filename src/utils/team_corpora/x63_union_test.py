"""``x63_union_v1`` (X63's union corpus) is ``x62_recomb_v1@ALL`` + ``n1_proc_v1@K3`` + ``d3_setmut_v1@ALL``, verbatim
and in that order; it holds no duplicate six and no team within radius 3 of an ``rc72b`` team (the builder REFUSES
both: the planted-copy tests below fail if either check is removed); its manifest records the components' shas, the
panel histograms and the Frustration share; its label-path soak passes."""
import json

import pytest

from main import heldout_panels as H
from main.anchors import team_overlap as TO
from utils import team_corpora as TC
from utils.team_corpora import build_x63_union as B

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
def rc72b():
    return list(H.panel_team_texts("rc72b"))


def test_the_union_is_its_three_components_verbatim_in_order(corpus, rows, manifest):
    assert B.COMPONENTS == (("x62_recomb_v1", "ALL"), ("n1_proc_v1", "K3"), ("d3_setmut_v1", "ALL"))
    at = 0
    for (name, tier), rec in zip(B.COMPONENTS, manifest["components"]):
        c = TC.load_corpus(name, tier)
        assert (rec["name"], rec["tier"], rec["n"], rec["first_union_i"]) == (name, tier, len(c), at)
        assert (rec["tier_sha256"], rec["data_sha256"]) == (c.tier_sha256, c.data_sha256)
        part = rows[at:at + len(c)]
        assert [r["text"] for r in part] == list(c.texts) and [r["component_id"] for r in part] == list(c.ids)
        assert {r["curator"] for r in part} == {name} and {r["component_tier"] for r in part} == {tier}
        at += len(c)
    assert at == len(corpus) == len(rows) == 17_876 == manifest["tiers"]["ALL"]["n"]
    assert [r["id"] for r in rows] == [f"x63u_{i:05d}" for i in range(len(rows))]
    assert manifest["engine"]["output_unplayable"] == {k: [] for k in ("ability", "item", "move", "species")}


def test_no_duplicate_six_and_no_rc72b_neighbour_at_radius_three(corpus, rc72b, manifest):
    teams = [TO.parse_team(t) for t in corpus.texts]
    assert len({t.key for t in teams}) == len(teams)
    ref = TO.Reference(TO.parse_items(rc72b))
    assert max(ref.max_shared(t)[0] for t in teams) < B.RC72B_REFUSE
    hist = manifest["describe"]["max_shared_hist_by_panel"]
    assert set(hist) == {"heldout72", "heldout72_hpfix", "forum72", "p72", "forum216b", "n1p72", "rc72b"}
    assert hist["rc72b"]["max"] <= 2
    assert sum(hist["rc72b"]["all"].values()) == len(corpus)


def test_the_frustration_share_is_recorded_and_below_the_backport_threshold(rows, manifest):
    """x35_pack_regime_2026-10-27 condition (a): above 2 % of a corpus's teams the arm needs a packer-backport pin."""
    f = manifest["describe"]["frustration_by_text"]
    no_h0 = [r["id"] for r in rows if B.frustration_teams(r["text"])[1]]
    assert f["ids_without_happiness0"] == no_h0 and f["teams_without_happiness0"] == len(no_h0)
    assert f["share_without_happiness0"] < 0.02


def test_frustration_teams_counts_by_block():
    t = "Snorlax\n- Frustration\n\nGyarados\nHappiness: 0\n- Frustration  \n\nBlissey\n- Return"
    assert B.frustration_teams(t) == (2, 1)
    assert B.frustration_teams("Blissey\n- Return") == (0, 0)


def test_the_committed_label_path_soak_passes_the_union(corpus):
    from main.x35.kit import D1_SOAK_SCHEMA

    soak = json.loads((TC.CORPORA[NAME] / "soak_selfcheck.json").read_text())
    assert soak["schema"] == D1_SOAK_SCHEMA and soak["labels"] and soak["verdict"] == "PASS"
    assert soak["core_errors"] == [] and soak["battles_finished"] == soak["battles_planned"] >= 10 * len(corpus)
    assert soak["corpus"]["data_sha256"] == corpus.data_sha256 and soak["corpus"]["tier"] == "ALL"


# ----------------------------------------------------------------------------------------- the refusals, planted
@pytest.fixture(scope="module")
def clean():
    """Small clean components: the first 40 teams of each real component (they pass, by the committed union)."""
    out = []
    for name, tier in B.COMPONENTS:
        c = TC.load_corpus(name, tier)
        out.append((name, tier, list(c.ids[:40]), list(c.texts[:40])))
    return out


def _blocks(text):
    return [b for b in text.strip().split("\n\n") if b.strip()]


def test_a_clean_union_assembles(clean, rc72b):
    rows = B.assemble(clean, rc72b)
    assert len(rows) == 120 and [r["curator"] for r in rows[::40]] == [n for n, _ in B.COMPONENTS]


@pytest.mark.parametrize("shared", [6, 3])
def test_a_planted_rc72b_near_copy_is_refused(clean, rc72b, shared):
    """THE m3 CHECK: a union team sharing >= 3 sets with one rc72b team raises (removing the check fails this)."""
    target = _blocks(rc72b[0][1])
    filler = _blocks(clean[1][3][0])           # an n1_proc team: shares <= 2 with any rc72b team
    planted = "\n\n".join(target[:shared] + filler[shared:])
    k, lab = TO.Reference(TO.parse_items(rc72b)).max_shared(TO.parse_team(planted))
    assert k >= shared >= B.RC72B_REFUSE and lab == rc72b[0][0]
    comps = [clean[0], (clean[1][0], clean[1][1], clean[1][2] + ["planted"], clean[1][3] + [planted]), clean[2]]
    with pytest.raises(B.UnionError, match=r"rc72b near-copies at radius 3 \['n1_proc_v1:planted shares"):
        B.assemble(comps, rc72b)


@pytest.mark.parametrize("how", ["same_text", "reordered_blocks"])
def test_a_duplicate_six_across_components_is_refused(clean, rc72b, how):
    """By six-set identity, not text: the same team with its blocks in another order is still a duplicate."""
    text = clean[0][3][5]
    dup = text if how == "same_text" else "\n\n".join(reversed(_blocks(text))).strip()
    assert TO.parse_team(dup).key == TO.parse_team(text).key
    comps = [clean[0], clean[1], (clean[2][0], clean[2][1], clean[2][2] + ["dup"], clean[2][3] + [dup])]
    with pytest.raises(B.UnionError, match=r"1 duplicate six\(es\) \['d3_setmut_v1:dup == x62_recomb_v1:"):
        B.assemble(comps, rc72b)


def test_an_unstripped_component_text_is_refused(clean, rc72b):
    comps = [(clean[0][0], clean[0][1], clean[0][2][:1], [clean[0][3][0] + "  \n"])]
    with pytest.raises(B.UnionError, match="not .strip"):
        B.assemble(comps, rc72b)


@pytest.mark.slow  # Node (the pinned validator over 17,876 teams) + the engine's coverage oracle
@pytest.mark.integration
def test_the_committed_union_reproduces():
    assert B.main(["--check"]) == 0
