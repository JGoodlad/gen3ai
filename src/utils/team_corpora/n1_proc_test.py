"""``n1_proc_v1`` (X61 / N1's procedural corpus) and its held-out panel ``n1p72`` are what the manifests and the design
say: >= 10,000 whole teams drawn from SMOGON usage only, level 100, prefix tiers, no held-out panel neighbour at REVIEW
A's radius 3 (``n1p72`` included), no pool copy, no duplicate, soaked with the label path; the draw is deterministic."""
import hashlib
import json

import pytest

from main import heldout_panels as H
from utils import team_corpora as TC
from utils.paths import repo_path
from utils.team_corpora import build_n1_proc as B

NAME = B.NAME


@pytest.fixture(scope="module")
def corpus():
    return TC.load_corpus(NAME)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((TC.CORPORA[NAME] / TC.MANIFEST_NAME).read_text())


@pytest.fixture(scope="module")
def panel():
    d = repo_path(*B.PANEL_REL.split("/"))
    return [(p.name, p.read_text()) for p in sorted(d.iterdir())]


def test_the_corpus_has_ten_thousand_teams_and_prefix_tiers(corpus, manifest):
    assert corpus.tier == B.DEFAULT_TIER and len(corpus) == B.N_CORPUS >= 10_000
    for name, n in B.TIERS:
        c = TC.load_corpus(NAME, name)
        assert len(c) == n == manifest["tiers"][name]["n"] and c.texts == corpus.texts[:n]
    _, rows, _, _ = TC.read_dir(TC.CORPORA[NAME])
    assert [r["i"] for r in rows] == list(range(len(rows)))
    assert [r["candidate"] for r in rows] == sorted(r["candidate"] for r in rows), "rows are in candidate order"


def test_no_team_is_near_a_held_out_panel_at_radius_three_nor_a_pool_copy_nor_a_duplicate(corpus, panel, manifest):
    from main.anchors.team_overlap import Reference, parse_items, parse_team
    from utils.team_loader import TeamLoader

    items = list(zip(corpus.ids, corpus.texts))
    assert H.hits(items, H.panel_names(held_out_only=True), min_shared=B.PANEL_MIN_SHARED) == []
    own = Reference(parse_items(panel))
    teams = [parse_team(t) for t in corpus.texts]
    assert max(own.max_shared(t)[0] for t in teams) < B.PANEL_MIN_SHARED, "a corpus team is near n1p72"
    assert manifest["panel_registry"]["sha256"] == H.registry_sha256(), "the registry changed: rebuild"
    pool = TeamLoader(str(repo_path("data", "teams"))).get_all_teams()
    ref = Reference(parse_items([(str(i), t) for i, t in enumerate(pool)]))
    assert not [i for i, t in zip(corpus.ids, teams) if ref.keys[t.key]]
    keys = {t.key for t in teams} | {t.key for t in own.teams}
    assert len(keys) == len(teams) + len(own.teams), "a duplicate inside the corpus or between the corpus and n1p72"
    assert not [i for i, t in zip(corpus.ids, corpus.texts) if TC.label_path_defects(t) or "Level:" in t]


def test_the_panel_is_registered_by_its_manifest_and_held_out_from_the_registered_panels(panel):
    man = json.loads(repo_path(*B.PANEL_MANIFEST_REL.split("/")).read_text())
    assert man["schema"] == B.PANEL_SCHEMA and man["n"] == len(panel) == B.N_PANEL
    assert {k: v["sha256"] for k, v in man["files"].items()} == \
        {k: hashlib.sha256(t.encode()).hexdigest() for k, t in panel}
    assert H.hits(panel, H.panel_names(held_out_only=True), min_shared=B.PANEL_MIN_SHARED) == []


def test_every_set_comes_from_the_smogon_priors_of_its_species(corpus):
    """Each mon: a species with a Smogon usage entry; moves from its move prior (Hidden Power: a type of its prior or
    the format's, by the repair rule); the item from its item prior; the ability from its ability prior; nature + EVs
    ONE of its usage spreads; six distinct species."""
    from agents.gen3_data import priors
    from main.anchors.team_overlap import parse_team

    usage = priors.species_usage()
    for text in corpus.texts[:600]:
        team = parse_team(text)
        assert len(team.mons) == 6 and len({m.species for m in team.mons}) == 6
        for m, block in zip(team.mons, text.split("\n\n")):
            sp = m.species
            assert sp in usage, sp
            mp = priors.moves(sp)
            for mv in m.moves:
                assert mv in mp or (mv.startswith(B.HP) and any(k.startswith(B.HP) for k in mp)), (sp, mv)
            assert (m.item or "nothing") in priors.items(sp), (sp, m.item)
            ab = block.split("\n")[1]
            assert ab.startswith("Ability: ") and any(
                B.ability_name(a) == ab[len("Ability: "):] for a in priors.ability(sp)), (sp, ab)
            assert (m.nature, m.evs) in {(n.lower(), tuple(e)) for n, e, _ in priors.spreads(sp)}, (sp, m.nature)


def test_the_manifest_stamps_the_priors_it_drew_from_and_describes_coverage(manifest):
    for f, sha in manifest["priors"].items():
        assert hashlib.sha256(repo_path("data", "pokemon", f).read_bytes()).hexdigest() == sha, f"{f} moved: rebuild"
    d = manifest["describe"]
    assert d["n_teams"] == B.N_CORPUS and d["species"]["drawn"] >= 150
    assert d["species"]["pearson_team_freq_vs_smogon_usage_pct"] > 0.9
    assert max(int(k) for k in d["max_shared_mon_sets_with_any_heldout_panel_hist"]) < B.PANEL_MIN_SHARED


def test_the_committed_label_path_soak_passes_this_corpus(corpus):
    from main.x35.kit import D1_SOAK_SCHEMA

    soak = json.loads((TC.CORPORA[NAME] / "soak_selfcheck.json").read_text())
    assert soak["schema"] == D1_SOAK_SCHEMA and soak["labels"] and soak["verdict"] == "PASS"
    assert soak["core_errors"] == [] and soak["battles_finished"] == soak["battles_planned"] >= 2 * len(corpus)
    assert soak["corpus"]["data_sha256"] == corpus.data_sha256 and soak["corpus"]["tier"] == B.DEFAULT_TIER


# ----------------------------------------------------------------------------------------- the draw, pure
@pytest.fixture(scope="module")
def universe():
    empty = frozenset()
    return B.build_universe({"species": empty, "move": empty, "item": empty, "ability": empty})


def test_the_draw_is_deterministic_per_stream_and_index_and_streams_differ(universe):
    a, b = B.draw_team("corpus", 17, universe), B.draw_team("corpus", 17, universe)
    assert a == b
    assert B.draw_team("panel", 17, universe)[0] != a[0]
    assert len({B.draw_team("corpus", j, universe)[0] for j in range(20)}) == 20


def test_species_are_distinct_and_a_lead_without_teammates_falls_back_to_usage(universe):
    import random

    for j in range(50):
        sp = B.draw_species(random.Random(j), universe)
        assert len(sp) == 6 == len(set(sp)) and set(sp) <= set(universe.species)
    lone = B.Universe(species=("a", "b", "c", "d", "e", "f", "g"), usage=(1.0,) * 7, names={}, sets={}, abilities={},
                      teammates={k: {} for k in "abcdefg"})
    sp = B.draw_species(random.Random(0), lone)
    assert len(set(sp)) == 6


def test_a_drawn_hidden_power_is_typed_and_ivs_are_written(universe):
    hp_sets = [r for j in range(200) for r in B.draw_team("corpus", j, universe)[1] if r["hidden_power"]]
    assert hp_sets, "200 teams draw at least one Hidden Power"
    text = next(B.draw_team("corpus", j, universe)[0] for j in range(200)
                if any(r["hidden_power"] for r in B.draw_team("corpus", j, universe)[1]))
    assert "- Hidden Power [" in text and "IVs:" in text and "Level:" not in text


@pytest.mark.slow  # Node (the pinned validator) over ~13k teams + the engine's coverage oracle; minutes
@pytest.mark.integration
def test_the_committed_corpus_and_panel_reproduce_from_the_priors():
    assert B.main(["--check"]) == 0


def test_soak_chunks_merge_only_when_they_tile_the_rounds_with_distinct_seeds():
    from utils.team_corpora import soak as S

    def chunk(r0, k, seed, verdict="PASS", errors=()):
        c = {"schema": S.SOAK_SCHEMA, "corpus": {"corpus": "x"}, "labels": ["spread"], "profile": "selfcheck",
             "n_envs": 8, "threads": 3, "turn_limit": 250, "opponent": S.BOT, "rounds": k, "seed": seed,
             "battles_planned": 10 * k, "battles_finished": 10 * k, "decisions_p1": 100 * k, "seats": "s",
             "forfeits": 1, "end_turn": {"median": 20 + r0, "max": 250}, "core_errors": list(errors), "wall_s": 1.0,
             "verdict": verdict}
        if r0:
            c["first_round"] = r0
        return c

    m = S.merge([chunk(5, 5, 3803), chunk(0, 5, 3802)])
    assert m["verdict"] == "PASS" and m["rounds"] == 10 and m["battles_finished"] == m["battles_planned"] == 100
    assert m["seed"] == [3802, 3803] and m["forfeits"] == 2 and [c["first_round"] for c in m["chunks"]] == [0, 5]
    assert S.merge([chunk(0, 5, 1), chunk(5, 5, 2, "FAIL", [{"error": "e"}])])["verdict"] == "FAIL"
    for bad in ([chunk(0, 5, 1), chunk(4, 5, 2)], [chunk(0, 5, 1), chunk(6, 5, 2)], [chunk(0, 5, 1), chunk(5, 5, 1)],
                [chunk(1, 5, 1)]):
        with pytest.raises(ValueError):
            S.merge(bad)
