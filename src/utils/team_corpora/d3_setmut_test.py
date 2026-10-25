"""``d3_setmut_v1`` (X41 / D3's set-mutation corpus) is what its manifest and its design say: every team is a POOL team
with 1-2 mons' sets redrawn from SMOGON usage, the other mons byte-identical, no held-out panel near, level 100, soaked
with the label path; and the mutation itself is deterministic per seed."""
import collections
import json
import random

import pytest

from main import heldout_panels as H
from utils import team_corpora as TC
from utils.paths import repo_path
from utils.team_corpora import build_d3_setmut as B

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
def pool():
    from utils.team_loader import TeamLoader

    return {B.source_id(t): t for t in TeamLoader(str(repo_path("data", "teams"))).get_all_teams()}


def test_the_corpus_verifies_and_its_manifest_accounts_for_every_source_variant(corpus, rows, manifest):
    assert TC.load_corpus(NAME, B.TIER) == corpus and len(corpus) == manifest["tiers"][B.TIER]["n"] > 2000
    assert [r["i"] for r in rows] == list(range(len(rows)))
    src = manifest["pool"]["sources"] - manifest["pool"]["source_drops"]["source_no_eligible_mon"]["n"]
    assert len(rows) + manifest["exhausted"]["n"] == src * B.VARIANTS_PER_TEAM
    assert len({(r["source"], r["variant"]) for r in rows}) == len(rows)


def test_every_variant_is_its_pool_source_with_k_mons_redrawn_and_the_rest_byte_identical(rows, pool):
    from main.anchors.team_overlap import parse_team

    ks = collections.Counter()
    for r in rows:
        src = pool[r["source"]]
        sb, vb = B.split_blocks(src), B.split_blocks(r["text"])
        slots = r["slots"]
        assert len(vb) == 6 and 1 <= len(slots) <= max(B.K_CHOICES), r["id"]
        assert [vb[i] for i in range(6) if i not in slots] == [sb[i] for i in range(6) if i not in slots], r["id"]
        st, vt = parse_team(src), parse_team(r["text"])
        assert [m.species for m in st.mons] == [m.species for m in vt.mons], r["id"]
        assert all(st.mons[i].key != vt.mons[i].key for i in slots), r["id"]      # a NEW mon set per mutated slot
        assert len(st.key & vt.key) == 6 - len(slots), r["id"]
        ks[len(slots)] += 1
    assert set(ks) == set(B.K_CHOICES)


def test_every_redrawn_set_comes_from_the_smogon_priors_of_its_species(rows):
    """Moves from the species' Smogon move prior (Hidden Power: a type in its prior or the format's, by the repair
    rule), the item from its item prior, nature + EVs ONE of its usage spreads; the ability is the source's."""
    from agents.gen3_data import priors
    from main.anchors.team_overlap import parse_team

    for r in rows[:600]:
        team = parse_team(r["text"])
        for s in r["slots"]:
            m, block = team.mons[s], B.split_blocks(r["text"])[s]
            sp = B.block_species(block)
            mp = priors.moves(sp)
            for mv in m.moves:
                assert mv in mp or (mv.startswith(B.HP) and any(k.startswith(B.HP) for k in mp)), (r["id"], sp, mv)
            assert (m.item or "nothing") in priors.items(sp), (r["id"], sp, m.item)
            spreads = {(n.lower(), tuple(e)) for n, e, _ in priors.spreads(sp)}
            assert (m.nature, m.evs) in spreads, (r["id"], sp, m.nature, m.evs)


def test_the_corpus_holds_no_held_out_panel_neighbour_no_pool_copy_no_duplicate_and_no_level_line(corpus, manifest):
    from main.anchors.team_overlap import Reference, parse_items, parse_team
    from utils.team_loader import TeamLoader

    items = list(zip(corpus.ids, corpus.texts))
    assert H.contamination(items) == []
    assert manifest["panel_registry"]["sha256"] == H.registry_sha256(), "the registry changed: rebuild"
    pool = TeamLoader(str(repo_path("data", "teams"))).get_all_teams()
    ref = Reference(parse_items([(str(i), t) for i, t in enumerate(pool)]))
    keys = [parse_team(t).key for t in corpus.texts]
    assert not [i for i, k in zip(corpus.ids, keys) if ref.keys[k]]
    assert len(set(keys)) == len(keys)
    assert not [i for i, t in zip(corpus.ids, corpus.texts) if TC.label_path_defects(t) or "Level:" in t]


def test_the_manifest_stamps_the_priors_it_drew_from(manifest):
    import hashlib

    for f, sha in manifest["priors"].items():
        assert hashlib.sha256(repo_path("data", "pokemon", f).read_bytes()).hexdigest() == sha, f"{f} moved: rebuild"


def test_the_committed_label_path_soak_passes_this_corpus(corpus):
    from main.x35.kit import D1_SOAK_SCHEMA

    soak = json.loads((TC.CORPORA[NAME] / "soak_selfcheck.json").read_text())
    assert soak["schema"] == D1_SOAK_SCHEMA and soak["labels"] and soak["verdict"] == "PASS"
    assert soak["core_errors"] == [] and soak["battles_finished"] == soak["battles_planned"] >= 2 * len(corpus)
    assert soak["corpus"]["data_sha256"] == corpus.data_sha256 and soak["corpus"]["tier"] == B.TIER


# --------------------------------------------------------------------------------------- the mutation, pure
_SKARM = "Skarmory @ Leftovers\nAbility: Keen Eye\nEVs: 252 HP / 252 SpD / 4 Spe\nCareful Nature\n- Spikes\n- Roar\n" \
         "- Toxic\n- Drill Peck"


def _prior(**kw):
    base = dict(moves=(("drillpeck", 0.5), ("hiddenpower", 0.9), ("protect", 0.6), ("roar", 0.9), ("spikes", 1.0),
                       ("toxic", 0.6)),
                items=(("leftovers", 1.0),), spreads=(("Careful", (252, 0, 4, 0, 252, 0), 1.0),))
    base.update(kw)
    return B.SpeciesPrior(**base)


def test_the_mutation_is_deterministic_per_seed_and_keeps_the_species_ability_and_unmutated_mons():
    blocks = [_SKARM] * 6
    pri = [_prior()] * 6
    a, b = B.mutate_team(blocks, pri, "pool_x", 0, 0), B.mutate_team(blocks, pri, "pool_x", 0, 0)
    assert a == b
    outs = {B.mutate_team(blocks, pri, "pool_x", v, t)[0] for v in range(4) for t in range(3)}
    assert len(outs) > 1, "variants and attempts draw from different streams"
    text, slots, recs = a
    assert len(slots) in B.K_CHOICES and len(recs) == len(slots)
    for i, blk in enumerate(B.split_blocks(text)):
        assert blk.startswith("Skarmory @ Leftovers") and "Ability: Keen Eye" in blk
        if i not in slots:
            assert blk == _SKARM


def test_a_drawn_hidden_power_is_typed_by_the_repair_rule_and_ineligible_mons_are_never_drawn():
    pri = _prior(moves=(("hiddenpower", 1.0), ("roar", 1.0), ("spikes", 1.0), ("toxic", 1.0)))
    new, rec = B.mutate_block(_SKARM, pri, random.Random(1), 7)
    assert rec["hidden_power"] and "- Hidden Power [" in new and "IVs:" in new
    assert "Level:" not in new and "Careful Nature" in new and "EVs: 252 HP / 4 Def / 252 SpD" in new
    thin = _prior(moves=(("roar", 1.0), ("spikes", 1.0), ("toxic", 1.0)))
    assert not thin.eligible
    assert B.mutate_team([_SKARM] * 6, [thin] * 6, "pool_x", 0, 0) is None
    text, slots, _ = B.mutate_team([_SKARM] * 6, [thin] * 5 + [_prior()], "pool_x", 0, 0)
    assert slots == [5]


@pytest.mark.slow  # Node (the pinned validator) + the engine's coverage oracle; ≈ 10 s
@pytest.mark.integration
def test_the_committed_corpus_reproduces_from_the_pool_and_the_priors():
    assert B.main(["--check"]) == 0
