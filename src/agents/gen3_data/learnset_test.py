"""Unit tests for the gen3 legal-movepool facade (`gen3_data.learnset`)."""
import pytest

from agents import gen3_data
from agents.gen3_data import learnset


def test_known_species_legal_and_illegal_moves():
    # Skarmory: Spikes / Drill Peck / Roar are its staples; it cannot learn Fire Blast or Surf.
    assert learnset.is_legal("skarmory", "spikes")
    assert learnset.is_legal("skarmory", "drillpeck")
    assert not learnset.is_legal("skarmory", "fireblast")
    assert not learnset.is_legal("skarmory", "surf")
    # Snorlax: Body Slam / Earthquake / Rest legal; Spikes illegal.
    assert learnset.is_legal("snorlax", "bodyslam")
    assert learnset.is_legal("snorlax", "earthquake")
    assert not learnset.is_legal("snorlax", "spikes")


def test_hidden_power_is_legal_for_hp_users():
    # HP users carry the bare 'hiddenpower' learnset entry (the typed variant is an IV choice).
    assert learnset.is_legal("zapdos", "hiddenpower")


def test_get_legal_moves_returns_set_for_known_species():
    moves = learnset.get_legal_moves("salamence")
    assert moves is not None
    assert "fireblast" in moves and "brickbreak" in moves
    assert "spikes" not in moves
    # Every legal id is a real gen3 move (no later-gen / typo leak through the builder).
    for mid in moves:
        assert gen3_data.moves.get(mid) is not None


def test_unknown_species_is_tolerant_no_constraint():
    # The tolerance contract: unknown species → None / True, so the gate never wrongly prunes.
    assert learnset.get_legal_moves("notarealmon") is None
    assert learnset.is_legal("notarealmon", "anything")


def test_legal_moves_for_species_raises_on_unknown():
    with pytest.raises(KeyError):
        learnset.legal_moves_for_species("notarealmon")


def test_full_gen3_pokedex_covered():
    # Every gen3 species with a movepool should be present (386 base forms in the pokedex).
    raw = learnset.raw()
    assert len(raw) == 386
    # A representative sample of OU mons all have non-empty movepools.
    for sid in ("tyranitar", "blissey", "metagross", "suicune", "gengar"):
        assert len(raw[sid]) > 0


# --- gen3_learnset_prevo_chain_v1 -------------------------------------------------------------- #
# A species' legal movepool is the union over its whole PRE-EVOLUTION chain (Showdown's own validator walks
# `getFullLearnset`), plus the whole sketchable dex for a Sketch learner. The table used to be each species' OWN
# entry alone, so every move reachable only through a pre-evolution (egg / event / level-up) read as ILLEGAL:
# 1151 (species, move) pairs over 182 species, 9.2 % of the move uses of the training team pools and 928 of the
# Smogon usage pairs, with Dragon Dance on Tyranitar the named case (its own entry says `["9M","8M"]`; Larvitar's
# says `[..., "3E", "3S0"]`). The ground truth is `src/rust_sim/harness/dump_gen3_learnset.js --check`, which
# asks Showdown's validator; these tests are the always-on, node-free guards.

def test_dragon_dance_tyranitar_is_legal_via_larvitar():
    """THE named regression: Tyranitar hatches as a Larvitar (gen-3 egg move `3E`, event `3S0`) and keeps the move."""
    assert learnset.is_legal("tyranitar", "dragondance")
    assert learnset.is_legal("pupitar", "dragondance")      # the middle stage inherits it too
    assert learnset.is_legal("larvitar", "dragondance")     # the source


@pytest.mark.parametrize("species,move", [
    ("tyranitar", "pursuit"),       # Larvitar egg move
    ("swampert", "hydropump"),      # Mudkip / Marshtomp level-up
    ("swampert", "curse"),
    ("breloom", "spore"),           # Shroomish level-up
    ("gengar", "willowisp"),        # Gastly level-up
    ("jolteon", "batonpass"),       # Eevee
    ("cloyster", "rapidspin"),      # Shellder
    ("blissey", "wish"),            # Chansey egg move
    ("wobbuffet", "encore"),        # Wynaut
    ("azumarill", "bellydrum"),     # Marill egg move
])
def test_a_move_learned_by_a_pre_evolution_is_legal_after_evolving(species, move):
    assert learnset.is_legal(species, move)


@pytest.mark.parametrize("species,move", [
    ("larvitar", "icebeam"),        # an evolution's TM does not flow BACKWARD
    ("larvitar", "flamethrower"),
    ("pupitar", "icebeam"),
])
def test_an_evolutions_move_is_not_legal_for_its_pre_evolution(species, move):
    assert not learnset.is_legal(species, move)
    assert learnset.is_legal("tyranitar", move)             # the precondition: the evolution really has it


def test_the_chain_stops_at_a_pre_evolution_newer_than_gen_3():
    """Snorlax's Munchlax and Chansey's Happiny are gen 4: Showdown cuts the chain there, so a move only
    Munchlax learns (Wish) is NOT a gen-3 Snorlax move — while Chansey (gen 3) still passes its own to Blissey."""
    assert gen3_data.species.get("munchlax") is None, "munchlax became a gen-3 species: the cut moved"
    assert not learnset.is_legal("snorlax", "wish")
    assert learnset.is_legal("blissey", "wish")


def test_smeargle_can_sketch_every_sketchable_move():
    """Sketch: any move that is not `nosketch` (Struggle / Sketch itself are the gen-3 exceptions)."""
    moves = learnset.get_legal_moves("smeargle")
    assert moves is not None
    for mid in ("spore", "spikes", "explosion", "batonpass", "destinybond", "hiddenpower"):
        assert mid in moves, mid
    assert "struggle" not in moves
    # the typed Hidden Powers are OUR ids: the table carries only the bare move
    assert not any(m.startswith("hiddenpower") and m != "hiddenpower" for m in moves)
    assert len(moves) > 300


def _training_pool_sets():
    """(label, species id, move id) for every set of every team the trainer / eval bots can draw (the
    `valid`-marked manifest entries) and of the specialist bot teams."""
    import json

    from agents.gen3_data import team_legality
    from utils.paths import repo_path

    data = repo_path("data")
    files = []
    for manifest in sorted((data / "teams").rglob("teams.json")):
        for entry in json.loads(manifest.read_text()):
            if entry.get("file") and entry.get("valid") is not False:
                files.append(data / entry["file"])
    files += sorted((data / "teams" / "specialist").glob("*.txt"))
    aliases = gen3_data.moves.move_aliases()
    for path in files:
        for s in team_legality.parse_paste(path.read_text(encoding="utf-8")):
            for mid in s.moves:
                mid = aliases.get(mid, mid)
                yield f"{path.name}", s.species, "hiddenpower" if mid.startswith("hiddenpower") else mid


def test_every_move_a_training_pool_team_carries_is_legal_for_its_species():
    """THE POOL GUARD: a move a real, Showdown-validated team runs can never be called illegal by the table. Before
    the fix 1664 of the pool's 18,113 move uses (55 distinct pairs, every pool) failed this."""
    uses = list(_training_pool_sets())
    assert len(uses) > 10_000, "the pool scan found no teams: the guard would be vacuous"
    bad = sorted({(sp, mv) for _, sp, mv in uses
                  if gen3_data.species.get(sp) is not None and not learnset.is_legal(sp, mv)})
    assert not bad, f"{len(bad)} pool (species, move) pairs the learnset calls illegal: {bad[:25]}"
    assert {sp for _, sp, _ in uses if gen3_data.species.get(sp) is None} == set(), "a pool set names an unknown species"


def test_every_move_in_the_smogon_usage_data_is_legal_for_its_species():
    """The same guard over the Smogon chaos moves (`priors.moves`): a move the metagame runs on a species must be
    one the species can learn. Before the fix 928 pairs (usage mass 73 move-slots) failed — e.g. Pupitar Dragon
    Dance at 99.95 % — and the move-belief prior only survived because a recorded usage OVERRIDES the legality
    gate (`build_move_prior_logits`)."""
    checked = 0
    bad = []
    for sid in gen3_data.species.base_form_ids():
        for mid in gen3_data.priors.moves(sid):
            checked += 1
            base = "hiddenpower" if mid.startswith("hiddenpower") else mid
            if not learnset.is_legal(sid, base):
                bad.append((sid, mid))
    assert checked > 5_000, "the usage scan found no moves: the guard would be vacuous"
    assert not bad, f"{len(bad)} Smogon usage pairs the learnset calls illegal: {bad[:25]}"
