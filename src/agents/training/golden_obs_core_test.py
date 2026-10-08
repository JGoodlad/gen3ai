"""THE OBS GOLDEN on the Rust core (the value-neutrality linchpin, P6 of the poke-env retirement): the core, replaying
the banked golden battles, writes the trainee's rows with EXACTLY the committed hashes, in order — and the check has
teeth (a changed command moves the hashes; a changed hash is caught at its decision)."""
import json

import pytest

from agents.training import golden_obs_core as G

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def test_the_core_reproduces_every_committed_hash():
    golden = json.loads(G.HASHES_PATH.read_text())
    got = G.vector_hashes(G.core_rows())
    assert len(got) == golden["n_decisions"] == len(golden["hashes"]), (len(got), golden["n_decisions"])
    first, n = G.compare(got, golden["hashes"])
    assert n == 0, f"{n} decisions differ from the committed golden, first at {first}"
    assert len(G.load_battles()) == json.loads(G.BATTLES_PATH.read_text())["n_battles"] >= 6


def test_a_changed_battle_moves_the_hashes():
    battles = G.load_battles()
    b = battles[0]
    # swap one of p1's moves for another legal-looking choice late in the battle: the game (and the rows) diverge
    k = max(i for i, c in enumerate(b.commands) if c[0] == "p1" and c[1].startswith("move "))
    b.commands = b.commands[:k]
    golden = json.loads(G.HASHES_PATH.read_text())["hashes"]
    first, n = G.compare(G.vector_hashes(G.core_rows([b])), golden)
    assert n > 0 and first >= 0


def test_compare_names_the_first_difference():
    assert G.compare(["a", "b", "c"], ["a", "b", "c"]) == (-1, 0)
    assert G.compare(["a", "x", "c"], ["a", "b", "c"]) == (1, 1)
    assert G.compare(["a", "b"], ["a", "b", "c"]) == (2, 1)
