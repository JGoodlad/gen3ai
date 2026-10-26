"""Every dex number lies under its embedding table's rows (``gen3_vocab_bounds_v1``, GIGO audit C-F1).

The audit measured the headroom (species 386 of 400, moves 370 of 400, items 499 of 600, abilities 76 of 100,
types 18 of 19 in use) but nothing pinned it: a regenerated dex, or a new species past the table, would reach the
embedding as an out-of-range index (IndexError on CPU, a device assert on CUDA) or, through any clamp, collide
silently with another row. This file is the STATIC half (the shipped data, the layout, the constants and the
table the model actually builds) and the throwing guard's own cases; the Rust encoder holds the twin
(``encoder/data.rs``, a ``cargo test``) and ``rust_core_obs_layout_test`` holds the two sides' constants equal.
"""
from __future__ import annotations

import copy
import json

import pytest

from agents.observation import constants as C
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.types import TypeEncoder
from agents.observation.vocab_bounds import VOCABS, VocabBoundError, check_vocab_bounds
from utils.paths import repo_path


@pytest.fixture(scope="module")
def mappings():
    return load_mappings()


def test_the_layout_reads_the_declared_vocabularies():
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    assert (layout["max_species"], layout["max_moves"], layout["max_items"], layout["max_abilities"],
            layout["max_types"]) == (C.VOCAB_SPECIES, C.VOCAB_MOVES, C.VOCAB_ITEMS, C.VOCAB_ABILITIES, C.VOCAB_TYPES)
    # the removed, unused `MAX_*_ID` constants disagreed with the layout; nothing re-introduces them
    assert not any(hasattr(C, n) for n in ("MAX_SPECIES_ID", "MAX_MOVE_ID", "MAX_ITEM_ID", "MAX_ABILITY_ID"))


@pytest.mark.parametrize("category", sorted(VOCABS))
def test_every_dex_num_is_under_its_embedding_table(mappings, category):
    rows, key = VOCABS[category]
    nums = {name: e["num"] for name, e in mappings[category].items() if isinstance(e, dict) and "num" in e}
    assert nums, f"no {category} rows were read"
    worst = max(nums.items(), key=lambda kv: kv[1])
    assert min(nums.values()) >= 0
    assert worst[1] < rows, f"{category} {worst[0]!r} has num {worst[1]} but the {key} table has {rows} rows"
    # the guard itself agrees, and reports the headroom it measured
    assert check_vocab_bounds(mappings)[category] == worst[1]


def test_the_embedding_tables_are_built_from_the_declared_rows():
    import torch
    from agents.model.extractor_ctx import Embeddings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    emb = Embeddings(layout)
    assert emb.species_embedding.num_embeddings == C.VOCAB_SPECIES
    assert emb.move_embedding.num_embeddings == C.VOCAB_MOVES
    assert emb.item_embedding.num_embeddings == C.VOCAB_ITEMS
    assert emb.ability_embedding.num_embeddings == C.VOCAB_ABILITIES
    assert emb.type_embedding.num_embeddings == C.VOCAB_TYPES
    # the last legal row of each table indexes; one past it raises (nothing clamps on the way in)
    for table, rows in ((emb.species_embedding, C.VOCAB_SPECIES), (emb.item_embedding, C.VOCAB_ITEMS)):
        table(torch.tensor([rows - 1]))
        with pytest.raises(IndexError):
            table(torch.tensor([rows]))


def test_type_ids_are_under_the_ops_type_axis_and_the_table():
    from agents.model.damage_tables import N_TYPE_IDX

    ids = TypeEncoder.TYPE_TO_IDX.values()
    assert min(ids) >= 1
    assert max(ids) < N_TYPE_IDX <= C.VOCAB_TYPES


def test_the_dex_files_the_rust_encoder_reads_hold_the_same_bounds():
    """The Rust encoder reads ``data/pokemon/gen3_*.json`` itself (``encoder/data.rs``), not the facade: pin the
    raw files, so a num the facade would drop but Rust would write still fails here."""
    for fname, category in (("gen3_species.json", "species"), ("gen3_moves.json", "moves"),
                            ("gen3_items.json", "items"), ("gen3_abilities.json", "abilities")):
        rows = VOCABS[category][0]
        raw = json.loads(repo_path("data", "pokemon", fname).read_text())
        nums = [v["num"] for v in raw.values() if isinstance(v, dict)]
        assert nums and all(isinstance(n, int) and 0 <= n < rows for n in nums), (fname, max(nums), rows)


@pytest.mark.parametrize("category", sorted(VOCABS))
def test_the_guard_throws_on_a_num_at_or_past_the_table_and_on_a_bad_num(mappings, category):
    rows = VOCABS[category][0]
    name = next(iter(mappings[category]))
    for bad in (rows, rows + 100, -1, 1.5, None):
        broken = copy.copy(mappings)
        broken[category] = {**mappings[category], name: {**mappings[category][name], "num": bad}}
        with pytest.raises(VocabBoundError, match=category):
            check_vocab_bounds(broken)
        with pytest.raises(VocabBoundError):          # the encoder refuses it at construction
            Gen3ObservationEncoder(broken)
    # NON-VACUITY: the last legal row passes
    ok = copy.copy(mappings)
    ok[category] = {**mappings[category], name: {**mappings[category][name], "num": rows - 1}}
    check_vocab_bounds(ok)
