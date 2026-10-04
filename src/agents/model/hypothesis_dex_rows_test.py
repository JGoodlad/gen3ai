"""The X5 dex-row table's LOADER (`gen3_x5_dex_rows_v1`, U1): the committed artifact loads into a
dense, num-indexed array whose shape is the observation layout's, one row per base-form species.

Pure unit (no Rust): the byte gate that regenerates the table through the encoder, and the
real-state cross-check, are ``hypothesis_dex_rows_sim_test.py``.
"""
import numpy as np
import pytest

from agents.gen3_data import species as gs
from agents.model import hypothesis_dex_rows as H
from agents.observation import constants as C
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


@pytest.fixture(scope="module")
def layout():
    return Gen3ObservationEncoder(load_mappings()).get_layout()


@pytest.fixture(scope="module")
def table(layout):
    return H.load_hypothesis_dex_rows(layout["max_species"])


def test_the_table_has_the_layouts_shape(layout, table):
    n_slots, slot_dim = layout["parts"]["opp_team"]["reshape"]
    assert table.rows.shape == (layout["max_species"], slot_dim)
    assert table.rows.dtype == np.float32
    assert table.valid.shape == (layout["max_species"],)
    assert not table.rows.flags.writeable and not table.valid.flags.writeable


def test_one_row_per_base_form_species_at_its_num(table):
    base = {gs.species_data(sid).num: sid for sid in gs.base_form_ids()}
    got = {num: sid for num, sid in enumerate(table.species) if sid is not None}
    assert got == base
    assert int(table.valid.sum()) == len(base)
    assert not table.valid[0], "the sentinel 0 is never a species"
    assert not table.rows[~table.valid].any(), "a num no species holds is an all-zero row"


def test_every_row_is_its_species_populated_unrevealed_and_benched(table):
    nums = np.flatnonzero(table.valid)
    rows = table.rows[nums]
    assert np.array_equal(rows[:, C.POKEMON_SPECIES_OFFSET], nums.astype(np.float32)), "the species cell is the num"
    assert (rows[:, C.POKEMON_SPECIES_KNOWN_OFFSET] == 1.0).all()
    assert (rows[:, C.POKEMON_HP_OFFSET] == 1.0).all(), "full HP"
    assert (rows[:, C.POKEMON_ACTIVE_OFFSET] == 0.0).all()
    assert not rows[:, C.POKEMON_MOVES_OFFSET:C.POKEMON_HP_OFFSET].any(), "no move revealed"
    assert not rows[:, C.POKEMON_ITEMS_OFFSET:C.POKEMON_TYPES_OFFSET].any(), "no item revealed"
    assert not rows[:, C.POKEMON_CONDITION_OFFSET:C.POKEMON_MOVES_OFFSET].any(), "no status"
    assert np.isfinite(rows).all()


def test_the_declared_cells_tile_the_slot_on_the_layouts_offsets(table):
    at = 0
    for c in table.cells:
        assert c.lo == at and c.hi > c.lo, c
        assert c.klass in ("species", "default", "revealed_on_field", "on_field"), c
        assert (c.flag is not None) == (c.klass == "revealed_on_field"), c
        at = c.hi
    assert at == C.POKEMON_FULL_DIM
    by = {c.name: c for c in table.cells}
    assert by["species"].lo == C.POKEMON_SPECIES_OFFSET
    assert by["moves"].lo == C.POKEMON_MOVES_OFFSET
    assert by["hp_fraction"].lo == C.POKEMON_HP_OFFSET
    assert by["hidden_power"].lo == C.POKEMON_HP_BLOCK_OFFSET
    assert by["recency"].lo == C.POKEMON_RECENCY_OFFSET
    assert by["active"].lo == C.POKEMON_ACTIVE_OFFSET


def test_the_file_is_what_its_own_renderer_writes():
    """Format determinism without the encoder: re-rendering the loaded rows reproduces the file's
    bytes (so the sim gate's byte comparison is a comparison of VALUES, never of formatting)."""
    import json

    text = H.ARTIFACT.read_text(encoding="utf-8")
    doc = json.loads(text)
    rows = [np.asarray(e[2], dtype=np.float32) for e in doc["rows"]]
    head = {"row_dim": doc["row_dim"], "cells": doc["cells"]}
    assert H.render([(e[0], e[1]) for e in doc["rows"]], rows, head) == text


def test_a_hand_edited_cell_is_refused(layout):
    text = H.ARTIFACT.read_text(encoding="utf-8")
    i = text.index('[227, "skarmory", [227.0, ')
    edited = text[:i] + text[i:].replace("[227.0, ", "[227.0, 0.5, ", 1).replace(", 0.0]]", "]]", 1)
    with pytest.raises(ValueError, match="sha256"):
        H._parse(edited, layout["max_species"])
    with pytest.raises(ValueError, match="outside"):
        H._parse(text, 200)
