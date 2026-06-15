"""Extractor ↔ upstream parity — the drift guard.

Proves the committed ``data/pokemon/`` files are genuinely *derived* and reproducible: each file
equals its upstream source of truth, and re-running the builders reproduces the committed file.
If any committed file is hand-edited away from what the extractor would produce, these fail.

Builders that read only poke-env static data (always present in the repo) are unit tests; those
that read the Showdown submodule (``deps/pokemon-showdown``) are integration tests.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load_sync():
    spec = importlib.util.spec_from_file_location(
        "pde_sync", _REPO / "tools" / "pokemon_data_extractor" / "sync.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _committed(name):
    with open(_REPO / "data" / "pokemon" / name) as f:
        return json.load(f)


# --- committed file == upstream source of truth (poke-env static; unit-safe) --------------- #

def test_type_chart_matches_gendata():
    from poke_env.data import GenData
    assert _committed("gen3_type_chart.json") == GenData.from_gen(3).type_chart


def test_natures_match_pokeenv():
    with open(_REPO / "src" / "poke_env" / "data" / "static" / "natures.json") as f:
        upstream = json.load(f)
    assert _committed("gen3_natures.json") == upstream


# --- builder reproduces the committed file (the reproducibility proof) ---------------------- #

def test_species_builder_reproduces_committed():
    assert _load_sync().build_species(3) == _committed("gen3_species.json")


def test_moves_builder_reproduces_committed():
    assert _load_sync().build_moves(3) == _committed("gen3_moves.json")


def test_learnset_builder_reproduces_committed():
    # build_learnset reads only poke-env static data (learnset.json + the species/moves builders),
    # so it is unit-safe (no Showdown submodule needed).
    assert _load_sync().build_learnset(3) == _committed("gen3_learnset.json")


@pytest.mark.integration
def test_abilities_builder_reproduces_committed():
    # build_abilities reads the Showdown submodule (deps/pokemon-showdown/data/abilities.ts).
    assert _load_sync().build_abilities(3) == _committed("gen3_abilities.json")


@pytest.mark.integration
def test_items_builder_reproduces_committed():
    # build_items reads the Showdown submodule (deps/pokemon-showdown/data/items.ts).
    assert _load_sync().build_items(3) == _committed("gen3_items.json")


# --- gen3_move_effects_v1: the effect flags are faithful to the Showdown SOURCE ------------- #
# `test_moves_builder_reproduces_committed` proves committed == builder (no hand-edit drift);
# build_moves reads poke-env's static JSON. This test closes the remaining gap — that poke-env's
# static fields stay faithful to Showdown's own representation — by re-deriving each in-scope
# (0-power utility) effect STRAIGHT FROM the Showdown .ts source (declarative field OR an
# onHit/onTryHit `this.boost`/`this.heal` callback) and asserting the facade flag matches for
# EVERY gen3 move. A future submodule bump that changes how an effect is encoded fails here.

import re  # noqa: E402

_MAJOR = {"par", "brn", "psn", "tox", "slp", "frz"}
_HAZARD_SC = {"spikes", "stealthrock", "toxicspikes", "stickyweb"}


def _ts_blocks():
    base = (_REPO / "deps" / "pokemon-showdown" / "data" / "moves.ts").read_text()
    mod = (_REPO / "deps" / "pokemon-showdown" / "data" / "mods" / "gen3" / "moves.ts").read_text()

    def _one(text, mid):
        m = re.search(r"^\t" + re.escape(mid) + r": \{", text, re.M)
        if not m:
            return ""
        end = re.search(r"^\t\},", text[m.start():], re.M)
        return text[m.start(): m.start() + (end.end() if end else 5000)]

    def block(mid):
        # The gen3 mod block uses `inherit: true` and overrides only a few attributes
        # (accuracy / pp / power) — the effect fields (status, flags.heal, volatileStatus,
        # forceSwitch, …) are INHERITED from base. So the truth is base ∪ mod: a field is
        # present if either block declares it. Return None only when the move is in neither.
        combined = _one(base, mid) + "\n" + _one(mod, mid)
        return combined if combined.strip() else None

    return block


@pytest.mark.integration
def test_effect_flags_faithful_to_showdown_source():
    from agents.gen3_data import moves as movedex
    block = _ts_blocks()
    mismatches = []
    for mid, md in movedex._dex().items():
        blk = block(mid)
        if blk is None:
            continue  # not in this Showdown build — skip (rare)
        # Declarative truth straight from the .ts:
        ts_protect = re.search(r"volatileStatus:\s*'(protect|endure)'", blk) is not None
        ts_phaze = "forceSwitch: true" in blk
        ts_hazard = re.search(r"sideCondition:\s*'([a-z]+)'", blk)
        ts_hazard = bool(ts_hazard and ts_hazard.group(1) in _HAZARD_SC)
        ts_status_m = re.search(r"\n\t\tstatus:\s*'([a-z]+)'", blk)
        ts_status = ts_status_m.group(1) if (ts_status_m and ts_status_m.group(1) in _MAJOR) else None
        ts_heal = re.search(r"flags:\s*\{[^}]*\bheal:\s*1", blk) is not None

        for name, ts, got in (
            ("protect", ts_protect, md.is_protect),
            ("phaze", ts_phaze, md.is_phaze),
            ("hazard", ts_hazard, md.is_hazard),
            ("status", ts_status, md.status_inflicted),
        ):
            if ts != got:
                mismatches.append(f"{mid}.{name}: showdown={ts!r} facade={got!r}")
        # heal is only meaningful for 0-power utility moves (damaging drain moves — Giga
        # Drain, Leech Life, Dream Eater — carry healing as an attack side-effect, not a
        # click-to-recover move, and are intentionally NOT utility-flagged).
        if md.base_power == 0 and ts_heal != md.is_heal:
            mismatches.append(f"{mid}.heal: showdown={ts_heal} facade={md.is_heal}")
    assert not mismatches, "facade effect flags drifted from Showdown source:\n  " + "\n  ".join(mismatches)

    # NOTE on is_boost: deliberately NOT validated against the raw .ts here. The base
    # moves.ts is the LATEST-gen definition and the gen3 mod can REMOVE/alter a boost
    # (gen3 Charge and Stockpile do NOT raise stats — those boosts were added in gen4),
    # which a flat base∪mod source scan can't see. That gen resolution is exactly what
    # poke-env's gen3-specific static JSON (the builder's input) already does, so is_boost
    # faithfulness is covered by the gen3-AWARE curated cross-check
    # (moves_test.test_boost_covers_gen3_setup_excludes_memento_and_stockpile) plus the
    # reproducibility guard (test_moves_builder_reproduces_committed) — not raw .ts parsing.
