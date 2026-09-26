"""gen3_event_record_v2 (E12, the event-block reshape) + E4 — one CONSTRUCTED battle per mechanic of
`designs/endstate/obs_enrichment_backlog.md` §1a, each asserting the fact survives into the
observation's event window UNFLATTENED.

Every battle runs through the Rust core (`core_events --trackers --obs`) AND the Python path in one
pass: slice T holds the core's event window equal to `EventWindowTracker`'s row for row, and slice O
holds the ENCODED row byte-equal, so each assertion below (made on the Python rows) is also a
statement about what the core ships to training. Each assertion FAILS on the pre-E12 fold (which
had no entry / denial / rel / caller / stat / layers / pursuit column and no DENIED row type).

The teams and scripts are the Rust core M3 catalogue's constructed fixtures
(`src/rust_sim/tests/window_record_test.rs`, exported to
`designs/research_state/measurements/rust_core_m3_2026-09-24/catalogue/e12_fixtures.jsonl`), inlined
so this gate never reads a measurement directory, plus one TRAPPING battle for E4.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]

SNORLAX = "snorlax|||immunity|rest|Hardy|||||100|"


def _fixture_specs():
    """The M3 fixture specs by name (the JSONL is the record of what `window_record_test.rs` built)."""
    import json
    from utils.paths import repo_path
    path = repo_path("designs", "research_state", "measurements", "rust_core_m3_2026-09-24",
                     "catalogue", "e12_fixtures.jsonl")
    return {f["name"]: f for f in (json.loads(ln) for ln in path.read_text().splitlines() if ln.strip())}


_SPECS = None


def _spec(name):
    global _SPECS
    if _SPECS is None:
        _SPECS = _fixture_specs()
    return _SPECS[name]


def _run(name, p1=None, p2=None, script=None):
    """Replay one constructed battle through the core + the Python path; return the Python event
    window rows at every decision of each viewer, after asserting slices T and O are clean."""
    from agents.battle import rust_core_parity as P
    from agents.battle import rust_core_parity_trackers as T
    from agents.battle.rust_core_parity_obs import ObsCensus

    logging.getLogger("poke-env").setLevel(logging.ERROR)
    if p1 is None:
        f = _spec(name)
        p1, p2, script, seed = f["p1"], f["p2"], f["script"], f["seed"]
    else:
        seed = "1,2,3,4"
    b = P.RecordedBattle(label=f"e12v2.{name}", format_id="gen3customgame", seed=seed,
                         p1={"name": "P1", "team": p1}, p2={"name": "P2", "team": p2},
                         commands=[[f"p{s + 1}", tok, "if_open"] for s, tok in script])
    real_scope = T.SCOPE_FORMATS
    T.SCOPE_FORMATS = frozenset(real_scope | {"gen3customgame"})
    try:
        t, o, got = T.TrackerCensus(), ObsCensus(), {}
        P.check_battles([b], P.Census(), trackers=t, obs=o,
                        on_result=lambda _b, res: got.update(res))
    finally:
        T.SCOPE_FORMATS = real_scope
    assert t.decisions > 0 and not t.divergences and not t.refused, t.render()
    assert o.decisions > 0 and not o.divergences, o.render()
    out = []
    for viewer in got.get("trackers") or []:
        caps = [c for c in viewer if "trackers" in c]
        out.append(caps[-1]["trackers"]["window"]["rows"] if caps else [])
    return out


def _rows(rows, **match):
    return [r for r in rows if all(r.get(k) == v for k, v in match.items())]


# --------------------------------------------------------------------------------- ACTION DENIAL
def test_denial_fainted_first_names_the_ko_and_masks_nothing_of_the_choice():
    """Mewtwo outspeeds and KOs Rattata: Rattata's chosen Tackle never happens. The window holds a
    DENIED(fainted first) row naming the KOer and its move — and no field for WHAT Rattata chose."""
    p1_rows, p2_rows = _run("faster_ko")
    d = _rows(p1_rows, t=11)
    assert [(r["actor"], r["side"], r["denial"], r["rel"], r["move_id"], r["faint_cause"]) for r in d] == \
        [("rattata", "opp", 1, "mewtwo", "psychic", "attack")]
    # the information boundary: the denied side's CHOICE is not a key the row has at all
    assert "choice" not in d[0]
    # the victim's own view names the same denial, from its side
    assert [(r["actor"], r["side"], r["denial"]) for r in _rows(p2_rows, t=11)] == [("rattata", "ours", 1)]


def test_denial_turn_cut_by_a_self_ko_explosion():
    """gen 3: ANY faint cancels every queued action. A faster Electrode's Explosion KOs only itself,
    and the Snorlax that had not moved yet is denied by the TURN CUT, attributed to the self-KO."""
    p1_rows, _ = _run("explosion_cuts_turn")
    assert [(r["actor"], r["denial"], r["rel"], r["rel_side"], r["faint_cause"], r["move_id"])
            for r in _rows(p1_rows, t=11)] == [("snorlax", 2, "electrode", "ours", "selfko", "explosion")]


def test_denial_turn_cut_by_recoil():
    """Recoil suicide (Double-Edge from a 1-HP Shedinja) denies the opponent's Softboiled."""
    p1_rows, _ = _run("recoil_cuts_softboiled")
    assert [(r["actor"], r["denial"], r["faint_cause"]) for r in _rows(p1_rows, t=11)] == \
        [("snorlax", 2, "recoil")]


# ------------------------------------------------------------- multiple faints: cause and ORDER
def test_trade_ko_by_destiny_bond_keeps_both_faints_in_order_with_their_causes():
    p1_rows, _ = _run("destiny_bond")
    assert [(r["actor"], r["faint_cause"], r["rel"]) for r in _rows(p1_rows, t=3)] == \
        [("gengar", "attack", "tyranitar"), ("tyranitar", "destinybond", None)]


def test_perish_song_faints_read_perishsong():
    p1_rows, _ = _run("perish")
    assert [(r["actor"], r["faint_cause"]) for r in _rows(p1_rows, t=3)] == \
        [("lapras", "perishsong"), ("blissey", "perishsong")]


# ----------------------------------------------------------------------------------- PHAZING
def test_roar_is_a_forced_entry_with_its_phazer_and_the_spikes_chip_and_layers():
    p1_rows, _ = _run("roar_spikes")
    drags = _rows(p1_rows, t=2, entry=3)
    assert [(r["actor"], r["target"], r["move_id"], r["rel"], r["layers"]) for r in drags] == [
        ("blissey", "skarmory", "roar", "snorlax", 1), ("snorlax", "skarmory", "roar", "blissey", 1)]
    # the dragged mon's Spikes chip is ON its entry row (1 layer = 1/8 of max HP)
    assert all(-0.13 < r["hp_delta"] < -0.12 for r in drags)


# ------------------------------------------------------------------------------- BATON PASS
def test_baton_pass_names_passer_and_receiver_and_a_pass_into_spikes():
    p1_rows, _ = _run("baton_pass")
    bp = _rows(p1_rows, t=2, entry=4)
    assert [(r["actor"], r["rel"], r["move_id"], r["layers"]) for r in bp] == \
        [("snorlax", "ninjask", "batonpass", 1)]
    assert bp[0]["hp_delta"] < 0          # the receiver's Spikes chip
    # boost rows name their STAT (the passed stages themselves ride the receiver's active context)
    assert [(r["stat"], r["hp_delta"]) for r in _rows(p1_rows, t=6)] == [(1, 2.0), (5, 1.0)]


# ------------------------------------------------------------------- ITEM transfer / removal
def test_thief_has_a_direction_and_both_parties():
    p1_rows, _ = _run("thief")
    assert [(r["actor"], r["item_tr"], r["rel"]) for r in _rows(p1_rows, t=7)] == \
        [("blissey", 4, "sneasel"), ("sneasel", 5, "blissey")]


def test_trick_is_two_receipts_each_naming_the_other_party():
    p1_rows, _ = _run("trick")
    assert [(r["actor"], r["item_tr"], r["rel"]) for r in _rows(p1_rows, t=7)] == \
        [("blissey", 5, "alakazam"), ("alakazam", 5, "blissey")]


def test_knock_off_is_a_removal_naming_the_knocker():
    p1_rows, _ = _run("knock_off")
    assert [(r["actor"], r["item_tr"], r["rel"]) for r in _rows(p1_rows, t=7)] == \
        [("blissey", 3, "snorlax")]


# ------------------------------------------------------------------ SACKS and the FREE switch
def test_a_hazard_ko_on_entry_then_the_free_replacement():
    """A chosen Shedinja dies to one layer of Spikes on entry (a 'Spikes sack'): its FAINT reads
    `hazard` with the layers, it cuts the turn (the opponent's queued move is denied), and the next
    entry is the free REPLACEMENT naming the mon it replaces."""
    p1_rows, _ = _run("shedinja_spikes")
    assert [(r["actor"], r["faint_cause"], r["layers"]) for r in _rows(p1_rows, t=3)] == \
        [("shedinja", "hazard", 1)]
    assert [(r["actor"], r["denial"], r["rel"]) for r in _rows(p1_rows, t=11)] == \
        [("skarmory", 2, "shedinja")]
    assert [(r["actor"], r["entry"], r["rel"]) for r in _rows(p1_rows, t=2)] == \
        [("shedinja", 1, "snorlax"), ("blissey", 2, "shedinja")]


# ---------------------------------------------------------------------------------- PURSUIT
def test_pursuit_on_a_switching_target_is_flagged_and_precedes_the_switch():
    p1_rows, _ = _run("pursuit")
    moves = _rows(p1_rows, t=1, move_id="pursuit")
    assert [r["pursuit"] for r in moves] == [True, False]
    i = p1_rows.index(moves[0])
    assert p1_rows[i + 1]["t"] == 2 and p1_rows[i + 1]["rel"] == "alakazam"   # then the switch


# ----------------------------------------------------------------------------- CALLED MOVES
def test_a_called_move_names_its_caller():
    p1_rows, _ = _run("sleep_talk")
    assert [(r["move_id"], r["caller"]) for r in _rows(p1_rows, t=1, actor="snorlax")
            if r["caller"]] == [("rest", "sleeptalk"), ("bodyslam", "sleeptalk")]


# ------------------------------------------------------------ E4: the refused switch's target
def test_e4_a_refused_switch_names_the_bench_mon_it_aimed_at():
    """Dugtrio's Arena Trap: p2's Snorlax tries to switch to Blissey, the server refuses
    (`|error|[Unavailable choice]`), and the SWITCH_REJECTED row names Blissey as the target."""
    p1 = "dugtrio|||arenatrap|earthquake|Hardy|||||100|]" + SNORLAX
    p2 = "snorlax|||immunity|rest|Hardy|||||100|]blissey|||naturalcure|softboiled|Hardy|||||100|"
    _, p2_rows = _run("arena_trap", p1, p2, [[0, "move 1"], [1, "switch 2"], [1, "move 1"],
                                             [0, "move 1"], [1, "move 1"]])
    rej = _rows(p2_rows, t=9)
    assert [(r["actor"], r["target"]) for r in rej] == [("snorlax", "blissey")]


def test_the_new_columns_reach_the_encoded_row():
    """The row writer puts each E12 field in its declared column (a fold that computed them but a
    writer that dropped them would pass every test above)."""
    from agents import gen3_data
    from agents.observation.assembler import write_event_row
    from agents.observation.constants import EVENT_TOKEN_DIM, EventCol as C

    p1_rows, _ = _run("roar_spikes")
    drag = _rows(p1_rows, t=2, entry=3)[0]
    vec = np.zeros(EVENT_TOKEN_DIM, dtype=np.float32)
    write_event_row(vec, 0, drag, drag["turn"])
    assert vec[C.ENTRY] == 3.0 and vec[C.LAYERS] == pytest.approx(1 / 3)
    assert vec[C.REL_SPECIES] == float(gen3_data.species.get("snorlax").num) and vec[C.REL_SIDE] == 1.0
    assert vec[C.MOVE] == float(gen3_data.moves.get("roar").num)
