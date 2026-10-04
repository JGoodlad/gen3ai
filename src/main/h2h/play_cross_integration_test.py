"""The TWO-ARCHITECTURE head-to-head engine (``main.h2h.arch``; F-U6-1) on the REAL engine (CPU, tiny): the X5 A/B's
cross — an X5 ``fixed_mass`` checkpoint against ``blob`` checkpoints — plays on ONE engine and writes valid v2 rows; a
same-architecture cell on the two-group engine plays the games a single-group engine plays (rows, outcome digests over
the clean games AND over every game, and game for game on the engine itself), whichever group it sits in; a cross
cell's games replay exactly on a re-run (and on an engine whose groups are declared in the other order); a re-run
resumes; the family reads across the look's cells; a third architecture, and a cell the declaration has no slot or core
for, are refused; and a stale slot is caught in EACH group.

Seeded PERTURBED-fresh checkpoints (``conftest``): the numbers are not the point, the identity is. Every comparison is
between two plays in ONE process (cross-process near-tie flips, F-U6, cannot enter). The plays are MODULE fixtures
(the tier budget is per test call); the tests only read what they recorded."""
from __future__ import annotations

import pytest

from agents.inference.service import InferenceService
from agents.inference.service.slots import SlotGroup, served_state_dict
from agents.inference.service.spec import CopyParityFailure, ParityFailure
from agents.training import eval_ledger as L
from main.h2h import many as MANY
from main.h2h import play as PL

pytestmark = [pytest.mark.sim, pytest.mark.integration]

COMPUTE = PL.Compute(device="cpu", backend="eager", n_envs=8, threads=2, torch_threads=2, front="ffi",
                     profile="selfcheck")
SEED = 5
FAMILY = "h2h_cross_test_family"
LOOK = "h2h_cross_test_look1"
QUIET = dict(emit=lambda _m: None)
ROWS = L.ReaderDecl(name="h2h_cross_test", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                    selection="include", flags_ok=frozenset(), inference="conditional")
FAMILY_READ = L.ReaderDecl(name="h2h_cross_test_family", purposes=frozenset({"ab"}),
                           regime=L.RegimeFilter(protocol=PL.PROTOCOL, play="greedy", opponent_play="greedy",
                                                 mirrored=True),
                           requests="family", selection="include", flags_ok=frozenset(), inference="conditional",
                           decision_kind="ab_verdict")
#: what must be equal between two plays of one cell batch (the per-batch seed block, the counts, every digest)
SAME = ("counts", "pairs", "teams", "seed", "regime")
SAME_COMPUTE = ("outcome_digest", "outcome_digest_all", "near_tie_games", "trainee_decisions", "p2_policy_decisions",
                "near_tie_decisions", "near_tie_decisions_wide")


def rows_of(out):
    return list(L.read(ROWS, root=out).rows)


def by_cell_batch(rows):
    return {(r["player"]["sha256"], r["opponent"]["sha256"], r["seed"]["batch"]): r for r in rows}


def assert_same_rows(got, want, what):
    assert set(got) == set(want) and got, what
    for k, w in want.items():
        g = got[k]
        for f in SAME:
            assert g[f] == w[f], f"{what}: cell batch {k}: {f} differs"
        for f in SAME_COMPUTE:
            assert g["compute"][f] == w["compute"][f], f"{what}: cell batch {k}: compute.{f} differs"


def all_games_digest(games):
    return L.outcome_digest(PL.outcome_vector(games), [])


@pytest.fixture(scope="module")
def refs(checkpoints, foreign):
    (b1, b2), fm = checkpoints, foreign
    return {"b1": PL.resolve_player(b1), "b2": PL.resolve_player(b2), "fm": PL.resolve_player(fm)}


@pytest.fixture(scope="module")
def played(built, refs, tmp_path_factory):
    """(1) the blob cell (b1, b2) single-cell (``play_edge``: a single-group engine); (2) ONE two-group engine over the
    look [(b1, b2), (fm, b1), (fm, b2)] of a registered family — the blob cell in group 0, the X5 player in group 1;
    (3) the same call re-run (resume); (4) the two cross cells alone on a fresh ledger, default requests — an engine
    whose groups are declared the other way round (fm group 0, blob group 1)."""
    b1, b2, fm = refs["b1"], refs["b2"], refs["fm"]
    base = tmp_path_factory.mktemp("h2h_cross")
    single, many, rerun = base / "single", base / "many", base / "rerun"
    PL.play_edge(str(single), b1, b2, pairs=2, batch_pairs=2, schedule_seed=SEED, run_label="test-study",
                 compute=COMPUTE, **QUIET)
    w = L.LedgerWriter(many, producer="h2h")
    w.register_family(FAMILY, decision_kind="ab_verdict", rule="test rule v1", protocol=PL.PROTOCOL)
    w.close()
    look = [(b1, b2), (fm, b1), (fm, b2)]
    kw = dict(batch_pairs=2, schedule_seed=SEED, run_label="test-study", compute=COMPUTE, **QUIET)
    first = MANY.play_cells(str(many), look, pairs=2, purpose="ab", family=FAMILY, request_id=LOOK, **kw)
    rows_first = rows_of(many)
    again = MANY.play_cells(str(many), look, pairs=2, purpose="ab", family=FAMILY, request_id=LOOK, **kw)
    cross = MANY.play_cells(str(rerun), [(fm, b1), (fm, b2)], pairs=2, **kw)
    return {"look": look, "single": rows_of(single), "many": many, "first": first, "rows_first": rows_first,
            "again": again, "rows_again": rows_of(many), "cross": cross, "rows_cross": rows_of(rerun)}


def test_the_look_is_one_engine_of_two_slot_groups_each_with_only_the_slots_its_cells_need(played, refs):
    eng = played["first"]["engine"]
    assert eng["decl"] == {
        "groups": [{"name": "eval", "source": refs["b1"].id, "roles": ["player", "opponent"],
                    "fingerprint": eng["decl"]["groups"][0]["fingerprint"]},
                   {"name": "eval_b", "source": refs["fm"].id, "roles": ["player"],
                    "fingerprint": eng["decl"]["groups"][1]["fingerprint"]}],
        "combos": [[0, 0], [1, 0]]}
    assert eng["decl"]["groups"][0]["fingerprint"] != eng["decl"]["groups"][1]["fingerprint"]
    assert [c["rows"] for c in eng["cells"]] == [1, 1, 1]
    # the cross alone: the X5 player's group first with ONE player slot, the blob group ONE opponent slot
    d = played["cross"]["engine"]["decl"]
    assert [(g["source"], g["roles"]) for g in d["groups"]] == [(refs["fm"].id, ["player"]),
                                                               (refs["b1"].id, ["opponent"])]
    assert d["combos"] == [[0, 1]]


def test_a_cross_cell_plays_and_writes_valid_v2_rows(played, refs):
    rows = by_cell_batch(played["rows_first"])
    fm, b1, b2 = refs["fm"], refs["b1"], refs["b2"]
    for opp in (b1, b2):
        r = rows[(fm.sha256, opp.sha256, 0)]
        assert L.validate_row(r) == []
        assert r["schema"] == L.SCHEMA and r["regime"]["protocol"] == PL.PROTOCOL
        assert r["player"]["sha256"] == fm.sha256 and r["opponent"]["sha256"] == opp.sha256
        assert r["pairs"]["n_pairs"] == 2 and sum(r["counts"][k] for k in ("w", "l", "d")) == 4
        assert r["compute"]["trainee_decisions"] > 0 and r["compute"]["p2_policy_decisions"] > 0
        assert r["request"]["id"] == LOOK and r["request"]["family"] == FAMILY and r["purpose"] == "ab"


def test_a_same_architecture_cell_on_the_two_group_engine_writes_the_single_group_rows(played, refs):
    """The blob cell (b1, b2) on the two-group engine (group 0, beside the X5 group) == on `play`'s single-group engine:
    the seed block, the counts, the pairs, the team counters and BOTH outcome digests (clean games; every game)."""
    key = (refs["b1"].sha256, refs["b2"].sha256, 0)
    single = by_cell_batch(played["single"])
    many = {k: v for k, v in by_cell_batch(played["rows_first"]).items() if k == key}
    assert_same_rows(many, single, "two-group vs single-group engine")


def test_a_cross_cell_replays_exactly_on_a_rerun_with_its_groups_declared_the_other_way(played, refs):
    fm = refs["fm"].sha256
    first = {k: v for k, v in by_cell_batch(played["rows_first"]).items() if k[0] == fm}
    again = by_cell_batch(played["rows_cross"])
    assert len(first) == 2
    assert_same_rows(again, first, "a cross cell's re-run")
    assert len({r["compute"]["outcome_digest_all"] for r in first.values()}) == 2, \
        "the two cross cells played different games (the comparison is not vacuous)"


def test_a_rerun_of_the_look_resumes_and_builds_no_engine(played):
    assert played["again"]["engine"] == {"startup_s": None, "preflight_s": None, "cells": [], "decl": None}
    assert len(played["rows_again"]) == len(played["rows_first"]) == 3


def test_the_family_reads_across_the_cross_cells_of_its_look(played):
    got = L.read(FAMILY_READ, root=played["many"], family=FAMILY)
    looks = L.looks(got, min_pairs=2)
    assert [rid for rid, _c in looks] == [LOOK]
    want = {(p.sha256, o.sha256) for p, o in played["look"]}
    assert {(c.player, c.opponent) for c in looks[0][1]} == want
    assert all(c.verdict == "OK" and c.n_pairs == 2 for c in looks[0][1])


# ------------------------------------------------------------------------------------------------ the declaration
def test_the_declaration_sorts_every_side_into_at_most_two_groups(built, refs):
    b1, b2, fm = refs["b1"], refs["b2"], refs["fm"]
    d = PL.declare_engine([(fm, b1), (fm, b2)], PL._load_host)
    assert (d.roles, d.combos, dict(d.members)) == (
        (("player",), ("opponent",)), ((0, 1),), {fm.sha256: 0, b1.sha256: 1, b2.sha256: 1})
    assert (d.n_slots, d.slot_of(0, "player"), d.slot_of(1, "opponent")) == (2, 0, 1)
    assert d.archs[0].terminal is not None and d.archs[1].terminal is None, "the terminal binds a PLAYER only"
    with pytest.raises(PL.CellArchMismatch, match="declares no opponent slot"):
        d.slot_of(0, "opponent")
    one = PL.declare_engine([(b1, b2)], PL._load_host)
    assert (one.roles, one.combos, one.n_slots) == ((("player", "opponent"),), ((0, 0),), 2), \
        "a single-architecture cell declares today's engine: one group of two slots, one core"


def test_a_third_architecture_is_refused_by_the_declaration(built, refs, third):
    t = PL.resolve_player(third)
    with pytest.raises(PL.CellArchMismatch, match="matches neither") as ei:
        PL.declare_engine([(refs["fm"], refs["b1"]), (t, refs["b1"])], PL._load_host)
    assert t.id in str(ei.value) and "move_prior_fusion" in str(ei.value)


# ------------------------------------------------------------------------------------------------ the engine by hand
@pytest.fixture(scope="module")
def engine_run(built, refs, third):
    """ONE engine declared from [(fm, b1), (b1, b2)] (fm group 0 with a player slot; blob group 1 with a player and an
    opponent slot), driven by hand: the cross cell, then the blob cell (in group 1), each one batch at the single-cell
    seeds; refusals (a third architecture, a combination with no slot); a stale slot written into EACH group after a
    cycle (the engine's own check must name it); last, a slot copy in group 1 that silently does nothing (the load's
    check must catch it and poison the service). Then the blob cell on a single-group engine of its own."""
    b1, b2, fm = refs["b1"], refs["b2"], refs["fm"]
    out = {}
    decl = PL.declare_engine([(fm, b1), (b1, b2)], PL._load_host)
    eng = PL.H2HEngine(fm, b1, COMPUTE, decl=decl, **QUIET)
    try:
        out["decl"], out["n_cores"] = decl, len(eng.cores)
        out["storage"] = [g.storage_bytes() for g in eng.svc.groups]
        games, _st = eng.play_batch(3, PL.cycle_seed(SEED, PL.schedule_key_of(fm, b1), 0))
        out["cross"] = sorted(games, key=lambda g: g["game"])
        for name, (p, o) in (("third", (PL.resolve_player(third), b1)), ("no_slot", (b1, fm))):
            try:
                eng.set_cell(p, o)
            except PL.CellArchMismatch as e:
                out[name] = str(e)
        out["cell_after_refusals"] = (eng.player.sha256, eng.opponent.sha256)
        eng.set_cell(b1, b2)
        games, _st = eng.play_batch(3, PL.cycle_seed(SEED, PL.schedule_key_of(b1, b2), 0))
        out["blob"] = sorted(games, key=lambda g: g["game"])
        out["blob_combo"] = eng.combo
        # a STALE slot in each group, written behind the service's back after a cycle: the after-cycle check names it
        eng.set_cell(fm, b1)
        eng.play_batch(1, PL.cycle_seed(SEED, PL.schedule_key_of(fm, b1), 1))
        out["stale"] = {}
        for gi, slot, wrong in ((0, decl.slot_of(0, "player"), "noise"), (1, decl.slot_of(1, "opponent"), b2)):
            g, i = eng.svc.groups[gi], decl.local(slot)[1]
            right = served_state_dict((eng.pm if gi == 0 else eng.om).policy)
            if wrong == "noise":
                bad = {k: v.clone() for k, v in right.items()}
                k0 = next(k for k, v in bad.items() if v.is_floating_point() and v.numel())
                bad[k0] = bad[k0] + 1e-3
            else:
                bad = served_state_dict(eng._host(wrong).policy)
            g.copy_in(i, bad)
            try:
                eng.verify_slots("test")
            except CopyParityFailure as e:
                out["stale"][gi] = str(e)
            g.copy_in(i, right)
            eng.verify_slots("restored")                              # the right weights back: it passes again
        eng.set_cell(b1, b2)
        orig = SlotGroup.copy_in
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(SlotGroup, "copy_in",
                       lambda self, slot, sd: None if self.name == "eval_b" else orig(self, slot, sd))
            try:
                eng.play_batch(1, PL.cycle_seed(SEED, PL.schedule_key_of(b1, b2), 1))
            except ParityFailure as e:
                out["load_noop"] = e
        try:
            eng.play_batch(1, PL.cycle_seed(SEED, PL.schedule_key_of(b1, b2), 1))
        except Exception as e:                                                     # noqa: BLE001 - recorded
            out["after_poison"] = e
    finally:
        eng.close()
    e2 = PL.H2HEngine(b1, b2, COMPUTE, **QUIET)
    try:
        out["single_groups"] = len(e2.svc.groups)
        games, _st = e2.play_batch(3, PL.cycle_seed(SEED, PL.schedule_key_of(b1, b2), 0))
        out["blob_single"] = sorted(games, key=lambda g: g["game"])
    finally:
        e2.close()
    return out


def test_a_blob_cell_in_the_second_group_plays_game_for_game_what_a_single_group_engine_plays(engine_run):
    assert engine_run["blob_combo"] == (1, 1) and engine_run["single_groups"] == 1
    assert engine_run["blob"] == engine_run["blob_single"], "the same game log: results, teams, end turns, near-ties"
    assert all_games_digest(engine_run["blob"]) == all_games_digest(engine_run["blob_single"])
    assert all_games_digest(engine_run["cross"]) != all_games_digest(engine_run["blob"]), "not vacuous"


def test_the_engine_declares_one_core_per_combination_and_one_slot_group_per_architecture(engine_run):
    d = engine_run["decl"]
    assert d.roles == (("player",), ("player", "opponent")) and d.combos == ((0, 1), (1, 1))
    assert engine_run["n_cores"] == 2
    s0, s1 = engine_run["storage"]
    assert s1 > 1.5 * s0, "group 1 stacks two slots, group 0 one (each slot's served weights, once per unique tensor)"


def test_a_third_architecture_and_an_undeclared_combination_are_refused_and_leave_the_cell(engine_run, refs):
    assert "matches none of the engine's 2 slot group(s)" in engine_run["third"]
    assert "move_prior_fusion" in engine_run["third"]
    assert "declares no opponent slot" in engine_run["no_slot"]
    assert engine_run["cell_after_refusals"] == (refs["fm"].sha256, refs["b1"].sha256)


def test_a_stale_slot_is_caught_in_each_group(engine_run, refs):
    stale = engine_run["stale"]
    assert set(stale) == {0, 1}
    assert "eval[0]" in stale[0] and refs["fm"].id in stale[0], "the X5 player's slot, in group 0"
    assert "eval_b[1]" in stale[1] and refs["b1"].id in stale[1], "the blob opponent's slot, in group 1"


def test_a_slot_copy_that_does_nothing_in_the_second_group_is_caught_and_poisons(engine_run):
    assert isinstance(engine_run.get("load_noop"), CopyParityFailure)
    assert "eval_b" in str(engine_run["load_noop"])
    assert "after_poison" in engine_run, "a poisoned service serves no later cell"


@pytest.fixture(scope="module")
def noop_load(built, refs, tmp_path_factory):
    """``InferenceService.load`` made a no-op over a two-cell cross; records the refusal and the rows written."""
    fm, b1, b2 = refs["fm"], refs["b1"], refs["b2"]
    out = tmp_path_factory.mktemp("h2h_cross_noop") / "ledger"
    got = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(InferenceService, "load", lambda self, slot, policy, model_id: None)
        try:
            MANY.play_cells(str(out), [(fm, b1), (fm, b2)], pairs=2, batch_pairs=2, run_label="s", compute=COMPUTE,
                            **QUIET)
        except Exception as e:                                                     # noqa: BLE001 - recorded
            got["error"] = e
    got["rows"] = rows_of(out)
    return got


def test_a_load_that_never_reached_a_cross_cells_slot_writes_no_row(noop_load, refs):
    """The FIRST cell's slots hold their groups' templates — which ARE its two checkpoints (each group's source), so
    its row is right — and the second cell's opponent slot still holds the first cell's opponent: the after-cycle check
    refuses that batch before it is scored."""
    fm, b1 = refs["fm"], refs["b1"]
    assert isinstance(noop_load.get("error"), CopyParityFailure)
    assert "eval_b[0]" in str(noop_load["error"]) and refs["b2"].id in str(noop_load["error"])
    assert [(r["player"]["sha256"], r["opponent"]["sha256"]) for r in noop_load["rows"]] == [(fm.sha256, b1.sha256)]
