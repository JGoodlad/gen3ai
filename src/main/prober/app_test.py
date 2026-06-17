"""Textual Pilot tests for ProberApp.

The skeleton tests mount the three-region shell. The analysis tests inject a
FakeProbeModel (no checkpoint, no torch) and drive a real tmp trace through the
selection → worker → render pipeline, asserting the panels populate — fast and
deterministic.
"""

import json
import os

import numpy as np
from textual.widgets import Collapsible, DataTable, ListView, Static, Tree

from agents.action.constants import MOVE_START
from main.prober.app import _DISABLED_GREY, PaneSplitter, ProberApp, _append_belief, _field_text, _hp_bar
from main.prober.engine import BeliefSlotView, BeliefView
from main.prober.model import ObsOffsets
from rich.text import Text


def test_hp_bar_disabled_is_grey_not_red():
    """A fainted / illegal slot renders grey; an alive low-HP mon keeps its (red) HP gradient."""
    dead = _hp_bar("faint", disabled=True)
    assert all(sp.style == _DISABLED_GREY for sp in dead.spans)
    alive_low = _hp_bar("8%")                       # alive but low → red-ish, NOT the disabled grey
    assert all(sp.style != _DISABLED_GREY for sp in alive_low.spans)


def test_field_text_shows_pending_wish():
    """gen3_wish_wired_v1: the FIELD line surfaces a pending Wish (per side) when decoded, and omits
    it otherwise — so a human walking the prober sees the floating heal coming."""
    base = {"weather": "NONE", "our_spikes": 0, "opp_spikes": 0, "turn": 7}
    assert "wish" not in _field_text(base).plain.lower()
    ours = _field_text({**base, "wish_our": True}).plain
    assert "wish" in ours.lower() and "our" in ours
    both = _field_text({**base, "wish_our": True, "wish_opp": True}).plain
    assert "our & opp" in both


def test_append_belief_renders_hidden_slot_guesses():
    """The believed-hidden section lists each hidden slot's top species + confidence."""
    out = Text("OPP TEAM\nmetagross 46%")
    belief = BeliefView(slots=(
        BeliefSlotView(slot=2, top=(("tyranitar", 0.41), ("skarmory", 0.19))),
        BeliefSlotView(slot=3, top=(("celebi", 0.33),)),
    ))
    _append_belief(out, belief)
    s = out.plain
    assert "believed hidden" in s
    assert "tyranitar 41%" in s and "skarmory 19%" in s and "celebi 33%" in s


def test_append_belief_noop_when_off_or_empty():
    """No believed section when the belief is off (None) or carries no hidden slots — off-runs are
    unchanged."""
    for belief in (None, BeliefView(slots=())):
        out = Text("OPP TEAM")
        _append_belief(out, belief)
        assert out.plain == "OPP TEAM"


def test_append_belief_truth_renders_truth_and_matched_guess():
    """The privileged view lists revealed mons, then each hidden mon with ✓/✗ + its matched guess and
    (when wrong) the rank the model gave the true species."""
    from main.prober.app import _append_belief_truth
    from main.prober.engine import BeliefTruthView, OppMonTruth

    out = Text("OPP")
    view = BeliefTruthView(mons=(
        OppMonTruth(species="metagross", revealed=True),
        OppMonTruth(species="tyranitar", revealed=False,                       # ✓ top-1 right
                    guess=(("tyranitar", 0.55), ("skarmory", 0.20)), guessed_right=True, true_rank=1),
        OppMonTruth(species="celebi", revealed=False,                          # ≈ in belief, not top-1
                    guess=(("blissey", 0.40), ("celebi", 0.25)), guessed_right=False, true_rank=2),
        OppMonTruth(species="jirachi", revealed=False,                         # ✗ not in the top-k
                    guess=(("snorlax", 0.30), ("blissey", 0.20)), guessed_right=False, true_rank=-1),
    ), n_hidden=3, n_correct=1)
    _append_belief_truth(out, view)
    s = out.plain
    assert "truth + belief" in s and "1/3 top-1" in s
    assert "metagross" in s                       # revealed mon under 'seen'
    # three distinct markers: ✓ top-1, ≈ matched-but-not-top-1, ✗ not found
    assert "✓" in s and "≈" in s and "✗" in s
    assert "tyranitar 55%" in s and "celebi 25%" in s
    assert "(#2)" in s                            # celebi's true-species rank, since it wasn't top-1


def test_append_belief_truth_noop_when_empty():
    from main.prober.app import _append_belief_truth
    from main.prober.engine import BeliefTruthView

    out = Text("OPP")
    _append_belief_truth(out, None)
    _append_belief_truth(out, BeliefTruthView(mons=()))
    assert out.plain == "OPP"

# Small synthetic obs layout (mirrors engine_test).
_OFF = ObsOffsets(mm_off=10, om_off=20, tm_off=164, active_block_dim=5,
                  turn_history_offset=200, turn_history_dim=10)
_OBS_LEN = 256


class _FakeModel:
    def __init__(self):
        self.offsets = _OFF

    def action_dist(self, obs, mask):
        w = np.ones(len(mask), dtype=np.float64)
        for s in range(4):
            w[MOVE_START + s] = 1.0 + float(obs[_OFF.mm_off + s]) * 10.0
        w = w * mask.astype(np.float64)
        return w / w.sum(), np.arange(len(mask), dtype=np.float64)

    def logit_grad(self, obs, mask, idx):
        return np.arange(len(obs), dtype=np.float64)

    def value(self, obs, mask):
        return 0.0

    def describe_global(self, obs):
        return {"weather": "SUN", "weather_permanent": False, "weather_turns_left": 3,
                "our_spikes": 2, "opp_spikes": 0, "turn": 5,
                "our_reflect": True, "opp_reflect": False}

    def describe_team(self, obs):
        # our active (zapdos) + opp active (jynx) revealed items + movesets, decoded from the obs.
        return {"zapdos": {"item": "choiceband", "moves": ("thunderbolt", "hiddenpower")},
                "jynx": {"item": "leftovers", "moves": ("icebeam",)}}

    def describe_turn_outcome(self, obs):
        return {"our_crit": False, "opp_crit": False, "our_cant": None, "opp_cant": None}


def _write_trace(tmp_path, chosen="thunderbolt", has_state=1):
    actions = {f"switch:m{i}": {"prob": "1.0%", "valid": True} for i in range(6)}
    actions.update({
        "thunderbolt": {"prob": "92.1%", "valid": True},
        "earthquake": {"prob": "2.8%", "valid": True},
        "move2": {"prob": "0.0%", "valid": False},
        "move3": {"prob": "0.0%", "valid": False},
        "struggle": {"prob": "0.0%", "valid": False},
    })
    summary = {
        "meta": {"step": 1000, "battle_id": "b", "result": "WIN", "turns": 5, "invocations": 1},
        "invocations": [{
            "i": 1, "turn": 3, "phase": "move_selection", "chosen": chosen,
            "our": {"species": "zapdos", "hp": "80%"},
            "opp": {"species": "jynx", "hp": "55%"}, "actions": actions,
            "outcome": {"our": {"action": chosen, "hp_delta": "-10%"},
                        "opp": {"action": "switch:gengar", "hp_delta": "+0%"},
                        "reward": {"total": -1.4, "base": "hp_ours=-1.2"},
                        "events": []},
        }],
    }
    d = tmp_path / "run" / "eval_traces" / "step_1000" / "Test"
    os.makedirs(d, exist_ok=True)
    with open(d / "win_001_summary.json", "w") as f:
        json.dump(summary, f)
    obs = np.zeros((1, _OBS_LEN), dtype=np.float32)
    obs[0, _OFF.mm_off:_OFF.mm_off + 4] = [0.5, 0.25, 0.0, 0.125]
    np.savez(d / "win_001_states.npz", obs=obs,
             values=np.array([1.5], dtype=np.float32),   # → a.value so the CRITIC line renders
             win_probs=np.array([0.62], dtype=np.float32),  # → a.win_prob so the WIN-PROB line renders
             has_state=np.array([has_state], dtype=np.int8))
    return str(tmp_path / "run")


def _write_run_with_models(tmp_path):
    """A run dir with a step-2,000,000 trace + manifest+snapshot, a checkpoint, and
    best_model — so the resolution ladder has all three tiers available."""
    run = tmp_path / "run"
    et = run / "eval_traces" / "step_2000000"
    bd = et / "Test"
    os.makedirs(bd, exist_ok=True)
    summary = {
        "meta": {"step": 2000000, "result": "WIN", "turns": 5, "invocations": 1},
        "invocations": [{
            "i": 1, "turn": 3, "phase": "move_selection", "chosen": "thunderbolt",
            "our": {"species": "zapdos"}, "opp": {"species": "jynx"},
            "actions": {**{f"switch:m{i}": {"prob": "1.0%", "valid": True} for i in range(6)},
                        "thunderbolt": {"prob": "92.1%", "valid": True},
                        "earthquake": {"prob": "2.8%", "valid": True},
                        "move2": {"prob": "0.0%", "valid": False},
                        "move3": {"prob": "0.0%", "valid": False},
                        "struggle": {"prob": "0.0%", "valid": False}},
        }],
    }
    with open(bd / "win_001_summary.json", "w") as f:
        json.dump(summary, f)
    obs = np.zeros((1, _OBS_LEN), dtype=np.float32)
    np.savez(bd / "win_001_states.npz", obs=obs, has_state=np.array([1], dtype=np.int8))
    with open(et / "eval_manifest.json", "w") as f:
        json.dump({"step": 2000000, "git_hash": "deadbeef", "arch_signature": "gen3_x",
                   "snapshot": "snapshot.zip"}, f)
    (et / "snapshot.zip").write_text("")            # retained exact snapshot
    (run / "checkpoint_3200000_steps.zip").write_text("")
    os.makedirs(run / "best_model", exist_ok=True)
    (run / "best_model" / "best_model.zip").write_text("")
    return str(run)


async def test_skeleton_mounts_three_regions():
    app = ProberApp(root="/tmp/prober-nonexistent")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#trace-tree", Tree) is not None
        assert app.query_one("#invocation-list", ListView) is not None
        # analysis is a scroll of collapsible sections (multiple open at once)
        assert app.query_one("#sec-board", Collapsible) is not None
        for tid in ("faith-table", "matchups-table", "sweep-table", "saliency-table"):
            assert app.query_one(f"#{tid}", DataTable) is not None
        # default open set
        assert app.query_one("#sec-board", Collapsible).collapsed is False
        assert app.query_one("#sec-matchups", Collapsible).collapsed is True


async def test_sections_toggle_multiple_open(tmp_path):
    run = _write_rich_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        matchups = app.query_one("#sec-matchups", Collapsible)
        assert matchups.collapsed is True
        app.action_toggle_section("sec-matchups")     # open it (Board/Faith already open)
        await pilot.pause()
        assert matchups.collapsed is False
        # several sections open simultaneously
        open_now = [s for s in ("sec-board", "sec-faith", "sec-matchups", "sec-outcome")
                    if not app.query_one(f"#{s}", Collapsible).collapsed]
        assert len(open_now) >= 3


async def test_obs_mismatch_warns_across_panels(tmp_path):
    """The engine's obs_mismatch flag must surface in the TUI everywhere (the 'stays in sync' guard):
    the Summary banner, the always-visible battle header, and the affected Matchups/Saliency panels."""
    import dataclasses
    run = _write_trace(tmp_path)
    fake = _FakeModel()
    fake.offsets = dataclasses.replace(_OFF, total_dim=_OBS_LEN + 2)   # encoder grew → trace is 'old obs'
    app = ProberApp(root=run, injected_model=fake)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "OBS MISMATCH" in str(app.query_one("#summary-head", Static).render())
        assert "OBS MISMATCH" in str(app.query_one("#battle-header", Static).render())
        assert "OBS MISMATCH" in str(app.query_one("#matchups-threat", Static).render())
        assert app.query_one("#saliency-table", DataTable).get_row_at(0)[0].plain.startswith("⚠ OBS")
    # and NO banner when the lengths match
    fake2 = _FakeModel()
    fake2.offsets = dataclasses.replace(_OFF, total_dim=_OBS_LEN)
    app2 = ProberApp(root=_write_trace(tmp_path), injected_model=fake2)
    async with app2.run_test() as pilot:
        await pilot.pause()
        app2._select_battle(app2._tree_model.all_battles()[0])
        await pilot.pause()
        await app2.workers.wait_for_complete()
        await pilot.pause()
        assert "OBS MISMATCH" not in str(app2.query_one("#battle-header", Static).render())


async def test_quit_binding():
    app = ProberApp(root="/tmp/prober-nonexistent")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
    assert app.return_code == 0


async def test_tree_left_right_expand_collapse(tmp_path):
    run = _write_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        tree = app.query_one("#trace-tree", Tree)
        tree.focus()
        await pilot.pause()
        opp = tree.root.children[0].children[0]   # step → opponent (collapsed)
        tree.move_cursor(opp)
        await pilot.pause()
        assert not opp.is_expanded
        await pilot.press("right")                 # right expands
        await pilot.pause()
        assert opp.is_expanded
        await pilot.press("left")                  # left collapses
        await pilot.pause()
        assert not opp.is_expanded


async def test_pane_splitter_drag_resizes(tmp_path):
    run = _write_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        sp = app.query_one("#split-sidebar", PaneSplitter)
        sidebar = app.query_one("#sidebar")
        w0 = sidebar.region.width

        class _Ev:
            def __init__(self, x):
                self.screen_x = x
            def stop(self):
                pass

        sp.on_mouse_down(_Ev(w0))          # grab
        sp.on_mouse_move(_Ev(w0 + 10))     # drag 10 cells right
        await pilot.pause()
        assert int(sidebar.styles.width.value) == w0 + 10
        sp.on_mouse_up(_Ev(w0 + 10))


async def test_browse_tree_built(tmp_path):
    run = _write_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app._tree_model.is_empty
        assert len(app._tree_model.all_battles()) == 1


async def test_select_battle_populates_panels(tmp_path):
    run = _write_trace(tmp_path, chosen="thunderbolt")
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        battle = app._tree_model.all_battles()[0]
        app._select_battle(battle)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#faith-table", DataTable).row_count == 11
        assert app.query_one("#matchups-table", DataTable).row_count == 4
        # chosen is a move → 4-point sweep
        assert app.query_one("#sweep-table", DataTable).row_count == 4
        assert app.query_one("#saliency-table", DataTable).row_count == 5  # +their_matchups block
        # MOVES is still a DataTable: 5 moves (4 + struggle).
        assert app.query_one("#summary-moves", DataTable).row_count == 5
        # SWITCHES / OPP / Team are now custom-rendered Static panels (so movesets can span).
        switches = str(app.query_one("#summary-switches", Static).render())
        assert "m0" in switches and "m5" in switches and "risk-in" in switches  # 6 switch mons listed
        opp = str(app.query_one("#summary-opp", Static).render())
        assert "jynx (leftovers)" in opp and "icebeam" in opp  # inline (item) + revealed moveset
        # Team tab: our active (zapdos) moveset + opp (jynx) revealed moveset
        team_our = str(app.query_one("#team-our", Static).render())
        assert "thunderbolt" in team_our and "earthquake" in team_our
        team_opp = str(app.query_one("#team-opp", Static).render())
        assert "icebeam" in team_opp
        head = str(app.query_one("#summary-head", Static).render())
        assert "zapdos" in head and "jynx" in head and "thunderbolt" in head  # context header
        assert "FIELD" in head and "SUN" in head  # field line (weather/hazards/screens) present
        assert "@choiceband" in head and "@leftovers" in head  # our + opp revealed items in header
        assert "RESULT" in head and "switch:gengar" in head  # what-happened line on the summary
        assert "REWARD" in head and "hp_ours=-1.2" in head    # reward breakdown on the summary
        assert "█" in head                                    # colour-graded HP bar (80% our active)
        # section titles carry their 1-indexed hotkey (Team inserted at 2)
        assert app.query_one("#sec-summary", Collapsible).title == "1  Summary"
        assert app.query_one("#sec-team", Collapsible).title == "2  Team"
        assert app.query_one("#sec-flow", Collapsible).title == "9  Flow"
        assert app.query_one("#sec-outcome", Collapsible).title == "0  Outcome"
        # header line order: FIELD · CHOSE · RESULT · REWARD · CRITIC (critic moved to last)
        assert (head.index("FIELD") < head.index("CHOSE") < head.index("RESULT")
                < head.index("REWARD") < head.index("CRITIC"))


def _write_status_move_trace(tmp_path):
    """A trace whose move slots mix a non-damaging move (spikes, slot 0) with a damaging one
    (thunderbolt, slot 1) — both carry a 2.0× type multiplier in the obs, but only thunderbolt's
    is meaningful. Lets us assert the 'n/a' rendering for the phantom multiplier."""
    actions = {f"switch:m{i}": {"prob": "1.0%", "valid": True} for i in range(6)}
    actions.update({
        "spikes": {"prob": "42.9%", "valid": True},       # non-damaging → eff n/a
        "thunderbolt": {"prob": "40.0%", "valid": True},   # damaging → real ×mult
        "toxic": {"prob": "5.0%", "valid": True},          # non-damaging
        "roar": {"prob": "2.0%", "valid": True},           # non-damaging (phazing)
        "struggle": {"prob": "0.0%", "valid": False},
    })
    summary = {
        "meta": {"step": 1000, "result": "LOSS", "turns": 3, "invocations": 1},
        "invocations": [{
            "i": 1, "turn": 3, "phase": "move_selection", "chosen": "spikes",
            "our": {"species": "skarmory", "hp": "75%"},
            "opp": {"species": "metagross", "hp": "100%"}, "actions": actions,
            "outcome": {"our": {"action": "spikes", "hp_delta": "+0%"},
                        "opp": {"action": "meteormash", "hp_delta": "+0%"}, "events": []},
        }],
    }
    d = tmp_path / "run" / "eval_traces" / "step_1000" / "Test"
    os.makedirs(d, exist_ok=True)
    with open(d / "loss_001_summary.json", "w") as f:
        json.dump(summary, f)
    obs = np.zeros((1, _OBS_LEN), dtype=np.float32)
    # spikes 2.0×, thunderbolt 2.0×, toxic 0.0×, roar 0.5× (all stored /4 in the obs)
    obs[0, _OFF.mm_off:_OFF.mm_off + 4] = [0.5, 0.5, 0.0, 0.125]
    np.savez(d / "loss_001_states.npz", obs=obs,
             values=np.array([1.5], dtype=np.float32),
             has_state=np.array([1], dtype=np.int8))
    return str(tmp_path / "run")


async def test_non_damaging_move_eff_is_na(tmp_path):
    """The phantom type multiplier on a non-damaging move renders 'n/a' / '—', while a damaging
    move in the same panel keeps its real ×mult — the GIGO-display fix."""
    run = _write_status_move_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Matchups table: spikes/toxic/roar → 'n/a'; thunderbolt → a real '×' multiplier.
        mt = app.query_one("#matchups-table", DataTable)
        eff_by_move = {str(mt.get_row_at(r)[0]): str(mt.get_row_at(r)[1])
                       for r in range(mt.row_count)}
        assert "n/a" in eff_by_move["spikes"] and "×" not in eff_by_move["spikes"]
        assert "n/a" in eff_by_move["toxic"] and "n/a" in eff_by_move["roar"]
        assert "×" in eff_by_move["thunderbolt"] and "n/a" not in eff_by_move["thunderbolt"]

        # MOVES summary panel: spikes' eff cell is the '—' placeholder, thunderbolt keeps '×'.
        moves = app.query_one("#summary-moves", DataTable)
        move_eff = {str(moves.get_row_at(r)[0]).lstrip("▶ ").strip(): str(moves.get_row_at(r)[1])
                    for r in range(moves.row_count)}
        assert move_eff["spikes"] == "—"
        assert "×" in move_eff["thunderbolt"]


async def test_summary_renders_win_prob(tmp_path):
    """The WIN-PROB line (calibrated P(win) from the head) renders in the Summary when the trace has
    a win_probs array, placed after CRITIC (the same critic lens, in [0,1] units)."""
    run = _write_trace(tmp_path, chosen="thunderbolt")
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        head = str(app.query_one("#summary-head", Static).render())
        assert "WIN-PROB" in head and "P(win)" in head and "62%" in head
        assert head.index("CRITIC") < head.index("WIN-PROB")   # appended after the critic line


async def test_switch_decision_has_no_sweep(tmp_path):
    run = _write_trace(tmp_path, chosen="switch:m0")
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # the "not a move" placeholder row
        assert app.query_one("#sweep-table", DataTable).row_count == 1


async def test_no_state_invocation_warns(tmp_path):
    run = _write_trace(tmp_path, has_state=0)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # faithfulness shows a single warning row; matchups/saliency stay empty
        assert app.query_one("#faith-table", DataTable).row_count == 1
        assert app.query_one("#matchups-table", DataTable).row_count == 0


def _write_rich_trace(tmp_path):
    """A 2-invocation trace (one move, one switch with a faint event) + values, so
    the Outcome panel and flags/jump have something to show."""
    run = tmp_path / "run"
    bd = run / "eval_traces" / "step_1000" / "Test"
    os.makedirs(bd, exist_ok=True)
    actions = {f"switch:m{i}": {"prob": "1.0%", "valid": True} for i in range(6)}
    actions.update({"thunderbolt": {"prob": "92.1%", "valid": True},
                    "earthquake": {"prob": "2.8%", "valid": True},
                    "move2": {"prob": "0.0%", "valid": False},
                    "move3": {"prob": "0.0%", "valid": False},
                    "struggle": {"prob": "0.0%", "valid": False}})
    invs = [
        {"i": 1, "turn": 1, "phase": "move_selection", "chosen": "thunderbolt",
         "our": {"species": "zapdos", "hp": "100%", "bench": "celebi(100%), snorlax(faint)"},
         "opp": {"species": "jynx", "hp": "80%", "bench": "tyranitar(faint)"}, "actions": actions,
         "outcome": {"our": {"action": "thunderbolt", "hp_delta": "-20%"},
                     "reward": {"total": 1.2, "base": "hp=1.0"}, "events": []}},
        {"i": 2, "turn": 2, "phase": "move_selection", "chosen": "switch:m0",
         "our": {"species": "zapdos"}, "opp": {"species": "jynx"}, "actions": actions,
         "outcome": {"reward": {"total": -0.5}, "events": ["opp:jynx:fainted"]}},
    ]
    summary = {"meta": {"step": 1000, "result": "WIN", "turns": 4, "invocations": 2},
               "invocations": invs}
    with open(bd / "win_001_summary.json", "w") as f:
        json.dump(summary, f)
    obs = np.zeros((2, _OBS_LEN), dtype=np.float32)
    np.savez(bd / "win_001_states.npz", obs=obs,
             has_state=np.array([1, 1], dtype=np.int8),
             values=np.array([2.0, -1.0], dtype=np.float32))
    return str(run)


async def test_outcome_panel_and_flags(tmp_path):
    run = _write_rich_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # the switch+faint invocation (idx 1) is jump-flagged
        assert app._flagged == [1]
        # Outcome tab: reward breakdown rows + a value summary line
        assert app.query_one("#reward-table", DataTable).row_count >= 1
        summ = str(app.query_one("#outcome-summary", Static).render())
        assert "V(s)" in summ
        assert "ΔV" in summ and "TD δ" in summ      # critic surprise surfaced for parity with the CLI
        # jump lands on the flagged invocation
        app.action_next_flagged()
        await pilot.pause()
        assert app.query_one("#invocation-list", ListView).index == 1


async def test_outcome_td_residual_uses_run_gamma(tmp_path):
    run = _write_rich_trace(tmp_path)
    with open(os.path.join(run, "metadata.json"), "w") as f:
        json.dump({"gamma": 0.9}, f)                # γ read from run metadata, like ProbeSession
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._gamma == 0.9
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        summ = str(app.query_one("#outcome-summary", Static).render())
        # inv0: TD δ = r + γV(s') − V(s) = 1.2 + 0.9·(−1.0) − 2.0 = −1.70
        assert "TD δ" in summ and "-1.70" in summ


def _write_threat_trace(tmp_path):
    """A trace whose obs is long enough to hold their_matchups, with a 4× incoming cell."""
    run = tmp_path / "run"
    bd = run / "eval_traces" / "step_1000" / "Test"
    os.makedirs(bd, exist_ok=True)
    actions = {f"switch:m{i}": {"prob": "1.0%", "valid": True} for i in range(6)}
    actions.update({"thunderbolt": {"prob": "92.1%", "valid": True},
                    "earthquake": {"prob": "2.8%", "valid": True},
                    "move2": {"prob": "0.0%", "valid": False},
                    "move3": {"prob": "0.0%", "valid": False},
                    "struggle": {"prob": "0.0%", "valid": False}})
    summary = {"meta": {"step": 1000, "result": "LOSS", "turns": 5, "invocations": 1},
               "invocations": [{"i": 1, "turn": 3, "phase": "move_selection",
                                "chosen": "thunderbolt", "our": {"species": "zapdos"},
                                "opp": {"species": "claydol"}, "actions": actions}]}
    with open(bd / "loss_006_summary.json", "w") as f:
        json.dump(summary, f)
    obs = np.zeros((1, 320), dtype=np.float32)         # ≥ tm_off(164)+144
    block = np.zeros((6, 4, 6), dtype=np.float32)
    block[1, 1, 2] = 4.0 / 4.0                          # a 4× incoming hit somewhere
    obs[0, _OFF.tm_off:_OFF.tm_off + 144] = block.reshape(-1)
    np.savez(bd / "loss_006_states.npz", obs=obs, has_state=np.array([1], dtype=np.int8))
    return str(run)


async def test_matchups_panel_shows_incoming_threat(tmp_path):
    run = _write_threat_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        threat = str(app.query_one("#matchups-threat", Static).render())
        assert "incoming" in threat and "4.00" in threat   # decoded worst incoming hit
        # their_matchups is now its own saliency row
        assert app.query_one("#saliency-table", DataTable).row_count == 5


# A synthetic layout WITH the incoming-damage belief block + a model exposing value_grad,
# so the belief line + the critic (V) saliency rows render. Active flag = last dim of each
# pokemon_full_dim(8) our-mon block; slot 1 is active (idx 1*8+7=15).
_OFF_INC = ObsOffsets(
    mm_off=10, om_off=20, tm_off=164, active_block_dim=5,
    turn_history_offset=300, turn_history_dim=10,
    incoming_off=350, incoming_dim=33, incoming_per_mon=5, incoming_recovery=3,
    pokemon_full_dim=8,
)


class _FakeModelInc(_FakeModel):
    def __init__(self):
        super().__init__()
        self.offsets = _OFF_INC

    def value_grad(self, obs, mask):
        return np.ones(len(obs), dtype=np.float64)   # uniform critic |grad| → renders V rows


def _write_belief_trace(tmp_path):
    """A trace whose obs carries an incoming-damage belief (active slot 1, spec P(KO)=0.9)."""
    run = tmp_path / "run"
    bd = run / "eval_traces" / "step_1000" / "Test"
    os.makedirs(bd, exist_ok=True)
    actions = {f"switch:m{i}": {"prob": "1.0%", "valid": True} for i in range(6)}
    actions.update({"thunderbolt": {"prob": "92.1%", "valid": True},
                    "earthquake": {"prob": "2.8%", "valid": True},
                    "move2": {"prob": "0.0%", "valid": False},
                    "move3": {"prob": "0.0%", "valid": False},
                    "struggle": {"prob": "0.0%", "valid": False}})
    summary = {"meta": {"step": 1000, "result": "LOSS", "turns": 5, "invocations": 1},
               "invocations": [{"i": 1, "turn": 3, "phase": "move_selection",
                                "chosen": "thunderbolt", "our": {"species": "zapdos"},
                                "opp": {"species": "salamence"}, "actions": actions}]}
    with open(bd / "loss_006_summary.json", "w") as f:
        json.dump(summary, f)
    obs = np.zeros((1, 512), dtype=np.float32)
    obs[0, 1 * 8 + 7] = 1.0                              # active flag on our slot 1
    obs[0, 350 + 1 * 5 + 3] = 0.9                        # slot1 spec_pko
    obs[0, 350 + 1 * 5 + 4] = 0.3                        # slot1 p_outspeed
    np.savez(bd / "loss_006_states.npz", obs=obs, has_state=np.array([1], dtype=np.int8))
    return str(run)


async def test_matchups_and_saliency_show_incoming_belief(tmp_path):
    run = _write_belief_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModelInc())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # Matchups panel surfaces the calibrated P(KO) belief for our active slot.
        threat = str(app.query_one("#matchups-threat", Static).render())
        assert "incoming P(KO)" in threat and "active 90%" in threat
        # Saliency table shows BOTH heads (π policy + V critic), incl. the incoming_damage block.
        sal = app.query_one("#saliency-table", DataTable)
        labels = [str(sal.get_row_at(r)[0]) for r in range(sal.row_count)]
        assert any(l.startswith("π ") for l in labels) and any(l.startswith("V ") for l in labels)
        assert any("incoming_damage" in l and l.startswith("V ") for l in labels)


class _FakeModelMB(_FakeModel):
    """Exposes the move-belief + damage-op views so the DAMAGE OP belief block renders in Matchups."""

    def damage_op_view(self, obs, mask):
        hot = {"low": 0.4, "high": 0.8, "crit": 1.0, "pko": 0.9, "acc": 1.0}
        cold = {"low": 0.05, "high": 0.1, "crit": 0.2, "pko": 0.0, "acc": 1.0}
        inc = [{"phys": cold, "spec": hot, "p_outspeed": 0.5, "provenance": 0.6}]   # team slot 0 = our active
        inc += [{"phys": cold, "spec": cold, "p_outspeed": 0.4, "provenance": 0.3}]  # slot 1
        inc += [{"phys": cold, "spec": cold, "p_outspeed": 0.0, "provenance": 0.0} for _ in range(4)]
        return {"incoming": inc, "effect": {}, "incoming_secondary": {},
                "choice_band": {"phys_high_cb": [0.0] * 6, "phys_pko_cb": [0.0] * 6, "p_cb": 0.0},
                "outgoing": None}

    def move_belief(self, obs, mask):
        from agents import gen3_data
        g = gen3_data.moves
        nmoves = max(g.get(m).num for m in g.raw()) + 1
        probs = np.zeros((6, nmoves), dtype=np.float64)
        probs[2, g.get("thunderbolt").num] = 0.85    # opp active = slot 2; believes T-bolt unseen
        probs[2, g.get("toxic").num] = 0.40
        probs[2, g.get("icebeam").num] = 0.95        # already revealed → filtered out of the belief
        empty = {"species": "", "revealed_moves": (), "known": False, "active": False}
        opp = [dict(empty), dict(empty),
               {"species": "blissey", "revealed_moves": ("icebeam",), "known": True, "active": True},
               dict(empty), dict(empty), dict(empty)]
        our = [{"species": "magneton", "active": True}, {"species": "skarmory", "active": False}]
        our += [{"species": "", "active": False} for _ in range(4)]
        return {"opp_probs": probs, "opp_slots": opp, "our_slots": our}


async def test_matchups_shows_move_belief_and_op_incoming(tmp_path):
    """The DAMAGE OP belief block renders the model's guess at the revealed opp's UNSEEN moves
    (already-revealed moves filtered out) + the per-OUR-mon op incoming damage labeled by species."""
    run = _write_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModelMB())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        threat = str(app.query_one("#matchups-threat", Static).render())
        assert "move belief" in threat                            # the new block header
        assert "blissey" in threat and "thunderbolt" in threat     # believed UNSEEN move
        assert "toxic" in threat
        assert "icebeam" in threat                                 # REVEALED move shown with its pinned belief
        assert "incoming (op)" in threat and "magneton" in threat  # per-our-mon op damage, labeled


def _flow_text(app) -> str:
    """The Flow Static's rendered plain text (the diagram is one Text built by _render_flow)."""
    return str(app.query_one("#flow-flow", Static).render())


async def test_flow_shows_both_heads(tmp_path):
    """The Flow section renders both head lanes (π POLICY / V VALUE) with their attribution blocks
    (incl. incoming_damage + a 100% dominant block) — layout-agnostic content check."""
    run = _write_belief_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModelInc())
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#sec-flow", Collapsible).collapsed is False   # default-open
        text = _flow_text(app)
        assert "π POLICY" in text and "V VALUE" in text, text
        assert "incoming_damage" in text and "100%" in text, text


def test_flow_head_lane_sorted():
    """_flow_head_lane sorts AND sizes the attribution bars by mean_abs (|grad|/dim), NOT total_abs
    (sum). The two metrics here order OPPOSITELY (the highest-mean_abs block has the lowest sum and
    vice-versa) so the assertion would fail if the lane still read total_abs."""
    import re

    from main.prober.engine import Saliency, SaliencyBlock
    # total_abs descending order: history > mid > small; mean_abs descending order: small > mid > history.
    # The big-but-diluted "history" block (huge sum, tiny per-dim) must NOT win the lane.
    sal = Saliency(overall_mean_abs=1.0, blocks=(
        SaliencyBlock(name="history", mean_abs=0.1, total_abs=1000.0),
        SaliencyBlock(name="small", mean_abs=0.9, total_abs=9.0),
        SaliencyBlock(name="mid", mean_abs=0.5, total_abs=50.0),
    ))
    app = ProberApp(root="/tmp/prober-nonexistent")
    lane = app._flow_head_lane("V VALUE", "value", sal, "bold magenta", "", {})
    # Bars sort by mean_abs descending → small (100%) → mid → history; the top row parses to 100%.
    pcts = [int(m.group(1)) for ln in lane
            for m in [re.search(r"(\d+)%\s*$", ln.plain)] if m]
    assert pcts == sorted(pcts, reverse=True) and pcts[0] == 100, pcts
    # The top (100%) row must be the highest-mean_abs block ("small"), not the highest-sum ("history").
    top_row = next(ln for ln in lane if re.search(r"100%\s*$", ln.plain))
    assert "small" in top_row.plain and "history" not in top_row.plain, top_row.plain


def test_flow_combine_lanes_side_by_side():
    """_flow_combine_lanes zips two lanes with the gutter, left line padded to LANE_W so the
    divider lands at the same column (the read-across alignment)."""
    from main.prober.app import GUTTER, LANE_W
    app = ProberApp(root="/tmp/prober-nonexistent")
    combined = app._flow_combine_lanes([Text("L1"), Text("L2")], [Text("R1")])
    assert len(combined) == 2
    line0 = combined[0].plain
    assert line0.startswith("L1") and "R1" in line0          # both lanes on one line
    assert line0[LANE_W:LANE_W + len(GUTTER)] == GUTTER       # divider at the fixed column
    assert combined[1].plain.startswith("L2") and combined[1].plain.rstrip().endswith("│")  # right lane empty


async def test_flow_handles_model_free_trace(tmp_path):
    """No captured state / no gradients → the Flow diagram shows a graceful hint, never crashes."""
    run = _write_trace(tmp_path, has_state=0)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "no per-decision attribution" in _flow_text(app)


class _FakeModelPopArt(_FakeModel):
    """A model exposing PopArt stats + a value gradient, so the normalized V companion + the
    Flow V-head lane render (the recorded V=1.5 from _write_trace → norm (1.5-0.5)/2 = +0.50)."""

    def popart_stats(self):
        return (0.5, 2.0)

    def value_grad(self, obs, mask):
        return np.ones(len(obs), dtype=np.float64)


async def test_popart_normalized_value_in_outcome_and_flow(tmp_path):
    """A --use-popart model: the engine attaches the normalized V to ValueView, and BOTH the
    Outcome panel and the Flow V-head caption surface it alongside the real-return V."""
    run = _write_trace(tmp_path)                       # recorded V = 1.5
    app = ProberApp(root=run, injected_model=_FakeModelPopArt())
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        a = app._last_analysis
        assert a.value is not None and a.value.popart_sigma == 2.0
        assert abs(a.value.normalized_recorded - 0.5) < 1e-6       # (1.5 - 0.5) / 2.0
        out = str(app.query_one("#outcome-summary", Static).render())
        assert "norm +0.50" in out
        assert "norm" in _flow_text(app)                           # V-lane caption shows it too


async def test_flow_no_popart_omits_normalized(tmp_path):
    """A model WITHOUT PopArt → normalized fields stay None and the captions show only the real V
    (no 'norm') — the no-PopArt path is unchanged."""
    run = _write_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        a = app._last_analysis
        assert a.value is not None and a.value.normalized_recorded is None
        assert "norm" not in str(app.query_one("#outcome-summary", Static).render())


class _FakeModelArch(_FakeModelInc):
    """Exposes a model architecture() so the Flow pipeline branch renders (one active optional
    phase + one OFF optional, to exercise both the numbered + greyed rows)."""

    def architecture(self):
        # Mirrors the real v27 taxonomy: trunk · side (BeliefHead/WinProbHead) · fork (CLSPool) ·
        # shared (post-fork, feeds both heads) · policy/value heads. Mixes active + off optionals.
        return [
            {"name": "Embeddings", "active": True, "optional": False, "stage": "trunk", "role": "shared tables"},
            {"name": "ObsUnpack", "active": True, "optional": False, "stage": "trunk", "role": "flat obs"},
            {"name": "PokemonEncoder", "active": True, "optional": False, "stage": "trunk", "role": "role tokens"},
            {"name": "BeliefSlots", "active": False, "optional": True, "stage": "trunk", "role": "hidden slots"},
            {"name": "TeamTransformer", "active": True, "optional": False, "stage": "trunk",
             "role": "SELF-ATTENTION over 23 tokens", "attn": True},
            {"name": "BeliefHead", "active": True, "optional": True, "stage": "side", "role": "aux species/moves"},
            {"name": "MoveBelief", "active": True, "optional": True, "stage": "trunk", "role": "reinject moves"},
            {"name": "SpreadBelief", "active": False, "optional": True, "stage": "trunk", "role": "reinject spread"},
            {"name": "CLSPool", "active": True, "optional": False, "stage": "fork", "role": "forks → π · V"},
            {"name": "WinProbHead", "active": True, "optional": True, "stage": "side",
             "role": "P(win) readout off value_pooled (read_only)"},
            {"name": "DamageOperator", "active": False, "optional": True, "stage": "shared", "role": "KO calc"},
            {"name": "ProjectionAssembler", "active": True, "optional": False, "stage": "shared",
             "role": "→ (pi_combined, vf_combined)"},
            {"name": "π policy head", "active": True, "optional": False, "stage": "policy",
             "role": "pi_combined → norm → proj(512)"},
            {"name": "V value head", "active": True, "optional": False, "stage": "value",
             "role": "vf_combined → norm → proj(512)"},
        ]


async def test_flow_renders_architecture_pipeline(tmp_path):
    """The Flow diagram draws the live model's forward DATAFLOW: stage bands (ENCODE/BELIEF/⑂ FORK)
    off a spine, active phases + greyed '⌀off' optionals + '┄▷ … ✗→heads' side readouts, the π/V
    heads as LANES (not pipeline rows), and each head's '↳ reads …' tied back to the arch."""
    run = _write_belief_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModelArch())
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _flow_text(app)
        assert "ENCODE" in text and "FORK" in text, text                  # stage band headers (▼ BAND)
        assert "TeamTransformer" in text and "⊛" in text, text            # active phase + attention marker
        assert "SpreadBelief" in text, text                               # off optional still listed (dim)
        assert "✗→heads" in text, text                                    # side-readout annotation
        assert "ProjectionAssembler" in text, text                        # post-fork shared phase
        assert "π POLICY" in text and "V VALUE" in text, text             # the two head lanes
        # the heads render as LANES, not as pipeline rows (their arch names are excluded)
        assert "policy head" not in text and "value head" not in text, text
        # the V lane's reads must be the VALUE inputs, NOT the WinProbHead role (stage-collision guard)
        assert "vf_combined" in text and "↳ P(win)" not in text, text


async def test_board_tab_populates(tmp_path):
    run = _write_rich_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # our team = active (zapdos) + 2 bench (celebi, snorlax-fainted)
        assert app.query_one("#board-our", DataTable).row_count == 3
        assert app.query_one("#board-opp", DataTable).row_count == 2   # jynx + tyranitar
        summary = str(app.query_one("#board-summary", Static).render())
        assert "zapdos" in summary and "jynx" in summary and "thunderbolt" in summary
        # field line decoded from the (fake) model: weather + spikes
        field = str(app.query_one("#board-field", Static).render())
        assert "SUN" in field and "spikes" in field


async def test_battle_filter_cycles(tmp_path):
    run = _write_rich_trace(tmp_path)  # a single WIN battle
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        tree = app.query_one("#trace-tree", Tree)
        assert "1" in str(tree.root.label)         # 1 battle shown
        app.action_cycle_filter()                   # all → loss (none)
        await pilot.pause()
        assert app._battle_filter == "loss"
        assert len(app._tree_model.all_battles()) == 1  # model unchanged
        # the tree now hides the win battle (loss-only filter)
        assert "loss only" in str(app.query_one("#trace-tree", Tree).root.label)
        app.action_cycle_filter()                   # → win (shows it again)
        await pilot.pause()
        assert app._battle_filter == "win"


async def test_resolves_exact_snapshot_for_battle(tmp_path):
    run = _write_run_with_models(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        choice = app._current_choice
        assert choice.tier == "exact"
        assert choice.path.endswith(os.path.join("step_2000000", "snapshot.zip"))
        assert choice.manifest["git_hash"] == "deadbeef"


async def test_tier_cycle_walks_exact_nearest_recent(tmp_path):
    run = _write_run_with_models(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        assert app._current_choice.tier == "exact"     # auto → exact
        app.action_cycle_model()                        # → nearest
        await pilot.pause()
        assert app._tier == "nearest"
        assert app._current_choice.tier == "nearest"
        assert app._current_choice.path.endswith("checkpoint_3200000_steps.zip")
        app.action_cycle_model()                        # → recent
        await pilot.pause()
        assert app._tier == "recent"
        assert app._current_choice.tier == "recent"
        assert app._current_choice.path.endswith("best_model.zip")
        app.action_cycle_model()                        # → back to auto/exact
        await pilot.pause()
        assert app._tier == "auto" and app._current_choice.tier == "exact"


async def test_review_flag_persists_and_shows_glyph(tmp_path):
    from textual.widgets import Input, Label
    from main.prober.review import ReviewStore
    run = _write_trace(tmp_path, chosen="thunderbolt")
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # the "what the model expected" card populated
        assert str(app.query_one("#review-card", Static).render()).strip() != ""
        # flag the current decision → store + glyph + status update
        app.action_toggle_review_flag()
        await pilot.pause()
        bid = app._battle_id()
        assert app._review_store.flag(bid, 0) is True
        item = app.query_one("#invocation-list", ListView).children[0]
        assert "⚑" in str(item.query_one(Label).render())
        assert "FLAGGED" in str(app.query_one("#review-status", Static).render())
        # note via the input-submitted handler → persists to disk
        note_in = app.query_one("#review-note", Input)
        note_in.value = "explosion at full HP"
        app.on_input_submitted(Input.Submitted(note_in, "explosion at full HP"))
        await pilot.pause()
        assert app._review_store.note(bid, 0) == "explosion at full HP"
        assert ReviewStore(run).flag(bid, 0) is True            # a fresh store reads the file
        # toggling the flag off keeps the note (still annotated → glyph stays)
        app.action_toggle_review_flag()
        await pilot.pause()
        assert app._review_store.flag(bid, 0) is False
        assert 0 in app._review_store.annotated_invs(bid)       # note still annotates it
        # export writes markdown
        path = app._review_store.export_markdown()
        assert path and "explosion at full HP" in open(path).read()


# ── RESULT timeline rendering (battle-log, one line per action, no «1st» tag) ─────────────────────
from types import SimpleNamespace                         # noqa: E402
from main.prober.app import _append_happened              # noqa: E402


def _outcome_with_timeline():
    from main.prober.engine import build_result_timeline
    out = {"our": {"action": "icebeam", "hp_delta": "-72%"},
           "opp": {"action": "hiddenpower → metagross_sent_in", "hp_delta": "-100%"},
           "events": ["opp:salamence:fainted"], "opp_crit": True, "move_order": "opp_first"}
    out["timeline"] = build_result_timeline(
        out, "tyranitar", "salamence", "move_selection",
        our_hp_before="100%", opp_hp_before="100%", our_hp_after="28%", opp_hp_after="100%")
    return out


def test_append_happened_renders_ordered_battle_log():
    a = SimpleNamespace(outcome=_outcome_with_timeline(), our_species="tyranitar",
                        opp_species="salamence", phase="move_selection", next_board=None)
    t = Text()
    _append_happened(t, a, "\nRESULT  ")
    txt = t.plain
    assert "«1st»" not in txt                                     # the confusing order tag is gone
    # The damage we took is on the OPPONENT's move, with before→after HP.
    assert "opp hiddenpower did 72%" in txt and "(tyranitar 100% → 28%)" in txt
    assert "⚡CRIT" in txt
    # Our KO is visible on its own line; the forced replacement on another.
    assert "we icebeam did 100%" in txt and "salamence 100% → faint" in txt
    assert "opp sends in metagross" in txt
    # One line per action (3 actions: opp move, our move, opp send-in), opp first.
    lines = [ln for ln in txt.split("\n") if ln.strip()]
    assert len(lines) == 3
    assert lines[0].startswith("RESULT  opp hiddenpower")          # opp moved first → on top


def test_append_happened_noop_without_outcome():
    a = SimpleNamespace(outcome={}, our_species="a", opp_species="b", phase="", next_board=None)
    t = Text()
    _append_happened(t, a, "\nRESULT  ")
    assert t.plain == ""                                           # nothing recorded → nothing drawn


def test_append_happened_flags_unknown_order():
    from main.prober.engine import build_result_timeline
    out = {"our": {"action": "surf", "hp_delta": "-18%"},
           "opp": {"action": "earthquake", "hp_delta": "-12%"}, "events": []}   # no move_order
    out["timeline"] = build_result_timeline(out, "milotic", "swampert", "move_selection",
                                            our_hp_after="82%", opp_hp_after="88%")
    a = SimpleNamespace(outcome=out, our_species="milotic", opp_species="swampert",
                        phase="", next_board=None)
    t = Text()
    _append_happened(t, a, "\nRESULT  ")
    assert "(move order not recorded)" in t.plain      # honest about the unknown
    assert "· " in t.plain                             # neutral bullets, not an implied 1st/2nd


def test_timeline_renders_resulting_and_no_effect():
    from main.prober.app import _append_timeline_entry
    t = Text()
    _append_timeline_entry(t, {"side": "we", "kind": "move", "move": "rockslide", "crit": True,
                               "target": "celebi", "resulting": True, "hp_after": "11%"})
    assert "rockslide → celebi (now 11%)" in t.plain and "⚡CRIT" in t.plain
    t2 = Text()
    _append_timeline_entry(t2, {"side": "opp", "kind": "move", "move": "seismictoss",
                                "no_effect": "immune"})
    assert "seismictoss — no effect (immune)" in t2.plain
    t3 = Text()
    _append_timeline_entry(t3, {"side": "we", "kind": "move", "move": "hypnosis",
                                "no_effect": "missed"})
    assert "hypnosis — missed" in t3.plain


def test_protocol_text_renders_lines_and_empty():
    from main.prober.app import _protocol_text
    t = _protocol_text(("|turn|4", "|move|p1a: Aerodactyl|Rock Slide|p2a: Celebi",
                        "|-crit|p2a: Celebi", "|-damage|p2a: Celebi|11/100"), 4)
    assert "raw log · turn 4" in t.plain and "Rock Slide" in t.plain and "11/100" in t.plain
    assert "no replay.html" in _protocol_text((), 4).plain


async def test_replay_path_bar_reveals_path_on_own_line(tmp_path):
    """`y` reveals the replay path on a dedicated full-width bar (its own line, selectable under
    `v`), not crammed/wrapped into a toast — and a new battle drops the stale bar."""
    run = _write_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#replay-path-bar", Static)
        assert bar.display is False                              # hidden until used
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        app.action_copy_replay_path()                           # the `y` action
        await pilot.pause()
        assert bar.display is True
        lines = str(bar.render()).split("\n")
        assert "replay path" in lines[0]                        # the hint
        assert lines[-1].endswith("_replay.html")               # the path ALONE on its own line
        # selecting another battle clears the stale path
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        assert bar.display is False


async def test_team_columns_get_real_width_not_collapsed(tmp_path):
    """The Team tab's two panels split the analysis pane (~1fr each). Regression for the auto-width
    collapse that squished each column to ~2 cells, wrapping every word ('po/ké/mo/n') vertically."""
    run = _write_trace(tmp_path)
    app = ProberApp(root=run, injected_model=_FakeModel())
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause()
        app._select_battle(app._tree_model.all_battles()[0])
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.action_toggle_section("sec-team")          # open Team (collapsed by default)
        await pilot.pause()
        our = app.query_one("#team-our", Static)
        opp = app.query_one("#team-opp", Static)
        assert our.size.width > 20 and opp.size.width > 20   # not collapsed to ~2 cells
