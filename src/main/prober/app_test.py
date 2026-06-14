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
from main.prober.app import _DISABLED_GREY, PaneSplitter, ProberApp, _hp_bar
from main.prober.model import ObsOffsets


def test_hp_bar_disabled_is_grey_not_red():
    """A fainted / illegal slot renders grey; an alive low-HP mon keeps its (red) HP gradient."""
    dead = _hp_bar("faint", disabled=True)
    assert all(sp.style == _DISABLED_GREY for sp in dead.spans)
    alive_low = _hp_bar("8%")                       # alive but low → red-ish, NOT the disabled grey
    assert all(sp.style != _DISABLED_GREY for sp in alive_low.spans)

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
        assert app.query_one("#sec-outcome", Collapsible).title == "9  Outcome"
        # header line order: FIELD · CHOSE · RESULT · REWARD · CRITIC (critic moved to last)
        assert (head.index("FIELD") < head.index("CHOSE") < head.index("RESULT")
                < head.index("REWARD") < head.index("CRITIC"))


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
