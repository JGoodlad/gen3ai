"""Pure unit tests for the counterfactual runner's SCRIPTING logic — fake player + fake battle, no
bridge, no Node. The real re-roll/seed/obs faithfulness is proven end-to-end against real battles in
``counterfactual_fuzz_test.py``; this pins the per-decision mode machine on top of it."""

from types import SimpleNamespace

import numpy as np
import pytest

from poke_env.player.battle_order import SingleBattleOrder

from utils.bridge.counterfactual import (
    _battle_outcome, _force_switch, _invert_choice, _passthrough, install_scripted_prefix,
    summarize_trajectory)


def test_passthrough_message():
    assert _passthrough("move 3").message == "/choose move 3"
    assert _passthrough("switch 2").message == "/choose switch 2"


def test_force_switch_bool_and_list():
    assert _force_switch(SimpleNamespace(force_switch=True)) is True
    assert _force_switch(SimpleNamespace(force_switch=False)) is False
    assert _force_switch(SimpleNamespace(force_switch=[False, True])) is True
    assert _force_switch(SimpleNamespace()) is False        # missing attr → False


def test_battle_outcome_maps_win_loss_tie():
    def player(won, finished):
        b = SimpleNamespace(won=won, finished=finished, turn=12)
        return SimpleNamespace(_battles={"t": b})
    assert _battle_outcome(player(True, True), "me")["outcome"] == "win"
    assert _battle_outcome(player(True, True), "me")["winner"] == "me"
    assert _battle_outcome(player(False, True), "me")["outcome"] == "loss"
    assert _battle_outcome(player(None, True), "me")["outcome"] == "tie"
    assert _battle_outcome(player(None, False), "me")["outcome"] == "unfinished"
    assert _battle_outcome(SimpleNamespace(_battles={}), "me")["outcome"] == "unfinished"


class _FakeTracker:
    def __init__(self):
        self.last_ctx = SimpleNamespace(mask=np.ones(11, dtype=np.int8), legal=None)
        self.advanced = []

    def advance(self, idx):
        self.advanced.append(int(idx))


class _FakePlayer:
    """A Gen3Player-shaped stand-in: ``action_to_order(idx)`` → ``move <idx>`` so a recorded command
    ``"move k"`` inverts to index k. ``choose_move`` is the LIVE policy."""

    def __init__(self):
        self.tracker = _FakeTracker()
        self.live_calls = []
        self.embeds = []

    def _handle_stall(self, battle, suffix):
        return None

    def embed_battle(self, battle):
        self.embeds.append(battle.turn)
        return {"observation": np.array([float(battle.turn)], dtype=np.float32),
                "action_mask": np.ones(11, dtype=np.int8)}

    def _get_tracker(self, battle):
        return self.tracker

    def action_to_order(self, idx, battle):
        return SingleBattleOrder(f"/choose move {idx}")

    def choose_default_move(self):
        return SingleBattleOrder("/choose default")

    def choose_move(self, battle):                          # the LIVE policy (post-handoff)
        self.live_calls.append(battle.turn)
        return SingleBattleOrder("/choose LIVE")


def _battle(turn, force_switch=False):
    return SimpleNamespace(turn=turn, force_switch=force_switch)


def _record(commands):
    return SimpleNamespace(commands=commands)


def test_invert_choice_recovers_index():
    p = _FakePlayer()
    assert _invert_choice(p, _battle(1), "move 4") == 4
    assert _invert_choice(p, _battle(1), "switch 9") is None   # not a producible message → no match


def test_our_side_scripts_then_substitutes_then_goes_live():
    p = _FakePlayer()
    cmds = [("p1", "move 0"), ("p1", "move 1"), ("p1", "move 2")]   # recorded p1 line
    install_scripted_prefix(p, side="p1", record=_record(cmds), divergence_turn=3,
                            substitute_choice="move 7", is_our_side=True)

    # turns 1, 2 < divergence → replay the recorded commands (and advance the tracker faithfully).
    assert p.choose_move(_battle(1)).message == "/choose move 0"
    assert p.choose_move(_battle(2)).message == "/choose move 1"
    assert p.tracker.advanced == [0, 1]
    # turn 3 == divergence move round → the SUBSTITUTE, tracker advanced with its index, then LIVE.
    assert p.choose_move(_battle(3)).message == "/choose move 7"
    assert p.tracker.advanced == [0, 1, 7]
    # turn 4 → live policy (NOT a recorded passthrough).
    assert p.choose_move(_battle(4)).message == "/choose LIVE"
    assert p.live_calls == [4]


def test_our_side_forced_switch_at_divergence_goes_live_off_script():
    # A FORCED SWITCH at the divergence turn is OFF-SCRIPT (the board diverged) → hand to the live
    # policy, NEVER pop the recorded command (which would desync the flat deque). Distinct from the
    # normal move-round substitute (tested above), which has force_switch False.
    p = _FakePlayer()
    cmds = [("p1", "move 0"), ("p1", "switch 5"), ("p1", "move 2")]
    install_scripted_prefix(p, side="p1", record=_record(cmds), divergence_turn=2,
                            substitute_choice="move 7", is_our_side=True)
    assert p.choose_move(_battle(1)).message == "/choose move 0"          # prefix recorded
    # turn 2, FORCED SWITCH (off-script) → live, NOT the recorded "switch 5".
    assert p.choose_move(_battle(2, force_switch=True)).message == "/choose LIVE"
    assert 2 in p.live_calls


def test_opponent_forced_switch_at_divergence_goes_live_not_scripted():
    # The faithfulness fix (finding #1): when OUR substitute KOs the opponent at turn T, the opp gets
    # an OFF-SCRIPT forced switch at turn T. It must go LIVE (the reloaded policy switches), NOT pop the
    # recorded turn-(T+1) move for a forceSwitch (the desync bug).
    p = _FakePlayer()
    cmds = [("p2", "move 0"), ("p2", "move 1"), ("p2", "switch 3")]
    install_scripted_prefix(p, side="p2", record=_record(cmds), divergence_turn=2,
                            substitute_choice=None, is_our_side=False)
    assert p.choose_move(_battle(1)).message == "/choose move 0"          # prefix recorded
    assert p.choose_move(_battle(2)).message == "/choose move 1"          # opp turn-T MOVE round (recorded)
    # opp forced switch at turn T (off-script) → LIVE, must NOT pop the recorded "switch 3".
    assert p.choose_move(_battle(2, force_switch=True)).message == "/choose LIVE"
    assert 2 in p.live_calls


def test_opponent_side_scripts_through_divergence_then_live():
    p = _FakePlayer()
    cmds = [("p2", "move 0"), ("p2", "move 1"), ("p2", "move 2")]
    install_scripted_prefix(p, side="p2", record=_record(cmds), divergence_turn=2,
                            substitute_choice=None, is_our_side=False)
    # opponent replays recorded for turns <= divergence (it can't react to our change on turn T).
    assert p.choose_move(_battle(1)).message == "/choose move 0"
    assert p.choose_move(_battle(2)).message == "/choose move 1"   # turn == divergence → still recorded
    # turn 3 > divergence → live.
    assert p.choose_move(_battle(3)).message == "/choose LIVE"
    assert p.live_calls == [3]


def test_full_replay_mode_scripts_everything():
    p = _FakePlayer()
    cmds = [("p1", "move 0"), ("p1", "move 1")]
    install_scripted_prefix(p, side="p1", record=_record(cmds), divergence_turn=None,
                            substitute_choice=None, is_our_side=True)
    assert p.choose_move(_battle(5)).message == "/choose move 0"   # divergence_turn=None → never live
    assert p.choose_move(_battle(9)).message == "/choose move 1"
    assert p.live_calls == []


def test_obs_sink_captures_scripted_obs():
    p = _FakePlayer()
    sink = []
    cmds = [("p1", "move 0"), ("p1", "move 1")]
    install_scripted_prefix(p, side="p1", record=_record(cmds), divergence_turn=None,
                            substitute_choice=None, is_our_side=True, obs_sink=sink)
    p.choose_move(_battle(1))
    p.choose_move(_battle(2))
    assert [float(o[0]) for o in sink] == [1.0, 2.0]              # one obs per scripted decision


def test_non_gen3_player_just_passes_through():
    """A bot opponent (no embed_battle) scripts by pure pass-through — no tracker, no obs."""
    class _Bot:
        def __init__(self):
            self.live = []

        def choose_move(self, battle):
            self.live.append(battle.turn)
            return SingleBattleOrder("/choose LIVE")

    b = _Bot()
    install_scripted_prefix(b, side="p2", record=_record([("p2", "move 0")]),
                            divergence_turn=1, substitute_choice=None, is_our_side=False)
    assert b.choose_move(_battle(1)).message == "/choose move 0"   # recorded passthrough
    assert b.choose_move(_battle(2)).message == "/choose LIVE"     # live after divergence


def test_summarize_trajectory_parses_protocol():
    side = "p1"   # trainee is p1; opp is p2
    chunks = [
        ("p1", "|turn|5\n|switch|p1a: Gengar|Gengar, M|100/100\n"
               "|move|p2a: Swampert|Earthquake|p1a: Gengar\n|-immune|p1a: Gengar\n"),
        ("p2", "|turn|5\n|move|p2a: Swampert|Earthquake|p1a: Gengar\n"),   # opp-side chunk → must be IGNORED
        ("p1", "|turn|6\n|move|p1a: Gengar|Ice Beam|p2a: Swampert\n|-supereffective|p2a: Swampert\n"
               "|-crit|p2a: Swampert\n|-damage|p2a: Swampert|0 fnt\n|faint|p2a: Swampert\n|win|TraineeName\n"),
    ]
    turns = {t["turn"]: t["events"] for t in summarize_trajectory(side, chunks)}
    assert set(turns) == {5, 6}
    # turn 5: we switch to Gengar; opp Earthquakes; Gengar immune (no double-count from the p2 chunk).
    assert any("we sent in Gengar" in e for e in turns[5])
    assert sum("opp used Earthquake" in e for e in turns[5]) == 1   # the p2-side chunk was ignored
    assert any("immune" in e for e in turns[5])
    # turn 6: we Ice Beam → super-effective crit → Swampert faints → we win.
    assert any("we used Ice Beam" in e for e in turns[6])
    assert any("super-effective" in e for e in turns[6])
    assert any("crit" in e for e in turns[6])
    assert any("Swampert FAINTED" in e for e in turns[6])
    assert any("WINS" in e for e in turns[6])


def test_summarize_trajectory_hp_fraction_and_malformed():
    side = "p2"   # trainee is p2 this time
    chunks = [("p2", "|turn|3\n|move|p2a: Milotic|Surf|p1a: Tyranitar\n"
                     "|-damage|p1a: Tyranitar|140/300\n|garbage line no pipe\n|\n")]
    turns = {t["turn"]: t["events"] for t in summarize_trajectory(side, chunks)}
    assert any("we used Surf" in e for e in turns[3])
    assert any("47% hp" in e for e in turns[3])   # 140/300 → 47%
