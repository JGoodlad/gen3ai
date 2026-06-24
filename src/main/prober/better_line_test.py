"""Pure unit tests for the better-line SEARCH backup logic (terminal sentinels, max-backup,
beam pruning, principal variation) — no Node, no torch. The full re-roll → materialize → V pipeline
is exercised against a real battle in ``better_line_integration_test.py``; this pins the tree logic."""

from main.prober.better_line import (
    _WIN, _LOSS, _Node, _backup, _keep_beam, _pv_nodes, _terminal_label)


def _node(label, depth=1, value=None, terminal=None):
    return _Node(label=label, depth=depth, parent=None, sid="n", action=int(label.split(">")[-1]),
                 our_suffixes=[], opp_suffixes=[], our_actions=[], opp_actions=[],
                 ended=terminal is not None, terminal=terminal, value=value)


def test_terminal_label():
    assert _terminal_label({"winner": "me"}, "me") == "win"
    assert _terminal_label({"winner": "you"}, "me") == "loss"
    assert _terminal_label({"winner": None}, "me") == "tie"
    assert _terminal_label({}, "me") == "tie"


def test_backup_leaf_is_its_value():
    n = _node("0>3", value=1.5)
    assert _backup(n) == 1.5


def test_backup_terminal_sentinels():
    assert _backup(_node("0>1", terminal="win")) == _WIN
    assert _backup(_node("0>2", terminal="loss")) == _LOSS
    assert _backup(_node("0>3", terminal="tie")) == 0.0


def test_backup_is_max_over_children():
    root = _node("0>1", value=0.0)
    root.children = [_node("1>2", depth=2, value=-1.0), _node("1>3", depth=2, value=2.5),
                     _node("1>4", depth=2, value=0.7)]
    # max over continuations — we pick the best line, the opponent is the fixed conditional response.
    assert _backup(root) == 2.5


def test_backup_terminal_win_child_floats_up():
    root = _node("0>1", value=0.0)
    root.children = [_node("1>2", depth=2, value=5.0), _node("1>3", depth=2, terminal="win")]
    assert _backup(root) == _WIN


def test_keep_beam_takes_top_value_nonterminal():
    nodes = [_node("0>1", value=0.2), _node("0>2", value=0.9), _node("0>3", value=0.5),
             _node("0>4", terminal="loss"), _node("0>5", value=None)]
    kept = _keep_beam(nodes, beam=2)
    assert [n.action for n in kept] == [2, 3]   # top-2 by value; terminal + value-less excluded


def test_pv_follows_argmax_backup():
    root = _node("0>1", value=0.0)
    a = _node("1>2", depth=2, value=1.0)
    b = _node("1>3", depth=2, value=3.0)
    b.children = [_node("3>7", depth=3, value=4.0)]
    root.children = [a, b]
    _backup(root)
    pv = _pv_nodes(root)
    assert [n.action for n in pv] == [1, 3, 7]   # root → best child (b) → its best grandchild
