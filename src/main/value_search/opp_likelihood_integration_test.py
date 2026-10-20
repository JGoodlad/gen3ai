"""The OPPONENT-CHOICE LIKELIHOOD on real banked decisions (``opp_likelihood``; ``design_world_sampler.md`` §4 (f)):
it scores the OPPONENT's own rows (its seat, its world team), so a world that differs from the truth only in an
unrevealed move scores differently; it never scores a choice the searched side did not see (a slot-rewritten choice,
or the opponent's choice for the root round); and a world replayed off the history its gate accepted RAISES.

The forward is a fixed random projection of the row (model-free, deterministic): the likelihood's plumbing is under
test, not a network."""
from __future__ import annotations

import numpy as np
import pytest

from main.value_search import opp_likelihood as OL
from main.value_search import worlds as W

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: the BANK_V1 decision whose opponent chose its root-round switch BEFORE the searched side (found by a scan)
PENDING_ID = "ai_v14_01_base/step_36000000/random/win_s2_001#p1#29"


class Rec:
    """A row-dependent forward that records every (row, mask) it is asked for."""

    def __init__(self):
        from agents.battle.core_obs import obs_dim

        rng = np.random.default_rng(7)
        self.P = rng.normal(0.0, 0.05, size=(obs_dim(), 12)).astype(np.float64)
        self.rows: list = []

    def __call__(self, rows, masks):
        self.rows.extend(np.asarray(r).tobytes() for r in rows)
        return (np.asarray(rows, dtype=np.float64) @ self.P).astype(np.float32)


def _bank():
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1

    return B.load_bank(BANK_V1)


def _decision(bank, d):
    from main.policy_spectrum.truth import cmd_index, to_log

    bat = bank.battles[bank.battle_index[d["battle"]]]
    return to_log(bat), cmd_index(bat, d["side"], d["n"]), d["side"]


def _with_team(log: dict, root: W.Root, team: str) -> dict:
    cmds = [f"CHOOSE {s_} {tok}" for s_, tok in W.render(root.tmpl, team)]
    return dict(log, teams=[team if s == root.oi else t for s, t in enumerate(log["teams"])],
                cmds=cmds + list(log["cmds"][root.at:]))


def _same_view(root: W.Root, team: str) -> bool:
    """The world gate, independent of the module under test: the searched side's rows up to the root are the truth's."""
    from agents.battle.core_replay import run_core

    res = run_core([root.rec(team, "v", W.render(root.tmpl, team))], trackers=True, obs=True)[0]
    try:
        return bool(res["ok"]) and W.reveals(res, root.viewer, root.n)[1] == root.ref
    except W.WorldError:
        return False


@pytest.fixture(scope="module")
def case():
    """The first BANK_V1 decision with ≥ 3 scored opponent steps and a seen opponent mon holding an unrevealed move
    that a stand-in move can replace without moving the searched side's rows (deterministic)."""
    bank = _bank()
    for d in bank.decisions[:200]:
        log, at, side = _decision(bank, d)
        try:
            root = W.prepare_root(log, at, side, d["id"])
        except W.WorldError:
            continue
        steps = OL.opponent_steps(root)
        if sum(1 for s in steps if s.status == "scored") < 3:
            continue
        for i, sid in enumerate(root.sids):
            if sid not in root.kept:
                continue
            f = W.parse_set(root.entries[i])
            moves = f[4].split(",")
            unrev = [k for k, m in enumerate(moves) if W.toid(m) not in root.kept[sid] and not W.is_hp(W.toid(m))]
            for k in unrev:
                for sub in ("protect", "toxic", "substitute", "rest"):
                    if sub in {W.toid(m) for m in moves}:
                        continue
                    g = list(f)
                    g[4] = ",".join(sub if j == k else m for j, m in enumerate(moves))
                    team = "]".join(root.entries[:i] + ["|".join(g)] + root.entries[i + 1:])
                    wlog = _with_team(log, root, team)
                    if not _same_view(root, team):
                        continue
                    return dict(log=log, at=at, side=side, root=root, steps=steps, world=wlog, key=d["id"])
    pytest.fail("precondition: no banked decision with a replaceable unrevealed move and 3 scored opponent steps")


def test_the_rows_scored_are_the_opponents_own(case):
    """FAILS ON REVERT of the seat (scoring the searched side's rows): every forwarded row is the opponent's frame at a
    scored step of the truth's replay, in order, and none is a frame of ours."""
    from agents.battle.core_obs import wrap_row
    from agents.battle.core_replay import run_core

    root, steps = case["root"], case["steps"]
    rec = Rec()
    res = OL.world_logliks(root, [OL.truth_log(root)], rec, steps)[0]
    true = run_core([root.rec(None, "t", root.cmds)], trackers=True, obs=True)[0]
    opp = [wrap_row(e["obs"]).tobytes() for e in true["trackers"][root.oi] if "obs" in e]
    ours = {wrap_row(e["obs"]).tobytes() for e in true["trackers"][root.viewer] if "obs" in e}
    want = [opp[s.k] for s in steps if s.status == "scored"]
    assert res.n_scored == len(want) >= 3
    assert sorted(set(rec.rows)) == sorted(set(want))
    assert not (set(rec.rows) & ours)


def test_a_world_differing_only_in_an_unrevealed_move_scores_differently(case):
    """The evidence channel: the opponent's own row shows its moves, so a world with another unrevealed move gives
    its observed choices another likelihood; a model blind to the row (constant logits) gives the same one."""
    root, steps = case["root"], case["steps"]
    t, w = OL.world_logliks(root, [OL.truth_log(root), case["world"]], Rec(), steps)
    assert t.n_scored == w.n_scored >= 3
    assert abs(t.ll - w.ll) > 1e-6

    def flat(rows, masks):
        return np.zeros((len(rows), 12), dtype=np.float32)

    t0, w0 = OL.world_logliks(root, [OL.truth_log(root), case["world"]], flat, steps)
    assert t0.ll == w0.ll


def test_the_likelihood_is_deterministic(case):
    root, steps = case["root"], case["steps"]
    a = OL.world_logliks(root, [OL.truth_log(root), case["world"]], Rec(), steps)
    b = OL.world_logliks(root, [OL.truth_log(root), case["world"]], Rec(), steps)
    assert [x.ll for x in a] == [x.ll for x in b] and [x.conf for x in a] == [x.conf for x in b]


def test_a_world_off_its_gated_history_raises(case):
    """A world whose REVEALED move differs replays another history for the searched side: a defect, never a score."""
    root = case["root"]
    for i, sid in enumerate(root.sids):
        if sid not in root.kept or not root.kept[sid]:
            continue
        f = W.parse_set(root.entries[i])
        moves = f[4].split(",")
        k = next((j for j, m in enumerate(moves) if W.toid(m) in root.kept[sid]), None)
        if k is None:
            continue
        g = list(f)
        g[4] = ",".join("splash" if j == k else m for j, m in enumerate(moves))
        team = "]".join(root.entries[:i] + ["|".join(g)] + root.entries[i + 1:])
        bad = dict(case["world"], teams=[team if s == root.oi else t for s, t in enumerate(case["log"]["teams"])])
        with pytest.raises(OL.OppLikelihoodError):
            OL.world_logliks(root, [bad], Rec(), case["steps"])
        return
    pytest.fail("precondition: the case has a seen mon with a revealed move")


def test_slot_rewritten_choices_are_never_scored():
    """A choice the searched side never saw (rewritten to a SLOT by the template) is ``unobserved``: no row of it is
    forwarded, in the truth or in a world whose token there differs."""
    bank = _bank()
    for d in bank.decisions[:200]:
        log, at, side = _decision(bank, d)
        try:
            root = W.prepare_root(log, at, side, d["id"])
        except W.WorldError:
            continue
        opp = "p2" if side == "p1" else "p1"
        rewritten = {i for i, (s_, t) in enumerate(root.tmpl) if s_ == opp and isinstance(t, tuple)}
        if not rewritten:
            continue
        steps = OL.opponent_steps(root)
        un = [s for s in steps if s.status == "unobserved"]
        assert {s.cmd for s in un} == rewritten - {s.cmd for s in steps if s.status == "pending"}
        if not any(s.status == "scored" for s in steps):
            continue
        rec = Rec()
        res = OL.world_logliks(root, [OL.truth_log(root)], rec, steps)[0]
        assert res.n_unobserved == len(un)
        assert res.n_scored == sum(1 for s in steps if s.status == "scored") == len(rec.rows)
        return
    pytest.fail("precondition: no banked decision with a slot-rewritten opponent choice")


def test_the_opponents_root_round_choice_is_pending_and_never_scored():
    """FAILS ON REVERT of the pending rule: the opponent chose its root-round action before the searched side, which
    has not seen it. It is ``pending``, and replacing it by another legal choice leaves the likelihood unchanged."""
    from agents.battle.core_replay import run_core

    bank = _bank()
    d = next(x for x in bank.decisions if x["id"] == PENDING_ID)
    log, at, side = _decision(bank, d)
    root = W.prepare_root(log, at, side, d["id"])
    steps = OL.opponent_steps(root)
    assert steps[-1].status == "pending" and steps[-1].cmd == at - 1
    assert all(s.status != "pending" for s in steps[:-1])
    base = OL.world_logliks(root, [OL.truth_log(root)], Rec(), steps)[0]
    assert base.n_pending == 1
    true = run_core([root.rec(None, "t", root.cmds)], trackers=True, obs=True)[0]
    ent = [e for e in true["trackers"][root.oi] if "obs" in e][steps[-1].k]
    other = sorted(t for t in ent["tokens"].values() if t != ent["choice"])
    assert other, "precondition: the pending decision has another legal choice"
    cmds = list(log["cmds"])
    cmds[at - 1] = f"CHOOSE {'p2' if side == 'p1' else 'p1'} {other[0]}"
    log2 = dict(log, cmds=cmds)
    root2 = W.prepare_root(log2, at, side, d["id"])
    steps2 = OL.opponent_steps(root2)
    assert [s.status for s in steps2] == [s.status for s in steps]
    alt = OL.world_logliks(root2, [OL.truth_log(root2)], Rec(), steps2)[0]
    assert alt.ll == base.ll and alt.n_scored == base.n_scored
