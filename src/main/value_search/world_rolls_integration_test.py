"""F1 TIER A on a REAL banked decision (``design_world_sampler.md`` §3.3.1; the Rust half: ``pokesim::rolls``,
``src/rust_sim/tests/rolls_test.rs``): with ``dice="rolls"`` the honest builder's gate accepts a world whose damage
numbers fit the log only on OTHER damage rolls than the battle seed's, the world carries the rolls it took (its log's
``"rolls"``, ``Worlds.rolls``) and its likelihood weight (``Worlds.log_w``), and every replay of it — the info audit's
regate, the search's playout tool — reproduces what the searched side saw. ``dice="seed"`` (the default) is today.

THE GUARDS: the truth's own match replay is byte-identical to the plain truth and takes no other roll (weight > 0);
a control that would exclude the truth RAISES ``RollMatchError`` (a defect, never an exclusion)."""
from __future__ import annotations

import json

import pytest

from main.value_search import worlds as W

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def _gate(root: W.Root, res: dict) -> bool:
    if not res["ok"]:
        return False
    try:
        _, rows = W.reveals(res, root.viewer, root.n)
    except W.WorldError:
        return False
    return rows == root.ref and W.protocol_lines(res, root.viewer, root.n) == root.ref_proto


def _shift(entry: str, stat: int, d: int) -> str:
    """The entry with ``d`` EVs moved from Def (or HP) into stat ``stat`` (the total kept)."""
    f = W.parse_set(entry)
    ev = [int(x) if x else 0 for x in (f[6].split(",") + [""] * 6)[:6]]
    src = 2 if stat != 2 else 0
    if not (0 <= ev[stat] + d <= 252 and 0 <= ev[src] - d <= 252):
        return entry
    ev[stat] += d
    ev[src] -= d
    f[6] = ",".join(str(x) for x in ev)
    return "|".join(f)


@pytest.fixture(scope="module")
def case():
    """The first BANK_V1 decision with an opponent SEEN mon whose one-stat EV shift today's gate rejects and the
    tier-A gate accepts (deterministic: the bank, the order and the shifts are fixed)."""
    from agents.battle.core_replay import run_core
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1
    from main.policy_spectrum.truth import cmd_index, to_log

    bank = B.load_bank(BANK_V1)
    ws = W.WorldSampler(reveal="off", dice="rolls")
    for d in bank.decisions[:120]:
        bat = bank.battles[bank.battle_index[d["battle"]]]
        log, at = to_log(bat), cmd_index(bat, d["side"], d["n"])
        try:
            root = W.prepare_root(log, at, d["side"], d["id"])
        except W.WorldError:
            continue
        match = ws._roll_match(root, d["id"], lambda t: W.render(root.tmpl, t), lambda r, c: (_gate(root, r), 0, None))
        if sum(1 for t in match["targets"] if t[1] == root.oi and t[2] == root.viewer and not t[5]) < 2:
            continue                                   # needs the opponent's hits on our mons (exact HP)
        seen = [i for i, s in enumerate(root.sids) if s in root.kept]
        cands = []
        for i in seen:
            for stat in (1, 3):                        # Atk, SpA
                for dd in (4, 8, -4, -8, 12, -12):
                    e = _shift(root.entries[i], stat, dd)
                    if e != root.entries[i]:
                        cands.append("]".join(root.entries[:i] + [e] + root.entries[i + 1:]))
        if not cands:
            continue
        cmds = [W.render(root.tmpl, t) for t in cands]
        plain = run_core([root.rec(t, f"c{j}", c) for j, (t, c) in enumerate(zip(cands, cmds))], trackers=True, obs=True)
        tier = run_core([root.rec(t, f"m{j}", c, match) for j, (t, c) in enumerate(zip(cands, cmds))],
                        trackers=True, obs=True)
        for t, c, p, m in zip(cands, cmds, plain, tier):
            if not _gate(root, p) and _gate(root, m):
                return dict(log=log, at=at, side=d["side"], key=d["id"], root=root, match=match, team=t, cmds=c,
                            res=m, ws=ws, plain_res=p)
    pytest.fail("precondition: no banked decision with a seen mon whose EV shift only tier A accepts")


def test_the_truth_passes_its_own_match_byte_for_byte(case):
    from agents.battle.core_replay import run_core

    root = case["root"]
    plain = run_core([root.rec(None, "t", W.render(root.tmpl, root.team))], trackers=True, obs=True)[0]
    m = run_core([root.rec(None, "m", W.render(root.tmpl, root.team), case["match"])], trackers=True, obs=True)[0]
    assert m["chunks"] == plain["chunks"] and m["trackers"] == plain["trackers"]     # the identity
    assert m["rolls"]["feasible"] and m["rolls"]["force"] == [] and m["rolls"]["log_w"] <= 0.0
    # Force with the truth's own drawn rolls at EVERY site is the plain replay too
    every = [[s[0], s[4]] for s in m["rolls"]["site_log"]]
    assert len(every) >= 2
    f = run_core([root.rec(None, "f", W.render(root.tmpl, root.team),
                           {"viewer": root.viewer, "mode": "force", "force": every})], trackers=True, obs=True)[0]
    assert f["chunks"] == plain["chunks"] and f["trackers"] == plain["trackers"]


def test_tier_a_accepts_the_shifted_world_on_other_rolls_and_its_table_replays_it(case):
    from agents.battle.core_replay import run_core

    root, res = case["root"], case["res"]
    rr = res["rolls"]
    assert not _gate(root, case["plain_res"]) and _gate(root, res)    # FAILS on revert (no control: the seed's rolls)
    assert rr["feasible"] and rr["force"], "the acceptance took a roll other than the stream's"
    assert rr["log_w"] < 0.0
    f = run_core([root.rec(case["team"], "f", case["cmds"], {"viewer": root.viewer, "mode": "force",
                                                             "force": rr["force"]})], trackers=True, obs=True)[0]
    assert _gate(root, f) and f["chunks"] == res["chunks"]


def test_the_playout_tool_replays_the_worlds_rolls(case):
    """The search's own replay (``rust_env_playout_open``) of the world's log: WITH its ``rolls`` the searched side's
    prefix is the truth's; without them it is another battle (the playout must honour the table, or search plays a
    world the gate never accepted)."""
    root, ws = case["root"], case["ws"]
    lw = dict(case["log"], teams=[case["team"] if s == root.oi else t for s, t in enumerate(case["log"]["teams"])],
              cmds=[f"CHOOSE {c0} {tok}" for c0, tok in case["cmds"]] + list(case["log"]["cmds"][case["at"]:]))
    assert ws.playout_refusal(lw, case["at"], case["side"]) is None
    core = ws._playout_core

    def prefix(lg: dict):
        req = {"log": lg, "at": int(case["at"]), "side": case["side"], "actions": None, "seeds": [None],
               "stall": None, "max_turns": 999, "keep_cmds": False, "text": case["side"]}
        core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())
        return core._cstr(core.lib.rust_env_playout_results)["prefix_text"]

    truth = prefix(case["log"])
    assert prefix(dict(lw, rolls=case["res"]["rolls"]["force"])) == truth
    assert prefix(lw) != truth


def test_the_builder_returns_weighted_worlds_and_the_audit_regates_them(case):
    from main.info_audit import checks as C

    w = case["ws"].sample(case["log"], case["at"], case["side"], 3, case["key"], screen_cache={})
    assert w.dice == "rolls" and len(w.log_w) == len(w.rolls) == len(w.teams) == 3
    assert all(x <= 0.0 for x in w.log_w)
    for lg, r in zip(w.logs, w.rolls):
        assert lg.get("rolls", []) == r                       # the log carries exactly the rolls the world took
    assert C.regate(case["root"], w)["pass"]
    seed = W.WorldSampler(reveal="off").sample(case["log"], case["at"], case["side"], 3, case["key"], screen_cache={})
    assert seed.dice == "seed" and seed.log_w == [] and seed.rolls == []
    assert not any("rolls" in lg for lg in seed.logs)          # today's worlds are untouched


def test_a_control_that_excludes_the_truth_raises(case, monkeypatch):
    from agents.battle import core_replay as CR

    real = CR.run_core

    def corrupt(battles, *a, **k):
        out = real(battles, *a, **k)
        for b, r in zip(battles, out):
            if b.rolls is not None and b.rolls.get("mode") == "record" and r.get("ok"):
                t = r["rolls"]["targets"]
                i = next(j for j, x in enumerate(t) if not x[5])
                t[i] = t[i][:4] + ["no such sight"] + t[i][5:]      # a target the truth cannot print
        return out

    monkeypatch.setattr(CR, "run_core", corrupt)
    with pytest.raises(W.RollMatchError, match="TRUTH-PASSES GUARD"):
        case["ws"]._roll_match(case["root"], "planted", lambda t: W.render(case["root"].tmpl, t),
                               lambda r, c: (_gate(case["root"], r), 0, None))
    assert not issubclass(W.RollMatchError, W.WorldError), "a defect is never an exclusion"


def test_dice_is_validated():
    with pytest.raises(ValueError, match="dice"):
        W.WorldSampler(dice="fresh")
