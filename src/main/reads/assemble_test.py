"""``main.reads.assemble``: the regime id is ALWAYS recorded (anchor units and S ledgers), a mixed regime or a unit
played on other bytes is never pooled, and S is ``population_score`` plus the declared se(S)."""
from __future__ import annotations

import math

import pytest

from agents.training import eval_ledger as L
from agents.training.eval_ledger import testkit as K
from main.reads import assemble as AS

PANELS = ("heldout72_hpfix", "forum72")


def _summary(**cell):
    c = {"battle_format": "gen3ou", "opponent": "metamon:Kakuna", "opponent_version": "Kakuna@ckpt34",
         "our_regime": "greedy", "their_regime": "greedy", "our_team_digest": "d1", "their_team_digest": "d1",
         "server_version": "ws_frontend@aaa", "model_zip": "/a/final_model.zip", "our_team_source": "/pin/a"}
    c.update(cell)
    return {"cell": c, "declared_team_cell": "symmetric", "team_digest_schema": "gen3_team_set_digest_v1"}


def test_the_anchor_regime_id_ignores_the_commit_and_paths_but_not_the_regime():
    a = AS.anchor_regime_id(_summary())
    assert a is not None and len(a) == 16
    assert AS.anchor_regime_id(_summary(server_version="ws_frontend@bbb", model_zip="/b.zip",
                                        our_team_source="/pin/b")) == a
    assert AS.anchor_regime_id(_summary(their_regime="t1")) != a
    assert AS.anchor_regime_id(_summary(our_team_digest="d2")) != a
    assert AS.anchor_regime_id({"status": "OK"}) is None, "a summary that stamps no cell has no regime"


def _row(label, panel, seed, *, wins=10, n=20, counted=True, regime="r" * 16, sha=None):
    return {"unit": f"{label}__{panel}__{seed:02d}", "label": label, "panel": panel, "unit_seed": seed, "wins": wins,
            "n": n, "ties": 0, "counted": counted, "regime_id": regime, "model_sha256": sha}


def _grid(**kw):
    return [_row(lab, ts, s, **kw) for lab in ("a", "b") for ts in PANELS for s in (0, 1)]


def test_an_anchor_group_pools_counted_units_per_team_set_with_one_regime_each():
    rows = _grid()
    rows[0] = dict(rows[0], counted=False)
    g = AS.anchor_group(rows, labels=["a", "b"], teamsets=PANELS, unit_seeds=(0, 1))
    assert g["cells"]["a"]["heldout72_hpfix"] == {"wins": 10, "ties": 0, "n": 20, "units": 1, "planned": 2,
                                                  "skipped": ["a__heldout72_hpfix__00"]}
    assert g["regime_id"] == AS.group_regime_id({ts: "r" * 16 for ts in PANELS}) and not g["inconclusive"]
    assert g["unplayed"] == 0


def test_a_mixed_regime_within_a_team_set_is_never_conclusive():
    rows = _grid()
    rows[1] = dict(rows[1], regime_id="s" * 16)
    g = AS.anchor_group(rows, labels=["a", "b"], teamsets=PANELS, unit_seeds=(0, 1))
    assert g["regime_id"] is None and any("span 2 regimes" in w for w in g["inconclusive"])
    assert any("no regime id" in w for w in AS.conclusive({"anchor": {"u": g}}))


def test_a_unit_without_a_regime_is_never_conclusive():
    g = AS.anchor_group(_grid(regime=None), labels=["a", "b"], teamsets=PANELS, unit_seeds=(0, 1))
    assert g["regime_id"] is None and g["inconclusive"]


def test_a_unit_on_other_bytes_or_twice_or_foreign_refuses():
    with pytest.raises(AS.AssembleError, match="played sha"):
        AS.anchor_group(_grid(sha="x" * 64), labels=["a", "b"], teamsets=PANELS, unit_seeds=(0, 1),
                        shas={"a": "y" * 64, "b": "x" * 64})
    with pytest.raises(AS.AssembleError, match="twice"):
        AS.anchor_group(_grid() + [_row("a", PANELS[0], 0)], labels=["a", "b"], teamsets=PANELS, unit_seeds=(0, 1))
    with pytest.raises(AS.AssembleError, match="not a unit of this group"):
        AS.anchor_group([_row("a", "p72", 0)], labels=["a"], teamsets=PANELS, unit_seeds=(0, 1))


# ------------------------------------------------------------------------------------------------ S
def _ledger(root, family, request, cells, reg=None):
    clock = K.Clock()
    w = L.LedgerWriter(root, "h2h", clock=clock, host="testhost", pid=11, alive=lambda _p: True)
    w.register_family(family, decision_kind="ab_verdict", rule="t", protocol=K.PROTO)
    req = w.open_request(request, kind="ab_cell", purpose="ab", family=family, spec={})
    reg = reg or K.regime()
    for i, (p, o, pc) in enumerate(cells):
        clock.advance(60)
        c = w.claim(req["request_id"], batch=0, player=p, opponent=o, regime_id=reg["regime_id"], expected_wall_s=60)
        w.append_row(K.make_row(w.next_row_id(), request=req, batch=0, p=p, o=o, reg=reg, pc=pc, ts=clock.iso(),
                                purpose="ab", cycle_seed=1000 + i), c)
    return reg["regime_id"]


MEMBERS = {"m1": K.SHA_B, "m2": K.SHA_C}


def test_an_s_group_records_the_family_reads_regime_and_the_declared_se(tmp_path):
    pc1, pc2 = (50, 100, 150, 100, 100), (100, 100, 100, 100, 100)
    rid = _ledger(tmp_path / "led", "fam", "g_a_b100", [(K.SHA_A, K.SHA_B, pc1), (K.SHA_A, K.SHA_C, pc2)])
    g = AS.s_group(tmp_path / "led", "fam", {"a": "g_a_b100"}, {"a": K.SHA_A}, MEMBERS)
    assert g["regime_id"] == rid and len(rid) == 16 and g["ledger_digest"] and not g["inconclusive"]
    n = g["nodes"]["a"]
    s1 = sum(i * c for i, c in enumerate(pc1)) / (4 * 500)
    assert n["per_member_pp"] == {"m1": 100 * s1, "m2": 50.0} and n["s_pp"] == (100 * s1 + 50.0) / 2
    se = math.sqrt(n["per_member_se_pp"]["m1"] ** 2 + n["per_member_se_pp"]["m2"] ** 2) / 2
    assert n["se_pp"] == se and n["ci95_pp"] == [n["s_pp"] - 1.959963985 * se, n["s_pp"] + 1.959963985 * se]


def test_an_s_group_on_another_regime_or_panel_is_inconclusive(tmp_path):
    pc = (100, 100, 100, 100, 100)
    _ledger(tmp_path / "led", "fam", "q", [(K.SHA_A, K.SHA_B, pc), (K.SHA_A, K.SHA_C, pc)])
    g = AS.s_group(tmp_path / "led", "fam", {"a": "q"}, {"a": K.SHA_A}, MEMBERS, want_regime="0" * 16)
    assert any("not the registered" in w for w in g["inconclusive"])
    g = AS.s_group(tmp_path / "led", "fam", {"a": "q"}, {"a": K.SHA_A}, MEMBERS, panel="home72")
    assert any("not panel home72" in w for w in g["inconclusive"])
    g = AS.s_group(tmp_path / "led", "fam", {"a": "q2"}, {"a": K.SHA_A}, MEMBERS)
    assert any("request q2" in w for w in g["inconclusive"]) and g["nodes"]["a"]["s_pp"] is None


def test_a_short_cell_leaves_no_s(tmp_path):
    _ledger(tmp_path / "led", "fam", "q", [(K.SHA_A, K.SHA_B, (0, 0, 499, 0, 0)), (K.SHA_A, K.SHA_C, (0, 0, 500, 0, 0))])
    n = AS.s_group(tmp_path / "led", "fam", {"a": "q"}, {"a": K.SHA_A}, MEMBERS)["nodes"]["a"]
    assert n["s_pp"] is None and any("499 completed pairs < 500" in w for w in n["inconclusive"])


def test_an_unregistered_family_is_a_named_refusal(tmp_path):
    _ledger(tmp_path / "led", "fam", "q", [])
    with pytest.raises(AS.AssembleError, match="other"):
        AS.s_group(tmp_path / "led", "other", {"a": "q"}, {"a": K.SHA_A}, MEMBERS)
