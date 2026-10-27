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


# ------------------------------------------------------------------------------------ models with different loads
LAM = "+belief_prior_mix=0.25"


def _counted_summary(**cell):
    s = _summary(**cell)
    s.update(status="OK", n=20, wins=10, losses=10, ties=0, regime_verified_decisions=True, peer_clean=True,
             team_source_asymmetry=False)
    return s


def _unit(label, panel, seed, lam, **cell):
    """A played unit's row as `assemble.unit_row` builds it, for a model that declared load option ``lam``."""
    from types import SimpleNamespace as NS

    stamp = f"greedy{LAM}" if lam else "greedy"
    u = NS(name=f"{label}__{panel}__{seed:02d}", label=label, teamset=panel, unit_seed=seed, games=20,
           model_sha256="0" * 64, belief_prior_mix=lam)
    return AS.unit_row(u, _counted_summary(our_regime=cell.pop("our_regime", stamp), teamset=panel, **cell))


def _mixed_rows(**b_cell):
    return ([_unit("a", ts, s, None) for ts in PANELS for s in (0, 1)]
            + [_unit("b", ts, s, 0.25, **b_cell) for ts in PANELS for s in (0, 1)])


LOADS = {"a": {}, "b": {"belief_prior_mix": 0.25}}


def _group(rows, loads=LOADS):
    return AS.anchor_group(rows, labels=["a", "b"], teamsets=PANELS, unit_seeds=(0, 1), loads=loads)


def test_the_base_id_of_a_loaded_unit_is_the_id_of_the_same_cell_without_the_load():
    plain = _counted_summary()
    loaded = _counted_summary(our_regime=f"greedy{LAM}")
    assert AS.anchor_regime_id(loaded) != AS.anchor_regime_id(plain), "a load changes the full id (it never pools)"
    assert AS.anchor_regime_id(loaded, strip_load=LAM) == AS.anchor_regime_id(plain)
    assert AS.anchor_regime_id(plain, strip_load=LAM) is None, "a unit that declared a load and did not stamp it"
    assert _unit("b", PANELS[0], 0, 0.25)["base_regime_id"] == _unit("a", PANELS[0], 0, None)["regime_id"]


def test_one_group_holds_models_with_different_loads_with_a_regime_id_per_load_and_one_base():
    rows = _mixed_rows()
    g = _group(rows)
    assert g["inconclusive"] == [] and g["unplayed"] == 0 and g["regime_id"] is not None
    plain = {ts: _unit("a", ts, 0, None)["regime_id"] for ts in PANELS}
    loaded = {ts: _unit("b", ts, 0, 0.25)["regime_id"] for ts in PANELS}
    assert g["regime_ids"] == plain, "the group's per-team-set id is the BASE id every model shares"
    assert g["regime_ids_by_load"] == {"none": plain, LAM: loaded} and plain != loaded
    assert g["loads"] == {"a": "", "b": LAM}
    assert g["regime_id"] == AS.mixed_group_regime_id(plain, {"none": plain, LAM: loaded})
    assert g["regime_id"] != AS.group_regime_id(plain)
    legacy = _group(rows, loads=None)             # what the group was before: every load a "mixed regime"
    assert legacy["regime_id"] is None and any("span 2 regimes" in w for w in legacy["inconclusive"])


def test_a_different_lambda_is_a_different_group_regime():
    a = _group(_mixed_rows())
    rows = ([_unit("a", ts, s, None) for ts in PANELS for s in (0, 1)]
            + [dict(_unit("b", ts, s, 0.5, our_regime="greedy+belief_prior_mix=0.5")) for ts in PANELS for s in (0, 1)])
    b = _group(rows, loads={"a": {}, "b": {"belief_prior_mix": 0.5}})
    assert a["regime_id"] and b["regime_id"] and a["regime_id"] != b["regime_id"]


def test_models_with_one_shared_load_keep_the_group_id_they_always_had():
    rows = [_unit(lab, ts, s, 0.25) for lab in ("a", "b") for ts in PANELS for s in (0, 1)]
    same = {"a": {"belief_prior_mix": 0.25}, "b": {"belief_prior_mix": 0.25}}
    g = _group(rows, loads=same)
    assert g["regime_id"] == AS.group_regime_id({ts: _unit("a", ts, 0, 0.25)["regime_id"] for ts in PANELS})
    assert "regime_ids_by_load" not in g and g == _group(rows, loads=None)
    plain = [_unit(lab, ts, s, None) for lab in ("a", "b") for ts in PANELS for s in (0, 1)]
    assert _group(plain, loads={"a": {}, "b": {}}) == _group(plain, loads=None)


def test_a_mixed_load_group_is_not_conclusive_when_a_model_differs_in_more_than_its_load():
    other = _group(_mixed_rows(their_regime="t1"))
    assert other["regime_id"] is None and any("BASE regimes" in w for w in other["inconclusive"]), other["inconclusive"]
    unstamped = _group(_mixed_rows(our_regime="greedy"))      # b declared lambda = 0.25 but its units stamp nothing
    assert AS.conclusive({"anchor": {"u": unstamped}}), "the id may exist, the group is still not conclusive"
    assert any("does not end with" in w and LAM in w for w in unstamped["inconclusive"]), unstamped["inconclusive"]
    rows = _mixed_rows()
    rows[0] = _unit("a", PANELS[0], 0, None, their_team_digest="dX")
    drift = _group(rows)
    assert drift["regime_id"] is None and any("units of load 'none' span 2 regimes" in w or "BASE regimes" in w
                                              for w in drift["inconclusive"]), drift["inconclusive"]
    only_a = _group([r for r in _mixed_rows() if r["label"] == "a"])
    assert only_a["regime_id"] is None and any(f"no counted unit of load '{LAM}'" in w for w in only_a["inconclusive"])


def test_assemble_reads_each_models_load_from_the_spec(tmp_path, run_archive_env, monkeypatch):
    """End to end over a planned state: the units the kakuna plan carries get their stamps, the group gets its ids."""
    import json
    import time

    from main.reads import api, layout as LY, runner as RN
    from main.reads.runner_test import CODE, _spec as _rspec, _step, _zip

    p = _rspec(tmp_path, run_archive_env)
    doc = json.loads(p.read_text())
    doc["models"]["new"]["load"] = {"belief_prior_mix": 0.25}
    p.write_text(json.dumps(doc))
    _zip(run_archive_env / "rb_new", age=3600)
    state = tmp_path / "state"
    RN.plan(p, state, code=CODE, step_of=_step)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)

    def unit(st, u):
        d = st / "cells" / u.name
        d.mkdir(parents=True, exist_ok=True)
        stamp = "greedy" + (f"+belief_prior_mix={u.belief_prior_mix:g}" if u.belief_prior_mix else "")
        s = _counted_summary(our_regime=stamp, teamset=u.teamset)
        (d / "summary.json").write_text(json.dumps(s))
        return 0

    assert RN.run(state, "anchor", time.time() + 3600, run_unit=unit, sleep=lambda s: None) == 0
    g = api.assemble(state)["anchor"]["u"]
    assert g["inconclusive"] == [] and g["regime_id"] and g["loads"] == {"old": "", "new": LAM}
    assert set(g["regime_ids_by_load"]) == {"none", LAM}
    assert {u["label"]: u["our_regime"] for u in g["units"]} == {"old": "greedy", "new": f"greedy{LAM}"}
    assert json.loads(LY.cells_file(state).read_text())["anchor"]["u"]["regime_id"] == g["regime_id"]


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


# ------------------------------------------------------------------ the Metamon WEIGHTS in the regime id (FR-1 FINDING 3)
_WCELL = {"battle_format": "gen3ou", "opponent": "metamon:Kakuna", "opponent_agent": "Kakuna", "opponent_kind": "metamon",
          "opponent_version": "Kakuna@ckpt34", "our_regime": "greedy", "their_regime": "greedy",
          "regime_matched": True, "teamset": "away", "our_team_digest": "d1", "their_team_digest": "d1",
          "our_side": "model"}
#: ``anchor_regime_id`` of ``_WCELL`` computed by the code BEFORE the weights entered the id (main @ 4d8d9eef): the id
#: every banked Kakuna read of this shape carries. It must never move.
_BANKED_ID = "1f40bc89740bcf0f"


def _wsummary(**cell):
    return {"cell": {**_WCELL, **cell}, "declared_team_cell": "symmetric",
            "team_digest_schema": "gen3_team_set_digest_v1"}


def test_banked_regime_ids_do_not_move_and_the_same_weights_re_read_comparably() -> None:
    """No weights stamped (every banked read) -> the id is the one the old code made; the legacy weights stamped (a NEW
    read of the same bytes) -> the SAME id, so a new read stays comparable with the banked ones."""
    assert AS.anchor_regime_id(_wsummary()) == _BANKED_ID
    assert AS.anchor_regime_id(_wsummary(hf_revision=AS.LEGACY_HF_REVISION,
                                         weights_sha256=AS.LEGACY_WEIGHTS["Kakuna"])) == _BANKED_ID


def test_different_weights_or_revision_are_a_different_regime() -> None:
    """Fails on revert: if the revision / digest are not regime inputs, these ids equal the banked one."""
    legacy = {"hf_revision": AS.LEGACY_HF_REVISION, "weights_sha256": AS.LEGACY_WEIGHTS["Kakuna"]}
    other_rev = AS.anchor_regime_id(_wsummary(**{**legacy, "hf_revision": "b" * 40}))
    other_sha = AS.anchor_regime_id(_wsummary(**{**legacy, "weights_sha256": "c" * 64}))
    # another agent's weights under Kakuna's name are not Kakuna's legacy weights either
    swapped = AS.anchor_regime_id(_wsummary(**{**legacy, "weights_sha256": AS.LEGACY_WEIGHTS["SmallRL"]}))
    ids = {_BANKED_ID, other_rev, other_sha, swapped}
    assert len(ids) == 4, ids
    assert AS.anchor_regime_id(_wsummary(**{**legacy, "hf_revision": "b" * 40})) == other_rev, "and it is deterministic"


def test_our_metamon_peers_weights_are_regime_inputs_too() -> None:
    ours = {"our_side": "metamon:Kakuna28"}
    base = AS.anchor_regime_id(_wsummary(**ours))
    legacy = AS.anchor_regime_id(_wsummary(**ours, our_hf_revision=AS.LEGACY_HF_REVISION,
                                           our_weights_sha256=AS.LEGACY_WEIGHTS["Kakuna28"]))
    assert legacy == base
    assert AS.anchor_regime_id(_wsummary(**ours, our_hf_revision="b" * 40,
                                         our_weights_sha256=AS.LEGACY_WEIGHTS["Kakuna28"])) != base


def test_the_legacy_table_is_the_committed_pins_while_they_are_the_legacy_revision() -> None:
    """The frozen legacy digests are exactly what ``anchors.json`` pins for the legacy revision (a repin to a new
    revision leaves LEGACY_WEIGHTS alone and is a NEW regime; this test then has nothing to say)."""
    from main.anchors.config import load_config

    meta = load_config().metamon
    if meta.hf_revision != AS.LEGACY_HF_REVISION:
        pytest.skip("anchors.json is repinned past the legacy revision: that is a new regime by design")
    assert {a: meta.pin(a).digest for a in meta.agents} == AS.LEGACY_WEIGHTS
