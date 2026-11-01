"""``main.reads.peer_divergence``: AM-DL-PEER's classifier moved into ``src``. The dlive script's selftest cases (one
synthetic capture per class, with each class's branch / null drop verdict), and the frozen dlive script's own
``classify`` run beside this module's on the same captures: equal output, case for case."""
from __future__ import annotations

import importlib.util
import json
from typing import Any, Dict, Optional

import pytest

from main.reads import peer_divergence as PD
from utils.paths import repo_path

DLIVE_SCRIPT = repo_path("designs/research_state/measurements/dlive_read_2026-10-28/scripts/peer_divergence.py")


def synthetic(turns: int, *, peer_at: Optional[int] = None, ours_at: Optional[int] = None,
              engine_at: Optional[int] = None, ours: str = "p1") -> Dict[str, Any]:
    """A capture of ``turns`` turns (the dlive script's ``_synthetic``, plus ``ours``: the side our anchor plays).
    ``peer_at`` / ``ours_at``: the turn from which that side chooses differently; ``engine_at``: the protocol differs
    from that turn with every command equal."""
    peer = "p2" if ours == "p1" else "p1"
    names = {ours: "Gen3AIAnchor1", peer: "MetaKakuna1"}
    start = (f'START {{"formatid": "gen3ou", "p1": {{"name": "{names["p1"]}", "team": "x"}}, '
             f'"p2": {{"name": "{names["p2"]}", "team": "y"}}, "seed": [1, 2, 3, 4]}}')
    cmds, chunks = [start], []
    diverged = False
    for t in range(1, turns + 1):
        for side in ("p1", "p2"):
            chunks.append([side, f'|request|{{"active": [{{}}], "rqid": {t}, "v": {int(diverged)}}}'])
        o = "move b" if ours_at is not None and t >= ours_at else "move a"
        q = "move d" if peer_at is not None and t >= peer_at else "move c"
        cmds += [f"CHOOSE {ours} {o}", f"CHOOSE {peer} {q}"]
        diverged = diverged or o != "move a" or q != "move c" or (engine_at is not None and t >= engine_at)
        for side in ("p1", "p2"):
            chunks.append([side, f"|\n|move|{o}|{q}|{int(diverged)}\n|turn|{t + 1}"])
    for side in ("p1", "p2"):
        chunks.append([side, "|win|Gen3AIAnchor1"])
    return {"tag": "battle-gen3ou-1", "seed": [1, 2, 3, 4], "commands": cmds, "chunks": chunks}


#: the dlive script's selftest: (replay, class, branch drop, null drop), raw = 20 turns, branch turn 10
CASES = [
    (dict(), "identical", False, False),
    (dict(peer_at=3), "peer_first", True, True),
    (dict(peer_at=12), "peer_first_late", False, True),
    (dict(peer_at=10), "peer_first_late", False, True),
    (dict(ours_at=10), "ours_first", False, False),
    (dict(ours_at=3, peer_at=3), "ours_first", False, False),
    (dict(engine_at=4), "unattributable", False, False),
]


def _dlive():
    spec = importlib.util.spec_from_file_location("dlive_peer_divergence_frozen", DLIVE_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("kw, cls, b_drop, n_drop", CASES)
def test_the_selftest_cases(kw, cls, b_drop, n_drop):
    got = PD.classify(synthetic(20), synthetic(20, **kw), 10)["class"]
    assert got == cls
    assert PD.is_peer_drop("branch", got) is b_drop and PD.is_peer_drop("null", got) is n_drop


def test_the_peer_side_is_named_from_the_start_document_either_way_round():
    got = PD.classify(synthetic(20, ours="p2"), synthetic(20, ours="p2", peer_at=3), 10)
    assert got["class"] == "peer_first" and got["sides"] == {"ours": "p2", "peer": "p1"}
    assert got["peer"]["divergence_turn"] == 3 and got["ours"]["diverged"] is False


def test_an_unreadable_capture_or_another_deal_is_unattributable_never_a_drop():
    assert PD.classify({"commands": []}, synthetic(20), 10)["class"] == "unattributable"
    other = synthetic(20)
    other["commands"][0] = other["commands"][0].replace('"team": "y"', '"team": "z"')
    assert PD.classify(synthetic(20), other, 10) == {"class": "unattributable", "why": "the START documents differ"}
    with pytest.raises(ValueError):
        PD.is_peer_drop("searched", "peer_first")


def test_a_result_line_is_never_compared():
    a, b = synthetic(20), synthetic(20)
    b["chunks"] = [[s, c.replace("|win|Gen3AIAnchor1", "|win|MetaKakuna1")] for s, c in b["chunks"]]
    assert PD.classify(a, b, 10)["class"] == "identical"


EXTRA = [dict(ours="p2", peer_at=3), dict(ours="p2", ours_at=5), dict(peer_at=1), dict(ours_at=1),
         dict(engine_at=1), dict(peer_at=20), dict(ours_at=15, peer_at=4)]


@pytest.mark.parametrize("kw", [c[0] for c in CASES] + EXTRA)
@pytest.mark.parametrize("turn", [1, 4, 10, 1000])
def test_equal_to_the_frozen_dlive_script_on_the_same_captures(kw, turn):
    old = _dlive()
    side = kw.get("ours", "p1")
    raw, rep = synthetic(20, ours=side), synthetic(20, **kw)
    if side == "p1":           # the builder IS the script's, case for case
        assert old._synthetic(20, **{k: v for k, v in kw.items() if k != "ours"}) == rep
    assert json.dumps(PD.classify(raw, rep, turn), sort_keys=True) == json.dumps(old.classify(raw, rep, turn),
                                                                                 sort_keys=True)
    for kind in ("branch", "null"):
        c = PD.classify(raw, rep, turn)["class"]
        assert PD.is_peer_drop(kind, c) == old.is_peer_drop(kind, c)


def test_the_prefix_verb_classifies_each_mismatch_of_a_state_outcome_blind(tmp_path, monkeypatch, capsys):
    """M5: ``python -m main.reads.peer_divergence prefix`` on a synthetic searched state: each pair whose captures
    differ before the search's first turn, classified; a pair that matches is not listed; no result is printed."""
    import time

    from main.reads import runner as RN
    from main.reads import search_read_test as SR
    from utils.hashing import sha256_file

    z = tmp_path / "ckpt" / "checkpoint_280000170_steps.zip"
    z.parent.mkdir()
    (z.parent / "model_config.json").write_text("{}")
    z.write_bytes(b"pstar")
    spec_p = tmp_path / "read_spec.json"
    spec_p.write_text(json.dumps(SR._doc(str(z), sha256_file(z))))
    state = tmp_path / "state"
    RN.plan(spec_p, state, code=SR.CODE, step_of=lambda p: 280000170)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: SR.CODE)
    assert RN.run(state, "anchor", time.time() + 3600, run_unit=SR._fake_unit([], diverge=(("forum72", 0), 9)),
                  sleep=lambda s: None) == 0
    # the home72 pair matched; make it a Kakuna-first divergence on an identical view (ours equal) instead
    for lab, cap in (("pstar", synthetic(20)), ("pstar_s", synthetic(20, peer_at=3))):
        p = state / "anchor" / "su" / lab / "cells" / f"{lab}__home72__00__g00" / "capture" / "battle-gen3ou-1.json"
        p.write_text(json.dumps(cap))
    capsys.readouterr()
    assert PD.main(["prefix", "--state", str(state), "--group", "su", "--spec", str(state / "spec.json")]) == 0
    out = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    rows = {r["deal"]: r for r in out[:-1]}
    assert set(rows) == {"home72:0:0", "forum72:0:0"} and out[-1] == {"mismatches": 2, "peer_drops": 1}
    assert rows["home72:0:0"]["class"] == "peer_first" and rows["home72:0:0"]["peer_drop"] is True
    assert rows["home72:0:0"]["peer_divergence_turn"] == 3 and rows["home72:0:0"]["ours_diverged"] is False
    assert rows["forum72:0:0"]["class"] == "unattributable" and rows["forum72:0:0"]["peer_drop"] is False
    assert "win" not in json.dumps(out) and "|move|" not in json.dumps(out)
    assert PD.main(["prefix", "--state", str(state), "--group", "nope"]) == 2
