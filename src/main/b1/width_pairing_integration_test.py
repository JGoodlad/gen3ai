"""BOUNDARY-1's primary meter pairs a 128-wide network with a 256-wide one in ONE head-to-head engine (PREREG §5): a
look of 24 cells holds cross-width cells (W or WB against C or Br) beside same-width cells of BOTH widths (Br against
C, WB against W). This plays that shape on the REAL engine (CPU, tiny): two checkpoints built from the control arm's own
resolved config and two from the width arm's (``main.b1.arms``), one ``play_cells`` call over the four kinds of cell.

It pins what no other test does: that the two-architecture engine (``main.h2h.arch``; its own tests cross a network of
another DEPTH) serves two trunk WIDTHS, each group with a player and an opponent slot. Seeded perturbed-fresh
checkpoints: the numbers are not the point, the pairing is. The GPU inference path (graph capture, bf16) is NOT covered
here; it is PREREG §13's item for the leased window."""
from __future__ import annotations

import argparse
import json

import pytest

from agents.training import eval_ledger as L
from main.b1 import arms as A
from main.h2h import many as MANY
from main.h2h import play as PL

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]

PIN = "0" * 40
#: one intra-op torch thread, as the head-to-head engine's own CPU tests play (designs/ops/testing.md)
COMPUTE = PL.Compute(device="cpu", backend="eager", n_envs=8, threads=2, torch_threads=1, front="ffi",
                     profile="selfcheck")
ROWS = L.ReaderDecl(name="b1_width_pairing_test", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                    selection="include", flags_ok=frozenset(), inference="conditional")
WIDTH = {"c": 128, "w": 256}


@pytest.fixture(scope="module")
def built():
    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    return True


def _arm_checkpoints(dst, arm):
    """Two perturbed-fresh checkpoints of ``arm``'s registered architecture, beside their ``model_config.json``."""
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.snapshot import recorded_model_version
    from agents.training.rust_eval import offline as PAR
    from main.fresh_checkpoint import build_fresh_model

    dst.mkdir(parents=True)
    out = []
    with PAR.declared_torch_state(1):
        for i in range(2):
            ns = argparse.Namespace(**A.resolved_namespace(A.build_argv(arm, A.SEEDS[arm][i], PIN)))
            model, _, _ = build_fresh_model(i, args=ns)
            perturb_(model.policy, seed=5100 + 10 * i + len(arm), scale=PERTURB_SCALE)
            path = dst / f"snapshot_{(i + 1) * 1000:012d}.zip"
            model.save(str(path))
            out.append(str(path))
    (dst / "model_config.json").write_text(recorded_model_version(model).to_json())
    return out


@pytest.fixture(scope="module")
def played(built, tmp_path_factory):
    base = tmp_path_factory.mktemp("b1_width_pairing")
    refs = {}
    for arm in ("c", "w"):
        z1, z2 = _arm_checkpoints(base / f"run_b1_{arm}", arm)
        refs[arm] = (PL.resolve_player(z1), PL.resolve_player(z2))
    (c1, c2), (w1, w2) = refs["c"], refs["w"]
    # the four kinds of cell a look holds: wide measured, narrow measured, and the same-width cell of each width
    look = [(w1, c1), (c2, w2), (c1, c2), (w1, w2)]
    out = base / "ledger"
    summary = MANY.play_cells(str(out), look, pairs=2, batch_pairs=2, schedule_seed=5, run_label="b1-width-pairing",
                              compute=COMPUTE, emit=lambda _m: None)
    return {"refs": refs, "look": look, "summary": summary, "rows": list(L.read(ROWS, root=out).rows), "base": base}


def test_the_two_arms_checkpoints_record_the_two_registered_widths(played):
    for arm, width in WIDTH.items():
        cfg = json.loads((played["base"] / f"run_b1_{arm}" / "model_config.json").read_text())
        found = _find(cfg, "d_model")
        assert found == {width}, f"arm {arm}: model_config.json records d_model {found}, registered {width}"


def _find(node, key):
    """Every value stored under ``key`` anywhere in a JSON tree."""
    out = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and not isinstance(v, (dict, list)):
                out.add(v)
            out |= _find(v, key)
    elif isinstance(node, list):
        for v in node:
            out |= _find(v, key)
    return out


def test_one_engine_serves_both_widths_each_as_player_and_as_opponent(played):
    decl = played["summary"]["engine"]["decl"]
    (c1, _c2), (w1, _w2) = played["refs"]["c"], played["refs"]["w"]
    assert len(decl["groups"]) == 2, decl
    by_source = {g["source"]: g for g in decl["groups"]}
    assert set(by_source) == {c1.id, w1.id}, "one group an architecture, named by the first checkpoint that needs it"
    assert all(sorted(g["roles"]) == ["opponent", "player"] for g in decl["groups"]), decl
    assert decl["groups"][0]["fingerprint"] != decl["groups"][1]["fingerprint"]
    # the four combinations of (player's group, opponent's group): both cross directions and both same-width cells
    assert sorted(map(tuple, decl["combos"])) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_every_cell_of_the_look_writes_one_valid_row_of_its_two_pairs(played):
    rows = {(r["player"]["sha256"], r["opponent"]["sha256"]): r for r in played["rows"]}
    assert len(played["rows"]) == len(played["look"]) == len(rows) == 4
    for player, opponent in played["look"]:
        r = rows[(player.sha256, opponent.sha256)]
        assert L.validate_row(r) == []
        assert r["regime"]["protocol"] == PL.PROTOCOL
        assert r["pairs"]["n_pairs"] == 2 and sum(r["counts"][k] for k in ("w", "l", "d")) == 4
        assert r["compute"]["trainee_decisions"] > 0 and r["compute"]["p2_policy_decisions"] > 0
