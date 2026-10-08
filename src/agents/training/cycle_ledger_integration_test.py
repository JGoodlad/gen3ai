"""The in-loop eval cycle on the REAL Rust eval core writes ledger rows that ARE its games (eval unit U2).

One seeded in-loop cycle (``eval_launch.launch_rust_eval_cycle``, the callbacks' own call) with a ``CycleLedger``,
against the same plan replayed with the executor's FULL game log and against the same cycle with NO ledger:

* every row validates, reads through a declared ``ReaderDecl``, and its OUTCOME DIGEST, near-tie list and W / L / D
  are the full log's for its opponent (the row is the games, game for game);
* the ledger changes no game: the cycle's published shard results are identical with and without it.

Seeded perturbed fresh policies (``rust_eval.offline.build_models``), CPU eager, the emission self-check build —
the harness of ``designs/research_state/measurements/eval_ledger_u2_2026-10-04/digest_proof.py`` (loaded by path),
whose before/after run is the cross-commit half of the proof. FAILS on a revert of the producer (no rows)."""
from __future__ import annotations

import importlib.util

import pytest

from utils.paths import repo_path

# `slow`: ~40-55 s of real games (perturbed fresh policies play long), over the routine tier's 30 s budget.
pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]

PROOF = repo_path("designs", "research_state", "measurements", "eval_ledger_u2_2026-10-04", "digest_proof.py")


@pytest.fixture(scope="module")
def proof():
    spec = importlib.util.spec_from_file_location("u2_digest_proof", PROOF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def host(proof, tmp_path_factory):
    import torch

    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    threads = torch.get_num_threads()
    h = proof.build_host(tmp_path_factory.mktemp("u2host"))
    yield h
    h["ev"].close()
    torch.set_num_threads(threads)


def test_the_cycles_ledger_rows_are_its_games_and_the_ledger_changes_no_game(proof, host, tmp_path):
    from agents.training import eval_ledger as L
    from agents.training.cycle_ledger import PROTOCOL
    from agents.training.eval_ledger import audit as AU

    led_root = tmp_path / "ledger"
    plan = {"games": 2, "bots": 1, "sentinels": 1, "fixed": False}      # 1 mirrored pair per opponent: CPU-cheap
    on = proof.play_cycle(host, tmp_path / "on", mirrored=True, sentinel_greedy=False, ledger_root=led_root, plan=plan)
    off = proof.play_cycle(host, tmp_path / "off", mirrored=True, sentinel_greedy=False, ledger_root=None,
                           full_log=False, plan=plan)
    assert on["merged"] == off["merged"], "the ledger changed what the cycle published"
    rows = on["ledger"]
    assert sorted(rows) == sorted(on["full_log"]), (sorted(rows), sorted(on["full_log"]))
    for key, row in rows.items():
        fl = on["full_log"][key]
        assert row["digest"] == fl["digest"] and row["near_tie_games"] == fl["near_tie_games"], key
        assert row["digest_all"] == fl["digest_all"], key                # every game, no near-tie exclusion
        wld = tuple(sum(1 for _g, r, _t in fl["outcomes"] if r == x) for x in "WLD")
        assert (row["counts"]["w"], row["counts"]["l"], row["counts"]["d"]) == wld, (key, row["counts"], wld)
        assert row["pairs"] == on["merged"][key]["pairs"] and row["mirrored"] is True
    decl = L.ReaderDecl(name="cycle_ledger_integration", purposes=frozenset({"cycle"}),
                        regime=L.RegimeFilter(protocol=PROTOCOL, play="greedy", mirrored=True, seat_rule="fixed_p1"),
                        requests="any", selection="include", flags_ok=frozenset(), inference="conditional")
    got = [r for rd in L.read_by_regime(decl, root=led_root).values() for r in rd.rows]
    assert len(got) == len(rows) and all(L.validate_row(r) == [] for r in got)
    assert all("d" in r["counts"] for r in got)              # draws are a column, never folded (F-ED-8)
    rep = AU.audit(led_root)
    assert rep.ok, rep.problems
