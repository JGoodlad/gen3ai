"""INTEGRATION (the Rust core; ~1 CPU-minute): POOL's builder runs the world consistency guard on the teams it returns
(``world_guard.check_teams``, 2026-10-15; the builder bypasses ``WorldSampler.sample``, where the guard otherwise runs).

* a clean ``pool_sample`` on one registered decision (end2, k = 3) returns worlds, and the guard finds every one consistent;
* a PLANTED defect (the builder hands back a team whose seen mons sit at the wrong slots, and the gate and the playout
  check are waved through so the draw reaches the guard) makes ``pool_sample`` RAISE ``WorldConsistencyError``, which is
  not a ``WorldError`` (a defect is never an exclusion);
* with the guard call removed the planted defect is returned silently (the test fails on revert: it asserts the
  precondition that the defect passes everything but the guard).

    PYTHONPATH=<worktree>/src python -m pytest designs/research_state/measurements/hidden_info_tier1_2026-10-13/pool_worlds_guard_test.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pool_worlds as PW  # noqa: E402
import t1_common as T  # noqa: E402

pytestmark = [pytest.mark.sim, pytest.mark.integration]
_REAL_BUILD = PW.build_world       # captured once: a second monkeypatch must wrap the REAL builder, not the first plant

DID = "rb_es_end2_s2001/step_10000009/aggressive/win_s3_002#p1#13"
MAX_GATED = 40                 # a cost bound for the test (the production draw gates every candidate)


@pytest.fixture(scope="module")
def decision():
    bp = T.bank_path("end2", "registered")
    if not (bp / "decisions.jsonl.gz").exists():
        pytest.skip("the registered bank is absent")
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log

    bank = load_bank(bp)
    d = next(x for x in bank.decisions if x["id"] == DID)
    b = bank.battles[bank.battle_index[d["battle"]]]
    return to_log(b), cmd_index(b, d["side"], d["n"]), d["side"]


def _sample(decision, monkeypatch=None, plant=False):
    import pool_teams as PT
    from main.value_search import worlds as W

    log, at, side = decision
    ws = W.WorldSampler(reveal="off")
    if plant:
        assert monkeypatch is not None
        real_build = _REAL_BUILD

        def swapped(root, team, rng, pad):
            tw, why = real_build(root, team, rng, pad)
            if tw is None:
                return tw, why
            ents = tw.split("]")
            seen = [i for i, s in enumerate(root.sids) if s in root.kept]
            ents[seen[0]], ents[seen[1]] = ents[seen[1]], ents[seen[0]]       # two seen mons out of their true slots
            return "]".join(ents), ""

        monkeypatch.setattr(PW, "build_world", swapped)
        monkeypatch.setattr(PW, "strict_gate", lambda root, teams, key, verdicts=None: [True] * len(teams))
        monkeypatch.setattr(ws, "playout_refusal", lambda *a, **k: None)
    return PW.pool_sample(log, at, side, 3, key=DID, pool=PT.load_pool(), ws=ws, max_gated=MAX_GATED)


def test_a_clean_pool_draw_passes_the_guard(decision):
    pw = _sample(decision)
    assert len(pw.teams) == 3 and pw.meta["passed"] >= 1


def test_a_planted_defect_raises_and_is_not_an_exclusion(decision, monkeypatch):
    from main.value_search import world_guard as WG
    from main.value_search import worlds as W

    assert issubclass(WG.WorldConsistencyError, Exception) and not issubclass(WG.WorldConsistencyError, W.WorldError)
    with pytest.raises(WG.WorldConsistencyError, match="true slot|fact record|engine"):
        _sample(decision, monkeypatch, plant=True)
    # the precondition: the same planted defect is returned silently without the guard (so the raise above IS the guard)
    monkeypatch.setattr(WG, "check_teams", lambda *a, **k: None)
    pw = _sample(decision, monkeypatch, plant=True)
    assert len(pw.teams) == 3
