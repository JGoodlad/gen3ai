"""INTEGRATION (the Rust core; ~1 CPU-minute): on one registered decision (end2, k = 3),

* FN on F's cache and FN on a FRESH cache (F built in "another process") return the SAME worlds, and every seen-mon
  sampler the fresh FN built draws F's samples (its ``hp_any`` reset from the network's to Smogon's);
* POOL's ``strict_verdict`` agrees with the builder's own strict gate on every team F judged (pass and fail).

    PYTHONPATH=<worktree>/src python -m pytest designs/research_state/measurements/hidden_info_tier1_2026-10-13/fn_worlds_screen_test.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t1_common as T  # noqa: E402

DID = "rb_es_end2_s2001/step_10000009/aggressive/win_s3_002#p1#13"


@pytest.fixture(scope="module")
def built():
    bp = T.bank_path("end2", "registered")
    bel = T.cp_dir("end2") / "fn_belief_registered.jsonl"
    if not (bp / "decisions.jsonl.gz").exists() or not bel.exists():
        pytest.skip("the registered bank or the end2 belief rows are absent")
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import worlds as W

    import fn_worlds as FNW

    bank = load_bank(bp)
    d = next(x for x in bank.decisions if x["id"] == DID)
    b = bank.battles[bank.battle_index[d["battle"]]]
    at, log = cmd_index(b, d["side"], d["n"]), to_log(b)
    row = FNW.load_beliefs("end2")[DID]
    cache_f: dict = {}
    wf = W.WorldSampler(reveal="off", max_tries=60).sample(log, at, d["side"], 8, key=DID, screen_cache=cache_f)
    shared = FNW.fn_sampler(row).sample(log, at, d["side"], 8, key=DID, screen_cache=cache_f)
    fresh_cache: dict = {}
    fresh = FNW.fn_sampler(row).sample(log, at, d["side"], 8, key=DID, screen_cache=fresh_cache)
    root = W.prepare_root(log, at, d["side"], DID)
    return {"W": W, "root": root, "F": wf, "shared": shared, "fresh": fresh, "cache_f": cache_f,
            "fresh_cache": fresh_cache}


def test_fn_on_a_fresh_cache_equals_fn_on_F_cache(built):
    assert built["fresh"].teams == built["shared"].teams
    assert built["shared"].fn["screen"]["rebuilt_on_F_parameters"] == []      # F's samplers were read
    rb = built["fresh"].fn["screen"]["rebuilt_on_F_parameters"]
    assert rb and len(rb) == len(built["root"].kept)


def test_the_fresh_samplers_draw_F_samples(built):
    sf = built["cache_f"][("samplers", DID)]
    sn = built["fresh_cache"][("samplers", DID)]
    assert set(sn) <= set(sf)
    for k, m in sn.items():
        L = min(len(m.samples), len(sf[k].samples))
        assert L >= 8 and m.samples[:L] == sf[k].samples[:L]
        assert m.hp_any == sf[k].hp_any


def test_pool_strict_gate_equals_the_builder_gate(built):
    from agents.battle.core_replay import run_core

    import pool_worlds as PW

    W, root = built["W"], built["root"]
    v = built["cache_f"][("verdict", DID)]
    strict = [(k[: -len("##strict")], ok[0]) for k, ok in v.items() if k.endswith("##strict")]
    lenient = [k[: -len("##lenient")] for k, ok in v.items() if k.endswith("##lenient") and not ok[0]]
    teams = [t for t, _ in strict] + lenient[:20]           # screening rejects: worlds the strict gate must fail
    res = run_core([root.rec(t, f"eq{j}", W.render(root.tmpl, t)) for j, t in enumerate(teams)], trackers=True,
                   obs=True)
    mine = [PW.strict_verdict(root, r) for r in res]
    assert mine[: len(strict)] == [ok for _, ok in strict]
    assert not any(mine[len(strict):])
    assert any(not x for x in mine)                          # the comparison saw a failing world
