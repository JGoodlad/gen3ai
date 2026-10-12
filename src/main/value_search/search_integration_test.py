"""``search_decision`` and the world sampler on real banked turns (the core, a stub forward)."""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from main.policy_spectrum import bank as B
from main.policy_spectrum.bank_test import BANK_V1
from main.policy_spectrum.truth import cmd_index, select_subset, to_log
from main.value_search import worlds as W
from main.value_search.playout import LeafEngine
from main.value_search.search import SearchConfig, search_decision
from utils.paths import src_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def lib():
    from utils.rust_env import ffi

    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed")
    crate = src_path("rust_env")
    r = subprocess.run([cargo, "build", "--lib", "--profile", "selfcheck", "--features", "emission-selfcheck",
                        "--manifest-path", str(crate / "Cargo.toml")],
                       env=dict(os.environ, CARGO_TARGET_DIR=str(crate / "target")),
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-4000:]
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


def _stub(rows: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Logits from a fixed projection of the row (greedy = a deterministic legal action), V a fixed
    function of the row."""
    rng = np.random.default_rng(7)
    proj = rng.standard_normal((rows.shape[1], 11)).astype(np.float32) * 0.05
    out = np.zeros((len(rows), 12), dtype=np.float32)
    out[:, :11] = rows @ proj
    out[:, 11] = 1.0 / (1.0 + np.exp(-rows.astype(np.float64).mean(axis=1) * 10.0))
    return out


@pytest.fixture(scope="module")
def turns():
    """Three free turns (turn ≥ 4, ≥ 3 legal) of the committed Lane S bank, with their root rows."""
    from main.policy_spectrum.replay import replay

    bank = B.load_bank(BANK_V1)
    dd = {d["id"]: d for d in bank.decisions}
    ids = [i for i in select_subset(bank) if dd[i]["kind"] == "free" and dd[i]["turn"] >= 4
           and dd[i]["n_legal"] >= 3][:3]
    assert len(ids) == 3
    out = []
    for i in ids:
        d = dd[i]
        b = bank.battles[bank.battle_index[d["battle"]]]
        rep = replay([b.recorded()])[0]
        dec = rep.decisions[0 if d["side"] == "p1" else 1][d["n"]]
        out.append({"id": i, "log": to_log(b), "at": cmd_index(b, d["side"], d["n"]), "side": d["side"],
                    "row": np.array(dec.row, dtype=np.float32), "mask": np.asarray(dec.mask, dtype=np.uint8)})
    return out


def test_oracle_search_is_legal_deterministic_and_within_budget(lib, turns):
    eng = LeafEngine(_stub, n_cores=4, lib=lib)
    try:
        for t in turns:
            for cfg in (SearchConfig(per_action=2, depth=2, n_worlds=4, oracle=True),
                        SearchConfig(per_action=2, depth=None, n_worlds=4, oracle=True, allocation="uniform")):
                a = search_decision(eng, t["log"], t["at"], t["side"], t["row"], t["mask"], cfg, key=t["id"])
                b = search_decision(eng, t["log"], t["at"], t["side"], t["row"], t["mask"], cfg, key=t["id"])
                legal = set(np.flatnonzero(t["mask"]).tolist())
                assert a.action in legal
                assert a.policy_target.sum() == pytest.approx(1.0)
                assert all(a.policy_target[x] == 0.0 for x in range(11) if x not in legal)
                assert a.spent <= 2 * len(legal) + 4       # max(1, ⌊B / (R·|S|)⌋) may round a round up
                assert all(0.0 <= v <= 1.0 for v in a.q.values())
                assert (a.action, a.q, a.visits) == (b.action, b.q, b.visits)       # seeded end to end
    finally:
        eng.close()


def test_worlds_pass_the_observation_gate_and_a_wrong_world_does_not(lib, turns):
    from agents.battle.core_replay import RecordedBattle, run_core

    t = turns[0]
    sampler = W.WorldSampler()
    ws = sampler.sample(t["log"], t["at"], t["side"], 4, t["id"])
    true_opp = t["log"]["teams"][1 if t["side"] == "p1" else 0]
    assert len(ws.logs) == 4 and any(team != true_opp for team in ws.teams)   # it really sampled
    viewer = 0 if t["side"] == "p1" else 1
    n = sum(1 for s, _ in W.log_commands(t["log"], t["at"]) if s == t["side"])

    def rows_of(log):
        res = run_core([RecordedBattle(label="x", format_id=log["format_id"], seed=log["seed"],
                                       p1={"name": log["names"][0], "team": log["teams"][0]},
                                       p2={"name": log["names"][1], "team": log["teams"][1]},
                                       commands=W.log_commands(log, t["at"]))], trackers=True, obs=True)[0]
        return (W.reveals(res, viewer, n)[1], W.protocol_lines(res, viewer, n)) if res["ok"] else None

    ref = rows_of(t["log"])
    for log in ws.logs:
        assert rows_of(log) == ref       # the gate, re-checked outside: rows AND the searched side's protocol bytes
    assert all(not (W.team_problems(team) - W.team_problems(true_opp)) for team in ws.teams)   # gen3ou-legal
    # teeth: the opponent's SEEN lead with another species is not a consistent world
    oi = 1 - viewer
    ents = t["log"]["teams"][oi].split("]")
    lead = W.parse_set(ents[0])
    other = "Blissey||leftovers|naturalcure|softboiled,seismictoss,toxic,aromatherapy|Bold|252,,252,,4,||||||"
    if W.toid(lead[0]) == "blissey":
        other = other.replace("Blissey", "Chansey")
    bad = dict(t["log"], teams=[*t["log"]["teams"]])
    bad["teams"][oi] = "]".join([other] + ents[1:])
    assert rows_of(bad) != ref


def test_honest_search_runs_on_sampled_worlds(lib, turns):
    eng = LeafEngine(_stub, n_cores=4, lib=lib)
    try:
        t = turns[1]
        r = search_decision(eng, t["log"], t["at"], t["side"], t["row"], t["mask"],
                            SearchConfig(per_action=2, depth=4, n_worlds=4), key=t["id"], sampler=W.WorldSampler())
        assert r.action in set(np.flatnonzero(t["mask"]).tolist())
        assert len(r.world_tries) == 4 and min(r.world_tries) >= 1
        assert 0.0 <= r.value_target <= 1.0
    finally:
        eng.close()
