"""The X4 pre-read's one-ply branching on real banked turns (integration: Lane I's playout core)."""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from main.policy_spectrum import bank as B
from main.policy_spectrum import qhat as Q
from main.policy_spectrum.bank_test import BANK_V1
from main.policy_spectrum.truth import select_subset
from utils.paths import src_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def core_factory():
    from utils.rust_env import ffi
    from utils.rust_env.successors import SearchCore

    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed")
    crate = src_path("rust_env")
    r = subprocess.run([cargo, "build", "--lib", "--profile", "selfcheck", "--features", "emission-selfcheck",
                        "--manifest-path", str(crate / "Cargo.toml")],
                       env=dict(os.environ, CARGO_TARGET_DIR=str(crate / "target")),
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-4000:]
    lib = ffi.load(ffi.default_path("selfcheck"), nan_poison=True)
    return lambda: SearchCore(lib=lib)


def _stub(rows: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Logits 0 (greedy = the lowest legal index) and a V that is a fixed function of the row."""
    out = np.zeros((len(rows), 12), dtype=np.float32)
    out[:, 11] = 1.0 / (1.0 + np.exp(-rows.astype(np.float64).mean(axis=1) * 10.0))
    return out


@pytest.fixture(scope="module")
def bank():
    return B.load_bank(BANK_V1)


def test_every_action_x_seed_is_scored_deterministically_and_R_closes_the_root(bank, core_factory):
    ids = select_subset(bank)[:4]
    dd = {d["id"]: d for d in bank.decisions}
    with core_factory() as core:
        first = [Q.qhat_turn(bank, i, _stub, core, 2, "stub") for i in ids]
        again = [Q.qhat_turn(bank, i, _stub, core, 2, "stub") for i in ids]
    assert first == again                                            # CRN + a deterministic scorer
    saw_open = False
    for i, rows in zip(ids, first):
        by = {r["variant"]: r for r in rows}
        assert by["R"]["ok"], by["R"].get("error")
        for r in rows:
            assert set(r["actions"]) == set(dd[i]["tokens"])        # every legal action branched
            for a in r["actions"].values():
                assert len(a["q"]) == 2 and len(a["src"]) == 2
                assert all(-1.0 <= x <= 1.0 for x in a["q"])
        if by["R"]["other_open"]:
            saw_open = True
            assert "M" in by and by["M"]["ok"]
            assert by["R"]["at_eff"] > by["R"]["at"]                # the recorded answer moved in front
            # M answers the open root with the checkpoint's greedy choice on EVERY branch; R holds the
            # recorded answer in the prefix, so it answers only later (replacement) decisions
            m_opp = [int(ch) for a in by["M"]["actions"].values() for ch in a["opp"]]
            r_opp = [int(ch) for a in by["R"]["actions"].values() for ch in a["opp"]]
            assert min(m_opp) >= 1 and sum(m_opp) > sum(r_opp)
    assert saw_open, "no turn with an open opponent root in the slice — the R / M split went untested"


def test_the_played_action_under_the_recorded_dice_reproduces_the_next_row(bank, core_factory):
    """Teeth: seed None + our played action + the opponent's recorded answers = the replay's next
    decision, BYTE-EQUAL (so the captured row is the trainee's own observation of the successor)."""
    from main.policy_spectrum.replay import replay

    ids = select_subset(bank)[:6]
    battles = sorted({next(d for d in bank.decisions if d["id"] == i)["battle"] for i in ids})
    reps = dict(zip(battles, replay([bank.battles[bank.battle_index[b]].recorded() for b in battles])))
    captured = 0
    with core_factory() as core:
        for i in ids:
            b = next(d for d in bank.decisions if d["id"] == i)["battle"]
            r = Q.sanity_turn(bank, i, _stub, core, reps[b])
            if r["status"] == "captured":
                captured += 1
                assert r["byte_equal"], r
                assert r["v_branch"] == r["v_recorded"]
    assert captured >= 4
