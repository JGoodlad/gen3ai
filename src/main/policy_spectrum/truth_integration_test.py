"""Gate ④ through Lane I's ``play_out`` on real banked turns (integration)."""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from main.policy_spectrum import bank as B
from main.policy_spectrum import truth as T
from main.policy_spectrum.bank_test import BANK_V1
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


def _first_legal(rows, masks, who):
    return np.array([int(np.flatnonzero(m)[0]) for m in masks], dtype=np.int32)


def test_branch_turns_are_crn_deterministic_and_token_checked(core_factory, tmp_path):
    bank = B.load_bank(BANK_V1)
    ids = T.select_subset(bank)[:3]
    out = tmp_path / "rows.jsonl"
    assert T.run(bank, ids, _first_legal, "first-legal (test)", out, s=2, log=lambda m: None,
                 core_factory=core_factory) == 3
    rows = T.load_rows(out)
    assert all(r["ok"] for r in rows), [r.get("error") for r in rows]
    dd = {d["id"]: d for d in bank.decisions}
    for r in rows:
        assert set(r["outcomes"]) == set(dd[r["id"]]["tokens"])       # every legal action branched
        assert all(len(v) == 2 and set(v) <= {-1.0, 0.0, 1.0} for v in r["outcomes"].values())
    # resumable: a second run writes nothing; a re-branch is identical (CRN, deterministic policy)
    assert T.run(bank, ids, _first_legal, "first-legal (test)", out, s=2, log=lambda m: None,
                 core_factory=core_factory) == 0
    with core_factory() as core:
        again = T.branch_turn(bank, ids[0], _first_legal, T.turn_seeds(ids[0], 2), "first-legal (test)",
                              core=core)
    assert again["outcomes"] == rows[0]["outcomes"] and again["ends"] == rows[0]["ends"]
