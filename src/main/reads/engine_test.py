"""``main.reads.engine``: the frozen LEGACY tables reproduce from git, the reconstruction reproduces the core stamps the
eval ledger RECORDED, and an engine comparison refuses a mismatch or an unknown side (REVIEW F F-M2)."""
from __future__ import annotations

import subprocess

import pytest

from main.reads import engine as EN
from utils.paths import repo_root


def _have(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(repo_root()), "cat-file", "-e", f"{commit}^{{commit}}"],
                          capture_output=True).returncode == 0


#: Core stamps the eval ledger's rows RECORDED (``compute.core_stamp``: ``src`` / ``nfiles``) at three commits, one
#: per banked engine family: D1's P lane, D3 look 1's S lane, N1-P look 1's S lane.
RECORDED_CORE = {"6c428453a59b4b64b44df64b879b56c105c49e5a": "env=7c53280566d666ae/125",
                 "1ed71c9dd5d849ad6a3e56b9fba2c103b6b1f28c": "env=9ecb4249afb76504/125",
                 "46b9e572cc0fa237f62ddd948fb5fa4d115e44a4": "env=393631b886965b0c/126"}


@pytest.mark.parametrize("commit", sorted(RECORDED_CORE))
def test_the_reconstruction_reproduces_the_core_stamps_the_ledger_recorded(commit):
    if not _have(commit):
        pytest.skip(f"commit {commit[:12]} is not in this clone's object store")
    assert EN.engine_at(commit).split(";")[1] == RECORDED_CORE[commit]


@pytest.mark.parametrize("commit", sorted(EN.LEGACY_ENGINE))
def test_every_legacy_engine_reproduces_from_its_commit(commit):
    if not _have(commit):
        pytest.skip(f"commit {commit[:12]} is not in this clone's object store")
    assert EN.engine_at(commit) == EN.LEGACY_ENGINE[commit]


def test_a_server_version_names_its_legacy_engine_by_prefix():
    assert EN.unit_engine({"server_version": "ws_frontend[in-process]@2cc864f2f+rust:cargo:src/rust_sim"}) == EN.E2
    assert EN.unit_engine({"server_version": "ws_frontend[in-process]@63f97a57+rust:cargo:src/rust_sim"}) == EN.E0
    assert EN.unit_engine({"server_version": "showdown:e0551883f"}) is None
    assert EN.unit_engine({"server_version": "ws_frontend@aaa", "engine_stamp": "port=1/1;env=1/1"}) == "port=1/1;env=1/1"
    assert EN.row_engine({"compute": {"core_stamp": "stamp=v2;kind=env;commit=x;src=7c53280566d666ae;nfiles=125;"
                                                    "nan_poison=0"}}) == "env=7c53280566d666ae/125"
    assert EN.row_engine({"compute": {}}) is None


def test_one_engine_and_the_import_rule():
    assert EN.one([EN.E1, EN.E1]) == (EN.E1, None)
    assert EN.one([None, None]) == (None, None)
    assert "span 2 engines" in str(EN.one([EN.E1, EN.E2])[1])
    assert "unknown engine" in str(EN.one([EN.E1, None])[1])
    EN.require_same("x", EN.E1, EN.E1)
    with pytest.raises(EN.EngineError, match="never crosses engines"):
        EN.require_same("x", EN.E2, EN.E1)
    with pytest.raises(EN.EngineError, match="engine unknown"):
        EN.require_same("x", None, None)


def test_a_bank_names_its_engine_by_its_rows_then_its_registration_then_the_frozen_table():
    d1 = {"path": "d1.json", "sha256": "8c4884aafcb5fa77a15b2e8be517460cf64f411a43a6ffa67bf02d1f4f2170b7"}
    assert EN.bank_engine(d1, "anchor") == EN.E1 and EN.bank_engine(d1, "s") == "env=7c53280566d666ae/125"
    assert EN.bank_engine(dict(d1, engines={"anchor": EN.E2}), "anchor") == EN.E2
    x51 = [{"server_version": "ws_frontend[in-process]@63f97a575+rust:cargo:src/rust_sim"}] * 3
    assert EN.bank_engine(d1, "anchor", x51) == EN.E0, "rows that name their engine win"
    with pytest.raises(EN.EngineError, match="span 2 engines"):
        EN.bank_engine(d1, "anchor", x51 + [{"engine_stamp": EN.E1}])
    assert EN.bank_engine({"path": "p", "sha256": "0" * 64}, "anchor") is None
