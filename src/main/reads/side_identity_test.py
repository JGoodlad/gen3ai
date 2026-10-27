"""``main.reads.side_identity`` and ``peer_scripts/peer_code``: the Metamon peer's code and our policy's load mode enter
the regime id only when they are not what the banked reads played; the frozen LEGACY tables reproduce from git; a
comparison refuses a difference or an unknown side (TECH_DEBT P1 (b), 2026-10-27)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from main.anchors.peer_scripts import peer_code as PC
from main.reads import assemble as AS
from main.reads import engine as EN
from main.reads import side_identity as SI
from main.reads.assemble_test import _BANKED_ID, _wsummary
from utils.paths import repo_root, src_path

X51 = "ws_frontend[in-process]@63f97a575+rust:cargo:src/rust_sim"
N1P = "ws_frontend[in-process]@de15dfc7b+rust:cargo:src/rust_sim"
NEW = "f" * 16


def _have(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(repo_root()), "cat-file", "-e", f"{commit}^{{commit}}"],
                          capture_output=True).returncode == 0


# ------------------------------------------------------------------------------------------------ the id
def test_a_banked_id_is_unchanged_and_the_legacy_stamps_keep_it():
    """No stamp (every banked summary), or the legacy peer code and the "train" mode stamped: the banked id."""
    assert AS.anchor_regime_id(_wsummary()) == _BANKED_ID
    for legacy in SI.LEGACY_PEER_CODE:
        assert AS.anchor_regime_id(_wsummary(peer_code=legacy, policy_mode="train")) == _BANKED_ID
    assert AS.anchor_regime_id(_wsummary(policy_mode="", peer_code=None)) == _BANKED_ID


def test_another_peer_code_or_load_mode_is_another_regime():
    """Fails on revert of ``SI.regime_identity`` in ``anchor_regime_id``: these ids would all equal the banked one."""
    ids = {AS.anchor_regime_id(_wsummary(peer_code=NEW)),
           AS.anchor_regime_id(_wsummary(peer_code="e" * 16)),
           AS.anchor_regime_id(_wsummary(our_side="metamon:Kakuna28", our_peer_code=NEW)),
           AS.anchor_regime_id(_wsummary(policy_mode="eval")),
           AS.anchor_regime_id(_wsummary(policy_mode="mixed")),
           AS.anchor_regime_id(_wsummary(peer_code=NEW, policy_mode="eval"))}
    assert len(ids) == 6 and _BANKED_ID not in ids


# ------------------------------------------------------------------------------------------------ per unit
def test_a_unit_names_its_sides_stamped_or_from_its_banked_commit():
    cell = dict(_wsummary()["cell"])
    assert SI.unit_sides(dict(cell, server_version=X51)) == {"peer_code": "their=legacy", "policy_mode": "train"}
    assert SI.unit_sides(dict(cell, server_version=N1P)) == {"peer_code": "their=legacy", "policy_mode": "eval"}
    assert SI.unit_sides(dict(cell, server_version="ws_frontend@" + "e" * 9 + "+rust")) == \
        {"peer_code": None, "policy_mode": None}, "a commit outside the tables is unknown"
    stamped = dict(cell, peer_code=NEW, policy_mode="eval")
    assert SI.unit_sides(stamped) == {"peer_code": f"their={NEW}", "policy_mode": "eval"}
    legacy = next(iter(SI.LEGACY_PEER_CODE))
    assert SI.unit_peer_code(dict(cell, peer_code=legacy)) == "their=legacy"
    assert SI.unit_peer_code(dict(cell, opponent_kind="foulplay")) == "", "no Metamon side: nothing to name"
    both = dict(cell, our_side="metamon:Kakuna28", peer_code=NEW, our_peer_code=legacy)
    assert SI.unit_peer_code(both) == f"their={NEW};our=legacy" and SI.unit_policy_mode(both) == ""
    assert SI.unit_policy_mode(dict(cell, our_side="bot:random")) == ""
    row = AS.unit_row(_U(), dict(_wsummary(peer_code=NEW, policy_mode="eval"), status="OK", n=20, wins=10, losses=10))
    assert (row["peer_code"], row["policy_mode"]) == (f"their={NEW}", "eval")


class _U:
    name, label, teamset, unit_seed, games, model_sha256 = "u", "a", "away", 0, 20, None


def test_a_group_spanning_two_sides_is_inconclusive():
    rows = [{"counted": True, "peer_code": "their=legacy", "policy_mode": "train"},
            {"counted": True, "peer_code": "their=legacy", "policy_mode": "eval"},
            {"counted": False, "peer_code": None, "policy_mode": None}]
    sides, why = SI.group_sides(rows)
    assert sides == {"peer_code": "their=legacy", "policy_mode": None} and len(why) == 1 and "policy mode" in why[0]
    _, why = SI.group_sides([dict(rows[0]), dict(rows[0], peer_code=None)])
    assert why and "unknown" in why[0]


def test_a_comparison_refuses_a_difference_or_an_unknown_side():
    a = {"peer_code": "their=legacy", "policy_mode": "train"}
    SI.require_same("x", a, dict(a))
    for there, why in ((dict(a, policy_mode="eval"), "policy mode"), (dict(a, peer_code=f"their={NEW}"), "peer code"),
                       (dict(a, policy_mode=None), "unknown")):
        with pytest.raises(SI.SideError, match=why):
            SI.require_same("x", a, there)


def test_a_bank_is_read_by_its_registration_its_rows_or_the_legacy_table():
    legacy = {"peer_code": "their=legacy", "policy_mode": "train"}
    assert SI.bank_sides({"sha256": next(iter(SI.LEGACY_BANK_SIDE))}) == legacy
    assert SI.bank_sides({"sha256": "0" * 64}) == {"peer_code": None, "policy_mode": None}
    assert SI.bank_sides({"sha256": "0" * 64}, [{"server_version": X51}, {"server_version": X51}]) == legacy
    assert SI.bank_sides({"sides": {"peer_code": "their=legacy", "policy_mode": "eval"}})["policy_mode"] == "eval"
    with pytest.raises(SI.SideError, match="span"):
        SI.bank_sides({"path": "b"}, [{"server_version": X51}, {"server_version": N1P}])


# ------------------------------------------------------------------------------------------------ legacy tables
@pytest.mark.parametrize("commit", sorted(EN.LEGACY_ENGINE))
def test_every_banked_commit_reconstructs_to_a_legacy_peer_code_and_its_mode(commit):
    """The frozen tables reproduce from git: each banked commit's peer scripts (with the legacy checkout and env) give
    a :data:`SI.LEGACY_PEER_CODE` digest, and :data:`SI.LEGACY_COMMIT_MODE` is the ancestry rule's mode."""
    if not _have(commit):
        pytest.skip(f"commit {commit[:12]} is not in this clone's object store")
    assert SI.peer_code_at(commit) in SI.LEGACY_PEER_CODE
    assert SI.policy_mode_at(commit) == SI.LEGACY_COMMIT_MODE[commit]


def test_the_legacy_peer_codes_are_exactly_the_banked_commits_and_this_tree_is_not_one():
    if not all(_have(c) for c in EN.LEGACY_ENGINE):
        pytest.skip("a banked commit is not in this clone's object store")
    assert {SI.peer_code_at(c) for c in EN.LEGACY_ENGINE} == set(SI.LEGACY_PEER_CODE)
    here = PC.hashes_from(lambda where, name: _read_here(where, name))
    assert "peer_scripts/peer_code.py" in here and "bridge/showdown_pack.js" in here
    assert PC.digest({"schema": PC.SCHEMA, "scripts": here, "metamon": SI.LEGACY_METAMON,
                      **SI.LEGACY_PEER_ENV}) not in SI.LEGACY_PEER_CODE, "the peer's scripts changed (b0441623, this)"


def _read_here(where: str, name: str):
    p = src_path("main", "anchors", "peer_scripts", name) if where == "peer_scripts" else src_path("utils", "bridge", name)
    return Path(p).read_bytes() if Path(p).is_file() else None


# ------------------------------------------------------------------------------------------------ peer_code.py
def _git(repo: Path, *args: str) -> None:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, env=env)


def test_the_checkout_state_sees_a_tracked_change_and_new_package_code_but_not_the_cache(tmp_path):
    repo = tmp_path / "metamon"
    (repo / "metamon").mkdir(parents=True)
    (repo / "metamon" / "a.py").write_text("A = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    clean = PC.checkout_state(str(repo))
    assert clean["dirty"] == "" and len(clean["commit"]) == 40
    (repo / "cache").mkdir()
    (repo / "cache" / "w.json").write_text("{}")
    (repo / "metamon" / "__pycache__").mkdir()
    (repo / "metamon" / "__pycache__" / "a.pyc").write_bytes(b"x")
    assert PC.checkout_state(str(repo))["dirty"] == "", "the weights cache and bytecode are not the peer's code"
    (repo / "metamon" / "b.py").write_text("B = 2\n")
    new_file = PC.checkout_state(str(repo))
    (repo / "metamon" / "b.py").unlink()
    (repo / "metamon" / "a.py").write_text("A = 2\n")
    edited = PC.checkout_state(str(repo))
    assert len({clean["dirty"], new_file["dirty"], edited["dirty"]}) == 3 and edited["dirty_paths"] == 1
    with pytest.raises(RuntimeError):
        PC.checkout_state(str(tmp_path / "nowhere"))


def test_the_bridge_scripts_count_only_beside_the_team_pack():
    files = {("peer_scripts", "metamon_side.py"): b"m", ("bridge", "pack_oracle.js"): b"o"}
    assert set(PC.hashes_from(lambda w, n: files.get((w, n)))) == {"peer_scripts/metamon_side.py"}
    files[("peer_scripts", "team_pack.py")] = b"t"
    assert set(PC.hashes_from(lambda w, n: files.get((w, n)))) == \
        {"peer_scripts/metamon_side.py", "peer_scripts/team_pack.py", "bridge/pack_oracle.js"}


def test_the_identity_runs_on_any_interpreter_and_digests_only_identity(tmp_path):
    repo = tmp_path / "m"
    (repo / "metamon").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "x")
    rec = PC.identity(str(src_path("main", "anchors", "peer_scripts")), str(src_path("utils", "bridge")), str(repo))
    assert rec["schema"] == PC.SCHEMA and rec["n_packages"] > 0 and rec["metamon"]["dirty"] == ""
    assert PC.digest(dict(rec, n_packages=0, metamon=dict(rec["metamon"], dirty_paths=9))) == PC.digest(rec)
