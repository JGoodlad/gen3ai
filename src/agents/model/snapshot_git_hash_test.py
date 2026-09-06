"""A checkpoint must be stamped with the commit of the code THAT RAN — and with its history.

🚨 THE INCIDENT (2026-09-05). A launcher-pinned run recorded pin `eb5261ff` in its run-level
`metadata.json` while every checkpoint SIDECAR recorded `fff95a16`, the ambient HEAD of the
main checkout. Two independent causes, both fixed here:

  1. `record_checkpoint` resolved `git_hash or get_git_hash()` — never consulting
     `$LAUNCHER_GIT_HASH` — and the resulting truthy value then WON the `git_hash or env`
     chain inside `_build_snapshot_entry` too, so the env fallback there was dead code.
  2. `utils.git.get_git_hash()` ran `git rev-parse HEAD` in the process CWD. The launcher
     puts the pinned worktree on the child's PYTHONPATH but spawns it with **no `cwd=`**, so
     the child imports the pin while standing in the un-pinned main checkout.

Then the resume read the sidecar and pinned the restart to the wrong commit.

And a third, structural one (same day): the run-level scalar `git_hash` is REWRITTEN on every
save, so on a run that restarts every 3 h it names the LAST code to touch the run rather than
the code that ran most of it. `pin_history` is the append-only record that does not lose that.

Run: python -m pytest src/agents/model/snapshot_git_hash_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import json
import os
import subprocess

import pytest

from agents.model import snapshot as sn
from agents.model.snapshot import (
    GitHashMismatchError, LAUNCHER_GIT_HASH_ENV, resolve_git_hash,
)
from utils.git import get_git_hash
from utils.paths import repo_root


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture(autouse=True)
def _no_launcher_env(monkeypatch):
    """Every test states its own env — a stray inherited pin would silently change answers."""
    monkeypatch.delenv(LAUNCHER_GIT_HASH_ENV, raising=False)
    monkeypatch.delenv("LAUNCHER_PIN_SOURCE", raising=False)


@pytest.fixture
def other_repo(tmp_path):
    """A DIFFERENT one-commit git repo, so 'the cwd's HEAD' is a distinguishable answer."""
    root = tmp_path / "elsewhere"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "f.txt").write_text("x")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "unrelated")
    return str(root), _git(root, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------------------
# (a) get_git_hash answers about the IMPORTED checkout, not the cwd
# ---------------------------------------------------------------------------------------

def test_a_get_git_hash_ignores_the_process_cwd(other_repo, monkeypatch):
    """This is cause #2 above, reproduced: stand in another repo, ask for our hash."""
    root, other_head = other_repo
    ours = _git(str(repo_root()), "rev-parse", "HEAD")
    assert other_head != ours, "fixture must produce a distinguishable commit"

    monkeypatch.chdir(root)
    assert get_git_hash() == ours, (
        "get_git_hash() answered about the CWD's repo — that is exactly how the ambient main "
        "HEAD got stamped into a pinned run's sidecars")
    assert get_git_hash() != other_head
    assert get_git_hash(short=True) == ours[:len(get_git_hash(short=True))]


# ---------------------------------------------------------------------------------------
# (b)/(c) the launcher pin and the imported tree must AGREE, or the write refuses
# ---------------------------------------------------------------------------------------

def test_b_a_disagreeing_launcher_pin_raises_at_the_sidecar_write(tmp_path, monkeypatch):
    monkeypatch.setenv(LAUNCHER_GIT_HASH_ENV, "deadbeef" * 5)
    ckpt = tmp_path / "checkpoint_100_steps.zip"
    ckpt.write_text("")
    with pytest.raises(GitHashMismatchError) as e:
        sn.write_checkpoint_metadata(str(ckpt), lr=3e-4, n_epochs=5)
    msg = str(e.value)
    assert "deadbeef" in msg and get_git_hash()[:8] in msg, "name BOTH hashes"
    assert LAUNCHER_GIT_HASH_ENV in msg, "name the env var so the reader can act"
    assert not os.path.exists(str(tmp_path / "checkpoint_100_steps.json")), (
        "a refused write must leave no half-truth behind")


def test_c_an_agreeing_launcher_pin_passes(tmp_path, monkeypatch):
    monkeypatch.setenv(LAUNCHER_GIT_HASH_ENV, get_git_hash())
    ckpt = tmp_path / "checkpoint_100_steps.zip"
    ckpt.write_text("")
    sn.write_checkpoint_metadata(str(ckpt), lr=3e-4, n_epochs=5)
    entry = json.load(open(str(tmp_path / "checkpoint_100_steps.json")))
    assert entry["git_hash"] == get_git_hash()


def test_c_a_short_pin_agrees_with_the_full_hash(monkeypatch):
    """`--pin-commit` may be a prefix; a prefix is not a disagreement."""
    monkeypatch.setenv(LAUNCHER_GIT_HASH_ENV, get_git_hash()[:8])
    assert resolve_git_hash() == get_git_hash()[:8]


def test_c_an_explicit_hash_bypasses_the_guard(monkeypatch):
    """A caller that already knows (the round-trip smoke's sentinel) is not second-guessed."""
    monkeypatch.setenv(LAUNCHER_GIT_HASH_ENV, "deadbeef" * 5)
    assert resolve_git_hash("roundtrip-test") == "roundtrip-test"


# ---------------------------------------------------------------------------------------
# (d) with no env var, the sidecar's hash is repo_root()'s HEAD — from ANY cwd
# ---------------------------------------------------------------------------------------

def test_d_with_no_env_the_sidecar_hash_is_the_imported_checkouts_head(
        tmp_path, other_repo, monkeypatch):
    root, other_head = other_repo
    monkeypatch.chdir(root)                       # stand somewhere else entirely
    ckpt = tmp_path / "checkpoint_7_steps.zip"
    ckpt.write_text("")
    sn.write_checkpoint_metadata(str(ckpt), lr=1e-4, n_epochs=3)
    entry = json.load(open(str(tmp_path / "checkpoint_7_steps.json")))
    assert entry["git_hash"] == _git(str(repo_root()), "rev-parse", "HEAD")
    assert entry["git_hash"] != other_head


def test_d_record_checkpoint_stamps_the_same_hash_in_both_places(tmp_path, monkeypatch):
    """The sidecar and snapshot_history must never disagree — one resolver, one answer."""
    monkeypatch.setenv(LAUNCHER_GIT_HASH_ENV, get_git_hash())
    ckpt = tmp_path / "checkpoint_500_steps.zip"
    ckpt.write_text("")
    sn.record_checkpoint(str(tmp_path), str(ckpt), 3e-4, 5)
    side = json.load(open(str(tmp_path / "checkpoint_500_steps.json")))
    hist = json.load(open(str(tmp_path / "metadata.json")))["snapshot_history"]
    assert side["git_hash"] == get_git_hash()
    assert hist["checkpoint_500_steps.zip"]["git_hash"] == get_git_hash()


# ---------------------------------------------------------------------------------------
# pin_history — the append-only "which commit ran which steps"
# ---------------------------------------------------------------------------------------

def _save(model_dir, *, git_hash, step, pin_source=None, monkeypatch=None):
    if monkeypatch is not None:
        if pin_source:
            monkeypatch.setenv("LAUNCHER_PIN_SOURCE", pin_source)
        else:
            monkeypatch.delenv("LAUNCHER_PIN_SOURCE", raising=False)
    version = _FakeVersion()
    sn.save_model_snapshot(str(model_dir), version, git_hash=git_hash, num_timesteps=step)


class _FakeVersion:
    """`save_model_snapshot` only ever calls `.to_json()` on the version it is handed."""
    def to_json(self) -> str:
        return json.dumps({"config_version": 1})


def test_pin_history_a_two_saves_under_one_hash_make_ONE_entry(tmp_path, monkeypatch):
    _save(tmp_path, git_hash="aaaa1111", step=1000, pin_source="checkpoint",
          monkeypatch=monkeypatch)
    _save(tmp_path, git_hash="aaaa1111", step=5000, pin_source="checkpoint",
          monkeypatch=monkeypatch)
    hist = json.load(open(str(tmp_path / "metadata.json")))["pin_history"]
    assert len(hist) == 1, f"same commit ⇒ one span, got {hist}"
    assert hist[0] == {"git_hash": "aaaa1111", "pin_source": "checkpoint",
                       "first_step": 1000, "last_step": 5000}


def test_pin_history_b_a_new_hash_appends_and_leaves_the_first_untouched(tmp_path, monkeypatch):
    _save(tmp_path, git_hash="aaaa1111", step=1000, pin_source="head", monkeypatch=monkeypatch)
    _save(tmp_path, git_hash="aaaa1111", step=9000, pin_source="head", monkeypatch=monkeypatch)
    _save(tmp_path, git_hash="bbbb2222", step=9500, pin_source="pin_commit",
          monkeypatch=monkeypatch)
    hist = json.load(open(str(tmp_path / "metadata.json")))["pin_history"]
    assert len(hist) == 2
    assert hist[0] == {"git_hash": "aaaa1111", "pin_source": "head",
                       "first_step": 1000, "last_step": 9000}, "the first span was rewritten"
    assert hist[1] == {"git_hash": "bbbb2222", "pin_source": "pin_commit",
                       "first_step": 9500, "last_step": 9500}
    # ...and the SCALAR still says "current", which is precisely why the list has to exist.
    assert json.load(open(str(tmp_path / "metadata.json")))["git_hash"] == "bbbb2222"


def test_pin_history_c_a_legacy_metadata_is_seeded_and_marked_derived(tmp_path, monkeypatch):
    """A pre-2026-09-05 run has a scalar git_hash and no history. Absence must not read as
    'this run only ever ran one commit' — the old hash gets a `derived: true` span."""
    (tmp_path / "metadata.json").write_text(json.dumps({
        "git_hash": "old0old0", "pin_source": "checkpoint", "saved_at": "2026-08-01",
    }))
    _save(tmp_path, git_hash="new1new1", step=42000, pin_source="head", monkeypatch=monkeypatch)
    hist = json.load(open(str(tmp_path / "metadata.json")))["pin_history"]
    assert len(hist) == 2, f"legacy hash + current hash = two spans, got {hist}"
    assert hist[0]["git_hash"] == "old0old0"
    assert hist[0]["derived"] is True, "the seeded span must say its bounds were inferred"
    assert hist[0]["first_step"] == 42000, "seeded at the step we NOTICED it"
    assert hist[1]["git_hash"] == "new1new1" and hist[1]["first_step"] == 42000


def test_pin_history_rides_into_the_checkpoint_sidecar(tmp_path, monkeypatch):
    _save(tmp_path, git_hash="aaaa1111", step=1000, monkeypatch=monkeypatch)
    ckpt = tmp_path / "checkpoint_1000_steps.zip"
    ckpt.write_text("")
    sn.record_checkpoint(str(tmp_path), str(ckpt), 3e-4, 5, git_hash="aaaa1111")
    side = json.load(open(str(tmp_path / "checkpoint_1000_steps.json")))
    assert side["pin_history"] == [{"git_hash": "aaaa1111", "pin_source": None,
                                    "first_step": 1000, "last_step": 1000}]


# ---------------------------------------------------------------------------------------
# num_timesteps — HOW FAR THE RUN TRAINED, in the same two files, from the same save path
#
# Found by the fold_displacement probe (2026-09-05): a run's step count lived in NO
# metadata.json field, so `main.lineage` / `main.sidecar_audit` / `main.dose` — all
# deliberately JSON-only, no torch, no .zip opened — could not read it at all. Unlike
# `original_command` / `lineage` / `pin_history` this key is "latest": overwritten on every
# save. And unlike them, ABSENT must read as UNKNOWN rather than as 0.
# ---------------------------------------------------------------------------------------

def _meta(model_dir):
    return json.load(open(str(model_dir / "metadata.json")))


def test_steps_a_the_run_level_key_is_written_and_OVERWRITTEN(tmp_path, monkeypatch):
    _save(tmp_path, git_hash="aaaa1111", step=1000, monkeypatch=monkeypatch)
    assert _meta(tmp_path)["num_timesteps"] == 1000
    _save(tmp_path, git_hash="aaaa1111", step=9000, monkeypatch=monkeypatch)
    assert _meta(tmp_path)["num_timesteps"] == 9000, "'latest' — not immutable like lineage"


def test_steps_b_a_save_that_knows_no_step_CARRIES_FORWARD_rather_than_clobbering(
        tmp_path, monkeypatch):
    """`_max_recorded_step`'s stale-but-ordered guess feeds pin_history's spans only. Writing a
    guess HERE would make an inferred number indistinguishable from a recorded one, and writing
    nothing would lose the fact the run already stated."""
    _save(tmp_path, git_hash="aaaa1111", step=4242, monkeypatch=monkeypatch)
    sn.save_model_snapshot(str(tmp_path), _FakeVersion(), git_hash="aaaa1111")  # no step
    assert _meta(tmp_path)["num_timesteps"] == 4242


def test_steps_c_a_LEGACY_run_reads_as_UNKNOWN_never_as_zero(tmp_path, monkeypatch):
    monkeypatch.setenv(LAUNCHER_GIT_HASH_ENV, "aaaa1111")
    sn.save_model_snapshot(str(tmp_path), _FakeVersion(), git_hash="aaaa1111")
    assert "num_timesteps" not in _meta(tmp_path), "absent is UNKNOWN; 0 would be a claim"


def test_steps_d_the_explicit_argument_wins_over_a_colliding_hparam(tmp_path, monkeypatch):
    monkeypatch.setenv(LAUNCHER_GIT_HASH_ENV, "aaaa1111")
    sn.save_model_snapshot(str(tmp_path), _FakeVersion(), git_hash="aaaa1111",
                           hparams={"num_timesteps": 111}, num_timesteps=222)
    assert _meta(tmp_path)["num_timesteps"] == 222


def test_steps_e_every_sidecar_and_history_row_carries_it(tmp_path, monkeypatch):
    """One save path, both places — the same contract `git_hash` and `pin_history` already have."""
    monkeypatch.setenv(LAUNCHER_GIT_HASH_ENV, "aaaa1111")
    ckpt = tmp_path / "checkpoint_777000_steps.zip"
    ckpt.write_text("")
    sn.record_checkpoint(str(tmp_path), str(ckpt), 3e-4, 5, git_hash="aaaa1111",
                         hparams={"num_timesteps": 777000})
    side = json.load(open(str(tmp_path / "checkpoint_777000_steps.json")))
    hist = _meta(tmp_path)["snapshot_history"]["checkpoint_777000_steps.zip"]
    assert side["num_timesteps"] == 777000
    assert hist["num_timesteps"] == 777000, "sidecar and history must never disagree"


def test_steps_f_the_zips_own_name_is_the_last_resort(tmp_path):
    """A caller that passes neither the value nor an hparams block still gets a real number
    out of `checkpoint_<N>_steps.zip` — the name the checkpoint callback writes FROM
    `model.num_timesteps`."""
    ckpt = tmp_path / "checkpoint_31337_steps.zip"
    ckpt.write_text("")
    sn.write_checkpoint_metadata(str(ckpt), lr=1e-4, n_epochs=3)
    assert json.load(open(str(tmp_path / "checkpoint_31337_steps.json")))["num_timesteps"] == 31337


def test_steps_g_a_name_with_no_step_in_it_stays_UNKNOWN(tmp_path):
    """`final_model.zip` / `best_model.zip` encode no step. Inventing one would be worse than
    saying nothing — the readers render an absent key as unknown."""
    ckpt = tmp_path / "final_model.zip"
    ckpt.write_text("")
    sn.write_checkpoint_metadata(str(ckpt), lr=1e-4, n_epochs=3)
    assert "num_timesteps" not in json.load(open(str(tmp_path / "final_model.json")))
