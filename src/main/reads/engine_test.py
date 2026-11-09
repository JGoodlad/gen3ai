"""``main.reads.engine``: the frozen LEGACY tables reproduce from git, the reconstruction reproduces the core stamps the
eval ledger RECORDED, and an engine comparison refuses a mismatch or an unknown side (REVIEW F F-M2)."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from main.reads import engine as EN
from utils.paths import repo_root, src_path


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


# ---------------------------------------------------------------------------------------- the Cargo.lock files (2026-11-09)
#
# The build stamp hashes each root crate's Cargo.lock. It was git-IGNORED, so only a `cargo` run wrote it: a fresh worktree
# bootstrapped without `--with-rust` stamped `port=…/94;env=…/129` where main's `/95` and `/130`, and a registered read
# (which pins the engine stamp) played on another engine. The locks are tracked now; these tests fail on a revert.
_CRATES = (("port", "rust_sim"), ("env", "rust_env"))


def _g(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                        *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"git {args}: {r.stderr}")
    return r


def _mini_repo(root: Path, *, lock: bool = True) -> Path:
    """A git repo laid out like the two crates (a lib and a manifest each), the locks committed when ``lock``."""
    for _tag, crate in _CRATES:
        (root / "src" / crate / "src").mkdir(parents=True)
        (root / "src" / crate / "Cargo.toml").write_text(f'[package]\nname = "{crate}"\nversion = "0.0.0"\n')
        (root / "src" / crate / "src" / "lib.rs").write_text("// lib\n")
        if lock:
            (root / "src" / crate / "Cargo.lock").write_text(f"# lock of {crate}\n")
    _g(root, "init", "-q", "-b", "main")
    _g(root, "add", "-A")
    _g(root, "commit", "-q", "-m", "init")
    return root


def test_the_two_cargo_locks_are_tracked_not_ignored_and_the_banked_bytes():
    """A lock that git does not hold makes a fresh clone stamp another engine. The bytes are the ones EVERY banked build
    hashed (both crates are std-only, so cargo writes the same file): a changed lock is a dependency added (see
    ``src/rust_sim/CLAUDE.md``: std-only on purpose) and a NEW ENGINE, to be decided, not slipped in."""
    for tag, crate in _CRATES:
        rel = f"src/{crate}/Cargo.lock"
        assert _g(repo_root(), "ls-files", "--error-unmatch", rel, check=False).returncode == 0, f"{rel} is not tracked"
        assert _g(repo_root(), "check-ignore", "--no-index", "-q", rel, check=False).returncode == 1, f"{rel} is git-ignored"
        assert EN.lock_blob(repo_root() / rel) == EN.LEGACY_LOCK_BLOBS[tag], f"{rel} is not the lock every banked engine hashed"


def test_a_fresh_checkout_stamps_the_engine_its_commit_names(tmp_path):
    """THE CHECK THAT WOULD HAVE CAUGHT 2026-11-09: export what a fresh clone of HEAD carries (``git archive``: tracked
    files only) and stamp it from disk; it must equal the engine computed from HEAD's git objects. With the locks
    untracked the export has none and the two disagree (``/94`` and ``/129`` against ``/95`` and ``/130``)."""
    held = _g(repo_root(), "ls-tree", "HEAD", "--", "src/rust_sim/Cargo.lock", "src/rust_env/Cargo.lock").stdout.splitlines()
    assert len(held) == 2, "HEAD does not hold both Cargo.lock files, so a fresh clone stamps another engine"
    tar = subprocess.run(["git", "-C", str(repo_root()), "archive", "--format=tar", "HEAD", "--", "src/rust_sim", "src/rust_env",
                          ":(exclude)src/rust_sim/tests", ":(exclude)src/rust_sim/harness", ":(exclude)src/rust_env/tests"],
                         capture_output=True, check=True).stdout        # ~4.5 MB of the 87 MB; the stamp hashes none of the rest
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(tmp_path, filter="data")
    assert EN.tree_engine(tmp_path / "src") == EN.engine_at("HEAD")


def test_the_server_engine_stamp_is_the_trees_engine(monkeypatch):
    """``main.anchors.server.engine_stamp`` (what every unit stamps) and ``tree_engine`` (what the check prints, for any
    tree) are one definition: a play-path file and a reads file cannot share code, so this holds them equal."""
    from main.anchors import server

    monkeypatch.delenv("POKESIM_ALLOW_STALE_BUILD", raising=False)
    assert server.engine_stamp() == EN.tree_engine()


def test_engine_at_reads_the_tracked_lock_and_a_commit_without_one_takes_the_banked_blob(tmp_path):
    without = _mini_repo(tmp_path / "a", lock=False)
    old = _g(without, "rev-parse", "HEAD").stdout.strip()
    assert not any(r.endswith("/Cargo.lock") for scope in ("port", "env") for r, _b in EN.committed_listing(old, without, scope)), \
        "no fallback given: a commit without a tracked lock lists none"
    banked = EN.engine_at(old, without)
    assert banked == EN.engine_at(old, without, lock_blobs=EN.LEGACY_LOCK_BLOBS), "the default fallback is the banked blobs"
    assert {r: b for r, b in EN.committed_listing(old, without, "env", lock_blobs=EN.LEGACY_LOCK_BLOBS) if r.endswith("Cargo.lock")} \
        == {"env/Cargo.lock": EN.LEGACY_LOCK_BLOBS["env"]}
    with_lock = _mini_repo(tmp_path / "b", lock=True)
    new = _g(with_lock, "rev-parse", "HEAD").stdout.strip()
    tracked = {r: b for r, b in EN.committed_listing(new, with_lock, "env") if r.endswith("Cargo.lock")}
    assert tracked == {"env/Cargo.lock": EN.lock_blob(with_lock / "src" / "rust_env" / "Cargo.lock")}, \
        "a commit that tracks the lock is judged by ITS blob, not the banked one"
    assert "port/Cargo.lock" not in [r for r, _b in EN.committed_listing(new, with_lock, "env")], "only the ROOT crate's lock"
    assert "port/Cargo.lock" in [r for r, _b in EN.committed_listing(new, with_lock, "port")], "the port's own scope roots at the port"
    assert EN.engine_at(new, with_lock) != banked


def test_the_input_check_names_a_missing_an_untracked_and_a_filtered_lock(tmp_path):
    ok = _mini_repo(tmp_path / "ok")
    assert EN.engine_input_problems(ok / "src") == []
    # (1) a lock that is not there: what a fresh worktree that never ran cargo looked like
    (ok / "src" / "rust_env" / "Cargo.lock").unlink()
    (miss,) = EN.engine_input_problems(ok / "src")
    assert "src/rust_env/Cargo.lock is missing" in miss and "--install-locks" in miss
    _g(ok, "checkout", "--", "src/rust_env/Cargo.lock")
    assert EN.engine_input_problems(ok / "src") == []
    # (2) a lock that git ignores: the old .gitignore
    ign = _mini_repo(tmp_path / "ign")
    _g(ign, "rm", "-q", "--cached", "src/rust_sim/Cargo.lock")
    (ign / ".gitignore").write_text("Cargo.lock\n")
    (untracked,) = EN.engine_input_problems(ign / "src")
    assert "src/rust_sim/Cargo.lock" in untracked and "does not track it" in untracked
    # (3) a checkout filter: clean status, but the bytes on disk are not the committed ones
    flt = tmp_path / "flt"
    flt.mkdir()
    (flt / ".gitattributes").write_text("Cargo.lock text eol=crlf\n")
    _mini_repo(flt)
    (flt / "src" / "rust_env" / "Cargo.lock").unlink()
    _g(flt, "checkout", "--", "src/rust_env/Cargo.lock")
    assert b"\r\n" in (flt / "src" / "rust_env" / "Cargo.lock").read_bytes() and not _g(flt, "status", "--porcelain").stdout.strip()
    (one,) = EN.engine_input_problems(flt / "src")
    assert "differs from HEAD's committed" in one
    # uncommitted edits to a crate: only (1) and (2) apply, so an edit in flight is not a drift
    (ok / "src" / "rust_sim" / "src" / "lib.rs").write_text("// edited\n")
    assert EN.engine_input_problems(ok / "src") == []


def test_install_locks_puts_the_banked_bytes_into_an_old_pin_and_never_overwrites(tmp_path):
    source = _mini_repo(tmp_path / "source")
    pin = _mini_repo(tmp_path / "pin", lock=False)
    assert "Cargo.lock is missing" in "".join(EN.engine_input_problems(pin / "src"))
    lines = EN.install_locks(pin, repo=source)
    assert [ln.split(":")[0] for ln in lines] == ["src/rust_sim/Cargo.lock", "src/rust_env/Cargo.lock"] and all("installed" in x for x in lines)
    for _tag, crate in _CRATES:
        assert (pin / "src" / crate / "Cargo.lock").read_bytes() == (source / "src" / crate / "Cargo.lock").read_bytes()
    assert all("already there" in x for x in EN.install_locks(pin, repo=source)), "idempotent"
    (pin / "src" / "rust_env" / "Cargo.lock").write_text("# another lock\n")
    with pytest.raises(EN.EngineError, match="exists and differs"):
        EN.install_locks(pin, repo=source)
    (source / "src" / "rust_sim" / "Cargo.lock").write_text("# edited, uncommitted\n")
    with pytest.raises(EN.EngineError, match="not committed unchanged"):
        EN.install_locks(tmp_path / "elsewhere", repo=source)


def test_the_command_line_checks_prints_and_refuses_on_an_engine_mismatch(tmp_path):
    ok = _mini_repo(tmp_path / "ok")
    pin = _mini_repo(tmp_path / "pin", lock=False)
    env = {**os.environ, "PYTHONPATH": str(src_path())}

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "main.reads.engine", *args], capture_output=True, text=True, env=env, timeout=120)

    good = cli("--check-inputs", "--tree", str(ok))
    assert good.returncode == 0 and good.stderr == ""
    bad = cli("--check-inputs", "--tree", str(pin))
    assert bad.returncode == 1 and "Cargo.lock is missing" in bad.stderr
    notgit = tmp_path / "notgit"
    (notgit / "src").mkdir(parents=True)
    assert cli("--check-inputs", "--tree", str(notgit)).returncode == 2, "git cannot answer: the third verdict"
    key = cli("--engine", "--tree", str(ok)).stdout.strip()
    assert key == EN.tree_engine(ok / "src") and key.startswith("port=") and ";env=" in key
    assert cli("--engine", "--tree", str(ok), "--expect", key).returncode == 0
    miss = cli("--engine", "--tree", str(pin), "--expect", key)
    assert miss.returncode == 1 and "ENGINE MISMATCH" in miss.stderr and miss.stdout.strip() != key
    assert cli("--install-locks").returncode == 2, "argparse: --install-locks needs --tree"


def _cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the Cargo.lock sync check cannot run (install rustup)")
    return cargo


@pytest.mark.integration
def test_each_committed_lock_is_what_cargo_would_write_for_its_manifest(tmp_path):
    """``--locked`` is the build's own check, so the committed lock must satisfy it; and it has teeth: a manifest edited
    without its lock (here the port's version, which the env crate's lock records too) fails it."""
    cargo = _cargo()

    def metadata(root: Path, crate: str) -> subprocess.CompletedProcess:
        return subprocess.run([cargo, "metadata", "--locked", "--offline", "--format-version", "1",
                               "--manifest-path", str(root / crate / "Cargo.toml")], capture_output=True, text=True, timeout=300)

    real = src_path()
    for _tag, crate in _CRATES:
        r = metadata(real, crate)
        assert r.returncode == 0, f"src/{crate}/Cargo.lock is out of sync with its manifest:\n{r.stderr[-600:]}"
    for crate in ("rust_sim", "rust_env"):
        shutil.copytree(real / crate, tmp_path / crate, ignore=shutil.ignore_patterns("target", "tests", "harness", "examples",
                                                                                       "__pycache__"))
    assert metadata(tmp_path, "rust_env").returncode == 0, "the copy is a faithful stand-in"
    manifest = tmp_path / "rust_sim" / "Cargo.toml"
    manifest.write_text(manifest.read_text().replace('name = "pokesim"\nversion = "0.0.0"', 'name = "pokesim"\nversion = "0.0.1"', 1))
    for crate in ("rust_sim", "rust_env"):
        r = metadata(tmp_path, crate)
        assert r.returncode != 0 and "lock file" in r.stderr, f"{crate}: --locked let an out-of-sync lock through:\n{r.stderr[-400:]}"
