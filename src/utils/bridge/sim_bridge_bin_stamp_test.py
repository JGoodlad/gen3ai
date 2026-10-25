"""The Rust drivers are judged by their BUILD STAMP before they are spawned (2026-10-20).

The defect: main's ``target/`` binaries predated two Rust commits, and any process that took one by path
(``$POKESIM_SIM_BRIDGE_BIN`` / ``_SEARCH_DRIVER_BIN`` / ``_CORE_EVENTS_BIN`` / ``_LIVE_READER_BIN``, a pre-built
``target/``) ran behaviour no commit pins; only the env core's library carried a source check. Now every binary answers
``--build-stamp`` and ``sim_bridge_bin._checked`` compares it with a recompute over THIS checkout's Rust sources at
every resolve — the override path included, which is the one that does no build.

Unit half: fake binaries that answer ``--build-stamp`` (the check stays ACTIVE; nothing is stubbed). Integration half: the
REAL self-check binaries this session builds carry this tree's stamp, a copy of the tree with one source byte changed
refuses them, and a test-only edit does not.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from utils.bridge import sim_bridge_bin as sb
from utils.paths import src_path
from utils.rust_env import stamp as S
from utils.rust_env.stamp_fakes import make_stamp, write_fake_binary

#: resolver name -> (override env var, cargo bin name)
DRIVERS = {
    "sim_bridge": ("POKESIM_SIM_BRIDGE_BIN", sb.resolve_sim_bridge_bin),
    "search_driver": ("POKESIM_SEARCH_DRIVER_BIN", sb.resolve_search_driver_bin),
    "core_events": ("POKESIM_CORE_EVENTS_BIN", sb.resolve_core_events_bin),
    "live_reader": ("POKESIM_LIVE_READER_BIN", sb.resolve_live_reader_bin),
    # the three OFFLINE tools (2026-10-25): the fuzzers' replayers and the coverage oracle were taken by path unjudged
    "ab_replay": (sb.AB_REPLAY_ENV, sb.resolve_ab_replay_bin),
    "scan_move_probe": (sb.SCAN_MOVE_PROBE_ENV, sb.resolve_scan_move_probe_bin),
    "bridge_replay": (sb.BRIDGE_REPLAY_ENV, sb.resolve_bridge_replay_bin),
}
OFFLINE_TOOLS = ("ab_replay", "scan_move_probe", "bridge_replay")


@pytest.fixture(autouse=True)
def _cold(monkeypatch):
    monkeypatch.setattr(sb, "_rust_bin_cache", {})
    monkeypatch.delenv(S.ALLOW_STALE_ENV, raising=False)
    for var, _ in DRIVERS.values():
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_an_override_naming_a_stale_binary_is_refused_and_a_current_one_passes(name, tmp_path, monkeypatch):
    """The override is taken by path with no build, so it is where staleness hid. A stamp of other sources REFUSES
    (one line: the built-at hash, this checkout's, the exact rebuild command); this tree's passes. Fails on revert."""
    var, resolve = DRIVERS[name]
    good = write_fake_binary(tmp_path / name, make_stamp("port"))
    monkeypatch.setenv(var, str(good))
    assert resolve() == str(good.resolve())
    stale = write_fake_binary(tmp_path / f"stale_{name}", make_stamp("port", src="0123456789abcdef", nfiles="3", commit="19d5bad"))
    monkeypatch.setenv(var, str(stale))
    with pytest.raises(S.StampMismatch) as ei:
        resolve()
    msg = str(ei.value)
    assert "\n" not in msg and "STALE RUST BUILD" in msg
    assert "src=0123456789abcdef" in msg and "19d5bad" in msg, "the built-at hash and commit"
    assert f"src={S.source_hash(scope='port')[0]}" in msg, "this checkout's hash"
    assert sb.build_command(name) in msg and f"--bin {name}" in msg, "the rebuild command"


def test_the_spawn_argv_helpers_refuse_a_stale_override_too(tmp_path, monkeypatch):
    """`bridge_spawn_argv("rust")`, `search_driver_spawn_argv()` and `resolve_and_publish_sim_bridge_bin()` — what the
    spawners and the trainer call — go through the same check: a stale binary is never published to children."""
    stale = write_fake_binary(tmp_path / "sim_bridge", make_stamp("port", src="0123456789abcdef"))
    monkeypatch.setenv("POKESIM_SIM_BRIDGE_BIN", str(stale))
    with pytest.raises(S.StampMismatch):
        sb.bridge_spawn_argv("rust")
    with pytest.raises(S.StampMismatch):
        sb.resolve_and_publish_sim_bridge_bin()
    stale_driver = write_fake_binary(tmp_path / "search_driver", make_stamp("port", src="0123456789abcdef"))
    monkeypatch.setenv("POKESIM_SEARCH_DRIVER_BIN", str(stale_driver))
    with pytest.raises(S.StampMismatch):
        sb.search_driver_spawn_argv()


def test_the_ladder_corpus_oracle_is_taken_through_the_stamped_resolver(tmp_path, monkeypatch):
    """`utils.ladder_corpus.build.scan_probe_bin` (the coverage oracle that decides which moves are "unplayable") once ran
    a bare `cargo build` + path; it now resolves through the stamp check, so a stale oracle is refused."""
    from utils.ladder_corpus import build as corpus_build

    stale = write_fake_binary(tmp_path / "scan_move_probe", make_stamp("port", src="0123456789abcdef"))
    monkeypatch.setenv(sb.SCAN_MOVE_PROBE_ENV, str(stale))
    with pytest.raises(S.StampMismatch, match="scan_move_probe"):
        corpus_build.scan_probe_bin()
    good = write_fake_binary(tmp_path / "good_scan_move_probe", make_stamp("port"))
    monkeypatch.setenv(sb.SCAN_MOVE_PROBE_ENV, str(good))
    assert corpus_build.scan_probe_bin() == str(good.resolve())


def test_a_binary_that_predates_the_flag_is_refused(tmp_path, monkeypatch):
    old = write_fake_binary(tmp_path / "sim_bridge", None)           # `--build-stamp` prints nothing: an old build
    monkeypatch.setenv("POKESIM_SIM_BRIDGE_BIN", str(old))
    with pytest.raises(S.StampMismatch, match="reports no build stamp"):
        sb.resolve_sim_bridge_bin()


def test_the_opt_out_runs_a_stale_binary_with_a_warning(tmp_path, monkeypatch, capsys):
    stale = write_fake_binary(tmp_path / "search_driver", make_stamp("port", src="0123456789abcdef"))
    monkeypatch.setenv("POKESIM_SEARCH_DRIVER_BIN", str(stale))
    with pytest.raises(S.StampMismatch):
        sb.resolve_search_driver_bin()
    monkeypatch.setenv(S.ALLOW_STALE_ENV, "1")
    capsys.readouterr()
    assert sb.resolve_search_driver_bin() == str(stale.resolve())
    err = capsys.readouterr().err
    assert "STALE RUST BUILD ALLOWED" in err and "search_driver" in err and "0123456789abcdef" in err, err


def test_a_foreign_checkouts_binary_is_judged_against_THIS_checkouts_sources(tmp_path, monkeypatch):
    """The pinned-run property: the tree hashed is the one `utils.paths` finds this code in — the PIN's in a pinned run —
    never main's. A binary stamped for another tree's sources is refused here however current it is THERE."""
    other = tmp_path / "other_src"
    for crate in ("rust_sim", "rust_env"):
        shutil.copytree(src_path(crate, "src"), other / crate / "src", ignore=shutil.ignore_patterns("target"))
        for f in ("Cargo.toml", "build.rs", "build_support.rs"):
            if src_path(crate, f).is_file():
                shutil.copy2(src_path(crate, f), other / crate / f)
    victim = other / "rust_sim" / "src" / "turn.rs"
    victim.write_bytes(victim.read_bytes() + b"\n// a commit main does not have\n")
    theirs = write_fake_binary(tmp_path / "sim_bridge", make_stamp("port", tree=other))     # current for ITS tree
    monkeypatch.setenv("POKESIM_SIM_BRIDGE_BIN", str(theirs))
    with pytest.raises(S.StampMismatch):
        sb.resolve_sim_bridge_bin()
    # ...and it is accepted by a checkout that has exactly those sources (the stamp's own tree)
    S.check_binary_stamp(theirs, "sim_bridge", kind="port", src=other)


# ------------------------------------------------------------------ the REAL binaries


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_the_real_built_driver_carries_this_trees_stamp(name, monkeypatch):
    """Resolving builds (or reuses) this checkout's self-check binary, and the resolver has already judged it; here the
    stamp itself is read back: kind port, this tree's `src` and file count, a self-check build (the session's)."""
    _, resolve = DRIVERS[name]
    path = Path(resolve())
    st = S.parse_stamp(S.binary_stamp(path) or "")
    assert st["kind"] == "port"
    h, n = S.source_hash(scope="port")
    assert (st["src"], st["nfiles"]) == (h, str(n))
    assert st["selfcheck"] == "1" and path.parent.name == "selfcheck"


@pytest.mark.integration
def test_a_real_binary_has_teeth_one_source_byte_refuses_and_a_test_only_edit_does_not(tmp_path):
    """Against the REAL `sim_bridge`: a copy of the tree with one `.rs` byte changed REFUSES it; with only a test module
    edited it passes. (Fails on revert of either the hash or the test-only rule.)"""
    path = Path(sb.resolve_sim_bridge_bin())
    copy = tmp_path / "src"
    for crate in ("rust_sim", "rust_env"):
        shutil.copytree(src_path(crate, "src"), copy / crate / "src", ignore=shutil.ignore_patterns("target"))
        for f in ("Cargo.toml", "Cargo.lock", "build.rs", "build_support.rs"):
            if src_path(crate, f).is_file():
                shutil.copy2(src_path(crate, f), copy / crate / f)
    S.check_binary_stamp(path, "sim_bridge", kind="port", src=copy)
    tests_rs = copy / "rust_sim" / "src" / "present" / "tests.rs"
    tests_rs.write_bytes(tests_rs.read_bytes() + b"// edited test\n")
    S.check_binary_stamp(path, "sim_bridge", kind="port", src=copy)
    turn = copy / "rust_sim" / "src" / "turn.rs"
    turn.write_bytes(turn.read_bytes() + b" ")
    with pytest.raises(S.StampMismatch, match="STALE RUST BUILD"):
        S.check_binary_stamp(path, "sim_bridge", kind="port", src=copy)


# ------------------------------------------------------------------ bot_reader: the ENV crate's binary (kind env)


def test_bot_reader_is_judged_against_the_env_scope(tmp_path, monkeypatch):
    """`bot_reader` is built from `rust_env` (the port included), so its stamp is the ENV one: a port stamp is refused
    (wrong kind), a stale env stamp is refused, and this tree's env stamp passes."""
    from utils.rust_env import bot_reader_bin as BR

    monkeypatch.setattr(BR, "_cache", {})
    good = write_fake_binary(tmp_path / "bot_reader", make_stamp("env"))
    monkeypatch.setenv(BR.ENV_OVERRIDE, str(good))
    assert BR.resolve_bot_reader_bin() == str(good.resolve())
    stale = write_fake_binary(tmp_path / "stale_bot_reader", make_stamp("env", src="0123456789abcdef"))
    monkeypatch.setenv(BR.ENV_OVERRIDE, str(stale))
    with pytest.raises(S.StampMismatch, match="STALE RUST BUILD") as ei:
        BR.resolve_bot_reader_bin()
    assert "cargo build" in str(ei.value) and "--bin bot_reader" in str(ei.value)
    wrong = write_fake_binary(tmp_path / "port_bot_reader", make_stamp("port"))
    monkeypatch.setenv(BR.ENV_OVERRIDE, str(wrong))
    with pytest.raises(S.StampMismatch, match="kind"):
        BR.resolve_bot_reader_bin()


@pytest.mark.integration
def test_the_real_env_binaries_carry_this_trees_env_stamp():
    """The REAL self-check `bot_reader` and `rust_env_proc` print the ENV stamp (the port + `rust_env` sources, schema,
    data) and this tree accepts it — the resolver already judged `bot_reader`; `rust_env_proc` is judged here."""
    from utils.rust_env import bot_reader_bin as BR
    from utils.rust_env import build as env_build

    reader = Path(BR.resolve_bot_reader_bin())
    child, _lib = env_build.ensure_built("selfcheck")
    for binary in (reader, child):
        st = S.check_binary_stamp(binary, binary.name, kind="env")
        assert st is not None and st["kind"] == "env" and (st["src"], int(st["nfiles"])) == S.source_hash(scope="env")
