"""Gate ⑤ — the build stamp's TEETH (M5 Lane 0, widened 2026-10-20 to every Rust artifact Python runs).

The stamp refuses a build whose sources, column schema, data directory or build kind are not this
tree's. Proven on a temporary COPY of the sources that reach the build (never the checkout itself):
the copy hashes like the tree, a one-byte edit or an added source flips the verdict to a refusal,
a TEST-ONLY file's edit does not, and restoring the file restores it. The port's drivers carry a stamp of
their own (``--build-stamp``), checked at every resolve: ``check_binary_stamp`` is proven on fake binaries
here and on the REAL built drivers in ``sim_bridge_bin_stamp_test.py``. The cross-language agreement (the
Rust ``build.rs`` hash == this module's) is proven with rustc on a copy below and on a real cargo build by
``core_cargo_test.py``.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from utils.paths import src_path
from utils.rust_env import columns
from utils.rust_env import stamp as S
from utils.rust_env.stamp_fakes import make_stamp, write_fake_binary


def _copy_tree(root: Path) -> Path:
    """The sources that reach a build, copied under ``root/src`` (+ a ``root/data`` link the env build script wants)."""
    for crate in ("rust_sim", "rust_env"):
        shutil.copytree(src_path(crate, "src"), root / "src" / crate / "src", ignore=shutil.ignore_patterns("target"))
        for f in ("Cargo.toml", "Cargo.lock", "build.rs", "build_support.rs"):
            if src_path(crate, f).is_file():
                shutil.copy2(src_path(crate, f), root / "src" / crate / f)
    (root / "data").symlink_to(src_path().parent / "data")
    return root / "src"


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return _copy_tree(tmp_path_factory.mktemp("stamp_src"))


def _stamp(src_hash, n, **kw):
    fields = dict(stamp="v2", kind="env", commit="c0ffee", src=src_hash, nfiles=str(n), nan_poison="1",
                  schema=columns.schema_id(), data=S.data_dir())
    fields.update(kw)
    return ";".join(f"{k}={v}" for k, v in fields.items())


def test_a_copy_hashes_like_the_tree(tree):
    assert S.source_hash(tree) == S.source_hash()
    rels = [r for r, _ in S.source_listing()]
    assert "env/build.rs" in rels and "port/Cargo.toml" in rels and "env/src/core/columns.rs" in rels
    assert "port/build.rs" in rels and "port/build_support.rs" in rels
    assert sum(r.startswith("port/src/") for r in rels) > 50, "the port's sources are in the listing"
    assert S.source_hash(tree, "port") == S.source_hash(scope="port")
    assert not any(r.startswith("env/") for r, _ in S.source_listing(scope="port")), "the port's scope holds no env file"


def test_the_stamp_has_teeth(tree):
    """A matching stamp passes; ONE source byte changed or one source added refuses; restoring passes again."""
    h, n = S.source_hash(tree)
    good = _stamp(h, n)
    S.check_stamp(good, "t", src=tree)
    f = tree / "rust_env" / "src" / "core" / "pool.rs"
    orig = f.read_bytes()
    try:
        f.write_bytes(orig + b"// one comment\n")
        with pytest.raises(S.StampMismatch, match="sources"):
            S.check_stamp(good, "t", src=tree)
    finally:
        f.write_bytes(orig)
    S.check_stamp(good, "t", src=tree)
    added = tree / "rust_sim" / "src" / "zz_new_module.rs"
    added.write_text("// a source nobody built\n")
    try:
        with pytest.raises(S.StampMismatch, match="sources"):
            S.check_stamp(good, "t", src=tree)
    finally:
        added.unlink()
    S.check_stamp(good, "t", src=tree)


def test_schema_data_and_build_kind_are_refused(monkeypatch):
    monkeypatch.delenv(S.DATA_DIR_ENV, raising=False)
    h, n = S.source_hash()
    S.check_stamp(_stamp(h, n), "t")
    with pytest.raises(S.StampMismatch, match="column schema"):
        S.check_stamp(_stamp(h, n, schema="0" * 16), "t")
    with pytest.raises(S.StampMismatch, match="data"):
        S.check_stamp(_stamp(h, n, data="/elsewhere/data/pokemon"), "t")
    with pytest.raises(S.StampMismatch, match="build kind"):
        S.check_stamp(_stamp(h, n), "t", nan_poison=False)
    S.check_stamp(_stamp(h, n), "t", nan_poison=True)
    with pytest.raises(S.StampMismatch):
        S.check_stamp("garbage", "t")
    with pytest.raises(S.StampMismatch, match="version"):
        S.check_stamp(_stamp(h, n, stamp="v1"), "t")      # an older build script hashed other files: rebuild
    with pytest.raises(S.StampMismatch, match="kind"):
        S.check_stamp(_stamp(h, n, kind="port"), "t")     # a port driver's stamp where the env core's is demanded


def _fake_repo(root):
    """A directory shaped like a checkout as far as the Rust data-dir rule can tell."""
    (root / "data" / "pokemon").mkdir(parents=True)
    (root / "data" / "pokemon" / "gen3_species.json").write_text("{}")
    (root / "src" / "rust_env" / "target" / "selfcheck").mkdir(parents=True)
    return root / "src" / "rust_env" / "target" / "selfcheck" / "libpokesim_env.so"


def test_the_data_check_follows_the_RUN_TIME_dir_not_the_build_time_one(tmp_path, monkeypatch):
    """Since 805c63fc (`gen3_dex_runtime_data_dir_v1`) the Rust side READS `$POKESIM_DATA_DIR` -> the repo holding the
    module -> the compile-time dir. The stamp's `data=` is only that last default, so the check resolves the same rule.
    Fails on revert (the old `stamp.data != this checkout` check refuses the first case and accepts the second)."""
    monkeypatch.delenv(S.DATA_DIR_ENV, raising=False)
    h, n = S.source_hash()
    here_so = src_path("rust_env", "target", "selfcheck", "libpokesim_env.so")      # a module IN this checkout
    foreign_so = _fake_repo(tmp_path / "other_checkout")                             # a module in ANOTHER repo
    built_elsewhere = _stamp(h, n, data=str(tmp_path / "deleted_worktree" / "data" / "pokemon"))
    built_here = _stamp(h, n)
    # built in a (now gone) worktree, sitting in THIS checkout's target/: it reads this checkout's data -> accepted
    S.check_stamp(built_elsewhere, "t", module=here_so)
    # built HERE but sitting in another checkout's target/: it reads THAT data -> refused, naming the dir and its source
    with pytest.raises(S.StampMismatch, match=r"data: the build reads .*other_checkout.*\(module\)"):
        S.check_stamp(built_here, "t", module=foreign_so)
    # a module outside any checkout falls back to the compile-time dir: the stamp's, as before
    outside = tmp_path / "bin" / "libpokesim_env.so"
    S.check_stamp(built_here, "t", module=outside)
    with pytest.raises(S.StampMismatch, match=r"compile-time"):
        S.check_stamp(built_elsewhere, "t", module=outside)
    # no module named at all (a caller that cannot say): the compile-time dir, i.e. the old check
    with pytest.raises(S.StampMismatch, match="data"):
        S.check_stamp(built_elsewhere, "t")
    # `$POKESIM_DATA_DIR` is authoritative: another dir is refused, this checkout's own is accepted
    monkeypatch.setenv(S.DATA_DIR_ENV, str(tmp_path / "override"))
    with pytest.raises(S.StampMismatch, match=r"override.*\(env\)"):
        S.check_stamp(built_here, "t", module=here_so)
    monkeypatch.setenv(S.DATA_DIR_ENV, S.data_dir())
    S.check_stamp(built_elsewhere, "t", module=foreign_so)
    monkeypatch.setenv(S.DATA_DIR_ENV, "")                                           # empty counts as unset
    S.check_stamp(built_elsewhere, "t", module=here_so)


def test_runtime_data_dir_is_the_rust_rule(tmp_path):
    """`dex/data_dir.rs::resolve_data_dir` re-stated: env, then the first ancestor holding the marker, then the default."""
    so = _fake_repo(tmp_path / "repo")
    default = str(tmp_path / "compile" / "data" / "pokemon")
    want = str((tmp_path / "repo" / "data" / "pokemon").resolve())
    assert S.runtime_data_dir(default, so, {}) == (want, "module")
    assert S.runtime_data_dir(default, so, {S.DATA_DIR_ENV: "/x/pokemon"}) == ("/x/pokemon", "env")
    assert S.runtime_data_dir(default, so, {S.DATA_DIR_ENV: ""}) == (want, "module")
    link = tmp_path / "elsewhere" / "libpokesim_env.so"                              # /proc/self/maps shows the REAL path
    link.parent.mkdir()
    link.symlink_to(so)
    assert S.runtime_data_dir(default, link, {}) == (want, "module")
    assert S.runtime_data_dir(default, tmp_path / "bin" / "x", {}) == (str(Path(default).resolve()), "compile-time")
    assert S.runtime_data_dir(default, None, {}) == (str(Path(default).resolve()), "compile-time")


# ------------------------------------------------------------------ what is hashed: test-only files are not


def test_a_test_only_file_changed_still_passes_and_one_source_byte_refuses(tree):
    """THE FILE SET: the stamp covers what a NON-TEST build compiles. A `#[cfg(test)]` module file (`tests.rs`,
    `*_tests.rs`), a new one, `tests/`, docs and goldens are not in it, so editing them never stales a build; one byte of a
    real source does. Fails on revert (hash every `.rs`: the test-file edit then refuses)."""
    for scope in ("port", "env"):
        h, n = S.source_hash(tree, scope)
        good = make_stamp(scope, tree=tree)
        S.check_stamp(good, "t", src=tree, kind=scope)
        test_file = tree / "rust_sim" / "src" / "present" / "tests.rs"
        orig = test_file.read_bytes()
        extra = tree / "rust_sim" / "src" / "present" / "zz_new_tests.rs"
        try:
            test_file.write_bytes(orig + b"// an edited test\n")
            extra.write_text("// a new test-only module\n")
            (tree / "rust_sim" / "tests").mkdir(exist_ok=True)
            (tree / "rust_sim" / "tests" / "zz_it.rs").write_text("// an integration test\n")
            (tree / "rust_sim" / "NOTES.md").write_text("docs\n")
            assert S.source_hash(tree, scope) == (h, n), "test-only files, tests/ and docs are outside the hash"
            S.check_stamp(good, "t", src=tree, kind=scope)                              # still passes
            real = tree / "rust_sim" / "src" / "present" / "mod.rs"
            real_orig = real.read_bytes()
            real.write_bytes(real_orig + b" ")
            try:
                with pytest.raises(S.StampMismatch, match="STALE RUST BUILD"):
                    S.check_stamp(good, "t", src=tree, kind=scope)                      # one source byte refuses
            finally:
                real.write_bytes(real_orig)
        finally:
            test_file.write_bytes(orig)
            for f in (extra, tree / "rust_sim" / "tests" / "zz_it.rs", tree / "rust_sim" / "NOTES.md"):
                f.unlink(missing_ok=True)
        S.check_stamp(good, "t", src=tree, kind=scope)


def test_the_port_scope_ignores_the_env_crate_and_the_env_scope_does_not(tree):
    """A port driver is judged by the port's files alone: an env-crate edit must not stale `sim_bridge`; it does stale
    the env core (which links both)."""
    port_h, env_h = S.source_hash(tree, "port"), S.source_hash(tree, "env")
    f = tree / "rust_env" / "src" / "core" / "pool.rs"
    orig = f.read_bytes()
    try:
        f.write_bytes(orig + b"// env only\n")
        assert S.source_hash(tree, "port") == port_h
        assert S.source_hash(tree, "env") != env_h
    finally:
        f.write_bytes(orig)


def test_only_the_root_crates_lock_file_is_hashed(tree):
    """`Cargo.lock` is gitignored and cargo-written, and cargo writes the ROOT crate's only: the port's lock is in the
    port scope, not the env scope (a later port build would otherwise flip the env core's hash), and the env's is in
    the env scope. Fails on revert (hash every lock)."""
    port_lock, env_lock = tree / "rust_sim" / "Cargo.lock", tree / "rust_env" / "Cargo.lock"
    had = {p: p.read_bytes() if p.exists() else None for p in (port_lock, env_lock)}
    try:
        port_lock.write_text("# lock A\n")
        env_lock.write_text("# lock A\n")
        base = {k: S.source_hash(tree, k) for k in ("port", "env")}
        assert "port/Cargo.lock" in [r for r, _ in S.source_listing(tree, "port")]
        assert "port/Cargo.lock" not in [r for r, _ in S.source_listing(tree, "env")]
        assert "env/Cargo.lock" in [r for r, _ in S.source_listing(tree, "env")]
        port_lock.write_text("# lock B\n")
        assert S.source_hash(tree, "port") != base["port"] and S.source_hash(tree, "env") == base["env"]
        port_lock.write_text("# lock A\n")
        env_lock.write_text("# lock B\n")
        assert S.source_hash(tree, "env") != base["env"] and S.source_hash(tree, "port") == base["port"]
    finally:
        for p, b in had.items():
            p.unlink(missing_ok=True)
            if b is not None:
                p.write_bytes(b)


_CFG_TEST_MOD = re.compile(r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;")


def _cfg_test_module_files():
    """Every file a `#[cfg(test)] mod x;` declaration pulls in, as a crate-relative `/` path."""
    out = set()
    for crate in ("rust_sim", "rust_env"):
        root = src_path(crate)
        for f in sorted((root / "src").rglob("*.rs")):
            rel = f.relative_to(root)
            is_root = f.name in ("mod.rs", "lib.rs", "main.rs") or rel.parts[:2] == ("src", "bin") and len(rel.parts) == 3
            base = f.parent if is_root else f.parent / f.stem
            for name in _CFG_TEST_MOD.findall(f.read_text()):
                hits = [c for c in (base / f"{name}.rs", base / name / "mod.rs") if c.is_file()]
                assert len(hits) == 1, f"{rel}: `#[cfg(test)] mod {name};` resolves to {hits}"
                out.add((crate, hits[0].relative_to(root).as_posix()))
    return out


def test_the_test_only_rule_is_exactly_the_cfg_test_modules():
    """`is_test_only` is a NAME rule (Python and Rust twins). This pins it to the truth in both directions: every
    `#[cfg(test)] mod x;` file in the tree is excluded (else an edit to it demands a rebuild: the spurious-refusal
    direction), and no excluded file is real code (else an edit to compiled code would be invisible to the stamp: the
    silent-staleness direction)."""
    declared = _cfg_test_module_files()
    assert declared, "the tree has `#[cfg(test)] mod x;` declarations (the regex found none: the pin is vacuous)"
    for crate, rel in sorted(declared):
        assert S.is_test_only(rel), f"{crate}/{rel} is a `#[cfg(test)]` module but not named like one: call it tests.rs / *_tests.rs"
    named = set()
    for crate in ("rust_sim", "rust_env"):
        root = src_path(crate)
        named |= {(crate, f.relative_to(root).as_posix()) for f in (root / "src").rglob("*.rs")
                  if S.is_test_only(f.relative_to(root).as_posix())}
    assert named == declared, f"files named like tests but not `#[cfg(test)] mod`-declared: {sorted(named - declared)}"
    # the predicate's own table (the Rust twin asserts the same rows in `rust_sim/tests/build_stamp_test.rs`)
    for rel in ("src/present/tests.rs", "src/present/called_move_tests.rs", "src/x/foo_test.rs", "src/tests/a.rs",
                "tests/regression_test.rs"):
        assert S.is_test_only(rel), rel
    for rel in ("src/present/mod.rs", "src/latest.rs", "src/contest.rs", "src/dex/test_support.rs", "Cargo.toml", "build.rs"):
        assert not S.is_test_only(rel), rel


# ------------------------------------------------------------------ the refusal's words, and the opt-out


def test_the_refusal_is_one_line_naming_built_at_hash_source_hash_and_fix(tree):
    h, n = S.source_hash(tree, "port")
    stale = make_stamp("port", tree=tree, src="deadbeefdeadbeef", commit="abc1234")
    with pytest.raises(S.StampMismatch) as ei:
        S.check_stamp(stale, "sim_bridge", src=tree, kind="port", fix="cargo build --release --bin sim_bridge")
    msg = str(ei.value)
    assert "\n" not in msg, "ONE line"
    assert "src=deadbeefdeadbeef" in msg and "abc1234" in msg, "the built-at hash and commit"
    assert f"src={h}" in msg, "this checkout's hash"
    assert "rebuild it: cargo build --release --bin sim_bridge" in msg, "the fix"
    assert S.ALLOW_STALE_ENV in msg, "the opt-out is named"


def test_the_opt_out_passes_with_a_loud_warning_and_waives_only_staleness(tree, monkeypatch, capsys):
    stale = make_stamp("port", tree=tree, src="ffffffffffffffff")
    monkeypatch.delenv(S.ALLOW_STALE_ENV, raising=False)
    with pytest.raises(S.StampMismatch):
        S.check_stamp(stale, "stale-one", src=tree, kind="port")
    for off in ("0", "true", "yes", ""):                         # only the exact value 1 is the opt-out
        monkeypatch.setenv(S.ALLOW_STALE_ENV, off)
        with pytest.raises(S.StampMismatch):
            S.check_stamp(stale, "stale-one", src=tree, kind="port")
    monkeypatch.setenv(S.ALLOW_STALE_ENV, "1")
    capsys.readouterr()
    st = S.check_stamp(stale, "stale-one", src=tree, kind="port")
    assert st["src"] == "ffffffffffffffff"
    err = capsys.readouterr().err
    assert "STALE RUST BUILD ALLOWED" in err and "stale-one" in err and "NO commit pins" in err, err
    S.check_stamp(stale, "stale-one", src=tree, kind="port")      # once per artifact and process
    assert capsys.readouterr().err == ""
    # the opt-out waives STALENESS only: a foreign column schema or build kind still refuses
    bad_schema = _stamp("ffffffffffffffff", 1, schema="0" * 16)
    with pytest.raises(S.StampMismatch, match="column schema"):
        S.check_stamp(bad_schema, "t", src=tree)
    h2, n2 = S.source_hash(tree)
    with pytest.raises(S.StampMismatch, match="build kind"):
        S.check_stamp(_stamp(h2, n2), "t", src=tree, nan_poison=False)


# ------------------------------------------------------------------ binaries invoked from Python


def test_a_binary_is_judged_by_its_build_stamp_before_it_is_spawned(tree, tmp_path, monkeypatch, capsys):
    """`check_binary_stamp`: matching passes; a stale stamp refuses (one line: both hashes + fix); a binary that reports no
    stamp (it predates the flag) refuses; a rebuilt binary is re-asked; the opt-out passes with the loud warning."""
    monkeypatch.delenv(S.ALLOW_STALE_ENV, raising=False)
    good = write_fake_binary(tmp_path / "good", make_stamp("port", tree=tree))
    st = S.check_binary_stamp(good, "sim_bridge", kind="port", src=tree)
    assert st["kind"] == "port"
    stale = write_fake_binary(tmp_path / "stale", make_stamp("port", tree=tree, src="0123456789abcdef"))
    with pytest.raises(S.StampMismatch, match=r"STALE RUST BUILD — .*src=0123456789abcdef .*rebuild it: ") as ei:
        S.check_binary_stamp(stale, "sim_bridge", kind="port", src=tree, fix="cargo build --release --bin sim_bridge")
    assert "\n" not in str(ei.value) and str(stale) in str(ei.value)
    silent = write_fake_binary(tmp_path / "silent", None)
    with pytest.raises(S.StampMismatch, match="reports no build stamp"):
        S.check_binary_stamp(silent, "sim_bridge", kind="port", src=tree)
    not_a_binary = tmp_path / "plain.txt"
    not_a_binary.write_text("not executable\n")
    with pytest.raises(S.StampMismatch, match="reports no build stamp"):
        S.check_binary_stamp(not_a_binary, "sim_bridge", kind="port", src=tree)
    # an env-kind stamp is not a port driver's
    with pytest.raises(S.StampMismatch, match="kind"):
        S.check_binary_stamp(write_fake_binary(tmp_path / "envk", make_stamp("env", tree=tree)), "sim_bridge", kind="port", src=tree)
    # a rebuild at the same path is re-asked (cached per binary identity: path, mtime, size)
    box = write_fake_binary(tmp_path / "box", make_stamp("port", tree=tree, src="1111111111111111"))
    with pytest.raises(S.StampMismatch):
        S.check_binary_stamp(box, "sim_bridge", kind="port", src=tree)
    write_fake_binary(box, make_stamp("port", tree=tree))
    os.utime(box, ns=(1, 2))                                      # a distinct mtime, whatever the clock's tick
    S.check_binary_stamp(box, "sim_bridge", kind="port", src=tree)
    # the opt-out: stale and silent binaries run, loudly
    monkeypatch.setenv(S.ALLOW_STALE_ENV, "1")
    capsys.readouterr()
    assert S.check_binary_stamp(stale, "sim_bridge", kind="port", src=tree)["src"] == "0123456789abcdef"
    assert S.check_binary_stamp(silent, "sim_bridge", kind="port", src=tree) is None
    assert capsys.readouterr().err.count("STALE RUST BUILD ALLOWED") == 2


def test_a_binary_that_does_not_answer_in_time_is_unverified_never_stale(tree, tmp_path, monkeypatch):
    """A timeout is never a semantic outcome: a `--build-stamp` that times out raises `StampUnavailable(timed_out=True)`
    (not the staleness message), is not cached (the next check asks again), and the opt-out skips it loudly."""
    good = write_fake_binary(tmp_path / "good", make_stamp("port", tree=tree))
    monkeypatch.delenv(S.ALLOW_STALE_ENV, raising=False)

    def hangs(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    with pytest.raises(S.StampUnavailable) as ei:
        S.check_binary_stamp(good, "sim_bridge", kind="port", src=tree, run=hangs)
    assert ei.value.timed_out and "STALE RUST BUILD" not in str(ei.value) and "not a verdict" in str(ei.value)
    S.check_binary_stamp(good, "sim_bridge", kind="port", src=tree)                  # the next ask is answered: nothing was cached
    assert S.check_binary_stamp(tmp_path / "good", "sim_bridge", kind="port", src=tree, allow_stale=True, run=hangs) is not None


def test_a_binary_is_exec_d_once_per_identity(tree, tmp_path):
    good = write_fake_binary(tmp_path / "good", make_stamp("port", tree=tree))
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.run(argv, **kw)

    for _ in range(3):
        S.check_binary_stamp(good, "sim_bridge", kind="port", src=tree, run=run)
    assert calls == [[str(good), "--build-stamp"]], "one exec per binary identity, not per check"


# ------------------------------------------------------------------ Rust and Python agree (rustc on a copy)


def _rustc() -> str:
    rustc = shutil.which("rustc") or os.path.expanduser("~/.cargo/bin/rustc")
    if not os.path.exists(rustc):
        pytest.fail("rustc is not installed — the build stamp's cross-language gate cannot run (install rustup)")
    return rustc


def _run_build_script(rustc: str, crate_dir: Path, out: Path, *, selfcheck: bool) -> str:
    """Compile `<crate_dir>/build.rs` and run it as cargo does (CARGO_MANIFEST_DIR); return the stamp it emits."""
    exe = out / f"build_{crate_dir.name}"
    if not exe.exists():                                       # the script itself never changes within a test: compile once
        r = subprocess.run([rustc, "--edition", "2021", str(crate_dir / "build.rs"), "-o", str(exe)],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CARGO_")}
    env["CARGO_MANIFEST_DIR"] = str(crate_dir)
    if selfcheck:
        env["CARGO_FEATURE_EMISSION_SELFCHECK"] = "1"
    r = subprocess.run([str(exe)], capture_output=True, text=True, env=env, cwd=str(out), timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    lines = [ln.split("=", 2)[2] for ln in r.stdout.splitlines() if ln.startswith("cargo:rustc-env=POKESIM_")]
    assert len(lines) == 1, r.stdout[-2000:]
    return lines[0]


@pytest.mark.integration
def test_the_rust_build_scripts_and_this_module_hash_the_same_files(tmp_path):
    """THE CROSS-LANGUAGE GATE, without a cargo build: compile each `build.rs` with rustc, run it over a copy of the real
    tree, and demand the stamp it prints is the one `check_stamp` accepts for that copy — the same `src`, the same file
    count, test-only files and all. Then edit the copy and demand Rust and Python agree on the new answer too. Fails
    on any drift between `build_support.rs` and `stamp.py` (the file set, the test-only rule, the blob ids, the FNV)."""
    rustc = _rustc()
    src = _copy_tree(tmp_path / "copy")
    port = _run_build_script(rustc, src / "rust_sim", tmp_path, selfcheck=True)
    envs = _run_build_script(rustc, src / "rust_env", tmp_path, selfcheck=True)
    p, e = S.parse_stamp(port), S.parse_stamp(envs)
    assert (p["kind"], e["kind"]) == ("port", "env") and p["selfcheck"] == "1" and e["nan_poison"] == "1"
    assert (p["src"], int(p["nfiles"])) == S.source_hash(src, "port")
    assert (e["src"], int(e["nfiles"])) == S.source_hash(src, "env")
    S.check_stamp(port, "port build.rs", src=src, kind="port")
    S.check_stamp(envs, "env build.rs", src=src, kind="env", nan_poison=True)
    # release shape: no selfcheck
    assert S.parse_stamp(_run_build_script(rustc, src / "rust_sim", tmp_path, selfcheck=False))["selfcheck"] == "0"
    # one source byte: both languages move together; a test-only edit: neither moves
    (src / "rust_sim" / "src" / "present" / "tests.rs").write_text("// a test-only edit\n")
    assert _run_build_script(rustc, src / "rust_sim", tmp_path, selfcheck=True) == port
    victim = src / "rust_sim" / "src" / "turn.rs"
    victim.write_bytes(victim.read_bytes() + b"\n")
    moved = _run_build_script(rustc, src / "rust_sim", tmp_path, selfcheck=True)
    assert moved != port
    S.check_stamp(moved, "port build.rs (edited)", src=src, kind="port")
    with pytest.raises(S.StampMismatch):
        S.check_stamp(port, "port build.rs (before the edit)", src=src, kind="port")


# ------------------------------------------------------------------ pinned runs


def test_a_pinned_checkout_is_judged_against_its_own_sources_never_the_importing_trees(tmp_path):
    """THE PINNED-RUN PROPERTY. A pinned trainer imports `utils` from its pin's worktree (the launcher puts `<pin>/src` first on
    PYTHONPATH) and loads the library built in THAT worktree. `stamp` hashes `utils.paths.src_path()` — the tree the module was
    imported from — so the pin's build is judged against the pin's sources and cannot be refused for being unlike main's; and
    main's (newer or older) build, handed to the pin, IS refused. Proven with a stand-in pin in a subprocess: a copy of this
    checkout's python `utils` and Rust sources with one Rust byte different."""
    import json
    import sys

    pin = tmp_path / "pin"
    src = _copy_tree(pin)                                                         # pin/src/{rust_sim,rust_env}
    shutil.copytree(src_path("utils"), src / "utils",
                    ignore=shutil.ignore_patterns("__pycache__", "*_test.py", "*.json", "*.png", "*.csv"))
    victim = src / "rust_sim" / "src" / "turn.rs"
    victim.write_bytes(victim.read_bytes() + b"\n// a commit main does not have\n")
    main_stamp = make_stamp("port")                                               # a build of THIS (main-like) tree
    pin_stamp = make_stamp("port", tree=src)                                      # a build of the PIN's tree
    code = (
        "import json\n"
        "from utils.paths import src_path\n"
        "from utils.rust_env import stamp as S\n"
        "def refused(stamp):\n"
        "    try:\n"
        "        S.check_stamp(stamp, 'lib', kind='port')\n"
        "        return False\n"
        "    except S.StampMismatch:\n"
        "        return True\n"
        "import sys\n"
        "print(json.dumps({'src': str(src_path()), 'hash': S.source_hash(scope='port'),\n"
        "                  'pin_build_refused': refused(sys.argv[1]), 'main_build_refused': refused(sys.argv[2])}))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", S.ALLOW_STALE_ENV)}
    env.update(PYTHONPATH=str(src), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", code, pin_stamp, main_stamp], capture_output=True, text=True, env=env,
                       cwd=str(tmp_path), timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["src"] == str(src.resolve()), "the child imported the PIN's utils"
    assert tuple(got["hash"]) == S.source_hash(src, "port") != S.source_hash(scope="port")
    assert got["pin_build_refused"] is False, "a correct pinned build must never be refused"
    assert got["main_build_refused"] is True, "main's build handed to the pin is refused"
