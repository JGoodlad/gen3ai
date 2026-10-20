"""Gate ⑤ — the build stamp's TEETH (M5 Lane 0).

The stamp refuses a build whose sources, column schema, data directory or build kind are not this
tree's. Proven on a temporary COPY of the sources that reach the build (never the checkout itself):
the copy hashes like the tree, a one-byte edit or an added source flips the verdict to a refusal,
and restoring the file restores it. The cross-language agreement (the Rust ``build.rs`` hash == this
module's) is proven on a real build by ``core_cargo_test.py``.
"""
import shutil
from pathlib import Path

import pytest

from utils.paths import src_path
from utils.rust_env import columns
from utils.rust_env import stamp as S


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("stamp_src")
    for crate, extra in (("rust_sim", ()), ("rust_env", ("build.rs",))):
        shutil.copytree(src_path(crate, "src"), root / crate / "src", ignore=shutil.ignore_patterns("target"))
        for f in ("Cargo.toml",) + extra:
            shutil.copy2(src_path(crate, f), root / crate / f)
    return root


def _stamp(src_hash, n, **kw):
    fields = dict(stamp="v1", commit="c0ffee", src=src_hash, nfiles=str(n), nan_poison="1",
                  schema=columns.schema_id(), data=S.data_dir())
    fields.update(kw)
    return ";".join(f"{k}={v}" for k, v in fields.items())


def test_a_copy_hashes_like_the_tree(tree):
    assert S.source_hash(tree) == S.source_hash()
    rels = [r for r, _ in S.source_listing()]
    assert "env/build.rs" in rels and "port/Cargo.toml" in rels and "env/src/core/columns.rs" in rels
    assert sum(r.startswith("port/src/") for r in rels) > 50, "the port's sources are in the listing"


def test_the_stamp_has_teeth(tree):
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
    with pytest.raises(S.StampMismatch):
        S.check_stamp(_stamp(h, n, stamp="v0"), "t")


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
