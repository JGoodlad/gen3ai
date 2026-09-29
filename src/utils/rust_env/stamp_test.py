"""Gate ⑤ — the build stamp's TEETH (M5 Lane 0).

The stamp refuses a build whose sources, column schema, data directory or build kind are not this
tree's. Proven on a temporary COPY of the sources that reach the build (never the checkout itself):
the copy hashes like the tree, a one-byte edit or an added source flips the verdict to a refusal,
and restoring the file restores it. The cross-language agreement (the Rust ``build.rs`` hash == this
module's) is proven on a real build by ``core_cargo_test.py``.
"""
import shutil

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


def test_schema_data_and_build_kind_are_refused():
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
