"""The guarded remove (fsguard.py) and the static scan that every arm-chain module deletes only through it (CB-28)."""
import ast
import os

import pytest

from main.ops.arm_chain import fsguard

HERE = os.path.dirname(os.path.abspath(__file__))
#: Removal calls by attribute name: os.remove / os.unlink / os.rmdir / os.removedirs, shutil.rmtree, Path.unlink /
#: Path.rmdir (and list.remove, which this module never needs).
_REMOVERS = {"remove", "unlink", "rmdir", "removedirs", "rmtree"}


def _production_modules():
    for n in sorted(os.listdir(HERE)):
        if n.endswith(".py") and not n.endswith("_test.py") and n not in ("testing_fakes.py", "fsguard.py"):
            yield n


def _violations(src: str):
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name in _REMOVERS:
                out.append(f"line {node.lineno}: {name}()")
        if isinstance(node, (ast.List, ast.Tuple)):       # an argv such as ["rm", "-rf", p] or ["git", "rm", p]
            for e in node.elts:
                if isinstance(e, ast.Constant) and e.value in ("rm", "rmdir"):
                    out.append(f"line {node.lineno}: an argv carrying {e.value!r}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                node.value.startswith("rm ") or " rm -" in node.value or ";rm " in node.value):
            out.append(f"line {node.lineno}: a shell string running rm")
    return out


def test_the_module_never_removes_a_variable_path_outside_the_guarded_helper():
    """CB-28 (operator review 10-27): no ``rm`` / ``shutil.rmtree`` / ``os.remove`` on a variable path anywhere in
    the module except :func:`fsguard.remove_under`, which refuses a path not strictly inside its root."""
    bad = {n: v for n in _production_modules() if (v := _violations(open(os.path.join(HERE, n)).read()))}
    assert bad == {}


def test_the_scan_catches_each_removal_shape():
    for src in ["import shutil\nshutil.rmtree(p)\n", "import os\nos.remove(p)\n", "os.unlink(p)\n", "p.unlink()\n",
                "run(['rm', '-rf', p])\n", "run(['git', 'rm', p])\n", "x = 'rm -rf ' + p\n", "os.rmdir(p)\n"]:
        assert _violations(src), src
    assert _violations("from main.ops.arm_chain.fsguard import remove_under\nremove_under(p, root)\n") == []


def test_remove_under_removes_only_strictly_inside_its_root(tmp_path):
    root = tmp_path / "work"
    (root / "scratch" / "deep").mkdir(parents=True)
    (root / "f.txt").write_text("x")
    outside = tmp_path / "models"
    outside.mkdir()
    (outside / "run").mkdir()
    fsguard.remove_under(str(root / "f.txt"), str(root))
    assert not (root / "f.txt").exists()
    fsguard.remove_under(str(root / "scratch"), str(root), tree=True)
    assert not (root / "scratch").exists()
    fsguard.remove_under(str(root / "absent"), str(root))          # a no-op
    for bad, kw in [(str(outside / "run"), {"tree": True}), (str(root / ".." / "models" / "run"), {"tree": True}),
                    (str(root), {"tree": True}), (str(root) + "_sibling", {"tree": True})]:
        if bad.endswith("_sibling"):
            os.makedirs(bad)
        with pytest.raises(fsguard.UnsafeRemove):
            fsguard.remove_under(bad, str(root), **kw)
    assert (outside / "run").is_dir() and root.is_dir()


def test_remove_under_never_follows_a_symlink_out_of_its_root(tmp_path):
    root, target = tmp_path / "work", tmp_path / "models" / "run"
    root.mkdir()
    target.mkdir(parents=True)
    (target / "ckpt.zip").write_text("keep")
    (root / "link").symlink_to(target)
    with pytest.raises(fsguard.UnsafeRemove, match="symlink"):
        fsguard.remove_under(str(root / "link"), str(root), tree=True)
    fsguard.remove_under(str(root / "link"), str(root))             # the LINK itself is inside: removed, not its target
    assert not (root / "link").exists() and (target / "ckpt.zip").read_text() == "keep"
    (root / "dirlink_parent").symlink_to(tmp_path / "models")
    with pytest.raises(fsguard.UnsafeRemove, match="not strictly inside"):
        fsguard.remove_under(str(root / "dirlink_parent" / "run"), str(root), tree=True)
    assert (target / "ckpt.zip").exists()


@pytest.mark.parametrize("root", ["relative/dir", "/", "~"])
def test_remove_under_refuses_a_root_that_is_not_a_safe_directory(tmp_path, root):
    with pytest.raises(fsguard.UnsafeRemove):
        fsguard.remove_under(str(tmp_path / "x"), os.path.expanduser(root))
