"""``main.reads.playpath``: the tracked files a Kakuna unit's play reads, at a COMMIT, and their digest. A miniature git
repo shaped like the real layout (``mini_repo``) carries each rule; the real repo's HEAD pins that scheduling code is
off the path (what lets a scheduling fix reach a running read)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import pytest

from main.reads import playpath as PP

BASE: Dict[str, str] = {
    "src/main/__init__.py": "",
    "src/main/anchors/__init__.py": "",
    "src/main/anchors/__main__.py": "from utils.play_core import go\n",
    "src/main/anchors/peer_scripts/side.py": "PEER = 1\n",
    "src/main/anchors/cli_test.py": "TEST = 1\n",
    "src/main/x35/__init__.py": "",
    "src/main/x35/kakuna.py": "def lazy():\n    from utils.late import x\n    return x\n",
    "src/main/reads/__init__.py": "",
    "src/main/reads/runner.py": "SCHED = 1\n",
    "src/main/reads/anchor_pool.py": "POOL = 1\n",
    "src/utils/__init__.py": "",
    "src/utils/play_core.py": "from . import sibling\n\ndef go():\n    return sibling.S\n",
    "src/utils/sibling.py": "S = 1\n",
    "src/utils/late.py": "x = 1\n",
    "src/utils/mem_cap.py": "",
    "src/utils/heavy_slots.py": "",
    "src/utils/unrelated.py": "U = 1\n",
    "src/utils/bridge/__init__.py": "",
    "src/utils/bridge/ws_frontend.py": "",
    "src/rust_sim/Cargo.toml": "[package]\n",
    "src/rust_sim/src/bridge.rs": "// bridge\n",
    "src/rust_sim/tests/b_test.rs": "// test\n",
    "data/teams/a.txt": "team a\n",
    "scripts/ops/mem_cap.sh": "#!/bin/sh\n",
    "designs/tools/reads.md": "# reads\n",
    "designs/ops/anchors.json": "{}\n",
}


def git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env)
    return r.stdout.strip()


def write(repo: Path, files: Dict[str, Optional[str]]) -> None:
    for rel, text in files.items():
        p = repo / rel
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)


def commit_on(repo: Path, base: str, files: Dict[str, Optional[str]], *, chmod_x: Optional[str] = None) -> str:
    """A commit with ``files`` changed (None deletes) on top of ``base`` (the working tree is reset to ``base`` first)."""
    git(repo, "checkout", "-q", "--detach", base)
    write(repo, files)
    if chmod_x:
        (repo / chmod_x).chmod(0o755)
    git(repo, "add", "-A")
    git(repo, "-c", "core.fileMode=true", "commit", "-q", "-m", "x", "--allow-empty")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def mini_repo(tmp_path) -> Path:
    repo = tmp_path / "mini"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.fileMode", "true")
    write(repo, dict(BASE))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def base_of(repo: Path) -> str:
    return git(repo, "rev-list", "--max-parents=0", "HEAD")


def test_the_play_path_is_the_closure_plus_the_named_files_and_never_tests_or_docs(mini_repo):
    base = base_of(mini_repo)
    got = set(PP.play_files(base, mini_repo))
    assert got == {
        "src/main/__init__.py", "src/main/anchors/__init__.py", "src/main/anchors/__main__.py",
        "src/main/anchors/peer_scripts/side.py",                       # the anchors package, even unimported
        "src/main/x35/__init__.py", "src/main/x35/kakuna.py",
        "src/utils/__init__.py", "src/utils/play_core.py", "src/utils/sibling.py",       # a relative import
        "src/utils/late.py",                                           # a LAZY import inside a function
        "src/utils/mem_cap.py", "src/utils/heavy_slots.py", "src/utils/bridge/__init__.py",
        "src/utils/bridge/ws_frontend.py", "src/rust_sim/Cargo.toml", "src/rust_sim/src/bridge.rs",
        "data/teams/a.txt", "scripts/ops/mem_cap.sh", "designs/ops/anchors.json"}
    assert "src/main/anchors/cli_test.py" not in got and "src/rust_sim/tests/b_test.rs" not in got
    assert "designs/tools/reads.md" not in got
    assert not {"src/main/reads/runner.py", "src/main/reads/anchor_pool.py", "src/utils/unrelated.py"} & got


def test_a_change_off_the_path_leaves_the_digest_and_each_change_on_it_moves_it(mini_repo):
    base = base_of(mini_repo)
    d0 = PP.digest(PP.play_files(base, mini_repo))
    off = commit_on(mini_repo, base, {"src/main/reads/runner.py": "SCHED = 2\n", "designs/tools/reads.md": "# new\n",
                                      "src/main/anchors/cli_test.py": "TEST = 2\n", "src/utils/unrelated.py": "U = 2\n",
                                      "src/main/reads/new_module.py": "N = 1\n"})
    assert PP.digest(PP.play_files(off, mini_repo)) == d0
    for files in ({"src/rust_sim/src/bridge.rs": "// changed\n"}, {"data/teams/a.txt": "team b\n"},
                  {"src/utils/late.py": "x = 2\n"}, {"src/utils/sibling.py": "S = 2\n"},
                  {"scripts/ops/mem_cap.sh": "#!/bin/sh\n# x\n"}, {"src/main/anchors/peer_scripts/side.py": "PEER = 2\n"},
                  {"data/teams/b.txt": "new team\n"}, {"data/teams/a.txt": None}):
        on = commit_on(mini_repo, base, files)
        assert PP.digest(PP.play_files(on, mini_repo)) != d0, files
    mode = commit_on(mini_repo, base, {}, chmod_x="scripts/ops/mem_cap.sh")
    assert PP.digest(PP.play_files(mode, mini_repo)) != d0, "a mode change is a change"


def test_a_new_import_from_the_path_widens_it_and_so_differs(mini_repo):
    """kakuna newly importing a module that was off the path: that module is now on it, and the two paths differ."""
    base = base_of(mini_repo)
    c = commit_on(mini_repo, base, {"src/main/x35/kakuna.py": "from utils.unrelated import U\n"})
    diffs = PP.differences(PP.play_files(base, mini_repo), PP.play_files(c, mini_repo))
    assert "added src/utils/unrelated.py" in diffs and "changed src/main/x35/kakuna.py" in diffs
    assert "removed src/utils/late.py" in diffs, "and the module it stopped importing left the path"


def test_differences_name_changed_added_and_removed_paths():
    a = {"p": ("100644", "1"), "q": ("100644", "2"), "r": ("100644", "3")}
    b = {"p": ("100644", "1"), "q": ("100755", "2"), "s": ("100644", "4")}
    assert PP.differences(a, b) == ["changed q", "removed r", "added s"]
    assert PP.differences(a, a) == []
    assert PP.digest(a) == PP.digest(dict(reversed(list(a.items())))), "the order of the dict is not the digest"
    assert PP.digest(a) != PP.digest(b)


def test_relative_imports_resolve_against_the_files_own_package():
    src = b"from . import a\nfrom .b import c\nfrom .. import d\nfrom ..e import f\nimport g.h\n"
    assert PP._imports("src/p/q/mod.py", src) >= {"p.q", "p.q.a", "p.q.b", "p.q.b.c", "p", "p.d", "p.e", "p.e.f", "g.h"}
    assert PP._imports("src/p/q/__init__.py", b"from . import a\n") >= {"p.q.a"}, "an __init__ is its own package"
    assert PP._imports("src/p/bad.py", b"def (:\n") == set(), "a file that does not parse names no import"


def test_a_commit_that_is_not_a_gen3ai_tree_or_not_a_commit_is_a_named_error(mini_repo):
    empty = commit_on(mini_repo, base_of(mini_repo), {"src/main/x35/kakuna.py": None})
    with pytest.raises(PP.PlayPathError, match="not a gen3ai commit"):
        PP.play_files(empty, mini_repo)
    for bad in ("nosuchref", "", "-x"):
        with pytest.raises(PP.PlayPathError):
            PP.resolve(bad, mini_repo)


# ------------------------------------------------------------------------- the real repo
def test_the_real_play_path_holds_the_named_files_and_no_scheduling_code():
    """What lets a scheduling fix reach a running read: the supervisor's verbs are OFF the path (``supervisors.py`` and
    the package ``__init__`` are the two ``main.reads`` files ``main.x35.kakuna`` reaches, and stay on it)."""
    try:
        head = PP.resolve("HEAD")
    except PP.PlayPathError:
        pytest.skip("not a git checkout")
    got = PP.play_files(head)
    for must in ("src/main/x35/kakuna.py", "src/main/anchors/__main__.py", "scripts/ops/mem_cap.sh",
                 "src/utils/heavy_slots.py", "src/utils/bridge/ws_frontend.py", "designs/ops/anchors.json"):
        assert must in got, must
    assert any(p.startswith("src/rust_sim/src/") for p in got) and any(p.startswith("data/") for p in got)
    on = sorted(p for p in got if p.startswith("src/main/reads/"))
    assert on == ["src/main/reads/__init__.py", "src/main/reads/supervisors.py"], on
    assert not any(p.endswith(("_test.py", ".md")) or p.endswith("conftest.py") for p in got)


def test_the_real_play_path_holds_every_rust_source_the_engine_stamp_hashes():
    """REVIEW F F-M2: a repin is sound only if an identical play path is an identical ENGINE. Every tracked file the
    build stamp hashes for the ``sim_bridge`` (scope ``port``) or the env core (scope ``env``) is on the path; the
    untracked ``Cargo.lock`` git cannot hold (``main.reads.engine.LEGACY_LOCK_BLOBS``)."""
    from utils.rust_env import stamp as ST

    try:
        head = PP.resolve("HEAD")
    except PP.PlayPathError:
        pytest.skip("not a git checkout")
    got = PP.play_files(head)
    tracked = set(PP.tree(head))
    for scope in ("port", "env"):
        for tag, path in ST.source_listing(scope=scope):
            rel = str(path.relative_to(ST.src_path().parent))
            if rel not in tracked:
                assert rel.endswith("Cargo.lock"), f"{rel}: untracked yet in the {scope} stamp"
                continue
            assert rel in got, f"{rel} ({tag}) is hashed by the {scope} engine stamp but is off the play path"
