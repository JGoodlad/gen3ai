"""THE ANCHOR LANE'S PLAY PATH, as git sees it (``designs/tools/reads.md`` §3, ``repin``).

A Kakuna unit's play is ``python -m main.anchors`` run by ``main.x35.kakuna.run_unit`` under ``mem_cap.sh``, over a
websocket front end backed by the Rust ``sim_bridge``. This module names the TRACKED files that decide what such a unit
plays, at any COMMIT (read from git objects, never the working tree), and digests them. Two commits whose play-path
digests are equal play the same units: whatever differs between them (scheduling, the supervisor's verbs, tests, docs)
cannot change a game. ``main.reads.repin`` uses that to let a running read adopt a newer checkout.

THE PLAY PATH (a deliberate over-approximation: a false "differs" refuses, a false "same" would be silent):

* the static IMPORT CLOSURE of :data:`ENTRY_MODULES` (every ``import`` statement anywhere in a module, lazy ones and
  ``TYPE_CHECKING`` ones included, resolved against the commit's own tree);
* the whole of :data:`PREFIXES` (the anchors package and its peer scripts, the front end, the model, inference,
  observation and data-facade packages, the Rust sources that build ``sim_bridge``, ``data/``, the submodule pointers);
* the exact files of :data:`EXACT` (``mem_cap.sh``, the registry ``anchors.json``, the committed cell configs and
  manifest, the panel registry, the Rust build inputs, the env file).

Tests (``*_test.py``, ``conftest.py``) and prose (``*.md``) are never on it. NOT covered, because git does not hold
them: the Metamon cache and weights, the panel team dirs outside the repo, the model copies, the toolchain. Each unit
verifies those itself at play (the preflight's team digests, the weights pins, the model's sha256).

``main/reads/runner.py``, ``api.py``, ``cli.py``, ``repin.py``, ``anchor_pool.py``, ``lanectl.py``, ``speccheck.py`` and this
module are NOT in the closure (only ``main/reads/__init__.py`` and ``supervisors.py`` are), which is what lets a scheduling
change reach a running read; ``playpath_test`` pins that.
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

SCHEMA = "gen3_read_playpath_v1"
#: The modules a unit (or its supervisor-side launcher) runs; the closure starts here.
ENTRY_MODULES: Tuple[str, ...] = ("main.anchors.__main__", "main.anchors", "main.x35.kakuna", "utils.bridge.ws_frontend",
                                  "utils.mem_cap", "utils.heavy_slots")
#: Every tracked file under these is on the path (a file name is a prefix too).
PREFIXES: Tuple[str, ...] = (
    "src/main/anchors/", "src/utils/bridge/", "src/utils/rust_env/", "src/agents/model/", "src/agents/inference/",
    "src/agents/observation/", "src/agents/gen3_data/", "src/rust_sim/src/", "src/rust_env/src/", "data/", "deps/",
    "designs/research_state/measurements/n0_endofrun_2026-09-27/teams_home72/",
    "designs/research_state/measurements/x37a_eval_matrix_2026-10-18/configs/",
    "designs/research_state/measurements/hctx_rc72_2026-10-26/teams_rc72/",
    "designs/research_state/measurements/hctx_rc72_2026-10-26/teams_rc72b/",
)
EXACT: Tuple[str, ...] = (
    "src/main/x35/kakuna.py", "scripts/ops/mem_cap.sh", "scripts/ops/_common.sh", "environment_torch28.yml",
    ".gitmodules", "src/rust_sim/Cargo.toml", "src/rust_sim/Cargo.lock", "src/rust_sim/build.rs",
    "src/rust_sim/build_support.rs", "src/rust_env/Cargo.toml", "src/rust_env/build.rs", "designs/ops/anchors.json",
    "designs/research_state/heldout_panels.json", "designs/research_state/diagnostic_panels.json",
    "designs/research_state/measurements/hctx_rc72_2026-10-26/anchors_rc72_kakuna.json",
    "designs/research_state/measurements/hctx_rc72_2026-10-26/rc72_manifest.json",
    "designs/research_state/measurements/hctx_rc72_2026-10-26/anchors_rc72b_kakuna.json",
    "designs/research_state/measurements/hctx_rc72_2026-10-26/rc72b_manifest.json",
    "designs/research_state/measurements/x35_breakers_stage_2026-10-17/anchors_home72_kakuna.json",
    "designs/research_state/measurements/x35_breakers_stage_2026-10-17/home72_manifest.json",
    "designs/research_state/measurements/heldout72_hpfix_kakuna_2026-10-18/anchors_heldout72_hpfix_kakuna.json",
)
#: Never on the path: tests and prose.
EXCLUDED_SUFFIXES: Tuple[str, ...] = ("_test.py", ".md")
EXCLUDED_NAMES: Tuple[str, ...] = ("conftest.py",)

Entry = Tuple[str, str]                     # (git mode, blob sha)


class PlayPathError(RuntimeError):
    """The play path of a commit cannot be established (named)."""


def _git(repo: Path, *args: str, stdin: Optional[bytes] = None) -> bytes:
    r = subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, check=False)
    if r.returncode != 0:
        raise PlayPathError(f"git {' '.join(args[:3])}: {r.stderr.decode(errors='replace').strip()[:400]}")
    return r.stdout


def default_repo() -> Path:
    from utils.paths import repo_root

    return Path(repo_root())


def resolve(ref: str, repo: Optional[Path] = None) -> str:
    """The full commit sha ``ref`` names in this repo (any commit-ish: a sha, a tag, a branch)."""
    if not ref or ref.startswith("-"):
        raise PlayPathError(f"{ref!r} is not a commit")
    from utils import git_sha

    # a spec's `pin` and a state's `code.json` commit are RECORDED shas: the history-rewrite map first, git second
    full = git_sha.commit_of(ref, repo=str(repo or default_repo()))
    if not full:
        raise PlayPathError(f"{ref!r} names no commit in {repo or default_repo()}")
    return full


def tree(commit: str, repo: Optional[Path] = None) -> Dict[str, Entry]:
    """``{path: (mode, blob sha)}`` of every tracked file at ``commit`` (submodules are their gitlink commit)."""
    out = _git(repo or default_repo(), "ls-tree", "-r", "-z", "--full-tree", commit)
    files: Dict[str, Entry] = {}
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        mode, _typ, sha = meta.decode().split(" ")
        files[path.decode()] = (mode, sha)
    return files


def read_blobs(shas: Iterable[str], repo: Optional[Path] = None) -> Dict[str, bytes]:
    """``{sha: content}`` of blobs, in one ``git cat-file --batch``."""
    uniq = sorted(set(shas))
    if not uniq:
        return {}
    buf = _git(repo or default_repo(), "cat-file", "--batch", stdin="".join(f"{s}\n" for s in uniq).encode())
    out: Dict[str, bytes] = {}
    i = 0
    for s in uniq:
        j = buf.index(b"\n", i)
        head = buf[i:j].split()
        if len(head) != 3 or head[1] != b"blob":
            raise PlayPathError(f"blob {s}: {buf[i:j].decode(errors='replace')}")
        size = int(head[2])
        out[s] = buf[j + 1:j + 1 + size]
        i = j + 1 + size + 1
    return out


def excluded(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in EXCLUDED_NAMES or path.endswith(EXCLUDED_SUFFIXES)


def _module_file(name: str, files: Dict[str, Entry]) -> Optional[str]:
    base = "src/" + name.replace(".", "/")
    for cand in (base + "/__init__.py", base + ".py"):
        if cand in files:
            return cand
    return None


def _imports(path: str, source: bytes) -> Set[str]:
    """Every module name a file's import statements can name (a ``from m import n`` names ``m`` and ``m.n``)."""
    try:
        tree_ = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    pkg = path[len("src/"):-len(".py")].split("/")[:-1]      # the package the file's relative imports start from
    out: Set[str] = set()
    for n in ast.walk(tree_):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                mod = ".".join(pkg[:len(pkg) - (n.level - 1)] + ([n.module] if n.module else []))
            else:
                mod = n.module or ""
            if not mod:
                continue
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in n.names)
    return out


def closure(files: Dict[str, Entry], repo: Optional[Path] = None,
            entries: Iterable[str] = ENTRY_MODULES) -> Set[str]:
    """The static import closure (as tracked paths under ``src/``) of ``entries`` in the tree ``files``."""
    seen: Dict[str, str] = {}                          # module -> path
    todo: List[str] = list(entries)
    content: Dict[str, bytes] = {}
    while todo:
        wave = []
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            path = _module_file(name, files)
            if path is None:
                continue
            seen[name] = path
            wave.append((name, path))
            parts = name.split(".")
            todo.extend(".".join(parts[:i]) for i in range(1, len(parts)))      # the parent packages run too
        need = {files[p][1] for _n, p in wave} - set(content)
        content.update(read_blobs(need, repo))
        for _n, p in wave:
            todo.extend(_imports(p, content[files[p][1]]))
    return set(seen.values())


def play_files(commit: str, repo: Optional[Path] = None) -> Dict[str, Entry]:
    """The play path at ``commit``: ``{path: (mode, blob sha)}``."""
    repo = repo or default_repo()
    files = tree(commit, repo)
    clos = closure(files, repo)
    if "src/main/x35/kakuna.py" not in clos:
        raise PlayPathError(f"{commit[:12]}: src/main/x35/kakuna.py is not in its tree (not a gen3ai commit?)")
    keep = {p for p in files if p in EXACT or p.startswith(PREFIXES)} | clos
    return {p: files[p] for p in sorted(keep) if not excluded(p)}


def digest(files: Dict[str, Entry]) -> str:
    """sha256 over ``<mode> <blob sha> <path>`` lines in path order (so a rename, a mode change or one byte moves it)."""
    h = hashlib.sha256(SCHEMA.encode() + b"\n")
    for p in sorted(files):
        mode, sha = files[p]
        h.update(f"{mode} {sha} {p}\n".encode())
    return h.hexdigest()


def differences(a: Dict[str, Entry], b: Dict[str, Entry]) -> List[str]:
    """Every path whose play-path entry differs, sorted, each as ``changed|added|removed <path>`` (a -> b)."""
    out = [f"changed {p}" for p in a.keys() & b.keys() if a[p] != b[p]]
    out += [f"removed {p}" for p in a.keys() - b.keys()]
    out += [f"added {p}" for p in b.keys() - a.keys()]
    return sorted(out, key=lambda s: s.split(" ", 1)[1])


def compare(old: str, new: str, repo: Optional[Path] = None) -> Tuple[List[str], str, int]:
    """``(differences, digest of new, number of play-path files)`` between two commits' play paths."""
    a, b = play_files(old, repo), play_files(new, repo)
    return differences(a, b), digest(b), len(b)
