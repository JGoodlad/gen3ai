"""The two conda env files (`environment_torch28.yml` and its CUDA 12.8 twin `environment_torch28_cu128.yml`)
declare everything the code imports, and stay the same environment except for the CUDA build.

Two defects this pins (both found 2026-10-12 by the first cloud rental):

1. **UNDECLARED IMPORTS.** `environment_torch28.yml` did not declare `scipy` or `scikit-learn`, which the training
   path imports (`ridealong_terms.rank_auroc` died on `from scipy.stats import rankdata`). The shared env had
   them from its `--clone gen3ai_stable` history, so nothing noticed until a FRESH build from the file. The
   test below scans every `import` of `src/` and `tools/` and fails when a third-party module's distribution is
   neither declared in the env file nor required (transitively) by something declared.
2. **THE TWIN DRIFTING.** `gen3ai_torch28_cu128` exists so a cu126-vs-cu128 measurement compares ONE variable. If
   the twin's pins drift from the main file (a new library bumped in one file only), the comparison silently
   gains a second variable. Every non-CUDA pin of the main file must appear in the twin at the same version.

Env-dependent on purpose in one place only: the import scan maps module names to distributions through the
INTERPRETER's installed metadata (`importlib.metadata.packages_distributions`), exactly as an env built from the
file would have them.
"""
from __future__ import annotations

import ast
import functools
import importlib.metadata as md
import os
import re
from typing import Dict, List, Set

import pytest

from utils import env_file as ef
from utils.paths import repo_path

_MAIN = "environment_torch28.yml"
_TWIN = "environment_torch28_cu128.yml"

#: Third-party modules the code imports that NO env file may be asked to carry, each with its reason. Keep this
#: tiny: a module that is not here and not declared is the bug this test exists to catch.
_NOT_OURS_TO_DECLARE = {
    # The Metamon peer script runs in METAMON's own interpreter (root CLAUDE.md, Python Environment).
    "amago": "main/anchors/peer_scripts/metamon_side.py: Metamon's interpreter",
    "gin": "main/anchors/peer_scripts/metamon_side.py: Metamon's interpreter",
    "huggingface_hub": "main/anchors/peer_scripts/metamon_side.py: Metamon's interpreter (the weight pins patch its download)",
    "metamon": "main/anchors/peer_scripts/metamon_side.py: Metamon's interpreter",
    "poke_env": "main/anchors/peer_scripts/metamon_side.py: upstream poke-env, in Metamon's interpreter",
    # The Foul Play pack dump runs in FOUL PLAY's own interpreter (`foulplay.python` in anchors.json); `fp` is its package.
    "fp": "main/anchors/peer_scripts/fp_pack_dump.py: Foul Play's interpreter",
    # `main/visualize_arch.py` imports netron lazily inside one function: an optional dev tool, installed in the
    # shared env by hand (the twin pins it as a dev extra).
    "netron": "main/visualize_arch.py: optional, lazy",
}

#: Skipped directories while scanning.
_SKIP_DIRS = {"node_modules", "target", ".git", "__pycache__", "deps", "models"}


def _text(name: str) -> str:
    with open(repo_path(name), encoding="utf-8") as fh:
        return fh.read()


def _local_top_levels() -> Set[str]:
    names: Set[str] = {"src", "tools"}
    for sub in ("src", "tools"):
        for entry in os.listdir(repo_path(sub)):
            names.add(entry[:-3] if entry.endswith(".py") else entry)
    return names


@functools.lru_cache(maxsize=None)
def _third_party_imports() -> Dict[str, str]:
    """top-level module -> one file that imports it, for every non-stdlib, non-local import in src/ and tools/."""
    import sys
    stdlib = set(sys.stdlib_module_names)
    local = _local_top_levels()
    found: Dict[str, str] = {}
    for sub in ("src", "tools"):
        for dirpath, dirnames, filenames in os.walk(repo_path(sub)):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, encoding="utf-8") as fh:
                        tree = ast.parse(fh.read())
                except (OSError, SyntaxError, ValueError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        mods = [a.name.split(".")[0] for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        mods = [node.module.split(".")[0]]
                    else:
                        continue
                    for m in mods:
                        if m not in stdlib and m not in local:
                            found.setdefault(m, os.path.relpath(path, repo_path()))
    return found


def _closure(declared: Set[str]) -> Set[str]:
    """``declared`` plus every installed distribution that a declared one requires, transitively (an
    ``extra == ...`` requirement is not followed: an env built from the file would not install it)."""
    seen: Set[str] = set()
    todo = list(declared)
    while todo:
        name = ef.norm(todo.pop())
        if name in seen:
            continue
        seen.add(name)
        try:
            reqs = md.requires(name) or []
        except md.PackageNotFoundError:
            continue
        for r in reqs:
            if re.search(r"extra\s*==", r):
                continue
            m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", r)
            if m:
                todo.append(m.group(1))
    return seen


# LINCHPIN (ship gate, 2026-10-19): kept in the routine gate at any cost — designs/ops/testing.md "The SHIP GATE"
@pytest.mark.linchpin
@pytest.mark.parametrize("env_file", [_MAIN, _TWIN])
def test_every_third_party_import_is_declared_by_the_env_file(env_file):
    declared = {n for n, _, _ in ef.pip_pins(_text(env_file))} | {n for n, _, _ in ef.conda_pins(_text(env_file))}
    have = _closure(declared)
    mod_to_dists = md.packages_distributions()
    missing: List[str] = []
    for mod, where in sorted(_third_party_imports().items()):
        if mod in _NOT_OURS_TO_DECLARE or mod in {"_pytest"}:
            continue
        dists = {ef.norm(d) for d in mod_to_dists.get(mod, [])}
        if not dists:
            missing.append(f"{mod} (imported by {where}) is not installed in the interpreter running this test")
        elif not dists & have:
            missing.append(f"{mod} -> {sorted(dists)} (imported by {where}) is declared by neither {env_file} "
                           f"nor anything it declares")
    assert not missing, ("imports no env file would install (declare the distribution in BOTH env files, "
                         "environment_torch28.yml first):\n  " + "\n  ".join(missing))


def test_the_exemptions_stay_honest():
    """An exemption for a module nothing imports any more is a stale entry: delete it."""
    imported = _third_party_imports()
    stale = [m for m in _NOT_OURS_TO_DECLARE if m not in imported]
    assert not stale, f"exempted but no longer imported anywhere: {stale}"


# ------------------------------------------------------------------------- the twin

_CUDA_PIP = re.compile(r"^(torch|torchvision|torchaudio|nvidia-.*-cu12)$")
_CUDA_CONDA = re.compile(r"^(cuda-.*|cuda)$")


def _by_name(text: str, pins) -> Dict[str, str]:
    return {n: (v or "") for n, v, _ in pins(text)}


def test_the_twin_pins_every_non_cuda_package_of_the_main_file_at_the_same_version():
    main, twin = _text(_MAIN), _text(_TWIN)
    tp, mp = _by_name(twin, ef.pip_pins), _by_name(main, ef.pip_pins)
    drift = [f"{n}: main {v!r}, twin {tp.get(n)!r}" for n, v in mp.items()
             if not _CUDA_PIP.match(n) and tp.get(n) != v]
    tc, mc = _by_name(twin, ef.conda_pins), _by_name(main, ef.conda_pins)
    drift += [f"conda {n}: main {v!r}, twin {tc.get(n)!r}" for n, v in mc.items()
              if not _CUDA_CONDA.match(n) and tc.get(n) != v]
    assert not drift, "the twin differs from the main file by more than the CUDA build:\n  " + "\n  ".join(drift)


def test_the_twin_is_on_cuda_128_everywhere_and_the_main_file_on_126():
    main, twin = _text(_MAIN), _text(_TWIN)
    assert re.search(r"^name:\s*gen3ai_torch28\s*$", main, re.M)
    assert re.search(r"^name:\s*gen3ai_torch28_cu128\s*$", twin, re.M)
    for text, tag, label in ((main, "cu126", "cuda-12.6.3"), (twin, "cu128", "cuda-12.8.1")):
        pins = _by_name(text, ef.pip_pins)
        for n in ("torch", "torchvision", "torchaudio"):
            assert pins[n].endswith(f"+{tag}"), f"{n} is {pins[n]!r}, wanted +{tag}"
        assert f"download.pytorch.org/whl/{tag}" in text
        assert f"nvidia/label/{label}" in text
        other = "cu128" if tag == "cu126" else "cu126"
        # a header COMMENT may name the other build; a PIN, index or channel may not
        live = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
        assert other not in live and ("12.8.1" if tag == "cu126" else "12.6.3") not in live, \
            f"a {other} pin or channel is left in the {tag} file"
    mp, tp = _by_name(main, ef.pip_pins), _by_name(twin, ef.pip_pins)
    for n in ("torch", "torchvision", "torchaudio"):
        assert mp[n].split("+")[0] == tp[n].split("+")[0], f"{n}: the two files name different RELEASES"
    assert mp["triton"] == tp["triton"]


def test_the_twin_declares_the_same_set_of_cuda_libraries_so_none_is_silently_missing():
    main, twin = _text(_MAIN), _text(_TWIN)
    nv = lambda t: {n for n in _by_name(t, ef.pip_pins) if n.startswith("nvidia-")}
    assert nv(main) == nv(twin) and len(nv(main)) >= 14
    cc = lambda t: {n for n in _by_name(t, ef.conda_pins) if _CUDA_CONDA.match(n)}
    assert cc(main) == cc(twin) and cc(main), "the CUDA header packages differ between the files"


def test_neither_file_re_adds_the_deliberately_absent_packages():
    for f in (_MAIN, _TWIN):
        names = {n for n, _, _ in ef.pip_pins(_text(f))} | {n for n, _, _ in ef.conda_pins(_text(f))}
        assert not names & {"poke-env", "asyncio"}, f"{f} re-adds a deliberately absent package"
