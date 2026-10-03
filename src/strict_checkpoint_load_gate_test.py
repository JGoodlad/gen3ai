"""Can any module load a PPO checkpoint through sb3's NON-STRICT retry? The static gate.

**The rule** (`gen3_strict_checkpoint_load_v1`, `agents/training/instrumented_ppo/strict_load.py`,
`designs/training/learner_lifecycle.md`). sb3's ``BaseAlgorithm.load`` retries with
``exact_match=False`` whenever the strict ``set_parameters`` error names ``pi_features_extractor`` — and
any missing extractor key does, through the alias — leaving that submodule at FRESH INIT behind one
warning (P10 review, F4). Every algorithm class this repo loads a checkpoint into therefore carries
``StrictCheckpointLoad`` (a strict, retry-refusing ``set_parameters``): the learner and the opponent /
reader classes via ``OwnedLoop``, every other reader via ``StrictMaskablePPO``, built by
``agents.model.snapshot.load_checkpoint_strict``. So this gate FAILS on, in any non-test module under
``src/``, ``tools/`` or ``scripts/``:

* ``<Algorithm>.load(...)`` where the class is one sb3 / sb3_contrib exports (``MaskablePPO``, ``PPO``,
  ``BaseAlgorithm``, ... — see :data:`ALGORITHM_NAMES`), however it was imported or aliased
  (``from sb3_contrib import MaskablePPO as M``, ``sb3_contrib.MaskablePPO.load``, ``P = PPO; P.load``);
* an explicit non-strict request — a ``set_parameters`` call (or any call) with ``exact_match=False``;
* a CLASS statement that subclasses one of those algorithm classes without ``StrictCheckpointLoad`` or
  ``OwnedLoop`` among its bases (its ``.load`` would then be sb3's, reachable through a name this
  gate cannot flag at the call site).

A repo class (``StrictMaskablePPO``, ``InstrumentedMaskablePPO``, ``InferenceMaskablePPO``) loads
strictly by construction, so ``.load`` on it is the legal spelling. Test files (``*_test.py``) are out
of scope: they build throwaway fixtures, several of them to prove that a BARE load fails.

**The allowlist is EMPTY.** Route the load through ``load_checkpoint_strict``.

What it cannot see, stated: ``getattr(MaskablePPO, "load")``, a class handed to a function as a value
(``loader(MaskablePPO)``), an alias of the sb3 MODULE under a name the import tracking does not follow,
and a ``load_state_dict(strict=False)`` the caller writes by hand.

Opt out explicitly (never silently): ``GEN3AI_SKIP_STRICT_LOAD_GATE=1``.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import List, Set

import pytest

from utils.paths import repo_root

pytestmark = pytest.mark.static   # the `static` budget tier (conftest._STATIC_BUDGET_BASE_S)

#: The algorithm classes sb3 / sb3_contrib export (a ``load`` classmethod on any of them is sb3's).
ALGORITHM_NAMES = frozenset({
    "A2C", "ARS", "BaseAlgorithm", "CrossQ", "DDPG", "DQN", "HER", "MaskablePPO", "OffPolicyAlgorithm",
    "OnPolicyAlgorithm", "PPO", "QRDQN", "RecurrentPPO", "SAC", "TD3", "TQC", "TRPO",
})
#: A subclass of an algorithm must list one of these among its bases (module docs).
STRICT_BASES = frozenset({"StrictCheckpointLoad", "OwnedLoop"})
_SB3_ROOTS = ("stable_baselines3", "sb3_contrib")

_SKIP = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_STRICT_LOAD_GATE") == "1",
                           reason="GEN3AI_SKIP_STRICT_LOAD_GATE=1 (explicit opt-out)")


def _dotted(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _is_sb3(module: str) -> bool:
    return module.split(".")[0] in _SB3_ROOTS


def _sb3_names(tree: ast.AST) -> "tuple[Set[str], Set[str]]":
    """``(algorithm names, sb3 module names)`` bound in this module: ``from sb3_contrib import
    MaskablePPO as M`` -> ``M``; ``import sb3_contrib`` / ``import stable_baselines3 as sb3`` -> the
    module name; ``P = PPO`` (a plain alias of a bound name) -> ``P``."""
    algos: Set[str] = set()
    mods: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and _is_sb3(node.module):
            for a in node.names:
                if a.name in ALGORITHM_NAMES:
                    algos.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if _is_sb3(a.name):
                    mods.add(a.asname or a.name.split(".")[0])
    for node in ast.walk(tree):          # plain aliases, `X = <bound algorithm>` (one pass is enough here)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if _algorithm_expr(node.value, algos, mods):
                algos.add(node.targets[0].id)
    return algos, mods


def _algorithm_expr(node: ast.AST, algos: Set[str], mods: Set[str]) -> bool:
    """Is ``node`` an sb3 algorithm CLASS: a bound name, or ``<sb3 module>.<...>.<Algorithm>``?"""
    if isinstance(node, ast.Name):
        return node.id in algos
    if isinstance(node, ast.Attribute):
        root = _dotted(node).split(".")[0]
        return root in mods and node.attr in ALGORITHM_NAMES
    return False


def violations_in(source: str, where: str) -> List[str]:
    tree = ast.parse(source)
    algos, mods = _sb3_names(tree)
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "load" and _algorithm_expr(node.func.value, algos, mods):
                out.append(f"{where}:{node.lineno} `{_dotted(node.func)}(...)` is sb3's load, with its "
                           "non-strict retry — use `agents.model.snapshot.load_checkpoint_strict`")
            nonstrict = [k for k in node.keywords if k.arg == "exact_match"
                         and isinstance(k.value, ast.Constant) and k.value.value is False]
            if node.func.attr == "set_parameters" and len(node.args) >= 2:
                a = node.args[1]
                if isinstance(a, ast.Constant) and a.value is False:
                    nonstrict.append(ast.keyword(arg="exact_match", value=a))
            if nonstrict:
                out.append(f"{where}:{node.lineno} `exact_match=False` — an explicit non-strict "
                           "checkpoint load (leaves a missing tensor at FRESH INIT)")
        elif isinstance(node, ast.ClassDef):
            bases = [b for b in node.bases]
            if any(_algorithm_expr(b, algos, mods) for b in bases):
                names = {_dotted(b).split(".")[-1] for b in bases}
                if not names & STRICT_BASES:
                    out.append(f"{where}:{node.lineno} `class {node.name}` subclasses an sb3 algorithm "
                               "without `StrictCheckpointLoad` / `OwnedLoop` — its `.load` is sb3's, "
                               "with the non-strict retry")
    return out


def algorithm_subclasses_in(source: str) -> List[str]:
    """The names of the classes in ``source`` that subclass an sb3 algorithm (the gate's R3 scope)."""
    tree = ast.parse(source)
    algos, mods = _sb3_names(tree)
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
            and any(_algorithm_expr(b, algos, mods) for b in n.bases)]


def _modules() -> List[Path]:
    root = repo_root()
    out: List[Path] = []
    for top in ("src", "tools", "scripts"):
        base = root / top
        if base.is_dir():
            out.extend(p for p in base.rglob("*.py")
                       if not p.name.endswith("_test.py") and "__pycache__" not in p.parts
                       and "node_modules" not in p.parts and "target" not in p.parts)
    return sorted(out)


@_SKIP
def test_no_module_loads_a_checkpoint_through_sb3s_non_strict_retry() -> None:
    root = repo_root()
    bad: List[str] = []
    for p in _modules():
        try:
            src = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "stable_baselines3" not in src and "sb3_contrib" not in src and "exact_match" not in src:
            continue
        try:
            bad.extend(violations_in(src, str(p.relative_to(root))))
        except SyntaxError:
            continue
    assert not bad, ("a checkpoint load that sb3 may retry NON-STRICTLY (src/strict_checkpoint_load_gate_"
                     "test.py):\n  " + "\n  ".join(bad))


@_SKIP
def test_the_gate_scans_something() -> None:
    mods = _modules()
    assert len(mods) > 500, f"the scan found only {len(mods)} modules — the scope is broken"
    by_name = {p.name: p for p in mods}
    # a positive control for the class rule: the gate SEES the repo's algorithm subclasses and the
    # strict mixin on each (a scan that resolved no sb3 name would pass for the wrong reason)
    strict = by_name["strict_load.py"].read_text(encoding="utf-8")
    assert algorithm_subclasses_in(strict) == ["StrictMaskablePPO"]
    assert violations_in(strict, "strict_load.py") == []
    ppo = by_name["ppo.py"].read_text(encoding="utf-8")
    assert "InstrumentedMaskablePPO" in algorithm_subclasses_in(ppo)
    assert violations_in(ppo, "ppo.py") == []


@pytest.mark.parametrize("snippet,n", [
    # a bare load of an sb3 algorithm class, however spelled
    ("from sb3_contrib import MaskablePPO\nm = MaskablePPO.load('x.zip')\n", 1),
    ("from sb3_contrib import MaskablePPO as M\nm = M.load('x.zip', env=None)\n", 1),
    ("from stable_baselines3 import PPO\nm = PPO.load('x')\n", 1),
    ("from stable_baselines3.common.base_class import BaseAlgorithm\nm = BaseAlgorithm.load('x')\n", 1),
    ("import sb3_contrib\nm = sb3_contrib.MaskablePPO.load('x')\n", 1),
    ("import stable_baselines3 as sb3\nm = sb3.PPO.load('x')\n", 1),
    ("from sb3_contrib import MaskablePPO\nP = MaskablePPO\nm = P.load('x')\n", 1),
    # an explicit non-strict request
    ("m.set_parameters(sd, exact_match=False)\n", 1),
    ("m.set_parameters(sd, False)\n", 1),
    # a subclass that does not carry the strict mixin
    ("from sb3_contrib import MaskablePPO\nclass A(MaskablePPO):\n    pass\n", 1),
    ("import sb3_contrib\nclass A(sb3_contrib.MaskablePPO):\n    pass\n", 1),
    ("from stable_baselines3 import PPO\nclass A(Mixin, PPO):\n    pass\n", 1),
    # legal: the strict classes, the strict loader, unrelated loads, a strict passthrough
    ("from sb3_contrib import MaskablePPO\nclass A(StrictCheckpointLoad, MaskablePPO):\n    pass\n", 0),
    ("from sb3_contrib import MaskablePPO\nclass A(Foo, OwnedLoop, MaskablePPO):\n    pass\n", 0),
    ("m = StrictMaskablePPO.load('x')\nn = InferenceMaskablePPO.load('y', env=None)\n", 0),
    ("from agents.model.snapshot import load_checkpoint_strict\nm = load_checkpoint_strict('x')\n", 0),
    ("import torch\nsd = torch.load('x')\nimport json\nj = json.load(f)\n", 0),
    ("from stable_baselines3.common.vec_env import VecNormalize\nv = VecNormalize.load('x', env)\n", 0),
    ("from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy\n"
     "class P(MaskableActorCriticPolicy):\n    pass\n", 0),
    ("m.set_parameters(sd, exact_match=True)\nm.set_parameters(sd, exact_match=flag)\nm.set_parameters(sd)\n", 0),
    # `.load` on a name that is NOT an imported sb3 algorithm (a user variable, a repo class)
    ("def f(MaskablePPO):\n    return MaskablePPO.load('x')\n", 0),
])
def test_the_gate_judges_each_spelling(snippet: str, n: int) -> None:
    assert len(violations_in(snippet, "snippet.py")) == n, violations_in(snippet, "snippet.py")
