"""Can any module SEED a process-global RNG through a name the runtime guard cannot see? The static gate.

**The rule** (`gen3_no_global_reseed_v1`, `agents/training/global_rng_guard.py`,
`designs/training/learner_lifecycle.md` "No global reseed after the freeze"). After the learner
freezes, a seed of Python `random`, NumPy's global `RandomState` or a torch generator is a typed FATAL
— the runtime guard wraps the seeding functions ON THEIR MODULES (`random.seed`, `numpy.random.seed`,
`torch.manual_seed`, `torch.random.manual_seed`, `torch.seed`, `torch.cuda.manual_seed[_all]`,
`torch.cuda.seed[_all]`). A name bound to one of them BEFORE the guard armed holds the unwrapped
function, and a seed through it passes silently. So this gate FAILS on, in any non-test module under
``src/``:

* ``from random import seed`` / ``from numpy.random import seed`` / ``from torch import manual_seed``
  (and every other (module, seeding function) pair above, aliased or not);
* a seeding function taken as a VALUE rather than called through its module — ``f = random.seed``,
  ``hooks.append(np.random.seed)``, ``partial(torch.manual_seed, 3)``.

A call through the module attribute (``random.seed(3)``, ``np.random.seed(3)``, ``th.manual_seed(3)``)
is what the guard sees, so it is the one legal spelling. Test files (``*_test.py``) are out of scope:
they assert the guard's own wrapping by identity, and no test runs a frozen learner of its own.

**The allowlist is EMPTY.** Call it through its module.

What it cannot see, stated: ``getattr(random, "seed")``, an alias of the MODULE under a name not listed
in :data:`MODULE_ALIASES`, and `random.Random` / `np.random.RandomState` instances (private
generators — not global streams, so not this rule's business).

Opt out explicitly (never silently): ``GEN3AI_SKIP_GLOBAL_RNG_GATE=1``.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict, List, Set

import pytest

from utils.paths import src_root

pytestmark = pytest.mark.static   # the `static` budget tier (conftest._STATIC_BUDGET_BASE_S)

#: module path -> the seeding functions the runtime guard wraps on it.
SEEDERS: Dict[str, Set[str]] = {
    "random": {"seed"},
    "numpy.random": {"seed"},
    "torch": {"manual_seed", "seed"},
    "torch.random": {"manual_seed", "seed"},
    "torch.cuda": {"manual_seed", "manual_seed_all", "seed", "seed_all"},
}
#: dotted spellings of those modules as the repo writes them.
MODULE_ALIASES: Dict[str, str] = {
    "random": "random", "np.random": "numpy.random", "numpy.random": "numpy.random",
    "torch": "torch", "th": "torch", "torch.random": "torch.random", "th.random": "torch.random",
    "torch.cuda": "torch.cuda", "th.cuda": "torch.cuda",
}

_SKIP = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_GLOBAL_RNG_GATE") == "1",
                           reason="GEN3AI_SKIP_GLOBAL_RNG_GATE=1 (explicit opt-out)")


def _dotted(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def violations_in(source: str, where: str) -> List[str]:
    tree = ast.parse(source)
    out: List[str] = []
    called: Set[int] = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in SEEDERS and node.level == 0:
            for a in node.names:
                if a.name in SEEDERS[node.module]:
                    out.append(f"{where}:{node.lineno} `from {node.module} import {a.name}` binds an "
                               f"unguarded seeding function — call `{node.module}.{a.name}(...)`")
        elif isinstance(node, ast.Attribute) and id(node) not in called:
            mod = MODULE_ALIASES.get(_dotted(node.value))
            if mod is not None and node.attr in SEEDERS[mod]:
                out.append(f"{where}:{node.lineno} `{_dotted(node)}` taken as a value — a name bound "
                           f"to it bypasses the guard; call it through its module")
    return out


def _modules() -> List[Path]:
    root = src_root()
    return sorted(p for p in root.rglob("*.py")
                  if not p.name.endswith("_test.py") and "__pycache__" not in p.parts)


@_SKIP
def test_no_module_binds_a_global_seeding_function_by_name() -> None:
    root = src_root()
    bad: List[str] = []
    for p in _modules():
        try:
            src = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "seed" not in src:
            continue
        bad.extend(violations_in(src, str(p.relative_to(root))))
    assert not bad, ("a global seeding function bound by name (the runtime reseed guard cannot see a "
                     "call through it — src/global_rng_seed_gate_test.py):\n  " + "\n  ".join(bad))


@_SKIP
def test_the_gate_scans_something() -> None:
    mods = _modules()
    assert len(mods) > 500, f"the scan found only {len(mods)} modules — the scope is broken"
    assert any(p.name == "global_rng_guard.py" for p in mods)


@pytest.mark.parametrize("snippet,n", [
    ("from random import seed\n", 1),
    ("from numpy.random import seed as s\n", 1),
    ("from torch import manual_seed\n", 1),
    ("from torch.cuda import manual_seed_all\n", 1),
    ("import random\nf = random.seed\n", 1),
    ("import numpy as np\nhooks = [np.random.seed]\n", 1),
    ("import torch as th\nimport functools\np = functools.partial(th.manual_seed, 3)\n", 1),
    # legal: a call through the module attribute; private generators; unrelated `seed` names
    ("import random\nrandom.seed(3)\n", 0),
    ("import numpy as np\nnp.random.seed(3)\nimport torch as th\nth.manual_seed(3)\n", 0),
    ("import random\nr = random.Random(3)\nr.seed(4)\nx = r.seed\n", 0),
    ("from agents.training import keyed_draw as KD\nseed = KD.seed\n", 0),
    ("from random import Random, choice\n", 0),
])
def test_the_gate_judges_each_spelling(snippet: str, n: int) -> None:
    assert len(violations_in(snippet, "snippet.py")) == n
