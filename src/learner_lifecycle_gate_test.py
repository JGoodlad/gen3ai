"""Does any TRAINING-STEP code path build an optimizer, a Parameter or a Module? The static gate.

**The rule (the M5 DECLARED LIFECYCLE, `designs/endstate/program_rust_core.md`, "M5 DESIGN PRINCIPLE —
a DECLARED LIFECYCLE").** A training process acquires everything at STARTUP, then FREEZES; the steady
state acquires nothing. The K8 inventory found the ride-along heads' Adam built LAZILY on the first
update — by luck, not by any check. This gate is the STATIC twin of the runtime freeze guard: it
FAILS on any of

* an OPTIMIZER construction — any ``torch.optim`` optimizer class however spelled
  (``torch.optim.Adam(``, ``th.optim.AdamW(``, ``optim.SGD(``, ``from torch.optim import Adam`` then
  ``Adam(``), a repo subclass of one, and ``<anything>.optimizer_class(`` (SB3's spelling);
* a PARAMETER construction — ``torch.nn.Parameter`` however spelled (``nn.Parameter(``,
  ``th.nn.Parameter(``, ``nn.parameter.Parameter(``, an imported ``Parameter``);
* a MODULE construction — any ``torch.nn`` class that subclasses ``nn.Module`` (``nn.Linear(``,
  ``th.nn.LayerNorm(``, ``nn.ModuleList(`` …), and any REPO class that subclasses ``nn.Module``
  directly or transitively;

inside a TRAINING-STEP code path, except within a function decorated ``@startup_builder``
(`agents.training.lifecycle_decl`) or a class's ``__init__`` / ``_build`` / ``_setup_model`` (an
exemption covers every def / lambda nested inside it). Module-level and class-body code is import
time, i.e. startup, and is not scanned.

**The allowlist is EMPTY — a new entry is not a legal move.** Move the construction to startup (a
``@startup_builder``, ``_setup_model``, an ``__init__``); if the scope below wrongly includes a
genuine startup method, fix the SCOPE (and say why in its justification), never list the site.

**THE SCOPE — declared, reviewable lists; no call-graph walk.**

1. :data:`STEP_PACKAGES`: every function / method of every module in
   ``agents/training/instrumented_ppo/`` — the learner itself (``train()``, its mixins, the rollout
   collection override, the per-update probes and exports). The package's startup methods are the
   exemptions (``_setup_model``, ``__init__``).
2. :data:`STEP_MODULES`: the loss-term and probe modules the fold calls EVERY UPDATE — exactly the
   ``agents.*`` modules ``instrumented_ppo/*.py`` imports (top-level or deferred), minus the pure
   constant / mode readers. Each entry carries its reason. A new module the fold starts calling is
   added HERE, in the same change (the coverage test below fails when an ``instrumented_ppo``
   import names an ``agents`` module that is in neither list nor :data:`NOT_STEP_MODULES`).
3. CALLBACKS: the PER-STEP methods (:data:`CALLBACK_STEP_METHODS`) of every SB3 callback class (a
   class reaching ``BaseCallback`` through its bases, resolved by name) defined under
   ``src/agents/training/`` or ``src/main/train/``. ``_on_training_start`` / ``_init_callback`` run
   before the first rollout and count as startup.

**What it cannot see, stated** (the RUNTIME freeze guard covers every one of these):

* a construction in a helper module OUTSIDE the lists that a step path calls (e.g. a callback's
  ``_on_step`` calling ``self._rebuild()`` — only the per-step methods themselves are scanned —
  or a utility in ``utils/``);
* a ``@startup_builder`` (or an ``__init__``) CALLED from a step path — the decorator is a
  declaration, the guard is the proof;
* a class reached dynamically: ``type(m)(...)``, ``getattr(nn, name)(...)``, ``copy.deepcopy(module)``,
  a class held in a variable, a factory function returning a module;
* a repo class whose NAME collides with a Module subclass elsewhere: repo resolution is by class
  NAME over every class under ``src/`` (bases read as their last dotted component), so a same-named
  non-Module class is over-reported and a Module reached through an alias of its name is missed.

Opt out explicitly (never silently): ``GEN3AI_SKIP_LIFECYCLE_GATE=1``.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import pytest

from utils.paths import src_root

#: EMPTY by rule. A late construction is moved to startup; it is never listed here.
ALLOWLIST: Tuple[str, ...] = ()

#: Every module of these packages is a training-step path, every function in it.
STEP_PACKAGES: Tuple[str, ...] = ("agents/training/instrumented_ppo",)

#: The loss-term / probe modules the fold calls every update, with the reason each is in scope.
STEP_MODULES: Dict[str, str] = {
    "agents/training/loop_hooks.py": "the loop's DECLARED hook table (gen3_declared_loop_hooks_v1): "
                                     "`around` runs at every collect and update; it builds no learner object",
    "agents/training/belief_bank.py": "the belief-bank loss terms, computed inside the PPO minibatch loop",
    "agents/training/belief_bank_static.py": "the belief bank's static twins, called by the micro-step (R1) per minibatch",
    "agents/model/masked_categorical.py": "the functional masked distribution the micro-step evaluates per minibatch",
    "agents/model/region_calls.py": "the no-silent-eager route counters every region body and dispatcher touches per call",
    "agents/training/capacity_telemetry.py": "capacity telemetry the fold computes per update",
    "agents/training/cf_terms.py": "counterfactual-grounding loss terms, per minibatch",
    "agents/training/frozen_phi.py": "the frozen-phi actor-only seams, hit on every gradient step",
    "agents/training/q_winprob_terms.py": "the Q / win-prob loss terms, per minibatch",
    "agents/training/td_aux.py": "the TD auxiliary loss, per minibatch",
    "agents/training/grad_balance.py": "gradient-balance metrics + shared-trunk parameter lists, per update",
    "agents/training/rank_metrics.py": "the rank probe, run on the update's features",
    "agents/training/opp_intent_labels.py": "the opponent-intent labels built per rollout/update",
    "agents/training/winprob_pbrs.py": "the win-prob PBRS reward, applied to every rollout",
    "agents/training/fork_arm.py": "the fork-arm PG mask read per minibatch",
    "agents/training/scaffolding.py": "the live scaffolding gauge, computed per update",
    "agents/training/async_vec_env.py": "the async rollout collection, run every rollout",
    "agents/training/rust_rollout/consistency.py": "the behaviour-policy consistency gate, per update",
    "agents/training/teacher/buffer.py": "the teacher buffer the distill terms sample per minibatch",
    "agents/model/opp_intent.py": "the opponent-intent loss functions + label matching, per minibatch",
    "agents/model/dense_aux_head.py": "the dense-aux head's loss helpers, per minibatch",
    "agents/model/ridealong_heads.py": "the ride-along heads' forward + loss, per minibatch (epoch 0)",
}

#: ``agents`` modules ``instrumented_ppo`` imports that are NOT step paths, and why.
NOT_STEP_MODULES: Dict[str, str] = {
    "agents/model/critic_mode.py": "a pure mode predicate (`is_winprob`) read at setup; builds nothing",
    "agents/model/compile_trainer.py": "the compile wrapper, applied once at startup; the per-step "
                                       "`eager_extractor` is a context manager over existing modules",
    "agents/observation/constants.py": "layout constants only",
    "agents/model/arch_constants.py": "architecture constants only",
    "agents/training/lifecycle_decl.py": "the `@startup_builder` marker itself; builds nothing",
    "agents/model/compile_regions.py": "the compile regions' install / gate / prewarm run once at startup; "
                                       "the per-update `check_r1_declared` compares R1's inputs to the "
                                       "startup declaration and builds nothing",
}

#: The PER-STEP SB3 callback hooks. ``_on_training_start`` / ``_init_callback`` are startup.
CALLBACK_STEP_METHODS = frozenset({"_on_step", "_on_rollout_start", "_on_rollout_end", "on_step",
                                   "on_rollout_start", "on_rollout_end", "_on_training_end"})
#: Where a callback class is looked for.
CALLBACK_ROOTS: Tuple[str, ...] = ("agents/training", "main/train")
#: SB3 callback base names (the transitive resolution starts here).
CALLBACK_SEEDS = frozenset({"BaseCallback", "EventCallback", "CallbackList", "EvalCallback",
                            "CheckpointCallback", "EveryNTimesteps", "MaskableEvalCallback",
                            "StopTrainingOnRewardThreshold", "ProgressBarCallback"})

#: The exemptions: a METHOD with one of these names (its class's startup), and this decorator.
EXEMPT_METHODS = frozenset({"__init__", "_build", "_setup_model"})
STARTUP_DECORATOR = "startup_builder"


def _torch_class_names() -> Tuple[Set[str], Set[str]]:
    """``torch.nn`` Module classes and ``torch.optim`` Optimizer classes, read from torch itself —
    a missing torch FAILS the gate (an import error), it never skips it."""
    import torch

    nn_mods = {n for n, o in vars(torch.nn).items() if isinstance(o, type) and issubclass(o, torch.nn.Module)}
    optims = {n for n, o in vars(torch.optim).items()
              if isinstance(o, type) and issubclass(o, torch.optim.Optimizer)}
    return nn_mods, optims


TORCH_NN_MODULES, TORCH_OPTIMIZERS = _torch_class_names()


class Hit(NamedTuple):
    path: str
    line: int
    kind: str          # "optimizer" | "parameter" | "module"
    construct: str     # the call as spelled
    function: str      # the enclosing function's dotted name

    def render(self) -> str:
        return (f"{self.path}:{self.line}: {self.kind} construction `{self.construct}(...)` in "
                f"`{self.function}` — acquire it in a `@startup_builder` "
                "(agents.training.lifecycle_decl) / `_setup_model` / an `__init__`, i.e. move it to startup")


# --------------------------------------------------------------------------------------------- #
# the class universe: which repo class NAMES are Modules / Optimizers / callbacks (name-based)
# --------------------------------------------------------------------------------------------- #

def _base_name(b: ast.expr) -> str:
    if isinstance(b, ast.Subscript):          # Generic[...] and friends
        b = b.value
    if isinstance(b, ast.Name):
        return b.id
    if isinstance(b, ast.Attribute):
        return b.attr
    return ""


def class_bases(sources: Dict[str, str]) -> Dict[str, Set[str]]:
    """class NAME -> the union of its base names over every definition of that name."""
    out: Dict[str, Set[str]] = {}
    for rel, src in sources.items():
        try:
            tree = ast.parse(src, rel)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                out.setdefault(n.name, set()).update(x for x in map(_base_name, n.bases) if x)
    return out


def descendants(bases: Dict[str, Set[str]], seeds: Set[str]) -> Set[str]:
    """Every class name that reaches a seed through its bases (the fixpoint), seeds excluded."""
    hit: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, bs in bases.items():
            if name not in hit and bs & (seeds | hit):
                hit.add(name)
                changed = True
    return hit


class Universe(NamedTuple):
    repo_modules: Set[str]
    repo_optimizers: Set[str]
    callbacks: Set[str]


def build_universe(sources: Dict[str, str]) -> Universe:
    bases = class_bases(sources)
    return Universe(repo_modules=descendants(bases, set(TORCH_NN_MODULES)),
                    repo_optimizers=descendants(bases, set(TORCH_OPTIMIZERS)),
                    callbacks=descendants(bases, set(CALLBACK_SEEDS)))


# --------------------------------------------------------------------------------------------- #
# one module's scan
# --------------------------------------------------------------------------------------------- #

def _imports(tree: ast.AST) -> Dict[str, str]:
    """local name -> the dotted name it binds (every import in the module, deferred ones included)."""
    out: Dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    root = a.name.split(".")[0]
                    out[root] = root
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for a in n.names:
                out[a.asname or a.name] = f"{n.module}.{a.name}"
    return out


def _dotted(e: ast.expr) -> Optional[List[str]]:
    if isinstance(e, ast.Name):
        return [e.id]
    if isinstance(e, ast.Attribute):
        head = _dotted(e.value)
        return None if head is None else [*head, e.attr]
    return None


def classify(func: ast.expr, imports: Dict[str, str], uni: Universe) -> Optional[Tuple[str, str]]:
    """(kind, spelling) when calling ``func`` constructs an optimizer / Parameter / Module."""
    parts = _dotted(func)
    if not parts:
        return None
    spelling = ".".join(parts)
    last = parts[-1]
    if last == "optimizer_class" and len(parts) > 1:
        return "optimizer", spelling
    root = imports.get(parts[0])
    q = ".".join([root, *parts[1:]]) if root else spelling
    last = q.split(".")[-1]               # the RESOLVED class name (`P` bound to `...Parameter`)
    if q == "torch" or q.startswith("torch."):
        if q.startswith("torch.nn.") and last == "Parameter":
            return "parameter", spelling
        if q.startswith("torch.nn.") and last in TORCH_NN_MODULES:
            return "module", spelling
        if q.startswith("torch.optim.") and last in TORCH_OPTIMIZERS:
            return "optimizer", spelling
        return None
    # a repo class: bare (defined here or imported by name), or through a non-torch module alias.
    if len(parts) == 1 or (root is not None and parts[0] != "self"):
        if last in uni.repo_modules:
            return "module", spelling
        if last in uni.repo_optimizers:
            return "optimizer", spelling
    return None


def _is_startup_builder(fn: ast.AST) -> bool:
    decs = getattr(fn, "decorator_list", [])
    for d in decs:
        d = d.func if isinstance(d, ast.Call) else d
        if _base_name(d) == STARTUP_DECORATOR:
            return True
    return False


def scan_source(src: str, rel: str, uni: Universe, *, whole_module: bool) -> List[Hit]:
    """Every violation in ``src``. ``whole_module``: every function is a step path; otherwise only
    the per-step methods of callback classes (``uni.callbacks``) are."""
    tree = ast.parse(src, rel)
    imports = _imports(tree)
    hits: List[Hit] = []

    def visit(node: ast.AST, qual: List[str], cls: Optional[ast.ClassDef], in_step: bool,
              exempt: bool) -> None:
        # ``cls`` is set only while walking a class BODY (so a def there is a method, even under an
        # ``if``); module-level and class-body statements are import time (startup): ``in_step``
        # starts False and only a step function's body turns it on.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, [*qual, child.name], child, in_step, exempt)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_method = cls is not None
                ex = exempt or _is_startup_builder(child) or (is_method and child.name in EXEMPT_METHODS)
                step = in_step or whole_module or (
                    cls is not None and cls.name in uni.callbacks and child.name in CALLBACK_STEP_METHODS)
                visit(child, [*qual, child.name], None, step, ex)
            else:
                if isinstance(child, ast.Call) and in_step and not exempt:
                    got = classify(child.func, imports, uni)
                    if got is not None:
                        hits.append(Hit(rel, child.lineno, got[0], got[1], ".".join(qual) or "<module>"))
                visit(child, qual, cls, in_step, exempt)

    visit(tree, [], None, False, False)
    return sorted(set(hits))


# --------------------------------------------------------------------------------------------- #
# the tree
# --------------------------------------------------------------------------------------------- #

def _is_src(rel: str) -> bool:
    return rel.endswith(".py") and not rel.endswith("_test.py") and "/target/" not in rel \
        and not rel.startswith("rust_sim/target")


def _read_all(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if not _is_src(rel):
            continue
        try:
            out[rel] = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
    return out


def step_modules(sources: Dict[str, str]) -> Set[str]:
    out = {r for r in sources for pkg in STEP_PACKAGES if r.startswith(pkg + "/")}
    return out | set(STEP_MODULES)


def scan_tree(root: Path) -> Tuple[List[Hit], Dict[str, int]]:
    """(every violation, coverage counters) over ``root`` (= ``src/``)."""
    sources = _read_all(root)
    uni = build_universe(sources)
    whole = step_modules(sources)
    hits: List[Hit] = []
    n_cb_files = 0
    for rel, src in sources.items():
        if rel in whole:
            hits.extend(scan_source(src, rel, uni, whole_module=True))
        elif any(rel.startswith(r + "/") for r in CALLBACK_ROOTS):
            n_cb_files += 1
            hits.extend(scan_source(src, rel, uni, whole_module=False))
    return sorted(set(hits)), {"step_modules": len(whole & set(sources)), "callback_files": n_cb_files,
                               "callback_classes": len(uni.callbacks), "repo_modules": len(uni.repo_modules)}


def _instrumented_ppo_agents_imports(sources: Dict[str, str]) -> Set[str]:
    """Every ``agents.*`` MODULE file ``instrumented_ppo/*.py`` imports (deferred imports included)."""
    out: Set[str] = set()
    for rel, src in sources.items():
        if not rel.startswith(STEP_PACKAGES[0] + "/"):
            continue
        for n in ast.walk(ast.parse(src, rel)):
            mods: List[str] = []
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("agents."):
                mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Import):
                mods = [a.name for a in n.names if a.name.startswith("agents.")]
            for m in mods:
                rel_m = m.replace(".", "/") + ".py"
                if rel_m in sources:
                    out.add(rel_m)
    return {m for m in out if not any(m.startswith(p + "/") for p in STEP_PACKAGES)}


# --------------------------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------------------------- #

_SKIP = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_LIFECYCLE_GATE") == "1",
                           reason="GEN3AI_SKIP_LIFECYCLE_GATE=1 (explicit opt-out)")


@pytest.fixture(scope="module")
def tree_scan() -> Tuple[List[Hit], Dict[str, int]]:
    return scan_tree(src_root())


@_SKIP
def test_no_training_step_path_constructs_an_optimizer_parameter_or_module(tree_scan: Tuple[List[Hit], Dict[str, int]]) -> None:
    hits = [h for h in tree_scan[0] if f"{h.path}:{h.line}" not in ALLOWLIST]
    assert not hits, (
        f"{len(hits)} construction(s) in a TRAINING-STEP code path — the declared lifecycle acquires "
        "everything at STARTUP and the steady state acquires nothing (designs/endstate/"
        "program_rust_core.md, 'M5 DESIGN PRINCIPLE — a DECLARED LIFECYCLE'). The allowlist is "
        "EMPTY; a new entry is not a legal move:\n  " + "\n  ".join(h.render() for h in hits))


@_SKIP
def test_the_scan_actually_walked_the_tree(tree_scan: Tuple[List[Hit], Dict[str, int]]) -> None:
    """A scan that found nothing because it looked at nothing reads exactly like a clean tree."""
    c = tree_scan[1]
    assert c["step_modules"] >= len(STEP_MODULES) + 15, c
    assert c["callback_files"] >= 20 and c["callback_classes"] >= 20, c
    assert c["repo_modules"] >= 20, c
    assert {"Linear", "LayerNorm", "ModuleList", "Sequential"} <= TORCH_NN_MODULES
    assert {"Adam", "AdamW", "SGD"} <= TORCH_OPTIMIZERS


@_SKIP
def test_every_declared_step_module_exists() -> None:
    root = src_root()
    missing = [m for m in [*STEP_MODULES, *NOT_STEP_MODULES] if not (root / m).is_file()]
    assert not missing, f"declared scope entries that no longer exist (renamed? fix the list): {missing}"
    assert all(len(why) >= 20 for why in [*STEP_MODULES.values(), *NOT_STEP_MODULES.values()])


@_SKIP
def test_every_agents_module_the_learner_imports_is_classified() -> None:
    """The scope is a DECLARED list, so it must grow with the fold: an ``agents`` module that
    ``instrumented_ppo`` starts importing is classified (STEP_MODULES or NOT_STEP_MODULES) in the
    same change."""
    sources = _read_all(src_root())
    unclassified = sorted(_instrumented_ppo_agents_imports(sources) - set(STEP_MODULES) - set(NOT_STEP_MODULES))
    assert not unclassified, (
        f"instrumented_ppo imports {unclassified}, which the lifecycle gate's scope does not classify. "
        "Add each to STEP_MODULES (the fold calls it per update) or NOT_STEP_MODULES (with why).")


def test_the_allowlist_is_empty() -> None:
    assert ALLOWLIST == ()


def test_the_decorator_is_the_one_the_gate_reads() -> None:
    from agents.training.lifecycle_decl import startup_builder

    assert startup_builder.__name__ == STARTUP_DECORATOR

    @startup_builder
    def f() -> int:
        return 1

    assert f() == 1 and getattr(f, "__gen3_startup_builder__") is True


# --------------------------------------------------------------------------------------------- #
# TEETH: synthetic sources
# --------------------------------------------------------------------------------------------- #

_UNI = build_universe({"x.py": "import torch.nn as nn\nclass MyHead(nn.Module):\n    pass\n"
                                "class DeepHead(MyHead):\n    pass\n"
                                "class MyCb(BaseCallback):\n    pass\n"
                                "class SubCb(MyCb):\n    pass\n"
                                "class MyOpt(torch.optim.Adam):\n    pass\n"})


def _kinds(src: str, *, whole_module: bool = True) -> List[str]:
    return [h.kind for h in scan_source(src, "t.py", _UNI, whole_module=whole_module)]


@pytest.mark.parametrize("src,kind", [
    ("import torch\ndef step(p):\n    return torch.optim.Adam(p)\n", "optimizer"),
    ("import torch as th\ndef step(p):\n    return th.optim.AdamW(p)\n", "optimizer"),
    ("from torch import optim\ndef step(p):\n    return optim.SGD(p, lr=1)\n", "optimizer"),
    ("from torch.optim import Adam\ndef step(p):\n    return Adam(p)\n", "optimizer"),
    ("import torch.optim as O\ndef step(p):\n    return O.RMSprop(p)\n", "optimizer"),
    ("class L:\n    def train(self):\n        self.optimizer = self.optimizer_class(self.p)\n", "optimizer"),
    ("def step(p):\n    return MyOpt(p)\n", "optimizer"),
    ("import torch.nn as nn\ndef step(t):\n    return nn.Parameter(t)\n", "parameter"),
    ("import torch\ndef step(t):\n    return torch.nn.Parameter(t)\n", "parameter"),
    ("import torch as th\ndef step(t):\n    return th.nn.Parameter(t)\n", "parameter"),
    ("from torch import nn\ndef step(t):\n    return nn.parameter.Parameter(t)\n", "parameter"),
    ("from torch.nn import Parameter\ndef step(t):\n    return Parameter(t)\n", "parameter"),
    ("from torch.nn.parameter import Parameter as P\ndef step(t):\n    return P(t)\n", "parameter"),
    ("import torch.nn as nn\ndef step():\n    return nn.Linear(2, 3)\n", "module"),
    ("import torch as th\ndef step():\n    return th.nn.LayerNorm(4)\n", "module"),
    ("from torch import nn\ndef step(ms):\n    return nn.ModuleList(ms)\n", "module"),
    ("from torch.nn import Sequential\ndef step():\n    return Sequential()\n", "module"),
    ("def step():\n    return MyHead()\n", "module"),
    ("def step():\n    return DeepHead()\n", "module"),          # transitive
    ("from agents.model import heads as H\ndef step():\n    return H.MyHead()\n", "module"),
    # nested inside a step function: still a step path
    ("import torch.nn as nn\ndef step():\n    def inner():\n        return nn.Linear(1, 1)\n    return inner\n", "module"),
    ("import torch.nn as nn\nclass T:\n    def train(self):\n        f = lambda: nn.Linear(1, 1)\n        return f\n", "module"),
    # a method named like an exemption but NOT a method (a free function) is not exempt
    ("import torch.nn as nn\ndef _build():\n    return nn.Linear(1, 1)\n", "module"),
    # a DIFFERENT decorator is not the marker
    ("import torch.nn as nn\nimport functools\n@functools.lru_cache\ndef step():\n    return nn.Linear(1, 1)\n", "module"),
])
def test_teeth_each_violation_kind_fails(src: str, kind: str) -> None:
    assert _kinds(src) == [kind], src


@pytest.mark.parametrize("src", [
    # the marker, bare and qualified, with nested defs under it
    "import torch\nfrom agents.training.lifecycle_decl import startup_builder\n@startup_builder\n"
    "def build(p):\n    return torch.optim.Adam(p)\n",
    "import torch.nn as nn\nfrom agents.training import lifecycle_decl\nclass T:\n"
    "    @lifecycle_decl.startup_builder\n    def acquire(self):\n        def mk():\n            return nn.Linear(1, 1)\n"
    "        self.h = mk()\n",
    # the class startup methods
    "import torch.nn as nn\nclass H(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
    "        self.l = nn.Linear(1, 1)\n        self.p = nn.Parameter(self.l.weight)\n",
    "import torch\nclass L:\n    def _setup_model(self):\n        self.opt = torch.optim.Adam(self.p)\n"
    "        self.o2 = self.optimizer_class(self.p)\n",
    "import torch.nn as nn\nclass H:\n    def _build(self):\n        self.l = nn.Linear(1, 1)\n",
    # import time / class body is startup
    "import torch.nn as nn\nHEAD = nn.Linear(1, 1)\nclass C:\n    proto = nn.Linear(1, 1)\n",
    # not a construction: functional ops, utils, a tensor
    "import torch\nfrom torch.nn import functional as F\ndef step(x, p):\n"
    "    torch.nn.utils.clip_grad_norm_(p, 1.0)\n    return F.linear(x, x) + torch.zeros(3)\n",
    # an unrelated name that merely shares the attribute with self
    "class L:\n    def step(self):\n        return self.MyHead()\n",
])
def test_teeth_each_exemption_passes(src: str) -> None:
    assert _kinds(src) == [], src


_CB = ("import torch.nn as nn\nimport torch\nclass Cb(SubCb):\n"
       "    def _init_callback(self):\n        self.h = nn.Linear(1, 1)\n"
       "    def _on_training_start(self):\n        self.opt = torch.optim.Adam(self.h.parameters())\n"
       "    def {hook}(self):\n        self.h2 = nn.Linear(1, 1)\n        return True\n"
       "    def helper(self):\n        return nn.Linear(1, 1)\n")


@pytest.mark.parametrize("hook", sorted(CALLBACK_STEP_METHODS))
def test_teeth_a_callbacks_per_step_hook_fails_and_its_startup_hooks_pass(hook: str) -> None:
    src = _CB.replace("{hook}", hook)
    uni = build_universe({"base.py": "class MyCb(BaseCallback):\n    pass\nclass SubCb(MyCb):\n    pass\n",
                          "cb.py": src})
    assert "Cb" in uni.callbacks                     # resolved TRANSITIVELY: Cb -> SubCb -> MyCb -> BaseCallback
    hits = scan_source(src, "cb.py", uni, whole_module=False)
    assert [(h.kind, h.function) for h in hits] == [("module", f"Cb.{hook}")]


def test_teeth_a_non_callback_classs_on_step_is_not_scanned_outside_the_step_modules() -> None:
    src = "import torch.nn as nn\nclass NotCb:\n    def _on_step(self):\n        return nn.Linear(1, 1)\n"
    assert scan_source(src, "x.py", _UNI, whole_module=False) == []


def test_teeth_a_def_inside_a_module_level_block_is_scanned() -> None:
    src = "import torch.nn as nn\ntry:\n    def step():\n        return nn.Linear(1, 1)\nexcept Exception:\n    pass\n"
    assert _kinds(src) == ["module"]


def test_teeth_the_message_names_file_line_construct_function_and_fix() -> None:
    (h,) = scan_source("import torch\nclass L:\n    def train(self):\n        x = 1\n"
                       "        self.o = torch.optim.Adam(self.p)\n", "a/b.py", _UNI, whole_module=True)
    msg = h.render()
    for part in ("a/b.py:5", "optimizer", "`torch.optim.Adam(...)`", "`L.train`", "@startup_builder",
                 "move it to startup"):
        assert part in msg, (part, msg)
