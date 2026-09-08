"""THE STUB-VACUITY SCANNER — does a patched symbol exist, in the module the patch names?

The ENGINE behind `src/test_stub_vacuity_gate_test.py`. Pure AST, no imports of the code under
test, no runtime patching: a scan must be able to read a module whose architecture drifted past
current code, and a gate that imported every test module would cost minutes and could run a
training job (`src/main/entry_point_guard_test.py`).

**What a patch site IS.** `monkeypatch.setattr` / `monkeypatch.delattr` / `mock.patch` /
`patch.object` / `patch` — in either of the two spellings the stdlib and pytest share:

    monkeypatch.setattr("agents.training.foo.helper", stub)      # STRING form
    monkeypatch.setattr(foo_mod, "helper", stub)                 # OBJECT form
    with patch.object(foo_mod, "helper"): ...

Both resolve to the pair `(module, attribute)`. Everything else — a patch on a CLASS attribute, on
an instance, on a third-party module, on a `dict` item — is out of scope and reported as SKIPPED
with its reason, never silently dropped: a scanner that cannot say what it declined to look at is
indistinguishable from one that found nothing.

**The two vacuity shapes, and why they are different failures.**

* `MISSING` — the module named does not have the attribute at all, under any binding. At RUNTIME
  this usually raises (`monkeypatch.setattr` and `mock.patch` both refuse an absent attribute), so
  it is loud *if the line runs*. It is silent when the site carries `raising=False` / `create=True`,
  and it is silent for every test the routine gate never selects (`slow`, `e2e`, a script). This
  scan sees all of them.
* `UNREAD` — the module HAS the name (it defines it, or imports it) but never READS it: no call, no
  reference, nothing that a patched value could reach. The stub installs cleanly, the code under
  test resolves the real symbol through its own direct import, and the test passes for the wrong
  reason. This is the shape that cost `ccd08003` four sites and a byte-identity test that compared
  two identical arms.

A module-level `__all__` entry counts as a READ: a re-export hub's whole job is to hand the name to
somebody else, and `hub.name` is exactly what such a patch is aimed at.

**The DEFINITION SITE is not automatically vacuous, and that is where a naive scan goes wrong.**
`main.launcher.ipc` defines `emit` and never calls it — yet patching `ipc.emit` works perfectly,
because its consumers hold the MODULE (`from main.launcher import ipc`) and reach `ipc.emit` at
call time. A patch on a name the owning module does not read is therefore only vacuous when NO
consumer reaches it that way; the whole point of the vacuity shape is that the consumer wrote
`from x import func` and now holds its own reference. So an `unread` candidate is settled by a
CONSUMER SEARCH over every non-test module in the tree, and the finding names the consumers it
did not find. Measured 2026-09-07: the search cleared 39 of 43 candidates.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

# Attribute names that mean "install a stub". `setitem` is deliberately ABSENT — it patches a dict
# entry, which this scan has no opinion about.
_SETATTR_ATTRS = frozenset({"setattr", "delattr"})
_PATCH_ATTRS = frozenset({"patch"})
_PATCH_OBJECT = "object"


@dataclass(frozen=True)
class Site:
    """One patch call: where it is, what it names, and what the scan made of it."""
    path: str          # repo-relative test file
    lineno: int
    call: str          # the spelling, e.g. "monkeypatch.setattr"
    target: str        # as written, e.g. "ppo_mod, 'shared_trunk_parameters'"
    module: str | None  # resolved dotted module path, when it is one of ours
    attr: str | None
    verdict: str       # "ok" | "missing" | "unread" | "skipped"
    reason: str = ""
    consumers: tuple[str, ...] = ()   # non-test modules that reach it as `<module>.<attr>`


def _dotted(node: ast.AST) -> str | None:
    """`a.b.c` -> "a.b.c"; anything else -> None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _absolute(package: str | None, node: ast.ImportFrom) -> str | None:
    """`from . import assembler` inside `agents.observation` -> "agents.observation".

    RELATIVE imports have to be resolved or the consumer search goes blind exactly where this tree
    uses them most — `state_encoder` reaches the assembler's `OBS_VERIFY` through
    `from . import assembler as _assembler`, and skipping `level > 0` reported that live consumer
    as absent.
    """
    if not node.level:
        return node.module
    if package is None:
        return None
    parts = package.split(".")
    if node.level - 1 > len(parts):
        return None
    base = ".".join(parts[:len(parts) - (node.level - 1)])
    return f"{base}.{node.module}" if node.module else base


def _import_bindings(tree: ast.AST, package: str | None = None) -> dict[str, str]:
    """Every name bound to a DOTTED PATH by an import, anywhere in the file.

    Function-local imports are included and share one namespace with the module-level ones. That is
    a deliberate over-approximation: this tree imports inside test functions constantly, and two
    different local aliases for the same name in one file would have to disagree for it to matter.
    """
    out: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom):
            mod = _absolute(package, n)
            if mod:
                for a in n.names:
                    out[a.asname or a.name] = f"{mod}.{a.name}"
    return out


def _local_module_aliases(tree: ast.AST, imports: dict[str, str]) -> dict[str, str]:
    """`m = some_module` / `m = pkg.mod` — the alias-of-an-alias indirection.

    `X = importlib.import_module("a.b")` is resolved too — it is how a test reaches a module whose
    plain import would be shadowed, and `main/launcher/dry_run_test.py` binds nine patch targets
    that way.

    Resolved the way `main/train/combination_checks_test.py` resolves its guard aliases, and for the
    same reason: an indirection whose NAME says nothing about what it points at is exactly where a
    scan goes blind.
    """
    out = dict(imports)
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call)
                and _dotted(n.value.func) in ("importlib.import_module", "import_module")
                and n.value.args and isinstance(n.value.args[0], ast.Constant)
                and isinstance(n.value.args[0].value, str)):
            out[n.targets[0].id] = n.value.args[0].value
    for _ in range(3):          # a fixed point in practice; bounded so a cycle cannot hang the gate
        grew = False
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)):
                continue
            src = _dotted(n.value)
            if src is None:
                continue
            head, _, rest = src.partition(".")
            if head in out:
                full = out[head] + (f".{rest}" if rest else "")
                if out.get(n.targets[0].id) != full:
                    out[n.targets[0].id] = full
                    grew = True
        if not grew:
            break
    return out


_DYNAMIC_LOOKUPS = frozenset({"getattr", "hasattr", "setattr", "delattr"})


class _ModuleFacts:
    """What a module's own source says about a NAME: is it bound here, and is it ever read?

    **A READ is a GLOBAL read, and nothing looser.** The only thing a patch on
    ``<module>.<attr>`` can reach inside that module is a load of ``attr`` as a global — an
    ``ast.Name`` in ``Load`` context. Three narrow additions, each because it really is such a
    load in disguise:

    * a string in ``__all__`` — a re-export hub's whole job is to hand the name out, and
      ``hub.name`` is exactly what a patch aimed at a hub means;
    * a string argument to ``getattr`` / ``hasattr`` / ``setattr`` / ``delattr`` — a genuine read
      the AST would otherwise see as prose;
    * ``globals()`` / ``vars()`` appearing anywhere — the module reaches its namespace by
      computation, so no static claim about which names it reads is safe. ``dynamic`` marks that,
      and every name in such a module is treated as read.

    An EARLIER draft of this counted every ``ast.Attribute``'s ``.attr`` and every string constant
    as a read. That is conservative in the safe direction — it can only suppress a finding, never
    invent one — but it is blind in a way that matters: ``self.foo`` is not a module global, and a
    docstring naming a function is not a call to it, so any module whose prose mentions its own
    symbol would clear every patch aimed at it. Measured 2026-09-07: exactly 2 of 397 sites rested
    on that looseness, and BOTH are cleared by the consumer search below on their own merits
    (``utils.team_loader.TeamLoader`` and ``obs_materializer.materialize_branches``, each reached
    by a DEFERRED import in its consumer). Precision here costs nothing and the consumer search is
    the real safety net, so the loose reads were dropped.
    """

    def __init__(self, tree: ast.AST):
        self.bound: set[str] = set()
        self.read: set[str] = set()
        self.dynamic = False
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.bound.add(n.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    self.bound.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    self.bound.add(a.asname or a.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    for sub in ast.walk(t):
                        if isinstance(sub, ast.Name):
                            self.bound.add(sub.id)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and isinstance(n.target, ast.Name):
                self.bound.add(n.target.id)
            elif isinstance(n, ast.Global):
                self.bound.update(n.names)

            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                self.read.add(n.id)
                if n.id in ("globals", "vars"):
                    self.dynamic = True
            elif isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
                self.read.update(c.value for c in ast.walk(n.value)
                                 if isinstance(c, ast.Constant) and isinstance(c.value, str))
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in _DYNAMIC_LOOKUPS and len(n.args) >= 2
                    and isinstance(n.args[1], ast.Constant)
                    and isinstance(n.args[1].value, str)):
                self.read.add(n.args[1].value)

    def reads(self, attr: str) -> bool:
        return self.dynamic or attr in self.read


def _qualified_reads(tree: ast.AST, package: str | None = None) -> set[str]:
    """Every `<module alias>.<attr>` in a file, expanded to its full dotted path.

    This is the READ a definition-site patch is aimed at: the consumer kept the MODULE and resolves
    the attribute at call time, so replacing the module's global reaches it.

    A DEFERRED import counts as the same reach: `from main.launcher.ipc import emit` written INSIDE
    a function body re-executes on every call, so it reads `ipc.emit` after the stub is installed.
    Only a MODULE-LEVEL `from M import A` takes its copy once, before any test runs — which is the
    vacuity shape this whole gate is about.
    """
    aliases = _local_module_aliases(tree, _import_bindings(tree, package))
    out: set[str] = set()
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for n in ast.walk(fn):
            if isinstance(n, ast.ImportFrom):
                mod = _absolute(package, n)
                if mod:
                    out.update(f"{mod}.{a.name}" for a in n.names)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Attribute):
            continue
        dotted = _dotted(n)
        if dotted is None:
            continue
        head, _, rest = dotted.partition(".")
        if head in aliases and rest:
            out.add(f"{aliases[head]}.{rest}")
        out.add(dotted)
    return out


def consumer_index(src_root: Path) -> dict[str, set[str]]:
    """`"pkg.mod.name"` -> the non-test modules that read it as a qualified attribute.

    Non-test on purpose: a patch exists to steer the CODE UNDER TEST, and a fixture reaching the
    same name proves nothing about whether the production path can see the stub.
    """
    out: dict[str, set[str]] = {}
    for p in sorted(src_root.rglob("*.py")):
        if any(part in ("poke_env", "rust_sim") for part in p.parts) or p.name.endswith("_test.py"):
            continue
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        name = str(p.relative_to(src_root)).removesuffix(".py").replace("/", ".")
        pkg = name.rsplit(".", 1)[0] if "." in name else ""
        for q in _qualified_reads(tree, pkg):
            out.setdefault(q, set()).add(name)
    return out


def _module_path(src_root: Path, dotted: str) -> Path | None:
    rel = dotted.replace(".", "/")
    for cand in (src_root / f"{rel}.py", src_root / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _follow_module_alias(src_root: Path, dotted: str) -> str:
    """`agents.training.snapshot_ladder.elo_mod` -> `agents.training.elo`.

    A test may reach a module through ANOTHER module's alias for it
    (`monkeypatch.setattr(snapshot_ladder.elo_mod, …)`). One hop is followed through the owning
    module's own import bindings, repeated to a fixed point and bounded so a cycle cannot hang.
    """
    for _ in range(3):
        if _module_path(src_root, dotted) is not None:
            return dotted
        parts = dotted.split(".")
        moved = False
        for cut in range(len(parts) - 1, 0, -1):
            head, rest = ".".join(parts[:cut]), parts[cut:]
            mpath = _module_path(src_root, head)
            if mpath is None:
                continue
            try:
                binds = _import_bindings(ast.parse(mpath.read_text()),
                                        head.rsplit(".", 1)[0] if "." in head else "")
            except SyntaxError:                                 # pragma: no cover - not in tree
                return dotted
            if rest[0] in binds:
                dotted = ".".join([binds[rest[0]]] + rest[1:])
                moved = True
            break
        if not moved:
            return dotted
    return dotted


def _split_target(src_root: Path, dotted: str) -> tuple[str, str] | None:
    """`"a.b.c"` -> ("a.b", "c") when `a.b` is a module of ours. Longest module prefix wins."""
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:cut])
        if _module_path(src_root, head) is not None:
            return head, ".".join(parts[cut:])
    return None


def _patch_calls(tree: ast.AST):
    """Every call that installs a stub, with the spelling it used."""
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr in _SETATTR_ATTRS:
            base = _dotted(f.value)
            if base and base.split(".")[-1] == "monkeypatch":
                yield n, f"{base}.{f.attr}", "object_or_string"
        elif isinstance(f, ast.Attribute) and f.attr == _PATCH_OBJECT:
            base = _dotted(f.value)
            if base and base.split(".")[-1] == "patch":
                yield n, f"{base}.object", "object"
        elif isinstance(f, ast.Attribute) and f.attr in _PATCH_ATTRS:
            yield n, f"{_dotted(f.value)}.patch", "string"
        elif isinstance(f, ast.Name) and f.id == "patch":
            yield n, "patch", "string"


def _patch_assignments(tree: ast.AST):
    """`mod.name = stub` — the HAND-ROLLED stub, save-and-restore around a `try`/`finally`.

    Not a `monkeypatch` call, and exactly as vacuous when it names the wrong module: two of the
    four sites `ccd08003` repointed are this shape, not the call shape
    (`scaffolding_test.py`'s `live_gauge_metrics` and `signal_metrics_test.py`'s
    `advantage_density_metrics`, both of which had to move from `ppo` to the mixin that ended up
    holding the read). A gate that only understood `monkeypatch.setattr` would have caught one of
    the four and reported the other two as clean.

    Only an assignment whose TARGET HEAD resolves to a module of ours is yielded; `self.x = 1` and
    every ordinary local attribute assignment resolve to nothing and never reach the scan.
    """
    for n in ast.walk(tree):
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if isinstance(t, ast.Attribute):
                yield n, t


def scan_file(path: Path, src_root: Path, module_cache: dict,
              consumers: dict[str, set[str]] | None = None) -> list[Site]:
    rel = str(path.relative_to(src_root.parent))
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:                                    # pragma: no cover - not in tree
        return [Site(rel, getattr(e, "lineno", 0) or 0, "-", "-", None, None, "skipped",
                     f"unparseable: {e}")]
    pkg = str(path.relative_to(src_root)).removesuffix(".py").replace("/", ".")
    pkg = pkg.rsplit(".", 1)[0] if "." in pkg else ""
    aliases = _local_module_aliases(tree, _import_bindings(tree, pkg))
    sites: list[Site] = []

    for call, spelling, form in _patch_calls(tree):
        kwargs = {k.arg for k in call.keywords if k.arg}
        soft = bool(kwargs & {"raising", "create", "new_callable", "autospec"})
        dotted = attr = None
        shown = ", ".join(ast.unparse(a) for a in call.args[:2])

        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(
                call.args[0].value, str):
            full = call.args[0].value
            split = _split_target(src_root, full)
            if split is None:
                sites.append(Site(rel, call.lineno, spelling, shown, None, None, "skipped",
                                  f"not a module of ours: {full!r}"))
                continue
            dotted, attr = split
        elif len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and isinstance(
                call.args[1].value, str):
            base = _dotted(call.args[0])
            if base is None:
                sites.append(Site(rel, call.lineno, spelling, shown, None, None, "skipped",
                                  "target object is an expression, not a name"))
                continue
            head, _, rest = base.partition(".")
            resolved = _follow_module_alias(
                src_root, aliases.get(head, head) + (f".{rest}" if rest else ""))
            if _module_path(src_root, resolved) is None:
                sites.append(Site(rel, call.lineno, spelling, shown, None, None, "skipped",
                                  f"target {resolved!r} is not a module of ours"))
                continue
            dotted, attr = resolved, call.args[1].value
        else:
            sites.append(Site(rel, call.lineno, spelling, shown, None, None, "skipped",
                              "no literal target"))
            continue

        sites.append(_judge(rel, call.lineno, spelling, shown, src_root, dotted, attr,
                            module_cache, consumers, soft))

    seen: set[tuple[str, str]] = set()
    for assign, target in _patch_assignments(tree):
        base = _dotted(target.value)
        if base is None:
            continue
        head, _, rest = base.partition(".")
        if head not in aliases:
            continue                      # a local object, not a module of ours — not our subject
        resolved = _follow_module_alias(src_root, aliases[head] + (f".{rest}" if rest else ""))
        if _module_path(src_root, resolved) is None:
            continue
        if (resolved, target.attr) in seen:
            continue                      # the save/restore pair names one target, not two
        seen.add((resolved, target.attr))
        sites.append(_judge(rel, assign.lineno, "<assignment>",
                            f"{base}.{target.attr} = …", src_root, resolved, target.attr,
                            module_cache, consumers, soft=False))
    return sites


def _judge(rel: str, lineno: int, spelling: str, shown: str, src_root: Path,
           dotted: str, attr: str, module_cache: dict,
           consumers: dict[str, set[str]] | None, soft: bool) -> Site:
    """The verdict for one resolved `(module, attr)` pair. Shared by both patch shapes."""
    if "." in attr:
        return Site(rel, lineno, spelling, shown, dotted, attr, "skipped",
                    "a CLASS attribute, not a module global")

    mpath = _module_path(src_root, dotted)
    assert mpath is not None
    if dotted not in module_cache:
        module_cache[dotted] = _ModuleFacts(ast.parse(mpath.read_text()))
    facts = module_cache[dotted]

    if not facts.reads(attr) and attr not in facts.bound:
        return Site(rel, lineno, spelling, shown, dotted, attr, "missing",
                    "soft (raising=False/create=True)" if soft else "")
    if not facts.reads(attr):
        reached = sorted((consumers or {}).get(f"{dotted}.{attr}", ()))
        return Site(rel, lineno, spelling, shown, dotted, attr,
                    "ok" if reached else "unread", "", tuple(reached))
    return Site(rel, lineno, spelling, shown, dotted, attr, "ok")


def scan_tree(src_root: Path) -> list[Site]:
    cache: dict = {}
    consumers = consumer_index(src_root)
    out: list[Site] = []
    for p in sorted(src_root.rglob("*_test.py")):
        if any(part in ("poke_env", "rust_sim") for part in p.parts):
            continue
        out.extend(scan_file(p, src_root, cache, consumers))
    return out


if __name__ == "__main__":                                        # report mode: the census
    import collections
    import sys

    from utils.paths import src_root as _src_root

    sites = scan_tree(_src_root())
    by = collections.Counter(s.verdict for s in sites)
    print(f"{len(sites)} patch sites  |  " + "  ".join(f"{k}={v}" for k, v in sorted(by.items())))
    for want in ("missing", "unread"):
        rows = [s for s in sites if s.verdict == want]
        if not rows:
            continue
        print(f"\n=== {want.upper()} ({len(rows)}) ===")
        for s in rows:
            print(f"  {s.path}:{s.lineno}  {s.call}({s.target})  -> {s.module}.{s.attr}"
                  + (f"   [{s.reason}]" if s.reason else ""))
    if "-v" in sys.argv:
        print("\n=== SKIPPED reasons ===")
        for k, v in collections.Counter(s.reason for s in sites
                                        if s.verdict == "skipped").most_common():
            print(f"  {v:5d}  {k}")
