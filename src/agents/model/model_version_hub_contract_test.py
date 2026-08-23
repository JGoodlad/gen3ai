"""The CONTRACT the `agents.model.model_version` decomposition rests on.

On 2026-08-23 `model_version.py` (2,000 lines — exactly AT the size gate's hard bound) became a
PACKAGE whose `__init__.py` is a pure re-export hub. Four things make that safe rather than
merely tidier, and none of them is self-enforcing:

**1. The hub still exports everything it used to.** ~48 modules and tests do
`from agents.model.model_version import <name>`. The list below is the module-level *definitions*
of the file as it stood at the commit before the split, recovered by AST rather than typed by
hand. It may GROW; a name leaving it is a deliberate deletion that should fail here first.

⚠️ Pure import BINDINGS are deliberately not pinned (`model_version.json`, `.math`, `.dataclass`,
…). They were reachable as module attributes before and are not all now; nothing in the tree
reads them that way, and pinning them would freeze one module's private imports as another's
public surface. (Same rule as `main/prober/hub_contract_test.py`.)

**2. `ModelVersion` is assembled from MIXINS, so it has a BASE LIST** — and a base list can lose
an entry without any import failing. The class would still construct, still round-trip through
JSON, and simply stop having `check_compatible`, which is the project's load-bearing safety net.
Pinned directly below.

**3. No submodule imports its own hub.** The hub imports the leaves; a leaf importing the hub back
closes a cycle whose symptom is an `AttributeError` on a name that plainly exists.

**4. Every submodule stands up on its own.** A module that only imports once its siblings have is
a cycle that has not bitten yet.

Milliseconds — attribute lookups and an AST walk. No marker; runs in the fast inner loop.
"""
import ast
import dataclasses
import importlib
import pathlib

import pytest

import agents.model.model_version as hub

_PKG = "agents.model.model_version"
_DIR = pathlib.Path(hub.__file__).parent

# Every module-level name `src/agents/model/model_version.py` DEFINED at ea5cd98, the commit
# before the decomposition. Recovered by AST, transcribed verbatim.
_PRE_SPLIT = (
    "ARCH_SIGNATURE", "MIGRATION_FLOOR", "MODEL_CONFIG_VERSION", "ModelVersion",
    "ModelVersionError", "SIGNATURE_FIRST_VERSION", "_BELIEF_GRAD_MODE_EFFECT",
    "_REWARD_FIELD_FLAGS", "_REWARD_IMMUTABLE_FIELDS", "_migrate_config", "_reward_flag_repr",
)

# The mixins `ModelVersion` is assembled from, innermost last. Dropping one is silent.
_BASES = ("ModelVersionConstruction", "ModelVersionCompatibility", "ModelVersionResumeChecks",
          "ModelVersionFields")


@pytest.mark.parametrize("name", _PRE_SPLIT)
def test_the_hub_still_exports_every_pre_split_name(name):
    assert hasattr(hub, name), (
        f"`{_PKG}.{name}` no longer resolves. The hub's whole job is that the decomposition "
        f"changed no import path — re-export it from whichever module now owns it, or delete "
        f"this entry deliberately if the name is genuinely gone."
    )


def test_model_version_carries_every_gate_family():
    """`ModelVersion` is fields + one mixin per family; a family can vanish from the bases silently.

    This is the base-list cost of splitting a 1,510-line CLASS. Nothing else in the tree fails at
    import time if `ModelVersionCompatibility` drops out — a resume would simply stop being gated,
    which is the exact failure the version system exists to prevent.
    """
    names = [c.__name__ for c in hub.ModelVersion.__mro__]
    for base in _BASES:
        assert base in names, (
            f"`{base}` is gone from `ModelVersion`'s bases (MRO is {names}). A dropped mixin "
            f"removes its gates without breaking a single import.")
    for method in ("from_layout_and_policy_kwargs", "to_json", "from_json_file",
                   "check_compatible", "check_opponent_compatible", "check_vf_coef",
                   "check_belief_grad_mode", "check_value_from_dist", "check_value_tail_weight",
                   "check_value_dist", "check_reward_config"):
        assert callable(getattr(hub.ModelVersion, method, None)), (
            f"`ModelVersion.{method}` is gone — a mixin dropped out of the base list in "
            f"`model_version/spec.py`, and nothing else in the tree would fail at import time.")


def test_the_field_block_still_lives_on_the_dataclass():
    """The fields are declared on `ModelVersionFields` and INHERITED. Their ORDER is the
    constructor's positional order and `asdict()`'s key order, so it is part of the contract."""
    fields = dataclasses.fields(hub.ModelVersion)
    assert len(fields) > 100, f"only {len(fields)} fields — the field block did not come across"
    assert fields[0].name == "config_version" and fields[1].name == "arch_signature", (
        "the first two fields are the schema identity and must stay first")
    assert dataclasses.is_dataclass(hub.ModelVersion)


def test_the_hub_re_exports_from_every_module():
    """A module the hub forgot is a module no historical import path can reach.

    Not a size check — a COVERAGE one: every `.py` beside the hub must be named in one of its
    import blocks, so adding a module without wiring it up fails here rather than at the first
    caller that happens to want a name from it. `fields` / `construct` / `compat` /
    `resume_checks` are reached transitively through `spec`, but the hub names them anyway so
    this check stays a flat statement about the directory.
    """
    modules = {p.stem for p in _DIR.glob("*.py") if p.stem != "__init__"}
    src = (_DIR / "__init__.py").read_text(encoding="utf-8")
    indirect = {"construct", "compat", "resume_checks"}   # reached through `spec`
    missing = sorted(m for m in modules - indirect if f"{_PKG}.{m} import" not in src)
    assert not missing, f"the model_version hub re-exports nothing from: {missing}"


def test_no_submodule_imports_its_own_hub():
    """The cycle guard. Submodules may import each other; none may import the package itself."""
    offenders = []
    for path in sorted(_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = " ".join(a.name for a in node.names)
            if mod == _PKG:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"submodule(s) import the `{_PKG}` hub back: {offenders}. That closes an import cycle — "
        f"the hub imports the submodules — and its symptom is an AttributeError on a name that "
        f"plainly exists. Import from the sibling module that OWNS the name instead."
    )


def test_every_submodule_imports_on_its_own():
    for path in sorted(_DIR.glob("*.py")):
        if path.name != "__init__.py":
            importlib.import_module(f"{_PKG}.{path.stem}")


def test_the_pre_floor_history_archive_survived_the_move():
    """`_migrate_config`'s PRE-FLOOR HISTORY block is a DELIBERATE archive, not dead prose.

    It records what each deleted `if version < N` branch injected or popped, which is the only
    surviving statement of what an archived checkpoint's config meant. A decomposition is exactly
    the kind of change that quietly drops a 200-line comment, so this asserts it is still there
    and still spans the range it claims to.
    """
    src = (_DIR / "migrations.py").read_text(encoding="utf-8")
    assert "PRE-FLOOR MIGRATION HISTORY" in src
    for marker in ("#   v2:", "#   v50:", "#   v95:", "VERSION-INDEPENDENT SANITIZERS",
                   "POST-FLOOR MIGRATION BRANCHES"):
        assert marker in src, f"the migration archive lost {marker!r}"
