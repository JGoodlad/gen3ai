"""THE SLOW GOLDENS THE MODEL'S NUMERICS MOVE — one registry, so re-recording a ROUTINE golden forces them.

WHY (2026-11-10). A golden pinned in a ``slow`` file is not run by the ship gate (``-m "not slow and not e2e"``); the
gate adds "the slow tests of code you touched", chosen by hand. A commit that changes what the model computes touches
``src/agents/model/`` and re-records the routine goldens it turns red (the K9 learner golden, the played-bits
sentinel), and nothing made it run a slow file in another package that pins the same numerics. Twice that left
``rust_rollout/overlap_integration_test.py`` red on main: ``d4af9f1e2`` (2026-10-20, found the next day) and then
config v165 … v168 (``83272a8f9``, ``7a0153b46``, ``566d878b2``, ``3b1733f50``; 2026-11-06 … 11-09, found 2026-11-10
by a slow-tier sweep). The slow record could not show it either: a row is what the test did the last time it RAN.

THE RULE, mechanical. Every registered slow file carries a stamp, ``NUMERICS_ANCHOR``: the :func:`anchor` it was
last RUN under. The routine test (``numerics_goldens_test.py``, in the ship gate) fails while any stamp differs from
the current anchor. So the commit that re-records a routine golden, or raises ``OBS_SEMANTICS_VERSION``, cannot pass
its gate until each registered slow file has been run at the new tree, re-recorded where it moved (by that file's
own procedure, with the cause), and re-stamped — and the re-stamp is an edit to the slow file, which is what "the
slow tests of code you touched" keys on.

    python -m agents.training.numerics_goldens     # the current anchor, each file's stamp, what is stale

WHAT THE ANCHOR READS (files only, no torch): the last row of ``OBS_SEMANTICS_HISTORY`` (the config version at which
the MEANING of a model input last changed); every torch build's ``post_params_sha256`` (production and the fp32
opt-out) in ``learner_golden.json``; the sentinel's ``PACKED_DIGEST`` / ``PLAYED_DIGEST``. The semantics marker is
in it because the routine goldens are small samples: config v166 (``7a0153b46``: our stats floor the EV, Leech Seed
drains an integer) moved the overlap file's goldens and NEITHER routine golden (one row of its first rollout holds a
Leech-Seeded active; the K9 golden's buffer holds none; measured 2026-11-10). All four commits of v165 … v168
raised the marker. Neither trigger is enough alone: T62 (``bd3126e0``) changed a solve with no marker and moved both
routine goldens. NOT ``MODEL_CONFIG_VERSION``: it is also bumped for a new per-run setting that changes no numerics
(v169, v170), and each false alarm would cost a run of every registered file (the reveal file is 30 minutes).

BORN LIVE: a ``slow`` test that reads a module-level sha256 constant must be registered here or exempted with a
reason (:func:`unregistered`), so the next slow golden cannot be added outside the rule.

LIMITS. The stamp is a STATEMENT that the file was run; nothing proves the run (the slow record's row for the file
is the evidence). Not seen: a change that moves a slow golden while no routine golden moves and the semantics
marker is not raised — an ENGINE fix that changes one of these games, or a model-internal change whose author
skipped the marker's by-hand rule (``model_version/constants.py``). The slow tier's sweep is what finds those.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

from utils.paths import repo_path, src_path

#: slow test file (repo-relative) -> the digest constants in it that the model's numerics move
SLOW_GOLDENS: Dict[str, Tuple[str, ...]] = {
    "src/agents/training/rust_rollout/overlap_integration_test.py": ("OFF_GOLDEN", "PRE_X14_GOLDEN_FP64"),
    "src/main/h2h/play_reveal_integration_test.py": ("OFF_BITS", "OFF_OUTCOME_ALL"),
}
#: slow test file -> why its sha256 constant is NOT a numerics golden (empty today)
EXEMPT: Dict[str, str] = {}

STAMP = "NUMERICS_ANCHOR"
VERSION_CONSTANTS = "src/agents/model/model_version/constants.py"
LEARNER_GOLDEN = "src/agents/training/learner_golden.json"
SENTINEL = "src/agents/training/played_bits_sentinel_test.py"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def constants(rel: str) -> Dict[str, Any]:
    """Module-level ``NAME = <literal>`` assignments of a repo file, read from its source (never imported)."""
    out: Dict[str, Any] = {}
    for node in ast.parse(repo_path(rel).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.value
    return out


def semantics_version() -> int:
    """``OBS_SEMANTICS_VERSION``, read from the source: the config version of ``OBS_SEMANTICS_HISTORY``'s last row."""
    for node in ast.parse(repo_path(VERSION_CONSTANTS).read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "OBS_SEMANTICS_HISTORY" for t in node.targets):
            try:
                version = ast.literal_eval(node.value)[-1][0]
            except (ValueError, IndexError, TypeError) as e:
                raise ValueError(f"OBS_SEMANTICS_HISTORY is no longer a literal of rows in {VERSION_CONSTANTS}") from e
            if type(version) is not int:
                raise ValueError(f"OBS_SEMANTICS_HISTORY's last row does not start with an int: {version!r}")
            return version
    raise ValueError(f"no OBS_SEMANTICS_HISTORY in {VERSION_CONSTANTS}")


def anchor() -> str:
    """16 hex chars over the obs-semantics marker and the routine goldens that pin the numerics (module docs)."""
    entries = json.loads(repo_path(LEARNER_GOLDEN).read_text())["entries"]
    parts: List[str] = [f"obs semantics v{semantics_version()}"]
    for key in sorted(entries):
        parts += [key, entries[key]["post_params_sha256"], entries[key]["fp32_opt_out"]["post_params_sha256"]]
    sentinel = constants(SENTINEL)
    parts += [sentinel["PACKED_DIGEST"], sentinel["PLAYED_DIGEST"]]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def stale() -> List[str]:
    """One line per registered file whose stamp is not the current anchor, or whose registered constant is gone."""
    now, out = anchor(), []
    for rel, names in sorted(SLOW_GOLDENS.items()):
        consts = constants(rel)
        missing = [n for n in names if not _is_sha256(consts.get(n))]
        if missing:
            out.append(f"{rel}: registered golden(s) {missing} are not module-level sha256 constants")
        if consts.get(STAMP) != now:
            out.append(f"{rel}: {STAMP} is {consts.get(STAMP)!r}, the current anchor is {now!r}")
    return out


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_slow_mark(node: ast.AST) -> bool:
    return "pytest.mark.slow" in ast.unparse(node)


def slow_read_digests(rel: str) -> List[str]:
    """The module-level sha256 constants of a test file that a ``slow`` test reads: all that any function reads when
    the module's ``pytestmark`` declares ``slow``, else those read inside a function decorated ``slow``."""
    tree = ast.parse(repo_path(rel).read_text())
    digests = {k for k, v in constants(rel).items() if _is_sha256(v)}
    if not digests:
        return []
    whole = any(isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "pytestmark" for t in n.targets)
                and _is_slow_mark(n.value) for n in tree.body)
    read = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                whole or any(_is_slow_mark(d) for d in fn.decorator_list)):
            read |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} & digests
    return sorted(read)


def unregistered() -> List[str]:
    """``path: NAMES`` for every slow-read sha256 constant that is neither registered nor exempt."""
    root, out = src_path(), []
    for p in sorted(root.rglob("*_test.py")):
        rel = "src/" + p.relative_to(root).as_posix()
        if "mark.slow" not in p.read_text() or rel in EXEMPT:
            continue
        loose = [n for n in slow_read_digests(rel) if n not in SLOW_GOLDENS.get(rel, ())]
        if loose:
            out.append(f"{rel}: {', '.join(loose)}")
    return out


def main() -> int:
    print(f"numerics anchor (obs semantics v{semantics_version()} + routine goldens): {anchor()}")
    for rel in sorted(SLOW_GOLDENS):
        print(f"  {constants(rel).get(STAMP)}  {rel}  [{', '.join(SLOW_GOLDENS[rel])}]")
    problems = [f"STALE: {s}" for s in stale()] + [f"UNREGISTERED slow golden: {u}" for u in unregistered()]
    print("\n".join(problems) if problems else "every registered slow golden is stamped with the current anchor")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
