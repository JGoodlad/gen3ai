"""What a training child needs from ``deps/pokemon-showdown`` - and the preflight that checks it.

WHY THIS EXISTS (2026-10-09, ``designs/research_state/measurements/gpu_checks_endstate_2026-10-09/`` F-GE-4).
The launcher's pinned worktree links ``deps/pokemon-showdown`` to the LAUNCHING checkout's submodule
(``main/launcher/worktree.py``). From an agent worktree whose submodule was never initialised (or never built)
that is an empty placeholder, and the run died minutes after the launch, in the child's team validation:
:class:`utils.teambuilder.Teambuilder` runs ``utils/bridge/validate_team.js`` under ``node``, which does
``require('<repo>/deps/pokemon-showdown')`` and swallows ``Cannot find module`` into ``{"valid": False}``, so the
first symptom is "No valid teams found" after the model and the T2 service are already up. The first P attempt of
the end-state GPU check died this way.

The training child needs NOTHING else from that directory (the Rust core embeds its own data; the live front end and
the anchors' node server are separate entry points with their own errors). So the closed list below is the files
:func:`~utils.bridge.team_validator.validate_teams_locally` reads, MEASURED by tracing ``require`` + ``readFileSync``
across ``new TeamValidator('gen3ou').validateTeam(...)`` (``showdown_deps_test.py`` repeats that trace and fails if a
listed file is no longer read, so the list cannot hold a phantom). It is deliberately the SENTINELS of each of the three
things that can be wrong, not every module of the dex:

    checkout   the submodule is not checked out        (``git submodule update --init``)
    build      ``dist/`` is missing or a dangling link (``node build`` / a worktree's link to main's ``dist``)
    modules    ``node_modules/`` is missing            (``npm ci`` / a worktree's link to main's ``node_modules``)

``./scripts/bootstrap.sh --skip-env`` repairs all three in a linked worktree (it initialises the submodule and links
main's ``dist`` and ``node_modules``) without touching the shared conda env.

:func:`check` is pure filesystem (``os.path.exists`` follows symlinks, so a dangling ``dist`` link is "missing"); the
launcher asks it before any worktree exists (``main/launcher/submodule_gate.py``) and ``--dry-run`` prints the same
verdict.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

#: Where the submodule lives, relative to a checkout root.
SUBMODULE_RELPATH = os.path.join("deps", "pokemon-showdown")

#: The three things that can be wrong, in the order a fix has to be applied.
KIND_CHECKOUT = "checkout"
KIND_BUILD = "build"
KIND_MODULES = "modules"

_WHAT = {
    KIND_CHECKOUT: ("the submodule was never initialised in this checkout (deps/pokemon-showdown is an empty "
                    "placeholder)"),
    KIND_BUILD: "the submodule has no built dist/ (not built, or a dangling link)",
    KIND_MODULES: "the submodule has no node_modules/ (not installed, or a dangling link)",
}


@dataclass(frozen=True)
class Need:
    """One file the team validation reads."""
    rel: str            # relative to ``deps/pokemon-showdown``
    kind: str           # KIND_*
    why: str


#: The closed list. Every entry is read by ``new TeamValidator('gen3ou').validateTeam(Teams.import(...))`` through
#: ``require(<dir>)`` (``showdown_deps_test.py`` traces it). Add an entry only with the trace that shows the read.
NEEDS: Tuple[Need, ...] = (
    Need("package.json", KIND_CHECKOUT, "require(<dir>) resolves `main` (dist/sim/index.js) from it"),
    Need("dist/sim/index.js", KIND_BUILD, "the package's `main`"),
    Need("dist/sim/team-validator.js", KIND_BUILD, "TeamValidator"),
    Need("dist/sim/dex.js", KIND_BUILD, "the Dex the validator builds"),
    Need("dist/config/formats.js", KIND_BUILD, "the format table `gen3ou` is looked up in"),
    Need("dist/data/pokedex.js", KIND_BUILD, "species data"),
    Need("dist/data/moves.js", KIND_BUILD, "move data"),
    Need("dist/data/learnsets.js", KIND_BUILD, "learnsets (move legality)"),
    Need("dist/data/formats-data.js", KIND_BUILD, "tiers / formats-data"),
    Need("dist/data/mods/gen3/scripts.js", KIND_BUILD, "the gen3 mod `[Gen 3] OU` resolves through"),
    Need("dist/data/mods/gen3/rulesets.js", KIND_BUILD, "the gen3 mod's rulesets"),
    Need("node_modules/sql-template-strings/index.js", KIND_MODULES, "required by dist/lib/sql.js on the sim's import"),
    Need("node_modules/ts-chacha20/build/src/chacha20.js", KIND_MODULES, "required by dist/sim/prng.js"),
)


@dataclass(frozen=True)
class DepsVerdict:
    """The preflight's answer for one ``deps/pokemon-showdown`` directory."""
    root: str                           # the directory checked (``<checkout>/deps/pokemon-showdown``)
    missing: Tuple[Need, ...]           # NEEDS that are not there, in list order
    node: Optional[str]                 # the `node` executable the child will run, or None

    @property
    def refused(self) -> bool:
        return bool(self.missing) or self.node is None

    @property
    def kinds(self) -> Tuple[str, ...]:
        """The distinct KIND_* of what is missing, checkout first (fix order)."""
        have = {n.kind for n in self.missing}
        return tuple(k for k in (KIND_CHECKOUT, KIND_BUILD, KIND_MODULES) if k in have)

    def fix(self) -> List[str]:
        """The command(s) that repair it, naming the checkout. Empty when nothing is wrong."""
        out: List[str] = []
        checkout = os.path.dirname(os.path.dirname(self.root))
        link_note = ("(in a linked worktree it links the main checkout's built dist/ and node_modules/, and it "
                     "leaves the shared conda env alone)")
        if KIND_CHECKOUT in self.kinds:
            out.append(f"in {checkout}: `git submodule update --init`, then `./scripts/bootstrap.sh --skip-env` "
                       f"{link_note}")
        elif self.missing:
            out.append(f"in {checkout}: `./scripts/bootstrap.sh --skip-env` {link_note}")
            if KIND_BUILD in self.kinds:
                out.append(f"or build it: `cd {self.root} && node build`")
            if KIND_MODULES in self.kinds:
                out.append(f"or install it: `cd {self.root} && npm ci`")
        if self.node is None:
            out.append("put `node` on PATH (team validation runs it)")
        return out

    def lines(self) -> List[str]:
        """``--dry-run`` / refusal text. The first line is the verdict."""
        if not self.refused:
            return [f"showdown deps : ✓ {self.root} serves team validation "
                    f"({len(NEEDS)} required files present, node {self.node})"]
        if self.missing:
            names = ", ".join(n.rel for n in self.missing[:4]) + (
                f" (+{len(self.missing) - 4} more)" if len(self.missing) > 4 else "")
            head = (f"showdown deps : ✗ REFUSED - {self.root} cannot serve team validation: "
                    f"{len(self.missing)} of {len(NEEDS)} required files missing ({names})")
        else:
            head = "showdown deps : ✗ REFUSED - `node` is not on PATH (team validation runs it)"
        lines = [head]
        for k in self.kinds:
            lines.append(f"                {_WHAT[k]}")
        lines.append("                the child would die minutes after the launch in team validation "
                     "(\"Cannot find module .../deps/pokemon-showdown\")")
        for f in self.fix():
            lines.append(f"                fix: {f}")
        return lines


def check(root: str, *, which: Callable[[str], Optional[str]] = shutil.which) -> DepsVerdict:
    """Check ``root`` (a ``.../deps/pokemon-showdown`` directory) against :data:`NEEDS`, and ``node`` on PATH.

    ``os.path.exists`` follows symlinks on purpose: a linked worktree reaches ``dist/`` and ``node_modules/`` through
    links to the main checkout, and a link whose target is gone must read as missing."""
    missing = tuple(n for n in NEEDS if not os.path.exists(os.path.join(root, *n.rel.split("/"))))
    return DepsVerdict(root=root, missing=missing, node=which("node"))


def source_dir(checkout_root: str) -> str:
    """``<checkout_root>/deps/pokemon-showdown`` - the one place the path is spelled."""
    return os.path.join(checkout_root, SUBMODULE_RELPATH)


def write_stub_checkout(checkout_root: str) -> str:
    """TEST SUPPORT: make ``<checkout_root>/deps/pokemon-showdown`` satisfy :func:`check` with empty files, so a test
    that fakes a repo root for the launcher's pin logic (a throwaway git repo) does not trip the preflight it is not
    about. Returns the directory. Never used outside ``*_test.py``."""
    root = source_dir(checkout_root)
    for n in NEEDS:
        path = os.path.join(root, *n.rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8"):
            pass
    return root
