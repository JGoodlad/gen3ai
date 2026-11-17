"""``plan --check-pins``: each arm's OWN pin is judged, not only named.

The chain file gives every arm its own full 40-hex ``pin`` (``spec.py``), and the kit's GO-time pre-checks judge it
(``launch_x35.sh``: the required commits, ``kit pin-check``, ``checkargs --pin``, the launcher ``--dry-run``). Those run
when the chain REACHES the arm, which can be days after the file was written, and a refusal there HOLDs the chain with
the GPU idle. This module runs the cheap half of that at PLAN time, in three layers, cheapest first:

1. **It resolves**: the pin (an arm's) or ``require_commit`` (a window's) is a commit in the main checkout.
2. **It is on main, or the chain file says why not.** An arm's pin is an ancestor of ``main_ref`` (``origin/main``),
   or it is named in the chain file's top-level ``off_main_pins`` ({pin: reason}) AND some branch or tag holds it
   (a commit nothing refers to is garbage-collectable mid-campaign). Pins ARE usually off main here: the X35 pins
   are a side pin plus a commit or two, kept by a tag. The allowance is chain-level on purpose: a new field on an
   arm would change ``spec.as_dict`` and so every STARTED item's saved snapshot, and a restart of a live supervisor
   would refuse the whole chain (``chain._load_state``: a started item never changes).
3. **The pinned parser accepts the arm's argv**: ``python -m main.checkargs --pin <pin> --argv <the kit's describe
   argv>`` run from the main checkout, exactly the kit's pre-check 5 (``launch_x35.sh``). Nothing is re-implemented:
   ``checkargs`` materialises the pin with ``git archive`` (no worktree) and judges the argv by the PIN's own
   ``build_parser()``. A verdict from a static scan or "unavailable" is a refusal here, never a pass: the launcher
   proceeds on those with a warning, and a plan that proceeds on them has judged nothing.

WHAT IS DELIBERATELY NOT CHECKED: that the files the argv names exist at the pin. A repo-relative path in an argv
(``--trainee-teams designs/…/sfv2_0070.txt``) is read from the LAUNCHING checkout (the child runs with the launcher's
cwd, ``src/main/launcher/CLAUDE.md``), not from the pin: X52's practised teams are absent at its pin ``f4b25238`` and
present on main, and X52 launched there. A ``git cat-file -e <pin>:<path>`` would refuse a launch that works.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import List, Mapping, Optional

from main.launcher.pinned_argv import ParseReport
from main.ops.arm_chain.launch import Describe
from main.ops.arm_chain.runner import Runner, cpu_env
from main.ops.arm_chain.spec import ArmItem, ChainSpec, WindowItem

DEFAULT_MAIN_REF = "origin/main"
#: The kit's own bound on its ``checkargs --pin`` pre-check (``launch_x35.sh``).
CHECKARGS_TIMEOUT_S = 600
_MODE = re.compile(r"\[mode=(\w+)")


def _git(repo: str, *args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def is_commit(repo: str, rev: str) -> bool:
    return _git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").returncode == 0


def who(it: ArmItem) -> str:
    """How a problem line names an arm: its key, arm, seed and the pin the line is about."""
    return f"{it.key} (ARM {it.arm} s{it.seed}) at pin {it.pin}"


def _off_main_problem(spec: ChainSpec, it: ArmItem, main_ref: str) -> Optional[str]:
    """None when the arm's pin is on main, or is allowed off it and held by a ref; else why not."""
    p = _git(spec.repo, "merge-base", "--is-ancestor", it.pin, main_ref)
    if p.returncode == 0:
        return None
    if p.returncode != 1:
        return f"cannot tell whether the pin is an ancestor of {main_ref}: git merge-base exit {p.returncode}: {p.stderr.strip()}"
    why = spec.off_main_pins.get(it.pin)
    if why is None:
        return (f"is not an ancestor of {main_ref}, and the chain file's off_main_pins does not name it "
                f"(a pin kept off main is allowed only with a written reason: {{\"{it.pin}\": \"<why>\"}})")
    held = _git(spec.repo, "for-each-ref", "--contains", it.pin, "--count=1", "--format=%(refname)",
                "refs/heads", "refs/tags", "refs/remotes")
    if not held.stdout.strip():
        return (f"is allowed off main ({why!r}) but no branch or tag holds it: git can garbage-collect a commit "
                "nothing refers to, in the middle of the campaign (tag it)")
    return None


def check_pins(spec: ChainSpec, main_ref: str = DEFAULT_MAIN_REF) -> List[str]:
    """Layers 1 and 2 (git only): the problems found, each naming the arm (or window), the pin and the reason."""
    bad: List[str] = []
    main_ok = is_commit(spec.repo, main_ref)
    if spec.arms() and not main_ok:
        bad.append(f"chain: cannot judge whether the arms' pins are on main: {main_ref} is not a commit in {spec.repo} "
                   "(fetch it, or name the ref with --main-ref)")
    for it in spec.items:
        if isinstance(it, WindowItem):
            if not is_commit(spec.repo, it.require_commit):
                bad.append(f"{it.key}: {it.require_commit} is not a commit in {spec.repo}")
            continue
        if not is_commit(spec.repo, it.pin):
            bad.append(f"{who(it)}: {it.pin} is not a commit in {spec.repo}")
        elif main_ok:
            why = _off_main_problem(spec, it, main_ref)
            if why:
                bad.append(f"{who(it)}: {why}")
    return bad


def judge_checkargs(rc: int, text: str) -> Optional[str]:
    """Layer 3's verdict on one ``checkargs --pin`` run (its exit code and transcript): None = the PINNED parser
    answered authoritatively and accepted the argv; else the reason."""
    if rc != 0:
        marks = []
        for line in text.splitlines():
            line = line.strip()
            if "✗" in line and line not in marks:
                marks.append(line)
        reason = "; ".join(m[:200] for m in marks[:2]) or " ".join(text.split())[-200:] or "no output"
        return f"main.checkargs --pin exits {rc}: {reason}"
    for mode in _MODE.findall(text):
        if mode not in ParseReport.AUTHORITATIVE_MODES:
            return (f"main.checkargs --pin exits 0 but the pinned parser answered by mode={mode}, which is not "
                    "authoritative: the argv is UNVALIDATED at this pin")
    return None


def check_argv_at_pin(spec: ChainSpec, it: ArmItem, d: Describe, runner: Runner, base_env: Mapping[str, str],
                      ) -> List[str]:
    """Layer 3: the kit's describe argv for this arm, judged by the arm's own pin's parser, run the way the kit runs it
    (``cd <main checkout> && PYTHONPATH=<main>/src python -m main.checkargs --pin <pin> --argv "<argv>"``)."""
    out = os.path.join(spec.work_dir, f"plan_checkargs.{it.arm}.{it.seed}.txt")
    argv = [spec.python, "-m", "main.checkargs", "--pin", it.pin, "--argv", shlex.join(d.argv)]
    rc = runner.run(argv, cwd=spec.repo, env=cpu_env(base_env, spec.repo, models_dir=spec.models_dir),
                    timeout_s=CHECKARGS_TIMEOUT_S, out_path=out)
    try:
        text = open(out, errors="replace").read()
    except OSError:
        text = ""
    why = judge_checkargs(rc, text)
    return [] if why is None else [f"{who(it)}: {why} (transcript {out})"]

