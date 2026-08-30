"""The ai_v12 CLEAN-WORLD launch runbook's commands must still LAUNCH.

`designs/ai_v12/launch_runbook.md` is the document a generation-scale launch is executed from, and
its four argvs are ~110 flags each. A recorded command outlives the flags in it: argparse reports
only the FIRST unrecognized flag, so relaunching a stale argv is a launch-crash-fix loop at ~40 s
and one stray run dir per dead flag — the exact failure `python -m main.checkargs` was built to
answer offline. A runbook that has silently rotted reads identical to one that has not.

So the runbook is checked the way `checkargs` checks a run's recorded command: parse the shell
block OUT OF THE DOCUMENT, compose the five argvs the same way the document does, and run every
flag against the LIVE parser plus the `flag_registry` `requires` graph. Deleting a flag anywhere in
the tree fails here with the runbook named, instead of at a launch two days later.

Two things this deliberately does NOT do, because it cannot without becoming a different test:
it does not run training (a smoke is the runbook's own §5, and its evidence is recorded there), and
it does not check the VALUES are still the right experiment — only that the command is accepted.

Run:
    python -m pytest src/main/launch_runbook_test.py -q
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import re

import pytest

from main.checkargs import check
from utils.paths import repo_path

RUNBOOK = repo_path("designs", "ai_v12", "launch_runbook.md")

#: The shell variables the runbook defines, and the arms it composes from them. Mirrors §2 — if the
#: document grows an arm, this list is where the test learns about it.
_ARMS = {
    "cw1_sparse":     ("TRAIN", "ARCH", "CLEAN"),
    "cw2_self_phi":   ("TRAIN", "ARCH", "CLEAN"),
    "cw3_frozen_phi": ("TRAIN", "ARCH", "CLEAN"),
    "pt_sparse":      ("TRAIN", "ARCH", "CLEAN"),
    "pt_shaped":      ("TRAIN", "ARCH", "SHAPED"),
}
#: The per-arm tail — the flags that are the whole experiment, so they are spelled out here rather
#: than scraped: a parser that stopped accepting them is exactly what this file exists to catch.
_TAILS = {
    "cw2_self_phi":   ["--win-prob-pbrs-coef", "0.3"],
    "cw3_frozen_phi": ["--win-prob-pbrs-coef", "0.3",
                       "--win-prob-pbrs-source", "models/x/final_model.zip"],
}


def _shell_vars() -> dict:
    """The `NAME="…"` assignments in the runbook's bash block, with `\\`-continuations joined."""
    text = RUNBOOK.read_text()
    body = re.sub(r"\\\n\s*", " ", text)                       # join line continuations
    out = {}
    for name, val in re.findall(r'^([A-Z_]+)="([^"]*)"', body, flags=re.MULTILINE):
        out[name] = val.split()
    return out


def _argv(arm: str) -> list:
    v = _shell_vars()
    argv = ["--run-name", arm, "--steps", "25000000"]
    for part in _ARMS[arm]:
        argv += v[part]
    return argv + _TAILS.get(arm, [])


def test_the_runbook_still_defines_every_block_its_commands_reference():
    v = _shell_vars()
    for name in ("ARCH", "TRAIN", "CLEAN", "SHAPED"):
        assert name in v and v[name], f"the runbook's ${name} block went missing or empty"


@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_every_runbook_argv_is_still_launchable(arm):
    """The whole point: `checkargs` clean, exit-0-equivalent, on the live parser."""
    res = check(_argv(arm))
    assert not res["unknown"], (
        f"{arm}: designs/ai_v12/launch_runbook.md names flags the parser no longer knows: "
        f"{[f for f, _ in res['unknown']]}. Update the runbook (a deleted flag may have a "
        f"replacement — dropping it silently would change the arm).")
    assert not res["unsatisfiable"], (
        f"{arm}: the runbook's flag COMBINATION is refused by flag_registry's requires graph: "
        f"{res['unsatisfiable']}. That crash lands inside Gen3FeaturesExtractor.__init__, later "
        f"and dearer than an argparse error.")
    assert res["n_flags"] > 80, f"{arm}: only {res['n_flags']} flags parsed — the block did not extract"


def test_the_clean_arms_carry_the_four_flags_that_MAKE_them_clean():
    """A runbook whose clean composition quietly lost a flag would still launch — and would train
    the incumbent reward under a clean-world run name."""
    clean = _shell_vars()["CLEAN"]
    for flag in ("--no-hand-shaping", "--victory-value", "--draw-penalty", "--win-prob-mode"):
        assert flag in clean, f"${{CLEAN}} no longer carries {flag}"
    assert clean[clean.index("--victory-value") + 1] == "1.0"
    assert clean[clean.index("--draw-penalty") + 1] == "-1.0", (
        "draw must equal a LOSS: at draw > -victory_value the 250-turn stall is the best "
        "non-winning outcome and the clean arm has no anti-stall term (ledger cfbc9bf).")


def test_PopArt_is_retired_on_the_clean_arms_and_kept_on_the_shaped_control():
    """PopArt is retired by OMISSION (`--use-popart` is opt-in and a fresh run resolves it False),
    so the guard has to be that the flag is absent — there is nothing else to assert."""
    v = _shell_vars()
    assert "--use-popart" not in v["TRAIN"] and "--use-popart" not in v["CLEAN"]
    assert "--use-popart" in v["SHAPED"], "the 5M pre-test's control is the INCUMBENT recipe"
    assert "--clip-range-vf" in v["TRAIN"]
    assert v["TRAIN"][v["TRAIN"].index("--clip-range-vf") + 1] == "none"


def test_the_critic_SUPPORT_is_sized_to_the_clean_terminal():
    """The launch-blocker of §6.3: with PopArt off the atom support is in RAW return units, so a
    ±30-era [-12, +12] leaves the whole ±1 outcome axis inside ~4 of 51 bins."""
    clean = _shell_vars()["CLEAN"]
    vmin = float(clean[clean.index("--value-dist-vmin") + 1])
    vmax = float(clean[clean.index("--value-dist-vmax") + 1])
    assert vmax >= 1.0 and vmin <= -1.0, "the support must at least reach the ±1 terminal"
    assert (vmax - vmin) <= 8.0, "a support >8x the outcome range is the resolution collapse"
