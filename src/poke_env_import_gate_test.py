"""Does any file import poke-env that the retirement has not already counted? The static gate (T27 / P0).

**What this is for.** The owner's 2026-10-06 direction is to retire poke-env to ONE stack — "not maintain
the dual stack" (every fact the Rust reader already reads is a second copy in the Python battle layer: the
change amplification). The retirement is phased (``designs/research_state/measurements/
pokeenv_and_hotpath_survey_2026-10-06/README.md`` §A4.5, backlog ``T27``), and its first phase is this RATCHET:
the set of files outside the vendored fork (``src/poke_env/``) that import ``poke_env`` may only SHRINK, so a
phase's progress is a number that falls and a NEW importer is an event somebody has to look at.

**The rule.**

* The GENERATED list is ``designs/ops/poke_env_import_allowlist.txt`` (scanner + CLI:
  ``utils.poke_env_importers``). An importer — an ``import`` / ``from … import`` at module level, inside a
  function, or under ``TYPE_CHECKING``, or a literal poke-env module STRING handed to ``import_module`` /
  ``patch`` / ``monkeypatch.setattr`` — that is NOT on the list FAILS here.
* **THE LIST MAY ONLY SHRINK.** A listed file that no longer imports ``poke_env`` FAILS too (a stale entry
  would let the import come back unnoticed), and :data:`FROZEN_NON_TEST_COUNT` / :data:`FROZEN_TEST_COUNT`
  are the CEILING: the list may hold no more entries than they say, and exactly that many, so every shrink
  lowers them. ``python -m utils.poke_env_importers --shrink`` does the mechanical part (drops the stale
  entries, rewrites the two constants, and REFUSES if a live importer is not listed). **Raising a count, or
  adding a line, is not a legal move** — it is the one thing this file exists to stop. A file a build is
  genuinely FORCED to add (a new file in the Python encoder that Rust-parity work needs) is named, with the
  reason, in the commit body and decided by the owner; it is never waved through by a green gate.
* Scope: ``src/``, ``tools/`` and ``scripts/``, minus the vendored fork. ``designs/`` measurement scripts are
  frozen history and are out of scope; a module name built at run time is a blind spot (see the scanner's
  docstring).
* The ONE permanent entry is ``PEER_PROCESS_PERMANENT``: a script that runs in Metamon's interpreter against
  UPSTREAM poke-env and is never imported by this repo. It is not on the list and is not counted; the end
  state of the retirement is an empty list plus that file.

**The end state** is an empty allowlist and the vendored fork deleted (P6). Until then every phase's commit
lowers the two counts, and this file's docstring is the place a reader learns why.

Opt out explicitly (never silently): ``GEN3AI_SKIP_POKE_ENV_IMPORT_GATE=1``.
"""
from __future__ import annotations

import os

import pytest

from utils import poke_env_importers as pei
from utils.paths import repo_path

pytestmark = [pytest.mark.static,   # the `static` budget tier (conftest._STATIC_BUDGET_BASE_S)
              pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_POKE_ENV_IMPORT_GATE") == "1",
                                 reason="GEN3AI_SKIP_POKE_ENV_IMPORT_GATE=1 (explicit opt-out)")]

#: The CEILING on the allowlist. They only go DOWN: `python -m utils.poke_env_importers --shrink` rewrites
#: these two lines (it matches them by a strict regex — keep each on its own line, in this exact form).
#: Frozen 2026-10-06 at 74 non-test + 97 test importers (+ the permanent peer-process script); every shrink since
#: has lowered them (the two lines below are the CURRENT ceiling).
FROZEN_NON_TEST_COUNT = 50
FROZEN_TEST_COUNT = 88

#: A floor under the scan itself, so a wrong root or a broken walk cannot make every check below vacuous.
MIN_FILES_SCANNED = 1000


def _inventory():
    return pei.scan_repo()


def _fix_hint() -> str:
    return ("The list only SHRINKS. Fix the import (use `agents.gen3_data` for data tables, or the Rust path), or — "
            "if a build is genuinely forced to add a file — name the file and the reason in the commit body and "
            "get the OWNER's decision; do not raise the FROZEN counts yourself.")


def test_the_scan_walks_the_tree_and_sees_the_live_importers():
    inv = _inventory()
    assert inv.files_scanned >= MIN_FILES_SCANNED, (
        f"the scan saw only {inv.files_scanned} .py files — a wrong root would make every check vacuous")
    assert not any(p.startswith(pei.VENDORED_FORK + "/") for p in inv.importers), (
        "the vendored fork must be pruned from the scan — it is what is being retired, not a user of it")
    for permanent in pei.PEER_PROCESS_PERMANENT:
        assert repo_path(*permanent.split("/")).is_file(), f"{permanent} is declared permanent but is gone"


def test_no_new_poke_env_importer():
    inv = _inventory()
    listed = set(pei.read_allowlist())
    new = sorted(set(inv.allowlistable()) - listed)
    detail = [f"{p}  [{', '.join(inv.importers[p].modules)}]  ({inv.importers[p].scope})" for p in new]
    assert not new, (
        f"{len(new)} file(s) import poke_env and are NOT on designs/ops/poke_env_import_allowlist.txt. "
        + _fix_hint() + "\n  " + "\n  ".join(detail))


def test_no_stale_allowlist_entry():
    inv = _inventory()
    stale = sorted(set(pei.read_allowlist()) - set(inv.allowlistable()))
    assert not stale, (
        f"{len(stale)} allowlist entr{'y' if len(stale) == 1 else 'ies'} no longer import poke_env (or the "
        "file is gone) — a retired importer must LEAVE the list in the commit that retires it, or it could "
        "silently come back. Run `python -m utils.poke_env_importers --shrink`:\n  " + "\n  ".join(stale))


def test_the_allowlist_never_grows_beyond_the_frozen_counts():
    listed = pei.read_allowlist()
    assert len(listed) == len(set(listed)), "duplicate allowlist entries"
    non_test, test = pei.counts(listed)
    assert non_test <= FROZEN_NON_TEST_COUNT and test <= FROZEN_TEST_COUNT, (
        f"the allowlist holds {non_test} non-test / {test} test entries, over the frozen ceiling "
        f"{FROZEN_NON_TEST_COUNT} / {FROZEN_TEST_COUNT}: an entry was ADDED. " + _fix_hint())
    assert (non_test, test) == (FROZEN_NON_TEST_COUNT, FROZEN_TEST_COUNT), (
        f"the allowlist SHRANK to {non_test} non-test / {test} test but the frozen ceiling still says "
        f"{FROZEN_NON_TEST_COUNT} / {FROZEN_TEST_COUNT} — lower FROZEN_NON_TEST_COUNT / FROZEN_TEST_COUNT to the "
        "new numbers (`python -m utils.poke_env_importers --shrink` does it), or the freed headroom would let "
        "a new importer in later.")


def test_the_permanent_peer_process_entries_are_not_on_the_list():
    assert not set(pei.PEER_PROCESS_PERMANENT) & set(pei.read_allowlist()), (
        "a permanent peer-process entry is allowed by declaration, not by the list — keep it off the list so "
        "the shrink target (an empty list) stays meaningful")


def test_every_allowlist_entry_is_in_scope_and_sorted():
    listed = pei.read_allowlist()
    for p in listed:
        assert p.split("/", 1)[0] in pei.SCAN_ROOTS and not p.startswith(pei.VENDORED_FORK + "/"), (
            f"{p} is outside the scanned roots {pei.SCAN_ROOTS} (or inside the vendored fork)")
        assert p.endswith(".py"), p
    assert listed == sorted(listed, key=lambda x: (pei.is_test_path(x), x)), (
        "the allowlist is written sorted (non-test first, then test); `--shrink` keeps it so")
