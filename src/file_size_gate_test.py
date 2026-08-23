"""Is any file getting too big to read? The static gate for SIZE.

Sits beside `ruff_gate_test.py` and `strict_api_lock_test.py` at the `src/` root because, like
those, its subject is the whole tree rather than any one package.

**The owner rule (2026-08-22).** For core source files, **under 1,000 lines is the TARGET and
under 2,000 is the STRONG bound.** This gate encodes exactly that asymmetry: 2,000 HARD-FAILS,
while the 1,000-2,000 band fails nobody and is instead REPORTED (`test_target_band_census`, visible
under `-s`) so the number stays in front of people without blocking anyone's commit. A target that
fails the build is not a target, it is a bound; and a target nothing ever prints is not a target
either, it is a wish.

**Test files are exempt when they exercise a SINGLE subject, or are fuzz.** A 2,200-line test file
that pins one 400-line module is a GOOD artifact — it is the module's specification, and splitting
it scatters that specification across files that must then be kept in sync by hand. A 2,200-line
test file that sprawls across six subsystems is the same liability as a 2,200-line source file, so
it gets the same bound. See `is_exempt_test` for how the two are told apart and what to do when the
heuristic misfires.

**The ratchet only turns one way.** Files already over the bound are grandfathered with their
measured line count. They may shrink freely, they may drift up by a few lines, but growing 10%
past the recorded count FAILS. The point is not to punish a bug fix that adds four lines to
`train_rl_agent.py`; it is to make "this file was already huge" stop working as a licence to keep
feeding it.

**A grandfathered file that drops back under the bound must LEAVE the list.** That is the
c-family lesson recorded in the root `CLAUDE.md`: an allowlist entry can outlive its own fix and
then mislead every reader after it, including agents briefed from it. So the list may only ever
SHRINK, and this gate fails when an entry stops being load-bearing. Same house rule as the ruff
temporary-handoff list, which shrank to nothing and is now CLOSED: **a new finding has nowhere to
be parked.** There is no "add it to the list" escape for a file you just wrote — decompose it, the
way `features_extractor.py` was decomposed into per-phase modules on 2026-08-16 with a re-export
hub that kept every import path working.

**Scope** mirrors the ruff gate exactly: `src/agents`, `src/main`, `src/utils`. `src/poke_env` (a
vendored FORK) and `src/rust_sim` (a Rust crate whose Python is harness scratch) are out of scope
— neither is ours to shape.

**Cost: milliseconds** — it reads 592 files and counts newlines, so there is no tier question. No
cost marker, runs in the fast inner loop. Opt out explicitly:

    GEN3AI_SKIP_SIZE_GATE=1 pytest src/ -q
"""
import os
import pathlib
from typing import Dict, List, Optional

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Mirrored from `ruff_gate_test.py::_RUFF_ARGV`. If one gate's scope changes, change both — two
# static gates that disagree about what "the tree" means will eventually let a file live in the
# gap between them.
_SOURCE_ROOTS = ("src/agents", "src/main", "src/utils")

HARD_LIMIT = 2000       # over this FAILS
TARGET = 1000           # over this is REPORTED, never a failure
RATCHET_SLACK = 0.10    # a grandfathered file may drift up by <10% of its recorded count

# ---------------------------------------------------------------------------------------------
# Grandfathered SOURCE files. Measured 2026-08-22 at ce77c86. The list may only SHRINK.
#
# It already has: `src/main/train_rl_agent.py` (4574 lines) was decomposed into the `main/train/`
# package on 2026-08-22 and is now ~350 lines, so its entry is GONE rather than lowered — which is
# the rule this file states twice and the reason there is no "add it to the list" escape. Taking
# an entry off is welcome piecemeal work: it needs no permission, no design doc and no
# coordination, and every removal is a permanent reduction in how much a reader has to hold at
# once.
# ---------------------------------------------------------------------------------------------
GRANDFATHERED_SOURCE: Dict[str, int] = {
    "src/main/prober/engine.py": 3058,
    "src/main/prober/session.py": 2573,
    "src/agents/model/features_extractor.py": 2237,
    "src/agents/training/instrumented_ppo.py": 2134,
}

# ---------------------------------------------------------------------------------------------
# Grandfathered non-exempt TEST files.
#
# EMPTY, and that is a measurement rather than an oversight: the three test files over 2,000 lines
# (`prober/web/app_test.py`, `training/reward_manager_test.py`, `prober/engine_test.py`) each name
# a source sibling, so each is exempt as a single-subject specification. Verified 2026-08-22.
# ---------------------------------------------------------------------------------------------
GRANDFATHERED_TESTS: Dict[str, int] = {}


def is_fuzz_test(rel_path: str) -> bool:
    """Fuzz tests are exempt outright.

    A fuzz test plays real battles and validates against the live protocol stream; its length is
    the scenario coverage, not accumulated sprawl. They are also not pytest-collected (no
    `test_*` functions — they run as scripts), so they are not even part of the suite whose
    readability this gate protects.
    """
    name = os.path.basename(rel_path)
    return name.endswith("_fuzz_test.py") or name.endswith("_fuzz_e2e_test.py")


def source_sibling_for(rel_path: str, exists) -> Optional[str]:
    """Which source file does `X_test.py` appear to be the specification OF?

    ⚠️ **This is a NAME heuristic, not an import-graph analysis**, and it is deliberately so: cost
    arrives transitively through helpers, so an import graph classifies nearly every unit test as
    cross-cutting (the root `CLAUDE.md` records that exact measurement for the tier markers — 30
    test files transitively reach the battle runner). A name, by contrast, is a DECLARATION by the
    author of what the file is about, which is the thing being exempted.

    Three spellings are accepted, all relative to the test file's own directory, for
    `<stem>_test.py`:

    1. `<stem>.py`                — the ordinary case (`reward_manager_test.py` ↔ `reward_manager.py`)
    2. `<stem>/__init__.py`       — the test's subject is a PACKAGE of that name
                                    (`main/launcher_test.py` ↔ `main/launcher/`)
    3. `<a>/<b...>.py`            — underscores re-read as directory separators, for a test that
                                    sits one level above its subject
                                    (`main/launcher_app_test.py` ↔ `main/launcher/app.py`)

    **When it misfires** (a genuinely single-subject test the heuristic calls cross-cutting): the
    fix is to RENAME the test to match its subject, or to split it. Do not add it to
    `GRANDFATHERED_TESTS` — that list is for files that were already oversized on 2026-08-22, and
    a name that does not say what the file is about is itself the problem the gate found.

    `exists` is injected (a `str -> bool` callable) so the meta-tests can pin this against a made-up
    tree instead of the real one.
    """
    name = os.path.basename(rel_path)
    if not name.endswith("_test.py"):
        return None
    directory = os.path.dirname(rel_path)
    stem = name[: -len("_test.py")]

    candidates = [f"{stem}.py", f"{stem}/__init__.py"]
    parts = stem.split("_")
    for cut in range(1, len(parts)):
        candidates.append("/".join(parts[:cut]) + "/" + "_".join(parts[cut:]) + ".py")

    for candidate in candidates:
        full = f"{directory}/{candidate}" if directory else candidate
        if exists(full):
            return full
    return None


def is_exempt_test(rel_path: str, exists) -> bool:
    """Exempt = a fuzz test, or a `*_test.py` whose name names a single source subject."""
    if not os.path.basename(rel_path).endswith("_test.py"):
        return False
    return is_fuzz_test(rel_path) or source_sibling_for(rel_path, exists) is not None


# ---------------------------------------------------------------------------------------------
# The pure core. Everything the gate MEANS lives here, in one function over a plain dict, so the
# meta-tests can pin all four semantics without a filesystem.
# ---------------------------------------------------------------------------------------------

_DECOMP_PRECEDENT = (
    "The precedent is `features_extractor.py` (2026-08-16): it was split into one file per phase "
    "group (extractor_ctx / encoders / team_transformer / pools / belief_heads / aux_value_heads / "
    "pointer_head / value_readouts) with the original kept as a re-export hub, so every existing "
    "import path still resolved and no caller changed."
)


def check_sizes(counts: Dict[str, int], grandfathered: Dict[str, int], kind: str) -> List[str]:
    """Return one human-readable violation per broken rule (empty list == the gate passes).

    `counts` maps repo-relative path -> line count, and holds ONLY the files this call judges
    (source, or non-exempt tests). `kind` is the word used in the messages.
    """
    problems: List[str] = []

    for path, count in sorted(counts.items()):
        recorded = grandfathered.get(path)
        if recorded is None:
            if count > HARD_LIMIT:
                problems.append(
                    f"NEW OVERSIZED {kind.upper()} FILE: {path} is {count} lines, over the "
                    f"{HARD_LIMIT}-line hard bound.\n"
                    f"  The owner rule: under {TARGET} lines is the TARGET for a core file, under "
                    f"{HARD_LIMIT} is the STRONG bound. This file is over the bound.\n"
                    f"  There is no 'add it to the allowlist' escape — that list is frozen at its "
                    f"2026-08-22 membership and may only shrink. Decompose instead.\n"
                    f"  {_DECOMP_PRECEDENT}"
                )
            continue

        ceiling = int(recorded * (1 + RATCHET_SLACK))
        if count <= HARD_LIMIT:
            problems.append(
                f"GRANDFATHERED ENTRY IS STALE: {path} is now {count} lines, under the "
                f"{HARD_LIMIT}-line bound — REMOVE IT from the allowlist in "
                f"{os.path.basename(__file__)}.\n"
                f"  The list may only shrink, and this is the shrink: the file no longer needs an "
                f"exception, so the entry now only misleads the next reader into thinking it does. "
                f"(Same reason the ruff handoff list had to be closed rather than left standing.)"
            )
        elif count >= ceiling:
            problems.append(
                f"THE RATCHET ONLY TURNS ONE WAY: {path} has grown to {count} lines from its "
                f"recorded {recorded} (+{count - recorded}), past the {ceiling}-line ratchet "
                f"ceiling ({int(RATCHET_SLACK * 100)}% slack).\n"
                f"  Being over the bound already is not a licence to keep growing. Either take "
                f"this addition somewhere else, or spend the change making the file smaller and "
                f"lower its recorded count.\n"
                f"  Do NOT simply raise the number — that turns the ratchet into a rubber stamp."
            )

    missing = sorted(set(grandfathered) - set(counts))
    if missing:
        problems.append(
            f"GRANDFATHERED {kind.upper()} ENTRIES NAME FILES THAT ARE NOT THERE: "
            f"{', '.join(missing)}.\n"
            f"  If they were deleted or renamed, delete the entries too — a stale allowlist entry "
            f"is the failure mode this gate exists to prevent."
        )
    return problems


# ---------------------------------------------------------------------------------------------
# Walking the real tree
# ---------------------------------------------------------------------------------------------

def _line_count(path: pathlib.Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def measure_tree(repo_root: pathlib.Path):
    """-> (source_counts, non_exempt_test_counts, exempt_tests) as repo-relative path dicts."""
    source: Dict[str, int] = {}
    tests: Dict[str, int] = {}
    exempt: Dict[str, int] = {}

    files = []
    for root in _SOURCE_ROOTS:
        files.extend(sorted((repo_root / root).rglob("*.py")))
    rel_all = {p.relative_to(repo_root).as_posix() for p in files}

    def exists(rel: str) -> bool:
        return rel in rel_all or (repo_root / rel).is_file()

    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        count = _line_count(path)
        if os.path.basename(rel).endswith("_test.py"):
            (exempt if is_exempt_test(rel, exists) else tests)[rel] = count
        else:
            source[rel] = count
    return source, tests, exempt


_SKIP = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_SIZE_GATE") == "1",
                           reason="GEN3AI_SKIP_SIZE_GATE=1")


@pytest.fixture(scope="module")
def tree():
    return measure_tree(_REPO_ROOT)


@_SKIP
def test_source_files_respect_the_size_ratchet(tree):
    source, _, _ = tree
    assert source, f"walked {_SOURCE_ROOTS} and found no source files — the gate is not running"
    problems = check_sizes(source, GRANDFATHERED_SOURCE, "source")
    assert not problems, "\n\n".join(problems)


@_SKIP
def test_cross_cutting_test_files_respect_the_size_ratchet(tree):
    _, tests, _ = tree
    problems = check_sizes(tests, GRANDFATHERED_TESTS, "test")
    assert not problems, "\n\n".join(problems) + (
        "\n\nA test file is EXEMPT from this bound when it is a fuzz test, or when its name names "
        "a single source subject (`X_test.py` beside `X.py`). This one names none, so it reads as "
        "cross-cutting. Rename it to match its subject if that is wrong, or split it."
    )


@_SKIP
def test_target_band_census(capsys, tree):
    """REPORT the 1,000-2,000 band. Never fails — print it with `-s` to see it.

    The band is the whole point of the owner rule and the half a hard limit cannot express: these
    files are within the bound and over the target, so they are where the next decomposition
    should come from, and where a file about to become a problem shows up first.
    """
    source, tests, exempt = tree
    judged = {**source, **tests}
    band = sorted(((c, p) for p, c in judged.items() if TARGET <= c <= HARD_LIMIT), reverse=True)
    over = sorted(((c, p) for p, c in judged.items() if c > HARD_LIMIT), reverse=True)

    with capsys.disabled():
        print(f"\n[file-size] {len(judged)} judged files "
              f"({len(exempt)} test files exempt as fuzz / single-subject)")
        print(f"[file-size] over the {HARD_LIMIT} bound (all grandfathered): {len(over)}")
        for c, p in over:
            print(f"[file-size]   {c:>5}  {p}")
        print(f"[file-size] in the {TARGET}-{HARD_LIMIT} TARGET band (reported, not a failure): "
              f"{len(band)}")
        for c, p in band:
            print(f"[file-size]   {c:>5}  {p}")


# ---------------------------------------------------------------------------------------------
# META-TESTS — pin what the gate MEANS. Every rule in the docstring is asserted here against a
# synthetic tree, so a future edit that softens one of them fails loudly instead of quietly.
# ---------------------------------------------------------------------------------------------

def test_meta_a_new_oversized_file_fails():
    problems = check_sizes({"src/agents/huge.py": HARD_LIMIT + 1}, {}, "source")
    assert len(problems) == 1
    assert "NEW OVERSIZED" in problems[0]
    assert str(TARGET) in problems[0] and str(HARD_LIMIT) in problems[0]
    assert "features_extractor" in problems[0], "the message must name the decomposition precedent"


def test_meta_a_file_exactly_at_the_bound_passes():
    assert check_sizes({"src/agents/ok.py": HARD_LIMIT}, {}, "source") == []


def test_meta_a_grandfathered_file_may_drift_but_not_grow_ten_percent():
    allow = {"src/main/big.py": 4000}
    assert check_sizes({"src/main/big.py": 4000}, allow, "source") == []
    assert check_sizes({"src/main/big.py": 4399}, allow, "source") == [], "small drift is allowed"
    problems = check_sizes({"src/main/big.py": 4400}, allow, "source")
    assert len(problems) == 1 and "RATCHET ONLY TURNS ONE WAY" in problems[0]
    assert "Do NOT simply raise the number" in problems[0]


def test_meta_a_grandfathered_file_may_shrink_freely_until_it_crosses_the_bound():
    allow = {"src/main/big.py": 4000}
    assert check_sizes({"src/main/big.py": 2001}, allow, "source") == []


def test_meta_a_grandfathered_file_back_under_the_bound_must_leave_the_list():
    problems = check_sizes({"src/main/big.py": HARD_LIMIT}, {"src/main/big.py": 4000}, "source")
    assert len(problems) == 1 and "STALE" in problems[0]
    assert "REMOVE IT" in problems[0]


def test_meta_a_grandfathered_entry_for_a_deleted_file_fails():
    problems = check_sizes({}, {"src/main/gone.py": 4000}, "source")
    assert len(problems) == 1 and "NOT THERE" in problems[0]


def test_meta_the_test_bound_is_the_same_three_rules():
    assert "NEW OVERSIZED TEST FILE" in check_sizes({"a_test.py": 2001}, {}, "test")[0]
    assert "RATCHET" in check_sizes({"a_test.py": 3400}, {"a_test.py": 3000}, "test")[0]
    assert "STALE" in check_sizes({"a_test.py": 1000}, {"a_test.py": 3000}, "test")[0]


def test_meta_fuzz_tests_are_exempt():
    tree = {"src/agents/battle/event_log_fuzz_test.py",
            "src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py"}
    for path in tree:
        assert is_fuzz_test(path)
        assert is_exempt_test(path, tree.__contains__), "fuzz needs no sibling to be exempt"


def test_meta_a_name_sibling_exempts_and_its_absence_does_not():
    tree = {"src/agents/training/reward_manager.py",
            "src/agents/training/reward_manager_test.py",
            "src/agents/training/reward_redesign_test.py"}
    exists = tree.__contains__
    assert is_exempt_test("src/agents/training/reward_manager_test.py", exists)
    assert not is_exempt_test("src/agents/training/reward_redesign_test.py", exists), (
        "no `reward_redesign.py` exists, so the file is cross-cutting by the name heuristic"
    )


def test_meta_the_package_and_subpackage_spellings_resolve():
    tree = {"src/main/launcher/__init__.py", "src/main/launcher/app.py",
            "src/main/launcher_test.py", "src/main/launcher_app_test.py"}
    exists = tree.__contains__
    assert source_sibling_for("src/main/launcher_test.py", exists) == \
        "src/main/launcher/__init__.py"
    assert source_sibling_for("src/main/launcher_app_test.py", exists) == "src/main/launcher/app.py"


def test_meta_a_non_test_file_is_never_exempt():
    tree = {"src/agents/model/features_extractor.py"}
    assert not is_exempt_test("src/agents/model/features_extractor.py", tree.__contains__)


def test_meta_the_scope_matches_the_ruff_gate():
    """The two static gates must agree on what "the tree" is."""
    import ruff_gate_test
    ruff_roots = [a for a in ruff_gate_test._RUFF_ARGV if a.startswith("src/")
                  and "--exclude" not in a]
    excluded = {"src/poke_env", "src/rust_sim"}
    assert sorted(r for r in ruff_roots if r not in excluded) == sorted(_SOURCE_ROOTS)


@_SKIP
def test_meta_the_live_allowlists_are_exactly_the_files_over_the_bound(tree):
    """No entry may be aspirational, and no oversized file may be missing.

    This is the one meta-test that touches the real tree, and it is what makes the two lists a
    MEASUREMENT rather than a claim.
    """
    source, tests, _ = tree
    assert sorted(GRANDFATHERED_SOURCE) == sorted(p for p, c in source.items() if c > HARD_LIMIT)
    assert sorted(GRANDFATHERED_TESTS) == sorted(p for p, c in tests.items() if c > HARD_LIMIT)
