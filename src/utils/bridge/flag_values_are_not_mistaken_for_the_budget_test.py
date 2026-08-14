"""A value-taking flag's VALUE must never be parsed as the fuzz budget.

`bridge_session_fuzz_test.py` takes its budget as the first BARE token ("2000" episodes,
"90m" of wall clock). Flags that carry a separate value token therefore have to be declared
in `_VALUE_FLAGS`, or the value is bare too and gets read as the budget.

`--impl` was read by `main()` but never declared, so the DOCUMENTED gate command

    python src/utils/bridge/bridge_session_fuzz_test.py --impl rust

died on `int('rust')` before a single battle ran. That command is the durable end-to-end gate
for `gen3_bridge_forfeit_win_v1` — the bug that wedged `--use-bridge=rust` training (episodes
finished, ZERO PPO iterations) and was read as a "multi-env stall" for months. A gate that
cannot be invoked is indistinguishable from a gate that passes, which is the whole hazard.

The class test (`every_value_flag_main_reads_is_declared`) is the one that matters: it derives
the flag set from `main()`'s own source, so a NEW value-flag added later cannot reintroduce
this by omission — the failure mode that let `--impl` through in the first place.
"""
import re
from pathlib import Path

import pytest

from utils.bridge.bridge_session_fuzz_test import _VALUE_FLAGS, _parse_budget

_SRC = Path(__file__).with_name("bridge_session_fuzz_test.py")


def test_impl_value_is_not_read_as_the_budget():
    """The exact documented invocation must parse, and must yield the DEFAULT budget."""
    assert _parse_budget(["--impl", "rust"]) == ("count", 1000)


def test_impl_with_an_explicit_budget_keeps_both():
    assert _parse_budget(["--impl", "rust", "60"]) == ("count", 60)
    assert _parse_budget(["60", "--impl", "rust"]) == ("count", 60)
    assert _parse_budget(["--impl", "rust", "90m"]) == ("time", 90 * 60.0)


@pytest.mark.parametrize("flag,value", [("--workers", "4"), ("--slow-ms", "250"),
                                        ("--impl", "rust"), ("--impl", "node")])
def test_no_value_flag_leaks_into_the_budget(flag, value):
    assert _parse_budget([flag, value]) == ("count", 1000)


def test_every_value_flag_main_reads_is_declared():
    """Derive the flags from main()'s source so omission cannot reintroduce the bug.

    `main()` reads a value-taking flag as `sys.argv[sys.argv.index("--x") + 1]`. Every flag
    matched that way must be in `_VALUE_FLAGS`; `--impl` was not, which is this whole file.
    """
    src = _SRC.read_text()
    read_with_value = set(re.findall(r'sys\.argv\[sys\.argv\.index\(\s*"(--[a-z0-9-]+)"\s*\)\s*\+\s*1\]', src))
    assert read_with_value, "expected to find the argv value-reads in main()"
    missing = read_with_value - set(_VALUE_FLAGS)
    assert not missing, (
        f"{sorted(missing)} take a separate value token but are not in _VALUE_FLAGS, so that "
        f"value will be parsed as the fuzz budget (int('rust') → ValueError, or worse, a "
        f"SILENTLY wrong episode count for a numeric value)")


def test_a_numeric_flag_value_would_silently_corrupt_the_budget_if_undeclared():
    """The quiet half of the bug: a NUMERIC value parses fine and silently wins.

    `--impl rust` failed loudly. `--workers 4` undeclared would instead run a 4-EPISODE fuzz
    that reports success — so this test pins the skip behaviour, not merely the exception.
    """
    assert _parse_budget(["--workers", "4", "2000"]) == ("count", 2000)
    # And with the declaration removed, the value would win instead — the silent failure.
    assert _parse_budget(["4", "2000"]) == ("count", 4)
