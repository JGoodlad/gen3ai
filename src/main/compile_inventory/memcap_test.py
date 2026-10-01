"""The stack-profile refusal reads the REAL cap of the process's cgroup chain."""
from __future__ import annotations

import pytest

from main.compile_inventory.memcap import (StackProfileRefused, memory_cap_bytes, own_cgroup,
                                           require_cap_for_stack_profile)


def _tree(tmp_path, values):
    for rel, v in values.items():
        d = tmp_path / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.max").write_text(v + "\n")
    return tmp_path


def test_uncapped_chain_reads_none(tmp_path):
    root = _tree(tmp_path, {"user.slice": "max", "user.slice/app.slice": "max",
                            "user.slice/app.slice/tmux.scope": "max"})
    assert memory_cap_bytes("/user.slice/app.slice/tmux.scope", root=root) is None


def test_a_cap_on_an_ANCESTOR_counts_and_the_tightest_wins(tmp_path):
    root = _tree(tmp_path, {"user.slice": "max", "user.slice/heavy.slice": str(40 << 30),
                            "user.slice/heavy.slice/run.scope": str(12 << 30)})
    assert memory_cap_bytes("/user.slice/heavy.slice/run.scope", root=root) == 12 << 30
    assert memory_cap_bytes("/user.slice/heavy.slice", root=root) == 40 << 30


def test_own_cgroup_parses_the_v2_line(tmp_path):
    f = tmp_path / "cgroup"
    f.write_text("0::/user.slice/app.slice/run-p1.scope\n")
    assert own_cgroup(f) == "/user.slice/app.slice/run-p1.scope"


def test_refusal_needs_both_the_flag_and_a_cap():
    with pytest.raises(StackProfileRefused):
        require_cap_for_stack_profile(False, 12 << 30)
    with pytest.raises(StackProfileRefused):
        require_cap_for_stack_profile(True, None)
    require_cap_for_stack_profile(True, 12 << 30)
