"""`--limit` caps NEW work, never the id list: E2's two resumable drivers (pure; no archive, model or engine).

phase1_e2.py sliced its sorted id list to ``--limit`` BEFORE dropping the done ids, so once the first 30 truth rows
existed every call found 0 to go and the queue re-launched it every minute: 570 of 600 truth rows per checkpoint sat
empty for a day (fixed in 6a1cf301, 2026-10-17). build_e2_worlds.py held the same slice (its line 80) but never ran with
``--limit``; it now selects through `select_ids`. Both tests fail on a revert of either fix.

    python -I -m pytest designs/research_state/measurements/hidden_info_e2_competent_2026-10-16/build_e2_worlds_test.py -q
"""
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_e2_worlds as B  # noqa: E402

IDS = [f"s{i}" for i in range(6)]
ARMS = ["F", "F+U"]


def test_limit_caps_new_work_not_the_id_list():
    done = {a: set(IDS[:3]) for a in ARMS}
    # the original `ids[: limit]` re-picked s0..s2 (all built) on every call and never reached s3..s5
    assert B.select_ids(IDS, done, ARMS, 3) == ["s3", "s4", "s5"]


def test_a_state_with_one_arm_still_to_build_is_selected():
    done = {"F": set(IDS), "F+U": set(IDS) - {"s1"}}
    assert B.select_ids(IDS, done, ARMS, 2) == ["s1"]


def test_no_limit_takes_every_pending_state_in_order():
    done = {a: {"s0", "s4"} for a in ARMS}
    assert B.select_ids(IDS, done, ARMS, None) == ["s1", "s2", "s3", "s5"]


def test_everything_built_selects_nothing():
    assert B.select_ids(IDS, {a: set(IDS) for a in ARMS}, ARMS, 3) == []


def limit_slices(script: str) -> list:
    """The expressions a script slices with an upper bound that names ``limit``."""
    tree = ast.parse((HERE / script).read_text())
    return sorted(ast.unparse(n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice) and n.slice.upper is not None
                  and "limit" in ast.unparse(n.slice.upper))


def test_only_the_not_done_list_is_ever_capped():
    # a `ids[: a.limit]` (either script's original) reappearing anywhere fails here
    assert limit_slices("build_e2_worlds.py") == ["todo"]
    assert limit_slices("phase1_e2.py") == ["todo"]
