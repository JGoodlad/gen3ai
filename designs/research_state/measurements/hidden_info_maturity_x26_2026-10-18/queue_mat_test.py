"""The queue and the archive layout on a SYNTHETIC archive (amendment A1: two banks, per-(anchor, bank) modes, the
hide-only arms). No model, no engine, no real archive: a temporary ``HIB_MAT_ARCHIVE`` and temporary states files.

    python -I queue_mat_test.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="hibmat_queue_test_"))
os.environ["HIB_MAT_ARCHIVE"] = str(TMP / "arch")
os.environ.pop("HIB_MAT_PILOT", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402
import queue_mat as Q  # noqa: E402

assert E.ARCH == TMP / "arch", "mat_common must read the temporary archive"


def _states(name: str, n: int, lean: int) -> None:
    st = {f"{name}/b{i:03d}#p1#{i}": {"lean": i < lean, "battle": f"{name}/b{i:03d}"} for i in range(n)}
    p = TMP / f"states_{name}.json"
    p.write_text(json.dumps({"states": st}))
    E.BANKS[name] = dataclasses.replace(E.BANKS[name], states_file=str(p))


def _ckpts(anchors: dict) -> None:
    E.ARCH.mkdir(parents=True, exist_ok=True)
    E.ckpts_path().write_text(json.dumps({"anchors": anchors}))


def _rows(dd: Path, ids, name="rows_0_of_1.jsonl", **extra) -> None:
    dd.mkdir(parents=True, exist_ok=True)
    with open(dd / name, "a") as f:
        for x in ids:
            f.write(json.dumps({"id": x, "ok": True, **extra}) + "\n")


def setup() -> None:
    _states("interim", 10, 4)
    _states("plateau", 6, 2)
    _ckpts({
        "x26_025M": {"step": 25, "mode": "lean", "priority": 1,
                     "banks": {"interim": {"mode": "full", "priority": 20}, "plateau": {"mode": "full", "priority": 40}}},
        "x26_225M": {"step": 225, "mode": "lean", "priority": 2},                     # the original registration
        "x26_P280M": {"step": 280, "mode": "full_hide", "priority": 10,
                      "banks": {"interim": {"mode": "full_hide", "priority": 10},
                                "plateau": {"mode": "full_hide", "priority": 30}}},
    })


def test_banks_have_their_own_namespace_and_ids():
    setup()
    assert E.anchor_mode("x26_225M", "interim") == "lean" and E.anchor_mode("x26_225M", "plateau") is None
    with E.using_bank("plateau"):
        assert E.d("rows").parent == E.ARCH / "plateau" and E.bank_path() == E.ARCH / "plateau" / "bank"
        assert len(E.mode_ids("full_hide")) == 6 and len(E.mode_ids("lean")) == 2
        assert E.d("ckpt").parent == E.ARCH                                         # shared across banks
        assert all(x.startswith("interim/") for x in E.identity_ids())             # the identity: interim only
    assert E.bank().name == "interim" and E.d("rows").parent == E.ARCH
    assert E.arm_dir("O-M") == "OmM" and E.worlds_dir(None, "O-U") == E.ARCH / "worlds" / "shared" / "OmU"


def test_units_follow_the_plateau_step_order():
    setup()
    us = Q.units()
    assert Q.jobs() == [("interim", "x26_225M"), ("interim", "x26_P280M"), ("interim", "x26_025M"),
                        ("plateau", "x26_P280M"), ("plateau", "x26_025M")]
    pos = {u: i for i, u in enumerate(us)}
    # 1 (P* interim) -> 2 (25M interim) -> 3 (the plateau bank) -> 5 (hide-only), every hide-only unit last
    assert pos[("truth", "interim", "x26_P280M")] < pos[("truth", "interim", "x26_025M")] \
        < pos[("build", "plateau", "base")] < pos[("truth", "plateau", "x26_P280M")] \
        < pos[("truth", "plateau", "x26_025M")] < pos[("build", "interim", "hide")]
    hide = [u for u in us if u[0] == "play" and u[3] in E.HIDE_ARMS]
    assert hide == [("play", b, "x26_P280M", x) for b in ("interim", "plateau") for x in ("O-M", "O-U")]
    assert all(pos[u] > pos[("play", "plateau", "x26_025M", "F+U")] for u in hide)
    assert ("vplay", "interim", "x26_P280M") in pos and not any(u[0] == "vplay" and u[1] == "plateau" for u in us)
    assert not any(u[3:] == ("F+M",) for u in us if u[2:3] == ("x26_225M",))     # LEAN: O, F, FN only


def test_pending_counts_each_bank_and_arm_set():
    setup()
    with E.using_bank("interim"):
        ids = E.mode_ids("full")
        _rows(E.worlds_dir(None, "F"), ids, "worlds_0_of_1.jsonl")
        _rows(E.worlds_dir(None, "F+U"), ids[:7], "worlds_0_of_1.jsonl")
        _rows(E.worlds_dir(None, "F+M"), ids, "worlds_0_of_1.jsonl")
        _rows(E.worlds_dir(None, "O-M"), ids[:2], "worlds_0_of_1.jsonl")
        _rows(E.rows_dir("x26_P280M", "O"), ids[:5], "shard_0_of_1.jsonl")
    assert Q.pending(("build", "interim", "base")) == 3                   # F+U missing on 3 of the 10
    assert Q.pending(("build", "interim", "hide")) == 10                  # O-U missing everywhere
    assert Q.pending(("build", "plateau", "base")) == 6                   # nothing built on the plateau bank
    assert Q.pending(("truth", "interim", "x26_P280M")) == 5 and Q.pending(("truth", "plateau", "x26_P280M")) == 6
    assert Q.pending(("truth", "interim", "x26_225M")) == 4               # LEAN ids only
    assert Q.pending(("play", "interim", "x26_P280M", "O-M")) == 2        # the built O-M worlds not yet played
    assert Q.done_count(("build", "interim", "base")) == 7 and Q.wanted(("build", "interim", "hide")) == 10
    assert Q.upstream(("play", "plateau", "x26_P280M", "O-U")) == ("build", "plateau", "hide")
    assert Q.upstream(("play", "plateau", "x26_P280M", "FN")) == ("fnbuild", "plateau", "x26_P280M")


def test_commands_name_the_bank_and_the_arm_sets():
    setup()
    c = Q.command(("build", "interim", "base"))
    assert c[c.index("--shared") + 1] == "F,F+U,F+M" and c[c.index("--bank") + 1] == "interim"
    c = Q.command(("build", "plateau", "hide"))
    assert c[c.index("--shared") + 1] == "O-M,O-U" and c[c.index("--mode") + 1] == "full_hide"
    c = Q.command(("truth", "plateau", "x26_025M"))
    assert c[c.index("--mode") + 1] == "full" and c[c.index("--bank") + 1] == "plateau"
    assert "--bank" not in Q.command(("vplay", "interim", "x26_P280M"))       # the identity: interim (default)


def test_write_row_stamps_driver_and_bank():
    setup()
    os.environ["HIB_MAT_DRIVER_SHA"] = "d" * 40
    try:
        with E.using_bank("plateau"):
            p = E.d("probe") / "r.jsonl"
            with open(p, "w") as f:
                E.write_row(f, {"id": "x"})
        row = json.loads(p.read_text())
        assert row == {"id": "x", "driver": "d" * 40, "bank": "plateau"}
    finally:
        os.environ.pop("HIB_MAT_DRIVER_SHA")


if __name__ == "__main__":
    import shutil

    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_"):
                shutil.rmtree(E.ARCH, ignore_errors=True)
                fn()
                print("ok", name)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
