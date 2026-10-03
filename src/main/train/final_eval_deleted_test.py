"""The post-training FINAL EVAL is DELETED (deletion pass P6, owner decision D5, 2026-10-02).

`main/train/final_eval.py` played 9 bots x `--eval-battles` greedy at batch 1 after `Training
complete` and nothing read its output; the last periodic Rust eval cycle is the end-of-run readout.
Its flags went with it. An argv that still carries one must be REFUSED, never quietly accepted —
`--eval-battles` in particular looked like a live-eval power lever and sized nothing in a live
cycle (that is `--eval-games`).
"""
import importlib.util

import pytest

from main.train.parser import build_parser

DELETED = ("--final-eval", "--no-final-eval", "--eval-only", "--no-eval-only", "--eval-battles")


@pytest.mark.parametrize("flag", DELETED)
def test_a_deleted_final_eval_flag_is_refused(flag, capsys):
    argv = [flag] if flag != "--eval-battles" else [flag, "5"]
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_the_module_is_gone():
    assert importlib.util.find_spec("main.train.final_eval") is None


def test_no_dest_survives_on_the_namespace():
    ns = build_parser().parse_args([])
    for dest in ("final_eval", "eval_only", "eval_battles"):
        assert not hasattr(ns, dest), dest
