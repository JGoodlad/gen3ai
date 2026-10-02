"""The legacy POST-TRAINING eval is OFF by default (owner 2026-10-01; `--final-eval` opts in).

`main/train/final_eval.py` plays 9 bots x `--eval-battles` greedy at batch 1 through the per-game
path (~20 min per run) after `Training complete`, and nothing reads its output; the in-run eval
cycles already measure the same bots. Both post-training call sites in `model_build` must be gated
on the flag, and the flag must default False. `--eval-only` still runs it (explicitly requested).
"""
import ast
import inspect

from main.train import model_build
from main.train.parser import build_parser


def test_the_flag_defaults_off_and_opts_in():
    parser = build_parser()
    assert parser.parse_args([]).final_eval is False
    assert parser.parse_args(["--final-eval"]).final_eval is True


def test_every_post_training_call_is_gated_on_the_flag():
    """Every `await evaluate_model_random(model)` outside the `--eval-only` branch sits under an
    `if` whose test reads `final_eval` — a new ungated call site fails here."""
    tree = ast.parse(inspect.getsource(model_build))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "evaluate_model_random"]
    assert len(calls) == 3, f"expected the eval-only site + 2 post-training sites, got {len(calls)}"
    gated = eval_only = 0
    for call in calls:
        node = call
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                text = ast.unparse(node.test)
                if "eval_only" in text:
                    eval_only += 1
                    break
                if "final_eval" in text:
                    gated += 1
                    break
    assert eval_only == 1 and gated == 2, (eval_only, gated)
