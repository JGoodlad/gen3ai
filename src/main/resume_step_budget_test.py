"""The resume must not silently DOUBLE its step budget.

THE BUG THIS PINS (found on gen-10, 2026-08-15). `train_rl_agent` computed `remaining_steps` on the
resume path, printed it, and then called::

    model.learn(total_timesteps=args.steps, reset_num_timesteps=False)

SB3's `BaseAlgorithm._setup_learn` does `total_timesteps += self.num_timesteps` whenever
`reset_num_timesteps=False`. So the ABSOLUTE target re-adds every step already trained: a resume at
24.08M under `--steps 25M` retargets to ~49M. The child printed *"915,520 remaining of 25,000,000"*
and then trained straight through it — gen-10 reached 26.05M before it was stopped by hand, and
gen-9 hit the same thing at 26M against the same 25M budget.

It stayed invisible because nothing FAILS: the run keeps training, the metrics look healthy, and the
only symptom is a step counter drifting past a number nobody re-reads. That is exactly the shape of
defect a test has to catch, because a human watching a dashboard will not.

These tests read the SOURCE rather than running a 25M-step job — the arithmetic is the contract, and
the contract is "which variable is passed", which is statically checkable.
"""
import ast

from main.train import entry_source, entry_source_files


def _learn_calls():
    """Every `*.learn(...)` call in the entry point, as (`file:line`, keyword -> arg-source).

    Scans the WHOLE entry point (the `train_rl_agent.py` hub + the `main/train/` phase modules),
    not one file: the two `learn()` sites moved into `main/train/model_build.py` with the
    2026-08-22 decomposition, and a gate that reads a single path would have gone quietly vacuous.
    """
    out = []
    for path in entry_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "learn"):
                kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
                out.append((f"{path.name}:{node.lineno}", kw))
    return out


def test_every_learn_call_passes_reset_num_timesteps_false():
    """The premise of the whole fix: with reset_num_timesteps=True SB3 would ZERO the counter and a
    resume would lose its history. Both sites rely on False, which is what makes the += apply."""
    calls = _learn_calls()
    assert calls, "no model.learn(...) call found — did the entry point move?"
    for lineno, kw in calls:
        assert kw.get("reset_num_timesteps") == "False", (
            f"learn() at line {lineno} does not pass reset_num_timesteps=False; the step-budget "
            f"arithmetic in this file assumes it does")


def test_the_resume_call_passes_the_REMAINING_budget_not_the_absolute_target():
    """THE regression. `args.steps` on the resume path doubles the budget (see the module docstring).

    Identified by position: the resume site is the one guarded by a `remaining_steps` computation.
    """
    src = entry_source()
    assert "remaining_steps = args.steps - model.num_timesteps" in src, (
        "the resume path no longer computes remaining_steps — re-check this test's assumption")
    calls = _learn_calls()
    budgets = [kw.get("total_timesteps") for _, kw in calls]
    assert "remaining_steps" in budgets, (
        f"no learn() call passes the REMAINING budget; found {budgets}. On the resume path "
        f"`total_timesteps=args.steps` retargets to args.steps + num_timesteps — gen-10 ran to "
        f"26.05M against a 25M budget that way.")


def test_at_most_one_learn_call_uses_the_absolute_target():
    """Only the FRESH-run site may pass `args.steps`, and only because num_timesteps is 0 there so
    SB3's `+= num_timesteps` is a no-op. A second one would be the bug coming back."""
    absolute = [ln for ln, kw in _learn_calls() if kw.get("total_timesteps") == "args.steps"]
    assert len(absolute) <= 1, (
        f"{len(absolute)} learn() calls pass the absolute target (lines {absolute}). Only the "
        f"fresh-run path may — a resume must pass the remaining budget.")


def test_the_printed_remaining_matches_what_is_trained():
    """The message said '915,520 remaining' while the run trained 25M more. A number a human is
    shown must be the number the code acts on, or it actively misleads during an incident."""
    src = entry_source()
    printed = "Steps: {remaining_steps:,} remaining" in src
    used = any(kw.get("total_timesteps") == "remaining_steps" for _, kw in _learn_calls())
    assert printed == used or not printed, (
        "train_rl_agent prints `remaining_steps` to the operator but does not pass it to learn(); "
        "that is the gen-10 overrun exactly — the displayed budget was fiction.")
