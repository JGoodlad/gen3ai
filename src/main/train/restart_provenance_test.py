"""A launcher RESTART keeps the run's PROVENANCE tags — `arch_source` and `recipe_source` (cutover loose end 3).

`--arch production` stamps `arch_source` (recorded in `model_config.json`) and `recipe_source`
(recorded in `metadata.json:cli_args`) in the one `desugar_umbrella_flags` branch the launcher's
restart strips with `--arch`. Every later save records the NAMESPACE's values, so the run's FIRST
restart wrote `arch_source: null` over `production_config@…` — the field that exists to say "we meant
this architecture" (the 2026-09-06 incident) erased by the run's own routine restart. A same-run
restart now keeps both from the run's own record; a FORK into a new run dir is untouched.

End to end on cutover-prep's restart mechanism (`68850f27`): fresh → save → the launcher's OWN restart
argv (`resume_child_args`) → `resolve_config` → the ModelVersion the restarted child would record, and
again for a SECOND restart. Every assertion on a kept tag fails on revert.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

_FRESH = ["--arch", "production", "--steps", "1000"]


def _save(run: pathlib.Path, args, *, first: bool) -> None:
    """What a save writes for `args`: model_config.json (the ModelVersion) and metadata.json's
    `cli_args` (vars(args), as `train_rl_agent` records them) beside the immutable original_command."""
    from main.train.derived_toggle_resume_test import _saved_version

    saved = dataclasses.replace(_saved_version(args), terminal_indicator=args.terminal_indicator,
                                victory_value=args.victory_value, draw_penalty=args.draw_penalty)
    (run / "model_config.json").write_text(json.dumps(dataclasses.asdict(saved)))
    meta_path = run / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if first:
        meta["original_command"] = "launcher " + " ".join(_FRESH)
    meta["cli_args"] = json.loads(json.dumps(vars(args), default=str))
    meta_path.write_text(json.dumps(meta))


def _recorded_arch_source(args):
    from main.train.derived_toggle_resume_test import _saved_version
    return _saved_version(args).arch_source


def _run(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "checkpoint_1024_steps.zip"
    ckpt.write_bytes(b"")
    return run, ckpt


def test_a_fresh_launch_then_restarts_keep_arch_source_and_recipe_source(tmp_path, capsys):
    from main.launcher.checkpoint import resume_child_args
    from main.train.derived_toggle_resume_test import _resolve

    fresh = _resolve(list(_FRESH))
    assert fresh.arch_source and fresh.arch_source.startswith("production_config@")
    assert fresh.recipe_source and fresh.recipe_source.startswith("production_config@")
    assert _recorded_arch_source(fresh) == fresh.arch_source          # the fresh save records it

    run, ckpt = _run(tmp_path)
    _save(run, fresh, first=True)
    argv, stripped = resume_child_args(list(_FRESH), str(ckpt), str(run))
    assert stripped and "--arch" not in argv
    capsys.readouterr()
    first = _resolve(argv)
    assert getattr(first, "arch_source", None) == fresh.arch_source  # was None: nulled at the 1st restart
    assert _recorded_arch_source(first) == fresh.arch_source          # ...and so in model_config.json
    assert getattr(first, "recipe_source", None) == fresh.recipe_source  # ...and cli_args
    assert f"arch_source={fresh.arch_source!r} kept from model_config.json" in capsys.readouterr().out

    _save(run, first, first=False)                                    # the restarted child's own save
    second = _resolve(resume_child_args(list(_FRESH), str(ckpt), str(run))[0])
    assert (second.arch_source, second.recipe_source) == (fresh.arch_source, fresh.recipe_source)


def test_a_restart_never_overwrites_a_tag_the_namespace_already_carries(tmp_path):
    from main.train.arch_surface import inherit_arch_source_on_restart

    run, ckpt = _run(tmp_path)
    saved = type("V", (), {"arch_source": "production_config@aaaaaaaaaaaa"})()
    ns = type("NS", (), {"model": str(ckpt), "arch_source": "nonproduction (x)"})()
    assert inherit_arch_source_on_restart(ns, str(run), saved) is None and ns.arch_source == "nonproduction (x)"
    ns.arch_source = None
    assert inherit_arch_source_on_restart(ns, str(run), saved) == "production_config@aaaaaaaaaaaa"


def test_a_fork_into_a_new_run_dir_does_not_take_its_parents_tag(tmp_path):
    """A fork's surface is its PARENT's (recorded through `lineage`); its own tag says how ITS argv
    chose the surface, so nothing is copied across run dirs."""
    from main.train.arch_surface import inherit_arch_source_on_restart

    parent, ckpt = _run(tmp_path)
    saved = type("V", (), {"arch_source": "production_config@aaaaaaaaaaaa"})()
    ns = type("NS", (), {"model": str(ckpt), "arch_source": None})()
    child = tmp_path / "fork"
    child.mkdir()
    assert inherit_arch_source_on_restart(ns, str(child), saved) is None and ns.arch_source is None
    assert inherit_arch_source_on_restart(ns, None, saved) is None
