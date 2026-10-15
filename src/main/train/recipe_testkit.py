"""A RECORDED RECIPE for a test's fake run directory — what `recipe_surface.inherit_recorded_recipe` reads.

Since the 2026-10-14 fork-recipe fix, EVERY `--model` launch (a restart, a fork, a fork's resume) restores its
untyped recipe knobs from the record of the run that wrote the checkpoint (`metadata.json:cli_args`), and a
record that lacks one REFUSES (`RecipeRecordError`, `FATAL_CONFIG`) rather than fall back to the parser. A test
whose subject is something else (an eval regime, a pin, a torch switch) but which launches from a hand-built
checkpoint therefore gives its fake run the record a real run would carry — through this one helper, so the set
of recorded knobs follows `recipe_surface.ROWS` and never drifts into a hand list.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict


def recorded_cli_args(**overrides: Any) -> Dict[str, Any]:
    """Every recipe row recorded nowhere but `cli_args`, at the production fresh recipe's values, plus `overrides`.
    (The INERT rows and the recorded `model_config.json` fields are not cli_args-route rows; they are left out.)"""
    from main.train import recipe_surface as rs
    from main.train.parser import build_parser
    p = build_parser()
    fields = rs._model_version_fields()
    want = rs.production_recipe()
    out = {r.dest: want[r.dest] for r in rs.ROWS
           if rs.restart_route(r.dest, p.get_default(r.dest), fields) == "cli_args"}
    out.update(overrides)
    return out


def write_recorded_recipe(run_dir: "str | os.PathLike[str]", **overrides: Any) -> Dict[str, Any]:
    """Merge `recorded_cli_args(**overrides)` into `<run_dir>/metadata.json`'s `cli_args` (creating either)."""
    path = os.path.join(str(run_dir), "metadata.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as fh:
            meta = json.load(fh)
    cli = dict(meta.get("cli_args") or {})
    for k, v in recorded_cli_args(**overrides).items():
        cli.setdefault(k, v)
    cli.update(overrides)
    meta["cli_args"] = cli
    os.makedirs(str(run_dir), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(meta, fh)
    return cli


def fake_record(monkeypatch: Any, role: str = "fork", **overrides: Any) -> Dict[str, Any]:
    """For a test that resolves a `--model` argv against a STAND-IN checkpoint (no run dir on disk): make
    `recipe_surface.record_source` answer with a recorded `cli_args` (`recorded_cli_args(**overrides)`)."""
    from main.train import recipe_surface as rs
    cli = recorded_cli_args(**overrides)
    monkeypatch.setattr(rs, "record_source", lambda model, run_dir: rs.RecordSource(
        role, "<stand-in run>", "<stand-in run>/metadata.json", {"cli_args": dict(cli)}))
    return cli
