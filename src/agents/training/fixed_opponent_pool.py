"""Stable (cross-run) fixed opponents.

A *stable opponent* is a frozen model from ANOTHER, already-finished run, used as a fixed
opponent in the current run (eval yardstick + training mix). It is loaded inference-only via
``agents.model.snapshot.load_foreign_opponent`` and gated on the OBSERVATION FAMILY only — the
same ``arch_signature`` (see ``designs/ai_v5/design_stable_opponents.md`` §3). A mismatch is a
hard, surfaced-at-startup FATAL (``[StableOpponent] FATAL`` → ``TrainExitCode.FATAL_CONFIG``,
shown in the TUI), never a silent wrong-obs feed.

This module owns the CLI-spec parsing + path/arch resolution. The per-worker loading + LRU cache
and the floor/challenge sampling lifecycle (Stage 2) layer on top of ``FixedOpponentEntry``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from agents.model.model_version import ModelVersion

# Reserved label namespace for cross-run EXTernal opponents — distinct from the bot roster and
# from the ``sentinel_<i>`` pool snapshots. The separator is an UNDERSCORE (``ext_<run>``) so the
# emitted metric tags are uniform with the rest (``eval/win_rate_vs_ext_<run>``, like
# ``eval/win_rate_vs_sentinel_0``) — no colons in TensorBoard tags. The ``ext_`` prefix lets
# ``is_external`` keep these out of ``win_rate_vs_bots`` / the ELO fit. (The aggregate key
# ``external`` is NOT external by this test — ``"external".startswith("ext_")`` is False — and it
# is never a dict key anyway.)
EXT_PREFIX = "ext_"


def is_external(name: str) -> bool:
    """True if ``name`` is a stable cross-run opponent label (``ext_...``)."""
    return name.startswith(EXT_PREFIX)


@dataclass
class FixedOpponentEntry:
    """One resolved + arch-validated stable opponent."""
    label: str          # ext_<...>, unique per opponent (eval/TB/TUI key)
    zip_path: str       # absolute path to the opponent's .zip weights
    config_path: str    # absolute path to its model_config.json (provenance + arch gate)
    arch_signature: str
    temperature: float = 1.0  # TRAINING-mix play temperature (eval plays greedy regardless)

    def to_cfg(self) -> dict:
        """The subset threaded to an eval worker via ``base_cfg['fixed_opponents']`` (eval plays the
        opponent greedy, so temperature isn't needed worker-side)."""
        return {
            "label": self.label,
            "path": self.zip_path,
            "config_path": self.config_path,
        }


def parse_stable_opponents(spec: str) -> list[dict]:
    """Parse a ``--stable-opponents`` string into raw specs (no filesystem / arch checks).

    Grammar: comma-separated ``path[@step][:label]``. The path must not contain ``@`` or ``:``
    (filesystem paths don't). Returns ``{"path", "step", "label"}`` dicts (step/label ``None`` when
    absent). Raises ``ValueError`` on a malformed token — including a ``=<weight>`` (per-opponent
    weights only matter for the training mix, which is Stage 2; rejected with a clear message so the
    syntax isn't silently ignored).
    """
    out: list[dict] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        core, _, label = tok.partition(":")
        if "=" in core:
            raise ValueError(
                f"--stable-opponents token {tok!r}: per-opponent weights (=<weight>) are not "
                "supported yet — they only matter for the training mix (Stage 2). Drop the "
                "'=<weight>'.")
        path, _, step_s = core.partition("@")
        path = path.strip()
        if not path:
            raise ValueError(f"--stable-opponents token {tok!r} has no path")
        step = None
        if step_s:
            try:
                step = int(step_s)
            except ValueError:
                raise ValueError(
                    f"--stable-opponents token {tok!r}: step {step_s!r} is not an integer")
        out.append({"path": path, "step": step, "label": (label.strip() or None)})
    return out


def _resolve_zip_and_config(path: str, step: int | None) -> tuple[str, str, str]:
    """Resolve a spec path to ``(zip_path, config_path, run_basename)``, all absolute.

    Accepts: a direct ``.zip``; a run directory (→ ``best_model/best_model.zip``, then
    ``final_model.zip``, then ``best_model.zip``); or ``@step`` (→
    ``<run>/checkpoint_<step>_steps.zip``). ``model_config.json`` is searched next to the zip,
    then the run dir, then the zip's parent — so a ``best_model/best_model.zip`` still finds the
    run-level config. Raises ``FileNotFoundError`` if no zip or no config resolves.
    """
    apath = os.path.abspath(path)
    zip_path: str | None = None
    run_dir: str
    if step is not None:
        zip_path = os.path.join(apath, f"checkpoint_{step}_steps.zip")
        run_dir = apath
    elif apath.endswith(".zip") and os.path.isfile(apath):
        zip_path = apath
        run_dir = os.path.dirname(apath)
    elif os.path.isdir(apath):
        for cand in (os.path.join(apath, "best_model", "best_model.zip"),
                     os.path.join(apath, "final_model.zip"),
                     os.path.join(apath, "best_model.zip")):
            if os.path.isfile(cand):
                zip_path = cand
                break
        run_dir = apath
    elif os.path.isfile(apath + ".zip"):
        zip_path = apath + ".zip"
        run_dir = os.path.dirname(apath)
    else:
        run_dir = apath

    if zip_path is None or not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"--stable-opponents: no model .zip found for {path!r}"
            + (f" @step {step}" if step is not None else "")
            + " (expected a run dir with best_model/best_model.zip, a direct .zip, "
              "or a run dir + @step)."
        )

    zip_dir = os.path.dirname(zip_path)
    config_path: str | None = None
    for d in (zip_dir, run_dir, os.path.dirname(zip_dir)):
        cand = os.path.join(d, "model_config.json")
        if os.path.isfile(cand):
            config_path = cand
            break
    if config_path is None:
        raise FileNotFoundError(
            f"--stable-opponents: {path!r} resolved to {zip_path!r} but no sibling "
            "model_config.json was found (a stable opponent must carry its arch provenance)."
        )
    # The run NAME is the default label — the dir the user thinks of as "the run"
    # (ai_v5_5_popart_N_0607), NOT a "best_model"/"snapshots" subfolder the zip happens to live in.
    name_dir = os.path.dirname(zip_dir) if os.path.basename(zip_dir) in ("best_model", "snapshots") \
        else zip_dir
    return zip_path, config_path, os.path.basename(os.path.normpath(name_dir))


def resolve_stable_opponents(
    spec: "str | None",
    current_version: ModelVersion,
    default_temperature: float = 1.0,
) -> list[FixedOpponentEntry]:
    """Resolve + VALIDATE a ``--stable-opponents`` string into ``FixedOpponentEntry`` objects.

    For each spec: resolve the zip + its ``model_config.json``, then assert the opponent shares
    the live run's architecture family (``current_version.check_opponent_compatible`` — same
    ``arch_signature`` = same observation layout). A mismatch raises ``ModelVersionError``; the
    caller turns it into a startup FATAL surfaced to the TUI exactly like a checkpoint arch
    mismatch. ``None``/empty ``spec`` → ``[]``. Raises ``ValueError`` on a malformed spec /
    duplicate label, ``FileNotFoundError`` if a path / config can't be resolved.
    """
    if not spec:
        return []
    entries: list[FixedOpponentEntry] = []
    seen_labels: set[str] = set()
    for raw in parse_stable_opponents(spec):
        zip_path, config_path, run_base = _resolve_zip_and_config(raw["path"], raw["step"])
        foreign = ModelVersion.from_json_file(config_path)
        # Raises ModelVersionError on an arch_signature (obs-family) mismatch — the FATAL gate.
        current_version.check_opponent_compatible(foreign)

        label = raw["label"]
        if label:
            label = label if is_external(label) else EXT_PREFIX + label
        else:
            suffix = f"@{raw['step']}" if raw["step"] is not None else ""
            label = f"{EXT_PREFIX}{run_base}{suffix}"
        if label in seen_labels:
            raise ValueError(
                f"--stable-opponents: duplicate label {label!r} — give each opponent a "
                "unique :label.")
        seen_labels.add(label)

        entries.append(FixedOpponentEntry(
            label=label, zip_path=zip_path, config_path=config_path,
            arch_signature=foreign.arch_signature, temperature=default_temperature,
        ))
    return entries
