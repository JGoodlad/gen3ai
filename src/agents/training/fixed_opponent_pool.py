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

import json
import os
from dataclasses import dataclass

from agents.model.model_version import ModelVersion
from agents.training.run_spec import split_run_spec

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
    source_elo: "float | None" = None  # the opponent's OWN recorded ELO (its run's metadata.json), if any
    # The opponent's OWN pinned team (the fold-back contract): a specialist trained on ONE team
    # (--trainee-team, read from its run's metadata) must PILOT that team when it plays here as an
    # opponent — in training (the wrapper's per-episode agent2._team switch) AND in eval (the
    # worker's FIXED branch). None = the run had no pin → it pilots the shared pool (a generalist).
    team_str: "str | None" = None   # the raw Showdown export (the FIRST pin; back-compat mirror of team_strs[0])
    team_file: "str | None" = None  # provenance (the first pin file recorded in the opponent's metadata)
    # A MULTI-team specialist (--trainee-teams / pin_multi) trained piloting a WHOLE z-cluster, so as an
    # opponent it must sample among ALL of them — piloting just one (or worse, the shared pool) throws
    # away the pressure it was trained to apply. `team_str`/`team_file` mirror element 0 for back-compat.
    team_strs: "tuple[str, ...]" = ()
    team_files: "tuple[str, ...]" = ()

    def to_cfg(self) -> dict:
        """The subset threaded to an eval worker via ``base_cfg['fixed_opponents']`` (eval plays the
        opponent greedy, so temperature isn't needed worker-side)."""
        return {
            "label": self.label,
            "path": self.zip_path,
            "config_path": self.config_path,
            "team_str": self.team_str,
            "team_strs": list(self.team_strs),
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
        # THE canonical `path[@step]` split (`agents.training.run_spec`) — this parser used to be
        # the ONLY consumer that did it, which is how every other run-spec flag ended up handing
        # an `@step` suffix to a reader that takes a directory.
        path, step = split_run_spec(core, what=f"--stable-opponents token {tok!r}")
        out.append({"path": path, "step": step, "label": (label.strip() or None)})
    return out


def _resolve_zip_and_config(path: str, step: int | None) -> tuple[str, str, str]:
    """Resolve a spec path to ``(zip_path, config_path, run_basename)``, all absolute.

    Accepts: a direct ``.zip``; a run directory (→ ``best_model/best_model.zip``, then
    ``final_model.zip``, then ``best_model.zip``); or ``@step`` (→
    ``<run>/checkpoints/checkpoint_<step>_steps.zip``, falling back to the legacy
    ``<run>/checkpoint_<step>_steps.zip``). ``model_config.json`` is searched next to the zip,
    then the run dir, then the zip's parent — so a ``best_model/best_model.zip`` still finds the
    run-level config. Raises ``FileNotFoundError`` if no zip or no config resolves.

    🚨 **THE `@step` SPLIT HAPPENS HERE, at the ONE choke point every run-spec consumer reaches**
    (`gen3_run_spec_split_v1`). `--stable-opponents` parsed its own `@step` and passed it in;
    every OTHER caller — `--distill-teacher`, `--win-prob-pbrs-source`,
    `--distill-anchor-parent`, `--warmstart-consensus` — calls with ``step=None`` and a raw spec
    string, so before this they could only ever resolve a run DIR. A spec that still carries a
    suffix is split here rather than refused, which makes `<run>@<step>` mean the same thing on
    every flag; an explicit ``step`` that DISAGREES with an embedded one is a ``ValueError``,
    never a silent winner.
    """
    embedded_step: "int | None"
    path, embedded_step = split_run_spec(path, what="run spec")
    if embedded_step is not None:
        if step is not None and step != embedded_step:
            raise ValueError(
                f"run spec {path!r} names step {embedded_step} but step {step} was also passed — "
                "give the step once.")
        step = embedded_step
    apath = os.path.abspath(path)
    zip_path: str | None = None
    run_dir: str
    if step is not None:
        # Current layout puts checkpoints in <run>/checkpoints/; fall back to the legacy
        # root for older runs. run_dir is the run root either way (config search below
        # already covers checkpoints/ via zip_dir + run_dir + parent).
        ckpt_name = f"checkpoint_{step}_steps.zip"
        in_subdir = os.path.join(apath, "checkpoints", ckpt_name)
        zip_path = in_subdir if os.path.isfile(in_subdir) else os.path.join(apath, ckpt_name)
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
    # (ai_v5_5_popart_N_0607), NOT a "best_model"/"snapshots"/"checkpoints" subfolder the zip
    # happens to live in (an @step checkpoint now resolves under <run>/checkpoints/).
    name_dir = os.path.dirname(zip_dir) if os.path.basename(zip_dir) in ("best_model", "snapshots", "checkpoints") \
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

        team_strs, team_files = _read_trainee_pin(config_path)
        entries.append(FixedOpponentEntry(
            label=label, zip_path=zip_path, config_path=config_path,
            arch_signature=foreign.arch_signature, temperature=default_temperature,
            source_elo=_read_source_elo(config_path),
            team_str=(team_strs[0] if team_strs else None),
            team_file=(team_files[0] if team_files else None),
            team_strs=tuple(team_strs), team_files=tuple(team_files),
        ))
    return entries


def register_exploiter_for_eval(
    fixed: "list[FixedOpponentEntry]",
    exploiter: "FixedOpponentEntry | None",
) -> "tuple[list[FixedOpponentEntry], bool]":
    """AUTO-register the ``--exploiter`` target as an eval opponent (opponent-parity Proposal A).

    The exploiter's whole point is beating ONE target, but ``--exploiter`` alone never produced the
    verdict metric (``eval/win_rate_vs_ext_<target>``) — you had to remember to ALSO pass
    ``--stable-opponents <same target>`` (the audited footgun). This appends the resolved exploiter
    entry to the eval-side fixed-opponent list, DEDUP-GUARDED: if the target is already registered
    (same resolved ``zip_path``, or a colliding label), the list is returned unchanged — so a config
    that passes both flags (the historical recipe) is byte-identical. Training-mix participation is
    untouched (exploiter mode excludes ``--self-play``, so appended entries are eval-only).

    Returns ``(entries, appended)``; pure (no I/O) → unit-tested.
    """
    if exploiter is None:
        return fixed, False
    if any(e.zip_path == exploiter.zip_path for e in fixed):
        return fixed, False
    if any(e.label == exploiter.label for e in fixed):
        return fixed, False
    return list(fixed) + [exploiter], True


def _read_trainee_pin(config_path: str) -> "tuple[list[str], list[str]]":
    """The opponent run's OWN pinned team(s) ``(team_strs, team_files)`` — the fold-back contract.

    A specialist run records its pin in ``metadata.json:cli_args`` — ``trainee_team`` (one team) or
    ``trainee_teams`` (a MULTI-team z-cluster exploiter). When that run is later used as a
    stable/exploiter OPPONENT it must pilot THOSE teams, not the shared pool (otherwise a trapper
    exploiter folds back piloting random teams and the pressure it was trained to apply evaporates —
    the realized-matchup lesson applied to the opponent side). A multi-team specialist SAMPLES among
    its own teams, mirroring how it trained.

    Delegates to ``matchup_spec.read_recorded_trainee_teams`` — the SINGLE provenance reader shared
    with ``--distill-teacher '<model>:*'``, so the two consumers cannot drift. It is FAIL-LOUD: a
    recorded team file that is missing raises ``FileNotFoundError``; one whose content no longer
    matches the run's recorded fingerprint raises ``ValueError``. Generalist run → ``([], [])``.
    """
    from agents.training.matchup_spec import read_recorded_trainee_teams
    files = read_recorded_trainee_teams(config_path)
    strs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            strs.append(fh.read())
    return strs, files


def _read_source_elo(config_path: str) -> "float | None":
    """The stable opponent's OWN recorded ELO (``latest_eval.elo``) — a well-fit, bot-anchored rating,
    far better than a single-edge live estimate, and directly comparable since the bot anchors are
    cross-run-stable. Read from the co-located **``best_model.json``** sidecar first (the self-contained
    location ``write_best_model_sidecar`` produces), then ``metadata.json`` next to the config, then the
    run dir / parent (older runs keep it only at the run level). Best-effort → ``None`` if none found."""
    d = os.path.dirname(config_path)
    for cand in (os.path.join(d, "best_model.json"),               # the co-located best-model sidecar
                 os.path.join(d, "metadata.json"),                 # best_model/ metadata (if any)
                 os.path.join(os.path.dirname(d), "metadata.json")):  # run-level metadata (old runs)
        if not os.path.isfile(cand):
            continue
        try:
            elo = json.load(open(cand)).get("latest_eval", {}).get("elo")
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if isinstance(elo, (int, float)):
            return float(elo)
    return None
