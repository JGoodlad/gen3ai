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

import glob
import json
import os
import sys
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
    # WHICH FILE, and HOW it was chosen (gen3_last_snapshot_resolution_v1). A bare run dir names a
    # run, not a file, and until 2026-09-06 it silently meant the BOT-WIN-RATE `best_model` export —
    # which for 2 of 8 R5F teachers was a ~0.93M-step checkpoint rather than the ~2.93M final, with
    # nothing on disk recording it (ledger 2026-09-06, probe H8). These three ride every entry so a
    # startup line and a `lineage` block can both state the answer instead of implying it.
    # `resolution_rung` is the specific rung that fired (`latest_txt` / `highest_checkpoint` /
    # `final_model` / `best_model_fallback` / `explicit_step` / `explicit_zip`); `resolution_rule`
    # is its coarse class; `num_timesteps` is None when the zip declares no step ("unknown", never 0).
    resolution_rung: "str | None" = None
    resolution_rule: "str | None" = None
    num_timesteps: "int | None" = None

    def provenance(self) -> str:
        """``<zip> @<steps> [rung=… rule=…]`` — the one line a startup log prints per opponent."""
        steps = f"{self.num_timesteps:,} steps" if self.num_timesteps is not None else "steps unknown"
        return (f"{self.zip_path} @{steps} "
                f"[rung={self.resolution_rung or '?'} rule={self.resolution_rule or '?'}]")

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


# ---------------------------------------------------------------------------------------------
# WHICH FILE A BARE RUN DIRECTORY MEANS — the ONE resolution rule
# (``gen3_last_snapshot_resolution_v1``, 2026-09-06)
# ---------------------------------------------------------------------------------------------
#
# THE OWNER DECISION. A bare run directory resolves to the run's **LAST SNAPSHOT**, not to
# ``best_model/best_model.zip``. Verbatim: *"I would either prefer us do best against target or
# just do the last snapshot. I feel like best against target will always have a nuance that we
# need to keep track of, whereas the last one is probably what our metrics would measure anyway."*
#
# WHY IT CHANGED. ``best_model/best_model.zip`` is exported on **BOT win rate** — an opponent set
# that has nothing to do with what a teacher is being distilled FOR. Ledger 2026-09-06 (probe H8)
# measured the consequence: for 2 of 8 unfunded R5F teachers (``ai_v9_94_R5F02``,
# ``ai_v9_98_R5F06``) the exported file was a ~0.93M-step exploiter rather than the ~2.93M final,
# so "the teacher" a fold distilled from was neither the last snapshot nor the best against its
# target — and *nothing recorded which file was used*. Every meter this programme banks scores a
# run at its END, so the last snapshot is what the metrics already measure.
#
# THE RUNGS, in the order they are tried for a BARE run dir (no ``@step``):
#
#   1. ``latest_txt``          — ``<run>/latest.txt``, a run-RELATIVE path (root CLAUDE.md)
#   2. ``highest_checkpoint``  — the highest-step ``checkpoints/checkpoint_<N>_steps.zip``
#                                (and the SIGUSR1 ``checkpoint_forced_<N>_<HHMMSS>.zip``), legacy
#                                run-root copies included
#   3. ``final_model``         — ``final_model.zip`` / ``final_model_interrupted.zip``
#   4. ``best_model_fallback`` — ``best_model/best_model.zip`` (legacy ``<run>/best_model.zip``),
#                                LAST, only for a run that has nothing else, and it says so on
#                                stderr when it fires
#
# 🚨 **DISAGREEMENT: the higher ``num_timesteps`` wins, not the earlier rung.** Rungs 1-3 are
# every name for "the end of this run", and they disagree in two directions:
#
#   * a COMPLETED run writes ``latest.txt -> final_model.zip`` *after* its last periodic
#     checkpoint, so ``latest.txt`` is AHEAD of ``checkpoints/`` (measured on the eight R5F runs
#     2026-09-06: ``final_model.zip`` @28,115,184 vs the highest checkpoint @28,067,760 — 47,424
#     steps apart, and rung 1 fires for every one of them);
#   * an INTERRUPTED / crashed run can leave ``latest.txt`` pointing at a file that a later
#     ``final_model_interrupted.zip`` has since passed.
#
# Taking the earlier rung would be right in the first case and wrong in the second, so neither
# ordering is the rule: the rule is **the file that trained furthest**, with the rung order used
# only to break a tie (or when no candidate declares a step at all, which is the unreadable-zip
# case). ``num_timesteps`` is read from the SB3 zip's plain-JSON ``data`` member — no torch, no
# model load — falling back to the ``checkpoint_<N>_steps.zip`` filename.
#
# EXPLICIT NAMES BYPASS THE LADDER ENTIRELY. ``<run>@<step>`` is an explicit checkpoint
# (``explicit_step``) and a path that names a ``.zip`` — INCLUDING ``best_model/best_model.zip`` —
# is used verbatim (``explicit_zip``). Naming the file is how you pin it; the ladder is only for
# the caller who did not.
#
# EVERY CONSUMER GOES THROUGH HERE. ``--distill-teacher`` (``main/train/model_build.py``),
# ``--stable-opponents`` and ``--exploiter`` (via :func:`resolve_stable_opponents`),
# ``--exploiter-ladder``, ``--warmstart-consensus`` (``agents/training/warmstart.py``),
# ``--distill-anchor-parent`` (``main/train/callbacks.py``) and ``--win-prob-pbrs-source``
# (``main/train/model_build.py``) all call this function and no other — the census in
# ``run_spec_test.py`` fails when one of them stops.
#
# 🚨 **EVERY TEACHER LOADED BEFORE THIS CHANGE WENT THROUGH THE OLD RULE** (``best_model`` first,
# then ``final_model.zip``, then ``<run>/best_model.zip``). No run on disk today recorded which
# file it got, so a pre-change run's teacher identity is not recoverable from its metadata —
# ``main.lineage`` says "resolved file not recorded" rather than re-resolving it under the new
# rule and presenting today's answer as history. The probe scripts under
# ``designs/research_state/measurements/arch_transfer_2026-09-05/`` (``content_locality_v2``,
# ``exploiter_competence``) import this function deliberately to reproduce what a fold loaded;
# they measured the OLD rule's files and stay as records of it.
#
# NOT VERSIONED. This changes which FILE a run loads, never a weight shape, so it is absent from
# ``ModelVersion.check_compatible`` / ``arch_signature`` by design.

#: The rungs, in the order they are TRIED. The two ``explicit_*`` members are not rungs of the
#: directory ladder — they are the ways a caller names a file outright — but they share the
#: vocabulary so a recorded provenance string always answers "how was this file chosen?".
RESOLUTION_RUNGS = ("explicit_step", "explicit_zip", "latest_txt", "highest_checkpoint",
                    "final_model", "best_model_fallback")

#: The COARSE class each rung reports as ``rule`` (the four names the ledger quotes).
_RUNG_RULE = {
    "explicit_step": "explicit_step",
    "explicit_zip": "explicit_zip",
    "latest_txt": "last_snapshot",
    "highest_checkpoint": "last_snapshot",
    "final_model": "last_snapshot",
    "best_model_fallback": "best_model_fallback",
}

#: Rungs whose files are the ladder's "end of the run" tier — ranked by ``num_timesteps``, with the
#: tuple ORDER breaking a tie. ``best_model_fallback`` is tried only after this tier is empty.
_LAST_SNAPSHOT_RUNGS = ("latest_txt", "highest_checkpoint", "final_model")


@dataclass(frozen=True)
class ResolvedModel:
    """WHICH FILE a run spec named, and HOW it was chosen — the provenance a fold must record.

    ``rung`` is the specific ladder step that fired (one of :data:`RESOLUTION_RUNGS`); ``rule`` is
    its coarse class (``explicit_step`` / ``explicit_zip`` / ``last_snapshot`` /
    ``best_model_fallback``). Both are recorded because a reader needs the class to compare two
    fleets and the rung to reproduce one file. ``num_timesteps`` is ``None`` when the zip declares
    no step and its name does not carry one — a real answer ("unknown"), never a substituted 0.
    """
    zip_path: str
    config_path: str
    run_base: str
    run_dir: str
    rung: str
    rule: str
    num_timesteps: "int | None"

    def describe(self) -> str:
        """One line for a startup log / a provenance print."""
        steps = f"{self.num_timesteps:,} steps" if self.num_timesteps is not None else "steps unknown"
        return f"{self.zip_path} @{steps} [rung={self.rung} rule={self.rule}]"


def _checkpoint_steps(zip_path: str) -> "int | None":
    """``num_timesteps`` out of an SB3 zip without importing torch (delegates to the ONE reader,
    ``agents.training.lineage.checkpoint_num_timesteps``). Imported lazily: ``lineage`` reaches
    back into this module for the same resolution, and a module-level pair would be a cycle."""
    from agents.training.lineage import checkpoint_num_timesteps
    try:
        return checkpoint_num_timesteps(zip_path)
    except Exception:  # noqa: BLE001 — provenance must never break a load
        return None


def _checkpoint_candidates(run_dir: str) -> "list[str]":
    """Every resumable checkpoint zip under ``run_dir`` — current ``checkpoints/`` and the legacy
    run-root layout, periodic AND the SIGUSR1 ``checkpoint_forced_*`` (which ``cf_producer``
    learned the hard way to include: globbing only the periodic form ranks a forced save below
    every periodic one and walks a reader BACKWARDS)."""
    out: "list[str]" = []
    for root in (os.path.join(run_dir, "checkpoints"), run_dir):
        out += glob.glob(os.path.join(root, "checkpoint_*_steps.zip"))
        out += glob.glob(os.path.join(root, "checkpoint_forced_*.zip"))
    return sorted(set(out))


def _step_sort_key(path: str) -> "tuple[int, int, float]":
    """Rank zips by DECLARED step; an unparseable one falls back to mtime so a hand-placed file
    stays reachable rather than invisible."""
    s = _checkpoint_steps(path)
    return (0 if s is None else 1, s or 0, os.path.getmtime(path))


def _run_dir_candidates(run_dir: str) -> "list[tuple[str, str]]":
    """``[(rung, abs_path)]`` for a BARE run dir, in rung order, deduplicated by realpath.

    Dedup keeps the EARLIER rung: ``latest.txt`` normally names the very file ``final_model`` or
    ``highest_checkpoint`` would find, and reporting that as ``latest_txt`` is the honest answer —
    it is the rung that actually fired.
    """
    cands: "list[tuple[str, str]]" = []

    rel = ""
    try:
        with open(os.path.join(run_dir, "latest.txt"), encoding="utf-8") as fh:
            rel = fh.read().strip()
    except OSError:
        rel = ""
    if rel:
        p = rel if os.path.isabs(rel) else os.path.join(run_dir, rel)
        if os.path.isfile(p):
            cands.append(("latest_txt", os.path.abspath(p)))

    ckpts = _checkpoint_candidates(run_dir)
    if ckpts:
        cands.append(("highest_checkpoint", os.path.abspath(max(ckpts, key=_step_sort_key))))

    finals = [os.path.join(run_dir, n)
              for n in ("final_model.zip", "final_model_interrupted.zip")]
    finals = [p for p in finals if os.path.isfile(p)]
    if finals:
        cands.append(("final_model", os.path.abspath(max(finals, key=_step_sort_key))))

    for p in (os.path.join(run_dir, "best_model", "best_model.zip"),
              os.path.join(run_dir, "best_model.zip")):
        if os.path.isfile(p):
            cands.append(("best_model_fallback", os.path.abspath(p)))
            break

    seen: "set[str]" = set()
    out: "list[tuple[str, str]]" = []
    for rung, p in cands:
        key = os.path.realpath(p)
        if key in seen:
            continue
        seen.add(key)
        out.append((rung, p))
    return out


def _pick_run_dir_zip(run_dir: str) -> "tuple[str, str, int | None] | None":
    """``(rung, zip_path, num_timesteps)`` for a bare run dir, or None when the run holds nothing.

    THE RULE: among the LAST-SNAPSHOT rungs present, the highest ``num_timesteps`` wins; rung
    order breaks a tie and decides when no candidate declares a step at all.
    ``best_model_fallback`` is consulted only when that tier is empty.
    """
    cands = _run_dir_candidates(run_dir)
    tier = [(r, p) for r, p in cands if r in _LAST_SNAPSHOT_RUNGS]
    if not tier:
        tier = [(r, p) for r, p in cands if r == "best_model_fallback"]
    if not tier:
        return None
    scored = [(rung, p, _checkpoint_steps(p)) for rung, p in tier]

    def _key(i: int) -> "tuple[bool, int, int]":
        _rung, _p, steps = scored[i]
        # (a step is known, how far it trained, EARLIER rung) — the last element breaks a tie and
        # is what decides when nothing declares a step at all.
        return (steps is not None, steps if steps is not None else 0, -i)

    rung, zip_path, steps = scored[max(range(len(scored)), key=_key)]
    return rung, zip_path, steps


def resolve_model_ref(path: str, step: "int | None" = None, *, warn: bool = True) -> ResolvedModel:
    """THE choke point: a run spec → the file to load, its config, and HOW it was chosen.

    See the block comment above for the rung order and the disagreement rule.
    :func:`_resolve_zip_and_config` is the 3-tuple wrapper every existing caller (and the probe
    scripts that import it) still uses; this is the same resolution with the provenance attached.

    🚨 **THE ``@step`` SPLIT HAPPENS HERE, at the ONE choke point every run-spec consumer reaches**
    (``gen3_run_spec_split_v1``). ``--stable-opponents`` parses its own ``@step`` and passes it in;
    every OTHER caller — ``--distill-teacher``, ``--win-prob-pbrs-source``,
    ``--distill-anchor-parent``, ``--warmstart-consensus`` — calls with ``step=None`` and a raw
    spec string, so before that fix they could only ever resolve a run DIR. A spec that still
    carries a suffix is split here rather than refused, which makes ``<run>@<step>`` mean the same
    thing on every flag; an explicit ``step`` that DISAGREES with an embedded one is a
    ``ValueError``, never a silent winner.

    ``model_config.json`` is searched next to the zip, then the run dir, then the zip's parent — so
    a ``best_model/best_model.zip`` still finds the run-level config. Raises ``FileNotFoundError``
    if no zip or no config resolves.
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
    zip_path: "str | None" = None
    rung: "str | None" = None
    num_timesteps: "int | None" = None
    run_dir: str
    if step is not None:
        # Current layout puts checkpoints in <run>/checkpoints/; fall back to the legacy
        # root for older runs. run_dir is the run root either way (config search below
        # already covers checkpoints/ via zip_dir + run_dir + parent).
        ckpt_name = f"checkpoint_{step}_steps.zip"
        in_subdir = os.path.join(apath, "checkpoints", ckpt_name)
        zip_path = in_subdir if os.path.isfile(in_subdir) else os.path.join(apath, ckpt_name)
        run_dir = apath
        rung, num_timesteps = "explicit_step", step
    elif apath.endswith(".zip") and os.path.isfile(apath):
        zip_path, run_dir, rung = apath, os.path.dirname(apath), "explicit_zip"
    elif os.path.isdir(apath):
        run_dir = apath
        picked = _pick_run_dir_zip(apath)
        if picked is not None:
            rung, zip_path, num_timesteps = picked
    elif os.path.isfile(apath + ".zip"):
        zip_path, run_dir, rung = apath + ".zip", os.path.dirname(apath), "explicit_zip"
    else:
        run_dir = apath

    if zip_path is None or not os.path.isfile(zip_path) or rung is None:
        raise FileNotFoundError(
            f"run spec: no model .zip found for {path!r}"
            + (f" @step {step}" if step is not None else "")
            + " (expected a run dir carrying latest.txt / checkpoints/ / final_model.zip / "
              "best_model/best_model.zip, a direct .zip, or a run dir + @step)."
        )
    if num_timesteps is None:
        num_timesteps = _checkpoint_steps(zip_path)

    zip_dir = os.path.dirname(zip_path)
    config_path: "str | None" = None
    for d in (zip_dir, run_dir, os.path.dirname(zip_dir)):
        cand = os.path.join(d, "model_config.json")
        if os.path.isfile(cand):
            config_path = cand
            break
    if config_path is None:
        raise FileNotFoundError(
            f"run spec: {path!r} resolved to {zip_path!r} but no sibling "
            "model_config.json was found (a stable opponent must carry its arch provenance)."
        )
    # The run NAME is the default label — the dir the user thinks of as "the run"
    # (ai_v5_5_popart_N_0607), NOT a "best_model"/"snapshots"/"checkpoints" subfolder the zip
    # happens to live in (an @step checkpoint now resolves under <run>/checkpoints/).
    name_dir = os.path.dirname(zip_dir) if os.path.basename(zip_dir) in ("best_model", "snapshots", "checkpoints") \
        else zip_dir
    if rung == "best_model_fallback" and warn:
        # PRINTED, not merely recorded: this is the BOT-WIN-RATE export, reached only because the
        # run carries no latest.txt, no checkpoints/ and no final_model — the H8 finding's shape.
        print(f"[run-spec] {path!r}: FALLING BACK to {zip_path} — this run has no latest.txt, no "
              "checkpoints/ and no final_model*.zip, so the only file left is the BOT-WIN-RATE "
              "best_model export (see gen3_last_snapshot_resolution_v1).", file=sys.stderr)
    return ResolvedModel(
        zip_path=zip_path, config_path=config_path,
        run_base=os.path.basename(os.path.normpath(name_dir)), run_dir=run_dir,
        rung=rung, rule=_RUNG_RULE[rung], num_timesteps=num_timesteps,
    )


def _resolve_zip_and_config(path: str, step: "int | None") -> "tuple[str, str, str]":
    """``(zip_path, config_path, run_basename)`` — :func:`resolve_model_ref` without the provenance.

    THE SIGNATURE IS FROZEN. Offline probe scripts under
    ``designs/research_state/measurements/arch_transfer_2026-09-05/`` import this by name to
    reproduce exactly the call ``main/train/model_build.py`` makes for a ``--distill-teacher``; a
    3-tuple is what they unpack. New call sites that want the rung / ``num_timesteps`` should call
    :func:`resolve_model_ref` instead.
    """
    r = resolve_model_ref(path, step)
    return r.zip_path, r.config_path, r.run_base


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
        ref = resolve_model_ref(raw["path"], raw["step"])
        zip_path, config_path, run_base = ref.zip_path, ref.config_path, ref.run_base
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
            resolution_rung=ref.rung, resolution_rule=ref.rule,
            num_timesteps=ref.num_timesteps,
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
