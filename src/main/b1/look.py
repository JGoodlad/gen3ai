"""BOUNDARY-1's LOOKS: which checkpoints a look plays, the guards a run must pass to count, and the read.

    python -m main.b1.look names --look 50M --pin <sha>              # per panel: family, request, pairs, rule
    python -m main.b1.look plan  --look 50M --pin <sha> --out <W>    # W/cells_plan.json + W/look.json; exit 3 REFUSED
    python -m main.b1.look read  --work <W>                          # W/result/verdict.json (and stdout)
    python -m main.b1.look read  --work <W> --kakuna width=POSITIVE --kakuna breadth=SPANS-ZERO   # the 100M look
    python -m main.b1.look plan  --look 100M --pin <sha> --out <W> --survivor breadth   # after width's STOP-HARM

The registration is ``designs/research_state/measurements/boundary1_scratch_2026-11-09/PREREG.md`` (§5, §8, §10);
the pin is registered THERE (§3) and is an argument here, never a constant of this tree (a commit cannot name
itself). The statistic and every class are :mod:`main.b1.cross`; the measurement folder's ``looks/drive_look.sh``
plays the plan.

**After a STOP-HARM** (``--survivor <the factor still read>``): the look holds the two arms that went on (the
surviving factor's ON arm and the control), 4 cells at two runs an arm, its own request names, and its read is
:func:`main.b1.cross.read_survivor`: one simple effect on every view, NO class, to the owner.

**A look** is one matched step: ``50M`` (the harm stop), ``75M`` (REPORTED), ``100M`` (the decision), and ``EQ``, the
REPORTED equal-compute read (the wide arms at ``--eq-step``, the narrow arms at 100M).

**A look's checkpoint** (:func:`find_checkpoint`) is the first ``checkpoints/checkpoint_<N>_steps.zip`` a run saved
at or after the look's step; ``N`` must be within one checkpoint interval of it. ``final_model.zip`` is never read:
it is overwritten by the phase-2 resume, and a look must read the same bytes whenever it is read again.

**A run counts** (:func:`check_run`, PREREG §10) iff its record says: it was LAUNCHED by its arm's registered argv,
token for token (``original_command``, which no restart or resume rewrites, against :func:`main.b1.arms.build_argv`
at a registered step target); it is a FRESH run; ONE pin, the registered one, from step 0 to the checkpoint;
``ent_coef`` 0.05 and ten epochs; and its LAST process still resolved the arm's lever keys (:data:`CONFIG_KEYS`) as
the arm's own argv resolves them, so a resume that dropped the width or the team mix is refused. Across the look,
every run's last process must record the SAME public ``cli_args`` as the first control run's, except its own keys
(:data:`RUN_OWN_KEYS`) and its arm's lever keys (:func:`recipe_drift`: any other knob that moved in one run and not
in the control; the keys a resume rewrites are compared only between processes of one kind, and the rollout shape
they stand for at ``metadata.json``'s top level, :data:`TRUE_KEYS`), and all runs must share one ``torch_version``
and one ``init_num_threads``. Anything else REFUSES the plan: a run that exists and is
not clean is a question for the orchestrator, never a silent drop.

**The cells** are :meth:`main.b1.cross.Design.cells` in order (24 at two runs an arm), each ``[player .zip, opponent
.zip]``, the same list for every panel. One ledger FAMILY a panel, one REQUEST a look and panel (:func:`family`,
:func:`request`), so a re-run resumes and a look never reads another look's games.

**The read** maps each panel's ledger cells to runs by checkpoint sha256 (:func:`main.b1.cross.scores_from_cells`);
any defective cell on any panel makes the look INCONCLUSIVE, never a class. So does a ROW that was not played as
registered (:func:`row_problems`): at another commit than the pin, on another panel's team set, or on a second
device or backend within one look (a near-tie greedy decision can flip with the device, ``main.h2h.play``).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from main.b1 import arms as A
from main.b1 import cross as X

LOOK_STEPS: Dict[str, int] = {"50M": 50_000_000, "75M": 75_000_000, "100M": 100_000_000}
#: The REPORTED equal-compute look: the wide arms at ``--eq-step``, the others at 100M.
EQ = "EQ"
WIDE_ARMS: Tuple[str, ...] = X.FACTORS["width"][0]
#: cross arm -> the kit's arm key (``main.b1.arms``).
KIT_ARM: Dict[str, str] = {"C": "c", "W": "w", "Br": "br", "WB": "wb"}
PROTOCOL = "gen3_eval_protocol_v1_h2h"
ENT_COEF, N_EPOCHS = 0.05, 10
INCONCLUSIVE = "INCONCLUSIVE"
#: The keys of a run's recorded ``cli_args`` (its LAST process's resolved namespace) that must equal what its arm's
#: argv resolves to. ``allow_nonproduction_arch`` is a permission, not a configuration, and is not compared.
CONFIG_KEYS: Tuple[str, ...] = ("d_model", "team_mix", "team_mix_sides", "trunk_ffn_dim")
#: The recorded ``cli_args`` keys that are a run's OWN (its identity, the file it resumed from, its step target):
#: every other public key must equal the first control run's, except the arm's lever keys (:func:`recipe_drift`).
#: Measured: sibling runs of one recipe differ in exactly these (x35t and x60t's seeds, 2026-11; the kit's smoke).
RUN_OWN_KEYS: Tuple[str, ...] = ("model", "run_dir", "run_name", "seed", "steps")
#: ``cli_args`` keys a RESUME rewrites whatever the run is: afterwards they hold the parser's default or an
#: "inherited" label, never the run's value (measured on the kit's stop-and-resume smoke, 2026-11-09: a fresh process
#: against its own phase-2 resume). They are compared only between two last processes of ONE kind
#: (:func:`process_kind`); what they stand for is compared at its source instead (:data:`TRUE_KEYS`; the lever keys).
RESUME_REWRITES: Tuple[str, ...] = (
    "allow_nonproduction_arch", "arch", "batch_size", "eval_mirrored_pairs_source", "eval_sentinel_greedy_source",
    "n_steps", "promote_threshold_source", "promotion_sprt_source", "unified_damage", "unified_moves")
#: ``metadata.json``'s TOP-LEVEL fields that hold the run's true rollout shape after any resume (``cli_args``' copies
#: are inert on one): every run's must equal the first control run's.
TRUE_KEYS: Tuple[str, ...] = ("batch_size", "n_steps")
_CKPT = re.compile(r"^checkpoint_(\d+)_steps\.zip$")


class LookError(ValueError):
    """An unknown look or panel, or a malformed argument."""


def family(panel: str) -> str:
    if panel not in X.PANELS:
        raise LookError(f"panel {panel!r} is not one of {X.PANELS}")
    return f"b1_scratch_h2h_{panel}"


def request(look: str, panel: str, *, eq_step: Optional[int] = None, survivor: Optional[str] = None) -> str:
    """The request id of one look on one panel (also the rows' label): unique across the whole ledger. The EQ
    look's id carries its step (``b1_eq61m_<panel>``), so a plan at another step can never resume into its games;
    a survivor look's carries the surviving factor (``b1_100m_breadthonly_<panel>``)."""
    if look not in X.LOOKS:
        raise LookError(f"look {look!r} is not one of {X.LOOKS}")
    family(panel)
    if survivor is not None and survivor not in X.SURVIVOR_ARMS:
        raise LookError(f"--survivor {survivor!r} is not one of {sorted(X.SURVIVOR_ARMS)}")
    arms = X.SURVIVOR_ARMS[survivor] if survivor else X.ARMS
    look_steps(look, eq_step, arms)  # validates --eq-step against the look
    tag = f"eq{eq_step // A.CHECKPOINT_EVERY}m" if look == EQ and eq_step is not None else look.lower()
    return f"b1_{tag}{f'_{survivor}only' if survivor else ''}_{panel}"


def names(look: str, pin: str, *, eq_step: Optional[int] = None, survivor: Optional[str] = None
          ) -> Dict[str, Dict[str, Any]]:
    """panel -> what the driver registers and plays: family, request, pairs a cell, the family's rule line."""
    return {panel: {"family": family(panel), "request": request(look, panel, eq_step=eq_step, survivor=survivor),
                    "pairs": X.PAIRS[panel], "rule": family_rule(panel, pin)} for panel in X.PANELS}


def family_rule(panel: str, pin: str) -> str:
    """The rule line ``python -m main.eval_ledger family-register --rule`` records for a panel's family."""
    return (f"boundary1_scratch_2026-11-09/PREREG.md §5 (pin {pin[:8]}): the 2 x 2 cross on panel {panel}, "
            f"{X.PAIRS[panel]} mirrored pairs a cell, one request a look; a factor's statistic is "
            f"main.b1.cross.factor_stat (D = mean - 50, within-arm variances floored at {X.FLOOR_PP} pp, t on "
            f"4(n-1) df, {int(round(100 * X.LEVEL))} % interval); classes main.b1.cross.read_look "
            f"(width on N, breadth on forum72 with the net rule)")


def look_steps(look: str, eq_step: Optional[int] = None, arms: Sequence[str] = X.ARMS) -> Dict[str, int]:
    """cross arm -> the step its checkpoint is taken at for ``look`` (``arms``: a survivor look's two)."""
    if look == EQ:
        if eq_step is None or not (0 < eq_step < LOOK_STEPS["100M"]) or eq_step % A.CHECKPOINT_EVERY:
            raise LookError(f"the EQ look needs --eq-step, a positive multiple of {A.CHECKPOINT_EVERY:,} below "
                            f"{LOOK_STEPS['100M']:,} (got {eq_step!r})")
        if not any(a in WIDE_ARMS for a in arms):
            raise LookError(f"the EQ look needs a wide arm; {tuple(arms)} holds none")
        return {a: (eq_step if a in WIDE_ARMS else LOOK_STEPS["100M"]) for a in arms}
    if look not in LOOK_STEPS:
        raise LookError(f"look {look!r} is not one of {X.LOOKS}")
    if eq_step is not None:
        raise LookError("--eq-step belongs to the EQ look alone")
    return {a: LOOK_STEPS[look] for a in arms}


def look_arms(look: str, survivor: Optional[str] = None) -> Tuple[str, ...]:
    """The arms a look holds, in :data:`main.b1.cross.ARMS` order: all four, or a survivor look's two. A survivor
    look exists only AFTER the 50M look (the stop is read there on all eight runs)."""
    if survivor is None:
        return X.ARMS
    if survivor not in X.SURVIVOR_ARMS:
        raise LookError(f"--survivor {survivor!r} is not one of {sorted(X.SURVIVOR_ARMS)}")
    if look == "50M":
        raise LookError("the 50M look is read on all eight runs; a survivor look comes after it")
    return tuple(a for a in X.ARMS if a in X.SURVIVOR_ARMS[survivor])


def design_seeds(replace: Optional[int] = None, arms: Sequence[str] = X.ARMS) -> Dict[str, Tuple[int, ...]]:
    """cross arm -> its seeds, with ``replace`` swapped for the spare in its own arm. ``replace`` is the REPLACED
    registered seed (for example 5103), never the spare's own number."""
    seeds = {a: tuple(A.SEEDS[KIT_ARM[a]]) for a in arms}
    if replace is None:
        return seeds
    hit = [a for a in arms if replace in seeds[a]]
    if not hit:
        raise LookError(f"--replace {replace} is not a registered seed of {tuple(arms)}: give the seed the spare "
                        f"({A.SPARE_SEED}) REPLACED")
    seeds[hit[0]] = tuple(A.SPARE_SEED if s == replace else s for s in seeds[hit[0]])
    return seeds


# ----------------------------------------------------------------------------------------------- the guards
def find_checkpoint(run_dir: Path, step: int) -> Tuple[Optional[Path], Optional[int], Optional[str]]:
    """``(zip, N, None)`` for the first checkpoint at or after ``step``, or ``(None, None, why)``."""
    best: Optional[Tuple[int, Path]] = None
    ck = run_dir / "checkpoints"
    for p in (sorted(ck.iterdir()) if ck.is_dir() else []):
        m = _CKPT.match(p.name)
        if m and int(m.group(1)) >= step and (best is None or int(m.group(1)) < best[0]):
            best = (int(m.group(1)), p)
    if best is None:
        return None, None, f"no checkpoint at or after {step:,}"
    if best[0] >= step + A.CHECKPOINT_EVERY:
        return None, None, (f"its first checkpoint at or after {step:,} is {best[0]:,}, more than one interval "
                            f"({A.CHECKPOINT_EVERY:,}) late")
    return best[1], best[0], None


def _json(v: Any) -> Any:
    return json.loads(json.dumps(v, default=str))


def expected_levers(arm: str, seed: int, pin: str) -> Dict[str, Any]:
    """What the run's recorded ``cli_args`` must hold on :data:`CONFIG_KEYS`: the arm's own argv, resolved."""
    ns = A.resolved_namespace(A.build_argv(KIT_ARM[arm], seed, pin))
    return {k: _json(ns.get(k)) for k in CONFIG_KEYS}


def launched_as_registered(original_command: Any, arm: str, seed: int, pin: str) -> Optional[str]:
    """``None`` iff the run's first launch was the arm's registered argv at a registered step target."""
    try:
        got = shlex.split(str(original_command or ""))[1:]
    except ValueError:
        return "original_command cannot be parsed"
    if any(got == A.build_argv(KIT_ARM[arm], seed, pin, steps) for steps in A.PHASE_STEPS):
        return None
    return f"original_command is not the registered argv of arm {arm} seed {seed}: {' '.join(got)!r}"


def _one_pin(history: Any, pin: str, upto: int) -> Optional[str]:
    seen = [(str(p.get("git_hash"))[:8], p.get("first_step"), p.get("last_step")) for p in (history or [])]
    if len(seen) != 1 or history[0].get("git_hash") != pin:
        return f"pin_history {seen} is not one entry at {pin[:8]}"
    first, last = history[0].get("first_step"), history[0].get("last_step")
    if first != 0:
        return f"its pin entry starts at {first!r}, not step 0"
    if not isinstance(last, int) or last < upto:
        return f"its pin entry ends at {last!r}, before the checkpoint's {upto:,}"
    return None


def check_run(run_dir: Path, arm: str, seed: int, pin: str, step: int, levers: Mapping[str, Any]
              ) -> Tuple[List[str], Dict[str, Any]]:
    """``(problems, facts)`` for one run at one look step (module docstring, "A run counts")."""
    probs: List[str] = []
    facts: Dict[str, Any] = {"arm": arm, "seed": seed, "run": run_dir.name}
    try:
        md = json.loads((run_dir / "metadata.json").read_text())
    except (OSError, ValueError) as e:
        return [f"metadata.json cannot be read ({type(e).__name__})"], facts
    zip_path, n, why = find_checkpoint(run_dir, step)
    if zip_path is None or n is None:
        return [why or "no checkpoint"], facts
    try:
        side = json.loads(zip_path.with_suffix(".json").read_text())
    except (OSError, ValueError) as e:
        return [f"{zip_path.name}'s sidecar cannot be read ({type(e).__name__})"], facts
    if side.get("num_timesteps") != n:
        probs.append(f"{zip_path.name}'s sidecar says num_timesteps {side.get('num_timesteps')!r}")
    for where, hist in (("the run", md.get("pin_history")), (zip_path.name, side.get("pin_history"))):
        bad = _one_pin(hist, pin, n)
        if bad:
            probs.append(f"{where}: {bad}")
    lin = md.get("lineage") or {}
    if lin.get("role") != "fresh" or lin.get("fork_parent") is not None:
        probs.append(f"lineage is not a FRESH run (role {lin.get('role')!r})")
    cli = md.get("cli_args") or {}
    bad = launched_as_registered(md.get("original_command"), arm, seed, pin)
    if bad:
        probs.append(bad)
    if cli.get("n_epochs") != N_EPOCHS:
        probs.append(f"n_epochs {cli.get('n_epochs')!r} != {N_EPOCHS}")
    for where, got in (("metadata.ent_coef", md.get("ent_coef")), ("cli_args.ent_coef", cli.get("ent_coef")),
                       (f"{zip_path.name} ent_coef", side.get("ent_coef"))):
        if isinstance(got, bool) or not isinstance(got, (int, float)) or abs(float(got) - ENT_COEF) > 1e-12:
            probs.append(f"{where} {got!r} != {ENT_COEF}")
    for k in CONFIG_KEYS:
        if _json(cli.get(k)) != levers.get(k):
            probs.append(f"cli_args.{k} {cli.get(k)!r} is not the arm's {levers.get(k)!r}")
    facts.update({"checkpoint": str(zip_path), "steps": n, "torch_version": md.get("torch_version"),
                  "init_num_threads": md.get("init_num_threads"),
                  "_cli": {k: v for k, v in cli.items() if not str(k).startswith("_")},
                  "_true": {k: md.get(k) for k in TRUE_KEYS}})
    return probs, facts


def process_kind(cli: Mapping[str, Any]) -> str:
    """What launched a run's LAST process: ``fresh`` (no ``--model``), ``restart`` (the launcher's own six-hour
    restart child, which keeps the fresh argv's ``--run-name`` beside its ``--model``) or ``resume`` (a typed
    ``--model`` command: phase 2, and every restart child after it)."""
    if cli.get("model") is None:
        return "fresh"
    return "restart" if cli.get("run_name") is not None else "resume"


def recipe_drift(arm: str, cli: Mapping[str, Any], control_cli: Mapping[str, Any]) -> List[str]:
    """The public ``cli_args`` keys in which a run's LAST process differs from the first control run's, beyond its
    own keys (:data:`RUN_OWN_KEYS`) and its arm's lever keys: any knob at all that moved in one run and not in the
    control (a recipe flag typed on one resume, a default that resolved differently). Recorded against recorded, so
    nothing here depends on the machine the look is planned on. Between two processes of different kinds
    (:func:`process_kind`) the keys a resume rewrites (:data:`RESUME_REWRITES`) are not compared: they differ by
    kind, never by run."""
    allowed = set(RUN_OWN_KEYS) | set(A.LEVER_DESTS[KIT_ARM[arm]])
    if process_kind(cli) != process_kind(control_cli):
        allowed |= set(RESUME_REWRITES)
    return sorted(k for k in set(cli) | set(control_cli)
                  if k not in allowed and _json(cli.get(k)) != _json(control_cli.get(k)))


def _design(runs: Mapping[str, Sequence[str]], survivor: Optional[str]) -> Any:
    return X.pair_design_of(runs) if survivor else X.design_of(runs)


def plan(models: Path, look: str, pin: str, *, eq_step: Optional[int] = None, replace: Optional[int] = None,
         survivor: Optional[str] = None, levers: Optional[Mapping[Tuple[str, int], Mapping[str, Any]]] = None
         ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """``(plan, [])`` or ``(None, problems)``. ``survivor``: the factor still read after the other's STOP-HARM (the
    look then holds two arms). ``levers`` (tests) overrides :func:`expected_levers` per (arm, seed)."""
    from utils.hashing import sha256_file

    arms = look_arms(look, survivor)
    steps = look_steps(look, eq_step, arms)
    seeds = design_seeds(replace, arms)
    probs: List[str] = []
    runs: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    for arm in arms:
        for seed in seeds[arm]:
            run_dir = models / A.run_name(KIT_ARM[arm], seed)
            if not run_dir.is_dir():
                probs.append(f"{run_dir.name}: no run folder under {models}")
                continue
            want = levers[(arm, seed)] if levers is not None else expected_levers(arm, seed, pin)
            p, facts = check_run(run_dir, arm, seed, pin, steps[arm], want)
            probs.extend(f"{run_dir.name}: {x}" for x in p)
            runs[arm].append(facts)
    for what in ("torch_version", "init_num_threads"):
        seen = sorted({str(f.get(what)) for fs in runs.values() for f in fs if "checkpoint" in f})
        if len(seen) > 1:
            probs.append(f"{what} differs across the runs: {seen}")
    clis = {(a, f["run"]): (f.pop("_cli"), f.pop("_true")) for a in arms for f in runs[a] if "_cli" in f}
    control = next((c for (a, _), c in clis.items() if a == "C"), None)
    for (a, run), c in clis.items():
        if control is not None and c is not control:
            drift = recipe_drift(a, c[0], control[0])
            if drift:
                probs.append(f"{run}: its last process differs from the control's in {drift} "
                             f"(beyond its own keys and its arm's lever)")
            moved = sorted(k for k in TRUE_KEYS if _json(c[1].get(k)) != _json(control[1].get(k)))
            if moved:
                probs.append(f"{run}: metadata.json's {moved} differ from the control's "
                             f"({ {k: c[1].get(k) for k in moved} } against { {k: control[1].get(k) for k in moved} })")
    if probs:
        return None, probs
    for fs in runs.values():
        for f in fs:
            f["sha256"] = sha256_file(f["checkpoint"])
    design = _design({a: tuple(f["run"] for f in runs[a]) for a in arms}, survivor)
    zip_of = {f["run"]: f["checkpoint"] for fs in runs.values() for f in fs}
    shas = [f["sha256"] for fs in runs.values() for f in fs]
    if len(set(shas)) != len(shas):
        return None, ["two runs' checkpoints have the same sha256"]
    return {"look": look, "pin": pin, "steps": steps, "eq_step": eq_step, "replaced_seed": replace,
            "survivor": survivor, "runs": runs, "cells": [[zip_of[p], zip_of[o]] for p, o in design.cells()],
            "names": names(look, pin, eq_step=eq_step, survivor=survivor)}, []


# -------------------------------------------------------------------------------------------------- the read
def draw_summary(cells: Sequence[Any]) -> Dict[str, Any]:
    """One panel's drawn games (the head-to-head's timeouts), REPORTED: the total and the worst cell's share."""
    games = sum(int(c.games) for c in cells)
    draws = sum(int(c.d) for c in cells)
    worst = max((int(c.d) / int(c.games) for c in cells if int(c.games) > 0), default=0.0)
    return {"games": games, "draws": draws, "share": (draws / games if games else None), "worst_cell_share": worst}


def read(look_doc: Mapping[str, Any], cells_by_panel: Mapping[str, Sequence[Any]],
         kakuna: Optional[Mapping[str, str]] = None, row_reasons: Sequence[str] = ()) -> Dict[str, Any]:
    """The look's verdict from each panel's ledger cells. ``kakuna`` (the 100M look only): factor -> the class of
    its confirming contrast, which adds ``factors.<f>.confirming`` and never changes an outcome. ``row_reasons``
    (:func:`row_problems`): rows not played as registered, which make the look INCONCLUSIVE like a defective cell."""
    look = str(look_doc["look"])
    survivor = look_doc.get("survivor")
    arms = look_arms(look, survivor)
    design = _design({a: tuple(f["run"] for f in look_doc["runs"][a]) for a in arms}, survivor)
    run_of = {f["sha256"]: f["run"] for a in arms for f in look_doc["runs"][a]}
    by_panel: Dict[str, Dict[Tuple[str, str], float]] = {}
    reasons: List[str] = list(row_reasons)
    for panel in X.PANELS:
        scores, why = X.scores_from_cells(cells_by_panel.get(panel, ()), run_of, design, X.PAIRS[panel])
        if scores is None:
            reasons.extend(f"{panel}: {w}" for w in why)
        else:
            by_panel[panel] = scores
    if kakuna and (look != "100M" or survivor):
        raise LookError("the confirming meter belongs to the full 100M look alone")
    if kakuna and set(kakuna) != set(X.FACTORS):
        raise LookError(f"--kakuna needs one class for each of {sorted(X.FACTORS)}")
    head = {"look": look, "pin": look_doc["pin"], "steps": dict(look_doc["steps"]),
            "checkpoints": {f["run"]: {"steps": f["steps"], "sha256": f["sha256"]}
                            for a in arms for f in look_doc["runs"][a]},
            "draws": {panel: draw_summary(cells_by_panel.get(panel, ())) for panel in X.PANELS}}
    if reasons:
        return {**head, "outcome": INCONCLUSIVE, "reasons": reasons}
    if survivor:
        return {**head, **X.read_survivor(design, by_panel, look)}
    out = {**head, **X.read_look(design, by_panel, look)}
    for f, cls in (kakuna or {}).items():
        lab, to_owner = X.confirm(out["factors"][f]["outcome"], cls)
        out["factors"][f]["confirming"] = {"contrast": X.CONFIRMING[f], "kakuna_class": cls, "label": lab,
                                           "to_owner": to_owner}
    return out


def row_problems(rows_by_panel: Mapping[str, Sequence[Mapping[str, Any]]], pin: str,
                 team_sets: Mapping[str, str]) -> List[str]:
    """Why a look's ledger ROWS were not played as registered (``[]`` = they were): every row at the pin; every
    row of a panel on that panel's registered team set (``team_sets``); ONE device and backend across the look."""
    out: List[str] = []
    where: Dict[Tuple[str, str], int] = {}
    for panel in X.PANELS:
        rows = rows_by_panel.get(panel, ())
        other = sorted({str(r.get("commit"))[:12] for r in rows if r.get("commit") != pin})
        if other:
            out.append(f"{panel}: rows played at {other}, not the pin {pin[:12]}")
        sets = sorted({str((r.get("regime") or {}).get("team_set")) for r in rows} - {team_sets[panel]})
        if sets:
            out.append(f"{panel}: rows on team set(s) {[x[:16] for x in sets]}, not the panel's "
                       f"{team_sets[panel][:16]}")
        for r in rows:
            c = r.get("compute") or {}
            k = (str(c.get("device")), str(c.get("backend")))
            where[k] = where.get(k, 0) + 1
    if len(where) > 1:
        out.append(f"the look was played on more than one device/backend: {sorted(where.items())}")
    return out


#: A look's read: one panel's A/B family at the h2h protocol, greedy on both sides, mirrored, across runs (the terms
#: of ``main.h2h.cross.FAMILY_READ_OFF``, declared here because the reader gate wants each reader's own declaration).
LOOK_READ = L.ReaderDecl(
    name="main.b1.look.family", purposes=frozenset({"ab"}),
    regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h", play="greedy", opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="across_runs",
    decision_kind="ab_verdict")


def ledger_cells(root: Path, look_doc: Mapping[str, Any]) -> Tuple[Dict[str, List[Any]], List[str]]:
    """``(each panel's cells, row problems)`` of a planned look from the eval COUNT ledger; a family with no row yet
    reads as no cells. The request ids are the plan's own (``look.json``'s ``names``)."""
    from main.h2h import panel as PN

    cells: Dict[str, List[Any]] = {}
    rows: Dict[str, List[Mapping[str, Any]]] = {}
    for panel in X.PANELS:
        req = str(look_doc["names"][panel]["request"])
        try:
            got = L.read(LOOK_READ, root=root, family=family(panel))
            cells[panel] = list(dict(L.looks(got)).get(req, []))
            rows[panel] = [r for r in got.rows if (r.get("request") or {}).get("id") == req]
        except (ValueError, OSError) as e:  # an unreadable family is named, and the look reads INCONCLUSIVE
            print(f"NOTE: {panel}: no cells read ({type(e).__name__}: {e})", file=sys.stderr)
            cells[panel], rows[panel] = [], []
    return cells, row_problems(rows, str(look_doc["pin"]), {p: PN.panel_team_set_id(p) for p in X.PANELS})


# ------------------------------------------------------------------------------------------------------ CLI
def _models() -> Path:
    from utils.paths import main_models_dir

    m = main_models_dir()
    if m is None:
        raise LookError("no models/ archive here (set GEN3AI_MODELS_DIR)")
    return m


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.b1.look", description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("names", help="per panel: family, request, pairs, and the family's rule line")
    p = sub.add_parser("plan", help="check every run and write the cells of a look")
    for q in (n, p):
        q.add_argument("--look", required=True, choices=X.LOOKS)
        q.add_argument("--pin", required=True, help="the registered pin (PREREG §3)")
        q.add_argument("--eq-step", type=int, help="the EQ look: the wide arms' step")
        q.add_argument("--survivor", choices=sorted(X.SURVIVOR_ARMS),
                       help="after a STOP-HARM of the OTHER factor: the factor still read (PREREG §8)")
    p.add_argument("--out", required=True, help="the look's work directory")
    p.add_argument("--replace", type=int,
                   help="the REGISTERED seed the spare replaced, for example 5103 (PREREG §10); never the spare's")
    r = sub.add_parser("read", help="read a played look and print its verdict")
    r.add_argument("--work", required=True, help="the look's work directory (holds look.json)")
    r.add_argument("--ledger", help="the ledger root (default: <models>/_ledger)")
    r.add_argument("--kakuna", action="append", default=[], metavar="FACTOR=CLASS",
                   help="the 100M look: the class of each factor's confirming contrast (give both factors)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "names":
            look_arms(a.look, a.survivor)
            print(json.dumps(names(a.look, a.pin, eq_step=a.eq_step, survivor=a.survivor), indent=1))
            return 0
        if a.cmd == "plan":
            doc, probs = plan(_models(), a.look, a.pin, eq_step=a.eq_step, replace=a.replace, survivor=a.survivor)
            if doc is None:
                print("REFUSED: " + "; ".join(probs), file=sys.stderr)
                return 3
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "cells_plan.json").write_text(json.dumps(doc["cells"], indent=1))
            (out / "look.json").write_text(json.dumps(doc, indent=1))
            print(len(doc["cells"]))
            return 0
        work = Path(a.work)
        doc = json.loads((work / "look.json").read_text())
        kak = dict(kv.split("=", 1) for kv in a.kakuna) if a.kakuna else None
        root = Path(a.ledger) if a.ledger else _models() / "_ledger"
        cells, row_reasons = ledger_cells(root, doc)
        verdict = read(doc, cells, kak, row_reasons)
        (work / "result").mkdir(parents=True, exist_ok=True)
        (work / "result" / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
        print(json.dumps(verdict, default=str))
        return 0
    except (LookError, A.ArmError, X.CrossInputError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
