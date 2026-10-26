"""X35's G-E READERS (F27 closed for the readers, 2026-10-20): the exploitability guard's fresh best responders, as
kit arms ``launch_x35.sh`` can check and launch (``design_compute_efficiency.md`` §5.6.4 "R2" / "G-E"; launch sheet §0
"G-E readers"; review F-RV-7). Nothing here launches, trains or plays.

WHAT A READER IS (§5.6.4). For an arm's seed-3501 final (b-ON) and for C0's seed-3501 final (b-OFF), ONE fresh best
responder each: a fork of that final trained against that final ALONE (``--exploiter``), piloting the pinned OFFENSE
5-team archetype (``exploiter_teamsets_2026-09-22.json``), for 5M steps, the two readers at matched budget, dose and
regime. ``gap = P(the reader beats its target) − 0.5`` is its POOLED post-fork greedy-vs-greedy vs-target rate (the
``ext_<target run>`` series of its own eval cycles, ``main.best_response_gap``'s default ``pooled`` statistic, as
population-loop round 1 read it). G-E FAILs iff the Newcombe 95 % LOWER bound of gap(ON) − gap(OFF) > 0
(``main.x35.readout.guard_e``): the arm is detected MORE exploitable than C0.

THE ARMS (``kit.ARMS``, phase :data:`READER_PHASE`): ``ge_a1`` / ``ge_a2`` / ``ge_d1`` / ``ge_d3`` / ``ge_n1p`` read A1 /
A2 / D1 / D3 / N1-P's final, and
``ge_c0`` reads C0's, the b-OFF reader EVERY arm's G-E shares. The kit seed is the TARGET's seed (3501, the only one
§5.6.4 registers); the reader trains at seed :data:`READER_SEED` (§5.6.1: "best-response readers 3531+"), the same on
both sides so the two readers share the dice. Run ``rb_x35_ge_<target tag>_s<target seed>``, e.g.
``rb_x35_ge_a1arch_s3501`` reads ``rb_x35_a1arch_s3501``.

THE ARGV (:func:`build_reader_argv`): ``--model F --exploiter F --no-self-play --trainee-teams <offense 5>
--fork-lr 5.6e-05 --fork-lr-freeze --behaviour-check warn --steps <F's step + 5M> --checkpoint-every-steps 1000000
--device cuda --seed 3531 --run-name …``, F = ``models/<target run>/final_model.zip``. ``--no-self-play`` is F16 (the
parent's ``self_play=True`` is restored typed and ``--exploiter`` excludes it); ``warn`` is F14. The arms' levers are
NOT inherited by a reader: ``--stable-opponents``, ``--pfsp-scale``, ``--n-sentinels`` and ``--team-mix`` live in
neither ``model_config.json`` nor the recipe surface (checked on X26's records, 2026-10-20), so every reader best-responds
in the SAME regime and only its TARGET differs.

THE TRIGGER (:func:`trigger`, ``kit stage-check`` = ``launch_x35.sh`` pre-check 9). A reader of arm X is ALLOWED only
when X's PRIMARY (A1 / A2: R1; D1 / D3 / N1-P: H) has a saved FINAL class GAIN (``reads/<arm>_look<k>.json``, the latest look);
not read yet, CONTINUE, or an INCONCLUSIVE primary is PENDING; a final NULL / LOSS / UNDECIDED is REFUSED (G-E decides
only between ADOPT-CANDIDATE and TRADE, so a reader of a non-GAIN arm buys nothing: launch sheet §0). ``ge_c0`` is
ALLOWED as soon as ANY of A1 / A2 / D1 / D3 / N1-P reads GAIN. Production readers also wait for the side's FP32_ALT commit (AM-8),
and run at the SIDE'S pin (``kit pin-check`` judges the reader runs with the side's batch-1 runs).

THE READ (:func:`r2_input`, ``python -m main.x35.ge r2-input --arm a1``): the ``r2`` block of ``readout.decide`` /
``decide-d1``'s input, ``{"on": [wins, games], "off": [wins, games]}``, from the two reader runs through
``main.best_response_gap``'s own reader (``read_exploiter``) and its matched gate (budget 2 %, dose 10 % from one
full-resolution LR record, regime): a mismatch is a REFUSAL, never a number. Each reader must have FINISHED
(``final_model.zip``), recorded the right target FILE at the target's step (``lineage.exploiter_target``), piloted
the offense archetype and trained >= 5M post-fork steps.

DECLARED HERE (choices §5.6.4 leaves open; each is a FINDING of this build, the launch sheet lists them):
* the readers fork FROM their target (``--model F``), as population-loop round 1's RB / RC did, not from P\\*;
* no ``--exploiter-keep-bots``: the target is the sole opponent (as ``a3_mx``); round 1's readers kept bots at 0.5;
* the dose is PINNED by construction: ``--fork-lr 5.6e-05 --fork-lr-freeze`` (production's ``recipe.fork``, E5),
  because each target's KL controller ends at its own rate and an inherited one could fail the 10 % dose gate;
* production only: X47 registers no G-E (``main.x35.x47``), and the toy calibration (AM-3) reads R1's class only;
* the 5M budget at the inherited 2M eval cadence gives ~2–3 vs-target cycles (~200–300 games per reader at the
  inherited 100 per cycle, UNVERIFIED until a reader runs), not §5.6.4's "n ≈ 400";
* a reader's own K9(b) is not judged (AM-6 covers the batch-1 ARMS).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: kit.Arm.phase of a reader (0 single, 1 exploiters, 2 folded in).
READER_PHASE = 3
#: Reader arm -> the arm whose seed final it reads. ``ge_c0`` is every arm's shared b-OFF reader.
READER_ARMS: Dict[str, str] = {"ge_a1": "a1", "ge_a2": "a2", "ge_d1": "d1", "ge_d3": "d3", "ge_n1p": "n1p",
                               "ge_c0": "c0"}
#: The arms whose PRIMARY GAIN needs G-E before a verdict (``readout.decide``: A1 / A2; ``d1.d1_verdict``: D1).
GAIN_ARMS = ("a1", "a2", "d1", "d3", "n1p")
#: §5.6.4: "for each arm's seed-3501 final … 5M steps".
L_READER = 5_000_000
TARGET_SEEDS = (3501,)
#: §5.6.1: "best-response readers 3531+"; the same on both sides (b-ON and b-OFF share the dice).
READER_SEED = 3531
#: production's ``recipe.fork`` fork rate (``designs/production_config.json``, E5), frozen: the dose is a constant.
READER_FORK_LR = "5.6e-05"
ARCHETYPE = "offense"
STAT = "pooled"
GAIN = "GAIN"
FINAL_PRIMARY = ("GAIN", "NULL", "LOSS", "UNDECIDED")


class ReaderError(ValueError):
    """A reader cannot be built, triggered or read (named; the launch script HOLDs on it)."""


def target_run(reader_arm: str, seed: int) -> str:
    """The run a reader reads: ``ge_a1`` s3501 -> ``rb_x35_a1arch_s3501``."""
    from main.x35.kit import run_name

    if reader_arm not in READER_ARMS:
        raise ReaderError(f"{reader_arm!r} is not a reader arm ({sorted(READER_ARMS)})")
    return run_name(READER_ARMS[reader_arm], seed)


def reader_run(reader_arm: str, seed: int) -> str:
    """The reader's own run: ``rb_x35_ge_<target tag>_s<target seed>`` (``kit.ARMS``' format)."""
    from main.x35.kit import run_name

    target_run(reader_arm, seed)
    return run_name(reader_arm, seed)


def offense_teams(teamsets: Optional[str] = None) -> str:
    """The pinned OFFENSE archetype's ``--trainee-teams`` value, from the COMMITTED teamsets file the meter reads."""
    from agents.training.best_response_gap import load_teamsets

    block = load_teamsets(teamsets).get(ARCHETYPE)
    arg = (block or {}).get("trainee_teams_arg")
    if not arg or len(str(arg).split(",")) != 5:
        raise ReaderError(f"the teamsets file names no 5-team {ARCHETYPE!r} trainee_teams_arg")
    return str(arg)


def target_final(reader_arm: str, seed: int, models_dir: str, *, min_step: Optional[int] = None) -> Dict[str, Any]:
    """The target's FINISHED final: ``{run, path, step, sha256}``. A missing ``final_model.zip`` (a running,
    interrupted or not-yet-launched arm) or one short of ``min_step`` (P*'s step + the arm's length) is a refusal."""
    from main.x35.kit import KitError, _final_step
    from main.x35.refpop import sha256_file

    run = target_run(reader_arm, seed)
    z = Path(models_dir) / run / "final_model.zip"
    if not z.is_file():
        raise ReaderError(f"{run}: no final_model.zip under {models_dir} (a reader reads a FINISHED arm only)")
    try:
        step = _final_step(str(z))
    except KitError as exc:
        raise ReaderError(str(exc)) from exc
    if min_step is not None and step < min_step:
        raise ReaderError(f"{run}: final at {step:,} < {min_step:,} (the arm did not reach its registered length)")
    return {"run": run, "path": str(z.resolve()), "step": step, "sha256": sha256_file(z), "standin": False}


def standin_target(path: str, step: int) -> Dict[str, Any]:
    """A VALIDATION-ONLY target (``validate_batch1.sh --reader-standin``): never a launch value."""
    return {"run": "STANDIN", "path": path, "step": int(step), "sha256": None, "standin": True}


def build_reader_argv(reader_arm: str, seed: int, target: Mapping[str, Any], *, behaviour: str = "warn",
                      ckpt_every: str = "1000000") -> List[str]:
    """The trainer argv (the launcher's own flags excluded) of one reader (module docs)."""
    from main.x35.kit import _arm

    _arm(reader_arm, seed)
    f = str(target["path"])
    return ["--model", f, "--exploiter", f, "--no-self-play", "--trainee-teams", offense_teams(),
            "--fork-lr", READER_FORK_LR, "--fork-lr-freeze", "--behaviour-check", behaviour,
            "--steps", str(int(target["step"]) + L_READER), "--checkpoint-every-steps", ckpt_every,
            "--device", "cuda", "--seed", str(READER_SEED), "--run-name", reader_run(reader_arm, seed)]


# ------------------------------------------------------------------------------------------------------ the trigger
def primary_class(stage: str, arm: str) -> Tuple[Optional[str], str]:
    """``(class, why)``: the arm's FINAL primary class from its latest saved look; ``None`` = not final yet (no read,
    a CONTINUE, or an INCONCLUSIVE primary that states no class)."""
    from main.x35.batch import PRIMARY_KEY, read_record

    key = PRIMARY_KEY.get(arm, "r1")      # readout.Verdict: r1; d1.D1Verdict: h
    for look in (2, 1):
        rec = read_record(stage, arm, look)
        if rec is None:
            continue
        p = rec.get(key) or {}
        if not p.get("final"):
            return None, f"{arm} look {look}: the primary is not final (outcome {rec.get('verdict')})"
        cls = p.get("class")
        if cls not in FINAL_PRIMARY:
            return None, f"{arm} look {look}: the primary states no class ({cls!r}; outcome {rec.get('verdict')})"
        return str(cls), f"{arm} look {look}: primary FINAL class {cls}"
    return None, f"{arm}: no saved read"


def trigger(stage: str, reader_arm: str) -> Tuple[str, str]:
    """ALLOWED / PENDING / REFUSED for a reader now (module docs), with the reason."""
    from main.x35.batch import ALLOWED, PENDING, REFUSED, side_gate

    if reader_arm not in READER_ARMS:
        raise ReaderError(f"{reader_arm!r} is not a reader arm")
    filled, why = side_gate(stage, "production")
    if not filled:
        return PENDING, why
    t = READER_ARMS[reader_arm]
    if t in GAIN_ARMS:
        cls, why = primary_class(stage, t)
        if cls is None:
            return PENDING, f"G-E reader of {t}: {why}"
        if cls == GAIN:
            return ALLOWED, f"G-E reader of {t}: {why}"
        return REFUSED, (f"G-E reader of {t}: {why}, not GAIN: G-E decides only ADOPT-CANDIDATE vs TRADE, a reader "
                         "of a non-GAIN arm buys nothing")
    reads = [(a,) + primary_class(stage, a) for a in GAIN_ARMS]
    gains = [a for a, cls, _ in reads if cls == GAIN]
    if gains:
        return ALLOWED, f"the shared b-OFF reader: {gains} read GAIN"
    whys = "; ".join(w for _, _, w in reads)
    if any(cls is None for _, cls, _ in reads):
        return PENDING, f"the shared b-OFF reader waits for an arm's GAIN: {whys}"
    return REFUSED, f"no arm reads GAIN, so no G-E is needed: {whys}"


# ---------------------------------------------------------------------------------------------------------- the read
def _check_reader(run: Any, reader_arm: str, seed: int, models_dir: str) -> List[str]:
    probs: List[str] = []
    want = target_run(reader_arm, seed)
    if run.run != reader_run(reader_arm, seed):
        probs.append(f"{run.run}: not the registered reader run {reader_run(reader_arm, seed)}")
    if not os.path.isfile(os.path.join(run.run_dir, "final_model.zip")):
        probs.append(f"{run.run}: no final_model.zip (an unfinished reader is not read)")
    z = Path(models_dir) / want / "final_model.zip"
    if not run.target_file or not z.is_file() or os.path.realpath(run.target_file) != os.path.realpath(z):
        probs.append(f"{run.run}: its recorded target {run.target_file!r} is not {z}")
    elif run.target_step is not None:
        from main.x35.kit import _final_step

        if int(run.target_step) != _final_step(str(z)):
            probs.append(f"{run.run}: recorded target step {run.target_step} != {want}'s final step")
    if run.archetype != ARCHETYPE:
        probs.append(f"{run.run}: archetype {run.archetype!r} ({run.membership}), not {ARCHETYPE!r}")
    if run.budget is None or run.budget < L_READER:
        probs.append(f"{run.run}: post-fork budget {run.budget} < {L_READER:,}")
    return probs


def r2_input(arm: str, seed: int = TARGET_SEEDS[0], *, models_dir: str,
             reader: Any = None) -> Dict[str, Any]:
    """The ``r2`` block for ``arm``'s read (``readout.read`` / ``d1.read_d1``): ON = ``ge_<arm>``'s reader, OFF =
    ``ge_c0``'s. ``reader`` (tests) replaces ``best_response_gap.read_exploiter``. Refuses (:class:`ReaderError`)
    rather than return counts the registered G-E is not defined on."""
    from agents.training import best_response_gap as E

    if arm not in GAIN_ARMS:
        raise ReaderError(f"arm {arm!r} has no G-E (readers exist for {list(GAIN_ARMS)})")
    on_arm, off_arm = f"ge_{arm}", "ge_c0"
    teamsets = E.load_teamsets()
    read = reader or E.read_exploiter
    runs = []
    probs: List[str] = []
    for ra in (on_arm, off_arm):
        rd = os.path.join(models_dir, reader_run(ra, seed))
        try:
            r = read(rd, teamsets)
        except E.BestResponseGapError as exc:
            raise ReaderError(f"{reader_run(ra, seed)}: {exc}") from exc
        probs += _check_reader(r, ra, seed, models_dir)
        runs.append(r)
    if probs:
        raise ReaderError("; ".join(probs))
    try:
        E.check_matched(runs)
    except E.UnmatchedError as exc:
        raise ReaderError(str(exc)) from exc
    on, off = runs
    return {"on": list(on.cell(STAT)), "off": list(off.cell(STAT)), "stat": STAT,
            "series_regime": E.SERIES_REGIME,
            "readers": {side: {"run": r.run, "target_file": r.target_file, "target_step": r.target_step,
                               "budget": r.budget, "dose_rate": r.dose_rate, "dose_source": r.dose_source,
                               "endpoint": list(r.cell("endpoint")), "cycles": len(r.post_fork)}
                        for side, r in (("on", on), ("off", off))}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x35.ge", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("r2-input", help="the r2 block of an arm's read from its G-E readers (exit 2 = REFUSED)")
    r.add_argument("--arm", required=True, choices=GAIN_ARMS)
    r.add_argument("--seed", type=int, default=TARGET_SEEDS[0], choices=TARGET_SEEDS)
    r.add_argument("--models-dir", default=None)
    t = sub.add_parser("trigger", help="may this reader launch now? exit 0 ALLOWED, 1 not")
    t.add_argument("--arm", required=True, choices=sorted(READER_ARMS))
    t.add_argument("--stage", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "trigger":
            st, why = trigger(a.stage, a.arm)
            print(f"[x35 ge] {a.arm}: {st}: {why}")
            return 0 if st == "ALLOWED" else 1
        from utils.paths import main_models_dir

        md = a.models_dir or str(main_models_dir() or "")
        print(json.dumps(r2_input(a.arm, a.seed, models_dir=md), indent=1))
        return 0
    except ReaderError as exc:
        print(f"[x35 ge] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
