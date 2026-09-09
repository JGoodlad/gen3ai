"""``python -m main.ops.critic_read <arm> --control <control> --out <dir>`` — the CRITIC LADDER's
registered read, as ONE command.

Registered 2026-09-08 (ledger *REGISTRATION · THE CRITIC LADDER*, design note
``designs/research_state/winprob_critic_ladder_2026-09-08.md``): every ladder arm is read at 10M
by the CRITIC meters and NEVER by strength, as a **DELTA against the control with the delta's
own confidence interval**. Before this module that read was three bespoke agent efforts per arm
(the 75M read's tests 1-3 took 2-7 h each and produced three differently-shaped reports). It is
now one invocation and one report, so two arms are read by identical code and their numbers are
comparable by construction.

**What it composes** — it invents no statistic:

1. **IDENTITY** — ``agents.training.cf_audit`` on the arm's last COMPLETE eval cycle with NO
   ``--checkpoint`` (ledger ``1a1ad063``: the anchor and the label rollouts both use the trace
   dir's own ``snapshot.zip``), then the 75M read's readout, promoted into
   :mod:`main.ops.critic_readouts`: reproduction rate, bias ``V - p_hat`` overall / by turn
   bucket / by stratum, Murphy with the base-rate cap, and the ``corr(turn,V) - corr(turn,MC)``
   clock-tracking contrast.
2. **RESOLUTION** — ``python -m main.critic_gate`` for the registered G1-G4 rows against the
   committed calibration baseline.
3. **CONDITIONING** (added 2026-09-09) — :mod:`main.ops.conditioning_meters`, promoted from the
   mixture diagnostic and the probe read: the between-opponent SPREAD IDENTITY of ``V`` against
   the outcome (target 1.0), the bias-on-opponent-Elo slope, the own-team leave-one-battle-out
   win-rate R² of ``V`` and the opponent-class AUC of ``V``. Computed on the RECORDED ``V`` of
   the read cycle, so no model forward is needed.
4. **THE DELTA** — every quantity recomputed as ARM - CONTROL with a battle-clustered
   **difference of independent bootstraps**, labelled DETECTED / WITHIN FLOOR / NOT DETECTED.
5. **QUOTA MATCHING** (added 2026-09-09, ON by default) — :mod:`main.ops.quota_match`. The ladder's
   arms are traced at different outcome quotas, and a conditioning row whose estimator is a FIT on
   the frame (or an uncorrected second moment) has an expectation that moves with the frame's
   SIZE — which Horvitz-Thompson reweighting does not touch. Those rows are recomputed on the
   richer side subsampled to the poorer side's REALIZED per-opponent capture profile, over
   `--quota-match-seeds` seeded draws, and the label is decided on the MATCHED delta. The
   as-traced value is printed beside it, marked UNMATCHED and never labelled. `--no-quota-match`
   opts out and then those rows carry a hard `UNMATCHED — not a reading` marker INSTEAD of a
   label: the 2026-09-09 RETRACTION withdrew a DETECTED that was entirely this artefact, and the
   tool does not print that label again on an unequal frame.

**REFUSALS OVER SILENCE.** A missing manifest, a cycle that has not collected, a trace dir whose
npz carries no ``win_probs`` (the privileged arm's channel must be present AT EVAL — that is what
this check is for), an anchor reproduction rate under the label-trust gate, or a draw/timeout
share over the cap each print a REFUSAL naming the cause and exit 2. A partial table is never
emitted as if it were complete.

**THE FLOOR.** DETECTED requires the delta's CI to clear the replicate floor, and the ladder's
floor does not exist until a second control replicate does. Absent ``--floor-json`` the tool
prints that in the report and labels detections explicitly as *vs ZERO — NO FLOOR*.

Nothing is written under ``models/``: the run archive is opened read-only and every output goes
under ``--out``.
"""
from __future__ import annotations

import argparse
import re
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from main.ops import conditioning_meters as CM
from main.ops import critic_readouts as R
from main.ops import quota_match as QM
from main.ops.run_ref import interpreter, refuse, resolve_run_dir

# The report's own constants live with the report; re-exported here so `main.ops.critic_read`
# stays the one name a caller has to know.
from main.ops.critic_read_render import (HEADLINES, IDENTITY_STRATA,  # noqa: E402
                                         TOOL, TOOL_VERSION, WEIGHTING_NOTE, WEIGHTINGS,
                                         ledger_line, render_md)


# --------------------------------------------------------------------------- read roots

from main.ops import eval_trace_gen as ETG  # noqa: E402  (the provenance vocabulary, one copy)


def resolve_read_root(run_dir: Path, override: Optional[str], *, role: str) -> Path:
    """Where this side's TRACES are read from — the run itself, or an override.

    An override may be spelled either way, because both are natural: the SHADOW RUN DIR that
    ``main.ops.eval_trace_gen --out`` produced, or the ``eval_traces/step_<N>`` cycle inside it.
    Both normalise to the run-shaped root, because that is what every consumer downstream wants:
    ``cf_audit`` takes a run dir positionally and has no ``--traces`` flag, and the generated dir
    is laid out to satisfy it (its own ``model_config.json``, ``metadata.json``, a read-only
    ``snapshots/`` symlink, and the cycle's own ``snapshot.zip``).

    🚨 The override does NOT redirect ``main.critic_gate``. That reads the run's LADDER,
    ``eval_results.jsonl`` and TensorBoard — run-level history that an offline cycle neither has
    nor could have — so the registered G1-G4 rows keep coming from the real run and the report
    says so. Everything that is a function of the CYCLE (cf_audit's identity labels, the capture
    weights, the gauge's calibration arrays, the conditioning meters, the quota match) follows
    the override.
    """
    if not override:
        return run_dir
    d = Path(override).resolve()
    if not d.is_dir():
        refuse(f"REFUSING: --{role}-traces {d} is not a directory.")
    if (d / "eval_traces").is_dir():
        return d
    if re.match(r"^step_\d+$", d.name) and (d.parent.parent / "eval_traces").is_dir():
        return d.parent.parent
    refuse(f"REFUSING: --{role}-traces {d} is not a generated cycle.",
           "  Pass either the directory `main.ops.eval_trace_gen --out` wrote (it holds an "
           "`eval_traces/` subdirectory), or the `eval_traces/step_<N>` cycle inside it.")
    raise AssertionError("unreachable")  # pragma: no cover


def check_comparable(arm: Dict[str, Any], ctl: Dict[str, Any]) -> None:
    """REFUSE a delta between two cycles that are not samples of the same population.

    🚨 THIS IS THE WHOLE POINT OF THE PROVENANCE BLOCK. An offline cycle and a live one differ in
    games per opponent, in how many pool sentinels there are, and — decisively — in whether the
    traced battles are a random sample or the live outcome QUOTA's loss-enriched slice. Every
    conditioning row is a statistic OF that frame. Differencing one against the other produces a
    number that is mostly the difference between the two designs, and it would carry a CI that
    describes neither. The v3 quota match corrects a difference in capture RATE between two frames
    of the same shape; it cannot turn a 400-game full-capture frame into a 100-game quota one.

    Two OFFLINE frames must additionally agree on the spec — games, opponent set, capture, and
    the sentinel regime — for the same reason, and for none of the reasons a seed differs: two
    seeds are two draws from one population, which is exactly what a delta wants.
    """
    a_man, c_man = arm.get("manifest") or {}, ctl.get("manifest") or {}
    a_gen, c_gen = ETG.is_generated(a_man), ETG.is_generated(c_man)
    if a_gen != c_gen:
        off, live = ("arm", "control") if a_gen else ("control", "arm")
        refuse(
            "REFUSING: the two sides are DIFFERENT POPULATIONS — "
            f"the {off} cycle is OFFLINE-GENERATED and the {live} cycle is LIVE.",
            f"  {off:8s} {ETG.population_of(a_man if a_gen else c_man)}",
            f"  {live:8s} {ETG.population_of(c_man if a_gen else a_man)}",
            "  A delta across that boundary is mostly the difference between the two eval "
            "DESIGNS — more games, a different sentinel count, and a random sample against the "
            "live outcome quota's loss-enriched slice — and its CI describes neither side.",
            "  THE FIX: read both sides from the same kind of cycle. Generate the missing one "
            f"with `python -m main.ops.eval_trace_gen <{live} run>@<step> --games N --sentinels K "
            "--out DIR` at the SAME --games/--sentinels as the other side, then pass both "
            "--arm-traces and --control-traces. Nothing is read and nothing is concluded.")
    if not a_gen:
        return
    # 🚨 AN INCOMPLETE CYCLE IS A SMALLER FRAME, whatever its `n_games` says. This is the case the
    # spec check below CANNOT catch on its own: a cycle whose workers died mid-plan still records
    # the nominal games and the full opponent set, so it compares EQUAL to a complete one while
    # carrying 71% of its battles. That is precisely the frame-size artefact the 2026-09-09
    # RETRACTION was about, arriving through the provenance block instead of the quota.
    for role, man in (("arm", a_man), ("control", c_man)):
        comp = ETG.completeness(man)
        if comp["complete"] is None:
            refuse(
                f"REFUSING: the {role} cycle cannot be certified COMPLETE — its manifest records "
                "neither the battle plan it was given nor the battles it played.",
                "  A cycle whose workers died part-way still writes a well-formed manifest — "
                "nominal games, the full opponent set, capture rates of 1.0 — so without those "
                "counts nothing on disk separates a finished cycle from a truncated one, and a "
                "truncated one is a SMALLER FRAME whose fitted rows do not compare.",
                f"  THE FIX: re-generate the {role} cycle with the current tool "
                "(`python -m main.ops.eval_trace_gen … --force`). Nothing is read and nothing is "
                "concluded.")
        if comp["complete"] is False:
            refuse(
                f"REFUSING: the {role} cycle is INCOMPLETE — "
                f"{comp['battles_played']:,} of {comp['battles_expected']:,} battles played "
                f"({comp['shortfall']:,} short).",
                "  Its workers died, or it was killed, part-way through the plan. The frame is "
                "therefore SMALLER than its games-per-opponent spec advertises, and a fitted "
                "conditioning row's expectation moves with frame size — so a delta against a "
                "complete cycle would be partly the difference between the two frames.",
                f"  THE FIX: re-generate the {role} cycle "
                "(`python -m main.ops.eval_trace_gen … --force`) and check its log ends without "
                "an INCOMPLETE line. Nothing is read and nothing is concluded.")
    a_spec, c_spec = ETG.spec_of(a_man), ETG.spec_of(c_man)
    diff = {k: (a_spec[k], c_spec[k]) for k in a_spec if a_spec[k] != c_spec[k]}
    if diff:
        refuse("REFUSING: both cycles are offline-generated, but to DIFFERENT specs — they are "
               "not two draws from one population.",
               *[f"  {k}: arm={av!r}  control={cv!r}" for k, (av, cv) in sorted(diff.items())],
               "  A differing --seed is fine and is deliberately not checked: two seeds are two "
               "draws from the SAME population. Games, the opponent set, the capture rule and "
               "the sentinel regime are not.",
               "  Regenerate the two sides at one spec. Nothing is read and nothing is concluded.")


# --------------------------------------------------------------------------- cycle selection

def live_pids_for_run(run_base: str) -> List[int]:
    """PIDs whose cmdline names this run — a launcher, its trainer child, or an eval worker.

    Read straight from ``/proc`` rather than by shelling out, so this cannot match its OWN argv
    (which contains the run name) the way a ``pgrep -f`` would. That confusion has cost this tree
    a session before; the rule is kill/inspect by explicit PID, never by pattern over your own
    command line.
    """
    me = {os.getpid(), os.getppid()}
    hits: List[int] = []
    for entry in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            pid = int(entry.split("/")[2])
            if pid in me:
                continue
            parts = open(entry, "rb").read().decode("utf-8", "replace").split("\0")
        except (OSError, ValueError):
            continue
        if not parts or TOOL in " ".join(parts):
            continue
        for i, tok in enumerate(parts):
            if tok == "--run-name" and i + 1 < len(parts) and parts[i + 1] == run_base:
                hits.append(pid)
                break
            if tok == "--run-dir" and i + 1 < len(parts) and \
                    os.path.basename(parts[i + 1].rstrip("/")) == run_base:
                hits.append(pid)
                break
    return sorted(set(hits))


def cycle_status(trace_dir: str) -> Tuple[bool, str, Optional[dict]]:
    """``(complete, why, manifest)`` for one ``eval_traces/step_N`` directory.

    A cycle is COMPLETE when its ``eval_manifest.json`` exists **and carries a ``selection``
    block**. The manifest is written before a single battle is played, with ``selection: null``,
    and PATCHED at COLLECT (``agents.training.eval_callback.record_eval_selection``) — so the
    presence of the manifest alone says only that the cycle STARTED, while the selection block is
    written after every trace of the cycle is on disk. Rule 16 asks for the manifest; this is the
    check that actually answers "is anything still writing here".
    """
    path = os.path.join(trace_dir, "eval_manifest.json")
    if not os.path.exists(path):
        return False, "no eval_manifest.json — SELECTION UNKNOWN (rule 16)", None
    try:
        with open(path) as fh:
            man = json.load(fh)
    except (OSError, ValueError) as exc:
        return False, f"eval_manifest.json unreadable: {exc}", None
    if not isinstance(man.get("selection"), dict):
        return False, ("eval_manifest.json records no `selection` block — the cycle has not "
                       "COLLECTED, so traces may still be arriving"), man
    return True, "manifest + selection recorded", man


def draw_share(manifest: dict) -> Optional[float]:
    """Share of PLAYED battles that ended in a draw — a tie or a 250-turn timeout.

    The manifest's own per-opponent counts, so it is the population rate and not the capture
    quota's. A schema that does not record ``battles_drawn`` returns ``None``, which the caller
    reports as NOT RECORDED — never as zero.
    """
    per = (manifest.get("selection") or {}).get("opponents") or {}
    played = drawn = 0
    for rec in per.values():
        if "battles_drawn" not in rec:
            return None
        played += int(rec.get("battles_played", 0))
        drawn += int(rec.get("battles_drawn", 0))
    return (drawn / played) if played else None


def winprob_coverage(trace_dir: str) -> Dict[str, int]:
    """How many of the cycle's state npz files actually carry a ``win_probs`` column.

    This is the check the privileged-critic arm needs: ``--value-true-team`` rides a separate
    obs key that must be present AT EVAL, and both critic meters take V and P(win) from these
    npz. A tree missing the column would otherwise read as "no data" rather than as the arm
    having been measured on a V it never trained.
    """
    import numpy as np

    seen = with_wp = 0
    for npz in sorted(glob.glob(os.path.join(trace_dir, "*", "*_states.npz"))):
        seen += 1
        try:
            with np.load(npz) as d:
                if "win_probs" in d and "values" in d:
                    with_wp += 1
        except (OSError, ValueError, KeyError):
            pass
    return {"n_npz": seen, "n_with_winprob": with_wp, "n_missing": seen - with_wp}


def pick_cycle(run_dir: Path, *, on_live: str, step: Optional[int]) -> Dict[str, Any]:
    """The run's LAST COMPLETE eval cycle (or the pinned ``--step``), with why it was chosen."""
    base = run_dir.name
    dirs = sorted(glob.glob(str(run_dir / "eval_traces" / "step_*")),
                  key=lambda p: int(os.path.basename(p).split("_")[1]))
    if not dirs:
        refuse(f"REFUSING: {base} has no eval_traces/step_* directories — nothing to read.")
    steps = [int(os.path.basename(p).split("_")[1]) for p in dirs]
    pids = live_pids_for_run(base)
    if pids and on_live == "refuse":
        refuse(f"REFUSING: {base} is LIVE (pid {', '.join(map(str, pids))}) and --on-live is "
               "`refuse`. A cycle another process may still be writing is not a measurement. "
               "Re-run when the arm finishes, or pass --on-live skip-newest / use.")
    candidates = list(zip(steps, dirs))
    dropped_newest = False
    if pids and on_live == "skip-newest" and len(candidates) > 1 and step is None:
        candidates = candidates[:-1]
        dropped_newest = True
    if step is not None:
        candidates = [(s, d) for s, d in candidates if s == step]
        if not candidates:
            refuse(f"REFUSING: {base} has no eval_traces/step_{step} "
                   f"(available: {', '.join(str(s) for s in steps)}).")
    tried: List[str] = []
    for s, d in reversed(candidates):
        ok, why, man = cycle_status(d)
        if ok:
            return {"run": base, "step": s, "trace_dir": d, "manifest": man,
                    "live_pids": pids, "on_live": on_live,
                    "dropped_newest_because_live": dropped_newest,
                    "steps_on_disk": steps, "why": why}
        tried.append(f"step_{s}: {why}")
    refuse(f"REFUSING: {base} has no COMPLETE eval cycle to read.", *[f"  {t}" for t in tried],
           "  A cycle is complete when eval_manifest.json exists AND carries its `selection` "
           "block (written at COLLECT, after every trace of the cycle is on disk).")
    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- subprocesses

def _run(argv: Sequence[str], log_path: str, *, nice: int, cwd: Optional[str] = None) -> int:
    cmd = (["nice", "-n", str(nice)] if nice else []) + list(argv)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only: the GPU belongs to the live arm
    with open(log_path, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env=env)


def _tail(path: str, n: int = 25) -> str:
    try:
        return "".join(open(path).readlines()[-n:])
    except OSError:
        return "(no log)"


# --------------------------------------------------------------------------- per-run read

#: the IDENTITY + GATE readout's cache-key version. 🚨 DELIBERATELY DECOUPLED from
#: ``TOOL_VERSION``: the identity half costs a ~25-minute ``cf_audit`` run, and folding the
#: REPORT's version into its key throws every readout on disk away whenever a new SECTION is
#: added — which is not a change to any number this key covers. Bump this one only when something
#: that changes the identity or gate NUMBERS changes.
READOUT_FINGERPRINT_VERSION = 1


def _fingerprint(cycle: Dict[str, Any], args) -> Dict[str, Any]:
    return {"run_dir": str(cycle["run_dir"]),
            # 🚨 The READ ROOT is part of the key. Without it an offline read of (run, step) has a
            # fingerprint identical to the LIVE read of the same (run, step), and the second
            # invocation silently serves the first one's artifacts — a 400-game cycle reported
            # under a 100-game readout, with nothing in the output saying so.
            "read_root": str(cycle.get("read_root") or cycle["run_dir"]),
            "step": cycle["step"],
            "states": args.states, "anchors": args.anchors, "rollouts": args.rollouts,
            "impl": args.impl, "seed": args.seed,
            "anchor_tolerance": args.anchor_tolerance, "bins": args.bins,
            "saved_at": (cycle.get("manifest") or {}).get("saved_at"),
            "tool_version": READOUT_FINGERPRINT_VERSION}


def _cond_fingerprint(cycle: Dict[str, Any], args) -> Dict[str, Any]:
    """The CONDITIONING block's own cache key.

    Deliberately SEPARATE from :func:`_fingerprint`. The identity half costs a `cf_audit` run
    (~25 min); folding a new parameter into its key would invalidate every readout already on
    disk and re-pay that for a statistic that costs seconds. Two keys, two caches.
    """
    return {"run_dir": str(cycle["run_dir"]),
            "read_root": str(cycle.get("read_root") or cycle["run_dir"]),
            "step": cycle["step"],
            "boot": args.cond_boot, "seed": args.seed, "ladder": args.cond_ladder,
            "saved_at": (cycle.get("manifest") or {}).get("saved_at"),
            "meters": list(CM.METER_KEYS), "block_version": 1}


def identity_block(rows: List[dict], payload: dict, cap, *, boot: int,
                   seed: int, bins: int) -> Dict[str, Any]:
    """Every identity statistic, under all three weightings, with its raw bootstrap draws."""
    import numpy as np

    def weights(sub: Sequence[dict]) -> Dict[str, Any]:
        w_pop, cover = R.pop_weights(sub, payload["frame_cells"])
        w_ipw, ipw_cover = R.apply_capture(sub, w_pop, cap)
        return {"raw": (np.ones(len(sub)), 1.0), "pop": (w_pop, cover),
                "ipw": (w_ipw, ipw_cover)}

    out: Dict[str, Any] = {"strata": {}, "n_labels": len(rows),
                           "n_battles": len({r["battle_key"] for r in rows}),
                           "n_rollouts": int(sum(r["n"] for r in rows))}
    for name in IDENTITY_STRATA:
        if name == "ALL":
            sub = list(rows)
        elif name in R.TURN_BUCKETS:
            sub = [r for r in rows if R.turn_bucket(r["turn"]) == name]
        else:
            sub = [r for r in rows if (r["opp_class"] == "bot" if name == "bot"
                                       else r["opp_class"] != "bot")]
        if not sub:
            out["strata"][name] = {"n": 0}
            continue
        wt = weights(sub)
        entry: Dict[str, Any] = {"n": len(sub),
                                 "n_battles": len({r["battle_key"] for r in sub}),
                                 "mean_v": float(np.mean([r["v"] for r in sub])),
                                 "mean_mc": float(np.mean([r["mc"] for r in sub])),
                                 "coverage": {k: v[1] for k, v in wt.items()},
                                 "bias": {}}
        f = R.bias_stat(sub)
        for wname in WEIGHTINGS:
            w, _ = wt[wname]
            pt, draws = R.boot_draws(sub, f, w=w, draws=boot, seed=seed)
            entry["bias"][wname] = {"point": pt, "ci": R.ci_of(pt, draws),
                                    "n_draws": int(draws.size)}
            entry.setdefault("_draws", {})[f"bias.{wname}"] = draws
        out["strata"][name] = entry

    wt_all = weights(rows)
    out["murphy"] = {}
    for wname in WEIGHTINGS:
        w, _ = wt_all[wname]
        out["murphy"][wname] = R.murphy(rows, w, bins)
    for term in ("resolution", "skill_score", "resolution_cap_share", "reliability"):
        st = R.murphy_stat(rows, term, bins)
        for wname in WEIGHTINGS:
            w, _ = wt_all[wname]
            pt, draws = R.boot_draws(rows, st, w=w, draws=boot, seed=seed + 11)
            out["murphy"].setdefault("ci", {})[f"{term}.{wname}"] = {
                "point": pt, "ci": R.ci_of(pt, draws)}
            out.setdefault("_draws", {})[f"murphy.{term}.{wname}"] = draws

    tc = R.turn_contrast_stat(rows)
    out["turn"] = {}
    for wname in WEIGHTINGS:
        w, _ = wt_all[wname]
        pt, draws = R.boot_draws(rows, tc, w=w, draws=boot, seed=seed + 22)
        out["turn"][wname] = dict(R.turn_corr_parts(rows, w),
                                  contrast=pt, ci=R.ci_of(pt, draws))
        out.setdefault("_draws", {})[f"turn_contrast.{wname}"] = draws
    out["reliability_cells"] = R.reliability_cells(rows, wt_all["ipw"][0], bins)
    return out


def gate_block(run_dir: Path, step: int, *, boot: int, bins: int, seed: int,
               say) -> Dict[str, Any]:
    """The per-stratum calibration metrics of the CHOSEN cycle, with their bootstrap draws.

    Same estimator as ``main.critic_gate``'s table — ``agents.training.scaffolding``'s
    ``reliability_table`` on the scaffolding gauge's own capture-rate-weighted arrays — so the
    absolute numbers here and in ``gate/critic_gate.json`` agree row for row. What the gate does
    not publish, and this needs, is the raw DRAWS: a delta between two runs is the difference of
    their independent bootstraps and cannot be assembled from two published intervals.
    """
    arrays, cov = R.gauge_arrays(str(run_dir), step, seed=seed, say=say)
    out: Dict[str, Any] = {"coverage": cov, "strata": {}, "_draws": {}}
    for name in R.GATE_STRATA:
        arr = arrays.get(name)
        if arr is None:
            continue
        row: Dict[str, Any] = {"n_states": int(arr["p"].size),
                               "n_battles": int(len(set(arr["battles"].tolist())))}
        for metric in R.GATE_METRICS:
            pt, draws = R.gate_metric_draws(arr, metric, bins=bins, draws=boot,
                                            seed=seed + 300)
            row[metric] = {"point": pt, "ci": R.ci_of(pt, draws)}
            out["_draws"][f"{metric}.{name}"] = draws
        out["strata"][name] = row
    return out


def _serialisable(identity: Dict[str, Any]) -> Dict[str, Any]:
    """The identity block minus every raw bootstrap draw — what goes into the cached JSON.

    The draws are recomputed in seconds from the cached label join; carrying tens of thousands of
    floats through a cache file would make it unreadable and would fix a bootstrap SEED into an
    artifact that outlives the run that produced it.
    """
    out = {k: v for k, v in identity.items() if k != "_draws"}
    out["strata"] = {name: {k: v for k, v in entry.items() if k != "_draws"}
                     for name, entry in identity["strata"].items()}
    return out


def cache_hit(dirs: Sequence[Path], fp: Dict[str, Any]) -> Optional[Path]:
    """The ARTIFACT directory of a previous readout with an identical fingerprint, or ``None``.

    A readout stamp records where its artifacts actually live, so the pair directory a report is
    written into and the per-run cache the NEXT pair looks in can be different paths without
    either copying gigabytes or reusing the wrong cycle. The fingerprint covers the run, the
    cycle, the cycle's own ``saved_at``, and every parameter that changes a number — so a cache
    hit means the same computation, not merely the same run.
    """
    for d in dirs:
        stamp = Path(d) / "run_readout.json"
        if not stamp.exists():
            continue
        try:
            prior = json.loads(stamp.read_text())
        except (OSError, ValueError):
            continue
        if prior.get("fingerprint") == fp:
            art = Path(prior.get("artifact_dir") or d)
            if (art / "identity" / "bias_map.json").exists():
                return art
    return None


def read_run(run_dir: Path, cache_dir: Path, args, *, say, step: Optional[int] = None,
             alt_dirs: Sequence[Path] = (), read_root: Optional[Path] = None) -> Dict[str, Any]:
    """One run's whole readout — cached on the (run, read root, cycle, parameters) fingerprint.

    ``read_root`` overrides where the CYCLE is read from (see :func:`resolve_read_root`); it
    defaults to the run itself, which is every live read. ``run_dir`` still names the real run,
    and is what ``main.critic_gate`` is pointed at, because the registered G1-G4 rows are a
    function of the run's ladder and history rather than of any one cycle.
    """
    source = Path(read_root) if read_root is not None else run_dir
    cycle = pick_cycle(source, on_live=args.on_live, step=step)
    cycle["run_dir"] = str(run_dir)
    cycle["read_root"] = str(source)
    cycle["traces_overridden"] = source != run_dir
    cycle["pinned_step"] = step
    cycle["why_read"] = _cycle_why(cycle, step)
    fp = _fingerprint(cycle, args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    commands: List[str] = []

    hit = None if args.no_cache else cache_hit([cache_dir, *alt_dirs], fp)
    reused = hit is not None
    work = hit or cache_dir
    if reused:
        say(f"REUSING {run_dir.name}'s readout at step_{cycle['step']} from {work} "
            "(same run, same cycle, same parameters)")

    identity_dir = work / "identity"
    gate_dir = work / "gate"
    payload_path = work / "identity_payload.json"

    # ---- refusal gates that are cheap and must run on EVERY invocation, cache or not
    cov = winprob_coverage(cycle["trace_dir"])
    if cov["n_npz"] == 0:
        refuse(f"REFUSING: {run_dir.name} step_{cycle['step']} holds no *_states.npz — there is "
               "nothing to read.")
    if cov["n_missing"] > args.allow_missing_winprob * max(1, cov["n_npz"]):
        refuse(f"REFUSING: {cov['n_missing']} of {cov['n_npz']} state files at "
               f"{run_dir.name} step_{cycle['step']} carry NO `win_probs` column.",
               "  Both critic meters take V and P(win) from these npz. A head measured without "
               "its win-prob channel is not the head that trained — this is the check that the "
               "privileged arm's channel was present AT EVAL.",
               f"  Raise --allow-missing-winprob above "
               f"{cov['n_missing'] / cov['n_npz']:.3f} only with a reason.")
    ds = draw_share(cycle["manifest"] or {})
    if ds is not None and ds > args.max_draw_share:
        refuse(f"REFUSING: {run_dir.name} step_{cycle['step']} drew/timed out on "
               f"{ds * 100:.1f}% of played battles (cap {args.max_draw_share * 100:.0f}%).",
               "  A TIMEOUT IS NEVER A SEMANTIC OUTCOME. Above the cap this cycle is "
               "INCONCLUSIVE, not a measurement.")

    # ---- (1) identity: cf_audit + the readout
    if not reused:
        argv = [interpreter(), "-m", "agents.training.cf_audit", str(source),
                "--step", str(cycle["step"]), "--impl", args.impl,
                "--rollouts", str(args.rollouts), "--states", str(args.states),
                "--anchors", str(args.anchors), "--seed", str(args.seed),
                "--anchor-tolerance", str(args.anchor_tolerance),
                "--out", str(identity_dir)]
        if args.deadline_min:
            argv += ["--deadline-min", str(args.deadline_min)]
        commands.append(" ".join(argv))
        say(f"cf_audit on {source.name} step_{cycle['step']} "
            f"({args.states} states / {args.anchors} anchors) -> {identity_dir}")
        log = str(identity_dir / "cf_audit.log")
        rc = _run(argv, log, nice=args.nice)
        if rc != 0:
            refuse(f"REFUSING: cf_audit exited {rc} on {run_dir.name} step_{cycle['step']}.",
                   f"  log: {log}", *[f"  | {ln}" for ln in _tail(log).splitlines()[-12:]])

    bias_map_path = identity_dir / "bias_map.json"
    if not bias_map_path.exists():
        refuse(f"REFUSING: cf_audit wrote no bias_map.json under {identity_dir}.")
    bias_map = json.loads(bias_map_path.read_text())
    # cf_audit files the anchor arm under `accounting`; a flat file (an older producer) is read
    # too, because falling back to 0.0 would turn a schema change into a fabricated REFUSAL.
    acct = bias_map.get("accounting") if isinstance(bias_map.get("accounting"), dict) else bias_map
    if "anchor_rate" not in acct:
        refuse(f"REFUSING: {bias_map_path} records no `anchor_rate` — the label-trust arm cannot "
               "be read, so no bias may be reported from these labels.")
    anchor_rate = float(acct.get("anchor_rate", 0.0))
    if anchor_rate < args.anchor_tolerance:
        refuse(f"REFUSING: label trust — {acct.get('anchors_reproduced')}/"
               f"{acct.get('anchors_issued')} anchors reproduced the recorded outcome "
               f"({anchor_rate * 100:.1f}% < {args.anchor_tolerance * 100:.0f}%) on "
               f"{run_dir.name} step_{cycle['step']}.",
               "  Below the gate the MC labels do not describe the recorded battles, so no bias "
               "number may be reported from them.")

    if payload_path.exists() and reused:
        payload = json.loads(payload_path.read_text())
    else:
        labels = sorted(glob.glob(str(identity_dir / "cf_labels" / "labels_*.jsonl")))
        pinned = [p for p in labels if p.endswith(f"_{cycle['step']}.jsonl")]
        if not (pinned or labels):
            refuse(f"REFUSING: no cf_audit label file under {identity_dir / 'cf_labels'}.")
        payload = R.build_identity_payload((pinned or labels)[-1], cycle["trace_dir"])
        payload_path.write_text(json.dumps(payload, indent=1))
    rows = payload["rows"]
    if not rows:
        refuse(f"REFUSING: the cf_audit label join produced 0 rows for {run_dir.name}.")

    cap = R.capture_weights(str(source), cycle["step"])
    identity = identity_block(rows, payload, cap, boot=args.boot, seed=args.seed, bins=args.bins)
    identity["anchor"] = {"issued": acct.get("anchors_issued"),
                          "reproduced": acct.get("anchors_reproduced"),
                          "errors": acct.get("anchor_errors"),
                          "rate": anchor_rate, "tolerance": args.anchor_tolerance,
                          "note": R.anchor_note()}
    identity["capture_weights_available"] = cap is not None

    # ---- (2) main.critic_gate, for the registered G1-G4 rows
    gate_json = gate_dir / "critic_gate.json"
    if not reused or not gate_json.exists():
        gate_dir.mkdir(parents=True, exist_ok=True)
        argv = [interpreter(), "-m", "main.critic_gate", str(run_dir),
                "--parent", args.parent, "--famine-comparator", args.famine_comparator,
                "--skip-meter", "--boot", str(args.gate_boot), "--seed", str(args.seed),
                "--reliability-bins", str(args.bins),
                "--json", str(gate_json), "--md", str(gate_dir / "critic_gate.md")]
        commands.append(" ".join(argv))
        say(f"main.critic_gate on {run_dir.name} -> {gate_dir}"
            + ("  (the REAL run, not the generated cycle — the registered G1-G4 rows are a "
               "function of the run's ladder and history)" if source != run_dir else ""))
        log = str(gate_dir / "critic_gate.log")
        # exit 1 == a MIXED/failing VERDICT, which is a RESULT. Only 2 (GateRefusal) is a refusal.
        rc = _run(argv, log, nice=args.nice)
        if rc not in (0, 1) or not gate_json.exists():
            # exit 1 is a MIXED/failing VERDICT, which is a RESULT. Exit 2 is a GateRefusal —
            # most often a fresh arm whose ladder is not rated, which the ladder/G7 sections
            # need and the DELTA does not. Refuse by default anyway: the registered read names
            # the G1-G4 rows, and quietly dropping them would ship a table missing its primary.
            lines = [f"REFUSING: main.critic_gate exited {rc} on {run_dir.name} and wrote no "
                     "usable JSON.", f"  log: {log}",
                     *[f"  | {ln}" for ln in _tail(log).splitlines()[-12:]]]
            if not args.allow_gate_refusal:
                refuse(*lines, "  Pass --allow-gate-refusal to proceed WITHOUT the registered "
                               "G1-G4 rows; the deltas do not depend on them, and the report "
                               "will say in print that they are missing and why.")
            gate_refusal = "\n".join(lines)
        else:
            gate_refusal = None
    else:
        gate_refusal = None
    gate_doc = json.loads(gate_json.read_text()) if gate_json.exists() else {}
    gate_rows = next((c for c in gate_doc.get("calibration", {}).get("checkpoints", [])
                      if int(c["step"]) == cycle["step"]), None)

    gate = gate_block(source, cycle["step"], boot=args.gate_boot, bins=args.bins,
                      seed=args.seed, say=say)
    gate["critic_gate_verdict"] = gate_doc.get("calibration", {}).get("verdict")
    gate["critic_gate_rows_at_step"] = ({s["stratum"]: {
        k: s[k] for k in ("resolution", "resolution_ci", "baseline_resolution",
                          "delta_resolution", "reliability", "ece", "skill", "skill_ci",
                          "G1_resolution", "G2_reliability", "G3_ece", "G4_skill", "gated")
        if k in s} for s in gate_rows["strata"]} if gate_rows else None)
    gate["baseline_artifact"] = gate_doc.get("calibration", {}).get("artifact")
    gate["critic_gate_refusal"] = gate_refusal

    # ---- (3) CONDITIONING, on the RECORDED V of this cycle. Its own cache file, so a new meter
    # never invalidates the cf_audit half.
    cond: Optional[Dict[str, Any]] = None
    cond_refusal: Optional[str] = None
    if not args.no_conditioning:
        cfp = _cond_fingerprint(cycle, args)
        cond_path = work / "cond_readout.json"
        cached = None
        if not args.no_cache and cond_path.exists():
            try:
                prior = json.loads(cond_path.read_text())
                if prior.get("fingerprint") == cfp and prior.get("draws"):
                    cached = prior
            except (OSError, ValueError):
                cached = None
        if cached is not None:
            say(f"REUSING {run_dir.name}'s conditioning block at step_{cycle['step']} "
                f"from {cond_path}")
            cond = cached["block"]
            cond["_draws"] = {k: __import__("numpy").asarray(v, dtype=float)
                              for k, v in cached["draws"].items()}
        else:
            say(f"conditioning meters on {run_dir.name} step_{cycle['step']} "
                f"({args.cond_boot} battle-clustered draws, ladder={args.cond_ladder})")
            try:
                cond = CM.conditioning_block(str(source), cycle["step"], boot=args.cond_boot,
                                             seed=args.seed, ladder=args.cond_ladder, say=say)
            except CM.ConditioningRefusal as exc:
                cond, cond_refusal = None, str(exc)
                if not args.allow_conditioning_refusal:
                    refuse(f"REFUSING: the CONDITIONING block cannot be computed on "
                           f"{run_dir.name} step_{cycle['step']}.", f"  {exc}",
                           "  Pass --allow-conditioning-refusal to emit the report WITHOUT the "
                           "conditioning rows; it will say in print that they are missing and "
                           "why. Pass --no-conditioning to skip them deliberately.")
            if cond is not None:
                cond_path.parent.mkdir(parents=True, exist_ok=True)
                cond_path.write_text(json.dumps(
                    {"fingerprint": cfp,
                     "block": {k: v for k, v in cond.items() if k != "_draws"},
                     "draws": {k: [float(x) for x in v] for k, v in cond["_draws"].items()}},
                    indent=1, default=float))
        if cond is not None:
            cond["ci"] = {k: R.ci_of(cond["points"][k], cond["_draws"].get(k, __import__(
                "numpy").empty(0))) for k in cond["points"]}
            cond["refusal"] = None
        else:
            cond = {"points": {}, "ci": {}, "_draws": {}, "omitted": {}, "frame": {},
                    "refusal": cond_refusal}

    manifest = cycle.get("manifest") or {}
    doc = {"fingerprint": fp, "artifact_dir": str(work),
           "run": run_dir.name, "run_dir": str(run_dir),
           "read_root": str(source), "traces_overridden": source != run_dir,
           # The population IN WORDS, for both kinds of cycle. A header that describes only the
           # unusual side invites the reader to treat the other as the neutral default.
           "population": ETG.population_of(manifest),
           "generated": ETG.is_generated(manifest),
           "generated_by": manifest.get(ETG.GENERATED_KEY),
           "spec": ETG.spec_of(manifest),
           "manifest": manifest,
           "step": cycle["step"], "trace_dir": cycle["trace_dir"],
           "cycle": {k: v for k, v in cycle.items() if k != "manifest"},
           "npz_coverage": cov, "draw_share": ds,
           "identity": _serialisable(identity),
           "gate": {k: v for k, v in gate.items() if k != "_draws"},
           "conditioning": (None if cond is None else
                            {k: v for k, v in cond.items() if k != "_draws"}),
           "commands": commands, "reused": reused,
           "paths": {"identity": str(identity_dir), "gate": str(gate_dir),
                     "identity_payload": str(payload_path)}}
    (work / "run_readout.json").write_text(json.dumps(doc, indent=1, default=float))
    if work != cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "run_readout.json").write_text(json.dumps(doc, indent=1, default=float))
    doc["_identity_draws"] = identity.get("_draws", {})
    for name, entry in identity["strata"].items():
        for k, v in (entry.get("_draws") or {}).items():
            doc["_identity_draws"][f"{name}|{k}"] = v
    doc["_gate_draws"] = gate["_draws"]
    doc["_cond_draws"] = (cond or {}).get("_draws", {})
    doc["_identity_points"] = identity
    doc["_gate_points"] = gate
    return doc


# --------------------------------------------------------------------------- the delta

def _cycle_why(cycle: Dict[str, Any], pinned: Optional[int]) -> str:
    """One sentence naming WHY this cycle was read — printed in the log and in the report header.

    The tdaux read (backlog 2026-09-09) took the PREVIOUS cycle because the arm's launcher
    process was still alive and `--on-live skip-newest` dropped the newest one; the report named
    the step it actually read, but nothing said the newest had been dropped or why. It does now.
    """
    if pinned is not None:
        return f"PINNED by --step {pinned}"
    live = cycle.get("live_pids") or []
    if cycle.get("dropped_newest_because_live"):
        return (f"last complete cycle AFTER DROPPING THE NEWEST — the run is LIVE "
                f"(pid {', '.join(map(str, live))}) and --on-live is `{cycle['on_live']}`; "
                f"steps on disk: {', '.join(str(x) for x in cycle.get('steps_on_disk') or [])}. "
                f"🚨 Pin --step to read the newest anyway.")
    if live:
        return (f"last complete cycle; the run is LIVE (pid {', '.join(map(str, live))}) and "
                f"--on-live is `{cycle['on_live']}` — {cycle['why']}")
    return f"last complete cycle, run not live — {cycle['why']}"


def _floor_for(key: str, floors: Optional[Dict[str, float]]) -> Optional[float]:
    return None if not floors else floors.get(key)


def _stamp(cache_dir: Path, doc: Dict[str, Any]) -> None:
    """Drop this run's readout STAMP in its per-run cache directory.

    The arm's artifacts live under ``--out``, which is the PAIR's directory and is different for
    every pair. The stamp is a pointer at those artifacts filed under the RUN's name, which is
    where the next pair looks — so today's arm is tomorrow's free control and the expensive half
    of a ladder read is paid once per run, not once per pair.
    """
    art = Path(doc.get("artifact_dir") or cache_dir)
    if art.resolve() == cache_dir.resolve():
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "run_readout.json").write_text(
        json.dumps({k: v for k, v in doc.items() if not k.startswith("_")},
                   indent=1, default=float))


def compute_deltas(arm: Dict[str, Any], ctl: Dict[str, Any],
                   floors: Optional[Dict[str, float]], *, seed: int,
                   qm: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every registered quantity as ARM - CONTROL, each with the difference of the two runs'
    INDEPENDENT battle-clustered bootstraps and the registration's three-way label.

    ``qm`` is :func:`main.ops.quota_match.match`'s document. When it reports the two frames
    UNEQUAL, every FRAME-SENSITIVE conditioning row (the flag is declared on the meter, not
    matched on its name) is re-read on the equalised frames and the label is decided there; the
    as-traced value rides along under ``quota_match.unmatched`` and is never labelled. With
    ``qm`` absent or its plan not needed, every row is exactly what the tool printed before.
    """
    import numpy as np

    rows: List[Dict[str, Any]] = []

    def add(key: str, family: str, quantity: str, stratum: str, weighting: Optional[str],
            a_pt: float, a_draws, c_pt: float, c_draws, registered: bool) -> None:
        d = R.independent_delta(a_pt, a_draws, c_pt, c_draws, seed=seed)
        floor = _floor_for(key, floors)
        rows.append({"key": key, "family": family, "quantity": quantity, "stratum": stratum,
                     "weighting": weighting, "registered": registered,
                     "arm": a_pt, "control": c_pt, **d,
                     **R.label_delta(d["delta"], d["ci"], floor),
                     "headline": key in HEADLINES})

    for metric in R.GATE_METRICS:
        for stratum in R.GATE_STRATA:
            k = f"{metric}.{stratum}"
            if k not in arm["_gate_draws"] or k not in ctl["_gate_draws"]:
                continue
            add(f"gate.{k}", "gate", metric, stratum, None,
                arm["_gate_points"]["strata"][stratum][metric]["point"],
                arm["_gate_draws"][k],
                ctl["_gate_points"]["strata"][stratum][metric]["point"],
                ctl["_gate_draws"][k], registered=(metric in ("resolution", "skill")))

    for stratum in IDENTITY_STRATA:
        a_s = arm["_identity_points"]["strata"].get(stratum, {})
        c_s = ctl["_identity_points"]["strata"].get(stratum, {})
        if not a_s.get("n") or not c_s.get("n"):
            continue
        for wname in WEIGHTINGS:
            ak, ck = f"{stratum}|bias.{wname}", f"{stratum}|bias.{wname}"
            if ak not in arm["_identity_draws"] or ck not in ctl["_identity_draws"]:
                continue
            add(f"identity.bias.{stratum}" + ("" if wname == "ipw" else f".{wname}"),
                "identity", "bias V - p_hat", stratum, wname,
                a_s["bias"][wname]["point"], arm["_identity_draws"][ak],
                c_s["bias"][wname]["point"], ctl["_identity_draws"][ck],
                registered=(wname == "ipw"))

    for term in ("resolution", "skill_score", "resolution_cap_share", "reliability"):
        for wname in WEIGHTINGS:
            k = f"murphy.{term}.{wname}"
            if k not in arm["_identity_draws"] or k not in ctl["_identity_draws"]:
                continue
            add(f"identity.{term}" + ("" if wname == "ipw" else f".{wname}"),
                "identity", f"Murphy {term}", "ALL", wname,
                arm["_identity_points"]["murphy"]["ci"][f"{term}.{wname}"]["point"],
                arm["_identity_draws"][k],
                ctl["_identity_points"]["murphy"]["ci"][f"{term}.{wname}"]["point"],
                ctl["_identity_draws"][k], registered=(wname == "ipw"))

    a_cond = (arm.get("conditioning") or {}).get("points") or {}
    c_cond = (ctl.get("conditioning") or {}).get("points") or {}
    qm_plan = (qm or {}).get("plan") or {}
    qm_rungs = (qm or {}).get("rungs") or {}
    for m in CM.METER_SPECS:
        key = m.key
        if key not in a_cond or key not in c_cond:
            continue
        registered = key in ("cond.spread_ratio.t1_3", "cond.own_team_r2.t1")
        a_pt, a_dr = a_cond[key], arm["_cond_draws"].get(key, np.empty(0))
        c_pt, c_dr = c_cond[key], ctl["_cond_draws"].get(key, np.empty(0))
        if not (m.frame_sensitive and qm_plan.get("needed")):
            add(key, "conditioning", m.quantity, m.stratum, None,
                a_pt, a_dr, c_pt, c_dr, registered=registered)
            if m.frame_sensitive and qm_plan:
                # SYMMETRIC: the frames are already equal, so the row is the as-traced one and
                # is bit-for-bit what the pre-matching tool printed.
                rows[-1]["quota_match"] = {"status": "SYMMETRIC", "why": qm_plan.get("why")}
            continue

        # ---- frame-sensitive AND the two frames are unequal.
        base = R.independent_delta(a_pt, a_dr, c_pt, c_dr, seed=seed)
        unmatched = {"arm": a_pt, "control": c_pt, **base}
        if not qm_rungs:
            # --no-quota-match. 🚨 NO LABEL. The 2026-09-09 RETRACTION withdrew a DETECTED that
            # was entirely this artefact; a row read on an unequal frame is not a reading, and
            # printing NOT DETECTED here would be just as much of a claim as printing DETECTED.
            rows.append({"key": key, "family": "conditioning", "quantity": m.quantity,
                         "stratum": m.stratum, "weighting": None, "registered": registered,
                         "arm": a_pt, "control": c_pt, **base,
                         "label": QM.UNMATCHED_LABEL,
                         "qualifier": (f"{qm_plan.get('why')}; this row's estimator is "
                                       f"frame-size dependent ({m.why}). Re-run WITHOUT "
                                       "--no-quota-match."),
                         "floor": _floor_for(key, floors), "clears_zero": False,
                         "clears_floor": False,
                         "quota_match": {"status": "UNMATCHED", "unmatched": unmatched,
                                         "why": qm_plan.get("why"), "reason": m.why},
                         "headline": key in HEADLINES})
            continue

        variants: Dict[str, Any] = {}
        for rung_name in ("battle", "decoder"):
            aa_pt, aa_dr, cc_pt, cc_dr = a_pt, a_dr, c_pt, c_dr
            sides: Dict[str, Any] = {}
            ok = True
            for side, rr in qm_rungs.items():
                r = (rr.get(rung_name) or {}).get("rows", {}).get(key)
                if r is None:
                    ok = False
                    break
                if side == "arm":
                    aa_pt, aa_dr = r["point"], r["_draws"]
                else:
                    cc_pt, cc_dr = r["point"], r["_draws"]
                sides[side] = {"spread": r["spread"], "median_seed": r["median_seed"],
                               "caps": rr[rung_name]["caps"],
                               "n_seeds": rr[rung_name]["n_seeds"],
                               "median_battles": rr[rung_name]["median_battles"],
                               "median_decoder_battles":
                                   rr[rung_name]["median_decoder_battles"]}
            if not ok:
                continue
            d = R.independent_delta(aa_pt, aa_dr, cc_pt, cc_dr, seed=seed)
            variants[rung_name] = {"arm": aa_pt, "control": cc_pt, **d,
                                   **R.label_delta(d["delta"], d["ci"],
                                                   _floor_for(key, floors)),
                                   "sides": sides}
        primary = variants.get("battle") or variants.get("decoder")
        if primary is None:
            add(key, "conditioning", m.quantity, m.stratum, None,
                a_pt, a_dr, c_pt, c_dr, registered=registered)
            rows[-1]["label"] = QM.UNMATCHED_LABEL
            rows[-1]["qualifier"] = ("the matched rungs produced no usable value for this row "
                                     "(too few battles after the cut) — not a reading")
            rows[-1]["quota_match"] = {"status": "UNMATCHED", "unmatched": unmatched,
                                       "why": qm_plan.get("why"), "reason": m.why}
            continue
        rows.append({"key": key, "family": "conditioning", "quantity": m.quantity,
                     "stratum": m.stratum, "weighting": None, "registered": registered,
                     **{k: v for k, v in primary.items() if k != "sides"},
                     "quota_match": {"status": "MATCHED", "rung": "battle",
                                     "variants": variants, "unmatched": unmatched,
                                     "why": qm_plan.get("why"), "reason": m.why,
                                     "caps": qm_plan.get("caps"),
                                     "sides": qm_plan.get("sides")},
                     "headline": key in HEADLINES})

    for wname in WEIGHTINGS:
        k = f"turn_contrast.{wname}"
        if k not in arm["_identity_draws"] or k not in ctl["_identity_draws"]:
            continue
        add("identity.turn_contrast" + ("" if wname == "raw" else f".{wname}"),
            "identity", "corr(turn,V) - corr(turn,MC)", "ALL", wname,
            arm["_identity_points"]["turn"][wname]["contrast"], arm["_identity_draws"][k],
            ctl["_identity_points"]["turn"][wname]["contrast"], ctl["_identity_draws"][k],
            registered=(wname == "raw"))
    return rows


# --------------------------------------------------------------------------- CLI

def load_floors(path: Optional[str]) -> Dict[str, Any]:
    """``{"path": …, "floors": {key: magnitude} | None, "provenance": …}``.

    The floor file is ``{"floors": {"gate.resolution.bot": 0.012, …}, "provenance": "…"}`` (a
    bare ``{key: magnitude}`` mapping is also accepted). A key absent from the file has NO floor
    and its delta is labelled against zero with the explicit qualifier.
    """
    if not path:
        return {"path": None, "floors": None, "provenance": R.NO_FLOOR_NOTE}
    with open(path) as fh:
        doc = json.load(fh)
    floors = doc.get("floors", doc) if isinstance(doc, dict) else None
    if not isinstance(floors, dict):
        refuse(f"REFUSING: {path!r} is not a floor file — expected "
               '{"floors": {"<key>": <magnitude>}} or a bare {key: magnitude} mapping.')
    return {"path": os.path.abspath(path),
            "floors": {str(k): abs(float(v)) for k, v in floors.items()},
            "provenance": (doc.get("provenance") if isinstance(doc, dict) else None)
            or "(the floor file records no provenance)"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"python -m main.ops.{TOOL}",
        description="The critic ladder's registered read as ONE command: identity (cf_audit) + "
                    "G1-G4 (main.critic_gate) + the clock-tracking contrast, every quantity as "
                    "ARM - CONTROL with the delta's CI and the registered label.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Refusals over silence: a missing manifest, a cycle that has not collected, npz "
               "without `win_probs`, an anchor rate under the label-trust gate, or a "
               "draw/timeout share over the cap exits 2 with the cause named. Nothing is ever "
               "written under models/.")
    ap.add_argument("arm", help="the ladder arm — a run NAME under models/ or a run DIRECTORY")
    ap.add_argument("--control", required=True,
                    help="the control arm every delta is taken against "
                         "(the ladder's is `ai_v12_11_ladder_ctrl10M`)")
    ap.add_argument("--out", required=True, help="output directory for THIS pair's report")
    ap.add_argument("--step", type=int, default=None, metavar="N",
                    help="PIN the eval cycle read for BOTH runs to eval_traces/step_N, instead of "
                         "resolving each run's last COMPLETE cycle. 🚨 Pin it whenever the arm's "
                         "launcher process is still alive: `--on-live skip-newest` (the default) "
                         "drops the newest cycle, so a finished 10M arm whose launcher had not "
                         "yet exited is read at 8M and the report names 8M while the caller "
                         "believes it read 10M. That happened to the tdaux read (backlog "
                         "2026-09-09). A run with no step_N REFUSES, naming the steps it has.")
    ap.add_argument("--arm-traces", default=None, metavar="DIR",
                    help="read the ARM's cycle from this OFFLINE-GENERATED directory (what "
                         "`main.ops.eval_trace_gen --out` wrote, or the eval_traces/step_<N> "
                         "inside it) instead of from the run. The registered G1-G4 rows still "
                         "come from the REAL run's ladder. Both sides must be the same KIND of "
                         "cycle — an offline-vs-live delta is REFUSED, and two offline frames "
                         "must share games / opponent set / capture rule / sentinel regime.")
    ap.add_argument("--control-traces", default=None, metavar="DIR",
                    help="the same, for the CONTROL side")
    ap.add_argument("--control-step", type=int, default=None, metavar="N",
                    help="pin ONLY the control's cycle (default: --step if given, else the "
                         "control's own last complete cycle). The registered read is "
                         "arm@10M vs control@10M; a cross-step pair is labelled CROSS-STEP in "
                         "the ledger quote.")
    ap.add_argument("--baseline-arm", default=None,
                    help="the historical arm whose committed measurement supplies the "
                         "matched-stratum comparator when the fresh arm has no parent "
                         "(default: the `--parent` baseline name)")
    ap.add_argument("--parent", default="v9_fold_parent", metavar="REF|BASELINE",
                    help="passed to main.critic_gate, which requires one. A FRESH arm has no "
                         "parent, and the ladder/G7 sections that use it are NOT reported here "
                         "— G1-G4 read their bars from the committed calibration baseline, not "
                         "from this. Default: the `v9_fold_parent` registry name.")
    ap.add_argument("--famine-comparator", default="off",
                    help="passed through to main.critic_gate (default `off`: the famine "
                         "pre-test is a STRENGTH read and the ladder does not read strength)")
    ap.add_argument("--states", type=int, default=800, help="cf_audit --states (default 800, "
                                                            "the 75M read's)")
    ap.add_argument("--anchors", type=int, default=150, help="cf_audit --anchors (default 150)")
    ap.add_argument("--rollouts", type=int, default=8, help="cf_audit --rollouts (default 8)")
    ap.add_argument("--impl", choices=("rust", "node"), default="rust",
                    help="cf_audit offline driver (default rust)")
    ap.add_argument("--anchor-tolerance", type=float, default=0.90,
                    help="label-trust gate: below this anchor reproduction rate the read "
                         "REFUSES rather than report a bias (default 0.90)")
    ap.add_argument("--deadline-min", type=float, default=0,
                    help="cf_audit --deadline-min (0 = no bound)")
    ap.add_argument("--boot", type=int, default=R.N_BOOT,
                    help=f"identity cluster-bootstrap draws over BATTLES (default {R.N_BOOT})")
    ap.add_argument("--gate-boot", type=int, default=400,
                    help="gate cluster-bootstrap draws (default 400, main.critic_gate's own)")
    ap.add_argument("--bins", type=int, default=10, help="reliability bins (default 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor-json", default=None, metavar="PATH",
                    help="the REPLICATE FLOOR, {\"floors\": {key: magnitude}}. Absent, no delta "
                         "may be DETECTED against a floor and the report says so.")
    ap.add_argument("--on-live", choices=("skip-newest", "refuse", "use"), default="skip-newest",
                    help="what to do when a run still has a live launcher: drop its newest cycle "
                         "(default), refuse outright, or read the newest complete cycle anyway")
    ap.add_argument("--max-draw-share", type=float, default=0.25,
                    help="refuse a cycle whose played battles drew/timed out above this share "
                         "(default 0.25 — a timeout is never a semantic outcome)")
    ap.add_argument("--allow-missing-winprob", type=float, default=0.0,
                    help="tolerated share of state npz with no `win_probs` column (default 0.0)")
    ap.add_argument("--cond-boot", type=int, default=CM.N_BOOT,
                    help=f"CONDITIONING battle-clustered bootstrap draws (default {CM.N_BOOT}). "
                         "Battles are resampled WITHIN their opponent cell — the roster is a "
                         "fixed pinned set, not a sample — and the outcome side is redrawn from "
                         "its own Binomial(battles_played, true win rate).")
    ap.add_argument("--cond-ladder", choices=("refit", "off"), default="refit",
                    help="the CONDITIONING strength axis: `refit` builds it from the bot anchors "
                         "plus an all-steps bot-anchored refit of the run's snapshot ladder and "
                         "REFUSES an unanchored fit; `off` skips it and the Elo-slope row is "
                         "omitted with that reason. Every other conditioning row is unaffected — "
                         "the spread identity needs no strength axis.")
    ap.add_argument("--no-conditioning", action="store_true",
                    help="skip the CONDITIONING section entirely")
    ap.add_argument("--no-quota-match", action="store_true",
                    help="do NOT equalise the two trace frames before reading the FRAME-SENSITIVE "
                         "conditioning rows (the fitted decoders and the uncorrected spread "
                         "ratios). Matching is ON by default because a decoder's out-of-fold "
                         "score rises with the frame it was fit on and the ladder's arms are "
                         "traced at different quotas — that is what produced, and then withdrew, "
                         "the 2026-09-09 own-team detection. With this flag those rows are "
                         "printed with a hard `UNMATCHED — not a reading` marker INSTEAD of a "
                         "label; no DETECTED is ever printed on an unequal frame.")
    ap.add_argument("--quota-match-seeds", type=int, default=QM.DEFAULT_SEEDS, metavar="N",
                    help=f"subsample seeds per matched rung (default {QM.DEFAULT_SEEDS}; the "
                         "standing consequence asks for >= 20). ODD by default so the reported "
                         "median is an exact order statistic and the interval printed beside it "
                         "is that same draw's own battle-clustered bootstrap.")
    ap.add_argument("--quota-match-boot", type=int, default=None, metavar="N",
                    help="bootstrap draws for the matched rows (default: --cond-boot)")
    ap.add_argument("--allow-conditioning-refusal", action="store_true",
                    help="proceed when the conditioning block REFUSES (a cycle with no manifest "
                         "selection block, or no usable states). The report says in print that "
                         "the rows are missing, and why.")
    ap.add_argument("--nice", type=int, default=10, help="niceness for the subprocesses")
    ap.add_argument("--allow-gate-refusal", action="store_true",
                    help="proceed when main.critic_gate REFUSES (typically a fresh arm whose "
                         "ladder is not rated). The deltas are unaffected; the report says in "
                         "print that the registered G1-G4 rows are missing, and why.")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute both runs even when a matching cached readout exists")
    ap.add_argument("--ledger-line", action="store_true",
                    help="also print the one-line ledger QUOTE in the registered form")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve both runs and their cycles, print the plan, compute nothing")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    say = (lambda _m: None) if args.quiet else (
        lambda m: print(f"[{TOOL}] {m}", flush=True))
    started = time.time()

    arm_dir = resolve_run_dir(args.arm)
    ctl_dir = resolve_run_dir(args.control)
    arm_root = resolve_read_root(arm_dir, args.arm_traces, role="arm")
    ctl_root = resolve_read_root(ctl_dir, args.control_traces, role="control")
    out = Path(args.out).resolve()
    cache_root = out.parent
    floor = load_floors(args.floor_json)

    ctl_step = args.control_step if args.control_step is not None else args.step

    def stamp_dir(run_d: Path, root: Path) -> Path:
        """This side's per-run cache directory.

        An OFFLINE read gets its own, because the stamp is what the NEXT pair picks up as a free
        control — and a 400-game readout filed under the plain run name would be handed to a pair
        that asked for the live one, with nothing in its output saying so."""
        return cache_root / (run_d.name if root == run_d else f"{run_d.name}__offline")

    arm_cache, ctl_cache = stamp_dir(arm_dir, arm_root), stamp_dir(ctl_dir, ctl_root)
    same_side = arm_dir == ctl_dir and arm_root == ctl_root

    if args.dry_run:
        for role, d, root, st in (("arm", arm_dir, arm_root, args.step),
                                  ("control", ctl_dir, ctl_root, ctl_step)):
            c = pick_cycle(root, on_live=args.on_live, step=st)
            print(f"{role:8s} {d.name}  ->  step_{c['step']}  ({_cycle_why(c, st)}; "
                  f"live pids {c['live_pids'] or 'none'}; "
                  f"policy {c['on_live']}{'; newest dropped' if c['dropped_newest_because_live'] else ''})")
            print(f"{'':8s} population: {ETG.population_of(c.get('manifest'))}")
            if root != d:
                print(f"{'':8s} traces READ FROM {root}")
                print(f"{'':8s} (the REAL run still supplies the registered G1-G4 rows — an "
                      f"offline cycle has no ladder)")
        print(f"out      {out}")
        print(f"cache    {arm_cache}  |  {ctl_cache}")
        print(f"floor    {floor['path'] or 'NONE — ' + R.NO_FLOOR_NOTE}")
        return 0

    if same_side:
        say("arm and control are the SAME run AND the same cycle — this is the zero-delta "
            "self-consistency plant; every delta must be exactly 0.0 and every label NOT "
            "DETECTED.")

    out.mkdir(parents=True, exist_ok=True)
    # The ARM's artifacts go under --out, as the read registers them; a STAMP is also dropped in
    # the per-run sibling `<out>/../<run>/`, which is where the next pair's cache looks. So the
    # control of one invocation is the free arm of the next without anything being copied.
    # 🚨 The chosen cycle and WHY are printed BEFORE anything expensive runs, and land in the
    # report header — the tdaux read silently took the previous cycle because the launcher
    # process was still alive, and nothing in the output said so until the artifacts were read.
    # 🚨 THE POPULATION CHECK RUNS BEFORE ANYTHING EXPENSIVE. A cross-population pair is refused
    # here, not after a ~25-minute cf_audit has already been paid for on each side.
    picked = {}
    for role, d, root, st in (("arm", arm_dir, arm_root, args.step),
                              ("control", ctl_dir, ctl_root, ctl_step)):
        c = pick_cycle(root, on_live=args.on_live, step=st)
        picked[role] = c
        say(f"CYCLE  {role:8s} {d.name} -> step_{c['step']}  ({_cycle_why(c, st)})")
        say(f"POP    {role:8s} {ETG.population_of(c.get('manifest'))}")
    check_comparable(picked["arm"], picked["control"])

    arm = read_run(arm_dir, out, args, say=say, step=args.step,
                   alt_dirs=[arm_cache], read_root=arm_root)
    _stamp(arm_cache, arm)
    ctl = (arm if same_side and ctl_step in (None, arm["step"])
           else read_run(ctl_dir, ctl_cache, args, say=say, step=ctl_step,
                         alt_dirs=[out], read_root=ctl_root))

    qm = QM.build_quota_match(arm, ctl, args, say=say)
    deltas = compute_deltas(arm, ctl, floor["floors"], seed=args.seed, qm=qm)
    doc = {
        "tool": TOOL, "tool_version": TOOL_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 1),
        "invocation": " ".join([f"python -m main.ops.{TOOL}"] + list(argv or sys.argv[1:])),
        "out": str(out), "floor": floor,
        "params": {k: getattr(args, k) for k in
                   ("states", "anchors", "rollouts", "impl", "anchor_tolerance", "boot",
                    "gate_boot", "bins", "seed", "on_live", "max_draw_share",
                    "allow_missing_winprob", "parent", "famine_comparator", "baseline_arm",
                    "step", "control_step", "cond_boot", "cond_ladder", "no_conditioning",
                    "no_quota_match", "quota_match_seeds", "quota_match_boot",
                    "arm_traces", "control_traces")},
        "registration": ("ledger 2026-09-08 · REGISTRATION · THE CRITIC LADDER; design note "
                         "designs/research_state/winprob_critic_ladder_2026-09-08.md"),
        "quota_match": QM.serialisable(qm),
        "arm": {k: v for k, v in arm.items() if not k.startswith("_")},
        "control": {k: v for k, v in ctl.items() if not k.startswith("_")},
        "deltas": deltas,
        "headlines": list(HEADLINES),
        "notes": {"weightings": WEIGHTING_NOTE, "anchor": R.anchor_note(),
                  "no_floor": R.NO_FLOOR_NOTE},
    }
    doc["ledger_line"] = ledger_line(doc)
    (out / "critic_read.json").write_text(json.dumps(doc, indent=1, default=float))
    (out / "critic_read.md").write_text(render_md(doc))
    say(f"wrote {out / 'critic_read.md'} and {out / 'critic_read.json'} "
        f"({doc['elapsed_sec']} s)")

    print()
    by = {r["key"]: r for r in deltas}
    for key in HEADLINES:
        r = by.get(key)
        print(f"  {key:38s} " + ("NOT COMPUTED" if r is None else
                                 f"Δ {r['delta']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  "
                                 f"{r['label']}"
                                 f"{' (' + r['qualifier'] + ')' if r.get('qualifier') else ''}"))
    if floor["path"] is None:
        print(f"\n  {R.NO_FLOOR_NOTE}")
    if args.ledger_line:
        print()
        print(doc["ledger_line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
