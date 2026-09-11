"""Generate an eval-traces cycle for a SAVED checkpoint, offline, on CPU.

    python -m main.ops.eval_trace_gen <run>[@<step>] --games N --sentinels K --out DIR

WHY THIS EXISTS. A live eval cycle is sized for a training run, not for a measurement: 100 games
against 12 opponents, of which a per-opponent outcome QUOTA persists ~200 traced battles a side.
The critic ladder then reads a 10M arm against its control from that one cycle — and five arms in,
with zero detected registered rows, the binding constraint was shown to be POWER, not effect size:
the turn-1-3 between-opponent spread ratio carries a battle-clustered CI of ~+/-0.4 around a
control value of ~0.12, and the run-to-run floor on the low-variance rows sits BELOW the
battle-level CI width. More eval per read buys real power on exactly the rows the ladder headlines.

Every finished arm still has its 10M checkpoint, so that read is possible without retraining
anything. This tool plays the cycle again, offline, as large as you are willing to pay for.

WHAT IT IS NOT. It is not a re-implementation of eval. It drives the SAME entry point a live
cycle drives — ``python -m main.eval_worker`` over a ``ShardedEvalPool`` plan, hence the same
``LocalBattleRunner`` / ``EvalRLPlayer`` / ``BattleRecorder`` path — and writes the same npz keys,
the same ``*_summary.json``, and an ``eval_manifest.json`` at ``selection_schema`` 2 with the
per-opponent capture rates rule 17 asks for. The regime is READ from the run's recorded config
(``eval_sentinel_greedy``, the reward, the trainee's team pin), never assumed; a run that recorded
no regime is REFUSED rather than measured under a guess.

🚨 THE OUTPUT IS A DIFFERENT POPULATION FROM A LIVE CYCLE, and that is the point of generating it.
More games, possibly more sentinels, and — by default — NO capture quota at all, so every battle
played is a battle traced. A delta between an offline frame and a live frame is therefore a delta
between two different samples of two different populations, and ``main.ops.critic_read`` REFUSES
to form one. The manifest carries a ``generated_by`` block naming the tool, the games, the
sentinel spec, the seed, the worker count and the checkpoint sha so no reader can mistake this
cycle for a live one, and so two offline frames can be checked for the SAME spec before they are
differenced.

NOTHING IS EVER WRITTEN UNDER ``models/``. The generated cycle is a self-contained SHADOW RUN
DIR under ``--out``: the run's ``model_config.json`` and ``metadata.json`` are copied (with the
sentinel block rewritten to the sentinels this cycle actually used), ``snapshots/`` is symlinked
back to the real run read-only, and the checkpoint is copied to
``<out>/eval_traces/step_<N>/snapshot.zip`` — which is exactly where ``cf_audit`` looks for the
network that played the traces, so the identity half resolves against the generated cycle with no
special-casing.

REPRODUCIBILITY. ``--seed S`` pins every stream a shard unit draws from, keyed on
``(seed, opponent, shard index)`` — never on the worker id, which work-stealing decides in a race.
The worker count is therefore FREE: ``--workers 1`` and ``--workers 8`` generate the identical
cycle. ``--concurrency`` is NOT free: above 1, several battles of one unit share the process-global
`random` stream that the scripted bots draw from, and the order they draw in is a timing race. The
tool prints the caveat and records both numbers in the manifest, because this project would rather
refuse an unreproducible configuration than emit a number that wanders without saying so.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from main.ops.run_ref import refuse, resolve_run_dir

#: `gen3_offline_eval_cycle_v1` — the provenance contract. A cycle carrying this block was
#: GENERATED; one without it was played by a live training run. Bumped when the block's own
#: shape changes, never when a number in it changes.
GENERATED_SCHEMA = 1

#: The manifest key. One name, imported by every reader (``critic_read``), never re-typed.
GENERATED_KEY = "generated_by"

#: The tool tag inside that block.
GENERATOR_TAG = "eval_trace_gen"


def is_generated(manifest: Optional[dict]) -> bool:
    """True when this manifest describes an OFFLINE-GENERATED cycle.

    The one place the question is answered, so a consumer never open-codes a key lookup that a
    schema change could silently turn into a permanent False — which would read as "live" and let
    exactly the cross-population delta this block exists to forbid through.
    """
    return isinstance((manifest or {}).get(GENERATED_KEY), dict)


def population_of(manifest: Optional[dict]) -> str:
    """The cycle's population, in WORDS, for a report header.

    Stated for a live cycle too, because "12 opponents x 100 games, outcome-quota traced" is just
    as much a population as the generated one — and a header that describes only the unusual side
    invites the reader to treat the other as the neutral default.
    """
    man = manifest or {}
    gen = man.get(GENERATED_KEY)
    if isinstance(gen, dict) and gen.get("population"):
        text = str(gen["population"])
        # 🚨 INCOMPLETE LEADS. A cycle whose workers died mid-plan is a SMALLER frame than the one
        # its `n_games` advertises, and frame size moves every fitted conditioning row on its own.
        # Saying so at the front of the population string is what stops it being read as the
        # nominal cycle. (2026-09-09: the worktree hosting a generation was removed under it by
        # `land.sh`; all four workers died, the cycle landed at 71% with a correct manifest, and
        # nothing in the provenance said the frame had shrunk.)
        # Both branches ask :func:`completeness`, never the raw key — it DERIVES the counts from
        # the manifest's own `n_games` / `opponents` / `selection` when they were not recorded,
        # and a branch that read the key directly would call a derived-INCOMPLETE cycle UNKNOWN.
        comp = completeness(man)
        if comp["complete"] is False:
            text = (f"INCOMPLETE ({comp['battles_played']:,} of {comp['battles_expected']:,} "
                    f"battles played — workers died or the cycle was killed) " + text)
        elif comp["complete"] is None:
            text = ("COMPLETENESS UNKNOWN (the manifest records neither its battle plan nor the "
                    "battles played, so it cannot be certified complete) " + text)
        return text
    opponents = man.get("opponents") or []
    n_games = man.get("n_games")
    if not opponents or not n_games:
        return "UNKNOWN population (the manifest records no opponents/n_games)"
    n_sent = sum(1 for o in opponents if str(o).startswith("sentinel_"))
    n_bot = len(opponents) - n_sent
    return (f"LIVE cycle: {len(opponents)} opponents ({n_bot} scripted bots + {n_sent} pool "
            f"sentinels) x {n_games} games, traced under the per-opponent OUTCOME QUOTA "
            f"(loss-enriched, not a random subsample)")


def completeness(manifest: Optional[dict]) -> Dict[str, Any]:
    """Did this generated cycle play every battle its plan called for?

    Returns ``{complete, battles_played, battles_expected, shortfall}``; a LIVE cycle (no
    provenance block) answers ``complete: True`` with no counts, because a live cycle's own
    partial-coverage warning is the trainer's business and its frame is whatever it is.
    """
    gen = (manifest or {}).get(GENERATED_KEY)
    if not isinstance(gen, dict):
        return {"complete": True, "battles_played": None, "battles_expected": None,
                "shortfall": 0}
    played = gen.get("battles_played")
    want = gen.get("battles_expected")
    if want is None:
        # DERIVED, not guessed. A cycle written before `battles_expected` existed still records
        # everything the number is made of: the plan is `n_games` games against each of
        # `opponents`, and the `selection` block counts what was actually played. Deriving it is
        # strictly better than answering UNKNOWN — the same arithmetic would have caught the 71%
        # frame — and it is arithmetic on the manifest's OWN fields, not an assumption about them.
        man = manifest or {}
        n_games, opponents = man.get("n_games"), man.get("opponents")
        if n_games and opponents:
            want = int(n_games) * len(opponents)
        if played is None:
            per = ((man.get("selection") or {}).get("opponents") or {})
            if per:
                played = sum(int(v.get("battles_played", 0)) for v in per.values())
    if want is None or played is None:
        # 🚨 UNKNOWN, and deliberately NOT True. Nothing on disk can certify this cycle finished,
        # and "it looks fine" is exactly the reading that let the 71% frame through. None is a
        # third value every caller must handle, which is the point: `bool(None)` is False, so a
        # caller that forgets gets the SAFE answer rather than the flattering one.
        return {"complete": None, "battles_played": played, "battles_expected": want,
                "shortfall": None}
    played, want = int(played), int(want)
    complete = bool(gen["complete"]) if "complete" in gen else (played >= want > 0)
    return {"complete": complete, "battles_played": played,
            "battles_expected": want, "shortfall": max(0, want - played)}


def spec_of(manifest: Optional[dict]) -> Dict[str, Any]:
    """The comparability SPEC of a cycle — what two frames must share to be differenced.

    Games per opponent, the opponent SET (order-insensitive), whether every battle was traced,
    and the sentinel regime. NOT the seed, the worker count or the wall clock: two offline frames
    generated with different seeds are two draws from the SAME population, which is exactly what a
    delta wants; two generated with different GAME COUNTS are not.
    """
    man = manifest or {}
    gen = man.get(GENERATED_KEY) if isinstance(man.get(GENERATED_KEY), dict) else {}
    return {
        "generated": is_generated(man),
        "n_games": man.get("n_games"),
        "opponents": sorted(str(o) for o in (man.get("opponents") or [])),
        "capture": (gen or {}).get("capture", "quota"),
        "eval_sentinel_greedy": (gen or {}).get("eval_sentinel_greedy"),
        # Completeness, not the raw battle count: two cycles at one spec that both COMPLETED have
        # equal realized frames by construction, and comparing the counts would refuse a pair over
        # a single timed-out battle. An incomplete side is refused outright by `check_comparable`,
        # which is the stronger statement.
        "complete": (gen or {}).get("complete", True),
    }


# --------------------------------------------------------------------------- resolution

_RUN_REF = re.compile(r"^(?P<run>.*?)(?:@(?P<step>\d+))?$")


def parse_run_ref(arg: str) -> Tuple[Path, Optional[int]]:
    """``<run>`` or ``<run>@<step>`` -> (run dir, pinned step or None)."""
    m = _RUN_REF.match(arg)
    assert m is not None  # the pattern matches every string
    run = m.group("run")
    step = m.group("step")
    return resolve_run_dir(run), (int(step) if step else None)


def resolve_checkpoint(run_dir: Path, step: Optional[int]) -> Tuple[Path, int]:
    """The weights to replay, and the step they are at.

    PREFERS ``eval_traces/step_<N>/snapshot.zip`` — the bit-exact network that played the LIVE
    cycle at that step, so the offline cycle and the live one it is compared against are the same
    weights and not merely the same step. Falls back to ``checkpoints/`` when that snapshot was
    pruned (``--keep-eval-snapshots``), which is a real and common case, and says which it used.
    """
    if step is None:
        steps = sorted(int(os.path.basename(p).split("_")[1])
                       for p in glob.glob(str(run_dir / "eval_traces" / "step_*")))
        if not steps:
            refuse(f"REFUSING: {run_dir.name} has no eval_traces/step_* — pass @<step> naming a "
                   "checkpoint, or point at a run that evaluated at least once.")
        step = steps[-1]
    snap = run_dir / "eval_traces" / f"step_{step}" / "snapshot.zip"
    if snap.exists():
        return snap, step
    ckpts = sorted(glob.glob(str(run_dir / "checkpoints" / "*.zip")))
    exact = [p for p in ckpts if str(step) in os.path.basename(p)]
    if exact:
        return Path(exact[-1]), step
    refuse(f"REFUSING: no weights for {run_dir.name} at step {step}.",
           f"  looked for {snap}",
           f"  and for a checkpoint under {run_dir / 'checkpoints'} naming {step} "
           f"(found {len(ckpts)} checkpoint(s), none matching).",
           "  A cycle generated from the WRONG weights is not a re-read of this arm.")
    raise AssertionError("unreachable")  # pragma: no cover


def recoverable_regime(run_dir: Path) -> Optional[bool]:
    """What the run's ``metadata.json`` says its eval regime was, or ``None``.

    ``model_config.json`` only began carrying ``eval_sentinel_greedy`` at config_version 111
    (2026-09-07, the boundary itself). A PRE-BOUNDARY run therefore records the regime nowhere
    that ``read_regime`` looks — but it does record its whole resolved argparse namespace in
    ``metadata.json``'s ``cli_args``, written by the run itself at its own pin. That is a
    recovery, not a guess, and naming it in the refusal is what lets a caller DECLARE the regime
    knowingly instead of inventing one.

    It is deliberately NOT used as a silent fallback: ``cli_args`` is a large, weakly-typed blob
    and the value it carries for a flag whose DEFAULT moved is exactly the kind of thing that
    should be read by a human once and then stated out loud on the command line.
    """
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except ValueError:
        return None
    cli = meta.get("cli_args")
    if not isinstance(cli, dict) or "eval_sentinel_greedy" not in cli:
        return None
    val = cli["eval_sentinel_greedy"]
    return None if val is None else bool(val)


def read_regime(run_dir: Path, declared: Optional[bool] = None) -> Dict[str, Any]:
    """The run's RECORDED eval regime, or the one the caller DECLARED for a pre-boundary run.

    🚨 ``eval_sentinel_greedy`` carries an OPPONENT-REGIME BOUNDARY (2026-09-07) worth ~8.9 pp to
    the trainee. Guessing it would silently generate the cycle under the OTHER regime from the one
    the arm was measured in, and every number would look like a result. A run that recorded no
    regime is refused; there is no default that is not a lie.

    A PRE-BOUNDARY run records none — ``model_config.json`` only gained the key at the boundary —
    and such a run is not unreadable, it is undeclared. ``--eval-sentinel-greedy`` /
    ``--no-eval-sentinel-greedy`` is how the caller says which, and the refusal NAMES the value
    recoverable from ``metadata.json``'s ``cli_args`` so the declaration is informed rather than
    invented. The source ("recorded" or "declared") is returned, logged and written into the
    manifest, so no reader can mistake one for the other. A declaration that CONTRADICTS a recorded
    regime is refused outright: the run's own record wins, and a caller who disagrees with it is
    confused about which run they are reading.
    """
    cfg_path = run_dir / "model_config.json"
    if not cfg_path.exists():
        refuse(f"REFUSING: {run_dir.name} has no model_config.json — the eval REGIME cannot be "
               "read, and it must not be assumed.")
    try:
        cfg = json.loads(cfg_path.read_text())
    except ValueError as exc:
        refuse(f"REFUSING: {cfg_path} is unreadable: {exc}")
    source = "recorded"
    if "eval_sentinel_greedy" in cfg:
        recorded = bool(cfg["eval_sentinel_greedy"])
        if declared is not None and declared != recorded:
            refuse(
                f"REFUSING: --{'' if declared else 'no-'}eval-sentinel-greedy CONTRADICTS "
                f"{run_dir.name}'s own record (`eval_sentinel_greedy` = {recorded}).",
                "  The run's model_config.json is the record of the regime it was EVALUATED "
                "under; a declaration exists only for a pre-boundary run that recorded none.",
                "  Drop the flag to generate under the recorded regime. Nothing is generated.")
        greedy = recorded
    elif declared is None:
        rec = recoverable_regime(run_dir)
        hint = (["  ITS OWN metadata.json RECORDS `cli_args.eval_sentinel_greedy` = "
                 f"{rec} — the resolved namespace the run saved at its own pin. If that is the "
                 f"regime you mean, say so: pass "
                 f"--{'' if rec else 'no-'}eval-sentinel-greedy."]
                if rec is not None else
                ["  Its metadata.json records no `cli_args.eval_sentinel_greedy` either, so "
                 "nothing on disk answers the question. Do not generate this cycle."])
        refuse(f"REFUSING: {run_dir.name}'s model_config.json records no `eval_sentinel_greedy`.",
               "  That key names an OPPONENT-REGIME BOUNDARY (2026-09-07): greedy sentinels "
               "drawing the trainee's own teams, or the old asymmetric pair. The two differ by "
               "~8.9 pp to the trainee at equal skill.",
               "  A pre-boundary run cannot be re-read under either regime without saying which, "
               "so nothing is generated by default.",
               *hint,
               "  The cycle is then marked `eval_sentinel_greedy_source: declared` in its "
               "manifest, and `critic_read` still refuses a delta against a frame at the other "
               "regime. Nothing is read and nothing is concluded.")
        raise AssertionError("unreachable")  # pragma: no cover
    else:
        greedy = bool(declared)
        source = "declared"
    return {
        "eval_sentinel_greedy": greedy,
        "eval_sentinel_greedy_source": source,
        "self_play_temp": float(cfg.get("self_play_temp", 1.0) or 1.0),
        "gamma": float(cfg.get("gamma") or 0.99),
        "trainee_team_str": cfg.get("trainee_team_str"),
        "config": cfg,
    }


def pick_sentinels(run_dir: Path, step: int, k: int, *,
                   include_current: bool = False) -> Tuple[List[dict], List[str]]:
    """Up to ``k`` pool snapshots, spread across the run's rating range, as sentinel items.

    Drawn from the run's OWN ``snapshots/``, which is the pool a live cycle drew from. Snapshots
    at or above the read step are excluded by default: a live cycle's pool holds only snapshots
    OLDER than the trainee, and a self-mirror is a 50%-by-construction cell that would change what
    the pool row means. ``--include-current-snapshot`` opts in deliberately.

    Fewer than ``k`` available is CLAMPED and reported, never padded: a short run simply has fewer
    distinct opponents, and repeating one to reach a requested count would inflate the
    between-opponent spread with a duplicated cell.
    """
    notes: List[str] = []
    paths = []
    for p in sorted(glob.glob(str(run_dir / "snapshots" / "snapshot_*.zip"))):
        m = re.search(r"snapshot_(\d+)\.zip$", os.path.basename(p))
        if not m:
            continue
        s = int(m.group(1))
        if s >= step and not include_current:
            continue
        paths.append((s, p))
    paths.sort()
    if not paths:
        refuse(f"REFUSING: {run_dir.name} has no pool snapshot below step {step} to use as a "
               "sentinel.", f"  looked under {run_dir / 'snapshots'}.",
               "  Pass --sentinels 0 to generate a BOTS-ONLY cycle, which is a different "
               "population and is recorded as one.")
    if k <= 0:
        return [], ["sentinels DISABLED (--sentinels 0): this is a bots-only population"]
    if len(paths) < k:
        notes.append(
            f"CLAMPED: {k} sentinels requested, {len(paths)} pool snapshots exist below step "
            f"{step:,} — using all {len(paths)}. The count is NOT padded by repeating a "
            f"snapshot; a duplicated cell would inflate the between-opponent spread.")
        chosen = paths
    elif len(paths) == k or k == 1:
        # k == 1 takes the STRONGEST pool snapshot, not an arbitrary one: with a single cell the
        # pool row is that opponent, and the newest snapshot is the one a live cycle's pool
        # weights heaviest.
        chosen = paths if len(paths) == k else [paths[-1]]
    else:
        # Evenly spaced across the step range = evenly spaced across the rating range, because
        # the pool's rating is monotone in step over a healthy run (`pool.monotonicity`). Endpoints
        # always included, so the weakest and the strongest pool opponent are both measured.
        idx = [round(i * (len(paths) - 1) / (k - 1)) for i in range(k)]
        chosen = [paths[i] for i in sorted(set(idx))]
        if len(chosen) < k:  # pragma: no cover — only when rounding collides
            notes.append(f"CLAMPED to {len(chosen)} distinct snapshots after even spacing.")
    return ([{"label": f"sentinel_{i}", "path": str(p), "step": s}
             for i, (s, p) in enumerate(chosen)], notes)


def sha256_of(path: str | Path, *, limit: int = 1 << 22) -> str:
    """A checkpoint fingerprint. Truncated read: the zip's first 4 MiB plus its size is a
    sufficient identity for "is this the same file", at a small fraction of the I/O of hashing
    27 MB five times."""
    h = hashlib.sha256()
    h.update(str(os.path.getsize(path)).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- the shadow run

def build_shadow_run(out_dir: Path, run_dir: Path, step: int, ckpt: Path,
                     sentinels: Sequence[dict]) -> Path:
    """Lay out ``out_dir`` so every reader that takes a RUN DIR resolves against this cycle.

    Four things have to be there, because four different readers look for them:
      * ``model_config.json`` — the eval worker builds the run's REWARD from it, and
        ``write_eval_manifest`` reads the arch identity out of it.
      * ``metadata.json`` — ``cf_audit.sentinel_snapshots`` reads ``latest_eval.pool.sentinels``
        to pin which network each sentinel cell was. It is copied and then REWRITTEN to this
        cycle's sentinels; leaving the source run's block would pin the wrong networks under the
        right labels, which is worse than none.
      * ``snapshots/`` — where those pins resolve. A SYMLINK back to the real run: read-only, and
        nothing is ever written under ``models/``.
      * ``eval_traces/step_<N>/snapshot.zip`` — the network that played the traces, which is where
        ``cf_audit`` looks. Copied, not linked, so the cycle survives the source being groomed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    step_dir = out_dir / "eval_traces" / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(run_dir / "model_config.json", out_dir / "model_config.json")

    meta: Dict[str, Any] = {}
    src_meta = run_dir / "metadata.json"
    if src_meta.exists():
        try:
            meta = json.loads(src_meta.read_text())
        except ValueError:
            meta = {}
    latest = dict(meta.get("latest_eval") or {})
    pool = dict(latest.get("pool") or {})
    pool["sentinels"] = [{"step": s["step"], "snapshot": os.path.basename(s["path"])}
                         for s in sentinels]
    pool["snapshot_count"] = len(sentinels)
    latest["pool"] = pool
    latest["step"] = step
    meta["latest_eval"] = latest
    # An honest marker inside the metadata too: this file is a DERIVED copy, not a run's record.
    meta["derived_from"] = {"tool": GENERATOR_TAG, "source_run_dir": str(run_dir.resolve()),
                            "note": "metadata.json copied from the source run; latest_eval.pool"
                                    ".sentinels REWRITTEN to this generated cycle's sentinels."}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    link = out_dir / "snapshots"
    if not link.exists() and not link.is_symlink():
        os.symlink(str((run_dir / "snapshots").resolve()), str(link))

    dst_ckpt = step_dir / "snapshot.zip"
    if not dst_ckpt.exists():
        shutil.copy2(ckpt, dst_ckpt)
    return step_dir


# --------------------------------------------------------------------------- generation

def _log(msg: str) -> None:
    print(f"[eval_trace_gen] {msg}", flush=True)


def generate(args) -> Dict[str, Any]:
    """Play the cycle and return its manifest. The whole tool, in one readable pass."""
    # Imports deferred: they pull torch and the whole training package, which a --help or a
    # spec-comparison caller has no reason to pay for.
    from agents.training.eval_callback import (
        ForensicQuota, eval_opponent_names, merge_eval_results, record_eval_selection,
        spawn_eval_workers, write_eval_manifest,
    )
    from agents.training.eval_sharding import BOT, SENTINEL, EvalItem, ShardedEvalPool

    run_dir, pinned = parse_run_ref(args.run)
    ckpt, step = resolve_checkpoint(run_dir, pinned)
    regime = read_regime(run_dir, declared=getattr(args, "eval_sentinel_greedy", None))
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        refuse(f"REFUSING: {out_dir} exists and is not empty.",
               "  A generated cycle written on top of another is two samples in one directory, "
               "and nothing downstream could tell them apart. Pass --force to replace it, or "
               "name a fresh --out.")
    models_root = os.environ.get("GEN3AI_MODELS_DIR") or ""
    for forbidden in [p for p in (str(run_dir.parent), models_root) if p]:
        if str(out_dir) == forbidden or str(out_dir).startswith(forbidden.rstrip("/") + "/"):
            refuse(f"REFUSING: --out {out_dir} is inside the run archive ({forbidden}).",
                   "  models/ is the RECORD of what training did. A generated cycle placed in it "
                   "would be indistinguishable from one a run played, forever.")
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)

    sentinels, sentinel_notes = pick_sentinels(run_dir, step, args.sentinels,
                                               include_current=args.include_current_snapshot)
    for note in sentinel_notes:
        _log(f"⚠️  {note}")

    bots = ([b.strip() for b in args.opponents.split(",") if b.strip()]
            if args.opponents else eval_opponent_names())
    known = set(eval_opponent_names())
    unknown = [b for b in bots if b not in known]
    if unknown:
        refuse(f"REFUSING: unknown bot(s) {unknown}. The roster is: {', '.join(sorted(known))}.")

    step_dir = build_shadow_run(out_dir, run_dir, step, ckpt, sentinels)
    claim_dir = step_dir / "claims"
    claim_dir.mkdir(parents=True, exist_ok=True)

    # THE READ CYCLE CAPTURES EVERYTHING. A live cycle's outcome quota exists to bound a training
    # run's disk; here the traces ARE the measurement, and a loss-enriched subsample is precisely
    # what costs the low-variance rows their power. `--quota` restores a live-shaped capture for
    # parity work, and whichever it is, the manifest states it in the selection rule.
    quota = (ForensicQuota(win=args.games, loss=args.games, draw=args.games)
             if args.quota is None else
             ForensicQuota(win=args.quota, loss=args.quota, draw=args.quota))
    capture = "ALL" if args.quota is None else f"quota={args.quota}"

    items = [EvalItem(name, BOT, args.games) for name in bots]
    items += [EvalItem(s["label"], SENTINEL, args.games, path=s["path"], step=s["step"])
              for s in sentinels]
    names = [it.key for it in items]
    pool = ShardedEvalPool(items, args.shard_games, step=step)
    pool.write_plan(str(step_dir))

    write_eval_manifest(str(out_dir), step, opponents=names, n_games=args.games,
                        snapshot="snapshot.zip",
                        trainee_team_str=regime["trainee_team_str"],
                        opponent_pins={}, quota=quota)

    reproducible = args.seed is not None and args.concurrency == 1
    traced = ("ALL battles traced (no capture quota)" if args.quota is None
              else f"traced under a per-opponent outcome quota of {args.quota}")
    sent_regime = ("sentinels GREEDY, drawing the trainee's own team distribution"
                   if regime["eval_sentinel_greedy"] else
                   "sentinels ASYMMETRIC (stochastic, flat pool builder — the pre-2026-09-07 "
                   "regime, worth ~+8.9 pp to the trainee)")
    if regime["eval_sentinel_greedy_source"] == "declared":
        sent_regime += " [regime DECLARED on the command line — the run recorded none]"
    population = (
        f"OFFLINE-GENERATED cycle: {len(names)} opponents ({len(bots)} scripted bots + "
        f"{len(sentinels)} pool sentinels) x {args.games} games, {traced}; {sent_regime}")

    n_workers = max(1, min(args.workers, pool.n_units))
    base_cfg = {
        "snapshot": str(step_dir / "snapshot.zip"),
        "port": None,
        "use_showdown_bridge": True,       # serverless: no Showdown server is touched
        "bridge_impl": args.impl,
        "compile_extractor": False,        # a one-shot offline cycle never amortises the compile
        "model_dir": str(out_dir),         # traces land under the SHADOW dir, never models/
        "step": step,
        "self_play_temp": regime["self_play_temp"],
        "eval_sentinel_greedy": regime["eval_sentinel_greedy"],
        "claim_dir": str(claim_dir),
        "result_dir": str(step_dir),
        "concurrency": args.concurrency,
        "device": "cpu",
        "cycle_tag": f"g{int(time.time()) % 100000:05d}",
        "gamma": regime["gamma"],
        "forensic_quota": quota._asdict(),
        "arch_toggles": _arch_toggles(str(step_dir / "snapshot.zip")),
        "trainee_team_str": regime["trainee_team_str"],
        "seed_base": args.seed,
        # See the worker: a `periodic` checkpoint print()s a DEEP TRACE banner from inside the
        # forward, which at read-cycle volume is cost and noise, not a debugging aid.
        "disable_obs_debugger": True,
    }

    _log(f"{run_dir.name} @ {step:,}: {len(names)} opponents x {args.games} games "
         f"= {len(names) * args.games:,} battles, {pool.n_units} shard units, "
         f"{n_workers} worker(s), concurrency {args.concurrency}, capture {capture}")
    _log(f"checkpoint: {ckpt}  (sha {sha256_of(ckpt)})")
    _src = regime["eval_sentinel_greedy_source"]
    _log(f"regime: eval_sentinel_greedy={regime['eval_sentinel_greedy']} "
         + ("(RECORDED, not assumed)" if _src == "recorded" else
            "(DECLARED on the command line — this run's model_config.json records none; the "
            "declaration is written into the manifest and bounds every delta this cycle enters)"))
    if not reproducible:
        _log("⚠️  NOT REPRODUCIBLE: " + (
            "no --seed was given, so every stream is unseeded."
            if args.seed is None else
            f"--concurrency {args.concurrency} > 1 interleaves several battles of one shard over "
            "the process-global `random` stream the scripted bots draw from, and the order they "
            "draw in is a timing race. The seed pins the teams and the sim dice; it cannot pin "
            "that interleave. Re-run at --concurrency 1 for a reproducible cycle."))
    else:
        _log(f"reproducible: seed {args.seed} at concurrency 1 (the worker count is free — a "
             "shard's streams are keyed on the PLAN, not on which worker claimed it)")

    env_note = _cpu_env(args)
    started = time.time()
    procs = spawn_eval_workers(str(step_dir), base_cfg, n_workers)
    _wait(procs, args)
    elapsed = time.time() - started

    merged, missing = merge_eval_results(str(step_dir), names)
    if missing:
        _log(f"⚠️  no results at all for {missing} — a worker died holding every one of its "
             f"claims. See {step_dir}/worker_*.log")
    selection = record_eval_selection(str(out_dir), step, merged, quota=quota)
    if selection is None:
        refuse("REFUSING: the cycle collected no per-opponent counts, so no `selection` block "
               "could be recorded — every consumer would read this tree as SELECTION UNKNOWN.",
               f"  worker logs: {step_dir}/worker_*.log")

    n_battles = sum(v.get("battles_played", 0)
                    for v in (selection.get("opponents") or {}).values())
    expected_battles = len(names) * args.games
    manifest_path = step_dir / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[GENERATED_KEY] = {
        "tool": GENERATOR_TAG,
        "schema": GENERATED_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": run_dir.name,
        "source_run_dir": str(run_dir.resolve()),
        "checkpoint": str(ckpt),
        "checkpoint_sha": sha256_of(ckpt),
        "games_per_opponent": args.games,
        "bots": list(bots),
        "sentinels_requested": args.sentinels,
        "sentinels_used": len(sentinels),
        "sentinel_steps": [s["step"] for s in sentinels],
        "sentinel_notes": sentinel_notes,
        "capture": capture,
        "quota": quota._asdict(),
        "eval_sentinel_greedy": regime["eval_sentinel_greedy"],
        "eval_sentinel_greedy_source": regime["eval_sentinel_greedy_source"],
        "seed": args.seed,
        "workers": n_workers,
        "concurrency": args.concurrency,
        "shard_games": args.shard_games,
        "impl": args.impl,
        "reproducible": reproducible,
        "reproducibility_note": (
            "seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the "
            "worker count does not enter it" if reproducible else
            "NOT reproducible: " + ("no seed" if args.seed is None else
                                    f"concurrency {args.concurrency} > 1 races the scripted bots' "
                                    "shared `random` stream within a shard")),
        "battles_played": n_battles,
        "battles_expected": expected_battles,
        # 🚨 The frame's own completeness, recorded rather than left to be inferred from two other
        # numbers. A cycle can end short for reasons that have nothing to do with the model — a
        # worker crash, a kill for overrunning, or (2026-09-09) the WORKTREE the generation was
        # running out of being removed under it — and a short frame is a different population from
        # the one `n_games` advertises.
        "complete": n_battles >= expected_battles,
        "shortfall": max(0, expected_battles - n_battles),
        "wall_seconds": round(elapsed, 1),
        "games_per_sec": round(n_battles / elapsed, 3) if elapsed > 0 else None,
        "environment": env_note,
        "population": population,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    shutil.rmtree(claim_dir, ignore_errors=True)
    for shard in glob.glob(str(step_dir / "shard__*.json")):
        os.remove(shard)

    _log(f"DONE in {elapsed / 60:.1f} min — {n_battles:,} battles, "
         f"{n_battles / elapsed:.2f} games/sec, "
         f"{sum(1 for _ in glob.glob(str(step_dir / '*' / '*_states.npz'))):,} traces")
    if n_battles < expected_battles:
        _log(f"🚨 INCOMPLETE: {n_battles:,} of {expected_battles:,} battles "
             f"({n_battles / expected_battles * 100:.0f}%). This cycle is a SMALLER FRAME than "
             f"its {args.games}-game spec advertises, and frame size moves every fitted "
             f"conditioning row on its own — `main.ops.critic_read` will REFUSE to read it "
             f"against a complete one. Re-generate before reading. Worker logs: "
             f"{step_dir}/worker_*.log")
    _log(f"cycle: {step_dir}")
    return manifest


def _arch_toggles(snapshot: str) -> dict:
    """This checkpoint's arch toggles, for the workers' sentinel version gate.

    Read by LOADING the snapshot once here rather than guessing from the config, because that is
    what a live cycle does (``arch_toggles_from_model`` on the live policy) and a toggle-OFF
    default FATALs on the run's own belief-ON sentinels.
    """
    from sb3_contrib import MaskablePPO
    from agents.model.snapshot import arch_toggles_from_model
    model = MaskablePPO.load(snapshot, env=None, device="cpu")
    return arch_toggles_from_model(model)


def _cpu_env(args) -> dict:
    """Pin this process (and hence its children) to CPU and a polite niceness."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        os.nice(max(0, args.nice - os.nice(0)))
    except OSError:
        pass
    return {"CUDA_VISIBLE_DEVICES": "", "nice": args.nice,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}


def _wait(procs: List[dict], args) -> None:
    """Block until every worker exits, reporting progress on a slow cadence."""
    deadline = time.time() + args.timeout_min * 60 if args.timeout_min else None
    last = 0.0
    while True:
        if all(w["proc"].poll() is not None for w in procs):
            break
        if deadline and time.time() > deadline:
            _log(f"⚠️  TIMEOUT after {args.timeout_min} min — killing workers by explicit PID and "
                 "collecting whatever landed. A timed-out cycle is INCONCLUSIVE, not a result.")
            for w in procs:
                if w["proc"].poll() is None:
                    w["proc"].kill()
            break
        if time.time() - last > 300:
            alive = sum(1 for w in procs if w["proc"].poll() is None)
            _log(f"... {alive}/{len(procs)} worker(s) alive")
            last = time.time()
        time.sleep(5)
    for w in procs:
        w["log"].close()
    bad = [w for w in procs if w["proc"].returncode not in (0, None)]
    for w in bad:
        _log(f"⚠️  worker exited {w['proc'].returncode}; see {w['log_path']}")


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.ops.eval_trace_gen",
        description="Generate an eval-traces cycle for a saved checkpoint, OFFLINE, on CPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("run", help="a run NAME or DIRECTORY, optionally @<step> (default: last "
                               "evaluated step)")
    p.add_argument("--out", required=True,
                   help="output directory for the generated SHADOW RUN. Must NOT be inside "
                        "models/ — the run archive is the record of what training did.")
    p.add_argument("--games", type=int, default=400,
                   help="games per opponent (default 400; a live cycle plays 100)")
    p.add_argument("--sentinels", type=int, default=6,
                   help="pool snapshots to use as sentinels, spread across the rating range "
                        "(default 6; CLAMPED to what the run has, never padded). 0 = bots only.")
    p.add_argument("--opponents", default=None,
                   help="comma-separated bot subset (default: the full 9-bot roster)")
    p.add_argument("--seed", type=int, default=None,
                   help="pin every stream a shard draws from. Reproducible at --concurrency 1, "
                        "for ANY --workers. Omit for an unseeded cycle (and the manifest says so).")
    p.add_argument("--workers", type=int, default=4,
                   help="eval worker processes (default 4 — a training arm normally shares this "
                        "box)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="battles in flight per worker (default 1). Above 1 the cycle is NOT "
                        "reproducible; see the module docstring.")
    p.add_argument("--nice", type=int, default=15,
                   help="niceness for this process and its workers (default 15)")
    p.add_argument("--shard-games", type=int, default=25,
                   help="games per work-steal shard unit (default 25)")
    p.add_argument("--impl", default="rust", choices=["node", "rust"],
                   help="which in-process sim bridge (default rust — serverless)")
    p.add_argument("--quota", type=int, default=None,
                   help="per-outcome trace quota per opponent. DEFAULT IS NO QUOTA: every battle "
                        "is traced, because here the traces ARE the measurement.")
    p.add_argument("--include-current-snapshot", action="store_true",
                   help="allow a pool snapshot at or above the read step (a self-mirror cell)")
    greedy = p.add_mutually_exclusive_group()
    greedy.add_argument("--eval-sentinel-greedy", "--eval_sentinel_greedy",
                        dest="eval_sentinel_greedy", action="store_true", default=None,
                        help="DECLARE the sentinel regime for a PRE-BOUNDARY run whose "
                             "model_config.json records none (config_version < 111). Refused when "
                             "the run DOES record one — the run's record wins. The cycle is marked "
                             "`eval_sentinel_greedy_source: declared`, and critic_read still "
                             "refuses a delta against a frame at the other regime.")
    greedy.add_argument("--no-eval-sentinel-greedy", "--no_eval_sentinel_greedy",
                        dest="eval_sentinel_greedy", action="store_false", default=None,
                        help="the same declaration, for the PRE-2026-09-07 asymmetric regime "
                             "(stochastic sentinels, the flat pool teambuilder).")
    p.add_argument("--timeout-min", type=float, default=0.0,
                   help="kill the workers after this many minutes (0 = no bound)")
    p.add_argument("--force", action="store_true", help="replace a non-empty --out")
    p.add_argument("--json", default=None, help="also write the manifest to this path")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games < 1:
        refuse("REFUSING: --games must be >= 1.")
    manifest = generate(args)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
