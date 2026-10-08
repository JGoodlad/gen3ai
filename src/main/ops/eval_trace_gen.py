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

WHAT IT IS NOT. It is not a re-implementation of eval. It plays the cycle on the SAME executor a
live cycle plays on — ``RustEvalCore.run_cycle`` over a ``ShardedEvalPool`` plan, on an eval core and
T2 slots this tool declares the way the trainer does (``agents.training.rust_eval.offline``) — and the
executor writes the same CORE traces (``rust_eval.traces``: records + reconstruction + states + a
``meta``-only ``*_summary.json``, which the prober expands), the same shard results the collect merges,
and an ``eval_manifest.json`` at ``selection_schema`` 2 with the per-opponent capture rates rule 17 asks
for. The regime is READ from the run's recorded config (``eval_sentinel_greedy``, the mirrored-pair
regime, the observation mode, the reward's terminal, the trainee's team pin), never assumed; a run that
recorded no sentinel regime is REFUSED rather than measured under a guess. (Until poke-env retirement
P6 slice 6c it drove ``python -m main.eval_worker`` processes on the poke-env eval path; a cycle
generated then carries no ``transport`` in its provenance block, and ``spec_of`` keeps the two engines'
cycles apart.)

🚨 THE OUTPUT IS A DIFFERENT POPULATION FROM A LIVE CYCLE, and that is the point of generating it.
More games, possibly more sentinels, and — by default — NO capture quota at all, so every battle
played is a battle traced. A delta between an offline frame and a live frame is therefore a delta
between two different samples of two different populations, and ``main.ops.critic_read`` REFUSES
to form one. The manifest carries a ``generated_by`` block naming the tool, the games, the
sentinel spec, the seed, the eval core's compute flags and the checkpoint sha so no reader can mistake this
cycle for a live one, and so two offline frames can be checked for the SAME spec before they are
differenced.

NOTHING IS EVER WRITTEN UNDER ``models/``. The generated cycle is a self-contained SHADOW RUN
DIR under ``--out``: the run's ``model_config.json`` and ``metadata.json`` are copied (with the
sentinel block rewritten to the sentinels this cycle actually used), ``snapshots/`` is symlinked
back to the real run read-only, and the checkpoint is copied to
``<out>/eval_traces/step_<N>/snapshot.zip`` — where the offline readers look for the network that
played the traces, so they resolve against the generated cycle with no special-casing. (The offline
identity reader, ``cf_audit``, that first set this layout was deleted in P6 slice 6d-1; the layout stays.)

REPRODUCIBILITY. Every game on the Rust eval core is seeded by the GAME (``gen3_eval_game_seed_v1``,
``rust_eval.seeds``): its two teams, its battle seed, the bots' streams and a sampled sentinel's draws are
a pure function of (``--seed``, opponent, game index) — never of the env that played it, the schedule or
a thread count. So the same seed, checkpoint, plan and compute flags give the IDENTICAL cycle; a run with
no ``--seed`` draws one and RECORDS it (there is no unseeded mode). What the seed does NOT pin is the
float rounding of the batched forward, whose batch composition follows ``--n-envs`` and the schedule: a
greedy decision whose top-2 log-prob margin sits within a rounding error of a tie can resolve the other
way under different compute flags, another env-core build or another torch. The manifest records the
compute flags and the cycle's near-tie count beside the seed, and says all of this in words.
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
        # THE ENGINE: a cycle generated on the poke-env eval worker (before the Rust port; no `transport` recorded)
        # and one generated on the Rust eval core are different engines and encoders, never one population.
        "transport": ((gen or {}).get("transport", "python_bridge") if is_generated(man) else None),
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
        # the other two recorded eval-regime fields the in-loop cycle plays at (absent = their defaults: a run
        # that predates the field never ran the other value)
        "eval_mirrored_pairs": bool(cfg.get("eval_mirrored_pairs", False)),
        "oracle_reveal": str(cfg.get("oracle_reveal") or "off"),
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
      * ``model_config.json`` — the eval core's TERMINAL is built from it, and
        ``write_eval_manifest`` reads the arch identity out of it.
      * ``metadata.json`` — a reader of ``latest_eval.pool.sentinels`` pins which network each
        sentinel cell was (the deleted ``cf_audit.sentinel_snapshots`` was the one that did). It is copied and then REWRITTEN to this
        cycle's sentinels; leaving the source run's block would pin the wrong networks under the
        right labels, which is worse than none.
      * ``snapshots/`` — where those pins resolve. A SYMLINK back to the real run: read-only, and
        nothing is ever written under ``models/``.
      * ``eval_traces/step_<N>/snapshot.zip`` — the network that played the traces, which is where
        the offline readers look. Copied, not linked, so the cycle survives the source being groomed.
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


class CycleTimeout(RuntimeError):
    """``--timeout-min`` ran out: raised from the cycle's safe point, between two host steps."""


def draw_seed() -> int:
    """A fresh cycle seed for a run that named none. It is RECORDED (``seed`` + ``seed_source: drawn``), so the
    cycle is replayable from its own manifest — the Rust core has no unseeded mode: every game is seeded by the GAME."""
    import secrets

    return secrets.randbits(62)


def generate(args) -> Dict[str, Any]:
    """Play the cycle and return its manifest. The whole tool, in one readable pass."""
    # Imports deferred: they pull torch and the whole training package, which a --help or a
    # spec-comparison caller has no reason to pay for.
    from agents.training.eval_collect import merge_eval_results, record_eval_selection
    from agents.training.eval_launch import write_eval_manifest
    from agents.training.eval_quota import ForensicQuota
    from agents.training.eval_schedule import eval_opponent_names
    from agents.training.rust_eval import offline as OFF
    from agents.training.rust_eval import seeds as SD
    from main.h2h.play import check_team_pool
    from utils.rust_env.build import ensure_built

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
    mirrored = bool(regime["eval_mirrored_pairs"])
    if mirrored and args.games % 2:
        refuse(f"REFUSING: --games {args.games} is odd, and {run_dir.name} recorded the MIRRORED-PAIR eval regime "
               "(`eval_mirrored_pairs`): a pair is two games, and half a pair is not a measurement.",
               f"  Pass --games {args.games + 1} (or {args.games - 1}).")
    try:
        check_team_pool()
    except Exception as exc:  # noqa: BLE001 — a typed H2HError; restated as this tool's refusal
        refuse(f"REFUSING: {exc}")
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

    # THE READ CYCLE CAPTURES EVERYTHING. A live cycle's outcome quota exists to bound a training
    # run's disk; here the traces ARE the measurement, and a loss-enriched subsample is precisely
    # what costs the low-variance rows their power. `--quota` restores a live-shaped capture for
    # parity work, and whichever it is, the manifest states it in the selection rule.
    quota = (ForensicQuota(win=args.games, loss=args.games, draw=args.games)
             if args.quota is None else
             ForensicQuota(win=args.quota, loss=args.quota, draw=args.quota))
    capture = "ALL" if args.quota is None else f"quota={args.quota}"

    items = OFF.plan_items(list(bots), [s["path"] for s in sentinels], args.games,
                           sentinel_steps=[s["step"] for s in sentinels])
    names = [it.key for it in items]
    expected_labels = [*bots, *[s["label"] for s in sentinels]]
    if names != expected_labels:    # the plan's sentinel keys ARE the shadow metadata's labels (the sentinel pins read them)
        raise AssertionError(f"plan keys {names} != the shadow run's labels {expected_labels}")

    write_eval_manifest(str(out_dir), step, opponents=names, n_games=args.games,
                        snapshot="snapshot.zip",
                        trainee_team_str=regime["trainee_team_str"],
                        opponent_pins={}, quota=quota, mirrored_pairs=mirrored)

    seed_source = "given" if args.seed is not None else "drawn"
    seed = int(args.seed) if args.seed is not None else draw_seed()
    traced = ("ALL battles traced (no capture quota)" if args.quota is None
              else f"traced under a per-opponent outcome quota of {args.quota}")
    sent_regime = ("sentinels GREEDY, drawing the trainee's own team distribution"
                   if regime["eval_sentinel_greedy"] else
                   "sentinels SAMPLED at the run's self-play temperature, drawing the flat pool "
                   "(the pre-2026-09-07 asymmetric regime, worth ~+8.9 pp to the trainee)")
    if regime["eval_sentinel_greedy_source"] == "declared":
        sent_regime += " [regime DECLARED on the command line — the run recorded none]"
    if mirrored:
        sent_regime += "; MIRRORED team pairs"
    population = (
        f"OFFLINE-GENERATED cycle on the Rust eval core: {len(names)} opponents ({len(bots)} scripted bots + "
        f"{len(sentinels)} pool sentinels) x {args.games} games, {traced}; {sent_regime}")

    n_battles_plan = len(names) * args.games
    _log(f"{run_dir.name} @ {step:,}: {len(names)} opponents x {args.games} games "
         f"= {n_battles_plan:,} battles on the Rust eval core ({args.n_envs} envs, {args.threads} core thread(s), "
         f"{args.torch_threads} torch thread(s), front {args.front}, build {args.profile}), capture {capture}")
    _log(f"checkpoint: {ckpt}  (sha {sha256_of(ckpt)})")
    _src = regime["eval_sentinel_greedy_source"]
    _log(f"regime: eval_sentinel_greedy={regime['eval_sentinel_greedy']} "
         + ("(RECORDED, not assumed)" if _src == "recorded" else
            "(DECLARED on the command line — this run's model_config.json records none; the "
            "declaration is written into the manifest and bounds every delta this cycle enters)")
         + f", eval_mirrored_pairs={mirrored}, oracle_reveal={regime['oracle_reveal']}")
    _log(f"seed {seed} ({seed_source}) — every game is a pure function of (seed, opponent, game index): "
         "re-run with this --seed and the same compute flags for the identical cycle")

    env_note = _cpu_env(args)
    ensure_built(args.profile, emit=_log)
    deadline = time.time() + args.timeout_min * 60 if args.timeout_min else None

    def safe_point(_where: str) -> None:
        if deadline is not None and time.time() > deadline:
            raise CycleTimeout(f"--timeout-min {args.timeout_min} ran out")

    timed_out = False
    stats: Dict[str, Any] = {}
    started = time.time()
    try:
        played = OFF.run_rust(
            run_dir=step_dir, model_dir=out_dir, trainee=str(step_dir / "snapshot.zip"),
            sentinels=[s["path"] for s in sentinels], items=items, shard_games=args.shard_games, step=step,
            cycle_seed=seed, quota=quota._asdict(), device="cpu", backend="eager", n_envs=args.n_envs,
            buckets=_cpu_buckets(args.n_envs), front=args.front, profile=args.profile,
            sentinel_greedy=bool(regime["eval_sentinel_greedy"]), self_play_temp=float(regime["self_play_temp"]),
            safe_point=safe_point, trainee_team_str=regime["trainee_team_str"], mirrored=mirrored,
            core_threads=args.threads, torch_threads=args.torch_threads, forensic_root=step_dir, keep_games=False,
            oracle_reveal=regime["oracle_reveal"], emit=_log)
        stats = played["stats"]
    except CycleTimeout:
        timed_out = True
        _log(f"⚠️  TIMEOUT after {args.timeout_min} min — the cycle was abandoned at a host step; collecting the "
             "shard units that finished. A timed-out cycle is INCOMPLETE, not a result.")
    elapsed = time.time() - started

    merged, missing = merge_eval_results(str(step_dir), names)
    if missing:
        _log(f"⚠️  no results at all for {missing} — the cycle ended before any of their shard units finished")
    selection = record_eval_selection(str(out_dir), step, merged, quota=quota)
    if selection is None:
        refuse("REFUSING: the cycle collected no per-opponent counts, so no `selection` block "
               "could be recorded — every consumer would read this tree as SELECTION UNKNOWN.",
               f"  cycle dir: {step_dir}")

    n_battles = sum(v.get("battles_played", 0)
                    for v in (selection.get("opponents") or {}).values())
    expected_battles = n_battles_plan
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
        "eval_mirrored_pairs": mirrored,
        "oracle_reveal": regime["oracle_reveal"],
        "self_play_temp": regime["self_play_temp"],
        # THE ENGINE. A cycle generated before the Rust port (no `transport`) played on the poke-env eval worker:
        # a different engine and encoder, so `spec_of` keeps the two apart.
        "transport": TRANSPORT,
        "eval_core": {"n_envs": args.n_envs, "threads": args.threads, "torch_threads": args.torch_threads,
                      "front": args.front, "profile": args.profile, "device": "cpu", "backend": "eager"},
        "seed": seed,
        "seed_source": seed_source,
        "seed_rule": SD.SCHEMA,
        "shard_games": args.shard_games,
        "reproducible": not timed_out,
        "reproducibility_note": REPRODUCIBILITY_NOTE if not timed_out else
        "NOT reproducible as a whole: the cycle hit --timeout-min, and where it stopped is wall-clock",
        "near_ties": stats.get("near_ties"),
        "trainee_decisions": stats.get("trainee_decisions"),
        "battles_played": n_battles,
        "battles_expected": expected_battles,
        # 🚨 The frame's own completeness, recorded rather than left to be inferred from two other
        # numbers. A cycle can end short for reasons that have nothing to do with the model — a
        # timeout, a kill, or (2026-09-09) the WORKTREE the generation was running out of being
        # removed under it — and a short frame is a different population from the one `n_games` advertises.
        "complete": n_battles >= expected_battles,
        "shortfall": max(0, expected_battles - n_battles),
        "wall_seconds": round(elapsed, 1),
        "games_per_sec": round(n_battles / elapsed, 3) if elapsed > 0 else None,
        "environment": env_note,
        "population": population,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for shard in glob.glob(str(step_dir / "shard__*.json")):
        os.remove(shard)

    _log(f"DONE in {elapsed / 60:.1f} min — {n_battles:,} battles, "
         f"{n_battles / max(elapsed, 1e-9):.2f} games/sec, "
         f"{sum(1 for _ in glob.glob(str(step_dir / '*' / '*_states.npz'))):,} traces")
    if n_battles < expected_battles:
        _log(f"🚨 INCOMPLETE: {n_battles:,} of {expected_battles:,} battles "
             f"({n_battles / expected_battles * 100:.0f}%). This cycle is a SMALLER FRAME than "
             f"its {args.games}-game spec advertises, and frame size moves every fitted "
             f"conditioning row on its own — `main.ops.critic_read` will REFUSE to read it "
             f"against a complete one. Re-generate before reading.")
    _log(f"cycle: {step_dir}")
    return manifest


#: The engine stamp of a cycle this tool generates (the ladder's / the untaught meter's vocabulary).
TRANSPORT = "rust_eval"

#: What a seed pins on the Rust eval core, said once and written into every manifest.
REPRODUCIBILITY_NOTE = (
    "seeded by the GAME (gen3_eval_game_seed_v1): each game's two teams, battle seed, bot streams and a sampled "
    "sentinel's draws are a pure function of (seed, opponent, game index), never of the env, the schedule or the "
    "thread count — so the same seed, checkpoint, plan (--games/--sentinels/--opponents/--shard-games) and compute "
    "flags give the identical cycle. NOT pinned by the seed: the float rounding of the batched forward, whose batch "
    "composition follows --n-envs and the schedule — a greedy decision whose top-2 log-prob margin is within a "
    "rounding error of a tie (`near_ties` counts those under 2e-5) can resolve differently under other compute "
    "flags or another core build / torch")


def _cpu_buckets(n_envs: int) -> Tuple[int, ...]:
    """The served batch buckets for an eager CPU forward of ``n_envs`` rows (one small, one covering every env)."""
    return tuple(sorted({min(8, int(n_envs)), int(n_envs)}))


def _cpu_env(args) -> dict:
    """Pin this process to CPU and a polite niceness."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        os.nice(max(0, args.nice - os.nice(0)))
    except OSError:
        pass
    return {"CUDA_VISIBLE_DEVICES": "", "nice": args.nice,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.ops.eval_trace_gen",
        description="Generate an eval-traces cycle for a saved checkpoint, OFFLINE, on CPU, on the Rust eval core.",
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
                   help="the cycle seed (every game is seeded by the GAME from it). Omitted: one is drawn and "
                        "RECORDED in the manifest (`seed_source: drawn`), so the cycle is still replayable.")
    p.add_argument("--n-envs", type=int, default=32,
                   help="envs of the eval core (default 32 — a training arm normally shares this box)")
    p.add_argument("--threads", type=int, default=4, help="the Rust eval core's worker threads (default 4)")
    p.add_argument("--torch-threads", type=int, default=4, help="intra-op threads of the CPU forward (default 4)")
    p.add_argument("--front", default="proc", choices=("proc", "ffi"),
                   help="the eval core's front end (default proc, as the trainer)")
    p.add_argument("--profile", default="release", choices=("release", "selfcheck"),
                   help="the env core build (default release; selfcheck = the emission self-check build)")
    p.add_argument("--nice", type=int, default=15,
                   help="niceness for this process (default 15)")
    p.add_argument("--shard-games", type=int, default=25,
                   help="games per shard unit — the unit the capture quota is split over (default 25)")
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
                   help="abandon the cycle after this many minutes at its next host step and keep the shard units "
                        "that finished — recorded INCOMPLETE (0 = no bound)")
    p.add_argument("--force", action="store_true", help="replace a non-empty --out")
    p.add_argument("--json", default=None, help="also write the manifest to this path")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games < 1:
        refuse("REFUSING: --games must be >= 1.")
    if args.n_envs < 1 or args.threads < 1 or args.torch_threads < 1:
        refuse("REFUSING: --n-envs, --threads and --torch-threads must be >= 1.")
    manifest = generate(args)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
