"""``main.ops.quota_match`` — equalise two ladder arms' TRACE FRAMES before a fitted conditioning
row is compared between them.

**The defect this exists for.** The critic ladder's read (:mod:`main.ops.critic_read`) compares an
arm against a control on the conditioning meters. Every gauge row is Horvitz-Thompson reweighted by
the cycle's own capture rates, so the loss-ENRICHMENT of the eval quota is handled (rule of
evidence 17). What reweighting does not touch is **frame SIZE**: an out-of-fold score of a decoder
FIT on the frame has an expectation that rises with the number of battles it was fit on, and an
UNCORRECTED second moment carries noise that grows as the cells shrink. When the two sides are
traced at different outcome quotas — the ladder's arms run 40/40/10 against the control's 5/10/5 —
those rows are not comparable however they are weighted.

That is not a hypothesis. On 2026-09-09 ``ai_v12_12_ladder_cflabels``'s ``cond.own_team_r2.t1``
read **Δ +0.0841 [+0.0323, +0.1825] DETECTED**; cut to the control's realized profile it read
**Δ −0.0013 [−0.2436, +0.3461] NOT DETECTED**, and the arm's own value tracked its decoder's
battle count monotonically (62 → −0.025, 102 → +0.004, 176 → +0.037, 353 → +0.068, 429 → +0.060).
The detection was WITHDRAWN (ledger 2026-09-09 · RETRACTION), and the standing consequence is that
a decoder-based conditioning row may not be compared across arms traced at different quotas. This
module makes the equalisation part of the read, so that consequence is enforced by the tool rather
than remembered by the reader.

**How it matches.**

1. **The realized profile, never the nominal quota.** Under battle-level work-stealing each shard
   unit carries ``max(1, ceil(quota / n_shards))``, so a nominal 5/10/5 lands on disk as 8 traced
   wins and up to 12 traced losses per opponent. Matching the nominal numbers would over-shrink the
   richer side by ~35% and make the artefact look larger than it is. The profile is read from the
   manifest's ``selection`` block and CROSS-CHECKED against the battles actually on disk; where
   they disagree the DISK wins, because the disk is what the read consumes.
2. **The CAP is what a quota sets, and the cap is what is matched.** Each side's cap is the
   per-class maximum over opponents of its realized traced counts. The richer side is cut, per
   opponent, to the poorer side's cap. Residual per-opponent differences BELOW the cap are the two
   runs' own outcome mixes (one arm simply lost more games to one bot), not a selection asymmetry
   — equalising those would shrink both frames for no power gain, so they are reported and left
   alone.
3. **Subsampling is IN MEMORY.** The richer side's cycle is extracted once and the kept battles are
   selected from that array; the capture rates are RECOMPUTED for each subsample against the
   manifest's own denominators, so the HT weights describe the view actually read. Nothing is
   copied, symlinked or written — in particular nothing is written under ``models/``.
4. **Two rungs, both reported.** ``battle`` matches the battle count. ``decoder`` matches the
   frame the own-team decoder is actually FIT on: ``MIN_TEAM_BATTLES = 4`` makes that frame a
   nonlinear function of team diversity, and an arm carrying more distinct teams per battle
   under-fills its decoder at equal battle counts (62 against the control's 104 in the cflabels
   read). Battle-matching is then *unfair to the arm*, so the decoder-matched rung is printed
   beside it.

Draws never enter a conditioning row — :func:`~main.ops.conditioning_meters.extract_cycle` drops
every battle whose result is not WIN or LOSS — so the draw cap is cosmetic and is applied only so
the reported profile and the applied caps are the same object.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from main.ops import conditioning_meters as CM

#: the subsample RNG's third word, so a subsample stream cannot collide with any other stream
#: seeded from the same (seed, index) pair. Kept identical to the 2026-09-09 measurement's, which
#: is what makes this module's numbers comparable with the committed matched-quota read.
SALT = 0xC0FFEE
#: subsample seeds per rung. ODD by default so the median is an exact order statistic: the point
#: printed and the interval printed beside it then describe the SAME draw, rather than pairing an
#: interpolated median with one arbitrary seed's bootstrap. The 2026-09-09 read used 30; the
#: standing consequence asks for >= 20.
DEFAULT_SEEDS = 21
#: the across-seed spread reported beside the battle-clustered CI.
SPREAD_Q = (2.5, 97.5)
#: multipliers on the battle-matched caps searched when hunting the DECODER-matched rung.
DECODER_SCALE_GRID = tuple(round(1.0 + 0.125 * i, 3) for i in range(0, 25))
#: seeds averaged when sizing a candidate decoder rung (the search only needs a stable count).
DECODER_PROBE_SEEDS = (0, 1, 2)
#: a cap-equal pair whose traced totals still differ by more than this is REPORTED (never matched
#: on) — equal caps with very unequal totals means something other than the quota shaped the tree.
TOTAL_SKEW_REPORT = 0.25

UNMATCHED_LABEL = "UNMATCHED — not a reading"


class Profile(NamedTuple):
    """One cycle's REALIZED per-opponent capture profile."""

    run: str
    step: int
    trace_dir: str
    #: opponent -> {"wins", "losses", "draws"} as counted ON DISK, plus the manifest's own
    #: ``battles_played`` / ``battles_won`` / ``battles_drawn`` denominators and what the
    #: manifest CLAIMED it wrote.
    opponents: Dict[str, Dict[str, int]]
    #: (cap_win, cap_loss, cap_draw) — the per-class maximum over opponents.
    cap: Tuple[int, int, int]
    n_traced: int
    #: where the manifest's ``traces_*`` and the files on disk disagree, one line each.
    disk_mismatches: List[str]

    def totals(self) -> Tuple[int, int, int]:
        return (sum(o["wins"] for o in self.opponents.values()),
                sum(o["losses"] for o in self.opponents.values()),
                sum(o["draws"] for o in self.opponents.values()))

    def caps_str(self) -> str:
        return "/".join(str(c) for c in self.cap)


# --------------------------------------------------------------------------- the profile

def classify_on_disk(trace_dir: str, opp: str) -> Dict[str, List[str]]:
    """The opponent's traced battle basenames split WIN / LOSS / DRAW by the summary's result.

    Only the ``_summary.json`` siblings are opened — a battle with no summary is not readable by
    the meters either, so it is not counted.
    """
    out: Dict[str, List[str]] = {"WIN": [], "LOSS": [], "DRAW": []}
    odir = os.path.join(trace_dir, opp)
    if not os.path.isdir(odir):
        return out
    for fn in sorted(os.listdir(odir)):
        if not fn.endswith("_states.npz"):
            continue
        base = fn[: -len("_states.npz")]
        spath = os.path.join(odir, base + "_summary.json")
        if not os.path.exists(spath):
            continue
        try:
            with open(spath) as fh:
                res = (json.load(fh).get("meta") or {}).get("result")
        except (OSError, ValueError):
            continue
        out[res if res in ("WIN", "LOSS") else "DRAW"].append(base)
    return out


def realized_profile(run: str, step: int, trace_dir: str) -> Profile:
    """The cycle's realized per-opponent capture profile, from the manifest AND the disk.

    ⚠️ A nominal quota is not a realized one. The manifest's ``selection`` block records what the
    trainer intended to write (``traces_written`` / ``traces_won`` / ``traces_drawn``); the files
    beside it are what a read actually consumes. Both are carried, the DISK counts are the ones
    matched on, and every disagreement is named in :attr:`Profile.disk_mismatches` rather than
    silently resolved.
    """
    man_path = os.path.join(trace_dir, "eval_manifest.json")
    with open(man_path) as fh:
        man = json.load(fh)
    sel = ((man.get("selection") or {}).get("opponents")) or {}
    opponents: Dict[str, Dict[str, int]] = {}
    mismatches: List[str] = []
    for opp in sorted(sel):
        rec = sel[opp]
        disk = classify_on_disk(trace_dir, opp)
        played = int(rec.get("battles_played", 0))
        won = int(rec.get("battles_won", 0))
        drawn = int(rec.get("battles_drawn", 0))
        claim_w = int(rec.get("traces_won", 0))
        claim_d = int(rec.get("traces_drawn", 0))
        claim_l = int(rec.get("traces_written", 0)) - claim_w - claim_d
        entry = {"wins": len(disk["WIN"]), "losses": len(disk["LOSS"]),
                 "draws": len(disk["DRAW"]),
                 "battles_played": played, "battles_won": won, "battles_drawn": drawn,
                 "battles_lost": max(played - won - drawn, 0),
                 "manifest_wins": claim_w, "manifest_losses": claim_l,
                 "manifest_draws": claim_d}
        if (claim_w, claim_l, claim_d) != (entry["wins"], entry["losses"], entry["draws"]):
            mismatches.append(
                f"{opp}: manifest claims {claim_w}/{claim_l}/{claim_d} traced W/L/D, disk holds "
                f"{entry['wins']}/{entry['losses']}/{entry['draws']} — the DISK is matched on")
        opponents[opp] = entry
    cap = (max((o["wins"] for o in opponents.values()), default=0),
           max((o["losses"] for o in opponents.values()), default=0),
           max((o["draws"] for o in opponents.values()), default=0))
    n_traced = sum(o["wins"] + o["losses"] + o["draws"] for o in opponents.values())
    return Profile(run=run, step=int(step), trace_dir=os.path.abspath(trace_dir),
                   opponents=opponents, cap=cap, n_traced=n_traced,
                   disk_mismatches=mismatches)


# --------------------------------------------------------------------------- the plan

def plan(arm: Profile, ctl: Profile) -> Dict[str, Any]:
    """Which side (if any) is the RICHER one, and to what per-class caps it must be cut.

    The matched caps are the ELEMENTWISE MINIMUM of the two sides' caps, and a side is subsampled
    only where its own cap exceeds that minimum — which makes the rule symmetric: a control
    re-traced at the higher quota is the side that gets cut, and a pair that is richer on
    different classes has both sides cut.
    """
    caps = tuple(min(a, c) for a, c in zip(arm.cap, ctl.cap))
    # the DRAW class never reaches a conditioning row (extract_cycle drops non-WIN/LOSS battles),
    # so it is applied but never used to decide that matching is needed.
    sides = [name for name, p in (("arm", arm), ("control", ctl))
             if p.cap[0] > caps[0] or p.cap[1] > caps[1]]
    notes: List[str] = []
    if not sides:
        skew = (max(arm.n_traced, ctl.n_traced) / max(1, min(arm.n_traced, ctl.n_traced))) - 1.0
        if skew > TOTAL_SKEW_REPORT:
            notes.append(
                f"⚠️ the two caps are EQUAL ({arm.caps_str()}) but the traced totals differ by "
                f"{skew * 100:.0f}% ({arm.n_traced} vs {ctl.n_traced}) — the quota is not what "
                "shaped this difference, so no subsample can equalise it. REPORTED, not matched.")
    for name, p in (("arm", arm), ("control", ctl)):
        if p.disk_mismatches:
            notes.extend(f"{name}: {m}" for m in p.disk_mismatches)
    return {
        "needed": bool(sides), "caps": tuple(int(c) for c in caps), "sides": sides,
        "arm_cap": list(arm.cap), "control_cap": list(ctl.cap),
        "arm_traced": arm.n_traced, "control_traced": ctl.n_traced,
        "why": (f"the realized per-opponent caps differ — arm {arm.caps_str()}, control "
                f"{ctl.caps_str()} (traced W/L/D per opponent, counted on disk); "
                f"{' and '.join(sides)} cut to {'/'.join(str(c) for c in caps)}"
                if sides else
                f"both sides carry the SAME realized cap ({arm.caps_str()}) — the frames are "
                "already equalised and nothing is subsampled"),
        "notes": notes,
    }


# --------------------------------------------------------------------------- the subsample

def subsample(arr: np.ndarray, meta: Dict[str, Any], caps: Sequence[int], *,
              seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cut one cycle's extracted frame to at most ``caps`` traced (win, loss, draw) per opponent.

    IN MEMORY: the returned array is a view's copy of ``arr``'s kept rows with the Horvitz-Thompson
    weight ``w`` RECOMPUTED for the subsample — ``capture_rate_win = kept_wins / battles_won`` and
    likewise for losses, against the manifest's own denominators, exactly as the trainer computes
    them. Without that recomputation every downstream row would be reweighted for a tree that is no
    longer there, which is rule 17 violated in the other direction.

    Draws are not present in ``arr`` at all, so ``caps[2]`` never changes a number; it is carried
    so the applied caps and the reported profile are one object.
    """
    cap_w, cap_l = int(caps[0]), int(caps[1])
    names, first = np.unique(arr["battle"], return_index=True)
    b_opp = arr["opponent"][first]
    b_y = arr["y"][first]
    per_opp = meta.get("opponents") or {}
    order = sorted(per_opp) or sorted(set(b_opp.tolist()))

    keep_names: List[str] = []
    kept_counts: Dict[str, Dict[str, int]] = {}
    for i, opp in enumerate(order):
        rng = np.random.default_rng([seed, i, SALT])
        picked: Dict[str, List[str]] = {}
        for cls, cap, want in (("wins", cap_w, 1.0), ("losses", cap_l, 0.0)):
            pool = sorted(names[(b_opp == opp) & (b_y == want)].tolist())
            picked[cls] = (pool if len(pool) <= cap
                           else sorted(rng.choice(np.array(pool), size=cap,
                                                  replace=False).tolist()))
            keep_names.extend(picked[cls])
        kept_counts[opp] = {"wins": len(picked["wins"]), "losses": len(picked["losses"])}

    keep = np.isin(arr["battle"], np.array(sorted(keep_names), dtype=arr["battle"].dtype))
    out = arr[keep].copy()

    # ---- recomputed capture rates, and the weights that follow from them
    new_opps: Dict[str, Any] = {}
    w_of: Dict[Tuple[str, float], float] = {}
    for opp, rec in per_opp.items():
        kept = kept_counts.get(opp, {"wins": 0, "losses": 0})
        won = int(rec.get("battles_won", 0))
        played = int(rec.get("battles_played", 0))
        drawn = int(rec.get("battles_drawn", 0))
        lost = max(played - won - drawn, 0)
        crw = (kept["wins"] / won) if (won > 0 and kept["wins"]) else None
        crl = (kept["losses"] / lost) if (lost > 0 and kept["losses"]) else None
        new_opps[opp] = {**rec, "capture_rate_win": crw, "capture_rate_loss": crl,
                         "battles_loaded": kept["wins"] + kept["losses"],
                         "traces_won": kept["wins"],
                         "traces_written": kept["wins"] + kept["losses"]}
        if crw:
            w_of[(opp, 1.0)] = 1.0 / crw
        if crl:
            w_of[(opp, 0.0)] = 1.0 / crl
    if out.size:
        out["w"] = np.array([w_of.get((o, y), np.nan)
                             for o, y in zip(out["opponent"].tolist(), out["y"].tolist())])
        good = np.isfinite(out["w"])
        out = out[good]

    new_meta = dict(meta)
    new_meta["opponents"] = new_opps
    new_meta.update(n_states=int(out.size),
                    n_battles=int(np.unique(out["battle"]).size) if out.size else 0,
                    n_opponents=int(np.unique(out["opponent"]).size) if out.size else 0,
                    n_teams=int(np.unique(out["team"]).size) if out.size else 0)
    new_meta["subsample"] = {"caps": [int(c) for c in caps], "seed": int(seed),
                             "kept": kept_counts, "source": meta.get("trace_dir")}
    return out, new_meta


def decoder_battles(arr: np.ndarray, *, seed: int = 0) -> int:
    """How many battles the ``cond.own_team_r2.t1`` decoder would be FIT on for this frame.

    The same three steps :func:`~main.ops.conditioning_meters.conditioning_block` takes before it
    fits anything — the leave-one-battle-out team label (which needs ``MIN_TEAM_BATTLES``), the
    turn-1 mask, and the per-battle state cap — and none of the fitting, so a candidate rung can be
    SIZED for the price of a few array operations.
    """
    if arr.size == 0:
        return 0
    b = CM.rollup(arr)
    y = CM.loo_team_wr(b["team"], b["y"], b["w"])[b["_state_battle_inv"]]
    mask = (arr["turn"] == 1) & np.isfinite(y)
    idx = CM._cap_states(arr, mask, CM.STATES_PER_BATTLE_CAP, seed)
    return int(np.unique(arr["battle"][idx]).size) if idx.size else 0


def decoder_matched_caps(arr: np.ndarray, meta: Dict[str, Any], base_caps: Sequence[int],
                         target: int) -> Tuple[Tuple[int, int, int], int]:
    """Caps whose own-team DECODER frame is as close as possible to ``target`` battles.

    Battle-matching is not decoder-matching: ``MIN_TEAM_BATTLES = 4`` makes the decoder frame a
    nonlinear function of team diversity, so the side carrying more distinct teams per battle
    under-fills its decoder at equal battle counts and battle-matching runs AGAINST it. The rung
    is found by SEARCH rather than by scaling, because the count is superlinear in the caps (more
    battles put more teams over the threshold as well as more battles per team).
    """
    best: Optional[Tuple[int, Tuple[int, int, int], int]] = None
    for scale in DECODER_SCALE_GRID:
        caps = (max(1, int(round(base_caps[0] * scale))),
                max(1, int(round(base_caps[1] * scale))),
                int(base_caps[2]))
        got = int(np.median([decoder_battles(subsample(arr, meta, caps, seed=s)[0])
                             for s in DECODER_PROBE_SEEDS]))
        cand = (abs(got - target), caps, got)
        if best is None or cand[0] < best[0]:
            best = cand
        if got >= target:
            break
    assert best is not None
    return best[1], best[2]


# --------------------------------------------------------------------------- the matched read

def _q(points: Sequence[float]) -> List[float]:
    a = np.asarray([p for p in points if np.isfinite(p)], dtype=float)
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.percentile(a, SPREAD_Q)]


def rung(run_dir: str, step: int, arr: np.ndarray, meta: Dict[str, Any],
         caps: Sequence[int], *, keys: Sequence[str], seeds: int, boot: int, block_seed: int,
         say: Callable[[str], None] = lambda _m: None) -> Dict[str, Any]:
    """One matched rung: ``seeds`` subsamples scanned for their POINTS, then the median seed of
    each row re-read WITH its battle-clustered bootstrap.

    Two passes, because they cost differently. A point costs no bootstrap; the interval printed
    beside it must be the registered ``--cond-boot`` one, and only the seed whose point IS the
    reported median needs it. Scanning 21 seeds and bootstrapping the two or three seeds the rows
    actually land on keeps a whole pair inside a few seconds of CPU.

    The strength axis is DISABLED for a subsample (``ladder="off"``): ``cond.elo_slope`` is not
    frame-sensitive and is never read from here, and a snapshot-ladder refit per seed would
    dominate the cost of the entire read.
    """
    frames: List[Dict[str, Any]] = []
    points: Dict[str, List[float]] = {k: [] for k in keys}
    for s in range(int(seeds)):
        sarr, smeta = subsample(arr, meta, caps, seed=s)
        blk = CM.conditioning_block(run_dir, step, boot=0, seed=block_seed, ladder="off",
                                    frame=(sarr, smeta))
        for k in keys:
            points[k].append(float(blk["points"].get(k, float("nan"))))
        frames.append({"seed": s, "n_battles": blk["frame"]["n_battles"],
                       "n_teams": blk["frame"]["n_teams"],
                       "n_states": blk["frame"]["n_states"],
                       "decoder_battles": blk["frame"]["score_frames"].get(
                           "cond.own_team_r2.t1", {}).get("n_battles", 0)})

    #: the median SEED of each row — an order statistic, so the point printed and the interval
    #: printed beside it describe the same draw.
    median_seed: Dict[str, int] = {}
    for k in keys:
        arr_pts = np.asarray(points[k], dtype=float)
        finite = np.where(np.isfinite(arr_pts))[0]
        if finite.size == 0:
            continue
        median_seed[k] = int(finite[np.argsort(arr_pts[finite])[finite.size // 2]])

    boots: Dict[int, Dict[str, Any]] = {}
    for s in sorted(set(median_seed.values())):
        sarr, smeta = subsample(arr, meta, caps, seed=s)
        boots[s] = CM.conditioning_block(run_dir, step, boot=int(boot), seed=block_seed,
                                         ladder="off", frame=(sarr, smeta))
    say(f"  rung caps {'/'.join(str(c) for c in caps)}: {seeds} subsample seeds scanned, "
        f"{len(boots)} bootstrapped (median battles "
        f"{int(np.median([f['n_battles'] for f in frames]))}, median decoder battles "
        f"{int(np.median([f['decoder_battles'] for f in frames]))})")

    rows: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        if k not in median_seed:
            continue
        s = median_seed[k]
        blk = boots[s]
        rows[k] = {"point": float(blk["points"].get(k, float("nan"))),
                   "spread": _q(points[k]),
                   "median_seed": s, "points": points[k],
                   "_draws": blk["_draws"].get(k, np.empty(0))}
    return {"caps": [int(c) for c in caps], "n_seeds": int(seeds), "rows": rows,
            "frames": frames,
            "median_battles": float(np.median([f["n_battles"] for f in frames])),
            "median_teams": float(np.median([f["n_teams"] for f in frames])),
            "median_decoder_battles": float(np.median([f["decoder_battles"] for f in frames]))}


def match(*, arm_profile: Profile, control_profile: Profile,
          load_frame: Callable[[str], Tuple[np.ndarray, Dict[str, Any]]],
          run_dirs: Dict[str, str], steps: Dict[str, int],
          seeds: int = DEFAULT_SEEDS, boot: int = CM.N_BOOT, block_seed: int = 0,
          keys: Sequence[str] = CM.FRAME_SENSITIVE_KEYS,
          say: Callable[[str], None] = lambda _m: None) -> Dict[str, Any]:
    """The whole equalisation for one pair: the plan, then both rungs on each richer side.

    ``load_frame(side)`` returns that side's extracted ``(arr, meta)``; it is a callback so the
    richer side's trace tree is read exactly once and the poorer side's is never read at all.
    """
    p = plan(arm_profile, control_profile)
    out: Dict[str, Any] = {"plan": p, "profiles": {
        "arm": _profile_doc(arm_profile), "control": _profile_doc(control_profile)},
        "rungs": {}, "seeds": int(seeds), "boot": int(boot), "block_seed": int(block_seed),
        "keys": list(keys),
        # reaching :func:`match` at all means matching is ENABLED — the opt-out never gets here.
        # Set on the document itself so a caller that composes it directly (a test, a measurement
        # script) renders the same header the CLI does.
        "enabled": True}
    if not p["needed"]:
        return out
    target = {"arm": control_profile, "control": arm_profile}
    for side in p["sides"]:
        say(f"quota-match: subsampling the {side} to caps "
            f"{'/'.join(str(c) for c in p['caps'])} over {seeds} seeds")
        arr, meta = load_frame(side)
        rungs: Dict[str, Any] = {}
        rungs["battle"] = rung(run_dirs[side], steps[side], arr, meta, p["caps"],
                               keys=keys, seeds=seeds, boot=boot, block_seed=block_seed, say=say)
        poorer_arr, poorer_meta = load_frame("control" if side == "arm" else "arm")
        want = decoder_battles(poorer_arr)
        caps2, got = decoder_matched_caps(arr, meta, p["caps"], want)
        rungs["decoder"] = rung(run_dirs[side], steps[side], arr, meta, caps2,
                                keys=keys, seeds=seeds, boot=boot, block_seed=block_seed, say=say)
        rungs["decoder"]["target_decoder_battles"] = int(want)
        rungs["decoder"]["search_decoder_battles"] = int(got)
        rungs["poorer_decoder_battles"] = int(want)
        out["rungs"][side] = rungs
        _ = target  # the poorer side's profile is already in `profiles`
    return out


def _profile_doc(p: Profile) -> Dict[str, Any]:
    w, ll, d = p.totals()
    return {"run": p.run, "step": p.step, "trace_dir": p.trace_dir,
            "cap": list(p.cap), "n_traced": p.n_traced,
            "totals": {"wins": w, "losses": ll, "draws": d},
            "opponents": p.opponents, "disk_mismatches": p.disk_mismatches}


# --------------------------------------------------------------- the critic_read entry point

def serialisable(qm: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The quota-match document minus every raw bootstrap draw array."""
    if qm is None:
        return None
    out = {k: v for k, v in qm.items() if k != "rungs"}
    out["rungs"] = {
        side: {rn: ({"caps": r["caps"], "n_seeds": r["n_seeds"], "frames": r["frames"],
                     "median_battles": r["median_battles"], "median_teams": r["median_teams"],
                     "median_decoder_battles": r["median_decoder_battles"],
                     "target_decoder_battles": r.get("target_decoder_battles"),
                     "rows": {k: {kk: vv for kk, vv in row.items() if kk != "_draws"}
                              for k, row in r["rows"].items()}}
                    if isinstance(r, dict) and "rows" in r else r)
               for rn, r in rungs.items()}
        for side, rungs in (qm.get("rungs") or {}).items()}
    return out


def build_quota_match(arm: Dict[str, Any], ctl: Dict[str, Any], args, *,
                      say) -> Optional[Dict[str, Any]]:
    """The pair's frame-equalisation document, or ``None`` when there is no conditioning to match.

    The PLAN is always computed — it costs one pass over each side's ``_summary.json`` files — so
    even ``--no-quota-match`` can print the two realized capture profiles and mark the rows it
    refuses to label. Only when the plan says the frames are UNEQUAL, and matching is enabled, is
    the richer side's cycle extracted and subsampled.

    🚨 Nothing here reads or writes anything under ``models/`` beyond the same trace files the
    conditioning block already opens read-only, and every subsample lives in memory.
    """
    if arm.get("conditioning") is None or ctl.get("conditioning") is None:
        return None
    if not (arm["conditioning"].get("points") and ctl["conditioning"].get("points")):
        return None
    profiles = {role: realized_profile(d["run"], d["step"], d["trace_dir"])
                for role, d in (("arm", arm), ("control", ctl))}
    for role, pr in profiles.items():
        w, ll, dr = pr.totals()
        say(f"PROFILE {role:8s} {pr.run} step_{pr.step}: cap {pr.caps_str()} traced W/L/D per "
            f"opponent, {pr.n_traced} traced battles ({w}W / {ll}L / {dr}D over "
            f"{len(pr.opponents)} opponents)")
    pln = plan(profiles["arm"], profiles["control"])
    say(f"quota-match: {pln['why']}")
    for note in pln["notes"]:
        say(f"  {note}")
    if not pln["needed"] or args.no_quota_match:
        doc = {"plan": pln, "rungs": {}, "enabled": not args.no_quota_match,
               "profiles": {r: _profile_doc(pr) for r, pr in profiles.items()},
               "seeds": int(args.quota_match_seeds),
               "boot": int(args.quota_match_boot or args.cond_boot),
               "keys": list(CM.FRAME_SENSITIVE_KEYS)}
        if pln["needed"]:
            say("🚨 --no-quota-match: the FRAME-SENSITIVE conditioning rows will be printed "
                "with an UNMATCHED marker and NO label.")
        return doc
    cache: Dict[str, Any] = {}

    def load_frame(side: str):
        if side not in cache:
            d = arm if side == "arm" else ctl
            cache[side] = CM.extract_cycle(d["trace_dir"])
        return cache[side]

    t0 = time.time()
    doc = match(arm_profile=profiles["arm"], control_profile=profiles["control"],
                   load_frame=load_frame,
                   run_dirs={"arm": arm["run_dir"], "control": ctl["run_dir"]},
                   steps={"arm": arm["step"], "control": ctl["step"]},
                   seeds=int(args.quota_match_seeds),
                   boot=int(args.quota_match_boot or args.cond_boot),
                   block_seed=int(args.seed), say=say)
    doc["enabled"] = True
    doc["elapsed_sec"] = round(time.time() - t0, 1)
    say(f"quota-match: done in {doc['elapsed_sec']} s")
    return doc
