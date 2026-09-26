"""POLICY DRIFT meter — is training REFINING the current strategy or ADOPTING a new one?

Extends the function-space churn probe (`churn_probe.py`: KL between two checkpoints on a FROZEN
probe-state set) from ONE pair into a per-snapshot SERIES with several references each:

  * references per snapshot — (a) the PREVIOUS processed snapshot, (b) the one about
    ``back_steps`` (default 10M) earlier, (c) a fixed ANCHOR (the first snapshot the meter recorded
    — snapshots only exist once the self-play pool is seeded — or one named on the CLI);
  * masked KL(current ‖ reference), mean + median;
  * GREEDY FLIP RATE bucketed by the OLDER policy's top-1 − top-2 probability margin: flips in
    low-margin states read as refinement, flips in confident states as a strategy change;
  * ACTION-MIX shares of the greedy choice by class (switch vs move; moves split into attack /
    status / setup / hazard / recovery / phazing / self_ko from the dex via `agents.gen3_data`),
    and each class's delta vs each reference;
  * a CYCLING indicator: the current policy is CLOSER to an older reference than the previous
    snapshot was.

🚨 It is a DESCRIPTOR, not a test. The thresholds in `verdict()` are reading aids chosen by
inspection, not calibrated against a null — no p-value, no replicate noise floor.

This module holds the PURE part (numpy + the dex; no model, no bridge) plus the resumable
per-snapshot driver, which takes its probability function INJECTED so the resume logic is unit-
tested without a model. The CLI and the model-loading glue live in `main/policy_drift.py`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from agents.action.constants import MOVE_END, MOVE_START, SWITCH_END

# ── action classes ────────────────────────────────────────────────────────────────────────────
MOVE_CLASSES: Tuple[str, ...] = ("attack", "status", "setup", "hazard", "recovery", "phazing", "self_ko")
CLASSES: Tuple[str, ...] = ("switch",) + MOVE_CLASSES + ("other",)   # other = struggle / unknown move
# The dex carries no self-faint flag (data/ has none; `turn_delta.SELF_KO_MOVES` curates the two
# boom moves the same way). Memento faints the user too, so it belongs here as a STRATEGY class.
SELF_KO_MOVE_IDS = frozenset({"explosion", "selfdestruct", "memento"})
# Curse is type-conditional (non-Ghost → +Atk/+Def setup). `is_boost` is False for it in the dex;
# every gen-3 OU Curse user of note (Snorlax, Regirock, …) is non-Ghost, so read it as setup.
_SETUP_OVERRIDE = frozenset({"curse"})

# ── flip-rate buckets on the OLDER policy's top-1 − top-2 margin ──────────────────────────────
MARGIN_EDGES: Tuple[float, ...] = (0.1, 0.3)

# ── verdict thresholds (reading aids, NOT calibrated) ─────────────────────────────────────────
SHIFT_ABS = 0.05          # a class share moved ≥ 5 pp vs the long reference …
CONF_FLIP_RATE = 0.15     # … or ≥ 15% of CONFIDENT (margin > last edge) states flipped vs it
CYCLE_REL_TOL = 0.05      # current KL to an old ref is ≥ 5% below the previous snapshot's …
CYCLE_ABS_TOL = 1e-3      # … and ≥ 0.001 nats below it


def classify_move(md) -> str:
    """One class per dex move record (`gen3_data.moves.MoveData`). Order matters: Explosion has
    base power 250, so the self-KO test runs before the damaging test."""
    if md.id in SELF_KO_MOVE_IDS:
        return "self_ko"
    if md.is_hazard:
        return "hazard"
    if md.is_phaze:
        return "phazing"
    if md.is_heal:
        return "recovery"
    if md.is_boost or md.self_boosts or md.id in _SETUP_OVERRIDE:
        return "setup"
    if md.base_power > 0:
        return "attack"
    return "status"


def move_class_by_num() -> Dict[int, str]:
    """dex move NUM → class — the obs's `active_req_moves` block carries move NUMs (Hidden Power
    of any type → its one num), and the dex's nums are unique."""
    from agents.gen3_data import moves
    out: Dict[int, str] = {}
    for mid in moves.raw():
        md = moves.get(mid)
        if md is not None and md.num > 0:
            out[md.num] = classify_move(md)
    return out


def req_move_nums(obs: np.ndarray, layout: dict) -> np.ndarray:
    """[N, 4] OUR active's move NUMs in REQUEST order (slot k ↔ action 6+k), read through the
    schema's validated slice map — never a hardcoded offset."""
    from agents.observation.schema import build_schema
    sl = build_schema(layout).slices()["reactive.active_req_moves"]
    per = (sl.stop - sl.start) // 3
    return np.rint(obs[:, sl.start: sl.start + per]).astype(np.int64)


def action_classes(actions: np.ndarray, move_nums: np.ndarray, table: Dict[int, str]) -> List[str]:
    """Class of each chosen action: 0..5 → switch; 6..9 → the class of that request slot's move;
    anything else (struggle) or an unknown/empty move → "other"."""
    out = []
    for a, nums in zip(np.asarray(actions).tolist(), move_nums):
        if a < SWITCH_END:
            out.append("switch")
        elif MOVE_START <= a < MOVE_END:
            out.append(table.get(int(nums[a - MOVE_START]), "other"))
        else:
            out.append("other")
    return out


def class_shares(classes: Sequence[str]) -> Dict[str, float]:
    n = len(classes)
    return {c: (sum(1 for x in classes if x == c) / n if n else 0.0) for c in CLASSES}


def share_deltas(cur: Dict[str, float], ref: Dict[str, float]) -> Dict[str, float]:
    return {c: cur.get(c, 0.0) - ref.get(c, 0.0) for c in CLASSES}


# ── greedy / margin / flips ───────────────────────────────────────────────────────────────────
def greedy_actions(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0.5, p, -1.0).argmax(-1)


def top2_margin(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """top-1 minus top-2 probability over LEGAL actions ([N]); 1.0 where only one is legal."""
    q = np.sort(np.where(mask > 0.5, p, 0.0), axis=-1)
    return q[:, -1] - q[:, -2]


def bucket_labels(edges: Sequence[float] = MARGIN_EDGES) -> List[str]:
    e = list(edges)
    return [f"<{e[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(e, e[1:])] + [f">{e[-1]:g}"]


def flip_by_margin(p_old: np.ndarray, p_new: np.ndarray, mask: np.ndarray,
                   edges: Sequence[float] = MARGIN_EDGES) -> dict:
    """Share of states whose greedy action changed, bucketed by the OLDER policy's margin. States
    with fewer than two legal actions cannot flip and are excluded (they would dilute every bucket).
    A margin exactly on an edge goes to the HIGHER bucket."""
    choice = (mask > 0.5).sum(-1) >= 2
    flips = greedy_actions(p_old, mask) != greedy_actions(p_new, mask)
    idx = np.digitize(top2_margin(p_old, mask), list(edges), right=False)
    buckets = {}
    for b, lab in enumerate(bucket_labels(edges)):
        sel = choice & (idx == b)
        n = int(sel.sum())
        k = int((flips & sel).sum())
        buckets[lab] = {"n": n, "flips": k, "rate": (k / n) if n else None}
    n = int(choice.sum())
    k = int((flips & choice).sum())
    return {"n": n, "flips": k, "rate": (k / n) if n else None, "buckets": buckets}


def masked_kl(pa: np.ndarray, pb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from agents.training.churn_probe import masked_kl as _kl
    return _kl(pa, pb, mask)


# ── references + cycling ──────────────────────────────────────────────────────────────────────
def pick_prev(steps: Iterable[int], cur: int) -> Optional[int]:
    older = [s for s in steps if s < cur]
    return max(older) if older else None


def pick_back_ref(steps: Iterable[int], cur: int, back: int) -> Optional[int]:
    """The LATEST recorded step at least ``back`` steps before ``cur`` (None if none is that old)."""
    older = [s for s in steps if s <= cur - back]
    return max(older) if older else None


def cycling_refs(kl_cur: Dict[str, Optional[float]], kl_prev: Dict[str, Optional[float]],
                 rel_tol: float = CYCLE_REL_TOL, abs_tol: float = CYCLE_ABS_TOL) -> List[str]:
    """References the CURRENT policy is closer to than the PREVIOUS snapshot was (both KLs measured
    against the SAME reference). Moving back toward an older self is the self-play cycling
    signature; moving away (the normal case) or a within-tolerance wobble flags nothing."""
    out = []
    for name, cur in kl_cur.items():
        prev = kl_prev.get(name)
        if cur is None or prev is None:
            continue
        if cur < prev * (1.0 - rel_tol) and cur < prev - abs_tol:
            out.append(name)
    return out


def verdict(row: dict) -> str:
    """"cycling?" > "shifting (…)" > "refining". The LONG reference is back (≈10M earlier) when one
    exists, else the anchor, else the previous snapshot. A class counts as shifted when its share
    moved ≥ SHIFT_ABS vs the long reference AND the step vs the previous snapshot does not point the
    other way (coherent, not a one-snapshot reversal)."""
    if row.get("status") != "ok":
        return row.get("status", "?")
    if row.get("cycling"):
        return "cycling? (closer to " + ", ".join(row["cycling"]) + ")"
    refs = row.get("refs", {})
    long_ref = next((refs[k] for k in ("back", "anchor", "prev") if refs.get(k)), None)
    if long_ref is None:
        return "first (no reference)"
    prev = refs.get("prev") or {}
    shifted = []
    for c, d in long_ref["share_delta"].items():
        dp = (prev.get("share_delta") or {}).get(c, 0.0)
        if abs(d) >= SHIFT_ABS and not (dp * d < 0 and abs(dp) >= SHIFT_ABS / 2):
            shifted.append((abs(d), f"{c} {d * 100:+.0f}pp"))
    conf = long_ref["flip"]["buckets"].get(bucket_labels()[-1], {})
    if conf.get("rate") is not None and conf["rate"] >= CONF_FLIP_RATE:
        shifted.append((conf["rate"], f"confident flips {conf['rate'] * 100:.0f}%"))
    if shifted:
        return "shifting (" + ", ".join(t for _, t in sorted(shifted, reverse=True)) + ")"
    return "refining"


# ── the per-snapshot unit (resumable, append-only rows) ───────────────────────────────────────
ANCHOR_FILE = "anchor.npz"      # probs + step of an external anchor, written once by the CLI
_SNAP_RE = re.compile(r"^snapshot_(\d+)\.zip$")
_CKPT_RE = re.compile(r"^checkpoint_(\d+)_steps\.zip$")


def list_run_models(run_dir: Path, source: str = "snapshots") -> Dict[int, Path]:
    """step → zip for a run's pool snapshots and/or periodic checkpoints (read-only listing)."""
    out: Dict[int, Path] = {}
    dirs = {"snapshots": [("snapshots", _SNAP_RE)], "checkpoints": [("checkpoints", _CKPT_RE)],
            "both": [("snapshots", _SNAP_RE), ("checkpoints", _CKPT_RE)]}[source]
    for sub, rx in dirs:
        d = run_dir / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            m = rx.match(f.name)
            if m:
                out.setdefault(int(m.group(1)), f)
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(out_dir: Path) -> List[dict]:
    p = out_dir / "rows.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a torn last line from a kill mid-write is dropped, then redone
    return rows


def append_row(out_dir: Path, row: dict) -> None:
    p = out_dir / "rows.jsonl"
    torn = False
    if p.exists() and p.stat().st_size:
        with open(p, "rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"         # a kill mid-write left a partial line: fence it off
    with open(p, "a") as f:
        f.write(("\n" if torn else "") + json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _probs_path(out_dir: Path, step: int) -> Path:
    return out_dir / "probs" / f"{step:012d}.npz"


def cached_probs(out_dir: Path) -> Dict[int, Path]:
    d = out_dir / "probs"
    if not d.is_dir():
        return {}
    return {int(f.stem): f for f in d.glob("*.npz") if f.stem.isdigit()}


def _ref_block(p_cur, p_ref, mask, ref_step, cls_cur, cls_ref_shares) -> dict:
    kl = masked_kl(p_cur, p_ref, mask)
    return {
        "step": int(ref_step),
        "kl_mean": float(np.mean(kl)),
        "kl_median": float(np.median(kl)),
        "flip": flip_by_margin(p_ref, p_cur, mask),
        "share_delta": share_deltas(cls_cur, cls_ref_shares),
    }


def _shares_for(p, mask, move_nums, table, choice) -> Dict[str, float]:
    acts = greedy_actions(p, mask)[choice]
    return class_shares(action_classes(acts, move_nums[choice], table))


def process_pending(run_dir: Path, out_dir: Path, obs: np.ndarray, mask: np.ndarray,
                    move_nums: np.ndarray, probs_fn: Callable[[Path], np.ndarray], *,
                    source: str = "snapshots", back_steps: int = 10_000_000,
                    anchor_step: Optional[int] = None, settle_s: float = 60.0,
                    table: Optional[Dict[int, str]] = None, log=print) -> List[dict]:
    """Process every model in ``run_dir`` that has no row yet, oldest first; one durable row each.

    RESUMABLE at two grains: a step with a row is skipped outright; a step whose action
    probabilities are already cached under ``out_dir/probs`` (the row write was interrupted) is not
    re-forwarded. The cached probabilities are also what makes pruned snapshots usable as
    references — the pool is a sliding window, so a 10M-old zip is usually gone from disk.
    ``probs_fn(zip) → [N, A]`` is injected (the CLI passes the model forward). Never writes under
    ``run_dir``. A zip modified within ``settle_s`` is left for the next poll (may be mid-write)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probs").mkdir(exist_ok=True)
    table = table if table is not None else move_class_by_num()
    done = {int(r["step"]) for r in load_rows(out_dir)}
    avail = list_run_models(run_dir, source)
    choice = ((mask[:, :SWITCH_END] > 0.5).any(-1) & (mask[:, MOVE_START:MOVE_END] > 0.5).any(-1))
    new_rows = []
    for step in sorted(s for s in avail if s not in done):
        zp = avail[step]
        pp = _probs_path(out_dir, step)
        if not pp.exists():
            try:
                if time.time() - zp.stat().st_mtime < settle_s:
                    log(f"[policy_drift] {zp.name} still settling; next poll")
                    continue
                p = np.asarray(probs_fn(zp), np.float32)
            except FileNotFoundError:
                row = {"step": step, "ckpt": str(zp), "status": "pruned before read"}
                append_row(out_dir, row); new_rows.append(row)
                continue
            except Exception as e:                       # noqa: BLE001 — recorded, not swallowed
                row = {"step": step, "ckpt": str(zp), "status": "error",
                       "error": f"{type(e).__name__}: {e}"[:2000]}
                append_row(out_dir, row); new_rows.append(row)
                log(f"[policy_drift] ERROR on {zp.name}: {row['error']}")
                continue
            if p.shape != mask.shape:
                raise ValueError(f"probs {p.shape} != mask {mask.shape} for {zp}")
            tmp = pp.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, probs=p)
            os.replace(tmp, pp)
        cache = cached_probs(out_dir)
        load = lambda s: np.load(cache[s])["probs"]    # noqa: E731
        p_cur = load(step)
        older = [s for s in cache if s < step]
        ext = out_dir / ANCHOR_FILE             # an EXTERNAL anchor (--anchor <zip>) wins
        if ext.exists():
            with np.load(ext) as z:
                anc = int(z["step"])
                ext_probs = z["probs"]
            load = lambda s, _l=load: ext_probs if s == anc else _l(s)   # noqa: E731
            anc_ok = anc < step
        else:
            anc = anchor_step if anchor_step is not None else (min(cache) if cache else None)
            anc_ok = anc is not None and anc < step and anc in cache
        refs_steps = {"prev": pick_prev(older, step),
                      "back": pick_back_ref(older, step, back_steps),
                      "anchor": anc if anc_ok else None}
        cls_cur = _shares_for(p_cur, mask, move_nums, table, choice)
        shares_cache: Dict[int, Dict[str, float]] = {}
        refs = {}
        kl_cur, kl_prev = {}, {}
        for name, rs in refs_steps.items():
            if rs is None:
                refs[name] = None
                continue
            p_ref = load(rs)
            if rs not in shares_cache:
                shares_cache[rs] = _shares_for(p_ref, mask, move_nums, table, choice)
            refs[name] = _ref_block(p_cur, p_ref, mask, rs, cls_cur, shares_cache[rs])
            prev = refs_steps["prev"]
            if name != "prev" and prev is not None and rs < prev:
                kl_cur[name] = refs[name]["kl_mean"]
                kl_prev[name] = float(np.mean(masked_kl(load(prev), p_ref, mask)))
        row = {
            "step": step, "ckpt": str(zp), "status": "ok", "processed_at": time.time(),
            "n_states": int(len(mask)), "n_choice_states": int(choice.sum()),
            "shares": cls_cur, "refs": refs,
            "kl_prev_vs_ref": kl_prev,
            "cycling": cycling_refs(kl_cur, kl_prev),
        }
        row["verdict"] = verdict(row)
        append_row(out_dir, row)
        new_rows.append(row)
        log(f"[policy_drift] step {step:,}: {row['verdict']}")
    return new_rows
