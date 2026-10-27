"""G-ENC (F-1b-6; ``design_x50_rung1b_human_opponent_model.md`` §4): the HOM's training rows must be encoded byte for
byte as X26's pin encodes them. Re-asserted at THIS commit, before any training-date row is extracted, on R1's traces
(X26's eval traces at cycles >= 360M): the live reader's FULL read of the trainee's own stream (``extract.read_full``,
the reader every extraction runs) must equal the rows and masks the trainee played on (``*_states.npz``: ``obs``,
``action_mask``), bit for bit.

    python -m main.x50_r1.genc --run <main>/models/rb_x26_s2601 --min-step 360000000 --out <archive>/genc.json

The one tolerated difference is R1's: a trace whose ``states`` stop one row short of the read (a 250-turn draw's last
row is not recorded), the common rows equal. Anything else is a FAIL (exit 1): a defect, and nothing is extracted.

``--exclude <list.json>`` (``main.x50_r1.fr2_exclude``, X59-S Amendment 4c) skips the battles it names: counted as
``excluded``, and the list's path, sha256 and count are recorded.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_READER = None


def _reader():
    global _READER
    if _READER is None:
        from main.live.reader import LiveReader
        _READER = LiveReader()
    return _READER


def one(sp: str) -> Tuple[str, str]:
    """(verdict, detail) of one trace: ``equal`` / ``short_by_one`` / ``skipped_refused_choice`` / ``DIFF`` / ``ERROR``."""
    from main.live.replay_scan import battle_lines
    from main.x50_r1 import extract as E
    from main.x50_r1 import spectate as S

    try:
        p = Path(sp)
        ts = json.loads(p.read_text())["meta"]["trace_source"]
        d = p.parent
        seat = ts["trainee_side"]
        _, own = S.read_stream(d / ts["records"][seat])
        own = battle_lines("\n".join(own))
        if any(ln.startswith("|error|") for ln in own):
            return "skipped_refused_choice", sp
        rec = json.loads((d / ts["reconstruction"]).read_text())
        team = [json.loads(x.split(" ", 2)[2])["team"] for x in rec["input_log"] if x.startswith(f">player {seat}")][0]
        name = E.player_info(own).get(seat, {}).get("name") or seat
        full = E.read_full(_reader(), own, seat, name, team)
        st = np.load(d / ts["states"])
        obs, msk = st["obs"].astype(np.float32), st["action_mask"].astype(np.int8)
        rows = np.stack(full.rows).astype(np.float32) if full.rows else np.zeros((0, obs.shape[1]), np.float32)
        masks = np.stack(full.masks).astype(np.int8) if full.masks else np.zeros((0, 11), np.int8)
        n = min(len(rows), len(obs))
        same = np.array_equal(rows[:n].view(np.uint32), obs[:n].view(np.uint32)) and np.array_equal(masks[:n], msk[:n])
        if not same:
            k = next(i for i in range(n) if not (np.array_equal(rows[i].view(np.uint32), obs[i].view(np.uint32))
                                                  and np.array_equal(masks[i], msk[i])))
            return "DIFF", f"{sp}: row {k} of {n}"
        if len(rows) == len(obs):
            return "equal", sp
        if len(rows) == len(obs) + 1:
            return "short_by_one", sp
        return "DIFF", f"{sp}: {len(rows)} read rows, {len(obs)} played"
    except Exception as exc:                       # a read that cannot be made is a FAIL, never a skip
        return "ERROR", f"{sp}: {type(exc).__name__}: {str(exc)[:300]}"


def select(items: List[str], exclude: Optional[str]) -> Tuple[List[str], int, Optional[Dict[str, Any]]]:
    """(the traces to check, how many the list excluded, the list's record) — every trace when ``exclude`` is None."""
    if exclude is None:
        return items, 0, None
    from main.x50_r1 import fr2_exclude

    keys, sha = fr2_exclude.load(Path(exclude))
    kept = [sp for sp in items if fr2_exclude.norm(sp) not in keys]
    return kept, len(items) - len(kept), {"path": str(exclude), "sha256": sha, "count": len(keys)}


def main(argv: Optional[List[str]] = None) -> int:
    from main.x50_r1.run import trace_summaries
    from utils.git import get_git_hash

    ap = argparse.ArgumentParser(prog="python -m main.x50_r1.genc", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--exclude", default=None, help="a main.x50_r1.fr2_exclude list: its battles are skipped")
    a = ap.parse_args(argv)
    items, n_excluded, exclusion = select(trace_summaries(Path(a.run), a.min_step), a.exclude)
    with Pool(a.workers) as pool:
        res = pool.map(one, items, chunksize=8)
    c = collections.Counter(v for v, _ in res)
    if n_excluded:
        c["excluded"] = n_excluded
    bad = [d for v, d in res if v in ("DIFF", "ERROR")]
    body: Dict[str, Any] = {"gate": "G-ENC", "commit": get_git_hash(), "run": a.run, "min_step": a.min_step,
                            "traces": len(items) + n_excluded, "exclude": exclusion, "counts": dict(c), "failures": bad[:50],
                            "short_by_one": [d for v, d in res if v == "short_by_one"],
                            "verdict": "PASS" if not bad and c["equal"] > 0 else "FAIL"}
    Path(a.out).write_text(json.dumps(body, indent=1))
    print(f"[genc] {body['verdict']}: {dict(c)}", flush=True)
    return 0 if body["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
