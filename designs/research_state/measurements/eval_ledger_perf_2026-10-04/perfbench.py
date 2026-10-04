"""F-ED-22: the ledger writer's and reader's cost as the archive grows. Synthesizes an archive of N requests-stream
events in the in-loop producers' shape (one cycle = 3 opens + 15 claims + 15 row events + 3 dones = 36 events, 15
rows in 3 regimes; ~40 cycles per writer file, ~1,440 events = one 75M run) by writing the streams DIRECTLY (a
claim-by-claim build is itself quadratic in the old code), then times, on a fresh ``LedgerWriter`` (a fresh trainer
process), the operations an eval cycle does.

    PYTHONPATH=src python perfbench.py --events 5100 --reps 5 --out result.json

Runs on the code BEFORE the fix and AFTER it (it uses only the public writer / reader API and the testkit).
``cold_first_op_ms`` is the first operation on an archive that has never been indexed (before the fix: a fold).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from agents.training import eval_ledger as L
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import testkit as K

HOST, OPPS, REGIMES = "benchhost", 15, 3
PER_CYCLE = 3 + OPPS + OPPS + 3
CYCLES_PER_FILE = 40


def synth(root: Path, n_events: int) -> dict:
    (root / "requests").mkdir(parents=True)
    (root / "rows" / "inloop").mkdir(parents=True)
    (root / "decisions").mkdir()
    (root / "references").mkdir()
    regs = [K.regime(turn_limit=100 + r) for r in range(REGIMES)]
    n_cycles = max(1, n_events // PER_CYCLE)
    t0 = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    seq, n_rows, files = 0, 0, 0
    ev_f = rows_f = None
    wid = ""

    def ev(event_kind: str, **kw) -> dict:
        nonlocal seq
        seq += 1
        e = {"schema": S.EVENT_SCHEMA, "event": event_kind, "seq": seq,
             "ts": (t0 + dt.timedelta(seconds=seq)).isoformat(timespec="seconds"), "writer_id": wid, **kw}
        ev_f.write(S.canonical(e) + "\n")
        return e

    row_k = 0
    for c in range(n_cycles):
        if c % CYCLES_PER_FILE == 0:
            for f in (ev_f, rows_f):
                if f:
                    f.close()
            files += 1
            wid = f"20260901T{files // 60:02d}{files % 60:02d}00Z.0-{HOST}-{20000 + files}-inloop"
            ev_f = open(root / "requests" / f"events.{wid}.jsonl", "w")
            rows_f = open(root / "rows" / "inloop" / f"ledger.{wid}.jsonl", "w")
            row_k = 0
        player = f"{c + 1:064x}"
        opens = []
        for r in range(REGIMES):
            opens.append(ev("open", request_id=f"run{files}:cycle:{c}:{regs[r]['regime_id']}", kind="cycle",
                            purpose="cycle", family=None, regime_id=regs[r]["regime_id"], protocol=K.PROTO,
                            spec={"c": c}))
        claims = []
        for o in range(OPPS):
            r, opp = o % REGIMES, f"{o + 1:064x}"
            unit = {"request_id": opens[r]["request_id"], "batch": 0, "player": player, "opponent": opp,
                    "regime_id": regs[r]["regime_id"]}
            ce = ev("claim", **unit, producer="inloop", host=HOST, pid=20000 + files,
                    expires_at=(t0 + dt.timedelta(days=1)).isoformat(timespec="seconds"))
            claims.append((ce, r, opp, unit))
        for ce, r, opp, unit in claims:
            row = K.make_row(f"{wid}:{row_k}", request=opens[r], batch=0, p=player, o=opp, reg=regs[r],
                             ts=(t0 + dt.timedelta(seconds=seq)).isoformat(timespec="seconds"), purpose="cycle",
                             cycle_seed=c * 1000 + int(opp[-2:], 16))
            rows_f.write(S.canonical(row) + "\n")
            row_k += 1
            n_rows += 1
            ev("row", **unit, claim_seq=ce["seq"], row_id=row["row_id"], seed_key=S.key_list(S.seed_key(row)))
        for o_ in opens:
            ev("done", request_id=o_["request_id"])
    for f in (ev_f, rows_f):
        f.close()
    return {"events": seq, "rows": n_rows, "event_files": files}


OWN = L.ReaderDecl(name="perfbench.own", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="own",
                   selection="include", flags_ok=frozenset(), inference="conditional")


def med(xs):
    return round(statistics.median(xs) * 1e3, 3)


def bench(root: Path, reps: int) -> dict:
    reg = K.regime(turn_limit=100)
    out: dict = {}
    t = time.perf_counter()
    w0 = L.LedgerWriter(root, producer="inloop")
    w0.open_request("bench:cold", kind="cycle", purpose="cycle", regime_id=reg["regime_id"], protocol=K.PROTO,
                    spec={})
    out["cold_first_op_ms"] = round((time.perf_counter() - t) * 1e3, 3)
    fresh, steady, appends, opens, dones, reads = [], [], [], [], [], []
    for rep in range(reps):
        w = L.LedgerWriter(root, producer="inloop")           # a fresh trainer process
        rid = f"bench:{rep}"
        t = time.perf_counter()
        req = w.open_request(rid, kind="cycle", purpose="cycle", regime_id=reg["regime_id"], protocol=K.PROTO,
                             spec={"rep": rep})
        opens.append(time.perf_counter() - t)
        claims = []
        for o in range(OPPS):
            opp = f"{0xB0000 + o:064x}"
            t = time.perf_counter()
            c = w.claim(rid, batch=0, player=f"{0xA0000 + rep:064x}", opponent=opp, regime_id=reg["regime_id"],
                        expected_wall_s=1.0)
            (fresh if o == 0 else steady).append(time.perf_counter() - t)
            claims.append((c, opp))
        for i, (c, opp) in enumerate(claims):
            row = K.make_row(w.next_row_id(), request=req, batch=0, p=f"{0xA0000 + rep:064x}", o=opp, reg=reg,
                             ts=S.utc_now(), purpose="cycle", cycle_seed=900000 + rep * 100 + i)
            t = time.perf_counter()
            w.append_row(row, c)
            appends.append(time.perf_counter() - t)
        t = time.perf_counter()
        w.finish_request(rid)
        dones.append(time.perf_counter() - t)
        t = time.perf_counter()
        got = L.read(OWN, root=root, request_id=rid)
        reads.append(time.perf_counter() - t)
        assert len(got) == OPPS, len(got)
        w.close()
    out.update({"fresh_first_claim_ms": med(fresh), "steady_claim_ms": med(steady), "append_ms": med(appends),
                "open_ms": med(opens), "done_ms": med(dones), "read_own_ms": med(reads), "reps": reps})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="ledgerperf_") as td:
        root = Path(td) / "ledger"
        shape = synth(root, a.events)
        res = {"label": a.label, **shape, **bench(root, a.reps)}
    print(json.dumps(res, sort_keys=True))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
