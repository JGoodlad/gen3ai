"""F-ED-22: how long does one ledger CLAIM take as the archive's requests stream grows? Every claim (and every claimed
append) re-folds every events file under the lock. Builds a scratch ledger of <writers> x <cycles per writer> in-loop
shaped cycles (open, 15 claims, done) and prints the per-claim time of the last cycle.

    PYTHONPATH=src python designs/research_state/measurements/eval_ledger_u2_2026-10-04/foldbench.py 10 30
"""
import time, tempfile, os, sys
from pathlib import Path
from agents.training import eval_ledger as L
from agents.training.eval_ledger import testkit as K
root = Path(tempfile.mkdtemp())/"ledger"
reg = K.regime()
n_writers, cycles_per_writer, opps = int(sys.argv[1]), int(sys.argv[2]), 15
t_last = []
for w_i in range(n_writers):
    w = L.LedgerWriter(root, producer="inloop", pid=10_000+w_i)
    for c in range(cycles_per_writer):
        rid = f"run{w_i}:cycle:{c}"
        req = w.open_request(rid, kind="cycle", purpose="cycle", regime_id=reg["regime_id"], protocol=K.PROTO)
        t0 = time.perf_counter()
        for o in range(opps):
            w.claim(rid, batch=0, player=K.SHA_A, opponent=f"{o:064x}", regime_id=reg["regime_id"], expected_wall_s=1)
        t_last.append((time.perf_counter()-t0)/opps)
        w.finish_request(rid)
ev = sum(1 for _ in root.glob("requests/events.*"))
print(f"writers={n_writers} cycles={n_writers*cycles_per_writer} event_files={ev} last claim {t_last[-1]*1e3:.1f} ms; first {t_last[0]*1e3:.1f} ms")
