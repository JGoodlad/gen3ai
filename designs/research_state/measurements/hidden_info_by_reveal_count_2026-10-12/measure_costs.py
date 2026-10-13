"""The MEASURED CPU cost per state of the pilot's arms (for the amendment's cost plan): the sum of the rows' own `cost`
fields (CPU seconds of the playouts and the critic, as the driver measured them on the contended box) over the states
each archive directory holds, per checkpoint. World building is a separate file (`worlds_*.jsonl`) and is added when it
carries a CPU field.

    python -I measure_costs.py   →  costs.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = Path.home() / "gen3ai_archive" / "value_disc_pilot"


def rows(path_glob):
    out = {}
    for p in sorted(path_glob):
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[r["id"]] = r
    return out


res = {}
for cp in ("end2", "prod"):
    res[cp] = {}
    for name, pat in (("phase1_oracle", "rows/shard_*.jsonl"), ("det2_A", "det2/rows_*.jsonl"), ("det2_C", "det2_c/rows_*.jsonl"),
                      ("det2_CT", "det2_ct/rows_*.jsonl")):
        rs = rows((ARCH / cp).glob(pat))
        ok = [r for r in rs.values() if r.get("ok")]
        play = sum(r["cost"]["play"]["cpu_s"] for r in ok)
        crit = sum(r["cost"]["crit"]["cpu_s"] for r in ok)
        res[cp][name] = {"states": len(ok), "play_cpu_s_per_state": round(play / len(ok), 2),
                         "crit_cpu_s_per_state": round(crit / len(ok), 2), "total_cpu_s_per_state": round((play + crit) / len(ok), 2)}
    w = {}
    for p in sorted((ARCH / cp / "det2").glob("worlds_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                w[x["id"]] = x
    tot = [x["cpu_s"] for x in w.values() if isinstance(x.get("cpu_s"), (int, float))]
    res[cp]["det2_worlds"] = {"states": len(w), "world_build_cpu_s_per_state": round(sum(tot) / max(1, len(tot)), 2)}
(HERE / "costs.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
print(json.dumps(res, indent=1, sort_keys=True))
