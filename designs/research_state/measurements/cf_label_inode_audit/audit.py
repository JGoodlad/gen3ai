"""CfLabelBuffer inode-reuse impact audit (boundary a7627744). READ-ONLY on models/.

Run: PYTHONPATH=src python designs/research_state/measurements/cf_label_inode_audit/audit.py > .../audit.json
"""
import glob
import json
import os
import re
import sys

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

sys.path.insert(0, "src")
from utils.paths import main_models_dir  # noqa: E402

M = str(main_models_dir()) + "/"
CF = ("cf/labels_ingested_total", "cf/labels_expired_total", "cf/labels_skipped_total",
      "cf/labels_replaced_total", "cf/rows_sampled", "train/cf_loss")


def scalars(f):
    out = {}
    for ev in EventFileLoader(f).Load():
        for v in ev.summary.value:
            if v.tag in CF:
                val = v.simple_value if v.HasField("simple_value") else (
                    v.tensor.float_val[0] if v.tensor.float_val else 0.0)
                out.setdefault(v.tag, []).append((ev.wall_time, ev.step, val))
    return out


report = {"boundary": "a7627744", "runs": {}, "config_live": []}
for r in sorted(os.listdir(M)):
    cfg = os.path.join(M, r, "model_config.json")
    if os.path.exists(cfg):
        try:
            d = json.load(open(cfg))
        except Exception:
            d = {}
        live = {k: d.get(k) for k in ("cf_winprob_coef", "cf_evidential_coef", "cf_twin_coef",
                                      "cf_shadow_coef") if d.get(k) not in (None, 0, 0.0)}
        if live:
            report["config_live"].append(r)
    if not (os.path.isdir(M + r + "/cf_labels") or os.path.isdir(M + r + "/cf_records")):
        continue
    labels = sorted(glob.glob(M + r + "/cf_labels/labels_*.jsonl"))
    files = [(os.stat(f).st_mtime, sum(1 for _ in open(f)), os.path.basename(f)) for f in labels]
    segs = []
    for f in sorted(glob.glob(M + r + "/tb/**/events*", recursive=True)):
        if "inherited" in f:
            continue
        s = scalars(f)
        ing = s.get("cf/labels_ingested_total")
        if not ing:
            continue
        i, e = ing[-1][2], s["cf/labels_expired_total"][-1][2]
        k = s.get("cf/labels_skipped_total", [(0, 0, 0.0)])[-1][2]
        disk = sum(n for m, n, _ in files if m <= ing[-1][0])
        segs.append({"tb": os.path.basename(f), "steps": [ing[0][1], ing[-1][1]],
                     "ingested": i, "expired": e, "skipped": k,
                     "rows_sampled_max": max(x[2] for x in s.get("cf/rows_sampled", [(0, 0, 0.0)])),
                     "cf_loss_points": len(s.get("train/cf_loss", [])),
                     "disk_rows_by_last_log": disk,
                     # parsed = ingested + expired_at_ingest + skipped;  expired_resident <= ingested
                     "parsed_lo": max(i, e) + k, "parsed_hi": i + e + k,
                     "max_unexplained_rows": max(0, disk - (max(i, e) + k))})
    if not segs or all(sg["ingested"] == 0 for sg in segs):
        if files or any(sg["ingested"] for sg in segs):
            pass
        report["runs"][r] = {"ever_ingested": False, "label_files": len(files)} if segs else None
        if report["runs"][r] is None:
            del report["runs"][r]
        continue
    names = [n for _, _, n in files]
    seqs = [int(re.search(r"_(\d+)\.jsonl$", n).group(1)) for n in names]
    st = {}
    if os.path.exists(M + r + "/cf_producer_state.json"):
        st = json.load(open(M + r + "/cf_producer_state.json"))
    report["runs"][r] = {
        "ever_ingested": True, "label_files": len(files), "rows_on_disk": sum(n for _, n, _ in files),
        "distinct_seqs": len(set(seqs)), "max_seq": max(seqs),
        "producer_state_seq": st.get("seq"), "producer_state_labels_total": st.get("labels_total"),
        "segments": segs,
    }
report["consumers"] = [r for r, v in report["runs"].items() if v and v["ever_ingested"]]
report["config_live_count"] = len(report["config_live"])
json.dump(report, sys.stdout, indent=1, default=str)
