"""Extract the bf16 compiled arm's log-prob SHIFT per checkpoint from the T36 GPU diagnostic's output.

    python canary_shift_extract.py ~/.cache/gen3ai/diag_bf16_canary/87e493c35e6a > canary_shift_87e493c3.json

For every ``gpu_<tag>.json`` the diagnostic wrote (`main.ops.diag_bf16_canary gpu`, rb_toy_ent_s4001's checkpoints),
per fixture slice and per arm (Ebf = eager bf16, Cbf = compiled bf16, C32 = compiled fp32), against the fp32 eager
reference E32:
* ``taken``  = ``forward_vs_E32.<arm>.log_prob_max_abs``   — the largest |Δ log π| of the TAKEN action. This is the
  quantity `compile_regions._clip_margin_rule` judges (the PPO clip reads only the taken action's ratio) and the
  quantity ``R1_LOGPROB_SHIFT_MAX`` (0.18) bounds.
* ``legal``  = ``forward_vs_E32.<arm>.legal_logp_max_abs`` — the largest over EVERY legal action (a superset; never
  smaller than ``taken``). Descriptive only.
Pure read: no torch, no GPU.
"""
from __future__ import annotations

import json
import os
import re
import sys

ARMS = ("Ebf", "Cbf", "C32")


def main(d: str) -> dict:
    out = {"source": os.path.abspath(d), "rows": []}
    tags = sorted((m.group(1) for m in (re.match(r"gpu_(.+)\.json$", n) for n in os.listdir(d)) if m
                   and m.group(1) not in ("meta", "status")),
                  key=lambda t: (t == "fatal", float(t[:-1]) if t[:-1].replace(".", "").isdigit() else 0.0))
    for tag in tags:
        doc = json.load(open(os.path.join(d, f"gpu_{tag}.json")))
        for sl, s in sorted(doc.get("slices", {}).items()):
            fv = s.get("forward_vs_E32", {})
            row = {"checkpoint": tag, "slice": sl}
            for a in ARMS:
                if a in fv:
                    row[f"{a}_taken"] = fv[a].get("log_prob_max_abs")
                    row[f"{a}_legal"] = fv[a].get("legal_logp_max_abs")
            out["rows"].append(row)
    cbf = [r["Cbf_taken"] for r in out["rows"] if r.get("Cbf_taken") is not None]
    leg = [r["Cbf_legal"] for r in out["rows"] if r.get("Cbf_legal") is not None]
    out["Cbf_taken_max"] = max(cbf) if cbf else None
    out["Cbf_legal_max"] = max(leg) if leg else None
    return out


if __name__ == "__main__":
    print(json.dumps(main(os.path.expanduser(sys.argv[1])), indent=1))
