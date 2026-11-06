"""Path check for the delivery gate: P*'s argmax through delivery._forward (the student's path) vs the recorded
pi_argmax, and the student's argmax vs P*'s offline argmax, on every label row of each read panel."""
import json
import sys
from pathlib import Path

import numpy as np

from main.x65 import bank as BK
from main.x65 import delivery as DV

A = Path("/home/goodlad/gen3ai_archive/x65_exit0_2026-11-05")
INIT = Path("/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip")
out = {}
for panel, stu in (("heldout72_hpfix", "s_h"), ("forum72", "s_f"), ("p72", "s_p")):
    z = BK.load_rows(A / "bank", [panel], DV.FIELDS, labels=[BK.LABELS["searched"]])
    lab = z["label"]
    obs, mask = z["obs"][lab], z["mask"][lab]
    p_arg = DV.masked_argmax(DV._forward(INIT, obs, mask), mask)
    zp = Path(json.loads((A / "students" / stu / "model" / "export.json").read_text())["zip"])
    s_arg = DV.masked_argmax(DV._forward(zp, obs, mask), mask)
    pia, played = z["pi_argmax"][lab].astype(int), z["played"][lab].astype(int)
    out[panel] = {"rows": int(lab.sum()),
                  "pstar_offline_eq_recorded_pi_argmax": float((p_arg == pia).mean()),
                  "pstar_offline_agree_played": float((p_arg == played).mean()),
                  "recorded_pi_argmax_agree_played": float((pia == played).mean()),
                  "student_eq_pstar_offline": float((s_arg == p_arg).mean()),
                  "student_agree_played": float((s_arg == played).mean())}
    print(panel, out[panel], flush=True)
(A / "delivery" / "forward_check.json").write_text(json.dumps(out, indent=1))
