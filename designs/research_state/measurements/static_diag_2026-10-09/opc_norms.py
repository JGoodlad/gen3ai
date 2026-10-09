"""How much does each static final's OPC route (`op_content.amount_proj`, zero-init) carry? Column norms of its
weight per input cell [entry_chip, pursuit_p, pursuit_eff, grounded, leftovers, weather_chip, status_tick, leech],
beside the S/D networks' typical input-column norm. DESCRIPTIVE. Run at the pin (run_at_pin.sh opc_norms.py)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = Path("/home/goodlad/dev/gen3ai/models")
COLS = ["entry_chip", "pursuit_p", "pursuit_eff", "grounded", "leftovers", "weather_chip", "status_tick", "leech"]


def main():
    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    out = {"schema": "static_diag_opc_norms_v1", "tag": "DESCRIPTIVE", "columns": COLS, "runs": {}}
    with inference_globals(2):
        for s in range(1001, 1006):
            m = load_checkpoint(MODELS / f"rb_st_static_s{s}" / "final_model.zip")
            fe = m.policy.features_extractor
            oc = fe.op_content
            W = oc.amount_proj.weight.detach().numpy()                  # [128, 8]
            # reference: the D network's first layer, per input column (what a "used" input column looks like)
            dnet = [mod for name, mod in fe.named_modules() if name.endswith("dynamic_encoder.0")]
            ref = None
            if dnet:
                ref = float(np.median(np.linalg.norm(dnet[0].weight.detach().numpy(), axis=0)))
            out["runs"][s] = {"amount_proj_col_norm": dict(zip(COLS, np.linalg.norm(W, axis=0).round(4).tolist())),
                              "d_net_first_layer_median_col_norm": ref}
            del m
    (HERE / "opc_norms.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
