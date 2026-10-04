"""Build a 20-snapshot self-play pool SERVABLE at HEAD, from archived trained weights (profile setup only).

    python transplant_pool.py --template <HEAD seed snapshot.zip> --sources <dir of archived .zip> --out <snapshots dir>

Why: T2 serves a pool snapshot only if its FORWARD fingerprint (constructor kwargs + critic runtime
attrs, `inference/service/slots.forward_fingerprint`) equals the trainee's. Every archived run predates the
deletion pass's constructor changes, so its zips are refused (`SlotArchMismatch`, "same weight shapes but a
different FORWARD fingerprint"). The template is a HEAD-built model (the step-0 pool seed a fresh
`--arch production` launch writes); each source's POLICY state dict (same shapes) is loaded into it
strictly and saved under the source's (offset) snapshot name. The weights are trained and distinct, so the
pool's T2 fan-out (one slot per snapshot) is the steady state's. Used only to profile; nothing trains on it
that is read as a result.
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3.common.save_util import load_from_zip_file

from agents.inference.service.slots import forward_fingerprint
from agents.model.snapshot import load_checkpoint_strict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = load_checkpoint_strict(a.template, device="cpu")
    fp0 = forward_fingerprint(model.policy)
    print("template fingerprint", fp0[:12])
    for name in sorted(os.listdir(a.sources)):
        if not name.endswith(".zip"):
            continue
        _, params, _ = load_from_zip_file(os.path.join(a.sources, name), device="cpu", print_system_info=False)
        model.policy.load_state_dict(params["policy"], strict=True)
        assert forward_fingerprint(model.policy) == fp0
        dst = os.path.join(a.out, name)
        model.save(dst[:-4])
        print("wrote", dst)


if __name__ == "__main__":
    main()
