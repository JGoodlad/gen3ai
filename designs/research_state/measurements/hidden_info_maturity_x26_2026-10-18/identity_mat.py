"""PREREGISTRATION §3.5 VALIDITY, per anchor (no arm outcome is printed: only match counts): the CROSS-COMMIT identity.
``oracle`` worlds built at the HEAD (`build_mat_worlds.py --validation oracle`) and played at the pin with the anchor's
checkpoint must reproduce that anchor's Phase 1 (rows/<label>/O) exactly: every legal action's outcomes on worlds
0..7 = Phase 1's leaf seeds 0..7, the value traces within Tier 1's tolerances, the one-ply critic's 8 successors =
Phase 1's seeds 40..47. Tier 1's ``identity_read.compare`` and E2's ``identity_e2.tally`` are IMPORTED, never copied.

    python -I identity_mat.py [--cp x26_025M,...] [--out identity_mat.json]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402

E.add_paths()


def e2_tally():
    p = E.MEAS / "hidden_info_e2_competent_2026-10-16" / "identity_e2.py"
    sys.path.insert(1, str(p.parent))
    spec = importlib.util.spec_from_file_location("_identity_e2_for_mat", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.tally


def run(labels) -> dict:
    from identity_read import compare          # Tier 1's (pure)

    tally = e2_tally()
    doc = {}
    for lab in labels:
        truth = E.read_dir(E.rows_dir(lab, "O"), "shard_*.jsonl")
        orc = E.read_dir(E.rows_dir(lab, "V:oracle"), "rows_*.jsonl")
        pairs = [(d, truth.get(d), r) for d, r in sorted(orc.items()) if d in truth]
        doc[lab] = {"oracle = Phase 1": tally(pairs, compare), "identity_states_played": len(orc)}
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels = a.cp.split(",") if a.cp else sorted(E.registered_ckpts())
    s = json.dumps(run(labels), indent=1, sort_keys=True)
    print(s)
    if a.out:
        (E.HERE / a.out).write_text(s + "\n")


if __name__ == "__main__":
    main()
