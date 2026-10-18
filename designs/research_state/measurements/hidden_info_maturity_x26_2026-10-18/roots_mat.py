"""PREREGISTRATION §3.1: the checkpoint's OWN ROOT read on every bank state (its logits, V, top-3 and top-2 gap), one CPU
forward at the series' pin. The policy's top-1 is the baseline every IMP is measured against (§12.13.7 (b): "its own
top-1"), and a top-2 gap < 1e-6 is that checkpoint's rule-8 exclusion. No playout, no outcome.

    run_mat.sh pin 1 8 2h roots_mat.py --cp x26_025M

The roots are re-derived by replaying the bank's battles (the bank stores inputs, never obs); every replayed mask must
equal the banked one. Writes ``<archive>/roots/<label>.jsonl`` (one row per bank state; idempotent: refuses to append
to a complete file, rewrites an incomplete one).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402

E.add_paths()


def roots(bank, run_core, viewer_decisions):
    """{id: (row, mask)} of every bank decision, by replay."""
    want = {d["id"]: d for d in bank.decisions}
    got = {}
    for i in range(0, len(bank.battles), 32):
        part = bank.battles[i:i + 32]
        for b, r in zip(part, run_core([x.recorded() for x in part], trackers=True, obs=True)):
            if not r["ok"]:
                raise SystemExit(f"bank battle {b.battle_id} refused on replay: {r.get('error')}")
            ti = 0 if b.traced_side == "p1" else 1
            for d in viewer_decisions(r, ti):
                did = f"{b.battle_id}#{b.banked_side}#{d.n}"
                if did in want:
                    if "".join(str(int(x)) for x in d.mask) != want[did]["mask"]:
                        raise SystemExit(f"{did}: the replayed mask differs from the banked one")
                    got[did] = (np.asarray(d.row, dtype=np.float32), np.asarray(d.mask, dtype=np.uint8))
    if set(got) != set(want):
        raise SystemExit(f"{len(set(want) - set(got))} bank roots not re-derived")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", required=True)
    a = ap.parse_args()
    E.check_pin()
    if a.cp not in E.registered_ckpts():
        raise SystemExit(f"{a.cp}: not a registered anchor")
    import common as C  # the pilot's

    from agents.battle.core_replay import run_core
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.replay import _viewer_decisions

    out = E.d("roots") / f"{a.cp}.jsonl"
    bank = load_bank(E.bank_path())
    ids = sorted(d["id"] for d in bank.decisions)
    have = E.read_rows(out)
    if sorted(r["id"] for r in have) == ids:
        print(f"[roots] {a.cp}: complete ({len(ids)})", flush=True)
        return
    t0, c0 = time.monotonic(), C.cpu_seconds()
    rr = roots(bank, run_core, _viewer_decisions)
    rows = np.stack([rr[x][0] for x in ids])
    masks = np.stack([rr[x][1] for x in ids])
    with inference_globals(E.threads()):
        model = E.load_model(a.cp)
        o = logits_and_values(model, rows, masks, batch=256)
    lg = np.where(masks.astype(bool), o[:, :11].astype(np.float64), -np.inf)
    srt = -np.sort(-lg, axis=1)
    gap = srt[:, 0] - srt[:, 1]
    tmp = out.with_suffix(".part")
    with open(tmp, "w") as f:
        for i, did in enumerate(ids):
            top = [int(x) for x in np.argsort(-lg[i], kind="stable")[:3]]
            E.write_row(f, {"id": did, "cp": a.cp, "logits": [round(float(x), 6) for x in o[i, :11]],
                            "v_root": round(float(o[i, 11]), 6), "top": top, "gap": round(float(gap[i]), 6),
                            "tie": bool(gap[i] < 1e-6), "pin": E.series().pin})
    tmp.replace(out)
    print(json.dumps({"cp": a.cp, "states": len(ids), "ties": int((gap < 1e-6).sum()),
                      "cpu_s": round(C.cpu_seconds() - c0, 1), "wall_s": round(time.monotonic() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
