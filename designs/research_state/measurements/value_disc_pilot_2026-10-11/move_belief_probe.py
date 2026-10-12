"""DEBUG AID (no outcome): print the network's per-mon move belief at a few evaluation roots beside the true sets.

    run_at_pin.sh <cp> 1 8 20m move_belief_probe.py [N]
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import move_belief as MB  # noqa: E402


def main():
    c = C.cp()
    C.check_pin(c)
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.replay import replay

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    od = C.out_dir(c)
    bank = load_bank(od / "bank")
    decs = sorted(bank.decisions, key=lambda x: x["id"])[:n]
    bids = sorted({d["battle"] for d in decs})
    reps = dict(zip(bids, replay([bank.battles[bank.battle_index[b]].recorded() for b in bids], workers=1)))
    dd = [reps[d["battle"]].decisions[0 if d["side"] == "p1" else 1][d["n"]] for d in decs]
    rows = np.stack([np.array(x.row, dtype=np.float32) for x in dd])
    masks = np.stack([np.asarray(x.mask, dtype=np.uint8) for x in dd])
    data = Path(os.environ["VDP_PIN"]) / "data" / "pokemon"
    n2m = MB.move_ids(data)
    with inference_globals(1):
        bel = MB.root_move_beliefs(C.load_model(c), rows, masks, n2m)
    for d, slots in zip(decs, bel):
        b = bank.battles[bank.battle_index[d["battle"]]]
        opp = (b.p2 if d["side"] == "p1" else b.p1)["team"]
        print("==", d["id"], "turn", d["turn"])
        for e in opp.split("]"):
            f = e.split("|")
            print("   TRUE", f[0], f[4])
        for s in slots:
            top = sorted(s["w"].items(), key=lambda kv: -kv[1])[:8]
            print("   slot num", s["num"], "hyp" if s["hyp"] else "rev", "revealed", s["revealed"],
                  "sum", round(sum(s["w"].values()), 3), [(k, round(v, 3)) for k, v in top])


if __name__ == "__main__":
    main()
