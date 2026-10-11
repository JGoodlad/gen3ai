"""DESCRIPTIVE (no outcome): is the network's OWN species belief a better world sampler than the Smogon prior?
At every sampled root: the X5 hypothesis set's species presence π (Σ π = k, the unrevealed count) and the
expected share of a π-proportional draw that lands in the TRUE hidden species set, Σ_{s ∈ true hidden} π_s / k —
beside the same quantity for the Smogon-prior sampler's accepted worlds (`world_accuracy_read.json`).

    run_at_pin.sh <cp> 2 8 1h belief_accuracy_read.py   →  belief_accuracy_<cp>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def toid(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def main():
    c = C.cp()
    C.check_pin(c)
    from agents.inference.belief_decode import _num_to_species_id
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.replay import replay

    od = C.out_dir(c)
    bank = load_bank(od / "bank")
    worlds = {r["id"]: r for q in sorted((od / "det").glob("worlds_*.jsonl")) for r in C.read_rows(q) if r.get("ok")}
    bids = sorted({d["battle"] for d in bank.decisions})
    reps = dict(zip(bids, replay([bank.battles[bank.battle_index[b]].recorded() for b in bids], workers=2)))
    num2id = _num_to_species_id()
    rows, masks, meta = [], [], []
    for d in sorted(bank.decisions, key=lambda x: x["id"]):
        if d["id"] not in worlds:
            continue
        rep = reps[d["battle"]]
        dec = rep.decisions[0 if d["side"] == "p1" else 1][d["n"]]
        b = bank.battles[bank.battle_index[d["battle"]]]
        opp = b.p2 if d["side"] == "p1" else b.p1
        team = [toid((e.split("|")[1] or e.split("|")[0])) for e in opp["team"].split("]")]
        hidden = set(team) - set(worlds[d["id"]]["revealed"])
        rows.append(np.array(dec.row, dtype=np.float32))
        masks.append(np.asarray(dec.mask, dtype=np.uint8))
        meta.append({"id": d["id"], "hidden": sorted(hidden)})
    hits, tops = [], []
    with inference_globals(C.threads()):
        model = C.load_model(c)
        ext = model.policy.features_extractor
        for i in range(0, len(rows), 64):
            logits_and_values(model, np.stack(rows[i:i + 64]), np.stack(masks[i:i + 64]), batch=64)
            hs = ext.last_hypothesis
            pi = hs.species.pi.float().cpu().numpy()
            k = hs.species.k.cpu().numpy()
            for j, m in enumerate(meta[i:i + 64]):
                if k[j] <= 0 or not m["hidden"]:
                    continue
                mass = sum(float(pi[j][n]) for n in range(pi.shape[1]) if num2id.get(n) in set(m["hidden"]))
                hits.append(mass / float(k[j]))
                top = [num2id.get(int(n)) for n in np.argsort(-pi[j])[: int(k[j])]]
                tops.append(len(set(top) & set(m["hidden"])) / len(m["hidden"]))
    out = {"cp": c, "roots": len(hits), "belief_expected_hit_share": round(float(np.mean(hits)), 4),
           "belief_topk_recall": round(float(np.mean(tops)), 4)}
    (C.HERE / f"belief_accuracy_{c}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
