"""TEETH for Phase 2's driver (run after its read, as a GIGO check): give ``run_det_playouts.one_state`` the TRUE
opponent team as every "world" (the true prefix commands) and require its outcomes to equal Phase 1's on the same
dice seeds 0..7, byte for byte. If the honest leaf's collapse were a driver bug (a wrong team, seed, side or prefix),
this would fail.

    run_at_pin.sh <cp> 1 6 1h det_identity_check.py [--n 20]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import run_det_playouts as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    c = C.cp()
    C.check_pin(c)
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    od = C.out_dir(c)
    bank = load_bank(od / "bank")
    p1 = {r["id"]: r for q in sorted((od / "rows").glob("shard_*.jsonl")) for r in C.read_rows(q) if r.get("ok")}
    ids = sorted(p1)[:: max(1, len(p1) // a.n)][: a.n]
    lib = F.load(F.default_path("release"))
    cores = [SearchCore(lib=lib) for _ in range(R.J)]
    off = C.clock_offset()
    out = {"cp": c, "states": 0, "actions_compared": 0, "mismatch": []}
    with inference_globals(C.threads()):
        fwd = R.Fwd(C.load_model(c))
        for did in ids:
            d = next(x for x in bank.decisions if x["id"] == did)
            b = bank.battles[bank.battle_index[d["battle"]]]
            opp = "p2" if d["side"] == "p1" else "p1"
            at = cmd_index(b, d["side"], d["n"])
            true_team = getattr(b, opp)["team"]
            w = {"teams": [true_team] * R.J, "cmds": [[list(x) for x in b.commands[:at]]] * R.J}
            r = R.one_state(cores, fwd, bank, did, w, off)
            out["states"] += 1
            for act, s in r["out"].items():
                out["actions_compared"] += 1
                if s != p1[did]["out"][act][: R.J]:
                    out["mismatch"].append({"id": did, "action": act, "det": s, "phase1": p1[did]["out"][act][: R.J]})
    for core in cores:
        core.close()
    out["pass"] = not out["mismatch"]
    (C.HERE / f"det_identity_check_{c}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v if k != "mismatch" else len(v)) for k, v in out.items()}))


if __name__ == "__main__":
    main()
