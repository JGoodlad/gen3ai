"""Debug aid for the determinizer (no playout): one state's sampled worlds and why the core refuses them."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import determinize as D  # noqa: E402


def main():
    c = C.cp()
    C.check_pin(c)
    from agents.battle.core_replay import RecordedBattle, run_core
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index

    bank = load_bank(C.out_dir(c) / "bank")
    P = D.Priors(Path(os.environ["VDP_PIN"]) / "data" / "pokemon")
    did = sys.argv[1]
    d = next(x for x in bank.decisions if x["id"] == did)
    b = bank.battles[bank.battle_index[d["battle"]]]
    at = cmd_index(b, d["side"], d["n"])
    opp = "p2" if d["side"] == "p1" else "p1"
    team = getattr(b, opp)["team"]
    print("TRUE", team)
    rng = D.world_rng(did, 0)
    for k in range(3):
        t, st = D.sample_world(rng, P, team, {"celebi": {"gigadrain"}})
        print("WORLD", st, t)
        p = {"p1": dict(b.p1), "p2": dict(b.p2)}
        p[opp] = dict(p[opp], team=t)
        r = run_core([RecordedBattle(label="dbg", format_id=b.format_id, seed=b.seed, p1=p["p1"], p2=p["p2"],
                                     commands=[list(x) for x in b.commands[:at]])], trackers=True, obs=True)[0]
        print("OK", r["ok"], str(r.get("error"))[:600])


if __name__ == "__main__":
    main()
