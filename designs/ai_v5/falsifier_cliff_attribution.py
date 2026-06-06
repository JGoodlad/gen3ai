"""Zero-retrain falsifier for the incoming-damage obs feature (ai_v5).

Question: would a never-blank, prior-aware INCOMING-DAMAGE feature have helped the
122M critic at the value cliffs that lose games? Model-free attribution: for each
big value drop (delta_v cliff) in the loss corpus, was it
  (A) ADDRESSABLE  — critic optimistic, a big INCOMING hit landed, and the obs
      incoming channel (their_matchups) was BLANK/low at decision time, OR
  (B) SELF_KO      — our Explosion/Self-Destruct (a reward/value issue, not info), OR
  (C) SETUP        — we set up into the hit (feature could flag the incoming KO), OR
  (D) INFO_PRESENT — the incoming coverage was already revealed (critic ignored it
      → a feature in the same lane won't help without a retrain/attention fix), OR
  (E) OTHER/RNG.
GO if the addressable bucket (A + most of C) dominates; NO-GO if B/D/E dominate.

Run:
  PYTHONPATH=<worktree>/src python3 designs/ai_v5/falsifier_cliff_attribution.py
"""
from __future__ import annotations
import glob, json, os
import numpy as np

RUN = "/home/goodlad/dev/gen3ai/models/run_20260601_193826"
SNAP_STEPS = [122000011, 120000001, 118000001, 116000024, 114000008]
TM_OFF, TM_DIM = 1612, 144          # their_matchups (verified live)
CLIFF = -8.0                         # delta_v threshold for "a cliff"
BIG_HIT = -33.0                      # our hp_delta% counted as a big incoming hit
SETUP = {"dragondance","calmmind","swordsdance","curse","bellydrum","agility",
         "irondefense","amnesia","cosmicpower","bulkup","howl","meditate","sharpen",
         "tailglow","growth","acidarmor","barrier","defensecurl","harden","withdraw"}
SELF_KO = {"explosion","selfdestruct"}


def pct(s):
    try: return float(str(s).rstrip("%"))
    except Exception: return 0.0


def base_move(action: str) -> str:
    a = (action or "").lower()
    if a.startswith("switch") or a.startswith("switched"): return "<switch>"
    return a


def battles(outcome):
    out = []
    for step in SNAP_STEPS:
        for f in glob.glob(f"{RUN}/eval_traces/step_{step}/*/{outcome}_*_summary.json"):
            npz = f.replace("_summary.json", "_states.npz")
            if os.path.exists(npz): out.append((f, npz))
    return out


def decisions(summary_path, npz_path):
    s = json.load(open(summary_path))
    invs = s["invocations"]
    with np.load(npz_path) as z:
        vals = z["values"] if "values" in z.files else None
        obs = z["obs"] if "obs" in z.files else None
        hs = z["has_state"] if "has_state" in z.files else None
    rows = []
    for i, inv in enumerate(invs):
        if vals is None or i + 1 >= len(vals): continue
        v, vn = float(vals[i]), float(vals[i + 1])
        oc = inv.get("outcome") or {}
        our = oc.get("our") or {}; opp = oc.get("opp") or {}
        ev = oc.get("events") or []
        rf = mx = None
        if obs is not None and hs is not None and i < len(hs) and hs[i] and obs.shape[1] >= TM_OFF + TM_DIM:
            seg = obs[i, TM_OFF:TM_OFF + TM_DIM]
            rf = float((seg > 0).mean()); mx = float(seg.max()) * 4.0
        rows.append(dict(
            i=i, turn=inv.get("turn"), v=v, dv=vn - v,
            our_action=base_move(our.get("action")), our_hp=pct(our.get("hp_delta")),
            opp_action=(opp.get("action") or "").lower(),
            faint_self=any(str(e).startswith("our:") and "faint" in str(e).lower() for e in ev),
            revealed_frac=rf, max_incoming=mx,
            our_active=inv.get("our", {}).get("species"), opp_active=inv.get("opp", {}).get("species"),
        ))
    return rows


def attribute(r):
    big_incoming = (r["our_hp"] <= BIG_HIT) or (r["faint_self"] and r["our_action"] not in SELF_KO)
    optimistic = r["v"] > -2.0
    blank = (r["revealed_frac"] is not None and r["revealed_frac"] < 0.33)
    if r["our_action"] in SELF_KO:
        return "B_SELF_KO"
    if r["our_action"] in SETUP and big_incoming:
        return "C_SETUP"
    if big_incoming and optimistic and blank:
        return "A_ADDRESSABLE"
    if big_incoming and r["revealed_frac"] is not None and r["revealed_frac"] >= 0.33:
        return "D_INFO_PRESENT"
    return "E_OTHER"


def summarize(name, rows):
    cliffs = [r for r in rows if r["dv"] < CLIFF]
    print(f"\n=== {name}: {len(rows)} decisions, {len(cliffs)} cliffs (dv<{CLIFF}) ===")
    if not cliffs: return
    buckets = {}
    for r in cliffs:
        b = attribute(r); buckets[b] = buckets.get(b, 0) + 1
    n = len(cliffs)
    for b in sorted(buckets):
        print(f"   {b:16s} {buckets[b]:4d}  ({100*buckets[b]/n:4.1f}%)")
    addr = buckets.get("A_ADDRESSABLE", 0) + buckets.get("C_SETUP", 0)
    print(f"   --> ADDRESSABLE (A+C) = {addr}/{n} = {100*addr/n:.1f}%")
    import statistics as st
    print(f"   cliff V_before: mean {st.mean(r['v'] for r in cliffs):+.2f}, %>0 "
          f"{100*sum(r['v']>0 for r in cliffs)/n:.0f}%")
    print(f"   cliff our_hp_delta: mean {st.mean(r['our_hp'] for r in cliffs):.1f}%, %faint "
          f"{100*sum(r['faint_self'] for r in cliffs)/n:.0f}%")
    rfs = [r["revealed_frac"] for r in cliffs if r["revealed_frac"] is not None]
    if rfs:
        print(f"   cliff revealed_frac: mean {st.mean(rfs):.2f}, %<0.33 {100*sum(x<0.33 for x in rfs)/len(rfs):.0f}%")
    return cliffs, buckets


def decisive_tp(rows):
    """First cliff while clearly winning: V_before>5 and dv<-15 — the cleanest blindspot."""
    for r in rows:
        if r["v"] > 5.0 and r["dv"] < -15.0:
            return r
    return None


def main():
    loss_rows, win_rows = [], []
    loss_tps = []
    for f, n in battles("loss"):
        rs = decisions(f, n); loss_rows += rs
        tp = decisive_tp(rs)
        if tp: loss_tps.append(tp)
    for f, n in battles("win"):  win_rows  += decisions(f, n)
    lc, _ = summarize("LOSS", loss_rows)
    summarize("WIN (control)", win_rows)

    # The decision that matters: per-loss-battle DECISIVE turning point (first cliff while winning).
    print(f"\n=== DECISIVE TURNING POINTS (one per loss battle with V>5 then dv<-15): {len(loss_tps)} ===")
    if loss_tps:
        bk = {}
        for r in loss_tps:
            b = attribute(r); bk[b] = bk.get(b, 0) + 1
        n = len(loss_tps)
        for b in sorted(bk): print(f"   {b:16s} {bk[b]:4d}  ({100*bk[b]/n:4.1f}%)")
        addr = bk.get("A_ADDRESSABLE", 0) + bk.get("C_SETUP", 0)
        info = bk.get("D_INFO_PRESENT", 0)
        big = [r for r in loss_tps if (r["our_hp"] <= BIG_HIT or (r["faint_self"] and r["our_action"] not in SELF_KO))]
        print(f"   --> ADDRESSABLE (A+C) = {addr}/{n} = {100*addr/n:.1f}%  | INFO_PRESENT(critic ignored) = {100*info/n:.0f}%")
        print(f"   --> decisive TP was a big INCOMING hit: {len(big)}/{n} = {100*len(big)/n:.0f}%  "
              f"(of those, coverage blank<0.33: {100*sum((r['revealed_frac'] or 0)<0.33 for r in big)/max(1,len(big)):.0f}%)")

    # CONTROL: do WIN battles have the same addressable decisive cliffs? (discrimination check)
    win_tps = []
    for f, n in battles("win"):
        tp = decisive_tp(decisions(f, n))
        if tp: win_tps.append(tp)
    nlb = len(battles("loss")); nwb = len(battles("win"))
    la = sum(attribute(r) in ("A_ADDRESSABLE", "C_SETUP") for r in loss_tps)
    wa = sum(attribute(r) in ("A_ADDRESSABLE", "C_SETUP") for r in win_tps)
    print(f"\n=== DISCRIMINATION (rate of addressable decisive cliff per battle) ===")
    print(f"   LOSS: {la}/{nlb} battles = {100*la/max(1,nlb):.0f}%   WIN: {wa}/{nwb} battles = {100*wa/max(1,nwb):.0f}%")

    # ROBUSTNESS: addressable% among loss decisive-TPs across thresholds.
    print(f"\n=== ROBUSTNESS (addressable A+C %% among loss decisive-TPs) ===")
    for vmin in (0.0, 5.0, 10.0):
        for blank in (0.25, 0.33, 0.50):
            tps = []
            for f, nn in battles("loss"):
                for r in decisions(f, nn):
                    if r["v"] > vmin and r["dv"] < -15.0:
                        tps.append(r); break
            if not tps: continue
            def addr2(r):
                big = (r["our_hp"] <= BIG_HIT) or (r["faint_self"] and r["our_action"] not in SELF_KO)
                if r["our_action"] in SELF_KO: return False
                if r["our_action"] in SETUP and big: return True
                return big and r["v"] > -2 and (r["revealed_frac"] is not None and r["revealed_frac"] < blank)
            a = sum(addr2(r) for r in tps)
            print(f"   V>{vmin:>4} blank<{blank:.2f}:  {a:4d}/{len(tps):4d} = {100*a/len(tps):4.0f}%")
    # Show the worst 18 loss cliffs for eyeballing.
    lc_sorted = sorted(lc, key=lambda r: r["dv"])[:18]
    print("\n=== worst 18 loss cliffs ===")
    print(f"{'dv':>7} {'V_bef':>6} {'ourhp':>6} {'revfr':>5} {'maxin':>5}  {'our_act':>14}  {'opp_act':>12}  our->opp  bucket")
    for r in lc_sorted:
        rf = f"{r['revealed_frac']:.2f}" if r['revealed_frac'] is not None else "  -  "
        mx = f"{r['max_incoming']:.1f}" if r['max_incoming'] is not None else " - "
        print(f"{r['dv']:7.1f} {r['v']:6.1f} {r['our_hp']:6.0f} {rf:>5} {mx:>5}  "
              f"{r['our_action'][:14]:>14}  {r['opp_action'][:12]:>12}  "
              f"{str(r['our_active'])[:8]}->{str(r['opp_active'])[:8]}  {attribute(r)}")


if __name__ == "__main__":
    main()
