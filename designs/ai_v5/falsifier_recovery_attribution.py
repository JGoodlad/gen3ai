"""Zero-retrain falsifier for the STALL / RECOVERY-RESET obs feature (ai_v5).

Sibling of `falsifier_cliff_attribution.py` (incoming-damage). That one split the
decisive value cliffs into ADDRESSABLE-by-incoming-damage (~56%) vs E_OTHER (~34%).
This one dissects E_OTHER — the bucket the incoming-damage feature does NOT fix —
to find the genuinely-biggest *fixable* residual.

Method (identical attribution skeleton; only the E_OTHER split is new):
  For each loss battle take its DECISIVE turning point (first cliff while clearly
  winning: V_before > V_MIN and dv < DV_MAX), bucket it with the ORIGINAL
  incoming-damage attribute() (A_ADDRESSABLE / B_SELF_KO / C_SETUP / D_INFO_PRESENT
  / E_OTHER), then sub-split the E_OTHER rows into:
    (a) RECOVERY_RESET   — opp used a recovery move OR opp HP jumped UP >= HEAL%
                           this turn (Rest/Recover/Softboiled/Wish-land/Pain Split
                           /Leftovers-stack), while the critic priced V high.
    (b) ALREADY_LOSING   — V_before < 0 (downstream of an earlier blindspot; NOT a
                           fresh category — excluded as a target). Cannot occur in a
                           decisive TP by construction (V>V_MIN); shows up only in the
                           all-cliffs view below.
    (c) TEMPO_SWING      — V dropped with no big incoming hit and no opp heal (a bad
                           matchup got revealed / a switch flipped the board).

Reports the split as a % of decisive turning points, a threshold sweep (HEAL%, V_MIN,
DV_MAX), and a WIN-vs-LOSS discrimination control (the same robustness battery the
incoming-damage falsifier used). GO for a recovery feature iff RECOVERY_RESET is the
biggest fixable E_OTHER sub-bucket AND it discriminates loss from win.

Run:
  PYTHONPATH=<worktree>/src python3 designs/ai_v5/falsifier_recovery_attribution.py
"""
from __future__ import annotations
import glob, json, os
import statistics as st
import numpy as np

RUN = "/home/goodlad/dev/gen3ai/models/run_20260601_193826"
SNAP_STEPS = [122000011, 120000001, 118000001, 116000024, 114000008]
TM_OFF, TM_DIM = 1612, 144           # their_matchups (verified live for this run's arch)
CLIFF = -8.0                          # delta_v threshold for "a cliff" (all-cliffs view)
BIG_HIT = -33.0                       # our hp_delta% counted as a big incoming hit
V_MIN = 5.0                           # "clearly winning" floor for a decisive TP
DV_MAX = -15.0                        # cliff depth for a decisive TP
HEAL = 20.0                           # opp HP jump (%) that counts as "healed a lot"

SETUP = {"dragondance","calmmind","swordsdance","curse","bellydrum","agility",
         "irondefense","amnesia","cosmicpower","bulkup","howl","meditate","sharpen",
         "tailglow","growth","acidarmor","barrier","defensecurl","harden","withdraw"}
SELF_KO = {"explosion","selfdestruct"}
# Direct self-heal moves in gen3 OU. HP-jump detection (heal_jump) is the primary,
# move-id is corroboration (catches Rest curing status, and Wish whose heal lands a
# turn later under a different action).
RECOVERY_MOVES = {"rest","recover","softboiled","milkdrink","slackoff","moonlight",
                  "morningsun","synthesis","wish","swallow","painsplit","healorder",
                  "moonlight"}


def pct(s):
    try: return float(str(s).rstrip("%"))
    except Exception: return 0.0


def base_move(action: str) -> str:
    a = (action or "").lower()
    if a.startswith("switch") or a.startswith("switched"): return "<switch>"
    return a


def opp_move(action: str) -> str:
    a = (action or "").lower()
    if a.endswith("_sent_in") or a.endswith("sent_in"): return "<switch>"
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
    opp_name = os.path.basename(os.path.dirname(summary_path))
    rows = []
    for i, inv in enumerate(invs):
        if vals is None or i + 1 >= len(vals): continue
        v, vn = float(vals[i]), float(vals[i + 1])
        oc = inv.get("outcome") or {}
        our = oc.get("our") or {}; opp = oc.get("opp") or {}
        ev = oc.get("events") or []
        rew = (oc.get("reward") or {}).get("base") or ""
        rf = mx = None
        if obs is not None and hs is not None and i < len(hs) and hs[i] and obs.shape[1] >= TM_OFF + TM_DIM:
            seg = obs[i, TM_OFF:TM_OFF + TM_DIM]
            rf = float((seg > 0).mean()); mx = float(seg.max()) * 4.0
        rows.append(dict(
            i=i, turn=inv.get("turn"), v=v, dv=vn - v,
            our_action=base_move(our.get("action")), our_hp=pct(our.get("hp_delta")),
            opp_action=opp_move(opp.get("action")), opp_hp=pct(opp.get("hp_delta")),
            faint_self=any(str(e).startswith("our:") and "faint" in str(e).lower() for e in ev),
            revealed_frac=rf, max_incoming=mx, reward_base=rew, opponent=opp_name,
            our_active=inv.get("our", {}).get("species"), opp_active=inv.get("opp", {}).get("species"),
            summary=summary_path,
        ))
    return rows


# ---- ORIGINAL incoming-damage attribution (unchanged, for parity) -------------
def attribute(r, big_hit=BIG_HIT, blank_thr=0.33):
    big_incoming = (r["our_hp"] <= big_hit) or (r["faint_self"] and r["our_action"] not in SELF_KO)
    optimistic = r["v"] > -2.0
    blank = (r["revealed_frac"] is not None and r["revealed_frac"] < blank_thr)
    if r["our_action"] in SELF_KO:
        return "B_SELF_KO"
    if r["our_action"] in SETUP and big_incoming:
        return "C_SETUP"
    if big_incoming and optimistic and blank:
        return "A_ADDRESSABLE"
    if big_incoming and r["revealed_frac"] is not None and r["revealed_frac"] >= blank_thr:
        return "D_INFO_PRESENT"
    return "E_OTHER"


# ---- NEW: split the E_OTHER residual ----------------------------------------
def heal_jump(r, heal=HEAL): return r["opp_hp"] >= heal
def recovery_move(r): return r["opp_action"] in RECOVERY_MOVES


def attribute_other(r, heal=HEAL):
    """Sub-classify a row that attribute() put in E_OTHER."""
    if recovery_move(r) or heal_jump(r, heal):
        return "a_RECOVERY_RESET"
    if r["v"] < 0:
        return "b_ALREADY_LOSING"
    return "c_TEMPO_SWING"


def decisive_tp(rows, v_min=V_MIN, dv_max=DV_MAX):
    for r in rows:
        if r["v"] > v_min and r["dv"] < dv_max:
            return r
    return None


def split_report(name, tps):
    n = len(tps)
    print(f"\n=== {name}: {n} decisive turning points ===")
    if not n: return
    bk = {}
    for r in tps:
        bk[attribute(r)] = bk.get(attribute(r), 0) + 1
    for b in sorted(bk): print(f"   {b:16s} {bk[b]:4d}  ({100*bk[b]/n:4.1f}%)")
    addr = bk.get("A_ADDRESSABLE", 0) + bk.get("C_SETUP", 0)
    print(f"   --> ADDRESSABLE-by-incoming (A+C) = {addr}/{n} = {100*addr/n:.1f}%")
    # split E_OTHER
    others = [r for r in tps if attribute(r) == "E_OTHER"]
    print(f"\n   E_OTHER = {len(others)}/{n} = {100*len(others)/n:.1f}%  --> sub-split:")
    sub = {}
    for r in others:
        sub[attribute_other(r)] = sub.get(attribute_other(r), 0) + 1
    for b in sorted(sub):
        print(f"      {b:18s} {sub[b]:4d}  ({100*sub[b]/n:4.1f}% of all TPs, {100*sub[b]/max(1,len(others)):4.1f}% of E_OTHER)")
    return others


def main():
    loss_b, win_b = battles("loss"), battles("win")
    print(f"corpus: {len(loss_b)} loss battles, {len(win_b)} win battles  (steps {SNAP_STEPS})")

    loss_tps, win_tps = [], []
    for f, n in loss_b:
        tp = decisive_tp(decisions(f, n))
        if tp: loss_tps.append(tp)
    for f, n in win_b:
        tp = decisive_tp(decisions(f, n))
        if tp: win_tps.append(tp)

    loss_others = split_report("LOSS", loss_tps)
    win_others = split_report("WIN (control)", win_tps)

    # -------- DISCRIMINATION: recovery-reset decisive-TP rate per battle ----------
    def rec_rate(tps, nb):
        rec = sum(attribute(r) == "E_OTHER" and attribute_other(r) == "a_RECOVERY_RESET" for r in tps)
        return rec, nb, 100 * rec / max(1, nb)
    lr = rec_rate(loss_tps, len(loss_b)); wr = rec_rate(win_tps, len(win_b))
    print("\n=== DISCRIMINATION (recovery-reset decisive-TP rate per battle) ===")
    print(f"   LOSS: {lr[0]}/{lr[1]} battles = {lr[2]:.0f}%    WIN: {wr[0]}/{wr[1]} battles = {wr[2]:.0f}%")

    # -------- ROBUSTNESS: recovery-reset % of decisive TPs over thresholds -------
    print("\n=== ROBUSTNESS (recovery-reset % among loss decisive-TPs) ===")
    print(f"   {'V_MIN':>6} {'DV_MAX':>7} {'HEAL%':>6}   recov/E_OTHER/TPs   %TP   %E_OTHER")
    for v_min in (0.0, 5.0, 10.0):
        for dv_max in (-10.0, -15.0, -25.0):
            for heal in (15.0, 20.0, 30.0):
                tps = []
                for f, n in loss_b:
                    tp = decisive_tp(decisions(f, n), v_min=v_min, dv_max=dv_max)
                    if tp: tps.append(tp)
                if not tps: continue
                others = [r for r in tps if attribute(r) == "E_OTHER"]
                rec = sum(attribute_other(r, heal) == "a_RECOVERY_RESET" for r in others)
                print(f"   {v_min:>6} {dv_max:>7} {heal:>6}   {rec:>3}/{len(others):>3}/{len(tps):>3}        "
                      f"{100*rec/len(tps):>4.0f}  {100*rec/max(1,len(others)):>5.0f}")

    # -------- ALL-CLIFFS view: surfaces ALREADY_LOSING (downstream) magnitude ----
    print("\n=== ALL-CLIFFS E_OTHER split (dv<%.0f, any V) — to size the downstream/losing share ===" % CLIFF)
    all_cliffs = []
    for f, n in loss_b:
        all_cliffs += [r for r in decisions(f, n) if r["dv"] < CLIFF]
    others = [r for r in all_cliffs if attribute(r) == "E_OTHER"]
    sub = {}
    for r in others:
        sub[attribute_other(r)] = sub.get(attribute_other(r), 0) + 1
    print(f"   {len(all_cliffs)} loss cliffs; E_OTHER={len(others)}")
    for b in sorted(sub):
        print(f"      {b:18s} {sub[b]:4d}  ({100*sub[b]/max(1,len(others)):4.1f}% of E_OTHER)")

    # -------- DUMP the recovery-reset decisive TPs for prober targeting ----------
    rec_rows = [r for r in loss_tps if attribute(r) == "E_OTHER" and attribute_other(r) == "a_RECOVERY_RESET"]
    rec_rows = sorted(rec_rows, key=lambda r: r["dv"])
    print(f"\n=== RECOVERY-RESET decisive turning points (loss), worst-first: {len(rec_rows)} ===")
    print(f"{'dv':>7} {'V_bef':>6} {'opphp':>6} {'ourhp':>6}  {'our_act':>13}  {'opp_act':>11}  {'matchup':>20}  {'opp':>14}  inv")
    for r in rec_rows:
        bid = f"step_{r['summary'].split('/step_')[1].split('/')[0]}/{r['opponent']}/{os.path.basename(r['summary']).replace('_summary.json','')}"
        print(f"{r['dv']:7.1f} {r['v']:6.1f} {r['opp_hp']:+6.0f} {r['our_hp']:6.0f}  "
              f"{r['our_action'][:13]:>13}  {r['opp_action'][:11]:>11}  "
              f"{str(r['our_active'])[:9]}->{str(r['opp_active'])[:9]:>9}  {r['opponent'][:14]:>14}  {r['i']}")
        print(f"          id={bid}")

    # Also dump ALL recovery resets across ALL loss cliffs (not just decisive) for breadth.
    rec_all = sorted([r for r in all_cliffs if attribute(r) == "E_OTHER"
                      and attribute_other(r) == "a_RECOVERY_RESET"], key=lambda r: r["dv"])[:25]
    print(f"\n=== RECOVERY-RESET among ALL loss cliffs (dv<{CLIFF}), worst 25 of {sum(attribute_other(r)=='a_RECOVERY_RESET' for r in others)} ===")
    print(f"{'dv':>7} {'V_bef':>6} {'opphp':>6}  {'our_act':>13}  {'opp_act':>11}  {'matchup':>20}  {'opp':>12}")
    for r in rec_all:
        print(f"{r['dv']:7.1f} {r['v']:6.1f} {r['opp_hp']:+6.0f}  {r['our_action'][:13]:>13}  {r['opp_action'][:11]:>11}  "
              f"{str(r['our_active'])[:9]}->{str(r['opp_active'])[:9]:>9}  {r['opponent'][:12]:>12}")

    battle_stall_report()


def battle_stall_report():
    """Battle-LEVEL recovery-stall view (the decisive-TP metric undercounts slow stalls).

    A 'recovery-stall' battle has >= STALL_MIN big opp self-heals (opp_hp jump >= HEAL,
    opp didn't switch) on a wall we were attacking. Also reports the per-wall-species heal
    counts — the load-bearing 'Suicune-Rest is the discriminator' signal (losses vs wins)."""
    from collections import Counter
    STALL_MIN = 2
    print("\n=== BATTLE-LEVEL recovery-stall (>=%d big opp self-heals while attacked) ===" % STALL_MIN)
    for oc in ("loss", "win"):
        total = stall = 0
        walls = Counter()
        for f, n in battles(oc):
            rows = decisions(f, n); total += 1
            heals = [r for r in rows if r["opp_hp"] >= HEAL and r["opp_action"] not in ("<switch>", "none", "")]
            for r in heals: walls[r["opp_active"]] += 1
            if len(heals) >= STALL_MIN: stall += 1
        print(f"   {oc.upper():4s}: {stall}/{total} = {100*stall/max(1,total):.1f}% stall-battles | "
              f"big-heal events by wall species: {walls.most_common(8)}")
    print("   (the 'Suicune ~127 in LOSS vs ~2 in WIN' split is the recovery discriminator — "
          "Rest cures the Toxic clock; Softboiled does not, so Blissey heals appear in both.)")


if __name__ == "__main__":
    main()
