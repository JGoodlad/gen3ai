"""STARMIE/TYRANITAR RISK PROBE v2 — two owner-ordered extensions on the committed v1
(`starmie_ttar_risk_probe.py`, verdict: the mask reads the KO boundary at ~1/3 amplitude,
~0.58 Surf floor, argmax never flips).

Extension 1 — REVEALED-THREAT condition. v1's Tyranitar had its moveset hidden (only EQ
revealed) at the engineered decision. v2 engineers a prelude in which Tyranitar REVEALS
ALL FOUR of its moves before the decision, and reads the same capture (mask, V, win-prob,
alpha/beta) in the revealed condition vs a MATCHED hidden condition.

Extension 2 — THE BIAS SWEEP. The v1 HP sweep rerun in BOTH conditions with micro-steps
bracketing the 80% payoff crossover (true indifference sits between 12/16 and 13/16 Surf
KO rolls), delivering the bias as numbers: mask-mass at true equality, the argmax flip
threshold (or its censored bound), and dP(Surf)/d(truth-delta).

Construction deltas vs v1 (everything else — the explosion-into-Protect parity chain, the
exact-HP Endeavor targeting, the fail-loud decision-state asserts — is v1's, imported):
  - Tyranitar carries FOUR moves in BOTH conditions: Earthquake / Rock Slide /
    Focus Punch / Double-Edge (a real CB set; every one is blocked by a first-use Protect
    and none self-damages when blocked — no recoil fires on a blocked Double-Edge).
  - Marshtomp gains Protect (Endeavor / Growl / Protect) and the Endeavor moves from
    turn 1 to turn 9: a deterministic 9-turn REVEAL PHASE precedes it in BOTH conditions.
    Three reveal stints, each a 3-turn cycle (T clicks a move into Marshtomp's fresh
    Protect -> T pivots out to Koffing -> Koffing pivots back to T), then Marshtomp
    Endeavors on the final pivot-in turn while Tyranitar is still at FULL HP (Endeavor
    requires user HP < target HP; Marshtomp takes zero damage all phase — Ground-immune
    to sand, every Tyranitar move Protect-blocked). The phase is roll-free (first-use
    Protect is 100%, no damage lands, no speed ties), so turn-indexing Marshtomp's
    script is safe; every invariant that matters is still asserted at the decision.
  - HIDDEN arm: the three stint clicks are all Earthquake (already revealed by the kill
    phase) — same choreography, same turn count, only EQ ever shown. REVEALED arm: the
    stints click Rock Slide / Focus Punch / Double-Edge; EQ is revealed killing
    Marshtomp/Grimer. Decision-turn state is identical either way: Tyranitar forced back
    at 5/5 faints, choice-locked into EQ (it EQ'd Grimer the turn before), Starmie full.
  - The conditions therefore differ in the opponent's revealed-move slots (the
    treatment) plus the reveal-phase |move| identities in deep history; the decision-turn
    board, faint pattern and damage stream are matched.

Run (from the repo root; needs deps/pokemon-showdown built — node bridge):
    python designs/research_state/measurements/starmie_ttar_risk_probe_v2.py [--phase all]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Phases: smoke | capture | sweep | mc | analyze | all. Writes
starmie_ttar_risk_probe_v2_2026-08-31.json next to this file (resumable, incremental).
CPU-only, <=2 cores. v1's artifacts are untouched (reads v1's JSON for the roll tables).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))

import starmie_ttar_risk_probe as v1  # noqa: E402  (the committed v1 construction)

import numpy as np  # noqa: E402

from poke_env.ps_client import AccountConfiguration  # noqa: E402
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration  # noqa: E402

from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402

V1_JSON = OUT_DIR / "starmie_ttar_risk_probe_2026-08-30.json"
OUT_JSON = OUT_DIR / "starmie_ttar_risk_probe_v2_2026-08-31.json"
FORMAT = v1.FORMAT
CKPT = v1.CKPT
CKPT2 = v1.CKPT2
BASE_H = v1.BASE_H
CRIT_P = v1.CRIT_P

# The two arms' stint plans (the ONLY choreography difference between conditions).
REVEAL_HIDDEN = ("earthquake", "earthquake", "earthquake")
REVEAL_SHOWN = ("rockslide", "focuspunch", "doubleedge")
EXPECT_HIDDEN = frozenset({"earthquake"})
EXPECT_SHOWN = frozenset({"earthquake", "rockslide", "focuspunch", "doubleedge"})

# Marshtomp's deterministic reveal-phase script (turns 1..9), then growl until death.
M_SCRIPT = ("protect", "growl", "growl",     # stint 1: T clicks, pivots out, pivots back
            "protect", "growl", "growl",     # stint 2
            "protect", "growl", "endeavor")  # stint 3, then Endeavor on the pivot-in turn

# ---------------------------------------------------------------------------- teams

THEIR_TEAM_V2 = (
    "Tyranitar @ Choice Band\nAbility: Sand Stream\nLevel: 100\n"
    "EVs: 4 HP / 252 Atk\nAdamant Nature\n"
    "- Earthquake\n- Rock Slide\n- Focus Punch\n- Double-Edge\n\n"
    "Koffing\nAbility: Levitate\nLevel: 100\n- Protect\n- Explosion\n\n"
    "Weezing\nAbility: Levitate\nLevel: 100\n- Protect\n- Explosion\n\n"
    "Graveler\nAbility: Sturdy\nLevel: 100\n- Protect\n- Explosion\n\n"
    "Pineco\nAbility: Sturdy\nLevel: 100\n- Protect\n- Explosion\n\n"
    "Exeggutor\nAbility: Chlorophyll\nLevel: 100\n- Protect\n- Explosion")


def our_team_v2(h: int) -> str:
    fillers = (
        "Golem\nAbility: Sturdy\nLevel: 100\n- Protect\n- Explosion\n\n"
        "Forretress\nAbility: Sturdy\nLevel: 100\n- Protect\n- Explosion\n\n"
        "Camerupt\nAbility: Magma Armor\nLevel: 100\n- Protect\n- Explosion\n\n"
        "Grimer\nAbility: Sticky Hold\nLevel: 100\n- Protect\n- Sludge")
    marsh = (f"Marshtomp\nAbility: Torrent\nLevel: 100\n"
             f"EVs: {v1.marsh_hp_ev_for(h)} HP / 252 Spe\nJolly Nature\n"
             f"- Endeavor\n- Growl\n- Protect")
    return f"{marsh}\n\n{fillers}\n\n{v1.STARMIE_FRAIL}"

# ---------------------------------------------------------------------------- players


def _fresh_st() -> dict:
    return {"m_step": 0, "endeavored": False, "captured": False}


class _CoreV2:
    """v1's _OurScriptCore.classify with the Marshtomp reveal-phase script in front."""

    @staticmethod
    def classify(battle, st) -> tuple:
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if battle.force_switch or not battle.available_moves or me is None:
            return ("switch", None)
        if me.species == "marshtomp":
            i = st["m_step"]
            st["m_step"] = i + 1
            mv = M_SCRIPT[i] if i < len(M_SCRIPT) else "growl"
            if mv == "protect":
                assert opp is not None and opp.species == "tyranitar", (
                    f"M step {i} (protect) expected Tyranitar active, got "
                    f"{None if opp is None else opp.species}")
            if mv == "endeavor":
                # Decision-time active is the PIVOT filler (it switches to Tyranitar
                # this same turn; switches resolve first, so Endeavor executes against
                # the incoming Tyranitar). Tyranitar must still be untouched: Endeavor
                # sets its HP to ours EXACTLY.
                assert opp is not None and opp.species in v1.THEIR_FILLERS, (
                    f"M step {i} (endeavor) expected a pivot filler active, got "
                    f"{None if opp is None else opp.species}")
                ttar = next((m for m in battle.opponent_team.values()
                             if m.species == "tyranitar"), None)
                assert (ttar is not None and not ttar.fainted
                        and ttar.current_hp == ttar.max_hp), (
                    "Endeavor turn but Ttar not at full HP: "
                    f"{None if ttar is None else (ttar.current_hp, ttar.max_hp)}")
                st["endeavored"] = True
            return ("move", mv)
        if me.species in v1.OUR_FILLERS:
            explode_now = v1._faints(battle.team) < v1._faints(battle.opponent_team)
            if opp is not None and opp.species == "tyranitar":
                # Only Grimer (always-last filler) reaches this in explode state — v1 §2.
                return ("move", "sludge" if explode_now else "protect")
            return ("move", "explosion" if explode_now else "protect")
        # Starmie
        if opp is not None and opp.species == "tyranitar":
            if not st["captured"]:
                return ("decision", None)
            return ("move", "surf")
        return ("move", "recover")


class TtarSideV2(v1.Player):
    """Their side: v1's state-driven script + the three reveal stints vs Marshtomp.

    Stint cycle (while Marshtomp is active): click plan[done] -> pivot out to a filler
    -> the filler pivots back to Tyranitar -> next stint (or the kill phase: EQ)."""

    def __init__(self, *args, reveal_plan, **kwargs):
        super().__init__(*args, **kwargs)
        self._plan = tuple(reveal_plan)
        self._st = {}

    def choose_move(self, battle):
        st = self._st.setdefault(battle.battle_tag, {"done": 0, "just": False})
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if battle.force_switch or not battle.available_moves:
            filler = v1._first(battle.available_switches,
                               lambda m: m.species in v1.THEIR_FILLERS)
            target = filler or (battle.available_switches[0]
                                if battle.available_switches else None)
            return self.create_order(target) if target else self.choose_default_move()
        if me is not None and me.species == "tyranitar":
            if opp is not None and opp.species == "marshtomp":
                if st["just"]:  # pivot out the turn after a stint click (clears the lock)
                    st["just"] = False
                    filler = v1._first(battle.available_switches,
                                       lambda m: m.species in v1.THEIR_FILLERS)
                    assert filler is not None, "no pivot filler for the reveal stint"
                    return self.create_order(filler)
                if st["done"] < len(self._plan):
                    mid = self._plan[st["done"]]
                    mv = v1._first(battle.available_moves, lambda m: m.id == mid)
                    assert mv is not None, (
                        f"reveal move {mid!r} unavailable: "
                        f"{[m.id for m in battle.available_moves]}")
                    st["done"] += 1
                    st["just"] = True
                    return self.create_order(mv)
                # reveals done -> kill phase: EQ until Marshtomp dies
            if opp is not None and opp.species in v1.OUR_FILLERS:
                filler = v1._first(battle.available_switches,
                                   lambda m: m.species in v1.THEIR_FILLERS)
                if filler:  # voluntary switch: start/continue the filler chain
                    return self.create_order(filler)
            eq = v1._first(battle.available_moves, lambda m: m.id == "earthquake")
            if eq:
                return self.create_order(eq)
            return self.choose_default_move()
        # a filler
        if opp is not None and opp.species == "marshtomp":
            # reveal phase: pivot straight back to Tyranitar, never explode into Marshtomp
            ttar = v1._first(battle.available_switches,
                             lambda m: m.species == "tyranitar")
            assert ttar is not None, "Tyranitar unavailable for the pivot-back"
            return self.create_order(ttar)
        explode_now = v1._faints(battle.team) <= v1._faints(battle.opponent_team)
        want = ("explosion", "protect") if explode_now else ("protect",)
        for mid in want:
            mv = v1._first(battle.available_moves, lambda m: m.id == mid)
            if mv:
                return self.create_order(mv)
        return self.choose_default_move()


class OurSideFastV2(v1.OurSideFast):
    """v1's model-free MC arm on the v2 choreography."""

    def choose_move(self, battle):
        st = self._st.setdefault(battle.battle_tag, _fresh_st())
        kind, payload = _CoreV2.classify(battle, st)
        if kind == "switch":
            nxt = (v1._first(battle.available_switches, lambda m: m.species != "starmie")
                   or (battle.available_switches[0] if battle.available_switches else None))
            return self.create_order(nxt) if nxt else self.choose_default_move()
        if kind == "decision":
            v1._OurScriptCore.assert_decision_state(battle, self._expected_h,
                                                    self._expected_our_hp)
            st["captured"] = True
            self.decisions_reached += 1
            payload = self._forced_move
        mv = v1._first(battle.available_moves, lambda m: m.id == payload)
        assert mv is not None, (  # fail loud — v1's pump-label lesson
            f"scripted move {payload!r} not available: "
            f"{[m.id for m in battle.available_moves]}")
        return self.create_order(mv)


class OurSideModelV2(v1.OurSideModel):
    """v1's model-backed capture arm on the v2 choreography, plus the reveal assert."""

    def __init__(self, *args, expected_reveals: frozenset, **kwargs):
        super().__init__(*args, **kwargs)
        self._expected_reveals = expected_reveals

    def choose_move(self, battle):
        st = self._st.setdefault(battle.battle_tag, _fresh_st())
        kind, payload = _CoreV2.classify(battle, st)
        obs_dict = self.embed_battle(battle)   # keeps tracker/obs history genuine
        legal = self._get_tracker(battle).last_ctx.legal

        if kind == "switch":
            idx = None
            for sw in legal.switches:
                if sw.species != "starmie":
                    idx = sw.slot
                    break
            if idx is None and legal.switches:
                idx = legal.switches[0].slot
            if idx is None:
                return self.choose_default_move()
            self._get_tracker(battle).advance(idx)
            return self.action_to_order(idx, battle)

        if kind == "decision":
            board = v1._OurScriptCore.assert_decision_state(
                battle, self._expected_h, self._expected_our_hp)
            revealed = frozenset(battle.opponent_active_pokemon.moves)
            assert revealed == self._expected_reveals, (
                f"revealed {sorted(revealed)} != expected "
                f"{sorted(self._expected_reveals)}")
            st["captured"] = True
            idx, capture = self._capture(battle, obs_dict, board)
            capture["opp_revealed_moves"] = sorted(revealed)
        else:
            if payload == "endeavor":
                st["endeavored"] = True
            idx = self._action_for_move(battle, payload)
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)

# ---------------------------------------------------------------------------- runners


def _accounts(tag):
    return (AccountConfiguration(f"R2r{tag}"[:17], None),
            AccountConfiguration(f"R2o{tag}"[:17], None))


def run_capture_v2(model, mappings, h, tag, *, revealed: bool,
                   seed=(7, 11, 13, 17)) -> dict:
    sink = []
    a1, a2 = _accounts(tag)
    ours = OurSideModelV2(
        model=model, team=our_team_v2(h), battle_format=FORMAT,
        server_configuration=LocalhostServerConfiguration, mappings=mappings,
        account_configuration=a1, start_listening=False, stochastic=False,
        expected_h=h, expected_our_hp=262, capture_sink=sink,
        expected_reveals=EXPECT_SHOWN if revealed else EXPECT_HIDDEN)
    theirs = TtarSideV2(battle_format=FORMAT, team=THEIR_TEAM_V2,
                        server_configuration=LocalhostServerConfiguration,
                        account_configuration=a2, start_listening=False,
                        reveal_plan=REVEAL_SHOWN if revealed else REVEAL_HIDDEN)
    asyncio.run(run_local_battles(ours, theirs, 1, battle_format=FORMAT,
                                  seed=list(seed), impl="node"))
    assert len(sink) == 1, f"decision reached {len(sink)} times (wanted exactly 1)"
    battle = next(iter(ours.battles.values()))
    cap = sink[0]
    cap["battle_won"] = bool(battle.won)
    cap["final_turn"] = int(battle.turn)
    return cap


def run_mc_v2(forced_move, h, n, tag) -> dict:
    """Model-free MC on the v2 (hidden-plan) choreography — the physics is identical in
    both conditions, so one MC verifies both."""
    a1, a2 = _accounts(tag)
    ours = OurSideFastV2(battle_format=FORMAT, team=our_team_v2(h),
                         server_configuration=LocalhostServerConfiguration,
                         account_configuration=a1, start_listening=False,
                         forced_move=forced_move, expected_h=h, expected_our_hp=262,
                         max_concurrent_battles=2)
    theirs = TtarSideV2(battle_format=FORMAT, team=THEIR_TEAM_V2,
                        server_configuration=LocalhostServerConfiguration,
                        account_configuration=a2, start_listening=False,
                        reveal_plan=REVEAL_HIDDEN, max_concurrent_battles=2)
    asyncio.run(run_local_battles(ours, theirs, n, battle_format=FORMAT,
                                  seed=None, impl="node", concurrency=2))
    assert ours.decisions_reached == n, (
        f"decision reached {ours.decisions_reached}/{n} battles")
    wins = sum(1 for b in ours.battles.values() if b.won)
    lo, hi = v1._wilson(wins, n)
    return {"n": n, "wins": wins, "p_win": round(wins / n, 4),
            "wilson95": [round(lo, 4), round(hi, 4)]}

# ---------------------------------------------------------------------------- sweep/analysis


def sweep_hs(surf_rolls) -> list:
    """v1's 17 points (16/16 .. 0/16) + micro-steps bracketing the k=13 -> k=12
    crossover (true indifference: k/16 = (0.8 - 1/16)/(15/16) = 0.7867)."""
    hs = [284] + [r + 1 for r in surf_rolls] + [294, 295, 297, 298, 299]
    return sorted(set(hs))


def row_from_capture(cap, h, surf_rolls) -> dict:
    k = sum(1 for r in surf_rolls if r >= h)
    slots = cap["move_slots"]
    alpha = {a["name"]: a["p"] for a in (cap.get("opp_intent") or {}).get("alpha", [])}
    return {"H": h, "hp_frac": round(h / v1.TTAR_MAXHP, 4),
            "opp_hp_display": cap["board"]["opp_hp_display"],
            "surf_ko_rolls": k,
            "e_ko_surf": round(CRIT_P + (1 - CRIT_P) * k / 16, 4),
            "e_ko_pump": 0.8,
            "p_surf": cap["probs"][6 + slots.index("surf")],
            "p_pump": cap["probs"][6 + slots.index("hydropump")],
            "argmax": (cap["move_slots"][cap["argmax_action"] - 6]
                       if cap["argmax_action"] >= 6 else cap["argmax_action"]),
            "value": cap["value"],
            "win_prob": round(cap["win_prob"], 4) if cap["win_prob"] is not None else None,
            "alpha_eq": alpha.get("Earthquake")}


def analyze_condition(rows) -> dict:
    rows = sorted(rows, key=lambda r: r["H"])
    delta = np.array([r["e_ko_surf"] - r["e_ko_pump"] for r in rows])
    p_surf = np.array([r["p_surf"] for r in rows])

    # (1) mask mass at true equality: interpolate P(surf) in delta across the H-adjacent
    # pair straddling delta = 0 (the k=13 -> k=12 truth step, H=295 -> H=296).
    lo = max((r for r in rows if r["e_ko_surf"] >= 0.8), key=lambda r: r["H"])
    hi = min((r for r in rows if r["e_ko_surf"] < 0.8), key=lambda r: r["H"])
    d0, d1 = lo["e_ko_surf"] - 0.8, hi["e_ko_surf"] - 0.8
    p_at_eq = lo["p_surf"] + (0.0 - d0) * (hi["p_surf"] - lo["p_surf"]) / (d1 - d0)

    # (2) argmax flip threshold (or censored bound)
    flips = [r for r in rows if r["argmax"] != "surf"]
    max_deficit = round(0.8 - min(r["e_ko_surf"] for r in rows), 4)
    if flips:
        first = min(flips, key=lambda r: 0.8 - r["e_ko_surf"])
        flip = {"flipped": True, "deficit_at_first_flip": round(0.8 - first["e_ko_surf"], 4),
                "H_at_first_flip": first["H"]}
    else:
        flip = {"flipped": False, "censored_lower_bound_wp": max_deficit,
                "note": "argmax stays Surf at every point; the flip threshold exceeds "
                        f"a true Surf deficit of {max_deficit} win prob"}

    # (3) amplitude: OLS slope of P(surf) on truth-delta, global + crossover-local
    def ols(mask):
        x, y = delta[mask], p_surf[mask]
        b = float(np.polyfit(x, y, 1)[0])
        return round(b, 4)
    local = np.abs(delta) <= 0.16
    span_ratio = float((p_surf.max() - p_surf.min()) / (delta.max() - delta.min()))
    # in-band HP resolution: P(surf) spread across micro-steps of constant truth
    bands = {}
    for r in rows:
        bands.setdefault(r["surf_ko_rolls"], []).append(r["p_surf"])
    band_spread = {str(k): round(max(v) - min(v), 4)
                   for k, v in sorted(bands.items()) if len(v) > 1}
    return {"p_surf_at_true_equality": round(float(p_at_eq), 4),
            "equality_bracket": {"H_below": lo["H"], "p_surf_below": lo["p_surf"],
                                 "H_above": hi["H"], "p_surf_above": hi["p_surf"]},
            "argmax_flip": flip,
            "slope_global": ols(np.ones_like(delta, bool)),
            "slope_local_pm016": ols(local),
            "span_ratio": round(span_ratio, 4),
            "in_band_p_surf_spread": band_spread}

# ---------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["smoke", "capture", "sweep", "mc", "analyze", "all"])
    ap.add_argument("--mc-n", type=int, default=250)
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    v1_data = json.loads(V1_JSON.read_text())
    surf_rolls = v1_data["tables"]["surf_vs_cbtar"]["rolls"]
    assert len(surf_rolls) == 16, "v1 roll table missing/short"

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    def save():
        out_path.write_text(json.dumps(results, indent=1))
        print(f"[saved {out_path}]")

    results.setdefault("meta", {
        "date": "2026-08-31", "format": FORMAT, "impl": "node",
        "checkpoint": CKPT, "checkpoint_lineage": CKPT2,
        "capture_seed": [7, 11, 13, 17],
        "v1_artifact": V1_JSON.name,
        "reveal_plans": {"hidden": list(REVEAL_HIDDEN), "revealed": list(REVEAL_SHOWN)},
        "ttar_moves": ["earthquake", "rockslide", "focuspunch", "doubleedge"],
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
    })

    mappings = load_mappings()
    ver = current_model_version(mappings)

    def load(ck):
        m, _ = load_foreign_opponent(ck, current_version=ver, device="cpu")
        return m

    model = None

    if args.phase == "smoke":
        model = load(CKPT)
        for rev in (False, True):
            cap = run_capture_v2(model, mappings, BASE_H, f"k{int(rev)}", revealed=rev)
            print(f"revealed={rev}: turn={cap['board']['turn']} "
                  f"reveals={cap['opp_revealed_moves']} probs={cap['probs'][6:10]} "
                  f"V={cap['value']} wp={cap['win_prob']:.4f}")
        return

    if args.phase in ("capture", "all"):
        print("== ext1: base captures, hidden vs revealed, both checkpoints ==")
        results.setdefault("ext1", {})
        if "hidden" not in results["ext1"] or "revealed" not in results["ext1"]:
            model = model or load(CKPT)
            for rev, name in ((False, "hidden"), (True, "revealed")):
                if name not in results["ext1"]:
                    results["ext1"][name] = run_capture_v2(
                        model, mappings, BASE_H, f"e{int(rev)}", revealed=rev)
                    print(name, results["ext1"][name]["probs"][6:10],
                          "V", results["ext1"][name]["value"])
                    save()
        if "rev1" not in results["ext1"]:
            try:
                m2 = load(CKPT2)
                results["ext1"]["rev1"] = {
                    "checkpoint": CKPT2,
                    "hidden": run_capture_v2(m2, mappings, BASE_H, "r0", revealed=False),
                    "revealed": run_capture_v2(m2, mappings, BASE_H, "r1", revealed=True),
                }
                del m2
            except Exception as e:  # lineage is optional; never blocks the probe
                results["ext1"]["rev1"] = {"error": str(e)[:400]}
            save()

    if args.phase in ("sweep", "all"):
        print("== ext2: the 2 x sweep grid ==")
        model = model or load(CKPT)
        hs = sweep_hs(surf_rolls)
        results.setdefault("sweep", {"hidden": [], "revealed": []})
        for rev, name in ((False, "hidden"), (True, "revealed")):
            done = {r["H"] for r in results["sweep"][name]}
            for i, h in enumerate(hs):
                if h in done:
                    continue
                cap = run_capture_v2(model, mappings, h, f"{name[0]}{i}", revealed=rev)
                row = row_from_capture(cap, h, surf_rolls)
                results["sweep"][name].append(row)
                print(name, row)
                save()
            results["sweep"][name].sort(key=lambda r: r["H"])
            save()

    if args.phase in ("mc", "all"):
        print("== MC verification: crossover-adjacent cells (k=13 and k=12) ==")
        results.setdefault("mc", {})
        cells = (("h295_surf", "surf", 295), ("h295_pump", "hydropump", 295),
                 ("h299_surf", "surf", 299), ("h299_pump", "hydropump", 299))
        for key, mv, h in cells:
            if key not in results["mc"]:
                results["mc"][key] = run_mc_v2(mv, h, args.mc_n, key)
                k = sum(1 for r in surf_rolls if r >= h)
                results["mc"][key]["analytic"] = (
                    round(CRIT_P + (1 - CRIT_P) * k / 16, 4) if mv == "surf" else 0.8)
                print(key, results["mc"][key])
                save()

    if args.phase in ("analyze", "all"):
        print("== bias numbers ==")
        bias = {}
        for name in ("hidden", "revealed"):
            bias[name] = analyze_condition(results["sweep"][name])
        hid, rev = bias["hidden"], bias["revealed"]
        bias["revealed_minus_hidden"] = {
            "p_surf_at_true_equality": round(
                rev["p_surf_at_true_equality"] - hid["p_surf_at_true_equality"], 4),
            "slope_global": round(rev["slope_global"] - hid["slope_global"], 4),
            "slope_local_pm016": round(
                rev["slope_local_pm016"] - hid["slope_local_pm016"], 4),
            "flip": "both censored" if not (hid["argmax_flip"]["flipped"]
                                            or rev["argmax_flip"]["flipped"])
                    else "see per-condition argmax_flip",
        }
        # the value-stack delta across the whole sweep (paired by H)
        h_rows = {r["H"]: r for r in results["sweep"]["hidden"]}
        pair = [(h_rows[r["H"]], r) for r in results["sweep"]["revealed"]
                if r["H"] in h_rows]
        bias["value_stack_delta_mean"] = {
            "d_p_surf": round(float(np.mean([b["p_surf"] - a["p_surf"]
                                             for a, b in pair])), 4),
            "d_value": round(float(np.mean([b["value"] - a["value"]
                                            for a, b in pair])), 4),
            "d_win_prob": round(float(np.mean([b["win_prob"] - a["win_prob"]
                                               for a, b in pair])), 4),
            "n_pairs": len(pair),
        }
        results["bias"] = bias
        print(json.dumps(bias, indent=1))
        save()

    print("done.")


if __name__ == "__main__":
    main()
