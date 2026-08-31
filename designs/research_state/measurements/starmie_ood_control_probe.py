"""STARMIE/TYRANITAR OOD CONTROL — the constructed risk probe repeated at COMMON faint counts.

The objection this answers (owner, 2026-08-31): `starmie_ttar_risk_probe{,_v2}` read the policy's
mask at a decision with FIVE fainted mons on each side (a 1v1 endgame — the extreme tail of the
attrition distribution) reached by an ENGINEERED prelude (a 22-turn explosion-into-Protect parade,
in v2 preceded by a nine-turn Protect/pivot reveal phase). Both the BOARD and the HISTORY blocks of
that observation are therefore off-distribution, and the dramatic findings (mask mass 0.787-0.803 at
true indifference vs 0.5 unbiased, argmax never flips, zero excess response at the roll boundary,
amplitude 0.19-0.40) may be artifacts of that.

This probe re-runs the SAME engineered gamble (Surf 100% acc / 13-of-16 KO rolls vs Hydro Pump 80%
acc / every roll KOs, at an Endeavor-set exact Tyranitar HP) at faint counts F = 5, 3, 2 (plus
base-cell reads at F = 4, 1) using ONE construction, so F is the only manipulated variable.

TWO controls in one, because the prelude is NEW:

  * F = 5 here vs the v2 F = 5 already banked  ->  HISTORY sensitivity at a matched board.
    The prelude is a different, much shorter, much more ordinary sequence: no Protect anywhere, no
    reveal phase, 4-8 turns instead of 22, made of switches / attacks / KOs / Explosions.
  * F = 2, 3 vs F = 5 here  ->  BOARD (faint-count) sensitivity at a matched prelude mechanism.

THE CONSTRUCTION (deterministic; every leg either probe-verified in v1 or asserted at the decision):

  T1  Marshtomp (Jolly, max HP == the engineered H, faster than the 0-Spe Tyranitar) ENDEAVORS
      Tyranitar -> Ttar HP := H exactly. Ttar's CB Earthquake takes Marshtomp to H - (165..193).
  T2  Marshtomp Growls; Tyranitar SWITCHES OUT to Koffing (the switch resolves first, so the Growl
      lands on Koffing). Tyranitar is never touched again until the decision, and its Choice lock
      clears on the switch.
  T3  Marshtomp Growls; Koffing EXPLODES -> both faint.                       faints 1/1
  T4  our filler 1 Tackles; Weezing EXPLODES -> both faint.                   faints 2/2
  T5  our filler 2 ...; Graveler EXPLODES -> both faint.                      faints 3/3
  T6  Pineco     -> 4/4
  T7  Exeggutor  -> 5/5
  At the F-th step BOTH sides force-switch in the same turn: we choose Starmie, they choose
  Tyranitar. That entry mechanism is IDENTICAL at every F (at F = 5 it is the only legal choice),
  so Starmie always enters after residuals at exactly 262 HP with zero sandstorm ticks, and
  Tyranitar always enters unlocked at exactly H. The decision is the next turn.

Faints come in PAIRS because every step is an Explosion that kills its target — no Protect, hence
no gen3 stall-counter coin flip (a second consecutive Protect succeeds only 1/3 of the time, which
is what made v1's chain stochastic in its middle). Their exploders carry 252 Atk / Adamant so every
Explosion is guaranteed overkill on our deliberately frail fillers and on the chipped Marshtomp.

WHAT CHANGES AT F < 5, STATED UP FRONT (these are properties of the question, not defects):
  1. SWITCHES ARE LEGAL at the decision, so probability mass leaks out of the two moves. The
     headline statistic is therefore the RENORMALIZED P(Surf | Surf or Hydro Pump); the raw mask is
     reported beside it. At F = 5 the two coincide to ~1e-5.
  2. A KO IS NO LONGER A WIN. The engineered x-axis is E[KO|Surf] - E[KO|Pump], which at F = 5
     equals the win-probability delta and at F < 5 does not. Every "truth" column is labelled as a
     KO probability.

Run (from the repo root; needs deps/pokemon-showdown built - node bridge):
    python designs/research_state/measurements/starmie_ood_control_probe.py [--phase all]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Phases: smoke | capture | sweep | v2obs | analyze | all. Writes
starmie_ood_control_2026-08-31.json (resumable, incremental) and starmie_ood_control_obs.npz
(the captured observation vectors, for the history-block OOD comparison) next to this file.
CPU-only, <=2 cores. v1's and v2's artifacts are READ ONLY and are never modified.
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

import starmie_ttar_risk_probe as v1        # noqa: E402  (the committed v1 construction)
import starmie_ttar_risk_probe_v2 as v2     # noqa: E402  (the committed v2 sweep + analysis)

import numpy as np                          # noqa: E402

from poke_env.ps_client import AccountConfiguration                              # noqa: E402
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration  # noqa: E402

from agents.model.snapshot import current_model_version, load_foreign_opponent   # noqa: E402
from agents.observation.state_encoder import load_mappings                       # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles                   # noqa: E402

V1_JSON = OUT_DIR / "starmie_ttar_risk_probe_2026-08-30.json"
V2_JSON = OUT_DIR / "starmie_ttar_risk_probe_v2_2026-08-31.json"
OUT_JSON = OUT_DIR / "starmie_ood_control_2026-08-31.json"
OUT_NPZ = OUT_DIR / "starmie_ood_control_obs.npz"

FORMAT = v1.FORMAT
CKPT = v1.CKPT
CKPT2 = v1.CKPT2
BASE_H = v1.BASE_H
CRIT_P = v1.CRIT_P
TTAR_MAXHP = v1.TTAR_MAXHP

# Seeds are tried in order; the first one whose battle satisfies every decision-state assert wins.
# A turn-1 CRITICAL Earthquake kills Marshtomp outright (2 x 193 > any swept H) and desynchronises
# the chain -- the assert catches it and the next seed is tried, so the result stays reproducible.
SEEDS = ([7, 11, 13, 17], [3, 5, 7, 11], [19, 23, 29, 31], [2, 3, 5, 7], [41, 43, 47, 53])

OUR_FILLERS = ("rattata", "sentret", "zigzagoon", "poochyena")
THEIR_FILLERS = ("koffing", "weezing", "graveler", "pineco", "exeggutor")

# ---------------------------------------------------------------------------- teams

_EXPLODER = ("{name}\nAbility: {ability}\nLevel: 100\n"
             "EVs: 252 Atk\nAdamant Nature\n- Explosion\n- Protect")

THEIR_TEAM = "\n\n".join([
    ("Tyranitar @ Choice Band\nAbility: Sand Stream\nLevel: 100\n"
     "EVs: 4 HP / 252 Atk\nAdamant Nature\n- Earthquake\n- Roar"),
    _EXPLODER.format(name="Koffing", ability="Levitate"),
    _EXPLODER.format(name="Weezing", ability="Levitate"),
    _EXPLODER.format(name="Graveler", ability="Sturdy"),
    _EXPLODER.format(name="Pineco", ability="Sturdy"),
    _EXPLODER.format(name="Exeggutor", ability="Chlorophyll"),
])

_FRAIL = "{name}\nAbility: {ability}\nLevel: 100\n- Tackle\n- Protect"


def our_team(h: int) -> str:
    marsh = (f"Marshtomp\nAbility: Torrent\nLevel: 100\n"
             f"EVs: {v1.marsh_hp_ev_for(h)} HP / 252 Spe\nJolly Nature\n- Endeavor\n- Growl")
    fillers = "\n\n".join([
        _FRAIL.format(name="Rattata", ability="Run Away"),
        _FRAIL.format(name="Sentret", ability="Run Away"),
        _FRAIL.format(name="Zigzagoon", ability="Pickup"),
        _FRAIL.format(name="Poochyena", ability="Run Away"),
    ])
    return f"{marsh}\n\n{fillers}\n\n{v1.STARMIE_FRAIL}"

# ---------------------------------------------------------------------------- our side


def _fresh() -> dict:
    return {"endeavored": False, "captured": False}


class _Core:
    """The scripted decision logic, shared by the model-backed and model-free arms."""

    @staticmethod
    def classify(battle, st) -> tuple:
        me = battle.active_pokemon
        if battle.force_switch or not battle.available_moves or me is None:
            return ("switch", None)
        if me.species == "marshtomp":
            if not st["endeavored"]:
                return ("move", "endeavor")
            return ("move", "growl")
        if me.species in OUR_FILLERS:
            return ("move", "tackle")
        # Starmie
        opp = battle.opponent_active_pokemon
        if opp is not None and opp.species == "tyranitar" and not st["captured"]:
            return ("decision", None)
        return ("move", "surf")

    @staticmethod
    def switch_target(battle, faints: int):
        """Starmie once BOTH sides have reached the target faint count, else the next filler."""
        ours = v1._faints(battle.team)
        theirs = v1._faints(battle.opponent_team)
        if ours >= faints and theirs >= faints:
            star = v1._first(battle.available_switches, lambda m: m.species == "starmie")
            if star is not None:
                return star
        for sp in OUR_FILLERS:
            m = v1._first(battle.available_switches, lambda m: m.species == sp)
            if m is not None:
                return m
        return battle.available_switches[0] if battle.available_switches else None

    @staticmethod
    def assert_decision_state(battle, h: int, faints: int) -> dict:
        ours = v1._faints(battle.team)
        theirs = v1._faints(battle.opponent_team)
        opp = battle.opponent_active_pokemon
        me = battle.active_pokemon
        assert ours == faints and theirs == faints, f"faints {ours}/{theirs} != {faints}/{faints}"
        assert opp is not None and opp.species == "tyranitar", "opp active is not Tyranitar"
        assert me is not None and me.species == "starmie", "our active is not Starmie"
        assert opp.max_hp == TTAR_MAXHP, f"ttar maxhp {opp.max_hp} != {TTAR_MAXHP}"
        assert opp.current_hp == h, f"opp hp {opp.current_hp} != engineered H={h}"
        assert me.current_hp == 262, f"our hp {me.current_hp} != 262 (full)"
        return {"our_hp": int(me.current_hp), "our_maxhp": int(me.max_hp),
                "opp_hp_display": int(opp.current_hp), "turn": int(battle.turn),
                "our_faints": int(ours), "opp_faints": int(theirs),
                "our_alive": int(6 - ours), "opp_alive": int(6 - theirs),
                "n_legal_switches": int(len(battle.available_switches)),
                "opp_revealed_moves": sorted(opp.moves)}


class OurSideModel(v1.OurSideModel):
    """v1's capture machinery on the OOD-control choreography, with a parameterised faint count."""

    def __init__(self, *args, faints: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._faints = faints

    def choose_move(self, battle):
        st = self._st.setdefault(battle.battle_tag, _fresh())
        kind, payload = _Core.classify(battle, st)
        obs_dict = self.embed_battle(battle)   # keeps tracker/obs history genuine
        legal = self._get_tracker(battle).last_ctx.legal

        if kind == "switch":
            tgt = _Core.switch_target(battle, self._faints)
            idx = None
            if tgt is not None:
                for sw in legal.switches:
                    if sw.species == tgt.species:
                        idx = sw.slot
                        break
            if idx is None and legal.switches:
                idx = legal.switches[0].slot
            if idx is None:
                return self.choose_default_move()
            self._get_tracker(battle).advance(idx)
            return self.action_to_order(idx, battle)

        if kind == "decision":
            board = _Core.assert_decision_state(battle, self._expected_h, self._faints)
            st["captured"] = True
            idx, capture = self._capture(battle, obs_dict, board)
            capture["faints"] = self._faints
        else:
            if payload == "endeavor":
                st["endeavored"] = True
            idx = self._action_for_move(battle, payload)
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)


class TtarSide(v1.Player):
    """Their side: Earthquake once, pivot out, then one Explosion per turn; Tyranitar returns as
    the forced replacement on the turn the target faint count is reached."""

    def __init__(self, *args, faints: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._faints = faints
        self._st = {}

    def choose_move(self, battle):
        st = self._st.setdefault(battle.battle_tag, {"ttar_acted": 0, "pivoted": False})
        me = battle.active_pokemon
        ours = v1._faints(battle.team)          # THEIR faints (this player's own team)
        theirs = v1._faints(battle.opponent_team)

        if battle.force_switch or not battle.available_moves:
            if ours >= self._faints and theirs >= self._faints:
                ttar = v1._first(battle.available_switches, lambda m: m.species == "tyranitar")
                if ttar is not None:
                    return self.create_order(ttar)
            for sp in THEIR_FILLERS:
                m = v1._first(battle.available_switches, lambda m: m.species == sp)
                if m is not None:
                    return self.create_order(m)
            return (self.create_order(battle.available_switches[0])
                    if battle.available_switches else self.choose_default_move())

        if me is not None and me.species == "tyranitar":
            if st["ttar_acted"] >= 1 and not st["pivoted"]:
                filler = v1._first(battle.available_switches,
                                   lambda m: m.species in THEIR_FILLERS)
                assert filler is not None, "no filler available for the Tyranitar pivot"
                st["pivoted"] = True
                return self.create_order(filler)
            eq = v1._first(battle.available_moves, lambda m: m.id == "earthquake")
            assert eq is not None, f"EQ unavailable: {[m.id for m in battle.available_moves]}"
            st["ttar_acted"] += 1
            return self.create_order(eq)

        boom = v1._first(battle.available_moves, lambda m: m.id == "explosion")
        assert boom is not None, f"explosion unavailable: {[m.id for m in battle.available_moves]}"
        return self.create_order(boom)

# ---------------------------------------------------------------------------- runner


def _accounts(tag):
    return (AccountConfiguration(f"Or{tag}"[:17], None),
            AccountConfiguration(f"Oo{tag}"[:17], None))


def run_capture(model, mappings, h, faints, tag) -> dict:
    """One battle per (H, faints) cell. Seeds are tried in a FIXED order until every
    decision-state assert holds, so the cell is reproducible without being luck-dependent."""
    last = None
    for si, seed in enumerate(SEEDS):
        sink = []
        a1, a2 = _accounts(f"{tag}{si}")
        ours = OurSideModel(
            model=model, team=our_team(h), battle_format=FORMAT,
            server_configuration=LocalhostServerConfiguration, mappings=mappings,
            account_configuration=a1, start_listening=False, stochastic=False,
            expected_h=h, expected_our_hp=262, capture_sink=sink, faints=faints)
        theirs = TtarSide(battle_format=FORMAT, team=THEIR_TEAM,
                          server_configuration=LocalhostServerConfiguration,
                          account_configuration=a2, start_listening=False, faints=faints)
        try:
            asyncio.run(run_local_battles(ours, theirs, 1, battle_format=FORMAT,
                                          seed=list(seed), impl="node"))
            assert len(sink) == 1, f"decision reached {len(sink)} times (wanted exactly 1)"
        except Exception as e:                     # noqa: BLE001 - retry on a different dice stream
            last = f"seed {seed}: {type(e).__name__}: {str(e)[:200]}"
            continue
        cap = sink[0]
        cap["seed"] = list(seed)
        cap["seed_attempt"] = si
        return cap
    raise RuntimeError(f"no seed produced the engineered state at H={h}, F={faints}: {last}")

# ---------------------------------------------------------------------------- rows / analysis


def row_from_capture(cap, h, surf_rolls) -> dict:
    k = sum(1 for r in surf_rolls if r >= h)
    slots = cap["move_slots"]
    p_surf = cap["probs"][6 + slots.index("surf")]
    p_pump = cap["probs"][6 + slots.index("hydropump")]
    alpha = {a["name"]: a["p"] for a in (cap.get("opp_intent") or {}).get("alpha", [])}
    argmax = (cap["move_slots"][cap["argmax_action"] - 6]
              if cap["argmax_action"] >= 6 else f"switch{cap['argmax_action']}")
    return {"H": h, "hp_frac": round(h / TTAR_MAXHP, 4),
            "opp_hp_display": cap["board"]["opp_hp_display"],
            "surf_ko_rolls": k,
            "e_ko_surf": round(CRIT_P + (1 - CRIT_P) * k / 16, 4),
            "e_ko_pump": 0.8,
            "p_surf": p_surf, "p_pump": p_pump,
            "p_surf_rn": round(p_surf / (p_surf + p_pump), 6),
            "p_move_total": round(sum(cap["probs"][6:10]), 6),
            "p_switch_total": round(sum(cap["probs"][0:6]), 6),
            "argmax": argmax,
            "argmax_between": "surf" if p_surf >= p_pump else "hydropump",
            "value": cap["value"],
            "win_prob": round(cap["win_prob"], 4) if cap["win_prob"] is not None else None,
            "alpha_eq": alpha.get("Earthquake"),
            "our_alive": cap["board"]["our_alive"], "opp_alive": cap["board"]["opp_alive"],
            "n_legal_switches": cap["board"]["n_legal_switches"],
            "turn": cap["board"]["turn"], "seed_attempt": cap.get("seed_attempt", 0)}


def _ols(x, y) -> float:
    return round(float(np.polyfit(x, y, 1)[0]), 4)


def analyze_arm(rows) -> dict:
    """The three v2 bias numbers, computed on BOTH the raw mask and the renormalized
    P(Surf | Surf or Pump). The renormalized column is the one comparable across faint counts."""
    rows = sorted(rows, key=lambda r: r["H"])
    delta = np.array([r["e_ko_surf"] - r["e_ko_pump"] for r in rows])
    out = {"n_points": len(rows),
           "p_switch_total_mean": round(float(np.mean([r["p_switch_total"] for r in rows])), 4),
           "p_move_total_mean": round(float(np.mean([r["p_move_total"] for r in rows])), 4)}

    lo = max((r for r in rows if r["e_ko_surf"] >= 0.8), key=lambda r: r["H"])
    hi = min((r for r in rows if r["e_ko_surf"] < 0.8), key=lambda r: r["H"])
    d0, d1 = lo["e_ko_surf"] - 0.8, hi["e_ko_surf"] - 0.8

    for key, label in (("p_surf", "raw"), ("p_surf_rn", "renorm")):
        y = np.array([r[key] for r in rows])
        p_at_eq = lo[key] + (0.0 - d0) * (hi[key] - lo[key]) / (d1 - d0)
        local = np.abs(delta) <= 0.16
        out[label] = {
            "p_surf_at_true_equality": round(float(p_at_eq), 4),
            "equality_bracket": {"H_below": lo["H"], "p_below": round(lo[key], 4),
                                 "H_above": hi["H"], "p_above": round(hi[key], 4)},
            "slope_global": _ols(delta, y),
            "slope_local_pm016": _ols(delta[local], y[local]),
            "span_ratio": round(float((y.max() - y.min()) / (delta.max() - delta.min())), 4),
            "p_at_best_surf": round(float(y[np.argmax(delta)]), 4),
            "p_at_worst_surf": round(float(y[np.argmin(delta)]), 4),
        }

    max_deficit = round(0.8 - min(r["e_ko_surf"] for r in rows), 4)
    for key, label in (("argmax", "argmax_global"), ("argmax_between", "argmax_between")):
        flips = [r for r in rows if r[key] != "surf"]
        if flips:
            first = min(flips, key=lambda r: 0.8 - r["e_ko_surf"])
            out[label] = {"flipped": True,
                          "deficit_at_first_flip": round(0.8 - first["e_ko_surf"], 4),
                          "H_at_first_flip": first["H"], "flipped_to": first[key],
                          "n_flipped": len(flips)}
        else:
            out[label] = {"flipped": False, "censored_lower_bound_ko_deficit": max_deficit}

    # per-HP-point steps inside a constant-truth band: the v2 "prices the HP bar, not the roll
    # table" reading, recomputed here.
    bands = {}
    for r in rows:
        bands.setdefault(r["surf_ko_rolls"], []).append((r["H"], r["p_surf_rn"]))
    out["in_band_p_surf_rn_spread"] = {
        str(k): round(max(p for _, p in v) - min(p for _, p in v), 4)
        for k, v in sorted(bands.items()) if len(v) > 1}
    steps = []
    for a, b in zip(rows, rows[1:]):
        if b["H"] == a["H"] + 1:
            steps.append({"H": f"{a['H']}->{b['H']}", "same_k": a["surf_ko_rolls"] == b["surf_ko_rolls"],
                          "d_p_surf_rn_pp": round((b["p_surf_rn"] - a["p_surf_rn"]) * 100, 3)})
    out["unit_hp_steps"] = steps
    return out

# ---------------------------------------------------------------------------- main


def _save_obs(store: dict):
    if store:
        np.savez_compressed(OUT_NPZ, **{k: np.asarray(v, dtype=np.float32)
                                        for k, v in store.items()})
        print(f"[saved {OUT_NPZ}: {sorted(store)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["smoke", "capture", "sweep", "v2obs", "seedcheck", "analyze", "all"])
    ap.add_argument("--faints", default="5,3,2", help="comma-separated F values to SWEEP")
    ap.add_argument("--base-faints", default="5,4,3,2,1",
                    help="comma-separated F values for the base-cell capture")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    # every capture also stashes its raw observation, for the history-block OOD comparison
    _orig_capture = v1.OurSideModel._capture

    def _capture_with_obs(self, battle, obs_dict, board):
        idx, cap = _orig_capture(self, battle, obs_dict, board)
        cap["_obs"] = obs_dict["observation"]
        return idx, cap

    v1.OurSideModel._capture = _capture_with_obs

    v1_data = json.loads(V1_JSON.read_text())
    surf_rolls = v1_data["tables"]["surf_vs_cbtar"]["rolls"]
    assert len(surf_rolls) == 16, "v1 roll table missing/short"

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    obs_store = {}
    if OUT_NPZ.exists():
        with np.load(OUT_NPZ) as z:
            obs_store = {k: z[k] for k in z.files}

    def save():
        out_path.write_text(json.dumps(results, indent=1))
        print(f"[saved {out_path}]")

    results.setdefault("meta", {
        "date": "2026-08-31", "format": FORMAT, "impl": "node",
        "checkpoint": CKPT, "checkpoint_lineage": CKPT2,
        "seeds_tried_in_order": [list(s) for s in SEEDS],
        "v1_artifact": V1_JSON.name, "v2_artifact": V2_JSON.name,
        "our_fillers": list(OUR_FILLERS), "their_fillers": list(THEIR_FILLERS),
        "construction": "Endeavor -> Ttar pivots out -> one guaranteed-overkill Explosion per "
                        "turn, each killing its target, so faints advance 1/1 per turn; both "
                        "principals enter as FORCED replacements on the F-th step",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
    })

    mappings = load_mappings()
    ver = current_model_version(mappings)
    model = None

    def load(ck):
        m, _ = load_foreign_opponent(ck, current_version=ver, device="cpu")
        return m

    def strip(cap, key):
        obs = cap.pop("_obs", None)
        if obs is not None:
            obs_store[key] = obs
        return cap

    if args.phase == "smoke":
        model = load(CKPT)
        for f in (5, 2):
            cap = run_capture(model, mappings, BASE_H, f, f"sm{f}")
            print(f"F={f} turn={cap['board']['turn']} faints={cap['board']['our_faints']}/"
                  f"{cap['board']['opp_faints']} alive={cap['board']['our_alive']}/"
                  f"{cap['board']['opp_alive']} switches={cap['board']['n_legal_switches']} "
                  f"reveals={cap['board']['opp_revealed_moves']} probs={cap['probs']} "
                  f"V={cap['value']} wp={cap['win_prob']:.4f} seed={cap['seed']}")
        return

    if args.phase in ("capture", "all"):
        print("== base cells (H=295) across faint counts ==")
        results.setdefault("base_cells", {})
        for f in [int(x) for x in args.base_faints.split(",")]:
            key = f"F{f}"
            if key in results["base_cells"]:
                continue
            model = model or load(CKPT)
            cap = run_capture(model, mappings, BASE_H, f, f"b{f}")
            results["base_cells"][key] = strip(cap, f"base_F{f}")
            print(key, cap["probs"], "V", cap["value"], "wp", cap["win_prob"],
                  "turn", cap["board"]["turn"])
            save()
            _save_obs(obs_store)
        # lineage checkpoint, base cell only, at the extreme and a common faint count
        if "lineage_rev1" not in results:
            try:
                m2 = load(CKPT2)
                results["lineage_rev1"] = {
                    "checkpoint": CKPT2,
                    "F5": strip(run_capture(m2, mappings, BASE_H, 5, "l5"), "lineage_F5"),
                    "F2": strip(run_capture(m2, mappings, BASE_H, 2, "l2"), "lineage_F2"),
                }
                del m2
            except Exception as e:                                        # noqa: BLE001
                results["lineage_rev1"] = {"error": str(e)[:400]}
            save()
            _save_obs(obs_store)

    if args.phase in ("sweep", "all"):
        print("== the sweep grid, one arm per faint count ==")
        model = model or load(CKPT)
        hs = v2.sweep_hs(surf_rolls)
        results.setdefault("sweep", {})
        for f in [int(x) for x in args.faints.split(",")]:
            name = f"F{f}"
            results["sweep"].setdefault(name, [])
            done = {r["H"] for r in results["sweep"][name]}
            for i, h in enumerate(hs):
                if h in done:
                    continue
                cap = run_capture(model, mappings, h, f, f"{f}s{i}")
                strip(cap, f"sweep_F{f}_H{h}")
                row = row_from_capture(cap, h, surf_rolls)
                results["sweep"][name].append(row)
                print(name, row["H"], "k", row["surf_ko_rolls"], "p_surf", row["p_surf"],
                      "rn", row["p_surf_rn"], "sw", row["p_switch_total"])
                save()
            results["sweep"][name].sort(key=lambda r: r["H"])
            save()
            _save_obs(obs_store)

    if args.phase in ("v2obs", "all"):
        # ONE v2-prelude capture, purely to recover its observation vector for the history-block
        # OOD comparison (the v2 JSON records only an obs sha). v2's artifacts are not touched.
        if "v2_prelude_base" not in results:
            print("== v2-prelude base capture (obs recovery only) ==")
            model = model or load(CKPT)
            sink = []
            a1, a2 = _accounts("v2o")
            ours = v2.OurSideModelV2(
                model=model, team=v2.our_team_v2(BASE_H), battle_format=FORMAT,
                server_configuration=LocalhostServerConfiguration, mappings=mappings,
                account_configuration=a1, start_listening=False, stochastic=False,
                expected_h=BASE_H, expected_our_hp=262, capture_sink=sink,
                expected_reveals=v2.EXPECT_HIDDEN)
            theirs = v2.TtarSideV2(battle_format=FORMAT, team=v2.THEIR_TEAM_V2,
                                   server_configuration=LocalhostServerConfiguration,
                                   account_configuration=a2, start_listening=False,
                                   reveal_plan=v2.REVEAL_HIDDEN)
            asyncio.run(run_local_battles(ours, theirs, 1, battle_format=FORMAT,
                                          seed=[7, 11, 13, 17], impl="node"))
            assert len(sink) == 1, f"v2 decision reached {len(sink)} times"
            cap = sink[0]
            results["v2_prelude_base"] = strip(cap, "v2_prelude_F5")
            print("v2 prelude", cap["probs"][6:10], "turn", cap["board"]["turn"])
            save()
            _save_obs(obs_store)

    if args.phase == "seedcheck":
        # How much of a cell's number is CONSTRUCTION NOISE? The prelude is deterministic in its
        # events but not in its damage MAGNITUDES (the turn-1 Earthquake roll rides in the event
        # window), so the same engineered decision state is reached with a different history under
        # each dice stream. This bounds that.
        print("== seed robustness at the base cell ==")
        model = model or load(CKPT)
        results.setdefault("seed_robustness", {})
        for f in (1, 2, 5):
            key = f"F{f}"
            if key in results["seed_robustness"]:
                continue
            vals = []
            for si, seed in enumerate(SEEDS):
                sink = []
                a1, a2 = _accounts(f"z{f}{si}")
                ours = OurSideModel(
                    model=model, team=our_team(BASE_H), battle_format=FORMAT,
                    server_configuration=LocalhostServerConfiguration, mappings=mappings,
                    account_configuration=a1, start_listening=False, stochastic=False,
                    expected_h=BASE_H, expected_our_hp=262, capture_sink=sink, faints=f)
                theirs = TtarSide(battle_format=FORMAT, team=THEIR_TEAM,
                                  server_configuration=LocalhostServerConfiguration,
                                  account_configuration=a2, start_listening=False, faints=f)
                try:
                    asyncio.run(run_local_battles(ours, theirs, 1, battle_format=FORMAT,
                                                  seed=list(seed), impl="node"))
                    assert len(sink) == 1
                except Exception as e:                                    # noqa: BLE001
                    vals.append({"seed": list(seed), "error": str(e)[:160]})
                    continue
                cap = sink[0]
                cap.pop("_obs", None)
                sl = cap["move_slots"]
                ps = cap["probs"][6 + sl.index("surf")]
                pp = cap["probs"][6 + sl.index("hydropump")]
                vals.append({"seed": list(seed), "p_surf": ps,
                             "p_surf_rn": round(ps / (ps + pp), 6),
                             "value": cap["value"], "win_prob": cap["win_prob"],
                             "obs_sha": cap["obs_sha"]})
            ok = [v["p_surf_rn"] for v in vals if "p_surf_rn" in v]
            results["seed_robustness"][key] = {
                "cells": vals, "n_ok": len(ok),
                "p_surf_rn_min": round(min(ok), 4) if ok else None,
                "p_surf_rn_max": round(max(ok), 4) if ok else None,
                "p_surf_rn_spread": round(max(ok) - min(ok), 4) if ok else None}
            print(key, results["seed_robustness"][key]["p_surf_rn_min"],
                  results["seed_robustness"][key]["p_surf_rn_max"])
            save()
        return

    if args.phase in ("analyze", "all"):
        print("== bias numbers, per faint count ==")
        bias = {}
        for name, rows in sorted(results.get("sweep", {}).items()):
            if rows:
                bias[name] = analyze_arm(rows)
        # the v2 reference, recomputed with the SAME code on v2's banked hidden-condition sweep
        try:
            v2_rows = json.loads(V2_JSON.read_text())["sweep"]["hidden"]
            for r in v2_rows:
                r.setdefault("p_surf_rn", round(r["p_surf"] / (r["p_surf"] + r["p_pump"]), 6))
                r.setdefault("p_switch_total", 0.0)
                r.setdefault("p_move_total", 1.0)
                r.setdefault("argmax_between", "surf" if r["p_surf"] >= r["p_pump"] else "hydropump")
            bias["v2_F5_hidden_reference"] = analyze_arm(v2_rows)
        except Exception as e:                                            # noqa: BLE001
            bias["v2_F5_hidden_reference"] = {"error": str(e)[:300]}
        results["bias"] = bias
        print(json.dumps({k: {"renorm_at_equality": v.get("renorm", {}).get(
            "p_surf_at_true_equality"),
            "raw_at_equality": v.get("raw", {}).get("p_surf_at_true_equality"),
            "slope_renorm": v.get("renorm", {}).get("slope_global"),
            "flip": v.get("argmax_between")} for k, v in bias.items()}, indent=1))
        save()

    _save_obs(obs_store)
    print("done.")


if __name__ == "__main__":
    main()
