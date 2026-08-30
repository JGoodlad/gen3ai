"""STARMIE/TYRANITAR CONSTRUCTED RISK PROBE — the constructed-scenario complement to
`designs/ai_v12/probe_risk_modulation_capstone.md` (theory: `designs/learning/
temperature_mixing_and_risk.md` §3).

A hand-engineered 1v1 endgame reached inside a REAL bridge battle (so the observation is
genuine): five mons fainted per side, our Starmie vs their Choice Band Tyranitar, with
Tyranitar's exact current HP set by a scripted Endeavor so that Surf (100% acc) KOs on
exactly 13/16 damage rolls while Hydro Pump (80% acc) KOs on every roll — the same first
moment (E[KO] 0.8242 vs 0.8000, as close as gen3's discrete rolls + 1/16 crit allow),
different failure modes. We read the policy's masked probability distribution at that
decision, plus V(s), the win-prob head, and the alpha/beta opponent-intent posterior.

Construction (state-driven scripts — parity rules on the FAINT COUNTS, never turn
numbers, so a crit-shifted timeline self-heals; every leg verified by the damage probe):
  - our side: Marshtomp (Jolly, max HP == the engineered H) leads, Endeavors Tyranitar on
    turn 1 (acts first: Ttar runs 0 Spe) -> Ttar's HP := H exactly; Marshtomp then dies to
    CB Earthquakes. Fillers are an EXPLOSION-INTO-PROTECT chain (all distinct species —
    poke-env's ident re-keying cannot track duplicate species on one side): each turn one
    side's filler Explodes into the other side's first-use Protect (100% success; the
    exploder faints even when blocked), alternating by the faint-count parity rule
    (ours: explode iff our_faints < their_faints; theirs: explode iff theirs <= ours).
    Our last filler (Grimer, Poison, slower than Ttar) dies to a guaranteed-overkill
    super-effective CB EQ before its own move executes, so Tyranitar is never touched
    after the Endeavor. Starmie (slot 6) enters on the decision turn itself — zero
    sandstorm ticks, full HP.
  - their side: CB Tyranitar leads (Sand Stream), kills Marshtomp, voluntarily switches
    out for the filler chain, and is forced back in — at H, unlocked, boosts cleared —
    for the decision turn.
  - retaliation: CB Earthquake (Crunch is SPECIAL in gen3 — Dark type — so CB doesn't boost
    it; measured, not assumed). Base Starmie carries Def IV 0 so EQ KOs on 96/96 probe
    rolls: symmetric fatality. The RECOVERABLE variant's bulky Starmie survives exactly one
    non-crit EQ.

Run (from the repo root; needs deps/pokemon-showdown built — node bridge):
    python designs/research_state/measurements/starmie_ttar_risk_probe.py [--phase all]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Phases: tables | capture | mc | sweep | all (default). Writes
starmie_ttar_risk_probe_2026-08-30.json next to this file. CPU-only, <=2 cores.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

torch.set_num_threads(2)

from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.inference.player import RLPlayer
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from utils.bridge.local_battle_runner import run_local_battles
from utils.paths import src_path

OUT_DIR = Path(__file__).resolve().parent
PROBE_JS = str(src_path("utils", "bridge", "damage_probe.js"))
FORMAT = "gen3customgame"  # no species clause (duplicate Shedinja fillers); protocol stream
                           # is structurally identical to gen3ou for the parser/encoder.
CKPT = "/home/goodlad/dev/gen3ai/models/ai_v9_70_R3ACTION_0828/final_model.zip"
CKPT2 = "/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/final_model.zip"

TTAR_MAXHP = 342          # 4 HP EVs, 31 IV, Adamant, L100
BASE_H = 295              # engineered current HP: 13/16 surf rolls KO
CRIT_P = 1.0 / 16.0       # gen3 base crit stage

# ---------------------------------------------------------------------------- teams

OUR_FILLERS = ("golem", "forretress", "camerupt", "grimer")
THEIR_FILLERS = ("koffing", "weezing", "graveler", "pineco", "exeggutor")


def our_team(starmie_block: str, marsh_hp_ev: int) -> str:
    fillers = (
        "Golem\nAbility: Sturdy\nLevel: 100\n- Protect\n- Explosion\n\n"
        "Forretress\nAbility: Sturdy\nLevel: 100\n- Protect\n- Explosion\n\n"
        "Camerupt\nAbility: Magma Armor\nLevel: 100\n- Protect\n- Explosion\n\n"
        "Grimer\nAbility: Sticky Hold\nLevel: 100\n- Protect\n- Sludge")
    marsh = (f"Marshtomp\nAbility: Torrent\nLevel: 100\n"
             f"EVs: {marsh_hp_ev} HP / 252 Spe\nJolly Nature\n- Endeavor\n- Growl")
    return f"{marsh}\n\n{fillers}\n\n{starmie_block}"


STARMIE_FRAIL = ("Starmie @ Leftovers\nAbility: Natural Cure\nLevel: 100\n"
                 "EVs: 4 HP / 252 SpA / 252 Spe\nIVs: 0 Def\nModest Nature\n"
                 "- Surf\n- Hydro Pump\n- Ice Beam\n- Recover")
STARMIE_BULKY = ("Starmie @ Leftovers\nAbility: Natural Cure\nLevel: 100\n"
                 "EVs: 252 HP / 252 SpA / 4 Spe\nModest Nature\n"
                 "- Surf\n- Hydro Pump\n- Ice Beam\n- Recover")

THEIR_TEAM = ("Tyranitar @ Choice Band\nAbility: Sand Stream\nLevel: 100\n"
              "EVs: 4 HP / 252 Atk\nAdamant Nature\n- Earthquake\n- Roar\n\n"
              "Koffing\nAbility: Levitate\nLevel: 100\n- Protect\n- Explosion\n\n"
              "Weezing\nAbility: Levitate\nLevel: 100\n- Protect\n- Explosion\n\n"
              "Graveler\nAbility: Sturdy\nLevel: 100\n- Protect\n- Explosion\n\n"
              "Pineco\nAbility: Sturdy\nLevel: 100\n- Protect\n- Explosion\n\n"
              "Exeggutor\nAbility: Chlorophyll\nLevel: 100\n- Protect\n- Explosion")


def _faints(team) -> int:
    return sum(1 for m in team.values() if m.fainted)


def marsh_hp_ev_for(h: int) -> int:
    """Marshtomp max HP = 250 + 31 (IV) + floor(EV/4); Endeavor transfers it exactly."""
    ev = 4 * (h - 281)
    assert 0 <= ev <= 252 and 281 + ev // 4 == h, f"H={h} unreachable"
    return ev

# ---------------------------------------------------------------------------- players

def _first(seq, pred):
    for x in seq:
        if pred(x):
            return x
    return None


class TtarSide(Player):
    """Their side: state-driven script (never turn-indexed, so crit-shifted faint timing
    cannot desync it)."""

    def choose_move(self, battle):
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if battle.force_switch or not battle.available_moves:
            filler = _first(battle.available_switches,
                            lambda m: m.species in THEIR_FILLERS)
            target = filler or (battle.available_switches[0]
                                if battle.available_switches else None)
            return self.create_order(target) if target else self.choose_default_move()
        if me is not None and me.species == "tyranitar":
            if opp is not None and opp.species in OUR_FILLERS:
                filler = _first(battle.available_switches,
                                lambda m: m.species in THEIR_FILLERS)
                if filler:  # voluntary switch: start/continue the filler chain
                    return self.create_order(filler)
            eq = _first(battle.available_moves, lambda m: m.id == "earthquake")
            if eq:
                return self.create_order(eq)
            return self.choose_default_move()
        # a filler: explode iff our own faints <= their faints, else first-use Protect
        explode_now = _faints(battle.team) <= _faints(battle.opponent_team)
        want = ("explosion", "protect") if explode_now else ("protect",)
        for mid in want:
            mv = _first(battle.available_moves, lambda m: m.id == mid)
            if mv:
                return self.create_order(mv)
        return self.choose_default_move()


class _OurScriptCore:
    """Shared scripted-decision logic for both our player classes.

    Returns the desired (kind, payload): ("move", move_id) | ("switch", None) |
    ("decision", None) — "decision" means the engineered capture point was reached.
    """

    @staticmethod
    def classify(battle, st) -> tuple:
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if battle.force_switch or not battle.available_moves or me is None:
            return ("switch", None)
        if me.species == "marshtomp":
            if not st["endeavored"]:
                return ("move", "endeavor")
            return ("move", "growl")
        if me.species in OUR_FILLERS:
            explode_now = _faints(battle.team) < _faints(battle.opponent_team)
            if opp is not None and opp.species == "tyranitar":
                # Only Grimer (the always-last filler) ever reaches this in explode state;
                # it is slower than Ttar and dies to a guaranteed-overkill SE CB EQ before
                # the move executes — Tyranitar is never chipped. Never click Explosion here.
                return ("move", "sludge" if explode_now else "protect")
            return ("move", "explosion" if explode_now else "protect")
        # Starmie
        if opp is not None and opp.species == "tyranitar":
            if not st["captured"]:
                return ("decision", None)
            return ("move", "surf")  # continuation policy: always Surf after the decision
        return ("move", "recover")

    @staticmethod
    def assert_decision_state(battle, expected_h: int, expected_our_hp: int):
        ours = sum(1 for m in battle.team.values() if m.fainted)
        theirs = sum(1 for m in battle.opponent_team.values() if m.fainted)
        opp = battle.opponent_active_pokemon
        me = battle.active_pokemon
        assert ours == 5 and theirs == 5, f"faints {ours}/{theirs} != 5/5"
        assert opp.species == "tyranitar" and me.species == "starmie"
        # opp HP: the local bridge reports exact hp/maxhp; a live server would send /100
        disp = opp.current_hp
        if opp.max_hp == TTAR_MAXHP:
            assert disp == expected_h, f"opp hp {disp} != engineered H={expected_h}"
        else:
            want = round(expected_h / TTAR_MAXHP * opp.max_hp)
            assert abs(disp - want) <= 1, f"opp hp display {disp} != ~{want}"
        assert me.current_hp == expected_our_hp, (
            f"our hp {me.current_hp} != {expected_our_hp}")
        return {"our_hp": int(me.current_hp), "our_maxhp": int(me.max_hp),
                "opp_hp_display": int(disp), "turn": int(battle.turn)}


class OurSideFast(Player):
    """Model-free scripted arm for Monte-Carlo rollouts: forced first move at the
    engineered decision, always-Surf continuation. No obs, no tracker — fast."""

    def __init__(self, *args, forced_move: str, expected_h: int, expected_our_hp: int,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_move = forced_move
        self._expected_h = expected_h
        self._expected_our_hp = expected_our_hp
        self._st = {}
        self.decisions_reached = 0

    def choose_move(self, battle):
        st = self._st.setdefault(battle.battle_tag,
                                 {"endeavored": False, "captured": False})
        kind, payload = _OurScriptCore.classify(battle, st)
        if kind == "switch":
            nxt = (_first(battle.available_switches, lambda m: m.species != "starmie")
                   or (battle.available_switches[0] if battle.available_switches else None))
            return self.create_order(nxt) if nxt else self.choose_default_move()
        if kind == "decision":
            _OurScriptCore.assert_decision_state(battle, self._expected_h,
                                                 self._expected_our_hp)
            st["captured"] = True
            self.decisions_reached += 1
            payload = self._forced_move
        if payload == "endeavor":
            st["endeavored"] = True
        mv = _first(battle.available_moves, lambda m: m.id == payload)
        # FAIL LOUD: a missing scripted move silently degrading to the default order is
        # exactly how the first MC "pump" arms measured Surf under a pump label
        # (the move id is "hydropump", not "pump").
        assert mv is not None, (
            f"scripted move {payload!r} not available: "
            f"{[m.id for m in battle.available_moves]}")
        return self.create_order(mv)


class OurSideModel(RLPlayer):
    """Model-backed arm: genuine RLPlayer obs pipeline on EVERY decision (tracker fed
    exactly as live play would), scripted actions, and a full capture at the engineered
    decision: masked probability distribution, V(s), win-prob, value-dist, alpha/beta."""

    def __init__(self, *args, expected_h: int, expected_our_hp: int,
                 capture_sink: list, **kwargs):
        super().__init__(*args, **kwargs)
        self._expected_h = expected_h
        self._expected_our_hp = expected_our_hp
        self._capture_sink = capture_sink
        self._st = {}

    def _action_for_move(self, battle, move_id: str) -> int:
        legal = self._get_tracker(battle).last_ctx.legal
        for k, m in enumerate(legal.move_slots):
            if m.id == move_id and not m.disabled:
                return 6 + k
        raise AssertionError(f"{move_id} not in legal move slots "
                             f"{[m.id for m in legal.move_slots]}")

    def choose_move(self, battle):
        st = self._st.setdefault(battle.battle_tag,
                                 {"endeavored": False, "captured": False})
        kind, payload = _OurScriptCore.classify(battle, st)
        obs_dict = self.embed_battle(battle)   # keeps tracker/obs history genuine
        legal = self._get_tracker(battle).last_ctx.legal

        if kind == "switch":
            # pick the first legal non-Starmie switch action (Starmie is slot-last)
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
            board = _OurScriptCore.assert_decision_state(
                battle, self._expected_h, self._expected_our_hp)
            st["captured"] = True
            idx, capture = self._capture(battle, obs_dict, board)
        else:
            if payload == "endeavor":
                st["endeavored"] = True
            idx = self._action_for_move(battle, payload)
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)

    def _capture(self, battle, obs_dict, board):
        obs = obs_dict["observation"]
        mask = obs_dict["action_mask"].astype(np.float32)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[None]).to(self.model.device)
            mask_t = torch.as_tensor(mask[None]).to(self.model.device)
            pin = {"observation": obs_t, "action_mask": mask_t}
            dist = self.model.policy.get_distribution(pin)
            logits = dist.distribution.logits[0].cpu().numpy().astype(np.float64)
            masked = logits + (mask - 1.0) * 1e9
            probs = torch.softmax(torch.as_tensor(masked), dim=0).numpy()
            value = float(self.model.policy.predict_values(pin)[0].item())
        legal = self._get_tracker(battle).last_ctx.legal
        idx = int(np.argmax(masked))
        capture = {
            "board": board,
            "mask": mask.astype(int).tolist(),
            "probs": [round(float(p), 6) for p in probs],
            "logits_masked_legal": {str(i): round(float(logits[i]), 4)
                                    for i in range(11) if mask[i] > 0},
            "move_slots": [m.id for m in legal.move_slots],
            "argmax_action": idx,
            "value": round(value, 4),
            "win_prob": self._win_prob(),
            "value_dist": self._value_dist(),
            "opp_intent": self._opp_intent(battle),
            "obs_sha": __import__("hashlib").sha1(
                np.ascontiguousarray(obs).tobytes()).hexdigest()[:16],
        }
        self._capture_sink.append(capture)
        return idx, capture

# ---------------------------------------------------------------------------- phase A

IVS = {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31}


def _pmon(species, moves, *, item="", ability="No Ability", nature="Serious",
          evs=None, ivs=None):
    base = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
    if evs:
        base.update(evs)
    return {"species": species, "item": item, "ability": ability, "moves": moves,
            "evs": base, "ivs": ivs or dict(IVS), "nature": nature,
            "level": 100, "gender": "N"}


def phase_tables() -> dict:
    """Exact roll tables from the omniscient damage probe (fresh sim per scenario,
    varied seeds; crits and misses separated by observation, not assumption)."""
    frail_iv = dict(IVS); frail_iv["def"] = 0
    starmie_atk = _pmon("Starmie", ["surf", "hydropump"], item="leftovers",
                        ability="Natural Cure", nature="Modest",
                        evs={"spa": 252, "spe": 252, "hp": 4}, ivs=frail_iv)
    cbtar_tgt = _pmon("Tyranitar", ["roar"], item="choiceband", ability="Sand Stream",
                      nature="Adamant", evs={"atk": 252, "hp": 4})
    cbtar_eq = _pmon("Tyranitar", ["earthquake"], item="choiceband",
                     ability="Sand Stream", nature="Adamant", evs={"atk": 252, "hp": 4})
    starmie_frail_tgt = _pmon("Starmie", ["recover"], ability="Natural Cure",
                              nature="Modest", evs={"spa": 252, "spe": 252, "hp": 4},
                              ivs=frail_iv)
    starmie_bulky_tgt = _pmon("Starmie", ["recover"], ability="Natural Cure",
                              nature="Modest", evs={"spa": 252, "hp": 252, "spe": 4})
    marsh = _pmon("Marshtomp", ["endeavor", "growl"], ability="Torrent",
                  nature="Jolly", evs={"spe": 252, "hp": marsh_hp_ev_for(BASE_H)})
    marsh_tgt = _pmon("Marshtomp", ["growl"], ability="Torrent", nature="Jolly",
                      evs={"spe": 252, "hp": marsh_hp_ev_for(BASE_H)})

    S = []

    def add(tag, p1, p2, mvname, n):
        for i in range(n):
            S.append({"id": f"{tag}:{i}", "formatid": FORMAT,
                      "seed": [i + 1, i * 7 + 3, i * 13 + 5, i * 29 + 11],
                      "p1": [p1], "p2": [p2],
                      "choices": [["p1", f"move {1 + p1['moves'].index(mvname)}"],
                                  ["p2", "move 1"]]})

    add("surf", starmie_atk, cbtar_tgt, "surf", 320)
    add("pump", starmie_atk, cbtar_tgt, "hydropump", 320)
    add("eq_frail", cbtar_eq, starmie_frail_tgt, "earthquake", 96)
    add("eq_bulky", cbtar_eq, starmie_bulky_tgt, "earthquake", 96)
    add("eq_marsh", cbtar_eq, marsh_tgt, "earthquake", 64)
    for i in range(6):  # turn-1 ordering + Endeavor exactness
        S.append({"id": f"order:{i}", "formatid": FORMAT, "seed": [i + 2, 3, 5, 7],
                  "p1": [marsh], "p2": [cbtar_eq],
                  "choices": [["p1", "move 1"], ["p2", "move 1"]]})
    # explosion-into-protect: exploder faints, protector untouched (the filler chain's axiom)
    koffing = _pmon("Koffing", ["explosion"], ability="Levitate")
    golem = _pmon("Golem", ["protect", "explosion"], ability="Sturdy")
    for i in range(6):
        S.append({"id": f"boomprot:{i}", "formatid": FORMAT, "seed": [i + 3, 5, 7, 11],
                  "p1": [koffing], "p2": [golem],
                  "choices": [["p1", "move 1"], ["p2", "move 1"]]})
    # CB EQ vs Grimer: guaranteed overkill (2x SE) — Grimer must die before its move executes
    grimer = _pmon("Grimer", ["sludge", "protect"], ability="Sticky Hold")
    for i in range(24):
        S.append({"id": f"eq_grimer:{i}", "formatid": FORMAT,
                  "seed": [i + 1, i * 5 + 3, i * 11 + 5, i * 17 + 7],
                  "p1": [cbtar_eq], "p2": [grimer],
                  "choices": [["p1", "move 1"], ["p2", "move 1"]]})

    res = subprocess.run(["node", PROBE_JS],
                         input=json.dumps({"scenarios": S}),
                         capture_output=True, text=True, timeout=1800)
    raw = collections.defaultdict(collections.Counter)
    order_ok = []
    for line in res.stdout.splitlines():
        r = json.loads(line)
        if "error" in r or "fatal" in r:
            raise RuntimeError(f"probe error {r.get('id')}: {r.get('error', r.get('fatal'))[:400]}")
        tag = r["id"].split(":")[0]
        if tag == "order":
            order_ok.append((r["p2"]["hp"], r["p2"]["maxhp"], r["p1"]["maxhp"]))
            continue
        if tag == "boomprot":
            assert r["p1"]["hp"] == 0, "exploder did not faint into Protect"
            assert r["p2"]["hp"] == r["p2"]["maxhp"], "protector was damaged by Explosion"
            continue
        raw[tag][r["p2"]["maxhp"] - r["p2"]["hp"]] += 1

    for hp, maxhp, marsh_maxhp in order_ok:
        assert hp == BASE_H == marsh_maxhp and maxhp == TTAR_MAXHP, (
            f"Endeavor ordering broken: ttar {hp}/{maxhp}, marsh max {marsh_maxhp}")
    assert set(raw["eq_grimer"]) == {max(raw["eq_grimer"])}, (
        f"CB EQ did not always KO Grimer: {sorted(raw['eq_grimer'])}")

    def split(tag, target_maxhp, ko_capped: bool):
        c = raw[tag]
        vals = sorted(c)
        misses = c.get(0, 0)
        # crits: KO-capped at target max HP, or the ~2x cluster above the 16-roll band
        body = [v for v in vals if v not in (0, target_maxhp)]
        # the non-crit band is the 16 lowest distinct values; anything above is crit
        noncrit = body[:16]
        crit_vals = [v for v in body[16:]] + ([target_maxhp] if target_maxhp in c else [])
        n_crit = sum(c[v] for v in crit_vals)
        n = sum(c.values())
        return {"n": n, "rolls": noncrit, "n_distinct": len(noncrit),
                "miss_n": misses, "crit_n": n_crit,
                "crit_frac": round(n_crit / max(1, n - misses), 4),
                "counts": {str(k): v for k, v in sorted(c.items())}}

    tables = {
        "surf_vs_cbtar": split("surf", TTAR_MAXHP, True),
        "pump_vs_cbtar": split("pump", TTAR_MAXHP, True),
        "cb_eq_vs_frail_starmie": split("eq_frail", 262, True),
        "cb_eq_vs_bulky_starmie": split("eq_bulky", 324, True),
        "cb_eq_vs_marshtomp": split("eq_marsh", BASE_H, True),
    }
    surf = tables["surf_vs_cbtar"]["rolls"]
    assert len(surf) == 16, f"surf rolls != 16: {surf}"
    # every recorded pump HIT was a full-HP KO (342 dmg) => all pump rolls KO at any H<=342
    pump_c = raw["pump"]
    assert set(pump_c) <= {0, TTAR_MAXHP}, f"pump produced non-KO hits: {sorted(pump_c)}"
    # CB EQ always KOs the Def-IV-0 Starmie (the base cell's symmetric fatality)
    assert set(raw["eq_frail"]) == {262}, f"EQ did not always KO frail Starmie: {sorted(raw['eq_frail'])}"
    ko13 = sum(1 for r in surf if r >= BASE_H)
    assert ko13 == 13, f"H={BASE_H} gives {ko13}/16 surf KO rolls, wanted 13"
    tables["engineering"] = {
        "ttar_maxhp": TTAR_MAXHP, "H": BASE_H,
        "hp_fraction": round(BASE_H / TTAR_MAXHP, 4),
        "surf_ko_rolls": f"{ko13}/16",
        "surf_noncrit_ko_frac": ko13 / 16,
        "e_ko_surf": round(CRIT_P + (1 - CRIT_P) * ko13 / 16, 4),
        "e_ko_pump": 0.8,
        "measured_crit_frac_surf": tables["surf_vs_cbtar"]["crit_frac"],
        "measured_miss_frac_pump": round(
            tables["pump_vs_cbtar"]["miss_n"] / tables["pump_vs_cbtar"]["n"], 4),
    }
    return tables

# ---------------------------------------------------------------------------- truth

def base_truth(surf_rolls, h) -> dict:
    k = sum(1 for r in surf_rolls if r >= h)
    p_surf = CRIT_P + (1 - CRIT_P) * k / 16
    return {"p_win_surf": round(p_surf, 4), "p_win_pump": 0.8,
            "surf_ko_rolls": k, "note": "failure of either move = certain loss (CB EQ "
            "KOs the Def-IV-0 Starmie on every roll; simultaneous-turn retaliation)"}


def v2_truth(surf_rolls, eq_rolls, h, star_max=324) -> dict:
    """Exact DP over integer HPs. Starmie acts first; continuation = always Surf.
    Crit model: surf crit always KOs (2x min roll 570 > 342, proven); EQ crit = 2x roll
    (approximation — sim caps at KO; MC is the ground truth cross-check)."""
    sand = star_max // 16  # leftovers refunds it unless the sand tick itself kills

    @lru_cache(maxsize=None)
    def pwin(ttar_hp, star_hp, first_move):
        if ttar_hp <= 0:
            return 1.0
        if star_hp <= 0:
            return 0.0
        move = first_move or "surf"
        total = 0.0
        # our move branches: (prob, ttar_hp')
        branches = []
        if move == "pump":
            branches.append((0.2, ttar_hp))            # miss
            branches.append((0.8, 0))                  # every pump roll KOs (measured)
        else:
            branches.append((CRIT_P, 0))               # surf crit always KOs
            for r in surf_rolls:
                branches.append(((1 - CRIT_P) / 16, ttar_hp - r))
        for p_us, t2 in branches:
            if t2 <= 0:
                total += p_us
                continue
            # Ttar CB EQ resolves after us (we are faster; choice was simultaneous)
            for is_crit in (False, True):
                p_c = CRIT_P if is_crit else 1 - CRIT_P
                for r in eq_rolls:
                    dmg = 2 * r if is_crit else r
                    s2 = star_hp - dmg
                    if s2 <= 0:
                        continue
                    if s2 <= sand:  # sandstorm tick kills before Leftovers
                        continue
                    total += p_us * p_c / 16 * pwin(t2, s2, None)
        return total

    return {"p_win_surf": round(pwin(h, star_max, "surf"), 4),
            "p_win_pump": round(pwin(h, star_max, "pump"), 4),
            "continuation": "always Surf after the first move",
            "crit_model_note": "EQ crit approximated as 2x roll (uncapped); "
                               "MC rollouts are the assumption-free cross-check"}

# ---------------------------------------------------------------------------- battles

def _mk_accounts(tag):
    return (AccountConfiguration(f"RPr{tag}"[:17], None),
            AccountConfiguration(f"RPo{tag}"[:17], None))


def run_capture(model, mappings, starmie_block, h, expected_our_hp, tag,
                seed=(7, 11, 13, 17)) -> dict:
    sink = []
    a1, a2 = _mk_accounts(tag)
    ours = OurSideModel(
        model=model, team=our_team(starmie_block, marsh_hp_ev_for(h)),
        battle_format=FORMAT, server_configuration=LocalhostServerConfiguration,
        mappings=mappings, account_configuration=a1, start_listening=False,
        stochastic=False, expected_h=h, expected_our_hp=expected_our_hp,
        capture_sink=sink)
    theirs = TtarSide(battle_format=FORMAT, team=THEIR_TEAM,
                      server_configuration=LocalhostServerConfiguration,
                      account_configuration=a2, start_listening=False)
    asyncio.run(run_local_battles(ours, theirs, 1, battle_format=FORMAT,
                                  seed=list(seed), impl="node"))
    assert len(sink) == 1, f"decision reached {len(sink)} times (wanted exactly 1)"
    battle = next(iter(ours.battles.values()))
    cap = sink[0]
    cap["battle_won"] = bool(battle.won)
    cap["final_turn"] = int(battle.turn)
    return cap


def run_mc(forced_move, starmie_block, h, expected_our_hp, n, tag) -> dict:
    a1, a2 = _mk_accounts(tag)
    ours = OurSideFast(battle_format=FORMAT,
                       team=our_team(starmie_block, marsh_hp_ev_for(h)),
                       server_configuration=LocalhostServerConfiguration,
                       account_configuration=a1, start_listening=False,
                       forced_move=forced_move, expected_h=h,
                       expected_our_hp=expected_our_hp,
                       max_concurrent_battles=2)
    theirs = TtarSide(battle_format=FORMAT, team=THEIR_TEAM,
                      server_configuration=LocalhostServerConfiguration,
                      account_configuration=a2, start_listening=False,
                      max_concurrent_battles=2)
    asyncio.run(run_local_battles(ours, theirs, n, battle_format=FORMAT,
                                  seed=None, impl="node", concurrency=2))
    assert ours.decisions_reached == n, (
        f"decision reached {ours.decisions_reached}/{n} battles")
    wins = sum(1 for b in ours.battles.values() if b.won)
    lo, hi = _wilson(wins, n)
    return {"n": n, "wins": wins, "p_win": round(wins / n, 4),
            "wilson95": [round(lo, 4), round(hi, 4)]}


def _wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (c - m) / d, (c + m) / d

# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["tables", "capture", "mc", "sweep", "all"])
    ap.add_argument("--mc-n", type=int, default=400)
    ap.add_argument("--mc-n-base", type=int, default=250)
    ap.add_argument("--out", default=str(OUT_DIR / "starmie_ttar_risk_probe_2026-08-30.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    def save():
        out_path.write_text(json.dumps(results, indent=1))
        print(f"[saved {out_path}]")

    results.setdefault("meta", {
        "date": "2026-08-30", "format": FORMAT, "impl": "node",
        "checkpoint": CKPT, "checkpoint_lineage": CKPT2,
        "capture_seed": [7, 11, 13, 17],
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
    })

    if args.phase in ("tables", "all"):
        print("== phase A: damage tables ==")
        results["tables"] = phase_tables()
        print(json.dumps(results["tables"]["engineering"], indent=1))
        save()

    tables = results["tables"]
    surf_rolls = tables["surf_vs_cbtar"]["rolls"]
    eq_bulky = tables["cb_eq_vs_bulky_starmie"]["rolls"]

    model = mappings = None

    def load(ck):
        nonlocal mappings
        if mappings is None:
            mappings = load_mappings()
        ver = current_model_version(mappings)
        m, _ = load_foreign_opponent(ck, current_version=ver, device="cpu")
        return m

    if args.phase in ("capture", "all"):
        print("== phase B: captures ==")
        model = load(CKPT)
        results.setdefault("variants", {})
        results["variants"]["base"] = {
            "starmie": "frail (4 HP/252 SpA/252 Spe Modest, Def IV 0; 262 HP)",
            "capture": run_capture(model, mappings, STARMIE_FRAIL, BASE_H, 262, "b1"),
            "truth": base_truth(surf_rolls, BASE_H),
        }
        save()
        results["variants"]["recoverable"] = {
            "starmie": "bulky (252 HP/252 SpA/4 Spe Modest; 324 HP; survives exactly one "
                       "non-crit CB EQ)",
            "capture": run_capture(model, mappings, STARMIE_BULKY, BASE_H, 324, "b2"),
            "truth_analytic": v2_truth(surf_rolls, eq_bulky, BASE_H),
        }
        save()
        # lineage checkpoint: base cell only
        try:
            model2 = load(CKPT2)
            results["variants"]["base_lineage_rev1"] = {
                "checkpoint": CKPT2,
                "capture": run_capture(model2, mappings, STARMIE_FRAIL, BASE_H, 262, "b3"),
            }
            del model2
        except Exception as e:  # lineage comparison is optional; never blocks the probe
            results["variants"]["base_lineage_rev1"] = {"error": str(e)[:400]}
        save()

    if args.phase in ("mc", "all"):
        print("== phase C: Monte-Carlo rollouts ==")
        results.setdefault("mc", {})
        move_ids = {"surf": "surf", "pump": "hydropump"}
        for arm, block, hp, n, tag in (
                ("surf", STARMIE_FRAIL, 262, args.mc_n_base, "m1"),
                ("pump", STARMIE_FRAIL, 262, args.mc_n_base, "m2")):
            key = f"base_{arm}"
            if key not in results["mc"]:
                results["mc"][key] = run_mc(move_ids[arm], block, BASE_H, hp, n, tag)
                print(key, results["mc"][key])
                save()
        for arm, tag in (("surf", "m3"), ("pump", "m4")):
            key = f"recoverable_{arm}"
            if key not in results["mc"]:
                results["mc"][key] = run_mc(move_ids[arm], STARMIE_BULKY, BASE_H, 324,
                                            args.mc_n, tag)
                print(key, results["mc"][key])
                save()

    if args.phase in ("sweep", "all"):
        print("== phase D: HP sweep (mask vs true KO fraction) ==")
        if model is None:
            model = load(CKPT)
        sweep_hs = [284] + [r + 1 for r in surf_rolls]  # 16/16 down to 0/16
        results.setdefault("sweep", [])
        done = {p["H"] for p in results["sweep"]}
        for i, h in enumerate(sweep_hs):
            if h in done:
                continue
            frac = sum(1 for r in surf_rolls if r >= h)
            cap = run_capture(model, mappings, STARMIE_FRAIL, h, 262, f"s{i}")
            slots = cap["move_slots"]
            p_surf = cap["probs"][6 + slots.index("surf")]
            p_pump = cap["probs"][6 + slots.index("hydropump")]
            row = {"H": h, "hp_frac": round(h / TTAR_MAXHP, 4),
                   "opp_hp_display": cap["board"]["opp_hp_display"],
                   "surf_ko_rolls": frac,
                   "e_ko_surf": round(CRIT_P + (1 - CRIT_P) * frac / 16, 4),
                   "e_ko_pump": 0.8,
                   "p_surf": p_surf, "p_pump": p_pump,
                   "argmax": cap["move_slots"][cap["argmax_action"] - 6]
                             if cap["argmax_action"] >= 6 else cap["argmax_action"],
                   "value": cap["value"], "win_prob": cap["win_prob"]}
            results["sweep"].append(row)
            print(row)
            save()
        results["sweep"].sort(key=lambda r: r["H"])
        save()

    print("done.")


if __name__ == "__main__":
    main()
