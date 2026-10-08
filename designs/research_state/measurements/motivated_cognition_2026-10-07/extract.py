"""Motivated-cognition read, EXTRACTION half (README §0): per checkpoint, CPU forwards on the Lane S bank.

Run AT THE PIN through ``run_pin.sh`` (cwd + PYTHONPATH = the 706fa536 checkout):

    run_pin.sh extract.py --out <dir>/rows --ckpt <zip>=<label> [...]

Writes, per label, ``<label>.npz`` (per-row arrays the self-serving tests and the bootstrap need) and
``<label>.calib.json`` (per-head calibration aggregates, ARM and PRIOR columns). RESUMABLE: a label whose
two files both exist is skipped. The bank is re-encoded once per invocation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

TEAM = 6
BATCH = 256
HP_NUM = 237
TYPED_HP0 = 355
N_HP = 16
TIE_EPS = 1e-6
MOVE_UNIT_MIN_PRIOR = 0.01       # §0.3: a move is a scored unit iff Smogon P(m | species) >= 0.01
CB_MIN_PRIOR = 0.02              # §0.4 T3
ECE_EDGES = np.linspace(0.0, 1.0, 11)
REL_EDGES = np.array([0, .01, .02, .05, .1, .2, .3, .5, .7, .9, 1.0])
HP_TYPE_NAMES = ["bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost", "grass",
                 "ground", "ice", "poison", "psychic", "rock", "steel", "water"]


# ───────────────────────────────────────── static tables ─────────────────────────────────────────
def to_id(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_packed(packed: str) -> List[dict]:
    """Showdown packed team → [{species, item, ability, moves, nature, evs}] (evs: atk,def,spa,spd,spe)."""
    out = []
    for mon in packed.split("]"):
        f = mon.split("|")
        sp = f[1] if len(f) > 1 and f[1] else f[0]
        evs_raw = (f[6].split(",") if len(f) > 6 and f[6] else [])
        evs = [int(x) if x else 0 for x in evs_raw] + [0] * (6 - len(evs_raw))
        out.append({"name": f[0], "species": sp, "item": f[2] if len(f) > 2 else "",
                    "ability": f[3] if len(f) > 3 else "",
                    "moves": [to_id(m) for m in (f[4].split(",") if len(f) > 4 and f[4] else []) if m],
                    "nature": f[5] if len(f) > 5 else "", "evs": evs[1:6]})
    return out


class Tables:
    """Dex facts the tests need, num-indexed (S species × M moves)."""

    def __init__(self, S: int, M: int):
        from agents import gen3_data
        from agents.gen3_data import type_chart

        self.S, self.M = S, M
        chart = type_chart.chart()
        tnames = sorted(chart.keys())
        self.tidx = {t: i for i, t in enumerate(tnames)}
        T = len(tnames)
        mult = np.ones((T, T))                              # [attacking, defending]
        for d in tnames:
            for a in tnames:
                mult[self.tidx[a], self.tidx[d]] = float(chart[d][a])
        # species → types (num-keyed, base forms)
        self.sp_types: Dict[int, Tuple[int, ...]] = {}
        for sid in gen3_data.species.base_form_ids():
            sd = gen3_data.species.get(sid)
            if 0 < sd.num < S and sd.num not in self.sp_types:
                ts = tuple(self.tidx[t.upper()] for t in sd.types if t.upper() in self.tidx)
                if len(ts) != len(sd.types):
                    raise KeyError(f"{sid}: a type off the chart {sd.types}")
                self.sp_types[sd.num] = ts
        # attack type a vs defender species s: product over s's types
        self.att_vs_sp = np.ones((T, S))
        for s, ts in self.sp_types.items():
            for t in ts:
                self.att_vs_sp[:, s] *= mult[:, t]
        # moves: type idx + damaging, by num; typed HP channels
        self.move_type = np.full(M, -1, dtype=np.int64)
        self.move_dmg = np.zeros(M, dtype=bool)
        for mid, md in gen3_data.moves._dex().items():
            if 0 < md.num < M and not mid.startswith("hiddenpower") and md.type.name in self.tidx:
                # (a type off the chart, e.g. Curse's '???', is no damaging threat: type stays -1)
                self.move_type[md.num] = self.tidx[md.type.name]
                self.move_dmg[md.num] = md.base_power > 0
        for t, n in enumerate(HP_TYPE_NAMES):
            self.move_type[TYPED_HP0 + t] = self.tidx[n.upper()]
            self.move_dmg[TYPED_HP0 + t] = True
        self.ground_moves = self.move_type == self.tidx["GROUND"]
        # SE_move[m, s]: damaging move m is super-effective (>= 2) vs species s
        mt = np.where(self.move_type >= 0, self.move_type, 0)
        self.se_move = (self.move_dmg[:, None] & (self.move_type[:, None] >= 0)
                        & (self.att_vs_sp[mt, :] >= 2.0 - 1e-12))
        # TH[z, s]: some type of species s is >= 2 vs species z (species-level STAB threat)
        self.th = np.zeros((S, S), dtype=bool)
        for s, ts in self.sp_types.items():
            for t in ts:
                self.th[:, s] |= self.att_vs_sp[t, :] >= 2.0 - 1e-12
        self.cb_num = int(gen3_data.items.get("choiceband").num)
        self.nature_num = {k.lower(): int(v["num"]) for k, v in gen3_data.natures.raw().items()}


def mon_record(m: dict, tb: "Tables") -> dict:
    from agents import gen3_data
    from main.belief_roles.bank_rows import move_event_num, species_num

    num = species_num(m["species"])
    mh = np.zeros(tb.M, dtype=bool)
    hp_t = -1
    for mv in m["moves"]:
        if mv.startswith("hiddenpower"):
            t = mv[len("hiddenpower"):]
            if t in HP_TYPE_NAMES:
                hp_t = HP_TYPE_NAMES.index(t)
                mh[TYPED_HP0 + hp_t] = True
            mh[HP_NUM] = True
        else:
            mh[move_event_num(mv)] = True
    item = to_id(m["item"])
    if item:
        it = gen3_data.items.get(item)
        if it is None:
            raise KeyError(f"item {item!r} has no num (never a silent 0)")
        item_num = int(it.num)
    else:
        item_num = 0
    nat = tb.nature_num.get(to_id(m["nature"]), -1) if m["nature"] else -1
    return {"num": num, "moves": mh, "hp_t": hp_t, "item": item_num, "nature": nat,
            "evs": np.array(m["evs"], dtype=np.float64), "ability": to_id(m["ability"]), "name": m["name"]}


# ───────────────────────────────────────── calibration accumulators ─────────────────────────────
class BinAcc:
    """Binary-unit calibration aggregates: ECE bins, reliability bins, Brier / log score / CITL sums."""

    def __init__(self):
        self.n = 0
        self.sp = self.sy = self.brier = self.logl = 0.0
        self.e_n = np.zeros(10); self.e_p = np.zeros(10); self.e_y = np.zeros(10)
        self.r_n = np.zeros(10); self.r_p = np.zeros(10); self.r_y = np.zeros(10)

    def add(self, p: np.ndarray, y: np.ndarray):
        p = p.astype(np.float64).ravel(); y = y.astype(np.float64).ravel()
        if p.size == 0:
            return
        self.n += p.size
        self.sp += p.sum(); self.sy += y.sum()
        self.brier += ((p - y) ** 2).sum()
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        self.logl += -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).sum()
        for edges, cn, cp, cy in ((ECE_EDGES, self.e_n, self.e_p, self.e_y),
                                  (REL_EDGES, self.r_n, self.r_p, self.r_y)):
            b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
            cn += np.bincount(b, minlength=10); cp += np.bincount(b, p, 10); cy += np.bincount(b, y, 10)

    def out(self) -> dict:
        n = max(self.n, 1)
        return {"n": self.n, "ece": float(np.abs(self.e_p - self.e_y).sum() / n),
                "brier": self.brier / n, "logscore": self.logl / n, "citl": (self.sp - self.sy) / n,
                "mean_p": self.sp / n, "mean_y": self.sy / n,
                "rel": {"edges": REL_EDGES.tolist(), "n": self.r_n.tolist(), "sum_p": self.r_p.tolist(),
                        "sum_y": self.r_y.tolist()}}


class CatAcc:
    """Categorical calibration aggregates: multi-class Brier, log score, top-1 ECE / CITL."""

    def __init__(self):
        self.n = 0
        self.brier = self.logl = self.conf = self.acc = 0.0
        self.e_n = np.zeros(10); self.e_c = np.zeros(10); self.e_a = np.zeros(10)

    def add(self, P: np.ndarray, y: np.ndarray):
        if len(y) == 0:
            return
        P = P.astype(np.float64)
        n = len(y)
        self.n += n
        oh = np.zeros_like(P); oh[np.arange(n), y] = 1.0
        self.brier += ((P - oh) ** 2).sum()
        self.logl += -np.log(np.clip(P[np.arange(n), y], 1e-12, 1.0)).sum()
        top = P.argmax(1); c = P.max(1); a = (top == y).astype(np.float64)
        self.conf += c.sum(); self.acc += a.sum()
        b = np.clip(np.searchsorted(ECE_EDGES, c, side="right") - 1, 0, 9)
        self.e_n += np.bincount(b, minlength=10); self.e_c += np.bincount(b, c, 10)
        self.e_a += np.bincount(b, a, 10)

    def out(self) -> dict:
        n = max(self.n, 1)
        return {"n": self.n, "brier": self.brier / n, "logscore": self.logl / n,
                "ece_top1": float(np.abs(self.e_c - self.e_a).sum() / n),
                "citl_top1": (self.conf - self.acc) / n, "conf": self.conf / n, "acc": self.acc / n,
                "bins": {"n": self.e_n.tolist(), "sum_conf": self.e_c.tolist(), "sum_acc": self.e_a.tolist()}}


# ───────────────────────────────────────── the per-checkpoint read ───────────────────────────────
def read_one(zip_path: Path, br, ctxinfo: dict, tb: Tables, threads: int) -> Tuple[dict, dict]:
    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.hypothesis_dex_rows import load_hypothesis_dex_rows
    from agents.model.hypothesis_set import (belief_head_team_scores, fixed_mass_presence,
                                             species_candidates)
    from agents.model.t0_species import species_team_prior_logits
    from main.belief_roles.forward import arm_of, load_strict
    from main.policy_spectrum.reader import inference_globals

    with inference_globals(threads), torch.no_grad():
        model = load_strict(zip_path)
        arm = arm_of(model)
        fe = model.policy.features_extractor
        S = int(fe.layout["max_species"])
        valid = torch.from_numpy(load_hypothesis_dex_rows(S).valid.copy())
        Pm = fe.move_belief.move_prior_probs.double()                    # [S,M]
        hp_prior = fe.hp_type_belief_head.hp_prior.double()               # [S,16]
        item_prior = fe.item_belief_head.item_prior.double()              # [S,I]
        item_prior = item_prior.clamp_min(1e-6)
        item_prior = item_prior / item_prior.sum(-1, keepdim=True)       # the head's log-prior, normalised
        nat_prior = torch.softmax(fe.spread_belief.nature_logprior.double(), -1)
        ev_prior = fe.spread_belief.ev_prior.double()
        t0 = fe.t0_species_prior
        M = Pm.shape[1]
        assert M == tb.M and S == tb.S, (M, S)
        typed = torch.arange(TYPED_HP0, TYPED_HP0 + N_HP)
        se_move = torch.from_numpy(tb.se_move)
        th = torch.from_numpy(tb.th).double()
        ground = torch.from_numpy(tb.ground_moves)
        acc = {f"{h}_{c}": (CatAcc() if h in ("item", "nature", "hptype") else BinAcc())
               for h in ("presence", "moves", "item", "nature", "hptype", "cb") for c in ("arm", "prior")}
        ev_abs = {"arm": 0.0, "prior": 0.0, "n": 0}
        N = br.n
        R = {k: np.full(N, np.nan) for k in ("e1_arm", "e1_prior", "n_theta", "d2_arm", "d2_prior",
                                           "e3_arm", "e3_prior", "margin")}
        R["top1"] = np.full(N, -1, dtype=np.int64)
        for k_ in ("s1_arm", "s1_prior", "y1", "p_stay"):
            R[k_] = np.full(N, np.nan)
        U: Dict[str, list] = {k_: [] for k_ in ("row", "m", "p_arm", "p_prior", "y")}
        for i in range(0, N, BATCH):
            sl = slice(i, i + BATCH)
            rows = torch.tensor(br.rows[sl]); mk = torch.tensor(br.masks[sl])
            B = rows.shape[0]
            ob = {"observation": rows, "action_mask": mk.float()}
            ob.update(zero_extra_obs(fe, batch=B))
            ctx = fe.unpack(ob)
            logits = model.policy.get_distribution(ob).distribution.logits.double()
            st = fe.stash
            # ---- the preference: greedy argmax of the masked policy, rule-8 margin
            pr = torch.softmax(logits.masked_fill(~mk, -math.inf), -1)
            t2v = pr.topk(2, -1)
            R["top1"][sl] = t2v.indices[:, 0].numpy()
            R["margin"][sl] = (t2v.values[:, 0] - t2v.values[:, 1]).numpy()
            opp_ids = ctx.species_ids[:, TEAM:].long()
            believed = ctx.opp_believed_mask.bool()
            rev = (opp_ids > 0) & ~believed
            cand, k = species_candidates(valid, opp_ids, believed)
            # ---- species presence (§0.3)
            t0lp = species_team_prior_logits(t0.species_prior_log_marginal, t0.species_prior_log_lift,
                                             opp_ids, believed).double()
            pi_prior = fixed_mass_presence(t0lp, cand, k).pi
            if arm == "fixed_mass":
                pi_arm = st.hypothesis.species.pi.double()
            else:
                pi_arm = fixed_mass_presence(belief_head_team_scores(st.belief_logits["species"].double(),
                                                                     believed), cand, k).pi
            ytrue = torch.zeros(B, S, dtype=torch.bool)
            ytrue.scatter_(1, torch.from_numpy(br.true_species[sl]).clamp(0, S - 1), True)
            yS = (ytrue & cand).double()
            live = cand & (k > 0).unsqueeze(-1)
            for c, pi in (("arm", pi_arm), ("prior", pi_prior)):
                acc[f"presence_{c}"].add(pi[live].numpy(), yS[live].numpy())
            # ---- per revealed opp slot: the truth mon (battle, num) → index into ctxinfo tables
            bidx = br.battle_index[sl]
            mon_ix = np.full((B, TEAM), -1, dtype=np.int64)
            oid = opp_ids.numpy(); rv = rev.numpy()
            for b in range(B):
                for j in range(TEAM):
                    if rv[b, j]:
                        mon_ix[b, j] = ctxinfo["opp_key"][(int(bidx[b]), int(br_side(br, i + b)), int(oid[b, j]))]
            has = torch.from_numpy(mon_ix >= 0)
            mi = torch.from_numpy(np.where(mon_ix >= 0, mon_ix, 0))
            Ymoves = torch.from_numpy(ctxinfo["opp_moves"])[mi]          # [B,6,M] bool
            # ---- moves (§0.3)
            p_arm = torch.sigmoid(st.move_belief_logits.double())       # [B,6,M] typed posterior
            sid = opp_ids.clamp(0, S - 1)
            pri = Pm[sid].clone()                                        # [B,6,M]
            pri[..., typed] = pri[..., HP_NUM].unsqueeze(-1) * hp_prior[sid]
            revm = torch.zeros(B, TEAM, M, dtype=torch.bool)
            amv = ctx.all_move_ids[:, TEAM:, :].long().clamp(0, M - 1)
            revm.scatter_(-1, amv, (ctx.all_move_ids[:, TEAM:, :] > 0))
            hp_rev = revm[..., HP_NUM] | revm[..., typed].any(-1)
            four = (ctx.all_move_ids[:, TEAM:, :] > 0).sum(-1) >= 4
            unit = (pri >= MOVE_UNIT_MIN_PRIOR) & ~revm
            unit[..., HP_NUM] = False
            unit[..., typed] &= ~hp_rev.unsqueeze(-1)
            unit &= (has & ~four).unsqueeze(-1)
            yM = Ymoves.double()
            for c, p in (("arm", p_arm), ("prior", pri)):
                acc[f"moves_{c}"].add(p[unit].numpy(), yM[unit].numpy())
            # ---- item / nature / HP type / EV (revealed slots)
            it_lab = torch.from_numpy(ctxinfo["opp_item"])[mi]
            nat_lab = torch.from_numpy(ctxinfo["opp_nature"])[mi]
            hp_lab = torch.from_numpy(ctxinfo["opp_hp_t"])[mi]
            ev_lab = torch.from_numpy(ctxinfo["opp_evs"])[mi]
            item_unrev = ctx.item_ids[:, TEAM:].long() == 0
            Pit = torch.softmax(st.item_logits.double(), -1)
            Pit_pr = item_prior[sid]
            u_it = has & item_unrev
            Pnat = torch.softmax(st.spread_nature_logits.double(), -1)
            u_nat = has & (nat_lab >= 0)
            Php = torch.softmax(st.hp_type_logits.double(), -1)
            u_hp = has & (hp_lab >= 0)
            for c, (Pi, Pn, Ph) in (("arm", (Pit, Pnat, Php)),
                                    ("prior", (Pit_pr, nat_prior[sid], hp_prior[sid]))):
                acc[f"item_{c}"].add(Pi[u_it].numpy(), it_lab[u_it].numpy())
                acc[f"cb_{c}"].add(Pi[..., tb.cb_num][u_it].numpy(),
                                   (it_lab[u_it] == tb.cb_num).double().numpy())
                acc[f"nature_{c}"].add(Pn[u_nat].numpy(), nat_lab[u_nat].numpy())
                acc[f"hptype_{c}"].add(Ph[u_hp].numpy(), hp_lab[u_hp].numpy())
            if bool(u_nat.any()):
                ev_abs["arm"] += float((st.spread_ev.double()[u_nat] - ev_lab[u_nat]).abs().sum())
                ev_abs["prior"] += float((ev_prior[sid][u_nat] - ev_lab[u_nat]).abs().sum())
                ev_abs["n"] += int(u_nat.sum()) * 5
            # ---- self-serving tests (§0.4)
            oa = ctx.opp_active_local.long().clamp(0, TEAM - 1)
            ar = torch.arange(B)
            our_ids = ctx.species_ids[:, :TEAM].long()
            our_act = our_ids[ar, ctx.our_active_idx.long().clamp(0, TEAM - 1)]
            lev = torch.tensor([ctxinfo["our_ability"].get((int(bidx[b]), int(br_side(br, i + b)),
                                                            int(our_act[b])), "") == "levitate"
                                for b in range(B)])
            theta = unit[ar, oa] & se_move[:, our_act.clamp(0, S - 1)].T
            theta &= ~(ground.unsqueeze(0) & lev.unsqueeze(-1))
            theta &= has[ar, oa].unsqueeze(-1)
            for c, p in (("arm", p_arm), ("prior", pri)):
                e = ((p[ar, oa] - yM[ar, oa]) * theta.double()).sum(-1)
                R[f"e1_{c}"][sl] = torch.where(theta.any(-1), e, torch.full_like(e, math.nan)).numpy()
            R["n_theta"][sl] = theta.sum(-1).double().numpy()
            # ---- post-hoc diagnostics (README §3; NOT part of the registered verdict): the T1 sums
            # split into belief and truth, the policy's STAY mass, and the per-unit (p, y) of Θ
            R["s1_arm"][sl] = (p_arm[ar, oa] * theta.double()).sum(-1).numpy()
            R["s1_prior"][sl] = (pri[ar, oa] * theta.double()).sum(-1).numpy()
            R["y1"][sl] = (yM[ar, oa] * theta.double()).sum(-1).numpy()
            R["p_stay"][sl] = pr[:, 6:].sum(-1).numpy()
            ub, um = theta.nonzero(as_tuple=True)
            U["row"].append((ub + i).numpy())
            U["m"].append(um.numpy())
            U["p_arm"].append(p_arm[ar, oa][ub, um].numpy())
            U["p_prior"].append(pri[ar, oa][ub, um].numpy())
            U["y"].append(yM[ar, oa][ub, um].numpy())
            # T2: E[b, z] = Σ_s cand · TH[z, s] · (π_s − y_s)
            for c, pi in (("arm", pi_arm), ("prior", pi_prior)):
                err = (pi - yS) * cand.double()
                E = err @ th.T                                            # [B,S_z]
                d2 = np.full(B, np.nan)
                En = E.numpy()
                for b in range(B):
                    g = i + b
                    if int(k[b]) < 1:
                        continue
                    opts = ctxinfo["options"][g]                          # {action: species num}
                    a = int(R["top1"][g])
                    act_num = int(our_act[b])
                    if a in opts:                                         # switch to opts[a]
                        z_star = opts[a]
                        alts = [z for aa, z in opts.items() if aa != a] + [act_num]
                    else:
                        z_star = act_num
                        alts = list(opts.values())
                    if not alts or z_star <= 0 or any(z <= 0 for z in alts):
                        continue
                    d2[b] = En[b, z_star] - float(np.mean([En[b, z] for z in alts]))
                R[f"d2_{c}"][sl] = d2
            # T3: Choice Band on the opp active
            cb_arm = Pit[ar, oa, tb.cb_num]
            cb_pr = Pit_pr[ar, oa, tb.cb_num]
            ok3 = has[ar, oa] & item_unrev[ar, oa] & (cb_pr >= CB_MIN_PRIOR)
            y3 = (it_lab[ar, oa] == tb.cb_num).double()
            nan = torch.full((B,), math.nan, dtype=torch.float64)
            R["e3_arm"][sl] = torch.where(ok3, cb_arm - y3, nan).numpy()
            R["e3_prior"][sl] = torch.where(ok3, cb_pr - y3, nan).numpy()
        for k_, v in U.items():
            R[f"u_{k_}"] = np.concatenate(v)
        calib = {k_: v.out() for k_, v in acc.items()}
        calib["ev_mae"] = {"arm": ev_abs["arm"] / max(ev_abs["n"], 1),
                           "prior": ev_abs["prior"] / max(ev_abs["n"], 1), "n": ev_abs["n"]}
        calib["arm"] = arm
        del model
    return R, calib


_SIDE = {}


def br_side(br, g: int) -> int:
    """0 = the decision's viewer is p1, 1 = p2."""
    return _SIDE[g]


def build_ctxinfo(br, tb: Tables) -> dict:
    """Truth tables keyed by (battle index, viewer side, species num): the viewer's OPPONENT's mons, the
    viewer's own mons' abilities, and per decision the switch options {action: species num}."""
    from main.belief_roles.bank_rows import _as_dict, species_num

    bank = br.bank
    opp_key: Dict[Tuple[int, int, int], int] = {}
    our_ability: Dict[Tuple[int, int, int], str] = {}
    mons = []
    name_num: Dict[Tuple[int, int], Dict[str, int]] = {}
    for bi, b in enumerate(bank.battles):
        for vs, (own, other) in enumerate((("p1", "p2"), ("p2", "p1"))):
            for m in parse_packed(_as_dict(getattr(b, other))["team"]):
                r = mon_record(m, tb)
                opp_key[(bi, vs, r["num"])] = len(mons)          # a later mon of the species wins (Rust rule)
                mons.append(r)
            nn = {}
            for m in parse_packed(_as_dict(getattr(b, own))["team"]):
                num = species_num(m["species"])
                our_ability[(bi, vs, num)] = to_id(m["ability"])
                nn[to_id(m["species"])] = num
                if m["name"]:
                    nn[to_id(m["name"])] = num
            name_num[(bi, vs)] = nn
    options = []
    for g, d in enumerate(bank.decisions):
        vs = 0 if d["side"] == "p1" else 1
        _SIDE[g] = vs
        bi = int(br.battle_index[g])
        o = {}
        for a, tok in d["tokens"].items():
            a = int(a)
            if a < 6 and d["mask"][a] == "1":
                nm = to_id(tok.split(" ", 1)[1])
                if nm not in name_num[(bi, vs)]:
                    raise KeyError(f"{d['id']}: switch target {tok!r} not on the viewer's team")
                o[a] = name_num[(bi, vs)][nm]
        options.append(o)
    M = tb.M
    return {"opp_key": opp_key, "our_ability": our_ability, "options": options,
            "opp_moves": np.stack([m["moves"] for m in mons]).astype(bool),
            "opp_item": np.array([m["item"] for m in mons], dtype=np.int64),
            "opp_nature": np.array([m["nature"] for m in mons], dtype=np.int64),
            "opp_hp_t": np.array([m["hp_t"] for m in mons], dtype=np.int64),
            "opp_evs": np.stack([m["evs"] for m in mons]).astype(np.float64), "M": M}


def main() -> None:
    from main.belief_roles.__main__ import default_bank
    from main.belief_roles.bank_rows import bank_rows_of
    from main.policy_spectrum.bank import load_bank

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="<zip>=<label>")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-battles", type=int, default=0, help="smoke only: the first N battles")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for spec in a.ckpt:
        z, lab = spec.rsplit("=", 1)
        if (out / f"{lab}.npz").exists() and (out / f"{lab}.calib.json").exists():
            print(f"[skip] {lab}: done", flush=True)
            continue
        todo.append((Path(z), lab))
    if not todo:
        return
    t = time.time()
    bank = load_bank(default_bank())
    if a.max_battles:
        import dataclasses
        keep = {b.battle_id for b in bank.battles[:a.max_battles]}
        bank = dataclasses.replace(bank, battles=bank.battles[:a.max_battles],
                                   decisions=[d for d in bank.decisions if d["battle"] in keep])
    br = bank_rows_of(bank, workers=a.workers)
    print(f"[bank] {br.n} rows re-encoded in {time.time() - t:.0f}s; gate {br.gate['obs_as_recorded']} "
          f"({br.gate['byte_equal']}/{br.gate['recorded_rows_checked']})", flush=True)
    tb = Tables(S=400, M=400)
    ci = build_ctxinfo(br, tb)
    np.savez_compressed(out / "_bank_meta.npz", battle_index=br.battle_index,
                        free=np.array([d["kind"] == "free" for d in bank.decisions]),
                        masks=br.masks)
    for z, lab in todo:
        t = time.time()
        R, calib = read_one(z, br, ci, tb, a.threads)
        calib.update({"label": lab, "checkpoint": str(z), "pin": os.environ.get("PYTHONPATH", ""),
                      "gate": br.gate, "wall_s": time.time() - t})
        np.savez_compressed(out / f"{lab}.npz", **R)
        (out / f"{lab}.calib.json").write_text(json.dumps(calib, indent=1, default=float))
        print(f"[done] {lab} ({calib['arm']}) {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
