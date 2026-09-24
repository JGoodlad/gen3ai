"""Play the paired battles of ONE worker shard and score the belief heads at every trainee decision.

PREDICTION.md §2-§3. Run with the PINNED tree (cwd = the pin worktree, PYTHONPATH=<pin>/src,
POKESIM_SIM_BRIDGE_BIN=<pin>/src/rust_sim/target/release/sim_bridge):

    python run_arm.py --manifest <work>/manifest_full.json --arms pool,ladder,procedural \
        --indices 0-399 --worker 0 --n-workers 4 --out <work>/rows

Battle i of every arm: same trainee team, same sim seed (seed_base), trainee = p1 (the model,
GREEDY), opponent = p2 (a SECOND copy of the model, GREEDY). Only the opponent's TEAM differs.
One JSON line per battle: {arm, i, status, n_dec, rows{family: [[k, ...head..., ...prior...], ...]}}.
Nothing is written under models/.
"""
import argparse
import asyncio
import itertools
import json
import os
import time
import traceback

import numpy as np
import torch

torch.set_num_threads(1)

from poke_env import AccountConfiguration, LocalhostServerConfiguration  # noqa: E402
from poke_env.data.normalize import to_id_str  # noqa: E402
from poke_env.data.gen_data import GenData  # noqa: E402
from poke_env.teambuilder import Teambuilder  # noqa: E402

from agents.inference.player import RLPlayer  # noqa: E402
from agents.training.gen3_env import Gen3Env  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402
from agents.observation.base import ObservationEncoder  # noqa: E402
from agents.observation.constants import (  # noqa: E402
    OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET, TEAM_SIZE)
from agents.model.t0_species import species_team_prior_logits  # noqa: E402
from agents.model.belief_heads import _REVEAL_LOGIT  # noqa: E402
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM  # noqa: E402
from main.capacity import load_policy  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402

MODEL = "/home/goodlad/dev/gen3ai/models/ai_v13_22_popr1_loop/final_model.zip"
RUN_DIR = "/home/goodlad/dev/gen3ai/models/ai_v13_22_popr1_loop"
CW_THRESH = 0.8
BATTLE_WALL_S = 1800.0


class FixedTB(Teambuilder):
    def __init__(self, packed):
        self.packed = packed

    def yield_team(self):
        return self.packed


class LabelShim:
    """The pinned Gen3Env label methods, bound onto (battle1=trainee view, battle2=opponent's own)."""
    _belief_labels = Gen3Env._belief_labels
    _spread_labels = Gen3Env._spread_labels
    _nature_ev_map = Gen3Env._nature_ev_map
    _hp_type_labels = Gen3Env._hp_type_labels
    _item_labels = Gen3Env._item_labels

    def __init__(self, mappings):
        self._species_num = {s: r["num"] for s, r in mappings.get("species", {}).items() if "num" in r}
        self._move_num = {m: r["num"] for m, r in mappings.get("moves", {}).items() if "num" in r}
        self._emit_known_moves = True
        self._nature_ev_cache_key = None
        self._nature_ev_cache = {}
        self.battle1 = None
        self.battle2 = None

    def labels(self, obs_vec, b1, b2):
        self.battle1, self.battle2 = b1, b2
        out = {}
        out.update(self._belief_labels(obs_vec))
        out.update(self._spread_labels(obs_vec))
        out.update(self._hp_type_labels(obs_vec))
        out.update(self._item_labels(obs_vec))
        return out


def _perms(k):
    return np.array(list(itertools.permutations(range(k))), dtype=np.int64)


_PERMS = {k: _perms(k) for k in range(1, 7)}


def hungarian(cost):
    """cost [k,k] (pred i, target j) → best permutation p (pred i ↔ target p[i]), exact."""
    k = cost.shape[0]
    P = _PERMS[k]
    tot = cost[np.arange(k)[None, :], P].sum(1)
    return P[int(np.argmin(tot))]


def log_softmax_np(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def bce_mean(logit_or_none, prob, target):
    """Mean over channels of BCE. Pass logits (exact, torch-equivalent) or probs."""
    if logit_or_none is not None:
        x = logit_or_none
        return float(np.mean(np.maximum(x, 0) - x * target + np.log1p(np.exp(-np.abs(x)))))
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(np.mean(-(target * np.log(p) + (1 - target) * np.log(1 - p))))


class Scorer:
    """Holds the Smogon-prior buffers (read from the loaded model's own non-persistent buffers)."""

    def __init__(self, model):
        fe = model.policy.features_extractor
        self.fe = fe
        t0 = fe.t0_species_prior
        self.sp_marg = t0.species_prior_log_marginal
        self.sp_lift = t0.species_prior_log_lift
        self.move_prior = fe.move_belief.move_prior_logits           # [S, M]
        self.hp_prior = fe.hp_type_belief_head.hp_prior               # [S, 16]
        self.item_prior = fe.item_belief_head.item_prior             # [S, I]
        sb = fe.spread_belief
        self.nat_logprior, self.ev_prior = sb.nature_logprior, sb.ev_prior
        self.nature_mult, self.base_nonhp = sb.nature_mult, sb.base_nonhp
        self.hp_typed_nums = fe.hp_type_belief_head.HP_TYPED_NUMS.numpy()
        S, M = self.move_prior.shape
        # Per-species typed Smogon move PROBABILITY table [S, M] (no reveals, no narrowing).
        with torch.no_grad():
            raw = self.move_prior.clone().unsqueeze(0)                                    # [1,S,M]
            post = self.hp_prior.clamp_min(1e-6)
            post = (post / post.sum(-1, keepdim=True)).unsqueeze(0)                     # [1,S,16]
            typed, _ = fe.hp_type_belief_head.compose_typed_hp(
                raw, post, torch.zeros(1, S, 16), torch.zeros(1, S, 4, dtype=torch.long))
            self.species_move_prob = torch.sigmoid(typed[0]).numpy()                      # [S,M]
        self.hidden_ref = {}          # slot position → first hidden-slot move logits seen (constancy check)
        self.hidden_maxdev = 0.0

    def score(self, obs_vec, mask_vec, lab, rows, counters):
        fe = self.fe
        with torch.no_grad():
            ctx = fe.unpack({"observation": torch.as_tensor(obs_vec[None]).float(),
                             "action_mask": torch.as_tensor(mask_vec[None]).float()})
        opp_ids = ctx.species_ids[0, TEAM_SIZE:2 * TEAM_SIZE].long()                     # [6]
        believed = ctx.opp_believed_mask[0].numpy().astype(bool)                          # [6]
        opp_mv = ctx.all_move_ids[0, TEAM_SIZE:, :].long()                                # [6,4]
        k_rev = int((~believed).sum())
        sk = np.array([obs_vec[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET]
                       for i in range(TEAM_SIZE)])
        assert np.array_equal(sk < 0.5, believed), "believed mask disagrees with obs species_known"
        stash = fe.stash
        assert bool(torch.equal(stash.opp_believed_mask[0].cpu(), ctx.opp_believed_mask[0])), \
            "stash belief mask is not this decision's"

        # ---------------- SPECIES (hidden slots; BeliefHead) ----------------
        bs = lab["belief_species"]
        slots = np.where(bs >= 0)[0]
        if len(slots):
            counters["species_dec"] += 1
            assert set(slots.tolist()) == set(np.where(believed)[0].tolist()) or len(slots) < believed.sum(), \
                "label believed slots differ from the model's believed slots"
            tgt = bs[slots]
            head_lp = log_softmax_np(stash.belief_logits["species"][0].double().numpy()[slots])   # [k,S]
            with torch.no_grad():
                prior_lp = species_team_prior_logits(self.sp_marg, self.sp_lift, opp_ids[None],
                                                     ctx.opp_believed_mask)[0].double().numpy()
            prior_lp = np.broadcast_to(prior_lp, head_lp.shape)
            tset = set(tgt.tolist())
            for lp, tag in ((head_lp, "h"), (prior_lp, "p")):
                cost = -lp[:, tgt]                                                          # [k,k]
                perm = hungarian(cost)
                matched = tgt[perm]
                nll = -lp[np.arange(len(slots)), matched]
                am = lp.argmax(-1)
                conf = np.exp(lp.max(-1))
                corr = (am == matched)
                cw_set = (conf > CW_THRESH) & np.array([a not in tset for a in am])
                cw_m = (conf > CW_THRESH) & ~corr
                if tag == "h":
                    h = (nll, corr, conf, cw_set, cw_m)
                else:
                    p = (nll, corr, conf, cw_set, cw_m)
            for j in range(len(slots)):
                rows["species"].append([k_rev, float(h[0][j]), int(h[1][j]), float(h[2][j]), int(h[3][j]),
                                        int(h[4][j]), float(p[0][j]), int(p[1][j]), float(p[2][j]),
                                        int(p[3][j]), int(p[4][j])])

        # ---------------- MOVES (typed MoveBelief posterior) ----------------
        ml = stash.move_belief_logits[0].double().numpy()                                   # [6,M]
        M = ml.shape[-1]
        hp_typed = set(self.hp_typed_nums.tolist())
        # prior posterior for revealed slots: species row, reveals pinned, typed with the HP prior
        with torch.no_grad():
            raw = self.move_prior[opp_ids].clone()                                          # [6,M]
            valid = opp_mv > 0
            ids = opp_mv.clamp(0, M - 1)
            rev = torch.zeros_like(raw, dtype=torch.bool)
            rev.scatter_(-1, ids, valid)
            raw = torch.where(rev, torch.tensor(_REVEAL_LOGIT, dtype=raw.dtype), raw)
            post = self.hp_prior[opp_ids].clamp_min(1e-6)
            post = post / post.sum(-1, keepdim=True)
            prior_rev, _ = self.fe.hp_type_belief_head.compose_typed_hp(
                raw[None], post[None], ctx.hp_probs[:, TEAM_SIZE:], opp_mv[None])
            prior_rev = prior_rev[0].double().numpy()                                       # [6,M] logits
        km = lab["known_moves"]
        for s in np.where(~believed)[0]:
            true = [int(x) for x in km[s] if x >= 0]
            if not true:
                continue
            revealed = set(int(x) for x in opp_mv[s].tolist() if x > 0)
            excl = set(revealed)
            if HIDDEN_POWER_MOVE_NUM in revealed:
                excl |= hp_typed
            hidden_true = [m for m in true if m not in excl]
            r = len(revealed)
            if not hidden_true or r >= 4:
                continue
            tvec = np.zeros(M)
            tvec[true] = 1.0
            out = [k_rev, len(hidden_true)]
            for lg in (ml[s], prior_rev[s]):
                cand = np.array([m for m in range(M) if m not in excl and m != HIDDEN_POWER_MOVE_NUM])
                top = cand[np.argsort(-lg[cand], kind="stable")[: 4 - r]]
                hits = len(set(top.tolist()) & set(hidden_true))
                out += [bce_mean(lg, None, tvec), hits]
            rows["moves_rev"].append(out)
        # hidden slots
        bm = lab["belief_moves"]
        hs = [s for s in np.where(believed)[0] if (bm[s] >= 0).any()]
        for s in np.where(believed)[0]:
            ref = self.hidden_ref.setdefault(int(s), ml[s].copy())
            self.hidden_maxdev = max(self.hidden_maxdev, float(np.abs(ml[s] - ref).max()))
        if hs:
            tg = np.zeros((len(hs), M))
            for j, s in enumerate(hs):
                tg[j, [int(x) for x in bm[s] if x >= 0]] = 1.0
            preds = ml[hs]                                                                  # [k,M]
            perm = hungarian(-(preds @ tg.T))
            with torch.no_grad():
                p_t0 = species_team_prior_logits(self.sp_marg, self.sp_lift, opp_ids[None],
                                                 ctx.opp_believed_mask)[0].exp().double().numpy()
            mix = p_t0 @ self.species_move_prob.astype(np.float64)                          # [M]
            top_mix = set(np.argsort(-mix, kind="stable")[:4].tolist())
            for j in range(len(hs)):
                t = tg[perm[j]]
                tl = np.where(t > 0)[0].tolist()
                top_h = set(np.argsort(-preds[j], kind="stable")[:4].tolist())
                rows["moves_hid"].append([k_rev, len(tl),
                                          bce_mean(preds[j], None, t), len(top_h & set(tl)),
                                          bce_mean(None, mix, t), len(top_mix & set(tl))])

        # ---------------- ITEM (revealed slots, item still unknown on the board) ----------------
        il, imask = lab["item_label"], lab["item_mask"]
        it_logits = stash.item_logits[0].double().numpy()
        rev_mons = [m for m in ObservationEncoder.get_team_list(counters["_b1"], is_opponent=True)
                    if m is not None]
        for s in np.where(imask > 0.5)[0]:
            mon = rev_mons[s] if s < len(rev_mons) else None
            if mon is None:
                counters["item_slot_nomon"] += 1
                continue
            if mon.item != GenData.UNKNOWN_ITEM:
                counters["item_known_skipped"] += 1
                continue
            y = int(il[s])
            h = log_softmax_np(it_logits[s])
            pr = log_softmax_np(np.log(np.clip(self.item_prior[opp_ids[s]].double().numpy(), 1e-6, None)))
            rows["item"].append([k_rev, float(-h[y]), int(h.argmax() == y), float(-pr[y]), int(pr.argmax() == y)])

        # ---------------- HP TYPE (revealed slots whose true mon runs HP) ----------------
        hl, hm = lab["hp_type_label"], lab["hp_type_mask"]
        hp_logits = stash.hp_type_logits[0].double().numpy()
        for s in np.where(hm > 0.5)[0]:
            y = int(hl[s])
            h = log_softmax_np(hp_logits[s])
            pr = log_softmax_np(np.log(np.clip(self.hp_prior[opp_ids[s]].double().numpy(), 1e-6, None)))
            rows["hp"].append([k_rev, float(-h[y]), int(h.argmax() == y), float(-pr[y]), int(pr.argmax() == y)])

        # ---------------- SPREAD (nature/EV generative head; revealed slots) ----------------
        nat_l = stash.spread_nature_logits[0].double().numpy()
        ev_h = stash.spread_ev[0].double().numpy()
        der_h = stash.spread_belief[0].double().numpy()
        nmask, nlab = lab["belief_nature_mask"], lab["belief_nature"]
        evmask, evlab = lab["belief_ev_mask"], lab["belief_ev"]
        spmask, splab = lab["belief_spread_mask"], lab["belief_spread"]
        with torch.no_grad():
            nlp = self.nat_logprior[opp_ids]                                               # [6,25]
            e_mult = torch.softmax(nlp, -1) @ self.nature_mult
            ev_p = self.ev_prior[opp_ids].clamp(0.0, 252.0)
            der_p = ((2.0 * self.base_nonhp[opp_ids] + 31.0 + ev_p / 4.0 + 5.0) * e_mult).clamp(min=1.0)
        nlp, ev_p, der_p = nlp.double().numpy(), ev_p.double().numpy(), der_p.double().numpy()
        for s in range(TEAM_SIZE):
            if nmask[s] < 0.5 or spmask[s] < 0.5 or evmask[s] < 0.5:
                continue
            y = int(nlab[s])
            h = log_softmax_np(nat_l[s])
            pr = log_softmax_np(nlp[s])
            rows["spread"].append([k_rev,
                                   float(-h[y]), int(h.argmax() == y), float(np.abs(ev_h[s] - evlab[s]).mean()),
                                   float(np.abs(der_h[s] - splab[s]).mean()),
                                   float(-pr[y]), int(pr.argmax() == y), float(np.abs(ev_p[s] - evlab[s]).mean()),
                                   float(np.abs(der_p[s] - splab[s]).mean())])


class ProbePlayer(RLPlayer):
    """The trainee: the model greedy, scoring its belief stashes at every COMMITTED decision."""

    def setup_probe(self, opp_player, scorer, shim):
        self._opp = opp_player
        self._scorer = scorer
        self._shim = shim
        self._pending = None
        self.reset_probe()

    def reset_probe(self):
        self.rows = {k: [] for k in ("species", "moves_rev", "moves_hid", "item", "hp", "spread")}
        self.counters = {"decisions": 0, "species_dec": 0, "label_race_skipped": 0, "item_known_skipped": 0,
                         "item_slot_nomon": 0, "species_labelset_mismatch": 0}

    def embed_battle(self, battle):
        out = super().embed_battle(battle)
        # Labels immediately after the obs snapshot, from the opponent player's OWN battle view.
        b2 = self._opp.battles.get(battle.battle_tag)
        self._probe_obs = (out, battle, b2)
        return out

    def _predict_best_action(self, battle, stochastic=False, need_aux=True, temperature=1.0):
        idx, probs, mask = super()._predict_best_action(battle, stochastic=stochastic, need_aux=False,
                                                        temperature=temperature)
        self._pending = None
        if idx is None:
            return idx, probs, mask
        out, b1, b2 = self._probe_obs
        if b2 is None:
            self.counters["label_race_skipped"] += 1
            return idx, probs, mask
        obs_vec = np.asarray(out["observation"], dtype=np.float32)
        n_known = int(sum(1 for i in range(TEAM_SIZE)
                          if obs_vec[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET] >= 0.5))
        n_rev_board = len([m for m in ObservationEncoder.get_team_list(b1, is_opponent=True) if m is not None])
        if n_known != n_rev_board:
            # the board advanced between the obs snapshot and now (POKE_LOOP race) — skip, count
            self.counters["label_race_skipped"] += 1
            return idx, probs, mask
        lab = self._shim.labels(obs_vec, b1, b2)
        # label sanity (the belief_labels fuzz invariant 3): labelled set == actual hidden set
        revealed = {to_id_str(m.species) for m in ObservationEncoder.get_team_list(b1, is_opponent=True) if m}
        hidden_nums = {self._shim._species_num[to_id_str(m.species)] for m in b2.team.values()
                       if to_id_str(m.species) not in revealed and to_id_str(m.species) in self._shim._species_num}
        if {int(x) for x in lab["belief_species"] if x >= 0} != hidden_nums:
            self.counters["species_labelset_mismatch"] += 1
        rows = {k: [] for k in self.rows}
        self.counters["_b1"] = b1
        self._scorer.score(obs_vec, np.asarray(out["action_mask"]), lab, rows, self.counters)
        self.counters.pop("_b1", None)
        self._pending = rows
        return idx, probs, mask

    def choose_move(self, battle):
        self._pending = None
        order = super().choose_move(battle)
        if self._pending is not None:
            self.counters["decisions"] += 1
            for k, v in self._pending.items():
                self.rows[k].extend(v)
            self._pending = None
        return order


def parse_indices(spec):
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arms", default="pool,ladder,procedural")
    ap.add_argument("--indices", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="print shapes/counters only, never metrics")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    packed = man["packed"]
    idxs = [i for i in parse_indices(args.indices) if i % args.n_workers == args.worker]
    arms = args.arms.split(",")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"rows_w{args.worker}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            r = json.loads(ln)
            done.add((r["arm"], r["i"]))

    trainee_model, _, _ = load_policy(MODEL, RUN_DIR, "cpu")
    opp_model, _, _ = load_policy(MODEL, RUN_DIR, "cpu")
    trainee_model.policy.set_training_mode(False)
    for _m in (trainee_model, opp_model):   # print-only board dumps; not part of the forward
        _m.policy.features_extractor.disable_observation_debugger()
    opp_model.policy.set_training_mode(False)
    mappings = load_mappings()
    scorer = Scorer(trainee_model)
    shim = LabelShim(mappings)
    fmt = "gen3ou"
    tag = f"bc{args.worker}"

    def make_players(gen):
        opp = RLPlayer(model=opp_model, team=FixedTB(""), battle_format=fmt, server_configuration=LocalhostServerConfiguration,
                       mappings=mappings, account_configuration=AccountConfiguration(f"{tag}o{gen}", None),
                       start_listening=False, max_concurrent_battles=1, stochastic=False)
        tr = ProbePlayer(model=trainee_model, team=FixedTB(""), battle_format=fmt, server_configuration=LocalhostServerConfiguration,
                         mappings=mappings, account_configuration=AccountConfiguration(f"{tag}t{gen}", None),
                         start_listening=False, max_concurrent_battles=1, stochastic=False)
        tr.setup_probe(opp, scorer, shim)
        return tr, opp

    gen = 0
    tr, opp = make_players(gen)
    with open(out_path, "a") as fh:
        for i in idxs:
            b = man["battles"][i]
            for arm in arms:
                if (arm, i) in done:
                    continue
                tr._team = FixedTB(packed[b["trainee"]])
                opp._team = FixedTB(packed[b["opp"][arm]])
                tr.reset_probe()
                t0 = time.time()
                status, err, winner, turns = "ok", None, None, None
                try:
                    asyncio.run(asyncio.wait_for(
                        run_local_battles(tr, opp, 1, seed_base=b["seed_base"], impl="rust"),
                        timeout=BATTLE_WALL_S))
                    bt = list(tr.battles.values())
                    if len(bt) != 1 or not bt[0].finished:
                        status = "unfinished"
                    else:
                        winner = 1 if bt[0].won else (0 if bt[0].lost else -1)
                        turns = bt[0].turn
                except asyncio.TimeoutError:
                    status, err = "timeout", f"wall>{BATTLE_WALL_S}s"
                except Exception as e:  # noqa: BLE001 — recorded as INCONCLUSIVE, never a result
                    status, err = "error", f"{type(e).__name__}: {e}"[:400]
                    traceback.print_exc()
                rec = {"arm": arm, "i": i, "status": status, "err": err, "wall_s": round(time.time() - t0, 1),
                       "winner": winner, "turns": turns,
                       "counters": {k: v for k, v in tr.counters.items() if not k.startswith("_")},
                       "hidden_move_maxdev": scorer.hidden_maxdev,
                       "rows": tr.rows}
                if args.smoke:
                    print(json.dumps({"arm": arm, "i": i, "status": status, "err": err, "wall_s": rec["wall_s"],
                                      "counters": rec["counters"],
                                      "n_rows": {k: len(v) for k, v in tr.rows.items()},
                                      "row_width": {k: (len(v[0]) if v else 0) for k, v in tr.rows.items()},
                                      "hidden_move_maxdev": scorer.hidden_maxdev}), flush=True)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if status == "ok":
                    tr.reset_battles()
                    opp.reset_battles()
                else:   # an unfinished battle cannot be reset — replace both players instead
                    gen += 1
                    tr, opp = make_players(gen)


if __name__ == "__main__":
    main()
