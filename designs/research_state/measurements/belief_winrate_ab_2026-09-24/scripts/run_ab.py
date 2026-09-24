"""Play the paired battles of ONE worker shard of the belief win-rate A/B (PREDICTION.md §2).

Run with the PINNED tree (cwd = the pin worktree, PYTHONPATH=<pin>/src,
POKESIM_SIM_BRIDGE_BIN=<pin>/src/rust_sim/target/release/sim_bridge):

    python run_ab.py --manifest /tmp/belief_wr/manifest_full.json \
        --cells pool:learned,pool:prior,ladder:learned,ladder:prior \
        --indices 0-1999 --worker 0 --n-workers 3 --out /tmp/belief_wr/rows

Cell = (column, belief). Battle i of every cell: same trainee (our) team, same sim seed, trainee = p1
(the model GREEDY, belief LEARNED or PRIOR-ONLY via `prior_switch`), opponent = p2 (a SEPARATE,
UNMODIFIED copy of the model, GREEDY). Only the opponent TEAM differs across columns and only the
switch differs across belief rows.

SIDE READ (learned cells): at every committed trainee decision the same obs is also run with the
switch ON ("all" and "move" scopes), and whether the greedy action changes is recorded with k = the
number of opponent species revealed in the obs.

PROOF (`--proof-decisions N`, learned cells): at the first N decisions of each battle, also checks
(a) switch OFF == an independently loaded unmodified model, byte for byte (pi logits, value, every
belief stash), (b) switch ON("all") ⇒ every covered belief output equals its prior buffer formula
exactly, (c) switch ON("move") ⇒ the move posterior equals the prior move row (reveals pinned,
composed with the LEARNED HP-type posterior). Failures are COUNTED, not raised.
Nothing is written under models/.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poke_env import AccountConfiguration, LocalhostServerConfiguration  # noqa: E402
from poke_env.teambuilder import Teambuilder  # noqa: E402

from agents.inference.player import RLPlayer  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402
from agents.observation.constants import (  # noqa: E402
    OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET, TEAM_SIZE)
from agents.model.belief_heads import _REVEAL_LOGIT  # noqa: E402
from main.capacity import load_policy  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402

from prior_switch import PriorOnlySwitch  # noqa: E402

MODEL = "/home/goodlad/dev/gen3ai/models/ai_v13_22_popr1_loop/final_model.zip"
RUN_DIR = "/home/goodlad/dev/gen3ai/models/ai_v13_22_popr1_loop"
BATTLE_WALL_S = 1800.0
STASH_KEYS = ("move_belief_logits", "hp_type_logits", "item_logits", "spread_belief",
              "spread_nature_logits", "spread_ev")


class FixedTB(Teambuilder):
    def __init__(self, packed):
        self.packed = packed

    def yield_team(self):
        return self.packed


def _policy_in(obs, mask):
    return {"observation": torch.as_tensor(np.asarray(obs, dtype=np.float32)[None]),
            "action_mask": torch.as_tensor(np.asarray(mask, dtype=np.float32)[None])}


def _forward(model, pin):
    """(masked greedy action, logits, value, stash snapshot) for one obs — the player's own recipe."""
    with torch.no_grad():
        dist = model.policy.get_distribution(pin)
        logits = dist.distribution.logits
        masked = logits + (pin["action_mask"] - 1.0) * 1e9
        idx = int(torch.argmax(masked, dim=1).item())
        st = model.policy.features_extractor.stash
        snap = {k: (getattr(st, k).clone() if getattr(st, k) is not None else None) for k in STASH_KEYS}
        snap["species"] = st.belief_logits["species"].clone() if st.belief_logits else None
        value = model.policy.predict_values(pin)
    return idx, logits.clone(), value.clone(), snap


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return bool(torch.equal(a, b))


class Prover:
    """Expected prior-only tensors, built from the model's own non-persistent prior buffers with the
    same ops the modules use (delta := 0)."""

    def __init__(self, model):
        self.fe = model.policy.features_extractor

    def expected(self, pin, hp_post_for_move=None):
        fe = self.fe
        with torch.no_grad():
            ctx = fe.unpack(pin)
            ids = ctx.species_ids[:, TEAM_SIZE:]
            mv = ctx.all_move_ids[:, TEAM_SIZE:, :]
            out = {}
            # species (BeliefHead side readout): 0 + team prior, broadcast over slots
            sp = fe.belief_head.species_prior_logits(ids, ctx.opp_believed_mask)
            out["species"] = torch.zeros(sp.shape[0], TEAM_SIZE, sp.shape[-1]) + sp
            # HP type
            hp_prior = fe.hp_type_belief_head.hp_prior[ids].clamp_min(1e-6)
            hp_logits = torch.zeros_like(hp_prior) + torch.log(hp_prior)
            out["hp_type_logits"] = hp_logits
            hp_post = torch.softmax(hp_logits, dim=-1) if hp_post_for_move is None else hp_post_for_move
            # moves: 0 + prior row, reveals pinned, typed-HP composed
            raw = torch.zeros(ids.shape[0], TEAM_SIZE, fe.move_belief.move_prior_logits.shape[-1]) \
                + fe.move_belief.move_prior_logits[ids]
            valid = mv > 0
            idc = mv.clamp(0, raw.shape[-1] - 1)
            rev = torch.zeros_like(raw, dtype=torch.bool)
            rev.scatter_(-1, idc, valid)
            raw = torch.where(rev, _REVEAL_LOGIT, raw)
            typed, _ = fe.hp_type_belief_head.compose_typed_hp(raw, hp_post, ctx.hp_probs[:, TEAM_SIZE:], mv)
            out["move_belief_logits"] = typed
            # item
            ip = fe.item_belief_head.item_prior[ids].clamp_min(1e-6)
            out["item_logits"] = torch.zeros_like(ip) + torch.log(ip)
            # spread (nature/EV generative)
            sb = fe.spread_belief
            nat = sb.nature_logprior[ids] + torch.zeros_like(sb.nature_logprior[ids])
            out["spread_nature_logits"] = nat
            e_mult = torch.softmax(nat, dim=-1) @ sb.nature_mult
            ev = (sb.ev_prior[ids] + torch.zeros_like(sb.ev_prior[ids]) * 64.0).clamp(0.0, 252.0)
            out["spread_ev"] = ev
            out["spread_belief"] = ((2.0 * sb.base_nonhp[ids] + 31.0 + ev / 4.0 + 5.0) * e_mult).clamp(min=1.0)
        return out


class ABPlayer(RLPlayer):
    def setup_ab(self, sw_all, sw_move, ref_model, prover):
        self._sw_all, self._sw_move = sw_all, sw_move
        self._ref, self._prover = ref_model, prover
        self.side_read = False
        self.proof_decisions = 0
        self.reset_ab()

    def reset_ab(self):
        self.side = []
        self.proof = {"n": 0, "off_eq_ref": 0, "on_all_eq_prior": 0, "on_move_eq_prior": 0,
                      "on_all_logits_differ": 0, "fail": []}
        self.n_dec = 0
        self._pending = None

    def embed_battle(self, battle):
        out = super().embed_battle(battle)
        self._last_obs = out
        return out

    def _predict_best_action(self, battle, stochastic=False, need_aux=True, temperature=1.0):
        idx, probs, mask = super()._predict_best_action(battle, stochastic=stochastic, need_aux=False,
                                                        temperature=temperature)
        self._pending = None
        if idx is None or not self.side_read:
            self._pending = {}
            return idx, probs, mask
        out = self._last_obs
        obs = np.asarray(out["observation"], dtype=np.float32)
        k = int(sum(1 for s in range(TEAM_SIZE)
                    if obs[OFFSET_OPP_TEAM + s * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET] >= 0.5))
        pin = _policy_in(obs, out["action_mask"])
        i_off, lg_off, v_off, st_off = _forward(self.model, pin)
        with self._sw_all:
            i_all, lg_all, v_all, st_all = _forward(self.model, pin)
        with self._sw_move:
            i_mv, _, _, st_mv = _forward(self.model, pin)
        rec = {"side": [k, int(i_off != idx), int(i_all != idx), int(i_mv != idx)]}
        if self.n_dec < self.proof_decisions:
            pr = {"off_eq_ref": 0, "on_all_eq_prior": 0, "on_move_eq_prior": 0, "on_all_logits_differ": 0,
                  "fail": []}
            i_ref, lg_ref, v_ref, st_ref = _forward(self._ref, pin)
            ok = i_ref == i_off and _eq(lg_ref, lg_off) and _eq(v_ref, v_off) and \
                all(_eq(st_ref[x], st_off[x]) for x in st_ref)
            pr["off_eq_ref"] = int(ok)
            if not ok:
                pr["fail"].append("off!=ref")
            exp = self._prover.expected(pin)
            bad = [x for x in exp if not _eq(exp[x], st_all[x])]
            pr["on_all_eq_prior"] = int(not bad)
            if bad:
                pr["fail"].append("all:" + ",".join(bad))
            learned_hp_post = torch.softmax(st_mv["hp_type_logits"], dim=-1)
            exp_mv = self._prover.expected(pin, hp_post_for_move=learned_hp_post)
            # The HP-type head reads the PRE-reinjection tokens, so it must be untouched by the move-only
            # switch. Item + spread read the tokens AFTER the move reinjection (extractor_forward
            # `_spread_hp_damage(role_tokens[:, TEAM_SIZE:])`), so they legitimately move and are not checked.
            okm = _eq(exp_mv["move_belief_logits"], st_mv["move_belief_logits"]) and \
                _eq(st_mv["hp_type_logits"], st_off["hp_type_logits"])
            pr["on_move_eq_prior"] = int(okm)
            if not okm:
                pr["fail"].append("move")
            pr["on_all_logits_differ"] = int(not _eq(lg_all, lg_off))
            rec["proof"] = pr
        self._pending = rec
        return idx, probs, mask

    def choose_move(self, battle):
        self._pending = None
        order = super().choose_move(battle)
        if self._pending is not None:
            self.n_dec += 1
            if "side" in self._pending:
                self.side.append(self._pending["side"])
            pr = self._pending.get("proof")
            if pr:
                self.proof["n"] += 1
                for key in ("off_eq_ref", "on_all_eq_prior", "on_move_eq_prior", "on_all_logits_differ"):
                    self.proof[key] += pr[key]
                self.proof["fail"].extend(pr["fail"])
            self._pending = None
        return order


def parse_indices(spec):
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", default="pool:learned,pool:prior,ladder:learned,ladder:prior")
    ap.add_argument("--indices", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-side-read", action="store_true")
    ap.add_argument("--proof-decisions", type=int, default=0)
    ap.add_argument("--tag", default="ab")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    packed = man["packed"]
    idxs = [i for i in parse_indices(args.indices) if i % args.n_workers == args.worker]
    cells = [tuple(c.split(":")) for c in args.cells.split(",")]
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"rows_w{args.worker}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            r = json.loads(ln)
            done.add((r["col"], r["belief"], r["i"]))

    trainee_model, _, _ = load_policy(MODEL, RUN_DIR, "cpu")
    opp_model, _, _ = load_policy(MODEL, RUN_DIR, "cpu")
    ref_model = None
    if args.proof_decisions:
        ref_model, _, _ = load_policy(MODEL, RUN_DIR, "cpu")
    for _m in (trainee_model, opp_model, ref_model):
        if _m is not None:
            _m.policy.set_training_mode(False)
            _m.policy.features_extractor.disable_observation_debugger()
    sw_all = PriorOnlySwitch(trainee_model, "all")
    sw_move = PriorOnlySwitch(trainee_model, "move")
    prover = Prover(trainee_model)
    print(json.dumps({"covered_all": sw_all.covered, "covered_move": sw_move.covered}), flush=True)
    mappings = load_mappings()
    fmt = "gen3ou"
    tag = f"{args.tag}{args.worker}"

    def make_players(gen):
        opp = RLPlayer(model=opp_model, team=FixedTB(""), battle_format=fmt,
                       server_configuration=LocalhostServerConfiguration, mappings=mappings,
                       account_configuration=AccountConfiguration(f"{tag}o{gen}", None),
                       start_listening=False, max_concurrent_battles=1, stochastic=False)
        tr = ABPlayer(model=trainee_model, team=FixedTB(""), battle_format=fmt,
                      server_configuration=LocalhostServerConfiguration, mappings=mappings,
                      account_configuration=AccountConfiguration(f"{tag}t{gen}", None),
                      start_listening=False, max_concurrent_battles=1, stochastic=False)
        tr.setup_ab(sw_all, sw_move, ref_model, prover)
        return tr, opp

    gen = 0
    tr, opp = make_players(gen)
    with open(out_path, "a") as fh:
        for i in idxs:
            b = man["battles"][i]
            for col, belief in cells:
                if (col, belief) != ("pool", "learned") and (col, belief) not in \
                        {("pool", "prior"), ("ladder", "learned"), ("ladder", "prior"),
                         ("pool", "move"), ("ladder", "move")}:
                    raise ValueError(f"unknown cell {col}:{belief}")
                if (col, belief, i) in done:
                    continue
                tr._team = FixedTB(packed[b["trainee"]])
                opp._team = FixedTB(packed[b["opp"][col]])
                tr.reset_ab()
                tr.side_read = (belief == "learned") and not args.no_side_read
                tr.proof_decisions = args.proof_decisions if belief == "learned" else 0
                sw = {"learned": None, "prior": sw_all, "move": sw_move}[belief]
                calls0 = sw.calls if sw else 0
                t0 = time.time()
                status, err, winner, turns = "ok", None, None, None
                try:
                    if sw is not None:
                        sw.on = True
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
                finally:
                    sw_all.on = False
                    sw_move.on = False
                rec = {"col": col, "belief": belief, "i": i, "status": status, "err": err,
                       "wall_s": round(time.time() - t0, 1), "winner": winner, "turns": turns,
                       "n_dec": tr.n_dec, "switch_calls": (sw.calls - calls0) if sw else 0,
                       "side": tr.side, "proof": tr.proof if tr.proof_decisions else None}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if status == "ok":
                    tr.reset_battles()
                    opp.reset_battles()
                else:
                    gen += 1
                    tr, opp = make_players(gen)


if __name__ == "__main__":
    main()
