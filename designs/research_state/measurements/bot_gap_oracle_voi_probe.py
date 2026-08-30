"""SI-2 measurement 2 — decision-level oracle-opponent VoI on recorded bot games.

For sampled decisions in the generalist's (ai_v9_29_rev1) bot games, build the full
(our action x opp action) one-ply grid with CRN dice via reroll_many, materialize each
successor's one-sided obs, and score it with the checkpoint's critic + win-prob head.

Then compare, per decision, three one-ply policies:
  - chosen        : the action the policy actually played
  - best_marginal : argmax_a  E_{b ~ alpha/beta (the model's RECORDED intent read)} WP(a,b)
  - best_oracle   : argmax_a  WP(a, b_true)  where b_true = the RECORDED bot action
                    (for `random`, the true distribution is uniform-over-legal instead)

VoI (the ceiling of skill-conditioning at this state, one-ply):
    voi_wp = WP(best_oracle, b_eval) - WP(best_marginal, b_eval)
evaluated under the ORACLE opponent model (b_eval = recorded action; uniform for random).
Also reported: oracle_gain_over_chosen (includes the pure one-ply-search effect) and
marginal_gain_over_chosen (one-ply search WITHOUT oracle knowledge) — their difference
is again the oracle-specific increment.

Run:
  export PYTHONPATH=$PYTHONPATH:src
  nice -n 15 python3 tmp/si2/oracle_voi.py --steps 24000000 [--max-loss-battles N] [--out FILE]
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
import traceback

import numpy as np

from agents.training.obs_materializer import Branch, materialize_branches, materialize_from_record
from main.prober.falsifier import select_anchors
from main.prober.session import ProbeSession
from utils.bridge.reconstruction import ReconstructionRecord, reroll_many

RUN = "/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823"
BOTS = ["heuristic", "heuristic2", "staller", "staller_v2", "aggressive",
        "aggressive_v2", "setup_sweep", "setup_sweep_v2", "random"]
MAX_OUR = 9
MAX_OPP = 9


def norm_move(name: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    if s.startswith("hiddenpower"):
        return "hiddenpower"
    return s


def opp_legal_from_request(req: dict) -> list[dict]:
    """Enumerate the opponent's legal sim choices from its decision request.

    Returns [{"choice": "move 2"|"switch 3", "kind": "move"|"switch",
              "move_id": str|None, "species": str|None}].
    """
    if not req or req.get("wait"):
        return []
    out = []
    force = bool(req.get("forceSwitch"))
    active = (req.get("active") or [{}])[0] if not force else {}
    trapped = bool(active.get("trapped"))
    if not force:
        for k, mv in enumerate(active.get("moves") or []):
            if mv.get("disabled"):
                continue
            pp = mv.get("pp")
            if pp is not None and pp <= 0:
                continue
            out.append({"choice": f"move {k + 1}", "kind": "move",
                        "move_id": norm_move(mv.get("id") or mv.get("move") or ""),
                        "species": None})
    side = (req.get("side") or {})
    if force or not trapped:
        for j, p in enumerate(side.get("pokemon") or []):
            if p.get("active"):
                continue
            cond = p.get("condition") or ""
            if cond.startswith("0") or cond.endswith("fnt"):
                continue
            det = (p.get("details") or "").split(",")[0]
            out.append({"choice": f"switch {j + 1}", "kind": "switch",
                        "move_id": None, "species": norm_move(det)})
    return out


def match_recorded_choice(rec_choice: str | None, cands: list[dict], req: dict) -> int | None:
    """Map the recorded opp choice string onto the enumerated candidate list.

    poke-env commits choices as ids/names ("move earthquake", "switch salamence");
    the enumerated candidates are positional ("move 2", "switch 3"). Match on the
    move id / species when non-numeric, on position when numeric.
    """
    if not rec_choice:
        return None
    toks = rec_choice.strip().split()
    if not toks:
        return None
    kind = toks[0].lower()
    arg = toks[1] if len(toks) > 1 else ""
    if kind == "move":
        if arg.isdigit():
            want = f"move {arg}"
            for i, c in enumerate(cands):
                if c["choice"] == want:
                    return i
            return None
        mid = norm_move(arg)
        for i, c in enumerate(cands):
            if c["kind"] == "move" and c["move_id"] == mid:
                return i
        return None
    if kind == "switch":
        if arg.isdigit():
            want = f"switch {arg}"
            for i, c in enumerate(cands):
                if c["choice"] == want:
                    return i
            return None
        sp = norm_move(arg)
        for i, c in enumerate(cands):
            if c["kind"] == "switch" and c["species"] == sp:
                return i
        # nickname form: match via side.pokemon ident
        for j, p in enumerate((req.get("side") or {}).get("pokemon") or []):
            ident = norm_move((p.get("ident") or "").split(":")[-1])
            if ident == sp:
                want = f"switch {j + 1}"
                for i, c in enumerate(cands):
                    if c["choice"] == want:
                        return i
        return None
    return None


def alpha_marginal(inv: dict, cands: list[dict]) -> list[float] | None:
    """Map the RECORDED alpha/beta intent read onto the opp's true legal choices."""
    oi = inv.get("opp_intent") or {}
    alpha = oi.get("alpha") or []
    if not alpha or not cands:
        return None
    p_switch = 0.0
    move_named: dict[str, float] = {}
    for row in alpha:
        nm = row.get("name") or ""
        p = float(row.get("p") or 0.0)
        if nm == "SWITCH":
            p_switch += p
        else:
            move_named[norm_move(nm)] = move_named.get(norm_move(nm), 0.0) + p
    move_idx = [i for i, c in enumerate(cands) if c["kind"] == "move"]
    sw_idx = [i for i, c in enumerate(cands) if c["kind"] == "switch"]
    w = [0.0] * len(cands)
    # --- move mass ---
    matched_mass = 0.0
    unmatched_named_mass = sum(move_named.values())
    unmatched_legal: list[int] = []
    for i in move_idx:
        mid = cands[i]["move_id"]
        if mid in move_named:
            w[i] = move_named[mid]
            matched_mass += move_named[mid]
            unmatched_named_mass -= move_named[mid]
        else:
            unmatched_legal.append(i)
    # alpha mass on believed moves the opp can't actually use, spread over the
    # legal moves alpha never named (belief error -> honest uniform residual).
    if unmatched_legal and unmatched_named_mass > 0:
        for i in unmatched_legal:
            w[i] += unmatched_named_mass / len(unmatched_legal)
    elif move_idx and unmatched_named_mass > 0:
        for i in move_idx:
            w[i] += unmatched_named_mass / len(move_idx)
    # --- switch mass, split by beta over matching species ---
    if sw_idx and p_switch > 0:
        beta = oi.get("beta") or []
        bmass: dict[str, float] = {}
        for row in beta:
            sp = norm_move(row.get("species") or "")
            if sp:
                bmass[sp] = bmass.get(sp, 0.0) + float(row.get("p") or 0.0)
        bw = []
        for i in sw_idx:
            bw.append(bmass.get(cands[i]["species"] or "", 0.0))
        tot = sum(bw)
        if tot > 0:
            for i, wt in zip(sw_idx, bw):
                w[i] += p_switch * (wt / tot)
        else:
            for i in sw_idx:
                w[i] += p_switch / len(sw_idx)
    elif not sw_idx:
        pass  # switch mass dies; renormalize below
    s = sum(w)
    if s <= 0:
        return None
    return [x / s for x in w]


def wp_of_terminal(outcome: dict, username: str) -> float:
    wname = outcome.get("winner")
    if not wname:
        return 0.5
    return 1.0 if wname == username else 0.0


def probe_decision(model, record, summary, npz, inv_index, impl="rust"):
    invs = summary["invocations"]
    inv = invs[inv_index]
    if inv.get("phase") != "move_selection":
        raise ValueError("not a move_selection anchor")
    turn = int(inv["turn"])
    actions = np.asarray(npz["actions"], dtype=int)
    chosen_idx = int(actions[inv_index])
    side = record.side_of(record.trainee_username)
    other = "p2" if side == "p1" else "p1"
    username = record.username(side)

    trace = materialize_from_record(record, actions=actions,
                                    map_actions_at=inv_index,
                                    stop_after_decision=inv_index, impl=impl)
    choice_map = dict(trace.action_choices or {})
    if chosen_idx not in choice_map:
        raise RuntimeError("chosen action not legal in replayed state")
    our = sorted(choice_map)[:MAX_OUR]
    if chosen_idx not in our:
        our.append(chosen_idx)

    rr0 = reroll_many(record, turn, arms=(), impl=impl)
    opp_req = rr0.requests.get(other) or {}
    cands = opp_legal_from_request(opp_req)[:MAX_OPP]
    if len(cands) < 2:
        raise ValueError(f"opp has {len(cands)} legal choices — no opponent-model question here")
    rec_choice = (rr0.recorded_choices or {}).get(other)
    rec_b = match_recorded_choice(rec_choice, cands, opp_req)
    if rec_b is None:
        raise ValueError(f"recorded opp choice {rec_choice!r} not among enumerated {[c['choice'] for c in cands]}")

    arms = []
    for a in our:
        a_str = "recorded" if a == chosen_idx else choice_map[a]
        for bi, c in enumerate(cands):
            b_str = "recorded" if bi == rec_b else c["choice"]
            arms.append({f"{side}_action": a_str, f"{other}_action": b_str,
                         "seed": "original", "label": f"{a}:{bi}"})
    rr = reroll_many(record, turn, arms, followup="random", impl=impl)
    prefix_chunks = rr.prefix_p1_chunks if side == "p1" else rr.prefix_p2_chunks
    prefix_actions = [int(x) for x in actions[:inv_index]]

    by_label = {arm.label: arm for arm in rr.arms}
    branch_keys, branch_list = [], []
    WP = {}
    for a in our:
        for bi in range(len(cands)):
            lab = f"{a}:{bi}"
            arm = by_label.get(lab)
            if arm is None or arm.outcome.get("stuck"):
                continue
            if arm.outcome.get("ended"):
                WP[(a, bi)] = ("terminal", wp_of_terminal(arm.outcome, username))
                continue
            branch_keys.append((a, bi))
            branch_list.append(Branch(
                chunks=list(arm.p1_chunks if side == "p1" else arm.p2_chunks),
                actions=[a], label=lab))
    if branch_list:
        mats = materialize_branches(
            list(prefix_chunks), branch_list, username=username,
            packed_team=record.packed_team(side), side=side,
            prefix_actions=prefix_actions, battle_format=record.format_id,
            battle_tag=record.battle_tag, stop_after_decision=inv_index + 1,
            encode_only_at={inv_index + 1})
        for key, mt in zip(branch_keys, mats):
            if len(mt.decisions) > inv_index + 1:
                d = mt.decisions[inv_index + 1]
                wp = model.win_prob_at(d.obs, d.mask)
                v = model.value(d.obs, d.mask)
                WP[key] = ("wp", float(wp) if wp is not None else None, float(v))

    def wp_val(a, bi):
        row = WP.get((a, bi))
        if row is None:
            return None
        if row[0] == "terminal":
            return row[1]
        return row[1]

    # candidate distributions
    marg = alpha_marginal(inv, cands)
    is_random_bot = summary.get("meta", {}).get("opponent") == "random"

    def exp_wp(a, dist):
        tot, mass = 0.0, 0.0
        for bi, p in enumerate(dist):
            v = wp_val(a, bi)
            if v is None or p <= 0:
                continue
            tot += p * v
            mass += p
        return tot / mass if mass > 0 else None

    uniform = [1.0 / len(cands)] * len(cands)
    oracle_dist = uniform if is_random_bot else [1.0 if i == rec_b else 0.0 for i in range(len(cands))]

    scores_oracle = {a: exp_wp(a, oracle_dist) for a in our}
    ok_actions = [a for a in our if scores_oracle[a] is not None]
    if chosen_idx not in ok_actions:
        raise ValueError("chosen action has no scorable successor")
    best_oracle = max(ok_actions, key=lambda a: scores_oracle[a])

    result = {
        "turn": turn, "inv": inv_index, "n_our": len(ok_actions), "n_opp": len(cands),
        "chosen": chosen_idx,
        "recorded_opp_choice": rec_choice,
        "oracle_is_uniform": is_random_bot,
        "wp_chosen_oracle": scores_oracle[chosen_idx],
        "wp_best_oracle": scores_oracle[best_oracle],
        "oracle_gain_over_chosen": scores_oracle[best_oracle] - scores_oracle[chosen_idx],
        "alpha_available": marg is not None,
    }
    if marg is not None:
        scores_marg = {a: exp_wp(a, marg) for a in ok_actions}
        ok2 = [a for a in ok_actions if scores_marg[a] is not None]
        best_marg = max(ok2, key=lambda a: scores_marg[a])
        result.update({
            "best_oracle_action": best_oracle, "best_marginal_action": best_marg,
            "flip": best_oracle != best_marg,
            "wp_best_marginal_under_oracle": scores_oracle[best_marg],
            "voi_wp": scores_oracle[best_oracle] - scores_oracle[best_marg],
            "marginal_gain_over_chosen": scores_oracle[best_marg] - scores_oracle[chosen_idx],
            "alpha_on_recorded": (None if is_random_bot else marg[rec_b]),
            "alpha_switch_mass": sum(p for p, c in zip(marg, cands) if c["kind"] == "switch"),
            "opp_actually_switched": cands[rec_b]["kind"] == "switch",
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[24000000])
    ap.add_argument("--max-loss-battles", type=int, default=999)
    ap.add_argument("--max-win-battles-per-bot", type=int, default=2)
    ap.add_argument("--worst-per-loss", type=int, default=1)
    ap.add_argument("--random-per-battle", type=int, default=1)
    ap.add_argument("--impl", default="rust")
    ap.add_argument("--out", default="tmp/si2/oracle_voi_results.jsonl")
    args = ap.parse_args()

    session = ProbeSession(RUN, impl=args.impl)
    rng = random.Random(20260830)
    out = open(args.out, "a")
    t00 = time.time()
    n_done = 0

    battles = []
    for step in args.steps:
        for bot in BOTS:
            d = f"{RUN}/eval_traces/step_{step}/{bot}"
            losses = sorted(glob.glob(f"{d}/loss_*_summary.json"))[: args.max_loss_battles]
            wins = sorted(glob.glob(f"{d}/win_*_summary.json"))
            rng.shuffle(wins)
            wins = wins[: args.max_win_battles_per_bot]
            battles += [(p, bot, step, "loss") for p in losses]
            battles += [(p, bot, step, "win") for p in wins]

    print(f"{len(battles)} battles queued", flush=True)
    for bnum, (spath, bot, step, outcome) in enumerate(battles):
        try:
            model, _choice = session.probe_model(spath)
            summary = json.load(open(spath))
            summary.setdefault("meta", {})["opponent"] = bot
            npz = dict(np.load(spath.replace("_summary.json", "_states.npz"), allow_pickle=True))
            record = ReconstructionRecord.load(spath.replace("_summary.json", "_reconstruction.json"))
        except Exception as e:
            print(f"[{bnum}] LOADFAIL {spath}: {e}", flush=True)
            continue

        invs = summary.get("invocations", [])
        move_invs = [i for i, v in enumerate(invs)
                     if v.get("phase") == "move_selection" and i + 1 < len(invs)]
        anchors = []
        if outcome == "loss":
            try:
                worst = select_anchors(summary, npz, gamma=0.99, worst=args.worst_per_loss)
                anchors += [(int(i), "crater") for i in worst]
            except Exception:
                pass
        pool = [i for i in move_invs if i not in {a for a, _ in anchors}]
        if pool:
            anchors += [(i, "random") for i in rng.sample(pool, min(args.random_per_battle, len(pool)))]

        for inv_index, stratum in anchors:
            t0 = time.time()
            row = {"battle": spath, "bot": bot, "step": step, "outcome": outcome,
                   "stratum": stratum}
            try:
                row.update(probe_decision(model, record, summary, npz, inv_index,
                                          impl=args.impl))
                row["ok"] = True
            except Exception as e:
                row.update({"ok": False, "inv": inv_index, "error": str(e)[:300]})
            row["secs"] = round(time.time() - t0, 2)
            out.write(json.dumps(row) + "\n")
            out.flush()
            n_done += 1
            if n_done % 10 == 0:
                print(f"[{n_done}] {time.time()-t00:.0f}s elapsed, last={row.get('secs')}s "
                      f"ok={row.get('ok')}", flush=True)
    print(f"DONE {n_done} decisions in {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
