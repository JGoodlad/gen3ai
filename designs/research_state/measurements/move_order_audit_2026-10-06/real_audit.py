"""REAL-DATA alignment audit on the Lane S bank (real eval-trace battles, re-played through the Rust core).

For every answered decision of BOTH viewers:
  A. the obs request block (ids) == the |request| JSON's active moves, by num, in request order
  B. action 6+k's choice token names request move k (by id)
  C. the move the SIM EXECUTED (the viewer's next `|move|` protocol line) == the chosen token's move
  D. the per-mon slots of our active: their order vs request order, and whether the PokemonEncoder's
     POSITIONAL application of request-order legality disagrees with the move-IDENTITY legality.
Usage: real_audit.py [n_battles]
"""
import collections
import re
import sys
from pathlib import Path

import numpy as np
import torch as th

from agents import gen3_data
from agents.battle.rust_core_parity import run_core
from agents.battle.core_obs import wrap_row
from main.policy_spectrum.bank import load_bank

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
bank = load_bank(Path("designs/research_state/measurements/m5_laneS/bank_v1"), verify=False)
battles = bank.battles[:N]
print("battles", len(battles), flush=True)


def norm(s):
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return "hiddenpower" if s.startswith("hiddenpower") else s


CALLERS = {"sleeptalk", "metronome", "mirrormove", "naturepower", "assist"}
rows, metas = [], []
st = collections.Counter()
examples = collections.defaultdict(list)
for i in range(0, len(battles), 32):
    res_list = run_core([b.recorded() for b in battles[i:i + 32]], trackers=True, obs=True)
    for res in res_list:
        if not res["ok"]:
            st["battle_not_ok"] += 1
            continue
        chunks = res["chunks"]
        for v in (0, 1):
            tag = f"p{v + 1}a: "
            request = None
            last_after = -1
            for entry in res["trackers"][v]:
                if "obs" not in entry:
                    continue
                after = int(entry["after"])
                for side, lines in chunks[last_after + 1: after + 1]:
                    if side == v:
                        for line in lines:
                            if line.startswith("|request|"):
                                import json
                                request = json.loads(line[len("|request|"):])
                last_after = after
                choice = entry.get("choice")
                if choice is None or request is None:
                    continue
                st["decisions"] += 1
                tokens = {int(k): t for k, t in entry["tokens"].items()}
                act = request.get("active") or [{}]
                req_moves = [m for m in act[0].get("moves", []) if m.get("id") != "struggle"]
                req_ids = [m["id"] for m in req_moves][:4]
                req_dis = [bool(m.get("disabled")) for m in req_moves][:4]
                # B: token of action 6+k names request move k
                for k, mid in enumerate(req_ids):
                    t = tokens.get(6 + k)
                    if t is None:
                        continue
                    st["B_checked"] += 1
                    tm = t[len("move "):]
                    ok = (norm(tm) == norm(mid)) or (mid == "recharge" and t == "move 1")
                    if not ok:
                        st["B_MISMATCH"] += 1
                        examples["B"].append((res["label"], v, k, mid, t))
                # C: executed move vs chosen token
                if choice.startswith("move ") and choice != "move struggle" and choice != "move 1":
                    chosen = norm(choice[len("move "):])
                    executed = None
                    seen_turn = False
                    for side, lines in chunks[after + 1:]:
                        if side != v:
                            continue
                        for line in lines:
                            if line.startswith(f"|move|{tag}") and "[from]" not in line:
                                executed = norm(line.split("|")[3])
                                break
                            if line.startswith("|turn|") or line.startswith("|win|") or line.startswith("|tie"):
                                seen_turn = True
                                break
                        if executed or seen_turn:
                            break
                    if executed is None:
                        st["C_no_move_line"] += 1
                    elif chosen in CALLERS:
                        st["C_caller_skip"] += 1
                    else:
                        st["C_checked"] += 1
                        if executed != chosen:
                            st["C_MISMATCH"] += 1
                            examples["C"].append((res["label"], v, entry.get("after"), choice, executed))
                rows.append(np.array(wrap_row(entry["obs"])))
                metas.append((res["label"], v, req_ids, req_dis, np.asarray(entry["mask"])))
    print("progress", i + 32, dict(st), flush=True)

from agents.training import learner_golden as LG
from main.train.production_args import production_args

m = LG.build_learner(args=production_args(), perturb_keyed=False)
fe = m.policy.features_extractor
obs = th.as_tensor(np.stack(rows))
HP_NUMS = {237} | {gen3_data.moves.get("hiddenpower" + t).num for t in ["bug", "dark", "dragon", "electric",
            "fighting", "fire", "flying", "ghost", "grass", "ground", "ice", "poison", "psychic", "rock", "steel", "water"]}
num = lambda mid: float(gen3_data.moves.get(mid).num) if gen3_data.moves.get(mid) else -1.0
for j0 in range(0, len(rows), 4096):
    ctx = fe.unpack({"observation": obs[j0:j0 + 4096]})
    ar = th.arange(ctx.batch_size)
    srt = ctx.all_move_ids[ar, ctx.our_active_idx].long().tolist()
    req = ctx.our_active_req_move_ids.long().tolist()
    leg = ctx.our_active_req_move_legal.tolist()
    for b in range(ctx.batch_size):
        label, v, req_ids, req_dis, mask = metas[j0 + b]
        s, r, l = srt[b], req[b], leg[b]
        # A: obs request block == |request| JSON, request order, by num
        want = [num(x) for x in req_ids] + [0.0] * (4 - len(req_ids))
        if req_ids and not mask[10]:
            st["A_checked"] += 1
            okA = all((float(rv) == w) or (x.startswith("hiddenpower") and int(rv) in HP_NUMS)
                      for rv, w, x in zip(r, want, req_ids)) and all(rv == 0 for rv in r[len(req_ids):])
            if not okA:
                st["A_MISMATCH"] += 1
                examples["A"].append((label, v, req_ids, r))
            if l != [0.0 if d else 1.0 for d in req_dis] + [0.0] * (4 - len(req_dis)):
                st["A_LEGAL_MISMATCH"] += 1
                examples["Al"].append((label, v, req_ids, req_dis, l))
        if not any(x > 0 for x in r):
            st["D_no_request_moves"] += 1
            continue
        st["D_rows"] += 1
        if s == r:
            st["D_order_equal"] += 1
        if any(x > 0 and x not in s for x in r):
            st["D_REQ_ID_NOT_IN_SLOTS"] += 1
            examples["Dreq"].append((label, v, s, r, req_ids))
        if sorted(x for x in s if x > 0) != sorted(x for x in r if x > 0):
            st["D_set_differs"] += 1
            examples["Dset"].append((label, v, s, r, req_ids))
        if any(x == 0 for x, rid in zip(l, r) if rid > 0):
            st["D_rows_with_illegal_move"] += 1
            n_ill = sum(1 for x, rid in zip(l, r) if rid > 0 and x == 0)
            st[f"D_rows_with_{n_ill}_illegal"] += 1
        ident = [l[r.index(x)] if (x > 0 and x in r) else 0.0 for x in s]
        if ident != l:
            st["D_LEGALITY_MISAPPLIED"] += 1
            # the worst case: a CHOOSABLE move's token says illegal (or vice versa)
            if any(a == 1.0 and b == 0.0 for a, b in zip(ident, l)):
                st["D_choosable_move_marked_illegal"] += 1
            if len(examples["D"]) < 8:
                examples["D"].append((label, v, s, r, l, ident))
print({k: st[k] for k in sorted(st)})
for k, ex in examples.items():
    print("EXAMPLES", k, len(ex))
    for e in ex[:6]:
        print("   ", e)
