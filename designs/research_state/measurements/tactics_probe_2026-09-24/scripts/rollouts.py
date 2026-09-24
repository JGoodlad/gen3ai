"""The ORACLE: true-world win probability of every legal action at every state, by rollouts to a terminal.

A rollout = the scripted prefix on the state's fixed seed -> at T our side plays action a (its sim token) ->
the sim PRNG is RESEEDED at the start of T (post_t_seed, so the dice after the decision vary by rollout
and not by action) -> both sides play on to the end as the FIXED CONTINUATION POLICY: G0
(ai_v13_12_plateau), stochastic at T = 1, each player with its own seeded sampler.
COMMON RANDOM NUMBERS: rollout r of a state-regime uses the same post_t_seed and the same two policy seeds
for EVERY action, so action differences are paired.

Regimes: LIVE = the opponent's turn-T move is chosen by the continuation policy (the true-world value vs
that opponent); PREMISE = the opponent's turn-T move is scripted (DG: Dragon Dance; DT: Explosion; B:
Soft-Boiled) -- the owner's scenario taken literally.

Each rollout row also holds, for G0 / B / C, the critic V and win-prob at OUR first decision after the
action (the "critic after the action"), and the turn-T protocol lines (who moved, who fainted).

UNITS: (sid, regime, block) = BLOCK rollouts x every legal action; minutes long. Rows are appended durably
per worker (rows/w<k>.jsonl) as each rollout finishes; a restart skips every (sid, regime, r, action)
already on disk. Static sharding: unit u goes to worker hash(u) % n_workers.

    python rollouts.py --manifest <states.json> --rows <archive>/rows --worker k --n-workers 3 [--max-units N]
"""
import argparse
import asyncio
import glob
import hashlib
import json
import os
import signal
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlib  # noqa: E402

# REGISTERED n (PREDICTION.md): rollouts per (state, regime, action)
R_CORE = 48          # DG_GL_*, DT_*, B_*
R_SECONDARY = 24     # DG_SL_*, L*, R* (amendment 1)
BLOCK = 4
LINE_TIMEOUT_S = 900.0
KEEP = ("|move|", "|-damage|", "|faint|", "|-boost|", "|cant|", "|switch|", "|-fail|", "|-miss|", "|-crit|",
        "|-immune|", "|-heal|", "|-status|", "|-activate|")


def regimes_for(sid, fam):
    if fam in ("L", "R") or sid.startswith("DG_SL_"):
        return ["live"]
    return ["live", "premise"]


def r_for(sid, fam):
    return R_SECONDARY if (fam in ("L", "R") or sid.startswith("DG_SL_")) else R_CORE


def units(man):
    out = []
    for sid, st in sorted(man["states"].items()):
        if st.get("refused") or not st.get("tokens"):
            continue
        R = r_for(sid, st["family"])
        for reg in regimes_for(sid, st["family"]):
            for b in range(R // BLOCK):
                out.append((sid, reg, b))
    # interleave: every state-regime's block 0 first, then block 1, ... (progress covers all states early)
    out.sort(key=lambda u: (u[2], u[0], u[1]))
    return out


def owner(u, n):
    return int(hashlib.sha256(f"{u[0]}|{u[1]}|{u[2]}".encode()).hexdigest(), 16) % n


def done_keys(rows_dir):
    keys = set()
    for p in glob.glob(os.path.join(rows_dir, "w*.jsonl")):
        with open(p) as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue            # a torn last line from a kill: that rollout is simply redone
                keys.add((r["sid"], r["regime"], r["r"], r["action"]))
    return keys


def make_capture_cls():
    """RLPlayer that stores the FIRST obs our side builds after it is armed (at the turn-T substitute)."""
    from agents.inference.player import RLPlayer

    class _Cap(RLPlayer):
        armed = False
        post = None

        def embed_battle(self, battle):
            out = super().embed_battle(battle)
            if self.armed and self.post is None and int(np.asarray(out["action_mask"]).sum()) > 0:
                fs = getattr(battle, "force_switch", False)
                fs = any(fs) if isinstance(fs, (list, tuple)) else bool(fs)
                self.post = {"turn": int(battle.turn), "force_switch": fs,
                             "obs": np.asarray(out["observation"], dtype=np.float32).copy(),
                             "mask": np.asarray(out["action_mask"], dtype=np.float32).copy()}
            return out
    return _Cap


def turn_facts(lines, T, our_side):
    opp = tlib.OTHER[our_side]
    blk = [ln for ln in tlib.turn_block(lines, T) if ln.startswith(KEEP)]
    f = {"lines": blk,
         "our_moved": [ln.split("|")[3] for ln in blk if ln.startswith(f"|move|{our_side}a:")],
         "opp_moved": [ln.split("|")[3] for ln in blk if ln.startswith(f"|move|{opp}a:")],
         "our_fainted": any(ln.startswith(f"|faint|{our_side}a:") for ln in blk),
         "opp_fainted": any(ln.startswith(f"|faint|{opp}a:") for ln in blk),
         "cant": [ln for ln in blk if ln.startswith("|cant|")]}
    return f


def value_only(model, obs, mask):
    import torch
    with torch.no_grad():
        pin = {"observation": torch.as_tensor(obs[None]).float(),
               "action_mask": torch.as_tensor(mask[None]).float()}
        v = float(model.policy.predict_values(pin)[0].item())
        wl = model.policy.features_extractor.last_win_prob_logits
        wp = float(torch.sigmoid(wl[0, 0]).item()) if wl is not None else None
    return v, wp


def run_unit(u, man, models, cap_cls, fh, have, stop):
    sid, reg, b = u
    st = man["states"][sid]
    our = st["our_side"]
    cmds = [tuple(c) for c in st["commands"]]
    if reg == "premise":
        cmds = cmds + [(tlib.OTHER[our], st["premise"])]
    rec = tlib.make_record(st["p1_team"], st["p2_team"], st["seed"], cmds, our)
    T = int(st["T"])
    toks = {int(k): v for k, v in st["tokens"].items()}
    n_new = 0
    for r in range(b * BLOCK, (b + 1) * BLOCK):
        post_seed = tlib.mint_seed(f"oracle:{sid}:{reg}:r{r}")
        ps = int(hashlib.sha256(f"pol:{sid}:{reg}:r{r}".encode()).hexdigest()[:8], 16)
        for a, tok in sorted(toks.items()):
            if (sid, reg, r, a) in have:
                continue
            if stop["flag"]:
                return n_new
            t0 = time.time()
            trainee = tlib.make_rl(models["G0"], prefix="ORt", policy_seed=ps, cls=cap_cls)
            opp = tlib.make_rl(models["G0"], prefix="ORo", policy_seed=ps + 1)

            def hook(battle, obs_dict, choice, _tr=trainee):
                if int(battle.turn) == T:
                    _tr.armed = True

            row = {"sid": sid, "regime": reg, "r": r, "action": a, "token": tok, "post_seed": post_seed,
                   "policy_seed": ps}
            try:
                res = tlib.run_line(rec, trainee=trainee, opponent=opp, T=T, substitute=tok,
                                    post_t_seed=post_seed, hook=hook, timeout=LINE_TIMEOUT_S)
                lines = tlib.side_lines(res.pop("sink"), our)
                row.update(status="ok", outcome=res["outcome"], capped=res["capped"], turns=res["turns"],
                           score=tlib.rollout_score(res), substitute=res["substitute"],
                           our_exhausted=res["our_exhausted"], opp_exhausted=res["opp_exhausted"],
                           tfacts=turn_facts(lines, T, our))
                post = trainee.post
                if post is not None:
                    row["post"] = {"turn": post["turn"], "force_switch": post["force_switch"],
                                   "v": {t: value_only(m, post["obs"], post["mask"]) for t, m in models.items()}}
                else:
                    row["post"] = None
            except asyncio.TimeoutError:
                row.update(status="timeout", err=f"line wall > {LINE_TIMEOUT_S}s")
            except Exception as e:  # noqa: BLE001 -- recorded as an ERROR row, never as an outcome
                row.update(status="error", err=f"{type(e).__name__}: {e}"[:500])
                traceback.print_exc()
            row["wall_s"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            have.add((sid, reg, r, a))
            n_new += 1
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of sids (pilot)")
    a = ap.parse_args()
    os.makedirs(a.rows, exist_ok=True)
    man = json.load(open(a.manifest))
    stop = {"flag": False}

    def _term(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _term)

    models = {t: tlib.load_model(t) for t in tlib.MODELS}
    cap_cls = make_capture_cls()
    have = done_keys(a.rows)
    only = set(x for x in a.only.split(",") if x)
    todo = [u for u in units(man) if owner(u, a.n_workers) == a.worker and (not only or u[0] in only)]
    print(f"[w{a.worker}] {len(todo)} units assigned; {len(have)} rollouts already on disk", flush=True)
    path = os.path.join(a.rows, f"w{a.worker}.jsonl")
    n_units = 0
    with open(path, "a") as fh:
        for u in todo:
            st = man["states"][u[0]]
            need = [(u[0], u[1], r, int(k)) for r in range(u[2] * BLOCK, (u[2] + 1) * BLOCK) for k in st["tokens"]]
            if all(k in have for k in need):
                continue
            t0 = time.time()
            n = run_unit(u, man, models, cap_cls, fh, have, stop)
            n_units += 1
            print(f"[w{a.worker}] unit {u} +{n} rollouts in {time.time() - t0:.0f}s", flush=True)
            if stop["flag"] or (a.max_units and n_units >= a.max_units):
                break
    print(f"[w{a.worker}] exit: {n_units} units this session; stop={stop['flag']}", flush=True)


if __name__ == "__main__":
    main()
