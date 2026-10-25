"""GIGO audit of the training data path (2026-10-26): an INDEPENDENT re-derivation of every buffer row.

Runs the REAL Rust collector (self-check build, CPU, T2 eager, a perturbed fresh production policy, p2 a
seeded random policy on an external route — `rust_rollout.testkit`), records every trainee decision AT PLAY
TIME through a collector hook (keyed by (env, episode, dec_n), never by arena slot, which is reused), every
ended game through the callbacks' infos, and after each fill re-derives each buffer cell from the record:

  A  obs / mask / every label column / action / behaviour log-prob / play-time value  == the record
  B  episode_starts == (dec_n == 0); win_target == 1{the game's reward > 0}; win_mask == 1
  C  advantages / returns == an independent float64 GAE(gamma, lambda) over the WHOLE game (rows from
     several updates included), the terminal reward on its last row, no bootstrap past it
  D  column adjacency: inside a buffer column, consecutive cells are the same game's dec_n, dec_n + 1, or a
     game's LAST row followed by another game's dec_n 0 (what the opponent-intent shift relies on)
  E  the learner's own forward on the CURRENT-version rows reproduces the stored log-prob AND the value
  F  every played row is consumed exactly once (no duplicate key across fills)
  G  (turn_limit run) a game cut at the turn limit is a terminal not-win (reward 0) and is reported as the
     stall forfeit, not as a tie

Usage: python harness.py [--turn-limit N] [--updates K] [--front ffi|proc]. Exit 0 = every check passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def gae64(rew: np.ndarray, val: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(rew)
    adv = np.zeros(n)
    last = 0.0
    for t in range(n - 1, -1, -1):
        nv = 0.0 if t == n - 1 else float(val[t + 1])
        nnt = 0.0 if t == n - 1 else 1.0
        delta = float(rew[t]) + gamma * nv * nnt - float(val[t])
        last = delta + gamma * lam * nnt * last
        adv[t] = last
    return adv, adv + val.astype(np.float64)


class Recorder:
    def __init__(self, label_keys):
        self.rows: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.label_keys = label_keys
        self.dupes = 0

    def episode_started(self, col, moved):
        pass

    def p2(self, col):
        pass

    def trainee(self, col, envs, logp, act, blogp, value, u, slots):
        c = col.cols
        for j, e in enumerate(envs.tolist()):
            key = (int(e), int(c["episode"][e]), int(c["dec_n"][e, 0]))
            if key in self.rows:
                self.dupes += 1
            self.rows[key] = dict(obs=c["obs"][e, 0].copy(), mask=c["mask"][e, 0].copy(),
                                  labels={k: np.asarray(c[k][e, 0]).copy() for k in self.label_keys},
                                  act=int(act[j]), blogp=np.float32(blogp[j]), value=np.float32(value[j]),
                                  turn=int(c["turn"][e]), version=int(col.slot_version[int(col.env_slot[e])]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-limit", type=int, default=250)
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--front", default="ffi")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--learn", action="store_true", help="run real train() updates (not a weight perturbation)")
    a = ap.parse_args()

    import torch as th

    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl
    from agents.training.rust_vec_env import RustVecEnv

    TK.build_selfcheck()
    _args, obs_space, act_space = TK.production_spaces()
    N, NS = a.n_envs, a.n_steps
    decl = RustEnvDecl(n_envs=N, threads=2, front=a.front, profile="selfcheck", n_steps=NS, micro_batch=NS,
                       device="cpu", backend="eager", run_seed=11, gamma=1.0, gae_lambda=0.8,
                       turn_limit=int(a.turn_limit))
    env = RustVecEnv(n_envs=N, observation_space=obs_space, action_space=act_space,
                     build=lambda m: TK.collector_for(m, obs_space, decl=decl))
    model = TK.fresh_model(env, n_steps=NS, batch_size=NS)
    col = env.startup(model)
    rec = Recorder([k for k in col.cfg.label_keys if k in obs_space.spaces])
    col.hooks.append(rec)
    cb = TK.NullCallback()
    fails: Dict[str, List[str]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    consumed = set()
    gamma, lam = float(col.cfg.gamma), float(col.cfg.gae_lambda)

    def fail(check: str, msg: str) -> None:
        if len(fails[check]) < 8:
            fails[check].append(msg)
        counts[f"FAIL_{check}"] += 1

    try:
        for k in range(a.updates):
            if k:
                TK.perturb_weights(model, k)
                col.after_update(model)
            assert col.collect(model, cb, model.rollout_buffer)
            print(f"[harness] update {k}: {len(cb.infos)} games ended, {len(rec.rows)} rows played", file=sys.stderr,
                  flush=True)
            buf = model.rollout_buffer
            prov = model._rust_row_provenance
            cur = int(model._rust_version)
            games = {(int(i["rust_env"]["env"]), int(i["rust_env"]["episode"])): i for i in cb.infos}
            ns, ne = buf.buffer_size, buf.n_envs
            for j in range(ne):
                for i in range(ns):
                    key = (int(prov["env"][i, j]), int(prov["episode"][i, j]), int(prov["dec_n"][i, j]))
                    counts["cells"] += 1
                    if key in consumed:
                        fail("F_dup", f"{key} consumed twice")
                    consumed.add(key)
                    r = rec.rows.get(key)
                    if r is None:
                        fail("A_missing", f"{key} not in the play record")
                        continue
                    if not np.array_equal(buf.observations["observation"][i, j], r["obs"]):
                        fail("A_obs", f"{key} obs differs")
                    if not np.array_equal(buf.observations["action_mask"][i, j], r["mask"]):
                        fail("A_mask", f"{key} mask differs")
                    for lk, lv in r["labels"].items():
                        if not np.array_equal(np.asarray(buf.observations[lk][i, j]).reshape(lv.shape), lv):
                            fail("A_label", f"{key} label {lk} differs")
                    if int(buf.actions[i, j, 0]) != r["act"]:
                        fail("A_act", f"{key} action {int(buf.actions[i, j, 0])} != {r['act']}")
                    if buf.log_probs[i, j] != r["blogp"]:
                        fail("A_logp", f"{key} logp {buf.log_probs[i, j]} != {r['blogp']}")
                    if buf.values[i, j] != r["value"]:
                        fail("A_value", f"{key} value {buf.values[i, j]} != {r['value']}")
                    if float(buf.episode_starts[i, j]) != float(key[2] == 0):
                        fail("B_start", f"{key} start {buf.episode_starts[i, j]}")
                    g = games.get(key[:2])
                    if g is None:
                        fail("B_game", f"{key} game not reported ended")
                        continue
                    won = 1.0 if float(g["episode"]["r"]) > 0 else 0.0
                    if float(buf.observations["win_target"][i, j, 0]) != won:
                        fail("B_win_target", f"{key} win_target {buf.observations['win_target'][i, j, 0]} != {won}")
                    if float(buf.observations["win_mask"][i, j, 0]) != 1.0:
                        fail("B_win_mask", f"{key}")
                    L = int(g["episode"]["l"])
                    try:
                        vals = np.array([rec.rows[(key[0], key[1], d)]["value"] for d in range(L)], dtype=np.float32)
                    except KeyError as exc:
                        fail("C_rows", f"{key} game rows not contiguous 0..{L - 1}: {exc}")
                        continue
                    rew = np.zeros(L, dtype=np.float32)
                    rew[-1] = np.float32(g["episode"]["r"])
                    if float(buf.rewards[i, j]) != (float(rew[-1]) if key[2] == L - 1 else 0.0):
                        fail("C_reward", f"{key} reward {buf.rewards[i, j]} (L {L}, r {rew[-1]})")
                    adv, ret = gae64(rew, vals, gamma, lam)
                    if abs(float(buf.advantages[i, j]) - adv[key[2]]) > 1e-5:
                        fail("C_adv", f"{key} adv {buf.advantages[i, j]} != {adv[key[2]]}")
                    if abs(float(buf.returns[i, j]) - ret[key[2]]) > 1e-5:
                        fail("C_ret", f"{key} ret {buf.returns[i, j]} != {ret[key[2]]}")
                    counts["checked"] += 1
                # D: adjacency inside the column
                for i in range(ns - 1):
                    k0 = (int(prov["env"][i, j]), int(prov["episode"][i, j]), int(prov["dec_n"][i, j]))
                    k1 = (int(prov["env"][i + 1, j]), int(prov["episode"][i + 1, j]), int(prov["dec_n"][i + 1, j]))
                    if k0[:2] == k1[:2]:
                        if k1[2] != k0[2] + 1:
                            fail("D_adjacent", f"col {j}: {k0} -> {k1}")
                    else:
                        g0 = games.get(k0[:2])
                        if g0 is None or k0[2] != int(g0["episode"]["l"]) - 1 or k1[2] != 0:
                            fail("D_adjacent", f"col {j}: game change {k0} -> {k1} not last->first")
                    counts["adjacent_pairs"] += 1
            # E: the learner's forward on the current-version rows
            vers = np.asarray(model._rust_row_versions)
            sel = np.argwhere(vers == cur)
            if sel.size:
                obs_t = {key_: th.as_tensor(arr[sel[:, 0], sel[:, 1]]) for key_, arr in buf.observations.items()}
                acts = th.as_tensor(buf.actions[sel[:, 0], sel[:, 1], 0]).long()
                masks = th.as_tensor(buf.action_masks[sel[:, 0], sel[:, 1]])
                model.policy.set_training_mode(False)
                with th.no_grad():
                    v, lp, _ent = model.policy.evaluate_actions(obs_t, acts, action_masks=masks)
                dl = np.abs(lp.numpy().reshape(-1) - buf.log_probs[sel[:, 0], sel[:, 1]]).max()
                dv = np.abs(v.numpy().reshape(-1) - buf.values[sel[:, 0], sel[:, 1]]).max()
                counts["E_rows"] += int(sel.shape[0])
                counts.setdefault("E_max_dlogp_e9", 0)
                counts["E_max_dlogp_e9"] = max(counts["E_max_dlogp_e9"], int(dl * 1e9))
                counts["E_max_dvalue_e9"] = max(counts.get("E_max_dvalue_e9", 0), int(dv * 1e9))
                if dl > 1e-4:
                    fail("E_logp", f"update {k}: max |d logp| {dl}")
                if dv > 1e-4:
                    fail("E_value", f"update {k}: max |d value| {dv} (stored {buf.values[sel[0, 0], sel[0, 1]]} "
                                    f"learner {float(v.reshape(-1)[0])})")
        # G: the turn-limit games
        for i in cb.infos:
            g = i["rust_env"]
            key = (int(g["env"]), int(g["episode"]))
            L = int(i["episode"]["l"])
            last = rec.rows.get((key[0], key[1], L - 1))
            if g["truncated"]:
                counts["G_truncated"] += 1
                if float(i["episode"]["r"]) != 0.0:
                    fail("G_reward", f"{key} truncated with reward {i['episode']['r']}")
                if a.turn_limit < 250 and last is not None and last["turn"] >= a.turn_limit - 1 and not g["forfeit"]:
                    fail("G_forfeit", f"{key} truncated at turn {last['turn']} (limit {a.turn_limit}) reported "
                                      f"forfeit={g['forfeit']} draw={i['win_draw']}")
            counts["games"] += 1
            counts["wins"] += int(float(i["episode"]["r"]) > 0)
        counts["dupes_at_play"] = rec.dupes
        counts["played"] = len(rec.rows)
    finally:
        env.close()
    print(json.dumps({"counts": counts, "fails": fails}, indent=1, sort_keys=True))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
