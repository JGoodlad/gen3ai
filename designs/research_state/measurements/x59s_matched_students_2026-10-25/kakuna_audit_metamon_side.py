#!/usr/bin/env python3
"""The Kakuna-family audit's METAMON side (X59-S REVIEW E B2; PREREG Amendment 2): Kakuna's log-likelihood of a
seat's actions, over the decisions Metamon's OWN replay parser reconstructs from a spectator log.

THIS FILE IS NEVER IMPORTED BY THIS REPO'S CODE. It runs under the Metamon interpreter
(``designs/ops/anchors.json`` ``opponents.metamon.python``) with ``PYTHONPATH=""``, launched by
``python -m main.human_data.audit kakuna`` (which sets the environment exactly as ``main.anchors.peers.MetamonPeer``
does). It loads ``src/main/anchors/peer_scripts/metamon_side.py`` by path for ``install_cpu_attention`` (the
windowed CPU attention every Kakuna anchor read uses) and imports nothing else of ours.

    <metamon python> kakuna_audit_metamon_side.py --in <export.jsonl> --out <scores.jsonl> [--deadline <epoch s>]

Input: one JSON object per battle, ``{"key", "gameid", "date": "YYYY-MM-DD", "text": <spectator log>,
"seats": ["p1", ...]}``. Output: one JSON object per (battle, seat), appended and flushed per battle, so a re-run
RESUMES (a (key, seat) already in ``--out`` is skipped). Exit 0 = every battle done, 3 = the deadline stopped it.

Per battle, exactly Metamon's offline-data path (``ReplayParser.parse_replay`` → ``povreplay_to_seq`` →
``ParsedReplayDataset.load_filename`` → ``MetamonAMAGODataset._process_data`` → ``RLData``):

1. ``forward.forward_fill`` then ``backward.backward_fill`` with the parser's default team predictor
   (``NaiveUsagePredictor``, which SAMPLES the unrevealed sets from usage stats: ``random`` and ``numpy.random``
   are seeded per battle from ``key``, and the process runs with ``PYTHONHASHSEED=0``, so a battle's
   reconstruction is a pure function of its log). Any exception here or in ``povreplay_to_seq`` leaves the seat
   UNSCORED with the exception's type recorded.
2. The states are round-tripped through ``to_dict`` / ``from_dict`` (the dataset's JSON), observed with Kakuna's
   ``TokenizedObservationSpace`` (reset first), rewarded with its ``AggressiveShapedReward``; the legal set is
   ``UniversalAction.maybe_valid_actions``; ``rl2s`` = (previous reward, previous action one-hot), time = 0..T-1.
3. ONE teacher-forced forward of Kakuna (checkpoint 34 at the anchors' hub revision, sha256 pinned below, the hub
   OFFLINE; ``policy.eval()``, no grad, fp32 CPU,
   ``WindowedVanillaAttention`` window (96, 0) as installed by ``install_cpu_attention``) over the first
   ``--max-steps`` (default 128 = ``max_seq_len``) states. The distribution read is the actor's own output for the
   test-time gamma (index -1): ``MetamonDiscrete``'s clipped distribution, the one ``get_actions`` samples from or
   argmaxes (PRIMARY, ``ll``), and the pre-clip masked softmax (``ll_raw``).
4. A decision is SCORED when it has an action (not the terminal state, not ``missing``) and >= 2 legal actions
   (the rowset's own filter). Decisions past ``--max-steps`` are counted as ``n_cut``.

``--check-cache N``: for the first N battles also run the live inference path step by step (the KV cache,
``get_state_embedding`` with a hidden state, as ``Experiment.interact`` does) and record the largest absolute
probability difference from the one-pass forward (``cache_max_abs_diff``).
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":            # set-iteration order must not vary between runs
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# the weights (and the team predictor's usage stats) come from the local cache or not at all (anchors' FR-1 FINDING 3)
os.environ["HF_HUB_OFFLINE"] = "1"
THREADS = int(os.environ.get("KAUDIT_THREADS", "2"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

AGENT = "Kakuna"
CHECKPOINT = 34
#: the anchors' pin (designs/research_state/measurements/kakuna_wiring_2026-09-28/README.md; X51's record)
CKPT_SHA256 = "7e7474d5b2ec6624afdeb29a7cf1dc5c84817d5aee8ec39c1d35a153b18df26c"
#: the hub revision the anchors pin (origin/main ``designs/ops/anchors.json`` ``opponents.metamon.hf_revision``)
HF_REVISION = "ac8e88e7da8dc2a2b85176d88f81fd8854b1445b"
MAX_STEPS = 128
SCHEMA = "gen3_kakuna_audit_metamon_v1"


def _repo_root():
    # designs/research_state/measurements/<dir>/<this file>: this runs outside our env (no utils.paths)
    return Path(__file__).resolve().parents[4]


def load_metamon_side():
    p = _repo_root() / "src" / "main" / "anchors" / "peer_scripts" / "metamon_side.py"
    spec = importlib.util.spec_from_file_location("gen3ai_metamon_side", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, str(p)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def battle_seed(key):
    return int(hashlib.sha256(f"kakuna_audit|{key}".encode()).hexdigest()[:8], 16)


class Scorer:
    def __init__(self, max_steps):
        import torch

        torch.set_num_threads(THREADS)
        ms, ms_path = load_metamon_side()
        self.attn = {}
        ms.install_cpu_attention("vanilla", self.attn)
        import metamon.rl.metamon_to_amago as m2a
        from metamon.backend.replay_parser.parse_replays import ReplayParser
        from metamon.rl.pretrained import get_pretrained_model

        # the pre-clip distribution: MetamonDiscrete.forward computes it and returns the clipped one
        inner = m2a.MetamonDiscrete.forward
        stash = self.raw = {}

        def forward(self_, vec, log_dict=None):
            import amago

            stash["probs"] = amago.nets.policy_dists._Categorical(logits=vec / self_.temperature).probs.detach()
            return inner(self_, vec, log_dict=log_dict)

        m2a.MetamonDiscrete.forward = forward
        self.model = get_pretrained_model(AGENT)
        ckpt = self.model.get_path_to_checkpoint(CHECKPOINT)
        got = sha256_file(ckpt)
        if f"/snapshots/{HF_REVISION}/" not in ckpt:
            raise SystemExit(f"{ckpt} is not under the pinned hub revision {HF_REVISION}")
        if got != CKPT_SHA256:
            raise SystemExit(f"{AGENT} checkpoint {CHECKPOINT} sha256 {got} != the pin {CKPT_SHA256}")
        exp = self.model.initialize_agent(checkpoint=CHECKPOINT, log=False)
        self.policy = exp.policy
        self.policy.eval()
        self.parser = ReplayParser()
        self.max_steps = max_steps
        self.n_actions = self.model.action_space.gym_space.n
        self.ident = {"agent": AGENT, "checkpoint": CHECKPOINT, "ckpt_sha256": got, "ckpt_path": ckpt,
                      "hf_revision": HF_REVISION, "hf_hub_offline": True, "script_sha256": sha256_file(__file__),
                      "metamon_side": ms_path,
                      "metamon_side_sha256": sha256_file(ms_path), "threads": THREADS, "max_steps": max_steps,
                      **self.attn}

    # ------------------------------------------------------------------ the parser (Metamon's own)
    def parse(self, rec):
        import numpy as np
        from metamon.backend.replay_parser import backward, forward

        d = datetime.date.fromisoformat(rec["date"])
        replay = forward.ParsedReplay(gameid=rec["gameid"], format="gen3ou",
                                      time_played=datetime.datetime(d.year, d.month, d.day))
        log = self.parser.clean_log({"log": rec["text"]})
        s = battle_seed(rec["key"])
        random.seed(s)
        np.random.seed(s)
        replay = forward.forward_fill(replay, log, verbose=False)
        p1, p2 = backward.backward_fill(replay, team_predictor=self.parser.team_predictor)
        return {"p1": p1, "p2": p2}

    def seq(self, pov):
        """``povreplay_to_seq`` (kept in pieces for the turn record), then the dataset's JSON round trip."""
        from metamon.backend.replay_parser import checks
        from metamon.interface import UniversalState

        states, actions = self.parser.povreplay_to_state_action(pov)
        us, idxs = self.parser.state_action_to_obs_action_reward(states, actions)
        checks.check_action_idxs(us, actions, idxs, gen=pov.gen)
        turns = [[int(t.turn_number) if t.turn_number is not None else -1, bool(t.is_force_switch)]
                 for t in pov.povturnlist]
        us = [UniversalState.from_dict(json.loads(json.dumps(u.to_dict()))) for u in us]
        return us, list(idxs), turns

    # ------------------------------------------------------------------ the dataset path + one forward
    def tensors(self, states, idxs):
        import numpy as np
        import torch
        from metamon.interface import UniversalAction

        osp, asp = self.model.observation_space, self.model.action_space
        osp.reset()
        obs = [osp.state_to_obs(s) for s in states]
        chosen, legal, missing = [], [], []
        for s, a in zip(states, idxs[:-1]):
            ua = UniversalAction(action_idx=a)
            chosen.append(asp.action_to_agent_output(s, ua))
            legal.append({asp.action_to_agent_output(s, x) for x in UniversalAction.maybe_valid_actions(s)})
            missing.append(ua.missing)
        rews = np.array([self.model.reward_function(a, b) for a, b in zip(states[:-1], states[1:])], np.float32)
        T = len(states)
        L = min(T, self.max_steps)
        A = self.n_actions
        ill = torch.ones((T, A), dtype=torch.bool)
        for i, lg in enumerate(legal):
            for x in lg:
                ill[i, x] = False
        acts = torch.nn.functional.one_hot(torch.tensor(chosen).long().clamp(min=0), num_classes=A).float()
        rl2s = torch.cat((torch.cat((torch.zeros(1, 1), torch.from_numpy(rews).unsqueeze(-1)), 0),
                          torch.cat((torch.zeros(1, A), acts), 0)), -1)
        o = {k: torch.from_numpy(np.stack([x[k] for x in obs], 0))[:L].unsqueeze(0) for k in obs[0]}
        o["illegal_actions"] = ill[:L].unsqueeze(0)
        time_idxs = torch.arange(T).long().unsqueeze(-1)[:L].unsqueeze(0)
        return o, rl2s[:L].unsqueeze(0), time_idxs, chosen, legal, missing, T, L

    def probs(self, o, rl2s, time_idxs):
        import torch

        keys = self.policy.pass_obs_keys_to_actor
        with torch.no_grad():
            emb, _ = self.policy.get_state_embedding(obs=o, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None)
            dist = self.policy.actor(emb, straight_from_obs={k: o[k] for k in keys})
        return dist.probs[0, :, -1, :].double(), self.raw["probs"][0, :, -1, :].double()

    def probs_cached(self, o, rl2s, time_idxs):
        """The live inference path: one step at a time through the KV cache."""
        import torch

        keys = self.policy.pass_obs_keys_to_actor
        hs = self.policy.traj_encoder.init_hidden_state(1, torch.device("cpu"))
        out = []
        with torch.no_grad():
            for t in range(o["illegal_actions"].shape[1]):
                ot = {k: v[:, t:t + 1] for k, v in o.items()}
                emb, hs = self.policy.get_state_embedding(obs=ot, rl2s=rl2s[:, t:t + 1],
                                                          time_idxs=time_idxs[:, t:t + 1], hidden_state=hs)
                dist = self.policy.actor(emb, straight_from_obs={k: ot[k] for k in keys})
                out.append(dist.probs[0, 0, -1, :].double())
        return torch.stack(out)

    def score_seat(self, pov, check_cache):
        import torch

        states, idxs, turns = self.seq(pov)
        o, rl2s, tix, chosen, legal, missing, T, L = self.tensors(states, idxs)
        p, praw = self.probs(o, rl2s, tix)
        ll = ll_raw = 0.0
        n = hits = n_cut = 0
        kinds = []
        for i in range(T - 1):
            ok = (not missing[i]) and len(legal[i]) >= 2
            kinds.append([turns[i][0], int(turns[i][1]), "x" if missing[i] else ("s" if 4 <= idxs[i] <= 8 else "m"),
                          int(ok)])
            if not ok:
                continue
            if i >= L:
                n_cut += 1
                continue
            a = chosen[i]
            ll += float(torch.log(p[i, a]))
            ll_raw += float(torch.log(praw[i, a]))
            hits += int(int(torch.argmax(p[i])) == a)
            n += 1
        out = {"n_states": T, "n_dec": n, "n_cut": n_cut, "ll_sum": ll, "ll_raw_sum": ll_raw, "argmax_hits": hits,
               "n_missing": int(sum(missing)), "decisions": kinds}
        if check_cache:
            pc = self.probs_cached(o, rl2s, tix)
            out["cache_max_abs_diff"] = float((pc - p[:L]).abs().max())
        return out


def done_keys(path):
    out = set()
    if os.path.exists(path):
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    r = json.loads(ln)
                    out.add((r["key"], r["seat"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: start no battle after it")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--check-cache", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ident-out", default=None, help="write the scorer's identity record here")
    a = ap.parse_args(argv)
    done = done_keys(a.out)
    todo = []
    with open(a.inp) as fh:
        for ln in fh:
            r = json.loads(ln)
            seats = [s for s in r["seats"] if (r["key"], s) not in done]
            if seats:
                todo.append({**r, "seats": seats})
    if a.limit is not None:
        todo = todo[:a.limit]
    print(f"[kaudit] {len(todo)} battles to score ({len(done)} seats already done)", flush=True)
    if not todo:
        return 0
    t0 = time.time()
    sc = Scorer(a.max_steps)
    print(f"[kaudit] scorer built in {time.time() - t0:.0f} s: {sc.ident}", flush=True)
    if a.ident_out:
        with open(a.ident_out, "w") as fh:
            json.dump({"schema": SCHEMA, **sc.ident}, fh, indent=1)
    n_checked = 0
    with open(a.out, "a") as out:
        for bi, rec in enumerate(todo):
            if a.deadline is not None and time.time() >= a.deadline:
                print(f"[kaudit] deadline reached after {bi} battles", flush=True)
                return 3
            tb = time.time()
            rows = []
            try:
                povs = sc.parse(rec)
                perr = None
            except Exception as exc:                  # noqa: BLE001 - every parse failure is counted by type
                povs, perr = None, exc
            for seat in rec["seats"]:
                row = {"schema": SCHEMA, "key": rec["key"], "seat": seat}
                if povs is None:
                    row.update({"ok": False, "stage": "parse", "error": type(perr).__name__,
                                "error_msg": str(perr)[:300]})
                else:
                    try:
                        row.update({"ok": True, **sc.score_seat(povs[seat], n_checked < a.check_cache)})
                    except Exception as exc:          # noqa: BLE001
                        row.update({"ok": False, "stage": "seq", "error": type(exc).__name__,
                                    "error_msg": str(exc)[:300], "tb": traceback.format_exc()[-600:]})
                rows.append(row)
            n_checked += 1
            dt = time.time() - tb
            for row in rows:
                row["secs"] = round(dt / len(rows), 3)
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
            out.flush()
            if bi % 25 == 0:
                print(f"[kaudit] {bi + 1}/{len(todo)} battles, last {dt:.1f} s, {time.time() - t0:.0f} s total",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
