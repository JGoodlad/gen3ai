"""The probe battery's MODEL-SIDE worker — run BY PATH under the CHECKPOINT'S OWN checkout (`gen3_probe_battery_v1`).

A checkpoint is read at the commit it trained at (its obs layout, its encoder, its model code), so this file imports
NOTHING from ``main.probe_battery``: the HEAD-side CLI (``python -m main.probe_battery``) spawns it as
``python <this file> <cmd> ...`` with ``PYTHONPATH=<checkout>/src`` and the checkout as the working directory, and it
reaches only the checkout's long-standing APIs (``main.h2h.play``, ``agents.battle.*core*``'s ``run_core``,
``main.policy_spectrum.reader``). Everything here is CPU only and never touches the GPU.

Commands (each writes ONE output and prints one JSON summary line on stdout):

* ``play``    — mirrored-pair games between checkpoints on the checkout's Rust eval core (``H2HEngine.play_batch``
  with a diagnostic sink, so every game keeps its INPUT LOG ``script``); one JSON line per game.
* ``replay``  — each game's input log through the checkout's ``core_events --views --trackers --obs``: per viewer,
  per answered decision, the obs row + mask + legal tokens + the BOTH sides' ``present()`` views at that decision
  (the viewer's own, and the opponent's — the opponent's TRUE state) + the opponent's choice at the same board.
* ``capture`` — one checkpoint (or a fresh RANDOM-INIT build of its architecture) over a bank's rows: the policy, the
  win-prob, and float16 tokens at fixed seats and depths (trunk input, after every trunk layer) + the policy state
  (the actor latent the pointer head reads) + the critic pool (``vf_features``).
* ``forward`` — policy probabilities and win-prob over a rows file (the behavioural probes' edited rows).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SCHEMA = "gen3_probe_battery_v1"


def _emit(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, sort_keys=True), flush=True)


def _checkout_head(root: Path) -> str:
    """The checkout's HEAD commit, read from its git admin files (no git process)."""
    g = root / ".git"
    gitdir = Path(g.read_text().split("gitdir:", 1)[1].strip()) if g.is_file() else g
    head = (gitdir / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        common = gitdir
        cd = gitdir / "commondir"
        if cd.exists():
            common = (gitdir / cd.read_text().strip()).resolve()
        for base in (gitdir, common):
            p = base / ref
            if p.exists():
                return p.read_text().strip()
        packed = common / "packed-refs"
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
        raise RuntimeError(f"cannot resolve {ref} in {gitdir}")
    return head


# ------------------------------------------------------------------------------------------------- play
def cmd_play(a: argparse.Namespace) -> int:
    from main.h2h.play import Compute, H2HEngine, resolve_player
    from main.h2h.arch import declare_engine
    import main.h2h.play as PL

    cells = json.loads(Path(a.cells).read_text())
    out = Path(a.out)
    done = set()
    if out.exists():
        with gzip.open(out, "rt") as f:
            for line in f:
                done.add(json.loads(line)["cell"])
    todo = [c for c in cells if c["cell"] not in done]
    if not todo:
        _emit({"cmd": "play", "games": 0, "cells_done": len(done), "note": "nothing to do"})
        return 0
    refs = {}
    for c in cells:
        for k in ("player", "opponent"):
            if c[k] not in refs:
                refs[c[k]] = resolve_player(c[k])
    pairs = [(refs[c["player"]], refs[c["opponent"]]) for c in todo]
    decl = declare_engine(pairs, PL._load_host, "off")
    compute = Compute(device="cpu", n_envs=int(a.n_envs), threads=int(a.threads), torch_threads=int(a.torch_threads))
    eng = H2HEngine(pairs[0][0], pairs[0][1], compute, decl=decl)
    n_games = 0
    t0 = time.time()
    try:
        for i, c in enumerate(todo):
            p, o = refs[c["player"]], refs[c["opponent"]]
            if i:
                eng.set_cell(p, o)
            sink: List[Dict[str, Any]] = []
            games, _st = eng.play_batch(int(c["pairs"]), seed=int(c["seed"]), sink=sink)
            lines = []
            for g in games:
                lines.append(json.dumps({
                    "schema": SCHEMA, "cell": c["cell"], "player": c["player"], "opponent": c["opponent"],
                    "player_label": c.get("player_label"), "opponent_label": c.get("opponent_label"),
                    "player_sha256": p.sha256, "opponent_sha256": o.sha256, "game": int(g["game"]),
                    "cycle_seed": int(c["seed"]), "seed": g.get("seed"), "swapped": bool(g.get("swapped")),
                    "teams": g.get("teams"), "winner": g.get("winner"), "end_turn": g.get("end_turn"),
                    "forfeit": g.get("forfeit"), "script": g["script"]}, sort_keys=True))
            # one cell's games land in ONE append, so a crash never leaves half a cell
            with gzip.open(out, "at") as f:
                f.write("\n".join(lines) + "\n")
            n_games += len(games)
            print(f"[probe_battery play] {c['cell']}: {len(games)} games ({time.time() - t0:.0f}s)",
                  file=sys.stderr, flush=True)
    finally:
        eng.close()
    _emit({"cmd": "play", "games": n_games, "cells": len(todo), "seconds": round(time.time() - t0, 1)})
    return 0


# ----------------------------------------------------------------------------------------------- replay
def _recorded(label: str, script: str):
    try:
        from agents.battle.core_replay import RecordedBattle
    except ImportError:                       # before the P6 move (2026-10-08)
        from agents.battle.rust_core_parity import RecordedBattle
    lines = script.split("\n")
    if not lines[0].startswith("START "):
        raise ValueError(f"{label}: a script starts with START")
    start = json.loads(lines[0][len("START "):])
    cmds: List[List[str]] = []
    for ln in lines[1:]:
        if ln.startswith("CHOOSE "):
            _, side, choice = ln.split(" ", 2)
            cmds.append([side, choice])
        elif ln.startswith("CHOOSEIF "):
            _, side, choice = ln.split(" ", 2)
            cmds.append([side, choice, "if_open"])
        elif ln.startswith("FORCELOSE "):
            cmds.append(["forcelose", ln.split(" ", 1)[1]])
        elif ln.strip() in ("END", ""):
            continue
        else:
            raise ValueError(f"{label}: unknown script line {ln[:60]!r}")
    return RecordedBattle(label=label, format_id=start["formatid"], seed=start["seed"], p1=start["p1"],
                          p2=start["p2"], commands=cmds, init_seed=bool(start.get("init_seed", False)),
                          quick_claw=bool(start.get("quick_claw", False)))


def _run_core(battles):
    try:
        from agents.battle.core_replay import run_core
    except ImportError:
        from agents.battle.rust_core_parity import run_core
    return run_core(battles, views=True, trackers=True, obs=True)


def _decode_row(obs: Any) -> np.ndarray:
    from agents.battle.core_obs import wrap_row

    return np.array(wrap_row(obs), dtype=np.float32, copy=True)


def _as_json(x: Any) -> Any:
    return json.loads(x) if isinstance(x, str) else x


def replay_game(game_id: str, script: str) -> Dict[str, Any]:
    """One game → its decisions (both viewers), each with the two views at its board."""
    res = _run_core([_recorded(game_id, script)])[0]
    if not res.get("ok"):
        raise RuntimeError(f"{game_id}: core_events refused: {res.get('error')}")
    views = res["views"]
    afters = [int(v["after"]) for v in views]
    per_view: Dict[int, Dict[int, Dict[str, Any]]] = {}
    decs: List[Dict[str, Any]] = []
    for s in (0, 1):
        n = 0
        for e in res["trackers"][s]:
            if "obs" not in e:
                continue
            choice = e.get("choice")
            if choice is None:            # an unanswered final request (the battle ended / forfeit)
                continue
            a = int(e["after"])
            k = next((i for i, va in enumerate(afters) if va > a), None)
            if k is None:
                raise RuntimeError(f"{game_id} p{s + 1} decision {n}: no view after chunk {a}")
            if not views[k]["new_request"][s]:
                raise RuntimeError(f"{game_id} p{s + 1} decision {n}: view {k} shipped no request to p{s + 1}")
            trk = e.get("trackers") or {}
            d = {"game": game_id, "side": s, "n": n, "after": a, "view": k,
                 "mask": [int(x) for x in e["mask"]], "tokens": {str(kk): vv for kk, vv in e["tokens"].items()},
                 "choice": choice, "our_slots": trk.get("our_slots"), "opp_slots": trk.get("opp_slots"),
                 "wish": trk.get("wish"), "row": _decode_row(e["obs"])}
            per_view.setdefault(k, {})[s] = d
            decs.append(d)
            n += 1
    for d in decs:
        k, s = d["view"], d["side"]
        v = views[k]
        d["V"] = _as_json(v["core"][s])
        d["W"] = _as_json(v["core"][1 - s])
        d["legal"] = _as_json(v["legal"][s])
        d["opp_legal"] = _as_json(v["legal"][1 - s])
        other = per_view.get(k, {}).get(1 - s)
        d["opp_choice"] = None if other is None else other["choice"]
        d["opp_tokens"] = None if other is None else other["tokens"]
    winner = res.get("winner")
    return {"game": game_id, "winner": winner, "decisions": decs}


def cmd_replay(a: argparse.Namespace) -> int:
    """games.jsonl.gz → decisions.jsonl.gz (one line per decision, no row) + rows.npy (float32, decision order)."""
    from concurrent.futures import ThreadPoolExecutor

    games = []
    with gzip.open(a.games, "rt") as f:
        for line in f:
            g = json.loads(line)
            gid = f"{g['cell']}#{g['game']}"
            games.append((gid, g))
    games.sort(key=lambda t: t[0])
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[np.ndarray] = []
    meta_f = gzip.open(out_dir / "decisions.jsonl.gz", "wt")
    t0 = time.time()
    n_dec = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(a.workers))) as ex:
            for gid_g, res in zip(games, ex.map(lambda t: replay_game(t[0], t[1]["script"]), games)):
                gid, g = gid_g
                for d in res["decisions"]:
                    rows.append(d.pop("row"))
                    d["winner"] = res["winner"]
                    d["cell"] = g["cell"]
                    d["seat_label"] = g["player_label"] if d["side"] == 0 else g["opponent_label"]
                    d["seat_ckpt"] = g["player"] if d["side"] == 0 else g["opponent"]
                    d["idx"] = n_dec
                    meta_f.write(json.dumps(d, sort_keys=True) + "\n")
                    n_dec += 1
    finally:
        meta_f.close()
    arr = np.stack(rows).astype(np.float32)
    np.save(out_dir / "rows.npy", arr)
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    lay = Gen3ObservationEncoder(load_mappings()).get_layout()
    (out_dir / "obs_layout.json").write_text(json.dumps(lay, indent=1, default=str))
    _emit({"cmd": "replay", "games": len(games), "decisions": n_dec, "obs_dim": int(arr.shape[1]),
           "seconds": round(time.time() - t0, 1)})
    return 0


# ---------------------------------------------------------------------------------------------- capture
def _load(ckpt: str, random_init: Optional[int]):
    import torch

    from main.policy_spectrum.reader import load_checkpoint

    model = load_checkpoint(Path(ckpt))
    if random_init is None:
        return model
    pol = model.policy
    torch.manual_seed(int(random_init))
    np.random.seed(int(random_init) % (2 ** 32))
    fresh = type(pol)(pol.observation_space, pol.action_space, model.lr_schedule, **model.policy_kwargs)
    fresh.eval()
    # the fresh build must be the SAME architecture (every key and shape) with DIFFERENT weights
    sa, sb = pol.state_dict(), fresh.state_dict()
    if list(sa) != list(sb) or any(sa[k].shape != sb[k].shape for k in sa):
        raise RuntimeError("the random-init build's state_dict differs in keys / shapes from the checkpoint's")
    moved = sum(1 for k in sa if sa[k].dtype.is_floating_point and not torch.equal(sa[k], sb[k]))
    if moved == 0:
        raise RuntimeError("the random-init build equals the checkpoint")
    model.policy = fresh
    return model


def _seat_layout(fe, n_tokens: int) -> Dict[str, Any]:
    """Which seat is which, read off the BUILT extractor (the prober's ``token_layout`` rule)."""
    tt = fe.team_transformer
    es = getattr(fe, "entity_seats", None)
    base = int(getattr(tt, "_total_tokens", 13) or 13)
    board = tuple(int(i) for i in (getattr(tt, "board_seats", None) or (12,)))
    if base == 13:
        board = (12, 12, 12)
    n_e3 = 4 if es is not None else 0
    k_e4 = int(getattr(es, "topk_seats", 0) or 0) if es is not None else 0
    n_tail = 6 if (es is not None and getattr(es, "tail_seats", False)) else 0
    has_other = getattr(fe, "hypothesis_builder", None) is not None
    rest = n_tokens - base - n_e3 - k_e4 - n_tail - (1 if has_other else 0)
    if rest < 0:
        raise RuntimeError(f"seat layout does not add up: n={n_tokens} base={base} e3={n_e3} e4={k_e4} tail={n_tail}")
    return {"n_tokens": n_tokens, "base": base, "board": list(board), "e3": [base + i for i in range(n_e3)],
            "e4": [base + n_e3 + i for i in range(k_e4)], "n_events": rest}


def _mon_seats(rows: np.ndarray) -> Dict[str, np.ndarray]:
    from agents.observation.constants import (OFFSET_OPP_TEAM, OFFSET_OUR_TEAM, POKEMON_ACTIVE_OFFSET,
                                              POKEMON_FULL_DIM, POKEMON_HP_OFFSET)

    P = POKEMON_FULL_DIM
    our_act = rows[:, [OFFSET_OUR_TEAM + i * P + POKEMON_ACTIVE_OFFSET for i in range(6)]] > 0.5
    opp_act = rows[:, [OFFSET_OPP_TEAM + i * P + POKEMON_ACTIVE_OFFSET for i in range(6)]] > 0.5
    our_hp = rows[:, [OFFSET_OUR_TEAM + i * P + POKEMON_HP_OFFSET for i in range(6)]]
    oa = np.where(our_act.sum(1) == 1, our_act.argmax(1), -1)
    ta = np.where(opp_act.sum(1) == 1, opp_act.argmax(1), -1)
    bench_ok = (~our_act) & (our_hp > 0)
    ob = np.where(bench_ok.any(1), bench_ok.argmax(1), -1)
    return {"OA": oa, "TA": ta, "OB": ob}


#: the token SITES a capture keeps, in this order (`board_*` are the legacy global seat for all three)
TOKEN_SITES = ("OA", "TA", "OB", "M0", "M1", "M2", "M3", "B_OURS", "B_THEIRS", "B_FIELD")


def cmd_capture(a: argparse.Namespace) -> int:
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from main.policy_spectrum.reader import inference_globals

    rows = np.load(Path(a.bank) / "rows.npy", mmap_mode="r")
    masks = np.load(Path(a.bank) / "masks.npy")
    N = len(rows) if a.limit is None else min(len(rows), int(a.limit))
    seats = _mon_seats(np.asarray(rows[:N]))
    t0 = time.time()
    with inference_globals(int(a.threads)):
        model = _load(a.ckpt, a.random_init)
        pol = model.policy
        fe = pol.features_extractor
        layers = trunk_rounds(fe)
        cur: Dict[str, List[Any]] = {"x": [], "out": [], "vf": [], "pi": []}

        def pre(mod, args, kwargs):
            cur["x"].append(args[0])

        def post(mod, args, kwargs, out):
            cur["out"].append(out)

        def fe_hook(mod, args, out):
            cur["vf"].append(out[1])

        def pi_hook(mod, args, out):
            cur["pi"].append(out)

        hs = []
        for L in layers:
            hs.append(L.register_forward_pre_hook(pre, with_kwargs=True))
            hs.append(L.register_forward_hook(post, with_kwargs=True))
        hs.append(fe.register_forward_hook(fe_hook))
        hs.append(pol.mlp_extractor.policy_net.register_forward_hook(pi_hook))
        n_depth = 1 + len(layers)
        probs = np.zeros((N, 11), np.float32)
        winp = np.zeros(N, np.float32)
        tok = None
        pi_lat = vf_lat = None
        lay = None
        try:
            for s in range(0, N, int(a.batch)):
                for v in cur.values():
                    v.clear()
                e = min(N, s + int(a.batch))
                mt = torch.as_tensor(masks[s:e].astype(np.float32))
                obs = {"observation": torch.as_tensor(np.array(rows[s:e], dtype=np.float32)), "action_mask": mt}
                obs.update(zero_extra_obs(fe, batch=e - s, device=torch.device("cpu")))
                with torch.no_grad():
                    lg = pol.get_distribution(obs).distribution.logits
                    lg = torch.where(mt.bool(), lg, torch.full_like(lg, -1e8))
                    probs[s:e] = torch.softmax(lg, 1).numpy()
                    winp[s:e] = torch.sigmoid(fe.last_win_prob_logits.reshape(-1)).numpy()
                    B, n, dm = cur["x"][0].shape
                    if lay is None:
                        lay = _seat_layout(fe, n)
                        tok = np.zeros((N, len(TOKEN_SITES), n_depth, dm), np.float16)
                        pi_lat = np.zeros((N, cur["pi"][0].shape[-1]), np.float16)
                        vf_lat = np.zeros((N, cur["vf"][0].shape[-1]), np.float16)
                    bi = torch.arange(e - s)
                    idx = {k: torch.as_tensor(np.maximum(v[s:e], 0)) for k, v in seats.items()}
                    seat_idx = [idx["OA"], 6 + idx["TA"], idx["OB"]] + \
                               [torch.full((e - s,), lay["e3"][i], dtype=torch.long) for i in range(4)] + \
                               [torch.full((e - s,), lay["board"][i], dtype=torch.long) for i in range(3)]
                    depths = [cur["x"][0]] + list(cur["out"])
                    for di, T in enumerate(depths):
                        for si, sx in enumerate(seat_idx):
                            tok[s:e, si, di] = T[bi, sx].numpy().astype(np.float16)
                    pi_lat[s:e] = cur["pi"][-1].numpy().astype(np.float16)
                    vf_lat[s:e] = cur["vf"][-1].numpy().astype(np.float16)
        finally:
            for h in hs:
                h.remove()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, probs=probs, win_prob=winp, tok=tok, pi=pi_lat, vf=vf_lat,
             OA=seats["OA"], TA=seats["TA"], OB=seats["OB"], sites=np.array(TOKEN_SITES),
             layout=np.array(json.dumps(lay)), n_layers=np.array(len(layers)))
    os.replace(tmp, out)
    _emit({"cmd": "capture", "rows": int(N), "label": a.label, "layers": len(layers), "layout": lay,
           "random_init": a.random_init, "seconds": round(time.time() - t0, 1)})
    return 0


def cmd_forward(a: argparse.Namespace) -> int:
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from main.policy_spectrum.reader import inference_globals

    d = np.load(a.rows)
    rows, masks = d["rows"].astype(np.float32), d["masks"]
    N = len(rows)
    probs = np.zeros((N, 11), np.float32)
    winp = np.zeros(N, np.float32)
    with inference_globals(int(a.threads)):
        model = _load(a.ckpt, None)
        pol = model.policy
        fe = pol.features_extractor
        for s in range(0, N, int(a.batch)):
            e = min(N, s + int(a.batch))
            mt = torch.as_tensor(masks[s:e].astype(np.float32))
            obs = {"observation": torch.as_tensor(rows[s:e]), "action_mask": mt}
            obs.update(zero_extra_obs(fe, batch=e - s, device=torch.device("cpu")))
            with torch.no_grad():
                lg = pol.get_distribution(obs).distribution.logits
                lg = torch.where(mt.bool(), lg, torch.full_like(lg, -1e8))
                probs[s:e] = torch.softmax(lg, 1).numpy()
                winp[s:e] = torch.sigmoid(fe.last_win_prob_logits.reshape(-1)).numpy()
    np.savez(a.out, probs=probs, win_prob=winp)
    _emit({"cmd": "forward", "rows": int(N), "label": a.label})
    return 0


# ------------------------------------------------------------------------------------- depth / capacity use
#: the truncation ranks of the capacity-use test (d_model = 128)
TRUNC_RANKS = (16, 32, 48, 64, 96)
#: the INPUT token projections truncated together (whichever the architecture builds)
INPUT_PROJ = ("pokemon_encoder.role_encoder.2", "pokemon_encoder.dynamic_encoder.2", "entity_seats.move_seat_proj",
              "entity_seats.threat_seat_proj", "team_transformer.global_proj", "history_events.proj")


#: The trunk's attention ROUNDS by class name, in EXECUTION order within the extractor's module tree: the post-LN
#: ``BiasedEncoderLayer``s, then (``--trunk-layers 3/4``, `agents.model.trunk_depth`) the PRE-LN ``IdentityInitRound``s
#: appended after them. A checkout that builds neither has no trunk this worker reads (refused).
ROUND_TYPES = ("BiasedEncoderLayer", "IdentityInitRound")
PRE_LN_ROUNDS = ("IdentityInitRound",)


def trunk_rounds(fe) -> List[Any]:
    rounds = [m for m in fe.modules() if type(m).__name__ in ROUND_TYPES]
    if not rounds:
        raise RuntimeError("the extractor builds no trunk round this worker knows "
                           f"({', '.join(ROUND_TYPES)}) — read its commit's trunk before probing it")
    return rounds


def _layer_forward(L, x, bias, *, mode: str = "full", head_mask=None, stats=None):
    """A REPLICA of ``BiasedEncoderLayer.forward`` (post-LN) with the battery's variants: ``full``; ``lens`` (the
    layer's output IS its input — the logit lens: the heads read the previous round); ``zero_update`` (both residual
    updates zeroed, the two LayerNorms kept); ``no_attn`` / ``no_ffn`` (one update zeroed); ``head_mask`` [H] scales
    each head's attention output. ``stats`` (a dict) receives the post-attention state and the FFN pre-activation."""
    import torch

    try:
        from agents.model.team_transformer import dense_attn_bias
    except ImportError:                      # an older trunk: the bias goes in as is
        def dense_attn_bias(b):
            return b
    if mode == "lens":
        return x
    B, n, d = x.shape
    if type(L).__name__ in PRE_LN_ROUNDS:
        # pre-LN (`IdentityInitRound`): x <- x + out(attn(LN1 x)); x <- x + W2 relu(W1 LN2 x). Zeroing BOTH updates
        # IS the identity, so `zero_update` equals `lens` here.
        if mode == "zero_update":
            return x
        x1 = x
        if mode != "no_attn":
            qkv = L.in_proj(L.norm1(x)).reshape(B, n, 3, L.n_heads, L.head_dim)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None if bias is None else dense_attn_bias(bias))
            if head_mask is not None:
                attn = attn * head_mask.view(1, -1, 1, 1).to(attn.dtype)
            x1 = x + L.out_proj(attn.transpose(1, 2).reshape(B, n, d))
        if mode == "no_ffn":
            return x1
        h = L.linear1(L.norm2(x1))
        if stats is not None:
            stats["x1"] = x1
            stats["h"] = h
        return x1 + L.linear2(torch.nn.functional.relu(h))
    if mode in ("zero_update", "no_attn"):
        x1 = L.norm1(x)
    else:
        qkv = L.in_proj(x).reshape(B, n, 3, L.n_heads, L.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None if bias is None else dense_attn_bias(bias))
        if head_mask is not None:
            attn = attn * head_mask.view(1, -1, 1, 1).to(attn.dtype)
        x1 = L.norm1(x + L.out_proj(attn.transpose(1, 2).reshape(B, n, d)))
    if mode in ("zero_update", "no_ffn"):
        return L.norm2(x1)
    h = L.linear1(x1)
    if stats is not None:
        stats["x1"] = x1
        stats["h"] = h
    return L.norm2(x1 + L.linear2(torch.nn.functional.relu(h)))


def _groups(lay: Dict[str, Any]) -> Dict[str, List[int]]:
    base, n = lay["base"], lay["n_tokens"]
    g = {"our_mons": list(range(0, 6)), "their_mons": list(range(6, 12)),
         "board": sorted(set(lay["board"])), "E3": list(lay["e3"]), "E4": list(lay["e4"])}
    after = (lay["e4"][-1] + 1) if lay["e4"] else (lay["e3"][-1] + 1 if lay["e3"] else base)
    ev0 = n - lay["n_events"]
    rest = list(range(after, ev0))
    if len(rest) >= 6:
        g["E5"] = rest[:6]
        if len(rest) > 6:
            g["OTHER"] = rest[6:]
    elif rest:
        g["OTHER"] = rest
    g["events"] = list(range(ev0, n))
    return g


def _spectrum(W) -> Dict[str, Any]:
    s = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
    e = s ** 2
    c = np.cumsum(e) / max(e.sum(), 1e-30)
    return {"shape": list(W.shape), "pr": float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30)),
            "n90": int(np.searchsorted(c, 0.90) + 1), "n99": int(np.searchsorted(c, 0.99) + 1),
            "sv": [round(float(x), 5) for x in s]}


def _truncate(W, r: int):
    import torch

    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r] if r < len(S) else W.double()


def cmd_depth(a: argparse.Namespace) -> int:
    """Depth-use and capacity-use diagnostics for one checkpoint on a deterministic bank subsample (every k-th row)."""
    import types

    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from main.policy_spectrum.reader import inference_globals

    rows_all = np.load(Path(a.bank) / "rows.npy", mmap_mode="r")
    masks_all = np.load(Path(a.bank) / "masks.npy")
    stride = max(1, int(np.ceil(len(rows_all) / int(a.n_rows))))
    idx = np.arange(0, len(rows_all), stride)
    rows = np.array(rows_all[idx], dtype=np.float32)
    masks = masks_all[idx]
    multi = masks.sum(1) >= 2
    t0 = time.time()
    out: Dict[str, Any] = {"label": a.label, "rows": int(len(idx)), "stride": int(stride)}
    with inference_globals(int(a.threads)):
        model = _load(a.ckpt, a.random_init)
        pol = model.policy
        fe = pol.features_extractor
        layers = trunk_rounds(fe)
        nl = len(layers)
        H = int(layers[0].n_heads)
        state: Dict[str, Any] = {"mode": {}, "head": {}, "rec": None}

        def make_fwd(i):
            def fwd(self, x, bias=None):
                st = {} if state["rec"] is not None else None
                y = _layer_forward(self, x, bias, mode=state["mode"].get(i, "full"),
                                   head_mask=state["head"].get(i), stats=st)
                if state["rec"] is not None:
                    state["rec"].append({"i": i, "x": x, "bias": bias, "y": y, **(st or {})})
                return y
            return fwd

        orig = [L.forward for L in layers]

        def run(record=False):
            P, Wp, recs = [], [], []
            for s in range(0, len(rows), int(a.batch)):
                e = min(len(rows), s + int(a.batch))
                mt = torch.as_tensor(masks[s:e].astype(np.float32))
                obs = {"observation": torch.as_tensor(rows[s:e]), "action_mask": mt}
                obs.update(zero_extra_obs(fe, batch=e - s, device=torch.device("cpu")))
                state["rec"] = [] if record else None
                with torch.no_grad():
                    lg = pol.get_distribution(obs).distribution.logits
                    lg = torch.where(mt.bool(), lg, torch.full_like(lg, -1e8))
                    P.append(torch.softmax(lg, 1).numpy())
                    Wp.append(torch.sigmoid(fe.last_win_prob_logits.reshape(-1)).numpy())
                if record:
                    recs.append(state["rec"])
            state["rec"] = None
            return np.concatenate(P), np.concatenate(Wp), recs

        # the replica must BE the layer before any variant is read
        with torch.no_grad():
            p_orig, w_orig, _ = run()
        try:
            for i, L in enumerate(layers):
                L.forward = types.MethodType(make_fwd(i), L)
            p_full, w_full, recs = run(record=True)
            order = [r["i"] for batch in recs for r in batch]
            if order != list(range(nl)) * len(recs):
                raise RuntimeError(f"the trunk rounds ran in order {order[:nl]}, not their module order — the "
                                   "per-round variant names would be wrong")
            rep = float(np.abs(p_full - p_orig).max()), float(np.abs(w_full - w_orig).max())
            if max(rep) > 1e-4:
                raise RuntimeError(f"the layer replica differs from BiasedEncoderLayer.forward by {rep}")
            out["replica_max_abs_diff"] = rep

            def delta(p, w):
                kl = np.sum(p_full * (np.log(np.clip(p_full, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1))), 1)
                return {"kl": float(kl[multi].mean()),
                        "top1_agree": float((p.argmax(1) == p_full.argmax(1))[multi].mean()),
                        "value_mae": float(np.abs(w - w_full).mean())}

            V: Dict[str, Any] = {}
            for i in range(nl):
                for mode in ("lens", "zero_update", "no_attn", "no_ffn"):
                    state["mode"] = {i: mode}
                    V[f"{mode}_L{i + 1}"] = delta(*run()[:2])
                state["mode"] = {}
                for h in range(H):
                    hm = torch.ones(H)
                    hm[h] = 0.0
                    state["head"] = {i: hm}
                    V[f"head_off_L{i + 1}_h{h}"] = delta(*run()[:2])
                state["head"] = {}
            # the low-rank truncation test
            named = dict(fe.named_modules())
            trunk_W = []
            for L in layers:
                trunk_W += [(L.in_proj, "qkv"), (L.out_proj, None), (L.linear1, None), (L.linear2, None)]
            input_W = [(named[n], None) for n in INPUT_PROJ if n in named]
            for gname, mats in (("trunk", trunk_W), ("input", input_W)):
                saved = [m.weight.detach().clone() for m, _ in mats]
                for r in TRUNC_RANKS:
                    with torch.no_grad():
                        for (m, kind), W0 in zip(mats, saved):
                            if kind == "qkv":
                                d = W0.shape[1]
                                m.weight.copy_(torch.cat([_truncate(W0[j * d:(j + 1) * d], r) for j in range(3)])
                                               .to(W0.dtype))
                            else:
                                m.weight.copy_(_truncate(W0, r).to(W0.dtype))
                    V[f"trunc_{gname}_r{r}"] = delta(*run()[:2])
                with torch.no_grad():
                    for (m, _), W0 in zip(mats, saved):
                        m.weight.copy_(W0)
            out["variants"] = V
            out["input_proj_truncated"] = [n for n in INPUT_PROJ if n in named]
        finally:
            for L, f in zip(layers, orig):
                L.forward = f
        # the per-token-type statistics, from the recorded full pass
        n_tok = recs[0][0]["x"].shape[1]
        lay = _seat_layout(fe, n_tok)
        G = _groups(lay)
        d = recs[0][0]["x"].shape[2]
        acc = {(g, dep): [np.zeros(d), np.zeros((d, d)), 0] for g in G for dep in range(nl + 1)}
        upd = {(g, i): [0.0, 0] for g in G for i in range(nl)}
        ffn_on = [None] * nl
        ffn_n = [0] * nl
        for batch in recs:
            keymask = (batch[0]["bias"][:, 0] <= -1e8).all(dim=1).numpy()          # [B, n] masked keys
            live = ~keymask
            for r in batch:
                i = r["i"]
                X = r["x"].numpy().astype(np.float64)
                Y = r["y"].numpy().astype(np.float64)
                hpos = (r["h"].numpy() > 0)
                lv = live.reshape(-1)
                hp = hpos.reshape(-1, hpos.shape[-1])[lv]
                ffn_on[i] = hp.sum(0) if ffn_on[i] is None else ffn_on[i] + hp.sum(0)
                ffn_n[i] += int(lv.sum())
                for g, seats in G.items():
                    m = live[:, seats]
                    xs, ys = X[:, seats][m], Y[:, seats][m]
                    if len(xs) == 0:
                        continue
                    rel = np.linalg.norm(ys - xs, axis=1) / np.maximum(np.linalg.norm(xs, axis=1), 1e-9)
                    upd[(g, i)][0] += float(rel.sum())
                    upd[(g, i)][1] += len(rel)
                    for dep, Z in ((i, xs), (i + 1, ys)) if i == 0 else ((i + 1, ys),):
                        A_ = acc[(g, dep)]
                        A_[0] += Z.sum(0)
                        A_[1] += Z.T @ Z
                        A_[2] += len(Z)
        ranks: Dict[str, Any] = {}
        for (g, dep), (s1, s2, c) in acc.items():
            if c < 2 * d:
                continue
            mu = s1 / c
            C = s2 / c - np.outer(mu, mu)
            ev = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
            cs = np.cumsum(ev) / max(ev.sum(), 1e-30)
            ranks[f"{g}@{['in', 'L1', 'L2', 'L3', 'L4'][dep]}"] = {
                "pr": float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30)), "n90": int(np.searchsorted(cs, 0.90) + 1),
                "n99": int(np.searchsorted(cs, 0.99) + 1), "tokens": int(c)}
        out["rank"] = ranks
        out["update_norm"] = {f"{g}@L{i + 1}": (v[0] / v[1] if v[1] else None) for (g, i), v in upd.items()}
        out["ffn"] = {f"L{i + 1}": {"units": int(len(ffn_on[i])), "dead": float((ffn_on[i] == 0).mean()),
                                    "always_on": float((ffn_on[i] >= 0.999 * ffn_n[i]).mean()),
                                    "mean_active": float((ffn_on[i] / max(ffn_n[i], 1)).mean())}
                      for i in range(nl)}
        spec = {}
        for li, L in enumerate(layers):
            W = L.in_proj.weight.detach().numpy()
            dd = W.shape[1]
            for j, nm in enumerate("qkv"):
                spec[f"L{li + 1}.{nm}_proj"] = _spectrum(W[j * dd:(j + 1) * dd])
            for nm in ("out_proj", "linear1", "linear2"):
                spec[f"L{li + 1}.{nm}"] = _spectrum(getattr(L, nm).weight.detach().numpy())
        named = dict(fe.named_modules())
        for n in INPUT_PROJ:
            if n in named:
                spec[n] = _spectrum(named[n].weight.detach().numpy())
        out["spectra"] = spec
        out["layout"] = lay
        out["groups"] = G
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=0))
    _emit({"cmd": "depth", "label": a.label, "rows": int(len(idx)), "seconds": out["seconds"]})
    return 0


def cmd_archetypes(a: argparse.Namespace) -> int:
    """``{packed team: archetype}`` over the checkout's pool: each EXPORT classified, keyed by its packing through the
    training teambuilder (the h2h team table's own rule; ``classify_team`` reads an export, never a packed string)."""
    from agents.training.team_archetypes import classify_team
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    loader = TeamLoader()
    texts = list(loader.get_all_teams()) + list(loader.get_sample_teams())
    packed = Gen3Teambuilder(texts).packed_teams
    if len(packed) != len(texts):
        raise SystemExit(f"[pin_worker] {len(texts) - len(packed)} pool teams failed the teambuilder")
    table = {p: str(classify_team(t)["archetype"]) for t, p in zip(texts, packed)}
    Path(a.out).write_text(json.dumps(table, sort_keys=True))
    _emit({"cmd": "archetypes", "teams": len(table)})
    return 0


# ------------------------------------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pin_worker")
    ap.add_argument("--checkout", required=True, help="the checkout whose code runs (its src/ is on PYTHONPATH)")
    ap.add_argument("--expect-commit", default=None, help="refuse unless the checkout's HEAD starts with this")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--cells", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--torch-threads", type=int, default=3)
    p = sub.add_parser("replay")
    p.add_argument("--games", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("capture")
    p.add_argument("--bank", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--random-init", type=int, default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("depth")
    p.add_argument("--bank", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--random-init", type=int, default=None)
    p.add_argument("--n-rows", type=int, default=4000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=256)
    p = sub.add_parser("archetypes")
    p.add_argument("--out", required=True)
    p = sub.add_parser("forward")
    p.add_argument("--rows", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=256)
    a = ap.parse_args(argv)
    root = Path(a.checkout).resolve()
    head = _checkout_head(root)
    if a.expect_commit and not head.startswith(a.expect_commit):
        raise SystemExit(f"[pin_worker] REFUSED: {root} is at {head[:12]}, not {a.expect_commit}")
    import agents
    if not str(Path(agents.__file__).resolve()).startswith(str(root)):
        raise SystemExit(f"[pin_worker] REFUSED: agents imports from {agents.__file__}, not {root}/src "
                         "(PYTHONPATH must put the checkout's src first)")
    os.chdir(root)                         # the team pool and data/ are read relative to the working directory
    return {"play": cmd_play, "replay": cmd_replay, "capture": cmd_capture, "forward": cmd_forward,
            "archetypes": cmd_archetypes, "depth": cmd_depth}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
