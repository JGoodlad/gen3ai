"""The PLASTICITY CHECK on X26's lineage (2026-10-26): CPU only, forward passes plus short fine-tunes, READ-ONLY
on ``models/``. A descriptive measurement script, not a registered instrument.

    python plasticity_probe.py bank      --out <dir>
    python plasticity_probe.py structure --bank <dir> --out <json> --feats <dir> --net <label>=<spec> [...]
    python plasticity_probe.py heads     --bank <dir> --feats <dir> --out <json>
    python plasticity_probe.py refit     --bank <dir> --out <json> --net <label>=<spec> [...] --target <t> [...]

``<spec>`` is a checkpoint ``.zip`` or ``fresh:<seed>:<config .zip>`` (the config checkpoint's own architecture,
rebuilt through the production init path: SB3's ``_build`` plus ``restore_identity_init``, then its weights checked
to differ from the checkpoint's).

THE BANKS (``bank``), every one seeded and reproducible:
  * ``fixture``: ``utils.rust_env.fixture_battles.play_rows`` exactly (p1 a seeded uniform draw over the mask against
    the episode-seeded heuristic bot, the whole 719-team pool);
  * ``golden``: the obs golden's 991 core rows (``agents.training.golden_obs_core.core_rows``; no recorded mask, so
    an all-legal mask: the extractor's taps do not read it);
  * ``unfam``: LABELLED games on teams the lineage never trained on (a seeded draw of the ladder corpus, which shares
    no packed team with the pool), self-play by a REFERENCE policy outside the compared set (X26 @50M, sampled at
    T = 1, seeded), recording p1's rows, the sampled action, the winner and both teams. Its games are split 3 : 1
    into train / held-out BY GAME (a game's rows share its label).

THE PROBES:
  * ``structure``: tau-dormant share (Sokar et al. 2023's score, mean |activation| over rows and token positions,
    normalised by the layer mean) on every ReLU output and transformer FFN hidden, plus the policy MLP's tanh units
    (dormant share AND saturated share: |a| > 0.99 on >= 99 % of rows); Kumar et al. 2021's srank (delta = 0.01,
    singular values of the UNCENTRED feature matrix) and the centred participation ratio at the penultimate taps;
    the weight norm; and per-example policy-gradient statistics on 64 fixed rows (grad of log pi(a|s) at the bank's
    own action: mean norm, mean off-diagonal cosine, the Gram matrix's participation ratio).
  * ``heads``: the trunk FROZEN, a fresh ZERO-initialised linear head on the cached penultimate features
    [latent_pi (512) ++ vf (128)] trained by Adam for a fixed budget on the unfam train games; loss at fixed steps
    and held-out loss / AUC.
  * ``refit``: the trunk UNFROZEN (extractor + policy MLP) under a fresh ZERO-initialised linear head, Adam at one
    fixed rate for a fixed budget, on a fixed new target; the held-out loss curve. The direct plasticity read.
  Targets: ``t_net`` (random projections of a fresh net's latent_pi), ``t_lin`` (the capacity battery's random
  function of the row), ``win``, ``species`` (heads only) and ``pstar`` (P*'s greedy action: the strategy doc's
  registered target).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REF_POLICY = "/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_50000106_steps.zip"
THREADS = 3
TAUS = (0.0, 0.025, 0.1)
N_GRAD_ROWS = 64
N_SPECIES = 8


# ------------------------------------------------------------------------------------------------ loading
def load_net(spec: str):
    """A checkpoint (inference load, arch signature verified) or a fresh build of one's architecture."""
    import torch as th

    from main.policy_spectrum.reader import load_checkpoint

    if not spec.startswith("fresh:"):
        m = load_checkpoint(Path(spec))
        return m.policy.eval()
    _, seed, cfg = spec.split(":", 2)
    m = load_checkpoint(Path(cfg))
    pol = m.policy
    th.manual_seed(int(seed))
    np.random.seed(int(seed))
    fresh = type(pol)(pol.observation_space, pol.action_space, m.lr_schedule, **m.policy_kwargs)
    sa, sb = pol.state_dict(), fresh.state_dict()
    if list(sa) != list(sb) or any(sa[k].shape != sb[k].shape for k in sa):
        raise RuntimeError("the fresh build's state_dict differs in keys / shapes from the checkpoint's")
    if not any(sa[k].dtype.is_floating_point and not th.equal(sa[k], sb[k]) for k in sa):
        raise RuntimeError("the fresh build equals the checkpoint")
    return fresh.eval()


def obs_dict(pol, obs: np.ndarray, mask: np.ndarray):
    import torch as th

    from agents.model.extra_obs_keys import zero_extra_obs

    ob = {"observation": th.as_tensor(obs), "action_mask": th.as_tensor(mask.astype(np.float32))}
    ob.update(zero_extra_obs(pol.features_extractor, batch=len(obs), device="cpu"))
    return ob


def penultimate(pol, ob):
    """(latent_pi [B, 512], vf_features [B, 128], pi_features, value_pooled): the inputs to the action and value
    heads (latent_pi is the policy MLP's output, the pointer head's context)."""
    fe = pol.features_extractor
    pi, vf = fe(ob)
    lat = pol.mlp_extractor.forward_actor(pi)
    return lat, vf, pi, fe.last_value_pooled


# ------------------------------------------------------------------------------------------------ bank
def _species(packed: str) -> List[str]:
    out = []
    for mon in packed.split("]"):
        f = mon.split("|")
        if len(f) > 1:
            out.append((f[1] or f[0]).strip().lower())
    return out


def collect_labelled(teams: List[str], n_games: int, seed: int, policy, n_envs: int = 16,
                     turn_limit: int = 250) -> Dict[str, np.ndarray]:
    """``play_rows``' own loop (self-play, the column contract's staging), recording per p1 row its game, the
    sampled action, and per game the winner and both team indices. The staged pair an env plays is the one staged
    BEFORE its last reset."""
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P
    from utils.rust_env.fixture_battles import NAMES, _lib

    spec = P.spec_json(n=n_envs, threads=1, teams=teams, names=NAMES, turn_limit=turn_limit, refusal_budget=4,
                       bank_dir=None, opponents=[{"kind": "external"}])
    core = ffi.FfiCore(spec, lib=_lib("release"))
    c = core.cols
    rng = np.random.default_rng(seed)
    staged: Dict[int, np.ndarray] = {}
    current: Dict[int, np.ndarray] = {}

    def stage(envs) -> None:
        for e in envs:
            pair = rng.choice(len(teams), 2, replace=False)
            c["ep_team"][e] = pair
            c["ep_seed"][e] = rng.integers(0, 65536, 4)
            c["ep_opp"][e] = 0
            staged[e] = pair

    stage(range(n_envs))
    core.reset()
    current.update({e: staged[e] for e in range(n_envs)})
    stage(range(n_envs))
    buf: Dict[int, list] = {e: [] for e in range(n_envs)}
    rows, masks, acts, gid = [], [], [], []
    winners, t1, t2 = [], [], []
    try:
        while len(winners) < n_games:
            pend = [(e, s) for e in range(n_envs) for s in (0, 1)
                    if c["need"][e, s] and not (s == 0 and int(c["turn"][e]) >= turn_limit)]
            if pend:
                obs = np.stack([c["obs"][e, s].copy() for e, s in pend]).astype(np.float32)
                mask = np.stack([c["mask"][e, s].astype(bool) for e, s in pend])
                a = policy(obs, mask)
                for k, (e, s) in enumerate(pend):
                    if not mask[k][a[k]]:
                        raise ValueError("illegal action")
                    if s == 0:
                        buf[e].append((obs[k], mask[k], int(a[k])))
                    c["action"][e, s] = int(a[k])
            core.step()
            done = core.finished()
            for f in done:
                e = f["env"]
                game, buf[e] = buf[e], []
                pair = current[e]
                current[e] = staged[e]
                if len(winners) < n_games and game:
                    g = len(winners)
                    winners.append(int(f["winner"]))
                    t1.append(int(pair[0]))
                    t2.append(int(pair[1]))
                    for o, m, ac in game:
                        rows.append(o)
                        masks.append(m)
                        acts.append(ac)
                        gid.append(g)
            stage(f["env"] for f in done)
    finally:
        core.close()
    return {"obs": np.stack(rows).astype(np.float32), "mask": np.stack(masks), "action": np.array(acts),
            "game": np.array(gid), "winner": np.array(winners), "team_p1": np.array(t1), "team_p2": np.array(t2)}


def cmd_bank(a) -> int:
    import torch as th

    from agents.training.churn_probe import masked_action_probs
    from agents.training.golden_obs_core import core_rows
    from main.policy_spectrum.reader import inference_globals, load_checkpoint
    from utils.rust_env.fixture_battles import play_rows
    from utils.team_sources import packed_teams

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with inference_globals(THREADS):
        g = play_rows(a.fixture_games, seed=20261026, n_teams=None)
        np.savez_compressed(out / "fixture.npz", obs=np.concatenate([x[0] for x in g]),
                            mask=np.concatenate([x[1] for x in g]))
        gold = np.stack(core_rows()).astype(np.float32)
        np.savez_compressed(out / "golden.npz", obs=gold, mask=np.ones((len(gold), 11), dtype=bool))
        print(f"fixture {sum(len(x[0]) for x in g)} rows, golden {len(gold)} rows, {time.time() - t0:.0f}s",
              flush=True)

        pool = set(packed_teams("pool"))
        ladder = [t for t in packed_teams("ladder") if t not in pool]
        pick = np.random.default_rng(20261026).choice(len(ladder), a.unfam_teams, replace=False)
        teams = [ladder[i] for i in sorted(pick)]
        ref = load_checkpoint(Path(REF_POLICY))
        prng = np.random.default_rng(7)

        def policy(obs, mask):
            p = masked_action_probs(ref, obs, mask.astype(np.float32))
            p = np.where(mask, p, 0.0)
            p = p / p.sum(1, keepdims=True)
            return np.array([int(prng.choice(p.shape[1], p=row)) for row in p])

        with th.no_grad():
            d = collect_labelled(teams, a.unfam_games, seed=11, policy=policy)
    sp = [_species(t) for t in teams]
    d["opp_species"] = np.array(["/".join(sp[i]) for i in d["team_p2"]])
    games = np.random.default_rng(3).permutation(len(d["winner"]))
    held = np.zeros(len(d["winner"]), dtype=bool)
    held[games[: len(games) // 4]] = True
    d["held_game"] = held
    np.savez_compressed(out / "unfam.npz", **d)
    (out / "unfam_teams.json").write_text(json.dumps(teams))
    meta = {"fixture_games": a.fixture_games, "unfam_games": int(len(d["winner"])), "unfam_rows": int(len(d["obs"])),
            "unfam_teams": len(teams), "ladder_teams_outside_pool": len(ladder), "ref_policy": REF_POLICY,
            "p1_win_rate": float(np.mean(d["winner"] == 1)), "seconds": round(time.time() - t0, 1)}
    (out / "bank_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta, flush=True)
    return 0


def load_bank(d: Path) -> Dict[str, Dict[str, np.ndarray]]:
    return {k: dict(np.load(d / f"{k}.npz")) for k in ("fixture", "golden", "unfam")}


# ------------------------------------------------------------------------------------------------ targets
def labels(u: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """win (p1 won; a tie is a loss), and the hidden-info label: the opponent's team holds species s, for the
    N_SPECIES species commonest among the TRAIN games' opponents (part of a team is revealed in the row, part is
    hidden)."""
    win = (u["winner"][u["game"]] == 1).astype(np.float32)
    opp = [s.split("/") for s in u["opp_species"]]
    train_games = set(np.flatnonzero(~u["held_game"]).tolist())
    counts: Dict[str, int] = {}
    for g in train_games:
        for s in opp[g]:
            counts[s] = counts.get(s, 0) + 1
    top = sorted(counts, key=lambda s: (-counts[s], s))[:N_SPECIES]
    sp = np.array([[float(s in opp[g]) for s in top] for g in u["game"]], dtype=np.float32)
    return {"win": win[:, None], "species": sp, "species_names": np.array(top)}


def random_net_targets(pol_target, u, seed: int = 99, k: int = 8) -> np.ndarray:
    """T_net: k fixed random projections of a FRESH network's (seed 12345) latent_pi, standardised on the train
    rows: a function the architecture can represent exactly and no checkpoint was trained toward."""
    import torch as th

    feats = []
    with th.no_grad():
        for i in range(0, len(u["obs"]), 256):
            lat, *_ = penultimate(pol_target, obs_dict(pol_target, u["obs"][i:i + 256], u["mask"][i:i + 256]))
            feats.append(lat.float().numpy())
    z = np.concatenate(feats).astype(np.float64)
    w = np.random.default_rng(seed).standard_normal((z.shape[1], k)) / np.sqrt(z.shape[1])
    t = z @ w
    tr = ~u["held_game"][u["game"]]
    return ((t - t[tr].mean(0)) / (t[tr].std(0) + 1e-8)).astype(np.float32)


def random_obs_targets(u, k: int = 8) -> np.ndarray:
    """T_lin: the capacity battery's random target family on the raw row, standardised on the train rows."""
    from agents.model.capacity_probes import random_targets

    t = random_targets(u["obs"], k, seed=0)
    tr = ~u["held_game"][u["game"]]
    return ((t - t[tr].mean(0)) / (t[tr].std(0) + 1e-8)).astype(np.float32)


PSTAR = "/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip"


def pstar_target(bank: Path, u) -> np.ndarray:
    """The strategy doc's registered target (``strategy_if_levers_null_2026-10-26.md`` §2): P*'s own GREEDY action on
    every unfam row, as [one-hot (11) ++ legal mask (11)]; computed once and cached beside the bank."""
    f = bank / "pstar_greedy.npy"
    if not f.exists():
        from agents.training.churn_probe import masked_action_probs
        from main.policy_spectrum.reader import inference_globals, load_checkpoint

        with inference_globals(THREADS):
            p = masked_action_probs(load_checkpoint(Path(PSTAR)), u["obs"], u["mask"].astype(np.float32))
        np.save(f, np.where(u["mask"], p, -1.0).argmax(1))
    g = np.load(f)
    if not u["mask"][np.arange(len(g)), g].all():
        raise RuntimeError("a P* greedy label is illegal")
    Y = np.zeros((len(g), 22), dtype=np.float32)
    Y[np.arange(len(g)), g] = 1.0
    Y[:, 11:] = u["mask"]
    return Y


# ------------------------------------------------------------------------------------------------ structure
def srank(Z: np.ndarray, delta: float = 0.01) -> int:
    s = np.linalg.svd(np.asarray(Z, dtype=np.float64), compute_uv=False)
    c = np.cumsum(s) / max(s.sum(), 1e-300)
    return int((c < 1 - delta).sum()) + 1


def structure_one(pol, bank: Dict[str, Dict[str, np.ndarray]], feats_out: Optional[Path], label: str) -> dict:
    import torch as th
    import torch.nn.functional as F

    from agents.training.rank_metrics import effective_rank

    res: Dict[str, Any] = {}
    for bname in ("fixture", "golden", "unfam"):
        b = bank[bname]
        sums: Dict[str, th.Tensor] = {}
        cnt: Dict[str, int] = {}
        sat: Dict[str, th.Tensor] = {}
        hooks = []

        def acc(name, y):
            y = y.detach().float()
            flat = y.reshape(-1, y.shape[-1]).abs()
            sums[name] = sums.get(name, 0) + flat.sum(0)
            cnt[name] = cnt.get(name, 0) + flat.shape[0]

        def acc_tanh(name, y):
            acc(name, y)
            s = (y.detach().float().abs() > 0.99).float().sum(0)
            sat[name] = sat.get(name, 0) + s

        for name, mod in pol.named_modules():
            if name.startswith("ridealong"):
                continue
            if isinstance(mod, th.nn.ReLU):
                hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: acc(n, o)))
            elif isinstance(mod, th.nn.Tanh) and name.startswith("mlp_extractor"):
                hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: acc_tanh(n, o)))
            elif isinstance(getattr(mod, "linear1", None), th.nn.Linear) and \
                    isinstance(getattr(mod, "linear2", None), th.nn.Linear):
                act = getattr(mod, "activation", F.relu)
                act = getattr(F, act) if isinstance(act, str) else act
                hooks.append(mod.linear1.register_forward_hook(
                    lambda m, i, o, n=name + ".ffn", f=act: acc(n, f(o))))
        taps: Dict[str, List[np.ndarray]] = {"latent_pi": [], "vf": [], "pi_features": [], "value_pooled": []}
        try:
            with th.no_grad():
                for i in range(0, len(b["obs"]), 256):
                    lat, vf, pi, vp = penultimate(pol, obs_dict(pol, b["obs"][i:i + 256], b["mask"][i:i + 256]))
                    for k, v in zip(taps, (lat, vf, pi, vp)):
                        taps[k].append(v.float().numpy())
        finally:
            for h in hooks:
                h.remove()
        T = {k: np.concatenate(v) for k, v in taps.items()}
        dorm = {}
        pooled = {t: [0, 0] for t in TAUS}
        tanh_rec = {}
        for name, s in sums.items():
            m = (s / cnt[name]).numpy()
            mu = float(m.mean())
            score = m / mu if mu > 0 else np.zeros_like(m)
            rec = {"units": int(m.size), "mean_abs": round(mu, 6)}
            for t in TAUS:
                k = int((score <= t).sum())
                rec[f"d{t}"] = round(k / m.size, 5)
                if name in sat:
                    continue
                pooled[t][0] += k
                pooled[t][1] += m.size
            if name in sat:
                frac = (sat[name] / cnt[name]).numpy()
                rec["saturated_units"] = round(float((frac >= 0.99).mean()), 5)
                rec["mean_sat_frac"] = round(float(frac.mean()), 5)
                tanh_rec[name] = rec
            else:
                dorm[name] = rec
        ranks = {}
        for k, Z in T.items():
            r = effective_rank(Z)
            ranks[k] = {"srank_kumar": srank(Z), "pr": round(float(r["pr"]), 3), "n99": int(r["n99"]),
                        "dim": int(Z.shape[1]), "feat_norm": round(float(np.linalg.norm(Z, axis=1).mean()), 4)}
        res[bname] = {"rows": int(len(b["obs"])),
                      "dormant_pooled": {f"d{t}": round(pooled[t][0] / max(pooled[t][1], 1), 5) for t in TAUS},
                      "dormant_units": pooled[TAUS[0]][1], "tanh": tanh_rec, "ranks": ranks, "dormant_layers": dorm}
        if bname == "unfam" and feats_out is not None:
            feats_out.mkdir(parents=True, exist_ok=True)
            np.savez(feats_out / f"{label}.npz", latent_pi=T["latent_pi"], vf=T["vf"])
    wn: Dict[str, float] = {}
    for name, p in pol.named_parameters():
        g = ".".join(name.split(".")[:2])
        wn[g] = wn.get(g, 0.0) + float(p.detach().float().pow(2).sum())
    res["weight_norm_total_excl_ridealong"] = round(sum(v for k, v in wn.items()
                                                        if not k.startswith("ridealong")) ** 0.5, 4)
    res["weight_norm_extractor"] = round(sum(v for k, v in wn.items()
                                             if k.startswith("features_extractor")) ** 0.5, 4)
    res["weight_norm_policy_mlp"] = round(sum(v for k, v in wn.items() if k.startswith("mlp_extractor")) ** 0.5, 4)
    res["grad"] = grad_stats(pol, bank["unfam"])
    return res


def grad_stats(pol, u) -> dict:
    """Per-example grad of log pi(a|s) (the bank's sampled action) on N_GRAD_ROWS fixed held-out rows, w.r.t. the
    extractor + policy MLP + pointer head."""
    import torch as th

    idx = np.flatnonzero(u["held_game"][u["game"]])
    idx = np.sort(np.random.default_rng(5).choice(idx, N_GRAD_ROWS, replace=False))
    params = [p for n, p in pol.named_parameters() if not n.startswith("ridealong")]
    G = []
    for i in idx:
        ob = obs_dict(pol, u["obs"][i:i + 1], u["mask"][i:i + 1])
        pol.zero_grad(set_to_none=True)
        pi, _ = pol.extract_features(ob)
        lat = pol.mlp_extractor.forward_actor(pi)
        logp = pol.masked_logp(lat, ob["action_mask"])
        logp[0, int(u["action"][i])].backward()
        G.append(th.cat([(p.grad if p.grad is not None else th.zeros_like(p)).reshape(-1) for p in params]))
    pol.zero_grad(set_to_none=True)
    M = th.stack(G).double()
    norms = M.norm(dim=1)
    gram = M @ M.T
    cos = gram / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
    off = cos[~th.eye(len(idx), dtype=th.bool)]
    ev = th.linalg.eigvalsh(gram).clamp_min(0)
    p = ev / ev.sum()
    return {"rows": int(len(idx)), "mean_norm": round(float(norms.mean()), 5),
            "median_norm": round(float(norms.median()), 5),
            "mean_offdiag_cos": round(float(off.mean()), 5), "mean_abs_offdiag_cos": round(float(off.abs().mean()), 5),
            "gram_pr": round(float(1.0 / (p ** 2).sum()), 3)}


def cmd_structure(a) -> int:
    from main.policy_spectrum.reader import inference_globals

    bank = load_bank(Path(a.bank))
    outp = Path(a.out)
    out = json.loads(outp.read_text()) if outp.exists() else {}
    with inference_globals(THREADS):
        for spec in a.net:
            label, path = spec.split("=", 1)
            if label in out:
                print("skip", label, flush=True)
                continue
            t0 = time.time()
            pol = load_net(path)
            r = structure_one(pol, bank, Path(a.feats), label)
            r["spec"] = path
            out[label] = r
            outp.write_text(json.dumps(out, indent=1))
            f = r["unfam"]
            print(label, f"{time.time() - t0:.0f}s dormant(unfam) {f['dormant_pooled']} srank lat/vf "
                  f"{f['ranks']['latent_pi']['srank_kumar']}/{f['ranks']['vf']['srank_kumar']} wn "
                  f"{r['weight_norm_total_excl_ridealong']} grad {r['grad']}", flush=True)
            del pol
    return 0


# ------------------------------------------------------------------------------------------------ heads (frozen)
def auc(y: np.ndarray, s: np.ndarray) -> float:
    from agents.model.capacity_probes import auc_score

    return float(auc_score(y, s))


def make_loss(kind: str):
    """(head width from Y's width, loss(pred, Y)). ``ce``: Y = [one-hot P*'s greedy action (11) ++ legal mask (11)];
    the loss is the cross-entropy of the masked softmax over the legal actions (an illegal logit gets -1e9)."""
    import torch as th

    if kind == "bce":
        return (lambda w: w), th.nn.BCEWithLogitsLoss()
    if kind == "mse":
        return (lambda w: w), th.nn.MSELoss()

    def ce(pred, Y):
        logits = pred + (Y[:, 11:] - 1.0) * 1e9
        return -(th.log_softmax(logits, dim=-1) * Y[:, :11]).sum(-1).mean()

    return (lambda w: w // 2), ce


def train_head(X: np.ndarray, Y: np.ndarray, tr: np.ndarray, kind: str, seed: int, steps: int, lr: float,
               batch: int, checkpoints=(30, 100, 300, 1000)) -> dict:
    import torch as th

    Xt = th.as_tensor(X, dtype=th.float32)
    Yt = th.as_tensor(Y, dtype=th.float32)
    width, lossf = make_loss(kind)
    head = th.nn.Linear(X.shape[1], width(Y.shape[1]))
    with th.no_grad():  # ZERO init: every net starts at the same loss (MSE 1 on a standardised target, BCE ln 2)
        head.weight.zero_()
        head.bias.zero_()
    opt = th.optim.Adam(head.parameters(), lr=lr)
    tri = np.flatnonzero(tr)
    tei = np.flatnonzero(~tr)
    rng = np.random.default_rng(seed)
    curve = {}

    def evaluate(ix):
        with th.no_grad():
            return float(lossf(head(Xt[ix]), Yt[ix]))

    for s in range(1, steps + 1):
        b = rng.choice(tri, batch, replace=False)
        opt.zero_grad()
        lossf(head(Xt[b]), Yt[b]).backward()
        opt.step()
        if s in checkpoints:
            curve[s] = {"train": round(evaluate(tri), 5), "held": round(evaluate(tei), 5)}
    rec: Dict[str, Any] = {"curve": curve}
    if kind == "bce":
        with th.no_grad():
            sc = head(Xt[tei]).numpy()
        rec["held_auc"] = round(float(np.mean([auc(Y[tei, j], sc[:, j]) for j in range(Y.shape[1])
                                               if 0 < Y[tei, j].mean() < 1])), 5)
    return rec


def cmd_heads(a) -> int:
    from utils.torch_state_guard import torch_globals

    bank = load_bank(Path(a.bank))
    u = bank["unfam"]
    lab = labels(u)
    tr = ~u["held_game"][u["game"]]
    tnet = np.load(Path(a.bank) / "t_net.npy")
    tlin = random_obs_targets(u)
    tasks = {"win": (lab["win"], "bce"), "species": (lab["species"], "bce"), "t_net": (tnet, "mse"),
             "t_lin": (tlin, "mse"), "pstar": (pstar_target(Path(a.bank), u), "ce")}
    out: Dict[str, Any] = {"species": lab["species_names"].tolist(), "train_rows": int(tr.sum()),
                           "held_rows": int((~tr).sum()), "reads": {}}
    with torch_globals(num_threads=THREADS):
        for f in sorted(Path(a.feats).glob("*.npz")):
            z = np.load(f)
            X = np.concatenate([z["latent_pi"], z["vf"]], 1)
            rec = {}
            for tname, (Y, kind) in tasks.items():
                rec[tname] = [train_head(X, Y, tr, kind, seed, a.steps, a.lr, a.batch) for seed in (0, 1)]
            out["reads"][f.stem] = rec
            print(f.stem, {t: (r[0]["curve"][a.steps]["held"], r[1]["curve"][a.steps]["held"])
                           for t, r in rec.items()}, flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def cmd_tnet(a) -> int:
    from main.policy_spectrum.reader import inference_globals

    bank = load_bank(Path(a.bank))
    with inference_globals(THREADS):
        pol = load_net(a.spec)
        np.save(Path(a.bank) / "t_net.npy", random_net_targets(pol, bank["unfam"]))
    return 0


# ------------------------------------------------------------------------------------------------ refit (unfrozen)
def refit_one(pol, u, Y: np.ndarray, kind: str, seed: int, steps: int, lr: float, batch: int,
              evals=(0, 25, 50, 100, 200, 300, 400)) -> dict:
    """The trunk UNFROZEN: Adam(lr) on the extractor + policy MLP + a fresh linear head over [latent_pi ++ vf]."""
    import torch as th

    th.manual_seed(seed)
    trunk = [p for n, p in pol.named_parameters() if n.startswith(("features_extractor", "mlp_extractor"))]
    for n, p in pol.named_parameters():
        p.requires_grad_(n.startswith(("features_extractor", "mlp_extractor")))
    width, lossf = make_loss(kind)
    head = th.nn.Linear(640, width(Y.shape[1]))
    with th.no_grad():  # ZERO init: every net starts at the same held-out loss; the trunk moves from step 2
        head.weight.zero_()
        head.bias.zero_()
    opt = th.optim.Adam(trunk + list(head.parameters()), lr=lr)
    rowtr = np.flatnonzero(~u["held_game"][u["game"]])
    rowte = np.flatnonzero(u["held_game"][u["game"]])
    rowte = np.sort(np.random.default_rng(17).choice(rowte, min(768, len(rowte)), replace=False))
    rng = np.random.default_rng(seed)
    Yt = th.as_tensor(Y)

    def fwd(ix):
        lat, vf, *_ = penultimate(pol, obs_dict(pol, u["obs"][ix], u["mask"][ix]))
        return head(th.cat([lat, vf], 1).float())

    def held():
        with th.no_grad():
            return float(np.mean([float(lossf(fwd(rowte[i:i + 256]), Yt[rowte[i:i + 256]]))
                                  for i in range(0, len(rowte), 256)]))

    curve = {}
    t0 = time.time()
    for s in range(0, steps + 1):
        if s in evals:
            curve[s] = round(held(), 5)
        if s == steps:
            break
        b = np.sort(rng.choice(rowtr, batch, replace=False))
        opt.zero_grad(set_to_none=True)
        loss = lossf(fwd(b), Yt[b])
        loss.backward()
        opt.step()
    return {"held_curve": curve, "seconds": round(time.time() - t0, 1)}


def cmd_refit(a) -> int:
    from main.policy_spectrum.reader import inference_globals

    bank = load_bank(Path(a.bank))
    u = bank["unfam"]
    targets = {"t_net": (np.load(Path(a.bank) / "t_net.npy"), "mse"), "t_lin": (random_obs_targets(u), "mse"),
               "pstar": (pstar_target(Path(a.bank), u), "ce"),
               "win": (labels(u)["win"], "bce")}
    outp = Path(a.out)
    out = json.loads(outp.read_text()) if outp.exists() else {}
    with inference_globals(THREADS):
        for spec in a.net:
            label, path = spec.split("=", 1)
            for t in a.target:
                for seed in a.seed:
                    key = f"{label}|{t}|{seed}"
                    if key in out:
                        continue
                    pol = load_net(path)
                    Y, kind = targets[t]
                    r = refit_one(pol, u, Y, kind, seed, a.steps, a.lr, a.batch)
                    r.update({"net": label, "target": t, "seed": seed, "lr": a.lr, "batch": a.batch, "spec": path})
                    out[key] = r
                    outp.write_text(json.dumps(out, indent=1))
                    print(key, r["held_curve"], r["seconds"], flush=True)
                    del pol
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--out", required=True)
    b.add_argument("--fixture-games", type=int, default=120)
    b.add_argument("--unfam-games", type=int, default=240)
    b.add_argument("--unfam-teams", type=int, default=400)
    s = sub.add_parser("structure")
    s.add_argument("--bank", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--feats", required=True)
    s.add_argument("--net", action="append", required=True)
    t = sub.add_parser("tnet")
    t.add_argument("--bank", required=True)
    t.add_argument("--spec", required=True)
    h = sub.add_parser("heads")
    h.add_argument("--bank", required=True)
    h.add_argument("--feats", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--steps", type=int, default=1000)
    h.add_argument("--lr", type=float, default=1e-3)
    h.add_argument("--batch", type=int, default=256)
    r = sub.add_parser("refit")
    r.add_argument("--bank", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--net", action="append", required=True)
    r.add_argument("--target", action="append", required=True)
    r.add_argument("--seed", action="append", type=int, default=None)
    r.add_argument("--steps", type=int, default=400)
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--batch", type=int, default=64)
    a = ap.parse_args(argv)
    if a.cmd == "refit" and not a.seed:
        a.seed = [0]
    return {"bank": cmd_bank, "structure": cmd_structure, "heads": cmd_heads, "refit": cmd_refit,
            "tnet": cmd_tnet}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
