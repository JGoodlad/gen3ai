"""Architecture diagnostic, the model-side reads (2026-11-09): census, saturation, gradients, tower ablations.

    model_read.py <out_dir>

P* (rb_x26_s2601 @ 280,000,170) read by ITS OWN code: run with PYTHONPATH=<checkout at 9c0334890>/src, CPU, one thread.
Writes <out_dir>/model_read.json (every number in the README) and <out_dir>/acts.npz (per-row value features, kept
out of git, for value_read.py). No training, no games; models/ and the archive are read-only."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SAT_A = 0.99                 # the plasticity probe's threshold on |tanh(x)|: |x| > 2.6467
SAT_PRE = float(np.arctanh(SAT_A))
GRAD_N = 256                 # rows per gradient batch
SEED = 20261109


# ---------------------------------------------------------------------------------------------- census
GROUPS = [   # (group, class) by parameter-name prefix, first match wins
    ("features_extractor.team_transformer.", "trunk: team_transformer (attention + FFN over the 12 mon tokens)"),
    ("features_extractor.cls_pool.", "pooling: cls_pool (three attention pools)"),
    ("features_extractor.value_entity_pool.", "pooling: value_entity_pool (critic route)"),
    ("features_extractor.hidden_opp_belief.", "pooling: hidden_opp_belief"),
    ("features_extractor.pre_proj_norm.", "post-pool flat: pre_proj_norm"),
    ("features_extractor.projection.", "post-pool flat: projection 1152->512 (ReLU)"),
    ("mlp_extractor.", "post-pool flat: actor MLP 512->512->512 (tanh)"),
    ("pointer_head.", "head: pointer (policy)"),
    ("features_extractor.win_head.", "head: win_head (critic)"),
    ("features_extractor.embeddings.", "pre-trunk: embeddings"),
    ("features_extractor.pokemon_encoder.", "pre-trunk: pokemon_encoder"),
]
BELIEF = ("belief_head", "move_belief", "spread_belief", "hp_type_belief_head", "item_belief_head", "t0_species_prior",
          "flat_intent_head", "hypothesis_builder")


def group_of(name: str) -> str:
    for pre, g in GROUPS:
        if name.startswith(pre):
            return g
    if name.startswith("ridealong."):
        return "ride-along (detached readouts; not in the 2,609,183)"
    if name.startswith("features_extractor."):
        child = name.split(".")[1]
        if child in BELIEF:
            return "head: belief / intent stack"
        return "pre-trunk / operator / tokens: other extractor modules"
    return "other"


def census(pol) -> dict:
    by_group, by_child = {}, {}
    for n, p in pol.named_parameters():
        by_group[group_of(n)] = by_group.get(group_of(n), 0) + p.numel()
        key = ".".join(n.split(".")[:2]) if n.startswith("features_extractor.") else n.split(".")[0]
        by_child[key] = by_child.get(key, 0) + p.numel()
    ride = sum(v for k, v in by_group.items() if k.startswith("ride-along"))
    total = sum(by_group.values()) - ride
    return {"total_with_ridealong": sum(by_group.values()), "ridealong": ride, "learner_total": total,
            "by_group": dict(sorted(by_group.items(), key=lambda kv: -kv[1])),
            "by_group_share": {k: v / total for k, v in sorted(by_group.items(), key=lambda kv: -kv[1])
                               if not k.startswith("ride-along")},
            "by_child": dict(sorted(by_child.items(), key=lambda kv: -kv[1]))}


# ---------------------------------------------------------------------------------------------- forward
class Taps:
    """Forward hooks on the bounded / gated layers of the policy and value paths."""

    def __init__(self, pol):
        self.pol, self.fe = pol, pol.features_extractor
        self.out = {}
        net = pol.mlp_extractor.policy_net
        self.h = [
            self.fe.projection.register_forward_hook(lambda m, i, o: self.out.__setitem__("proj_pre", o.detach())),
            net[0].register_forward_hook(lambda m, i, o: self.out.__setitem__("tanh1_pre", o.detach())),
            net[2].register_forward_hook(lambda m, i, o: self.out.__setitem__("tanh2_pre", o.detach())),
            self.fe.win_head.net[1].register_forward_hook(lambda m, i, o: self.out.__setitem__("win_hidden_pre", o.detach())),
        ]
        ph = pol.pointer_head
        self._orig = ph.forward

        def tapped(ctx_vec, move_tokens_req, move_valid, team_tokens, move_cells, switch_cells):
            c = ph.ctx_proj(ctx_vec)
            m_in = th.cat([move_tokens_req, move_cells], dim=-1)
            self.out["ptr_ctx_pre"] = c.detach()
            self.out["ptr_move_tok_pre"] = ph.move_proj(m_in).detach()        # the token half, before + c
            if ph.switch_extra_proj is None:
                s_tok = ph.switch_proj(th.cat([team_tokens, switch_cells], dim=-1))
            else:
                b = ph._switch_base_dim
                s_tok = (ph.switch_proj(th.cat([team_tokens, switch_cells[..., :b]], dim=-1))
                         + ph.switch_extra_proj(switch_cells[..., b:]))
            self.out["ptr_switch_tok_pre"] = s_tok.detach()
            return self._orig(ctx_vec, move_tokens_req, move_valid, team_tokens, move_cells, switch_cells)
        ph.forward = tapped

    def close(self):
        for h in self.h:
            h.remove()
        self.pol.pointer_head.forward = self._orig


def masked_logp(logits, mask):
    return th.log_softmax(logits + (th.as_tensor(mask, dtype=th.float32) - 1.0) * 1e9, dim=-1)


def forward_all(model, obs, mask, batch=256):
    pol = model.policy
    fe = pol.features_extractor
    taps = Taps(pol)
    keep = {k: [] for k in ("proj_pre", "tanh1_pre", "tanh2_pre", "win_hidden_pre", "ptr_ctx_pre", "ptr_move_tok_pre",
                            "ptr_switch_tok_pre", "value_pooled", "win_logit", "logp")}
    with th.no_grad():
        for b0 in range(0, len(obs), batch):
            sl = slice(b0, b0 + batch)
            logits = pol.get_distribution(C.obs_dict(model, obs[sl], mask[sl])).distribution.logits
            for k in ("proj_pre", "tanh1_pre", "tanh2_pre", "win_hidden_pre", "ptr_ctx_pre", "ptr_move_tok_pre",
                      "ptr_switch_tok_pre"):
                keep[k].append(taps.out[k].numpy().copy())
            keep["value_pooled"].append(fe.stash.value_pooled.detach().numpy().copy())
            keep["win_logit"].append(fe.last_win_prob_logits.reshape(-1).numpy().copy())
            keep["logp"].append(masked_logp(logits, mask[sl]).numpy().copy())
    taps.close()
    return {k: np.concatenate(v) for k, v in keep.items()}


# ---------------------------------------------------------------------------------------------- saturation
def tanh_stats(pre: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """pre [N, U] (or [N, A, U] with valid [N, A]): activation-level and unit-level saturation of tanh(pre)."""
    if pre.ndim == 3:
        pre = pre[valid] if valid is not None else pre.reshape(-1, pre.shape[-1])
    a = np.tanh(pre.astype(np.float64))
    sat_pos, sat_neg = a > SAT_A, a < -SAT_A
    sat = sat_pos | sat_neg
    up, un = sat_pos.mean(0), sat_neg.mean(0)
    stuck = (up >= 0.99) | (un >= 0.99)                    # one sign on >= 99 % of rows: a constant, no gradient
    always = (up + un) >= 0.99                             # the plasticity probe's "saturated unit" (either sign)
    switch = always & ~stuck                               # saturated on >= 99 % of rows but with BOTH signs: a bit
    minor = np.minimum(up, un)
    return {"n_rows": int(pre.shape[0]), "units": int(pre.shape[1]),
            "frac_act_abs_gt_0.99": float(sat.mean()),
            "frac_act_abs_gt_0.964_(|pre|>2)": float((np.abs(pre) > 2).mean()),
            "frac_act_abs_gt_0.9999_(|pre|>5)": float((np.abs(pre) > 5).mean()),
            "median_abs_pre": float(np.median(np.abs(pre))),
            "mean_tanh_prime": float((1 - a * a).mean()),
            "median_tanh_prime": float(np.median(1 - a * a)),
            "units_saturated_on_ge99pct_rows": float(always.mean()),
            "units_stuck_one_sign_ge99pct": float(stuck.mean()),
            "units_two_sided_switch": float(switch.mean()),
            "units_minor_sign_ge_5pct_rows": float((minor >= 0.05).mean()),
            "mean_unit_sign_entropy_bits": float(np.mean(_h2((a > 0).mean(0)))),
            "units_never_saturated_(lt1pct_rows)": float(((up + un) < 0.01).mean())}


def _h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def relu_stats(pre: np.ndarray) -> dict:
    on = pre > 0
    return {"n_rows": int(pre.shape[0]), "units": int(pre.shape[1]), "frac_act_zero": float((~on).mean()),
            "units_dead_(never_on)": float((on.mean(0) == 0).mean()),
            "units_on_lt1pct_rows": float((on.mean(0) < 0.01).mean()),
            "units_always_on": float((on.mean(0) == 1).mean())}


def pointer_stats(acts: dict, mask: np.ndarray) -> dict:
    """The pointer head's tanh(token_proj + ctx_proj): over LEGAL actions (mask cols 0-5 switch, 6-9 move)."""
    c = acts["ptr_ctx_pre"][:, None, :]
    mv = acts["ptr_move_tok_pre"] + c
    sw = acts["ptr_switch_tok_pre"] + c
    pre = np.concatenate([sw[mask[:, 0:6]], mv[mask[:, 6:10]]])
    out = tanh_stats(pre)
    # how much of the pre-activation's variance the tower's context supplies, per unit, over legal actions
    tok = np.concatenate([acts["ptr_switch_tok_pre"][mask[:, 0:6]], acts["ptr_move_tok_pre"][mask[:, 6:10]]])
    ctx = pre - tok
    out["ctx_share_of_pre_variance"] = float(ctx.var(0).sum() / pre.var(0).sum())
    out["token_share_of_pre_variance"] = float(tok.var(0).sum() / pre.var(0).sum())
    return out


# ---------------------------------------------------------------------------------------------- tower ablations
def tower_ablations(model, obs, mask, base_logp, mean_latent, batch=256) -> dict:
    """Function-preserving reads of the tanh tower: what does the policy lose if the tower is (a) BINARISED
    (tanh -> sign: a saturated tower that works as bits loses nothing), (b) replaced by its dataset MEAN output
    (the tower carries nothing row-specific)."""
    pol = model.policy
    net = pol.mlp_extractor.policy_net

    class Sign(th.nn.Module):
        def forward(self, x):
            return th.sign(x)

    def run(kind):
        saved = (net[1], net[3], pol.mlp_extractor.forward_actor)
        try:
            if kind in ("sign_both", "sign_l1"):
                net[1] = Sign()
            if kind in ("sign_both", "sign_l2"):
                net[3] = Sign()
            if kind == "mean_latent":
                ml = th.as_tensor(mean_latent)
                pol.mlp_extractor.forward_actor = lambda f: ml.expand(f.shape[0], -1)
            out = []
            with th.no_grad():
                for b0 in range(0, len(obs), batch):
                    sl = slice(b0, b0 + batch)
                    lg = pol.get_distribution(C.obs_dict(model, obs[sl], mask[sl])).distribution.logits
                    out.append(masked_logp(lg, mask[sl]).numpy())
            return np.concatenate(out)
        finally:
            net[1], net[3] = saved[0], saved[1]
            pol.mlp_extractor.forward_actor = saved[2]

    res = {}
    p0 = np.exp(base_logp)
    multi = mask.sum(1) > 1
    for kind in ("sign_l1", "sign_l2", "sign_both", "mean_latent"):
        lp = run(kind)
        p1 = np.exp(lp)
        kl = np.where(mask, p0 * (base_logp - lp), 0.0).sum(1)
        res[kind] = {"argmax_agreement": float((lp.argmax(1) == base_logp.argmax(1))[multi].mean()),
                     "mean_tv": float((0.5 * np.abs(p1 - p0).sum(1))[multi].mean()),
                     "mean_kl_nats": float(kl[multi].mean()), "n_rows_multi_legal": int(multi.sum())}
    return res


# ---------------------------------------------------------------------------------------------- gradients
class TanhSTE(th.nn.Module):
    """tanh in the forward, identity in the backward: the tower's gradient if no unit were saturated."""
    def forward(self, x):
        return x + (th.tanh(x) - x).detach()


def grad_read(model, obs, mask, y, seed) -> dict:
    """Gradient magnitudes on one batch. Policy pseudo-loss: -mean(A * log pi(a|s)), a ~ pi, A ~ N(0,1) (the shape
    of a PPO policy gradient with unit advantages); value loss: BCE(win logit, outcome)."""
    pol = model.policy
    fe = pol.features_extractor
    net = pol.mlp_extractor.policy_net
    g = th.Generator().manual_seed(seed)
    ob = C.obs_dict(model, obs, mask)
    with th.no_grad():
        lp0 = masked_logp(pol.get_distribution(ob).distribution.logits, mask)
    act = th.multinomial(lp0.exp(), 1, generator=g)
    adv = th.randn(len(obs), generator=g)
    yt = th.as_tensor(y, dtype=th.float32)
    params = [(n, p) for n, p in pol.named_parameters() if not n.startswith("ridealong.")]

    def run(kind, loss_kind):
        saved = (net[1], net[3])
        hooks, cap = [], {}
        try:
            if kind == "ste":
                net[1], net[3] = TanhSTE(), TanhSTE()
            if kind == "tower_blocked":
                hooks.append(net[0].register_forward_pre_hook(lambda m, i: (i[0].detach(),)))
            if kind == "real":
                def tap(name):
                    def f(m, gi, go):
                        cap[name] = (go[0].detach().norm(dim=1), gi[0].detach().norm(dim=1))
                    return f
                hooks.append(net[1].register_full_backward_hook(tap("tanh1")))
                hooks.append(net[3].register_full_backward_hook(tap("tanh2")))
            for _, p in params:
                p.grad = None
            feats = {}
            hooks.append(fe.projection.register_forward_hook(lambda m, i, o: (o.retain_grad(), feats.__setitem__("proj", o))[1] and None))
            logits = pol.get_distribution(ob).distribution.logits
            if loss_kind == "policy":
                loss = -(adv * masked_logp(logits, mask).gather(1, act).squeeze(1)).mean()
            else:
                loss = th.nn.functional.binary_cross_entropy_with_logits(fe.last_win_prob_logits.reshape(-1), yt)
            loss.backward()
            by = {}
            for n, p in params:
                gname = group_of(n)
                d = by.setdefault(gname, {"g2": 0.0, "w2": 0.0, "n": 0})
                if p.grad is not None:
                    d["g2"] += float((p.grad.double() ** 2).sum())
                d["w2"] += float((p.detach().double() ** 2).sum())
                d["n"] += p.numel()
            out = {k: {"grad_norm": d["g2"] ** 0.5, "grad_rms_per_param": (d["g2"] / d["n"]) ** 0.5,
                       "grad_norm_over_weight_norm": d["g2"] ** 0.5 / max(d["w2"] ** 0.5, 1e-30), "params": d["n"]}
                   for k, d in by.items()}
            # per-layer rows for the tower
            for nm in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.2.weight",
                       "features_extractor.projection.weight", "pointer_head.ctx_proj.weight"):
                p = dict(params)[nm]
                gr = p.grad if p.grad is not None else th.zeros_like(p)  # None: the layer is not in this loss's graph
                out[nm] = {"grad_norm": float(gr.norm()), "grad_rms_per_param": float((gr ** 2).mean().sqrt()),
                           "grad_norm_over_weight_norm": float(gr.norm() / p.detach().norm()), "params": p.numel(),
                           "in_graph": p.grad is not None}
            extra = {"proj_out_grad_norm": float(feats["proj"].grad.norm()) if feats["proj"].grad is not None else 0.0}
            if kind == "real" and loss_kind == "policy":
                for k, (go, gi) in cap.items():
                    r = (gi / go.clamp(min=1e-30)).numpy()
                    extra[k + "_backward_gain_(norm_in/norm_out)"] = {
                        "mean": float(r.mean()), "median": float(np.median(r)), "p10": float(np.quantile(r, 0.1)),
                        "p90": float(np.quantile(r, 0.9))}
                # per-unit gradient on layer-1's incoming weight rows vs the unit's mean tanh'
                w = dict(params)["mlp_extractor.policy_net.0.weight"]
                extra["tanh1_unit_row_grad_norm"] = w.grad.norm(dim=1).numpy().tolist()
            return out, extra
        finally:
            net[1], net[3] = saved
            for h in hooks:
                h.remove()

    res = {}
    for loss_kind, kinds in (("policy", ("real", "ste", "tower_blocked")), ("value", ("real",))):
        for kind in kinds:
            groups, extra = run(kind, loss_kind)
            res[f"{loss_kind}:{kind}"] = {"groups": groups, **extra}
    for _, p in params:
        p.grad = None
    return res


# ---------------------------------------------------------------------------------------------- main
def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    th.manual_seed(SEED)
    model = C.load_model()
    pol = model.policy
    res = {"checkpoint": C.PSTAR, "code": os.path.dirname(os.path.dirname(th.__file__)) and sys.modules["agents"].__file__,
           "sat_threshold_abs_a": SAT_A, "sat_threshold_abs_pre": SAT_PRE, "census": census(pol),
           "win_head": str(pol.features_extractor.win_head), "actor": str(pol.mlp_extractor.policy_net),
           "projection": str(pol.features_extractor.projection)}
    print("census", res["census"]["learner_total"], flush=True)

    # row sets
    sets = {}
    for p in C.PANELS:
        d = C.archgen_rows(p)
        if p == "home72":
            sets["selfplay:home72 (on-pool, both sides)"] = (d["obs"], d["mask"], d["y"], d["game"], d["side"], d["turn"])
        else:
            for s, lab in ((0, "pilots U (off-pool own team)"), (1, "faces U (home team vs off-pool)")):
                m = d["side"] == s
                sets[f"selfplay:{p} {lab}"] = (d["obs"][m], d["mask"][m], d["y"][m], d["game"][m], d["side"][m], d["turn"][m])
    obs, mask, src = C.tie_rate_rows()
    for pre, lab in (("eval", "tie-rate: eval traces (rb_x60t_s3502 vs bots / sentinels, pool teams)"),
                     ("rollout", "tie-rate: rollout scans (rb_x60t_s3502 training rows)"),
                     ("fixture", "tie-rate: compile-parity fixture")):
        m = np.char.startswith(src, pre)
        sets[lab] = (obs[m], mask[m], None, None, None, None)
    res["row_sets"] = {k: len(v[0]) for k, v in sets.items()}

    acts, sat = {}, {}
    for name, (o, m, *_r) in sets.items():
        a = forward_all(model, o, m)
        acts[name] = a
        sat[name] = {"actor tanh-1": tanh_stats(a["tanh1_pre"]), "actor tanh-2": tanh_stats(a["tanh2_pre"]),
                     "pointer tanh (legal actions)": pointer_stats(a, m),
                     "projection ReLU": relu_stats(a["proj_pre"]), "win_head hidden ReLU": relu_stats(a["win_hidden_pre"]),
                     "V (sigmoid)": {"frac_V_lt_0.01_or_gt_0.99": float((np.abs(a["win_logit"]) > np.log(99)).mean()),
                                     "mean_V": float((1 / (1 + np.exp(-a["win_logit"]))).mean())}}
        print("forward", name, len(o), round(time.time() - t0), "s", flush=True)
    res["saturation"] = sat

    # pooled on-pool vs off-pool
    def pool(names, key):
        return np.concatenate([acts[n][key] for n in names])
    on = [n for n in sets if n.startswith("selfplay:home72")]
    off_pilot = [n for n in sets if "pilots U" in n]
    off_face = [n for n in sets if "faces U" in n]
    res["saturation_pooled"] = {}
    for lab, names in (("on-pool (home72 self-play)", on), ("off-pool: piloting U", off_pilot),
                       ("home team facing U", off_face)):
        res["saturation_pooled"][lab] = {"actor tanh-1": tanh_stats(pool(names, "tanh1_pre")),
                                         "actor tanh-2": tanh_stats(pool(names, "tanh2_pre")),
                                         "projection ReLU": relu_stats(pool(names, "proj_pre")),
                                         "win_head hidden ReLU": relu_stats(pool(names, "win_hidden_pre"))}
    # unit-level agreement of the saturated set on-pool vs off-pool
    a_on, a_off = np.tanh(pool(on, "tanh1_pre")), np.tanh(pool(off_pilot, "tanh1_pre"))
    s_on, s_off = (np.abs(a_on) > SAT_A).mean(0), (np.abs(a_off) > SAT_A).mean(0)
    res["tanh1_unit_sat_rate_corr_on_vs_off"] = float(np.corrcoef(s_on, s_off)[0, 1])
    res["tanh1_unit_sat_rate"] = {"on": s_on.tolist(), "off_pilot": s_off.tolist(),
                                  "mean_tanh_prime_on": (1 - a_on ** 2).mean(0).tolist()}

    # per-row value features for value_read.py (self-play rows only: they carry outcomes)
    keep = {}
    for i, (name, (o, m, y, game, side, turn)) in enumerate(sets.items()):
        if y is None:
            continue
        a = acts[name]
        keep[f"{i}:name"] = np.array(name)
        for k, v in (("y", y), ("game", game), ("side", side), ("turn", turn), ("win_logit", a["win_logit"]),
                     ("value_pooled", a["value_pooled"]), ("latent_pi", np.tanh(a["tanh2_pre"]).astype(np.float32)),
                     ("proj", np.maximum(a["proj_pre"], 0))):
            keep[f"{i}:{k}"] = v
    np.savez_compressed(os.path.join(out_dir, "acts.npz"), **keep)
    # tower ablations on home72 and on piloting-U rows
    res["tower_ablation"] = {}
    all_latent = np.tanh(np.concatenate([acts[n]["tanh2_pre"] for n in sets]))
    mean_latent = all_latent.mean(0, keepdims=True).astype(np.float32)
    for lab, names in (("on-pool (home72 self-play)", on), ("off-pool: piloting U", off_pilot)):
        o = np.concatenate([sets[n][0] for n in names]); m = np.concatenate([sets[n][1] for n in names])
        res["tower_ablation"][lab] = tower_ablations(model, o, m, pool(names, "logp"), mean_latent)
        print("ablation", lab, round(time.time() - t0), "s", flush=True)

    # gradients: one batch on-pool, one off-pool (piloting U), fixed draw
    rng = np.random.default_rng(SEED)
    res["gradients"] = {}
    for lab, names in (("on-pool (home72 self-play)", on), ("off-pool: piloting U", off_pilot)):
        o = np.concatenate([sets[n][0] for n in names]); m = np.concatenate([sets[n][1] for n in names])
        y = np.concatenate([sets[n][2] for n in names])
        multi = np.flatnonzero(m.sum(1) > 1)
        idx = np.sort(rng.choice(multi, GRAD_N, replace=False))
        res["gradients"][lab] = grad_read(model, o[idx], m[idx], y[idx], SEED)
        print("grad", lab, round(time.time() - t0), "s", flush=True)

    res["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "model_read.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("done", res["wall_s"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
