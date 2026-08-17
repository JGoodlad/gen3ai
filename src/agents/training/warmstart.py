"""Disagreement-gated ensemble CONSENSUS warm-start (gen3_exploiter_consensus_warmstart_v1).

Produce a competent, archetype-NEUTRAL warm-start checkpoint for a NEW EXPLOITER by behavior-cloning the
CONSENSUS of N mature teachers (the existing exploiters) into a competent student init (the generalist),
in FUNCTION space (teacher OUTPUTS — basin-free, unlike weight-averaging, which FAILED: from-scratch
exploiters live in different loss basins → the average collapses). The target is SHARPENED where the
teachers AGREE (universal decisions the new exploiter should just inherit) and FLATTENED where they
DISAGREE (archetype-specific forks → left high-entropy so the new exploiter specializes FREELY, unbiased):

    consensus(s) = mean of the N teachers' masked action distributions      # the "aligned parts"
    d(s)         = mean pairwise Jensen-Shannon disagreement                # how much they diverge
    g(s)         = quantile-normalized d(s) ∈ [0, 1]                        # the gate
    T(s)         = 1 + (tmax − 1)·g(s)  ∈ [1, tmax]                         # temperature rises with d
    target(s)    = softmax( log consensus(s) / T(s) )   over legal actions  # sharp on agree, flat on fork

BC also carries a KL anchor toward the student init's OWN distribution (`anchor_coef`) so the warm start
RETAINS the generalist's competence instead of drifting toward the (weaker, narrower) consensus.

**EXPLOITER-ONLY (by design).** This SEEDS a new model with consensus competence + freedom to diverge.
It must NOT be used for GENERALIST training, whose objective is the OPPOSITE — absorb the DIVERGENT
per-team specializations (that is `--distill-teacher`, one teacher per team-masked state). Distilling
the consensus into the generalist would SHARPEN agreement and BLUR divergence — erasing exactly the
specialization the generalist is trying to learn (and the generalist already ≈ the consensus, so it is
near-circular). The CLI guards this: `--warmstart-consensus` requires `--exploiter` and is rejected with
`--self-play`.

Standalone: `python -m agents.training.warmstart --student <run_dir> --teachers <run_dir,...> --out <dir>`.
Ported/generalized from the offline prototype (validated: weight-average failed the heuristic bar 0.075;
the function-space gated distill holds the generalist's ~0.925 competence while raising fork entropy).
"""
from __future__ import annotations

import argparse
import os
import shutil
from typing import Optional

import numpy as np
import torch as th

_JS_EPS = 1e-12
_CHUNK = 512
_NEG_INF = -1e9


# ------------------------------------------------------------------ pure "aligned-parts" math (testable)

def pairwise_js_disagreement(teacher_probs: np.ndarray) -> np.ndarray:
    """``teacher_probs`` [T, N, A] (T teacher action-distributions over A actions for N states) →
    ``[N]`` per-state MEAN pairwise Jensen-Shannon divergence — how much the teachers DIVERGE at each
    state (0 = identical, larger = they fork). A single teacher (T<2) → all-zeros (no disagreement)."""
    T = teacher_probs.shape[0]

    def _kl(a, b):  # row-wise KL(a‖b) over the action axis
        return (a * (np.log(a + _JS_EPS) - np.log(b + _JS_EPS))).sum(-1)

    acc = None
    n_pairs = 0
    for i in range(T):
        for j in range(i + 1, T):
            p, q = teacher_probs[i], teacher_probs[j]
            m = 0.5 * (p + q)
            js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
            acc = js if acc is None else acc + js
            n_pairs += 1
    if n_pairs == 0:
        return np.zeros(teacher_probs.shape[1], dtype=np.float64)
    return acc / n_pairs


def build_consensus_target(teacher_probs: np.ndarray, mask: np.ndarray, tmax: float = 3.0,
                           d_lo_q: float = 0.2, d_hi_q: float = 0.9):
    """Build the disagreement-gated consensus BC target — the CORE of the warm start (pure numpy).

    ``teacher_probs`` [T, N, A] = each teacher's masked action distribution (sum-1 over LEGAL actions);
    ``mask`` [N, A] = the legal-action mask (1 legal / 0 illegal). Returns
    ``(target [N,A] float32, gate [N] float32, disagreement [N] float32)`` where ``target`` is
    ``softmax(log consensus / T)`` over legal actions with ``T = 1 + (tmax−1)·gate`` — SHARP where the
    teachers agree (gate≈0 ⇒ T≈1 ⇒ ≈ consensus) and FLAT where they disagree (gate≈1 ⇒ T≈tmax ⇒ toward
    uniform-over-legal). ``gate`` is ``d`` linearly normalized between its ``d_lo_q``/``d_hi_q`` quantiles
    and clipped to [0,1], so the gating is robust to the raw JS scale. Illegal actions get 0 mass."""
    if tmax < 1.0:
        raise ValueError(f"tmax must be >= 1 (1 = plain consensus, no gating); got {tmax}")
    consensus = teacher_probs.mean(0)                                    # [N,A] the aligned mean
    d = pairwise_js_disagreement(teacher_probs)                          # [N]
    d_lo, d_hi = np.quantile(d, d_lo_q), np.quantile(d, d_hi_q)
    gate = np.clip((d - d_lo) / (d_hi - d_lo + 1e-9), 0.0, 1.0)          # [N] 0..1
    temp = 1.0 + (tmax - 1.0) * gate                                     # [N] in [1, tmax]
    logc = np.log(consensus + _JS_EPS) / temp[:, None]                   # temper the log-consensus
    logc = np.where(mask > 0, logc, _NEG_INF)                            # keep only legal actions
    t = np.exp(logc - logc.max(-1, keepdims=True))
    target = (t / t.sum(-1, keepdims=True)).astype(np.float32)           # [N,A] renormalized over legal
    return target, gate.astype(np.float32), d.astype(np.float32)


def action_entropy(probs: np.ndarray) -> np.ndarray:
    """Row-wise entropy (nats) of an action distribution — the gate diagnostic (rises on disagreement)."""
    return -(probs * np.log(probs + _JS_EPS)).sum(-1)


# ------------------------------------------------------------------ model I/O + collection (needs bridge)

def masked_action_probs(model, obs: np.ndarray, mask: np.ndarray,
                        chunk: int = _CHUNK) -> np.ndarray:
    """Forward a model over ``obs``/``mask`` in chunks → masked softmax action probs [len(obs), A].

    The tensor device is derived FROM THE MODEL (``next(model.policy.parameters()).device``), not
    a caller-passed string — the ai_v7_22 launch crash was exactly this drift: the student loaded
    on ``--device cuda`` while the teachers took ``_load``'s cpu default, and a shared device
    param could not be right for both. Deriving per-model makes any student/teacher device mix
    correct by construction (CPU teachers stay VRAM-free; the CUDA student forwards on the GPU)."""
    dev = next(model.policy.parameters()).device
    out = []
    for i in range(0, len(obs), chunk):
        mb = th.tensor(mask[i:i + chunk], device=dev)
        ob = {"observation": th.tensor(obs[i:i + chunk], device=dev), "action_mask": mb}
        with th.no_grad():
            logits = model.policy.get_distribution(ob).distribution.logits
        out.append(th.softmax(logits + (mb - 1.0) * 1e9, dim=-1).cpu().numpy())
    return np.concatenate(out, 0)


async def run_consensus_warmstart(student_ckpt: str, student_cfg: str,
                            teachers: "dict[str, tuple[str, str]]", out_dir: str, current_version,
                            mappings, *, battles: int = 200, bc_steps: int = 4000, lr: float = 3e-4,
                            tmax: float = 3.0, anchor_coef: float = 0.5, kl_stop: float = 0.15,
                            batch: int = 256, device: str = "cpu", cache: Optional[str] = None,
                            smoke_battles: int = 0, verbose: bool = True) -> str:
    """Build the warm-start checkpoint. Loads the student init + N frozen teachers, collects on-policy
    states (student piloting the pool vs itself), builds the disagreement-gated consensus target +
    a competence anchor, BC-fits the student, and saves ``<out_dir>/warmstart_consensus.zip`` (+ the
    student's ``model_config.json``). Returns the saved checkpoint path. Idempotent-friendly: the caller
    should skip this if the output already exists (launcher restarts). Live (needs the local bridge)."""
    from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
    from agents.inference.player import RLPlayer
    from agents.model.snapshot import load_foreign_opponent
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder
    from utils.bridge.local_battle_runner import run_local_battles

    dev = th.device(device)

    def _load(ck, cfg, dv="cpu"):
        m, _ = load_foreign_opponent(ck, current_version=current_version, device=dv, config_path=cfg)
        fe = m.policy.features_extractor
        if hasattr(fe, "_debugger"):
            fe._debugger = None
        return m

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    pool_tb = Gen3Teambuilder(TeamLoader().get_all_teams())
    student = _load(student_ckpt, student_cfg, device)
    teacher_models = {k: _load(v[0], v[1]) for k, v in teachers.items()}
    _log(f"[warmstart] student + {len(teacher_models)} teachers loaded")

    class _Coll(RLPlayer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.O, self.M = [], []

        def embed_battle(self, b):
            d = super().embed_battle(b)
            self.O.append(np.asarray(d["observation"], np.float32).copy())
            self.M.append(np.asarray(d["action_mask"], np.float32).copy())
            return d

    if cache and os.path.exists(cache):
        z = np.load(cache)
        obs, mask, target, anchor = z["obs"], z["mask"], z["target"], z["anchor"]
        _log(f"[warmstart] reused cached dataset: {obs.shape[0]} states ({cache})")
    else:
        _acct = lambda tag: AccountConfiguration(tag, "pw")
        c = _Coll(model=student, team=pool_tb, battle_format="gen3ou",
                  server_configuration=LocalhostServerConfiguration, mappings=mappings,
                  account_configuration=_acct("WsA"), stochastic=False, start_listening=False)
        o = RLPlayer(model=student, team=pool_tb, battle_format="gen3ou",
                     server_configuration=LocalhostServerConfiguration, mappings=mappings,
                     account_configuration=_acct("WsB"), stochastic=False, start_listening=False)
        await run_local_battles(c, o, battles, concurrency=2)   # awaited: main() runs in an event loop
        obs, mask = np.stack(c.O), np.stack(c.M)
        _log(f"[warmstart] collected {obs.shape[0]} states")
        tp = np.stack([masked_action_probs(teacher_models[k], obs, mask) for k in teacher_models])
        target, gate, _d = build_consensus_target(tp, mask, tmax=tmax)
        anchor = masked_action_probs(student, obs, mask).astype(np.float32)   # competence anchor
        lo, hi = gate < 0.33, gate > 0.66
        if hi.any() and lo.any():
            _log(f"[warmstart] target entropy — AGREE {action_entropy(target[lo]).mean():.3f} | "
                 f"DISAGREE {action_entropy(target[hi]).mean():.3f} nats (should rise; gate working)")
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            np.savez(cache, obs=obs, mask=mask, target=target, anchor=anchor)

    # BC: gated-consensus KL + anchor KL (retain the student's competence)
    student.policy.to(dev).train()
    opt = th.optim.Adam(student.policy.parameters(), lr=lr)
    O, M = th.tensor(obs, device=dev), th.tensor(mask, device=dev)
    Tt, An = th.tensor(target, device=dev), th.tensor(anchor, device=dev)
    n, ema, first, last = len(obs), None, None, None
    for step in range(bc_steps):
        idx = th.randint(0, n, (batch,), device=dev)
        logits = student.policy.get_distribution(
            {"observation": O[idx], "action_mask": M[idx]}).distribution.logits
        logp = th.log_softmax(logits + (M[idx] - 1.0) * 1e9, dim=-1)
        tgt, anc = Tt[idx], An[idx]
        kl_gated = (tgt * (th.log(tgt + 1e-12) - logp)).sum(-1).mean()
        kl_anchor = (anc * (th.log(anc + 1e-12) - logp)).sum(-1).mean()
        loss = kl_gated + anchor_coef * kl_anchor
        opt.zero_grad()
        loss.backward()
        opt.step()
        val = kl_gated.item()
        first = val if first is None else first
        last = val
        ema = val if ema is None else 0.95 * ema + 0.05 * val
        if verbose and step % max(1, bc_steps // 10) == 0:
            _log(f"  step {step:4d}  gated-KL={val:.4f} (ema {ema:.4f})  anchor-KL={kl_anchor.item():.4f}")
        if ema is not None and ema < kl_stop and step > 20:
            _log(f"[warmstart] early-stop @ step {step}: gated-KL ema {ema:.4f} < {kl_stop}")
            break
    _log(f"[warmstart] BC done: gated-KL {first:.3f} -> {last:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    out_ckpt = os.path.join(out_dir, "warmstart_consensus.zip")
    student.policy.to("cpu").eval()
    student.save(out_ckpt)
    shutil.copy(student_cfg, os.path.join(out_dir, "model_config.json"))
    _log(f"[warmstart] saved {out_ckpt} (+ model_config.json)")

    if smoke_battles > 0:
        from poke_env.player import SimpleHeuristicsPlayer
        _acct = lambda tag: AccountConfiguration(tag, "pw")
        p = RLPlayer(model=student, team=pool_tb, battle_format="gen3ou",
                     server_configuration=LocalhostServerConfiguration, mappings=mappings,
                     account_configuration=_acct("WsW"), stochastic=False, start_listening=False)
        h = SimpleHeuristicsPlayer(battle_format="gen3ou", team=pool_tb,
                                   server_configuration=LocalhostServerConfiguration,
                                   account_configuration=_acct("WsH"), start_listening=False)
        await run_local_battles(p, h, smoke_battles, concurrency=2)
        wr = p.n_won_battles / max(1, p.n_finished_battles)
        _log(f"[warmstart] SMOKE vs SimpleHeuristics: {p.n_won_battles}/{p.n_finished_battles} = {wr:.3f}")
    return out_ckpt


def main(argv=None):
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings
    from agents.training.fixed_opponent_pool import _resolve_zip_and_config

    ap = argparse.ArgumentParser(description="Disagreement-gated ensemble consensus warm-start (exploiter init).")
    ap.add_argument("--student", required=True, help="student init run-dir or checkpoint (the generalist)")
    ap.add_argument("--teachers", required=True, help="comma-separated teacher run-dirs (mature exploiters)")
    ap.add_argument("--out", required=True, help="output dir for warmstart_consensus.zip + model_config.json")
    ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--bc-steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--anchor-coef", type=float, default=0.5)
    ap.add_argument("--kl-stop", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--smoke-battles", type=int, default=0)
    a = ap.parse_args(argv)

    mappings = load_mappings()
    cv = current_model_version(mappings)
    s_zip, s_cfg, _ = _resolve_zip_and_config(a.student, None)
    teachers = {}
    for i, t in enumerate([x.strip() for x in a.teachers.split(",") if x.strip()]):
        z, cfg, _ = _resolve_zip_and_config(t, None)
        teachers[f"t{i + 1}"] = (z, cfg)
    import asyncio
    asyncio.run(run_consensus_warmstart(
        s_zip, s_cfg, teachers, a.out, cv, mappings, battles=a.battles,
        bc_steps=a.bc_steps, lr=a.lr, tmax=a.tmax, anchor_coef=a.anchor_coef,
        kl_stop=a.kl_stop, batch=a.batch, device=a.device, cache=a.cache,
        smoke_battles=a.smoke_battles))


if __name__ == "__main__":
    main()
