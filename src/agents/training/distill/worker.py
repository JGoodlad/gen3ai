"""Distill-one-snapshot subprocess — the manager's ``run_distill_fn`` target (like ``eval_worker``).

Non-blocking + idempotent from the manager's view: it loads the teacher snapshot, collects
on-distribution states via the server-free bridge, runs the two-stage recipe at the requested
capacity rung, gates it (fidelity + a greedy head-to-head vs the teacher), and atomically writes
``<distilled>/snapshot_<step>.distilled.pt`` + ``.distilled.json`` (the manifest the pool/manager
read). Exit 0 always; ``passed`` lives in the manifest so a crash just leaves no manifest (the
reconcile loop re-triggers).

    python -m agents.training.distill.worker --snapshot <zip> --step N --distilled-dir <dir> \
        --config '{"enc_hidden":192,...}' [--states-battles 40 --h2h-battles 200 --device cpu ...]
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np
import torch

from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.model_version import ModelVersion
from agents.model.snapshot import load_model_snapshot
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.inference.player import RLPlayer
from agents.training.distill.recipe import distill_snapshot, _masked
from agents.training.distill.student import DistilledStudent, DistilledOpponentModel, save_distilled

FMT = "gen3ou"


# ── state collection (the teacher plays; we record its decision obs+mask) ──
class _Collector(RLPlayer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.obs, self.masks = [], []

    def _predict_best_action(self, battle, **kw):
        kw["need_aux"] = True
        idx, probs, mask = super()._predict_best_action(battle, **kw)
        lp = getattr(self, "_last_prediction", None)
        if lp is not None and mask is not None and int(np.asarray(mask).sum()) > 0:
            self.obs.append(np.asarray(lp["obs"], dtype=np.float32))
            self.masks.append(np.asarray(mask, dtype=np.float32))
        return idx, probs, mask


def _player(cls, model, teams, name, **extra):
    from poke_env import AccountConfiguration, LocalhostServerConfiguration
    from utils.teambuilder import Gen3Teambuilder
    return cls(model=model, team=Gen3Teambuilder(teams), battle_format=FMT,
               server_configuration=LocalhostServerConfiguration, start_listening=False,
               account_configuration=AccountConfiguration(name, "x"),
               max_concurrent_battles=10, **extra)


async def _collect_states(teacher, teams, n_battles, tag):
    from poke_env import AccountConfiguration, LocalhostServerConfiguration
    from poke_env.player import SimpleHeuristicsPlayer
    from utils.teambuilder import Gen3Teambuilder
    from utils.bridge.local_battle_runner import run_local_battles
    c1 = _player(_Collector, teacher, teams, f"dc1{tag}", stochastic=True, temperature=1.0)
    c2 = _player(_Collector, teacher, teams, f"dc2{tag}", stochastic=True, temperature=1.0)
    heur = SimpleHeuristicsPlayer(battle_format=FMT, team=Gen3Teambuilder(teams),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"dh{tag}", "x"), max_concurrent_battles=10)
    await run_local_battles(c1, c2, n_battles, battle_format=FMT)
    await run_local_battles(c2, heur, n_battles, battle_format=FMT)
    obs = np.stack(c1.obs + c2.obs)
    mask = np.stack(c1.masks + c2.masks)
    return obs, mask


async def _head_to_head(student, teacher, teams, n, tag):
    from utils.bridge.local_battle_runner import run_local_battles
    stu = _player(RLPlayer, DistilledOpponentModel(student), teams, f"ds{tag}", stochastic=False)
    tea = _player(RLPlayer, teacher, teams, f"dt{tag}", stochastic=False)
    await run_local_battles(stu, tea, n, battle_format=FMT)
    fin = max(stu.n_finished_battles, 1)
    wr = stu.n_won_battles / fin
    ci = 1.96 * (wr * (1 - wr) / fin) ** 0.5
    return wr, ci, stu.n_finished_battles


def _time_decode(teacher, student, obs, reps=200):
    o1 = {"observation": torch.as_tensor(obs[:1]), "action_mask": torch.ones(1, 11, dtype=torch.int8)}
    o2 = {"observation": torch.as_tensor(obs[:1])}
    prev = torch.get_num_threads(); torch.set_num_threads(1); student.eval()
    with torch.no_grad():
        for _ in range(20):
            teacher.policy.get_distribution(o1); student(o2)
        t0 = time.perf_counter()
        for _ in range(reps):
            teacher.policy.get_distribution(o1)
        tt = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            student(o2)
        st = (time.perf_counter() - t0) / reps
    torch.set_num_threads(prev)
    return tt / max(st, 1e-9)


def _atomic_write(path, write_fn):
    tmp = str(path) + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--distilled-dir", required=True)
    ap.add_argument("--config", default='{"enc_hidden":192,"head_hidden":384,"tf_heads":8,"tf_ffn":512}')
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--states-battles", type=int, default=40)
    ap.add_argument("--h2h-battles", type=int, default=200)
    ap.add_argument("--s1-epochs", type=int, default=60)
    ap.add_argument("--s2-epochs", type=int, default=200)
    ap.add_argument("--min-speedup", type=float, default=2.0)
    ap.add_argument("--h2h-lo", type=float, default=0.45)
    ap.add_argument("--h2h-hi", type=float, default=0.55)
    ap.add_argument("--ent-lo", type=float, default=0.9)
    ap.add_argument("--ent-hi", type=float, default=1.1)
    args = ap.parse_args(argv)
    config = json.loads(args.config)
    tag = f"{args.step % 100000}"

    mappings = load_mappings()
    enc = Gen3ObservationEncoder(mappings)
    ek = enc.get_features_extractor_kwargs()
    layout = ek["layout"]
    version = ModelVersion.from_layout_and_policy_kwargs(
        layout, {"features_extractor_class": Gen3FeaturesExtractor,
                 "features_extractor_kwargs": ek, "net_arch": NET_ARCH})
    teacher = load_model_snapshot(args.snapshot, env=None, current_version=version, device=args.device)
    teacher.policy.set_training_mode(False)

    from utils.team_loader import TeamLoader
    teams = TeamLoader().get_sample_teams()

    print(f"[distill] step {args.step}: collecting states ({args.states_battles} battles)...", flush=True)
    obs, mask = asyncio.run(_collect_states(teacher, teams, args.states_battles, tag))
    print(f"[distill] {len(obs)} states; distilling cfg={config}...", flush=True)

    student, m = distill_snapshot(layout, teacher, obs, mask, config,
                                  s1_epochs=args.s1_epochs, s2_epochs=args.s2_epochs,
                                  device=args.device, log=lambda s: print("  " + s, flush=True))
    speedup = _time_decode(teacher, student, obs)

    h2h = h2h_ci = h2h_n = None
    if m["top1"] >= 0.80:
        print(f"[distill] gate h2h ({args.h2h_battles} battles)...", flush=True)
        h2h, h2h_ci, h2h_n = asyncio.run(_head_to_head(student, teacher, teams, args.h2h_battles, tag))

    # gate (distill_integration.md §4): faithful-ish (h2h CI overlaps band) + ent in range + fast enough
    ent_ok = args.ent_lo <= m["ent_ratio"] <= args.ent_hi
    h2h_ok = h2h is not None and (h2h + h2h_ci) >= args.h2h_lo and (h2h - h2h_ci) <= args.h2h_hi
    passed = bool(h2h_ok and ent_ok and speedup >= args.min_speedup)

    os.makedirs(args.distilled_dir, exist_ok=True)
    pt = os.path.join(args.distilled_dir, f"snapshot_{args.step:012d}.distilled.pt")
    mf = os.path.join(args.distilled_dir, f"snapshot_{args.step:012d}.distilled.json")
    if passed:
        _atomic_write(pt, lambda p: save_distilled(student, p))
    manifest = dict(step=args.step, passed=passed, top1=m["top1"], kl=m["kl"],
                    ent_ratio=m["ent_ratio"], tok_nrmse=m.get("tok_nrmse"), speedup=speedup,
                    h2h=h2h, h2h_ci=h2h_ci, h2h_n=h2h_n, n_states=int(len(obs)), config=config)
    _atomic_write(mf, lambda p: open(p, "w").write(json.dumps(manifest, indent=2)))
    print(f"[distill] step {args.step}: passed={passed} top1={m['top1']:.3f} "
          f"h2h={h2h} speedup={speedup:.2f}x -> {mf}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
