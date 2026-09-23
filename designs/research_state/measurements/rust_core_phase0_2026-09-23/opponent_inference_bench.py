"""Phase-0 (b): what ONE per-env opponent forward costs on CPU at the training shape, and at a
Tier-2-style batch — the price list the inference tier is judged against.

    export PYTHONPATH=$PYTHONPATH:src
    CUDA_VISIBLE_DEVICES= python3 designs/research_state/measurements/rust_core_phase0_2026-09-23/opponent_inference_bench.py \
        --model /home/goodlad/dev/gen3ai/models/ai_v13_21_wcont_b/snapshots/snapshot_000072000000.zip

What it times is EXACTLY the opponent's per-decision network call (`RLPlayer._predict_best_action`
minus the obs build): `policy.get_distribution` on a Dict obs, under `torch.no_grad`, on CPU with
ONE torch thread — the env-worker setting (`main/train/env_factory.py` sets 1). The extractor is
compiled with the production helper `maybe_compile_extractor` (its own eager-vs-compiled gate
prints beside our numbers). The obs is the declared synthetic warm-up obs
(`compile_opponents._compile_warmup_obs`) — a DENSE network's cost does not depend on its values.

Arms, alternated round by round (the same-process A/B discipline):
  eager B=1 · compiled B=1 (the production opponent) · eager B=48 (one Tier-2-style CPU batch)
  · compiled B=48 (dynamic re-trace) — each a median over rounds of a min over reps.

CPU only. Nothing is written anywhere.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = os.environ.get("BENCH_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import copy  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from agents.model.compile_opponents import _compile_warmup_obs, maybe_compile_extractor  # noqa: E402
from agents.model.model_version import ModelVersion  # noqa: E402
from agents.model.snapshot import _resolve_paths, load_foreign_opponent  # noqa: E402
from utils.contention import describe_contention  # noqa: E402


def _batch(obs1: dict, b: int) -> dict:
    return {k: v.repeat(b, *([1] * (v.dim() - 1))) for k, v in obs1.items()}


def _time(fn, obs, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(obs)
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--npz", default=None,
                    help="a banked eval-trace *_states.npz: time REAL obs rows + masks")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "1")))
    zip_path, cfg_dir = _resolve_paths(a.model)
    cfg = os.path.join(os.path.dirname(cfg_dir) if not os.path.exists(
        os.path.join(cfg_dir, "model_config.json")) else cfg_dir, "model_config.json")
    ver = ModelVersion.from_json_file(cfg)
    eager_model, _ = load_foreign_opponent(zip_path, ver, device="cpu")
    comp_model = copy.deepcopy(eager_model)
    eager_model.policy.eval()
    comp_model.policy.eval()
    obs1 = {k: torch.as_tensor(v) for k, v in _compile_warmup_obs(
        eager_model.policy.features_extractor).items()}
    obs1 = {k: (v if v.dim() > 1 and v.shape[0] == 1 else v.unsqueeze(0)) for k, v in obs1.items()}
    obsb = _batch(obs1, a.batch)
    if a.npz:
        import numpy as np
        z = np.load(a.npz, allow_pickle=True)
        rows = np.asarray(z["obs"], dtype=np.float32)
        masks = np.asarray(z["action_mask"], dtype=np.float32)
        n = min(a.batch, len(rows))
        mid = len(rows) // 2
        obs1 = dict(obs1, observation=torch.as_tensor(rows[mid:mid + 1]),
                    action_mask=torch.as_tensor(masks[mid:mid + 1]))
        obsb = _batch(obs1, n)
        obsb["observation"] = torch.as_tensor(rows[:n])
        obsb["action_mask"] = torch.as_tensor(masks[:n])
        a.batch = n
        print(f"  obs: {n} REAL banked decisions from {a.npz}")

    print(f"OPPONENT INFERENCE (CPU, torch threads={torch.get_num_threads()}, model={zip_path})")
    print(f"  {describe_contention()}")
    t0 = time.perf_counter()
    applied = maybe_compile_extractor(comp_model, True, label="phase0-bench", hide_cuda=True)
    print(f"  compile applied={applied}  (compile + its own gate: {time.perf_counter() - t0:.1f} s)")

    def fwd(model):
        def f(obs):
            with torch.no_grad():
                d = model.policy.get_distribution(obs)
                return d.distribution.logits
        return f

    fe, fc = fwd(eager_model), fwd(comp_model)
    for f, o in ((fe, obs1), (fc, obs1), (fe, obsb), (fc, obsb)):
        for _ in range(3):
            f(o)
    if applied:
        with torch.no_grad():
            le = fe(obs1)
            lc = fc(obs1)
        print(f"  max|Δlogits| eager vs compiled at B=1: {float((le - lc).abs().max()):.3g}")
    arms = {"eager B=1": (fe, obs1), "compiled B=1": (fc, obs1),
            f"eager B={a.batch}": (fe, obsb), f"compiled B={a.batch}": (fc, obsb)}
    res = {k: [] for k in arms}
    names = list(arms)
    for r in range(a.rounds):
        order = names if r % 2 == 0 else list(reversed(names))
        for k in order:
            f, o = arms[k]
            res[k].append(_time(f, o, a.reps))
    print(f"  {'arm':<18} {'median ms/call':>15} {'ms per ROW':>11}   (rounds={a.rounds}, "
          f"each a min over {a.reps} calls)")
    med = {k: statistics.median(v) for k, v in res.items()}
    for k in names:
        b = a.batch if f"B={a.batch}" in k else 1
        print(f"  {k:<18} {med[k]:>15.3f} {med[k] / b:>11.4f}")
    print(f"  compiled/eager at B=1: {med['eager B=1'] / med['compiled B=1']:.2f}x   "
          f"B={a.batch} batch per-row vs compiled B=1: "
          f"{med['compiled B=1'] / (med[f'eager B={a.batch}'] / a.batch):.1f}x (eager batch), "
          f"{med['compiled B=1'] / (med[f'compiled B={a.batch}'] / a.batch):.1f}x (compiled batch)")
    print(f"  {describe_contention()}")


if __name__ == "__main__":
    main()
