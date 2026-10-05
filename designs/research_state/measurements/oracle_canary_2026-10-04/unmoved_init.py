"""Which parameters are BIT-IDENTICAL to their fresh-build init (`gen3_r1_unmoved_init_v1`).

Usage (repo root, CPU only, READ-ONLY on models/):
    unmoved_init.py checkpoint <run_dir> <ckpt.zip> <out.json>
        rebuild the run's fresh init from its recorded `original_command` (seed + arch flags, under
        `single_thread_build`, through the trainer's own `construct_fresh_learner` arguments) and
        list every policy parameter of the checkpoint that is bit-identical to it, plus every
        all-zero one (the 8b8fbac0 rule). Also checks the rebuild against parameters that cannot
        have moved for a structural reason, so a wrong rebuild reads as zero matches, not silence.
    unmoved_init.py predict <out.json> [seed]
        build fresh at `--oracle-reveal off|species|full` (production blob arm) and run ONE eager
        PPO micro-step on the K9 golden rows with the oracle's labels (all PAD) — list the parameters
        whose gradient is None or exactly zero (the ones training cannot move), per level.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from typing import Any, Dict, List

import torch

LAUNCHER_ONLY = {"--restart-interval-hours": 1, "--pin-commit": 1}


def _sha(p: torch.Tensor) -> str:
    t = p.detach().cpu().contiguous()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def _trainer_argv(cmd: str) -> List[str]:
    toks = cmd.split()
    toks = toks[toks.index(next(t for t in toks if t.endswith("__main__.py"))) + 1:]
    out: List[str] = []
    i = 0
    while i < len(toks):
        if toks[i] in LAUNCHER_ONLY:
            i += 1 + LAUNCHER_ONLY[toks[i]]
            continue
        out.append(toks[i])
        i += 1
    for j, t in enumerate(out):
        if t == "--device":
            out[j + 1] = "cpu"
    return out


def _resolve(argv: List[str]) -> Any:
    from main.train.config import resolve_config
    from main.train.parser import build_parser
    parser = build_parser()
    a = parser.parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, parser)
    return a


def _fresh(args: Any) -> Any:
    from main.fresh_checkpoint import build_fresh_model
    model, _, _ = build_fresh_model(int(args.seed), args=args)
    return model


def checkpoint(run_dir: str, ckpt: str, out: str) -> None:
    from agents.model.snapshot import load_checkpoint_strict
    meta = json.load(open(f"{run_dir}/metadata.json"))
    argv = _trainer_argv(meta["original_command"])
    args = _resolve(argv)
    fresh = dict(_fresh(args).policy.named_parameters())
    m = load_checkpoint_strict(ckpt, device="cpu")
    live = dict(m.policy.named_parameters())
    rows = []
    for n, p in live.items():
        f = fresh.get(n)
        same = f is not None and f.shape == p.shape and torch.equal(f.detach(), p.detach())
        rows.append({"name": n, "numel": p.numel(), "bit_identical_to_init": bool(same),
                     "all_zero": not bool(p.detach().any()),
                     "init_all_zero": bool(f is not None and not bool(f.detach().any())),
                     "max_abs_delta": (None if f is None or f.shape != p.shape
                                       else float((p.detach() - f.detach()).abs().max()))})
    ident = [r["name"] for r in rows if r["bit_identical_to_init"]]
    zero = [r["name"] for r in rows if r["all_zero"]]
    res = {"run_dir": run_dir, "checkpoint": ckpt, "trainer_argv": argv,
           "n_params": len(rows), "n_bit_identical": len(ident), "n_all_zero": len(zero),
           "bit_identical": ident, "all_zero": zero,
           "identical_not_zero": [n for n in ident if n not in zero],
           "zero_not_identical": [n for n in zero if n not in ident],
           "num_timesteps": int(getattr(m, "num_timesteps", 0)), "rows": rows}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def predict(out: str, seed: int = 1001) -> None:
    """The fresh build at `--oracle-reveal off|species|full` (the blob arm, the oracle runs' argv):
    the parameter set and every parameter's init bytes. The reveal is an OBSERVATION mode with no
    module and no weight, so equal sets + equal bytes mean the unmoved set at `full` is decided by
    which losses go to no-op (the labels), not by the build."""
    builds: Dict[str, Dict[str, str]] = {}
    for level in ("off", "species", "full"):
        argv = ["--steps", "1", "--arch", "production", "--belief-tokens", "blob", "--seed", str(seed),
                "--ridealong-ensemble", "5", "--ridealong-rnd", "--ridealong-adv", "5", "--ridealong-opp", "5",
                "--ridealong-rnd-variants", "all", "--device", "cpu", "--oracle-reveal", level]
        model = _fresh(_resolve(argv))
        builds[level] = {n: _sha(p) for n, p in model.policy.named_parameters()}
        del model
    res = {"seed": seed, "n_params": {k: len(v) for k, v in builds.items()},
           "same_names": {k: list(v) == list(builds["species"]) for k, v in builds.items()},
           "differing_init_bytes": {k: [n for n in v if builds["species"].get(n) != v[n]]
                                    for k, v in builds.items()}}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "checkpoint":
        checkpoint(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        predict(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1001)
