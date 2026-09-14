#!/usr/bin/env python3
"""Run a Metamon pretrained policy as one client on OUR Showdown server.

WHY THIS WRAPPER EXISTS — the hazard it closes
----------------------------------------------
``metamon.env.wrappers.PokeEnvWrapper.server_configuration`` is a PROPERTY that returns the
module-level ``LocalhostServerConfiguration``, which poke-env hardcodes to
``ws://localhost:8000/showdown/websocket``. There is no CLI flag, no env var and no constructor
argument for the port. Port 8000 is our shared DEV server and 8001 the live TRAINING server, and
this box must never have a stray client on either — so we REBIND the module global before any env
is constructed. The property reads it at call time, so the rebind takes effect.

Everything else is Metamon's own code path (``metamon.rl.evaluate``'s ``challenge`` eval type,
i.e. ``ChallengeByUsername``), so what plays here is the policy as its authors run it.

Runs on CPU only: ``CUDA_VISIBLE_DEVICES=""`` is set before torch is imported, because a live
training arm owns this box's GPU.

    python run_metamon_side.py --agent SmallRL --port 9217 \
        --username MetamonSmallRL --opponent_username Gen3AIv12 --role acceptor \
        --total_battles 60 --team_set gen3ai_pool --save_results_to ./results
"""

import argparse
import os
import sys
import time

# Before torch: a GPU this process can see is a GPU it may grab.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("METAMON_CACHE_DIR", "/home/goodlad/dev/metamon/cache")

RESERVED_PORTS = {8000: "the shared DEV server", 8001: "the live TRAINING server"}


def rebind_localhost(port: int) -> None:
    """Point every Metamon env at ``ws://localhost:<port>``."""
    if port in RESERVED_PORTS:
        raise SystemExit(f"refusing --port {port}: that is {RESERVED_PORTS[port]}.")
    from poke_env.ps_client.server_configuration import ServerConfiguration

    cfg = ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    import metamon.env.wrappers as wrappers

    wrappers.LocalhostServerConfiguration = cfg
    # Belt and braces: anything that imported the name directly.
    import poke_env.ps_client.server_configuration as sc

    sc.LocalhostServerConfiguration = cfg
    print(f"[metamon-side] server rebound to {cfg.websocket_url}", flush=True)


class MoveTimer:
    """Per-decision wall time for the policy's forward pass.

    Wraps ``amago.Experiment``'s inner agent call the cheap way — around the env ``step`` is
    wrong (it includes the opponent's think time and the server round trip), so we patch the
    metamon-side action selection instead and record only the seconds spent inside it.
    """

    def __init__(self):
        self.samples = []

    def wrap(self, obj, attr):
        inner = getattr(obj, attr)

        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return inner(*a, **k)
            finally:
                self.samples.append(time.perf_counter() - t0)

        setattr(obj, attr, timed)

    def report(self):
        if not self.samples:
            return {}
        s = sorted(self.samples)
        n = len(s)
        return {
            "n_decisions": n,
            "mean_s": sum(s) / n,
            "median_s": s[n // 2],
            "p90_s": s[int(0.9 * n)],
            "max_s": s[-1],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--checkpoint", type=int, default=None)
    ap.add_argument("--username", required=True)
    ap.add_argument("--opponent_username", required=True)
    ap.add_argument("--role", choices=("challenger", "acceptor"), default="acceptor")
    ap.add_argument("--total_battles", type=int, default=50)
    ap.add_argument("--battle_format", default="gen3ou")
    ap.add_argument("--team_set", default="gen3ai_pool")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--battle_backend", default=None,
                    help="default: the model's own (SmallRL/SyntheticRLV2 want 'poke-env')")
    ap.add_argument("--save_results_to", default=None)
    ap.add_argument("--attention", default="vanilla", choices=("vanilla", "flex", "flash"),
                    help="attention kernel; 'flash' needs a CUDA-only wheel, so CPU needs "
                         "'vanilla' (same math, slower)")
    ap.add_argument("--timing_out", default=None, help="write per-move timing JSON here")
    args = ap.parse_args()

    rebind_localhost(args.port)

    import json

    import metamon.env
    from metamon.rl.evaluate.__main__ import pretrained_vs_challenge
    from metamon.rl.pretrained import get_pretrained_model

    model = get_pretrained_model(args.agent)
    backend = args.battle_backend or model.battle_backend

    # 🚨 CPU HAZARD. amago's `TformerTrajEncoder` defaults to `FlashAttention`, which asserts on
    # a missing `flash-attn` install — a CUDA-only wheel. Every Metamon transformer policy is
    # therefore UNRUNNABLE on CPU out of the box (`AssertionError: Missing flash attention 2
    # install`). `VanillaAttention` is the same causal softmax attention computed the slow way,
    # so the policy's outputs are unchanged; only the speed is. Injected as a gin override on the
    # PretrainedModel, which is the supported seam for exactly this.
    if not args.attention == "flash":
        import amago.nets.transformer as _tf

        attn = {"vanilla": _tf.VanillaAttention, "flex": _tf.VanillaFlexAttention}[args.attention]
        overrides = dict(model.gin_overrides or {})
        overrides["traj_encoders.TformerTrajEncoder.attention_type"] = attn
        model.gin_overrides = overrides
        print(f"[metamon-side] attention -> {attn.__name__} (CPU; flash-attn is CUDA-only)",
              flush=True)
    team_set = metamon.env.get_metamon_teams(args.battle_format, args.team_set)

    timer = MoveTimer()
    # `amago.Experiment.interact`-level timing is not reachable from here without forking the
    # loop, so time the POLICY call itself: `Agent.get_actions` is what every metamon rollout
    # goes through. Patched on the CLASS so it covers the instance the eval builds internally.
    try:
        import amago.agent

        timer.wrap(amago.agent.Agent, "get_actions")
        print("[metamon-side] timing amago.agent.Agent.get_actions", flush=True)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[metamon-side] WARNING: could not install move timer: {exc}", flush=True)

    t0 = time.time()
    results = pretrained_vs_challenge(
        pretrained_model=model,
        username=args.username,
        opponent_username=args.opponent_username,
        role=args.role,
        battle_format=args.battle_format,
        team_set=team_set,
        total_battles=args.total_battles,
        checkpoint=args.checkpoint,
        battle_backend=backend,
        action_temperature=args.temperature,
        save_results_to=args.save_results_to,
    )
    elapsed = time.time() - t0
    print(json.dumps(results, indent=2, sort_keys=True, default=str), flush=True)
    timing = timer.report() | {"wall_s": elapsed, "agent": args.agent,
                               "checkpoint": args.checkpoint, "backend": backend}
    print("[metamon-side] TIMING " + json.dumps(timing), flush=True)
    if args.timing_out:
        with open(args.timing_out, "w") as f:
            json.dump(timing, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
