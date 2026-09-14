#!/usr/bin/env python3
"""Run a Metamon pretrained policy as one client on OUR Showdown server, at a CHOSEN regime.

This is the 2026-09-14 de-risk's ``run_metamon_side.py`` plus the three things a matched-regime
2x2 needs and the de-risk did not: a **greedy** setting, a **seeded** team draw, and an instrument
that proves the greedy setting took effect.

WHY THE PORT REBIND EXISTS (de-risk hazard H4)
----------------------------------------------
``metamon.env.wrappers.PokeEnvWrapper.server_configuration`` is a PROPERTY returning the
module-level ``LocalhostServerConfiguration``, which poke-env hardcodes to
``ws://localhost:8000`` — our shared DEV server, one port away from the live TRAINING server on
8001. There is no CLI flag, no env var and no constructor argument. We rebind the module global
before any env is constructed (the property reads it at call time) and refuse 8000/8001 in code.

HOW GREEDY IS SET — and why NOT with a temperature
--------------------------------------------------
``amago``'s rollout loop calls ``policy.get_actions(..., sample=Experiment.sample_actions_val)``,
and ``get_actions(sample=False)`` on a discrete policy returns ``argmax(dist.probs)``: exact
greedy. ``sample_actions_val`` is an ``Experiment`` field that Metamon's
``make_placeholder_experiment`` never passes, so we set it on the built experiment by wrapping
``PretrainedModel.initialize_agent`` — the one construction site, so no code path can miss it.

We do **NOT** drive ``action_temperature`` toward zero. ``MetamonDiscrete.forward`` computes
``vec / self.temperature`` (zero divides by zero) **and then clips the probabilities to
[clip_prob_low=0.001, clip_prob_high=0.99] before renormalising**. On the 9-way
``MinimalActionSpace`` that leaves the argmax action at 0.99/(0.99 + 8*0.001) ~= 0.992 no matter
how cold the temperature is — about one action in 125 would still be a random non-argmax move.
A temperature knob CANNOT express greedy in this codebase.

HOW IT IS VERIFIED (never assumed)
----------------------------------
``--verify`` installs two instruments and writes them to the timing JSON:

* ``sample_kwarg_values`` — the ``sample`` keyword each ``Agent.get_actions`` call actually
  received. Must be exactly ``[False]`` in a greedy run and ``[True]`` in a sampling run.
* ``argmax_match_rate`` — the fraction of decisions on which the action the policy EMITTED equals
  the argmax of the action distribution the actor produced for that decision. Must be 1.000 in a
  greedy run; in a sampling run it must be materially below 1.000, which is the POSITIVE CONTROL
  that the instrument has any power at all.

CPU HAZARD (de-risk H2)
-----------------------
``amago``'s ``TformerTrajEncoder`` defaults to ``FlashAttention``, a CUDA-only wheel, so every
Metamon transformer policy is unrunnable on CPU as shipped. ``VanillaAttention`` is the same exact
causal softmax attention computed the slow way; injected through the supported
``PretrainedModel.gin_overrides`` seam.

    python run_metamon_side.py --agent SmallRL --port 9350 --username MetaSmallRL \
        --opponent_username Gen3AIv12 --role acceptor --total_battles 50 \
        --team_set gen3ai_pool --regime greedy --team-seed 20260914 \
        --save_results_to ./out --timing_out ./out/metamon_timing.json
"""

import argparse
import json
import os
import random
import sys
import time

# Before torch: a GPU this process can see is a GPU it may grab, and a live training arm owns it.
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
    import poke_env.ps_client.server_configuration as sc

    sc.LocalhostServerConfiguration = cfg
    print(f"[metamon-side] server rebound to {cfg.websocket_url}", flush=True)


# --------------------------------------------------------------------------------------------
# Seeded team draws
# --------------------------------------------------------------------------------------------
def install_seeded_team_draw(seed: int, draw_log: list) -> None:
    """Give ``TeamSet.yield_team`` a PRIVATE seeded RNG over a SORTED file list.

    Metamon's shipped ``yield_team`` calls ``random.choice(self.team_files)`` on the
    process-global ``random`` module, over a file list whose order comes from a directory walk.
    Neither is reproducible across processes, so a "seeded" series would not in fact repeat. The
    patch is at CLASS level (``PokeEnvWrapper`` deep-copies a team set on one of its paths) and
    preserves ``_most_recent_team_file``, which is what Metamon's own per-battle CSV records.
    """
    import metamon.env.wrappers as wrappers

    def seeded_yield_team(self) -> str:
        files = getattr(self, "_sorted_files", None)
        if files is None:
            files = self._sorted_files = sorted(self.team_files)
            self._draw_rng = random.Random(seed)
        for attempt in range(100):
            file = self._draw_rng.choice(files)
            self._most_recent_team_file = file
            with open(file, "r") as f:
                team_data = f.read()
            candidate = self.join_team(self.parse_showdown_team(team_data))
            if not self.block_team(candidate):
                draw_log.append(os.path.basename(file))
                return candidate
        raise RuntimeError("Could not find valid team after 100 attempts")

    wrappers.TeamSet.yield_team = seeded_yield_team
    print(f"[metamon-side] team draw seeded with {seed} over a SORTED file list", flush=True)


# --------------------------------------------------------------------------------------------
# The regime, and the instruments that prove it
# --------------------------------------------------------------------------------------------
class RegimeProbe:
    """Per-decision timing + the two greedy-verification instruments."""

    def __init__(self, verify: bool):
        self.samples: list = []
        self.verify = verify
        self.sample_kwargs: set = set()
        self.n_checked = 0
        self.n_argmax_match = 0
        self._last_probs = None

    def install(self) -> None:
        import amago.agent
        import amago.nets.actor_critic as ac

        probe = self

        if self.verify:
            inner_actor_fwd = ac.BaseActorHead.forward

            def actor_forward(self_, *a, **k):
                dist = inner_actor_fwd(self_, *a, **k)
                try:
                    probe._last_probs = dist.probs.detach()
                except Exception:
                    probe._last_probs = None
                return dist

            ac.BaseActorHead.forward = actor_forward

        inner_get_actions = amago.agent.Agent.get_actions

        def get_actions(self_, *a, **k):
            probe.sample_kwargs.add(bool(k.get("sample", True)))
            t0 = time.perf_counter()
            try:
                actions, hidden = inner_get_actions(self_, *a, **k)
            finally:
                probe.samples.append(time.perf_counter() - t0)
            if probe.verify and probe._last_probs is not None:
                try:
                    # probs: (Batch, Length, Gammas, Actions); the emitted action is taken from
                    # the LAST gamma (`actions[..., -1, :]` inside get_actions), so that is the
                    # row to compare against.
                    greedy = probe._last_probs[..., -1, :].argmax(dim=-1)
                    emitted = actions.squeeze(-1)
                    probe.n_checked += int(emitted.numel())
                    probe.n_argmax_match += int((greedy == emitted).sum().item())
                except Exception:
                    pass
                probe._last_probs = None
            return actions, hidden

        amago.agent.Agent.get_actions = get_actions
        print("[metamon-side] probes installed on Agent.get_actions"
              + (" + BaseActorHead.forward" if self.verify else ""), flush=True)

    def report(self) -> dict:
        out: dict = {}
        if self.samples:
            s = sorted(self.samples)
            n = len(s)
            out = {"n_decisions": n, "mean_s": sum(s) / n, "median_s": s[n // 2],
                   "p90_s": s[int(0.9 * n)], "max_s": s[-1]}
        out["sample_kwarg_values"] = sorted(self.sample_kwargs)
        if self.verify:
            out["argmax_checked"] = self.n_checked
            out["argmax_matched"] = self.n_argmax_match
            out["argmax_match_rate"] = (self.n_argmax_match / self.n_checked
                                        if self.n_checked else None)
        return out


def install_regime(sample: bool) -> None:
    """Set ``Experiment.sample_actions_val`` on every agent this process builds.

    Wrapped at ``PretrainedModel.initialize_agent`` rather than passed through
    ``pretrained_vs_challenge``, because that helper builds the agent internally and exposes no
    seam. Everything else stays Metamon's own code path.
    """
    from metamon.rl.pretrained import PretrainedModel

    inner = PretrainedModel.initialize_agent

    def initialize_agent(self_, *a, **k):
        exp = inner(self_, *a, **k)
        exp.sample_actions_val = sample
        exp.sample_actions_train = sample
        print(f"[metamon-side] Experiment.sample_actions_val = {sample}", flush=True)
        return exp

    PretrainedModel.initialize_agent = initialize_agent


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
    ap.add_argument("--team-seed", dest="team_seed", type=int, default=None,
                    help="seed the team draw (a SORTED file list + a private RNG)")
    ap.add_argument("--regime", choices=("greedy", "t1"), required=True,
                    help="greedy = Agent.get_actions(sample=False), i.e. argmax; "
                         "t1 = Metamon's shipped default (sample, action_temperature 1.0)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="action_temperature for the t1 regime; IGNORED by greedy (see module "
                         "docstring: MetamonDiscrete clips probabilities and cannot express it)")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the argmax-match instrument (it costs one comparison per move)")
    ap.add_argument("--battle_backend", default=None,
                    help="default: the model's own (SmallRL/SyntheticRLV2 want 'poke-env')")
    ap.add_argument("--save_results_to", default=None)
    ap.add_argument("--attention", default="vanilla", choices=("vanilla", "flex", "flash"))
    ap.add_argument("--timing_out", default=None)
    ap.add_argument("--draws_out", default=None, help="write the ordered team draws here")
    args = ap.parse_args()

    rebind_localhost(args.port)

    import metamon.env
    from metamon.rl.evaluate.__main__ import pretrained_vs_challenge
    from metamon.rl.pretrained import get_pretrained_model

    sample = args.regime == "t1"
    install_regime(sample)

    draw_log: list = []
    if args.team_seed is not None:
        install_seeded_team_draw(args.team_seed, draw_log)

    model = get_pretrained_model(args.agent)
    backend = args.battle_backend or model.battle_backend

    if args.attention != "flash":
        import amago.nets.transformer as _tf

        attn = {"vanilla": _tf.VanillaAttention, "flex": _tf.VanillaFlexAttention}[args.attention]
        overrides = dict(model.gin_overrides or {})
        overrides["traj_encoders.TformerTrajEncoder.attention_type"] = attn
        model.gin_overrides = overrides
        print(f"[metamon-side] attention -> {attn.__name__} (CPU; flash-attn is CUDA-only)",
              flush=True)

    team_set = metamon.env.get_metamon_teams(args.battle_format, args.team_set)

    probe = RegimeProbe(verify=args.verify)
    probe.install()

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
        # Only consulted in the t1 regime; greedy never samples the distribution at all.
        action_temperature=args.temperature,
        save_results_to=args.save_results_to,
    )
    elapsed = time.time() - t0
    print(json.dumps(results, indent=2, sort_keys=True, default=str), flush=True)

    timing = probe.report() | {
        "wall_s": elapsed, "agent": args.agent, "checkpoint": args.checkpoint,
        "backend": backend, "regime": args.regime, "role": args.role,
        "team_set": args.team_set, "team_seed": args.team_seed,
        "action_temperature": args.temperature if sample else None,
        "n_team_draws": len(draw_log),
    }
    print("[metamon-side] TIMING " + json.dumps(timing), flush=True)

    # The regime check, stated loudly so a log tail cannot miss it.
    expect_sample = [sample]
    ok_kw = timing["sample_kwarg_values"] == expect_sample
    rate = timing.get("argmax_match_rate")
    ok_rate = (rate is None) or (rate == 1.0 if args.regime == "greedy" else rate < 1.0)
    print(f"[metamon-side] REGIME CHECK regime={args.regime} "
          f"sample_kwargs={timing['sample_kwarg_values']} (ok={ok_kw}) "
          f"argmax_match_rate={rate} (ok={ok_rate})", flush=True)
    if not (ok_kw and ok_rate):
        print("[metamon-side] 🚨 REGIME CHECK FAILED — this cell is NOT a measurement", flush=True)

    if args.timing_out:
        with open(args.timing_out, "w") as f:
            json.dump(timing | {"regime_check_ok": bool(ok_kw and ok_rate)}, f, indent=1)
    if args.draws_out:
        with open(args.draws_out, "w") as f:
            json.dump(draw_log, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
