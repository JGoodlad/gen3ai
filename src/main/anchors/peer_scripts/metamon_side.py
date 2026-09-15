#!/usr/bin/env python3
"""Run a Metamon pretrained policy as ONE websocket client, at a CHOSEN and VERIFIED regime.

🚨 **THIS FILE IS NEVER IMPORTED BY THIS REPO'S CODE.** It runs under the *Metamon* interpreter
(``opponents.metamon.python`` in ``designs/ops/anchors.json``), in an environment where
``import poke_env`` resolves to **upstream poke-env 0.8.3.3**, not our vendored fork. Importing it
from ``gen3ai_stable`` would fail on ``metamon``/``amago`` and, worse, mixing the two packages in
one process is the silent-wrong-results hazard ``src/poke_env_fork_gate_test.py`` exists to
prevent. ``main.anchors.peers.MetamonPeer`` launches it as a subprocess with ``PYTHONPATH=""``,
which is what keeps the two worlds apart. It is descended from
``designs/research_state/measurements/metamon_matched_regime_2026-09-14/run_metamon_side.py``.

WHY THE SERVER REBIND EXISTS (de-risk hazard H4)
------------------------------------------------
``metamon.env.wrappers.PokeEnvWrapper.server_configuration`` is a PROPERTY returning the
module-level ``LocalhostServerConfiguration``, which poke-env hardcodes to
``ws://localhost:8000`` — our shared DEV server, one port from the live TRAINING server on 8001.
There is no CLI flag, no env var and no constructor argument. We rebind the module global before
any env is constructed (the property reads it at call time) and refuse 8000/8001 in code.

HOW GREEDY IS SET — and why NOT with a temperature (hazard H-D)
---------------------------------------------------------------
``amago``'s rollout loop calls ``policy.get_actions(..., sample=Experiment.sample_actions_val)``,
and ``get_actions(sample=False)`` on a discrete policy returns ``argmax(dist.probs)``: exact
greedy. ``sample_actions_val`` is an ``Experiment`` field Metamon's ``make_placeholder_experiment``
never passes, so we set it by wrapping ``PretrainedModel.initialize_agent`` — the one construction
site, so no code path can miss it.

A temperature CANNOT express greedy here. ``MetamonDiscrete.forward`` computes
``vec / self.temperature`` **and then clips the probabilities to [0.001, 0.99] before
renormalising**, so on the 9-way ``MinimalActionSpace`` the argmax action tops out at
``0.99 / (0.99 + 8*0.001) ~= 0.992`` however cold the temperature — about one decision in 125
would still be a random non-argmax move, and nothing would say so.

HOW THE REGIME IS VERIFIED (never assumed)
------------------------------------------
Two instruments, both written to ``--report-out``:

* ``sample_kwarg_values`` — the ``sample`` keyword each ``Agent.get_actions`` call ACTUALLY
  received. Exactly ``[False]`` in greedy, ``[True]`` in t1.
* ``argmax_match_rate`` — the fraction of decisions where the EMITTED action equals the argmax of
  the distribution the actor produced. Must be 1.0000 in greedy; in t1 it must be materially below
  1.0000, which is the POSITIVE CONTROL that the instrument has any power at all. Measured
  2026-09-14 over 14,085 sampled decisions: 0.647 for ``SmallRL``, 0.880 for ``SyntheticRLV2`` —
  the finding that "temperature 1.0" is not one regime across models.

CPU HAZARD (de-risk H2)
-----------------------
``amago``'s ``TformerTrajEncoder`` defaults to ``FlashAttention``, a CUDA-only wheel, so every
Metamon transformer policy is unrunnable on CPU as shipped. ``VanillaAttention`` is the same exact
causal softmax attention computed the slow way, injected through the supported
``PretrainedModel.gin_overrides`` seam — so the policy's outputs are unchanged and only speed
differs.
"""

import argparse
import json
import os
import random
import re
import sys
import time

# Before torch: a GPU this process can see is a GPU it may grab, and a training arm owns it.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

RESERVED_PORTS = {8000: "the shared DEV server", 8001: "the live TRAINING server"}


def rebind_server(server_uri):
    """Point every Metamon env at ``server_uri`` — de-risk hazard H4."""
    m = re.match(r"^(wss?)://([^/:]+):(\d+)(/.*)?$", server_uri)
    if not m:
        raise SystemExit(f"--server-uri {server_uri!r} must look like ws://host:port/path")
    port = int(m.group(3))
    if port in RESERVED_PORTS:
        raise SystemExit(f"refusing port {port}: that is {RESERVED_PORTS[port]}.")
    from poke_env.ps_client.server_configuration import ServerConfiguration

    cfg = ServerConfiguration(server_uri, "https://play.pokemonshowdown.com/action.php?")
    import metamon.env.wrappers as wrappers

    wrappers.LocalhostServerConfiguration = cfg
    import poke_env.ps_client.server_configuration as sc

    sc.LocalhostServerConfiguration = cfg
    print(f"[peer] server rebound to {cfg.websocket_url}", flush=True)


def install_seeded_team_draw(seed, draw_log):
    """Give ``TeamSet.yield_team`` a PRIVATE seeded RNG over a SORTED file list.

    Metamon's shipped ``yield_team`` calls ``random.choice(self.team_files)`` on the process-global
    ``random`` module over a file list whose order comes from a directory walk. Neither is
    reproducible across processes, so a "seeded" series would not in fact repeat. Patched at CLASS
    level (``PokeEnvWrapper`` deep-copies a team set on one of its paths) and preserving
    ``_most_recent_team_file``, which is what Metamon's own per-battle CSV records.
    """
    import metamon.env.wrappers as wrappers

    def seeded_yield_team(self):
        files = getattr(self, "_sorted_files", None)
        if files is None:
            files = self._sorted_files = sorted(self.team_files)
            self._draw_rng = random.Random(seed)
        for _ in range(100):
            path = self._draw_rng.choice(files)
            self._most_recent_team_file = path
            with open(path, "r") as fh:
                team_data = fh.read()
            candidate = self.join_team(self.parse_showdown_team(team_data))
            if not self.block_team(candidate):
                draw_log.append(os.path.basename(path))
                return candidate
        raise RuntimeError("Could not find valid team after 100 attempts")

    wrappers.TeamSet.yield_team = seeded_yield_team
    print(f"[peer] team draw seeded with {seed} over a SORTED file list", flush=True)


class RegimeProbe:
    """Per-decision timing + the two regime-verification instruments."""

    def __init__(self, verify=True):
        self.samples = []
        self.verify = verify
        self.sample_kwargs = set()
        self.n_checked = 0
        self.n_argmax_match = 0
        self._last_probs = None

    def install(self):
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
                    # probs: (Batch, Length, Gammas, Actions); the emitted action comes from the
                    # LAST gamma (`actions[..., -1, :]` inside get_actions), so that is the row.
                    greedy = probe._last_probs[..., -1, :].argmax(dim=-1)
                    emitted = actions.squeeze(-1)
                    probe.n_checked += int(emitted.numel())
                    probe.n_argmax_match += int((greedy == emitted).sum().item())
                except Exception:
                    pass
                probe._last_probs = None
            return actions, hidden

        amago.agent.Agent.get_actions = get_actions
        print("[peer] probes installed on Agent.get_actions"
              + (" + BaseActorHead.forward" if self.verify else ""), flush=True)

    def report(self):
        out = {}
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


def install_regime(sample):
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
        print(f"[peer] Experiment.sample_actions_val = {sample}", flush=True)
        return exp

    PretrainedModel.initialize_agent = initialize_agent


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-uri", required=True, help="ws://host:port/showdown/websocket")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--checkpoint", type=int, default=None)
    ap.add_argument("--username", required=True)
    ap.add_argument("--opponent-username", required=True)
    ap.add_argument("--role", choices=("challenger", "acceptor"), default="acceptor")
    ap.add_argument("--total-battles", type=int, default=50)
    ap.add_argument("--battle-format", default="gen3ou")
    ap.add_argument("--team-set", default="gen3ai_pool")
    ap.add_argument("--team-seed", type=int, default=None)
    ap.add_argument("--regime", choices=("greedy", "t1"), required=True,
                    help="greedy = Agent.get_actions(sample=False), i.e. argmax; "
                         "t1 = Metamon's shipped default (sample, action_temperature 1.0)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="action_temperature for t1; IGNORED by greedy (see the module docstring: "
                         "MetamonDiscrete clips probabilities and cannot express it)")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--battle-backend", default=None,
                    help="default: the model's own (SmallRL/SyntheticRLV2 want 'poke-env')")
    ap.add_argument("--attention", default="vanilla", choices=("vanilla", "flex", "flash"))
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--report-out", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    rebind_server(args.server_uri)

    import metamon.env
    from metamon.rl.evaluate.__main__ import pretrained_vs_challenge
    from metamon.rl.pretrained import get_pretrained_model

    sample = args.regime == "t1"
    install_regime(sample)

    draw_log = []
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
        print(f"[peer] attention -> {attn.__name__} (CPU; flash-attn is CUDA-only)", flush=True)

    team_set = metamon.env.get_metamon_teams(args.battle_format, args.team_set)

    probe = RegimeProbe(verify=args.verify)
    probe.install()

    t0 = time.time()
    results = {}
    error = None
    try:
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
            # Only consulted in t1; greedy never samples the distribution at all.
            action_temperature=args.temperature,
            save_results_to=args.results_dir,
        )
    except BaseException as exc:            # noqa: BLE001 - the report must survive ANY death
        # 🚨 The whole point of writing the report from a `finally`: hazard H-B kills this process
        # with a RecursionError when OUR side forfeits at turn 250 (Metamon's long-tail handler
        # force-resets and then calls itself, ~985 levels deep). A report that only exists on the
        # happy path would leave the driver with a dead peer and NO regime record — which is
        # exactly the "reported nothing, looks like a result" failure this tool exists to refuse.
        error = f"{type(exc).__name__}: {exc}"
        print(f"[peer] FATAL {error}", flush=True)
    elapsed = time.time() - t0

    timing = probe.report()
    timing.update({
        "wall_s": elapsed, "agent": args.agent, "checkpoint": args.checkpoint,
        "backend": backend, "regime": args.regime, "role": args.role,
        "team_set": args.team_set, "team_seed": args.team_seed,
        "action_temperature": args.temperature if sample else None,
        "n_team_draws": len(draw_log), "team_draws": draw_log,
        "error": error,
    })
    # Metamon's own scoreboard, complementing ours — the independent cross-check that the win
    # accounting is right. Its `won` field is a BOOLEAN, so it books a TIE as its own loss
    # (de-risk H3); our side's poke-env flags are the authority.
    if isinstance(results, dict):
        timing["peer_results"] = {str(k): str(v) for k, v in results.items()}

    expect_sample = [sample]
    ok_kw = timing["sample_kwarg_values"] == expect_sample
    rate = timing.get("argmax_match_rate")
    ok_rate = (rate is None) or (rate == 1.0 if args.regime == "greedy" else rate < 1.0)
    timing["regime_check_ok"] = bool(ok_kw and ok_rate and error is None)
    print(f"[peer] REGIME CHECK regime={args.regime} "
          f"sample_kwargs={timing['sample_kwarg_values']} (ok={ok_kw}) "
          f"argmax_match_rate={rate} (ok={ok_rate})", flush=True)
    if not timing["regime_check_ok"]:
        print("[peer] 🚨 REGIME CHECK FAILED — this cell is NOT a measurement", flush=True)

    with open(args.report_out, "w") as fh:
        json.dump(timing, fh, indent=1, default=str)
    print("[peer] REPORT " + json.dumps({k: v for k, v in timing.items()
                                         if k != "team_draws"}, default=str), flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
