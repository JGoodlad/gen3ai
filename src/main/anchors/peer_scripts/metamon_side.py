#!/usr/bin/env python3
"""Run a Metamon pretrained policy as ONE websocket client, at a CHOSEN and VERIFIED regime.

🚨 **THIS FILE IS NEVER IMPORTED BY THIS REPO'S CODE.** It runs under the *Metamon* interpreter
(``opponents.metamon.python`` in ``designs/ops/anchors.json``), in an environment where
``import poke_env`` resolves to **upstream poke-env 0.8.3.3** (our tree has no poke-env at all since
the retirement's P6). Importing it from our env would fail on ``poke_env``/``metamon``/``amago``.
``main.anchors.peers.MetamonPeer`` launches it as a subprocess with ``PYTHONPATH=""``, which keeps our
``src/`` (and its generic top-level names) out of Metamon's interpreter. It is the ONE permanent entry of
the poke-env import gate (``PEER_PROCESS_PERMANENT``). It is descended from
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

🚨 **A GIN FILE BINDING BEATS THE ``gin_overrides`` DICT, and the newer models SLIDE A WINDOW**
(found 2026-09-28 wiring ``Kakuna``). ``amago.cli_utils.use_config`` binds the dict FIRST and then
parses the model's ``.gin`` files, so a file that itself says
``TformerTrajEncoder.attention_type = @transformer.FlashAttention`` (``superkazam.gin`` — Kakuna,
Superkazam — and ``alakazam*.gin``, ``smaller_multitaskagent*.gin``) silently re-wins and the peer
dies on the missing flash-attn wheel. Those same files also set
``FlashAttention.window_size = (96, 0)`` (or ``(32, 0)``): each query sees only the last 96 keys.
Plain ``VanillaAttention`` would attend to the whole 128-step cache — a DIFFERENT function from the
one the weights were trained under, on exactly the long games where it matters, and nothing would
say so. So :func:`install_cpu_attention` re-binds ``attention_type`` AFTER the files are parsed and
reads the window the file configured: no window → ``VanillaAttention`` (``SmallRL`` and
``SyntheticRLV2``, whose gin files bind neither key, are unchanged); a window →
:class:`WindowedVanillaAttention`, the same mask flash-attn applies (key ``j`` visible to query
``i`` iff ``i - left <= j <= i``). The choice and the window are written to ``--report-out``.
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
# ...and before torch for the same reason: the thread count is read at import. B=1 CPU inference
# with the default thread pool measured 210% CPU per peer on a shared box, which is two cores of
# synchronisation buying nothing. The driver also sets these in the peer env; this is the second
# lock, for anyone running the script by hand.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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


def install_sequence_team_draw(sequence_path, draw_log):
    """MIRRORED TEAM PAIRS (T17): yield the team FILES listed in ``sequence_path`` (a JSON list), in
    order, one per battle — the harness wrote them so that battle ``2k+1`` hands this side the team
    OUR side played in battle ``2k``. Every yielded file is logged (its basename), which is how the
    harness VERIFIES each pair was in fact swapped. A sequence that runs out, or a team Metamon blocks,
    RAISES: a silent substitute would be an unmirrored game reported as a mirrored one."""
    import metamon.env.wrappers as wrappers

    with open(sequence_path) as fh:
        files = list(json.load(fh))

    def sequence_yield_team(self):
        i = len(draw_log)
        if i >= len(files):
            raise RuntimeError(f"team sequence exhausted after {i} battles ({sequence_path})")
        path = files[i]
        self._most_recent_team_file = path
        with open(path, "r") as fh:
            candidate = self.join_team(self.parse_showdown_team(fh.read()))
        if self.block_team(candidate):
            raise RuntimeError(f"Metamon blocked mirrored-pair team {path}")
        draw_log.append(os.path.basename(path))
        return candidate

    wrappers.TeamSet.yield_team = sequence_yield_team
    print(f"[peer] team draw = the mirrored-pair SEQUENCE {sequence_path} ({len(files)} battles)", flush=True)


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


def _flash_window():
    """The ``(left, right)`` window the parsed gin config gave ``FlashAttention`` — ``(-1, -1)``
    (flash-attn's "full attention") when nothing bound it."""
    import gin

    try:
        return tuple(gin.query_parameter("transformer.FlashAttention.window_size"))
    except ValueError:
        return (-1, -1)


def make_windowed_vanilla_attention(left):
    """``VanillaAttention`` restricted to flash-attn's sliding window ``(left, 0)``.

    Eager (no ``torch.compile``): the masks differ per call length and this is B = 1 CPU work. The
    arithmetic is ``VanillaAttention``'s own, line for line, plus one extra mask term.
    """
    import math

    import torch
    from amago.nets.transformer import VanillaAttention

    class WindowedVanillaAttention(VanillaAttention):
        window_left = int(left)

        def _inference_with_cache(self, qkv, key_cache, val_cache, cache_seqlens):
            queries, keys, values = torch.unbind(qkv, dim=2)
            B, L, H, E = queries.shape
            assert L == 1
            scale = 1.0 / math.sqrt(E)
            cache_idxs = torch.arange(key_cache.shape[0], device=key_cache.device)
            key_cache[cache_idxs, cache_seqlens] = keys[:, 0]
            val_cache[cache_idxs, cache_seqlens] = values[:, 0]
            end = cache_seqlens + 1
            max_len = end.max()
            k_cache = torch.nan_to_num(key_cache[:, :max_len])
            v_cache = torch.nan_to_num(val_cache[:, :max_len])
            scores = scale * torch.einsum("blhe,blhe->blh", queries, k_cache)
            pos = torch.arange(max_len, device=cache_seqlens.device)[None, :]
            # the query sits at index cache_seqlens; it sees keys [q - left, q]
            mask = (pos >= end[:, None]) | (pos < (cache_seqlens[:, None] - self.window_left))
            scores.masked_fill_(mask[:, :, None], -torch.inf)
            A = self.dropout(torch.softmax(scores, dim=1))
            return torch.einsum("blh,blhd->bhd", A, v_cache).unsqueeze(1)

        def _forward_without_cache(self, qkv, mask):
            queries, keys, values = torch.unbind(qkv, dim=2)
            B, L, H, E = queries.shape
            scale = 1.0 / math.sqrt(E)
            scores = torch.einsum("blhe,bshe->bhls", queries, keys)
            # `mask` is VanillaAttention's causal triu; add the keys older than the window
            too_old = torch.tril(torch.ones((L, L), dtype=torch.bool, device=qkv.device),
                                 diagonal=-(self.window_left + 1))
            scores.masked_fill_(mask | too_old[None, None], -torch.inf)
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            return torch.einsum("bhls,bshd->blhd", A, values)

    return WindowedVanillaAttention


def install_cpu_attention(choice, record):
    """Re-bind the trajectory encoder's attention AFTER the model's gin files are parsed.

    See the module docstring: the ``gin_overrides`` dict loses to a file binding, and a sliding
    window must survive the swap. ``record`` receives ``attention`` and ``attention_window``.
    """
    import amago.cli_utils
    import gin
    import amago.nets.transformer as _tf

    inner = amago.cli_utils.use_config

    def use_config(custom_params, gin_configs=None, finalize=True):
        inner(custom_params, gin_configs, finalize=False)
        left, right = _flash_window()
        if right not in (0, -1):
            raise SystemExit(f"FlashAttention.window_size={(left, right)}: a right-hand window "
                             "has no CPU equivalent here")
        if left < 0:
            attn = {"vanilla": _tf.VanillaAttention, "flex": _tf.VanillaFlexAttention}[choice]
        elif choice == "vanilla":
            attn = make_windowed_vanilla_attention(left)
        else:
            raise SystemExit(f"--attention {choice} has no sliding-window form; this model's gin "
                             f"sets FlashAttention.window_size={(left, right)} — use vanilla")
        gin.bind_parameter("traj_encoders.TformerTrajEncoder.attention_type", attn)
        record["attention"] = attn.__name__
        record["attention_window"] = [left, right]
        print(f"[peer] attention -> {attn.__name__} window={(left, right)} "
              "(CPU; flash-attn is CUDA-only)", flush=True)
        if finalize:
            gin.finalize()

    amago.cli_utils.use_config = use_config


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
    ap.add_argument("--team-sequence", default=None,
                    help="MIRRORED TEAM PAIRS: a JSON list of team files, one per battle, in order "
                         "(overrides --team-seed)")
    ap.add_argument("--regime", choices=("greedy", "t1"), required=True,
                    help="greedy = Agent.get_actions(sample=False), i.e. argmax; "
                         "t1 = Metamon's shipped default (sample, action_temperature 1.0)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="action_temperature for t1; IGNORED by greedy (see the module docstring: "
                         "MetamonDiscrete clips probabilities and cannot express it)")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--battle-backend", default=None,
                    help="default: the model's own (SmallRL/SyntheticRLV2 want 'poke-env', Kakuna 'metamon')")
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
    if args.team_sequence is not None:
        install_sequence_team_draw(args.team_sequence, draw_log)
    elif args.team_seed is not None:
        install_seeded_team_draw(args.team_seed, draw_log)

    model = get_pretrained_model(args.agent)
    backend = args.battle_backend or model.battle_backend

    attention_record = {"attention": "FlashAttention" if args.attention == "flash" else None,
                        "attention_window": None}
    if args.attention != "flash":
        install_cpu_attention(args.attention, attention_record)

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
        **attention_record,
    })
    # Metamon's own scoreboard, complementing ours — the independent cross-check that the win
    # accounting is right. Its `won` field is a BOOLEAN, so it books a TIE as its own loss
    # (de-risk H3); our side's poke-env flags are the authority.
    if isinstance(results, dict):
        timing["peer_results"] = {str(k): str(v) for k, v in results.items()}

    expect_sample = [sample]
    ok_kw = timing["sample_kwarg_values"] == expect_sample
    rate = timing.get("argmax_match_rate")
    n_dec = timing.get("n_decisions") or 0
    # 🚨 TWO QUESTIONS, TWO FIELDS (gen3_anchor_regime_split_v1). "Did the regime take effect on
    # every decision?" and "did this process exit cleanly?" are different facts with different
    # consequences, and ANDing them cost the 2026-09-20 campaign 31 of 84 sub-cells: the peer's
    # post-game RecursionError set `error`, the composite went False, and every per-decision
    # argmax rate in those cells was 1.0000. A flag that reads FALSE on a clean regime is a flag
    # a reader learns to ignore.
    #
    # `ok_rate` is REGIME-APPROPRIATE, not literally "== 1.0": greedy demands exactly 1.0, t1
    # demands materially below it (the positive control that the instrument has any power). What
    # it no longer does is pass VACUOUSLY — `rate is None` or zero decisions now FAILS, because a
    # check that never ran reads exactly like one that passed.
    ok_rate = (rate is not None and n_dec >= 1
               and (rate == 1.0 if args.regime == "greedy" else rate < 1.0))
    timing["regime_verified_decisions"] = bool(ok_kw and ok_rate)
    timing["peer_error_free"] = error is None
    #: DEPRECATED (one release, gen3_anchor_regime_split_v1). The composite AND of the two fields
    #: above. Kept so an existing reader is not silently re-pointed at a different quantity; read
    #: `regime_verified_decisions` for the regime and `peer_error_free`/`peer_clean` for the exit.
    timing["regime_check_ok"] = bool(timing["regime_verified_decisions"] and error is None)
    print(f"[peer] REGIME CHECK regime={args.regime} "
          f"sample_kwargs={timing['sample_kwarg_values']} (ok={ok_kw}) "
          f"argmax_match_rate={rate} over {n_dec} decisions (ok={ok_rate}) "
          f"| peer_error_free={timing['peer_error_free']}", flush=True)
    if not timing["regime_verified_decisions"]:
        print("[peer] 🚨 REGIME CHECK FAILED — this cell is NOT a measurement", flush=True)
    elif not timing["peer_error_free"]:
        print("[peer] ⚠️ the regime VERIFIED on every decision, but this process did not exit "
              "cleanly — the games stand, the process does not", flush=True)

    with open(args.report_out, "w") as fh:
        json.dump(timing, fh, indent=1, default=str)
    print("[peer] REPORT " + json.dumps({k: v for k, v in timing.items()
                                         if k != "team_draws"}, default=str), flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
