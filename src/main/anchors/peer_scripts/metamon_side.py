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
``ws://localhost:8000`` — our shared DEV server, one port from the reserved training-tools server on 8001.
There is no CLI flag, no env var and no constructor argument. We rebind the module global before
any env is constructed (the property reads it at call time) and refuse 8000/8001 in code.

WEIGHTS ARE PINNED (FR-1 FINDING 3)
-----------------------------------
Metamon fetches a checkpoint with ``hf_hub_download(repo_id, filename, cache_dir)``: no revision, online. This peer
forces ``HF_HUB_OFFLINE=1`` at import, replaces that call by ``weight_pins.pinned_downloader`` (the pinned revision,
the local cache only, sha256-checked) and hashes every pinned file before the model is built
(:func:`install_weight_pins`); a mismatch or missing file is ``SystemExit``. See ``EXTERNAL_ANCHORS_SOP.md``.

TEAMS ARE PACKED BY SHOWDOWN (gen3_peer_showdown_pack_v1, 2026-10-27)
--------------------------------------------------------------------
Upstream poke-env's packer is not Showdown's reading of a paste: a mis-ordered Hidden Power table (12 of 16 types
became another type; the server refused the team), ``Happiness: 0`` packed as 255 (Frustration at base power 1), a
``Hidden Power: <Type>`` line ignored. So every team file this peer may draw is packed through ``showdown_pack.js``
(import, the validator, ``Teams.pack``: what a real server plays) and checked by ``pack_oracle.js`` at startup
(``team_pack.py``); a file either fails is ``SystemExit`` naming it, before the model is built. Every draw
(seeded, sequence, Metamon's own unseeded one) yields that pack. The report's ``peer_pack_changed`` says whether any
file plays differently from the legacy pack; it enters the cell's regime id (``EXTERNAL_ANCHORS_SOP.md`` H22).

THE PEER'S CODE IS REPORTED (gen3_peer_code_v1, 2026-10-27)
----------------------------------------------------------
The report's ``peer_code`` (``peer_code.py``) names what this process executed: these scripts, the Metamon checkout it
imports (HEAD and any local change), this interpreter and its packages. The cell stamps its digest and the regime id
carries it unless it is a banked one (``main.reads.side_identity``).

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
import threading
import time

# Before torch: a GPU this process can see is a GPU it may grab, and a training arm owns it.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ...and before torch for the same reason: the thread count is read at import. B=1 CPU inference
# with the default thread pool measured 210% CPU per peer on a shared box, which is two cores of
# synchronisation buying nothing. The driver also sets these in the peer env; this is the second
# lock, for anyone running the script by hand.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _load_sibling(filename, name):
    """A stdlib-only module beside this file (``weight_pins.py``, ``team_pack.py``), loaded BY PATH: the script runs
    with ``PYTHONPATH=""`` and may itself be imported by a test under another name, so a sibling ``import`` cannot be
    relied on."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WP = _load_sibling("weight_pins.py", "metamon_weight_pins")
TP = _load_sibling("team_pack.py", "metamon_team_pack")
PC = _load_sibling("peer_code.py", "metamon_peer_code")
#: ``{abspath: packed}`` for every team file this peer may draw: packed through Showdown and verified at startup
#: (:func:`install_showdown_pack`). Every yield reads it; a file not in it never plays.
_PACKS = {}
# Before ANY import of huggingface_hub (it reads these into constants at import): the weights come from the
# local cache at the pinned revision or not at all. Overrides whatever the caller exported.
WP.force_offline()

RESERVED_PORTS = {8000: "the shared DEV server",
                  8001: "the reserved TRAINING server (the trainer itself is in-process and connects to none)"}


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


def install_seeded_team_draw(seed, draw_log, skip=0):
    """Give ``TeamSet.yield_team`` a PRIVATE seeded RNG over a SORTED file list.

    Metamon's shipped ``yield_team`` calls ``random.choice(self.team_files)`` on the process-global
    ``random`` module over a file list whose order comes from a directory walk. Neither is
    reproducible across processes, so a "seeded" series would not in fact repeat. Patched at CLASS
    level (``PokeEnvWrapper`` deep-copies a team set on one of its paths) and preserving
    ``_most_recent_team_file``, which is what Metamon's own per-battle CSV records.

    ``skip`` (``--team-skip``, S-U's per-game units): the stream's first ``skip`` draws are made and DISCARDED (unlogged)
    the first time the set draws, so its first battle gets the team the same seed's draw ``skip`` gave game ``skip`` of
    a whole series. The draw itself, and the play, are unchanged.
    """
    import metamon.env.wrappers as wrappers

    def draw(self, log):
        for _ in range(100):
            path = self._draw_rng.choice(self._sorted_files)
            self._most_recent_team_file = path
            candidate = TP.pack_of(_PACKS, path)
            if not self.block_team(candidate):
                if log:
                    draw_log.append(os.path.basename(path))
                return candidate
        raise RuntimeError("Could not find valid team after 100 attempts")

    def seeded_yield_team(self):
        files = getattr(self, "_sorted_files", None)
        if files is None:
            self._sorted_files = sorted(self.team_files)
            self._draw_rng = random.Random(seed)
            for _ in range(int(skip)):
                draw(self, log=False)
        return draw(self, log=True)

    wrappers.TeamSet.yield_team = seeded_yield_team
    print(f"[peer] team draw seeded with {seed} over a SORTED file list"
          + (f", its first {int(skip)} draw(s) skipped" if skip else ""), flush=True)


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
        candidate = TP.pack_of(_PACKS, path)
        if self.block_team(candidate):
            raise RuntimeError(f"Metamon blocked mirrored-pair team {path}")
        draw_log.append(os.path.basename(path))
        return candidate

    wrappers.TeamSet.yield_team = sequence_yield_team
    print(f"[peer] team draw = the mirrored-pair SEQUENCE {sequence_path} ({len(files)} battles)", flush=True)


def install_default_team_draw():
    """The UNSEEDED draw (no ``--team-seed`` / ``--team-sequence``; the driver always passes a seed, a run by hand may
    not): Metamon's own ``yield_team`` (``random.choice`` over its file list, one draw per attempt, the same global
    RNG use), except that the pack is the verified Showdown pack (:data:`_PACKS`), never poke-env's."""
    import warnings

    import metamon.env.wrappers as wrappers

    def default_yield_team(self):
        for attempt in range(100):
            path = random.choice(self.team_files)
            self._most_recent_team_file = path
            candidate = TP.pack_of(_PACKS, path)
            if not self.block_team(candidate):
                return candidate
            if attempt % 10 == 0:
                warnings.warn(f"Failed to find valid team after {attempt + 1} attempts")
        raise RuntimeError("Could not find valid team after 100 attempts")

    wrappers.TeamSet.yield_team = default_yield_team


def legacy_pack(text):
    """How this peer packed a team file before ``gen3_peer_showdown_pack_v1``: upstream poke-env's ``Teambuilder``
    (what Metamon's shipped ``TeamSet.yield_team`` calls). Compared, never played: :func:`install_showdown_pack`."""
    from poke_env.teambuilder.teambuilder import Teambuilder

    return Teambuilder.join_team(Teambuilder.parse_showdown_team(text))


def install_showdown_pack(files, battle_format, bridge_dir):
    """Pack EVERY team file this peer may draw through Showdown (``team_pack.build``) before the first battle, or
    ``SystemExit`` naming the file and the set. Returns the record the report carries (``peer_pack_changed``: does any
    file play differently from the legacy pack; it enters the cell's regime id)."""
    try:
        packs, record = TP.build(files, battle_format, bridge_dir, legacy_pack=legacy_pack)
    except TP.TeamPackError as exc:
        raise SystemExit(f"[peer] FATAL team pack: {exc}") from None
    _PACKS.clear()
    _PACKS.update(packs)
    print(f"[peer] team pack {record['schema']}: {record['n_files']} file(s) packed through Showdown and verified; "
          f"{len(record['changed'])} play differently from the legacy pack "
          f"({', '.join(c['file'] for c in record['changed'][:5]) or 'none'})", flush=True)
    return record


def _rng_states():
    """The process-global RNGs a router could draw from whose STATE is compared: torch's CPU generator and
    NumPy's. Python's ``random`` is checked per THREAD instead (:func:`_install_python_draw_guard`)."""
    import numpy as np
    import torch

    return torch.get_rng_state(), np.random.get_state()


#: Python ``random``'s module-level draw functions. Their global state is SHARED with poke-env's event-loop thread,
#: which draws while the router computes (measured 2026-10-25: 1 call in ~23 moved it, always ``python`` only), so
#: a state compare would blame the router for another thread's draw.
_PY_DRAWS = ("random", "uniform", "triangular", "randint", "randrange", "choice", "choices", "shuffle", "sample",
             "gauss", "normalvariate", "lognormvariate", "expovariate", "vonmisesvariate", "gammavariate",
             "betavariate", "paretovariate", "weibullvariate", "getrandbits", "randbytes", "binomialvariate")
_ROUTER_THREAD = threading.local()
_PY_DRAWS_IN_ROUTER = [0]


def _install_python_draw_guard():
    """Wrap each ``random.<draw>`` so a draw made by the thread that is INSIDE a router call is counted
    (``_PY_DRAWS_IN_ROUTER``); draws from other threads pass through uncounted. Idempotent. (A module that bound a
    draw by ``from random import …`` before this ran is not seen; Metamon's router imports no ``random`` at all.)"""
    if getattr(random, "_gen3ai_draw_guard", False):
        return
    for name in _PY_DRAWS:
        inner = getattr(random, name, None)
        if inner is None:
            continue

        def draw(*a, _inner=inner, **k):
            if getattr(_ROUTER_THREAD, "active", False):
                _PY_DRAWS_IN_ROUTER[0] += 1
            return _inner(*a, **k)

        setattr(random, name, draw)
    random._gen3ai_draw_guard = True


def _rng_moved(a, b):
    """The names of the RNGs whose state differs between two :func:`_rng_states` snapshots ([] = none)."""
    import numpy as np
    import torch

    (ta, na), (tb, nb) = a, b
    moved = []
    if not torch.equal(ta, tb):
        moved.append("torch")
    if not (na[0] == nb[0] and np.array_equal(na[1], nb[1]) and tuple(na[2:]) == tuple(nb[2:])):
        moved.append("numpy")
    return moved


def wrap_router(cls, probe):
    """Instrument an ENSEMBLE ROUTER's ``get_actions`` (Metamon's ``HeuristicRouterEnsemblePolicy``).

    The router is not an ``amago`` ``Agent``, so the argmax probe never sees it, and it has no argmax: it
    scores its members' proposals and picks one. What makes it GREEDY-equivalent is that it is
    DETERMINISTIC: it ignores the ``sample`` keyword and draws no random number. So each call records the
    ``sample`` keyword, its time, and whether every RNG state is byte-equal before and after
    (``router_no_rng_rate``, exactly 1.0 required). A router that sampled would move a state and fail. Python's
    ``random`` is shared with poke-env's thread, so for it the check is per thread: no draw made BY THE ROUTER'S
    THREAD during the call (:func:`_install_python_draw_guard`).
    """
    inner = cls.get_actions
    _install_python_draw_guard()

    def get_actions(self_, *a, **k):
        probe.sample_kwargs.add(bool(k.get("sample", True)))
        before = _rng_states()
        py_before = _PY_DRAWS_IN_ROUTER[0]
        _ROUTER_THREAD.active = True
        t0 = time.perf_counter()
        try:
            return inner(self_, *a, **k)
        finally:
            _ROUTER_THREAD.active = False
            probe.samples.append(time.perf_counter() - t0)
            moved = _rng_moved(before, _rng_states())
            if _PY_DRAWS_IN_ROUTER[0] != py_before:
                moved.append("python")
            if moved:
                probe.router_rng_moved.append({"call": probe.n_router_checked, "rngs": moved})
            probe.n_router_checked += 1
            probe.n_router_no_rng += int(not moved)

    cls.get_actions = get_actions


def decisions_verified(regime, rate, n_dec, router_rate=None, n_router=0):
    """The per-decision regime rule. A single policy: greedy demands an argmax rate of exactly 1.0, t1
    materially below it (the positive control). An ENSEMBLE router (``n_router >= 1``): greedy demands a
    no-RNG rate of exactly 1.0 on every call; t1 is never verified, because the router cannot sample."""
    if n_router >= 1:
        return regime == "greedy" and router_rate == 1.0
    return (rate is not None and n_dec >= 1
            and (rate == 1.0 if regime == "greedy" else rate < 1.0))


def preset_sha256(members):
    """The identity of an ensemble preset: sha256 over its member list's canonical JSON."""
    import hashlib

    return hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_preset(model, expect_name, expect_sha):
    """Refuse to play unless ``model`` (a built Metamon nickname ensemble) routes over EXACTLY the pinned preset:
    the preset file's entry hashes to the pin, and the member specs the model will build equal that entry."""
    import dataclasses

    import metamon.rl.experimental.ensemble as ens

    with open(os.path.join(os.path.dirname(ens.__file__), "ensemble_presets.json")) as fh:
        members = json.load(fh).get(expect_name)
    got = preset_sha256(members) if members is not None else None
    specs = [{**dataclasses.asdict(m), "proposal_roles": list(m.proposal_roles)}
             for m in getattr(type(model), "MEMBER_SPECS", None) or []]
    built = preset_sha256(specs) if specs else None
    if got != expect_sha or built != expect_sha:
        raise SystemExit(f"ensemble {type(model).__name__}: preset file {expect_name!r} sha256 {got}, built members "
                         f"sha256 {built}; the pin is {expect_sha}")
    print(f"[peer] ensemble preset {expect_name} sha256 {got} == the members it builds (pinned)", flush=True)
    return {"preset": expect_name, "preset_sha256": got}


class RegimeProbe:
    """Per-decision timing + the regime-verification instruments (argmax for a policy, no-RNG for a router)."""

    def __init__(self, verify=True):
        self.samples = []
        self.verify = verify
        self.sample_kwargs = set()
        self.n_checked = 0
        self.n_argmax_match = 0
        self.n_router_checked = 0
        self.n_router_no_rng = 0
        self.router_rng_moved = []
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
        try:
            from metamon.rl.experimental.ensemble.ensemble import HeuristicRouterEnsemblePolicy
        except ImportError:
            HeuristicRouterEnsemblePolicy = None
        if HeuristicRouterEnsemblePolicy is not None:
            wrap_router(HeuristicRouterEnsemblePolicy, probe)
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
            # A router run has no argmax to compare with (its members' last actor call is not its choice).
            out["argmax_match_rate"] = (self.n_argmax_match / self.n_checked
                                        if self.n_checked and not self.n_router_checked else None)
        out["router_checked"] = self.n_router_checked
        out["router_no_rng"] = self.n_router_no_rng
        out["router_rng_moved"] = self.router_rng_moved
        out["router_no_rng_rate"] = (self.n_router_no_rng / self.n_router_checked
                                     if self.n_router_checked else None)
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


def peer_code_record(bridge_dir):
    """What this peer executes (``peer_code.py``, gen3_peer_code_v1): its scripts, the Metamon checkout ``metamon`` is
    imported from, this interpreter and its packages. An error is RECORDED, never raised: the harness voids the unit."""
    import metamon

    try:
        return PC.identity(os.path.dirname(os.path.abspath(__file__)), bridge_dir,
                           os.path.dirname(os.path.dirname(os.path.abspath(metamon.__file__))))
    except Exception as exc:                # noqa: BLE001 - recorded, and the unit does not count
        return {"schema": PC.SCHEMA, "error": f"{type(exc).__name__}: {exc}"}


def install_weight_pins(args):
    """Offline, pinned-revision, sha-verified weights (FR-1 FINDING 3): replace ``huggingface_hub.hf_hub_download``
    (which Metamon calls without a revision) by the pinned downloader, then hash EVERY pinned file BEFORE the model is
    built or a game played. Any unpinned file, missing file or hash mismatch raises ``SystemExit``."""
    import huggingface_hub
    import huggingface_hub.constants as hfc
    from metamon.rl.pretrained import MODEL_DOWNLOAD_DIR

    try:
        weights = json.loads(args.weights_json)
        WP.check_pins(args.hf_revision, weights)
        WP.assert_offline(hfc)
        loaded = []
        download = WP.pinned_downloader(huggingface_hub.hf_hub_download, repo=args.hf_repo,
                                        revision=args.hf_revision, weights=weights, loaded=loaded)
        verified = WP.verify_all(download, args.hf_repo, weights, MODEL_DOWNLOAD_DIR)
    except WP.WeightPinError as exc:
        raise SystemExit(f"REFUSING TO PLAY: {exc}")
    huggingface_hub.hf_hub_download = download
    loaded.clear()
    print(f"[peer] weights pinned: {args.hf_repo}@{args.hf_revision[:8]} offline, {len(verified)} file(s) sha256-verified "
          f"({WP.weights_digest(verified)[:12]})", flush=True)
    return {"hf_hub_offline": True, "hf_repo": args.hf_repo, "hf_revision": args.hf_revision,
            "weights_verified": verified, "weights_loaded": loaded}


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
    ap.add_argument("--team-skip", type=int, default=0,
                    help="with --team-seed: make and discard the seeded stream's first N draws, so the first battle "
                         "pilots the team game N of a whole seeded series drew (main.anchors --deal-game)")
    ap.add_argument("--team-sequence", default=None,
                    help="MIRRORED TEAM PAIRS: a JSON list of team files, one per battle, in order "
                         "(overrides --team-seed)")
    ap.add_argument("--regime", choices=("greedy", "t1"), required=True,
                    help="greedy = Agent.get_actions(sample=False), i.e. argmax; "
                         "t1 = Metamon's shipped default (sample, action_temperature 1.0)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="action_temperature for t1; IGNORED by greedy (see the module docstring: "
                         "MetamonDiscrete clips probabilities and cannot express it)")
    ap.add_argument("--expect-preset", default=None,
                    help="an ENSEMBLE agent: the member preset it must route over (refused otherwise)")
    ap.add_argument("--expect-preset-sha256", default=None)
    ap.add_argument("--hf-repo", required=True, help="the pinned Hugging Face repo the weights come from")
    ap.add_argument("--hf-revision", required=True, help="the pinned FULL commit sha of that repo")
    ap.add_argument("--weights-json", required=True,
                    help='JSON {"<file path in the repo>": "<sha256>"}: every weight file this agent may load')
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--battle-backend", default=None,
                    help="default: the model's own (SmallRL/SyntheticRLV2 want 'poke-env', Kakuna 'metamon')")
    ap.add_argument("--attention", default="vanilla", choices=("vanilla", "flex", "flash"))
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--bridge-dir", required=True,
                    help="our src/utils/bridge (showdown_pack.js, pack_oracle.js): every team file is packed and "
                         "verified through Showdown there (gen3_peer_showdown_pack_v1)")
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
    if args.team_skip and (args.team_seed is None or args.team_sequence is not None or args.team_skip < 0):
        raise SystemExit("--team-skip N needs --team-seed (and no --team-sequence), N >= 0")
    if args.team_sequence is not None:
        install_sequence_team_draw(args.team_sequence, draw_log)
    elif args.team_seed is not None:
        install_seeded_team_draw(args.team_seed, draw_log, skip=args.team_skip)
    else:
        install_default_team_draw()
    # Every team file packed through Showdown and verified BEFORE the weights and the model (minutes for Kakuna):
    # a bad file is a startup failure, never a game.
    team_set = metamon.env.get_metamon_teams(args.battle_format, args.team_set)
    if args.team_sequence is not None:
        with open(args.team_sequence) as fh:
            pack_files = list(json.load(fh))
    else:
        pack_files = list(team_set.team_files)
    pack_record = install_showdown_pack(pack_files, args.battle_format, args.bridge_dir)
    code_record = peer_code_record(args.bridge_dir)

    pin_record = install_weight_pins(args)
    model = get_pretrained_model(args.agent)
    preset_record = {}
    if args.expect_preset is not None:
        preset_record = check_preset(model, args.expect_preset, args.expect_preset_sha256)
    backend = args.battle_backend or model.battle_backend

    attention_record = {"attention": "FlashAttention" if args.attention == "flash" else None,
                        "attention_window": None}
    if args.attention != "flash":
        install_cpu_attention(args.attention, attention_record)

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
        "team_set": args.team_set, "team_seed": args.team_seed, "team_skip": args.team_skip,
        "action_temperature": args.temperature if sample else None,
        "n_team_draws": len(draw_log), "team_draws": draw_log,
        "error": error,
        **attention_record,
        **preset_record,
        **pin_record,
        "team_pack": pack_record,
        "peer_pack_changed": pack_record["peer_pack_changed"],
        "peer_code": code_record,
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
    # (`rate is not None and n_dec >= 1` for a policy; a router's no-RNG rate for an ensemble.)
    ok_rate = decisions_verified(args.regime, rate, n_dec, timing.get("router_no_rng_rate"),
                                 timing.get("router_checked") or 0)
    timing["regime_verified_decisions"] = bool(ok_kw and ok_rate)
    timing["peer_error_free"] = error is None
    #: DEPRECATED (one release, gen3_anchor_regime_split_v1). The composite AND of the two fields
    #: above. Kept so an existing reader is not silently re-pointed at a different quantity; read
    #: `regime_verified_decisions` for the regime and `peer_error_free`/`peer_clean` for the exit.
    timing["regime_check_ok"] = bool(timing["regime_verified_decisions"] and error is None)
    print(f"[peer] REGIME CHECK regime={args.regime} "
          f"sample_kwargs={timing['sample_kwarg_values']} (ok={ok_kw}) "
          f"argmax_match_rate={rate} router_no_rng_rate={timing.get('router_no_rng_rate')} "
          f"over {n_dec} decisions (ok={ok_rate}) "
          f"| peer_error_free={timing['peer_error_free']}", flush=True)
    if not timing["regime_verified_decisions"]:
        print("[peer] 🚨 REGIME CHECK FAILED — this cell is NOT a measurement", flush=True)
    elif not timing["peer_error_free"]:
        print("[peer] ⚠️ the regime VERIFIED on every decision, but this process did not exit "
              "cleanly — the unit is VOID if a long-tail line was logged (H21)", flush=True)

    with open(args.report_out, "w") as fh:
        json.dump(timing, fh, indent=1, default=str)
    print("[peer] REPORT " + json.dumps({k: v for k, v in timing.items()
                                         if k != "team_draws"}, default=str), flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
