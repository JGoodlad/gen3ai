"""Shared helpers of the static-port identity proof — imported by every script here AFTER it has put the
checkout under test's ``src/`` first on ``sys.path`` (``bootstrap``). Pure plumbing: nothing here is
checkout-specific except through the code it imports.

The configuration proved (design_static_tokens.md §8.2, the screen's static arm): the production surface
(`production_args()` = a fresh `--arch production`) with ``token_encoding='static'``, belief ``fixed_mass``
(a flag at the pin, the only belief representation at HEAD), readout ``tower``, ``move_resolution='off'``,
``value_threat_inject`` on, ``speed_physics='off'``; at HEAD also ``obs_facts='off'`` and ``op_reduction='max'``.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "gen3ai" / "static_port_identity"
FORWARD_KEYS = ("values", "log_prob", "entropy", "masked_logp", "masks_bool")


def bootstrap(argv: List[str]) -> Tuple[str, List[str]]:
    """Pop ``--src PATH`` from argv and put it FIRST on sys.path (a direct `python script.py` would otherwise
    import the main checkout's code through the editable install). CPU only."""
    if "--src" not in argv:
        raise SystemExit("--src <checkout>/src is required (the checkout whose code is under test)")
    i = argv.index("--src")
    src = str(Path(argv[i + 1]).resolve())
    rest = argv[:i] + argv[i + 2:]
    sys.path.insert(0, src)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return src, rest


def checkout_of(src: str) -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=src).stdout.strip()


def static_args() -> Tuple[Any, Dict[str, Any]]:
    """The screen's static arm at the running checkout. Returns (args, the overrides applied)."""
    from main.train.production_args import production_args
    a = production_args()
    applied: Dict[str, Any] = {"token_encoding": "static"}
    if hasattr(a, "belief_tokens"):           # the pin: X5 is a lever (production `blob`)
        applied["belief_tokens"] = "fixed_mass"
    if hasattr(a, "obs_facts"):               # HEAD: the OBS-FACTS lever, OFF
        applied["obs_facts"] = "off"
    if hasattr(a, "op_reduction"):            # HEAD: F6b's lever, `max`
        applied["op_reduction"] = "max"
    for k, v in applied.items():
        setattr(a, k, v)
    want = {"policy_readout": "tower", "move_resolution": "off", "value_threat_inject": True,
            "speed_physics": "off"}
    got = {k: getattr(a, k) for k in want}
    if got != want:
        raise SystemExit(f"REFUSED: the production surface is not the screen's static arm: {got} != {want}")
    return a, applied


def build(args: Any) -> Any:
    """The K9 golden's seeded, name-keyed-perturbed learner on ``args`` (one thread)."""
    from agents.training import learner_golden as LG
    return LG.build_learner(args=args, perturb_keyed=True)


def assert_static_fixed_mass_tower(fe: Any) -> Dict[str, Any]:
    """The built extractor IS static × fixed_mass × tower (read off the modules, not the args)."""
    from agents.model.static_tokens import StaticTokenEncoder
    facts = {
        "token_encoding": getattr(fe, "token_encoding", None),
        "pokemon_encoder": type(fe.pokemon_encoder).__name__,
        "hypothesis_builder": type(getattr(fe, "hypothesis_builder", None)).__name__,
        "flat_intent_head": type(getattr(fe, "flat_intent_head", None)).__name__,
        "policy_readout": getattr(fe, "policy_readout", None),
        "op_content": type(getattr(fe, "op_content", None)).__name__,
        "static_board": bool(getattr(fe.team_transformer, "static_board", False)),
        "move_resolution": type(getattr(fe, "move_resolution", None)).__name__,
        "op_reduction": getattr(fe.damage_op, "op_reduction", "<absent: pre-v146>"),
        "obs_facts_inject": type(getattr(fe, "obs_facts_inject", None)).__name__,
    }
    ok = (facts["token_encoding"] == "static" and isinstance(fe.pokemon_encoder, StaticTokenEncoder)
          and facts["hypothesis_builder"] == "HypothesisBuilder" and facts["flat_intent_head"] == "FlatIntentHead"
          and facts["policy_readout"] == "tower" and facts["static_board"] and facts["op_content"] == "OpContent"
          and facts["obs_facts_inject"] == "NoneType")
    if not ok:
        raise SystemExit(f"REFUSED: the build is not static x fixed_mass x tower: {facts}")
    return facts


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sd_sha(sd: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def maxdiff(a: Any, b: Any) -> float:
    import torch as th
    if a.dtype == th.bool:
        return float((a != b).sum())
    ad, bd = a.double(), b.double()
    fin = th.isfinite(ad) & th.isfinite(bd)
    same_nonfin = (ad == bd) & ~fin
    if bool((~fin & ~same_nonfin).any()):
        return float("inf")
    return float((ad - bd).abs()[fin].max()) if bool(fin.any()) else 0.0


def forward(pol: Any, obs: Dict[str, Any], acts: Any, masks: Any) -> Dict[str, Any]:
    import torch as th
    pol.set_training_mode(True)
    with th.no_grad():
        out = pol.evaluate_actions_functional(obs, acts, masks)
    return dict(zip(FORWARD_KEYS, out))
