"""The `--model` ARCHITECTURE check, shared by the trainer, `main.checkargs` and `launcher --dry-run`.

WHY (F-BI-2, `designs/endstate/design_bot_ideas_screen.md`, 2026-10-19). A fork typing an
architecture flag its parent was not trained with (a toy fork with `--trunk-layers 4`) read "✓ this
command still launches" in `checkargs` and "would launch" in the dry run, then died at the child's
start with `ModelVersionError: trunk_layers mismatch`. The comparison lived only in the child.

ONE IMPLEMENTATION. `resume_model_version` is the trainer's own construction of the current-code
`ModelVersion` for a `--model` load (`main.train.model_build.build_and_train` calls it), and the
verdict calls the trainer's own `ModelVersion.check_compatible` on the parent's recorded
`model_config.json`, found the way `agents.model.snapshot.load_model_snapshot` finds it. Nothing here
compares a field by hand, so the dry-run refusal text IS the child's FATAL text.

DECLARED CHANGES (the seam for B-GROW, `--grow-trunk-layers`). A fork may change an architecture
field on purpose only through a declared growth option. `DECLARED_GROWTH` maps such an option's
namespace dest to the `ModelVersion` fields it is allowed to change; when the option is set, those
fields are taken from the current version before `check_compatible` runs (the same `replace` move
`check_opponent_snapshot_compatible` makes for the ride-along heads). It is EMPTY today: no growth
option exists on main. The unit that adds one adds its row here AND makes the trainer's load path
honour the same table, or the dry run and the child disagree again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Optional

#: namespace dest of a declared growth option -> the ModelVersion fields it may change on a fork.
#: EMPTY until B-GROW lands (see the module docstring). Never a hand list of arch keys otherwise.
DECLARED_GROWTH: dict[str, tuple[str, ...]] = {}


def resume_model_version(args: Any, *, mappings: Any = None, reward_config: Any = None) -> Any:
    """The current-code `ModelVersion` a `--model` load is checked against (`resume_version_build`)."""
    return resume_version_build(args, mappings=mappings, reward_config=reward_config)[0]


def resume_version_build(args: Any, *, mappings: Any = None,
                         reward_config: Any = None) -> tuple[Any, Any, dict[str, Any]]:
    """``(version, layout, policy_kwargs)`` for a `--model` load — THE trainer's build.

    `args` is the resolved training namespace. `mappings` / `reward_config` default to what the
    trainer builds from the same namespace (`load_mappings()`, `RewardConfig.from_args(args)`).
    The layout and policy kwargs feed the trainer's round-trip smoke after the load."""
    from agents.model.extractor_arch import build_extractor_arch_kwargs
    from agents.model.features_extractor import NET_ARCH, Gen3FeaturesExtractor
    from agents.model.model_version import ModelVersion
    from agents.model.policy import POLICY_ACTIVATION_FN
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    if mappings is None:
        mappings = load_mappings()
    if reward_config is None:
        from agents.training.reward_config import RewardConfig
        reward_config = RewardConfig.from_args(args)
    load_encoder = Gen3ObservationEncoder(mappings)
    # ONE source of truth for every version-checked arch toggle
    # (agents.model.extractor_arch.build_extractor_arch_kwargs). The fresh-run path builds the SAME
    # dict from the SAME table, so a toggle cannot land on one path and not the other.
    extractor_kwargs = build_extractor_arch_kwargs(
        args, base=load_encoder.get_features_extractor_kwargs())
    policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": NET_ARCH,
        # gen3_policy_activation_pin_v1: carried for PARITY with the fresh-run dict, so the two
        # policy_kwargs sites cannot drift. ModelVersion records no activation field, and SB3
        # rebuilds the loaded policy from the ZIP's own saved policy_kwargs.
        "activation_fn": POLICY_ACTIVATION_FN,
        "critic": args.critic,  # gen3_winprob_critic_mode_v1: WHICH readout is the critic
    }
    version = ModelVersion.from_layout_and_policy_kwargs(
        extractor_kwargs["layout"], policy_kwargs, vf_coef=args.vf_coef,
        reward_config=reward_config,
        opp_belief_aux_coef=args.opp_belief_aux_coef,
        move_belief_coef=args.move_belief_coef,
        move_belief_latent_coef=args.move_belief_latent_coef,
        spread_belief_coef=args.spread_belief_coef,
        hp_type_belief_coef=args.hp_type_belief_coef,
        item_belief_coef=args.item_belief_coef,
        arch_source=getattr(args, "arch_source", None),
        policy_grad_coef=args.policy_grad_coef,
        intent_label_bot_weight=args.intent_label_bot_weight,
        policy_gae_lambda=args.policy_gae_lambda,
        diagnostics_every=args.diagnostics_every,
        opp_intent_coef=float(args.opp_intent_coef or 0.0),
        fork_fraction=args.fork_fraction,
        fork_branches=args.fork_branches,
        fork_contested_gap=args.fork_contested_gap,
        fork_contested_absv=args.fork_contested_absv,
        fork_max_per_battle=args.fork_max_per_battle,
        fork_crn=args.fork_crn,
        # gen3_eval_sentinel_greedy_default_v1 (v112): the EVAL REGIME + the gate it derives, so a
        # flagless resume reads its own regime back. Both are RESOLVED by `resolve_config`.
        eval_sentinel_greedy=args.eval_sentinel_greedy,
        promote_threshold=args.promote_threshold,
        eval_mirrored_pairs=args.eval_mirrored_pairs,
        promotion_sprt=args.promotion_sprt,
        trunk_precision=args.trunk_precision,
        canary_shift_max=args.canary_shift_max,
        capacity_telemetry=args.capacity_telemetry,
        canary_reset_steps=args.canary_reset_steps,
        capacity_cosine_every=args.capacity_cosine_every,
        capacity_velocity_every=args.capacity_velocity_every,
        rank_tripwire=args.rank_tripwire,
        rank_tripwire_drop=args.rank_tripwire_drop,
    )
    return version, extractor_kwargs["layout"], policy_kwargs


def declared_changes(ns: Any) -> tuple[str, ...]:
    """The ModelVersion fields this namespace DECLARES it changes (a set growth option)."""
    out: list[str] = []
    for dest, fields in DECLARED_GROWTH.items():
        if getattr(ns, dest, None):
            out.extend(fields)
    return tuple(out)


def check_against_saved(current: Any, saved: Any, ns: Any) -> None:
    """`current.check_compatible(saved)`, after the declared growth fields are taken as current.
    Raises the trainer's own `ModelVersionError`."""
    allowed = declared_changes(ns)
    if allowed:
        saved = replace(saved, **{k: getattr(current, k) for k in allowed})
    current.check_compatible(saved)


@dataclass(frozen=True)
class ForkArchVerdict:
    """The verdict on one `--model` argv. `refused` ⇔ the child would die `[ModelVersion] FATAL`
    on the architecture. `checked` False ⇔ it could not be judged here (`note` says why)."""
    checked: bool
    refused: bool
    reason: Optional[str] = None
    config_path: Optional[str] = None
    note: Optional[str] = None

    def lines(self, *, advisory: bool = False) -> list[str]:
        if not self.checked:
            return [f"⚠️  fork arch  : NOT checked here — {self.note} (the child checks it at start)"]
        if not self.refused:
            return [f"✓ fork arch  : compatible with {self.config_path} (check_compatible, the "
                    "trainer's own check)"]
        mark = ("ℹ️  advisory (CURRENT tree, NOT the pinned one)" if advisory
                else "✗ REFUSED (architecture vs the --model checkpoint, FATAL_CONFIG)")
        out = [f"{mark}: the child would die `[ModelVersion] FATAL` at start against "
               f"{self.config_path}:"]
        out += [f"      {ln}" for ln in (self.reason or "").splitlines() if ln.strip()]
        return out


def resolve_like_the_trainer(argv: list[str], model_path: str) -> Any:
    """The namespace the CHILD builds from `argv`: the trainer's parser, then
    `main.train.config.resolve_config` — the function the trainer runs, as `production_args` runs it
    (its launch banner captured). `--model` is set to `model_path`, the resolved checkpoint. A refusal
    of `resolve_config` raises `SystemExit`; the caller reports it as "not judged here"."""
    import contextlib
    import io

    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        # parse_known_args: a checkargs argv may carry the LAUNCHER-owned flags, which never reach the child
        args, _launcher_only = parser.parse_known_args(list(argv))
        args.model = model_path
        resolve_config(args, parser)
    return args


def _first_line(e: BaseException) -> str:
    return (str(e).splitlines() or [type(e).__name__])[0][:140]


def fork_arch_verdict(argv: list[str], model_path: Optional[str]) -> Optional[ForkArchVerdict]:
    """Judge a `--model` argv against the checkpoint's recorded config, as the child will.

    None when there is no `--model`. `checked=False` (never a refusal) when the config is absent
    (the trainer skips that load check with a WARNING), unreadable under THIS tree, or when the
    argv does not resolve here at all — each of those has its own checkargs finding (an unknown flag,
    a refused combination, the version break, the retired levers)."""
    if not model_path:
        return None
    from agents.model.model_version import ModelVersion, ModelVersionError
    from agents.model.snapshot import _resolve_paths, _run_config_path

    try:
        _zip, config_dir = _resolve_paths(model_path)
    except FileNotFoundError as e:
        return ForkArchVerdict(False, False, note=f"no checkpoint zip ({_first_line(e)})")
    config_path = _run_config_path(config_dir)
    if not os.path.exists(config_path):
        return ForkArchVerdict(False, False, note=f"no model_config.json beside {model_path}")
    try:
        saved = ModelVersion.from_json_file(config_path)
    except Exception as e:  # noqa: BLE001 — any read failure is "not judged here", never a guess
        return ForkArchVerdict(False, False, config_path=config_path,
                               note=f"{config_path} is not readable under this tree ({_first_line(e)})")
    try:
        ns = resolve_like_the_trainer(argv, model_path)
    except SystemExit:
        return ForkArchVerdict(False, False, config_path=config_path,
                               note="the trainer's resolve_config refuses this argv first (see the other findings)")
    try:
        current = resume_model_version(ns)
    except Exception as e:  # noqa: BLE001
        return ForkArchVerdict(False, False, config_path=config_path,
                               note=f"the current version could not be built ({type(e).__name__}: {_first_line(e)})")
    try:
        check_against_saved(current, saved, ns)
    except ModelVersionError as e:
        return ForkArchVerdict(True, True, reason=str(e), config_path=config_path)
    return ForkArchVerdict(True, False, config_path=config_path)
