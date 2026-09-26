"""The reward's DECLARATIONS — what a run is configured with, and what a turn's reward is made of.

**The reward is the TERMINAL alone.** The hand-shaped reward path — the eight PBRS potentials, the
~25 BIAS terms, the bias-additivity refund, the no-progress tax and the 14 flags that configured
them — was DELETED on 2026-09-26 (program_rust_core §4 M3 row; the flags are in
`designs/deleted_flags.md`). Production had trained on the terminal alone since the win-prob era
(`--critic winprob --terminal-indicator --victory-value 1.0`), and the Rust core's reward is the
win indicator.

* `RewardClass`     — the reward classes. ONE survives: TERMINAL.
* `RewardConfig`    — the per-run, resume-IMMUTABLE reward configuration, recorded in
                      `model_config.json` and reconstructed from it on every eval / resume path.
* `RewardBreakdown` — the per-turn record, and `_REGISTRY`: the field→class map the census and
                      the `reward/` export read.

A checkpoint TRAINED WITH SHAPING is recognised by `agents.model.model_version.shaped_reward`, so a
resume or fork of one refuses loudly instead of silently continuing on the terminal alone.

**Every public name here is re-exported by `reward_manager`**, so `from
agents.training.reward_manager import RewardConfig` still resolves.
"""
from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar

from agents.training.reward_weights import PBRS_GAMMA


class RewardClass(Enum):
    """The reward classes. Only TERMINAL survives the shaped-reward deletion (2026-09-26): the
    PBRS and BIAS classes went with every term they held."""
    TERMINAL = "terminal"


@dataclass
class RewardConfig:
    """Per-run reward configuration (resume-immutable; recorded in model_config.json).

    The TERMINAL's three knobs, the PPO discount, and the no-progress CLOCK's two switches — which
    live here because they are resume-immutable run facts read by BOTH the training env and the
    eval-side tracker, even though the clock now feeds only the `turns_since_progress` OBS scalar
    (its reward half, the no-progress tax, was deleted with the shaped path).
    """
    # Terminal reward for a DRAW / 250-turn timeout (no winner), under the SIGNED terminal
    # (`terminal_indicator=False`). More negative than a decisive loss makes stalling to the turn cap
    # strictly worse than losing cleanly. A decisive loss stays -victory_value. INAPPLICABLE under
    # `terminal_indicator` (refused there by `main.train.combination_checks` unless 0).
    draw_penalty: float = -35.0
    gamma: float = 0.9999
    # The no-progress clock's two intent-restoring fixes (both default OFF = byte-identical). The
    # mechanism lives on `ProgressClock`; turning either ON changes the `turns_since_progress` OBS
    # scalar — retrain-class (no dim moves, no ARCH_SIGNATURE bump), never a mid-run toggle.
    #   `progress_decision_tense` (F1) — point both window gates at the decision being judged.
    #   `progress_switch_freeze` (F2b) — a voluntary switch that fails the progress predicate
    #       FREEZES the clock instead of advancing it.
    progress_decision_tense: bool = False
    progress_switch_freeze: bool = False
    # The TERMINAL magnitude. A win scores +victory_value; under the SIGNED terminal a decisive loss
    # and a rare pre-cap tie score −victory_value and a 250-turn TIMEOUT scores `draw_penalty`.
    # Default 30.0 == `reward_weights.VICTORY_VALUE` (pinned equal by `reward_defaults_test.py`).
    victory_value: float = 30.0
    # gen3_winprob_critic_mode_v1 — the TERMINAL as a WIN INDICATOR, for the win-prob critic.
    # False: a win pays +victory_value, a decisive loss / pre-cap tie −victory_value, a timeout
    # `draw_penalty`. True: +victory_value on a WIN and **0.0 on everything else**. Under
    # `--critic winprob` the critic is sigmoid(logit) ∈ [0,1] and GAE mixes the REWARD with it, so
    # with this on and `victory_value == 1.0` the undiscounted return from any state is exactly
    # 1{win} and V(s) == P(win|s) with no approximation term. PRODUCTION.
    terminal_indicator: bool = False

    # --- single source of truth: build once, flow everywhere (training + eval + version record) ---
    # Adding a reward flag = add the field above + a matching `--field-name` CLI arg. `from_args`
    # picks it up (no hand-threading), `from_dict` reconstructs it for eval/resume, and the eval
    # reward then automatically matches what the policy was trained with. This DRY-ness exists because
    # a hand-threaded field was once silently MISSED on the eval path (eval measured the wrong reward).
    @classmethod
    def from_args(cls, args) -> "RewardConfig":
        """THE construction site from parsed CLI args. Every field whose name matches a CLI dest is
        pulled from ``args``; ``gamma`` is the fixed PPO discount (0.9999, asserted == model.gamma)."""
        vals = {f.name: getattr(args, f.name)
                for f in fields(cls) if f.name != "gamma" and hasattr(args, f.name)}
        # gen3_winprob_critic_mode_v1: `--gamma` is a flag. An UNSET --gamma resolves to the
        # historical 0.9999 in `main.train.config`, so a flagless run reads exactly as it always did.
        vals["gamma"] = float(getattr(args, "gamma", None) or PBRS_GAMMA)
        return cls(**vals)

    @classmethod
    def from_dict(cls, d: "dict | None") -> "RewardConfig":
        """Reconstruct from a ``model_config.json`` dict — the helper EVERY snapshot-loading consumer
        (eval workers, resume) uses so the reward the policy was TRAINED with is the reward used to
        MEASURE it. Unknown keys (arch fields / use_popart / the DELETED shaped-reward fields of a
        pre-deletion config / …) are ignored; any reward field absent from an older config falls
        back to its dataclass default. ⚠️ Ignoring a deleted shaped field is right for an eval or a
        frozen opponent and WRONG for a resume or fork — those refuse first
        (`agents.model.model_version.shaped_reward`)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class RewardBreakdown:
    """Per-turn reward breakdown. Stored on Gen3RewardManager as `_last_breakdown` after each
    `process_turn_reward()` call. One field survives the shaped-reward deletion — the terminal."""

    win_loss: float = 0.0          # TERMINAL — the win/loss (indicator or signed; see RewardConfig)

    # ---- The reward registry: field name → class. The census (`reward_composition`) and the
    # `reward/` export read it; coverage is exhaustive + 1:1 over the float fields. ----
    _REGISTRY: ClassVar[dict] = {
        "win_loss": RewardClass.TERMINAL,
    }

    _GROUPS: ClassVar[tuple] = (
        ("base", ("win_loss",)),
    )

    # Derived-once memos (per-PROCESS constants), lazy so they stay DERIVED from the registry.
    _REGISTRY_FIELDS: ClassVar[dict] = {}
    _TOTAL_FIELDS: ClassVar[tuple] = ()

    @classmethod
    def registry_fields(cls, reward_class: "RewardClass") -> tuple:
        """The breakdown fields belonging to ``reward_class`` (the registry is the source of truth)."""
        cached = cls._REGISTRY_FIELDS.get(reward_class)
        if cached is None:
            cached = tuple(name for name, c in cls._REGISTRY.items() if c is reward_class)
            cls._REGISTRY_FIELDS[reward_class] = cached
        return cached

    @classmethod
    def field_names(cls) -> tuple:
        """Every declared breakdown field, in declaration order (cached; see `_TOTAL_FIELDS`)."""
        names = cls._TOTAL_FIELDS
        if not names:
            names = cls._TOTAL_FIELDS = tuple(f.name for f in fields(cls))
        return names

    @property
    def total(self) -> float:
        """Sum of every registry term (= every dataclass float field)."""
        return sum(getattr(self, n) for n in RewardBreakdown.field_names())

    def to_dict(self) -> dict:
        """Grouped, compact JSON dict: 'total' always, plus each group's non-zero fields as one
        'key=±value' string (e.g. ``{'total': 1.0, 'base': 'win_loss=+1'}``)."""
        result: dict = {"total": round(self.total, 4)}
        for group_name, group_fields in self._GROUPS:
            parts = []
            for fname in group_fields:
                v = getattr(self, fname)
                if v != 0.0:
                    parts.append(f"{fname}={v:+.4g}")
            if parts:
                result[group_name] = " ".join(parts)
        return result
