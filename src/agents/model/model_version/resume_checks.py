"""The gates that are NOT `check_compatible`: resume-only, and opponent-only.

Two families, and the distinction is the whole reason they are not in `compat.py`:

* `check_opponent_compatible` is DELIBERATELY WEAKER -- a frozen stable opponent is a pure
  `observation -> action` function, so only the observation family has to match.
* the `check_*` hparam gates are resume-IMMUTABLE value-meaning checks (`vf_coef`,
  `belief_grad_mode`, `value_from_dist`, `value_tail_weight`, the value-dist support,
  `reward_config`). Their forward is bit-identical, so gating them in `check_compatible` would
  falsely reject the run's own snapshots. `flag_registry_test` asserts exactly that separation.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from agents.model.model_version.constants import (
    _BELIEF_GRAD_MODE_EFFECT,
    _REWARD_IMMUTABLE_FIELDS,
    ModelVersionError,
    _reward_flag_repr,
)
from agents.model.model_version.fields import ModelVersionFields

if TYPE_CHECKING:
    from agents.model.model_version.spec import ModelVersion


class ModelVersionResumeChecks(ModelVersionFields):
    """The resume-immutable + stable-opponent gates."""

    def check_opponent_compatible(self, foreign: "ModelVersion") -> None:
        """Gate for loading a frozen model from ANOTHER run as an inference-only OPPONENT
        (a "stable opponent"). Call as: ``current_version.check_opponent_compatible(foreign)``.

        A stable opponent is a pure ``observation -> action`` function: it consumes the obs the
        LIVE encoder produces and emits an action index that crosses into the shared battle. So the
        ONLY axis that must match is the OBSERVATION FAMILY — and ``arch_signature`` is the proxy
        for it: any obs-layout/meaning change bumps the signature, so equal signatures guarantee the
        same obs layout. (It ALSO bumps on pure network-structure refactors, making this stricter
        than strictly necessary — but in a safe direction, and same-arch ⟹ identical net sizes, so
        the foreign zip rebuilds its extractor at shapes matching its own weights with no further
        check needed. If an obs-identical-but-model-refactored opponent is ever wanted, split a
        dedicated ``obs_signature`` out of ``arch_signature`` and gate on that instead.)

        Deliberately DISTINCT from ``check_compatible`` (which gates the trainee's own resume + the
        self-play pool/sentinels, where every ``_WEIGHT_FIELD`` AND ``use_popart`` must match): an
        opponent never shares weights with the trainee and never reads its value head, so
        ``use_popart`` / ``vf_coef`` / the reward-config hparams are all irrelevant to its forward
        and are deliberately NOT checked here.
        """
        if self.arch_signature != foreign.arch_signature:
            raise ModelVersionError(
                f"Stable opponent architecture-family mismatch: "
                f"opponent='{foreign.arch_signature}', current='{self.arch_signature}'.\n"
                "A stable opponent must share the live run's arch_signature — i.e. the SAME "
                "observation layout (a different signature means the live encoder cannot feed it).\n"
                "Use an opponent trained at the current architecture, or start the new run at the "
                "opponent's architecture."
            )
        # Defensive: same arch_signature already implies these match, but a hand-edited config
        # could lie — and feeding the opponent a wrong-width obs would be a silent-garbage bug.
        for field in ("total_dim", "active_context_dim"):
            cur, opp = getattr(self, field), getattr(foreign, field)
            if cur != opp:
                raise ModelVersionError(
                    f"Stable opponent {field} mismatch: opponent={opp}, current={cur} "
                    "(arch_signature matched — the opponent's model_config.json looks hand-edited)."
                )

    def check_vf_coef(self, requested: float) -> None:
        """Raise ModelVersionError if `requested` (the resume `--vf-coef`) differs from this
        saved config's vf_coef.

        Call as: saved_version.check_vf_coef(args.vf_coef).

        vf_coef is a training-loss coefficient, not a weight-shape concern, so it is
        deliberately NOT part of check_compatible() — that gates EVERY checkpoint load,
        including the frozen eval / self-play-pool / distill opponents, where vf_coef is
        irrelevant (the forward pass is identical regardless of it). This check is invoked
        ONLY on the training-resume path: silently changing the value head's gradient scale
        mid-run would let a forgotten/typo'd flag drift training, so a resume with a
        different value is a hard error rather than a quiet change.
        """
        if not math.isclose(self.vf_coef, requested, rel_tol=1e-9, abs_tol=1e-12):
            raise ModelVersionError(
                f"vf_coef mismatch: saved={self.vf_coef!r}, requested={requested!r}.\n"
                "The PPO value-loss coefficient is fixed for the lifetime of a run — changing it on "
                "resume silently alters the value head's gradient scale.\n"
                f"Fix: resume with --vf-coef {self.vf_coef!r}, or start a fresh training run to use "
                f"{requested!r}."
            )

    def check_belief_grad_mode(self, requested: str, allow_change: bool = False) -> None:
        """Raise ModelVersionError if `requested` (the resume `--belief-grad-mode`) differs from this
        saved config's belief_grad_mode. Call as: saved_version.check_belief_grad_mode(args.belief_grad_mode).

        gen3_belief_grad_mode_v1: detach() is value-preserving, so the FORWARD (eval / inference / a frozen
        pool / distill opponent) is bit-identical regardless of the mode — only the TRAINING gradient (does
        the belief reshape the trunk) differs. So, like vf_coef, it is EXCLUDED from check_compatible (gating
        a frozen opponent on it would be a false rejection that breaks self-play) and enforced ONLY on the
        training-resume path: flipping shaping↔detached mid-run silently changes whether the belief
        gradient shapes the shared trunk, so a drift is a hard error rather than a quiet change.

        ``allow_change=True`` (--allow-belief-grad-mode-change) is the INTENTIONAL-migration escape hatch:
        because detach() is value-preserving, flipping the mode on a converged checkpoint is weight-safe —
        the gate exists to prevent ACCIDENTAL drift, not because the transition is unsound. A permitted
        mismatch prints a loud notice; the next checkpoint save records the new mode, so the flag is only
        needed once per migration (the staged shaping-flip experiment, next_run_plan item 5)."""
        if self.belief_grad_mode != requested:
            if allow_change:
                print(
                    f"[ModelVersion] NOTICE: belief_grad_mode MIGRATION {self.belief_grad_mode!r} -> "
                    f"{requested!r} (--allow-belief-grad-mode-change). Forward is bit-identical; "
                    + _BELIEF_GRAD_MODE_EFFECT.get(requested, "the belief gradient routing changed.")
                    + " The next checkpoint save records the new mode."
                )
                return
            raise ModelVersionError(
                f"belief_grad_mode mismatch: saved={self.belief_grad_mode!r}, requested={requested!r}.\n"
                "Whether the belief heads reshape the shared trunk is fixed for a run's lifetime — flipping "
                "it on resume silently changes the training signal.\n"
                f"Fix: resume with --belief-grad-mode {self.belief_grad_mode}, pass "
                "--allow-belief-grad-mode-change for an intentional migration, or start a fresh run."
            )

    def check_value_from_dist(self, requested: bool, allow_change: bool = False) -> None:
        """Raise ModelVersionError if `requested` (the resume `--value-from-dist`) differs from this
        saved config's value_from_dist. gen3_dist_critic_v1 (Phase B): swapping the GAE/bootstrap value
        source between the scalar value_net and the distributional E[Z] silently changes the training
        objective, so — like belief_grad_mode/vf_coef — a mid-run drift is a hard error, enforced ONLY on
        the training-resume path (a frozen opponent's ACTION selection is unchanged, so it's EXCLUDED from
        check_compatible). ``allow_change=True`` (--allow-value-from-dist-change) is the intentional
        warm-start-migration hatch (the offline probe confirmed E[Z]≈V, so the swap is near-seamless);
        it prints a loud notice and the next save records the new mode."""
        if bool(self.value_from_dist) != bool(requested):
            if allow_change:
                print(
                    f"[ModelVersion] NOTICE: value_from_dist MIGRATION {self.value_from_dist} -> {requested} "
                    "(--allow-value-from-dist-change). The GAE/bootstrap critic is now "
                    + ("the distributional E[Z] (scalar value_net frozen as fallback)." if requested
                       else "the scalar value_net.")
                    + " The next checkpoint save records the new mode."
                )
                return
            raise ModelVersionError(
                f"value_from_dist mismatch: saved={self.value_from_dist}, requested={requested}.\n"
                "Whether the critic is the scalar value_net or the distributional E[Z] is fixed for a run's "
                "lifetime — flipping it on resume silently changes the value objective + GAE source.\n"
                f"Fix: resume with --value-from-dist={self.value_from_dist}, pass "
                "--allow-value-from-dist-change for the intentional Phase-B migration, or start a fresh run."
            )

    def check_value_tail_weight(self, requested: float) -> None:
        """Raise ModelVersionError if `requested` (the resume `--value-tail-weight`) differs from this
        saved config's value_tail_weight. Call as: saved_version.check_value_tail_weight(args...).

        Same treatment as check_vf_coef: a value-loss hparam (the CVaR-blend weight), not weight-shape,
        so it is EXCLUDED from check_compatible (frozen eval/pool/distill opponents never run the value
        loss) and enforced ONLY on the training-resume path. Changing it mid-run silently reshapes the
        value objective (how hard the critic chases its tail), so a drift is a hard error."""
        if not math.isclose(self.value_tail_weight, requested, rel_tol=1e-9, abs_tol=1e-12):
            raise ModelVersionError(
                f"value_tail_weight mismatch: saved={self.value_tail_weight!r}, requested={requested!r}.\n"
                "The tail-weighted value-loss β is fixed for a run's lifetime — changing it on resume "
                "silently reshapes the value objective.\n"
                f"Fix: resume with --value-tail-weight {self.value_tail_weight!r}, or start a fresh run."
            )

    def check_value_dist(self, vmin: float, vmax: float) -> None:
        """Raise ModelVersionError if the resume `--value-dist-vmin/--value-dist-vmax` differ from this
        saved config's support. Call as: saved_version.check_value_dist(args.value_dist_vmin, ...).

        Same treatment as check_value_tail_weight: the atom support is VALUE-meaning (it is what the
        head's logits are read against — the loss target and the prober's atoms→return mapping), not
        weight-shape (the atoms buffer is non-persistent), so it is EXCLUDED from check_compatible
        (frozen eval/pool/distill opponents never read the value-dist head) and enforced ONLY on the
        training-resume path. Shifting the support mid-run silently re-targets the head."""
        problems = []
        if not math.isclose(self.value_dist_vmin, vmin, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"vmin saved={self.value_dist_vmin!r} requested={vmin!r}")
        if not math.isclose(self.value_dist_vmax, vmax, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"vmax saved={self.value_dist_vmax!r} requested={vmax!r}")
        if problems:
            raise ModelVersionError(
                "value_dist support mismatch: " + "; ".join(problems) + ".\n"
                "The distributional value head's atom support is fixed for a run's lifetime — changing "
                "it on resume silently re-targets the head.\n"
                f"Fix: resume with --value-dist-vmin {self.value_dist_vmin!r} --value-dist-vmax "
                f"{self.value_dist_vmax!r}, or start a fresh run."
            )

    def check_reward_config(self, reward_config: Any) -> None:
        """Raise ModelVersionError if the resume `reward_config` differs from this saved config's
        reward hparams (bias_additivity / mat_alive_weight / bias_redesign / …). Like check_vf_coef:
        these are VALUE-meaning (changing them mid-run silently shifts the reward), NOT weight-shape,
        so they are enforced ONLY on the training-resume path and excluded from check_compatible().
        Call as: saved_version.check_reward_config(args_reward_config).

        The error NAMES the fix. That matters more than usual since 2026-08-18, when
        `--all-shaping-pbrs` defaulted ON and `--draw-penalty` to -35.0: every pre-flip run now
        mismatches on a flagless resume, and a diff that only reports "saved=X, requested=Y" leaves
        the reader to reconstruct the flag spelling (including that the opt-out is
        `--no-all-shaping-pbrs`, not `--all-shaping-pbrs false`, and that the negation of a float
        flag is just the old number).
        """
        problems, repass, recorded_pairs = [], [], []
        for name, default in _REWARD_IMMUTABLE_FIELDS.items():
            wanted = getattr(reward_config, name, default)
            saved = getattr(self, name)
            if isinstance(default, bool):
                saved, wanted = bool(saved), bool(wanted)
                differs = saved != wanted
            else:
                saved, wanted = float(saved), float(wanted)
                differs = not math.isclose(saved, wanted, rel_tol=1e-9, abs_tol=1e-12)
            if differs:
                problems.append(f"  {name}: saved={saved!r}, requested={wanted!r}")
                recorded_pairs.append(f"{name}={saved!r}")
                repass.append(_reward_flag_repr(name, saved))
        if problems:
            recorded = ", ".join(recorded_pairs)
            raise ModelVersionError(
                "Reward-config mismatch on resume — these hparams are fixed for a run's lifetime "
                "(changing them silently shifts the reward / objective):\n" + "\n".join(problems) +
                f"\n\nThis run recorded {recorded}.\n"
                f"Fix: re-pass `{' '.join(repass)}` to resume it, or start a fresh run.\n"
                "(The reward DEFAULTS changed on 2026-08-18 — --all-shaping-pbrs now defaults ON "
                "and --draw-penalty to -35.0 — so a run started under the old defaults must state "
                "them explicitly on every resume.)"
            )
