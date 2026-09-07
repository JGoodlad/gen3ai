"""``ProbeSession`` itself — construction, the shared internals, and the model-resolution ladder.

The class is assembled from one mixin per command family (see `main/prober/session/__init__.py`
for the map). What lives HERE is what all of them share: the trace tree, the summary/npz readers,
the bounded model cache, and `exact → nearest → recent` resolution.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from main.prober.discovery import BattleTrace, ModelChoice, build_trace_tree, resolve_model_for_step
from main.prober.awareness import awareness_from_npz
from main.prober.engine import awareness_text, decode_incoming_belief

from main.prober.session.serialize import _short_id
from main.prober.session.aggregate import _AggregateMixin
from main.prober.session.analysis import _AnalysisMixin
from main.prober.session.counterfactual import _CounterfactualMixin
from main.prober.session.probes import _ProbesMixin
from main.prober.session.reading import _ReadingMixin
from main.prober.session.scans import _ScansMixin
from main.prober.session.trace_io import _TraceIOMixin


# How many loaded ProbeModels one session keeps. Two lets the common "compare this step against a
# neighbour" walk stay instant without pinning a whole run's worth of checkpoints in RAM; a third
# would cost ~27 MB more for a pattern nobody performs.
_MAX_CACHED_MODELS = 2

_DEFAULT_GAMMA = 0.99


class ProbeSession(_ReadingMixin, _ScansMixin, _TraceIOMixin, _AnalysisMixin,
                   _CounterfactualMixin, _AggregateMixin, _ProbesMixin):
    """Programmatic access to the probing infrastructure — see the package docstring.

    The command families are mixins; every method below is one all of them use.
    """

    def __init__(self, root: str, ckpt_override: "str | None" = None, tier: str = "auto",
                 model_loader=None, compile_extractor: bool = False,
                 impl: str = "node") -> None:
        self.tree = build_trace_tree(root)
        self.run_dir = self.tree.run_dir
        self._override = ckpt_override
        self._tier = tier
        self._model_loader = model_loader      # (path)->model; default ProbeModel.load (tests inject)
        self._models: dict = {}                 # checkpoint path → ProbeModel
        self._play_models: dict = {}            # checkpoint path → MaskablePPO (counterfactual replay players)
        self._cf_mappings = None                # lazily-loaded encoder mappings for the replay players
        # torch.compile the no-grad replay models (see _load below). OFF by default: a one-off
        # `summary`/`list` query would never amortize the compile. Worth it for the search-shaped
        # commands — better-line, falsify, falsify-scan, replay-counterfactual, lookahead.
        self._compile_extractor = bool(compile_extractor)
        # WHICH offline replay/search driver the re-roll-backed probes spawn: "node" (default,
        # the historical behavior) or "rust". SESSION-WIDE on purpose, exactly like
        # `compile_extractor` above: it is a transport choice for the whole investigation, not a
        # per-question semantic — two probes of the same run answering under different engines
        # would not be comparable, which is precisely what a per-call knob would invite. The CLI
        # surfaces it as `--impl` on the search-shaped commands; every probe below reads
        # `self._impl`.
        self._impl = impl
        self._summaries: "dict[str, dict]" = {}
        self._by_path = {b.summary_path: b for b in self.tree.all_battles()}
        self._by_short = {_short_id(b): b for b in self.tree.all_battles()}
        self._gamma = self._read_gamma()
        self._dist_support_cache: object = "unset"   # model-free dist support (see _dist_support)
        self._critic_mode_cache: object = "unset"    # model-free critic currency (see critic_mode)

    def close(self) -> None:
        """Drop the cached models/summaries. A long-lived caller — the persistent search-teacher
        worker builds one ``ProbeSession`` per generation iteration — would otherwise accumulate a
        ``ProbeModel`` (+ a counterfactual ``MaskablePPO``) per checkpoint forever. Idempotent."""
        self._models.clear()
        self._play_models.clear()
        self._summaries.clear()

    def __enter__(self) -> "ProbeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


    def _obs_offsets(self):
        """Lazily resolve the obs-block offsets once (builds the encoder; model-free). Cached on the
        session. None if resolution fails, so the belief decode degrades gracefully."""
        off = getattr(self, "_offsets_cache", "unset")
        if off == "unset":
            try:
                from main.prober.model import ObsOffsets
                off = ObsOffsets.resolve()
            except Exception:  # noqa: BLE001 — belief decode is best-effort
                off = None
            self._offsets_cache = off
        return off

    def _belief_at(self, npz: dict, i: int):
        """The decoded incoming-damage belief at decision ``i`` (or None — no obs / no captured
        state for that decision / offsets unresolvable)."""
        obs_arr = npz.get("obs")
        if obs_arr is None or i >= len(obs_arr):
            return None
        hs = npz.get("has_state")
        if hs is not None and i < len(hs) and not bool(hs[i]):
            return None
        off = self._obs_offsets()
        if off is None:
            return None
        return decode_incoming_belief(obs_arr[i].astype(np.float32), off)

    # -- internals -----------------------------------------------------------

    def _notable(self, rows: "list[dict]") -> dict:
        drops = sorted(((r["delta_v"], r["inv"]) for r in rows if r["delta_v"] is not None))
        return {
            "faints": [r["inv"] for r in rows if "faint" in r["flags"]],
            "switches": [r["inv"] for r in rows if "switch" in r["flags"]],
            "uncertain_count": sum("uncertain" in r["flags"] for r in rows),
            "biggest_value_drops": [{"inv": i, "delta_v": d} for d, i in drops[:3]],
            "disagreements_hint": "call find(battle, 'disagree') to load the model and list them",
        }

    def _td(self, reward_total, v, v_next) -> "float | None":
        """TD residual δ = r + γV(s') − V(s): how surprised the critic was."""
        if reward_total is None or v is None or v_next is None:
            return None
        return float(reward_total) + self._gamma * v_next - v

    def _read_gamma(self) -> float:
        if not self.run_dir:
            return _DEFAULT_GAMMA
        try:
            with open(os.path.join(self.run_dir, "metadata.json")) as f:
                return float(json.load(f).get("gamma", _DEFAULT_GAMMA))
        except (OSError, ValueError, TypeError):
            return _DEFAULT_GAMMA

    def _snapshot_available(self, step: int, manifest: dict) -> bool:
        if not (manifest.get("snapshot") and self.run_dir):
            return False
        return os.path.exists(os.path.join(
            self.run_dir, "eval_traces", f"step_{step}", manifest["snapshot"]))

    def _values(self, battle: BattleTrace):
        return self._npz(battle).get("values")

    def _dist_support(self) -> "tuple[float, float, int] | None":
        """The dist head's atom support ``(vmin, vmax, bins)`` read MODEL-FREE from the run
        root's ``model_config.json`` — the same numbers ``value_dist_support()`` reads off a
        loaded extractor, available without loading one (so `turns`/awareness stay ~20 ms and
        work on archived runs the current code can't re-load). None when the run trained no
        dist head or the config is absent. Cached (one file, immutable for a run)."""
        if self._dist_support_cache != "unset":
            return self._dist_support_cache
        support = None
        try:
            with open(os.path.join(self.run_dir or "", "model_config.json")) as f:
                cfg = json.load(f)
            if cfg.get("value_dist_mode", "none") != "none":
                support = (float(cfg["value_dist_vmin"]), float(cfg["value_dist_vmax"]),
                           int(cfg["value_dist_bins"]))
        except (OSError, KeyError, TypeError, ValueError):
            support = None
        self._dist_support_cache = support
        return support

    def critic_mode(self) -> str:
        """WHICH READOUT IS THE CRITIC, and therefore WHAT CURRENCY the recorded ``values`` are
        in — read MODEL-FREE from the run root's ``model_config.json`` (`--critic`), exactly as
        ``_dist_support`` reads the dist head's support. ``"shaped"`` or ``"winprob"``.

        **An ABSENT key means ``"shaped"``**, and that is a fact about the archive rather than a
        default chosen here: `--critic` landed with config version 109, so every run recorded
        before it (214 of the 215 on this box, measured 2026-09-06) has no key and every one of
        them is shaped. A run that cannot be read at all is also shaped — the historical
        behaviour, so an unreadable config can never silently re-scale an old run's numbers.

        Why this exists: under ``winprob`` the critic IS the win-prob head, so V(s) = P(win) ∈
        [0, 1] instead of a shaped, discounted return of roughly ±``victory_value`` (30). Every
        threshold expressed in *value units* therefore means something different, and a threshold
        carried across the two currencies unchanged does not merely lose precision — it silently
        stops firing. See ``calibration``'s ``overvalue_tau``, which at its shaped default of 5.0
        exceeds the entire representable range of a probability gap.
        """
        if self._critic_mode_cache != "unset":
            return self._critic_mode_cache            # type: ignore[return-value]
        mode = "shaped"
        try:
            with open(os.path.join(self.run_dir or "", "model_config.json")) as f:
                mode = str(json.load(f).get("critic") or "shaped")
        except (OSError, KeyError, TypeError, ValueError):
            mode = "shaped"
        self._critic_mode_cache = mode
        return mode

    def critic_currency(self) -> dict:
        """The recorded critic's currency as DATA, for any surface that must scale a threshold,
        label an axis, or centre a "was it winning?" split. One declaration, so the CLI and the
        browser cannot end up describing the same run's units differently.

        ``even`` is the value at which the critic rates a position 50/50 — the number a
        winning-vs-losing split must compare against. It is **not** 0 on a shaped critic: V there
        is a shaped, discounted return carrying a structural negative offset (a measured
        self-mirror 50/50 reads V ≈ −6.5), which is why the shaped ``v_even`` stays 0.0 as a
        documented over-counting fallback rather than pretending to be centred.
        """
        mode = self.critic_mode()
        if mode == "winprob":
            return {
                "mode": "winprob", "units": "P(win)", "low": 0.0, "high": 1.0, "even": 0.5,
                "span": 1.0, "is_probability": True,
                # 1/12 of span — the same fraction the shaped 5.0 is of 60. See `calibration`.
                "default_overvalue_tau": round(1.0 / 12.0, 4),
                "note": "V(s) = sigmoid(win-prob logit) ∈ [0,1]; `values` EQUALS `win_probs`. "
                        "PopArt absent; the realized return G(s) is the terminal win indicator "
                        "(0 or 1) at gamma 1.",
            }
        return {
            "mode": "shaped", "units": "shaped return", "low": -35.0, "high": 30.0, "even": 0.0,
            "span": 60.0, "is_probability": False,
            "default_overvalue_tau": 5.0,     # the historical bar; 1/12 of the ±30 span
            "note": "V(s) is a shaped, discounted return of roughly ±victory_value (30). Its zero "
                    "is NOT 'even' — a self-mirror 50/50 reads V ≈ −6.5 — so a V-based winning "
                    "split over-counts grinds unless re-centred.",
        }

    def _awareness(self, battle: BattleTrace, invs: "list[dict]",
                   npz: "dict | None" = None) -> "dict | None":
        """The battle's awareness verdict (awareness.py) as a JSON dict, or None when the run
        has no dist head / the trace carries <2 dist rows. Model-free.

        `npz` lets a caller that has ALREADY loaded the trace's arrays (scan, triage) pass them in
        — `_npz` re-opens and re-reads the file on every call, so folding awareness into a
        whole-run sweep would otherwise double its file IO for data already in hand."""
        support = self._dist_support()
        if support is None:
            return None
        v = awareness_from_npz(npz if npz is not None else self._npz(battle),
                               [inv.get("turn") for inv in invs],
                               battle.outcome or "?", support)
        if v is None:
            return None
        # LISTS, not the dataclass's tuples. `ProbeSession` promises JSON-serializable dicts, and a
        # tuple is serializable but does NOT round-trip — it comes back a list, so a caller
        # comparing this dict to its own JSON form (which is exactly how `web/app_test.py` enforces
        # "the page reshapes nothing") gets a spurious mismatch on every array here.
        d = {k: (list(x) if isinstance(x, tuple) else x) for k, x in asdict(v).items()}
        # The verdict's SENTENCE is rendered here, in the engine, exactly like `timeline`'s `text`:
        # a surface prints it rather than re-deriving it, so the CLI and the browser cannot end up
        # phrasing the same fold differently.
        d["text"] = awareness_text(d)
        return d

    @staticmethod
    def _awareness_by_decision(aw: "dict | None") -> dict:
        """`{inv: {p_loss, p_win, p_tail, knew}}` from the battle-level verdict — the per-DECISION view of
        the same fold, keyed by the decision index the verdict records (never by game turn: a
        faint puts two decisions on one turn). `knew` marks the decisions at or after the sustained
        onset, i.e. the stretch the model was already reading as lost — and it keys off the onset
        DECISION rather than its turn, so a decision sharing that turn but preceding the crossing
        is not swept in with it."""
        if not aw:
            return {}
        onset = aw.get("knew_from_decision")
        out = {}
        for inv, pl, pw, pt in zip(aw.get("decisions") or (),
                                   aw.get("p_loss") or (), aw.get("p_win") or (),
                                   aw.get("p_tail") or ()):
            # BOTH directions ride the row: `p_win` is what the replay renders (one direction per
            # card, higher = better, matching the win-prob head beside it) and `p_loss` is what the
            # thresholds are defined on. A surface flipping one into the other itself would be a
            # view deriving a number.
            out[inv] = {"p_loss": pl, "p_win": pw, "p_tail": pt,
                        "knew": onset is not None and inv >= onset}
        return out

    @staticmethod
    def _v(values, i: int) -> "float | None":
        return float(values[i]) if values is not None and 0 <= i < len(values) else None

    def _battle(self, battle_id: str) -> BattleTrace:
        b = self._by_path.get(battle_id) or self._by_short.get(battle_id)
        if b is None:  # allow a raw summary path not in the original tree
            extra = build_trace_tree(battle_id).all_battles()
            if not extra:
                raise FileNotFoundError(f"no trace found for {battle_id!r}")
            b = extra[0]
        return b

    def _summary(self, battle: BattleTrace) -> dict:
        s = self._summaries.get(battle.summary_path)
        if s is None:
            with open(battle.summary_path) as f:
                s = json.load(f)
            self._summaries[battle.summary_path] = s
        return s

    def _npz(self, battle: BattleTrace) -> dict:
        if battle.npz_path is None:
            return {}
        with np.load(battle.npz_path) as z:
            return {k: z[k] for k in z.files}


    def probe_model(self, battle_id: str):
        """``(ProbeModel, ModelChoice)`` for one battle — the PUBLIC face of the resolution ladder.

        Exists for out-of-package readers that need the loaded network itself rather than one of
        the analyses built on it (`agents.training.cf_audit` reads the evidential Beta head off the
        audited checkpoint this way). It is the same cached `_model_for`, so a caller that resolves
        once and reuses pays for one load; every battle in one `step_N` trace dir resolves to the
        same checkpoint, so that is the normal shape."""
        return self._model_for(self._battle(battle_id))

    def _resolve(self, battle: BattleTrace) -> ModelChoice:
        return resolve_model_for_step(self.tree, battle.step, self._override, self._tier)

    def _model_for(self, battle: BattleTrace):
        choice = self._resolve(battle)
        if choice.path is None:
            raise FileNotFoundError(choice.detail)
        model = self._models.get(choice.path)
        if model is None:
            if self._model_loader is not None:
                model = self._model_loader(choice.path)
            else:
                from main.prober.model import ProbeModel
                model = ProbeModel.load(choice.path)
            self._models[choice.path] = model
            # BOUNDED, oldest-first. One ProbeModel is a full MaskablePPO (~27 MB of weights plus
            # its graph), and this cache is keyed by CHECKPOINT — a run retains up to
            # `--keep-eval-trace-steps` (20) eval snapshots, so walking a run's steps used to pin
            # twenty of them. Unbounded was survivable for the TUI (one human, one process); it is
            # not for the web front end, where an anonymous request picks the step and the same
            # "anything an anonymous request can grow must have a bound" rule already governs the
            # session cache and the auth failure map (see `web/CLAUDE.md`).
            while len(self._models) > _MAX_CACHED_MODELS:
                self._models.pop(next(iter(self._models)))
        return model, choice
