"""RUN-LEVEL model-free scans: worst-turn `scan`, `loops`, and `triage`.

Each folds one verdict per battle and ranks across the run. All model-free — they read the traces
on disk, so they answer on any run regardless of architecture drift.
"""

from __future__ import annotations

import collections
import copy
import json
import os

from agents.training.trace_result import OUTCOMES
from main.prober.discovery import BattleTrace
from main.prober.loops import LOOP_BASELINES
from main.prober.loops import analyze_battle as _analyze_loops
from main.prober.engine import (SETUP_MOVES, WP_EVEN_DEFAULT, _npz_win_prob,
    attribute_turning_point, build_board, parse_pct, summary_flags)
from main.prober.session.serialize import _active_str, _r, _short_id
from main.prober.session.stats import _loop_aggregate


# How many triage results one session keeps (`_triage_cache`, an LRU). A result is a few KB; 16 covers a
# reader walking the step and opponent filters of one run without letting an anonymous walk grow it.
_MAX_CACHED_TRIAGES = 16


class ScanTooBroad(ValueError):
    """A run-level read would EXPAND (core-trace replay, ~0.1-0.2 s each on rb_x26_s2601) more battles than
    the caller's ``max_expand`` allows. Refused BEFORE any expansion; the message names the count, the cap
    and the CLI to run it from. The web service passes its cap (``web/app._MAX_EXPANDED_BATTLES``); the CLI
    passes none."""

    def __init__(self, what: str, n: int, cap: int, cli: str) -> None:
        super().__init__(
            f"{what} would expand {n:,} battles (a core-trace replay each, ~0.1-0.2 s) — over this "
            f"server's cap of {cap:,} per request. Narrow it (one step, an opponent, an outcome; a "
            f"value_drop scan with a limit of at most {cap:,} reads only stored values), or run it from a "
            f"shell, where there is no cap: {cli}")
        self.n, self.cap = n, cap


class _ScansMixin:
    def loops(self, *, outcome: "str | None" = None, opponent: "str | None" = None,
              step: "int | None" = None, max_battles: "int | None" = None,
              near_zero_frac: float = 0.01, top: int = 12) -> dict:
        """BAIT-LOOP scan (MODEL-FREE): "they pivoted something immune in, and we fired anyway".

        The gen-16 instrument for the pathology gen-15 measured (`loops.py` holds the definitions
        and the baselines; `designs/research_state/bait_loop_hunt.md` holds the pre-registered
        bars). Joins each battle's raw Showdown protocol — the ground truth, NOT the rendered
        timeline, which collapses immune/cant/small-hit into one phrase — to the recorder's
        per-decision summary, and reports three rates plus the α/β readout on the same pivots.

        ``opponent`` is an fnmatch PATTERN, so ``sentinel_*`` selects the self-play sentinels as one
        population (an exact name still matches exactly — a name with no wildcard is its own
        pattern). The pathology was measured on sentinels, and five separate calls would be five
        separate denominators.

        Every rate ships with its numerator and denominator, and the two registered CONFOUNDS are
        conditioned for rather than mentioned: loop rate rises with game LENGTH and concentrates in
        WINNING positions, so the per-pivot and per-decision rates sit beside the per-battle one and
        the win/loss split is always reported.
        """
        import fnmatch

        rows, folds, skipped = [], [], collections.Counter()
        for b in self.tree.all_battles():
            if outcome and b.outcome != outcome:
                continue
            if opponent and not fnmatch.fnmatchcase(b.opponent, opponent):
                continue
            if step is not None and b.step != step:
                continue
            if max_battles is not None and len(rows) + sum(skipped.values()) >= max_battles:
                break
            lines = self._protocol_lines(b)
            if not lines:
                skipped["no_replay_html"] += 1
                continue
            summary = self._summary(b)
            npz = self._npz(b)
            fold = _analyze_loops(
                lines, summary.get("invocations", []), outcome=b.outcome,
                n_turns=(summary.get("meta") or {}).get("turns"),
                values=(list(npz["values"]) if "values" in npz else None),
                win_probs=(list(npz["win_probs"]) if "win_probs" in npz else None),
                near_zero_frac=near_zero_frac)
            if fold.skipped:
                skipped[fold.skipped.split(":")[0]] += 1
                continue
            folds.append(fold)
            rows.append({
                "id": b.summary_path, "short_id": _short_id(b), "step": b.step,
                "opponent": b.opponent, "outcome": b.outcome, "turns": fold.n_turns,
                "our_side": fold.our_side,
                "moved_into_pivots": fold.moved_into_pivots, "whiffs": fold.whiffs,
                "reclicks": fold.reclicks, "worst_loop": fold.worst_loop,
                "loops": [dict(g) for g in fold.loops],
                "whiff_turns": [b2.turn for b2 in fold.baits if b2.whiff],
            })

        agg = _loop_aggregate(folds)
        agg["by_outcome"] = {k: _loop_aggregate([f for f in folds if f.outcome == k])
                             for k in OUTCOMES if any(f.outcome == k for f in folds)}
        by_step: "dict[int, list]" = collections.defaultdict(list)
        for r, f in zip(rows, folds):
            by_step[r["step"]].append(f)
        agg["by_step"] = [dict(step=s, **_loop_aggregate(by_step[s])) for s in sorted(by_step)]
        agg["coverage"] = {"n_matched": len(folds), "n_skipped": int(sum(skipped.values())),
                           "skipped_reasons": dict(skipped)}
        rows.sort(key=lambda r: (-r["worst_loop"], -r["reclicks"]))
        return {
            "run_dir": self.run_dir,
            "params": {"outcome": outcome, "opponent": opponent, "step": step,
                       "max_battles": max_battles, "near_zero_frac": near_zero_frac},
            "aggregate": agg,
            # gen-15's numbers, carried WITH the live ones so a reading is never quoted without its
            # reference point. A baseline is a reference, NOT a target — the bars are in the hunt doc.
            "baseline": dict(LOOP_BASELINES),
            "caveats": [
                f"Baselines are {LOOP_BASELINES['generation']}, measured "
                f"{LOOP_BASELINES['measured']} on {LOOP_BASELINES['n_battles']} "
                f"{LOOP_BASELINES['scope']} battles ({LOOP_BASELINES['source']}) — a reference "
                "point, NOT a target. Compare at MATCHED scope: a run-wide read includes the bot "
                "opponents the baseline excluded.",
                "CONFOUND 1 (length): loop_battle_rate rises with game LENGTH — a 200-turn game has "
                "more chances to repeat a pair than a 30-turn one. Read whiff_rate_per_pivot and "
                "whiff_rate_per_decision, which normalize the exposure, before the per-battle rate.",
                "CONFOUND 2 (winning positions): the loops concentrate in games we were WINNING (a "
                "won position is where a free turn is affordable), so by_outcome is always "
                "reported and an overall rate that moves with the win rate has moved for two "
                "reasons. Compare win-arm to win-arm.",
                "A MISS is never a whiff: it is dice, and taxing it would make this partly a luck "
                "reading. Misses are counted separately (aggregate.misses).",
                "beta_slot_accuracy is decidable only when the arriving mon was ALREADY revealed "
                "(the obs slot is the k-th REVEALED opponent mon), so its denominator is smaller "
                "than the pivot count and is skewed toward REPEAT pivots by construction.",
                "The mirror block (THEY whiff into OUR pivots) is the control, not a target: it "
                "measures the OPPONENT's policy, which in a sentinel matchup is a frozen self.",
            ],
            "battles": rows[:top],
        }


    def scan(self, *, outcome: "str | None" = None, opponent: "str | None" = None,
             step: "int | None" = None, limit: "int | None" = None,
             metric: str = "value_drop", max_expand: "int | None" = None) -> "list[dict]":
        """Cross-battle, MODEL-FREE turning-point scan. For every matching battle,
        find its single worst decision and return them **ranked globally** — the
        one-call version of "list losses → overview each → rank by the biggest
        value drop", which is the recurring first move of any loss investigation.

        ``metric`` ranks by ``value_drop`` (most negative ΔV(s→s'), the default) or
        ``td_residual`` (most negative critic surprise δ = r + γV(s') − V(s)).
        Filters mirror ``battles()``. No checkpoint is loaded, so this is fast even
        across a whole run. Each row is ``{id, short_id, opponent, step, outcome,
        turns, worst:{inv, turn, phase, chosen, our_active, opp_active, delta_v,
        td_residual, reward_total, events, flags}}``.

        ``max_expand`` bounds how many battles the call may EXPAND (a core-trace replay each); a call
        that would expand more raises `ScanTooBroad` before expanding any. ``None`` (the CLI) is
        unbounded. A ``value_drop`` scan with a ``limit`` expands only the rows it returns (plus any
        battle with no recorded decision count); every other scan expands every matching battle.
        """
        if metric not in ("value_drop", "td_residual"):
            raise ValueError(f"metric must be 'value_drop' or 'td_residual', got {metric!r}")
        battles = [b for b in self.tree.all_battles()
                   if not (outcome and b.outcome != outcome)
                   and not (opponent and b.opponent != opponent)
                   and not (step is not None and b.step != step)]
        cli = self._scan_cli(outcome=outcome, opponent=opponent, step=step, limit=limit, metric=metric)
        if metric == "value_drop" and limit:
            return self._scan_value_drop_top(battles, limit, max_expand=max_expand, cli=cli)
        if max_expand is not None and len(battles) > max_expand:
            raise ScanTooBroad(f"this {metric} scan", len(battles), max_expand, cli)
        rows = []
        for b in battles:
            tp = self._worst_turning_point(b, metric)
            if tp is not None:
                rows.append(tp)
        key = "delta_v" if metric == "value_drop" else "td_residual"
        rows.sort(key=lambda r: (r["worst"][key] if r["worst"][key] is not None else float("inf")))
        return rows[:limit] if limit else rows

    def _scan_cli(self, **kw) -> str:
        """The shell command that runs the same scan with no cap (named in `ScanTooBroad`)."""
        flags = "".join(f" --{k} {v}" for k, v in kw.items() if v is not None and v != "")
        return f"python -m main.prober.query scan {self.run_dir or '<run_dir>'}{flags}"

    def _scan_value_drop_top(self, battles: "list[BattleTrace]", limit: int, *,
                             max_expand: "int | None" = None, cli: str = "") -> "list[dict]":
        """`scan(metric="value_drop", limit=N)` without expanding every battle.

        A battle's worst ΔV needs only its recorded `values` and its decision COUNT — both on disk
        without the core-trace replay (`meta.invocations`, which the expansion itself asserts equals
        the npz row count). So every candidate is RANKED from those, and only the N it returns are
        expanded into rows. Measured on `rb_x26_s2601` (41,345 core traces, 2026-10-19): one
        expansion is ~0.1-0.2 s, so the full path was hours for a run-wide `/scan`; this reads two
        small files per battle. The rows, and their order, are exactly the full path's: the same
        stable sort on the same ΔV, then the same `_worst_turning_point` builds each row."""
        counts = []
        for b in battles:
            n = (self._meta(b) or {}).get("invocations")
            counts.append(n if isinstance(n, int) and not isinstance(n, bool) else None)
        if max_expand is not None:
            # Every battle with no recorded count is ranked the full way; of the rest only `limit` expand.
            n_uncounted = sum(1 for n in counts if n is None)
            would = n_uncounted + min(limit, len(battles) - n_uncounted)
            if would > max_expand:
                raise ScanTooBroad("this value_drop scan", would, max_expand, cli)
        ranked = []
        for order, (b, n) in enumerate(zip(battles, counts)):
            if n is None:
                # No recorded decision count: rank it the full way (a stored trace, not a replay).
                tp = self._worst_turning_point(b, "value_drop")
                if tp is not None:
                    ranked.append((tp["worst"]["delta_v"], order, b, tp))
                continue
            values = self._npz_values(b)
            best = None
            for i in range(n):
                v, v_next = self._v(values, i), self._v(values, i + 1)
                if v is not None and v_next is not None and (best is None or v_next - v < best):
                    best = v_next - v
            if best is not None:
                ranked.append((best, order, b, None))
        ranked.sort(key=lambda t: (t[0], t[1]))
        rows = []
        for dv, _, b, tp in ranked[:limit]:
            tp = tp if tp is not None else self._worst_turning_point(b, "value_drop")
            if tp is None or tp["worst"]["delta_v"] != dv:
                raise ValueError(
                    f"scan: {b.summary_path} ranked at ΔV {dv} from its stored values but its "
                    f"expanded row reads {None if tp is None else tp['worst']['delta_v']} — the "
                    "recorded decision count disagrees with the trace; refused rather than ranked")
            rows.append(tp)
        return rows

    def _worst_turning_point(self, battle: BattleTrace, metric: str) -> "dict | None":
        """The single worst decision in one battle by `metric` (model-free).
        Boards are built only for the chosen invocation, so a scan stays cheap."""
        summary = self._summary(battle)
        invs = summary.get("invocations", [])
        if not invs:
            return None
        npz = self._npz(battle)
        values = npz.get("values")
        best_i = best_score = best_dv = best_td = None
        for i, inv in enumerate(invs):
            v, v_next = self._v(values, i), self._v(values, i + 1)
            dv = (v_next - v) if (v is not None and v_next is not None) else None
            reward = (inv.get("outcome") or {}).get("reward")
            rtotal = reward.get("total") if isinstance(reward, dict) else reward
            td = self._td(rtotal, v, v_next)
            score = dv if metric == "value_drop" else td
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_i, best_score, best_dv, best_td = i, score, dv, td
        if best_i is None:
            return None
        inv = invs[best_i]
        board = build_board(inv)
        reward = (inv.get("outcome") or {}).get("reward")
        rtotal = reward.get("total") if isinstance(reward, dict) else reward
        worst = {
            "inv": best_i, "turn": inv.get("turn"), "phase": inv.get("phase"),
            "chosen": inv.get("chosen", ""),
            "our_active": _active_str(board.ours), "opp_active": _active_str(board.opp),
            "delta_v": best_dv, "td_residual": best_td, "reward_total": rtotal,
            "events": (inv.get("outcome") or {}).get("events") or [],
            "flags": list(summary_flags(inv)),
        }
        # Decode the incoming-damage / OHKO belief the obs HELD at this cliff (model-free, from the
        # saved obs). The decisive A/B for "did the feature fill the obs gap": a high active_pko at
        # a value cliff means the OHKO WAS in the obs (remaining error is downstream usage); a low
        # one where our active then faints means the belief is mis-calibrated.
        belief = self._belief_at(npz, best_i)
        if belief is not None:
            worst["incoming_active_pko"] = belief.active_pko
            worst["incoming_max_pko"] = belief.max_pko
            worst["incoming_active_outspeed"] = belief.active_outspeed
        return {
            "id": battle.summary_path, "short_id": _short_id(battle),
            "opponent": battle.opponent, "step": battle.step, "outcome": battle.outcome,
            "turns": (summary.get("meta") or {}).get("turns"),
            "worst": worst,
        }


    # -- loss attribution: rank failure categories by recoverable rating -----

    def _turning_point_features(self, battle: BattleTrace) -> "dict | None":
        """The DECISIVE turning point of one loss (worst ΔV) as a feature dict the engine taxonomy
        categorizes. Model-free (summary + npz). None on an empty trace."""
        summary = self._summary(battle)
        invs = summary.get("invocations", [])
        if not invs:
            return None
        npz = self._npz(battle)          # ONE read: `_npz` reopens the file on every call
        values = npz.get("values")
        best_i = best_dv = None
        for i in range(len(invs)):
            v, vn = self._v(values, i), self._v(values, i + 1)
            dv = (vn - v) if (v is not None and vn is not None) else None
            if dv is None:
                continue
            if best_dv is None or dv < best_dv:
                best_i, best_dv = i, dv
        if best_i is None:                      # no value series — fall back to the last decision
            best_i = len(invs) - 1
        inv = invs[best_i]
        # Critic surprise at the cliff: δ = r + γV(s') − V(s). A large |δ| means the value crater
        # CAUGHT THE CRITIC OFF GUARD (capacity / missing-obs lever); a small δ means it declined as
        # the critic already expected (a genuinely-lost position, not a critic fix). Splits value_cliff.
        _reward = (inv.get("outcome") or {}).get("reward")
        _rtotal = _reward.get("total") if isinstance(_reward, dict) else _reward
        v_at = self._v(values, best_i)
        # Recorded P(win) at the cliff — the CALIBRATED winning-vs-losing signal that re-centers the
        # grind/throw split (V's sign mis-centers it; see engine._was_winning). None on a no-win-prob run.
        wp_at = _npz_win_prob(npz, best_i)
        td = self._td(_rtotal, v_at, self._v(values, best_i + 1))
        board = build_board(inv)
        our = inv.get("our") or {}
        ocour = (inv.get("outcome") or {}).get("our") or {}

        def pp(s):
            try:
                return parse_pct(s)
            except Exception:  # noqa: BLE001
                return None

        our_hp, our_dhp = pp(our.get("hp")), pp(ocour.get("hp_delta"))
        chosen = str(inv.get("chosen", ""))
        is_switch = chosen.startswith("switch")
        move_id = "" if is_switch else chosen.lower().replace(" ", "")
        belief = self._belief_at(npz, best_i)
        psp = list(belief.per_slot_pko) if (belief is not None and belief.per_slot_pko) else None
        return {
            "short_id": _short_id(battle), "opponent": battle.opponent, "step": battle.step,
            "turns": (summary.get("meta") or {}).get("turns"),
            "inv": best_i, "turn": inv.get("turn"), "phase": inv.get("phase"), "chosen": chosen,
            "is_switch": is_switch, "is_setup": move_id in SETUP_MOVES,
            "our_species": our.get("species"), "our_hp": our_hp, "our_hp_delta": our_dhp,
            "faint": bool(our_hp is not None and our_dhp is not None and our_hp + our_dhp <= 0.02),
            "active_pko": (belief.active_pko if belief else None),
            "active_outspeed": (belief.active_outspeed if belief else None),
            "max_pko": (belief.max_pko if belief else None),
            "n_healthy_bench": sum(1 for m in board.ours.bench if not m.fainted),
            "min_other_pko": (min(psp) if psp else None),
            "delta_v": best_dv, "td": td, "v_at": v_at, "wp_at": wp_at,
        }

    def _win_rates(self, step: "int | None") -> dict:
        """Per-opponent TRUE win-rate from eval_results.jsonl (row nearest `step`, else latest) — the
        loss-rate weight for the attribution. {} if absent. (NOT the trace win/loss tally, which is
        loss-WEIGHTED and would over-state the loss volume.)"""
        try:
            with open(os.path.join(self.run_dir or "", "eval_results.jsonl")) as f:
                rows = [json.loads(ln) for ln in f if ln.strip()]
        except (OSError, ValueError, TypeError):
            return {}
        if not rows:
            return {}
        row = (min(rows, key=lambda r: abs(r.get("step", 0) - step)) if step is not None else rows[-1])
        return dict(row.get("bots") or {})

    def _triage_select(self, step: "int | None", opponent: "str | None", wp_even: float,
                       v_even: "float | None"):
        """`(step, losses, v_even, currency, key)` for a triage call: the step resolved (the newest
        with a loss when None), the losses it reads, the even-point resolved from the critic
        currency, and the CACHE KEY — the arguments plus each loss's summary + npz mtime and
        `eval_results.jsonl`'s, so a re-written trace or a new win-rate row is a miss. Stats only."""
        currency = self.critic_currency()
        if v_even is None:
            v_even = float(currency["even"])
        losses = [b for b in self.tree.all_battles() if b.outcome == "loss"]
        if step is None:
            steps = sorted({b.step for b in losses})
            step = steps[-1] if steps else None
        losses = [b for b in losses
                  if (step is None or b.step == step) and (not opponent or b.opponent == opponent)]

        def mtime(path):
            try:
                return os.stat(path).st_mtime_ns if path else None
            except OSError:
                return None

        key = (step, opponent, float(wp_even), float(v_even),
               mtime(os.path.join(self.run_dir or "", "eval_results.jsonl")),
               tuple((b.summary_path, mtime(b.summary_path), mtime(b.npz_path)) for b in losses))
        return step, losses, v_even, currency, key

    def triage_cached(self, *, step: "int | None" = None, opponent: "str | None" = None,
                      wp_even: float = WP_EVEN_DEFAULT, v_even: "float | None" = None) -> "dict | None":
        """`triage(...)`'s result if it is already cached and still current, else None — never computes.
        Lets a surface render a warm triage at once and fetch a cold one lazily."""
        key = self._triage_select(step, opponent, wp_even, v_even)[4]
        with self._summaries_lock:
            hit = self._triage_cache.get(key)
        return copy.deepcopy(hit) if hit is not None else None

    def triage(self, *, step: "int | None" = None, opponent: "str | None" = None,
               wp_even: float = WP_EVEN_DEFAULT, v_even: "float | None" = None,
               max_expand: "int | None" = None) -> dict:
        """Loss attribution: categorize every loss's decisive turning point into the engine taxonomy,
        aggregate, and RANK the failure categories by estimated recoverable win-rate (the lever
        prioritization). Model-free. Defaults to the latest step that has loss traces.

        The grind-vs-throw boundary (positional_grind vs critic_blindspot) splits on whether the model
        rated itself WINNING right before the cliff. That uses the calibrated win-prob head
        (``P(win) ≥ wp_even``, default 0.5) when the traces carry it, falling back to ``V > v_even``
        (default 0). NB: V's zero is NOT "even" — V is a shaped/discounted return with a structural
        negative offset (a self-mirror 50/50 reads V≈−6.5), so the V-fallback over-counts grinds; pass
        ``v_even`` = the checkpoint's structural even-point (its self-mirror V / PopArt μ) to re-center a
        no-win-prob run.

        ``v_even=None`` (the default) resolves that even-point from the run's CRITIC CURRENCY: 0.0
        on a shaped critic — the documented over-counting fallback above — but **0.5 under
        the win-prob critic**, where V *is* P(win) and 0.0 is not "even" but a certain loss. On a
        winprob run the fallback is also unreachable in practice (``values`` equals ``win_probs``,
        so the primary split always has its input), but a threshold that is wrong only where it is
        currently unused is still wrong, and the next reader will not know that.

        CACHED per (step, opponent, thresholds), invalidated by any read loss's summary / npz mtime
        or `eval_results.jsonl`'s (`_triage_select`); a hit is a copy. Each loss is a core-trace
        expansion, so on rb_x26_s2601 a cold step measured ~47 s. ``max_expand`` refuses a cold call
        that would expand more losses (`ScanTooBroad`, naming the CLI) before expanding any."""
        step, losses, v_even, currency, key = self._triage_select(step, opponent, wp_even, v_even)
        with self._summaries_lock:
            hit = self._triage_cache.get(key)
            if hit is not None:
                self._triage_cache.move_to_end(key)
        if hit is not None:
            return copy.deepcopy(hit)
        if max_expand is not None and len(losses) > max_expand:
            flags = "".join(f" --{k} {v}" for k, v in (("step", step), ("opponent", opponent)) if v)
            raise ScanTooBroad("this triage", len(losses), max_expand,
                               f"python -m main.prober.query triage {self.run_dir or '<run_dir>'}{flags}")
        out = self._triage_compute(step, opponent, wp_even, v_even, currency, losses)
        with self._summaries_lock:
            self._triage_cache[key] = copy.deepcopy(out)
            while len(self._triage_cache) > _MAX_CACHED_TRIAGES:
                self._triage_cache.popitem(last=False)
        return out

    def _triage_compute(self, step: "int | None", opponent: "str | None", wp_even: float, v_even: float,
                        currency: dict, losses: "list[BattleTrace]") -> dict:
        from collections import defaultdict
        wr = self._win_rates(step)

        per_opp = defaultdict(lambda: {"n": 0, "cats": defaultdict(list)})
        cat_meta: dict = {}
        n_wp_split = 0                                   # losses whose grind/throw split used the win-prob head
        for b in losses:
            feat = self._turning_point_features(b)
            if feat is None:
                continue
            feat["wp_even"], feat["v_even"] = wp_even, v_even   # the winning-thresholds the taxonomy reads
            if feat.get("wp_at") is not None:
                n_wp_split += 1
            a = attribute_turning_point(feat)
            cat_meta[a["category"]] = a
            po = per_opp[b.opponent]
            po["n"] += 1
            po["cats"][a["category"]].append(feat)

        bot_opps = [o for o in per_opp if o in wr]                    # only the fixed-bot anchor
        cat_total: dict = defaultdict(int)
        cat_recover: dict = defaultdict(float)
        cat_by_opp: dict = defaultdict(lambda: defaultdict(int))
        cat_examples: dict = defaultdict(list)
        for opp, po in per_opp.items():
            for cat, feats in po["cats"].items():
                cat_total[cat] += len(feats)
                cat_by_opp[cat][opp] = len(feats)
                for f in feats[:3]:
                    cat_examples[cat].append(
                        f"{f['short_id']} t{f.get('turn')}: {f.get('our_species')} chose "
                        f"{f.get('chosen')} (pko={_r(f.get('active_pko'),2)}, "
                        f"outspeed={_r(f.get('active_outspeed'),2)}, dHP={_r(f.get('our_hp_delta'),2)}, "
                        f"healthy_bench={f.get('n_healthy_bench')})")
        for opp in bot_opps:
            n = per_opp[opp]["n"] or 1
            loss_rate = max(0.0, 1.0 - float(wr.get(opp, 0.0)))
            for cat, feats in per_opp[opp]["cats"].items():
                cat_recover[cat] += loss_rate * (len(feats) / n) / max(1, len(bot_opps))

        total = sum(cat_total.values())

        # Rank by recoverable win-rate; break ties by raw volume so the order is stable AND meaningful
        # even when no bot win-rates exist (recover all 0 → fall back to "most losses first").
        cats = [{
            "category": c, "lever": cat_meta[c]["lever"], "blurb": cat_meta[c]["blurb"],
            "n": cat_total[c], "pct_of_sampled_losses": round(100 * cat_total[c] / max(1, total), 1),
            "est_recoverable_winrate_pct": round(100 * cat_recover.get(c, 0.0), 2),
            "by_opponent": dict(cat_by_opp[c]),
            "examples": cat_examples[c][:4],
        } for c in sorted(cat_total, key=lambda c: (-cat_recover.get(c, 0.0), -cat_total[c]))]

        ranking_metric = ("est_recoverable_winrate_pct = mean over BOT opponents of "
                          "loss_rate(opp) × category_share(opp); an UPPER BOUND (assumes fixing "
                          "the lever flips that loss). Ranked descending — this is the lever order.")
        wp_frac = round(n_wp_split / max(1, total), 2)
        split_signal = (
            f"grind-vs-throw boundary: P(win) ≥ {wp_even} when the win-prob head is recorded "
            f"({int(100 * wp_frac)}% of these losses), else V > {v_even}. The calibrated win-prob split "
            "re-centers the old V>0 test, which OVER-counted grinds (V's zero is ~6 units below 'even' — "
            "a shaped-return offset, measured via a self-mirror 50/50 reading V<0).")
        caveats = [
            "Eval traces are LOSS-WEIGHTED (~10 loss / 5 win per opponent), so category SHARES are "
            "per-opponent representative but raw counts are NOT the true loss volume — the recoverable "
            "estimate corrects for that via the true per-opponent win-rate.",
            "Each loss is attributed to its single worst-ΔV turning point; a loss can have several causes.",
            "Recoverable-winrate is over BOT opponents only (the ELO anchor); sentinel/ext losses are "
            "counted (pct_of_sampled_losses) but not rating-weighted (their win-rate is gate-pinned).",
            split_signal,
        ]
        if not bot_opps:
            ranking_metric = ("no eval_results.jsonl bot win-rates found — est_recoverable_winrate_pct "
                              "is 0 for every category; ranked by RAW loss volume instead.")
            caveats.append("No bot win-rates available: ranking fell back to loss count, NOT "
                           "rating-weighted recoverability. Run an eval cycle (writes eval_results.jsonl) "
                           "for the lever-prioritized order.")

        return {
            "run_dir": self.run_dir, "step": step,
            "n_losses_analyzed": total, "n_bot_opponents": len(bot_opps),
            "bot_win_rates": {o: wr[o] for o in bot_opps},
            "ranking_metric": ranking_metric,
            "winning_split": {"wp_even": wp_even, "v_even": v_even, "wp_coverage": wp_frac,
                              "critic_mode": currency["mode"], "v_units": currency["units"]},
            "caveats": caveats,
            "categories": cats,
        }
